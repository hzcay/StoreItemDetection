import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch.nn as nn
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from models.backbone.ResMobileNetV2 import ResMobileNetV2, res_mobilenet_conf
import io
# -------------------------------
# Qdrant configuration
# -------------------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "product_embeddings")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

EMBEDDING_DIM = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Module-level singletons to avoid reloading per request
_MODEL_CACHE: Optional[ResMobileNetV2] = None
_QDRANT_CLIENT: Optional[QdrantClient] = None
_DINO_MODEL_CACHE = None
_DINO_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_CLIP_MODEL_CACHE = None
_CLIP_PROCESSOR_CACHE = None

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_model():
    return get_model()


def get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    checkpoint_path = os.path.join("data", "checkpoints", "situ", "best.pth")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(DEVICE))
    
    embedding_size = checkpoint.get('embedding_size', EMBEDDING_DIM)
    num_classes = checkpoint.get('num_classes', 1000)
    use_color_embedding = checkpoint.get('use_color_embedding', False)
    color_embedding_size = checkpoint.get('color_embedding_size', 64)
    
    # Auto-detect color_encoder from checkpoint state_dict
    state_dict = checkpoint.get('model_state_dict', {})
    # Kiểm tra nhiều patterns để detect color encoder
    color_encoder_keys = [
        k for k in state_dict.keys() 
        if 'color_encoder' in k.lower() or 'colorencoder' in k.lower()
    ]
    
    if color_encoder_keys:
        use_color_embedding = True
        print(f"✅ Auto-detected color_encoder in checkpoint ({len(color_encoder_keys)} keys), enabling color embedding")
    elif use_color_embedding:
        print(f"✅ Using color_embedding from checkpoint config")
    else:
        # Late Fusion requires color_embedding, but checkpoint doesn't have it
        print(f"⚠️  WARNING: No color_encoder detected in checkpoint!")
        print(f"   Late Fusion requires color_embedding. Please retrain with --use-color-embedding flag.")
        print(f"   Attempting to enable color_embedding anyway (may not match pretrained weights)...")
        use_color_embedding = True  # Force enable for Late Fusion
    
    inverted_residual_setting, last_channel = res_mobilenet_conf(width_mult=1.0)
    model = ResMobileNetV2(
        inverted_residual_setting=inverted_residual_setting,
        embedding_size=embedding_size,
        num_classes=num_classes,
        use_color_embedding=use_color_embedding,
        color_embedding_size=color_embedding_size
    )   
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Verify color_encoder was loaded correctly
    if use_color_embedding and model.color_encoder is None:
        print("⚠️  WARNING: use_color_embedding=True but model.color_encoder is None!")
        print("   This may cause errors in Late Fusion. Re-initializing color_encoder...")
        from models.backbone.ResMobileNetV2 import HSVColorEncoder
        model.color_encoder = HSVColorEncoder(embedding_size=color_embedding_size)
        model.color_encoder.to(DEVICE)
        model.color_encoder.eval()
    
    # Load color_alpha if exists (from training config)
    if use_color_embedding and 'color_alpha' in checkpoint:
        model.color_alpha = checkpoint.get('color_alpha', 0.3)
    
    model.to(DEVICE)
    model.eval()
    
    # Final check
    has_color = model.color_encoder is not None
    print(f"✅ Model initialized successfully (color_encoder={'enabled' if has_color else 'disabled'})")

    _MODEL_CACHE = model
    return _MODEL_CACHE

def create_qdrant_client() -> Optional[QdrantClient]:
    return get_qdrant_client()


def get_qdrant_client() -> Optional[QdrantClient]:
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    try:
        client = QdrantClient(
            url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
            api_key=QDRANT_API_KEY,
            headers={"api-key": QDRANT_API_KEY}
        )
        client.get_collections()
        print("Connected to Qdrant successfully")
        _QDRANT_CLIENT = client
        return _QDRANT_CLIENT
    except Exception as e:
        print(f"Warning: Could not connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print("Please ensure the Qdrant service is running and API key is correct")
        print(f"Error details: {str(e)}")
        return None

# -------------------------------
# Extract embedding
# -------------------------------
def get_image_embedding(
    image_bytes: bytes,
    model: nn.Module,
    return_color: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Extract embedding(s) from image bytes for Late Fusion.

    Returns:
        If return_color=False:
            visual_embedding (embedding_size,)
        If return_color=True:
            (visual_embedding, color_embedding)
    """

    # Load image from memory
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply same transform used in training
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Model ở eval mode luôn trả về (visual_emb, color_emb)
        visual_emb, color_emb = model(img_tensor)
        
        visual_emb = visual_emb.squeeze()
        visual_emb = F.normalize(visual_emb, p=2, dim=0)
        visual_emb = visual_emb.cpu().numpy().astype(np.float32)
        
        if return_color:
            if color_emb is not None:
                color_emb = color_emb.squeeze()
                color_emb = F.normalize(color_emb, p=2, dim=0)
                color_emb = color_emb.cpu().numpy().astype(np.float32)
            else:
                color_emb = None

            return visual_emb, color_emb

        else:
            return visual_emb

def search_similar(
    model: nn.Module,
    client: QdrantClient,
    image_bytes: bytes,     # 🔒 BYTES ONLY
    top_k: int = 20,
    visual_weight: float = 0.7,  # Giảm từ 0.8 xuống 0.6 để linh hoạt hơn
    color_weight: float = 0.3     # Tăng từ 0.2 lên 0.4 vì color match tốt hơn
):
    if client is None:
        raise RuntimeError("Qdrant client not initialized")
    if model is None:
        raise RuntimeError("Model not initialized")
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise TypeError(f"image_bytes must be bytes, got {type(image_bytes)}")

    # Normalize weights
    total = visual_weight + color_weight
    if abs(total - 1.0) > 0.01:
        visual_weight /= total
        color_weight /= total

    # Check collection
    try:
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in names:
            return []

        info = client.get_collection(QDRANT_COLLECTION)
        if getattr(info, "points_count", 0) == 0:
            return []

        vectors_cfg = info.config.params.vectors
        if not isinstance(vectors_cfg, dict) or "visual" not in vectors_cfg:
            raise RuntimeError("Collection must use named vectors: visual + color")

    except Exception as e:
        print(f"❌ Collection check failed: {e}")
        return []

    # Extract embeddings
    try:
        # Check if model has color_encoder (not use_color_embedding attribute)
        has_color = model.color_encoder is not None
        if not has_color:
            raise RuntimeError("Model must have color_encoder for Late Fusion. Please ensure checkpoint was trained with --use-color-embedding")
        
        query_visual, query_color = get_image_embedding(image_bytes, model, return_color=True)
        if query_color is None:
            raise RuntimeError("Failed to extract color embedding from model")
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return []

    # Scroll points
    try:
        points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=True
        )
    except Exception as e:
        print(f"❌ Qdrant scroll failed: {e}")
        return []

    # Late Fusion scoring
    results = []
    for p in points:
        if not isinstance(p.vector, dict):
            continue

        v = np.asarray(p.vector.get("visual"), dtype=np.float32)
        c = np.asarray(p.vector.get("color"), dtype=np.float32)

        if v.size == 0 or c.size == 0:
            continue

        visual_sim = float(np.dot(query_visual, v))
        color_sim = float(np.dot(query_color, c))
        score = visual_weight * visual_sim + color_weight * color_sim

        results.append({
            "id": p.id,
            "score": score,
            "visual_score": visual_sim,
            "color_score": color_sim,
            "payload": p.payload or {}
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]




def save_embedding_to_qdrant(
    model: nn.Module,
    client: QdrantClient,
    product_id: int,
    image_bytes: bytes,
    additional_data: Optional[Dict[str, Any]] = None,
    point_id: Optional[Union[int, str]] = None,
) -> bool:
    if client is None:
        return False

    try:
        # ---- Ensure collection exists ----
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Check if model has color_encoder
        if model.color_encoder is None:
            raise RuntimeError("Model must have color_encoder for Late Fusion. Please ensure checkpoint was trained with --use-color-embedding")
        
        # Extract embeddings separately for Late Fusion
        visual_emb, color_emb = get_image_embedding(image_bytes, model, return_color=True)
        
        if color_emb is None:
            raise RuntimeError("Failed to extract color embedding from model")
        
        # Create collection if needed with named vectors
        if QDRANT_COLLECTION not in collection_names:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    "visual": VectorParams(
                        size=len(visual_emb),
                        distance=Distance.COSINE
                    ),
                    "color": VectorParams(
                        size=len(color_emb),
                        distance=Distance.COSINE
                    ),
                }
            )

        # ---- Prepare Qdrant point ----
        point = PointStruct(
            id=point_id or str(product_id),
            vector={
                "visual": visual_emb.tolist(),
                "color": color_emb.tolist(),
            },
            payload={
                "product_id": product_id,
                **(additional_data or {})
            }
        )

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point]
        )

        return True

    except Exception as e:
        print(f"❌ Error saving to Qdrant: {str(e)}")
        return False