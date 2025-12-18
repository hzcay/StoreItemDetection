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
    
    # Auto-detect color_encoder from checkpoint
    state_dict = checkpoint.get('model_state_dict', {})
    if not use_color_embedding and any('color_encoder' in k for k in state_dict.keys()):
        use_color_embedding = True
        print("Auto-detected color_encoder in checkpoint, enabling color embedding")
    
    inverted_residual_setting, last_channel = res_mobilenet_conf(width_mult=1.0)
    model = ResMobileNetV2(
        inverted_residual_setting=inverted_residual_setting,
        embedding_size=embedding_size,
        num_classes=num_classes,
        use_color_embedding=use_color_embedding,
        color_embedding_size=color_embedding_size
    )   
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Load color_alpha if exists (from training config)
    if use_color_embedding and 'color_alpha' in checkpoint:
        model.color_alpha = checkpoint.get('color_alpha', 0.3)
    
    model.to(DEVICE)
    model.eval()
    
    print(f"Model initialized successfully (use_color_embedding={use_color_embedding})")

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
def get_image_embedding(image_path: str, model: nn.Module, return_color: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Extract embedding(s) from image for Late Fusion
    
    Returns:
        If return_color=False: visual_embedding (embedding_size,)
        If return_color=True: (visual_embedding, color_embedding) - both normalized
    """
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        if return_color:
            visual_emb, color_emb = model(img_tensor, return_color=True)
            visual_emb = visual_emb.squeeze()
            visual_emb = F.normalize(visual_emb, p=2, dim=0)
            visual_emb = visual_emb.cpu().numpy().astype(np.float32)
            
            if color_emb is not None:
                color_emb = color_emb.squeeze()
                color_emb = F.normalize(color_emb, p=2, dim=0)
                color_emb = color_emb.cpu().numpy().astype(np.float32)
            else:
                color_emb = None
            
            return visual_emb, color_emb
        else:
            visual_emb = model(img_tensor).squeeze()
            visual_emb = F.normalize(visual_emb, p=2, dim=0)
            visual_emb = visual_emb.cpu().numpy().astype(np.float32)
            return visual_emb


def search_similar(
    model: nn.Module,
    client: QdrantClient,
    image_path: str,
    top_k: int = 20,
    visual_weight: float = 0.6,
    color_weight: float = 0.4
):
    if client is None:
        raise RuntimeError("Qdrant client not initialized")
    if model is None:
        raise RuntimeError("Model not initialized")
    
    # Validate weights
    if abs(visual_weight + color_weight - 1.0) > 0.01:
        total = visual_weight + color_weight
        visual_weight = visual_weight / total
        color_weight = color_weight / total
    
    # Check collection
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            return []
        
        collection_info = client.get_collection(QDRANT_COLLECTION)
        point_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
        if point_count == 0:
            return []
        
        config = collection_info.config if hasattr(collection_info, 'config') else None
        uses_named_vectors = False
        if config and hasattr(config, 'params') and hasattr(config.params, 'vectors'):
            vectors_config = config.params.vectors
            uses_named_vectors = isinstance(vectors_config, dict) and 'visual' in vectors_config
        
        if not uses_named_vectors:
            raise RuntimeError("Collection must use named vectors for Late Fusion")
    except Exception as e:
        print(f"Error checking collection: {e}")
        return []
    
    # Extract embeddings
    try:
        has_color = hasattr(model, 'use_color_embedding') and model.use_color_embedding
        if not has_color:
            raise RuntimeError("Model must have color embedding for Late Fusion")
        
        query_visual, query_color = get_image_embedding(image_path, model, return_color=True)
        if query_color is None:
            raise RuntimeError("Failed to extract color embedding")
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
        return []
    
    # Get all points and compute Late Fusion scores
    try:
        all_points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=10000,
            with_payload=True,
            with_vectors=True
        )
    except Exception as e:
        print(f"Error scrolling collection: {e}")
        return []
    
    # Compute Late Fusion scores
    results = []
    for point in all_points:
        if not hasattr(point, 'vector') or not isinstance(point.vector, dict):
            continue
        
        stored_visual = np.array(point.vector.get('visual', []), dtype=np.float32)
        stored_color = np.array(point.vector.get('color', []), dtype=np.float32)
        
        if len(stored_visual) == 0 or len(stored_color) == 0:
            continue
        
        # Cosine similarities (vectors already normalized)
        visual_sim = float(np.dot(query_visual, stored_visual))
        color_sim = float(np.dot(query_color, stored_color))
        
        # Late Fusion: score = α * visual_sim + β * color_sim
        final_score = visual_weight * visual_sim + color_weight * color_sim
        
        results.append({
            "id": point.id,
            "score": final_score,
            "visual_score": visual_sim,
            "color_score": color_sim,
            "payload": getattr(point, 'payload', {})
        })
    
    # Sort and return top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]



def save_embedding_to_qdrant(
    model: nn.Module,
    client: QdrantClient,
    product_id: int,
    image_path: str,
    additional_data: Optional[Dict[str, Any]] = None,
    point_id: Optional[Union[int, str]] = None,
) -> bool:
    if client is None:
        return False
    
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        # Extract embeddings separately for Late Fusion
        visual_emb, color_emb = get_image_embedding(image_path, model, return_color=True)
        
        if color_emb is None:
            raise RuntimeError("Model must have color embedding for Late Fusion")
        
        # Create collection if needed with named vectors
        if QDRANT_COLLECTION not in collection_names:
            vectors_config = {
                "visual": VectorParams(size=len(visual_emb), distance=Distance.COSINE),
                "color": VectorParams(size=len(color_emb), distance=Distance.COSINE)
            }
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=vectors_config
            )
        
        # Prepare payload and point with named vectors
        payload = {
            "product_id": product_id,
            "image_path": image_path,
            **(additional_data or {})
        }
        
        vector_dict = {
            "visual": visual_emb.tolist(),
            "color": color_emb.tolist()
        }
        
        point = PointStruct(
            id=point_id if point_id is not None else product_id,
            vector=vector_dict,
            payload=payload
        )
        
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[point]
        )
        
        return True
    except Exception as e:
        print(f"Error saving to Qdrant: {str(e)}")
        return False