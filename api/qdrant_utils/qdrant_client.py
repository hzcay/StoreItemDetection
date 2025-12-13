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
from qdrant_utils.color_rerank import compute_color_score
from qdrant_utils.local_feature_rerank import compute_local_feature_score

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
    
    inverted_residual_setting, last_channel = res_mobilenet_conf(width_mult=1.0)
    model = ResMobileNetV2(
        inverted_residual_setting=inverted_residual_setting,
        embedding_size=embedding_size,
        num_classes=num_classes
    )   
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.to(DEVICE)
    model.eval()
    
    print("Model initialized successfully")

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
def get_image_embedding(image_path: str, model: nn.Module) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(DEVICE)
    
    with torch.no_grad():
        embedding = model(img_tensor).squeeze() 
        embedding = F.normalize(embedding, p=2, dim=0)
        embedding = embedding.cpu().numpy().astype(np.float32)

    return embedding


def search_similar(
    model: nn.Module,
    client: QdrantClient,
    image_path: str,
    top_k: int = 20
):
    if client is None:
        raise RuntimeError("Qdrant client not initialized")

    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            print(f"Warning: Collection '{QDRANT_COLLECTION}' does not exist")
            return []
        
        collection_info = client.get_collection(QDRANT_COLLECTION)
        point_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
        print(f"Collection '{QDRANT_COLLECTION}' has {point_count} points")
        
        if point_count == 0:
            print("Warning: Collection is empty, no results to return")
            return []
    except Exception as e:
        print(f"Error checking collection: {e}")
        return []

    query_vec = get_image_embedding(image_path, model).tolist()
    print(f"Query vector shape: {len(query_vec)}, first 5 values: {query_vec[:5]}")
    
    search_result = None
    try:
        search_result = client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True
        )
        print(f"Search method: .search(), result type: {type(search_result)}")
    except (AttributeError, TypeError) as e:
        print(f".search() failed: {e}, trying alternatives...")
        try:
            search_result = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vec,
                limit=top_k,
                with_payload=True
            )
            print(f"Search method: .query_points(), result type: {type(search_result)}")
        except (AttributeError, TypeError) as e2:
            print(f".query_points() failed: {e2}, trying search_batch...")
            try:
                from qdrant_client.models import SearchRequest
                batch_result = client.search_batch(
                    collection_name=QDRANT_COLLECTION,
                    requests=[SearchRequest(vector=query_vec, limit=top_k, with_payload=True)]
                )
                if batch_result and len(batch_result) > 0:
                    search_result = batch_result[0]
                print(f"Search method: .search_batch(), result type: {type(search_result)}")
            except Exception as e3:
                print(f"All search methods failed. Last error: {e3}")
                raise RuntimeError(f"Could not find compatible search method. Errors: {e}, {e2}, {e3}")
    
    if search_result is None:
        print("Warning: Search returned None")
        return []
    
    if isinstance(search_result, tuple):
        hits = search_result[0] if len(search_result) > 0 else []
        print(f"Search result is tuple, extracted {len(hits)} hits")
    elif hasattr(search_result, 'points'):
        hits = search_result.points
        print(f"Search result has .points attribute, extracted {len(hits) if hasattr(hits, '__len__') else 'unknown'} hits")
    elif hasattr(search_result, 'result'):
        hits = search_result.result
        print(f"Search result has .result attribute, extracted {len(hits) if hasattr(hits, '__len__') else 'unknown'} hits")
    elif isinstance(search_result, list):
        hits = search_result
        print(f"Search result is list with {len(hits)} hits")
    elif hasattr(search_result, '__iter__') and not isinstance(search_result, (str, bytes)):
        hits = list(search_result)
        print(f"Search result is iterable, converted to list with {len(hits)} hits")
    else:
        hits = [search_result]
        print(f"Search result is single object, wrapped in list")

    results = []
    for hit in hits:
        if hasattr(hit, 'id') and hasattr(hit, 'score'):
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": getattr(hit, 'payload', {})
            })
        elif isinstance(hit, dict):
            results.append(hit)
    
    print(f"Returning {len(results)} results")
    return results


def get_dino_model():
    """
    Lazy load DINOv2-Small for visual rerank.
    """
    global _DINO_MODEL_CACHE
    if _DINO_MODEL_CACHE is not None:
        return _DINO_MODEL_CACHE
    try:
        dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        dino_model.to(DEVICE)
        dino_model.eval()
        _DINO_MODEL_CACHE = dino_model
        print("DINOv2 model loaded successfully")
        return dino_model
    except Exception as e:
        print(f"Error loading DINOv2 model: {e}")
        return None


def get_dino_embedding(image_path: str) -> Optional[np.ndarray]:
    model = get_dino_model()
    if model is None:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _DINO_TRANSFORM(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = model(img_tensor)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            feat = F.normalize(feat.squeeze(), p=2, dim=0)
        return feat.detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error computing DINO embedding for {image_path}: {e}")
        return None


def get_clip_model():
    """
    Lazy load CLIP model (chỉ load khi cần, vì nặng).
    """
    global _CLIP_MODEL_CACHE, _CLIP_PROCESSOR_CACHE
    
    if _CLIP_MODEL_CACHE is not None:
        return _CLIP_MODEL_CACHE, _CLIP_PROCESSOR_CACHE
    
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        
        _CLIP_MODEL_CACHE = model
        _CLIP_PROCESSOR_CACHE = preprocess
        
        print("CLIP model loaded successfully")
        return model, preprocess
    except ImportError:
        print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None, None
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return None, None


def get_clip_embedding(image_path: str) -> Optional[np.ndarray]:
    model, preprocess = get_clip_model()
    if model is None or preprocess is None:
        print("Warning: CLIP model not available")
        return None
    try:
        device = next(model.parameters()).device
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(img_tensor)
            features = F.normalize(features, p=2, dim=1)
        return features.squeeze().detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error computing CLIP embedding for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def search_with_rerank(
    model: nn.Module,
    client: QdrantClient,
    query_image_path: str,
    top_k: int = 200,
    local_weight: float = 0.45,  # BRAND
    color_weight: float = 0.35,  # VARIANT
    embed_weight: float = 0.20,  # SEMANTIC
    fusion_threshold: float = 0.0,
) -> List[Dict[str, Any]]:

    
    if client is None:
        raise RuntimeError("Qdrant client not initialized")

    initial_results = search_similar(model, client, query_image_path, top_k=top_k)
    if not initial_results:
        print("No initial results found")
        return []
    
    print(f"Found {len(initial_results)} initial candidates")
    
    reranked_results = []
    for idx, hit in enumerate(initial_results):
        candidate_image_path = hit.get("payload", {}).get("image_path")
        if not candidate_image_path or not os.path.exists(candidate_image_path):
            continue
        
        embed_score_raw = hit.get("score", 0.0)
        embed_score = max(0.0, min(1.0, float(embed_score_raw)))
        
        local_result = {}
        try:
            local_result = compute_local_feature_score(
                query_image_path,
                candidate_image_path
            )
            local_score = local_result.get("local_score", 0.0)
        except Exception as e:
            print(f"Error computing local feature score for candidate {idx}: {e}")
            local_score = 0.0
            local_result = {"error": str(e)}
        
        color_result = {}
        try:
            color_result = compute_color_score(
                query_image_path,
                candidate_image_path,
                color_space="HSV",
                k_colors=3
            )
            color_score = color_result.get("color_score", 0.5)
        except Exception as e:
            print(f"Error computing color score for candidate {idx}: {e}")
            color_score = 0.5
            color_result = {"error": str(e)}
        
        final_score = (
            local_weight * local_score +      # BRAND (0.55)
            color_weight * color_score +      # VARIANT (0.25)
            embed_weight * embed_score        # SEMANTIC (0.20)
        )
        
        result_item = {
            **hit,
            "embed_score": float(embed_score),
            "local_score": float(local_score),
            "color_score": float(color_score),
            "final_score": float(final_score),
            "local_details": local_result,
            "color_details": color_result,
        }
        
        reranked_results.append(result_item)
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(initial_results)} candidates...")
    
    reranked_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    if fusion_threshold is not None and fusion_threshold > 0:
        reranked_results = [r for r in reranked_results if r["final_score"] >= fusion_threshold]
    
    
    return reranked_results


def save_embedding_to_qdrant(
    model: nn.Module,
    client: QdrantClient,
    product_id: int,
    image_path: str,
    additional_data: Optional[Dict[str, Any]] = None,
    point_id: Optional[Union[int, str]] = None,
) -> bool:

    if client is None:
        print("Qdrant client not initialized")
        return False

    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if QDRANT_COLLECTION not in collection_names:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )

        embedding = get_image_embedding(image_path, model)
        payload = {
            "product_id": product_id,
            "image_path": image_path,
            **(additional_data or {})
        }

        point = PointStruct(
            id=point_id if point_id is not None else product_id,
            vector=embedding.tolist(),
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