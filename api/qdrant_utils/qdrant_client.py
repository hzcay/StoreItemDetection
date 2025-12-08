import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from models.backbone.ResMobileNetV2 import ResMobileNetV2, res_mobilenet_conf
import cv2

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

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def initialize_model():
    """
    Legacy initializer (kept for compatibility). Prefer get_model().
    """
    return get_model()


def get_model():
    """
    Initialize and load the ResMobileNetV2 model from checkpoint.
    The checkpoint is a dictionary containing model_state_dict, embedding_size, num_classes, etc.
    """
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
    """
    Legacy initializer (kept for compatibility). Prefer get_qdrant_client().
    """
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

    # Check collection exists and has data
    try:
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if QDRANT_COLLECTION not in collection_names:
            print(f"Warning: Collection '{QDRANT_COLLECTION}' does not exist")
            return []
        
        # Check collection info
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


def calculate_orb_score(image_path1: str, image_path2: str, max_features: int = 500) -> float:
    """
    Tính ORB feature matching score giữa 2 ảnh (0-1, càng gần 1 càng giống).
    ORB tốt cho phân biệt logo, chữ, chi tiết pixel.
    """
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0
        
        img1 = cv2.resize(img1, (640, 640))
        img2 = cv2.resize(img2, (640, 640))
        
        orb = cv2.ORB_create(nfeatures=max_features)
        
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) == 0:
            return 0.0
    
        avg_features = (len(kp1) + len(kp2)) / 2.0
        orb_score = len(matches) / max(avg_features, 1.0)
        
        orb_score = min(orb_score, 1.0)
        
        return orb_score
    except Exception as e:
        print(f"Error calculating ORB score: {e}")
        return 0.0

_CLIP_MODEL_CACHE = None
_CLIP_PROCESSOR_CACHE = None


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


def calculate_clip_score(image_path1: str, image_path2: str) -> float:
    """
    Tính CLIP similarity score giữa 2 ảnh (0-1, càng gần 1 càng giống).
    CLIP tốt cho tie-break khi cần phân biệt rất gắt (Coca Zero vs Light vs Nguyên bản).
    """
    model, preprocess = get_clip_model()
    if model is None or preprocess is None:
        print("Warning: CLIP model not available, returning 0.0")
        return 0.0
    
    try:
        import torch
        
        device = next(model.parameters()).device
        
        img1 = Image.open(image_path1).convert('RGB')
        img2 = Image.open(image_path2).convert('RGB')
        
        img1_tensor = preprocess(img1).unsqueeze(0).to(device)
        img2_tensor = preprocess(img2).unsqueeze(0).to(device)
        
        with torch.no_grad():
            img1_features = model.encode_image(img1_tensor)
            img2_features = model.encode_image(img2_tensor)
            
            img1_features = F.normalize(img1_features, p=2, dim=1)
            img2_features = F.normalize(img2_features, p=2, dim=1)
            
            clip_score_raw = (img1_features @ img2_features.T).item()
        
        clip_score = (clip_score_raw + 1) / 2.0
        
        print(f"CLIP: {image_path1} vs {image_path2} -> raw={clip_score_raw:.4f}, normalized={clip_score:.4f}")
        return clip_score
    except Exception as e:
        print(f"Error calculating CLIP score for {image_path1} vs {image_path2}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def search_with_rerank(
    model: nn.Module,
    client: QdrantClient,
    query_image_path: str,
    top_k: int = 20,
    orb_threshold: float = 0.3,
    cosine_weight: float = 0.4,
    orb_weight: float = 0.6,
    use_clip_tiebreak: bool = False,
    clip_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Search với rerank: ArcFace + Qdrant → ORB → CLIP (tie-break)
    
    Pipeline:
    1. ArcFace → query embedding
    2. Qdrant Top-K (cosine similarity từ embedding)
    3. Tính ORB score cho mỗi candidate (feature matching)
    4. Rerank theo combined_score = cosine_weight * cosine + orb_weight * ORB
    5. Filter theo ORB threshold
    6. (Optional) CLIP tie-break cho các case khó phân biệt
    
    Args:
        use_clip_tiebreak: Nếu True, dùng CLIP để tie-break các candidate có score gần nhau
        clip_threshold: Ngưỡng để trigger CLIP tie-break (combined_score >= threshold)
    """
    if client is None:
        raise RuntimeError("Qdrant client not initialized")
    
    query_embedding = get_image_embedding(query_image_path, model)
    
    initial_results = search_similar(model, client, query_image_path, top_k=top_k)
    
    if not initial_results:
        return []
    
    reranked_results = []
    for hit in initial_results:
        candidate_image_path = hit.get('payload', {}).get('image_path')
        if not candidate_image_path or not os.path.exists(candidate_image_path):
            continue
        
        cosine_score = hit.get('score', 0.0)
        
        try:
            orb_score = calculate_orb_score(query_image_path, candidate_image_path)
            
            normalized_cosine = cosine_score if cosine_score >= 0 else (cosine_score + 1) / 2
            
            combined_score = cosine_weight * normalized_cosine + orb_weight * orb_score
            
            result_item = {
                **hit,
                "cosine_score": cosine_score,
                "normalized_cosine": normalized_cosine,
                "orb_score": orb_score,
                "combined_score": combined_score
            }
            if use_clip_tiebreak:
                clip_score = calculate_clip_score(query_image_path, candidate_image_path)
                result_item["clip_score"] = clip_score
                print(f"CLIP score for {candidate_image_path} (combined_score={combined_score:.4f}): {clip_score:.4f}")
                result_item["combined_score"] = 0.3 * normalized_cosine + 0.3 * orb_score + 0.4 * clip_score
            else:
                result_item["clip_score"] = None
            
            reranked_results.append(result_item)
        except Exception as e:
            print(f"Error calculating ORB for {candidate_image_path}: {e}")
            continue
    
    reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    filtered_results = [
        r for r in reranked_results 
        if r['orb_score'] >= orb_threshold
    ]
    
    print(f"Reranked: {len(reranked_results)} candidates, {len(filtered_results)} passed ORB threshold {orb_threshold}")
    
    return filtered_results


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