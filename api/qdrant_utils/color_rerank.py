import os
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from sklearn.cluster import KMeans
from PIL import Image


def preprocess_image_for_color(image_path: str, target_size: int = 300) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = img.shape[:2]
    if max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img


def extract_dominant_colors(image_path: str, k: int = 3, color_space: str = "HSV") -> np.ndarray:
    img = preprocess_image_for_color(image_path)
    
    if color_space == "HSV":
        img_colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif color_space == "LAB":
        img_colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    else:
        img_colorspace = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    pixels = img_colorspace.reshape(-1, 3).astype(np.float32)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    dominant_colors = kmeans.cluster_centers_.astype(np.uint8)
    
    return dominant_colors


def color_distance(color1: np.ndarray, color2: np.ndarray, color_space: str = "HSV") -> float:
    if color_space == "HSV":
        h_diff = min(abs(int(color1[0]) - int(color2[0])), 180 - abs(int(color1[0]) - int(color2[0])))
        s_diff = abs(int(color1[1]) - int(color2[1]))
        v_diff = abs(int(color1[2]) - int(color2[2]))
        
        distance = np.sqrt(
            (h_diff / 180.0) ** 2 * 0.3 +  # H less important
            (s_diff / 255.0) ** 2 * 0.35 +  # S important
            (v_diff / 255.0) ** 2 * 0.35    # V important
        )
    else:
        distance = np.linalg.norm(color1.astype(float) - color2.astype(float))
        distance = distance / (100.0 * np.sqrt(3))
    
    return distance


def compute_color_score(
    query_image_path: str,
    candidate_image_path: str,
    color_space: str = "HSV",
    k_colors: int = 3
) -> Dict[str, Any]:
    try:
        query_colors = extract_dominant_colors(query_image_path, k=k_colors, color_space=color_space)
        candidate_colors = extract_dominant_colors(candidate_image_path, k=k_colors, color_space=color_space)
    except Exception as e:
        print(f"Error extracting colors: {e}")
        return {
            "color_score": 0.5,  # Neutral score on error
            "query_colors": [],
            "candidate_colors": []
        }
    
    total_distance = 0.0
    matched_pairs = []
    
    used_candidate_indices = set()
    
    for q_color in query_colors:
        best_distance = float('inf')
        best_candidate_idx = -1
        
        for idx, c_color in enumerate(candidate_colors):
            if idx in used_candidate_indices:
                continue
            
            dist = color_distance(q_color, c_color, color_space)
            if dist < best_distance:
                best_distance = dist
                best_candidate_idx = idx
        
        if best_candidate_idx >= 0:
            matched_pairs.append((q_color, candidate_colors[best_candidate_idx], best_distance))
            used_candidate_indices.add(best_candidate_idx)
            total_distance += best_distance
    
    if len(matched_pairs) > 0:
        avg_distance = total_distance / len(matched_pairs)
        color_score = max(0.0, 1.0 - avg_distance)
    else:
        color_score = 0.0
    
    return {
        "color_score": float(color_score),
        "query_colors": query_colors.tolist(),
        "candidate_colors": candidate_colors.tolist()
    }
