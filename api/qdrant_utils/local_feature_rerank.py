import os
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple


def extract_local_features(image_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    h, w = img.shape
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    orb = cv2.ORB_create(nfeatures=5000, scaleFactor=1.2, nlevels=8)
    
    keypoints, descriptors = orb.detectAndCompute(img, None)
    
    if keypoints is None or descriptors is None or len(keypoints) == 0:
        return None
    
    kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    
    return kp_array, descriptors


def match_features(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75
) -> np.ndarray:
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return np.array([])
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append([m.queryIdx, m.trainIdx])
    
    return np.array(good_matches) if good_matches else np.array([])


def compute_homography_ransac(
    kp1: np.ndarray,
    kp2: np.ndarray,
    matches: np.ndarray,
    ransac_threshold: float = 5.0
) -> Tuple[Optional[np.ndarray], int]:
    if len(matches) < 4:
        return None, 0
    
    pts1 = kp1[matches[:, 0]].astype(np.float32)
    pts2 = kp2[matches[:, 1]].astype(np.float32)
    
    H, mask = cv2.findHomography(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.995
    )
    
    num_inliers = int(np.sum(mask)) if mask is not None else 0
    return H, num_inliers


def compute_local_feature_score(
    query_image_path: str,
    candidate_image_path: str
) -> Dict[str, Any]:
    query_features = extract_local_features(query_image_path)
    candidate_features = extract_local_features(candidate_image_path)
    
    if query_features is None or candidate_features is None:
        return {
            "local_score": 0.0,
            "num_matches": 0,
            "num_inliers": 0,
            "num_keypoints_query": 0,
            "num_keypoints_candidate": 0
        }
    
    kp1, desc1 = query_features
    kp2, desc2 = candidate_features
    
    num_kp1 = len(kp1)
    num_kp2 = len(kp2)
    
    if num_kp1 == 0 or num_kp2 == 0:
        return {
            "local_score": 0.0,
            "num_matches": 0,
            "num_inliers": 0,
            "num_keypoints_query": num_kp1,
            "num_keypoints_candidate": num_kp2
        }
    
    good_matches = match_features(desc1, desc2, ratio_threshold=0.8)
    num_matches = len(good_matches)
    
    if num_matches < 4:
        min_kp = min(num_kp1, num_kp2)
        local_score = float(num_matches / min_kp) if min_kp > 0 else 0.0
        return {
            "local_score": min(1.0, local_score),
            "num_matches": num_matches,
            "num_inliers": 0,
            "num_keypoints_query": num_kp1,
            "num_keypoints_candidate": num_kp2
        }
    
    H, num_inliers = compute_homography_ransac(kp1, kp2, good_matches)
    
    min_kp = min(num_kp1, num_kp2)
    
    if num_matches > 0 and min_kp > 0:
        match_ratio = num_matches / min_kp
        
        inlier_ratio = num_inliers / num_matches
        
        local_score = 0.6 * inlier_ratio + 0.4 * match_ratio
    else:
        local_score = 0.0
    
    if num_inliers < 8:
        local_score *= 0.3
    
    if num_matches > 0 and num_inliers / num_matches < 0.2:
        local_score = 0.0
    
    local_score = min(1.0, max(0.0, local_score))
    
    return {
        "local_score": float(local_score),
        "num_matches": int(num_matches),
        "num_inliers": int(num_inliers),
        "num_keypoints_query": int(num_kp1),
        "num_keypoints_candidate": int(num_kp2)
    }