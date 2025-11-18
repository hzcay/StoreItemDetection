"""Evaluation metrics for object detection."""

import numpy as np
from typing import List, Dict, Tuple


def calculate_iou(
    box1: np.ndarray,
    box2: np.ndarray
) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes.
    
    Args:
        box1: Box 1 in [x1, y1, x2, y2] format
        box2: Box 2 in [x1, y1, x2, y2] format
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_ap(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5
) -> float:
    """
    Calculate Average Precision (AP) for a single class.
    
    Args:
        predictions: List of predictions with 'box', 'score' keys
        ground_truths: List of ground truth boxes
        iou_threshold: IoU threshold for matching
        
    Returns:
        Average Precision
    """
    if len(predictions) == 0:
        return 0.0
    
    if len(ground_truths) == 0:
        return 0.0
    
    # Sort predictions by score
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    # Track which ground truths have been matched
    matched_gt = [False] * len(ground_truths)
    
    tp = []
    fp = []
    
    for pred in predictions:
        pred_box = pred['box']
        max_iou = 0.0
        max_idx = -1
        
        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truths):
            if matched_gt[gt_idx]:
                continue
            
            iou = calculate_iou(pred_box, gt['box'])
            if iou > max_iou:
                max_iou = iou
                max_idx = gt_idx
        
        # Check if match is good enough
        if max_iou >= iou_threshold and max_idx >= 0:
            tp.append(1)
            fp.append(0)
            matched_gt[max_idx] = True
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    recalls = tp / len(ground_truths)
    precisions = tp / (tp + fp)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_metrics(
    all_predictions: Dict[int, List[Dict]],
    all_ground_truths: Dict[int, List[Dict]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate mAP and other metrics.
    
    Args:
        all_predictions: Predictions per class
        all_ground_truths: Ground truths per class
        num_classes: Number of classes
        iou_threshold: IoU threshold
        
    Returns:
        Dictionary with metrics
    """
    aps = []
    
    for class_id in range(num_classes):
        preds = all_predictions.get(class_id, [])
        gts = all_ground_truths.get(class_id, [])
        
        ap = calculate_ap(preds, gts, iou_threshold)
        aps.append(ap)
    
    return {
        'mAP': np.mean(aps),
        'AP_per_class': aps
    }
