"""Visualization utilities for store item detection."""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pathlib import Path


def visualize_predictions(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: List[str],
    score_threshold: float = 0.5,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Visualize detection predictions on image.
    
    Args:
        image: Input image (H, W, C) in RGB format
        boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format
        labels: Class labels (N,)
        scores: Confidence scores (N,)
        class_names: List of class names
        score_threshold: Minimum score to display
        save_path: Optional path to save visualization
        
    Returns:
        Annotated image
    """
    # Create copy of image
    vis_image = image.copy()
    
    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Draw boxes and labels
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        color = _get_color(int(label))
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label and score
        class_name = class_names[int(label)] if int(label) < len(class_names) else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        # Draw label background
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            vis_image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            vis_image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    return vis_image


def _get_color(class_id: int) -> Tuple[int, int, int]:
    """
    Get color for class ID.
    
    Args:
        class_id: Class identifier
        
    Returns:
        RGB color tuple
    """
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot mAP
    if 'val_map' in history:
        axes[1].plot(history['val_map'], label='Val mAP')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP')
        axes[1].set_title('Validation mAP')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
