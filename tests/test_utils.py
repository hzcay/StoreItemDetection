"""Tests for utility functions."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from store_detection.utils.metrics import calculate_iou, calculate_ap


def test_calculate_iou():
    """Test IoU calculation."""
    # Perfect overlap
    box1 = np.array([0, 0, 100, 100])
    box2 = np.array([0, 0, 100, 100])
    assert calculate_iou(box1, box2) == 1.0
    
    # No overlap
    box1 = np.array([0, 0, 50, 50])
    box2 = np.array([100, 100, 150, 150])
    assert calculate_iou(box1, box2) == 0.0
    
    # Partial overlap
    box1 = np.array([0, 0, 100, 100])
    box2 = np.array([50, 50, 150, 150])
    iou = calculate_iou(box1, box2)
    assert 0 < iou < 1


def test_calculate_ap():
    """Test AP calculation."""
    predictions = [
        {'box': np.array([0, 0, 100, 100]), 'score': 0.9},
        {'box': np.array([50, 50, 150, 150]), 'score': 0.8},
    ]
    
    ground_truths = [
        {'box': np.array([0, 0, 100, 100])},
        {'box': np.array([200, 200, 300, 300])},
    ]
    
    ap = calculate_ap(predictions, ground_truths, iou_threshold=0.5)
    assert 0 <= ap <= 1


def test_calculate_ap_empty():
    """Test AP calculation with empty inputs."""
    assert calculate_ap([], [], iou_threshold=0.5) == 0.0
    assert calculate_ap([{'box': np.array([0, 0, 1, 1]), 'score': 0.9}], [], iou_threshold=0.5) == 0.0


if __name__ == '__main__':
    pytest.main([__file__])
