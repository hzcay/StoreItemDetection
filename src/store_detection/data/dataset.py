"""Dataset class for store item detection."""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from torch.utils.data import Dataset


class StoreItemDataset(Dataset):
    """Dataset class for store item detection."""
    
    def __init__(
        self,
        data_dir: str,
        annotation_file: str,
        transform=None,
        image_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            annotation_file: Path to annotation file (COCO format)
            transform: Optional transforms to apply
            image_size: Target image size
        """
        self.data_dir = Path(data_dir)
        self.annotation_file = annotation_file
        self.transform = transform
        self.image_size = image_size
        
        self.images = []
        self.annotations = []
        self._load_annotations()
    
    def _load_annotations(self):
        """Load annotations from file."""
        if not os.path.exists(self.annotation_file):
            print(f"Warning: Annotation file not found: {self.annotation_file}")
            return
        
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        
        # Parse COCO format annotations
        if 'images' in data and 'annotations' in data:
            self.images = data['images']
            self.annotations = data['annotations']
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """
        Get item from dataset.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image, annotations)
        """
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range")
        
        # Load image
        image_info = self.images[idx]
        image_path = self.data_dir / image_info['file_name']
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.image_size)
        
        # Get annotations for this image
        image_annotations = [
            ann for ann in self.annotations 
            if ann['image_id'] == image_info['id']
        ]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, {
            'image_id': image_info['id'],
            'annotations': image_annotations
        }
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        # This would be loaded from the annotation file
        # Placeholder implementation
        return [f"class_{i}" for i in range(10)]
