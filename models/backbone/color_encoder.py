"""
Color encoder using shallow CNN (learnable, not hand-crafted histogram)
Input: RGB image (B, 3, H, W)
Output: 64-d color embedding (B, 64)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ColorEncoder(nn.Module):
    """
    Shallow CNN for color embedding extraction.
    
    Architecture:
    - Conv 1x1 (3 → 16): Learns color mixing
    - Conv 3x3 (16 → 32): Learns local color patterns
    - Global Average Pool: Removes shape, focuses on color
    - FC → 64-d: Final color embedding
    
    All operations are learnable with backpropagation.
    """
    
    def __init__(self, embedding_size: int = 64):
        super().__init__()
        self.embedding_size = embedding_size
        
        # Conv 1x1: Learns color mixing (RGB → color space transformation)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Conv 3x3: Learns local color patterns (e.g., gradients, edges in color space)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Global Average Pooling: Removes spatial information, focuses on color distribution
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC layer to color embedding
        self.fc = nn.Linear(32, embedding_size)
        self.bn_fc = nn.BatchNorm1d(embedding_size)
    
    def forward(self, image_tensor: Tensor) -> Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
        x = image_tensor * std + mean
        x = torch.clamp(x, 0, 1)  # Ensure [0, 1] range
        
        # Conv 1x1: Color mixing
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        
        # Conv 3x3: Local color patterns
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        
        # Global Average Pooling
        x = self.gap(x)  # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 32)
        
        # Final FC layer
        x = self.fc(x)
        x = self.bn_fc(x)
        x = F.relu(x, inplace=True)
        
        # L2 normalize
        color_embedding = F.normalize(x, p=2, dim=1)
        
        return color_embedding
