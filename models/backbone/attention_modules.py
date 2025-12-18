"""
Attention modules for focus on logo regions and attention-based pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialAttention(nn.Module):
    """
    Spatial Attention module to focus on important regions (e.g., logo)
    Works on feature maps to highlight spatial locations
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, 
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        
        Returns:
            attended: (B, C, H, W) attended feature map
        """
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and apply conv
        attention = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.conv(attention)  # (B, 1, H, W)
        attention = self.sigmoid(attention)  # (B, 1, H, W)
        
        # Apply attention
        attended = x * attention
        return attended


class AttentionPooling(nn.Module):
    """
    Attention-based pooling instead of Global Average Pooling
    Learns to weight spatial locations adaptively
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) feature map
        
        Returns:
            pooled: (B, C) pooled feature vector
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (B, 1, H, W)
        
        # Weighted sum
        pooled = (x * attn_weights).sum(dim=(2, 3))  # (B, C)
        
        return pooled

