"""
MobileNetV3 for benchmark comparison with ResMobileNetV2
✅ Same loss (ArcFace + SupCon)
✅ Same interface
✅ Fair comparison (both with/without pretrain)
"""
from typing import Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
except ImportError:
    raise ImportError("torchvision is required for MobileNetV3. Install with: pip install torchvision")

# Import shared components from ResMobileNetV2
from .ResMobileNetV2 import GeM, HSVColorEncoder, AdaptiveSubCenterArcFace


class MobileNetV3(nn.Module):
    """
    MobileNetV3 model for benchmark comparison.
    
    ✅ Same loss: ArcFace (visual) + SupCon (color)
    ✅ Same interface as ResMobileNetV2
    ✅ Fair comparison: both models use same training setup
    
    Args:
        embedding_size: Size of embedding vector (default: 512)
        num_classes: Number of classes for classification
        use_color_embedding: Whether to use color encoder (default: True)
        color_embedding_size: Size of color embedding (default: 64)
        arcface_s: ArcFace scale parameter (default: 30.0)
        class_counts: Class counts for adaptive margin (optional)
        dropout_rate: Dropout rate (default: 0.3)
        model_type: "large" or "small" (default: "large")
        pretrained: Whether to use ImageNet pretrained weights (default: False)
            ⚠️ IMPORTANT: For fair benchmark, set same pretrained value for both models
    """
    
    def __init__(
        self,
        embedding_size: int = 512,
        num_classes: int = 1000,
        use_color_embedding: bool = True,
        color_embedding_size: int = 64,
        arcface_s: float = 30.0,
        class_counts: Optional[list] = None,
        dropout_rate: float = 0.3,
        model_type: str = "large",  # "large" or "small"
        pretrained: bool = False,  # ⚠️ Set same as ResMobileNetV2 for fair comparison
        **kwargs: Any,
    ) -> None:
        super().__init__()
        
        # Load torchvision MobileNetV3 backbone
        if model_type == "large":
            mobilenet = mobilenet_v3_large(pretrained=pretrained)
            # MobileNetV3-Large last conv output: 960 channels
            backbone_features = 960
        elif model_type == "small":
            mobilenet = mobilenet_v3_small(pretrained=pretrained)
            # MobileNetV3-Small last conv output: 576 channels
            backbone_features = 576
        else:
            raise ValueError(f"model_type must be 'large' or 'small', got {model_type}")
        
        # Extract features (remove classifier)
        self.features = mobilenet.features
        
        # GeM pooling (same as ResMobileNetV2)
        self.gem_pool = GeM(p=3.0)
        
        # Color encoder (same as ResMobileNetV2)
        self.color_encoder = HSVColorEncoder(embedding_size=color_embedding_size) if use_color_embedding else None
        
        # ArcFace head (same as ResMobileNetV2)
        self.arcface_head = AdaptiveSubCenterArcFace(
            embedding_size, 
            num_classes, 
            s=arcface_s, 
            k=3, 
            class_counts=class_counts
        )
        
        # Embedding projection
        self.fc_1 = nn.Linear(backbone_features, embedding_size)
        self.batch_norm_1 = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Store config
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.use_color_embedding = use_color_embedding
        self.color_embedding_size = color_embedding_size
        self.model_type = model_type
        self.pretrained = pretrained
        
    def forward(self, x: torch.Tensor, label=None, x_color=None):
        """
        Forward pass - SAME INTERFACE as ResMobileNetV2.
        
        Args:
            x: Input tensor (B, 3, H, W)
            label: Optional labels for training
            x_color: Optional separate input for color encoder
        
        Returns:
            Training mode with label:
                (logits, visual_embedding, color_embedding)
            Eval mode:
                (visual_embedding, color_embedding)
        """
        # Extract features
        features = self.features(x)  # (B, C, H', W')
        
        # GeM pooling
        pooled = self.gem_pool(features)  # (B, C, 1, 1)
        pooled = torch.flatten(pooled, 1)  # (B, C)
        
        # Project to embedding space
        x_fc = self.dropout(self.batch_norm_1(self.fc_1(pooled)))
        visual_embedding = F.normalize(x_fc, p=2, dim=1)
        
        # Color embedding
        img_for_color = x_color if x_color is not None else x
        color_emb = self.color_encoder(img_for_color) if self.color_encoder else visual_embedding
        
        # Training mode with labels
        if self.training and label is not None:
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, device=x.device, dtype=torch.long)
            
            logits = self.arcface_head(visual_embedding, label)
            return logits, visual_embedding, color_emb
        
        # Eval mode
        return visual_embedding, color_emb
    
    def get_features(self, x: torch.Tensor) -> Tensor:
        """Extract features before pooling (for visualization/debugging)."""
        return self.features(x)


def mobilenet_v3_large_config(**kwargs):
    """Factory function for MobileNetV3-Large."""
    return MobileNetV3(model_type="large", **kwargs)


def mobilenet_v3_small_config(**kwargs):
    """Factory function for MobileNetV3-Small."""
    return MobileNetV3(model_type="small", **kwargs)


@torch.no_grad()
def get_embeddings(model, data_loader, device):
    """
    Extract embeddings from model (same interface as ResMobileNetV2).
    
    Args:
        model: MobileNetV3 model
        data_loader: DataLoader
        device: torch device
    
    Returns:
        (visual_embeddings, color_embeddings), labels
    """
    model.eval()
    visual_embeddings_list = []
    color_embeddings_list = []
    labels_list = []
    
    for images, labels in data_loader:
        images = images.to(device)
        visual_emb, color_emb = model(images)
        visual_embeddings_list.append(visual_emb.cpu())
        color_embeddings_list.append(color_emb.cpu())
        labels_list.append(labels.cpu())
    
    visual_embeddings_tensor = torch.cat(visual_embeddings_list, dim=0)
    color_embeddings_tensor = torch.cat(color_embeddings_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)
    
    return (visual_embeddings_tensor.numpy(), color_embeddings_tensor.numpy()), labels_tensor.numpy()


if __name__ == "__main__":
    # Test model
    print("=" * 60)
    print("Testing MobileNetV3 for Benchmark")
    print("=" * 60)
    
    print("\n1️⃣ Testing MobileNetV3-Large (no pretrain)...")
    model_large = MobileNetV3(
        embedding_size=512,
        num_classes=100,
        use_color_embedding=True,
        model_type="large",
        pretrained=False  # ⚠️ Same as ResMobileNetV2 for fair comparison
    )
    
    x = torch.randn(2, 3, 224, 224)
    visual_emb, color_emb = model_large(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Visual embedding shape: {visual_emb.shape}")
    print(f"   Color embedding shape: {color_emb.shape}")
    
    # Test with labels (training mode)
    model_large.train()
    labels = torch.randint(0, 100, (2,))
    logits, visual_emb, color_emb = model_large(x, label=labels)
    print(f"   Logits shape: {logits.shape}")
    print(f"   ✅ MobileNetV3-Large test passed!")
    
    print("\n2️⃣ Testing MobileNetV3-Small (no pretrain)...")
    model_small = MobileNetV3(
        embedding_size=512,
        num_classes=100,
        use_color_embedding=True,
        model_type="small",
        pretrained=False
    )
    
    visual_emb, color_emb = model_small(x)
    print(f"   Visual embedding shape: {visual_emb.shape}")
    print(f"   Color embedding shape: {color_emb.shape}")
    print(f"   ✅ MobileNetV3-Small test passed!")
    
    print("\n3️⃣ Counting parameters...")
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    params_large = count_parameters(model_large)
    params_small = count_parameters(model_small)
    print(f"   MobileNetV3-Large: {params_large:,} params ({params_large/1e6:.2f}M)")
    print(f"   MobileNetV3-Small: {params_small:,} params ({params_small/1e6:.2f}M)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Ready for benchmark.")
    print("=" * 60)
