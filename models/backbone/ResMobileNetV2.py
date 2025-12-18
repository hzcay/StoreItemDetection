from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Tuple, Dict, Union
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .utils.misc import Conv2dNormActivation, SqueezeExcitation
from .utils.utils import _make_divisible
from .attention_modules import SpatialAttention, AttentionPooling
from .color_encoder import ColorEncoder

class ArcMarginProduct(nn.Module):
    """
    ArcFace head: L2-norm weight + margin m + scale s.
    Numerically stable version to prevent NaN issues.
    """

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5, easy_margin: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin terms (updated via set_margin for warm-up)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def set_margin(self, m: float) -> None:
        """
        Update ArcFace margin at runtime (e.g., linear warm-up).
        IMPORTANT: must refresh cached trig terms, not just self.m.
        """
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, embedding: Tensor, label: Tensor) -> Tensor:
        # 1. Normalize inputs
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        
        # 2. Clamp cosine để tránh lỗi số học (quan trọng!)
        # float16 có thể làm cosine > 1.0 hoặc < -1.0 một chút xíu -> gây NaN
        cosine = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # 3. Tính Sine an toàn
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # 4. Công thức ArcFace: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # 5. Convert label to one-hot (use same device as cosine)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # 6. Final output
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 
        output *= self.s
        
        return output

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class InvertedResidualConfig:
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SqueezeExcitation, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: list[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        layers.append(
            Conv2dNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


def res_mobilenet_conf(
    width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5

    return inverted_residual_setting, last_channel


class ResMobileNetV2(nn.Module):
    """
    ResNet stem → MobileNet mid → ResNet tail architecture
    
    Advantages:
    - Stronger low-level feature extraction (ResNet 7x7 conv + maxpool)
    - Efficient mid-level processing (MobileNet inverted residuals)
    - Powerful high-level refinement (ResNet bottlenecks)
    """
    def __init__(
        self,
        inverted_residual_setting: list[InvertedResidualConfig],
        embedding_size: int,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_classes: int = 1000,
        use_attention: bool = True,
        use_color_embedding: bool = True,
        color_embedding_size: int = 128,
        store_original_input: bool = True,  # Store original input for color encoder
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        first_mobile_channels = inverted_residual_setting[0].input_channels
        if self.inplanes != first_mobile_channels:
            self.transition = nn.Sequential(
                conv1x1(self.inplanes, first_mobile_channels),
                norm_layer(first_mobile_channels),
            )
        else:
            self.transition = nn.Identity()

        layers: list[nn.Module] = []
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))
        self.mobile_features = nn.Sequential(*layers)

        input_channels = inverted_residual_setting[-1].out_channels
        bottleneck_planes = input_channels // 4

        self.res_block = Bottleneck(
            inplanes=input_channels,
            planes=bottleneck_planes,
            stride=1,
            downsample=None,
            norm_layer=norm_layer,
        )

        self.res_block_2 = Bottleneck(
            inplanes=input_channels,
            planes=bottleneck_planes,
            stride=1,
            downsample=None,
            norm_layer=norm_layer,
        )

        self.res_block_3 = Bottleneck(
            inplanes=input_channels,
            planes=bottleneck_planes,
            stride=1,
            downsample=None,
            norm_layer=norm_layer,
        )

        self.res_block_4 = Bottleneck(
            inplanes=input_channels,
            planes=bottleneck_planes,
            stride=1,
            downsample=None,
            norm_layer=norm_layer,
        )

        self.res_block_5 = Bottleneck(
            inplanes=input_channels,
            planes=bottleneck_planes,
            stride=1,
            downsample=None,
            norm_layer=norm_layer,
        )

        self.res_block_6 = Bottleneck(
            inplanes=input_channels,
            planes=bottleneck_planes,
            stride=1,
            downsample=None,
            norm_layer=norm_layer,
        )

        # Attention modules
        self.use_attention = use_attention
        if use_attention:
            self.spatial_attention = SpatialAttention(kernel_size=7)
            self.attention_pooling = AttentionPooling(input_channels)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Visual embedding head (thêm dropout để giảm overfitting)
        self.fc_1 = nn.Linear(input_channels, embedding_size)
        self.batch_norm_1 = nn.BatchNorm1d(embedding_size)
        self.relu_1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)  # Dropout để giảm overfitting

        # Color embedding (shallow CNN - learnable)
        # ⚠️ QUAN TRỌNG: Color KHÔNG đi vào ArcFace, chỉ dùng cho final embedding (retrieval)
        self.use_color_embedding = use_color_embedding
        if use_color_embedding:
            self.color_encoder = ColorEncoder(embedding_size=color_embedding_size)
            # Color weight for final embedding: α ≈ 0.3-0.5 (tune based on validation)
            self.color_alpha = 0.3  # Can be made learnable or tuned as hyperparameter

        # ArcFace head (CHỈ nhận visual embedding - không phải fused)
        # ArcFace = shape/identity classifier, color = attribute (phân tách rõ ràng)
        self.arcface_head = ArcMarginProduct(
            in_features=embedding_size, out_features=num_classes, s=30.0, m=0.35
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def resnet_stem(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.transition(x)
        return x

    def get_mobile_features(self, x: Tensor) -> Tensor:
        return self.mobile_features(x)

    def resnet_tail(self, x: Tensor) -> Tensor:
        x = self.res_block(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)
        x = self.res_block_5(x)
        x = self.res_block_6(x)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        """
        Forward pass returning VISUAL embedding only (for ArcFace training)
        
        ⚠️ QUAN TRỌNG: 
        - ArcFace CHỈ nhận visual embedding (shape/identity)
        - Color embedding KHÔNG đi vào ArcFace (tránh phá decision boundary)
        - Color chỉ dùng cho final embedding ở inference (retrieval)
        """
        # Visual branch: ResNet stem → MobileNet → ResNet tail
        features = self.resnet_stem(x)
        features = self.get_mobile_features(features)
        features = self.resnet_tail(features)

        # Apply spatial attention (focus on logo regions)
        if self.use_attention:
            features = self.spatial_attention(features)
            # Attention-based pooling instead of GAP
            pooled = self.attention_pooling(features)  # (B, C)
        else:
            pooled = self.avgpool(features)
            pooled = torch.flatten(pooled, 1)  # (B, C)

        # Visual embedding
        x = self.fc_1(pooled)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.dropout(x)  # Apply dropout để giảm overfitting
        visual_embedding = F.normalize(x, p=2, dim=1)  # (B, embedding_size)

        return visual_embedding
    
    def get_final_embedding(self, x: Tensor) -> Tensor:
        """
        Get final embedding for retrieval (visual + color weighted)
        
        This is used at inference time for similarity search.
        Color embedding được weighted với alpha để không làm lệch quá visual.
        
        Returns:
            final_embedding: (B, embedding_size + color_embedding_size) normalized
        """
        original_input = x
        
        # Get visual embedding
        visual_emb = self._forward_impl(x)  # (B, embedding_size)
        
        if self.use_color_embedding:
            color_emb = self.color_encoder(original_input)  # (B, color_embedding_size)
            
            # Weighted concat: visual + color * alpha
            # α = 0.3 nghĩa là color chiếm ~23% contribution (0.3/(1+0.3))
            weighted_color = color_emb * self.color_alpha
            final_emb = torch.cat([visual_emb, weighted_color], dim=1)  # (B, embedding_size + color_embedding_size)
            final_emb = F.normalize(final_emb, p=2, dim=1)
        else:
            final_emb = visual_emb
        
        return final_emb
    
    def _forward_dual(self, x: Tensor, original_input: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass returning both visual and color embeddings (for evaluation/inference)
        
        Args:
            x: Input tensor (B, 3, H, W) - normalized image
            original_input: Original input tensor for color encoder (if None, use x)
        
        Returns:
            visual_embedding: (B, embedding_size)
            color_embedding: (B, color_embedding_size) or None
        """
        # Store original input for color encoder
        if original_input is None:
            original_input = x
        
        # Visual branch: ResNet stem → MobileNet → ResNet tail
        features = self.resnet_stem(x)
        features = self.get_mobile_features(features)
        features = self.resnet_tail(features)

        # Apply spatial attention (focus on logo regions)
        if self.use_attention:
            features = self.spatial_attention(features)
            # Attention-based pooling instead of GAP
            pooled = self.attention_pooling(features)  # (B, C)
        else:
            pooled = self.avgpool(features)
            pooled = torch.flatten(pooled, 1)  # (B, C)

        # Visual embedding
        x = self.fc_1(pooled)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.dropout(x)  # Apply dropout để giảm overfitting
        visual_embedding = F.normalize(x, p=2, dim=1)  # (B, embedding_size)

        # Color embedding (separate branch - uses original input image)
        color_embedding = None
        if self.use_color_embedding:
            color_embedding = self.color_encoder(original_input)  # (B, color_embedding_size)

        return visual_embedding, color_embedding

    def forward(self, x: Tensor, return_color: bool = False) -> Union[Tensor, Tuple[Tensor, Optional[Tensor]]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W) - normalized image
            return_color: If True, return both visual and color embeddings
        
        Returns:
            If return_color=False: visual_embedding (B, embedding_size)
            If return_color=True: (visual_embedding, color_embedding)
        """
        if return_color:
            return self._forward_dual(x, original_input=x)
        else:
            return self._forward_impl(x)

@torch.no_grad()
def get_embeddings(model, data_loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    for images, labels in data_loader:
        images = images.to(device)

        embeddings = model(images)

        embeddings_list.append(embeddings.cpu())
        labels_list.append(labels.cpu())

    embeddings_tensor = torch.cat(embeddings_list, dim=0)
    labels_tensor = torch.cat(labels_list, dim=0)

    return embeddings_tensor.numpy(), labels_tensor.numpy()