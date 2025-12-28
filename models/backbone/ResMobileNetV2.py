from collections.abc import Sequence
from functools import partial
from typing import Any, Callable, Optional, Tuple, Dict, Union
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .utils.misc import Conv2dNormActivation, SqueezeExcitation
from .utils.utils import _make_divisible
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            return self.gem(x.float(), p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        p_val = self.p.clamp(min=1.0, max=10.0)
        eps_safe = max(eps, 1e-4)  
        x_clamped = x.clamp(min=eps_safe)
        pooled = F.avg_pool2d(x_clamped.pow(p_val), (x.size(-2), x.size(-1)))
        
        return pooled.clamp(min=1e-6).pow(1.0 / p_val)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class HSVColorEncoder(nn.Module):
    def __init__(self, embedding_size: int = 64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(2, 32, kernel_size=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc = nn.Linear(64 * 2, embedding_size)
        self.bn_fc = nn.BatchNorm1d(embedding_size)

    def rgb_to_hs(self, img):
        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        
        max_c, _ = img.max(dim=1)
        min_c, _ = img.min(dim=1)
        diff = max_c - min_c + 1e-6  # Add epsilon to avoid division by zero
        
        s = torch.where(max_c == 0, torch.zeros_like(diff), diff / (max_c + 1e-6))
       
        mask_r = (max_c == r)
        mask_g = (max_c == g)
        mask_b = (max_c == b)
        
        h = torch.zeros_like(max_c)
        # Add epsilon to diff[mask_*] to avoid division by zero
        mask_r_valid = mask_r & (diff > 1e-6)
        mask_g_valid = mask_g & (diff > 1e-6)
        mask_b_valid = mask_b & (diff > 1e-6)
        
        h[mask_r_valid] = (g[mask_r_valid] - b[mask_r_valid]) / (diff[mask_r_valid] + 1e-8)
        h[mask_g_valid] = 2.0 + (b[mask_g_valid] - r[mask_g_valid]) / (diff[mask_g_valid] + 1e-8)
        h[mask_b_valid] = 4.0 + (r[mask_b_valid] - g[mask_b_valid]) / (diff[mask_b_valid] + 1e-8)
        
        h = (h / 6.0) % 1.0
        
        return torch.stack([h, s], dim=1)

    def forward(self, image_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=image_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image_tensor.device).view(1, 3, 1, 1)
        x = image_tensor * std + mean
        x = torch.clamp(x, 0, 1)
        
        x = self.rgb_to_hs(x) # (B, 2, H, W)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) 
        
        mean_feat = torch.mean(x, dim=[2, 3]) 
        std_feat = torch.std(x, dim=[2, 3]) 
        
        x = torch.cat([mean_feat, std_feat], dim=1)
        
        x = self.fc(x)
        x = self.bn_fc(x)
        
        return F.normalize(x, p=2, dim=1)

class AdaptiveSubCenterArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, base_m=0.5, k=3, class_counts=None):
        super().__init__()
        self.s = s
        self.base_s = s  # Lưu base scale để dùng trong warmup
        self.k = k 
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            # Ensure counts are positive and avoid division by zero
            counts = counts.clamp(min=1.0)
            m_list = base_m * (1.0 / torch.pow(counts, 0.25))
            # Normalize to base_m range, avoid division by zero
            m_min = m_list.min()
            if m_min > 1e-6:
                m_list = m_list / m_min * base_m
            else:
                m_list = torch.ones_like(m_list) * base_m
            self.register_buffer('m', m_list)
        else:
            self.register_buffer('m', torch.ones(out_features) * base_m)

    def forward(self, embedding, label):
        # Ensure label is tensor and in valid range
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, device=embedding.device, dtype=torch.long)
        else:
            label = label.long()
        
        # Clamp labels to valid range
        label = torch.clamp(label, 0, self.out_features - 1)
        
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        cosine = cosine.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine, dim=2)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        
        m_batch = self.m[label].view(-1, 1)
        
        # [FIX] Tính sine với numerical stability tốt hơn
        # Đảm bảo giá trị trong sqrt không âm và không quá nhỏ
        sine_sq = (1.0 - torch.pow(cosine, 2)).clamp(min=1e-7, max=1.0)
        sine = torch.sqrt(sine_sq)
        
        # Tính phi với numerical stability
        phi = cosine * torch.cos(m_batch) - sine * torch.sin(m_batch)
        
        # Hard margin condition
        threshold = torch.cos(math.pi - m_batch)
        phi = torch.where(cosine > threshold, phi, cosine - torch.sin(math.pi - m_batch) * m_batch)
        
        # [FIX] Clamp phi để tránh numerical issues trước khi nhân s
        phi = torch.clamp(phi, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # [FIX] Tính output trước, sau đó mới nhân s
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # [FIX] Clamp output cuối cùng trước khi nhân s để tránh logits quá lớn
        # Logits quá lớn sẽ khiến Softmax (trong CrossEntropyLoss) bị tràn số
        output = torch.clamp(output, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        
        # Kiểm tra NaN/Inf trước khi scale
        if not torch.isfinite(output).all():
            # Fallback: chỉ dùng cosine nếu có NaN
            output = torch.where(torch.isfinite(output), output, cosine)
        
        return output * self.s

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
    width_mult: float = 1.0, reduced_tail: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    
    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    inverted_residual_setting = [
        bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  
        
        bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
        
        bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  
        bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
        
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 1, 1),  
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, 2), 
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, 2), 
    ]
    
    last_channel = adjust_channels(1280 // reduce_divider)

    return inverted_residual_setting, last_channel


class ResMobileNetV2(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: list[InvertedResidualConfig],
        embedding_size: int,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        num_classes: int = 1000,
        use_attention: bool = False, 
        use_color_embedding: bool = True,
        color_embedding_size: int = 64,
        store_original_input: bool = True,  
        arcface_s : float = 30,
        class_counts: list = None,
        dropout_rate: float = 0.3,
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

        # self.res_block_5 = Bottleneck(
        #     inplanes=input_channels,
        #     planes=bottleneck_planes,
        #     stride=1,
        #     downsample=None,
        #     norm_layer=norm_layer,
        # )

        # self.res_block_6 = Bottleneck(
        #     inplanes=input_channels,
        #     planes=bottleneck_planes,
        #     stride=1,
        #     downsample=None,
        #     norm_layer=norm_layer,
        # )

        self.gem_pool = GeM(p=3.0) 
        self.color_encoder = HSVColorEncoder(embedding_size=color_embedding_size) if use_color_embedding else None
        self.arcface_head = AdaptiveSubCenterArcFace(embedding_size, num_classes, s=arcface_s, k=3, class_counts=class_counts)

        self.fc_1 = nn.Linear(input_channels, embedding_size)
        self.batch_norm_1 = nn.BatchNorm1d(embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

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
        # x = self.res_block_5(x)
        # x = self.res_block_6(x)
        return x

    def forward(self, x: torch.Tensor, label=None, x_color=None):
        features = self.resnet_stem(x)
        features = self.get_mobile_features(features)
        features = self.resnet_tail(features)
        
        pooled = self.gem_pool(features)
        pooled = torch.flatten(pooled, 1)
        
        x_fc = self.dropout(self.batch_norm_1(self.fc_1(pooled)))
        visual_embedding = F.normalize(x_fc, p=2, dim=1)
        
        img_for_color = x_color if x_color is not None else x
        
        color_emb = self.color_encoder(img_for_color) if self.color_encoder else visual_embedding
        
        if self.training and label is not None:
            if not isinstance(label, torch.Tensor):
                label = torch.tensor(label, device=x.device, dtype=torch.long)
            
            logits = self.arcface_head(visual_embedding, label)
            return logits, visual_embedding, color_emb
        
        return visual_embedding, color_emb

@torch.no_grad()
def get_embeddings(model, data_loader, device):
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