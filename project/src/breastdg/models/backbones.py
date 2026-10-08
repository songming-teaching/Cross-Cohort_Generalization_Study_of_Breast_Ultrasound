"""四个空间骨干：各自带 GRL + 多域判别头。

改造要点
--------
- 判别头用 LayerNorm（修 A2），且**不再分两次前向**（由训练循环拼接 batch 保证）；
- 域判别为 **三分类**（BUSI / BUSBRA / BUS-UCLM），与"每台设备一个签名方向"的
  理论假设 (A2) 对齐；
- 特征提取的倒数第二层输出统一暴露为 `forward_features()`，
  供 E8 谱分析直接取用（理论文档要求"逐子带的倒数第二层特征"）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .layers import CBAM, ClassHead, DomainHead, SEBlock


class _BaseDANN(nn.Module):
    feature_dim: int

    def __init__(self, feature_dim: int, n_domains: int, n_classes: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.class_head = ClassHead(feature_dim, n_classes)
        self.domain_head = DomainHead(feature_dim, n_domains)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x: torch.Tensor, alpha: float | None = None):
        f = self.forward_features(x)
        cls = self.class_head(f)
        if alpha is None:
            return cls
        return cls, self.domain_head(f, alpha)


# --------------------------------------------------------------------------
class SwinDANN(_BaseDANN):
    name = "swin_t"

    def __init__(self, n_domains: int, n_classes: int = 2, pretrained: bool = True):
        super().__init__(768, n_domains, n_classes)
        w = models.Swin_T_Weights.DEFAULT if pretrained else None
        m = models.swin_t(weights=w)
        self.feature_dim = m.head.in_features
        m.head = nn.Identity()
        self.backbone = m
        # 重建 head 以匹配 feature_dim
        self.class_head = ClassHead(self.feature_dim, n_classes)
        self.domain_head = DomainHead(self.feature_dim, n_domains)

    def forward_features(self, x):
        return self.backbone(x)


# --------------------------------------------------------------------------
class DenseNetSE_DANN(_BaseDANN):
    name = "densenet121_se"

    def __init__(self, n_domains: int, n_classes: int = 2, pretrained: bool = True):
        super().__init__(1024, n_domains, n_classes)
        w = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.backbone = models.densenet121(weights=w).features
        self.se = SEBlock(1024)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        return torch.flatten(self.pool(self.se(self.backbone(x))), 1)


# --------------------------------------------------------------------------
class ResNetCBAM_DANN(_BaseDANN):
    name = "resnet50_cbam"

    def __init__(self, n_domains: int, n_classes: int = 2, pretrained: bool = True):
        super().__init__(2048, n_domains, n_classes)
        w = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=w)
        self.backbone = nn.Sequential(*list(m.children())[:-2])
        self.cbam = CBAM(2048)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        return torch.flatten(self.pool(self.cbam(self.backbone(x))), 1)


# --------------------------------------------------------------------------
class EfficientNetV2_DANN(_BaseDANN):
    name = "efficientnet_v2_s"

    def __init__(self, n_domains: int, n_classes: int = 2, pretrained: bool = True):
        super().__init__(1280, n_domains, n_classes)
        w = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=w).features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        return torch.flatten(self.pool(self.backbone(x)), 1)


REGISTRY = {
    SwinDANN.name: SwinDANN,
    DenseNetSE_DANN.name: DenseNetSE_DANN,
    ResNetCBAM_DANN.name: ResNetCBAM_DANN,
    EfficientNetV2_DANN.name: EfficientNetV2_DANN,
}

ALL_NAMES = list(REGISTRY)
