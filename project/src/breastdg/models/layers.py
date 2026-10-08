"""基础层：梯度反转、注意力模块、多域判别头。

与旧实现的差异（见《问题与现状总结》A2）
----------------------------------------
旧代码在两个地方削弱甚至破坏了对抗信号：
  1. 源域与目标域**分两次前向**，且判别头内含 `BatchNorm1d`；
     训练模式下 BN 会各自减去**纯单域 mini-batch** 的均值，
     于是域间的**一阶矩差被抹平**，判别器只剩高阶统计量可用。
  2. 这恰好摧毁了理论命题 2 赖以成立的一阶矩通道（w* ∝ Σ⁻¹(μ_T−μ_S)）。

修正：判别头一律用 **LayerNorm**（逐样本，不跨样本聚合），
训练时源/目标拼成同一 batch 单次前向。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x: torch.Tensor, alpha: float) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


def dann_alpha(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """DANN 的标准 sigmoid ramp（挂在**步数**上，不挂在 epoch 上；修 A1）。"""
    p = min(1.0, step / max(total_steps, 1))
    return float(2.0 / (1.0 + torch.exp(torch.tensor(-gamma * p)).item()) - 1.0)


class DomainHead(nn.Module):
    """多域判别头：LayerNorm（非 BatchNorm）+ 三分类 softmax。

    输出维度 = 源域数 + 1（无标签目标域）。
    """

    def __init__(self, in_dim: int, n_domains: int, hidden: int = 256, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        if alpha is not None:
            x = grad_reverse(x, alpha)
        return self.net(x)


class ClassHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int = 2, hidden: int = 256, p: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------
# 注意力模块
# --------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg, self.max = nn.AdaptiveAvgPool2d(1), nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self, k: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        return self.sig(self.conv(a))


class CBAM(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.ca, self.sa = ChannelAttention(in_planes, ratio), SpatialAttention()

    def forward(self, x):
        return x * self.ca(x) * self.sa(x)
