"""FDNet-v2：真正的频率门控。

与旧 FDNet 的差别（修 A4）
-------------------------
旧实现把 LL 与三个细节子带**在第一个卷积层就合并**（12 通道 conv1），
随后 GRL 作用于融合后的特征——因此论文与理论所述"对抗只施加于高频子带"
在代码中**根本不存在**，E8 的"逐子带通路特征"也无从测量。

FDNet-v2 拆成两条独立通路：
    LL 通路  → E_L → f_L   （形态；弱对抗 λ_L，可取 0）
    高频通路 → E_H → f_H   （纹理签名；GRL + 多域判别头，λ_H）

融合用**凸组合**而非拼接
------------------------
拼接后接线性分类器时，门 g 与对应权重可互相抵消，**g 不可辨识**；
凸组合 `z = (1−g)·LN(f_L) + g·LN(f_H)` 使 g 有明确含义（高频通路的信息占比），
可报告其学习轨迹与终值。两个分支各自 LayerNorm 后再门控。

输入分辨率
----------
128×128 输入下 ResNet18 末层仅 4×4，与其它骨干的有效分辨率不匹配，
故 FDNet-v2 单列配置默认 **512 输入** → 子带 256×256（见 v2.8.2）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .layers import ClassHead, DomainHead

# 正交 Haar 滤波器（能量守恒，用于 DWT）
_LL = [[0.5, 0.5], [0.5, 0.5]]
_LH = [[-0.5, -0.5], [0.5, 0.5]]
_HL = [[-0.5, 0.5], [-0.5, 0.5]]
_HH = [[0.5, -0.5], [-0.5, 0.5]]


class HaarDWT2D(nn.Module):
    """2D Haar 小波分解，输出 LL(1ch) 与细节(3ch)。无参数。"""

    def __init__(self, in_ch: int = 1):
        super().__init__()
        k = torch.tensor([_LL, _LH, _HL, _HH], dtype=torch.float32)  # (4,2,2)
        k = k.unsqueeze(1).repeat(in_ch, 1, 1, 1)                    # (4*C,1,2,2)
        self.register_buffer("filters", k)
        self.in_ch = in_ch

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = x.shape[1]
        out = F.conv2d(x, self.filters, stride=2, groups=c)          # (B,4C,H/2,W/2)
        out = out.view(x.shape[0], c, 4, *out.shape[-2:])
        ll = out[:, :, 0]                       # (B,C,h,w)
        hf = out[:, :, 1:].reshape(x.shape[0], c * 3, *out.shape[-2:])
        return ll, hf


def _resnet18_encoder(in_ch: int, pretrained: bool = True) -> tuple[nn.Module, int]:
    """ResNet18 主干，改首层以适配任意输入通道；保留 ImageNet 预训练。

    旧 FDNet 的 12 通道 conv1 只能随机初始化，丢掉了预训练——这是它
    DE 权重仅 0.062 的可能原因之一（修 K4 的另一半）。
    """
    w = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=w)
    old = m.conv1
    new = nn.Conv2d(in_ch, old.out_channels, kernel_size=7, stride=2,
                    padding=3, bias=False)
    if pretrained:
        with torch.no_grad():
            # 单通道：按原权重的通道均值复制；多通道：循环填充
            wk = old.weight.mean(dim=1, keepdim=True)
            for i in range(in_ch):
                new.weight[:, i:i + 1] = wk
    m.conv1 = new
    feat = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                         m.layer1, m.layer2, m.layer3, m.layer4)
    return feat, 512


class FDNetV2(nn.Module):
    """频率门控的 FDNet。"""

    name = "fdnet_v2"

    def __init__(self,
                 n_domains: int,
                 n_classes: int = 2,
                 *,
                 pretrained: bool = True,
                 lambda_l: float = 0.0,
                 hl_only: bool = True,
                 gate_init: float = 0.5,
                 seg_head: bool = False):
        super().__init__()
        self.lambda_l = float(lambda_l)      # LL 通路的对抗强度（0 = 不接 GRL）
        self.hl_only = hl_only               # True: 仅高频接 GRL；False: 双通路（消融）
        self.dwt = HaarDWT2D(in_ch=1)

        self.enc_l, dim = _resnet18_encoder(1, pretrained)
        self.enc_h, dim_h = _resnet18_encoder(3, pretrained)
        assert dim == dim_h
        self.dim = dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ln_l = nn.LayerNorm(dim)
        self.ln_h = nn.LayerNorm(dim)

        # 可学习门：g 越大表示越依赖高频通路
        self.gate_logit = nn.Parameter(torch.tensor(
            float(torch.logit(torch.tensor(gate_init)))))

        self.class_head = ClassHead(dim, n_classes)
        self.domain_head_h = DomainHead(dim, n_domains)   # 高频：始终接 GRL
        self.domain_head_l = (DomainHead(dim, n_domains) if not hl_only else None)

        self.seg_head = (nn.Sequential(
            nn.Conv2d(dim, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)) if seg_head else None)

    # ---- 特征 ----
    def forward_features(self, x: torch.Tensor) -> dict:
        if x.shape[1] != 1:                      # 允许 RGB 输入，转灰度
            x = x.mean(dim=1, keepdim=True)
        ll, hf = self.dwt(x)
        fl_map = self.enc_l(ll)
        fh_map = self.enc_h(hf)
        f_l = torch.flatten(self.pool(fl_map), 1)
        f_h = torch.flatten(self.pool(fh_map), 1)
        return dict(f_l=f_l, f_h=f_h, fl_map=fl_map, fh_map=fh_map)

    def gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logit)

    def fuse(self, f_l: torch.Tensor, f_h: torch.Tensor) -> torch.Tensor:
        g = self.gate()
        return (1.0 - g) * self.ln_l(f_l) + g * self.ln_h(f_h)

    # ---- 前向 ----
    def forward(self, x: torch.Tensor, alpha: float | None = None):
        d = self.forward_features(x)
        z = self.fuse(d["f_l"], d["f_h"])
        cls = self.class_head(z)
        seg = self.seg_head(d["fl_map"]) if self.seg_head is not None else None
        if alpha is None:
            return (cls, seg) if seg is not None else cls

        a_h = alpha
        a_l = alpha * self.lambda_l
        dom = [self.domain_head_h(d["f_h"], a_h)]
        if self.domain_head_l is not None:
            dom.append(self.domain_head_l(d["f_l"], a_l))
        out = (cls, torch.stack(dom, dim=0))
        if seg is not None:
            out = (cls, torch.stack(dom, dim=0), seg)
        return out
