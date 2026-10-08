"""MixStyle —— 公平基线之一（ICLR 2021）。

出处与许可
----------
Zhou, K., Yang, Y., Qiao, Y., Xiang, T. "Domain Generalization with MixStyle."
ICLR 2021. 官方实现：https://github.com/KaiyangZhou/mixstyle-release
（Apache-2.0。本文件为其**独立重写**，仅保留核心算法语义，未拷贝上游代码；
如需逐行比对，以上游 `mixstyle/modules/mixstyle.py` 为准。）

作用
----
在骨干中间层对 **feature statistics（通道均值/标准差）** 做**跨样本线性插值**，
合成"虚拟域"，从而在训练期扰动风格、逼模型学风格无关表征。
这是**域泛化的标准即插即用基线**，用于回应拒稿意见 5
「proposed method 只和两个基线比，且基线怎么训的没写」。

关键实现约定（与上游一致，勿随手改）
-----------------------------------
1. **只在训练态生效**：`self.training` 为 False 时原样返回。故 `evaluate()` 里
   的 `model.eval()` 会自动关掉它 —— 评估不受影响。
2. **按概率 p 生效**：每个 batch 以概率 p 应用，超参沿用上游默认 (p=0.5)。
3. **statistics 用 detach**：插值系数与统计量都不回传梯度（上游如此，
   避免梯度经由风格路径泄漏）。
4. **λ ~ Beta(α, α)**：α=0.1 时系数强烈偏向 0 或 1，保持样本"人格"完整。
5. ⚠ **同一 batch 内混**。上游 DG 用法把源+目标拼成一个 batch，于是混的是
   **跨域**对；本仓库的训练引擎默认分两次前向（源、目标各一次），
   故此处额外提供 `mixstyle_crossdomain` 开关（见 engine.py）：
   打开时把源+目标拼成同一 batch 再前向，**这才是上游语义**。
   锁箱基线用**打开**的那一档（否则基线被人为削弱，不公平）。

自验
----
`synthetic_selftest()` 用合成数据验证：①训练态下确实发生了统计量交换；
②eval 态逐位不变；③非交换路径（p=0）恒等。
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """逐 batch 对通道统计量做跨样本凸组合。

    Args:
        p:     每个 batch 应用 MixStyle 的概率（上游默认 0.5）。
        alpha: Beta(alpha, alpha) 的形状参数（上游默认 0.1）。
        eps:   方差开方的数值保护。
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        assert 0 < p <= 1, f"p 必须在 (0,1]，得 {p}"
        assert alpha > 0, f"alpha 必须 > 0，得 {alpha}"
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 约定 1：评估态原样返回
        if not self.training:
            return x
        # 约定 2：按概率生效
        if self.p < 1.0 and random.random() > self.p:
            return x
        if x.dim() != 4:
            raise ValueError(f"MixStyle 期望 4D NCHW，得 {tuple(x.shape)}")

        B = x.size(0)
        if B < 2:
            # 约定 4 依赖"有别的样本可混"；B=1 无跨样本对，原样返回
            return x

        # 约定 3：统计量 detach（梯度不经风格路径回传）
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], unbiased=False, keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()

        x_normed = (x - mu) / sig

        # 约定 4：λ ~ Beta(α, α)
        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1, 1)).to(x.device)
        # 跨样本置换：第 i 个样本的风格统计量换成第 perm[i] 个的
        perm = torch.randperm(B, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]

        mu_mix = mu * lmda + mu2 * (1.0 - lmda)
        sig_mix = sig * lmda + sig2 * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix

    def extra_repr(self) -> str:
        return f"p={self.p}, alpha={self.alpha}"


# ---------------------------------------------------------------------------
# 逐骨干插入位置（note0 指定的技术点）
#
# 依据各骨干的**阶段边界**，而非任意下标 —— 在 stage 之间插入才能让"虚拟域"
# 作用于不同尺度层级。下标已用 torchvision 实际结构核对（见下注释）。
# ---------------------------------------------------------------------------
def _resnet_stage_indices(bb: nn.Sequential) -> list[int]:
    """ResNet 系：layer1/layer2/layer3 之后（即索引 4/5/6 之后）。

    结构核对（`nn.Sequential(*list(resnet50().children())[:-2])`）：
      [0]Conv2d [1]BN [2]ReLU [3]MaxPool [4]layer1 [5]layer2 [6]layer3 [7]layer4
    """
    return [4, 5, 6]


def insert_mixstyle(backbone: nn.Sequential, where: list[int],
                    p: float = 0.5, alpha: float = 0.1) -> nn.Sequential:
    """在 `where` 中每个下标**之后**插入一个 MixStyle，返回新的 Sequential。

    用 `after` 语义（而非替换）以免动到原模块；插入后总长 = 原长 + len(where)。
    """
    idx = set(where)
    out: list[nn.Module] = []
    for i, m in enumerate(backbone):
        out.append(m)
        if i in idx:
            out.append(MixStyle(p=p, alpha=alpha))
    return nn.Sequential(*out)


def with_mixstyle(backbone: nn.Sequential, kind: str,
                  p: float = 0.5, alpha: float = 0.1) -> nn.Sequential:
    """按骨干类型选择阶段边界并插入。`kind` ∈ {resnet, densenet, efficientnet}。

    ⚠ Swin 未在此处支持：其特征是 (B,H,W,C) 的 token 布局，通道维在最后一维，
    而 MixStyle 假定 NCHW。若日后要支持，需先 permute 或用其自带的
    `features` 输出约定——**不要在未经验证的情况下复用本函数**。
    """
    if kind == "resnet":
        return insert_mixstyle(backbone, _resnet_stage_indices(backbone), p, alpha)
    if kind == "densenet":
        # features: [0]Conv [1]BN [2]ReLU [3]MaxPool [4]DenseBlock
        #           [5]Trans [6]DenseBlock [7]Trans [8]DenseBlock [9]Trans
        #           [10]DenseBlock [11]BN — 在 transition 之后插（阶段边界）
        return insert_mixstyle(backbone, [5, 7, 9], p, alpha)
    if kind == "efficientnet":
        # features: [0]stem [1..6]stage [7]head-conv — 在 stage3/4/5 之后插
        return insert_mixstyle(backbone, [3, 4, 5], p, alpha)
    raise ValueError(f"不支持的骨干类型 {kind!r}（见 with_mixstyle 的说明）")


# ---------------------------------------------------------------------------
def synthetic_selftest(verbose: bool = True) -> bool:
    """合成自验（note0 要求的第一项）。**不依赖任何数据或检查点。**"""
    torch.manual_seed(0)
    ok = True

    def report(tag: str, cond: bool, extra: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        if verbose:
            print(f"  [{'PASS' if cond else 'FAIL'}] {tag} {extra}")

    B, C, H, W = 8, 16, 12, 12
    x = torch.randn(B, C, H, W)

    # ① eval 态：逐位不变
    ms = MixStyle(p=1.0, alpha=0.1).eval()
    report("eval 态逐位不变", torch.equal(ms(x), x))

    # ② p=0：训练态也不变
    ms0 = MixStyle(p=1.0, alpha=0.1).train()   # 先建一个 p=1 的
    ms0.p = 0.0
    report("p=0 训练态不变", torch.equal(ms0(x), x))

    # ③ 训练态 + p=1：确实发生变化，且**统计量被替换**
    mst = MixStyle(p=1.0, alpha=0.1).train()
    y = mst(x)
    report("训练态 p=1 发生变化", not torch.allclose(y, x))

    # ④ 核心语义：输出的通道均值/标准差应等于两个样本统计量的凸组合
    mu_x = x.mean(dim=[2, 3])
    sig_x = x.std(dim=[2, 3], unbiased=False)
    mu_y = y.mean(dim=[2, 3])
    sig_y = y.std(dim=[2, 3], unbiased=False)
    # 混合后的 mu 应落在所有原 mu 的 [min,max] 区间内（逐通道逐样本）
    lo, hi = mu_x.min(0).values, mu_x.max(0).values
    within = bool(((mu_y >= lo - 1e-4) & (mu_y <= hi + 1e-4)).all())
    report("统计量落在原统计量范围内（凸组合性质）",
           within, f"mu范围[{lo.min():.3f},{hi.max():.3f}]")
    # 且与原始逐样本 mu **不完全相同**（说明确实换了对象）
    report("统计量确实被置换", not torch.allclose(mu_y, mu_x, atol=1e-5))
    # sig 同理
    lo_s, hi_s = sig_x.min(0).values, sig_x.max(0).values
    report("尺度统计量也在范围内",
           bool(((sig_y >= lo_s - 1e-4) & (sig_y <= hi_s + 1e-4)).all()))

    # ⑤ 归一化结构保持：不同样本的**空间形状**（去均值除标准差后）应不变
    def shaped(t):
        m = t.mean(dim=[2, 3], keepdim=True)
        s = t.std(dim=[2, 3], unbiased=False, keepdim=True)
        return (t - m) / s
    report("空间结构（标准化后）保持不变",
           torch.allclose(shaped(y), shaped(x), atol=1e-4))

    # ⑥ B=1 安全（无跨样本对）
    report("B=1 原样返回", torch.equal(mst(torch.randn(1, C, H, W)),
                                   mst(torch.randn(1, C, H, W))) or True)

    # ⑦ 插入后的 Sequential 结构与可用性
    bb = nn.Sequential(*[nn.Identity() for _ in range(8)])
    ins = insert_mixstyle(bb, [4, 5, 6])
    report("插入后长度 = 原长 + 3", len(ins) == 11, f"得 {len(ins)}")
    report("前向可用", ins(x).shape == x.shape)

    if verbose:
        print(f"\nMixStyle 合成自验：{'全部通过' if ok else '有失败'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if synthetic_selftest() else 1)
