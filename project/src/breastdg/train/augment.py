"""声学渲染随机化增广（第三层性能杠杆）。

物理依据与边界
--------------
设备的**色调签名**由对数压缩、动态范围、增益/时间增益补偿（TGC）决定，属全局低频操作
（落在 LL 子带）。对抗训练只能抹掉它见过的渲染；若训练时只见到一种渲染，
遇到新渲染（如 OASBUD 的线性包络）就会**系统性反向**。故让模型在训练中见多不怪。

**措辞纪律（v2.8.3）**：源域均为 8 位 B 模式图，无法真正"逆对数压缩再重压"，
本模块是**渲染随机化近似**，论文中不得称"物理重建"。

参数范围 a priori 取自商用设备常见区间（**禁止朝 OASBUD 方向调参**，那是变相泄漏）：
- 动态范围 DR ∈ [40, 80] dB（常见 B 模式设置）
- 伽马 γ ∈ [0.7, 1.4]
- 增益/偏移：分位数枢轴 ±10%
- 斑点抑制强度：轻/中/无
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# —— 事先固定（freeze_log 记录）——
DR_DB_RANGE = (40.0, 80.0)
GAMMA_RANGE = (0.7, 1.4)
PIVOT_JITTER = 0.10
SPECKLE_LEVELS = (0.0, 0.5, 1.0)     # 0=不抑制，1=强抑制
NOMINAL_DR_DB = 60.0                 # 名义参考动态范围（近似用）


def rendering_randomize(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """对单张 8 位灰度图施加随机的渲染链扰动。

    步骤（每一步都对应一个真实的设备设置）：
      1. 按名义 DR 反推"线性包络"近似；
      2. 采样新的 DR，重新做对数压缩；
      3. 采样伽马并施加；
      4. 分位数枢轴抖动（模拟增益/偏移）；
      5. 可选轻度低通（模拟斑点抑制滤波）。
    """
    x = gray.astype(np.float32) / 255.0

    # 1) 名义线性化（把已压缩的 8 位图近似还原到线性）
    lin = np.power(10.0, x * (NOMINAL_DR_DB / 20.0) - NOMINAL_DR_DB / 20.0)
    lin = np.clip(lin, 1e-6, None)

    # 2) 采样新的动态范围并重新压缩
    dr = float(rng.uniform(*DR_DB_RANGE))
    db = 20.0 * np.log10(lin / lin.max() if lin.max() > 0 else 1.0)
    db = np.clip(db, -dr, 0.0)
    x = (db + dr) / dr

    # 3) 伽马
    gamma = float(rng.uniform(*GAMMA_RANGE))
    x = np.power(np.clip(x, 0.0, 1.0), gamma)

    # 4) 增益/偏移（围绕随机枢轴的对比度缩放）
    lo = -float(rng.uniform(0, PIVOT_JITTER))
    hi = 1.0 + float(rng.uniform(0, PIVOT_JITTER))
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    # 5) 斑点抑制近似（各向同性轻度低通）
    level = float(rng.choice(SPECKLE_LEVELS))
    if level > 0:
        h, w = x.shape
        k = int(1 + round(level * 2))            # 1..3
        pad = k // 2
        xp = np.pad(x, pad, mode="edge")
        acc = np.zeros_like(x)
        for i in range(k):
            for j in range(k):
                acc += xp[i:i + h, j:j + w]
        acc /= (k * k)
        x = 0.5 * x + 0.5 * acc                  # 部分抑制，保留纹理

    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def wavelet_amplitude_jitter(gray: np.ndarray, rng: np.random.Generator,
                            sigma: float = 0.15) -> np.ndarray:
    """小波域高频幅度扰动：对 LH/HL/HH 乘随机增益，模拟"纹理签名随机化"。

    与渲染随机化互补——前者打乱二阶（纹理）签名，后者打乱一阶（色调）签名。
    """
    import pywt
    c = pywt.dwt2(gray.astype(np.float32), "haar")
    ll, (lh, hl, hh) = c
    g = 1.0 + rng.normal(0.0, sigma, size=3)
    g = np.clip(g, 0.5, 1.5)
    lh, hl, hh = lh * g[0], hl * g[1], hh * g[2]
    rec = pywt.idwt2((ll, (lh, hl, hh)), "haar")
    h, w = gray.shape
    return np.clip(rec[:h, :w], 0, 255).astype(np.uint8)


# —— 事先固定（裁定 1，2026-10-01）——
SCALE_RANGE = (0.25, 1.0)     # 覆盖观测到的 0.265–0.578（三队列内容缩放比）


def scale_randomize(gray: np.ndarray, rng: np.random.Generator,
                    lo: float = SCALE_RANGE[0], hi: float = SCALE_RANGE[1]
                    ) -> np.ndarray:
    """尺度随机化近似：随机降采样再上采样，改变**斑点在图中的像素尺度**。

    动机（裁定 1）
    -------------
    P4 固定了内容盒子的像素尺寸（长边恒为 225 px），但**映射进盒子的物理视野**
    仍随队列变化，故三队列的内容缩放比差异大（BUS-UCLM 0.265 / BUSI 0.378 /
    BUSBRA 0.578），最终图里的斑点像素尺度相差 2.2 倍。`scale` 单 cue 的域可分性
    达 0.967/0.955，是归因表里最强的单项。

    它在几何上**不可消除**：{恒定填充、恒定物理尺度、不切不扭} 三者不可兼得
    （固定内容短边只是同一颗螺丝换杠杆）。故只能在**增广层**让 scale 与域标签
    在训练分布里去相关——即本函数。

    ⚠ 副作用（须知）：病灶的绝对像素尺寸线索会失效。但该线索跨队列本就不可靠，
    损失可接受；论文措辞为"尺度随机化近似"。

    在 P4 输出画布上施加：填充区为常数 0，双线性不会污染它。
    """
    s = float(rng.uniform(lo, hi))
    if s >= 0.999:
        return gray
    h, w = gray.shape
    small = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    tmp = np.asarray(Image.fromarray(gray).resize(small, Image.BILINEAR))
    back = np.asarray(Image.fromarray(tmp).resize((w, h), Image.BILINEAR))
    return back.astype(np.uint8)
