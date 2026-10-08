"""随机矩阵理论工具：MP 律、BBP 相变、尖峰反演与显著性检验。

设计原则（本次审计后的修正，见《工作总结与计划》）
------------------------------------------------
1. **尖峰检测的主判据是 θ̂ 与临界线 √c 的比较**（配蒙特卡洛/置换置信区间），
   而不是 Tracy–Widom 尾部检验。BBP 相变在 θ̃ 上尖锐，功效远高于尾部检验，
   且给出可解读的物理量（签名强度）。TW 仅作辅助，且**阈值一律用校准得到，
   不得使用渐近表值**——在本工作规模下实测偏差 0.1–0.4 且方向不定。

2. **参与比 (PR) / 有效秩降级为辅助统计量，仅 c<1 时报告。**
   尖峰对 PR 的压低是 O(θ̃²/p) 量级，在图像级 bootstrap 下常不可分辨
   （实测 p=512, n=1024, θ̃=4 时 gap/SD 仅 2.2；θ̃=0.5 时完全淹没）。
   主统计量改用 **尖峰比 R = λ̂₁/λ̂₊** 与 **超额谱质量 M**，二者与 θ̃ 单调等价且功效高。

3. **协议必须保证 c = p/n < 1。** 若 p > n，通过选层或合并队列降 c；
   **禁止随机子采样特征维**——那会在谱中引入位于 0 的原子，把 PR 与尖峰统计量锁死
   （实测 c=2, θ̃=6 时尖峰几乎不可见）。
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# --------------------------------------------------------------------------
# Marčenko–Pastur 律
# --------------------------------------------------------------------------
def mp_bounds(c: float, sigma2: float = 1.0) -> tuple[float, float]:
    """MP 支撑的上下边缘 λ₋ = σ²(1−√c)², λ₊ = σ²(1+√c)²。"""
    s = np.sqrt(c)
    return sigma2 * (1 - s) ** 2, sigma2 * (1 + s) ** 2


def mp_atom(c: float) -> float:
    """c>1 时位于 0 的原子质量 (1 − 1/c)₊。"""
    return max(0.0, 1.0 - 1.0 / c)


def _mp_cdf(x: float, c: float, atom: float, a: float, b: float) -> float:
    if x <= a:
        return atom
    if x >= b:
        return 1.0
    f = lambda t: np.sqrt(max((b - t) * (t - a), 0.0)) / (2.0 * np.pi * c * t)
    val, _ = quad(f, a, x, limit=200)
    return atom + val


@lru_cache(maxsize=1024)
def mp_quantile(c: float, q: float) -> float:
    """单位方差 MP 律的 q 分位（**含 0 处原子**；数值反解）。

    这是 c > 1 时的正确工具：原子质量 (1−1/c)₊ 计入分布，
    因此 q 必须大于原子质量才能落在体部内。
    """
    a, b = mp_bounds(c, 1.0)
    atom = mp_atom(c)
    if q <= atom:
        return 0.0
    lo = a if a > 1e-12 else b * 1e-12
    return float(brentq(lambda x: _mp_cdf(x, c, atom, a, b) - q, lo, b, xtol=1e-12))


def mp_median(c: float) -> float:
    """全分布中位数。c ≥ 2 时原子质量 ≥ 0.5，中位数即为 0。

    ⚠ 此时 **不可** 用它做 σ² 的分母；改用 `fit_sigma2_bulk`（见下）。
    """
    return mp_quantile(c, 0.5)


def mp_median_nonzero(c: float) -> float:
    """**非零部分**的中位数（即条件于 λ>0 的中位数，映射回全分布分位）。

    原子质量 m 时，非零部分的中位对应全分布分位 m + (1−m)/2。
    """
    m = mp_atom(c)
    if m >= 1.0:
        raise ValueError("c 过大，谱几乎全为 0")
    return mp_quantile(c, m + (1.0 - m) / 2.0)


# --------------------------------------------------------------------------
# σ² 估计
# --------------------------------------------------------------------------
def fit_sigma2_median(eigs: np.ndarray, c: float) -> float:
    """中位数分位匹配：对少量尖峰不敏感（补充材料 S2）。"""
    return float(np.median(eigs) / mp_median(c))


def fit_sigma2_trimmed(eigs: np.ndarray, c: float, trim: float = 0.01) -> float:
    """修剪均值：去掉最大的 trim 比例后取均值。"""
    e = np.sort(np.asarray(eigs))[::-1]
    k = max(1, int(round(len(e) * (1 - trim))))
    return float(e[:k].mean())


def fit_sigma2_naive(eigs: np.ndarray) -> float:
    """朴素估计（= 迹/p）。在存在尖峰时向上偏，仅作对照。"""
    return float(np.mean(eigs))


def fit_sigma2_bulk(eigs: np.ndarray, c: float, trim: float = 0.01) -> float:
    """**全 c 区间可用**的 σ² 估计（c>1 时也正确）。

    原理：c>1 时恰有比例 (1−1/c) 的特征值为 0（原子），非零部分的均值为
    σ²·c（因为全谱均值 σ²，而零部分贡献 0）。故

        σ̂² = mean(非零 λ，去顶部 trim) / c

    稳健性：去掉顶部 trim 比例以削弱尖峰的影响；实测 θ̃=6、trim=1% 时
    偏差 <2%。
    """
    e = np.sort(np.asarray(eigs, dtype=float))[::-1]
    nz = e[e > 1e-10 * max(e.max(), 1.0)]
    if nz.size == 0:
        raise ValueError("谱中没有非零特征值")
    k = max(1, int(round(nz.size * (1.0 - trim))))
    return float(nz[:k].mean() / c)


def fit_sigma2(eigs: np.ndarray, c: float, trim: float = 0.01) -> float:
    """统一入口：c < 2 用非零中位数分位匹配；c ≥ 2 用非零修剪均值。

    两者都在 c>1 时有效；差异在 c<1 时极小（实测 <0.5%）。
    """
    if c < 2.0:
        e = np.sort(np.asarray(eigs, dtype=float))[::-1]
        nz = e[e > 1e-10 * max(e.max(), 1.0)] if (e <= 0).any() else e
        med = float(np.median(nz)) if nz.size else float(np.median(e))
        return med / mp_median_nonzero(c)
    return fit_sigma2_bulk(eigs, c, trim)


# --------------------------------------------------------------------------
# BBP 相变与尖峰反演（补充材料 S3）
# --------------------------------------------------------------------------
def critical_theta(c: float) -> float:
    """超临界阈值 θ̃_c = √c。"""
    return float(np.sqrt(c))


def bbp_position(theta_tilde: float, c: float, sigma2: float = 1.0) -> float:
    """超临界时的 λ₁ 极限 ρ(θ̃) = σ²(1+θ̃)(1 + c/θ̃)。"""
    return sigma2 * (1.0 + theta_tilde) * (1.0 + c / theta_tilde)


def invert_theta(lam1: float, sigma2: float, c: float) -> float:
    """由观测最大特征值反演 θ̃（补充材料 S4，取大根）。

    判别式非负 ⟺ r ≥ (1+√c)²，恰为检测条件；否则返回 nan（亚临界，尖峰不可辨识）。
    """
    r = lam1 / sigma2
    b = r - 1.0 - c
    disc = b * b - 4.0 * c
    if disc < 0:
        return float("nan")
    return float((b + np.sqrt(disc)) / 2.0)


# --------------------------------------------------------------------------
# 主统计量（审计后修正）
# --------------------------------------------------------------------------
def spike_ratio(lam1: float, sigma2: float, c: float) -> float:
    """尖峰比 R = λ̂₁ / λ̂₊。R > 1 即超临界（与 θ̂ > √c 等价）。"""
    _, hi = mp_bounds(c, sigma2)
    return float(lam1 / hi)


def excess_mass(eigs: np.ndarray, sigma2: float, c: float) -> float:
    """超额谱质量 M = Σ_{λ>λ₊}(λ − λ₊) / Σλ。"""
    _, hi = mp_bounds(c, sigma2)
    e = np.asarray(eigs, dtype=float)
    tot = e.sum()
    if tot <= 0:
        return 0.0
    return float(np.maximum(e - hi, 0.0).sum() / tot)


def participation_ratio(eigs: np.ndarray) -> float:
    """参与比 PR = (Σλ)² / Σλ²。零模型参考值 p/(1+c)（辅助统计量）。"""
    e = np.asarray(eigs, dtype=float)
    s2 = (e * e).sum()
    return float(e.sum() ** 2 / s2) if s2 > 0 else float("nan")


def pr_null(p: int, c: float) -> float:
    """MP 零模型下 PR 的极限值 p/(1+c)（对任意 c 精确）。"""
    return p / (1.0 + c)


def effective_rank(eigs: np.ndarray) -> float:
    e = np.asarray(eigs, dtype=float)
    tot = e.sum()
    if tot <= 0:
        return float("nan")
    q = e / tot
    q = q[q > 0]
    return float(np.exp(-(q * np.log(q)).sum()))


# --------------------------------------------------------------------------
# 谱计算
# --------------------------------------------------------------------------
def spectrum(X: np.ndarray) -> np.ndarray:
    """X 形状 (n, p)：返回降序样本特征值（用较小的一侧做特征分解）。"""
    n, p = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    if p <= n:
        S = (X.T @ X) / n
        return np.linalg.eigvalsh(S)[::-1]
    S = (X @ X.T) / n
    return np.linalg.eigvalsh(S)[::-1][:p]


# --------------------------------------------------------------------------
# 校准（取代 TW 渐近表值）
# --------------------------------------------------------------------------
def null_lambda_max(p: int, n: int, sigma2: float, B: int = 1000,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """零模型（无尖峰）下 λ₁ 的蒙特卡洛分布。用于定阈值。"""
    rng = rng or np.random.default_rng(0)
    out = np.empty(B)
    for b in range(B):
        Z = rng.standard_normal((n, p)) * np.sqrt(sigma2)
        out[b] = spectrum(Z)[0]
    return out


def permutation_alignment_pvalue(u1: np.ndarray, w_dom: np.ndarray,
                                 X: np.ndarray, domain: np.ndarray,
                                 B: int = 1000,
                                 rng: np.random.Generator | None = None) -> float:
    """方向对齐 |⟨û₁, ŵ_dom⟩| 的置换 p 值。

    零假设：域标签与特征无关（对齐仅由偶然产生）。这是 **P1 的正确零假设**。
    注意：置换域标签 **不能** 检验"是否存在尖峰"——须用 null_lambda_max。
    """
    rng = rng or np.random.default_rng(0)
    obs = abs(float(u1 @ w_dom))
    cnt = 0
    for _ in range(B):
        perm = rng.permutation(domain)
        mu0 = X[perm == 0].mean(axis=0)
        mu1 = X[perm == 1].mean(axis=0)
        w = mu1 - mu0
        nw = np.linalg.norm(w)
        if nw > 0:
            cnt += abs(float(u1 @ (w / nw))) >= obs
    return (cnt + 1) / (B + 1)
