#!/usr/bin/env python3
"""E8 谱分析脚本：设备签名尖峰的校准检测（理论子节 S8 协议的工程实现）。

依赖：numpy / scipy  only。与训练代码完全解耦：只读 §5 接口的 npz 特征文件。

冻结口径（freeze_log F10 + 2026-09-30 补遗 v2，勿改）：
  1. **主检出统计量为 R = λ̂₁/λ̂₊**（零分布紧致且不随 c 漂移，实测各 c 下 95% 分位
     1.00–1.03、永不为 NaN）；θ̂ 为**效应量**（未检出=NaN 映为 0，绝不丢弃后取均值——
     丢弃会人为抬高估计），附 cluster bootstrap CI；M 为佐证。一切阈值按实际 (p, n)
     蒙特卡洛校准；√c 仅为理论参考线，不作判定（零模型 θ̂ 的 95% 分位高于 √c 达
     1.24–1.53×，直接比较会把假阳性抬到 20–30%）。
  2. TW 表值禁用；TW 统计量仅辅助，阈值同样由校准产生。
  3. PR / er 仅当 c<1（按图像级 n）时报告；c>1 时零原子将其锁死。
  4. c>1 允许：零原子不影响上边缘统计量（实测 c=2, θ̃=6 检出率 100%）。
     **禁止泛型降维**（块平均/随机子采样/随机投影）：以 p'/p 稀释 θ̃
     （实测块平均 512→128 使 θ̂ 由 3.0 跌至 0.76）。
     **例外**：源域拟合 PCA（无泄漏）可作敏感性臂，但必须过保真门——
     在 BUS-UCLM 上原生 vs 投影 θ̂ 比值 ∈ [0.8, 1.25] 方允许用于锁箱队列
     （依据：逐域签名模型 A2 下 u_T≠u_S，源 PCA 对正交/去局域目标签名
     仅捕获 ~31%，θ̃=4 → 1.25，低于检出下限）。
  5. σ² 估计用**条件中位数匹配**（对非零部分），原 rmt.py 的 mp_median 在 c≥2 时
     返回 0 导致 σ̂²=inf（已修复，见 fit_sigma2）。
  6. OASBUD（及任何多切面队列）的 bootstrap / 置换 / 校准一律按病灶 cluster，
     校准用保守有效 n（病灶数）。
  7. 每次 deflation 后重新拟合 σ²；多重比较 BH 校正。
  8. θ̂ 用朴素反演（B=500 实测偏差 <1%，无需校正）；但 c>1 时分布左偏、
     sd≈0.7，**必须报 bootstrap CI**，且任何模拟结论须用足够重复数
     （B≥500）——小样本中位数噪声曾导致"偏差 7–9%"的误报。
     零模型下 θ̂ 有 70–93% 为 NaN（临界条件恰在零模型期望位置）：
     NaN="未检出"即为正确读数，映为 0 参与校准分布。

用法：
  python e8_rmt_analysis_v2.py selftest                # 合成数据全套协议自验（含 PCA 门）
  python e8_rmt_analysis_v2.py run FEATURES.npz [--out row.csv] [--conservative-n 100]
  python e8_rmt_analysis_v2.py run FEATURES.npz --project-from SOURCE.npz --project-k 160
  python e8_rmt_analysis_v2.py gate FEATURES.npz --project-from SOURCE.npz --project-k 160
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Marčenko–Pastur 律
# ---------------------------------------------------------------------------

def mp_bounds(c: float, sigma2: float = 1.0) -> tuple[float, float]:
    s = np.sqrt(c)
    return sigma2 * (1 - s) ** 2, sigma2 * (1 + s) ** 2


def mp_atom(c: float) -> float:
    return max(0.0, 1.0 - 1.0 / c)


def _mp_cdf(x: float, c: float) -> float:
    a, b = mp_bounds(c, 1.0)
    if x <= a:
        return mp_atom(c)
    if x >= b:
        return 1.0
    f = lambda t: np.sqrt(max((b - t) * (t - a), 0.0)) / (2.0 * np.pi * c * t)
    val, _ = quad(f, a, x, limit=200)
    return mp_atom(c) + val


@lru_cache(maxsize=1024)
def mp_median_nonzero(c: float) -> float:
    """MP_c 连续（非零）部分的条件中位数——σ² 中位数匹配的正确基准。

    c ≤ 1 时与无条件中位数一致；c > 1 时跳过位于 0 的原子。
    （修复 rmt.py mp_median 在 c ≥ 2 返回 0 的 bug。）
    """
    c = float(c)
    target = mp_atom(c) + 0.5 * min(1.0, 1.0 / c)
    a, b = mp_bounds(c, 1.0)
    lo = a if a > 0 else b * 1e-12
    return float(brentq(lambda x: _mp_cdf(x, c) - target, lo, b, xtol=1e-10))


# ---------------------------------------------------------------------------
# 谱与 σ² 估计
# ---------------------------------------------------------------------------

def spectrum(X: np.ndarray) -> np.ndarray:
    """X 形状 (n, p)；返回降序特征值，长度恒为 p（p>n 时尾部补零）。"""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    X = X - X.mean(axis=0, keepdims=True)
    if p <= n:
        S = (X.T @ X) / n
        return np.linalg.eigvalsh(S)[::-1]
    S = (X @ X.T) / n
    eigs = np.linalg.eigvalsh(S)[::-1]
    return np.concatenate([eigs, np.zeros(p - n)])


def fit_sigma2(eigs: np.ndarray, n: int, p: int) -> float:
    """条件中位数匹配：取前 min(n, p) 个（非零部分）特征值的中位数。"""
    k = min(n, p)
    top = np.sort(np.asarray(eigs, dtype=float))[::-1][:k]
    return float(np.median(top) / mp_median_nonzero(p / n))


# ---------------------------------------------------------------------------
# 尖峰统计量（S3–S6）
# ---------------------------------------------------------------------------

def theta_inversion(lam1: float, sigma2: float, c: float) -> float:
    """(S4) 反演，取大根；判别式为负（亚临界）返回 0.0（与校准分布兼容）。"""
    r = lam1 / sigma2
    b = r - 1.0 - c
    disc = b * b - 4.0 * c
    if disc < 0:
        return 0.0
    return float((b + np.sqrt(disc)) / 2.0)


def spike_ratio(lam1: float, sigma2: float, c: float) -> float:
    return float(lam1 / mp_bounds(c, sigma2)[1])


def excess_mass(eigs: np.ndarray, sigma2: float, c: float) -> float:
    hi = mp_bounds(c, sigma2)[1]
    e = np.asarray(eigs, dtype=float)
    tot = e.sum()
    return float(np.maximum(e - hi, 0.0).sum() / tot) if tot > 0 else 0.0


def tw_statistic(lam1: float, sigma2: float, n: int, p: int) -> float:
    """Johnstone 归一化 (S12–S13)。仅辅助，阈值由校准给出，禁用表值。"""
    mu = sigma2 / n * (np.sqrt(n - 1) + np.sqrt(p)) ** 2
    nu = (sigma2 / n * (np.sqrt(n - 1) + np.sqrt(p))
          * (1.0 / np.sqrt(n - 1) + 1.0 / np.sqrt(p)) ** (1.0 / 3.0))
    return float((lam1 - mu) / nu)


def participation_ratio(eigs: np.ndarray) -> float:
    e = np.asarray(eigs, dtype=float)
    s2 = (e * e).sum()
    return float(e.sum() ** 2 / s2) if s2 > 0 else float("nan")


def pr_null(p: int, c: float) -> float:
    return p / (1.0 + c)


def effective_rank(eigs: np.ndarray) -> float:
    e = np.asarray(eigs, dtype=float)
    tot = e.sum()
    if tot <= 0:
        return float("nan")
    q = e[e > 0] / tot
    return float(np.exp(-(q * np.log(q)).sum()))


# ---------------------------------------------------------------------------
# F34 主表征：零模型自由量（不依赖体部形状假设）
# ---------------------------------------------------------------------------
# 裁定依据（补遗 P-7，2026-10-02；验证见 工作日志/2026-10-05_主工作日志.md R-1）：
# 体部非 MP——零原子与 c>1 预测精确吻合（0.693/0.084），但非零体部重尾
# （0.99 分位 obs/MP = 9.74/8.09）。故一切以白体部零模型为基础的**绝对校准量**
# （R、θ̂、K̂）撤回，改以下列零模型自由量为主表征：
#   PR        参与率 = (Σλ)²/Σλ² ；零模型期望 PR_null = np/(n+p)，解析式、无体部依赖
#   PR_frac   PR/PR_null —— 归一化后跨 p 可比，是主统计量
#   er        有效秩 = exp(熵)；零模型下同样有稳定值
#   tk_frac   前 k 占迹比（k 预先冻结，见 TK_GRID）
#
# 为什么 c≥1 也报告（推翻 v2 的 c<1 守卫）：零特征值对 PR 与 er 均无贡献，
# 故"补齐到长度 p 的谱"与"仅非零谱"给出**逐位相同**的结果（实测，R-1 第 1 条）；
# `spectrum()` 本就补齐到长度 p，守卫在数学上无必要——它是从 R/θ̂ 校准框架
# 继承的保守做法，而 R/θ̂ 正是 F34 撤回的量，方向相反。解除后 p>n 的格子
# （大骨干 + 小队列）才有读数，全表可读。

# 前 k 占迹比的冻结取值（a priori，禁止按结果挑选；全报告，不作选择）
TK_GRID: tuple[int, ...] = (1, 5, 16, 32, 64)


def pr_null_frac(p: int, n: int) -> float:
    """零模型（白体部、无尖峰）下 PR 的解析期望 = p/(1+c) = np/(n+p)。

    实测偏差 −0.05% ~ −0.47%（c ∈ [0.81, 5.85]，B=30），见 R-1。
    """
    return float(p * n / (p + n))


def er_null_frac(p: int, n: int) -> float:
    """零模型下有效秩的期望（MP 律无解析式，用固定种子的 MC 估计）。

    主统计量用 PR_frac（有解析式 np/(n+p)）；er_frac 为归一化展示量。
    B=20 时 MC 标准误 <1%（自验 T11b）。
    """
    vals = [effective_rank(spectrum(np.random.default_rng(9000 + b)
                                    .standard_normal((n, p))))
            for b in range(20)]
    return float(np.mean(vals))


def topk_trace_frac(eigs: np.ndarray, ks: tuple[int, ...] = TK_GRID) -> dict:
    """前 k 占迹比（k 预先冻结）。零模型自由——不假设体部形状。"""
    e = np.asarray(eigs, dtype=float)
    tot = e.sum()
    if tot <= 0:
        return {f"tk{k}": float("nan") for k in ks}
    out = {}
    for k in ks:
        kk = min(k, len(e))
        out[f"tk{k}"] = float(e[:kk].sum() / tot)
    return out


def cluster_stats(eigs: np.ndarray, p: int, n: int) -> dict:
    """MP 边缘以上簇的规模与方差占比（K̂ 带"白体部零模型依赖"限定，见 v13 规格）。

    K̂ **不得写作"尖峰数"**——它依赖白体部零模型，而 F34 已证体部非 MP，
    故只作描述性统计量、不作检出判据。簇方差占比 = 边缘以上特征值占迹比，
    这个量与体部模型无关，可放心报告。
    """
    e = np.asarray(eigs, dtype=float)
    s2 = fit_sigma2(e, n, p)
    lam_plus = mp_bounds(p / n, s2)[1]
    k_above = int((e > lam_plus).sum())
    tot = e.sum()
    frac = float(e[:k_above].sum() / tot) if (k_above > 0 and tot > 0) else 0.0
    return {"K_above": k_above, "K_above_frac": frac, "lambda_plus_edge": float(lam_plus)}


# ---------------------------------------------------------------------------
# 校准：零模型蒙特卡洛（F10 补遗条款 1：一切阈值按实际 (p, n) 校准产生）
# ---------------------------------------------------------------------------

def mc_null(p: int, n: int, B: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """零模型（无尖峰，σ²=1；各统计量均尺度不变）下 R / M / θ̂ / TW 的经验分布。"""
    rng = np.random.default_rng(seed)
    c = p / n
    out = {k: np.empty(B) for k in ("R", "M", "theta", "TW")}
    for b in range(B):
        eigs = spectrum(rng.standard_normal((n, p)))
        s2 = fit_sigma2(eigs, n, p)
        out["R"][b] = spike_ratio(eigs[0], s2, c)
        out["M"][b] = excess_mass(eigs, s2, c)
        out["theta"][b] = theta_inversion(eigs[0], s2, c)
        out["TW"][b] = tw_statistic(eigs[0], s2, n, p)
    return out


def calibrated_thresholds(null: dict[str, np.ndarray], alpha: float = 0.05
                          ) -> dict[str, float]:
    return {k: float(np.quantile(v, 1 - alpha)) for k, v in null.items()}


def power_floor(p: int, n: int, alpha: float = 0.05, target: float = 0.8,
                grid=(1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24), B_null: int = 300,
                B_alt: int = 150, seed: int = 99) -> float:
    """该 (p, n) 下达到 target 功效所需的最小 θ̃（Table S2 功效列）。

    用本地零模型阈值 + 尖峰模型模拟估计；grid 上线性扫描，返回首个达标点，
    全部不达标返回 inf。
    """
    thr = calibrated_thresholds(mc_null(p, n, B_null, seed), alpha)["R"]
    rng = np.random.default_rng(seed + 1)
    c = p / n
    for th in grid:
        hit = 0
        for _ in range(B_alt):
            u = rng.standard_normal(p); u /= np.linalg.norm(u)
            Z = rng.standard_normal((n, p))
            X = Z + np.sqrt(th) * np.outer(rng.standard_normal(n), u)
            eigs = spectrum(X)
            s2 = fit_sigma2(eigs, n, p)
            hit += spike_ratio(eigs[0], s2, c) > thr
        if hit / B_alt >= target:
            return float(th)
    return float("inf")


# ---------------------------------------------------------------------------
# cluster bootstrap（θ̂ / R / M 的置信区间；OASBUD 按病灶）
# ---------------------------------------------------------------------------

def _cluster_resample_idx(clusters: np.ndarray, rng: np.random.Generator
                          ) -> np.ndarray:
    uniq = np.unique(clusters)
    draw = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.flatnonzero(clusters == d) for d in draw])


def bootstrap_ci(X: np.ndarray, clusters: np.ndarray | None, B: int = 500,
                 seed: int = 0, alpha: float = 0.05,
                 m_frac: float = 0.9, mode: str = "subsample") -> dict[str, tuple]:
    """θ̂ / R / M 的置信区间。

    ⚠ **不可用朴素有放回 bootstrap**（曾误用，见 F24）：
    λ₁ 是极值统计量，对**重复行**非正则。实测 n=600 时有放回重采样只得约 392 个
    唯一行，重复行使 λ₁ 由 5.541 虚高到 7.035（+27%），于是 θ̂ 的"95% CI"
    [8.5, 11.0] **完全不含点估计 3.379**。

    两条正确口径：

    mode="subsample"（默认，非参数）
        m-out-of-n **无放回**子采样，且 **c 必须随 m 修正**（c_m = p/m_images）。
        只在 clusters 层面无放回抽样，保持"同一病灶多切面同进同出"。
        实测（m=0.9n）：θ̂* [3.214, 3.608] 与 R* [1.401, 1.494] 均含点估计。

    mode="parametric"（模型内，作交叉核对）
        从拟合的尖峰模型 Σ̂ = σ̂²I + θ̂ ûûᵀ 模拟再抽取。实测 θ̂* [2.885, 3.938]，
        含点估计且均值几乎无偏。

    报告两者（非参数 + 模型内）与 F13 的双臂纪律一致。
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if clusters is None:
        clusters = np.arange(n)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    rng = np.random.default_rng(seed)
    stats = {k: [] for k in ("theta", "R", "M")}

    if mode == "subsample":
        m_cl = max(2, int(round(len(uniq) * m_frac)))
        for _ in range(B):
            draw = rng.choice(uniq, size=m_cl, replace=False)   # 无放回
            idx = np.concatenate([np.flatnonzero(clusters == d) for d in draw])
            nb = len(idx)
            if nb < 2:
                continue
            cm = p / nb                                          # **c 随 m 修正**
            eigs = spectrum(X[idx])
            s2 = fit_sigma2(eigs, nb, p)
            stats["theta"].append(theta_inversion(eigs[0], s2, cm))
            stats["R"].append(spike_ratio(eigs[0], s2, cm))
            stats["M"].append(excess_mass(eigs, s2, cm))
    elif mode == "parametric":
        # 点估计
        e0 = spectrum(X)
        s2_0 = fit_sigma2(e0, n, p)
        th0 = theta_inversion(e0[0], s2_0, p / n)
        u1 = top_eigvec(X)
        if not np.isfinite(th0) or th0 <= 0:
            return {k: (float("nan"), float("nan")) for k in ("theta", "R", "M")}
        for _ in range(B):
            Z = rng.standard_normal((n, p))
            # sqrt(Σ̂)Z = sqrt(σ̂²)Z + (sqrt(σ̂²+θ̂) − sqrt(σ̂²))(Z·u)u
            Xs = (np.sqrt(s2_0) * Z
                  + (np.sqrt(s2_0 + th0) - np.sqrt(s2_0)) * np.outer(Z @ u1, u1))
            eigs = spectrum(Xs)
            s2 = fit_sigma2(eigs, n, p)
            stats["theta"].append(theta_inversion(eigs[0], s2, p / n))
            stats["R"].append(spike_ratio(eigs[0], s2, p / n))
            stats["M"].append(excess_mass(eigs, s2, p / n))
    else:
        raise ValueError(f"未知 mode {mode!r}；应为 'subsample' 或 'parametric'")

    out = {}
    for k, v in stats.items():
        if not v:
            out[k] = (float("nan"), float("nan"))
            continue
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        out[k] = ((float(np.quantile(a, alpha / 2)),
                   float(np.quantile(a, 1 - alpha / 2))) if a.size
                  else (float("nan"), float("nan")))
    return out


# ---------------------------------------------------------------------------
# 置换校准的方向对齐检验（P1 的零假设：域标签与特征无关）
# ---------------------------------------------------------------------------

def permutation_alignment_pvalue(u1: np.ndarray, X: np.ndarray,
                                 domain: np.ndarray,
                                 clusters: np.ndarray | None = None,
                                 B: int = 1000, seed: int = 0) -> float:
    """|⟨û₁, ŵ_dom⟩| 的置换 p 值；ŵ 用均值差方向代理。cluster 级置换。

    单域数据（如 OASBUD 单独）无域标签变化，返回 nan——对齐检验不适用。
    注意：置换域标签不能检验"尖峰是否存在"（那是 mc_null 的职责）。
    """
    if len(np.unique(domain)) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = None
    dom = np.asarray(domain)
    if clusters is None:
        clusters = np.arange(len(dom))
    uniq = np.unique(clusters)
    # cluster -> domain 一致则按 cluster 置换，否则退回图像级
    cl_dom = np.array([dom[clusters == u][0] for u in uniq])
    cnt = 0
    u1 = u1 / np.linalg.norm(u1)
    def align(d):
        mu0 = X[d == 0].mean(axis=0)
        mu1 = X[d == 1].mean(axis=0)
        w = mu1 - mu0
        nw = np.linalg.norm(w)
        return abs(float(u1 @ (w / nw))) if nw > 0 else 0.0
    obs = align(dom)
    for _ in range(B):
        perm_cl = rng.permutation(cl_dom)
        d = np.empty(len(dom), dtype=int)
        for u, dv in zip(uniq, perm_cl):
            d[clusters == u] = dv
        cnt += align(d) >= obs
    return float((cnt + 1) / (B + 1))


# ---------------------------------------------------------------------------
# 可选敏感性臂：源域拟合 PCA 投影（须过保真门方可用于锁箱队列）
# ---------------------------------------------------------------------------

GATE_LO, GATE_HI = 0.8, 1.25   # 保真门容差（冻结）

# 允许使用 PCA 敏感性臂的队列白名单（I2 修正，冻结）。
# 理由：门在 BUS-UCLM（商用 B 模式）上评估，而按逐域签名模型 (A2)，u_d 因设备而异；
# BUS-UCLM 上 u≈u_S **不能推出** OASBUD 上 u≈u_S——OASBUD 是 RF 重建、与 BUS-UCLM
# 差异最大，恰是门最不可迁移的一个队列。若不过门就用，会**静默地系统性低估**
# 外部队列的签名强度（模拟：正交/去局域 u 下源 PCA 仅捕获 ~31%，θ̃=4 → 1.25）。
# 因此 PCA 臂只用于三个 B 模式队列之间；OASBUD 一律用**原生 p** 分析（c>1 允许），
# 并附 power_floor() 报告的最小可检出 θ̃。
PCA_ARM_ALLOWED_COHORTS = {"busi", "busbra", "busuclm", "blusg", "breast"}


def pca_arm_allowed(cohort: str) -> bool:
    """该队列是否允许使用 PCA 敏感性臂。"""
    return cohort.strip().lower() in PCA_ARM_ALLOWED_COHORTS


def pca_fit(X: np.ndarray, k: int) -> np.ndarray:
    """在（源域）特征上拟合前 k 主成分，返回 p×k 投影矩阵。无标签、无目标域泄漏。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T.copy()


def pca_gate(features: np.ndarray, source: np.ndarray, k: int,
             seed: int = 0) -> dict:
    """保真门：同一验证队列上比较原生 θ̂ 与源域 PCA 投影 θ̂。

    在 BUS-UCLM（无标签训练域）上执行；比值落在 [GATE_LO, GATE_HI] 才允许
    把 PCA 臂用于**同一族**（B 模式）队列。返回诊断字典。

    ⚠ 本门**不覆盖 OASBUD**：见 PCA_ARM_ALLOWED_COHORTS 的说明。
    对 OASBUD 请用原生 p 并报 power_floor()。
    """
    n, p = features.shape
    eigs = spectrum(features)
    th_native = theta_inversion(eigs[0], fit_sigma2(eigs, n, p), p / n)
    V = pca_fit(source, k)
    Xp = features @ V
    ep = spectrum(Xp)
    th_proj = theta_inversion(ep[0], fit_sigma2(ep, n, k), k / n)
    ratio = (th_proj / th_native) if th_native > 0 else float("nan")
    return {"theta_native": th_native, "theta_pca": th_proj, "ratio": ratio,
            "passed": bool(GATE_LO <= ratio <= GATE_HI),
            "scope": "B 模式队列之间；不覆盖 OASBUD（RF 重建）"}


# ---------------------------------------------------------------------------
# 多尖峰 deflation（每次重新拟合 σ²）
# ---------------------------------------------------------------------------

def deflate(X: np.ndarray, u1: np.ndarray) -> np.ndarray:
    u = u1 / np.linalg.norm(u1)
    return X - np.outer(X @ u, u)


def top_eigvec(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    Xc = X - X.mean(axis=0, keepdims=True)
    if p <= n:
        S = (Xc.T @ Xc) / n
        w, V = np.linalg.eigh(S)
        return V[:, -1]
    G = (Xc @ Xc.T) / n
    w, V = np.linalg.eigh(G)
    v = Xc.T @ V[:, -1]
    return v / np.linalg.norm(v)

# ---------------------------------------------------------------------------
# 主管线：一个 npz → Table S2 的一行（S8 协议步骤 1–5、9）
# ---------------------------------------------------------------------------

@dataclass
class S2Row:
    model: str; stage: str; branch: str; cohort: str; seed_cfg: str
    p: int; n: int; c: float
    n_eff_calib: int                 # 校准用的保守 n（多切面队列=病灶数）
    sigma2: float; lambda_plus: float
    R: float; M: float; theta: float
    R_thr: float; M_thr: float; theta_thr: float; TW: float; TW_thr: float
    detected: bool                   # 主判据：R > 校准阈值（θ̂ 阈值仅次级确认，见 F10）
    theta_ci_lo: float; theta_ci_hi: float
    R_ci_lo: float; R_ci_hi: float
    align_dom: float; align_dom_p: float
    # ---- F34 主表征（零模型自由量，全 c 域有效）----
    PR: float; er: float             # 参与率 / 有效秩（c≥1 亦报告，见 R-1）
    PR_null: float                   # 零模型解析期望 np/(n+p)
    PR_frac: float                   # PR/PR_null —— 归一化主统计量，跨 p 可比
    er_null: float                   # 零模型有效秩期望
    er_frac: float                   # er/er_null
    tk1: float; tk5: float; tk16: float; tk32: float; tk64: float   # 前 k 占迹比
    K_above: int                     # MP 边缘以上计数（白体部零模型依赖，仅描述性）
    K_above_frac: float              # 边缘以上簇的方差占比（体部模型无关）
    # ---- 撤回的绝对校准量（仅作研究内比较描述量，不带校准显著性标记）----
    sqrt_c_ref: float                # 理论参考线（不作判定）
    std_version: str                 # 标准化口径（源域统计 / 逐域）
    projection: str                  # native 或 source-pca:k（敏感性臂）


def analyze(feats: np.ndarray, labels: np.ndarray, domains: np.ndarray,
            lesion_id: np.ndarray | None, meta: dict,
            alpha: float = 0.05, B_mc: int = 1000, B_boot: int = 500,
            B_perm: int = 1000, seed: int = 0,
            conservative_n: int | None = None) -> S2Row:
    """S8 全流程。feats 假定已在导出侧完成协议规定的标准化，此处仅居中。

    conservative_n：校准用的保守有效样本量（如 OASBUD 病灶数 100）；
    None 时用图像级 n。
    """
    n, p = feats.shape
    c = p / n
    n_cal = conservative_n or n
    clusters = lesion_id if lesion_id is not None else None

    eigs = spectrum(feats)
    s2 = fit_sigma2(eigs, n, p)
    lam1 = float(eigs[0])

    null = mc_null(p, n_cal, B_mc, seed)
    thr = calibrated_thresholds(null, alpha)

    R = spike_ratio(lam1, s2, c)
    M = excess_mass(eigs, s2, c)
    th = theta_inversion(lam1, s2, c)         # 朴素反演即无偏（自验 T2）
    TW = tw_statistic(lam1, s2, n, p)

    ci = bootstrap_ci(feats, clusters, B_boot, seed + 1, alpha)

    u1 = top_eigvec(feats)
    align = abs(float(u1 @ _mean_diff_dir(feats, domains))) if len(np.unique(domains)) > 1 else float("nan")
    align_p = permutation_alignment_pvalue(u1, feats, domains, clusters, B_perm, seed + 2)

    # F34：零模型自由量在全 c 域计算（R-1 裁定：解除 c<1 守卫）
    PR = participation_ratio(eigs)
    er = effective_rank(eigs)
    pr_nul = pr_null_frac(p, n)
    er_nul = er_null_frac(p, n)
    tk = topk_trace_frac(eigs)
    cs = cluster_stats(eigs, p, n)

    return S2Row(
        model=meta.get("model", "?"), stage=meta.get("stage", "?"),
        branch=meta.get("branch", "?"), cohort=meta.get("cohort", "?"),
        seed_cfg=str(meta.get("seed", "?")),
        p=p, n=n, c=round(c, 4), n_eff_calib=n_cal,
        sigma2=s2, lambda_plus=mp_bounds(c, s2)[1],
        R=R, M=M, theta=th,
        R_thr=thr["R"], M_thr=thr["M"], theta_thr=thr["theta"],
        TW=TW, TW_thr=thr["TW"],
        detected=bool(R > thr["R"]),            # 主判据：R > 校准阈值（θ̂ 阈值仅次级确认）
        theta_ci_lo=ci["theta"][0], theta_ci_hi=ci["theta"][1],
        R_ci_lo=ci["R"][0], R_ci_hi=ci["R"][1],
        align_dom=align, align_dom_p=align_p,
        PR=PR, er=er,
        PR_null=pr_nul, PR_frac=float(PR / pr_nul) if pr_nul > 0 else float("nan"),
        er_null=er_nul, er_frac=float(er / er_nul) if er_nul > 0 else float("nan"),
        tk1=tk["tk1"], tk5=tk["tk5"], tk16=tk["tk16"],
        tk32=tk["tk32"], tk64=tk["tk64"],
        K_above=cs["K_above"], K_above_frac=cs["K_above_frac"],
        sqrt_c_ref=float(np.sqrt(c)),
        std_version=meta.get("std", "?"),
        projection=meta.get("projection", "native"),
    )


def _mean_diff_dir(X: np.ndarray, domain: np.ndarray) -> np.ndarray:
    d = np.asarray(domain)
    w = X[d == 1].mean(axis=0) - X[d == 0].mean(axis=0)
    nw = np.linalg.norm(w)
    return w / nw if nw > 0 else w


def bh_adjust(pvals: list[float]) -> list[float]:
    """Benjamini–Hochberg 校正后的 q 值。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]
        prev = min(prev, p[idx] * m / (i + 1))
        q[idx] = prev
    return q.tolist()


# ---------------------------------------------------------------------------
# IO（§5 增补版接口：feats / labels / domains / lesion_id / image_id / meta）
# ---------------------------------------------------------------------------

def read_npz(path: str) -> dict:
    z = np.load(path, allow_pickle=True)
    out = {"feats": z["feats"], "labels": z["labels"], "domains": z["domains"]}
    out["lesion_id"] = z["lesion_id"] if "lesion_id" in z.files else None
    out["image_id"] = z["image_id"] if "image_id" in z.files else None
    if "meta" in z.files:
        out["meta"] = json.loads(str(z["meta"]))
    else:
        out["meta"] = {}
    return out


def write_row(row: S2Row, path: str) -> None:
    import csv, os
    d = asdict(row)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(d.keys()))
        if not exists:
            w.writeheader()
        w.writerow(d)

# ---------------------------------------------------------------------------
# 自验（合成数据；对应阶段 1 门槛：不通过不得上真实特征）
# ---------------------------------------------------------------------------

def _spiked(n, p, th, rng, u=None):
    if u is None:
        u = rng.standard_normal(p); u /= np.linalg.norm(u)
    return rng.standard_normal((n, p)) + np.sqrt(th) * np.outer(rng.standard_normal(n), u)


def selftest() -> int:
    ok = True
    def report(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    # T1 σ² 条件中位数匹配：c ∈ {0.24, 0.64, 2.56, 10.24}，偏差 <3%
    for (p, n) in [(512, 2123), (128, 200), (512, 200), (2048, 200)]:
        vals = []
        for s in range(10):
            e = spectrum(np.random.default_rng(s).standard_normal((n, p)))
            vals.append(fit_sigma2(e, n, p))
        report(f"T1 σ̂² 无偏 c={p/n:.2f}", abs(np.mean(vals) - 1) < 0.03,
               f"mean={np.mean(vals):.4f}")

    # T2 θ̂ 朴素反演偏差 <5%（B=100；c>1 时分布左偏 sd≈0.7，小样本中位数噪声大，
    #    须足够重复数——曾误报"7–9% 偏差"，B=500 后实测偏差 <1%，校正机制已撤）
    for (p, n, th) in [(512, 2123, 3.0), (512, 200, 6.0)]:
        ests = []
        for s in range(500):
            X = _spiked(n, p, th, np.random.default_rng(s))
            e = spectrum(X)
            ests.append(theta_inversion(e[0], fit_sigma2(e, n, p), p / n))
        med = float(np.median(ests))
        report(f"T2 θ̂ 反演 c={p/n:.2f} θ̃={th}", abs(med - th) / th < 0.05,
               f"median={med:.3f} (B=500)")

    # T3 校准后的 I 类错误 ≈ α（零模型，c=0.64）
    p, n = 128, 200
    thr = calibrated_thresholds(mc_null(p, n, 400, 0), 0.05)["R"]
    hits = 0
    for s in range(200):
        e = spectrum(np.random.default_rng(100 + s).standard_normal((n, p)))
        s2 = fit_sigma2(e, n, p)
        hits += spike_ratio(e[0], s2, p / n) > thr
    report("T3 I 类错误率 ∈ [0.01, 0.10]", 0.01 <= hits / 200 <= 0.10,
           f"rate={hits/200:.3f}")

    # T4 功效：c=0.64, n=200, θ̃=2 → >0.9
    hits = 0
    for s in range(150):
        X = _spiked(n, p, 2.0, np.random.default_rng(200 + s))
        e = spectrum(X)
        hits += spike_ratio(e[0], fit_sigma2(e, n, p), p / n) > thr
    report("T4 功效 θ̃=2 @ c=0.64 > 0.9", hits / 150 > 0.9, f"power={hits/150:.2f}")

    # T5 双尖峰 deflation：θ̃ = (4, 2) 均检出，deflate 后无残余
    rng = np.random.default_rng(7)
    u1t = rng.standard_normal(p); u1t /= np.linalg.norm(u1t)
    u2t = rng.standard_normal(p); u2t -= u2t @ u1t * u1t; u2t /= np.linalg.norm(u2t)
    X = (rng.standard_normal((n, p)) + 2.0 * np.outer(rng.standard_normal(n), u1t)
         + np.sqrt(2.0) * np.outer(rng.standard_normal(n), u2t))
    e1 = spectrum(X); s2_1 = fit_sigma2(e1, n, p)
    d1 = theta_inversion(e1[0], s2_1, p / n) > calibrated_thresholds(mc_null(p, n, 300, 1), 0.05)["theta"]
    Xd = deflate(X, top_eigvec(X))
    e2 = spectrum(Xd); s2_2 = fit_sigma2(e2, n, p)
    th2 = theta_inversion(e2[0], s2_2, p / n)
    report("T5 deflation 双尖峰", d1 and th2 > 1.0,
           f"stage1_detected={d1}, θ̂₂={th2:.2f}")

    # T6 警示存档：线性降维稀释 θ̃（块平均 512→128 ≈ ×0.25）——禁止降维降 c
    p0, n0, th0 = 512, 2123, 3.0
    A = np.zeros((128, p0))
    for i in range(128):
        A[i, i * 4:(i + 1) * 4] = 0.5
    ests_raw, ests_prj = [], []
    for s in range(10):
        X = _spiked(n0, p0, th0, np.random.default_rng(s))
        e = spectrum(X)
        ests_raw.append(theta_inversion(e[0], fit_sigma2(e, n0, p0), p0 / n0))
        Xp = X @ A.T
        ep = spectrum(Xp)
        ests_prj.append(theta_inversion(ep[0], fit_sigma2(ep, n0, 128), 128 / n0))
    dil = np.median(ests_prj) / np.median(ests_raw)
    report("T6 降维稀释存档", dil < 0.5, f"θ̂ 512→128 稀释比 {dil:.2f}（≈p'/p=0.25）")

    # T7 PCA 保真门：共享签名方向时源域 PCA 保真（ratio ∈ [0.8, 1.25]）
    rng = np.random.default_rng(11)
    u_sh = rng.standard_normal(512); u_sh /= np.linalg.norm(u_sh)
    XS = _spiked(2123, 512, 4.0, rng, u_sh)
    XV = _spiked(200, 512, 4.0, rng, u_sh)
    g7 = pca_gate(XV, XS, 160)
    report("T7 PCA 门（共享 u 应通过）", g7["passed"], f"ratio={g7['ratio']:.2f}")

    # T8 正交签名时 PCA 臂稀释 >40%（门的必要性存档：A2 默认情形下必须拦下）
    rng = np.random.default_rng(12)
    uS = rng.standard_normal(512); uS /= np.linalg.norm(uS)
    uT = rng.standard_normal(512); uT -= (uT @ uS) * uS; uT /= np.linalg.norm(uT)
    XS2 = _spiked(2123, 512, 4.0, rng, uS)
    XV2 = _spiked(200, 512, 4.0, rng, uT)
    g8 = pca_gate(XV2, XS2, 160)
    report("T8 PCA 门（正交 u 应拦截）", (not g8["passed"]) and g8["ratio"] < 0.6,
           f"ratio={g8['ratio']:.2f}")

    # ------------------------------------------------------------------
    # F34 增补门槛（补遗 P-7 / 实施方案 v14 阶段 1）：三项回收测试
    # ------------------------------------------------------------------

    # T11 PR/er 在 p>n（c>1）下良定义，且零模型期望 = np/(n+p)
    #     ——这是解除 c<1 守卫的直接依据（工作日志 R-1）
    rows11 = []
    for (n_, p_) in [(629, 2047), (1875, 2046), (264, 1543)]:
        vals, ers = [], []
        for s in range(20):
            e = spectrum(np.random.default_rng(500 + s).standard_normal((n_, p_)))
            vals.append(participation_ratio(e)); ers.append(effective_rank(e))
        pred = pr_null_frac(p_, n_)
        dev = abs(np.mean(vals) - pred) / pred
        rows11.append(dev)
        report(f"T11a PR 零模型 c={p_/n_:.2f} 偏差<2%", dev < 0.02,
               f"PR={np.mean(vals):.1f} pred={pred:.1f} dev={dev*100:.2f}%")
        # er 的零模型参考须在本格 (n_, p_) 上取，不可跨格复用（曾写错）
        pred_er = er_null_frac(p_, n_)
        dev_er = abs(np.mean(ers) - pred_er) / pred_er
        report(f"T11b er 零模型 c={p_/n_:.2f} 偏差<5%", dev_er < 0.05,
               f"er={np.mean(ers):.1f} pred={pred_er:.1f} dev={dev_er*100:.2f}%")
    # 补零不变性：同一 X，补零谱 vs 仅非零谱应逐位一致
    Xz = np.random.default_rng(77).standard_normal((300, 900))
    ez = spectrum(Xz)
    report("T11c 补零不变性（PR 逐位一致）",
           abs(participation_ratio(ez) - participation_ratio(ez[ez > 0])) < 1e-9,
           f"{participation_ratio(ez):.6f} vs {participation_ratio(ez[ez>0]):.6f}")

    # T12 低秩簇（K≈30）情形：簇统计量正确回收，且 PR 显著低于零模型
    n12, p12, K = 800, 1024, 30
    rng = np.random.default_rng(2024)
    U = rng.standard_normal((p12, K))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    strengths = np.linspace(60.0, 8.0, K)              # 簇内强度递减（近似实测的衰减尾）
    X12 = (rng.standard_normal((n12, p12))
           + np.outer(rng.standard_normal(n12), np.ones(p12)) * 0.0
           + sum(np.sqrt(strengths[i]) * np.outer(rng.standard_normal(n12), U[:, i])
                 for i in range(K)))
    e12 = spectrum(X12)
    cs12 = cluster_stats(e12, p12, n12)
    pr12, prn12 = participation_ratio(e12), pr_null_frac(p12, n12)
    report("T12a 低秩簇 K̂ 回收（≥20 且 ≤2K）", 20 <= cs12["K_above"] <= 2 * K,
           f"K̂={cs12['K_above']}（真值 {K}）")
    report("T12b 低秩簇 PR 显著低于零模型", pr12 < 0.5 * prn12,
           f"PR={pr12:.1f} vs null={prn12:.1f}（PR_frac={pr12/prn12:.3f}）")
    report("T12c 簇方差占比 >30%", cs12["K_above_frac"] > 0.30,
           f"frac={cs12['K_above_frac']:.3f}")

    # T13 重尾体部情形：PR 系量稳健回收；R/θ̂ 在重尾下失准（如实标注，见 F34）
    #     构造真正的重尾体部：显式谱形 λ_i = C·i^{-a}（幂律衰减），再由该谱
    #     生成高斯样本。注意——"逐维随机尺度再标准化"造不出重尾：标准化会把
    #     尺度差除掉，谱又变回白体部（这是我第一版的错误，实测 R≈1.005 即证）。
    n13, p13, a13 = 600, 1024, 0.55
    rng = np.random.default_rng(31337)
    lam13 = (np.arange(1, p13 + 1, dtype=float)) ** (-a13)
    lam13 = lam13 / lam13.mean()                      # 归一化到平均谱质量 1
    # 由目标谱生成 X（n<p：用 X = Z·diag(√λ) 的 p×p 侧；此处走 p>n 路径，
    # 直接构造 n×n 的小侧协方差即可，等价谱由随机构型保证）
    Q13, _ = np.linalg.qr(rng.standard_normal((p13, p13)))
    X13 = rng.standard_normal((n13, p13)) @ (Q13 * np.sqrt(lam13)) @ Q13.T
    e13 = spectrum(X13)
    pr13, prn13 = participation_ratio(e13), pr_null_frac(p13, n13)
    report("T13a 重尾体部下 PR 仍反映低秩（PR_frac 不虚高）", pr13 / prn13 < 1.6,
           f"PR_frac={pr13/prn13:.3f}（零模型参考 1.0）")
    # R 在重尾体部下偏离零模型阈值 → 正是 F34 撤回它的理由，自验须如实可见
    s2_13 = fit_sigma2(e13, n13, p13)
    R13 = spike_ratio(e13[0], s2_13, p13 / n13)
    null13 = mc_null(p13, n13, 60, 3)
    R13_thr = float(np.quantile(null13["R"], 0.95))
    report("T13b 重尾体部下 R 失准可见（>阈值，假阳性）", R13 > R13_thr,
           f"R={R13:.3f} vs 零模型95%阈值={R13_thr:.3f} —— 如实标注，不带校准显著性")

    print("\n自验", "全部通过。" if ok else "存在失败项——禁止上真实特征。")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="E8 RMT 谱分析（校准判据版）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    rn = sub.add_parser("run")
    rn.add_argument("features")
    rn.add_argument("--out", default=None)
    rn.add_argument("--alpha", type=float, default=0.05)
    rn.add_argument("--conservative-n", type=int, default=None,
                    help="校准用保守有效 n（如 OASBUD 病灶数 100）")
    rn.add_argument("--project-from", default=None,
                    help="源域 npz：拟合 PCA 并投影（敏感性臂，须先过 gate）")
    rn.add_argument("--project-k", type=int, default=160)
    rn.add_argument("--seed", type=int, default=0)
    gt = sub.add_parser("gate")
    gt.add_argument("features")
    gt.add_argument("--project-from", required=True)
    gt.add_argument("--project-k", type=int, default=160)
    args = ap.parse_args()

    if args.cmd == "selftest":
        return selftest()

    if args.cmd == "gate":
        d = read_npz(args.features)
        src = read_npz(args.project_from)
        print(json.dumps(pca_gate(d["feats"], src["feats"], args.project_k),
                         indent=2, ensure_ascii=False))
        return 0

    d = read_npz(args.features)
    if args.project_from:
        src = read_npz(args.project_from)
        V = pca_fit(src["feats"], args.project_k)
        d["feats"] = d["feats"] @ V
        d["meta"]["projection"] = f"source-pca:{args.project_k}"
        print(f"[警告] PCA 敏感性臂：确认已过 BUS-UCLM 保真门，否则不得用于锁箱队列。",
              file=sys.stderr)
    row = analyze(d["feats"], d["labels"], d["domains"], d["lesion_id"],
                  d["meta"], alpha=args.alpha, seed=args.seed,
                  conservative_n=args.conservative_n)
    txt = json.dumps(asdict(row), indent=2, ensure_ascii=False)
    print(txt)
    if args.out:
        write_row(row, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
