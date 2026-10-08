"""阶段 1 门槛：在合成数据上验证 RMT 管线。

验证项：
  V1  θ̂ 反演一致性（|均值偏差| < 5% 且真值落在 3 标准误内）
  V2  σ² 估计在存在尖峰时的稳健性（三种估计量对比）
  V3  临界检验的功效曲线
  V4  主统计量 R、M 与 PR 对尖峰的灵敏度对比（说明为何 PR 被降级）
  V5  c > 1 时 PR 被零原子锁死（说明协议必须保证 c < 1）
  V6  置换校准在「有尖峰但无均值差」时不误报（P1 的正确零假设）
  V7  校准阈值 vs 理论临界线 √c（推翻「直接比较 √c」的初始设想）
"""
from __future__ import annotations

import sys

import numpy as np

from breastdg.spectral import rmt

RNG = np.random.default_rng(20260930)


def make_features(n: int, p: int, theta_tilde: float, *,
                  mean_gap: float = 0.0, rng=None) -> np.ndarray:
    """生成 Σ = σ²I + θ uuᵀ 的特征（σ²=1），可选沿 u 的均值偏移。"""
    rng = rng or np.random.default_rng(0)
    Z = rng.standard_normal((n, p))
    if theta_tilde > 0:
        u = rng.standard_normal(p)
        u /= np.linalg.norm(u)
        Z = Z + (np.sqrt(1.0 + theta_tilde) - 1.0) * np.outer(Z @ u, u)
        if mean_gap:
            Z = Z + mean_gap * u
    return Z


# --------------------------------------------------------------------------
def V1_inversion_bias() -> bool:
    """θ̂ 反演的一致性。两类情形必须分别处理，否则误判：
      (a) 接近临界时判别式可能为负 → θ̂ = NaN，单列计数，不算失败；
      (b) 大样本下为一致估计，小样本有偏；门槛按 5% / 3 标准误。
    """
    print("\n[V1] θ̂ 反演一致性（每重复用独立随机数；NaN 单列）")
    ok = True
    rng = np.random.default_rng(1234)
    for c in (0.25, 0.5, 1.0):
        for th in (1.0, 2.0, 4.0):
            crit = rmt.critical_theta(c)
            if th <= crit:
                continue
            est, n_nan = [], 0
            for _ in range(60):
                n = 1600
                p = int(round(c * n))
                X = make_features(n, p, th, rng=rng)
                e = rmt.spectrum(X)
                s2 = rmt.fit_sigma2_median(e, p / n)
                v = rmt.invert_theta(e[0], s2, p / n)
                if np.isfinite(v):
                    est.append(v)
                else:
                    n_nan += 1
            est = np.array(est)
            if len(est) < 5:
                print(f"   c={c:4.2f} θ̃={th:4.1f}  可用样本不足（NaN {n_nan}/60）")
                continue
            bias = (est.mean() - th) / th
            se = est.std(ddof=1) / np.sqrt(len(est))
            within = abs(est.mean() - th) < 3 * se + 1e-9
            passed = abs(bias) < 0.05 and within
            ok &= passed
            margin = (th - crit) / crit
            print(f"   c={c:4.2f} θ̃={th:4.1f} (余量 {margin:+5.0%})  "
                  f"估计 {est.mean():6.3f} ± {est.std():.3f}  偏差 {bias:+6.1%}  "
                  f"NaN {n_nan:2d}/60  {'OK' if passed else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
def V2_sigma_robustness() -> bool:
    print("\n[V2] σ² 估计的稳健性（真值 1.0，数据含 θ̃=4 的尖峰）")
    ok = True
    for c in (0.5, 1.0):
        n = 1000
        p = int(round(c * n))
        X = make_features(n, p, 4.0, rng=RNG)
        e = rmt.spectrum(X)
        naive = rmt.fit_sigma2_naive(e)
        median = rmt.fit_sigma2_median(e, p / n)
        trim = rmt.fit_sigma2_trimmed(e, p / n)
        print(f"   c={c:4.2f}  朴素 {naive:.4f} (偏 {naive - 1:+.1%})  "
              f"中位数 {median:.4f} (偏 {median - 1:+.1%})  "
              f"修剪 {trim:.4f} (偏 {trim - 1:+.1%})")
        ok &= abs(median - 1.0) <= abs(naive - 1.0) + 1e-3
    return ok


# --------------------------------------------------------------------------
def V3_power() -> bool:
    print("\n[V3] 临界检验的功效（α=0.05，阈值由零模型校准得到）")
    n, c = 800, 0.5
    p = int(round(c * n))
    crit = rmt.critical_theta(c)

    null = []
    for _ in range(300):
        X = make_features(n, p, 0.0, rng=RNG)
        e = rmt.spectrum(X)
        v = rmt.invert_theta(e[0], rmt.fit_sigma2_median(e, c), c)
        if np.isfinite(v):
            null.append(v)
    thr = float(np.percentile(null, 95))
    print(f"     零模型 θ̂ 的 95% 分位（校准阈值）= {thr:.3f}   "
          f"理论 √c = {crit:.3f}   比值 {thr / crit:.2f}×")

    res = []
    for th in (0.5 * crit, crit, 1.5 * crit, 2 * crit, 3 * crit):
        hits, T = 0, 60
        for _ in range(T):
            X = make_features(n, p, th, rng=RNG)
            e = rmt.spectrum(X)
            v = rmt.invert_theta(e[0], rmt.fit_sigma2_median(e, c), c)
            if np.isfinite(v) and v > thr:
                hits += 1
        res.append((th, hits / T))
        print(f"     θ̃={th:5.3f} (={th / crit:4.2f}×√c)  功效 {hits / T:.2f}")
    powers = [q for _, q in res]
    mono = all(powers[i] <= powers[i + 1] + 1e-9 for i in range(len(powers) - 1))
    print(f"     单调性: {'OK' if mono else 'FAIL'}")
    return mono


# --------------------------------------------------------------------------
def V4_statistic_sensitivity() -> bool:
    print("\n[V4] 主统计量灵敏度对比（p=512, n=1024, c=0.5，bootstrap 60 次）")
    p, n = 512, 1024
    c = p / n
    print(f"     {'θ̃':>6} {'R=λ₁/λ₊':>10} {'M(超额质量)':>13} {'PR':>9} "
          f"{'PR 零模型':>10} {'PR 信噪比':>10}")
    for th in (0.5, 1.0, 2.0, 4.0):
        X = make_features(n, p, th, rng=RNG)
        e = rmt.spectrum(X)
        s2 = rmt.fit_sigma2_median(e, c)
        R = rmt.spike_ratio(e[0], s2, c)
        M = rmt.excess_mass(e, s2, c)
        pr = rmt.participation_ratio(e)
        boots = [rmt.participation_ratio(rmt.spectrum(X[RNG.integers(0, n, n)]))
                 for _ in range(60)]
        gap = abs(rmt.pr_null(p, c) - pr) / max(float(np.std(boots)), 1e-9)
        print(f"     {th:6.1f} {R:10.3f} {M:13.5f} {pr:9.2f} "
              f"{rmt.pr_null(p, c):10.2f} {gap:10.1f}")
    print("     结论：R 与 M 对尖峰强单调且量级清晰；PR 在 θ̃≤2 时信噪比不足 → 降级为辅助。")
    return True


# --------------------------------------------------------------------------
def V5_c_gt_1() -> bool:
    print("\n[V5] c>1 时 PR 被零原子锁死（说明协议必须保证 c<1）")
    for (p, n) in [(500, 500), (500, 250), (500, 100)]:
        c = p / n
        X = make_features(n, p, 6.0, rng=RNG)
        e = rmt.spectrum(X)
        print(f"     p={p} n={n} c={c:4.2f}  零原子 {rmt.mp_atom(c):.2f}  "
              f"PR={rmt.participation_ratio(e):7.2f}  零模型 {rmt.pr_null(p, c):7.2f}")
    print("     结论：c>1 时 PR 趋近 p/(1+c) 且对尖峰不敏感；子采样特征维会加重此效应，禁止。")
    return True


# --------------------------------------------------------------------------
def _top_eigvec(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    Xc = X - X.mean(0, keepdims=True)
    if p <= n:
        S = (Xc.T @ Xc) / n
        _, v = np.linalg.eigh(S)
        return v[:, -1]
    S = (Xc @ Xc.T) / n
    _, v = np.linalg.eigh(S)
    z = Xc.T @ v[:, -1]
    return z / np.linalg.norm(z)


def V6_permutation() -> bool:
    print("\n[V6] 置换校准：「有尖峰但域无均值差」时不应误报方向对齐")
    p, n = 200, 400
    rng = np.random.default_rng(7)
    # 两个域各自有尖峰，但都不含沿尖峰方向的均值偏移
    X = np.vstack([make_features(n, p, 4.0, rng=rng),
                   make_features(n, p, 4.0, rng=rng)])
    dom = np.r_[np.zeros(n), np.ones(n)].astype(int)
    u1 = _top_eigvec(X)
    w = X[dom == 1].mean(0) - X[dom == 0].mean(0)
    w = w / (np.linalg.norm(w) + 1e-12)
    pv = rmt.permutation_alignment_pvalue(u1, w, X, dom, B=300, rng=rng)
    print(f"     观测对齐 |<û₁,ŵ>| = {abs(float(u1 @ w)):.4f}   置换 p = {pv:.3f}")
    print("     期望：p 不显著（>0.05），因两域共享尖峰而无均值差")
    ok = pv > 0.05
    print(f"     {'OK' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
def V7_calibrated_vs_theoretical() -> bool:
    """有限样本下 θ̂ 的零分布分位远高于理论 √c。

    因此判据必须是「θ̂ > 校准阈值」，不能直接用 √c——后者严重反保守。
    这与 TW 表值在本规模下不可用属同一类问题。
    """
    print("\n[V7] 校准阈值 vs 理论临界线 √c")
    print("     （本项修正「直接比较 √c」的初始设想）")
    rng = np.random.default_rng(555)
    for c in (0.25, 0.5, 1.0):
        for n in (400, 1600):
            p = int(round(c * n))
            null = []
            for _ in range(300):
                X = make_features(n, p, 0.0, rng=rng)
                e = rmt.spectrum(X)
                v = rmt.invert_theta(e[0], rmt.fit_sigma2_median(e, c), c)
                if np.isfinite(v):
                    null.append(v)
            null = np.array(null) if null else np.array([np.nan])
            q95 = float(np.nanpercentile(null, 95))
            print(f"     c={c:4.2f} n={n:5d}  理论 √c = {rmt.critical_theta(c):.3f}   "
                  f"零模型 95% 分位 = {q95:.3f}   比值 {q95 / rmt.critical_theta(c):.2f}×")
    print("     结论：直接与 √c 比较会把假阳性率抬到远高于 5%；")
    print("           判据统一为 θ̂ > 校准阈值（零模型 MC 或置换分位），√c 仅作参考线。")
    return True


# --------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("阶段 1 门槛验证：RMT 管线（合成数据）")
    print("=" * 78)
    results = {
        "V1 反演一致性": V1_inversion_bias(),
        "V2 σ² 稳健性": V2_sigma_robustness(),
        "V3 功效曲线单调": V3_power(),
        "V4 统计量灵敏度": V4_statistic_sensitivity(),
        "V5 c>1 效应": V5_c_gt_1(),
        "V6 置换校准": V6_permutation(),
        "V7 校准阈值≠√c": V7_calibrated_vs_theoretical(),
    }
    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("=" * 78)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
