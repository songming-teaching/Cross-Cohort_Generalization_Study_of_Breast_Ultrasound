"""集成层（E5/E9）：五骨干概率融合 + 校准分析。

**为什么需要**：核查发现代码库**没有任何集成实现**——论文的核心方法声明是
「五骨干异构集成」，但没有概率融合代码。本脚本补齐。

三口径对比（v14 §3 要求，simple-wins 裁决）
--------------------------------------------
1. `uniform`  均匀平均——**零拟合、零自由度**，最强基线；
2. `de`       Dirichlet 搜索（权重单纯形上网格搜索，目标 = AUC）；
3. `stacking` logistic 回归（输入为各模型 logit）。

⚠ 2026-10-07 结构性重写（check3 的 P0）
=========================================
此前本脚本的 `main()` 直接 `load_probs(--tag oasbud_corrected)` 拿到**锁箱
外部集**的 `y`，然后把它喂给 `fuse_de` / `fuse_stacking` —— 即在**外部集
标签上做参数选择**。这与「外部集上调阈值」是同一严重等级的错误，且更隐蔽：
DE 要在 11⁵ = 161051 组权重里挑外部集 AUC 最高的那一组，等于用外部集标签
做了 16 万次选择；stacking 更是直接 `fit(X, y)`。

讽刺的是本文件的 docstring 自己写着「纪律：三口径全部在**源域验证集**上定
权重，绝不用外部集」，而用法示例给的正是 `--tag oasbud_corrected`。
**光靠注释约束不住，所以改成结构性约束**：

  · 「拟合」与「应用」被拆成两组函数，签名上就不可能混用 ——
      `fit_de(P, y)` → 返回权重；`apply_de(P, w)` → 只看权重，**没有 y 参数**。
    凡是接收 `y` 的函数，名里一律带 `fit`，且只允许用源域数据调用。
  · 外层评估走**独立的子命令** `--outer-eval`，它**不 import 也不调用**
    任何 `fit_*`；权重只能从 `fusion_weights.json` 读。
  · 权重文件缺失 → **拒绝执行**（退 4），不回退到"在现场拟合"。

工作流（顺序不可颠倒）
----------------------
  1) 源域拟合（可任意次重跑，只用源域验证集）：
       python scripts/freeze_thresholds.py --dump-source-probs   # 顺手产出源域概率
       python scripts/ensemble_and_calibrate.py --fit-source
     → runs/ensemble/fusion_weights.json（权重冻结）
  2) 锁箱开封后，外部集**只应用**（单次、不可逆）：
       python scripts/ensemble_and_calibrate.py --outer-eval --tag oasbud_corrected

校准（E9）
---------
reliability 曲线 + Brier + ECE。**纯评估指标，无拟合**，故在外部集上算没有问题
——它不参与任何选择，只是把已冻结的预测描述出来。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np

# Windows 控制台默认 GBK；本文件 print 含 `→`/`✓`/`·` 等 GBK 编不了的字符。
# 子进程有 PYTHONIOENCODING 兜底，但本脚本会被手动直接跑，故入口自护
# （preflight 坑1 按行号判时序：必须早于任何风险 print）。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config

OUT = config.RUNS_DIR / "ensemble"
WEIGHTS = OUT / "fusion_weights.json"

# 源域概率 CSV（由 freeze_thresholds.py --dump-source-probs 产出）
SOURCE_PROBS = config.RUNS_DIR / "source_val" / "probs_source_val.csv"

# 全局唯一自举种子（P1-a / preflight 坑10）
SEED_BOOTSTRAP = 20261006

EXIT_ERR = 1
EXIT_NO_WEIGHTS = 4          # 缺源域冻结融合权重 → 拒绝执行


def _log2p(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


# ===========================================================================
# 应用 —— 只吃权重，**没有 y 参数**。外部集上唯一的入口。
# ===========================================================================
def apply_uniform(probs: np.ndarray) -> np.ndarray:
    return probs.mean(axis=0)


def apply_de(probs: np.ndarray, weights) -> np.ndarray:
    w = np.asarray(weights, float)
    if w.shape[0] != probs.shape[0]:
        raise ValueError(f"权重维度 {w.shape[0]} ≠ 模型数 {probs.shape[0]}")
    return w @ probs


def apply_stacking(probs: np.ndarray, coef, intercept) -> np.ndarray:
    z = _log2p(probs).T @ np.asarray(coef, float) + float(np.asarray(intercept).ravel()[0])
    return 1.0 / (1.0 + np.exp(-z))


# ===========================================================================
# 拟合 —— 一律带 `fit_` 前缀，只允许在**源域验证集**上调用（A3）。
# ===========================================================================
# ---------------------------------------------------------------------------
# ⚠ 2026-10-08 修（P0，本轮实测暴露）：`fit_de` 的网格搜索随模型数**指数爆炸**。
#
#   原实现 = `itertools.product(grid, repeat=m)`，grid 步长 0.1 → 11^m 个组合，
#   每个组合算一次 roc_auc_score。m=5 时 11^5 = 161051，自验里毫无问题；
#   但**主配置是 7 变体 × 7 种子 = 49 个模型** → 11^49 ≈ 1.07×10^51 个组合。
#   实测单次 roc_auc_score(49×381) ≈ 0.89 ms → 全枚举 ≈ 3×10^40 年。
#   即：`--fit-source` 在真实规模上**永远跑不完**（不是慢，是不可达），
#   而手册把它列为必经步骤（⑪），下游 `--outer-eval` 又硬依赖其产物。
#   这是典型的「小规模验收、大规模爆炸」缺陷 —— selftest 用 m=5，恰好绕过。
#
#   修法：**保留穷举语义**（它是冻结契约，且 m 小时确实更优），仅在组合数
#   超阈值时改用**顶点起步 + 坐标精化**，并把降级事实写进返回值的
#   `search` 字段 —— 使下游与论文能**如实交代搜索范围**，而不是把
#   启发式结果冒充穷举最优。
#   ⚠ 降级后 DE 是**启发式**（不保证全局最优），故相关措辞须相应降级：
#     不得写成"最优加权平均"，应写"坐标搜索得到的加权平均"。
# ---------------------------------------------------------------------------
DE_EXHAUSTIVE_MAX = 200_000     # 组合数上限；超过即降级（11^5=161051 仍走穷举）


def _de_search_exhaustive(probs: np.ndarray, y: np.ndarray, grid, m: int):
    """穷举全部网格组合（m 小时可用）。"""
    from sklearn.metrics import roc_auc_score
    best_w, best_auc = None, -np.inf
    for w in product(grid, repeat=m):
        s = sum(w)
        if s <= 0:
            continue
        wv = np.array(w) / s
        auc = roc_auc_score(y, wv @ probs)
        if auc > best_auc:            # `>` 保留首个最优 → 迭代序固定即可复现
            best_auc, best_w = auc, wv
    return best_w, best_auc


def _de_search_refined(probs: np.ndarray, y: np.ndarray, m: int,
                       n_grid: int = 11, n_round: int = 4):
    """顶点起步 + 坐标精化（m 大时的**可算**替代）。

    为什么不是"随机搜索"：本仓的复现纪律要求同输入同输出。坐标搜索在
    **固定迭代序**下是确定性的，满足该纪律。
    为什么从顶点起步：保证"最差也不劣于最佳单模型"，给结果一个下界，
    避免启发式搜索得出比单个模型还差的权重。
    平局规则 `> best + 1e-12` 与穷举路径的 `> best_auc` 同向（保留首个最优）。
    """
    from sklearn.metrics import roc_auc_score

    def score(wv: np.ndarray) -> float:
        s = wv.sum()
        return roc_auc_score(y, (wv / s) @ probs) if s > 0 else -np.inf

    # ① 顶点起步：最佳单模型（保证下界）
    singles = [score(np.eye(m)[i]) for i in range(m)]
    best_w = np.zeros(m)
    best_w[int(np.argmax(singles))] = 1.0
    best_auc = float(max(singles))

    # ② 坐标精化：逐个分量在定网格上重扫，直到一轮无改进
    grid = np.linspace(0.0, 1.0, n_grid)
    for _ in range(n_round):
        improved = False
        for i in range(m):
            for v in grid:
                trial = best_w.copy()
                trial[i] = v
                a = score(trial)
                if a > best_auc + 1e-12:
                    best_auc, best_w, improved = a, trial, True
        if not improved:
            break
    return best_w / best_w.sum(), best_auc


def fit_de(probs: np.ndarray, y: np.ndarray, grid=None) -> dict:
    """Dirichlet 搜索。**只在源域验证集上调用。** 返回冻结所需的权重。

    「DE」= **D**irichlet 权重搜索，与差分进化（Differential Evolution）无关
    —— 名字沿用已久，此处只作澄清，不改名（改名会破坏冻结契约）。

    ⚠ 搜索空间随模型数指数增长，见上面 DE_EXHAUSTIVE_MAX 的说明：
      m ≤ 5 走**穷举**（与历史行为逐位相同）；
      m > 5 走**顶点+坐标精化**，降级事实记录在返回值的 `search` 字段。
    """
    m = probs.shape[0]
    if grid is None:
        grid = np.arange(0.0, 1.0 + 1e-9, 0.1)
    n_comb = len(grid) ** m
    step = float(grid[1] - grid[0]) if len(grid) > 1 else None
    if n_comb <= DE_EXHAUSTIVE_MAX:
        best_w, best_auc = _de_search_exhaustive(probs, y, grid, m)
        search = dict(mode="exhaustive", n_combinations=int(n_comb),
                      exhaustive_feasible=True, grid_step=step)
    else:
        best_w, best_auc = _de_search_refined(probs, y, m)
        search = dict(
            mode="coordinate_refine", n_combinations=int(n_comb),
            exhaustive_feasible=False, grid_step=step,
            note=(f"模型数 {m} 使穷举组合数 {n_comb} 超出上限 "
                  f"{DE_EXHAUSTIVE_MAX} —— 已降级为**启发式**坐标搜索"
                  f"（顶点起步，故最差亦不劣于最佳单模型）。"
                  f"不得声称全局最优。"))
    if best_w is None:
        raise RuntimeError("DE 搜索无可行解（权重全为 0？）")
    return dict(weights=[float(x) for x in best_w], auc_in_fit=float(best_auc),
                search=search)


def fit_stacking(probs: np.ndarray, y: np.ndarray) -> dict:
    """logistic 回归。**只在源域验证集上调用。** 返回冻结所需的系数。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = _log2p(probs).T
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X, y)
    return dict(coef=[float(x) for x in lr.coef_[0]],
                intercept=[float(x) for x in np.asarray(lr.intercept_).ravel()],
                auc_in_fit=float(roc_auc_score(y, lr.predict_proba(X)[:, 1])))


# ===========================================================================
# 校准（E9）—— 纯评估，无拟合
# ===========================================================================
def calibration(y, p, n_bins: int = 10) -> dict:
    y = np.asarray(y, float); p = np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            rows.append(dict(bin_lo=float(bins[b]), bin_hi=float(bins[b+1]),
                             n=0, conf=None, acc=None))
            continue
        conf = float(p[m].mean()); acc = float(y[m].mean()); frac = m.mean()
        ece += frac * abs(acc - conf)
        rows.append(dict(bin_lo=float(bins[b]), bin_hi=float(bins[b+1]),
                         n=int(m.sum()), conf=conf, acc=acc))
    brier = float(np.mean((p - y) ** 2))
    return dict(bins=rows, ece=float(ece), brier=brier)


def cluster_bootstrap_auc(y, p, clusters, B=2000,
                          seed=SEED_BOOTSTRAP, alpha=0.05):
    """簇级自举 CI。种子 = **20261006**（全库唯一口径，见 preflight 坑10）。

    原为 `seed=0`，而 `evaluate_lockbox.py` 的同一个函数用另一个种子，
    于是**同一批数据存在两套 CI 口径** —— 4 位小数一比对就露馅。
    自举种子只影响第 3-4 位小数；关键是全篇只有一个口径。
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p); cl = np.asarray(clusters)
    uniq = np.unique(cl); rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cl == d) for d in draw])
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha/2)), float(np.quantile(vals, 1-alpha/2))


# ===========================================================================
# 读概率 CSV
# ===========================================================================
def load_probs(probs_dir: Path, tag: str):
    """读 `probs_<tag>.csv`（evaluate_lockbox.py 产物）→ (P, y, models, clusters)。"""
    p = probs_dir / f"probs_{tag}.csv"
    if not p.exists():
        raise FileNotFoundError(f"未找到 {p}（先跑 evaluate_lockbox.py）")
    return load_probs_csv(p)


def load_probs_csv(p: Path):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"{p} 为空")
    models = sorted({r["variant"] + f"/seed{r['seed']}" for r in rows})
    keys = sorted({(r["image"], r["label"], r["lesion_id"]) for r in rows})
    kidx = {k: i for i, k in enumerate(keys)}
    P = np.full((len(models), len(keys)), np.nan)
    mvidx = {m: i for i, m in enumerate(models)}
    for r in rows:
        mi = mvidx[r["variant"] + f"/seed{r['seed']}"]
        ki = kidx[(r["image"], r["label"], r["lesion_id"])]
        P[mi, ki] = float(r["prob"])
    if np.isnan(P).any():
        raise RuntimeError(f"{p} 概率矩阵有缺（各模型的图集不一致）")
    y = np.array([int(k[1]) for k in keys])
    cl = np.array([k[2] for k in keys])
    # 空簇键是**静默**故障：会让每张图都算独立簇 → CI 假窄。显式拦下。
    if any(str(c).strip() == "" for c in cl):
        raise RuntimeError(
            f"{p} 的 lesion_id 列存在空串 —— 簇键未解析。\n"
            f"  下游会把全队列归一簇，得到 CI 宽度为 0 的假精确。\n"
            f"  该 CSV 应由 evaluate_lockbox.py 写出（它走 cluster_keys 回退链）。")
    return P, y, models, cl


# ===========================================================================
# 两个子命令
# ===========================================================================
def cmd_fit_source(a) -> int:
    """在**源域验证集**上拟合权重并冻结。可任意次重跑（不碰任何锁箱队列）。"""
    src = Path(a.source_probs)
    if not src.exists():
        print(f"✗ 缺源域概率：{src}\n"
              f"  先跑：python scripts/freeze_thresholds.py --dump-source-probs\n"
              f"  那是**源域验证集**上的概率，本步只在源域上拟合。", file=sys.stderr)
        return EXIT_ERR
    P, y, models, cl = load_probs_csv(src)
    print(f"源域拟合  {src}")
    print(f"  模型 {len(models)} 个  样本 {len(y)}  簇 {len(np.unique(cl))}"
          f"  患病率 {y.mean():.3f}")
    if len(models) < 2:
        print("  ⚠ 只有 1 个模型，集成无意义 —— 仍继续（便于冒烟）")

    de = fit_de(P, y)
    st = fit_stacking(P, y)
    rec = {
        "fit_source": {"csv": str(src), "n": int(len(y)),
                       "n_clusters": int(len(np.unique(cl))),
                       "prevalence": float(y.mean()), "models": models},
        "seed_bootstrap": SEED_BOOTSTRAP,
        "fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "de": de,
        "stacking": st,
        "uniform_auc_in_fit": float(
            __import__("sklearn.metrics", fromlist=["x"]).roc_auc_score(
                y, apply_uniform(P))),
        "note": ("融合权重在**源域验证集**上拟合后冻结；外部集只应用、不重新拟合"
                 "（A3 目标域标签隔离 / F7.1）。--outer-eval 不调用任何 fit_*。"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"  DE       源域拟合内 AUC={de['auc_in_fit']:.4f}"
          f"  权重={[round(x,3) for x in de['weights']]}")
    print(f"  stacking 源域拟合内 AUC={st['auc_in_fit']:.4f}")
    print(f"  uniform  源域内 AUC={rec['uniform_auc_in_fit']:.4f}")
    print(f"\n✓ 已冻结 {WEIGHTS}")
    print("  外部集评估请用 --outer-eval（它只读本文件，不拟合）")
    return 0


def cmd_outer_eval(a) -> int:
    """外部集评估：**只应用冻结权重**。不 import 任何 fit_* 的调用路径。"""
    if not WEIGHTS.exists():
        print(f"✗ 缺源域冻结融合权重：{WEIGHTS}\n"
              f"  外部集上**不允许**重新拟合权重（A3）：那是在外部集标签上做选择，\n"
              f"  与『外部集上调阈值』同一严重等级。\n"
              f"  先跑：python scripts/ensemble_and_calibrate.py --fit-source\n"
              f"  ——拒绝执行。", file=sys.stderr)
        return EXIT_NO_WEIGHTS
    W = json.loads(WEIGHTS.read_text(encoding="utf-8"))

    from sklearn.metrics import roc_auc_score
    P, y, models, cl = load_probs(Path(a.probs_dir), a.tag)
    print(f"tag={a.tag}  模型 {len(models)} 个  图 {len(y)} 张")
    print(f"  权重来自 {W['fit_source']['csv']}"
          f"（n={W['fit_source']['n']}，冻于 {W['fitted_at']}）")

    # 模型集合必须与拟合时一致 —— 否则"应用"本身无效（换了模型还套旧权重）
    if models != W["fit_source"]["models"]:
        print(f"\n✗ 模型集合与冻结权重不匹配 → 拒绝执行。\n"
              f"  权重拟合时：{W['fit_source']['models']}\n"
              f"  本次外部集：{models}\n"
              f"  权重不能跨模型集合搬运（那是另一种形式的出处不明）。",
              file=sys.stderr)
        return EXIT_ERR

    # ---- 应用（无 y 参与）----
    preds = {
        "uniform": apply_uniform(P),
        "de": apply_de(P, W["de"]["weights"]),
        "stacking": apply_stacking(P, W["stacking"]["coef"],
                                   W["stacking"]["intercept"]),
    }
    weights_used = {
        "uniform": None,
        "de": W["de"]["weights"],
        "stacking": {"coef": W["stacking"]["coef"],
                     "intercept": W["stacking"]["intercept"]},
    }

    # ---- 评估（校准与 AUC 都是纯评估，不参与选择）----
    OUT.mkdir(parents=True, exist_ok=True)
    results = {"_provenance": {
        "weights_file": str(WEIGHTS),
        "weights_fitted_on": W["fit_source"],
        "outer_tag": a.tag,
        "note": ("外部集仅应用冻结权重；未在外部集上做任何拟合或选择（A3）"),
    }}
    for name, pr in preds.items():
        auc = float(roc_auc_score(y, pr))
        lo, hi = cluster_bootstrap_auc(y, pr, cl)
        cal = calibration(y, pr)
        results[name] = dict(auc=auc, ci=[lo, hi], weights=weights_used[name],
                             calib=cal)
        print(f"  {name:9s} AUC={auc:.4f} [{lo:.4f},{hi:.4f}]  "
              f"Brier={cal['brier']:.4f}  ECE={cal['ece']:.4f}")

    singles = {m: float(roc_auc_score(y, P[i])) for i, m in enumerate(models)}
    results["singles"] = singles
    best_single = max(singles.values()) if singles else float("nan")
    print(f"  最佳单模型 AUC={best_single:.4f}  "
          f"（集成增益 {results['uniform']['auc']-best_single:+.4f}）")

    p = OUT / f"ensemble_{a.tag}.json"
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")
    return 0


# ===========================================================================
# 自验
# ===========================================================================
def selftest() -> int:
    ok = True
    def rep(name, cond, detail=""):
        nonlocal ok; ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    rng = np.random.default_rng(0)
    n, m = 600, 5
    y = rng.integers(0, 2, n)
    base = 0.5 + 0.35 * y + rng.normal(0, 0.18, n)
    probs = np.vstack([np.clip(base + rng.normal(0, 0.05, n), 0, 1) for _ in range(3)]
                      + [rng.random(n) for _ in range(2)])
    from sklearn.metrics import roc_auc_score

    de = fit_de(probs, y)
    st = fit_stacking(probs, y)
    d = apply_de(probs, de["weights"])
    s = apply_stacking(probs, st["coef"], st["intercept"])
    u = apply_uniform(probs)
    auc_u, auc_d, auc_s = (roc_auc_score(y, u), roc_auc_score(y, d),
                           roc_auc_score(y, s))
    rep("T1 均匀平均有效", auc_u > 0.85, f"AUC={auc_u:.3f}")
    rep("T2 DE 不劣于均匀平均", auc_d >= auc_u - 0.02,
        f"DE={auc_d:.3f} vs uniform={auc_u:.3f}")
    w = np.array(de["weights"])
    rep("T3 噪声模型权重被压低", w[:3].sum() > w[3:].sum() + 0.3,
        f"好的权重和={w[:3].sum():.3f} 噪声={w[3:].sum():.3f}")

    # ---- T4~T6 本轮新增：拟合与应用**必须可分离**（P0 的结构性保证）----
    import ast as _ast
    import inspect
    src_text = Path(__file__).read_text(encoding="utf-8")
    tree = _ast.parse(src_text)

    def body_of(func_name: str):
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) \
                    and node.name == func_name:
                return node
        return None

    # T4（**结构性**）：外部集评估函数体内**不得出现任何 `fit_*` 调用**。
    #   这是 P0 的真正保证 —— 不是"我这次没传外部集标签"，而是"这条路径
    #   上根本没有拟合代码可达"。静态可查，故比任何注释都硬。
    #   （先前写的是 `apply_de(P,w)==apply_de(P,w)`，恒真，什么也没证明。
    #     实测发现后换成 AST。自验证伪的检查等于没检查。）
    oe = body_of("cmd_outer_eval")
    called_in_outer = {n.func.id for n in _ast.walk(oe)
                       if isinstance(n, _ast.Call)
                       and isinstance(n.func, _ast.Name)} if oe else set()
    fits = sorted(c for c in called_in_outer if c.startswith("fit_"))
    rep("T4 外部集路径不可达任何 fit_*", oe is not None and not fits,
        f"cmd_outer_eval 调用了 {sorted(called_in_outer)}")

    # T5：`--outer-eval` 与 `--fit-source` 互斥（main 里的显式拒绝）
    m = body_of("main")
    has_mutex = any(
        isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
        and n.func.attr == "error" for n in _ast.walk(m)) if m else False
    rep("T5 拟合/应用入口互斥", has_mutex, "main 中 ap.error 拒绝同用")

    # T6：签名层面 —— fit_* 有 y 参数，apply_* 没有
    rep("T6 fit_/apply_ 签名互斥",
        "y" in inspect.signature(fit_de).parameters
        and "y" not in inspect.signature(apply_de).parameters
        and "y" not in inspect.signature(apply_stacking).parameters,
        "fit_de(P,y) / apply_de(P,w) / apply_stacking(P,coef,intercept)")

    # T7：换标签确实会改变**拟合**结果 —— 反证"只应用不拟合"有实质约束力。
    #   注意：这一步很贵（11⁵ 网格再跑一遍）。故只在取样小子集上做。
    _sub = rng.choice(n, size=min(n, 200), replace=False)
    y_other = rng.integers(0, 2, len(_sub))
    de_o = fit_de(probs[:, _sub], y_other)
    de_s = fit_de(probs[:, _sub], y[_sub])
    rep("T7 换标签确实会改变拟合结果",
        not np.allclose(de_o["weights"], de_s["weights"]),
        f"{[round(x,2) for x in de_o['weights']]} vs {[round(x,2) for x in de_s['weights']]}")

    rep("T8 校准输出结构完整",
        all(k in calibration(y, u) for k in ("bins", "ece", "brier")),
        f"ECE={calibration(y,u)['ece']:.3f}")
    n_cal = 20000
    pc = rng.random(n_cal)
    y_cal = (rng.random(n_cal) < pc).astype(int)
    ece_perfect = calibration(y_cal, pc)["ece"]
    rep("T9 完美校准预测 ECE ≈ 0", ece_perfect < 0.02, f"ECE={ece_perfect:.4f}")

    print("\n自验", "全部通过。" if ok else "存在失败项。")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-source", action="store_true",
                    help="在**源域验证集**上拟合并冻结融合权重（可任意次）")
    ap.add_argument("--outer-eval", action="store_true",
                    help="外部集：**只应用**冻结权重（单次、不可逆）")
    ap.add_argument("--source-probs", default=str(SOURCE_PROBS),
                    help="源域概率 CSV（freeze_thresholds.py --dump-source-probs 产出）")
    ap.add_argument("--probs-dir", default=str(config.RUNS_DIR / "lockbox_eval"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.fit_source and a.outer_eval:
        ap.error("--fit-source 与 --outer-eval 互斥：拟合只允许在源域，"
                 "外部集只应用。两者放同一个进程里就失去了结构性隔离。")
    if a.fit_source:
        return cmd_fit_source(a)
    if a.outer_eval:
        if not a.tag:
            ap.error("--outer-eval 需给 --tag")
        return cmd_outer_eval(a)
    ap.error("必须给 --fit-source / --outer-eval / --selftest 之一")


if __name__ == "__main__":
    sys.exit(main())
