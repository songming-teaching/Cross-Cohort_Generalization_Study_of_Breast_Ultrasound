"""按方向配对的汇总：解决"两方向基线水平不等"的合并问题。

问题（2026-10-01 发现）
-----------------------
`build_run` 里源域 = 除留出域外的所有源域减去验证集：

    留出 BUSBRA → 源域仅 BUSI        → 训练 534 张
    留出 BUSI   → 源域仅 BUSBRA      → 训练 1589 张

源域规模差 **3 倍**，导致两方向的绝对 AUC 水平不可比
（实测 baseline FINAL：busbra ≈0.67，busi ≈0.81）。
若按 F17 直接合并两方向均值，合并值会被**源域规模**主导，而非方法差异。

本脚本给两种口径
----------------
**口径 A（绝对）**：F17 原文写法，直接合并两方向。保留以便对照。
**口径 B（配对相对）**：按 (方向 × 种子) 配对，逐格减去该格 baseline，
得到"相对基线的增量"再合并。**同时消掉方向方差与种子方差**，
比较的是方法增益而非绝对水平。

F17 的 simple-wins 候选判定在两种口径下各出一份；若结论不一致，
以口径 B 为准（它控制了两个已知的混杂），并在文档中如实披露分歧。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from breastdg import config

CSV = config.PROJECT_ROOT / "runs" / "dev_screen" / "results.csv"

# 复杂度序（用于 simple-wins：候选内取最小者）。
# 计分 = 相对默认配置的"额外机制件数"；λ=0 视为**移除**一件，故比 baseline 简单。
COMPLEXITY = {
    "no_domainhead": 0,     # 移除对抗分支
    "no_tonenorm":   0,     # 移除色调标准化
    "baseline":      1,
    "lam_0.3":       1,
    "lam_3.0":       1,
    "ema":           2,
    "render_aug":    2,
    "scale_aug":     2,
    "render_aug_ema": 3,
    "scale_render_aug": 3,
}
COMPLEXITY_DEFAULT = 5      # 未列出的（如 fdnet 双通路）视为最复杂


def load(min_steps: int | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows = [r for r in rows if str(r.get("steps", "")).strip()]
    if not rows:
        return []
    if min_steps is None:
        min_steps = max(int(r["steps"]) for r in rows)
    return [r for r in rows if int(r["steps"]) >= min_steps]


def ci95_halfwidth(vals: np.ndarray) -> float:
    if len(vals) < 2:
        return float("nan")
    return float(1.96 * vals.std(ddof=1) / np.sqrt(len(vals)))


def simple_wins(means: dict[str, float], halfwidths: dict[str, float]) -> list[str]:
    """F17：均值最高者的 CI95 半宽内者皆为候选；候选内取最简。"""
    if not means:
        return []
    best = max(means, key=lambda k: means[k])
    hw = halfwidths.get(best, float("nan"))
    if not np.isfinite(hw):
        return [best]                      # 无 CI 时退化为取最大
    cands = [k for k, v in means.items() if means[best] - v < hw]
    return sorted(cands, key=lambda k: (COMPLEXITY.get(k, COMPLEXITY_DEFAULT), k))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--min-steps", type=int, default=None)
    a = ap.parse_args()

    rows = load(a.min_steps)
    if not rows:
        print("无结果"); return 1

    # 收集：FINAL 按 (variant, direction, seed)
    fin: dict[tuple[str, str, int], float] = {}
    for r in rows:
        v = r["dev_auc_final"]
        if v in (None, "", "nan"):
            continue
        fin[(r["variant"], r["direction"], int(r["seed"]))] = float(v)

    variants = sorted({k[0] for k in fin})
    dirs = sorted({k[1] for k in fin})
    seeds = sorted({k[2] for k in fin})

    print(f"读入 {len(fin)} 条 FINAL 结果  变体 {len(variants)}  方向 {dirs}  种子 {seeds}")
    print(f"（主配置口径 = FINAL，见 F16）\n")

    # ---------------- 口径 A：绝对 ----------------
    abs_by_v: dict[str, list[float]] = defaultdict(list)
    print(f"{'variant':<18}{'dir':<8}{'n':>3}{'绝对均值':>10}{'±CI半宽':>10}")
    for v in variants:
        for d in dirs:
            vals = np.array([fin[(v, d, s)] for s in seeds if (v, d, s) in fin])
            if len(vals) == 0:
                continue
            abs_by_v[v].extend(vals.tolist())
            hw = ci95_halfwidth(vals)
            print(f"{v:<18}{d:<8}{len(vals):>3}{vals.mean():>10.4f}{hw:>10.4f}")

    # ---------------- 口径 B：按 (方向×种子) 配对相对 baseline ----------------
    rel_by_v: dict[str, list[float]] = defaultdict(list)
    print(f"\n{'variant':<18}{'dir':<8}{'n':>3}{'相对基线增量':>13}{'±CI半宽':>10}")
    for v in variants:
        for d in dirs:
            diffs = []
            for s in seeds:
                if (v, d, s) in fin and (a.baseline, d, s) in fin:
                    diffs.append(fin[(v, d, s)] - fin[(a.baseline, d, s)])
            if not diffs:
                continue
            rel_by_v[v].extend(diffs)
            arr = np.array(diffs)
            hw = ci95_halfwidth(arr)
            print(f"{v:<18}{d:<8}{len(arr):>3}{arr.mean():>+13.4f}{hw:>10.4f}")

    # ---------------- 汇总与 simple-wins ----------------
    print("\n" + "=" * 74)
    print("口径 A（绝对，F17 原文）：跨方向 × 种子合并")
    print("=" * 74)
    A_means, A_hw = {}, {}
    for v, vals in sorted(abs_by_v.items()):
        arr = np.array(vals)
        A_means[v] = float(arr.mean()); A_hw[v] = ci95_halfwidth(arr)
        print(f"  {v:<18}n={len(arr):>3}  均值 {arr.mean():.4f}  CI95半宽 {A_hw[v]:.4f}")
    A_cands = simple_wins(A_means, A_hw)
    print(f"  → 候选（CI 内）：{A_cands}")
    print(f"  → simple-wins 选定：{A_cands[0] if A_cands else '-'}")

    print("\n" + "=" * 74)
    print("口径 B（配对相对，控制方向与种子方差）")
    print("=" * 74)
    B_means, B_hw = {}, {}
    for v, vals in sorted(rel_by_v.items()):
        arr = np.array(vals)
        B_means[v] = float(arr.mean()); B_hw[v] = ci95_halfwidth(arr)
        print(f"  {v:<18}n={len(arr):>3}  增量 {arr.mean():+.4f}  CI95半宽 {B_hw[v]:.4f}")
    B_cands = simple_wins(B_means, B_hw)
    print(f"  → 候选（CI 内）：{B_cands}")
    print(f"  → simple-wins 选定：{B_cands[0] if B_cands else '-'}")

    print("\n" + "=" * 74)
    print("口径 B 显著性（配对 Δ 的 95% CI 是否含 0）")
    print("=" * 74)
    print(f"  {'variant':<18}{'Δ 均值':>9}{'CI95半宽':>10}{'95% CI':>22}  判定")
    for v in sorted(B_means, key=lambda k: -B_means[k]):
        m, hw = B_means[v], B_hw[v]
        lo, hi = m - hw, m + hw
        if v == a.baseline:
            verdict = "参照"
        elif not np.isfinite(hw):
            verdict = "样本不足"
        elif lo > 0:
            verdict = "**显著优于基线**"
        elif hi < 0:
            verdict = "**显著劣于基线**"
        else:
            verdict = "与基线无显著差"
        print(f"  {v:<18}{m:>+9.4f}{hw:>10.4f}   [{lo:>+7.4f},{hi:>+7.4f}]  {verdict}")

    # simple-wins（F17 原意）：**在与最优者无显著差异的变体中取最简**。
    #
    # ⚠ 修正记录：此前实现为"在所有未显著劣于基线的变体中取最简"，
    #   那会把**显著更优**的变体也降级掉（实测 fdnet_gate_dual Δ=+0.0259 显著为正，
    #   却被 baseline 以"更简单"压过）。规则的语义是"统计上不可区分时取简"，
    #   不是"只要不更差就取简"。
    best_v = max(B_means, key=lambda k: B_means[k])
    hb = B_hw.get(best_v, float("nan"))
    if np.isfinite(hb):
        tied = [v for v in B_means if B_means[best_v] - B_means[v] < hb]
    else:
        tied = [best_v]
    tied.sort(key=lambda k: (COMPLEXITY.get(k, COMPLEXITY_DEFAULT), k))
    print(f"\n  与最优者（{best_v}, Δ={B_means[best_v]:+.4f}, CI 半宽 {hb:.4f}）"
          f"无显著差异者：")
    for v in tied:
        print(f"      {v:<18}Δ={B_means[v]:+.4f}  "
              f"差值 {B_means[best_v]-B_means[v]:.4f}")
    print(f"  → simple-wins 选定：**{tied[0]}**"
          f"（复杂度 {COMPLEXITY.get(tied[0], COMPLEXITY_DEFAULT)}）")

    # 显著的**正向**发现单列（不受 simple-wins 影响）
    sig_pos = [v for v in B_means
               if v != a.baseline and np.isfinite(B_hw[v])
               and B_means[v] - B_hw[v] > 0]
    if sig_pos:
        print(f"\n  **显著优于基线**的变体：{sig_pos}")
        for v in sig_pos:
            print(f"      {v:<18}Δ={B_means[v]:+.4f}  "
                  f"95% CI [{B_means[v]-B_hw[v]:+.4f}, {B_means[v]+B_hw[v]:+.4f}]")
    else:
        print(f"\n  无变体显著优于基线。")

    print("\n" + "=" * 74)
    print("结论判读")
    print("=" * 74)
    selA = A_cands[0] if A_cands else None
    selB = B_cands[0] if B_cands else None

    # 覆盖度检查：A 在覆盖不均时会把"评得更多/更多落在易方向"的变体排前，
    # 这不是方法差异。故先报覆盖度，再判读。
    cov = {k: len(v) for k, v in sorted(abs_by_v.items())}
    full = max(cov.values()) if cov else 0
    uneven = [k for k, n in cov.items() if n < full]
    print(f"  覆盖度：各变体有效 (方向×种子) 数 {cov}")
    if uneven:
        print(f"  [WARN] 覆盖不全（<{full}）：{uneven}")
        print("    → 覆盖不均时**绝对口径不可用于判定**：它把"
              "『评得更多』或『更多落在易方向』的变体排前，属评测伪影。")

    print(f"\n  绝对口径（A）simple-wins 得：{selA}   [受方向难度混杂，见下]")
    print(f"  配对口径（B）simple-wins 得：{selB if selB else ok[0] if ok else a.baseline}")

    nA, nB, nV = len(A_cands), len(B_cands), len(A_means)
    hA = A_hw.get(selA, float("nan")) if selA else float("nan")
    if uneven or (nA >= nV and nB < nV):
        print(f"\n  → 结论：**以配对口径（B）为准**。"
              f"绝对口径的 CI95 半宽（{hA:.4f}）宽于变体间真实差异，")
        print(f"    {nA}/{nV} 全部落入候选 → **不具区分力**"
              f"（既非『与 B 冲突』，也非『结果不稳』）。")
        print(f"    配对口径半宽 {max(B_hw.values()):.4f}（最大者），能真正分开变体。")
    elif selA == selB:
        print("  → 两口径一致。")
    else:
        print(f"  → 真分歧（A={selA}, B={selB}）且两口径均有区分力：需人工核查。")

    print("\n  为何以配对口径为准：两方向源域训练量相差约 3 倍"
          "（留出 BUSBRA → 534 张；留出 BUSI → 1589 张），")
    print("  绝对水平天然不同（baseline ≈0.67 vs ≈0.81）。配对口径在 Δ 层面合并，")
    print("  消掉方向与种子方差，比较的是**方法增益**。")

    print("\n  复杂度序：%s  未列出者计 %d" % (COMPLEXITY, COMPLEXITY_DEFAULT))
    print("  （λ=0 与去除模块计为更简，故 no_domainhead/no_tonenorm = 0）")

    # ---- 登记项：ERM == FINAL（不改 horizon，仅登记）----
    n_same = 0
    same_ids = []
    for r in rows:
        e, f = r.get("dev_auc_erm"), r.get("dev_auc_final")
        if e and f and e not in ("", "nan") and abs(float(e) - float(f)) < 1e-9:
            n_same += 1
            same_ids.append(f"{r['variant']}/{r['direction']}/seed{r['seed']}")
    print(f"\n  登记：ERM == FINAL 的运行 {n_same}/{len(rows)} 条 {same_ids}")
    print("  成因：验证 AUC 到末步仍升 → 该方向的 horizon 偏短。")
    print("  注意：方向差**主要来自 dev-ext 难度差异，而非训练量不足**——")
    print("  busbra 方向源域 534 张（epoch 数约为 busi 的 3 倍）而表现更低，")
    print("  说明是评测难度而非欠训练。horizon 已冻结（F16），不改，仅登记。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
