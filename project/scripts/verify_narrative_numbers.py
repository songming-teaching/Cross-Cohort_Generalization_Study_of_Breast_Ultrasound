"""叙事数字核验：把要写进论文的每个数字**当场从磁盘重算**。

动机（2026-10-06 用户要求 + 本项目多次"凭记忆引用"事故）：
叙事文档里的每个数字都必须可由本脚本一键复现，且口径写明。

口径（与 F17 / 补遗 N-2 一致）
-----------------------------
- 配对相对增量：按 (方向 × 种子) 配对，逐格减去 **同 (方向,种子) 的 baseline**，
  再在 Δ 层面汇总。**不合并绝对 AUC**（两方向源域规模差 2.98×，绝对均值被
  方向难度主导，见 F29.7）。
- 配对 bootstrap：对 6 个单元有放回重抽样 B=10000，报 95% 分位 CI。
- 符号检验：6 个单元中 Δ 为正的个数，二项单侧 p。
- `steps` 取该变体最大值（排除 300 步冒烟行）。

用法
----
    python scripts/verify_narrative_numbers.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# 同 calibrate_f34.py：print 里有 Δ、−（U+2212）等 GBK 编不了的字符，
# 手动直接跑会 UnicodeEncodeError 打死进程。入口自护。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "runs" / "dev_screen" / "results.csv"
B = 10000


def load() -> dict[tuple[str, str, int], float]:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows = [r for r in rows if str(r.get("steps", "")).strip()]
    mx = max(int(r["steps"]) for r in rows)
    rows = [r for r in rows if int(r["steps"]) >= mx]
    out = {}
    for r in rows:
        key = (r["variant"], r["direction"], int(r["seed"]))
        out[key] = float(r["dev_auc_final"])
    return out


def paired_delta(data, variant: str) -> np.ndarray:
    """返回 6 个单元的 Δ = variant − baseline（同方向同种子配对）。"""
    units = sorted({(d, s) for (v, d, s) in data if v == "baseline"})
    deltas = []
    for d, s in units:
        if (variant, d, s) in data:
            deltas.append(data[(variant, d, s)] - data[("baseline", d, s)])
    return np.array(deltas)


def boot_ci(x: np.ndarray, rng) -> tuple[float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_p(x: np.ndarray) -> float:
    """单侧符号检验 p（H1: 中位数 > 0）。"""
    from math import comb
    k = int((x > 0).sum())
    n = len(x)
    return float(sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def main() -> int:
    data = load()
    variants = sorted({v for (v, _, _) in data if v != "baseline"})
    rng = np.random.default_rng(20261006)

    print("=" * 88)
    print("配对相对增量（Δ = variant − baseline，6 单元 = 2 方向 × 3 种子）")
    print("=" * 88)
    print(f"{'变体':24s}{'ΔAUC':>9s}{'95% CI':>22s}{'含0':>6s}{'同向':>7s}{'单侧p':>8s}")
    results = {}
    for v in variants:
        d = paired_delta(data, v)
        if len(d) == 0:
            continue
        lo, hi = boot_ci(d, rng)
        contains0 = "是" if lo <= 0 <= hi else "否"
        same = f"{int((d > 0).sum())}/{len(d)}"
        p = sign_p(d)
        results[v] = (d.mean(), lo, hi, contains0, same, p)
        print(f"{v:24s}{d.mean():+9.4f}   [{lo:+.4f}, {hi:+.4f}]"
              f"{contains0:>6s}{same:>7s}{p:>8.3f}")

    # ---- 方向效应 ----
    print()
    print("=" * 88)
    print("方向效应（源域样本量混杂）")
    print("=" * 88)
    by_dir = defaultdict(list)
    for (v, d, s), auc in data.items():
        by_dir[(v, d)].append(auc)
    diffs, same_sign = [], 0
    for v in sorted({v for (v, _, _) in data}):
        if ("busbra", v) in {(d, vv) for (vv, d) in by_dir}:
            pass
        small = [a for (vv, d), lst in by_dir.items() if vv == v and d == "busbra"
                 for a in lst]
        large = [a for (vv, d), lst in by_dir.items() if vv == v and d == "busi"
                 for a in lst]
        if small and large:
            dlt = statistics.mean(large) - statistics.mean(small)
            diffs.append(dlt)
            if dlt > 0:
                same_sign += 1
    print(f"  变体数 {len(diffs)}；同向 {same_sign}/{len(diffs)}")
    print(f"  中位差 {statistics.median(diffs):+.4f}  "
          f"范围 [{min(diffs):+.4f}, {max(diffs):+.4f}]")
    print(f"  源域样本量比 1589/534 = {1589/534:.2f}×")

    # ---- E7 对齐不对称 ----
    print()
    print("=" * 88)
    print("E7 对齐不对称（残差域可分性 7a，3 种子均值）")
    print("=" * 88)
    import json
    e7p = ROOT / "runs" / "diag" / "alignment" / "alignment_trajectory.jsonl"
    if e7p.exists():
        rows = [json.loads(l) for l in e7p.read_text(encoding="utf-8").splitlines() if l.strip()]
        agg = defaultdict(list)
        for r in rows:
            agg[(r["pair"], r["variant"])].append(r["auc_residual"])
        for pair, tag in (("busi-busuclm", "对齐对（GRL 直接作用）"),
                          ("busi-busbra", "留出对（GRL 看不到）")):
            off = statistics.mean(agg[(pair, "no_domainhead")])
            on = statistics.mean(agg[(pair, "baseline")])
            print(f"  {pair:14s} 对抗关 {off:.4f} → 对抗开 {on:.4f}  "
                  f"Δ={on-off:+.4f}   {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
