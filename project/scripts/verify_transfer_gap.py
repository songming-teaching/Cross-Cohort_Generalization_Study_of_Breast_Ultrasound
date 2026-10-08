#!/usr/bin/env python3
"""核验 v5 叙事的第一幕钩子：**同检查点的跨库迁移落差**。

为什么需要它
------------
v5 把开篇钩子从旧的 `SP = 0` 换成一个**完全来自公开队列**的量：
"同一次训练、同一个检查点，源库验证 AUC 与异库 AUC 的落差"。

换掉的理由（两条，都要写进论文）：
1. 旧钩子（SP）是**阈值相关**指标，必被患病率与阈值口径纠缠 —— 审稿人一句
   "是不是阈值没调好"就能反咬；新钩子用 **AUC（阈值无关）**，堵死这条反驳。
2. 旧钩子的数字出处不属于可用的公开队列，按数据边界**已拔除**。

口径（必须与 `dev_screen.run_one` 的实现逐字对应）
--------------------------------------------------
- `src_val_auc` = `res["best"]["auc"]` —— **best_erm.pt 在源域验证集上的 AUC**
  （按源域 AUC 选出的最优步）。
- `dev_auc_erm`  = 把**同一个 best_erm.pt** 在**留出域**上评出的 AUC。
  ⇒ 二者**同检查点**，相减即干净的迁移落差。（`dev_auc_final` 是 final.pt，
    与 `src_val_auc` 不同检查点，只作对照，不用于主口径。）
- 留出域完全不可见：`build_run` 中 `dev_ext = dom[direction]` 仅用于评估；
  DANN 的无标签目标域**恒为 BUS-UCLM**（`tgt = dom["busuclm"]`），与留出域无关。
- **按队列归位**，不按"方向"——方向只是实验的索引，队列才是被迁移的对象。
  按方向切会造出虚假的左右不对称（本脚本 tail 处有对照，实测过一次）。

用法
----
    python scripts/verify_transfer_gap.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "runs" / "dev_screen" / "results.csv"

# direction = 留出的域；由此确定谁是源、谁是留出
SRC_OF = {"busbra": "BUSI", "busi": "BUSBRA"}


def load() -> list[dict]:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows = [r for r in rows if str(r.get("steps", "")).strip()]
    mx = max(int(r["steps"]) for r in rows)
    return [r for r in rows if int(r["steps"]) >= mx]


def main() -> int:
    rows = load()
    g: dict[str, dict[str, dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict))
    for r in rows:
        g[r["variant"]][r["direction"]][int(r["seed"])] = (
            float(r["src_val_auc"]), float(r["dev_auc_erm"]))

    def m(v: str, d: str, i: int) -> float:
        return statistics.mean(x[i] for x in g[v][d].values())

    variants = sorted(g)
    # 按【队列】归位：某队列作源时的源域val，与它作留出时的异库AUC
    coh = {
        "BUSI":   ([m(v, "busbra", 0) for v in variants],   # BUSI 作源 → 源域val
                   [m(v, "busi", 1) for v in variants]),    # BUSI 作留出 → 异库
        "BUSBRA": ([m(v, "busi", 0) for v in variants],
                   [m(v, "busbra", 1) for v in variants]),
    }

    print("=" * 80)
    print("同检查点（best_erm）的跨库迁移落差 —— 全部来自公开队列")
    print("=" * 80)
    print(f"{'队列':8s}{'源域验证AUC':>13s}{'异库AUC':>11s}{'落差':>10s}"
          f"{'逐变体同号':>12s}")
    allin, allx = [], []
    for c, (ind, crs) in coh.items():
        gaps = [a - b for a, b in zip(ind, crs)]
        allin += ind
        allx += crs
        print(f"{c:8s}{statistics.mean(ind):13.4f}{statistics.mean(crs):11.4f}"
              f"{statistics.mean(gaps):+10.4f}"
              f"{sum(1 for x in gaps if x > 0):>8d}/{len(gaps)}")
    gaps = [a - b for a, b in zip(allin, allx)]
    print("-" * 80)
    print(f"{'合计':8s}{statistics.mean(allin):13.4f}{statistics.mean(allx):11.4f}"
          f"{statistics.mean(gaps):+10.4f}"
          f"{sum(1 for x in gaps if x > 0):>8d}/{len(gaps)}")
    print(f"\n  逐变体落差的 16 个值："
          f"[{min(gaps):+.4f}, {max(gaps):+.4f}]  中位 {statistics.median(gaps):+.4f}")
    print(f"  异库 AUC 最低单格 {min(allx):.4f}")

    # 对照：若错误地按"方向"归位，会造出虚假不对称（留作反例，防回归）
    print()
    print("=" * 80)
    print("反例对照：按【方向】归位会造出的虚假不对称（不要这样切）")
    print("=" * 80)
    for d in ("busbra", "busi"):
        ind = statistics.mean(m(v, d, 0) for v in variants)
        crs = statistics.mean(m(v, d, 1) for v in variants)
        print(f"  direction={d:8s} 源域val {ind:.4f} → 异库 {crs:.4f}  "
              f"落差 {ind-crs:+.4f}")

    # 与 final 检查点对照，证明钩子对检查点选择稳健
    print()
    print("=" * 80)
    print("稳健性：换用 final.pt 口径（不同检查点，仅对照）")
    print("=" * 80)
    fin = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        fin[r["variant"]][r["direction"]][int(r["seed"])] = float(r["dev_auc_final"])
    for c, dd in (("BUSI", "busi"), ("BUSBRA", "busbra")):
        crs = statistics.mean(
            statistics.mean(fin[v][dd].values()) for v in variants)
        print(f"  {c:8s} 异库 {crs:.4f}（erm 口径 {statistics.mean(coh[c][1]):.4f}）")

    # ------------------------------------------------------------------
    # 落盘（2026-10-08 加）：本脚本原先**只 print、不落盘** ——
    # 于是叙事里引用的 `0.8863 → 0.7246 / 32/32` **没有任何产物可指**，
    # 只有"跑一下屏幕上看"。要写方法学就必须能指到产物，故补 JSON。
    # 口径与上面打印的**完全一致**（同一批 `coh`/`gaps`，不另起一份计算），
    # 避免"打印是一份口径、落盘是另一份"这类分叉。
    # ------------------------------------------------------------------
    rec = {
        "script": "scripts/verify_transfer_gap.py",
        "input_artifact": "runs/dev_screen/results.csv",
        "input_producer": "src/breastdg/train/dev_screen.py",
        "checkpoint": "best_erm.pt（与 src_val_auc **同一检查点**）",
        "cohort_axis": "queue（**不按 direction** —— 见 counterexample_by_direction）",
        "stage": "dev",
        "n_variants": len(variants),
        "per_cohort": {},
        "total": {},
        "counterexample_by_direction": {},
        "final_pt_robustness": {},
    }
    for c, (ind, crs) in coh.items():
        gp = [a - b for a, b in zip(ind, crs)]
        rec["per_cohort"][c] = {
            "src_val_auc": statistics.mean(ind),
            "external_auc": statistics.mean(crs),
            "gap": statistics.mean(gp),
            "n_positive": sum(1 for x in gp if x > 0),
            "n_total": len(gp),
        }
    rec["total"] = {
        "src_val_auc": statistics.mean(allin),
        "external_auc": statistics.mean(allx),
        "gap": statistics.mean(gaps),
        "n_positive": sum(1 for x in gaps if x > 0),
        "n_total": len(gaps),
        "gap_min": min(gaps), "gap_max": max(gaps),
        "gap_median": statistics.median(gaps),
        "external_auc_min_cell": min(allx),
    }
    for d in ("busbra", "busi"):
        ind = statistics.mean(m(v, d, 0) for v in variants)
        crs = statistics.mean(m(v, d, 1) for v in variants)
        rec["counterexample_by_direction"][d] = {
            "src_val_auc": ind, "external_auc": crs, "gap": ind - crs}
    for c, dd in (("BUSI", "busi"), ("BUSBRA", "busbra")):
        crs = statistics.mean(statistics.mean(fin[v][dd].values()) for v in variants)
        rec["final_pt_robustness"][c] = {
            "external_auc_final_pt": crs,
            "external_auc_erm_pt": statistics.mean(coh[c][1])}
    outp = ROOT / "runs" / "diag" / "transfer_gap_hook.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
