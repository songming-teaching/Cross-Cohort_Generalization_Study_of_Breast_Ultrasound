#!/usr/bin/env python3
"""锁箱级「源域 → 外部」落差：与 `verify_transfer_gap.py` 并列的**第二级**。

两级落差的关系（写论文时务必分清，**不可当同一个数字**）
--------------------------------------------------------
| | 级 1（dev 钩子） | 级 2（本脚本，锁箱） |
|---|---|---|
| 脚本 | `verify_transfer_gap.py` | `verify_lockbox_gap.py`（本文件） |
| 数据 | `runs/dev_screen/results.csv` | `runs/lockbox/*/seed*/**/thresholds.json` + `runs/lockbox_eval/summary_full_card.csv` |
| 源域 | 单个库（BUSI 或 BUSBRA） | **BUSI + BUSBRA 合并** |
| 外部 | 另一个库（**双向互换**） | OASBUD / BrEaST（**单向**，不互换） |
| 检查点 | `best_erm.pt` | `final.pt` |
| 结果 | 0.8863 → 0.7246，+0.1617，32/32 | 0.8721 → 0.8098/0.6023/0.8029 |

> **为什么两级都要报**：级 1 是"同检查点、跨库互换"，回答"落差是不是某个
> 切分口径的产物"；级 2 是"源域合并 → 两个真正的外部队列"，回答"这个落差
> 在**没见过的**外部集上还在不在"。两级同向、量级不同，比单报一级更稳。

同检查点论证（本脚本最关键的一步）
----------------------------------
`thresholds.json` 的 `auc_val` 由 `freeze_thresholds.py` 产出，而该脚本的
`find_ckpts` **只收 `*/final.pt`**（见其 docstring 第 73–74 行：明确不收
val-selected 的 `best_erm.pt`）。`evaluate_lockbox.find_ckpts` 同样只收
`*/final.pt`。⇒ 两边**是同一个检查点**，相减才是干净的落差。

⚠ 一个易错点
------------
`thresholds.json` 里的 `seed` 字段是**自举种子**（恒为 20261006），
**不是**运行种子。运行种子在**路径**里（`.../seed<N>/<runid>/thresholds.json`）。
若按 json 的 `seed` 去配对，49 个会塌成 7 个键、且全部配对失败。

用法
----
    python scripts/verify_lockbox_gap.py
产物
----
    runs/diag/transfer_gap_lockbox.json
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")     # 入口自护（preflight 坑1）

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from breastdg import config                   # noqa: E402

LOCKBOX = config.RUNS_DIR / "lockbox"
SUMMARY = config.RUNS_DIR / "lockbox_eval" / "summary_full_card.csv"
OUT = config.RUNS_DIR / "diag" / "transfer_gap_lockbox.json"

# 自举种子（用于断言 json 里的 seed 确为自举种子，防误当运行种子）
SEED_BOOTSTRAP = 20261006


def load_src_val() -> dict[tuple[str, int], float]:
    """读锁箱 thresholds.json → {(variant, run_seed): auc_val}。

    `run_seed` 取自**路径**，不是 json 的 `seed`（见模块 docstring ⚠）。
    """
    out: dict[tuple[str, int], float] = {}
    files = sorted(LOCKBOX.rglob("thresholds.json"))
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        rel = p.relative_to(LOCKBOX).parts        # <variant>/seed<N>/<runid>/...
        variant = rel[0]
        run_seed = int(rel[1].replace("seed", ""))
        # 断言：json 的 seed 是自举种子（若哪天改成运行种子，这里会响）
        assert int(d["seed"]) == SEED_BOOTSTRAP, (
            f"{p} 的 json seed={d['seed']} 不是自举种子 {SEED_BOOTSTRAP}；"
            f"本脚本的 run_seed 取自路径，此断言防两者被混淆")
        out[(variant, run_seed)] = float(d["auc_val"])
    return out


def load_external() -> dict[tuple[str, int], dict[str, float]]:
    """读锁箱评估卡汇总 → {(variant, seed): {arm: auc}}。"""
    out: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    if not SUMMARY.exists():
        raise SystemExit(f"  ✗ 找不到 {SUMMARY}\n"
                         f"     先跑 evaluate_lockbox.py --full-card")
    for r in csv.DictReader(open(SUMMARY, encoding="utf-8")):
        out[(r["variant"], int(r["seed"]))][r["arm"]] = float(r["auc"])
    return out


def main() -> int:
    src = load_src_val()
    ext = load_external()
    arms = ["oasbud_corrected", "oasbud_uncorrected", "breast"]

    print("=" * 78)
    print("锁箱级落差：源域（BUSI+BUSBRA，final.pt）→ 外部队列（同一 final.pt）")
    print("=" * 78)
    print(f"  thresholds.json {len(src)} 个；评估卡 {len(ext)} 个检查点")

    keys = sorted(set(src) & set(ext))
    missing = sorted(set(src) ^ set(ext))
    print(f"  配对成功 {len(keys)} 个检查点"
          + (f"；⚠ 未配对 {len(missing)}: {missing[:5]}" if missing else ""))
    if not keys:
        raise SystemExit("  ✗ 无一配对 —— 检查两边的检查点集合是否一致")

    per_arm: dict[str, dict] = {}
    for arm in arms:
        d = [(src[k], ext[k][arm]) for k in keys if arm in ext[k]]
        if not d:
            continue
        gaps = [a - b for a, b in d]
        per_arm[arm] = {
            "src_val_auc": statistics.mean(a for a, _ in d),
            "external_auc": statistics.mean(b for _, b in d),
            "gap": statistics.mean(gaps),
            "gap_median": statistics.median(gaps),
            "gap_min": min(gaps), "gap_max": max(gaps),
            "n_positive": sum(1 for g in gaps if g > 0),
            "n_total": len(gaps),
            "n_cells": len(d),
        }
        v = per_arm[arm]
        print(f"\n  {arm}")
        print(f"    源域val {v['src_val_auc']:.4f} → 外部 {v['external_auc']:.4f}"
              f"   落差 {v['gap']:+.4f}   {v['n_positive']}/{v['n_total']} 同号")
        print(f"    逐检查点落差范围 [{v['gap_min']:+.4f}, {v['gap_max']:+.4f}]"
              f"  中位 {v['gap_median']:+.4f}")

    # 核心对比：两条渲染管线的**落差**之比（幕④ 的锁箱级佐证）
    contrast = None
    if "oasbud_corrected" in per_arm and "oasbud_uncorrected" in per_arm:
        c, u = per_arm["oasbud_corrected"], per_arm["oasbud_uncorrected"]
        contrast = {
            "gap_corrected": c["gap"], "gap_uncorrected": u["gap"],
            "gap_ratio_unc_over_cor": u["gap"] / c["gap"] if c["gap"] else None,
            "auc_delta_cor_minus_unc": c["external_auc"] - u["external_auc"],
        }
        print(f"\n  两臂落差对比（幕④ 的锁箱级佐证）：")
        print(f"    corrected   落差 {c['gap']:+.4f}")
        print(f"    uncorrected 落差 {u['gap']:+.4f}"
              f"   = corrected 的 {contrast['gap_ratio_unc_over_cor']:.2f}×")
        print(f"    外部 AUC 差 (cor − unc) = {contrast['auc_delta_cor_minus_unc']:+.4f}")

    rec = {
        "script": "scripts/verify_lockbox_gap.py",
        "level": 2,
        "level_note": ("级 2（锁箱）。级 1 见 verify_transfer_gap.py / "
                       "runs/diag/transfer_gap_hook.json（dev，跨库互换）。"
                       "两级口径不同，**不是同一个数字**。"),
        "inputs": {
            "src_val": "runs/lockbox/*/seed*/**/thresholds.json（字段 auc_val）",
            "external": "runs/lockbox_eval/summary_full_card.csv",
        },
        "same_checkpoint": ("两侧均为 final.pt —— freeze_thresholds 与 "
                            "evaluate_lockbox 的 find_ckpts 都只收 */final.pt；"
                            "故相减是同检查点落差（非同口径不可减）"),
        "source_domain": "BUSI + BUSBRA（合并，按 splits/source_busi_busbra.json 划分）",
        "run_seed_from": "路径（thresholds.json 的 seed 字段是自举种子，非运行种子）",
        "bootstrap_seed": SEED_BOOTSTRAP,
        "n_cells": len(keys),
        "per_arm": per_arm,
        "two_arm_contrast": contrast,
        "stage": "lockbox",
        "wording": ("报『consistent with』；级 2 是单向（源→外部，不互换），"
                    "与级 1 的跨库互换是不同对照，不得混用同一句话"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {OUT}")
    return 0


if __name__ == "__main__":      # Windows spawn 守卫（preflight 坑3）
    sys.exit(main())
