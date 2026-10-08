#!/usr/bin/env python3
"""E8 锁箱臂聚合：49 格 × 3 队列 → 逐变体 × 逐队列的谱量表。

数据来源
--------
`runs/diag/e8_lockbox/e8_<variant>_seed<seed>.json`，每格 3 条记录
（`oasbud_corrected` / `oasbud_uncorrected` / `breast`）。由
`env/run_e8_lockbox.sh` 驱动 `run_e8.py` 产出，**一条记录一个队列**。

为什么单独聚合（而不是各队列混在一起）
--------------------------------------
三条队列的语义**不同**，混起来算一个均值毫无意义：
  · `oasbud_corrected` / `oasbud_uncorrected` —— 同一批 200 图的**两条渲染管线**
    （对数压缩 B 模式 vs 线性包络），是**对照关系**，必须成对看；
  · `breast` —— 独立队列，分布与 OASBUD 不同（n=252）。
故本表**逐队列成列**，并**额外给出两臂之差**（同一检查点、同一批图）。

⚠ 两臂**同源检测**（承 2026-10-08 缺陷 #11）
-------------------------------------------
若某格内两臂 `R` 与 `theta` **逐位相同**，则说明该格的臂名解析又退化成子串判定
（`"corrected" in "oasbud_uncorrected"` 为真）→ 两臂实读同一目录。
本脚本**主动检出并标红**，不静默出表。preflight 坑11 亦有同款动态闸门。

用法
----
    python scripts/aggregate_e8_lockbox.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")     # 入口自护（preflight 坑1）

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from breastdg import config                   # noqa: E402

SRC = config.RUNS_DIR / "diag" / "e8_lockbox"
OUT = config.RUNS_DIR / "diag" / "e8_lockbox_summary.json"

COHORTS = ("oasbud_corrected", "oasbud_uncorrected", "breast")
# 用于聚合的标量量（全部为谱学量，C1 跨架构限定所需）
SCALARS = ("R", "R_thr", "theta", "theta_thr", "M", "PR", "PR_frac",
           "er", "er_frac", "tk1", "tk5", "tk16", "tk32", "tk64",
           "K_above", "c", "sqrt_c_ref")


def parse_name(stem: str) -> tuple[str, str] | None:
    """`e8_<variant>_seed<seed>` → (variant, seed)。解析失败返回 None。

    ⚠ 不用 `split("_seed")[0]` 之外的猜测：变体名自身可能含下划线
    （`fdnet_gate_dual`、`efficientnet_v2_s`），故以**最后一个** `_seed` 切分。
    """
    if not stem.startswith("e8_"):
        return None
    body = stem[3:]
    i = body.rfind("_seed")
    if i < 0:
        return None
    return body[:i], body[i + len("_seed"):]


def main() -> int:
    files = sorted(SRC.glob("e8_*.json"))
    # 排除闸门探针等非正式产物（名字不以 seed<digits> 结尾的一律跳过）
    cells = []
    for f in files:
        p = parse_name(f.stem)
        if p is None or not p[1].isdigit():
            print(f"  [skip] 非正式产物: {f.name}")
            continue
        cells.append((f, p[0], int(p[1])))
    if not cells:
        print(f"尚无 E8 产物于 {SRC}")
        return 1

    # {variant: {cohort: {scalar: [values across seeds]}}}
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    same_arm_hits = []          # 两臂同源的格（缺陷 #11 复发信号）
    dup = []                    # 同 (variant, seed) 重复

    seen = set()
    for f, variant, seed in cells:
        if (variant, seed) in seen:
            dup.append(f"{variant}/seed{seed}")
        seen.add((variant, seed))
        rows = json.loads(f.read_text(encoding="utf-8"))
        by = {r["cohort"]: r for r in rows}
        a, b = by.get("oasbud_corrected"), by.get("oasbud_uncorrected")
        if a and b and a.get("R") == b.get("R") and a.get("theta") == b.get("theta"):
            same_arm_hits.append(f.name)
        for r in rows:
            for k in SCALARS:
                v = r.get(k)
                if isinstance(v, (int, float)) and np.isfinite(v):
                    acc[variant][r["cohort"]][k].append(float(v))

    variants = sorted(acc)
    n_seed = {v: len({s for _, vv, s in cells if vv == v}) for v in variants}

    # ---------- 打印逐变体 × 逐队列的表 ----------
    for v in variants:
        print("=" * 78)
        print(f"变体 {v}   （{n_seed[v]} 个种子）")
        print(f"  {'队列':20s} {'n':>5s} {'R':>9s} {'θ̂':>10s} {'M':>9s} "
              f"{'PR_frac':>8s} {'er_frac':>8s} {'tk16':>7s}")
        for c in COHORTS:
            if c not in acc[v]:
                continue
            d = acc[v][c]
            r = float(np.mean(d["R"])) if d.get("R") else float("nan")
            t = float(np.mean(d["theta"])) if d.get("theta") else float("nan")
            m = float(np.mean(d["M"])) if d.get("M") else float("nan")
            pr = float(np.mean(d["PR_frac"])) if d.get("PR_frac") else float("nan")
            er = float(np.mean(d["er_frac"])) if d.get("er_frac") else float("nan")
            tk = float(np.mean(d["tk16"])) if d.get("tk16") else float("nan")
            rs = d.get("R", [])
            sd = float(np.std(rs)) if rs else float("nan")
            print(f"  {c:20s} {len(rs):5d} {r:9.4f} {t:10.3f} {m:9.5f} "
                  f"{pr:8.3f} {er:8.3f} {tk:7.3f}   (R 逐种子 sd={sd:.3f})")
        # 两臂之差（同检查点、同批图 → 差是**配对**的，用逐种子配对差）
        a, b = acc[v].get("oasbud_corrected"), acc[v].get("oasbud_uncorrected")
        if a and b and len(a.get("R", [])) == len(b.get("R", [])):
            dR = np.array(a["R"]) - np.array(b["R"])
            dt = np.array(a["theta"]) - np.array(b["theta"])
            print(f"  {'两臂之差 (cor−unc)':20s} {'':5s} {dR.mean():9.4f} "
                  f"{dt.mean():10.3f}   逐种子 dR={np.round(dR, 3).tolist()}")

    # ---------- 落盘 ----------
    summary = {
        "n_cells": len(cells),
        "n_variants": len(variants),
        "seeds_per_variant": n_seed,
        "cohorts": list(COHORTS),
        "note": ("逐队列成列，**不合并** —— 三条队列语义不同（OASBUD 两臂是同一批图"
                 "的两条渲染管线，breast 是独立队列）。两臂之差为配对差。"),
        "per_variant": {
            v: {c: {k: {"mean": float(np.mean(vals)), "sd": float(np.std(vals)),
                        "n_seed": len(vals)}
                    for k, vals in acc[v][c].items()}
                for c in acc[v]}
            for v in variants},
    }
    if same_arm_hits:
        summary["TWO_ARM_IDENTICAL_CELLS"] = same_arm_hits
    if dup:
        summary["DUPLICATE_CELLS"] = dup
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print(f"已写入 {OUT}")
    print(f"格数 {len(cells)} / 变体 {len(variants)}")
    if same_arm_hits:
        print(f"\n⚠ 两臂**逐位相同**的格 {len(same_arm_hits)} 个（缺陷 #11 复发！）：")
        for s in same_arm_hits[:10]:
            print(f"    {s}")
        return 2
    print("两臂同源检测：0 格命中（正常）")
    if dup:
        print(f"⚠ 重复格: {dup}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
