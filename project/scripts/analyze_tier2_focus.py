"""tier2 结果聚焦分析：分离"架构差异"与"频率门控"。

问题
----
`fdnet_gate_dual` 显著优于 `baseline`（Δ=+0.0259, CI=[+0.0080,+0.0437]）。
但二者**架构不同**：
  baseline        = ResNet50-CBAM, 256 输入, 单通路 GRL
  fdnet_gate_dual = FDNet-v2 (ResNet18 双通路), 512 输入, HF 全强度 + LL 弱对抗(0.1×)
故该差异**不能**归因于"频率门控"——架构、输入分辨率、主干都变了。

**干净的对照**只有一个：`fdnet_gate_hf` vs `fdnet_gate_dual`——
同架构（FDNet-v2）、同输入（512）、同主干，**只差 GRL 的作用范围**：
  hf   = GRL 仅挂高频通路（"频率门控"假说所主张的）
  dual = HF 全强度 + LL 弱对抗(λ_L=0.1)

本脚本回答：
  Q1 dual 优于 hf 吗？（= 频率门控假说的方向性检验）
  Q2 fdnet 的优势是否只来自一个方向/一个种子（稳健性）？
  Q3 tier2 内部各变体相对 baseline 的增量（同架构内可比）
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from breastdg import config

CSV = config.PROJECT_ROOT / "runs" / "dev_screen" / "results.csv"
DIRS = ["busbra", "busi"]
SEEDS = [42, 123, 2026]


def load() -> dict[tuple[str, str, int], float]:
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if str(r.get("steps", "")).strip() and int(r["steps"]) >= 3000]
    out = {}
    for r in rows:
        v = r.get("dev_auc_final")
        if v not in (None, "", "nan"):
            out[(r["variant"], r["direction"], int(r["seed"]))] = float(v)
    return out


def paired(fin, va, vb) -> np.ndarray:
    """逐 (方向×种子) 配对求 va − vb。"""
    d = [fin[(va, dr, s)] - fin[(vb, dr, s)]
         for dr in DIRS for s in SEEDS
         if (va, dr, s) in fin and (vb, dr, s) in fin]
    return np.array(d)


def hw(a: np.ndarray) -> float:
    return float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")


def report(tag: str, a: np.ndarray, vs: str = "") -> dict:
    m, h = float(a.mean()), hw(a)
    lo, hi = m - h, m + h
    if not np.isfinite(h):
        verdict = "样本不足"
    elif lo > 0:
        verdict = "**显著为正**"
    elif hi < 0:
        verdict = "**显著为负**"
    else:
        verdict = "无显著差"
    print(f"  {tag:<44}Δ={m:+.4f}  ±{h:.4f}  [{lo:+.4f},{hi:+.4f}]  {verdict}")
    return dict(tag=tag, mean=m, hw=h, lo=lo, hi=hi, verdict=verdict, n=len(a))


def main() -> int:
    fin = load()
    print(f"读入 {len(fin)} 条 FINAL 结果\n")

    print("=" * 96)
    print("Q1 频率门控的**干净**检验（同架构 FDNet-v2，只差 GRL 作用范围）")
    print("=" * 96)
    d_dual_hf = paired(fin, "fdnet_gate_dual", "fdnet_gate_hf")
    r1 = report("dual(GRL: HF+LL弱) − hf(GRL: 仅HF)", d_dual_hf)
    print(f"\n  注：正 = dual 更好 = 『GRL 也作用于 LL（弱）』有益；")
    print(f"      负 = hf 更好 = 『只在高频加对抗』有益（= 频率门控假说方向）")

    print("\n" + "=" * 96)
    print("Q2 稳健性：fdnet_gate_dual vs baseline 的增量是否只来自某方向/某种子")
    print("=" * 96)
    for dr in DIRS:
        a = np.array([fin[("fdnet_gate_dual", dr, s)] - fin[("baseline", dr, s)]
                      for s in SEEDS])
        print(f"  方向 {dr:<8}Δ={a.mean():+.4f}  各种子 "
              + " ".join(f"{x:+.4f}" for x in a))
    print()
    for s in SEEDS:
        a = np.array([fin[("fdnet_gate_dual", dr, s)] - fin[("baseline", dr, s)]
                      for dr in DIRS])
        print(f"  种子 {s:<8}Δ={a.mean():+.4f}  各方向 "
              + " ".join(f"{x:+.4f}" for x in a))
    # 留一检验：去掉任一条，结论是否翻转
    allp = paired(fin, "fdnet_gate_dual", "baseline")
    print(f"\n  留一检验（去掉任一格后，95% CI 下界是否仍 > 0）：")
    flips = []
    for i in range(len(allp)):
        rest = np.delete(allp, i)
        lo = rest.mean() - hw(rest)
        if lo <= 0:
            flips.append(i)
    print(f"    去掉 {len(flips)}/{len(allp)} 个格会使结论翻转"
          f"{'（脆弱）' if flips else '（稳健）'}")

    print("\n" + "=" * 96)
    print("Q3 tier2 各变体 vs baseline（fdnet 组与 ResNet 组分开看）")
    print("=" * 96)
    T2 = ["ema", "render_aug_ema", "scale_aug", "scale_render_aug",
          "fdnet_gate_hf", "fdnet_gate_dual"]
    print(f"  {'变体':<20}{'Δ vs baseline':>15}{'±CI95':>9}   判定")
    for v in T2:
        a = paired(fin, v, "baseline")
        m, h = float(a.mean()), hw(a)
        lo, hi = m - h, m + h
        verdict = ("**显著正**" if lo > 0 else "**显著负**" if hi < 0 else "无显著差")
        print(f"  {v:<20}{m:>+15.4f}{h:>9.4f}   {verdict}")

    print("\n" + "=" * 96)
    print("Q4 [WARN] 架构混淆的提醒（写论文必读）")
    print("=" * 96)
    print("  fdnet_gate_dual 相对 baseline 的增量**不可**归因于频率门控：")
    print("    · baseline        = ResNet50-CBAM, 256 输入, 单通路 GRL")
    print("    · fdnet_gate_dual = FDNet-v2(ResNet18 双通路), 512 输入, HF+LL 弱")
    print("  三处同时改变（主干 / 输入分辨率 / GRL 作用范围）。")
    print("  可归因于门控的**唯一**证据是 Q1（dual vs hf），其结论见上。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
