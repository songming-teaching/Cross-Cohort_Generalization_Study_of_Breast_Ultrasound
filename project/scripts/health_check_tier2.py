"""tier2 健康检查：真失败、dom loss 契约、结果是否异常。"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

LOG = Path("reports/phase3a_tier2.log")
T2 = ("ema", "render_aug_ema", "scale_aug", "scale_render_aug",
      "fdnet_gate_hf", "fdnet_gate_dual")

print("=" * 74)
print("1) 日志中的**真**失败（排除 lr_scheduler 的 UserWarning 噪声）")
print("=" * 74)
txt = LOG.read_text(encoding="utf-8", errors="replace")
lines = [l for l in txt.splitlines()
         if not l.startswith("C:\\") and "UserWarning" not in l
         and "warnings.warn" not in l]
real_fail = [l for l in lines if re.search(r"FAIL:|Traceback|RuntimeError|"
                                           r"ValueError|AttributeError|KeyError", l)]
print(f"  真失败行数: {len(real_fail)}")
for l in real_fail[-8:]:
    print("   ", l[:150])

print()
print("=" * 74)
print("2) F26.1 契约：dom loss 是否停在 ln2≈0.693 附近（防 GRL 再次静默失效）")
print("=" * 74)
rows = []
suspicious = []
for f in sorted(glob.glob("runs/dev_screen/*/*/seed*/**/history.json",
                          recursive=True)):
    parts = Path(f).parts            # ('runs','dev_screen',variant,dir,'seedN',...,'history.json')
    v, d, s = parts[2], parts[3], parts[4]
    if v not in T2:
        continue
    h = json.loads(Path(f).read_text(encoding="utf-8"))
    if not h:
        continue
    last = h[-1]
    dom = last.get("loss_dom", float("nan"))
    val = last.get("val_auc", float("nan"))
    g = last.get("gate")
    rows.append((v, d, s, dom, val, g))
    if dom < 0.1:
        suspicious.append((v, d, s, dom))

print(f"  {'变体':<18}{'方向':<8}{'种子':<7}{'末步dom':>9}{'末步val':>9}{'门g':>8}")
for v, d, s, dom, val, g in sorted(rows):
    gs = f"{g:.3f}" if g is not None else "-"
    flag = "  *** GRID 疑似失效 ***" if dom < 0.1 else ""
    print(f"  {v:<18}{d:<8}{s:<7}{dom:>9.4f}{val:>9.4f}{gs:>8}{flag}")
print(f"\n  dom<0.1 的可疑运行: {len(suspicious)}  {suspicious if suspicious else '✓ 无'}")

print()
print("=" * 74)
print("3) 结果分布（是否有异常值）")
print("=" * 74)
CSV = Path("runs/dev_screen/results.csv")
allrows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
           if str(r.get("steps", "")).strip() and int(r["steps"]) >= 3000]
t2 = [r for r in allrows if r["variant"] in T2]
print(f"  tier2 已完成: {len(t2)}/36")
import numpy as np
for v in T2:
    sub = [r for r in t2 if r["variant"] == v]
    if not sub:
        continue
    fin = np.array([float(r["dev_auc_final"]) for r in sub])
    erm = np.array([float(r["dev_auc_erm"]) for r in sub])
    print(f"  {v:<18}n={len(sub):<3}FINAL {fin.mean():.4f}±{fin.std():.4f}"
          f"   ERM {erm.mean():.4f}   Δ {(fin-erm).mean():+.4f}")
    if fin.min() < 0.45 or fin.max() > 0.95:
        print(f"      [WARN] 异常范围: [{fin.min():.4f}, {fin.max():.4f}]")

print()
print("=" * 74)
print("4) 登记：lr_scheduler 警告的含义")
print("=" * 74)
n_warn = txt.count("lr_scheduler.step()` before")
print(f"  出现次数: {n_warn}（每次运行约 1 次，属**非致命**）")
print("  成因：AMP 下若某步梯度为 inf/nan，scaler.step() 会**跳过** opt.step()，")
print("        而随后无条件调用 sched.step() → 触发该警告。")
print("  影响：极小（仅调度器多推进一格，且只在 inf/nan 步发生）。")
print("  处置：**不改**（tier2 运行中，F22），登记备查。")
