"""一次性：落**逐条目**二阶 cues 缓存，供任意子集组合复算。

为什么需要它
------------
`diagnose_trivial_cues_tier2.py` 只落**聚合**结果（各单 cue AUC + `ALL_2nd`），
不落逐条目特征。于是"去掉 scale 的纯纹理子集 AUC 是多少"这种问题
**无法从已有产物回答**，只能重算全部图像。

2026-10-07 需要这个数字来落 v5 的幕②限定口径：
  · `ALL_2nd` = 0.9864 / 0.9954 —— **但含 scale**
  · scale 本身 0.967 / 0.955，是**采集协议**线索（有效分辨率），不是纹理
  · 故不能笼统称"纯纹理统计量能够辨库"
本脚本把逐条目 cues 落盘，之后任何子集都能秒算，不必再动图像。

产物：runs/diag/tier2_cues_perentry.npz
      （每个队列的 cues 矩阵 + patient_id + 队列名，供分组 CV 用）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 入口自护（preflight 坑1）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from breastdg.data import registry                     # noqa: E402
from breastdg.data.dedup import dedupe_entries         # noqa: E402
from breastdg.data.preprocess import PreprocConfig     # noqa: E402
from diagnose_trivial_cues_tier2 import NAMES, cues2   # noqa: E402

OUT = Path("runs/diag/tier2_cues_perentry.npz")


def main() -> int:
    cfg = PreprocConfig(size=256, geometry="p4")
    busi, _ = dedupe_entries(registry.busi_entries())
    cohorts = [("BUSI", busi), ("BUSBRA", registry.busbra_entries()),
               ("BUS-UCLM", registry.busuclm_entries())]
    payload = {}
    print(f"逐条目落二阶 cues（列序 = {NAMES}）")
    for nm, ents in cohorts:
        C = np.array([cues2(e, cfg) for e in ents], float)
        pid = np.array([str(e.get("patient_id") or e["image"]) for e in ents])
        payload[f"{nm}__cues"] = C
        payload[f"{nm}__pid"] = pid
        print(f"  {nm:10s} n={len(ents):5d}  "
              + "  ".join(f"{k}={C[:, i].mean():9.3f}"
                          for i, k in enumerate(NAMES)), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    print(f"\n✓ 已写入 {OUT}（{OUT.stat().st_size / 1e6:.2f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
