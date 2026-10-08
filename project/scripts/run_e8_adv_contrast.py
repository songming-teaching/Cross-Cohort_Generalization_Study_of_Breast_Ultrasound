"""E8 对抗开/关对照：no_domainhead (λ=0) vs baseline (λ=1.0)。

**为什么需要这个脚本**（2026-10-05 核查发现的错误）
----------------------------------------------------
先前用 `best_erm → final` 当作 "Stage A（对抗前）→ Stage B（对抗后）" 的对照，
**这是错的**。核查证据：

1. `dann_alpha(step, 3000)` 实测：step 1000 → 0.931、1500 → 0.987、2500 → 0.9995。
   即 α 在第 1000 步已接近 1.0。
2. `best_erm` 是**按源域 val AUC 选**的，实测落点中位 = step 2000（baseline），
   对应 α ≈ 0.998——**已是对抗充分阶段**，不是"对抗前"。

故 `best_erm → final` 的谱量变化只反映"多训 500–1500 步"，
**不能**解释为"对抗对齐压掉签名"。

**正确的对照**：`no_domainhead`（λ_domain=0，无 GRL）vs `baseline`（λ=1.0）。
两者同配方（P4 几何 + 色调前端 + 3000 步），唯一差别是否施加对抗。
这才是"对抗开/关"，也正是配对 ΔAUC = −0.0414（最强单效应）的来源。

用法
----
  # GPU 忙时用 CPU 跑，避免与训练抢显存
  # ⚠ 必须用 -1，**不能用空串**：空串在 Git Bash 下对 torch 等同"未设置"，
  #   会静默继续用 GPU 并与训练抢显存（2026-10-05 实测踩到）。
  CUDA_VISIBLE_DEVICES=-1 python scripts/run_e8_adv_contrast.py

产物
----
  runs/diag/e8_adv_contrast/results.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "..")

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
import e8_rmt_analysis_v2 as E8
from run_e8 import build_model, extract, find_ckpt, standardize, _lesion_codes

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = config.PROJECT_ROOT / "runs" / "diag" / "e8_adv_contrast"

CONFIGS = ["no_domainhead", "baseline"]     # 对抗关 / 对抗开
COHORTS = ["busi", "busbra", "busuclm"]
SEEDS = [42, 123, 2026]


def load_cohort(name: str):
    if name == "busi":
        return dedupe_entries(registry.busi_entries())[0]
    return registry.ALL_PUBLIC[name]()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"设备 {DEV}", flush=True)
    allrows = []
    for variant in CONFIGS:
        for seed in SEEDS:
            for cohort in COHORTS:
                try:
                    ck = find_ckpt(variant, "busbra", seed, "final.pt")
                except FileNotFoundError as e:
                    print(f"[skip] {variant}/seed{seed}: {e}", flush=True)
                    continue
                sd = torch.load(ck, map_location="cpu")
                saved_cfg = sd.get("cfg") or {"model": "resnet50_cbam"}
                m = build_model(saved_cfg)
                m.load_state_dict(sd["model"])
                sz = int(saved_cfg.get("image_size", 256))
                cfg_pp = PreprocConfig(size=sz,
                                       geometry=saved_cfg.get("geometry", "p4"))
                ents = load_cohort(cohort)
                F_raw = extract(m, ents, cfg_pp, "penultimate")
                F = standardize(F_raw)
                y = np.array([int(e["label"]) for e in ents])
                dom = np.full(len(ents), -1)
                lesion, _ = _lesion_codes(ents)
                meta = {"model": saved_cfg.get("model", "?"), "stage": "final",
                        "branch": "penultimate", "cohort": cohort, "seed": seed,
                        "std": "per_domain_pruned", "projection": "native"}
                row = E8.analyze(F, y, dom, lesion, meta,
                                 B_mc=300, B_boot=200, B_perm=200)
                d = dict(row.__dict__)
                d["variant"] = variant
                allrows.append(d)
                print(f"[{variant}/seed{seed}/{cohort}] n={d['n']} p={d['p']} "
                      f"c={d['c']:.3f}  PR={d['PR']:.2f} "
                      f"PR_frac={d['PR_frac']:.4f}  er_frac={d['er_frac']:.4f}  "
                      f"tk16={d['tk16']:.3f}  K^={d['K_above']}", flush=True)
                del m
                if DEV.type == "cuda":
                    torch.cuda.empty_cache()

    p = OUT_DIR / "results.json"
    p.write_text(json.dumps(allrows, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    print(f"\n已写入 {p}（{len(allrows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
