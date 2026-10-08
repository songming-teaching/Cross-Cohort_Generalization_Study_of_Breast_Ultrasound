"""B：E8 谱分析（dev 侧队列 + 已完成的合成自验）。

范围限制（F25 / F26 裁定）
--------------------------
**只跑 dev 侧队列**：busi / busbra / busuclm。
**不碰 OASBUD / BrEaST** —— 它们的谱量（θ̂/R/M/d_A/对齐余弦）一律留锁箱后
由 E8 统一出数；提前看一眼就等于用锁箱信息污染后续分析设计。

用到的两处已修（否则结果无效）
------------------------------
1. `run_e8.standardize` —— 必须标准化**并剪掉死维**：
   penultimate 特征约 25% 的维方差 <1e-4、1.1% 接近 0，直接除以近零 sd 会把
   数值噪声放大成人为特征值（实测 λ₁ 由 1.861 暴涨到 342）。
2. `e8.bootstrap_ci` —— 必须用 m-out-of-n 无放回 + c 随 m 修正（F24）：
   朴素有放回使 λ₁ 虚高 27%，CI 完全不含点估计。

须报告的限定（F33，**未解决**）
-------------------------------
实测谱形**不是** (A2) 的「单尖峰 + 平坦体部」，而是「低秩簇 + 衰减尾部」：
MP 体部边缘之上的特征值数在 BUSI 为 **120**、BUSBRA 为 **332**（(A2) 预测 = 1）。
故本批产出的 R/θ̂ 必须按"**主导尖峰**"解读（理论 S10 第 1 条的"尖峰簇"退路），
不可按"唯一尖峰"解读。
"""
from __future__ import annotations

import json
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
OUT_DIR = config.PROJECT_ROOT / "runs" / "diag" / "e8_dev"

# 主配置（simple-wins 产出）+ 参照基线
CONFIGS = ["fdnet_gate_dual", "baseline"]
STAGES = [("final", "FINAL（主口径，F16）"), ("best_erm", "Stage A 参照")]
COHORTS = ["busi", "busbra", "busuclm"]


def load_cohort(name: str):
    if name == "busi":
        return dedupe_entries(registry.busi_entries())[0]
    return registry.ALL_PUBLIC[name]()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_pp = PreprocConfig(size=256, geometry="p4")
    allrows = []

    for variant in CONFIGS:
        for which, which_desc in STAGES:
            for cohort in COHORTS:
                try:
                    ck = find_ckpt(variant, "busbra", 42, f"{which}.pt")
                except FileNotFoundError as e:
                    print(f"[skip] {variant}/{which}: {e}", flush=True)
                    continue
                sd = torch.load(ck, map_location="cpu")
                saved_cfg = sd.get("cfg") or {"model": "resnet50_cbam"}
                m = build_model(saved_cfg)
                m.load_state_dict(sd["model"])

                # 尺寸取自检查点——FDNet-v2 训练口径 512，喂 256 会使
                # 子带 256→128、末层特征图 8×8→4×4，实测 PR 差 44%。
                sz = int(saved_cfg.get("image_size", 256))
                cfg_pp = PreprocConfig(size=sz,
                                       geometry=saved_cfg.get("geometry", "p4"))

                ents = load_cohort(cohort)
                F_raw = extract(m, ents, cfg_pp, "penultimate")
                F = standardize(F_raw)
                y = np.array([int(e["label"]) for e in ents])
                dom = np.full(len(ents), -1)
                lesion, lvl = _lesion_codes(ents)
                meta = {"model": sd.get("cfg", {}).get("model", "?"),
                        "stage": which, "branch": "penultimate",
                        "cohort": cohort, "seed": 42,
                        "std": "per_domain_pruned", "projection": "native"}
                # 蒙特卡洛次数：p=2048 且 n=1875 时单次谱分解约 1.5 s，
                # B 过大（400+300+300）会让单次分析耗时 ~25 min、整批超 5 h。
                # 取 300/200/200 在阈值与 CI 精度上足够（95% 分位的 MC 误差可控），
                # 且整批约 1.5–2.5 h。
                row = E8.analyze(F, y, dom, lesion, meta,
                                 B_mc=300, B_boot=200, B_perm=200)
                d = dict(row.__dict__)
                d["variant"] = variant
                allrows.append(d)
                print(f"[{variant}/{which}/{cohort}] n={d['n']} p={d['p']} "
                      f"c={d['c']:.3f}  R={d['R']:.4f} (thr {d['R_thr']:.4f}) "
                      f"det={d['detected']}  theta={d['theta']:.3f} "
                      f"[{d['theta_ci_lo']:.3f},{d['theta_ci_hi']:.3f}]  "
                      f"M={d['M']:.5f}", flush=True)
                del m
                torch.cuda.empty_cache()

    p = OUT_DIR / "e8_dev_results.json"
    p.write_text(json.dumps(allrows, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    print(f"\n已写入 {p}（{len(allrows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
