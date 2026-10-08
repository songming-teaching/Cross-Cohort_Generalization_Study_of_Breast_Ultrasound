"""一次性核查：E7 残差臂与 F13 归因表口径是否一致。

背景：E7 首跑给 AUC_res≈0.68，而归因表同一格给 7b 下界 0.8357。
怀疑 E7 把 trivial_cues 返回的**全部**列（含 scale_extra = speckle/grad/snr）
都回归掉了，而 scale_extra 是**纹理签名自身的载体**，不该当平凡线索回归。

本脚本在同一次特征提取上对比几种回归集，确定差异来源。
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from diagnose_domain_probe_p4 import (PCA_K, cv_auc, extract, find_ckpt,
                                      residualize, trivial_cues)
from run_e8 import build_model

def main() -> int:
    cfg = PreprocConfig(size=256, geometry="p4")
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    ca = trivial_cues(busi, cfg)
    cb = trivial_cues(busbra, cfg)
    print("trivial_cues 键:", list(ca.keys()))
    print("各键形状:", {k: v.shape for k, v in ca.items()}, flush=True)

    ck = find_ckpt("no_domainhead", "busbra", 42, "best_erm")
    sd = torch.load(ck, map_location="cpu")
    c = sd["cfg"]
    m = build_model(c)
    m.load_state_dict(sd["model"])
    pp = PreprocConfig(size=int(c.get("image_size", 256)),
                       geometry=c.get("geometry", "p4"))
    Fa = extract(m, busi, pp)
    Fb = extract(m, busbra, pp)
    y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]
    F = np.vstack([Fa, Fb])
    print(f"特征 F={F.shape}  y={y.shape}", flush=True)

    def cols(keys):
        return np.vstack([np.column_stack([cc[k] for k in keys])
                          for cc in (ca, cb)])

    sets = [
        ("深度特征（不回归）", None),
        ("7a 上界 = geom+tone(pad,aspect,bright)", ["pad", "aspect", "bright"]),
        ("7b 下界 = 7a+scale", ["pad", "aspect", "bright", "scale"]),
        ("E7 现状 = 全部键(含 scale_extra)",
         ["pad", "aspect", "bright", "scale", "scale_extra"]),
        ("仅 scale_extra（佐证）", ["scale_extra"]),
    ]
    print(f"\n{'回归集':44s} {'AUC':>8s}")
    for name, keys in sets:
        X = F if keys is None else residualize(F, cols(keys))
        auc = cv_auc(X, y, None, pca_k=PCA_K)
        print(f"{name:44s} {auc:8.4f}", flush=True)
    return 0


if __name__ == "__main__":      # Windows spawn 必需，否则 DataLoader 起 worker 即崩
    sys.exit(main())
