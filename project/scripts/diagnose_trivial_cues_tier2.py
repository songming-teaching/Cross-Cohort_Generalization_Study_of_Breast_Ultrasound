"""第二梯队平凡线索（无标签）——排查 BUSI↔BUS-UCLM 残差 0.8419 偏高。

背景（《阶段3a检查总结_P4.md》§3.3）
-----------------------------------
BUSI↔BUS-UCLM 的深度特征 AUC 高达 0.9988，回归掉第一梯队平凡线索
（黑边/长宽比/亮度）后仍有 0.8419，高于"高频纹理签名"通常预期（0.7–0.95）。
需排查是否还有**未被列入**的平凡线索可解释它。

为什么这组线索特别有用：它们**全部无标签**，故不受 §2.3（BUS-UCLM 标签隔离）
限制，可直接用于 BUSI↔BUS-UCLM 那对——而标签分层做不到（见 F19）。

线索（均在 P4 处理后空间上计算）
--------------------------------
  scale        内容在画布上的缩放比（决定有效分辨率/斑点尺度）
  speckle      高频能量占比（P95−P50 或用 Laplacian 方差），斑点尺度代理
  snr          内容区 mean/std（信噪比代理）
  entropy      灰度直方图熵（纹理复杂度）
  grad         平均梯度幅值（边缘锐度）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.geometry import content_bbox, plan
from breastdg.data.preprocess import PreprocConfig, preprocess_image

SIZE = 256


def cues2(entry: dict, cfg: PreprocConfig) -> list[float]:
    """返回 [scale, speckle, snr, entropy, grad]，全部无标签。"""
    with Image.open(entry["image"]) as im:
        g = np.asarray(im.convert("L"))
    if g.shape[0] > g.shape[1]:
        g = np.rot90(g, 1)
    p = plan(g, size=cfg.size)
    img = preprocess_image(entry["image"], cfg, as_rgb=False).astype(np.float64)

    # 内容区（去掉几何填充，避免填充主导统计）
    nh = max(1, int(round((p.bbox[1] - p.bbox[0]) * p.scale)))
    nw = max(1, int(round((p.bbox[3] - p.bbox[2]) * p.scale)))
    c = img[p.pad_h:p.pad_h + nh, p.pad_w:p.pad_w + nw]
    if c.size == 0:
        c = img

    # 1 缩放比
    scale = float(p.scale)
    # 2 斑点尺度代理：Laplacian 方差（高频能量）
    lap = np.zeros_like(c)
    lap[1:-1, 1:-1] = (4 * c[1:-1, 1:-1] - c[:-2, 1:-1] - c[2:, 1:-1]
                       - c[1:-1, :-2] - c[1:-1, 2:])
    speckle = float(lap.var())
    # 3 信噪比代理
    snr = float(c.mean() / (c.std() + 1e-6))
    # 4 直方图熵
    h, _ = np.histogram(c, bins=64, range=(0, 255))
    pr = h.astype(np.float64) / max(h.sum(), 1)
    pr = pr[pr > 0]
    entropy = float(-(pr * np.log2(pr)).sum())
    # 5 平均梯度
    gy, gx = np.gradient(c)
    grad = float(np.sqrt(gx ** 2 + gy ** 2).mean())
    return [scale, speckle, snr, entropy, grad]


NAMES = ["scale", "speckle", "snr", "entropy", "grad"]


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    cfg = PreprocConfig(size=SIZE, geometry="p4")
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()

    print("计算第二梯队线索（无标签）...", flush=True)
    C = {}
    for nm, ents in (("BUSI", busi), ("BUSBRA", busbra), ("BUS-UCLM", busuclm)):
        C[nm] = np.array([cues2(e, cfg) for e in ents], float)
        print(f"  {nm:10s} n={len(ents):5d}  " +
              "  ".join(f"{k}={C[nm][:,i].mean():9.3f}" for i, k in enumerate(NAMES)),
              flush=True)

    def cv_auc(X, y, g, seed=0):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        a = []
        for tr, te in sgkf.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
            if len(set(y[te].tolist())) > 1:
                a.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
        return float(np.mean(a))

    out = {}
    print("\n=== 各线索单独的域可分性 ===")
    for pair, other, ent_b in (("BUSI_vs_BUS-UCLM", "BUS-UCLM", busuclm),
                               ("BUSI_vs_BUSBRA", "BUSBRA", busbra)):
        print(f"\n[{pair}]")
        y = np.r_[np.zeros(len(busi), int), np.ones(len(ent_b), int)]
        g = np.array([e["patient_id"] for e in busi] +
                     [e["patient_id"] for e in ent_b])
        res = {}
        for i, k in enumerate(NAMES):
            X = np.concatenate([C["BUSI"][:, i:i + 1], C[other][:, i:i + 1]], 0)
            res[k] = cv_auc(X, y, g)
            print(f"  {k:9s} AUC = {res[k]:.4f}")
        Xall = np.concatenate([C["BUSI"], C[other]], 0)
        res["ALL_2nd"] = cv_auc(Xall, y, g)
        print(f"  {'全部 5 项':9s} AUC = {res['ALL_2nd']:.4f}")
        out[pair] = res

    p = Path("runs/diag/trivial_cues_tier2.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
