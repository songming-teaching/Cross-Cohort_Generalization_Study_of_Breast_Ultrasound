"""控制诊断：0.987 的域可分性来自"签名"，还是来自平凡线索（黑边/亮度）？

动机
----
域探针在冻结的 ERM 特征上给出 AUC 0.987（BUSI↔BUS-UCLM 与 BUSI↔BUSBRA 都是）。
这个数值高得可疑：真实的高频纹理签名在深度特征里通常 0.7–0.95。
更可疑的是它与判别器损失矛盾——判别器收敛到 ln2（随机水平），
若域真的可分到 0.987，判别器应轻松到 ≈0。

排查的平凡线索（letterbox 之后可直接影响特征）：
  L1 黑边占比（letterbox 的填充面积 —— 直接编码原始宽高比）
  L2 原始尺寸 / 宽高比
  L3 全局均值亮度
  L4 全局标准差（对比度）
  L5 亮度分位数（5/50/95）

判定：若仅用这些平凡线索就能复现 ≈0.98 的 AUC，则探针测到的不是"签名"，
而是**预处理引入的几何/色调线索**。这与审稿意见 2（OASBUD 预处理争议）
和理论子节的"色调签名"直接相关。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig, letterbox, tone_normalize

SIZE = 256
N_FOLDS = 5


def image_stats(path: str, tone_norm: bool = True) -> dict:
    with Image.open(path) as im:
        w0, h0 = im.size
        g = np.asarray(im.convert("L"))
        if g.shape[0] > g.shape[1]:
            g = np.rot90(g, 1)
            w0, h0 = h0, w0
        if tone_norm:
            g = tone_normalize(g, 5.0, 95.0)
        lb = np.asarray(letterbox(Image.fromarray(g), SIZE, 0))
    black = float((lb == 0).mean())
    return dict(w0=w0, h0=h0, aspect=w0 / max(h0, 1),
                pad_frac=black,
                mean=float(lb.mean()), std=float(lb.std()),
                p05=float(np.percentile(lb, 5)), p50=float(np.percentile(lb, 50)),
                p95=float(np.percentile(lb, 95)))


def collect(entries, tone_norm=True):
    rows = [image_stats(e["image"], tone_norm) for e in entries]
    return rows


def cv_auc(X, y, g, seed=0):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in sgkf.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        if len(set(y[te].tolist())) > 1:
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs)), aucs


def pair_probe(rows_a, rows_b, ent_a, ent_b, feats, name):
    X = np.array([[r[f] for f in feats] for r in rows_a + rows_b], dtype=float)
    y = np.r_[np.zeros(len(rows_a), int), np.ones(len(rows_b), int)]
    g = np.array([e["patient_id"] for e in ent_a + ent_b])
    auc, folds = cv_auc(X, y, g)
    print(f"  {name:34s} AUC = {auc:.4f}")
    return auc


def main():
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()

    print("=== 原始尺寸分布 ===")
    for nm, ent in (("BUSI", busi), ("BUSBRA", busbra), ("BUS-UCLM", busuclm)):
        dims = []
        for e in ent[:200]:
            with Image.open(e["image"]) as im:
                dims.append(im.size)
        a = np.array(dims)
        print(f"  {nm:10s} n={len(ent):5d}  w:{a[:,0].min()}–{a[:,0].max()}  "
              f"h:{a[:,1].min()}–{a[:,1].max()}  "
              f"宽高比 {a[:,0].mean()/a[:,1].mean():.3f}")

    print("\n=== 计算每图的平凡线索（tone_norm=True，与训练一致）===")
    r_busi = collect(busi)
    r_busbra = collect(busbra)
    r_busuclm = collect(busuclm)
    for nm, rows in (("BUSI", r_busi), ("BUSBRA", r_busbra), ("BUS-UCLM", r_busuclm)):
        pf = np.array([r["pad_frac"] for r in rows])
        mn = np.array([r["mean"] for r in rows])
        print(f"  {nm:10s} 黑边占比 {pf.mean():.4f}±{pf.std():.4f}   "
              f"亮度 {mn.mean():6.2f}±{mn.std():.2f}")

    print("\n=== 仅用平凡线索的域可分性 ===")
    groups = [
        ("BUSI vs BUS-UCLM", r_busi, r_busuclm, busi, busuclm),
        ("BUSI vs BUSBRA", r_busi, r_busbra, busi, busbra),
    ]
    FEATS = {
        "全 8 项": ["w0", "h0", "aspect", "pad_frac", "mean", "std", "p05", "p50", "p95"],
        "仅几何(黑边+宽高比)": ["aspect", "pad_frac"],
        "仅黑边": ["pad_frac"],
        "仅亮度统计": ["mean", "std", "p05", "p50", "p95"],
        "仅均值亮度": ["mean"],
    }
    out = {}
    for gname, ra, rb, ea, eb in groups:
        print(f"\n  [{gname}]")
        out[gname] = {}
        for fname, feats in FEATS.items():
            out[gname][fname] = pair_probe(ra, rb, ea, eb, feats, fname)

    # 对照：不做 tone_norm 时的亮度线索强度
    print("\n=== 对照：不施加 tone_norm（看色调标准化压掉了多少）===")
    nb = collect(busi, tone_norm=False)
    nu = collect(busuclm, tone_norm=False)
    nbr = collect(busbra, tone_norm=False)
    for gname, ra, rb, ea, eb in [
        ("BUSI vs BUS-UCLM", nb, nu, busi, busuclm),
        ("BUSI vs BUSBRA", nb, nbr, busi, busbra)]:
        for fname in ("仅均值亮度", "仅亮度统计"):
            pair_probe(ra, rb, ea, eb, FEATS[fname], f"{fname}（无tone_norm）")

    p = Path("runs/diag/trivial_cues.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
