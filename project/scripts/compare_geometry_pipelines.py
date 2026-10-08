"""用"平凡线索可分性"这把尺子，量三种几何候选管线。

判据：一个管线的好坏，看它把域标识符（黑边/宽高比/亮度）压到多低。
诊断已证明这条尺子有效（letterbox 下平凡线索 AUC 0.9862 vs 深度特征 0.9866）。

候选：
  P1 letterbox（现状）              ：保比例 + 补黑边
  P2 content_direct                ：裁内容区 → 各向同性 → 直接 resize 到 256
                                     （无补边，但宽高比失真，失真率本身可能成为线索）
  P3 content_square                ：裁内容区 → 各向同性 → 保比例缩放后**中心方裁**
                                     （无补边、无失真）

另报每张图在**处理后空间**的平凡线索，供人工核查。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.oasbud_rf import RFGeometry, reconstruct_case, load_cases
from breastdg.data.preprocess import tone_normalize

SIZE = 256
N_FOLDS = 5
N_SAMPLE = 220


# ---------------------------------------------------------------------------
def load_gray(path: str) -> np.ndarray:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L"))
    if g.shape[0] > g.shape[1]:
        g = np.rot90(g, 1)
    return g


def content_bbox(g: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    """非黑内容区边界框。阈值取 全局最小 + 2% 动态范围。"""
    v = g.astype(np.float32)
    thr = v.min() + max(1.0, 0.02 * (v.max() - v.min()))
    m = v > thr
    rows, cols = np.any(m, axis=1), np.any(m, axis=0)
    if not rows.any() or not cols.any():
        return 0, g.shape[0], 0, g.shape[1]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1


def isotropic(g: np.ndarray, ratio: float) -> np.ndarray:
    """把横向按 ratio 重采样，使像素各向同性（ratio=1 时不变）。"""
    if abs(ratio - 1.0) < 1e-3:
        return g
    h, w = g.shape
    nw = max(1, int(round(w * ratio)))
    return np.asarray(Image.fromarray(g).resize((nw, h), Image.BILINEAR))


def pipe_letterbox(g: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    g = isotropic(g, ratio)
    h, w = g.shape
    s = SIZE / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = np.asarray(Image.fromarray(g).resize((nw, nh), Image.BILINEAR))
    out = np.zeros((SIZE, SIZE), np.uint8)
    out[(SIZE - nh) // 2:(SIZE - nh) // 2 + nh, (SIZE - nw) // 2:(SIZE - nw) // 2 + nw] = r
    return out


def pipe_content_direct(g: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    r0, r1, c0, c1 = content_bbox(g)
    g = g[r0:r1, c0:c1]
    g = isotropic(g, ratio)
    return np.asarray(Image.fromarray(g).resize((SIZE, SIZE), Image.BILINEAR))


def pipe_content_square(g: np.ndarray, ratio: float = 1.0) -> np.ndarray:
    r0, r1, c0, c1 = content_bbox(g)
    g = g[r0:r1, c0:c1]
    g = isotropic(g, ratio)
    h, w = g.shape
    s = SIZE / min(h, w)                      # 短边缩到 256，长边随之 ≥256
    nh, nw = max(SIZE, round(h * s)), max(SIZE, round(w * s))
    g = np.asarray(Image.fromarray(g).resize((nw, nh), Image.BILINEAR))
    y0, x0 = (nh - SIZE) // 2, (nw - SIZE) // 2
    return g[y0:y0 + SIZE, x0:x0 + SIZE]


PIPES = {"P1_letterbox": pipe_letterbox,
         "P2_content_direct": pipe_content_direct,
         "P3_content_square": pipe_content_square}


# ---------------------------------------------------------------------------
def cues(img: np.ndarray) -> dict:
    v = img.astype(np.float32)
    return dict(pad_frac=float((v == 0).mean()),
                mean=float(v.mean()), std=float(v.std()),
                p05=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)))


def cv_auc(X, y, g, seed=0):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    a = []
    for tr, te in sgkf.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        if len(set(y[te].tolist())) > 1:
            a.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(a))


def main():
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()
    rng = np.random.default_rng(0)
    ent = {"BUSI": [busi[i] for i in rng.choice(len(busi), min(N_SAMPLE, len(busi)), False)],
           "BUSBRA": [busbra[i] for i in rng.choice(len(busbra), min(N_SAMPLE, len(busbra)), False)],
           "BUS-UCLM": [busuclm[i] for i in rng.choice(len(busuclm), len(busuclm), False)]}

    # 预载灰度
    print("载入灰度图 ...", flush=True)
    gray = {k: [tone_normalize(load_gray(e["image"]), 5.0, 95.0) for e in v]
            for k, v in ent.items()}

    # OASBUD：物理像素比校正前后各一版
    try:
        recs = load_cases()
        geo = RFGeometry()
        oa_unc, oa_cor = [], []
        for r in recs[:60]:
            for view in (0, 1):
                img, _ = reconstruct_case(r, "bmod", geo, 60.0, False, view)
                oa_unc.append(tone_normalize((img * 255).astype(np.uint8), 5, 95))
                img2, _ = reconstruct_case(r, "bmod", geo, 60.0, True, view)
                oa_cor.append(tone_normalize((img2 * 255).astype(np.uint8), 5, 95))
        have_oasbud = True
        print(f"OASBUD: 未校正 {len(oa_unc)} / 已校正 {len(oa_cor)}", flush=True)
    except Exception as e:
        print(f"OASBUD 跳过: {e}", flush=True)
        have_oasbud = False

    results = {}
    for pname, fn in PIPES.items():
        print(f"\n{'='*72}\n{pname}\n{'='*72}", flush=True)
        C = {k: [cues(fn(g)) for g in v] for k, v in gray.items()}
        for k, rows in C.items():
            pf = np.array([r["pad_frac"] for r in rows])
            ar = np.array([r["mean"] for r in rows])
            print(f"  {k:10s} 黑边 {pf.mean():.4f}±{pf.std():.4f}  均值亮度 {ar.mean():6.2f}±{ar.std():.2f}",
                  flush=True)
        results[pname] = {}
        for gname, other in (("BUSI vs BUS-UCLM", "BUS-UCLM"), ("BUSI vs BUSBRA", "BUSBRA")):
            A, B = C["BUSI"], C[other]
            ea, eb = ent["BUSI"], ent[other]
            for fname, keys in (("全 5 项", ["pad_frac", "mean", "std", "p05", "p50", "p95"]),
                                ("仅黑边", ["pad_frac"]),
                                ("仅亮度统计", ["mean", "std", "p05", "p50", "p95"])):
                X = np.array([[r[k] for k in keys] for r in A + B], float)
                y = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
                gg = np.array([e["patient_id"] for e in ea + eb])
                auc = cv_auc(X, y, gg)
                results[pname][f"{gname}|{fname}"] = auc
                print(f"  [{gname}] {fname:12s} AUC = {auc:.4f}", flush=True)
        if have_oasbud:
            for tag, imgs in (("OASBUD未校正", oa_unc), ("OASBUD已校正", oa_cor)):
                pf = np.array([cues(fn(g))["pad_frac"] for g in imgs])
                print(f"  {tag:14s} 黑边 {pf.mean():.4f}", flush=True)

    p = Path("runs/diag/geometry_pipelines.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")

    print("\n" + "=" * 72)
    print("汇总：平凡线索可分性（越低越好；目标接近 0.5）")
    print(f"{'管线':<22}{'BUSI-UCLM 全项':>16}{'BUSI-BUSBRA 全项':>18}")
    for pname in PIPES:
        a = results[pname].get("BUSI vs BUS-UCLM|全 5 项", float("nan"))
        b = results[pname].get("BUSI vs BUSBRA|全 5 项", float("nan"))
        print(f"{pname:<22}{a:>16.4f}{b:>18.4f}")


if __name__ == "__main__":
    main()
