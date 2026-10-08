"""几何方案定稿前的最后核查：内容裁剪是否切掉病灶 + 各管线保留多少内容。

决策依据（不是可分性汇总表，那张表在内容裁剪后已被"内部暗区"混淆）：
  1. 内容裁剪会丢掉多少面积？
  2. 保比例方裁(P3) 是否切到病灶？（用掩膜判定）
  3. OASBUD 各向同性校正后的尺寸是否已接近方形（决定校正能帮多少）？

判据：若 P3 在带掩膜的队列（BUSI / BrEaST）上不切病灶，则 P3 优先
（无失真，与"形态保留"的主张一致）；若切病灶比例不可忽略，则退回 P2 或 P1。
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
from breastdg.data.preprocess import tone_normalize

SIZE = 256


def load_gray_and_mask(img_path: str, mask_paths: list[str]):
    with Image.open(img_path) as im:
        g = np.asarray(im.convert("L"))
    m = np.zeros_like(g, bool)
    for mp in mask_paths:
        try:
            with Image.open(mp) as mf:
                mm = np.asarray(mf.convert("L")) > 0
            if mm.shape != g.shape:
                mm = np.asarray(Image.fromarray(mm.astype(np.uint8) * 255)
                                .resize((g.shape[1], g.shape[0]), Image.NEAREST)) > 0
            m |= mm
        except Exception:
            pass
    if g.shape[0] > g.shape[1]:
        g = np.rot90(g, 1); m = np.rot90(m, 1)
    return g, m


def content_bbox(g: np.ndarray):
    v = g.astype(np.float32)
    thr = v.min() + max(1.0, 0.02 * (v.max() - v.min()))
    mm = v > thr
    rows, cols = np.any(mm, axis=1), np.any(mm, axis=0)
    if not rows.any() or not cols.any():
        return 0, g.shape[0], 0, g.shape[1]
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(r0), int(r1) + 1, int(c0), int(c1) + 1


def square_crop_box(h: int, w: int, S: int = SIZE):
    """保比例缩放到短边=S 后的中心方裁，返回放大前的 (y0,y1,x0,x1)。"""
    s = S / min(h, w)
    nh, nw = max(S, round(h * s)), max(S, round(w * s))
    # 放大后中心裁 S×S，映射回原坐标
    y0f = (nh - S) / 2 / s
    x0f = (nw - S) / 2 / s
    return int(np.floor(y0f)), int(np.floor(y0f)) + int(np.ceil(S / s)), \
           int(np.floor(x0f)), int(np.floor(x0f)) + int(np.ceil(S / s))


def main():
    report = {}
    for name, fn in (("busi", lambda: dedupe_entries(registry.busi_entries())[0]),
                     ("breast", registry.breast_entries)):
        ents = fn()
        ents = [e for e in ents if e.get("masks")]
        if not ents:
            print(f"{name}: 无掩膜，跳过"); continue
        crop_frac, cut_frac, cut_cases = [], [], []
        for e in ents:
            g, m = load_gray_and_mask(e["image"], e["masks"])
            h, w = g.shape
            r0, r1, c0, c1 = content_bbox(g)
            crop_frac.append(1.0 - ((r1 - r0) * (c1 - c0)) / (h * w))
            # P3：内容区 → 中心方裁
            gc, mc = g[r0:r1, c0:c1], m[r0:r1, c0:c1]
            if mc.sum() == 0:
                continue
            hh, ww = gc.shape
            y0, y1, x0, x1 = square_crop_box(hh, ww)
            y0, x0 = max(0, y0), max(0, x0)
            kept = np.zeros_like(mc, bool)
            kept[y0:min(y1, hh), x0:min(x1, ww)] = True
            lost = mc & ~kept
            frac = lost.sum() / mc.sum()
            if frac > 0.02:
                cut_cases.append((Path(e["image"]).name, round(float(frac), 4)))
            cut_frac.append(frac)
        cut_frac = np.array(cut_frac)
        report[name] = dict(n=len(ents),
                            content_crop_frac=float(np.mean(crop_frac)),
                            lesion_cut_mean=float(cut_frac.mean()),
                            lesion_cut_gt2pct=int((cut_frac > 0.02).sum()),
                            worst=cut_cases[:8])
        print(f"\n=== {name} (n={len(ents)}) ===")
        print(f"  内容裁剪丢面积 : {np.mean(crop_frac):.3f} 均值")
        print(f"  P3 切到病灶比例: 均值 {cut_frac.mean():.4f}  "
              f"| >2% 的病例数 {int((cut_frac>0.02).sum())}")
        if cut_cases:
            print(f"  最严重: {cut_cases[:5]}")

    p = Path("runs/diag/geometry_lesion_check.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
