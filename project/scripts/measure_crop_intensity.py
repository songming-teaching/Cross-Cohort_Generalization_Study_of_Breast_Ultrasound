"""测量 P4 内容裁剪力度：三种阈值下的实际裁剪比例。

用户顾虑：宁可少剪边框，也不要过量剪切。
（P3 的核查已显示 BUSI 丢面积 0.000、BrEaST 0.002，但那是"面积"口径，
  需按"裁掉的行列占比"再量一次，并比较不同阈值。）

三种口径：
  S0 严格黑边：只有**整行/整列全为 0** 才裁 → 最保守
  S1 当前 P4 ：阈值 max(1, 1% 分位)
  S2 激进    ：阈值 max(1, 5% 分位)     → 仅作对照，看过度剪辑什么样

另报：被裁区域内是否含掩膜像素（若含，说明裁到了真内容）。
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

N_SAMPLE = 260


def load_pair(ip: str, mps: list[str]):
    with Image.open(ip) as im:
        g = np.asarray(im.convert("L"))
    m = np.zeros(g.shape, bool)
    for p in mps or []:
        try:
            with Image.open(p) as f:
                mm = np.asarray(f.convert("L")) > 0
            if mm.shape != g.shape:
                mm = np.asarray(Image.fromarray(mm.astype(np.uint8) * 255)
                                .resize((g.shape[1], g.shape[0]), Image.NEAREST)) > 0
            m |= mm
        except Exception:
            pass
    if g.shape[0] > g.shape[1]:
        g = np.rot90(g, 1); m = np.rot90(m, 1)
    return g, m


def bbox_strict(g: np.ndarray):
    """只裁整行/整列全黑。"""
    nz_r = np.where((g > 0).any(axis=1))[0]
    nz_c = np.where((g > 0).any(axis=0))[0]
    if len(nz_r) == 0 or len(nz_c) == 0:
        return 0, g.shape[0], 0, g.shape[1]
    return int(nz_r[0]), int(nz_r[-1]) + 1, int(nz_c[0]), int(nz_c[-1]) + 1


def bbox_pct(g: np.ndarray, pct: float):
    thr = max(1.0, float(np.percentile(g, pct)))
    m = g > thr
    rows, cols = np.any(m, axis=1), np.any(m, axis=0)
    if not rows.any() or not cols.any():
        return 0, g.shape[0], 0, g.shape[1]
    r = np.where(rows)[0]; c = np.where(cols)[0]
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def metrics(g, m, box):
    r0, r1, c0, c1 = box
    H, W = g.shape
    crop_h = 1.0 - (r1 - r0) / H
    crop_w = 1.0 - (c1 - c0) / W
    inside = np.zeros_like(m, bool)
    inside[r0:r1, c0:c1] = True
    lost = int((m & ~inside).sum())
    tot = int(m.sum())
    return crop_h, crop_w, (lost / tot if tot else 0.0), lost, tot


def main():
    cohorts = {}
    busi, _ = dedupe_entries(registry.busi_entries())
    cohorts["BUSI"] = busi
    cohorts["BrEaST"] = registry.breast_entries()
    cohorts["BUSBRA"] = registry.busbra_entries()[:N_SAMPLE]
    cohorts["BUS-UCLM"] = registry.busuclm_entries()

    out = {}
    for name, ents in cohorts.items():
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ents), min(N_SAMPLE, len(ents)), False)
        ents = [ents[i] for i in idx]
        acc = {"S0 严格黑边": [], "S1 当前P4": [], "S2 激进(5%)": []}
        lost_mask = {"S0 严格黑边": [], "S1 当前P4": [], "S2 激进(5%)": []}
        for e in ents:
            g, m = load_pair(e["image"], e.get("masks"))
            boxes = {"S0 严格黑边": bbox_strict(g),
                     "S1 当前P4": bbox_pct(g, 1.0),
                     "S2 激进(5%)": bbox_pct(g, 5.0)}
            for k, b in boxes.items():
                ch, cw, lm, lost, tot = metrics(g, m, b)
                acc[k].append(max(ch, cw))
                if tot:
                    lost_mask[k].append(lm)
        print(f"\n=== {name} (n={len(ents)}) ===")
        res = {}
        for k in acc:
            a = np.array(acc[k])
            lm = np.array(lost_mask[k]) if lost_mask[k] else np.array([0.0])
            res[k] = dict(crop_axis_mean=float(a.mean()), crop_axis_max=float(a.max()),
                          mask_lost_mean=float(lm.mean()), mask_lost_max=float(lm.max()),
                          n_mask_lost_gt0=int((lm > 1e-9).sum()))
            print(f"  {k:12s} 裁剪(最长轴) 均 {a.mean():.4f} 最大 {a.max():.4f} | "
                  f"裁到掩膜 均 {lm.mean():.5f} 最大 {lm.max():.5f} "
                  f"项数 {int((lm>1e-9).sum())}")
        out[name] = res

    p = Path("runs/diag/crop_intensity.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
