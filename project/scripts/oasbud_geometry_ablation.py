"""OASBUD 几何消融臂：量化"预处理伪偏移"（补充材料用）。

定位（《冻结日志》F12.5）
------------------------
锁箱主评估只跑统一管线（P4）。本脚本产出的是**几何消融臂**：
同一批 OASBUD 病灶，在 P4 几何下比较两种**像素比处理**：

  (a) 未校正（现状的最保守近似）：不做各向同性校正，直接进 P4
  (b) 已校正：按物理像素比重采样到各向同性后进 P4

二者共用 P4 几何，故差异**pure 来自渲染链/像素比**，这正是审稿意见 2 与 9 问的。
本脚本只出**几何量**（宽高比、填充占比、内容尺度），不需模型，零 GPU 成本；
模型评估（同一冻结模型在两条臂上的 AUC）留待锁箱前由评估脚本统一出数。

⚠ 物理参数（采样率、孔径）目前是**待核对默认值**，须以 OASBUD 原文为准；
本脚本把参数写进输出，便于核对后重算。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from breastdg.data import geometry as G
from breastdg.data.oasbud_rf import RFGeometry, load_cases, reconstruct_case
from breastdg.data.preprocess import tone_normalize

OUT = Path("runs/diag/oasbud_geometry_ablation.json")
N_LESIONS = 100          # 全部病灶


def stats(vals):
    a = np.array(vals, float)
    return dict(mean=float(a.mean()), std=float(a.std()),
                min=float(a.min()), max=float(a.max()))


def main():
    recs = load_cases()
    geo = RFGeometry()
    print(f"病灶 {len(recs)}  物理参数: fs={geo.fs_hz/1e6:.2f} MHz  "
          f"孔径={geo.aperture_m*1e3:.2f} mm  扫描线={geo.n_lines}")
    print(f"  轴向间距={geo.axial_m*1e6:.3f} um  横向间距={geo.lateral_m*1e6:.3f} um  "
          f"横向/轴向 = {geo.ratio:.4f}")
    print("  [WARN] 上述物理参数为待核对默认值，须以 OASBUD 原文为准\n")

    arms = {"a_uncorrected": 1.0, "b_corrected": geo.ratio}
    out = {"physics": dict(fs_hz=geo.fs_hz, aperture_m=geo.aperture_m,
                           n_lines=geo.n_lines, ratio=geo.ratio,
                           note="物理参数待以 OASBUD 原文核对"),
           "n_views": 2 * min(len(recs), N_LESIONS),
           "arms": {}}

    for arm, ratio in arms.items():
        aspects, pads, scales, means, n_content_px = [], [], [], [], []
        for r in recs[:N_LESIONS]:
            for view in (0, 1):
                # apply_pixel_ratio=False：像素比**只在 plan/apply_image 里施加一次**。
                # （此前传 True 且 plan 又传 ratio，导致重复计入，宽高比 1.094 → 18.7。）
                img, _ = reconstruct_case(r, "bmod", geo, 60.0,
                                          False, view)
                g = tone_normalize((img * 255).astype(np.uint8), 5.0, 95.0)
                p = G.plan(g, ratio=ratio, size=256)
                gi = G.apply_image(g, p)
                aspects.append(G.content_aspect(p))
                pads.append(G.pad_decomposition(gi, p)["pad_geom"])
                scales.append(p.scale)
                means.append(float(gi.mean()))
                r0, r1, c0, c1 = p.bbox
                n_content_px.append((r1 - r0) * (c1 - c0))
        out["arms"][arm] = dict(
            ratio_applied=float(ratio),
            content_aspect=stats(aspects),
            pad_frac=stats(pads),
            content_scale=stats(scales),
            content_source_px=stats(n_content_px),
            rendered_mean=stats(means))
        s = out["arms"][arm]
        print(f"=== {arm} (各向同性比 {ratio:.3f}) ===")
        print(f"  内容宽高比  {s['content_aspect']['mean']:.3f}  "
              f"[{s['content_aspect']['min']:.3f}, {s['content_aspect']['max']:.3f}]")
        print(f"  几何填充    {s['pad_frac']['mean']:.4f}  "
              f"[{s['pad_frac']['min']:.4f}, {s['pad_frac']['max']:.4f}]")
        print(f"  内容缩放比  {s['content_scale']['mean']:.4f}")
        print(f"  内容源像素  {s['content_source_px']['mean']:.0f}\n")

    a, b = out["arms"]["a_uncorrected"], out["arms"]["b_corrected"]
    print("=" * 72)
    print("几何消融对比（这是审稿意见 2 / 9 的回应材料）")
    print(f"  内容宽高比   (a) {a['content_aspect']['mean']:.3f}  →  "
          f"(b) {b['content_aspect']['mean']:.3f}")
    print(f"  几何填充     (a) {a['pad_frac']['mean']:.4f}  →  "
          f"(b) {b['pad_frac']['mean']:.4f}")
    print(f"  内容源像素   (a) {a['content_source_px']['mean']:.0f}  →  "
          f"(b) {b['content_source_px']['mean']:.0f}  "
          f"(×{(b['content_source_px']['mean']/a['content_source_px']['mean']):.2f})")
    print()
    print("  训练域参照：内容宽高比 1.256–1.407，几何填充 0.373–0.451")
    ia = a['content_aspect']['mean']; ib = b['content_aspect']['mean']
    print(f"  → (a) 宽高比 {ia:.3f} {'落在' if 1.0<ia<1.7 else '**落在区间外**'}训练域区间内")
    print(f"  → (b) 宽高比 {ib:.3f} {'落在' if 1.0<ib<1.7 else '**落在区间外**'}训练域区间内")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
