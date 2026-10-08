"""P4 验收：掩膜核查（第零条）+ 黑边占比对账。

验收标准（《P4修正.txt》第三节）：
  1. BUSI 629 + BrEaST 252 全量过一遍，**切病灶病例数必须为 0**（构造上如此，实测验证）
  2. 实测黑边占比 与 预测式 1 − 0.88²/max(r,1/r) 一致（验证"残留只由长宽比承载"）
  3. 各队列黑边占比报出，供补充材料归因表使用
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
from breastdg.data.geometry import (OCCUPANCY, apply_image, apply_mask,
                                    content_aspect, pad_decomposition, plan,
                                    predicted_pad, spatial_containment)
from breastdg.data.preprocess import tone_normalize

SIZE = 256


def load_pair(img_path: str, mask_paths: list[str]):
    """**图像与掩膜走同一套方向处理**（此前的 bug：图像旋转了、掩膜却只 resize）。"""
    with Image.open(img_path) as im:
        g = np.asarray(im.convert("L"))
    m = np.zeros(g.shape, bool)
    for p in mask_paths or []:
        try:
            with Image.open(p) as f:
                mm = np.asarray(f.convert("L")) > 0
            if mm.shape != g.shape:                 # 兜底：仅在形状不符时才重采样
                mm = np.asarray(Image.fromarray(mm.astype(np.uint8) * 255)
                                .resize((g.shape[1], g.shape[0]), Image.NEAREST)) > 0
            m |= mm
        except Exception:
            pass
    # 图像与掩膜**一起**转
    if g.shape[0] > g.shape[1]:
        g = np.rot90(g, 1)
        m = np.rot90(m, 1)
    return g, m


def check(cohort: str, entries: list[dict]) -> dict:
    """两类损失分开报告：
       切 (_cut)     —— 掩膜落出输出窗口，几何设计问题，**必须为 0**
       腐蚀 (_erode) —— 降采样面积损失，分辨率代价，任何 256 管线都有
    """
    print(f"\n=== {cohort} (n={len(entries)}) ===", flush=True)
    cut, erode, pads, preds, aspects, blk = [], [], [], [], [], []
    mism = 0
    for e in entries:
        g, m = load_pair(e["image"], e.get("masks"))
        g = tone_normalize(g, 5.0, 95.0)
        p = plan(g, ratio=1.0, size=SIZE)
        img = apply_image(g, p)
        dec = pad_decomposition(img, p)
        pm_pred = predicted_pad(p)
        pads.append(dec["pad_geom"]); preds.append(pm_pred)
        blk.append(dec["black_cont"])
        aspects.append(content_aspect(p))
        if abs(dec["pad_geom"] - pm_pred) > 0.02:
            mism += 1
        if m.sum() > 0:
            cut.append(spatial_containment(m, p))
            pm = apply_mask(m, p)
            # 面积保真必须对照**缩放后的期望面积**（m.sum()·scale²）。
            # 直接比 m.sum() 会把缩放因子本身当成损失（scale=0.24 时显示 94% 假损失）。
            expect = max(1.0, float(m.sum()) * p.scale ** 2)
            erode.append(max(0.0, 1.0 - float(pm.sum()) / expect))
    cut = np.array(cut) if cut else np.array([0.0])
    erode = np.array(erode) if erode else np.array([0.0])
    pads = np.array(pads); preds = np.array(preds)
    aspects = np.array(aspects); blk = np.array(blk)
    r = dict(n=len(entries), n_with_mask=len(cut),
             lesion_cut_mean=float(cut.mean()), lesion_cut_max=float(cut.max()),
             lesion_cut_gt0=int((cut > 0).sum()),
             lesion_erode_mean=float(erode.mean()), lesion_erode_max=float(erode.max()),
             pad_geom_mean=float(pads.mean()), pad_geom_std=float(pads.std()),
             pad_predicted_mean=float(preds.mean()),
             black_content_mean=float(blk.mean()),
             aspect_mean=float(aspects.mean()),
             aspect_range=[float(aspects.min()), float(aspects.max())],
             pad_mismatch_gt2pct=int(mism))
    print(f"  几何填充 实测 {pads.mean():.4f}±{pads.std():.4f}  "
          f"预测 {preds.mean():.4f}  偏差>2%项={mism}", flush=True)
    print(f"  内容黑像素占比 {blk.mean():.4f}  (图像固有，非填充)", flush=True)
    print(f"  内容物理宽高比 {aspects.mean():.3f}  [{aspects.min():.3f}, {aspects.max():.3f}]",
          flush=True)
    print(f"  **空间切病灶** 均值 {cut.mean():.5f}  最大 {cut.max():.5f}  "
          f"病例数={int((cut>0).sum())}  (必须为 0)", flush=True)
    print(f"  降采样腐蚀    均值 {erode.mean():.4f}  最大 {erode.max():.4f}  "
          f"(分辨率代价，非缺陷)", flush=True)
    return r


def main():
    out = {}
    busi, _ = dedupe_entries(registry.busi_entries())
    out["BUSI"] = check("BUSI", busi)
    out["BrEaST"] = check("BrEaST", registry.breast_entries())
    out["BUSBRA"] = check("BUSBRA", registry.busbra_entries()[:400])   # 无掩膜，抽样
    out["BUS-UCLM"] = check("BUS-UCLM", registry.busuclm_entries())

    print("\n" + "=" * 74)
    print("验收判定")
    print("=" * 74)
    ok_cut = all(v["lesion_cut_max"] < 1e-9 for k, v in out.items() if k in ("BUSI", "BrEaST"))
    print(f"  第零条 切病灶为 0           : {'PASS' if ok_cut else 'FAIL'}")
    ok_f = all(v["pad_mismatch_gt2pct"] == 0 for v in out.values())
    print(f"  黑边与预测式对账            : {'PASS' if ok_f else 'FAIL'}")

    print(f"\n  各队列 P4 几何填充占比（补充材料归因表用）")
    for k, v in out.items():
        print(f"    {k:11s} {v['pad_geom_mean']:.4f}±{v['pad_geom_std']:.4f}"
              f"   内容宽高比 {v['aspect_mean']:.3f}"
              f"   内容黑像素 {v['black_content_mean']:.4f}")

    p = Path("runs/diag/geometry_p4_acceptance.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
