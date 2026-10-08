"""把 OASBUD 从 .mat 物化为图像文件（P4 几何 + 物理像素比校正）。

为什么需要
----------
E8 谱分析与 E3 评估都要在**训练好的模型**上取特征，而取特征走 `BreastDataset`
（读文件路径）。OASBUD 只有 `.mat` 里的原始 RF，故须先物化为 PNG 文件。

两条臂（F12.5）
---------------
  corrected   —— 锁箱主评估用（P4 + 各向同性校正，原文参数）
  uncorrected —— 仅作**附录几何消融**（不校正，复现"预处理伪偏移"）
两臂共用 P4 几何，差异纯粹来自像素比处理。

物理参数（已核对，见《OASBUD参数核对.md》）
-------------------------------------------
  采样率 fs = 40 MHz；孔径 38 mm；扫描线 510 → 横向/轴向 = 3.88
  （Piotrzkowska-Wróblewska et al., Med Phys 44(11):6105–6109, 2017）

输出
----
  {OUT}/{arm}/images/{id}_view{n}.png      256×256 灰度
  {OUT}/{arm}/manifest.csv                 lesion_id,label,view,path,birads
  {OUT}/_meta.json                         物理参数与各臂统计（供论文引用）

⚠ 本脚本不修改 `src/`（F22 运行期约束）；待长跑结束后再把 `oasbud_entries()`
  加进 `registry.py`，届时可直接读此处的 manifest。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from breastdg import config
from breastdg.data import geometry as G
from breastdg.data.oasbud_rf import RFGeometry, load_cases, reconstruct_case
from breastdg.data.preprocess import tone_normalize

# OASBUD 原文参数（已核对）
FS_HZ = 40.0e6
APERTURE_M = 0.0380          # 38 mm（原文 L14-5/38）
N_LINES = 510
# 由此得到的比值 = (0.0380/510) / (1540/(2*40e6)) = 0.0745/0.01925 ≈ 3.87
SIZE = 256
ARMS = {"corrected": None, "uncorrected": 1.0}   # None → 用 RFGeometry 的比值


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="默认写到 config.OASBUD_P4_DIR（云上应在数据盘）")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--dr-db", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    a = ap.parse_args()
    # 云上产物须落数据盘（AutoDL 镜像只打包系统盘），故默认走 config 的
    # 可覆盖路径而不是硬编码相对路径。
    out_root = Path(a.out) if a.out else config.OASBUD_P4_DIR

    geo = RFGeometry(fs_hz=FS_HZ, aperture_m=APERTURE_M, n_lines=N_LINES)
    ratio = geo.ratio
    print(f"物理参数（原文核对值）: fs={FS_HZ/1e6:.0f} MHz  孔径={APERTURE_M*1e3:.1f} mm  "
          f"扫描线={N_LINES}")
    print(f"  轴向 {geo.axial_m*1e6:.4f} um  横向 {geo.lateral_m*1e6:.4f} um  "
          f"横向/轴向 = {ratio:.4f}\n")

    recs = load_cases()
    if a.limit:
        recs = recs[:a.limit]
    meta = {"physics": {"fs_hz": FS_HZ, "aperture_m": APERTURE_M, "n_lines": N_LINES,
                        "axial_um": geo.axial_m * 1e6, "lateral_um": geo.lateral_m * 1e6,
                        "ratio": ratio, "dr_db": a.dr_db, "size": a.size,
                        "source": "Piotrzkowska-Wroblewska et al., Med Phys 44(11):6105-6109, 2017"},
            "arms": {}}

    for arm, forced in ARMS.items():
        r_arm = ratio if forced is None else forced
        img_dir = out_root / arm / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        rows, aspects, pads = [], [], []
        for rec_idx, rec in enumerate(recs):
            lid = str(getattr(rec, "id")).strip()
            cls = int(np.ravel(getattr(rec, "class"))[0])
            birads = str(getattr(rec, "birads")).strip()
            # **唯一病灶键 = 记录索引**（不是 id 字符串）。
            # 原因：OASBUD.mat 有 100 条记录但唯一 id 只有 99 个——
            # '30nh' 出现在两条记录上（BI-RADS 3 与 4a，均良性），是两个不同病灶共用 id。
            # 若用 id 作文件名会互相覆盖（200 → 198 张）；若用 id 作 lesion_id，
            # 病灶级 cluster bootstrap 会把两个病灶并成一个簇。
            uid = f"{rec_idx:03d}_{lid}"
            for view in (0, 1):
                # reconstruct_case 的 apply_pixel_ratio 传 False：
                # 像素比只在 G.plan/apply 里施加**一次**（否则重复计入，曾致 1.094→18.7）
                img, _ = reconstruct_case(rec, "bmod", geo, a.dr_db, False, view)
                g = tone_normalize((img * 255).astype(np.uint8), 5.0, 95.0)
                p = G.plan(g, ratio=r_arm, size=a.size)
                gi = G.apply_image(g, p)
                fn = f"{uid}_view{view+1}.png"
                Image.fromarray(gi, mode="L").save(img_dir / fn)
                rows.append(dict(lesion_id=uid, src_id=lid, rec_idx=rec_idx,
                                 label=cls, view=view + 1, birads=birads,
                                 path=str(img_dir / fn)))
                aspects.append(G.content_aspect(p))
                pads.append(G.pad_decomposition(gi, p)["pad_geom"])
        with open(out_root / arm / "manifest.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        asp = np.array(aspects); pd = np.array(pads)
        lab = Counter(r["label"] for r in rows)
        n_files = len(list(img_dir.glob("*.png")))
        meta["arms"][arm] = dict(ratio_applied=float(r_arm), n_images=len(rows),
                                 n_files_on_disk=n_files,
                                 n_lesions=len({r["lesion_id"] for r in rows}),
                                 n_src_ids=len({r["src_id"] for r in rows}),
                                 benign=lab.get(0, 0), malignant=lab.get(1, 0),
                                 content_aspect_mean=float(asp.mean()),
                                 content_aspect_range=[float(asp.min()), float(asp.max())],
                                 pad_geom_mean=float(pd.mean()))
        print(f"=== {arm} (各向同性比 {r_arm:.3f}) ===")
        print(f"  图像 {len(rows)}  落盘 {n_files}  病灶(唯一键) "
              f"{meta['arms'][arm]['n_lesions']}  源 id 数 {meta['arms'][arm]['n_src_ids']}")
        print(f"  良 {lab.get(0,0)} : 恶 {lab.get(1,0)}")
        print(f"  内容宽高比 {asp.mean():.3f} [{asp.min():.3f}, {asp.max():.3f}]  "
              f"几何填充 {pd.mean():.4f}")
        print(f"  -> {img_dir}\n")

    c, u = meta["arms"]["corrected"], meta["arms"]["uncorrected"]
    print("=" * 72)
    print("几何消融对比（附录用）")
    print(f"  内容宽高比  {u['content_aspect_mean']:.3f} (未校正) -> "
          f"{c['content_aspect_mean']:.3f} (已校正)")
    print(f"  几何填充    {u['pad_geom_mean']:.4f} -> {c['pad_geom_mean']:.4f}")
    print(f"  训练域参照：宽高比 1.256–1.407，填充 0.373–0.451")

    (out_root / "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out_root/'_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
