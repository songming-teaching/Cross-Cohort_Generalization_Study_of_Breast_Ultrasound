"""掩膜几何一致性检查：同一张图在各几何管线下掩膜面积应相互一致。

预期：p4 掩膜像素 ≈ letterbox 掩膜像素（同一内容，同样缩放到 256）。
若 p4 远大于 letterbox，说明几何参数取错了源（曾发生：用掩膜自身算内容框，
框缩到病灶周围，缩放比被放大，掩膜撑满 19212 px 而正确值约 189）。
"""
from __future__ import annotations

import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.train.datasets import BreastDataset


def src_mask_px(mask_paths: list[str]) -> int:
    m = None
    for p in mask_paths or []:
        with Image.open(p) as f:
            mm = np.asarray(f.convert("L")) > 0
        m = mm if m is None else (m | mm)
    return int(m.sum()) if m is not None else 0


def check(name: str, entries: list[dict], n: int = 5):
    print(f"\n=== {name} ===")
    print(f"  {'文件':<24}{'源掩膜':>8}{'p4':>8}{'letterbox':>11}{'p4/letterbox':>14}")
    ratios = []
    for e in [x for x in entries if x.get("masks")][:n]:
        src = src_mask_px(e["masks"])
        got = {}
        for geo in ("p4", "letterbox"):
            ds = BreastDataset([e], PreprocConfig(geometry=geo),
                               train=False, with_mask=True)
            _, _, _, _, m = ds[0]
            got[geo] = int(m.sum())
        r = got["p4"] / max(got["letterbox"], 1)
        ratios.append(r)
        fn = e["image"].replace("\\", "/").split("/")[-1]
        print(f"  {fn:<24}{src:>8}{got['p4']:>8}{got['letterbox']:>11}{r:>14.3f}")
    a = np.array(ratios)
    print(f"  p4/letterbox 比值：均值 {a.mean():.3f}  范围 [{a.min():.3f}, {a.max():.3f}]")
    verdict = "OK" if 0.6 <= a.mean() <= 1.6 else "FAIL（几何参数疑似取错源）"
    print(f"  判定：{verdict}")
    return verdict.startswith("OK")


def main():
    busi, _ = dedupe_entries(registry.busi_entries())
    ok1 = check("BUSI", busi)
    ok2 = check("BrEaST", registry.breast_entries())
    print("\n" + "=" * 60)
    print(f"总体：{'PASS' if (ok1 and ok2) else 'FAIL'}")


if __name__ == "__main__":
    main()
