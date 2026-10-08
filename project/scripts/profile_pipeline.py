"""定位数据管线瓶颈：单图成本拆解 + DataLoader 实际吞吐。"""
from __future__ import annotations

import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig, letterbox, tone_normalize
from breastdg.train.datasets import BreastDataset, make_loader

N = 24


def timed(fn, n=N, label=""):
    fn()                                   # 预热
    t = time.time()
    for _ in range(n):
        fn()
    ms = (time.time() - t) / n * 1000
    print(f"  {label:28s} {ms:8.2f} ms", flush=True)
    return ms


def main():
    busi, _ = dedupe_entries(registry.busi_entries())
    tr = busi[:400]
    pp = PreprocConfig(size=256, tone_norm=True)
    print(f"样本数 {len(tr)}", flush=True)

    paths = [e["image"] for e in tr[:8]]
    sizes = []
    for p in paths:
        with Image.open(p) as im:
            sizes.append(im.size)
    print(f"原图尺寸样例 {sizes}", flush=True)

    p0 = tr[0]["image"]
    with Image.open(p0) as im:
        a = np.asarray(im.convert("L"))
    rgb = letterbox(Image.fromarray(a), 256, 0).convert("RGB")

    print("\n--- 单步拆解 ---", flush=True)
    timed(lambda: np.asarray(Image.open(p0).convert("L")), label="读图+转灰度")
    timed(lambda: tone_normalize(a, 5, 95), label="tone_normalize")
    timed(lambda: letterbox(Image.fromarray(a), 256, 0), label="letterbox")
    timed(lambda: letterbox(Image.fromarray(a), 256, 0).convert("RGB"), label="letterbox+RGB")

    import torchvision.transforms as T
    aug = T.Compose([T.RandomAffine(10, translate=(.05, .05), scale=(.95, 1.05)),
                     T.RandomHorizontalFlip()])
    timed(lambda: aug(rgb), label="torchvision 增广")

    ds = BreastDataset(tr, pp, train=True)
    timed(lambda: ds[0], n=16, label="完整 __getitem__(训练)")

    print("\n--- DataLoader 吞吐 ---", flush=True)
    for nw in (0, 4, 10):
        ld = make_loader(tr, batch_size=32, train=True, cfg=pp, num_workers=nw)
        it = iter(ld)
        for _ in range(2):
            next(it)
        t = time.time(); n = 0
        while time.time() - t < 6:
            b = next(it); n += b[0].shape[0]
        print(f"  workers={nw:2d}              {n/(time.time()-t):8.1f} 图/秒", flush=True)
        del ld, it


if __name__ == "__main__":
    main()
