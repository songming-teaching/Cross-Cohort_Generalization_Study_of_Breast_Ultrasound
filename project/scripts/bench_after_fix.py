"""改后复测：数据吞吐 vs 计算 vs 整步，定位 uint8 优化效果。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "src")

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.models.backbones import REGISTRY
from breastdg.train.datasets import make_loader
from breastdg.train.engine import to_device_uint8

BS, STEPS = 32, 30


def cyc(loader):
    while True:
        for b in loader:
            yield b


def main():
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    tgt = registry.busuclm_entries()
    sp = json.loads((config.SPLITS_DIR / "lodo_holdout_busbra.json").read_text("utf-8"))
    trn = set(sp["train"])
    src = [e for e in busi + busbra if Path(e["image"]).name in trn]
    pp = PreprocConfig(size=256, tone_norm=True)
    print(f"源域 {len(src)}  目标域 {len(tgt)}", flush=True)

    # --- 取数吞吐（uint8）---
    best_ld = None
    for nw in (12, 8, 6):
        ls = make_loader(src, batch_size=BS, train=True, cfg=pp, num_workers=nw, seed=42)
        lt = make_loader(tgt, batch_size=BS, train=True, with_label=False,
                         cfg=pp, num_workers=nw, seed=42)
        a, b = cyc(ls), cyc(lt)
        for _ in range(4):
            next(a); next(b)
        t = time.time()
        for _ in range(STEPS):
            ba, bb = next(a), next(b)
        dt = (time.time() - t) / STEPS
        print(f"  取数 workers={nw:2d}: {dt*1000:6.0f} ms/step  "
              f"dtype={ba[0].dtype} bytes/batch={ba[0].numel()/1e6:.2f}M", flush=True)
        if nw == 12:
            best_ld = (ls, lt, ba, bb)
        else:
            del ls, lt, a, b

    # --- 计算 + 整步（缓存数据）---
    ls, lt, ba, bb = best_ld
    dev = torch.device("cuda")
    m = REGISTRY["resnet50_cbam"](n_domains=3).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ce = nn.CrossEntropyLoss()

    def step(s, t, up=True):
        xs, ys, sdom, _, _ = s
        xt, _, tdom, _, _ = t
        xs = xs.to(dev, non_blocking=True); xt = xt.to(dev, non_blocking=True)
        if up:
            xs, xt = to_device_uint8(xs), to_device_uint8(xt)
        else:
            xs, xt = xs.float() / 255, xt.float() / 255
        ys = ys.to(dev); sdom = sdom.to(dev); tdom = tdom.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = ce(m.class_head(m.forward_features(xs)), ys)
            loss = loss + ce(m.domain_head(
                torch.cat([m.forward_features(xs), m.forward_features(xt)], 0), 1.0),
                torch.cat([sdom, tdom], 0))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        scaler.step(opt); scaler.update()

    for up in (True, False):
        for _ in range(5):
            step(ba, bb, up)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(STEPS):
            step(ba, bb, up)
        torch.cuda.synchronize()
        tag = "uint8→GPU归一化" if up else "GPU 上 float 归一化（对照）"
        print(f"  计算 {tag}: {(time.time()-t)/STEPS*1000:6.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
