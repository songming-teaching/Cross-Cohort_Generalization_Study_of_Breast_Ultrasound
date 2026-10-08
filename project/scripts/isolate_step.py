"""隔离训练步耗时：纯取数 vs 纯计算 vs 完整步。"""
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

BS = 32
STEPS = 25


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def main():
    t0 = time.time()
    busi, _ = dedupe_entries(registry.busi_entries())
    print(f"数据准备（含去重）耗时 {time.time()-t0:.1f} s", flush=True)

    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()
    sp = json.loads((config.SPLITS_DIR / "lodo_holdout_busbra.json").read_text("utf-8"))
    trn = set(sp["train"])
    src = [e for e in busi + busbra if Path(e["image"]).name in trn]
    print(f"源域 {len(src)}  目标域 {len(busuclm)}", flush=True)

    pp = PreprocConfig(size=256, tone_norm=True)
    for nw in (10, 6):
        ld_s = make_loader(src, batch_size=BS, train=True, cfg=pp,
                           num_workers=nw, seed=42)
        ld_t = make_loader(busuclm, batch_size=BS, train=True, with_label=False,
                           cfg=pp, num_workers=nw, seed=42)
        it_s, it_t = _cycle(ld_s), _cycle(ld_t)
        for _ in range(3):                     # 预热（worker 启动）
            next(it_s); next(it_t)
        t = time.time()
        for _ in range(STEPS):
            next(it_s); next(it_t)
        dt = (time.time() - t) / STEPS
        print(f"  纯取数 workers={nw:2d}: {dt*1000:7.0f} ms/step", flush=True)
        del ld_s, ld_t, it_s, it_t

    # 缓存一批数据后测纯计算
    ld_s = make_loader(src, batch_size=BS, train=True, cfg=pp, num_workers=6, seed=42)
    ld_t = make_loader(busuclm, batch_size=BS, train=True, with_label=False,
                       cfg=pp, num_workers=6, seed=42)
    it_s, it_t = _cycle(ld_s), _cycle(ld_t)
    sb = [next(it_s) for _ in range(4)]
    tb = [next(it_t) for _ in range(4)]

    dev = torch.device("cuda")
    m = REGISTRY["resnet50_cbam"](n_domains=3).to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ce = nn.CrossEntropyLoss()

    def step(s, t):
        xs, ys, sdom, _, _ = s
        xt, _, tdom, _, _ = t
        xs, ys, sdom = xs.to(dev), ys.to(dev), sdom.to(dev)
        xt, tdom = xt.to(dev), tdom.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            fs, ft = m.forward_features(xs), m.forward_features(xt)
            loss = ce(m.class_head(fs), ys)
            dom = m.domain_head(torch.cat([fs, ft], 0), 1.0)
            loss = loss + ce(dom, torch.cat([sdom, tdom], 0))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        scaler.step(opt); scaler.update()

    for k in range(5):
        step(sb[k % 4], tb[k % 4])            # 预热
    torch.cuda.synchronize()
    t = time.time()
    for k in range(STEPS):
        step(sb[k % 4], tb[k % 4])
    torch.cuda.synchronize()
    print(f"  纯计算(数据已缓存): {(time.time()-t)/STEPS*1000:7.0f} ms/step", flush=True)


if __name__ == "__main__":
    main()
