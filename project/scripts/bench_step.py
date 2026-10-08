"""实测单个训练步的耗时，定位 GPU 侧瓶颈（deterministic / AMP / 骨干）。"""
from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "src")

from breastdg.models.backbones import REGISTRY


def bench(model, bs=32, steps=20, amp=True, deterministic=True, model_name="r50"):
    dev = torch.device("cuda")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    m = model.to(dev).train()
    opt = torch.optim.AdamW(m.parameters(), 1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ce = nn.CrossEntropyLoss()
    xs = torch.randn(bs, 3, 256, 256, device=dev)
    xt = torch.randn(bs, 3, 256, 256, device=dev)
    ys = torch.randint(0, 2, (bs,), device=dev)
    ydom = torch.randint(0, 3, (2 * bs,), device=dev)

    def one():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            fs = m.forward_features(xs)
            ft = m.forward_features(xt)
            zs = m.fuse(fs["f_l"], fs["f_h"]) if isinstance(fs, dict) else fs
            loss = ce(m.class_head(zs), ys)
            dom = m.domain_head(torch.cat([fs, ft], 0), 0.5)
            loss = loss + ce(dom, ydom)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    for _ in range(5):
        one()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        one()
    torch.cuda.synchronize()
    dt = (time.time() - t) / steps
    print(f"  {model_name:14s} amp={str(amp):5s} determ={str(deterministic):5s} "
          f"{dt*1000:7.0f} ms/step   {2*bs/dt:7.1f} img/s", flush=True)
    del m, opt
    torch.cuda.empty_cache()
    return dt


def main():
    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    m = REGISTRY["resnet50_cbam"](n_domains=3)
    for det in (True, False):
        for amp in (True, False):
            bench(m, amp=amp, deterministic=det)
    del m
    print(flush=True)
    for name in ("densenet121_se", "efficientnet_v2_s", "swin_t"):
        bench(REGISTRY[name](n_domains=3), deterministic=False, model_name=name)


if __name__ == "__main__":
    main()
