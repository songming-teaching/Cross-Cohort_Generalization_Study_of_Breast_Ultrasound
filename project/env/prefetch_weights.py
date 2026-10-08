"""预下载 5 个骨干的 torchvision 预训练权重（一次性，固化进镜像）。

为什么必须做
------------
训练第 1 步就要用预训练权重。云端若拉不到，**不报错、只是等** ——
你会看到 GPU 占用 0%、日志不动，按小时计费白烧。

为什么要写成文件而不是一行 `python -c`
----------------------------------------
一条 `ssh autodl 'python -c "..."'` 或 heredoc 粘贴，都会因为
**引号嵌套 / 前导空格** 而变形（实测：heredoc 内容被加了缩进 →
`IndentationError: unexpected indent`）。写成文件 + rsync 最稳，
且可重复执行（已下过的会走缓存，秒过）。

用法（云上，或本机交互式 ssh 进去后）
--------------------------------------
    /root/miniconda3/bin/python env/prefetch_weights.py

前置：`source /etc/network_turbo`（AutoDL 上拉 download.pytorch.org 需要它）
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import time

# 入口自护（preflight 坑1：Windows GBK 控制台编不了 ✓ 等字符）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 与训练代码实际用到的骨干**一一对应**（见 dev_screen / retrain_lockbox 的
# 变体表）。少一个都会让对应变体在训练时静默卡住。
BACKBONES = [
    "resnet50",
    "resnet18",
    "densenet121",
    "efficientnet_v2_s",
    "swin_t",
]


def cache_dir() -> pathlib.Path:
    home = os.environ.get("TORCH_HOME", "/root/.cache/torch")
    return pathlib.Path(home) / "hub" / "checkpoints"


def main() -> int:
    import torchvision.models as M

    d = cache_dir()
    print(f"目标缓存目录：{d}")
    before = sorted(p.name for p in d.glob("*.pth")) if d.is_dir() else []
    print(f"已有 {len(before)} 个权重文件\n")

    ok, fail = 0, []
    for name in BACKBONES:
        fn = getattr(M, name)
        wcls = getattr(M, f"{name.upper()}_Weights", None)
        try:
            t0 = time.time()
            if wcls is None:                       # resnet18 等命名差异兜底
                fn(weights="DEFAULT")
            else:
                fn(weights=wcls.DEFAULT)
            print(f"  ✓ {name:20s} {time.time() - t0:6.1f}s", flush=True)
            ok += 1
        except Exception as e:                     # noqa: BLE001
            print(f"  ✗ {name:20s} {type(e).__name__}: {e}", flush=True)
            fail.append(name)

    after = sorted(p.name for p in d.glob("*.pth")) if d.is_dir() else []
    print(f"\n缓存里现有 {len(after)} 个 .pth 文件：")
    for f in after:
        p = d / f
        mb = p.stat().st_size / 1e6
        print(f"  {f:48s} {mb:7.1f} MB")

    print()
    if len(after) < len(BACKBONES):
        print(f"✗ 只有 {len(after)} 个，不足 {len(BACKBONES)} 个 —— "
              f"**不要开始训练**，对应变体会静默卡住。", file=sys.stderr)
        print("  处置：确认已 `source /etc/network_turbo`，然后重跑本脚本。",
              file=sys.stderr)
        return 1
    print(f"✓ 5 个骨干权重全部就位（成功 {ok}/{len(BACKBONES)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
