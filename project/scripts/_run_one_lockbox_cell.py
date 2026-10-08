"""跑锁箱全源重训的**单个格子**（一个 骨干×种子 组合）。

与 `_run_one_cell.py`（TIER4 用）同构，理由相同：
**一格一个进程**才能在挂死时只损失一格 —— 2026-10-05 那次事故正是
进程挂死、持有 6.5 GB 内存与 CUDA 上下文达 10 小时，GPU 全程空转。

安全边界
--------
- **不碰 OASBUD / BrEaST**（锁箱量，须待配方冻结后单次评估）
- **不碰任何私有队列**（F9 + 记忆硬规则）
- **不修改 `src/` 下任何文件**（F22）

多卡（2026-10-06 新增）
----------------------
本脚本持有**第二把锁**（`RunLock`，F22 单实例 + 代码漂移检测）。驱动的
分片只解决了驱动自己那把锁；若不改这里，第 2 张卡起的格会去抢
`runs/dev_screen/.running.json` 而**直接抛 RuntimeError 退出** ——
即"租了 N 张卡，实际只跑得动 1 张"。故 `--shard` 同时改写这两处。

`--gpu` 设置 `CUDA_VISIBLE_DEVICES` 把本格钉在一张卡上。
**注意**：本机记忆里有个已踩过的坑 —— `CUDA_VISIBLE_DEVICES=""`（空串）
在 Windows 上**静默失效**，必须用 `-1` 才真正禁用。故此处只在显式给了
`--gpu` 时才设置该变量；不给就完全不碰环境。

用法
----
  python scripts/_run_one_lockbox_cell.py <variant> <seed> <steps> <eval_every>
  python scripts/_run_one_lockbox_cell.py swin_t 42 3000 500 --shard 1 --gpu 1
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from breastdg import config                                          # noqa: E402
from breastdg.train.dev_screen import RunLock                        # noqa: E402
from retrain_lockbox import LOCKBOX_VARIANTS, run_one                # noqa: E402


def lock_path_for_shard(idx: int) -> pathlib.Path:
    """分片锁路径；idx=0 且未分片时等于 `RunLock` 的默认路径。

    与 `_lockbox_driver.apply_shard` 的命名保持同族，便于排障时对照。
    """
    return config.PROJECT_ROOT / "runs" / "dev_screen" / f".running.lockbox_shard{idx}.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("seed", type=int)
    ap.add_argument("steps", type=int)
    ap.add_argument("eval_every", type=int)
    ap.add_argument("--shard", type=int, default=0,
                    help="分片号；决定本格用哪把 RunLock（默认 0 = 传统单实例锁）")
    ap.add_argument("--gpu", default="",
                    help="CUDA_VISIBLE_DEVICES 值；留空则不碰该环境变量")
    a = ap.parse_args()

    if a.variant not in LOCKBOX_VARIANTS:
        print(f"未知变体 {a.variant}；可用 {list(LOCKBOX_VARIANTS)}")
        return 2

    if a.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    lockp = lock_path_for_shard(a.shard)

    print(f"[lockbox cell] {a.variant} / seed{a.seed} / {a.steps} steps "
          f"(shard={a.shard} gpu={a.gpu or 'inherit'} lock={lockp.name})", flush=True)
    with RunLock(lockp):                  # 单实例 + 代码漂移检测（F22）
        row = run_one(a.variant, a.seed, a.steps, a.eval_every)
    print(f"[lockbox cell] OK {a.variant}/seed{a.seed} "
          f"src_val_auc={row['src_val_auc']:.4f} {row['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
