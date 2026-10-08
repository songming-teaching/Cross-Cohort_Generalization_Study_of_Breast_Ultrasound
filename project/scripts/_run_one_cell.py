"""跑 TIER4 的**单个格子**（一次训练）。

用途：给 `_tier4_driver.py` 当子进程用——每格一个新进程，
好处是挂死能被超时杀掉而不牵连其它格子，且进程退出即释放
CUDA 上下文与内存（上一轮事故正是进程挂死、持有 6.5 GB
内存与 CUDA 上下文长达 10 小时，GPU 全程空转）。

本脚本**不修改** `src/` 下任何文件（F22），只做导入与调用。

用法
----
  python scripts/_run_one_cell.py <variant> <direction> <seed> <steps> <eval_every>
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, "src")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from breastdg.train.dev_screen import RunLock, run_one   # noqa: E402


def main() -> int:
    if len(sys.argv) != 6:
        print(__doc__)
        return 2
    variant, direction = sys.argv[1], sys.argv[2]
    seed, steps, eval_every = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    print(f"[cell] {variant} / {direction} / seed{seed} / {steps} steps", flush=True)
    with RunLock():                       # 单实例 + 代码漂移检测（F22）
        row = run_one(variant, direction, seed, steps, eval_every)
    print(f"[cell] OK {variant}/{direction}/seed{seed} "
          f"dev_erm={row['dev_auc_erm']:.4f} dev_final={row['dev_auc_final']:.4f} "
          f"{row['minutes']} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
