"""以**脱离式**方式启动 TIER4 续跑驱动。

为什么要脱离式：本会话的后台命令有 2 小时上限，而 TIER4 剩余 15 格需
~9–10 h。先前两次用普通后台跑都被上限杀掉。

为什么要 `DETACHED_PROCESS` 而不是 PowerShell 的 `-NoNewWindow`：
`-NoNewWindow` 把子进程绑到父控制台，父控制台销毁时会投递
CTRL_CLOSE_EVENT → 子进程被静默杀死。这几乎肯定是上一轮脱离式训练
在 05:31 无任何错误输出就停摆的机制（`.err` 里只有一条无关警告，
没有 traceback）。DETACHED_PROCESS 让子进程完全不拥有控制台。

用法
----
  python scripts/_launch_tier4_driver.py --deadline 22:15 \
      --variants densenet121_se,efficientnet_v2_s,swin_t

⚠ 本脚本会把**自己的全部命令行参数**透传给驱动。曾经忘记透传，
  导致 `--deadline` 与 `--variants` 被静默丢弃、按最贵的骨干先跑、
  且没有夜停（已实际发生并修正，见工作日志 5.9）。
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
DRIVER = ROOT / "scripts" / "_tier4_driver.py"
LOG = ROOT / "runs" / "_logs" / "tier4_driver.out"
PIDFILE = ROOT / "runs" / "_logs" / "tier4_driver.pid"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def main() -> int:
    if not DRIVER.exists():
        print(f"驱动不存在：{DRIVER}")
        return 1
    LOG.parent.mkdir(parents=True, exist_ok=True)

    extra = sys.argv[1:]                      # ← 必须透传，见上方注释
    cmd = [str(PY), "-u", str(DRIVER), *extra]

    lf = open(LOG, "a", encoding="utf-8")
    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    PIDFILE.write_text(str(p.pid), encoding="utf-8")
    print(f"driver pid={p.pid}")
    print(f"driver args: {' '.join(extra) if extra else '(无)'}")
    print(f"driver stdout -> {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
