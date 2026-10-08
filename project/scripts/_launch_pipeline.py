"""以**脱离式**方式启动主流水线（`_pipeline_master.py`）。

脱离式理由
----------
- 本会话后台命令有 2 h 上限，而流水线可长达数小时；
- 必须用 `DETACHED_PROCESS` 而非 PowerShell 的 `-NoNewWindow` ——
  后者把子进程绑到父控制台，父控制台销毁时投递 `CTRL_CLOSE_EVENT`
  会**静默杀掉**子进程。这几乎肯定是 2026-10-05 05:31 那次
  无错误输出停摆的机制（`.err` 里没有任何 traceback）。

用法
----
  python scripts/_launch_pipeline.py --deadline 22:10
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
MASTER = ROOT / "scripts" / "_pipeline_master.py"
LOG = ROOT / "runs" / "_logs" / "pipeline_master.out"
PIDFILE = ROOT / "runs" / "_logs" / "pipeline_master.pid"

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200


def main() -> int:
    if not MASTER.exists():
        print(f"主流水线不存在：{MASTER}")
        return 1
    LOG.parent.mkdir(parents=True, exist_ok=True)

    extra = sys.argv[1:]
    lf = open(LOG, "a", encoding="utf-8")
    p = subprocess.Popen(
        [str(PY), "-u", str(MASTER), *extra],
        cwd=str(ROOT),
        stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    PIDFILE.write_text(str(p.pid), encoding="utf-8")
    print(f"pipeline pid={p.pid}  args={' '.join(extra)}")
    print(f"stdout -> {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
