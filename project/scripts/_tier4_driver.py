"""TIER4 自动续跑驱动（Python，带**时间预算 + 夜停**）。

为什么要重写
------------
1. **原脱离式运行是挂死、不是崩溃**：PID 143892 自 2026-10-05 05:31 起 10 h
   内 CPU 累计停在 871 s、占 6.5 GB 内存并持有 CUDA 上下文，GPU 全程空转。
   崩溃可重试，挂死不可 —— 故**一格一个子进程 + 一格一个超时**。
2. PowerShell 版在本机 GBK 下解析非 ASCII 字节会语法错乱（已实际发生）。
3. 上一版重试循环每次崩溃都留陈旧锁，使后续尝试全部秒失败
   （把「一格坏」放大成「整批坏」）。本版每格前显式清陈旧锁。

时间预算与夜停（用户 2026-10-05 要求）
--------------------------------------
- `--deadline HH:MM`：本地墙钟截止时刻。**在截止前不再启动新格**，
  但**不打断已在跑的格**（宁可多跑一格，不可浪费已投入的 GPU 时间）。
- 每格的预计耗时由 `results.csv` 的历史 `minutes` **中位数**推得（数据驱动），
  据此判断"这一格还来不来得及"，而不是拍脑袋定一个常数。
- 全程记录每格实际耗时到 `runs/_logs/tier4_timing.json`，结束打印汇总表 ——
  即用户要求的「每个任务加时间监测，杜绝时间浪费」。

幂等性
------
`dev_screen` 的跳过逻辑以 `results.csv` 为准；`run_one` 在"目录已存在且非空"
时拒绝执行，故本驱动在每格前归档"无 final.pt 的僵尸目录"（归档而非删除）。

安全边界
--------
不碰 OASBUD / BrEaST / 任何私有队列（F9 + 记忆硬规则）；
**不修改 `src/` 下任何文件**（F22）。

用法
----
  python scripts/_tier4_driver.py --deadline 22:15
  python scripts/_tier4_driver.py --deadline 22:15 --min-remaining 40
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
RUNS = ROOT / "runs" / "dev_screen"
LOGDIR = ROOT / "runs" / "_logs"
CELLDIR = LOGDIR / "tier4_cells"
DRIVER_LOG = LOGDIR / "tier4_driver.log"
TIMING = LOGDIR / "tier4_timing.json"
LOCK = RUNS / ".running.json"
CSV_PATH = RUNS / "results.csv"
CELL_RUNNER = ROOT / "scripts" / "_run_one_cell.py"

VARIANTS = ["swin_t", "densenet121_se", "efficientnet_v2_s"]
DIRECTIONS = ["busbra", "busi"]
SEEDS = [42, 123, 2026]
STEPS = 3000
EVAL_EVERY = 500

CELL_TIMEOUT_S = 150 * 60          # 单格硬超时：最慢实测 52 min，留 ~3 倍余量
MAX_SWEEPS = 5

# 历史缺数据时的兜底估计（分钟）。一旦 results.csv 有该变体的记录就用实测中位数。
FALLBACK_MIN = {"swin_t": 40, "densenet121_se": 26, "efficientnet_v2_s": 30}


# --------------------------------------------------------------------------
def note(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Windows 分离进程 stdout 默认 GBK，遇到 ▶ ✅ 等字符会抛
        # UnicodeEncodeError 并**直接打死进程**（2026-10-05 实际发生，
        # 使流水线在启动 E7 的瞬间崩溃、GPU 空转）。降级为 ASCII。
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(DRIVER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _child_env() -> dict:
    """给子进程注入 PYTHONIOENCODING=utf-8。

    本机（Windows）子进程 stdout 默认 GBK；训练脚本里只要有一处 GBK 编不了码的
    字符（实测：θ̂ 的组合抑扬符 U+0302、U+2212、⭐、µm…）就会在 print 时抛
    UnicodeEncodeError 打死进程。2026-10-05 实际因此把流水线打死一次。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return str(pid) in out


def clear_stale_lock() -> None:
    """锁的持有进程若已不存在则删除；若仍存活则**等它结束**。

    上一版驱动的致命缺陷：崩溃后没清锁 → 后续每次尝试秒失败。
    """
    if not LOCK.exists():
        return
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
        pid = int(info.get("pid", -1))
    except Exception:
        pid = -1
    if pid > 0 and pid_alive(pid):
        note(f"锁被存活进程 {pid} 持有（很可能是上一格的训练），等它结束")
        t0 = time.time()
        while pid_alive(pid):
            if time.time() - t0 > CELL_TIMEOUT_S + 600:
                note(f"等待 pid={pid} 超过上限，强制清除锁")
                LOCK.unlink(missing_ok=True)
                return
            time.sleep(20)
        # 锁的持有者已退出：其 __exit__ 通常会删锁；若没删，下一轮进来清。
        return
    LOCK.unlink(missing_ok=True)
    note(f"清除陈旧锁（pid={pid} 已不存在）")


def archive_zombie(variant: str, direction: str, seed: int) -> None:
    """归档"非空但没有 final.pt"的崩溃残留，否则 run_one 拒绝启动。

    ⚠ 只在**确认无训练在跑**时调用（调用点都在 clear_stale_lock 之后）。
    """
    base = RUNS / variant / direction / f"seed{seed}"
    if not base.is_dir():
        return
    arch = RUNS / "_archive" / f"_orphans_{time.strftime('%Y%m%d')}"
    for d in sorted(base.iterdir()):
        if not d.is_dir() or (d / "final.pt").exists():
            continue
        dest = arch / f"{variant}_{direction}_seed{seed}_{d.name}"
        arch.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(d), str(dest))
        note(f"归档僵尸目录 {d.relative_to(RUNS)} → {dest.name}")


def done_cells() -> set[tuple[str, str, int]]:
    """以 results.csv 为准判断已完成（与 dev_screen 自身口径一致）。"""
    done: set[tuple[str, str, int]] = set()
    if not CSV_PATH.exists():
        return done
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("steps") or "").strip()
            if not s or int(s) < STEPS:
                continue
            try:
                t = (r["variant"], r["direction"], int(r["seed"]))
            except (KeyError, ValueError):
                continue
            if t[0] in VARIANTS and t[1] in DIRECTIONS and t[2] in SEEDS:
                done.add(t)
    return done


def hist_minutes(variant: str) -> tuple[float, int]:
    """该变体的历史单格耗时中位数（分钟）与样本数。数据驱动，不用常数。"""
    vals: list[float] = []
    if CSV_PATH.exists():
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("variant") != variant:
                    continue
                if (r.get("steps") or "").strip() != str(STEPS):
                    continue
                try:
                    m = float(r.get("minutes") or "")
                except ValueError:
                    continue
                if m > 0:
                    vals.append(m)
    if vals:
        return float(statistics.median(vals)), len(vals)
    return float(FALLBACK_MIN.get(variant, 40)), 0


def _kill_children() -> None:
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              "Where-Object { $_.CommandLine -like '*_run_one_cell*' } | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
              "-ErrorAction SilentlyContinue }")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=60)
    except Exception:
        pass


def run_cell(variant: str, direction: str, seed: int) -> tuple[bool, float]:
    # ⚠ 顺序关键：先等锁释放（可能是在跑的那一格），**再**重新判定是否已完成。
    #
    # 动机（否则必错）：main 在每轮开头算一次 `left`。若此刻恰好有一格在跑，
    # 它尚未写入 results.csv，因而被算进 `left`；等它跑完释放锁后再无条件
    # 启动一次，run_one 就会因"目录已有 final.pt"抛错，白白浪费一格的时间。
    # 故这里在拿到锁之后重新查一次 results.csv。
    clear_stale_lock()
    if (variant, direction, seed) in done_cells():
        note(f"跳过 {variant}/{direction}/seed{seed}：等待期间已完成")
        return True, 0.0

    archive_zombie(variant, direction, seed)

    CELLDIR.mkdir(parents=True, exist_ok=True)
    cpath = CELLDIR / f"{variant}_{direction}_seed{seed}.log"
    cmd = [str(PY), "-u", str(CELL_RUNNER), variant, direction,
           str(seed), str(STEPS), str(EVAL_EVERY)]
    note(f"启动 {variant}/{direction}/seed{seed}（硬超时 {CELL_TIMEOUT_S//60} min）")
    t0 = time.time()
    with open(cpath, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"{' '.join(cmd[2:])} =====\n")
        lf.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, env=_child_env(),
                                timeout=CELL_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            mins = (time.time() - t0) / 60
            note(f"⚠ 超时 {variant}/{direction}/seed{seed}（{mins:.0f} min）"
                 f" —— 判为挂死，杀掉后继续")
            _kill_children()
            clear_stale_lock()
            archive_zombie(variant, direction, seed)
            return False, mins
        except Exception as e:                       # noqa: BLE001
            note(f"异常 {type(e).__name__}: {e}")
            return False, (time.time() - t0) / 60

    mins = (time.time() - t0) / 60
    ok = rc == 0 and (variant, direction, seed) in done_cells()
    note(f"{'✅ 完成' if ok else '❌ 失败'} {variant}/{direction}/seed{seed} "
         f"rc={rc} {mins:.1f} min")
    if not ok:
        clear_stale_lock()
        archive_zombie(variant, direction, seed)
    return ok, mins


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="",
                    help="本地截止时刻 HH:MM（今天）。留空=不限时。")
    ap.add_argument("--min-remaining", type=float, default=0.0,
                    help="启动新格所需的最少剩余分钟（0=用该变体预计耗时）。")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="变体顺序（逗号分隔）。默认按'便宜的先跑'，"
                         "以便尽早把某个骨干跑满 3 种子 —— 论文要的是"
                         "'五骨干'整体成立，而非某个骨干多个种子。")
    a = ap.parse_args()

    order = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in order if v not in VARIANTS]
    if bad:
        ap.error(f"未知变体 {bad}；可用 {VARIANTS}")

    if not PY.exists():
        note(f"解释器不存在：{PY}")
        return 1

    deadline: datetime | None = None
    if a.deadline:
        hh, mm = (int(x) for x in a.deadline.split(":"))
        now = datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            note(f"截止时刻 {a.deadline} 已过（现在 {now:%H:%M}），不启动任何新格")
            return 0
        note(f"截止时刻 {deadline:%Y-%m-%d %H:%M}"
             f"（剩余 {((deadline-now).total_seconds()/60):.0f} min）")

    note(f"driver 启动 pid={os.getpid()}")
    total_cells = len(VARIANTS) * len(DIRECTIONS) * len(SEEDS)
    note(f"目标：{len(VARIANTS)} 变体 × {len(DIRECTIONS)} 方向 × {len(SEEDS)} 种子"
         f" = {total_cells} 格")
    note(f"执行顺序：{' → '.join(order)}")

    for v in order:
        med, n = hist_minutes(v)
        note(f"  预计耗时 {v}: {med:.0f} min（来自 {n} 条历史实测）")

    ledger: list[dict] = []
    if TIMING.exists():
        try:
            ledger = json.loads(TIMING.read_text(encoding="utf-8"))
        except Exception:
            ledger = []

    for sweep in range(1, MAX_SWEEPS + 1):
        done = done_cells()
        left = [(v, d, s) for v in order for d in DIRECTIONS for s in SEEDS
                if (v, d, s) not in done]
        note(f"--- 第 {sweep} 轮：已完成 {len(done)}/{total_cells}，待跑 {len(left)} ---")
        if not left:
            note("全部完成，driver 退出")
            break

        stopped_for_time = False
        for (v, d, s) in left:
            if deadline is not None:
                remain = (deadline - datetime.now()).total_seconds() / 60
                need = a.min_remaining or hist_minutes(v)[0]
                if remain < need:
                    note(f"⏹ 夜停：剩余 {remain:.0f} min < 该格预计 {need:.0f} min，"
                         f"不再启动新格（已完成的不浪费）")
                    stopped_for_time = True
                    break
                note(f"剩余预算 {remain:.0f} min，本格预计 {need:.0f} min → 继续")

            ok, mins = run_cell(v, d, s)
            ledger.append(dict(variant=v, direction=d, seed=s, ok=ok,
                               minutes=round(mins, 1),
                               at=time.strftime("%Y-%m-%d %H:%M:%S")))
            TIMING.write_text(json.dumps(ledger, indent=2, ensure_ascii=False),
                              encoding="utf-8")

        if stopped_for_time:
            break

    # ---- 时间汇总（用户要求：每个任务加时间监测）----
    n, tot = done_cells()
    note(f"结束：TIER4 完成 {n}/{total_cells}")
    if ledger:
        note("本 session 各格实测耗时：")
        for r in ledger:
            note(f"   {r['variant']}/{r['direction']}/seed{r['seed']} "
                 f"{'OK ' if r['ok'] else 'FAIL'} {r['minutes']:>6.1f} min")
        good = [r["minutes"] for r in ledger if r["ok"]]
        if good:
            note(f"成功格合计 {sum(good):.0f} min，中位 {statistics.median(good):.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
