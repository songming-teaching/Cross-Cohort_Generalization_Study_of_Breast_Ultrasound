"""主流水线：等 TIER4 跑完，自动接续全部可行阶段（带**时间预算 + 夜停**）。

为什么要它
----------
TIER4 若跑完后无人接手，GPU 会空转一整夜 —— 这正是 2026-10-05 05:31
那次事故的翻版（进程挂死 10 h、GPU 利用率 0%）。本流水线让 GPU 不停工。

时间预算与夜停（用户 2026-10-05 要求）
--------------------------------------
- 用户当天剩余工作时间约 6 h，**夜间不跑**（网络太差）。
- `--deadline HH:MM`：本地墙钟截止。截止前不再**启动**新阶段，但**不打断
  已在跑的阶段**（不浪费已投入的算力）。
- 每个阶段记录墙钟耗时，结束打印汇总表。

阶段（严格串行 —— 8 GB 显存同一时刻只能装一个活）
-------------------------------------------------
  1. E7 对齐轨迹（GPU 特征提取，用已有 dev 检查点）
  2. E6 Grad-CAM（GPU）
  3. 锁箱全源重训 5 骨干 × 3 种子 = 15 格
     → 直接调用既有入口 `scripts/retrain_lockbox.py`（唯一实现，不重复）

**刻意不含**「锁箱评估（OASBUD 双管线 + BrEaST）」：
那是**不可逆的一次性锁箱量**，且 `evaluate_lockbox.py` 从未在真实数据上跑过。
在验证脚本正确性之前烧掉锁箱，是无可挽回的科学损失 ——
故流水线到此停下并明确记录，等作者确认后单次执行。

加固（照搬 TIER4 事故的教训）
----------------------------
- 每阶段独立子进程 + 独立超时 → 挂死只损失一个阶段；
- 每阶段前清陈旧锁（校验持有 pid 存活）→ 杜绝「一格坏放大成整批坏」；
- 全程日志，随时可中断、可重入（幂等）。

安全边界
--------
不碰 OASBUD / BrEaST / 任何私有队列；不修改 `src/` 下任何文件（F22）。

用法
----
  python scripts/_pipeline_master.py --deadline 22:10
  python scripts/_launch_pipeline.py --deadline 22:10     # 脱离式（推荐）
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = ROOT.parent / ".venv" / "Scripts" / "python.exe"
RUNS = ROOT / "runs" / "dev_screen"
LOGDIR = ROOT / "runs" / "_logs"
MASTER_LOG = LOGDIR / "pipeline_master.log"
TIMING = LOGDIR / "pipeline_timing.json"
LOCK = RUNS / ".running.json"
CSV_PATH = RUNS / "results.csv"

TIER4_VARIANTS = ["swin_t", "densenet121_se", "efficientnet_v2_s"]
TIER4_DIRS = ["busbra", "busi"]
TIER4_SEEDS = [42, 123, 2026]
TIER4_STEPS = 3000

STAGE_TIMEOUT_S = 150 * 60      # 单阶段硬超时
WAIT_TIER4_MAX_H = 8.0          # 等 TIER4 的上限（当日 6 h 预算内）


def note(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # ⚠ 本机（Windows）分离进程的 stdout 默认是 GBK，遇到 ▶ ✅ 这类字符会抛
        # **UnicodeEncodeError**。2026-10-05 已实际因此把流水线打死：它在 TIER4
        # 跑满、正要启动 E7 的那一刻崩溃，导致 GPU 空转。
        # 故此处降级为 ASCII 安全输出 —— **日志的写入（下行）优先于控制台美观**。
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    LOGDIR.mkdir(parents=True, exist_ok=True)
    with open(MASTER_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return str(pid) in out


def clear_stale_lock() -> None:
    if not LOCK.exists():
        return
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
        pid = int(info.get("pid", -1))
    except Exception:
        pid = -1
    if pid > 0 and pid_alive(pid):
        note(f"锁被存活进程 {pid} 持有，等它结束")
        while pid_alive(pid):
            time.sleep(20)
        return
    LOCK.unlink(missing_ok=True)
    note(f"清除陈旧锁（pid={pid} 已不存在）")


# --------------------------------------------------------------------------
def tier4_done() -> tuple[int, int]:
    done: set[tuple[str, str, int]] = set()
    if CSV_PATH.exists():
        import csv
        with open(CSV_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                s = (r.get("steps") or "").strip()
                if not s or int(s) < TIER4_STEPS:
                    continue
                try:
                    t = (r["variant"], r["direction"], int(r["seed"]))
                except (KeyError, ValueError):
                    continue
                if t[0] in TIER4_VARIANTS and t[1] in TIER4_DIRS and t[2] in TIER4_SEEDS:
                    done.add(t)
    total = len(TIER4_VARIANTS) * len(TIER4_DIRS) * len(TIER4_SEEDS)
    return len(done), total


def wait_for_tier4(deadline: datetime | None) -> bool:
    t0 = time.time()
    last = None
    while True:
        n, total = tier4_done()
        if n != last:
            note(f"TIER4 进度 {n}/{total}")
            last = n
        if n >= total:
            note("TIER4 已满")
            return True
        if deadline is not None and datetime.now() >= deadline:
            note(f"到截止时刻，TIER4 仅 {n}/{total} —— 停止本日流水线")
            return False
        if (time.time() - t0) / 3600 > WAIT_TIER4_MAX_H:
            note(f"等 TIER4 超 {WAIT_TIER4_MAX_H} h（{n}/{total}），停止等待")
            return False
        time.sleep(60)


# --------------------------------------------------------------------------
def _kill_by_marker(marker: str) -> None:
    try:
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              f"Where-Object {{ $_.CommandLine -like '*{marker}*' }} | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
              "-ErrorAction SilentlyContinue }")
        subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       capture_output=True, timeout=60)
    except Exception:
        pass


def budget_left(deadline: datetime | None) -> float:
    """剩余预算（分钟）；无截止则视为无限。"""
    if deadline is None:
        return float("inf")
    return (deadline - datetime.now()).total_seconds() / 60


def run_stage(name: str, args: list[str], marker: str,
              deadline: datetime | None, est_min: float,
              ledger: list[dict]) -> bool:
    """跑一个阶段（独立子进程 + 硬超时）。预算不足则跳过。"""
    script = ROOT / "scripts" / args[0]
    if not script.exists():
        note(f"跳过 {name}：脚本不存在 {script}")
        return False

    remain = budget_left(deadline)
    if remain < est_min:
        note(f"⏹ 预算不足（剩 {remain:.0f} min < 预计 {est_min:.0f} min），"
             f"跳过 {name}")
        ledger.append(dict(stage=name, status="skipped_budget",
                           minutes=None, at=time.strftime("%Y-%m-%d %H:%M:%S")))
        return False

    clear_stale_lock()
    logp = LOGDIR / f"pipeline_{name}.log"
    # ⚠ 必须用**脚本的完整路径**，不能用裸文件名：cwd 是 project 根，
    # 而脚本在 scripts/ 下。2026-10-05 实际踩到 —— 三个阶段全部
    # `fail_rc2`（"can't open file .../project/alignment_trajectory.py"）。
    # args[0] 是相对 scripts/ 的文件名，其余是参数，故此处做替换而非拼接。
    cmd = [str(PY), "-u", str(script), *args[1:]]
    note(f"▶ 开始 {name}（预计 {est_min:.0f} min，剩 {remain:.0f} min）"
         f"：{' '.join(args)}")
    t0 = time.time()
    status, rc = "ok", None
    # ⚠ 子进程**必须**带 PYTHONIOENCODING=utf-8。
    # 本机（Windows）分离进程的 stdout 默认是 GBK；脚本里只要有一个 GBK 编不了码的
    # 字符就会在 print 时抛 UnicodeEncodeError 直接打死进程。实测这些字符全部编不了：
    #   θ̂(θ+组合抑扬) − (U+2212) ✅ ▶ ↔ ⚠
    # 2026-10-05 已实际因此把流水线打死一次（TIER4 跑满、正要启动 E7 的那一刻崩溃，
    # GPU 空转）。此处统一注入，**一次保护所有阶段**，比逐个脚本改 print 可靠。
    child_env = dict(os.environ)
    child_env["PYTHONIOENCODING"] = "utf-8"
    with open(logp, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)} =====\n")
        lf.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, env=child_env,
                                timeout=STAGE_TIMEOUT_S).returncode
            if rc != 0:
                status = f"fail_rc{rc}"
        except subprocess.TimeoutExpired:
            status = "timeout"
            note(f"⚠ {name} 超时（{STAGE_TIMEOUT_S//60} min），杀掉后继续")
            _kill_by_marker(marker)
        except Exception as e:                       # noqa: BLE001
            status = f"error:{type(e).__name__}"
            note(f"⚠ {name} 异常 {type(e).__name__}: {e}")

    mins = (time.time() - t0) / 60
    note(f"{'✅' if status == 'ok' else '❌'} {name} {status} {mins:.1f} min"
         f"  日志 {logp.name}")
    ledger.append(dict(stage=name, status=status, rc=rc,
                       minutes=round(mins, 1),
                       at=time.strftime("%Y-%m-%d %H:%M:%S")))
    TIMING.write_text(json.dumps(ledger, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return status == "ok"


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", default="",
                    help="本地截止时刻 HH:MM（今天）。留空=不限时。")
    ap.add_argument("--skip-wait", action="store_true",
                    help="不等待 TIER4（TIER4 由它自己的驱动负责）")
    ap.add_argument("--stages", default="e7,e6",
                    help="要跑的阶段，逗号分隔：e7,e6,lockbox。默认 e7,e6 —— "
                         "**不含 lockbox**，因为锁箱全源重训已定于云端执行"
                         "（裁决 R-7：迁移的单位是完整比较组，锁箱 15 格全是新建的）。"
                         "在本机跑锁箱会与明天的云端任务重复，纯属浪费。")
    a = ap.parse_args()

    stages = {s.strip() for s in a.stages.split(",") if s.strip()}
    unknown = stages - {"e7", "e6", "lockbox"}
    if unknown:
        ap.error(f"未知阶段 {sorted(unknown)}；可用 e7,e6,lockbox")

    if not PY.exists():
        note(f"解释器不存在：{PY}")
        return 1

    deadline: datetime | None = None
    if a.deadline:
        hh, mm = (int(x) for x in a.deadline.split(":"))
        now = datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        note(f"截止时刻 {deadline:%Y-%m-%d %H:%M}"
             f"（剩余 {budget_left(deadline):.0f} min）")

    note(f"流水线启动 pid={os.getpid()}")
    ledger: list[dict] = []
    if TIMING.exists():
        try:
            ledger = json.loads(TIMING.read_text(encoding="utf-8"))
        except Exception:
            ledger = []

    if a.skip_wait:
        note("按参数跳过等待 TIER4")
    else:
        note("阶段 0：等待 TIER4")
        if not wait_for_tier4(deadline):
            note("TIER4 未完成 —— 结束本日流水线（不启动后续阶段，避免争显存）")
            return 0

    # TIER4 驱动需要时间释放自己的锁与清理
    time.sleep(30)
    clear_stale_lock()

    # ---- 阶段 1：E7 对齐轨迹（估 15 min）----
    if "e7" in stages:
        run_stage("e7_alignment",
                  ["alignment_trajectory.py",
                   "--pairs", "busi-busbra,busi-busuclm",
                   "--variants", "baseline,no_domainhead",
                   "--direction", "busbra",
                   "--seeds", "42,123,2026",
                   "--stages", "best_erm,final"],
                  marker="alignment_trajectory", deadline=deadline,
                  est_min=15.0, ledger=ledger)
    else:
        note("按 --stages 跳过 E7")

    # ---- 阶段 2：E6 Grad-CAM（估 10 min × 2）----
    if "e6" in stages:
        for v in ("baseline", "no_domainhead"):
            run_stage(f"e6_gradcam_{v}",
                      ["gradcam_interpretability.py",
                       "--variant", v, "--direction", "busbra",
                       "--seed", "42", "--which", "final",
                       "--cohort", "busi"],
                      marker="gradcam_interpretability", deadline=deadline,
                      est_min=10.0, ledger=ledger)
    else:
        note("按 --stages 跳过 E6")

    # ---- 阶段 3：锁箱全源重训 ----
    # **默认不跑**：锁箱 15 格已定于云端执行（裁决 R-7 —— 迁移的单位是
    # 完整比较组，锁箱全是新建的，零混杂）。在本机跑会与明天的云端任务
    # 重复，纯属浪费。如需在本地跑，显式加 `--stages e7,e6,lockbox`。
    if "lockbox" in stages:
        note("⚠ 本机执行锁箱重训 —— 若已计划上云，这会重复劳动")
        run_lockbox_gated(deadline, ledger)
    else:
        note("按 --stages 跳过锁箱重训（已定于云端执行）")

    # ---- 时间汇总 ----
    note("=" * 70)
    note("本 session 阶段耗时汇总：")
    for r in ledger:
        m = f"{r['minutes']:>6.1f} min" if r.get("minutes") is not None else "     —"
        note(f"   {r.get('stage', '?'):24s} {r.get('status', '?'):14s} {m}")
    done = [r for r in ledger if r.get("status") == "ok" and r.get("minutes")]
    if done:
        note(f"   成功阶段合计 {sum(r['minutes'] for r in done):.0f} min，"
             f"中位 {statistics.median([r['minutes'] for r in done]):.1f} min")
    note("流水线结束。**锁箱评估（OASBUD 双管线 + BrEaST）未执行** ——")
    note("  那是不可逆的一次性锁箱量，须先验证 evaluate_lockbox.py 的正确性，")
    note("  经作者确认后单次执行。")
    return 0


def run_lockbox_gated(deadline: datetime | None, ledger: list[dict]) -> None:
    """逐格跑锁箱重训，精确控制时间预算。

    为什么不直接调 `retrain_lockbox.py --variants all`：
    它是"一口气跑 15 格"的形态，无法在截止时刻优雅停下，
    也无法把单格挂死隔离到一格。故这里**逐格**拉起
    `_run_one_lockbox_cell.py`（每格独立子进程 + 超时）。
    """
    variants = ["resnet50_cbam", "swin_t", "densenet121_se",
                "efficientnet_v2_s", "fdnet_gate_dual"]
    seeds = [42, 123, 2026]
    steps, eval_every = 3000, 500
    cell_timeout_s = 180 * 60
    est = {"resnet50_cbam": 25, "swin_t": 40, "densenet121_se": 26,
           "efficientnet_v2_s": 30, "fdnet_gate_dual": 45}

    CELL = ROOT / "scripts" / "_run_one_lockbox_cell.py"
    if not CELL.exists():
        note("跳过锁箱重训：_run_one_lockbox_cell.py 不存在")
        return

    note("阶段 3：锁箱全源重训（逐格，5 骨干 × 3 种子 = 15 格）")
    for v in variants:
        for s in seeds:
            remain = budget_left(deadline)
            if remain < est.get(v, 40):
                note(f"⏹ 预算不足（剩 {remain:.0f} min < {v} 预计 "
                     f"{est.get(v, 40)} min），锁箱重训本日到此为止")
                ledger.append(dict(stage=f"lockbox:{v}/seed{s}",
                                   status="skipped_budget", minutes=None,
                                   at=time.strftime("%Y-%m-%d %H:%M:%S")))
                TIMING.write_text(json.dumps(ledger, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
                return

            clear_stale_lock()
            logp = LOGDIR / "lockbox_cells" / f"{v}_seed{s}.log"
            logp.parent.mkdir(parents=True, exist_ok=True)
            name = f"lockbox:{v}/seed{s}"
            note(f"▶ {name}（预计 {est.get(v, 40)} min，剩 {remain:.0f} min）")
            t0 = time.time()
            status, rc = "ok", None
            with open(logp, "a", encoding="utf-8") as lf:
                lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                         f"{v} seed{s} =====\n")
                lf.flush()
                try:
                    rc = subprocess.run(
                        [str(PY), "-u", str(CELL), v, str(s),
                         str(steps), str(eval_every)],
                        cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                        env=child_env, timeout=cell_timeout_s).returncode
                    if rc != 0:
                        status = f"fail_rc{rc}"
                except subprocess.TimeoutExpired:
                    status = "timeout"
                    note(f"⚠ {name} 超时，判为挂死，杀掉后继续")
                    _kill_by_marker("_run_one_lockbox_cell")
                except Exception as e:               # noqa: BLE001
                    status = f"error:{type(e).__name__}"
                    note(f"⚠ {name} 异常 {type(e).__name__}: {e}")

            mins = (time.time() - t0) / 60
            note(f"{'✅' if status == 'ok' else '❌'} {name} {status} {mins:.1f} min")
            ledger.append(dict(stage=name, status=status, rc=rc,
                               minutes=round(mins, 1),
                               at=time.strftime("%Y-%m-%d %H:%M:%S")))
            TIMING.write_text(json.dumps(ledger, indent=2, ensure_ascii=False),
                              encoding="utf-8")
            est[v] = mins if status == "ok" else est[v]      # 用实测更新预计


if __name__ == "__main__":
    sys.exit(main())
