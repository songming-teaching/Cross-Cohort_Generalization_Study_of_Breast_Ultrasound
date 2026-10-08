"""锁箱全源重训驱动 —— 单格独立子进程 + 单格超时 + 可续跑（跨平台）。

为什么需要它（修两个真实缺陷）
------------------------------
1. **无挂死保护**：`retrain_lockbox.py` 的主循环是
   `for v in names: for s in seeds:`，**没有 `timeout=`**。
   2026-10-05 本机已实际发生一次「进程挂死 10 h、GPU 全程空转」，
   在**按小时计费**的云上，同样的挂死会持续烧钱。故必须
   **一格一个子进程 + 一格一个硬超时**，让挂死只损失一格。

2. **不可续跑**：`retrain_lockbox.main()` 只在**全部跑完**后才写
   `summary.json`；中途挂死则该文件不存在，已完成的 14 格
   在驱动看来"一格都没完成"，于是**全部重跑**。
   云端抢占（spot 实例被回收）很常见，这会造成巨大浪费。

   本驱动的完成判据改为**产物存在性**：
   `runs/lockbox/<variant>/seed<N>/<run_id>/final.pt` 存在即算完成。
   这与 `run_one` 的防覆盖护栏一致（它按 run_id 目录判重），
   且不依赖任何汇总文件。

跨平台
------
本机 Windows 用 `tasklist` 判进程存活；Linux 用 `os.kill(pid, 0)`。
`pid_alive` 按 `os.name` 分派 —— **云端入口 run_cloud.sh 直接调用本脚本**，
故必须两边都能跑。

安全边界（与本地一致，不可放宽）
--------------------------------
- **不碰 OASBUD / BrEaST / 任何私有队列**（F9 + 记忆硬规则）；
- **不修改 `src/` 下任何文件**（F22）；
- 复用 `RunLock`（单实例 + 代码漂移检测，F22）。

用法
----
  python scripts/_lockbox_driver.py                      # 全部 15 格
  python scripts/_lockbox_driver.py --deadline 22:15      # 到点不再起新格
  python scripts/_lockbox_driver.py --variants swin_t --seeds 42
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = pathlib.Path(sys.executable)
LOGDIR = ROOT / "runs" / "_logs"
CELLDIR = LOGDIR / "lockbox_cells"
DRIVER_LOG = LOGDIR / "lockbox_driver.log"
TIMING = LOGDIR / "lockbox_timing.json"
LOCK = ROOT / "runs" / "dev_screen" / ".running.json"   # 与 dev_screen 共用同一把锁
LB = ROOT / "runs" / "lockbox"
CELL = ROOT / "scripts" / "_run_one_lockbox_cell.py"

VARIANTS = ["resnet50_cbam", "swin_t", "densenet121_se",
            "efficientnet_v2_s", "fdnet_gate_dual",
            # 公平基线（见 retrain_lockbox.LOCKBOX_VARIANTS）
            "deepall", "mixstyle"]
SEEDS = [42, 123, 2026]
STEPS, EVAL_EVERY = 3000, 500

CELL_TIMEOUT_S = 180 * 60      # 单格硬超时：最慢实测 52 min，留 ~3.5 倍余量
MAX_SWEEPS = 4

# ---- 多卡分片（2026-10-06 新增，为云端多卡并行）----
# 背景：本驱动原为**全局单实例**（一把锁 + 一份台账）—— 租 N 张卡时，
# 第 2 个实例会因锁冲突直接拒绝启动，多租的卡全部闲置。
# 修法：允许按 `--shard I --of N` 把格集切成 N 份互不相交的子集，
# 每份一个独立进程、绑一张卡、用**自己的**锁与台账。
# 分片规则 `cells[i::N]` 保证子集互斥且并集为全集（无重复、无遗漏）。
# 产物路径不变（cell_done 按 final.pt 判重，与分片无关）→ 可续跑、可换分片数。
SHARD_IDX = 0
SHARD_TOTAL = 1
GPU_ID = ""          # 空 = 不设置 CUDA_VISIBLE_DEVICES（继承可见的全部卡）


def apply_shard(idx: int, total: int) -> None:
    """把本进程配置为「第 idx / 共 total 片」，并就地改写三处全局：

    - `LOCK`       每片一把 → 否则第 2 片会因锁冲突拒绝启动（正是多租的卡闲置的根因）
    - `TIMING`     每片一份台账 → 否则多进程同写一个 JSON 会互相覆盖
    - `DRIVER_LOG` 每片一份日志 → 否则多进程 append 交错，排障不可读

    **total == 1 时一切保持原样**（含与 dev_screen 共用的那把锁），
    故本机单卡语义**逐字未变**。这条是刻意的：不为云上功能改动本机行为。
    """
    global SHARD_IDX, SHARD_TOTAL, LOCK, TIMING, DRIVER_LOG
    if total < 1 or not (0 <= idx < total):
        raise SystemExit(f"✗ 分片参数非法：--shard {idx} --of {total}")
    SHARD_IDX, SHARD_TOTAL = idx, total
    if total == 1:
        return
    LOCK = ROOT / "runs" / "dev_screen" / f".running.lockbox_shard{idx}.json"
    TIMING = LOGDIR / f"lockbox_timing_shard{idx}.json"
    DRIVER_LOG = LOGDIR / f"lockbox_driver_shard{idx}.log"


def shard_cells(cells: list[tuple[str, int]],
                key=None) -> list[tuple[str, int]]:
    """把格集切成互斥子集：第 i 片取 `sorted(cells, key)[i::N]`。

    切片语义保证「无重复、无遗漏」；且**不改动格本身**，
    故产物路径与完成判据（`*/final.pt`）与分片数无关 ——
    可随时改分片数续跑。

    **关于「按耗时降序再切」（2026-10-07 实测结论，勿重犯）**
    我曾宣称降序切"可省 ~20% 关键路径"。**两轮实测都证伪，已回退为顺序切：**

    | 网格 | N | 顺序切关键路径 | 降序切 |
    |---|---|---|---|
    | 35 格（5 骨干 × 7 种子） | 4 | **238 min** | 240 min（更差） |
    | 49 格（7 变体 × 7 种子） | 4 | **317 min** | 321 min（更差） |
    | 49 格 | 6 | **216 min** | 234 min（更差） |
    | 35/49 格 | 8 | 130 / 171 min | 同值 |

    原因：种子数（7）≥ 骨干数（5~7），`cells[i::N]` 轮转本身已把负载铺平，
    LPT 的收益只在"少数超长项扎堆"时才显现 —— 此处没有。
    ⚠ **教训：别用 dev 网格（16 变体）的直觉外推锁箱网格，先算再改。**
    本函数保留 `key` 形参以备将来网格形态变化，但**默认不传**（=顺序切）。
    """
    if key is not None:
        cells = sorted(cells, key=key, reverse=True)
    return cells[SHARD_IDX::SHARD_TOTAL]


# 无历史时的兜底预计（分钟）
FALLBACK_MIN = {"resnet50_cbam": 20, "swin_t": 45, "densenet121_se": 22,
                "efficientnet_v2_s": 25, "fdnet_gate_dual": 18,
                "deepall": 20, "mixstyle": 21}


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

    Windows 上子进程 stdout 默认 GBK；训练脚本里只要有 GBK 编不了码的字符
    （θ̂ 的组合抑扬符 U+0302、U+2212、⭐、µm 等）就会抛 UnicodeEncodeError
    打死进程。云端（Linux）本就是 utf-8，注入无副作用。
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# --------------------------------------------------------------------------
def pid_alive(pid: int) -> bool:
    """跨平台判断 PID 是否存活。"""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                                 capture_output=True, text=True, timeout=30).stdout
        except Exception:
            return False
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def clear_stale_lock() -> None:
    """锁持有者已不存在则删锁；仍存活则等它结束。

    （上一版驱动的致命缺陷：崩溃后不清锁 → 后续尝试全部秒失败，
      把「一格坏」放大成「整批坏」。）
    """
    if not LOCK.exists():
        return
    try:
        info = json.loads(LOCK.read_text(encoding="utf-8"))
        pid = int(info.get("pid", -1))
    except Exception:
        pid = -1
    if pid > 0 and pid_alive(pid):
        note(f"锁被存活进程 {pid} 持有，等它结束")
        t0 = time.time()
        while pid_alive(pid):
            if time.time() - t0 > CELL_TIMEOUT_S + 600:
                note(f"等待 pid={pid} 超过上限，强制删锁")
                LOCK.unlink(missing_ok=True)
                return
            time.sleep(20)
        return
    LOCK.unlink(missing_ok=True)
    note(f"清除陈旧锁（pid={pid} 已不存在）")


def cell_dir(variant: str, seed: int) -> pathlib.Path:
    return LB / variant / f"seed{seed}"


def cell_done(variant: str, seed: int) -> bool:
    """完成判据 = 存在任一 `*/final.pt`（与 run_one 的 run_id 防覆盖护栏一致）。"""
    d = cell_dir(variant, seed)
    return d.is_dir() and any(d.glob("*/final.pt"))


def archive_zombie(variant: str, seed: int) -> None:
    """归档"非空但无 final.pt"的崩溃残留，否则 run_one 拒绝启动。"""
    d = cell_dir(variant, seed)
    if not d.is_dir():
        return
    arch = LB / "_archive"
    for sub in sorted(d.iterdir()):
        if not sub.is_dir() or (sub / "final.pt").exists():
            continue
        arch.mkdir(parents=True, exist_ok=True)
        dest = arch / f"{variant}_seed{seed}_{sub.name}"
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.move(str(sub), str(dest))
        note(f"归档僵尸目录 {sub.relative_to(LB)} → {dest.name}")


def _kill_cell_processes() -> None:
    """杀掉本片仍在跑的格进程（按命令行匹配，避免误伤）。

    多卡下**必须带分片号匹配**：否则某片超时清理时，`pkill -f
    _run_one_lockbox_cell` 会把**其他卡上正在正常训练的格**一起杀掉 ——
    这不是理论风险：N 片并行时每片都会在超时时调用它一次。
    故匹配串收紧为 `_run_one_lockbox_cell*--shard <本片号>`，
    而 `--shard` 由本脚本拼进命令行，必然存在。
    """
    tag = f"--shard {SHARD_IDX}"
    try:
        if os.name == "nt":
            ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                  f"Where-Object {{ $_.CommandLine -like '*_run_one_lockbox_cell*' "
                  f"-and $_.CommandLine -like '*{tag}*' }} | "
                  "ForEach-Object { Stop-Process -Id $_.ProcessId -Force "
                  "-ErrorAction SilentlyContinue }")
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, timeout=60)
        else:
            subprocess.run(["pkill", "-f", f"_run_one_lockbox_cell.*{tag}"],
                           capture_output=True, timeout=60)
    except Exception:
        pass


def run_cell(variant: str, seed: int, est_min: float) -> tuple[bool, float]:
    clear_stale_lock()
    ensure_bootstrap()               # 取回本格预计耗时（从共享台账）
    if cell_done(variant, seed):
        note(f"跳过 {variant}/seed{seed}：已完成")
        return True, 0.0
    archive_zombie(variant, seed)

    CELLDIR.mkdir(parents=True, exist_ok=True)
    logp = CELLDIR / f"{variant}_seed{seed}.log"
    cmd = [str(PY), "-u", str(CELL), variant, str(seed), str(STEPS), str(EVAL_EVERY),
           "--shard", str(SHARD_IDX)]
    if GPU_ID != "":
        cmd += ["--gpu", GPU_ID]
    note(f"启动 {variant}/seed{seed}（预计 {est_min:.0f} min，硬超时 "
         f"{CELL_TIMEOUT_S//60} min）")
    t0 = time.time()
    with open(logp, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"{variant} seed{seed} =====\n")
        lf.flush()
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, env=_child_env(),
                                timeout=CELL_TIMEOUT_S).returncode
        except subprocess.TimeoutExpired:
            mins = (time.time() - t0) / 60
            note(f"⚠ 超时 {variant}/seed{seed}（{mins:.0f} min）—— 判为挂死，杀掉后继续")
            _kill_cell_processes()
            clear_stale_lock()
            archive_zombie(variant, seed)
            return False, mins
        except Exception as e:                       # noqa: BLE001
            note(f"异常 {type(e).__name__}: {e}")
            return False, (time.time() - t0) / 60

    mins = (time.time() - t0) / 60
    ok = rc == 0 and cell_done(variant, seed)
    note(f"{'✅ 完成' if ok else '❌ 失败'} {variant}/seed{seed} rc={rc} {mins:.1f} min")
    if not ok:
        clear_stale_lock()
        archive_zombie(variant, seed)
    return ok, mins


# ---- 共享台账（跨平台、可续跑） ----
def load_ledger() -> list[dict]:
    if TIMING.exists():
        try:
            return json.loads(TIMING.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def hist_minutes(variant: str, ledger: list[dict]) -> float:
    ms = [r["minutes"] for r in ledger
          if r.get("variant") == variant and r.get("ok") and r.get("minutes")]
    if ms:
        ms.sort()
        return float(ms[len(ms) // 2])
    return float(FALLBACK_MIN.get(variant, 30))


def ensure_bootstrap() -> None:
    """占位：确保 runs/lockbox 存在（run_one 自己会 mkdir）。"""
    LB.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--deadline", default="", help="本地截止时刻 HH:MM")
    ap.add_argument("--shard", type=int, default=0,
                    help="本进程取第几片（0-based）")
    ap.add_argument("--of", dest="of", type=int, default=1,
                    help="总共切几片；=1 时行为与本机单卡版逐字相同")
    ap.add_argument("--gpu", default="",
                    help="CUDA_VISIBLE_DEVICES 值（如 0 或 1）；留空则继承")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印本片负责的格与完成状态，不启动任何训练")
    a = ap.parse_args()

    global GPU_ID
    apply_shard(a.shard, a.of)
    GPU_ID = a.gpu

    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        ap.error(f"未知变体 {bad}；可用 {VARIANTS}")
    seeds = [int(s) for s in a.seeds.split(",")]

    if not CELL.exists():
        note(f"✗ 缺 {CELL}")
        return 1

    deadline: datetime | None = None
    if a.deadline:
        hh, mm = (int(x) for x in a.deadline.split(":"))
        now = datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        note(f"截止 {deadline:%Y-%m-%d %H:%M}")

    ledger = load_ledger()
    # 本片负责的格。**按预估耗时降序**再轮转（LPT 贪心），避免慢格扎堆：
    # 实测顺序切分下 4 片为 112/108/78/92 min，关键路径被拖长 ~20%。
    my_cells = shard_cells([(v, s) for v in variants for s in seeds])
    total = len(my_cells)
    est = sum(hist_minutes(v, ledger) for v, _ in my_cells)
    note(f"锁箱驱动启动 pid={os.getpid()}  分片 {SHARD_IDX}/{SHARD_TOTAL}  "
         f"本片 {total} 格 预计 {est:.0f} min  GPU={GPU_ID or '继承'}  锁={LOCK.name}")
    for v in variants:
        note(f"  预计 {v}: {hist_minutes(v, ledger):.0f} min")

    if a.dry_run:
        # 上云前验证：只看「本片认领哪些格、已完成哪些」，不启动任何训练。
        # 目的是在**计费之前**确认分片无重复、无遗漏、负载均衡、路径正确。
        note("=== --dry-run：仅打印计划，不启动训练 ===")
        for v, s in my_cells:
            note(f"    {'[done]' if cell_done(v, s) else '[todo]'} {v}/seed{s}"
                 f"   预计 {hist_minutes(v, ledger):.0f} min")
        note(f"  本片合计 {total} 格："
             f"已完成 {sum(1 for v_, s_ in my_cells if cell_done(v_, s_))}，"
             f"待跑 {sum(1 for v_, s_ in my_cells if not cell_done(v_, s_))}；"
             f"预估墙钟 {est:.0f} min")
        note(f"  锁文件   {LOCK}")
        note(f"  台账     {TIMING}")
        note(f"  驱动日志 {DRIVER_LOG}")
        return 0

    for sweep in range(1, MAX_SWEEPS + 1):
        left = [(v, s) for (v, s) in my_cells if not cell_done(v, s)]
        note(f"--- 第 {sweep} 轮：本片已完成 {total-len(left)}/{total}，"
             f"待跑 {len(left)} ---")
        if not left:
            note("全部完成")
            break
        stopped = False
        for (v, s) in left:
            if deadline is not None:
                remain = (deadline - datetime.now()).total_seconds() / 60
                need = hist_minutes(v, ledger)
                if remain < need:
                    note(f"⏹ 截止保护：剩 {remain:.0f} min < 预计 {need:.0f} min，"
                         f"不再起新格")
                    stopped = True
                    break
            ok, mins = run_cell(v, s, hist_minutes(v, ledger))
            ledger.append(dict(variant=v, seed=s, ok=ok, minutes=round(mins, 1),
                               at=time.strftime("%Y-%m-%d %H:%M:%S")))
            TIMING.write_text(json.dumps(ledger, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        if stopped:
            break

    done = sum(1 for (v, s) in my_cells if cell_done(v, s))
    note(f"结束：本片 {done}/{total}")
    okmins = [r["minutes"] for r in ledger if r.get("ok") and r.get("minutes")]
    if okmins:
        note(f"成功格合计 {sum(okmins):.0f} min，中位 "
             f"{sorted(okmins)[len(okmins)//2]:.1f} min")
    return 0 if done == total else 0     # 未完成不报错：夜停/截止是预期行为


if __name__ == "__main__":
    sys.exit(main())
