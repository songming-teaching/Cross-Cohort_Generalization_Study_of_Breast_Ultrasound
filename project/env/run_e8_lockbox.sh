#!/usr/bin/env bash
# E8 锁箱臂：在**锁箱 49 个检查点**上跑 OASBUD 双臂 + BrEaST 的谱量。
#
# 为什么现在才跑
# --------------
# `run_e8_batch.py` 的 docstring 明确写着：只跑 dev 侧队列，
# **OASBUD / BrEaST 的谱量一律留锁箱后由 E8 统一出数** ——
# 提前看一眼就等于用锁箱信息污染后续分析设计。
# 锁箱评估（⑫）已完成，故现在正是"统一出数"的时机。
#
# 为什么这些路径此前从未跑过（本轮修了 3 个缺陷）
# -----------------------------------------------
#   1. `load_oasbud` 直接用 manifest 的 `path`（Windows 反斜杠）→ 跨平台读图失败
#   2. `load_oasbud` 根目录硬编码 256 那份 → fdnet(512) 会拿到错尺寸的图
#   3. 预处理 `PreprocConfig` **未指定 tone_norm** → 默认 True →
#      OASBUD 成品图被**二次归一化**（实测内容区 MAE ≈ 9–21）
#   现三处都已修（1/2 见 run_e8.load_oasbud，3 见 cohorts.preproc_for_cfg）。
#
# 本地可跑：单次 ~21–25 s，49 格 × 3 队列 ≈ 1 小时。**不需要云算力。**
#
# 用法
#   setsid nohup bash env/run_e8_lockbox.sh > runs/_logs/e8_lockbox.log 2>&1 < /dev/null &
set -uo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"

PY="../.venv/Scripts/python.exe"
OUT_DIR="runs/diag/e8_lockbox"
LOG_DIR="runs/_logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

say "=== E8 锁箱臂启动 ==="
say "输出：$OUT_DIR（每格一个 JSON）"

# 检查点清单：直接问 find_ckpts（**唯一出处**，避免自己 glob 出歧义）
#
# ⚠ `tr -d '\r'` **必需**（2026-10-08 实测踩到）：
#   Windows 上 Python 的 stdout 处于文本模式，即使 PYTHONIOENCODING=utf-8，
#   `print()` 仍把 `\n` 翻成 `\r\n`（实测 `print('a b') | od -c` → `a  b \r \n`）。
#   而 `mapfile -t` 只剥行尾 `\n`，`\r` 会留在最后一个字段里
#   → 文件名变成 `e8_deepall_seed123\r.json` → `OSError: [Errno 22] Invalid argument`。
#   这是"跨平台文本模式"类问题，不是编码问题。
mapfile -t CELLS < <(PYTHONIOENCODING=utf-8 "$PY" -c "
import sys; sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
from evaluate_lockbox import find_ckpts
for v,s,_ in find_ckpts():
    print(f'{v} {s}')
" | tr -d '\r')
n_total=${#CELLS[@]}
# 同一坑的第二道防线：逐个字段清一次（防别的来源再引入 \r）
CELLS=("${CELLS[@]//$'\r'/}")
say "共 $n_total 格 × 3 队列"

i=0; ok=0; fail=0
for cell in "${CELLS[@]}"; do
  set -- $cell
  v="$1"; s="$2"
  i=$((i + 1))
  out="$OUT_DIR/e8_${v}_seed${s}.json"
  if [ -f "$out" ]; then
    say "[$i/$n_total] $v/seed$s 已有，跳过"
    ok=$((ok + 1)); continue
  fi
  say "[$i/$n_total] $v/seed$s"
  if PYTHONIOENCODING=utf-8 "$PY" -u scripts/run_e8.py \
       --variant "$v" --seed "$s" --which final \
       --cohorts oasbud_corrected,oasbud_uncorrected,breast \
       --runs-root runs/lockbox --out "$out" 2>&1 | tail -6; then
    ok=$((ok + 1))
  else
    say "  ✗ FAIL $v/seed$s"
    fail=$((fail + 1))
  fi
done

say ""
say "==============================================================="
say "E8 锁箱臂结束：成功 $ok / 失败 $fail / 共 $n_total"
say "产物：$OUT_DIR/"
say "==============================================================="
[ "$fail" -eq 0 ] || exit 1
