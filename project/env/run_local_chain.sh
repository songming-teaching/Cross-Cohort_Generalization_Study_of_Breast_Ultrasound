#!/usr/bin/env bash
# 本地接力（不计费）：等锁箱评估 ⑫ 收尾 → 自动跑 ⑬ outer-eval ×3 → ⑭ e3 极性分析。
#
# 为什么需要它
# ------------
# ⑫ 是单次不可逆的长跑（~45 min）；⑬⑭ 依赖它的产物。若由人盯着再逐步启动，
# 一旦中途断网（本机断网不会杀死已启动的本地进程，但**无法再启动新命令**），
# ⑬⑭ 就会卡住。故把「等 → 跑 → 跑」串成一个**已启动**的脚本，
# 之后即使没有人（也没有网络），它仍会把流程走完。
#
# 判据（为什么看 summary_full_card.csv）
# -------------------------------------
# `evaluate_lockbox.write_summary` 在**三个臂全部跑完**之后才执行，
# 故 `summary_full_card.csv` 存在 = 三个正式卡都已落盘。
# 只查 evalcard_*.json 不够：臂 2/3 还没写时就可能已满足部分条件。
#
# 幂等：⑬ 若已有产物会自己判断；⑭ 覆盖写。重复执行本脚本无副作用。
#
# 用法（本地）
# ------------
#   setsid nohup bash env/run_local_chain.sh > runs/_logs/local_chain.log 2>&1 < /dev/null &
set -uo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"

PY="../.venv/Scripts/python.exe"
EVAL_DIR="runs/lockbox_eval"
LOG_DIR="runs/_logs"
mkdir -p "$LOG_DIR"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

say "=== 本地接力启动 ==="
say "等 ⑫ 锁箱评估收尾（判据：$EVAL_DIR/summary_full_card.csv）"

n=0; stale=0
while [ ! -f "$EVAL_DIR/summary_full_card.csv" ]; do
  # ⚠ 存活判据用**日志 mtime**，不用 pgrep（2026-10-08 实测教训）：
  #   本仓在 Git Bash（MSYS）下跑，`pgrep -f evaluate_lockbox.py` **看不见**
  #   Windows 原生 python 进程 → 判据恒为"进程已消失"，
  #   于是接力在 30 秒内就误判失败退出（实测发生：12:38:40 退出，
  #   而评估当时正跑到 77/147）。改用"日志多久没更新"：
  #   每格都会追加一行 AUC，故 10 分钟无更新才是真的死了。
  now=$(date +%s)
  mtime=$(stat -c %Y "$LOG_DIR/eval_full_card.log" 2>/dev/null || echo "$now")
  if [ $((now - mtime)) -gt 600 ]; then
    stale=$((stale + 1))
    if [ "$stale" -ge 2 ]; then
      say "⚠ 日志已 $(((now - mtime) / 60)) 分钟未更新且无 summary —— 评估疑似失败，停止接力。"
      say "  请查 $LOG_DIR/eval_full_card.log"
      exit 1
    fi
  else
    stale=0
  fi
  n=$((n + 1))
  [ $((n % 30)) -eq 0 ] && say "  仍在跑（已等 $((n / 6)) 分钟）…… 已出 AUC 行数 $(grep -c 'AUC=' $LOG_DIR/eval_full_card.log 2>/dev/null || echo 0)/147"
  sleep 10
done
say "✓ ⑫ 收尾（等了约 $((n / 6)) 分钟）"

for t in oasbud_corrected oasbud_uncorrected breast; do
  if [ -f "$EVAL_DIR/evalcard_$t.json" ]; then
    say "  ✓ evalcard_$t.json"
  else
    say "  ✗ 缺 evalcard_$t.json —— 评估不完整，停止接力"
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# ⑬ 外部集：**只应用**冻结权重（不重新拟合）
# ---------------------------------------------------------------------------
say ""
say "=== ⑬ outer-eval ×3（只应用源域冻结权重，不拟合）==="
rc=0
for t in oasbud_corrected oasbud_uncorrected breast; do
  say "--- tag=$t ---"
  if PYTHONIOENCODING=utf-8 "$PY" -u scripts/ensemble_and_calibrate.py \
       --outer-eval --tag "$t" 2>&1 | tee "$LOG_DIR/outer_eval_$t.log" | tail -8; then
    say "  ✓ $t"
  else
    say "  ✗ $t 失败（见 $LOG_DIR/outer_eval_$t.log）"
    rc=1
  fi
done

# ---------------------------------------------------------------------------
# ⑭ E3 极性分析（全文核心实验）
# ---------------------------------------------------------------------------
say ""
say "=== ⑭ E3 极性分析 ==="
if PYTHONIOENCODING=utf-8 "$PY" -u scripts/e3_polarity_analysis.py \
     --eval-dir "$EVAL_DIR" 2>&1 | tee "$LOG_DIR/e3_polarity.log" | tail -25; then
  say "  ✓ E3"
else
  say "  ✗ E3 失败（见 $LOG_DIR/e3_polarity.log）"
  rc=1
fi

say ""
say "==============================================================="
say "本地接力结束（rc=$rc）。产物："
say "  $EVAL_DIR/evalcard_*.json + summary_full_card.csv   ← ⑫"
say "  runs/ensemble/ensemble_*.json                       ← ⑬"
say "  runs/diag/e3/                                       ← ⑭"
say "==============================================================="
exit $rc
