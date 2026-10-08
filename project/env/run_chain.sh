#!/usr/bin/env bash
# 无人值守接力：等当前 35 格跑完 → 自动补跑公平基线（只跑缺的格）。
#
# 为什么需要它
# ------------
# `lockbox-exhaustive` = 5 骨干 × 7 种子 = 35 格（主配置）。
# `lockbox-full`       = 7 变体 × 7 种子 = 49 格（多加 deepall / mixstyle，
#                        回应"拒稿意见 5：只和两个基线比、且不说基线怎么训"）。
#
# 驱动是**幂等**的（`cell_done` 判据 = 存在任一 `*/final.pt`），
# 故 `lockbox-full` 会自动跳过已完成的 35 格，**只跑新增的 14 格**。
#
# 但两者**不能同时起**：分片锁是 `runs/dev_screen/.running.lockbox_shard{N}.json`，
# 同名锁 → 第二个 run_sharded 会立刻挂在"检测到另一实例正在运行"。
# 所以必须**串行**：本脚本先等第一个跑完，再起第二个。
#
# 用法（云上）
# ------------
#     bash env/run_chain.sh            # 前台（可配合 setsid nohup 后台化）
#     setsid nohup bash env/run_chain.sh > /root/autodl-tmp/chain.log 2>&1 < /dev/null &
#
# 产物
# ----
#   /root/autodl-tmp/chain.log          本脚本的总日志
#   runs/_logs/lockbox_driver_shard*.log 各分片驱动日志
#   runs/_logs/chain_stage1.json etc.    各阶段收尾时间戳（便于事后核对）
set -uo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"

LOG_DIR="$APP_ROOT/runs/_logs"
mkdir -p "$LOG_DIR"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# ---------------------------------------------------------------------------
# 等待：直到没有 `_lockbox_driver.py` 进程（= 前一个 stage 结束）
# ---------------------------------------------------------------------------
wait_for_drivers() {
  local label="$1" n=0
  say "等待：$label 收尾（轮询 进程 + 锁文件）..."
  # ⚠ 必须**同时**看进程和锁文件（2026-10-08 修）：
  #   只查 `pgrep _lockbox_driver.py` 有**竞态** —— 驱动在两格之间有
  #   几秒的空隙（收尾→起下一格），若恰好在此刻进入本循环，pgrep 为空
  #   → 本脚本会以为"跑完了"而立刻起 lockbox-full，
  #   而 lockbox-full 会撞上**仍然存在的**分片锁 →
  #   "检测到另一实例正在运行" → 阶段 2 直接失败（而阶段 1 还在跑）。
  #   锁文件只在驱动**退出**时删除，故它是可靠的判据。
  while pgrep -f "_lockbox_driver.py" >/dev/null 2>&1      || ls runs/dev_screen/.running.lockbox_shard*.json >/dev/null 2>&1; do
    n=$((n + 1))
    # 每 30 分钟打一次心跳，便于 tail 时知道它没死
    if [ $((n % 30)) -eq 0 ]; then
      local done_n
      done_n=$(find runs -name final.pt 2>/dev/null | wc -l)
      say "  仍在跑（已等 $n min）…… 已完成 final.pt  $done_n 个"
    fi
    sleep 60
  done
  say "✓ $label 结束（累计等待 $n min）"
}

# ---------------------------------------------------------------------------
# 阶段 1：等当前 35 格结束
# ---------------------------------------------------------------------------
say "=== 接力脚本启动 ==="
say "阶段 1：等待 lockbox-exhaustive（35 格）完成"
wait_for_drivers "lockbox-exhaustive"

n1=$(find runs/lockbox -name final.pt 2>/dev/null | wc -l)
say "  已完成 final.pt：$n1 个"
echo "{\"stage\":\"exhaustive\",\"at\":\"$(date -Is)\",\"final_pt\":$n1}" \
  > "$LOG_DIR/chain_stage1.json"

# 单独核对：预期 35 个
if [ "$n1" -lt 35 ]; then
  say "⚠ 只完成 $n1 / 35 —— 可能有格子失败。**仍继续补基线**，"
  say "   但请回看 runs/_logs/lockbox_driver_shard*.log 找 FAIL 行。"
fi

# ---------------------------------------------------------------------------
# 阶段 2：补公平基线（deepall / mixstyle）—— 只跑缺的 14 格
# ---------------------------------------------------------------------------
say ""
say "阶段 2：lockbox-full（幂等，只会跑缺的 deepall / mixstyle 格）"
bash env/run_cloud.sh lockbox-full

n2=$(find runs/lockbox -name final.pt 2>/dev/null | wc -l)
say "✓ 阶段 2 结束，final.pt 合计 $n2 个"
echo "{\"stage\":\"full\",\"at\":\"$(date -Is)\",\"final_pt\":$n2}" \
  > "$LOG_DIR/chain_stage2.json"

# ---------------------------------------------------------------------------
# 收尾汇总
# ---------------------------------------------------------------------------
say ""
say "==============================================================="
say "接力完成。各变体已完成格数："
for v in resnet50_cbam swin_t densenet121_se efficientnet_v2_s fdnet_gate_dual deepall mixstyle; do
  c=$(find "runs/lockbox/$v" -name final.pt 2>/dev/null | wc -l)
  printf '  %-20s %2d 格\n' "$v" "$c"
done
say ""
say "预期：主配置 5 变体各 7 格 = 35；基线 2 变体各 7 格 = 14；合计 49。"
say "下一步（本地）：bash env/pull_runs.sh <host> <port>"
say "==============================================================="
