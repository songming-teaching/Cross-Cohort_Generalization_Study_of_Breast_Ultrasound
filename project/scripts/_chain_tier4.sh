#!/usr/bin/env bash
# 等当前 TIER4 训练释放 RunLock 后，接着跑剩余的骨干臂。
#
# 为什么需要：GPU 是串行的（8 GB 卡，一次只能跑一个训练），而
# swin_t / densenet121_se / efficientnet_v2_s 三个骨干各 6 次运行、
# 合计约 5 小时。串接成一条链，避免人工盯着逐个启动。
#
# 幂等：dev_screen 会跳过 results.csv 里已完成的 (变体,方向,种子,步数)，
# 故本脚本可安全重入；已跑完的 swin_t 会被自动跳过。
#
# 带超时：若当前进程被硬杀（OOM/断电），锁文件不会释放，
# 无超时的 while 会永远等下去 —— 故最多等 120 分钟。
cd "C:/Users/15315/Desktop/paper1st/meliorate/project" || exit 1
LOCK=runs/dev_screen/.running.json
MAX_WAIT=7200          # 120 min
t0=$(date +%s)
echo "[chain] 开始等待 RunLock 释放 $(date '+%H:%M:%S')"
while [ -f "$LOCK" ]; do
  now=$(date +%s)
  if [ $((now - t0)) -gt $MAX_WAIT ]; then
    echo "[chain] ⚠ 等待超 120 min，锁仍在（当前训练可能被硬杀）。"
    echo "[chain] ⚠ 中止，避免与僵尸进程抢 GPU。"
    exit 2
  fi
  sleep 60
done
echo "[chain] 锁已释放 $(date '+%H:%M:%S')，启动 TIER4 剩余骨干"
PYTHONIOENCODING=utf-8 PYTHONPATH=src ../.venv/Scripts/python.exe -u \
  -m breastdg.train.dev_screen --queue tier4 \
  --directions busbra,busi --seeds 42,123,2026 --steps 3000 --eval-every 500
echo "[chain] TIER4 结束 $(date '+%H:%M:%S') exit=$?"
