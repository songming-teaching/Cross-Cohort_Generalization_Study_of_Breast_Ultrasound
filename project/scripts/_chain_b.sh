#!/usr/bin/env bash
# 等 tier3 结束后跑 E8 dev 侧批次。
# **带超时**：若 tier3 被硬杀（OOM/断电），锁文件不会释放，
# 无超时的 while 会永远等下去 —— 故最多等 90 分钟，之后强制继续并告警。
cd "C:/Users/15315/Desktop/paper1st/meliorate/project"
LOCK=runs/dev_screen/.running.json
MAX_WAIT=5400          # 90 min
t0=$(date +%s)
echo "[chain] 开始等待 tier3 $(date '+%H:%M:%S')"
while [ -f "$LOCK" ]; do
  now=$(date +%s)
  if [ $((now - t0)) -gt $MAX_WAIT ]; then
    echo "[chain] ⚠ 等待超 90 min，锁仍在（tier3 可能被硬杀）。"
    echo "[chain] ⚠ 继续跑 E8，但请人工确认 tier3 产物完整性。"
    break
  fi
  sleep 60
done
echo "[chain] tier3 已结束（或超时） $(date '+%H:%M:%S')，启动 E8 dev 侧批次"
PYTHONIOENCODING=utf-8 PYTHONPATH=src ../.venv/Scripts/python.exe scripts/run_e8_batch.py
echo "[chain] E8 批次结束 $(date '+%H:%M:%S') exit=$?"
