#!/usr/bin/env bash
# 接在 TIER4 骨干补齐之后：等 RunLock 释放，跑锁箱全源重训。
#
# 依赖链（v14 阶段 ③）：
#   TIER4 骨干补齐（_chain_tier4.sh） → 本脚本（全源重训）
#   → 锁箱单次评估（OASBUD / BrEaST，须另行启动，见 make_figures / evaluate_lockbox）
#
# 幂等：retrain_lockbox 会拒绝覆盖非空运行目录，故重入安全
# （已完成的配置会因目录已存在而报错跳过，不产生错误结果）。
#
# 等待上限 300 min（5 h）：TIER4 三个骨干 18 次运行，按 17 min/次 ≈ 5.1 h。
# 超时则中止，避免与僵尸进程抢 GPU。
cd "C:/Users/15315/Desktop/paper1st/meliorate/project" || exit 1
LOCK=runs/dev_screen/.running.json
MAX_WAIT=18000         # 300 min
t0=$(date +%s)
echo "[lockbox-chain] 等待 TIER4 结束 $(date '+%H:%M:%S')"
while [ -f "$LOCK" ]; do
  now=$(date +%s)
  if [ $((now - t0)) -gt $MAX_WAIT ]; then
    echo "[lockbox-chain] ⚠ 等待超 300 min，锁仍在。中止以避免抢 GPU。"
    exit 2
  fi
  sleep 120
done
echo "[lockbox-chain] 锁已释放 $(date '+%H:%M:%S')，启动全源重训"
PYTHONIOENCODING=utf-8 PYTHONPATH=src ../.venv/Scripts/python.exe -u \
  scripts/retrain_lockbox.py --variants all --seeds 42,123,2026 --steps 3000
echo "[lockbox-chain] 全源重训结束 $(date '+%H:%M:%S') exit=$?"
