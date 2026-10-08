#!/usr/bin/env bash
# 等锁箱评估 ⑫ 收尾 → 重跑 E6 Grad-CAM（真类 CAM 口径）。
#
# 为什么单独等：E6 要用 GPU（每例一次前向+反传）。与 ⑫ 并发会争卡，
# 而 ⑫ 是不可逆的单次评估，不值得为省几分钟去打扰它。
# ⑬/⑭ 是纯 CPU（概率矩阵上的 AUC/CI），故 E6 与它们并发无妨。
#
# 为什么必须重跑：`GradCAM.__call__` 的 `target` 由恶性类（=1）改为**真类**。
# 旧口径下良性病例 ReLU 后整张全零 → mask_energy_ratio 恒为 nan
# （实测 efficientnet_v2_s 良性列全缺）。两个口径**不可混用**，
# 故 7 个变体全部按新口径重算，产物里也写明口径。
#
# 用法（本地）
#   setsid nohup bash env/run_e6_after_eval.sh > runs/_logs/e6_after_eval.log 2>&1 < /dev/null &
set -uo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_ROOT"

PY="../.venv/Scripts/python.exe"
LOG_DIR="runs/_logs"
EVAL_DIR="runs/lockbox_eval"
mkdir -p "$LOG_DIR"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

say "=== 等 ⑫ 收尾（判据：$EVAL_DIR/summary_full_card.csv）==="
n=0; stale=0
while [ ! -f "$EVAL_DIR/summary_full_card.csv" ]; do
  now=$(date +%s)
  mtime=$(stat -c %Y "$LOG_DIR/eval_full_card.log" 2>/dev/null || echo "$now")
  if [ $((now - mtime)) -gt 600 ]; then
    stale=$((stale + 1))
    if [ "$stale" -ge 2 ]; then
      say "⚠ 日志 $(((now - mtime) / 60)) 分钟未更新且无 summary —— 停止。"
      exit 1
    fi
  else
    stale=0
  fi
  n=$((n + 1))
  [ $((n % 30)) -eq 0 ] && say "  等 $((n / 6)) 分钟…… cells=$(grep -c 'AUC=' $LOG_DIR/eval_full_card.log 2>/dev/null || echo 0)/147"
  sleep 10
done
say "✓ ⑫ 收尾（等了约 $((n / 6)) 分钟）"

# 锁箱布局：7 个变体都在 runs/lockbox/<variant>/seed<N>/，且**无 direction 层**
export BREASTDG_CKPT_ROOT="$APP_ROOT/runs/lockbox"

say ""
say "=== E6 重跑（真类 CAM；7 变体 × seed42 × final × busi）==="
rc=0
for v in resnet50_cbam swin_t densenet121_se efficientnet_v2_s fdnet_gate_dual deepall mixstyle; do
  say "--- $v ---"
  if PYTHONIOENCODING=utf-8 "$PY" -u scripts/gradcam_interpretability.py \
       --variant "$v" --direction busbra --seed 42 --which final \
       --cohort busi --max-per-class 20 2>&1 | tail -4; then
    say "  ✓ $v"
  else
    say "  ✗ $v 失败"
    rc=1
  fi
done

say ""
say "=== 自检：良性列是否仍有 nan ==="
PYTHONIOENCODING=utf-8 "$PY" -c "
import json, glob
bad = []
for f in sorted(glob.glob('runs/diag/gradcam/gradcam_*_busi_final.json')):
    d = json.load(open(f, encoding='utf-8'))
    if d.get('cam_target') != 'true_class':
        bad.append((f, '旧口径', d.get('cam_target'))); continue
    if d.get('mean_ratio_benign') is None or d['mean_ratio_benign'] != d['mean_ratio_benign']:
        bad.append((f, '良性仍缺', d.get('mean_ratio_benign')))
print(f'  检查 {len(glob.glob(\"runs/diag/gradcam/gradcam_*_busi_final.json\"))} 个产物')
if bad:
    print('  ✗ 仍有问题：')
    for b in bad: print('    ', b)
else:
    print('  ✓ 全部为真类口径且良性列有效')
"
say "=== E6 重跑结束（rc=$rc）==="
exit $rc
