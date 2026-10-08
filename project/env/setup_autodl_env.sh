#!/usr/bin/env bash
# AutoDL 环境核对（**不安装、不改动任何东西**）
#
# ===========================================================================
# ⚠⚠ 2026-10-07 重要停用说明 —— 本脚本原名 setup_autodl_env.sh，
#     作用是「覆盖安装 torch 2.6.0」。**该用途已废弃，理由见下。**
#
#     事实：AutoDL 的 PyTorch 镜像列表里**没有 2.6.0**（实测列表：
#       2.8.0 / 2.12.1 / 2.5.1 / 2.3.0 / 2.1.2 / 2.1.0 / 2.0.0 /
#       1.11.0 / 1.10.0 / 1.9.0 / 1.8.1 / 1.7.0 / 1.6.0 / 1.5.1 / 1.1.0）
#     云端可用的是 **torch 2.5.1 + torchvision 0.20.1**，
#     而这是**官方配对**（2.5.1↔0.20.1 / 2.6.0↔0.21.0，**不可交叉**）。
#
#     本机 dev 产物是 2.6.0 跑的，故存在环境差异。**已知并接受**：
#       · 锁箱这批 35 格全部在**同一云端环境**跑完 —— 批次内各变体之间的
#         比较（论文里用到的主要比较）**不受影响**；
#       ·「云端锁箱 vs 本机 dev 88 格」的跨批比较带一层环境混杂，入表时
#         须标注；若被问，答复 = 本批为独立重训批次，非同一环境复现；
#       · 已核本仓**零处**使用 2.6 专有 API，2.5.1 下无功能缺失。
#
#     **所以：不要再试图装 2.6.0。** 那会白费时间，且 pip 覆盖可能把
#     当前**可用的**环境弄坏 —— 而当前环境是对的。
# ===========================================================================
#
# 本脚本现在只做三件事（全部只读，幂等，随时可跑）：
#   ① 打印当前环境实际版本
#   ② 跑 verify_env（对照 env/verify_env.py 的目标基准）
#   ③ 核对路径假设（PROJECT_ROOT 在数据盘、splits/ 在位）
#
# 用法（云上）：
#   cd /root/autodl-tmp/project
#   bash env/setup_autodl_env.sh
#
# 退出码：0 = 环境可用；1 = 有问题（**不要开始训练**）
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="${APP_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
cd "$APP_ROOT"

# ---------------------------------------------------------------------------
# Python 定位（2026-10-07 修）
#   `ssh autodl '命令'` 是**非交互式**会话，**不读 .bashrc**，
#   故 conda 未激活、`python` 不在 PATH —— 报 "python: command not found"。
#   而 `ssh autodl` 交互登录后手输命令是**能**找到的（.bashrc 生效）。
#   同一个脚本两种会话行为不同，必须显式探测。
#   顺序：显式覆盖 → PATH 上的 python/python3 → AutoDL 常见 conda 位置。
# ---------------------------------------------------------------------------
_find_python() {
  local c
  for c in "${PYTHON_BIN:-}" python python3            /root/miniconda3/bin/python            /root/miniconda3/envs/*/bin/python            /opt/conda/bin/python "$HOME/miniconda3/bin/python"; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then command -v "$c"; return 0; fi
  done
  return 1
}
PY="$(_find_python)" || {
  echo "✗ 找不到 python（PATH 上没有，AutoDL 常见位置也没有）" >&2
  echo "  诊断：ssh autodl 'echo \$PATH; ls /root/miniconda3/bin/python*'" >&2
  exit 1
}
export PATH="$(dirname "$PY"):$PATH"

# ---------------------------------------------------------------------------
# 数据根探测（2026-10-07 修）
#   本脚本原先**不设 BREASTDG_DATA**，于是 config.py 退回它的硬编码默认值
#   （`C:/Users/15315/Desktop/paper1st/data` —— **本机 Windows 路径**），
#   导致核对时三个数据队列全部报 ✗ —— 那是**误报**，不是数据没传。
#   run_cloud.sh 会自己设这个变量，故不影响训练；但核对脚本必须同样探测，
#   否则"核对"本身在骗人。
# ---------------------------------------------------------------------------
if [ -z "${BREASTDG_DATA:-}" ]; then
  for _cand in /root/autodl-tmp/data "${APP_ROOT}/data" /data; do
    if [ -d "$_cand" ]; then BREASTDG_DATA="$_cand"; break; fi
  done
fi
if [ -n "${BREASTDG_DATA:-}" ]; then
  export BREASTDG_DATA
  echo "数据根（探测）: $BREASTDG_DATA"
fi

echo "==============================================================="
echo "AutoDL 环境核对（只读，不安装任何东西）"
echo "  代码根 $APP_ROOT"
echo "==============================================================="

echo ""
echo "--- ① 当前实际版本 ---"
"$PY" - <<'PY'
import sys
print(f"  python      {sys.version.split()[0]}")
try:
    import torch, torchvision
    print(f"  torch       {torch.__version__}")
    print(f"  torchvision {torchvision.__version__}")
    print(f"  cuda        {torch.version.cuda}")
    print(f"  cudnn       {torch.backends.cudnn.version()}")
    print(f"  cuda可用    {torch.cuda.is_available()}")
    print(f"  设备数      {torch.cuda.device_count()}")
except Exception as e:
    print(f"  ✗ 导入失败：{type(e).__name__}: {e}")
    sys.exit(1)
PY

echo ""
echo "--- ② verify_env（对照目标基准）---"
"$PY" env/verify_env.py

echo ""
echo "--- ③ 路径假设核对 ---"
"$PY" - <<'PY'
import sys, pathlib
sys.path.insert(0, 'src')
from breastdg import config
print(f"  PROJECT_ROOT {config.PROJECT_ROOT}")
print(f"  DATA_ROOT    {config.DATA_ROOT}")
print(f"  RUNS_DIR     {config.RUNS_DIR}")

ok = True
# 产物必须在数据盘：否则实例一释放就没了（锁箱检查点不可重来）
if not str(config.RUNS_DIR).startswith('/root/autodl-tmp/'):
    print(f"  ✗ RUNS_DIR 不在数据盘 —— 释放实例会丢产物")
    print(f"    处置：把代码放在 /root/autodl-tmp 下（PROJECT_ROOT 就跟着走）")
    ok = False
else:
    print("  ✓ RUNS_DIR 在数据盘上")

# splits 随代码走
sp = pathlib.Path(config.PROJECT_ROOT, 'splits')
if not sp.is_dir():
    print(f"  ✗ 缺 {sp} —— LODO 划分随代码走，训练必需")
    ok = False
else:
    print("  ✓ splits/ 随代码在位")

# 数据三个源队列
for d in ('busi', 'BUSBRA', 'BUSI-UCLM'):
    p = pathlib.Path(config.DATA_ROOT, d)
    print(f"  {'✓' if p.exists() else '✗'} 数据 {d}")
    ok = ok and p.exists()
sys.exit(0 if ok else 1)
PY

echo ""
echo "--- ④ 预训练权重是否已固化 ---"
"$PY" - <<'PY'
import os, pathlib
d = pathlib.Path(os.environ.get('TORCH_HOME', '/root/.cache/torch')) / 'hub' / 'checkpoints'
fs = sorted(p.name for p in d.glob('*.pth')) if d.is_dir() else []
print(f"  已固化 {len(fs)} 个权重文件")
for x in fs:
    print(f"    {x}")
if len(fs) < 5:
    print("  ! 不足 5 个 —— 训练首次会**在线拉权重**，若外网不通会静默卡住")
    print("    处置：source /etc/network_turbo && python -c \"import torchvision.models as M;\\")
    print("      [f(weights=w) for f,w in [(M.resnet50,M.ResNet50_Weights.DEFAULT),")
    print("       (M.resnet18,M.ResNet18_Weights.DEFAULT),(M.densenet121,M.DenseNet121_Weights.DEFAULT),")
    print("       (M.efficientnet_v2_s,M.EfficientNet_V2_S_Weights.DEFAULT),(M.swin_t,M.Swin_T_Weights.DEFAULT)]]\"")
PY

echo ""
echo "==============================================================="
echo "下一步："
echo "  bash env/run_cloud.sh smoke                              # 3 min，第一次必跑"
echo "  DRY_RUN=1 bash env/run_cloud.sh lockbox-exhaustive       # 看分片计划"
echo "  nohup bash env/run_cloud.sh lockbox-exhaustive > /root/autodl-tmp/train.log 2>&1 &"
echo "==============================================================="
