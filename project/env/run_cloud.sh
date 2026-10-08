#!/usr/bin/env bash
# 云端启动入口 —— 与本机 Windows 版驱动等价，但纯 POSIX。
#
# 本机版（scripts/_tier4_driver.py 等）用了 Windows 专有 API：
#   DETACHED_PROCESS / creationflags / tasklist / PowerShell。
# 那些在 Linux 上全部不可用，故这里重写；**科学代码（src/）一行不改** ——
# 已确认 src/breastdg/** 无任何 Windows 专有调用。
#
# 幂等性：与 dev_screen 相同，以 runs/dev_screen/results.csv 为准跳过已完成格，
# 故可安全重入（云实例被抢占后重跑即可续上）。
#
# ⚠ 边界（与本地一致，不可放宽）：
#   - 只跑公开队列；私有队列不得出现在挂载路径、日志或环境变量中（F9）；
#   - **不跑锁箱评估**（OASBUD / BrEaST）—— 那是不可逆的一次性锁箱量，
#     须经作者确认后单次执行，本脚本刻意不包含。

set -euo pipefail

# ---------------------------------------------------------------------------
# 运行环境自适应（2026-10-07 修）
#
# 原实现把路径**硬编码**为 Docker 布局（`/app`、`/data`）—— 那是按
# `env/Dockerfile` 写的假设。但 AutoDL 的**容器实例内不支持 docker**
# （官方文档明示："如需使用 Docker 请联系客服租用裸金属服务器"），
# 故实际部署走**平台原生实例 + 保存镜像**：代码在 `/root/autodl-tmp/project`，
# 数据在 `/root/autodl-tmp/data`。硬编码路径到了那头**第一行就死**。
#
# 现改为自适应：一切相对**本脚本自身位置**推导，环境变量仅作覆盖。
#   APP_ROOT  = 本脚本所在目录的上一级（即 project/）
#   这样 Docker 布局（/app/env/run_cloud.sh → /app）与
#   AutoDL 布局（/root/autodl-tmp/project/env/run_cloud.sh → /root/autodl-tmp/project）**同一份代码**都能跑。
# ---------------------------------------------------------------------------
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="${APP_ROOT:-$(cd "${_SCRIPT_DIR}/.." && pwd)}"
export APP_ROOT

# 数据根：优先环境变量 → 其次 AutoDL 数据盘 → 最后 Docker 的 /data
if [ -z "${BREASTDG_DATA:-}" ]; then
  for _cand in /root/autodl-tmp/data "${APP_ROOT}/data" /data; do
    if [ -d "$_cand" ]; then BREASTDG_DATA="$_cand"; break; fi
  done
fi
: "${BREASTDG_DATA:=/root/autodl-tmp/data}"
export BREASTDG_DATA

# ⚠ **不设 BREASTDG_RUNS**（2026-10-07 修正）
#   原实现把它设为 `${数据盘}/runs` —— 那是**多余且有害**的：
#     · 多余：`config.RUNS_DIR` 默认 = `PROJECT_ROOT/runs`，而云上
#       `PROJECT_ROOT` = `/root/autodl-tmp/project`（**已在数据盘上**），
#       故默认值本就落在数据盘，不需要覆盖；
#     · 有害：它会让 `RUNS_DIR` **偏离 `PROJECT_ROOT`**，而代码里所有
#       `config.PROJECT_ROOT / "runs" / ...` 的**硬拼写法**（例如
#       `evaluate_lockbox.LOCKBOX`、`_lockbox_driver` 的产物路径）
#       不会跟着变 —— 于是"写"与"找"用两个不同的根，
#       产物写在一个地方、脚本在另一个地方找。这类错**静默**，
#       且要到评估阶段才炸（那时检查点已花几小时训完）。
#   落地依据（check5 实测）：用户在 AutoDL 上跑的正是
#       `~/autodl-tmp/project` → PROJECT_ROOT 落在数据盘 ✓
#       且 `verify_env` 全绿、`splits/` 就在 project/ 下 ✓
#   保留环境变量作**逃生口**：若哪天代码放在系统盘，只需
#       `export BREASTDG_RUNS=/root/autodl-tmp/runs`
#   （但那就要连同"硬拼写路径"一起改，见 config._runs_dir 的注释）
if [ -n "${BREASTDG_RUNS:-}" ]; then
  echo "⚠ 检测到 BREASTDG_RUNS=${BREASTDG_RUNS}"
  echo "  它会覆盖默认的 \${PROJECT_ROOT}/runs。若该路径不在数据盘，"
  echo "  产物会随实例释放消失；且与代码里硬拼写的 runs 路径不一致。"
  export BREASTDG_RUNS
fi

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

export PYTHONPATH="${APP_ROOT}/src"
export PYTHONIOENCODING=utf-8

STAGE="${1:-lockbox}"   # lockbox | lockbox-exhaustive | lockbox-full | smoke
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# ---- 废弃 stage 必须在**任何检查之前**拦下（2026-10-07）----
# 原先 lockbox-eval 的废弃提示放在 case 分支里，而**数据检查在它之前** ——
# 于是云上跑 `lockbox-eval` 会先因"缺 OASBUD/BrEaST"退 2，
# 操作者以为只是**挂载没配好**，去补挂数据，而不是去改流程。
# 一个已废弃的 stage 不该要求先把数据传齐才能告诉你它废弃了。
if [ "$STAGE" = "lockbox-eval" ]; then
  cat >&2 <<'EOF'
✗ 本 stage 已废弃。锁箱评估**改在本地做**（2026-10-07，check4 修订）。

为什么改（三条理由）
--------------------
1. **不可逆操作应发生在产物直接落永久存储的地方。**
   云上评估写的是实例盘 —— 实例一释放就没了，而锁箱评估**不可重做**
   （单次纪律）。在本地做，评估卡落盘那一刻就已经在永久存储上。
2. **省掉"拉回 → 冻结 → 回推 → 再拉回"的环形搬运。**
   原设计的死结：冻结阈值在本地（只用源域），评估却在云上 ——
   于是 thresholds.json 必须**回推**到云端检查点旁边，否则前置检查退 4。
   而回推意味着本地源码被复制上云，两端都可能漂移。
   本地评估后，检查点、阈值、评估卡**全在同一台机器上**，无环形搬运。
3. **实例不必挂着计费等你本地走完冻结流程。**
   实例在【拉回 + 校验通过 + 双备份】之后即可**立即释放** ——
   多留一小时都是白烧。

正确顺序（全在本地，见《上云与锁箱操作手册》）
----------------------------------------------
  ① 云上：bash env/run_cloud.sh lockbox-exhaustive        （35 格训练）
  ② 本地：bash env/pull_runs.sh <host> <port>             （拉回 + SHA256 校验）
  ③ 本地：rsync -a runs/cloud/<戳>/lockbox/ runs/lockbox/ （归位）
  ④ 本地：**双备份** runs/lockbox/
  ⑤ **此处即可释放云实例**（已拉回、已校验、已备份）
  ⑥ 本地：python scripts/preflight.py                      （须 10/10）
  ⑦ 本地：python scripts/freeze_thresholds.py
          python scripts/freeze_thresholds.py --dump-source-probs
          python scripts/ensemble_and_calibrate.py --fit-source
  ⑧ 本地：python scripts/evaluate_lockbox.py --full-card    （单次不可逆）
  ⑨ 本地：python scripts/ensemble_and_calibrate.py --outer-eval --tag oasbud_corrected
                                                         --tag oasbud_uncorrected
                                                         --tag breast
          python scripts/e3_polarity_analysis.py            （E3 = 全文核心实验）

数据边界（因此评估队列不再上传到云）
------------------------------------
OASBUD / BrEaST / OASBUD_p4 **不必上传** —— 少传 760+ MB，
少一类挂载失败模式。本地已有全部评估所需数据。
EOF
  exit 2
fi

cd "${APP_ROOT}"

echo "==============================================================="
echo "云端训练  stage=$STAGE"
echo "  代码根 APP_ROOT     = ${APP_ROOT}"
echo "  数据   BREASTDG_DATA= ${BREASTDG_DATA}"
echo "  产物   BREASTDG_RUNS= ${BREASTDG_RUNS:-（未设，用默认 ${APP_ROOT}/runs）}"
echo "  Python ${PY}         = $(${PY} -c 'import sys;print(sys.executable)' 2>/dev/null || echo '(不可用!)')"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader || true
echo "==============================================================="

# 环境自检 —— 这是**第一道闸门**。
# 在 AutoDL 原生部署里它格外重要：基础镜像只有 2.5.1，是我们的
# 安装脚本把 torch 升到 2.6.0 的。这一步就是**证明那个升级真的成功**。
# 不通过 = 没升上去 = 环境与本机不一致 → 立即停，别开始训练。
if ! ${PY} "${APP_ROOT}/env/verify_env.py"; then
  echo "" >&2
  echo "✗ 环境自检未通过 —— 拒绝开始训练。" >&2
  echo "  在 AutoDL 原生部署下，最常见原因是 torch 覆盖安装没生效。" >&2
  echo "  注意：**不要**跑 env/setup_autodl_env.sh —— 它会试图装 torch 2.6.0，" >&2
  echo "        而云端可用的是 2.5.1 + torchvision 0.20.1（官方配对，正确）。" >&2
  echo "        本批 35 格全部在同一云端环境跑完，批次内比较不受影响。" >&2
  echo "        先看上面是哪几条不一致，再决定；不要盲目重装。" >&2
  exit 1
fi

# 数据存在性检查（避免跑到一半才发现没挂上）
# ⚠ 分级检查（2026-10-07，含 check4 的修订）：
#   · 训练必需：3 个源队列（BUSI / BUSBRA / BUSI-UCLM）
#     —— 实核实测：`build_full_source()` 只用 busi+busbra（train/val）
#        与 busuclm（无标签目标域）；**评估队列不参与训练**。
#   · 评估队列（OASBUD / BrEaST / OASBUD_p4）**训练阶段一律不检查**。
#     理由：锁箱评估已改到**本地**做（见下方 lockbox-eval 段的说明），
#     评估队列再也不需要上传到云上 —— 少传 760+ MB，且少一类挂载失败模式。
#     ⚠ 原实现（早前版本）对 `smoke` 也要求评估队列，导致"只想冒烟验证
#       环境连通性"也得先传齐全部数据。已修：smoke 只要 3 个源队列。
REQUIRED=(busi/Dataset_BUSI_with_GT BUSBRA BUSI-UCLM)

for d in "${REQUIRED[@]}"; do
  if [ ! -e "${BREASTDG_DATA}/${d}" ]; then
    echo "✗ 数据缺失：${BREASTDG_DATA}/${d}" >&2
    echo "  数据必须放在**数据盘**（/root/autodl-tmp），且目录名逐字母对上" >&2
    echo "  （Linux 区分大小写：busi 小写、BUSBRA/BUSI-UCLM 大写）。" >&2
    exit 2
  fi
done

# OASBUD 物化目录**只在上传了它时才检查**（训练不需要它）
if [ -d "${BREASTDG_DATA}/OASBUD_p4" ]; then
  for arm in corrected uncorrected; do
    [ -f "${BREASTDG_DATA}/OASBUD_p4/${arm}/manifest.csv" ] \
      || { echo "⚠ OASBUD_p4/${arm}/ 缺 manifest.csv（训练用不到，但若你要在云上评估就会炸）" >&2; }
  done
fi
echo "✓ 数据就绪（stage=$STAGE，检查 ${#REQUIRED[@]} 项）"

# 全部走 _lockbox_driver.py（单格独立进程 + 单格超时 + 可续跑），
# **不用** retrain_lockbox.main()（它无挂死保护、且只在全部跑完后才写
# summary.json，中途挂死即无法续跑）。
DRIVER_SCRIPT="scripts/_lockbox_driver.py"
DL=""
[ -n "${DEADLINE:-}" ] && DL="--deadline $DEADLINE"

# ---- 多卡分片（2026-10-06 新增）-----------------------------------------
# 原版只起**一个**驱动进程。_lockbox_driver 旧为全局单实例（一把锁 + 一份
# 台账），故租 N 张卡时第 2 个进程会因锁冲突退出 —— **多租的卡全闲置**。
# 现在驱动支持 `--shard i --of N --gpu k`：每片独立进程、绑一张卡、
# 用**自己的**锁与台账，格集按 `cells[i::N]` 互斥切分（已实测 8+7=15 无重复）。
#
# NGPU 自动取可见卡数；可用 NGPU=1 强制单卡（行为退化为原样）。
#
# ⚠ 三处刻意的防御写法（2026-10-06 自查抓到，均在 set -euo pipefail 下真会炸）：
#   1) `|| true`：本脚本是 `set -o pipefail`。若 nvidia-smi 不存在，管道返回
#      非零，而**赋值语句里的命令替换失败会触发 set -e 直接退出**。
#   2) 不用 `[ … ] || [ … ] && NGPU=1` 这种链：在"有 GPU"的正常路径下整条
#      求值为**假**，set -e 会因此杀死脚本 —— 即在**唯一重要的场景**里必炸。
#      改用显式 if。
#   3) 不依赖 bash≥4.3 的 `${arr[-1]}`；pid 用普通变量接。
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)}"
if [ -z "$NGPU" ] || [ "$NGPU" -lt 1 ] 2>/dev/null; then
  NGPU=1
fi
echo "检测到 ${NGPU} 张 GPU → 将起 ${NGPU} 个分片进程"

# 计划预览（DRY_RUN=1）：逐片打印认领的格，**不启动训练、不计费**。
# 上云第一件事就该跑它，确认分片无重复、锁与台账路径正确。
plan_sharded() {
  local i
  for i in $(seq 0 $((NGPU - 1))); do
    ${PY} -u "$DRIVER_SCRIPT" "$@" --shard "$i" --of "$NGPU" --gpu "$i" --dry-run
  done
}

# 并行起 N 片并等全部结束。逐片记 rc，不因某片失败而丢掉其他片的产物
# （产物按 final.pt 判重，与分片数无关 → 失败后重跑同一命令即可续上）。
run_sharded() {
  local pids=() i p rc=0
  for i in $(seq 0 $((NGPU - 1))); do
    ${PY} -u "$DRIVER_SCRIPT" "$@" --shard "$i" --of "$NGPU" --gpu "$i" &
    p=$!
    pids+=("$p")
    echo "  → shard $i 已起 (pid $p, gpu $i)"
  done
  for p in "${pids[@]}"; do wait "$p" || rc=$?; done
  echo "全部分片结束（最后非零 rc=$rc；未完成不视为错误，可重入续跑）"
  return 0
}

case "$STAGE" in
  smoke)
    # 冒烟：一次短跑，验证端到端可用（约 3 min）。不写正式 run_id 之外的东西。
    ${PY} -u -m breastdg.train.dev_screen \
      --queue tier4 --directions busi --seeds 42 --steps 300 --eval-every 150
    ;;

  # ---- 主路径：5 骨干 × 3 种子 = 15 格（不含公平基线）----
  # 若只想先拿到主配置的锁箱结果（最短工期），用这段。
  lockbox)
    LOCKBOX_ARGS=(--variants densenet121_se,efficientnet_v2_s,resnet50_cbam,fdnet_gate_dual,swin_t
                  --seeds 42,123,2026)
    if [ "${DRY_RUN:-}" = "1" ]; then
      plan_sharded "${LOCKBOX_ARGS[@]}"
    else
      run_sharded "${LOCKBOX_ARGS[@]}" $DL
    fi
    ;;

  # ---- 以时长换完整性（**用户已选定：跑 35 格**）----
  # 云上成本 = 挂钟 × 费率，而换更高吞吐的边际成本为零。
  # 故**不要**卡在 3 个种子上：加种子是最划算的杠杆 ——
  # 加种子直接加强经不起推敲的结论（如「对抗削弱尖峰」在 dev 期
  # 符号检验 p=0.25）。
  #
  # ⚠ 2026-10-07：本段**不含公平基线**（5 骨干 × 7 种子 = 35 格）。
  #   含基线的全量在 `lockbox-full` 段（7 变体 × 7 种子 = 49 格）。
  lockbox-exhaustive)
    LOCKBOX_ARGS=(--variants densenet121_se,efficientnet_v2_s,resnet50_cbam,fdnet_gate_dual,swin_t
                  --seeds 42,123,2026,7,777,1234,20250101)
    if [ "${DRY_RUN:-}" = "1" ]; then
      plan_sharded "${LOCKBOX_ARGS[@]}"
    else
      run_sharded "${LOCKBOX_ARGS[@]}" $DL
    fi
    ;;

  # ---- 全量：35 格主配置 + 14 格公平基线 = 49 格 ----
  # 公平基线（deepall / mixstyle）回应拒稿意见 5：论文此前只和两个基线比、
  # 且没说基线怎么训的。规格见 retrain_lockbox.LOCKBOX_VARIANTS 的注释。
  #
  # ⚠ 基线给 **7 个种子**（而非 3）：主配置是 7 种子，基线若只给 3 种子，
  #   两边的 CI 宽窄不同，审稿人会问"为什么基线样本少"。
  #   若工期紧张，可改 `--seeds 42,123,2026` 只给基线 3 种子，但**须在文中说明**。
  lockbox-full)
    LOCKBOX_ARGS=(--variants densenet121_se,efficientnet_v2_s,resnet50_cbam,fdnet_gate_dual,swin_t,deepall,mixstyle
                  --seeds 42,123,2026,7,777,1234,20250101)
    if [ "${DRY_RUN:-}" = "1" ]; then
      plan_sharded "${LOCKBOX_ARGS[@]}"
    else
      run_sharded "${LOCKBOX_ARGS[@]}" $DL
    fi
    ;;

  # ---- 锁箱评估：**已改到本地执行**（2026-10-07，check4 修订）----
  lockbox-eval)
    cat >&2 <<'EOF'
✗ 本 stage 已废弃。锁箱评估**改在本地做**。

为什么改（三条理由）
--------------------
1. **不可逆操作应发生在产物直接落永久存储的地方。**
   云上评估写的是实例盘 —— 实例一释放就没有了，而锁箱评估是**不可重做**的
   （单次纪律）。在本地做，评估卡落盘的那一刻就已经在永久存储上。
2. **省掉"拉回 → 冻结 → 回推 → 再拉回"的环形搬运。**
   原设计的死结：冻结阈值在本地（只用源域），评估却在云上 ——
   于是 thresholds.json 必须**回推**到云端检查点旁边，否则前置检查退 4。
   而回推意味着本地源码被复制到云上，两端都可能漂移。
   本地评估后，检查点、阈值、评估卡**全在同一台机器上**，无环形搬运。
3. **实例不必挂着计费等你本地走完冻结流程。**
   实例在【拉回 + 校验通过 + 双备份】之后就可以**立即释放** ——
   多留一小时都是白烧。

正确顺序（全在本地，见《上云与锁箱操作手册》）
----------------------------------------------
  ① 云上：bash env/run_cloud.sh lockbox-exhaustive      （35 格训练）
  ② 本地：bash env/pull_runs.sh <host> <port>           （拉回 + SHA256 校验）
  ③ 本地：rsync -a runs/cloud/<戳>/lockbox/ runs/lockbox/   （归位）
  ④ 本地：**双备份** runs/lockbox/
  ⑤ **此处即可释放云实例**（检查点已拉回、已校验、已备份）
  ⑥ 本地：python scripts/preflight.py                    （须 10/10）
  ⑦ 本地：python scripts/freeze_thresholds.py
          python scripts/freeze_thresholds.py --dump-source-probs
          python scripts/ensemble_and_calibrate.py --fit-source
  ⑧ 本地：python scripts/evaluate_lockbox.py --full-card  （单次不可逆）
  ⑨ 本地：python scripts/ensemble_and_calibrate.py --outer-eval --tag oasbud_corrected
                                                       --tag oasbud_uncorrected
                                                       --tag breast
          python scripts/e3_polarity_analysis.py          （E3 = 全文核心实验，勿靠记性）

数据边界（因此本 stage 不再检查评估队列）
----------------------------------------
评估队列（OASBUD / BrEaST / OASBUD_p4）**不必上传到云** —— 少传 760+ MB，
且少一类挂载失败模式。本地已有全部所需数据。
EOF
    exit 2
    ;;

  *)
    echo "未知 stage: $STAGE" >&2
    echo "可用：smoke | lockbox | lockbox-exhaustive | lockbox-full" >&2
    echo "  （lockbox-eval 已废弃 → 锁箱评估改到本地，见 run_cloud.sh 内说明）" >&2
    exit 2
    ;;
esac

echo "==============================================================="
echo "stage=$STAGE 完成。产物在 ${BREASTDG_RUNS:-${APP_ROOT}/runs}/"
case "${BREASTDG_RUNS:-${APP_ROOT}/runs}" in
  /root/autodl-tmp/*) echo "  ✓ 在数据盘上（实例释放后仍在，但**仍须拉回本地**）" ;;
  *) echo "  ⚠ 不在 /root/autodl-tmp 下 —— 释放实例即永久丢失！" >&2 ;;
esac
echo "  下一步（本地）：bash env/pull_runs.sh <host> <port>"
echo "==============================================================="
