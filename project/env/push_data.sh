#!/usr/bin/env bash
# 数据上传 —— **在本地机器上跑**（`pull_runs.sh` 的反向）。
#
# 为什么只传 3 个队列（约 1.4 GB，不是 2.1 GB）
# ------------------------------------------------
# 锁箱评估已改到**本地**做（见 run_cloud.sh 的 lockbox-eval 段说明），
# 故评估队列**一律不上传**：
#
#   上传 ✓  busi/   BUSBRA/   BUSI-UCLM/      ← 训练用，实核实测：
#                                              build_full_source() 只用
#                                              busi+busbra(train/val)
#                                              + busuclm(无标签目标域)
#   不上传 ✗ BrEaST/(464M)  OASBUD.mat/(297M)  OASBUD_p4/(9.6M)
#
# 少传 760+ MB，且**少一类挂载失败模式**（路径拼错/大小写不符）。
#
# 传输方式（2026-10-07 修：本机**没有 rsync**）
# ---------------------------------------------
# Git Bash 默认不带 rsync。实测本机：rsync ✗ / scp ✓ / tar ✓ / WSL ✓。
# 故本脚本**自动选路**：
#   · 有 rsync → rsync（可续传，最优）
#   · 无 rsync → **tar + scp**（单流，不是 scp -r）
# `scp -r` 传 6000+ 个小文件是灾难（每文件一次往返）。tar 打包成**一个流**，
# 快得多，而且能对整个包做 sha256 校验 —— 比逐文件计数更硬。
#
# 用法
# ----
#   bash env/push_data.sh autodl -                       # 走 ~/.ssh/config 别名
#   bash env/push_data.sh connect.xxx.seetacloud.com 12345
#   bash env/push_data.sh autodl - /root/autodl-tmp/data /c/Users/15315/Desktop/paper1st/data
#
# 参数：<ssh目标> <端口|-> [远端数据目录] [本地数据目录]
#
# 退出码：0 = 上传且校验通过；2 = 参数/连接/数据缺失；3 = 传输或校验失败
#
# 边界：只搬公开队列；不接触任何私有队列（F9 + 记忆硬规则）。
set -euo pipefail

TARGET="${1:-}"
PORT="${2:--}"
REMOTE_DIR="${3:-/root/autodl-tmp/data}"
LOCAL_DIR="${4:-}"

if [ -z "$TARGET" ]; then
  echo "用法: bash env/push_data.sh <ssh目标> <端口|-> [远端数据目录] [本地数据目录]" >&2
  echo "例:   bash env/push_data.sh autodl -" >&2
  echo "      bash env/push_data.sh connect.xxx.seetacloud.com 12345" >&2
  exit 2
fi

# 端口为 "-" 或空 → 不加 -p（走 ssh config 的 Host 别名）
if [ "$PORT" = "-" ] || [ -z "$PORT" ]; then
  SSH_OPTS="-o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new"
  RSYNC_E="ssh -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new"
  SCP_OPTS="-o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new"
else
  SSH_OPTS="-p ${PORT} -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new"
  RSYNC_E="ssh -p ${PORT} -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new"
  SCP_OPTS="-P ${PORT} -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new"
fi

# ---------------------------------------------------------------------------
# 0) 定位本地数据根
# ---------------------------------------------------------------------------
if [ -z "$LOCAL_DIR" ]; then
  for cand in \
    "/c/Users/15315/Desktop/paper1st/data" \
    "C:/Users/15315/Desktop/paper1st/data" \
    "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/../data" \
    "../data" ; do
    if [ -d "${cand}/busi" ] || [ -d "${cand}/BUSI" ]; then
      LOCAL_DIR="$cand"; break
    fi
  done
fi
if [ -z "$LOCAL_DIR" ] || [ ! -d "$LOCAL_DIR" ]; then
  echo "✗ 找不到本地数据根。请显式给第 4 参数，例如：" >&2
  echo "    bash env/push_data.sh ${TARGET} ${PORT} ${REMOTE_DIR} /c/Users/15315/Desktop/paper1st/data" >&2
  exit 2
fi

echo "==============================================================="
echo "数据上传（只传训练所需的 3 个源队列）"
echo "  本地  ${LOCAL_DIR}"
echo "  远端  ${TARGET}:${REMOTE_DIR}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# 1) 本地自检：缺任何一个都别开始传（传一半最难查）
# ---------------------------------------------------------------------------
echo "--- 本地自检 ---"
missing=0
for d in busi/Dataset_BUSI_with_GT BUSBRA BUSI-UCLM; do
  if [ ! -e "${LOCAL_DIR}/${d}" ]; then
    echo "  ✗ 缺 ${LOCAL_DIR}/${d}" >&2
    missing=1
  else
    printf "  ✓ %-28s %6s  %6d 文件\n" "$d" \
      "$(du -sh "${LOCAL_DIR}/${d}" 2>/dev/null | cut -f1)" \
      "$(find "${LOCAL_DIR}/${d}" -type f 2>/dev/null | wc -l)"
  fi
done
[ "$missing" = "0" ] || { echo "✗ 本地数据不全，终止。" >&2; exit 2; }

# ---------------------------------------------------------------------------
# 2) 远端可达性 + 目录创建
# ---------------------------------------------------------------------------
if ! ssh $SSH_OPTS "$TARGET" "mkdir -p '${REMOTE_DIR}'" 2>/dev/null; then
  echo "✗ 连不上 ${TARGET}（或无权创建 ${REMOTE_DIR}）" >&2
  echo "" >&2
  echo "  若用别名 'autodl'，先在本机自测：" >&2
  echo "      ssh -v autodl 'hostname'" >&2
  echo "  Windows 上最常见的三个原因：" >&2
  echo "    1) 文件被存成 config.txt（记事本会自动加 .txt）——" >&2
  echo "       必须是 %USERPROFILE%\\.ssh\\config，无扩展名；" >&2
  echo "       查：ls -la ~/.ssh/  （Git Bash 里 ~ = C:/Users/15315）" >&2
  echo "    2) config 里 Host 名拼错或缩进不对（Host/Port/HostName/User" >&2
  echo "       必须在同一 Host 段内，每段内**不能有空行**）；" >&2
  echo "    3) 实例已关机／端口变了（AutoDL 每次开机端口可能换）。" >&2
  echo "  自测命令：ssh -G autodl | head -20   （会打印实际生效的配置）" >&2
  exit 2
fi
echo "✓ 远端可达：$(ssh $SSH_OPTS "$TARGET" 'hostname' 2>/dev/null)"

remote_free=$(ssh $SSH_OPTS "$TARGET" \
  "df -BG --output=avail '${REMOTE_DIR}' 2>/dev/null | tail -1 | tr -dc '0-9'" || echo "")
if [ -n "$remote_free" ]; then
  local_size=$(du -sm "$LOCAL_DIR" 2>/dev/null | cut -f1)
  echo "  远端可用 ${remote_free}G / 本次约需 $((local_size / 1024 + 1))G"
  if [ "$remote_free" -lt "$((local_size / 1024 + 2))" ]; then
    echo "✗ 远端空间不足" >&2; exit 2
  fi
fi

# ---------------------------------------------------------------------------
# 3) 搬运 —— 自动选路
# ---------------------------------------------------------------------------
BUNDLE="breastdg_data_$(date +%Y%m%d_%H%M%S).tar"
LOCAL_TAR="/tmp/${BUNDLE}"

if command -v rsync >/dev/null 2>&1; then
  echo ""
  echo "--- 传输方式：rsync（可续传）---"
  for d in busi BUSBRA BUSI-UCLM; do
    echo "  → ${d}"
    rsync -a --partial --info=progress2 -e "$RSYNC_E" \
          "${LOCAL_DIR}/${d}/" "${TARGET}:${REMOTE_DIR}/${d}/"
  done
else
  echo ""
  echo "--- 传输方式：tar + scp（本机无 rsync）---"
  echo "  打包（不压缩：图已压缩，gzip 只费 CPU 不提速）..."
  tar -cf "$LOCAL_TAR" -C "$LOCAL_DIR" busi BUSBRA BUSI-UCLM
  sz=$(du -sh "$LOCAL_TAR" | cut -f1)
  echo "  已打包 ${sz} → ${LOCAL_TAR}"

  # 本地 sha256（用作远端校验基准）
  if command -v sha256sum >/dev/null 2>&1; then
    SHR="$(sha256sum "$LOCAL_TAR" | cut -d' ' -f1)"
  else
    SHR="$(shasum -a 256 "$LOCAL_TAR" | cut -d' ' -f1)"
  fi
  echo "  sha256 ${SHR:0:16}…"

  echo "  上传中（单流，比 scp -r 传数千小文件快得多）..."
  scp $SCP_OPTS "$LOCAL_TAR" "${TARGET}:${REMOTE_DIR}/"

  # 远端校验 + 解包 + 删包（一次 ssh）
  echo "  远端校验并解包..."
  if ! ssh $SSH_OPTS "$TARGET" "
      set -e
      cd '${REMOTE_DIR}'
      got=\$(sha256sum '${BUNDLE}' | cut -d' ' -f1)
      if [ \"\$got\" != '${SHR}' ]; then
        echo \"✗ 远端 sha256 不符：期望 ${SHR:0:16}… 实得 \${got:0:16}…\" >&2
        exit 3
      fi
      echo '  ✓ sha256 一致'
      tar -xf '${BUNDLE}' -C '${REMOTE_DIR}'
      rm -f '${BUNDLE}'
      echo '  ✓ 已解包并删除包体'
    "; then
    echo "✗ 传输或解包失败。本地包保留在 ${LOCAL_TAR}，可重传。" >&2
    exit 3
  fi
  rm -f "$LOCAL_TAR"
fi

# ---------------------------------------------------------------------------
# 4) 校验：文件数逐队列比对
# ---------------------------------------------------------------------------
echo ""
echo "--- 校验（文件数逐队列比对）---"
bad=0
for d in busi BUSBRA BUSI-UCLM; do
  n_local=$(find "${LOCAL_DIR}/${d}" -type f 2>/dev/null | wc -l)
  n_remote=$(ssh $SSH_OPTS "$TARGET" "find '${REMOTE_DIR}/${d}' -type f 2>/dev/null | wc -l")
  if [ "$n_local" = "$n_remote" ]; then
    printf "  ✓ %-12s 本地 %6d == 远端 %6d\n" "$d" "$n_local" "$n_remote"
  else
    printf "  ✗ %-12s 本地 %6d != 远端 %6d\n" "$d" "$n_local" "$n_remote" >&2
    bad=$((bad + 1))
  fi
done

echo ""
echo "==============================================================="
if [ "$bad" -gt 0 ]; then
  echo "✗ ${bad} 个队列文件数不一致 —— 重跑本脚本。" >&2
  exit 3
fi
echo "✓ 上传完成，3 个队列文件数全部一致"
echo ""
echo "下一步（**在云实例上**）："
echo "  1) 代码若还没同步：把 project/ rsync 上去（须含 src/ scripts/ env/ splits/）"
echo "  2) 环境自检（**第一道闸门**）："
echo "       cd <代码根> && python -u env/verify_env.py"
echo "       全绿才算数；不绿说明 torch 覆盖没生效 → 先跑 env/setup_autodl_env.sh"
echo "  3) 冒烟：bash env/run_cloud.sh smoke"
echo "  4) 开跑：bash env/run_cloud.sh lockbox-exhaustive"
echo "==============================================================="
