#!/usr/bin/env bash
# 云端产物拉回 —— **在本地机器上跑**，不是在容器里跑。
#
# 为什么必须有这一步（用户 2026-10-06 标为**硬阻断**）
# ----------------------------------------------------
# AutoDL 的实例存储随实例释放**一起消失**：实例一关，`/root/autodl-tmp/runs/`
# 连同里面的锁箱检查点、评估卡、逐图概率，全部不可找回。
# 锁箱评估是**不可逆的一次性操作**（F7.1）—— 拉不回来就意味着要么重冻阈值
# 重评（违反单次纪律），要么整章结果没有出处。故本步是**释放实例前的强制动作**。
#
# 为什么不在 run_cloud.sh 里做
# ---------------------------
# `run_cloud.sh` 在**容器内**执行，容器看不到本地，也没有可用的回程凭据。
# 拉回只能由本地主动发起（SSH/rsync 出站）。故拆成独立脚本，与 run_cloud.sh
# 成对使用：云上跑 → 本地拉 → 校验 → 才释放实例。
#
# 用法
# ----
#   bash env/pull_runs.sh <host> <ssh_port> [远端 runs 目录] [本地目标目录]
#
# 例（AutoDL 给的形如 `ssh -p 12345 root@connect.xxx.seetacloud.com`）：
#   bash env/pull_runs.sh connect.xxx.seetacloud.com 12345
#   → 拉到 runs/cloud/20261007_143000/
#
# 目标目录默认带**时间戳**，所以从不覆盖既有产物 —— 一天拉三次也不会互相踩。
#
# 退出码
# ------
# 0 = 拉回且校验通过；2 = 参数/连接问题；3 = 拉回成功但**校验失败**（勿释放实例！）
#
# 边界
# ----
# 只搬运产物；不接触任何私有队列（F9 + 记忆硬规则）。本脚本不引用、不创建、
# 不命名任何非公开队列的路径。

set -euo pipefail

HOST="${1:-}"
# ⚠ PORT 默认值**必须是空**，不能是 22（2026-10-08 修）：
#   若默认 22，则 `pull_runs.sh autodl`（用户忘写第二个 `-`）会走
#   "显式 host"分支 → 拼成 `root@autodl` → 别名解析不了 → 失败。
#   默认空 = 默认走别名（最常用的那条路）；要显式 host 就明确写端口。
PORT="${2:-}"
# ⚠ 2026-10-08 修：默认远端路径**改对**。
#   原为 /root/autodl-tmp/runs —— 但 run_cloud.sh 不设 BREASTDG_RUNS，
#   产物实际落在 PROJECT_ROOT/runs = **/root/autodl-tmp/project/runs**。
#   两个路径只差一个 `project/`，却会让探活直接失败 ——
#   而失败提示写的是"实例可能已释放"，**把人往完全错误的方向引**
#   （实际实例好好的，只是路径不对）。这是最难查的一类错。
REMOTE_DIR="${3:-/root/autodl-tmp/project/runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_DIR="${4:-runs/cloud/${STAMP}}"

if [ -z "$HOST" ]; then
  echo "用法: bash env/pull_runs.sh <host|别名> <端口|-> [远端runs目录] [本地目标目录]" >&2
  echo "  SSH 别名（推荐，~/.ssh/config 里已有 Host autodl）:" >&2
  echo "      bash env/pull_runs.sh autodl -" >&2
  echo "  显式 host + port:" >&2
  echo "      bash env/pull_runs.sh connect.xxx.seetacloud.com 12345" >&2
  exit 2
fi

# ⚠ 2026-10-08 修：支持 **SSH 别名**。
#   原实现无条件拼 `root@${HOST}` 并加 `-p ${PORT}` —— 别名会被拼成
#   `root@autodl`，而别名只在 ssh 客户端解析（`root@别名` 解析不了）→ 直接失败。
#   规则：PORT 为 `-` 或空 → 走别名（不加 root@、不加 -p）；
#         否则 → 显式 host（加 root@ 与 -p）。
if [ -z "$PORT" ] || [ "$PORT" = "-" ]; then
  TARGET="$HOST"                                  # 别名形式
  SSH="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  SCPO="-o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  RSYNC_E="ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  echo "连接方式：SSH 别名 '$HOST'"
else
  TARGET="root@${HOST}"                       # 显式 host（别名形式不能加 root@）
  SSH="ssh -p ${PORT} -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  SCPO="-P ${PORT} -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  RSYNC_E="ssh -p ${PORT} -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
  echo "连接方式：${HOST}:${PORT}（user=root）"
fi

echo "==============================================================="
echo "拉回云端产物"
echo "  远端  ${TARGET}:${REMOTE_DIR}"
echo "  本地  ${LOCAL_DIR}"
echo "==============================================================="

# ---------------------------------------------------------------------------
# 0) 先探活：连不上就别往下走（避免把一个空目录当成"拉完了"）
# ---------------------------------------------------------------------------
if ! $SSH "${TARGET}" "test -d '${REMOTE_DIR}'" 2>/dev/null; then
  echo "✗ 连不上或远端目录不存在：${TARGET}:${REMOTE_DIR}" >&2
  echo "  可能原因：实例已释放 / 端口错 / 公钥没配 / 路径拼错。" >&2
  echo "  ⚠ 若实例确已释放，产物**无法找回** —— 实例释放必须在本步成功之后。" >&2
  exit 2
fi
echo "✓ 远端目录可达"

# 先看规模，让人对"拉多少"有预期（锁箱 35 格的 final.pt 不小）
$SSH "${TARGET}" \
  "du -sh '${REMOTE_DIR}' 2>/dev/null; find '${REMOTE_DIR}' -name final.pt | wc -l" \
  || true

mkdir -p "$LOCAL_DIR"

# ---------------------------------------------------------------------------
# 1) 搬运：优先 rsync（可续传、可重入），退化到 **tar 单流**
#
# ⚠ 2026-10-07 修（实测本机无 rsync）：
#   本机 Git Bash **没有 rsync**（`command -v rsync` 空），故 scp 分支是
#   实际上会走的路径。原实现有两个问题：
#     ① **只取第一个子目录** —— `find ... | head -1` 在远端 `runs/` 下有
#        lockbox/ dev_screen/ lockbox_eval/ 等多个子目录时，**静默丢掉其余**，
#        而校验只查"有没有卡"，照样报成功。这是最危险的一类：拉回来的东西
#        "看起来对"，但缺了一部分。
#     ② **scp -r 传数千小文件极慢** —— 35 个检查点 + 各 config/history.json，
#        每文件一次协议往返。
#   改为：远端先 tar 成一个流 → scp 单文件 → 本地解包。
#   校验紧随其后（评估卡 sha256 + 检查点反校 + 0 落地守卫），不受影响。
# ---------------------------------------------------------------------------
if command -v rsync >/dev/null 2>&1; then
  echo "--- 传输方式：rsync（可续传）---"
  # --partial：断点续传（中断后重跑接着传，不从头来）
  # ⚠ 不用 --append-verify：那是给"追加型文件"的，断点续传场景有边界情况；
  #   本脚本在真机上实测过失败（`Bad port '-'`）。保守取标准组合。
  rsync -avz --partial --progress \
        --exclude='__pycache__/' --exclude='*.pyc' \
        -e "${RSYNC_E}" \
        "${TARGET}:${REMOTE_DIR}/" "${LOCAL_DIR}/"
else
  echo "--- 传输方式：tar + scp（本机无 rsync）---"
  BUNDLE="runs_bundle_${STAMP}.tar"
  TMP="${LOCAL_DIR}/../_pull_tmp_${STAMP}"
  rm -rf "$TMP"; mkdir -p "$TMP"

  # 远端打包（排除 __pycache__：体积小但数量多，且本地可重建）
  echo "  远端打包..."
  if ! $SSH "${TARGET}" \
        "cd '${REMOTE_DIR}' && tar -cf '/tmp/${BUNDLE}' \
           --exclude='__pycache__' --exclude='*.pyc' . \
         && ls -l '/tmp/${BUNDLE}' | awk '{print \"  远端包大小:\", \$5, \"字节\"}'"; then
    echo "✗ 远端打包失败 —— 未搬运任何文件。产物仍在云端，**不要释放实例**。" >&2
    rm -rf "$TMP"; exit 2
  fi

  echo "  下载中（单流）..."
  if ! scp $SCPO "${TARGET}:/tmp/${BUNDLE}" "$TMP/"; then
    echo "✗ scp 失败 —— 未搬运任何文件。产物仍在云端，**不要释放实例**。" >&2
    rm -rf "$TMP"; exit 2
  fi

  echo "  本地解包..."
  if ! tar -xf "$TMP/${BUNDLE}" -C "$LOCAL_DIR"; then
    echo "✗ 解包失败（下载可能被截断）—— 重跑本脚本。" >&2
    rm -rf "$TMP"; exit 2
  fi

  # 远端清理（包体占数据盘，别留着）
  $SSH "${TARGET}" "rm -f '/tmp/${BUNDLE}'" || true
  rm -rf "$TMP"
fi

# ---- 硬守卫：0 落地即失败（不管走的是 rsync 还是 scp）----
# 这条与上面 scp 分支是**两回事**：rsync 也可能"成功但拉了个空目录"
# （远端 runs/ 存在但里面什么都没有，例如训练全失败）。
_n_any=$(find "$LOCAL_DIR" -type f 2>/dev/null | wc -l)
if [ "$_n_any" -eq 0 ]; then
  echo "✗ 落地文件数为 0 —— 搬运未生效或远端本就没有产物。" >&2
  echo "  产物仍在云端，**不要释放实例**。先到云上确认 ${REMOTE_DIR} 的内容。" >&2
  exit 2
fi

echo
echo "--- 落地统计 ---"
echo "  final.pt        $(find "$LOCAL_DIR" -name 'final.pt'   2>/dev/null | wc -l) 个"
echo "  evalcard_*.json $(find "$LOCAL_DIR" -name 'evalcard_*.json' 2>/dev/null | wc -l) 个"
echo "  probs_*.csv     $(find "$LOCAL_DIR" -name 'probs_*.csv' 2>/dev/null | wc -l) 个"
echo "  thresholds.json $(find "$LOCAL_DIR" -name 'thresholds.json' 2>/dev/null | wc -l) 个"
echo "  总大小          $(du -sh "$LOCAL_DIR" 2>/dev/null | cut -f1)"

# ---------------------------------------------------------------------------
# 2) 校验：评估卡自带的 SHA256 必须对得上
#    这是**唯一**能证明"卡没在传输中被截断"的机械证据。锁箱评估不可重做，
#    所以"看起来拉回来了"不够 —— 必须逐张核对。
# ---------------------------------------------------------------------------
echo
echo "--- 校验评估卡 SHA256 ---"
bad=0; n=0
while IFS= read -r sha_file; do
  # ⚠ 卡的扩展名是 `.json`，旁路文件是 `.sha256` —— 剥后缀时**必须补回 .json**。
  #   2026-10-07 实测：漏了这一步 → `-f` 永远为假 → 校验 0 张却报"通过"退 0，
  #   是个**假绿**（看起来校过了，其实一张都没校）。锁箱评估不可重做，
  #   这种假绿比不校验更坏 —— 它把"没检查"伪装成"检查通过"。
  card="${sha_file%.sha256}.json"
  if [ ! -f "$card" ]; then
    echo "  ! $(basename "$sha_file") 没有配对的 .json 卡，跳过" >&2
    continue
  fi
  n=$((n + 1))
  want="$(tr -d ' \r\n' < "$sha_file")"
  # 优先 sha256sum（Linux/Git Bash），退化到 shasum（macOS）
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$card" | cut -d' ' -f1)"
  else
    got="$(shasum -a 256 "$card" | cut -d' ' -f1)"
  fi
  if [ "$want" = "$got" ]; then
    echo "  ✓ $(basename "$card")"
  else
    echo "  ✗ $(basename "$card")  sha256 不符" >&2
    echo "      记录 ${want:0:16}…  实算 ${got:0:16}…" >&2
    bad=$((bad + 1))
  fi
done < <(find "$LOCAL_DIR" -name 'evalcard_*.sha256' 2>/dev/null)

# 阈值文件与检查点也必须成套（缺任一即"这批评不出数"）
ck_n=$(find "$LOCAL_DIR" -name 'final.pt' 2>/dev/null | wc -l)
th_n=$(find "$LOCAL_DIR" -name 'thresholds.json' 2>/dev/null | wc -l)
if [ "$ck_n" -gt 0 ] && [ "$th_n" -lt "$ck_n" ]; then
  echo "  ! 检查点 ${ck_n} 个但冻结阈值只有 ${th_n} 个 —— 缺的那部分无法评估" >&2
fi

echo
echo "--- 反校检查点 SHA256（P2④：卡里已记，白送的校验）---"
# 锁箱评估卡内的 audit.checkpoints 记了每个 final.pt 的 sha256。
# 那是**训练时**算的；用它反校拉回的检查点，能独立发现"传输中损坏/截断"，
# 且不需要任何额外信息。有 python 就做，没有就明确跳过（不假装校过）。
if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
  PY_BIN="$(command -v python3 || command -v python)"
  "$PY_BIN" - "$LOCAL_DIR" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def sha(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()
# 建立 name → path 索引（检查点文件名可能重复，故用 basename 匹配）
final_by_name = {}
for p in root.rglob("final.pt"):
    final_by_name.setdefault(p.parent.name, p)     # run_id → path
n, bad, missing = 0, 0, 0
for card in root.rglob("evalcard_*.json"):
    try:
        d = json.loads(card.read_text(encoding="utf-8"))
    except Exception:
        continue
    for key, rec in (d.get("audit", {}).get("checkpoints", {}) or {}).items():
        want, path = rec.get("sha256"), rec.get("path")
        if not want or not path: continue
        run_id = Path(str(path).replace("\\", "/")).parent.name
        p = final_by_name.get(run_id)
        if p is None:
            missing += 1
            continue
        n += 1
        got = sha(p)
        if got == want:
            print(f"  ✓ {run_id}/final.pt")
        else:
            print(f"  ✗ {run_id}/final.pt  sha256 不符", file=sys.stderr)
            print(f"      卡内记录 {want[:16]}…  实算 {got[:16]}…", file=sys.stderr)
            bad += 1
print(f"  反校 {n} 个检查点，不符 {bad} 个，卡内有记录但未拉到 {missing} 个")
sys.exit(3 if (bad or missing) else 0)
PYEOF
  rc=$?
  if [ "$rc" = "3" ]; then
    echo "✗ 检查点反校未通过 —— **不要释放实例**。" >&2
    echo "  典型原因：某个格子的 final.pt 没拉全（评估卡需要它来复现）。" >&2
    exit 3
  fi
elif [ "$ck_n" -gt 0 ]; then
  echo "  ! 无 python，跳过检查点反校（**未校验**，不等于通过）" >&2
fi

echo
echo "==============================================================="
if [ "$bad" -gt 0 ]; then
  echo "✗ 校验失败 ${bad}/${n} 张卡 —— **不要释放实例**，重跑本脚本再拉一次。" >&2
  echo "  重跑是安全的：rsync --partial 会续传，不会重复下载已完整的文件。" >&2
  exit 3
fi
echo "✓ 拉回完成，${n} 张评估卡校验通过 → ${LOCAL_DIR}"
echo
echo "下一步（顺序不可颠倒）："
echo "  1) **归位**：本脚本落在 ${LOCAL_DIR}，而本地的"
echo "     freeze_thresholds / evaluate_lockbox 走 runs/lockbox/ ——"
echo "     中间差一步搬运，不做的话本地脚本看不到这批检查点："
echo "       rsync -a ${LOCAL_DIR}/lockbox/ runs/lockbox/     # 保留 seed/run_id 层级"
echo "     （归位后先跑 scripts/preflight.py，再跑 freeze_thresholds.py）"
echo "  2) 确认 runs/lockbox/ 里 final.pt 数量与跑过的格子数一致；"
echo "  3) 把 ${LOCAL_DIR} 归档/备份（它是锁箱量的唯一副本）；"
echo "  4) **此时**才去释放云实例。"
echo "==============================================================="
