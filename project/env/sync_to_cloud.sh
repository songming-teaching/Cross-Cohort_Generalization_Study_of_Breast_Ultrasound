#!/usr/bin/env bash
# 一键同步：把本地代码推到云端（**在 WSL Ubuntu 里跑**）
#
# 为什么需要它
# ------------
# 长命令在粘贴时**反复被换行截断**，已经造成四次事故：
#   ① `python` 后参数断到下一行 → 进了 REPL，看起来卡住
#   ② `autodl:/root/.../env/2>/dev/null` → `2>` 被吃掉，目标路径变成 `env/2>`，
#      文件**根本没传上去**，而 rsync 不报错 —— 最危险的一种
#   ③ heredoc 内容被加前导空格 → Python IndentationError
#   ④ 文件路径被拼进远端目标
#
# 故把命令都收进本脚本，你只需要输一条**极短**的命令。
#
# 用法（WSL）
# ----------
#     cd /mnt/c/Users/15315/Desktop/paper1st/meliorate/project
#     bash env/sync_to_cloud.sh
#
# 可选：只推单个路径
#     bash env/sync_to_cloud.sh env/run_cloud.sh
#
# 退出码：0 = 全部推送成功且远端内容已验证；2 = 有失败
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="autodl:/root/autodl-tmp/project"

# 默认推送清单。**只列会变的代码**（不动 data/ figures/ reports/）。
FILES=(
  "env/run_cloud.sh"
  "env/setup_autodl_env.sh"
  "env/verify_env.py"
  "env/pull_runs.sh"
  "env/push_data.sh"
  "env/prefetch_weights.py"
)
DIRS=(
  "scripts"
  "src"
  "splits"
)

if [ $# -gt 0 ]; then
  FILES=("$@")          # 用户指定了 → 只推这些
  DIRS=()
fi

cd "$SRC_ROOT"
echo "本地根: $SRC_ROOT"
echo "远端  : $REMOTE"
echo "==============================================================="

fail=0

# ---- 1) 推送前先做本机自检（省得把坏文件推上去）----
echo "--- 本机自检 ---"
if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else PY=""; fi

if [ -n "$PY" ]; then
  # 行尾符：CRLF 会让 bash 报 "invalid option name"
  crlf=$("$PY" - <<'PYEOF'
import pathlib, sys
bad = []
for p in list(pathlib.Path('env').glob('*.sh')) + list(pathlib.Path('scripts').glob('*.sh')):
    if p.read_bytes().count(b'\r\n'):
        bad.append(str(p))
print(" ".join(bad))
PYEOF
)
  if [ -n "$crlf" ]; then
    echo "  ✗ 以下文件是 CRLF（bash 会报 invalid option name），先修：$crlf" >&2
    echo "    修法：sed -i 's/\r$//' <文件>" >&2
    fail=1
  else
    echo "  ✓ 行尾符全为 LF"
  fi

  # set -u 未定义变量
  if [ -f scripts/_chk_sh_unbound.py ]; then
    if ! "$PY" scripts/_chk_sh_unbound.py >/dev/null 2>&1; then
      echo "  ✗ shell 脚本存在 set -u 未定义变量引用：" >&2
      "$PY" scripts/_chk_sh_unbound.py >&2 || true
      fail=1
    else
      echo "  ✓ 无未定义变量引用"
    fi
  fi

  # 语法
  if ! "$PY" -c "import ast,pathlib,sys; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8')) for f in sys.argv[1:]]" $(find env -name '*.py') >/dev/null 2>&1; then
    echo "  ✗ env/ 下有 Python 语法错误" >&2; fail=1
  else
    echo "  ✓ Python 语法通过"
  fi
fi

for f in "${FILES[@]}"; do
  # ⚠ 只对 **.sh** 跑 bash -n —— 推清单里也有 .py（verify_env / prefetch_weights）
  case "$f" in
    *.sh) bash -n "$f" 2>/dev/null || { echo "  ✗ $f bash 语法错误" >&2; fail=1; } ;;
  esac
done
for d in "${DIRS[@]}"; do
  [ -d "$d" ] || { echo "  ✗ 本地缺目录 $d" >&2; fail=1; }
done
[ "$fail" = "0" ] || { echo "✗ 自检未过，**不推送**。" >&2; exit 2; }
echo "  ✓ 全部通过"

# ---- 2) 推送 ----
echo ""
echo "--- 推送 ---"
for f in "${FILES[@]}"; do
  d="$(dirname "$f")"
  # 先确保远端目录存在（否则 rsync 报 mkdir failed）
  ssh autodl "mkdir -p '$d'" 2>/dev/null || true
  if rsync -az "$f" "autodl:/root/autodl-tmp/project/$f"; then
    echo "  ✓ $f"
  else
    echo "  ✗ $f" >&2; fail=1
  fi
done
for d in "${DIRS[@]}"; do
  if rsync -az --delete "$d/" "autodl:/root/autodl-tmp/project/$d/"; then
    echo "  ✓ $d/"
  else
    echo "  ✗ $d/" >&2; fail=1
  fi
done

# ---- 3) 远端验证：**内容真的对上了吗**（治"2> 被吃掉"那类静默失败）----
echo ""
echo "--- 远端验证（关键文件的 sha256 必须与本地一致）---"
vbad=0
for f in "${FILES[@]}"; do
  if command -v sha256sum >/dev/null 2>&1; then
    l=$(sha256sum "$f" | cut -d' ' -f1)
  else
    l=$(shasum -a 256 "$f" | cut -d' ' -f1)
  fi
  r=$(ssh autodl "sha256sum '/root/autodl-tmp/project/$f' 2>/dev/null | cut -d' ' -f1" || echo "")
  if [ -n "$r" ] && [ "$l" = "$r" ]; then
    echo "  ✓ $f"
  else
    echo "  ✗ $f  本地 ${l:0:12}…  远端 ${r:0:12}…" >&2
    vbad=$((vbad + 1))
  fi
done

echo ""
echo "==============================================================="
if [ "$vbad" -gt 0 ] || [ "$fail" != "0" ]; then
  echo "✗ 有 ${vbad} 个文件远端内容与本地不一致 —— **不要开始训练**。" >&2
  exit 2
fi
echo "✓ 全部推送成功，远端内容已逐文件核对一致"
echo ""
echo "下一步（WSL）："
echo "  ssh autodl 'cd /root/autodl-tmp/project && bash env/run_cloud.sh smoke'"
echo "==============================================================="
