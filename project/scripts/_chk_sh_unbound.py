"""机械检查：shell 脚本在 `set -u` 下的未定义变量引用（preflight 坑11）。

为什么需要它
------------
`set -euo pipefail` 里的 `-u` 遇到未定义变量**直接退出**。
本仓的云端脚本全带 `-u`，而我（助手）已经**在同一个坑里栽了三次**：

  ① `${BREASTDG_RUNS}` —— 我一个"不设该变量"的改动，却留了三处裸引用
  ② `${PROJECT_ROOT}` —— 外层 `${X:-...}` 有保护，但**嵌在默认值内部**的
     这个没有；`PROJECT_ROOT` 是 Python 的概念，shell 里根本不存在
  ③ （同类，见 git 历史）

共同点：**都是文本级的疏漏，肉眼很难发现**，而且都要到云端真跑才炸 ——
那时已经花了租卡钱。故必须机械化。

判据
----
对每个 `.sh`：扫描全部 `${VAR}` / `${VAR[...]}` 引用，逐个判：
  · 算子里有 `:-` / `:=` / `:+` / `:?` / `-` / `=` / `+` / `?` → 有默认，跳过
  · `${VAR[0]}` 这类数组下标 → 跳过（BASH_SOURCE 等）
  · 该 VAR 在脚本前文出现过 `VAR=` 或 `VAR=$(...)` 或 `local VAR`
    ／`read VAR` ／`for VAR in` → 已定义，跳过
  · 属于 shell 内建/自动变量（BASH_SOURCE、IFS、HOME、PATH…）→ 跳过
  · 其余 → **报告**

局限（诚实说明）：本检查是**文本级**的，不跟踪控制流。
例如 `if [ x ]; then V=1; fi` 之后用 `$V` 在 `x` 为假时仍会炸，
而本检查看不出。故它是"减少一类低级错误"，不是"证明无错"。
真正确凿的验证只有一条：**在 `set -u` 下真跑一次**（即 smoke）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# shell 自带 / 环境常见 / 本仓通用
SAFE = {
    "BASH_SOURCE", "BASH_VERSION", "IFS", "HOME", "PATH", "PWD", "OLDPWD",
    "USER", "SHELL", "TMPDIR", "LANG", "LC_ALL", "RANDOM", "SECONDS",
    "HOSTNAME", "LOGNAME", "TERM", "EDITOR", "PWD", "PS1", "PPID", "BASHPID",
    "$", "?", "!", "#", "@", "*", "-", "0", "1", "2", "3", "4", "5",
}
# 这些是脚本里通常由环境提供、或我们有意让其为空的
OPTIONAL = {"PYTHONIOENCODING", "PYTHONPATH", "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS", "DRY_RUN", "NGPU", "APP_ROOT", "PYTHON_BIN",
            "BREASTDG_DATA", "BREASTDG_RUNS", "BREASTDG_OASBUD_P4"}

# 引用：${VAR} 或 ${VAR[...]}，取 VAR 名与本体的"算子部分"
REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)((?:[^}]*))?\}")
# 赋值/定义：VAR= / local VAR / read VAR / for VAR in
DEF = re.compile(
    r"(?:^|[\s;|&(])(?:local\s+|export\s+|readonly\s+|declare\s+)?"
    r"([A-Za-z_][A-Za-z0-9_]*)=", re.M)
FORIN = re.compile(r"\bfor\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\b")
READ = re.compile(r"\bread\s+(?:-\w+\s+)*([A-Za-z_][A-Za-z0-9_]*)")


def has_default(modifier: str) -> bool:
    """`:-` `:=` `:+` `:?` 或非冒号版 `-` `=` `+` `?` → 有默认值。"""
    return bool(re.search(r"[:\-+=?]", modifier or ""))


def scan(path: Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()

    # 先收集本文件里定义过的名字（不区分先后 —— 保守，减少误报）
    defined = set()
    for m in DEF.finditer(src):
        defined.add(m.group(1))
    for m in FORIN.finditer(src):
        defined.add(m.group(1))
    for m in READ.finditer(src):
        defined.add(m.group(1))

    bad = []
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue                      # 注释不算
        for m in REF.finditer(line):
            # 跳过**转义**的引用：`\${VAR}` 在 shell 里是字面量 `${VAR}`，
            # 不展开、故不会因未定义而炸。本仓第 65 行就是这种用法
            # （提示信息里写 `\${PROJECT_ROOT}` 作为说明文本）。
            if m.start() > 0 and line[m.start() - 1] == "\\":
                # 数一下前面连续的反斜杠个数：奇数=转义
                k = 0
                j = m.start() - 1
                while j >= 0 and line[j] == "\\":
                    k += 1; j -= 1
                if k % 2 == 1:
                    continue
            name, mod = m.group(1), m.group(2) or ""
            if name in SAFE or name in defined or name in OPTIONAL:
                continue
            if has_default(mod):
                continue
            # 数组下标 ${X[0]} —— mod 形如 "[0]"，已由 SAFE 覆盖 BASH_SOURCE
            if mod.startswith("["):
                continue
            bad.append((i, name))
    return bad


def main() -> int:
    targets = sorted(ROOT.glob("env/*.sh")) + sorted(ROOT.glob("scripts/*.sh"))
    total = 0
    for f in targets:
        bad = scan(f)
        if bad:
            total += len(bad)
            for ln, name in bad:
                print(f"  {f.name}:L{ln}  ${{{name}}} 无默认值且未见定义")
    if total:
        print(f"\n✗ 共 {total} 处 —— `set -u` 下会直接退出。")
        print("  处置：加 `${VAR:-默认值}`，或确认变量名拼写。")
        return 1
    print(f"✓ {len(targets)} 个 shell 脚本无未定义变量引用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
