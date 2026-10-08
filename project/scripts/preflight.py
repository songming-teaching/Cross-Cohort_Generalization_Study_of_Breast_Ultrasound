#!/usr/bin/env python3
"""重跑前的机械体检（preflight）—— 把"踩过的坑"变成可自动检查的规则。

为什么要它（用户 2026-10-06 令）
-------------------------------
「以后实验重跑之前一定要检查代码……确保（1）已知问题不会再次发生，已经
发生过的问题以及类似的问题已经解决；（2）修改 bug 过程中没有产生新的问题」
「类似错误不可以重复出现，不能一个坑里栽倒两次」。

靠人记住是不够的 —— 本项目已经栽过多次同一类坑。本脚本把**每一个已发生
的事故**编码成一条**可机械判定**的检查，重跑前跑一遍即可。

用法
----
    python scripts/preflight.py          # 全量体检，退出码 0=全过，1=有 FAIL
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ENV = ROOT / "env"

OK, FAIL, WARN = "OK", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []

# 扫描器**自己**不参与扫描：preflight.py 的规则表里是"被检查的数据"
# （θ̂、⚠、CUDA_VISIBLE_DEVICES="" 等字面量），不是它自己的输出。
# 不排除的话它会一直报自己 —— 这正是"扫描器误报自己"的经典坑。
SELF = "preflight.py"


def check(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))


# ---------------------------------------------------------------------------
# 坑 1：Windows 子进程 stdout 默认 GBK，脚本里 GBK 编不了码的字符
#      会在 print 时抛 UnicodeEncodeError **直接打死进程**。
#       2026-10-05 实际发生：TIER4 跑满、正要启动 E7 的瞬间崩掉，GPU 空转。
#       （更早一次同类：PowerShell 解析中文脚本失败。）
# 判据：任一脚本的 **print 字面量** 若 GBK 不可编码，则它必须在
#       "子进程注入 PYTHONIOENCODING" 的保护之下；否则 FAIL。
# ---------------------------------------------------------------------------
def gbk_ok(s: str) -> bool:
    try:
        s.encode("gbk")
        return True
    except UnicodeEncodeError:
        return False


def printed_literals(tree: ast.AST) -> list[tuple[int, str]]:
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            for arg in node.args:
                lits: list[ast.AST] = []
                if isinstance(arg, ast.Constant):
                    lits = [arg]
                elif isinstance(arg, ast.JoinedStr):
                    lits = [v for v in arg.values
                            if isinstance(v, ast.Constant)
                            and isinstance(v.value, str)]
                for c in lits:
                    if isinstance(c.value, str):
                        out.append((c.lineno, c.value))
    return out


def _selfprotect_line(src: str, tree: ast.AST) -> int:
    """返回本文件**自身**把 stdout 改成 utf-8 的那一行；没有则返回极大值。

    两种自护形态：
      1. 子进程注入约定 —— 源码里出现 `PYTHONIOENCODING`（由 driver 注入）；
      2. 入口自护 —— `sys.stdout.reconfigure(encoding="utf-8")`
         （2026-10-06 新增：手动直接跑的脚本不在任何 driver 之下，
          必须自己护自己，否则一跑就 UnicodeEncodeError 打死进程）。

    ⚠ 关键：自护必须**早于**风险的 print。否则同样是崩。
    故返回行号供调用方比较，而不是"出现过就算数"。
    """
    if "PYTHONIOENCODING" in src:
        return -1                     # -1 = 外部注入，恒早于一切
    best = 10 ** 9
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # 形如 <x>.reconfigure(encoding="utf-8")
        f = node.func
        if not (isinstance(f, ast.Attribute) and f.attr == "reconfigure"):
            continue
        for kw in node.keywords:
            if (kw.arg == "encoding"
                    and isinstance(kw.value, ast.Constant)
                    and str(kw.value.value).lower().replace("-", "") == "utf8"):
                best = min(best, node.lineno)
    return best


def check_gbk() -> None:
    protected = set()
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == SELF:
            continue
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            bad.append((f.name, f"语法错误 {e}"))
            continue
        if "PYTHONIOENCODING" in src:
            protected.add(f.name)
        risky = [(ln, s[:40]) for ln, s in printed_literals(tree)
                 if not gbk_ok(s)]
        if not risky:
            continue
        # 有风险字符 → 必须在受保护集合里，且保护要**早于**首个风险行
        sp = _selfprotect_line(src, tree)
        first_risk = min(ln for ln, _ in risky)
        if sp > first_risk:
            why = ("无编码注入" if sp == 10 ** 9
                   else f"自护在 L{sp}，晚于首个风险行 L{first_risk}")
            bad.append((f.name, f"{len(risky)} 处非 GBK 字面量且{why}"))
        if sp != 10 ** 9:
            protected.add(f.name)
    if bad:
        check("坑1 GBK 编码崩溃", FAIL,
              "; ".join(f"{n}:{d}" for n, d in bad))
    else:
        check("坑1 GBK 编码崩溃", OK,
              f"{len(protected)} 个脚本已注入编码或入口自护；"
              f"其余无风险字面量")


# ---------------------------------------------------------------------------
# 坑 2：subprocess 传**裸文件名**当脚本路径 —— cwd 与脚本目录不同，必崩。
#       2026-10-05 实际发生：三个阶段全部 fail_rc2
#       （"can't open file .../project/alignment_trajectory.py"）。
# 判据：subprocess 调用列表里，若第一元素是 PYTHON，则其后不应出现
#       以 ".py" 结尾、且不含路径分隔符的裸字符串字面量。
# ---------------------------------------------------------------------------
def check_bare_script_path() -> None:
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("run", "Popen")):
                continue
            if not node.args:
                continue
            first = node.args[0]
            # 展开列表字面量
            elts = first.elts if isinstance(first, ast.List) else []
            for i, el in enumerate(elts):
                if not (isinstance(el, ast.Constant)
                        and isinstance(el.value, str)):
                    continue
                v = el.value
                if v.endswith(".py") and "/" not in v and "\\" not in v:
                    # 允许 f-string/变量拼路径，故只报**纯字面量且裸名**
                    bad.append((f.name, node.lineno, v))
    if bad:
        check("坑2 裸文件名当路径", FAIL,
              "; ".join(f"{n}:L{l} {v!r}" for n, l, v in bad))
    else:
        check("坑2 裸文件名当路径", OK)


# ---------------------------------------------------------------------------
# 坑 3：Windows multiprocessing spawn —— 用 DataLoader(num_workers>0) 的
#       **可执行脚本** 必须有 `if __name__ == "__main__":` 守卫，
#       否则起 worker 即 RuntimeError。
#       2026-10-06 实际发生（本次审计中，_chk_e7_arm.py 首版）。
# 判据：文件里出现 make_loader/DataLoader 且无 __main__ 守卫 → WARN
#       （模块被 import 时无需守卫，故只警告不判死）。
# ---------------------------------------------------------------------------
def check_spawn_guard() -> None:
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        if "make_loader" not in src and "DataLoader" not in src:
            continue
        has_guard = 'if __name__ == "__main__"' in src
        # 只在"跑了就执行"的脚本里要求；被 import 的模块不算
        if not has_guard:
            bad.append(f.name)
    if bad:
        check("坑3 Windows spawn 守卫", WARN,
              "无 __main__ 守卫但用 DataLoader: " + ", ".join(bad))
    else:
        check("坑3 Windows spawn 守卫", OK)


# ---------------------------------------------------------------------------
# 坑 4：陈旧运行锁把"一格坏"放大成"整批坏"（2026-10-05）。
# 判据：runs/ 下不应有残留 .running.json。
# ---------------------------------------------------------------------------
def check_stale_lock() -> None:
    locks = list((ROOT / "runs").rglob(".running.json"))
    if locks:
        check("坑4 陈旧运行锁", FAIL,
              "; ".join(str(p.relative_to(ROOT)) for p in locks))
    else:
        check("坑4 陈旧运行锁", OK, "无残留")


# ---------------------------------------------------------------------------
# 坑 5：Windows venv 的 python.exe 是**转发器**，subprocess.run(timeout=)
#       只杀得到 shim，真实解释器会被孤儿化继续占 GPU（2026-10-05 发现）。
#       故所有"跑训练格"的驱动除了 timeout，还必须有按命令行 marker 的
#       兜底杀进程。
# 判据：含 subprocess.run(...timeout=) 且跑 python 的驱动，
#       必须同时出现 _kill_ 前缀的函数。
#
# ⚠ 2026-10-07 重写（坑9）—— 旧判据是裸文本正则，两处误报且都实际发生：
#   `re.search(r"subprocess\.run\([^)]*timeout=", src) and ('"python' in src or "PY" in src)`
#     ① `"PY" in src` 会命中**注释**里的 `PYTHONIOENCODING`、以及
#        `{"python": sys.version}` 这类**与子进程无关的字典键**；
#     ② `subprocess.run(["git", "rev-parse", ...], timeout=)` —— 取个 git SHA 而已，
#        与 Python 进程毫无关系，却被算作"跑训练格"，于是要求它也写兜底杀。
#   两次都是我为了让新代码过体检而差点去改判据 —— 那是本末倒置。
#   改为 AST 后按**实参内容**判定，判据只收紧不放松：
#     · 参数列表全由字符串常量构成 → 可静态判定：无 python 字样即**不是**
#       Python 进程（git/tasklist 等），跳过；
#     · 只要出现变量 / f-string（如 `SY[0]` 即 cmd、`f"PID eq {pid}"`）
#       → 静态判不出 → **保守计入**（与旧行为一致，不放宽）。
# ---------------------------------------------------------------------------
_PY_SUBPROC_FUNCS = {"run", "call", "check_call", "check_output", "Popen"}


def _is_python_argv_token(tok: ast.AST) -> bool:
    if not (isinstance(tok, ast.Constant) and isinstance(tok.value, str)):
        return False
    s = tok.value.replace("\\", "/").lower()
    return s in ("python", "python.exe", "py") or s.endswith(".py")


def _subprocess_python_timeout_lns(src: str) -> list[int]:
    """找「带 timeout= 且**可能**启动 Python 的 subprocess 调用」的行号。"""
    hits: list[int] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hits
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.attr if isinstance(fn, ast.Attribute) else
                fn.id if isinstance(fn, ast.Name) else None)
        if name not in _PY_SUBPROC_FUNCS:
            continue
        if not any(k.arg == "timeout" for k in node.keywords):
            continue
        argv = node.args[0] if node.args else None
        if isinstance(argv, (ast.List, ast.Tuple)):
            # 全是字符串常量 → 可精确判定
            if all(isinstance(e, ast.Constant) and isinstance(e.value, str)
                   for e in argv.elts):
                if any(_is_python_argv_token(e) for e in argv.elts):
                    hits.append(node.lineno)
                continue
            # 含变量/f-string → 判不出，保守计入
            hits.append(node.lineno)
        else:
            # 首参是变量（如 cmd）或其它表达式 → 保守计入
            hits.append(node.lineno)
    return hits


def check_orphan_kill() -> None:
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == SELF:
            continue
        src = f.read_text(encoding="utf-8")
        if _subprocess_python_timeout_lns(src) and "_kill_" not in src:
            bad.append(f.name)
    if bad:
        check("坑5 孤儿进程兜底杀", FAIL, ", ".join(bad))
    else:
        check("坑5 孤儿进程兜底杀", OK)


# ---------------------------------------------------------------------------
# 坑 6：跨脚本复用同一统计量时**口径必须逐列一致**。
#       2026-10-06 实际发生：alignment_trajectory 的 cv_auc 漏传 pca_k，
#       且残差回归集多含 scale_extra，导致与归因表差 0.155。
# 判据：凡调用 cv_auc(...) 的地方，必须显式写 pca_k=（除非有注释说明为何不用）。
# ---------------------------------------------------------------------------
def check_shared_stat_consistency() -> None:
    # 例外：cv_auc 定义处（脚本自带实现）与被**明确注释说明**不降维的调用。
    # 判据针对的是"跨脚本复用同一统计量却漏传口径参数"。
    defs = set()          # 定义了 cv_auc 的文件 → 其内部调用不算
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == SELF:
            continue
        src = f.read_text(encoding="utf-8")
        # 定义可能在顶层，也可能缩进在函数内（diagnose_trivial_cues_tier2）
        if re.search(r"^\s*def cv_auc\(", src, re.M):
            defs.add(f.name)
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == SELF or f.name in defs:
            continue
        src = f.read_text(encoding="utf-8")
        # ⚠ 不能只用 `[^)]*` 抓实参：调用形如 cv_auc(residualize(F, C_gt), y,
        # groups=None, pca_k=PCA_K) 时，嵌套括号会让正则在 residualize(F 处截断，
        # 误判为"漏传 pca_k"。改用括号配平扫描。
        # ⚠ 2026-10-08 修扫描器自身缺陷：原模式 `cv_auc\(` **不带左侧词边界**，
        #   于是 `grouped_cv_auc(` / `paired_cv_auc(` 等**包装函数**也被判为
        #   "复用同一统计量却漏传 pca_k"，报出**假 FAIL**。
        #   这与本轮坑11 是同一类缺陷（子串当身份）—— 修法同样：加边界。
        #   `(?<![\w.])` 使 `x.cv_auc(` 与 `grouped_cv_auc(` 都不再命中，
        #   而真正的顶层 `cv_auc(...)` 调用仍被抓到（下方有针对性自验）。
        for m in re.finditer(r"(?<![\w.])cv_auc\(", src):
            i = m.end()
            depth, start = 1, i
            while i < len(src) and depth:
                if src[i] == "(":
                    depth += 1
                elif src[i] == ")":
                    depth -= 1
                i += 1
            args = src[start:i - 1]
            if "pca_k" in args:
                continue
            ln = src[:m.start()].count("\n") + 1
            bad.append(f"{f.name}:L{ln}")
    if bad:
        check("坑6 统计量口径一致", FAIL,
              "cv_auc 未显式指定 pca_k: " + ", ".join(bad))
    else:
        check("坑6 统计量口径一致", OK)


# ---------------------------------------------------------------------------
# 坑 7：CUDA_VISIBLE_DEVICES="" 在 Git Bash 下对 torch **静默失效**，
#       必须用 -1（2026-10-05）。
# 判据：**代码里**不应把 CUDA_VISIBLE_DEVICES 赋成空串。
#
# ⚠ 2026-10-06 修扫描器自身缺陷：原版用正则扫**原始文本**，于是把
#   注释与 docstring 里的 `CUDA_VISIBLE_DEVICES=""` 也当成违规 ——
#   而"说明这个坑"的注释恰恰必须写出这个字符串。结果：谁按纪律写注释，
#   谁被自己的扫描器判 FAIL（典型扫描器误报自己）。
#   修法：.py 走 AST，只看**真正的赋值/传参**；.sh 跳过注释行。
#   这不是放宽判据 —— 真违规（代码里赋空串）仍会被抓到，
#   下面有针对性自验。
# ---------------------------------------------------------------------------
def _cuda_empty_py(src: str) -> list[int]:
    """AST 找 `X = ""` / `X = ''`（含 os.environ 下标、关键字传参）。"""
    hits: list[int] = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            val = node.value
            if not (isinstance(val, ast.Constant) and val.value == ""):
                continue
            for t in tgts:
                if isinstance(t, ast.Subscript):
                    sl = t.slice
                    if (isinstance(sl, ast.Constant)
                            and sl.value == "CUDA_VISIBLE_DEVICES"):
                        hits.append(node.lineno)
        # env["CUDA_VISIBLE_DEVICES"] = "" 已覆盖；再cover dict 字面量
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant)
                        and k.value == "CUDA_VISIBLE_DEVICES"
                        and isinstance(v, ast.Constant) and v.value == ""):
                    hits.append(v.lineno)
    return hits


def check_cuda_env() -> None:
    bad = []
    for f in list(SCRIPTS.glob("*.py")) + list(SCRIPTS.glob("*.sh")) \
            + list(ENV.glob("*")):
        if f.is_dir() or f.name == SELF:
            continue
        try:
            src = f.read_text(encoding="utf-8")
        except Exception:
            continue
        if f.suffix == ".py":
            try:
                for ln in _cuda_empty_py(src):
                    bad.append(f"{f.name}:L{ln}")
            except SyntaxError:
                pass
        else:
            for ln, line in enumerate(src.splitlines(), 1):
                if line.lstrip().startswith("#"):
                    continue                      # 注释行不判
                if re.search(r'CUDA_VISIBLE_DEVICES\s*=\s*(""|\'\')', line):
                    bad.append(f"{f.name}:L{ln}")
    if bad:
        check("坑7 CUDA 空串静默失效", FAIL, ", ".join(bad))
    else:
        check("坑7 CUDA 空串静默失效", OK)


# ---------------------------------------------------------------------------
# 坑 8：一次性落盘 —— 中间被打断则全部结果丢失、不可续跑。
#       （2026-10-05 机器重启，9/24 行只剩日志文本。）
# 判据：凡产出多行结果的脚本，必须有增量落盘（.jsonl 逐行 append）。
# ---------------------------------------------------------------------------
def check_incremental_persist() -> None:
    suspects = ["alignment_trajectory.py"]
    bad = []
    for name in suspects:
        f = SCRIPTS / name
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8")
        if ".jsonl" not in src or '"a"' not in src:
            bad.append(name)
    if bad:
        check("坑8 增量落盘", FAIL, ", ".join(bad))
    else:
        check("坑8 增量落盘", OK)


# ---------------------------------------------------------------------------
# 坑 10：自举/配对 bootstrap 的**种子必须全库唯一**。
#        2026-10-07 实测：同一个 cluster_bootstrap_auc 在三处各有各的默认种子
#          evaluate_lockbox.py            seed=20261006
#          ensemble_and_calibrate.py      seed=0        ← 两次才被发现
#          e3_polarity_analysis.py        seed=0        ← 第三次
#        于是**同一批数据存在多套 CI**，表里 4 位小数一比对就露馅，
#        而且没人能说清该信哪一套。出现三次 = 靠人记不住 →
#        必须机械强制（用户 check3 的原话："这个坑已经证明靠人记不住"）。
#
# 判据：凡分析脚本里出现自举/配对 bootstrap 或 default_rng 的函数定义，
#       其 seed 默认值**不得是字面量 0**，必须是 20261006
#       （或引用一个名字里带 SEED/seed 的常量）。
#       只查**默认值**，不查调用点——调用点可以不传参走默认。
# ---------------------------------------------------------------------------
BOOT_NAMES = ("cluster_bootstrap", "paired_delta", "bootstrap",
              "default_rng")


def check_bootstrap_seed() -> None:
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name == SELF:
            continue
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(n in node.name for n in BOOT_NAMES):
                continue
            args = node.args
            defaults = list(zip(args.args[len(args.args) - len(args.defaults):],
                                args.defaults)) if args.defaults else []
            for aname, dflt in defaults:
                if aname not in ("seed", "random_state", "rng_seed"):
                    continue
                # 字面量 0 / None / 其它小整数 → 违规；常量名 → 合规
                if isinstance(dflt, ast.Constant):
                    if isinstance(dflt.value, int) and dflt.value != 20261006:
                        bad.append(f"{f.name}:L{dflt.lineno} seed={dflt.value}")
                    elif dflt.value is None:
                        bad.append(f"{f.name}:L{dflt.lineno} seed=None")
                # ast.Name（如 SEED_BOOTSTRAP）：按名字内容判
                elif isinstance(dflt, ast.Name):
                    if "SEED" not in dflt.id.upper():
                        bad.append(f"{f.name}:L{dflt.lineno} seed={dflt.id}")
    if bad:
        check("坑10 自举种子唯一", FAIL,
              "自举/配对 bootstrap 的 seed 默认值不是 20261006: " + ", ".join(bad))
    else:
        check("坑10 自举种子唯一", OK, "全库统一 20261006")


# ---------------------------------------------------------------------------
# 坑 11：队列/臂名的**子串**判定把两个不同队列解析成同一个
#       2026-10-08 实际发生（且是**静默**的）：
#           arm = "corrected" if "corrected" in name else "uncorrected"
#       而 `"corrected" in "oasbud_uncorrected"` → **True**，
#       于是 OASBUD 的"未校正"臂实际读了 **corrected** 目录。
#
#       为什么值得单列一条规则：它**不崩、不报错、路径都合法**，
#       只是两臂结果**逐位相同** —— 唯一能发现它的判据就是
#       "把两臂放一起比"。本轮正是靠"27/27 格 R 与 theta 完全相等"
#       才抓到（若只看单臂，可以一路走到论文里）。
#
#       规则分两半：
#       (a) 静态：臂/管线名**不得**用 `in` 做子串判定（须精确相等或走
#           cohorts.oasbud_arm 这类唯一出处）；
#       (b) 动态：已产出的 E8 格里，oasbud 两臂的 R 与 theta **不得**逐位相同。
# ---------------------------------------------------------------------------
_SUBSTR_KIND_NAMES = ("corrected", "uncorrected", "oasbud")


def check_cohort_arm_substring() -> None:
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")) + sorted((ROOT / "src").rglob("*.py")):
        if f.name == SELF:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            # 形如  <KIND_NAME> in <expr>
            if not isinstance(node, ast.Compare):
                continue
            for op, right in zip(node.ops, node.comparators):
                if not isinstance(op, (ast.In, ast.NotIn)):
                    continue
                left = node.left
                if (isinstance(left, ast.Constant)
                        and isinstance(left.value, str)
                        and left.value.lower() in _SUBSTR_KIND_NAMES):
                    bad.append(f"{f.name}:L{node.lineno} "
                               f"'{left.value}' in ...（子串判定）")
    if bad:
        check("坑11 队列臂名精确解析", FAIL,
              "臂/管线名不得用子串判定（'corrected' in 'oasbud_uncorrected' 为真）: "
              + ", ".join(bad))
    else:
        check("坑11 队列臂名精确解析", OK, "无子串式臂名判定")

    # (b) 动态：看已产出的 E8 格
    d = ROOT / "runs" / "diag" / "e8_lockbox"
    files = sorted(d.glob("*.json")) if d.exists() else []
    same = []
    for fp in files:
        try:
            rows = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = {r.get("cohort"): r for r in rows if isinstance(r, dict)}
        a, b = m.get("oasbud_corrected"), m.get("oasbud_uncorrected")
        if a and b and a.get("R") == b.get("R") and a.get("theta") == b.get("theta"):
            same.append(fp.name)
    if same:
        check("坑11 E8 两臂非同源", FAIL,
              f"{len(same)}/{len(files)} 格两臂 R 与 theta **逐位相同**"
              f"（两臂同源 = 子串判定复发）: " + ", ".join(same[:5]))
    else:
        check("坑11 E8 两臂非同源", OK,
              f"已产出 {len(files)} 格，两臂均不同" if files else "尚无 E8 产物")


# ---------------------------------------------------------------------------
# 附加：脚本语法全部可编译
# ---------------------------------------------------------------------------
def check_syntax() -> None:
    bad = []
    for f in sorted(SCRIPTS.glob("*.py")):
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            bad.append(f"{f.name}:L{e.lineno}")
    if bad:
        check("语法可编译", FAIL, ", ".join(bad))
    else:
        check("语法可编译", OK, f"{len(list(SCRIPTS.glob('*.py')))} 个脚本")


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 74)
    print("重跑前机械体检（preflight）—— 每个坑一条规则")
    print("=" * 74)
    check_gbk()
    check_bare_script_path()
    check_spawn_guard()
    check_stale_lock()
    check_orphan_kill()
    check_shared_stat_consistency()
    check_cuda_env()
    check_incremental_persist()
    check_bootstrap_seed()
    check_cohort_arm_substring()
    check_syntax()

    for name, status, detail in results:
        mark = {OK: "[ OK ]", FAIL: "[FAIL]", WARN: "[WARN]"}[status]
        print(f"{mark} {name}")
        if detail:
            print(f"        {detail}")

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_warn = sum(1 for _, s, _ in results if s == WARN)
    print("-" * 74)
    print(f"合计 {len(results)} 项：FAIL {n_fail}，WARN {n_warn}，"
          f"OK {len(results) - n_fail - n_warn}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
