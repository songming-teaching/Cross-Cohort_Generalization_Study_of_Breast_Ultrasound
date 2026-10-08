#!/usr/bin/env python3
"""Q-5：归因表第八行（二阶 cues）——**按预登记判据**出数并落盘。

预登记文件
----------
`Q-5_归因表第八行_锁箱前预登记.md`（SHA256 `765c986b…`，**锁箱评估之前**落盘）。
本脚本只做两件事：**算出预登记指定的量**、**按预登记的三选一规则判读**。
不得在此处新增判据 —— 判据若需改，须另开文件并写明理由。

为什么需要本脚本（而不是沿用 scaffold）
--------------------------------------
第八行需要的 5 个量里，只有 4 个此前落过盘：

| 量 | 符号 | 此前出处 | 是否落盘 |
|---|---|---|---|
| 深度特征全量 | A_full | `runs/diag/domain_probe_p4_v2.json` | ✅ |
| 深度−(几何+色调)〔上界〕 | A_res | 同上（键 `7a`） | ✅ |
| 深度−(几何+色调+尺度)〔下界〕 | A_res,lo | 同上（键 `7b`） | ✅ |
| 二阶 cues + 尺度 | A_2s | `runs/diag/trivial_cues_tier2.json`（`ALL_2nd`） | ✅ |
| **纯二阶纹理**（去尺度） | **A_2** | 只在 `_chk_tier2_texture.py` 的 **stdout** 里 | ❌ |

而 Q-5 §三 明确要求 `pure_2nd` 与 `ALL_2nd` **分列两栏**（措辞纪律禁止把含尺度的
`ALL_2nd` 笼统称为"纯纹理统计量"）。故本脚本把 A_2 落成**产物**，
并顺手核对 A_2s 与既有值一致（防止两处口径悄悄分叉）。

口径对齐（关键）
----------------
跨量比较的前提是**折法相同**，故此处 cv 与 `diagnose_domain_probe_p4.cv_auc`
取同一口径：`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)`，
分组单位 = 病灶/病人 id（F23 同源），每折 StandardScaler + LogisticRegression。

关于 `random_state=0`：这**不是自举种子**（preflight 坑10 只管 bootstrap 类），
而是**已冻结的 CV 折划分**，与既有两个 dev 表逐位一致；改它会与既有行不可比。

⚠ **本脚本是源域（dev）分析，不是锁箱臂。**
预登记 §三 写的是"设**锁箱臂**上"，但其 §一 声明的依赖是 dev 阶段的
`no_domainhead` 域探针表；而**锁箱 49 格里没有 `no_domainhead`**（也无
`baseline`），故锁箱重算**不可行**。处置与 E7 一致（裁定 3）：保持 dev、
正文写明。后果必须显式交代：本行数值即预登记 §二 用来**提出** H8 的那组数，
故它**不是独立确认**，只能是"与预登记判据一致"。

用法
----
    python scripts/q5_second_order_row.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# 入口自护（preflight 坑1）：本脚本会被**手动直接跑**，不总在 driver 之下，
# 而下面的 print 含非 GBK 字面量（⚠、≥、θ 等）—— 不护就在 Windows 控制台
# 一跑即 UnicodeEncodeError 打死进程。须**早于**一切 print。
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, "src")
PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))

from breastdg import config                             # noqa: E402
from breastdg.data import registry                      # noqa: E402
from breastdg.data.dedup import dedupe_entries          # noqa: E402
from breastdg.data.preprocess import PreprocConfig      # noqa: E402

SIZE = 256
SEED_CV = 0          # 见 docstring：冻结的折划分，非自举种子
N_FOLDS = 5
PAIRS = (("BUSI_vs_BUS-UCLM", "BUS-UCLM", "busuclm"),
         ("BUSI_vs_BUSBRA", "BUSBRA", "busbra"))


def _tier2():
    """按**文件路径**载入 tier2 模块，口径与既有产物完全一致。

    （不用 import：`scripts/` 不在包路径里，且该文件名以 `diagnose_` 开头
    也无包结构。importlib 是既有 scaffold 的同一做法。）
    """
    p = PROJ / "scripts" / "diagnose_trivial_cues_tier2.py"
    spec = importlib.util.spec_from_file_location("t2", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def grouped_cv_auc(X: np.ndarray, y: np.ndarray, groups, seed: int = SEED_CV) -> float:
    """按病灶分组的 5 折 CV AUC（口径同 diagnose_domain_probe_p4.cv_auc）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    sk = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in sk.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(X[tr]), y[tr])
        if len(set(y[te].tolist())) > 1:
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(X[te]))))
    return float(np.mean(aucs))


def main() -> int:
    t2 = _tier2()
    N = t2.NAMES
    TEX = [N.index(k) for k in ("speckle", "snr", "entropy", "grad")]   # 纯二阶纹理
    SCALE = N.index("scale")

    cfg = PreprocConfig(size=SIZE, geometry="p4")
    print("计算二阶 cues（无标签、与模型无关）...", flush=True)
    busi, _ = dedupe_entries(registry.busi_entries())
    cohorts, ents_of = {}, {}
    for nm, ents in (("BUSI", busi),
                     ("BUSBRA", registry.busbra_entries()),
                     ("BUS-UCLM", registry.busuclm_entries())):
        ents_of[nm] = ents          # 与 cues 同源同序，分组 id 取自此，避免二次载入错位
        cohorts[nm] = np.array([t2.cues2(e, cfg) for e in ents], float)
        print(f"  {nm:10s} n={len(ents):5d}", flush=True)

    # 既有的深度特征三界（dev, no_domainhead/busbra/seed42）——**不重算**
    # （重算需载模型提特征；此处只读既有产物，保证与归因表第 1–7 行同源）
    probe = json.loads((config.RUNS_DIR / "diag" / "domain_probe_p4_v2.json")
                       .read_text(encoding="utf-8"))["tables"]
    old_all2 = json.loads((config.RUNS_DIR / "diag" / "trivial_cues_tier2.json")
                          .read_text(encoding="utf-8"))

    result, verdicts = {}, {}
    for pair, other, _key in PAIRS:
        y = np.r_[np.zeros(cohorts["BUSI"].shape[0], int),
                  np.ones(cohorts[other].shape[0], int)]
        g = np.array([e["patient_id"] for e in ents_of["BUSI"]] +
                     [e["patient_id"] for e in ents_of[other]])
        def cat(cols):
            return np.concatenate([cohorts["BUSI"][:, cols],
                                   cohorts[other][:, cols]], 0)
        a_scale = grouped_cv_auc(cat([SCALE]), y, g)
        a_pure = grouped_cv_auc(cat(TEX), y, g)
        a_all2 = grouped_cv_auc(cat(list(range(len(N)))), y, g)

        tb = probe[f"ERM:{pair}"]
        a_full = tb["6 深度特征"]
        a_res = tb["7a 深度−(几何+色调) [上界]"]
        a_res_lo = tb["7b 深度−(几何+色调+尺度) [下界]"]

        # 判据（预登记 §三，互斥三选一）——**照抄，不新增**
        ok = (a_all2 >= a_res) and (a_pure >= a_res_lo)
        verdicts[pair] = ok

        result[pair] = {
            "A_full": a_full, "A_res_upper": a_res, "A_res_lower": a_res_lo,
            "A_2s_ALL_2nd": a_all2, "A_2_pure_2nd": a_pure,
            "scale_only": a_scale,
            "rule_satisfied": bool(ok),
            "criterion": "A_2s >= A_res AND A_2 >= A_res_lo",
            "ALL_2nd_prev": old_all2.get(pair, {}).get("ALL_2nd"),
        }
        print(f"\n[{pair}]")
        print(f"  A_full 深度特征            = {a_full:.4f}")
        print(f"  A_res  上界 深度−(几何+色调) = {a_res:.4f}")
        print(f"  A_res,lo 下界 深度−(+尺度)  = {a_res_lo:.4f}")
        print(f"  A_2s   ALL_2nd(含尺度)     = {a_all2:.4f}"
              f"   (既有值 {result[pair]['ALL_2nd_prev']:.4f})")
        print(f"  A_2    pure_2nd(去尺度)    = {a_pure:.4f}   ← 本轮新落盘")
        print(f"  仅 scale                  = {a_scale:.4f}")
        print(f"  A_2s ≥ A_res ? {a_all2 >= a_res}"
              f"    A_2 ≥ A_res,lo ? {a_pure >= a_res_lo}"
              f"    → 本方向{'满足' if ok else '不满足'}")

    if all(verdicts.values()):
        branch, wording = "supported", "剩余量在免模型统计量的上界之内"
    elif any(verdicts.values()):
        branch, wording = "partially_supported", "仅一个队列方向满足 —— 逐队列报告，不作合并结论"
    else:
        branch, wording = "not_supported", "残差确实超出手工线索"

    out = {
        "prereg": "Q-5_归因表第八行_锁箱前预登记.md",
        "prereg_sha256": "765c986b31df8d45c2106442a59217c73091eff6baafa0c57abd02fb3f6d530d",
        "stage": "dev",
        "stage_caveat": (
            "预登记 §三 写的是『锁箱臂上』，但锁箱 49 格中**无 no_domainhead**"
            "（也无 baseline），故锁箱重算不可行 —— 与 E7 同一原因、同一处置"
            "（保持 dev、正文写明）。**因此本行数值即预登记 §二 用来提出 H8 的那组数，"
            "不构成独立确认**，只能报告为『与预登记判据一致』。"),
        "cv": {"splitter": "StratifiedGroupKFold", "n_splits": N_FOLDS,
               "shuffle": True, "random_state": SEED_CV,
               "group": "patient_id/lesion", "scaler": "StandardScaler(per-fold)",
               "clf": "LogisticRegression(max_iter=3000)"},
        "cohort_sizes": {k: int(v.shape[0]) for k, v in cohorts.items()},
        "per_pair": result,
        "verdict": branch,
        "verdict_wording": wording,
        "note": ("A_2（pure_2nd）与 A_2s（ALL_2nd，含尺度）**必须分列两栏**，"
                 "不得把含尺度的 ALL_2nd 称为『纯纹理统计量』（预登记 §四）。"),
    }
    p = config.RUNS_DIR / "diag" / "q5_second_order_row.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'=' * 62}")
    print(f"预登记三选一判读：**{branch}** —— {wording}")
    print(f"{'=' * 62}")
    print(f"已写入 {p}")
    print("⚠ 阶段=dev（锁箱无 no_domainhead，重算不可行）："
          "本行**不是**独立确认，见产物内 stage_caveat")
    return 0


if __name__ == "__main__":      # Windows spawn 守卫
    sys.exit(main())
