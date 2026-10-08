"""E3：OASBUD 双管线极性反转分析（全文核心实验）。

预注册分析逻辑（v14 §5.1，防"看结果讲故事"）
--------------------------------------------
1. 标签 a priori 冻结（`.mat` 内 BI-RADS 交叉表，见 labels_audit.md）；
2. 两管线共享 P4 统一几何，(a) vs (b) 对比**纯净化为渲染链对比**
   （线性包络灰度 vs 对数压缩 B 模式），几何混杂从设计上移除；
3. 报告约定：AUC 同时报**有符号值**与 **|AUC−0.5|**（无符号判别力），
   符号即"极性"；阈值一律源域冻结；
4. 病灶级 cluster bootstrap（同一病灶两切面不可拆，簇键=记录索引，F23）。

两种预注册读法（v14 §5.2，锁箱后照报不临时编）
-----------------------------------------------
(i)  统一几何后 ERM 在 OASBUD 上**仍显著失效** → 伪偏移消掉后残余失败
     只能归于声学签名，P4 锚点升级为"**最干净证据**"；
(ii) 失效**大幅减弱** → 叙事重心转为"**朴素评估中伪偏移主导**"，
     校正后残余偏移才是签名；P4 退为五队列剂量-响应。

用法
----
  python scripts/e3_polarity_analysis.py --eval-dir runs/lockbox_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config

OUT = config.RUNS_DIR / "diag" / "e3"

# 全库唯一自举种子（preflight 坑10 会机械强制：分析脚本里的自举/配对 bootstrap
# 不得再出现 seed=0 之类的本地默认）。这个坑已经出现三次
# （evaluate_lockbox → ensemble_and_calibrate → 本文件），靠人记不住。
SEED_BOOTSTRAP = 20261006


def cluster_bootstrap(y, p, cl, B=2000, seed=SEED_BOOTSTRAP, alpha=0.05):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p); cl = np.asarray(cl)
    uniq = np.unique(cl); rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cl == d) for d in draw])
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(vals)), float(np.quantile(vals, alpha/2)),
            float(np.quantile(vals, 1-alpha/2)))


def paired_delta(y, p_a, p_b, cl, B=2000, seed=SEED_BOOTSTRAP):
    """配对 bootstrap：同一病灶两管线同时抽样，得 Δ(AUC_a − AUC_b) 的 CI。"""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); cl = np.asarray(cl)
    uniq = np.unique(cl); rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cl == d) for d in draw])
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(vals)), float(np.quantile(vals, 0.025)),
            float(np.quantile(vals, 0.975)))


def load_arm(eval_dir: Path, tag: str):
    """读 probs_<tag>.csv，返回 {model: (probs, labels, lesion_ids)}（按文件名配对）。"""
    p = eval_dir / f"probs_{tag}.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = {}
    for r in rows:
        key = f"{r['variant']}/seed{r['seed']}"
        out.setdefault(key, []).append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(config.RUNS_DIR / "lockbox_eval"))
    a = ap.parse_args()
    d = Path(a.eval_dir)

    arm_a = load_arm(d, "oasbud_corrected")     # (b) 标准 B 模式重建
    arm_b = load_arm(d, "oasbud_uncorrected")   # (a) 包络重建
    if arm_a is None or arm_b is None:
        print("缺 OASBUD 评估产物。先跑**单次**全卡（P0-3：一次进程内出全部臂）：\n"
              "  python scripts/evaluate_lockbox.py --full-card\n"
              f"  （本脚本读 {d}/probs_oasbud_corrected.csv 与 "
              f"probs_oasbud_uncorrected.csv）")
        return 1

    from sklearn.metrics import roc_auc_score
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'model':28s} {'AUC(a)':>8s} {'AUC(b)':>8s} {'Δ':>8s} {'ΔCI':>18s}  极性")
    print("-" * 82)
    for key in sorted(arm_a):
        if key not in arm_b:
            continue
        ra = {r["image"]: r for r in arm_a[key]}
        rb = {r["image"]: r for r in arm_b[key]}
        common = sorted(set(ra) & set(rb))
        y = np.array([int(ra[i]["label"]) for i in common])
        pa = np.array([float(ra[i]["prob"]) for i in common])   # corrected
        pb = np.array([float(rb[i]["prob"]) for i in common])   # uncorrected
        cl = np.array([ra[i]["lesion_id"] for i in common])
        auc_a = float(roc_auc_score(y, pa))
        auc_b = float(roc_auc_score(y, pb))
        ma, lo_a, hi_a = cluster_bootstrap(y, pa, cl)
        mb, lo_b, hi_b = cluster_bootstrap(y, pb, cl)
        dm, dlo, dhi = paired_delta(y, pa, pb, cl)
        inverted = auc_b < 0.5 < auc_a
        rows.append(dict(model=key, n=len(common),
                         auc_corrected=auc_a, ci_corrected=[lo_a, hi_a],
                         auc_uncorrected=auc_b, ci_uncorrected=[lo_b, hi_b],
                         auc_abs_corrected=abs(auc_a - 0.5) + 0.5,
                         auc_abs_uncorrected=abs(auc_b - 0.5) + 0.5,
                         delta=dm, delta_ci=[dlo, dhi],
                         polarity_inverted=bool(inverted)))
        print(f"{key:28s} {auc_a:8.4f} {auc_b:8.4f} {dm:+8.4f} "
              f"[{dlo:+.4f},{dhi:+.4f}]  {'★极性反转' if inverted else ''}")

    p = OUT / "polarity_analysis.json"
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    n_inv = sum(r["polarity_inverted"] for r in rows)

    # ------------------------------------------------------------------
    # 两分支预注册结论（照报，不临时编）
    #
    # ⚠ 2026-10-08 修（本轮实测暴露，且这是**全文核心实验**的判读）：
    #   原实现有两个缺陷，合起来会把两个**互斥**分支同时打印为"成立倾向"：
    #     ① `if n_inv:` 之后又跟一个独立 `if mean_b >= 0.5:` —— 不是 `elif`，
    #        故两个分支可同时触发；
    #     ② 判据 `mean_b >= 0.5` 近乎恒真：任何平均 AUC ≥ 0.5 都会命中，
    #        而"平均 AUC ≥ 0.5"是**常态**（而非"失效减弱"的证据）。
    #        实测 mean_b = 0.6023 → 恒真。
    #   修正为**二分且互斥**的判据，直接对应 §5.2 的文字：
    #     (i)  "仍**显著**失效"  → 存在模型的 corrected AUC **CI 上界 < 0.5**
    #          （即校正之后**仍**显著低于随机 = 残余失败）
    #     (ii) "失效**大幅减弱**" → 包络下确有极性翻转（AUC<0.5），
    #          且 corrected 下**没有任何**模型 CI 上界 < 0.5
    #   两者由 `n_still_fail` 是否 >0 完全二分，故不可能同时成立。
    # ------------------------------------------------------------------
    n_still_fail = sum(1 for r in rows if r["ci_corrected"][1] < 0.5)
    n_unc_below = sum(1 for r in rows if r["auc_uncorrected"] < 0.5)
    n_corr_below = sum(1 for r in rows if r["auc_corrected"] < 0.5)
    n_delta_gt0 = sum(1 for r in rows if r["delta_ci"][0] > 0)
    b_i = n_still_fail > 0
    b_ii = (n_unc_below > 0) and (n_still_fail == 0)

    print(f"\n极性反转（corrected>0.5 而 uncorrected<0.5）  {n_inv}/{len(rows)} 个模型")
    print(f"corrected 下 CI 上界 < 0.5（=校正后仍**显著**失效）  {n_still_fail}/{len(rows)}")
    print(f"uncorrected 下 AUC < 0.5（极性翻转）              {n_unc_below}/{len(rows)}")
    print(f"corrected 下 AUC < 0.5                          {n_corr_below}/{len(rows)}")
    print(f"ΔCI 完全 > 0（校正显著有用）                     {n_delta_gt0}/{len(rows)}")
    print("预注册读法（互斥，见 §5.2）：")
    if b_i:
        print("  → 分支(i) 成立：统一几何后 ERM 在 OASBUD 上**仍显著失败**"
              " → 残余失败只能归于声学签名，P4 升级为最干净证据")
    elif b_ii:
        print("  → 分支(ii) 成立：失效**大幅减弱**"
              " → 叙事重心转为『朴素评估中伪偏移主导』，P4 退为五队列剂量-响应")
    else:
        print("  → **两分支均不成立**（既无残余显著失败，也无极性翻转）"
              " → 预注册未覆盖此情形，须显式报告为未判定，不得二选一硬套")

    verdict = dict(
        n_models=len(rows), n_polarity_inverted=n_inv,
        n_corrected_ci_upper_below_half=n_still_fail,
        n_uncorrected_auc_below_half=n_unc_below,
        n_corrected_auc_below_half=n_corr_below,
        n_delta_ci_all_positive=n_delta_gt0,
        branch_i_residual_significant_failure=b_i,
        branch_ii_failure_largely_mitigated=b_ii,
        branch_selected=("i" if b_i else ("ii" if b_ii else "neither")),
        mean_auc_corrected=float(np.mean([r["auc_corrected"] for r in rows])) if rows else None,
        mean_auc_uncorrected=float(np.mean([r["auc_uncorrected"] for r in rows])) if rows else None,
        mean_delta=float(np.mean([r["delta"] for r in rows])) if rows else None,
        rule=("(i) 存在 corrected CI 上界 < 0.5；(ii) 有极性翻转且无该情形。"
              "两者互斥。—— 判据固定在看到锁箱数之前（§5.2）"),
        note=("本文件由 e3_polarity_analysis.py 依**预注册**判据生成；"
              "两个分支互斥，不得同时声称成立。"),
    )
    vp = OUT / "polarity_verdict.json"
    vp.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")
    print(f"已写入 {vp}  （判读：分支 {verdict['branch_selected']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
