#!/usr/bin/env python3
"""临床队列**评估**：在锁箱 49 个检查点上评临床数据，并输出混淆矩阵级产物。

⚠ 本脚本属于「临床数据附加层」，为论文**完整性服务**，不是核心结论的一部分。
   完全独立：**不修改任何既有产物**，只写到 `runs/clinical_eval/`。
   删除清单见 `env/clinical_layer_files.txt`。

为什么要它（与既有 `evaluate_lockbox.py` 的关系）
------------------------------------------------
**复用，不复制**：检查点枚举、阈值加载、前向、簇自举全部 `import` 自
`evaluate_lockbox`（`find_ckpts` / `load_thresholds` / `eval_one` / `cluster_keys`）。
理由与 `cohorts.py` 的立论同源 —— 同一件事有第二份实现，就会悄悄分叉；
外部集评估的口径（尤其 F7.1 阈值纪律）绝不允许有两份。

那为什么还要单独一个脚本：**产物形态不同**。`evaluate_lockbox` 出的是
AUC/灵敏度表；本脚本要出**逐图 × 逐检查点的混淆矩阵归类**，
以便按病例、按变体、按阈值切片分析（用户明确要求）。

阈值纪律（F7.1）不变
--------------------
三个运行点：
  · `fixed_0.5`  —— 常数，允许在外部集上算；
  · `youden` / `sens90` —— **只能**来自锁箱 `thresholds.json`
    （在**源域** BUSI+BUSBRA 验证集上冻结）。本脚本**不重新拟合**任何阈值。

产物（`runs/clinical_eval/`）
-----------------------------
| 文件 | 内容 |
|---|---|
| `per_image_<name>.csv` | **逐图逐检查点**：prob + 三个阈值下的预测与混淆单元 |
| `confusion_matrices_<name>.csv` | 每个(检查点 × 阈值)的 2×2 聚合矩阵 |
| `misclassified_<name>.csv` | **仅错分**（便于快速看失败模式） |
| `summary_<name>.csv` | 每检查点一行：AUC + CI + 三点 sens/spec |
| `evalcard_<name>.json` | 审计（代码/环境/检查点 SHA256）+ 全部行 |
| `cross_cohort.csv` | 多队列时：队列 × 检查点的 AUC 对照 |

用法
----
    python scripts/clinical_eval.py --name hospA
    python scripts/clinical_eval.py --name hospA,hospB --limit 3   # 先试 3 个检查点

⚠ `--limit` 只用于**冒烟**，会写进产物的 `smoke` 字段并拒绝覆盖正式产物。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")     # 入口自护（preflight 坑1）

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "scripts"))    # 按名复用 evaluate_lockbox
sys.path.insert(0, str(PROJ / "src"))

from breastdg import config                   # noqa: E402
from breastdg.data import cohorts as _coh     # noqa: E402

import evaluate_lockbox as EV                 # noqa: E402  ← 复用，不复制

CLINICAL_ROOT = config.DATA_ROOT / "_clinical"
OUT = config.RUNS_DIR / "clinical_eval"


# ---------------------------------------------------------------------------
def load_clinical(name: str) -> list[dict]:
    """读 `_clinical/<name>/manifest.csv` → entries（与 BrEaST 同 schema）。

    `lesion_id` 优先取 manifest 的 `lesion_id`，缺则退到 `case_id`。
    ⚠ 簇键是**病灶**（F23）——一例多图必须归为一簇，否则 CI 被低估。
    """
    root = CLINICAL_ROOT / name
    man = root / "manifest.csv"
    if not man.exists():
        raise SystemExit(
            f"  ✗ 找不到 {man}\n"
            f"     先用 clinical_ingest.py 接入：\n"
            f"       python scripts/clinical_ingest.py --name {name} "
            f"--src <原始目录> --layout dirs")
    out = []
    for r in csv.DictReader(open(man, encoding="utf-8-sig")):
        img = root / "images" / r["image"]
        if not img.exists():
            raise FileNotFoundError(f"manifest 指向的图不存在：{img}")
        case = str(r.get("case_id") or Path(r["image"]).stem)
        les = str(r.get("lesion_id") or case)
        out.append(dict(dataset=f"clinical_{name}", image=str(img),
                        label=int(r["label"]),
                        patient_id=f"clin-{name}-{case}",
                        lesion_id=f"clin-{name}-{les}",
                        masks=[], birads=(r.get("birads") or None), view=None))
    return out


def cm_cell(label: int, pred: int) -> str:
    """混淆单元名。良性=0 为负类，恶性=1 为正类（与点指标口径一致）。"""
    return {(1, 1): "TP", (0, 1): "FP", (0, 0): "TN", (1, 0): "FN"}[(label, pred)]


def write_products(name: str, rows: list[dict], ents: list[dict],
                   smoke: bool) -> Path:
    """把逐检查点结果摊成：逐图表 + 混淆矩阵 + 误分类 + 汇总。"""
    OUT.mkdir(parents=True, exist_ok=True)
    ck = EV.cluster_keys(ents)            # 与自举**同一**回退链
    assert len(ck) == len(ents)
    labels = np.array([int(e["label"]) for e in ents])
    cases = [str(e["patient_id"]).split("-")[-1] for e in ents]

    # 阈值名固定三档：fixed_0.5 恒有；youden/sens90 来自源域冻结
    thr_names = ["fixed_0.5"]
    if rows and "youden" in rows[0]["runs"]:
        thr_names += ["youden", "sens90"]

    # ---------- ① 逐图逐检查点 ----------
    p_img = OUT / f"per_image_{name}.csv"
    hdr = ["variant", "seed", "image", "case_id", "lesion_id", "label", "prob"]
    for t in thr_names:
        hdr += [f"thr_{t}", f"pred_{t}", f"cell_{t}", f"correct_{t}"]
    n_written = 0
    with open(p_img, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(hdr)
        for r in rows:
            assert len(r["probs"]) == len(ents), (
                f"{r['variant']}/seed{r['seed']} 概率数 {len(r['probs'])} "
                f"≠ 条目数 {len(ents)} —— 下标错位，拒绝写出可能错配的表")
            for e, pr, c, lab, case in zip(ents, r["probs"], ck, labels, cases):
                line = [r["variant"], r["seed"], Path(e["image"]).name,
                        case, c, int(lab), f"{pr:.6f}"]
                for t in thr_names:
                    tval = r["runs"][t]["thr"]
                    pred = int(pr >= tval)
                    line += [f"{tval:.8f}", pred, cm_cell(int(lab), pred),
                             int(pred == int(lab))]
                w.writerow(line); n_written += 1
    print(f"  已写入 {p_img.name}（{n_written} 行 = "
          f"{len(rows)} 检查点 × {len(ents)} 图）")

    # ---------- ② 混淆矩阵（聚合） ----------
    p_cm = OUT / f"confusion_matrices_{name}.csv"
    with open(p_cm, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "thr_name", "thr_value", "n", "n_cases",
                    "tp", "fp", "tn", "fn", "sens", "spec", "ppv", "npv",
                    "acc", "auc"])
        for r in rows:
            for t in thr_names:
                m = r["runs"][t]
                w.writerow([r["variant"], r["seed"], t, f"{m['thr']:.8f}",
                            r["n"], r["n_clusters"], m["tp"], m["fp"],
                            m["tn"], m["fn"], f"{m['sens']:.4f}",
                            f"{m['spec']:.4f}", f"{m['ppv']:.4f}",
                            f"{m['npv']:.4f}", f"{m['acc']:.4f}",
                            f"{r['auc']:.4f}"])
    print(f"  已写入 {p_cm.name}（{len(rows)*len(thr_names)} 行）")

    # ---------- ③ 仅错分（默认看 youden；无则 fixed_0.5） ----------
    focus = "youden" if "youden" in thr_names else "fixed_0.5"
    p_mis = OUT / f"misclassified_{name}.csv"
    n_mis = 0
    with open(p_mis, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "thr_name", "image", "case_id",
                    "lesion_id", "label", "prob", "pred", "cell"])
        for r in rows:
            tval = r["runs"][focus]["thr"]
            for e, pr, c, lab, case in zip(ents, r["probs"], ck, labels, cases):
                pred = int(pr >= tval)
                if pred != int(lab):
                    w.writerow([r["variant"], r["seed"], focus,
                                Path(e["image"]).name, case, c, int(lab),
                                f"{pr:.6f}", pred, cm_cell(int(lab), pred)])
                    n_mis += 1
    print(f"  已写入 {p_mis.name}（{n_mis} 条错分 @ {focus}）")

    # ---------- ④ 汇总（每检查点一行，与锁箱表同列） ----------
    p_sum = OUT / f"summary_{name}.csv"
    with open(p_sum, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "variant", "seed", "n", "n_clusters",
                    "prevalence", "auc", "auc_ci_lo", "auc_ci_hi",
                    "sens@0.5", "spec@0.5", "sens@youden", "spec@youden",
                    "sens@sens90", "spec@sens90", "smoke"])
        for r in rows:
            g = lambda k, q: r["runs"].get(k, {}).get(q, "")
            fmt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else ""
            w.writerow([name, r["variant"], r["seed"], r["n"],
                        r["n_clusters"], f"{r['prevalence']:.4f}",
                        f"{r['auc']:.4f}", f"{r['auc_ci_lo']:.4f}",
                        f"{r['auc_ci_hi']:.4f}",
                        fmt(g("fixed_0.5", "sens")), fmt(g("fixed_0.5", "spec")),
                        fmt(g("youden", "sens")), fmt(g("youden", "spec")),
                        fmt(g("sens90", "sens")), fmt(g("sens90", "spec")),
                        int(smoke)])
    print(f"  已写入 {p_sum.name}（{len(rows)} 行）")

    # ---------- ⑤ 审计卡 ----------
    slim = [{k: v for k, v in r.items() if k not in ("probs", "labels")}
            for r in rows]
    card = {
        "cohort": name,
        "layer": "clinical_附加层（非核心结论）",
        "written_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "smoke": bool(smoke),
        "audit": {"code": EV.code_fingerprint(), "env": EV.env_fingerprint(),
                  "checkpoints": {f"{r['variant']}/seed{r['seed']}":
                                  {"path": r["ckpt"], "sha256": r["ckpt_sha256"]}
                                  for r in rows}},
        "thresholds_policy": ("youden/sens90 全部来自**源域**冻结的 "
                              "thresholds.json；外部集上只跑 fixed_0.5（F7.1）"),
        "bootstrap_seed": EV.SEED_BOOTSTRAP,
        "n_images": len(ents), "n_cases": len(set(cases)),
        "rows": slim,
    }
    p_card = OUT / f"evalcard_{name}.json"
    p_card.write_text(json.dumps(card, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    (OUT / f"evalcard_{name}.sha256").write_text(
        EV.sha256_file(p_card) + "\n", encoding="utf-8")
    print(f"  已写入 {p_card.name} / evalcard_{name}.sha256")
    return p_card


# ---------------------------------------------------------------------------
def eval_cohort(name: str, limit: int | None) -> list[dict]:
    ents = load_clinical(name)
    cohort_key = f"clinical_{name}"
    # ⚠ 口径登记用**运行时**注入，而不是改 `cohorts.py` 的源文件。
    #   理由（用户 2026-10-08 约束）：临床数据用完要删干净、还原成"只有公开
    #   队列"的版本再上 git。若把 clinical_* 写进 COHORT_KIND 的字面量里，
    #   删除时就得**改回源文件**，容易漏改、也污染 git 历史。
    #   运行时注入 → 本层删掉后 `cohorts.py` **逐字节不变**。
    #   口径与 breast/busi 完全一致（"raw"）：尺寸由**检查点**决定，与队列无关。
    _coh.COHORT_KIND[cohort_key] = "raw"

    ckpts = EV.find_ckpts()
    if limit:
        ckpts = ckpts[:limit]
    n_pos = sum(int(e["label"]) for e in ents)
    print(f"\n[{cohort_key}]  图 {len(ents)}  恶性 {n_pos} "
          f"良性 {len(ents)-n_pos}  case {len({e['patient_id'] for e in ents})}  "
          f"检查点 {len(ckpts)}")

    rows = []
    for variant, seed, ck in ckpts:
        thr = EV.load_thresholds(ck, required=True)
        r = EV.eval_one(ck, ents, cohort_key, "", thr, smoke=False)
        r["variant"] = variant; r["seed"] = seed
        rows.append(r)
        print(f"  [{variant}/seed{seed}]  AUC={r['auc']:.4f}"
              f" [{r['auc_ci_lo']:.4f},{r['auc_ci_hi']:.4f}]"
              f"  prev={r['prevalence']:.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="队列名（`_clinical/<name>/`）。可逗号分隔多个")
    ap.add_argument("--limit", type=int, default=None,
                    help="只跑前 N 个检查点（**冒烟**；会写进产物 smoke 字段）")
    a = ap.parse_args()

    names = [n.strip() for n in a.name.split(",") if n.strip()]
    if not names:
        raise SystemExit("  ✗ --name 为空")
    print("=" * 74)
    print("临床队列评估（附加层）—— 锁箱 49 检查点，阈值源自源域冻结")
    print("=" * 74)
    if a.limit:
        print(f"  ⚠ 冒烟模式：只跑前 {a.limit} 个检查点（产物标 smoke=1）")

    cross = []
    for nm in names:
        rows = eval_cohort(nm, a.limit)
        write_products(nm, rows, load_clinical(nm), smoke=bool(a.limit))
        for r in rows:
            cross.append({"cohort": nm, "variant": r["variant"],
                          "seed": r["seed"], "auc": r["auc"],
                          "auc_ci_lo": r["auc_ci_lo"],
                          "auc_ci_hi": r["auc_ci_hi"]})

    if len(names) > 1:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / "cross_cohort.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["cohort", "variant", "seed",
                                              "auc", "auc_ci_lo", "auc_ci_hi"])
            w.writeheader(); w.writerows(cross)
        print(f"\n  已写入 {p.name}（{len(cross)} 行）")
        # 逐队列均值，便于"挑一套最好的"
        by = {}
        for c in cross:
            by.setdefault(c["cohort"], []).append(c["auc"])
        print("\n  逐队列 AUC 均值（挑最好的用这个）：")
        for nm, v in sorted(by.items(), key=lambda kv: -np.mean(kv[1])):
            print(f"    {nm:16s} {np.mean(v):.4f}  (n={len(v)})")

    print(f"\n产物目录：{OUT}")
    return 0


if __name__ == "__main__":      # Windows spawn 守卫（preflight 坑3）
    sys.exit(main())
