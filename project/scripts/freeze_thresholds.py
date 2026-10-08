"""源域阈值冻结 —— 锁箱评估的前置步骤（F7.1 的强制配套）。

为什么必须有这个脚本
--------------------
锁箱纪律里**唯一不能犯**的错是"在外部集上调阈值"。此前
`evaluate_lockbox.py` 在**外部集自己身上**算 Youden / sens90 —— 那是
标准的锁箱违规：一旦被审稿人发现，整张评估卡作废。

修法即本脚本：**在源域验证集上**把运行点算出来、写盘、冻结，
锁箱评估只**读取**它。外部集上从此只有固定的 `0.5` 与源域冻结的两个阈值。

边界（重要）
------------
本脚本只看**源域**（BUSI + BUSBRA 的 val 划分），**完全不接触**
OASBUD / BrEaST / 任何锁箱队列。故它可以在锁箱开封**之前**任意次重跑。

产物
----
与每个检查点**同目录**的 `thresholds.json`：
    {"thr_youden": …, "thr_sens90": …, "ckpt_sha256": …,
     "n_val": …, "seed": 20261006, "frozen_at": "...", ...}
`evaluate_lockbox.py` 读不到它就**拒绝执行**（非零退出），不静默回退。

用法
----
  python scripts/freeze_thresholds.py                    # 全部锁箱检查点
  python scripts/freeze_thresholds.py --ckpt runs/.../final.pt --out /tmp/t.json
  python scripts/freeze_thresholds.py --dry-run          # 只列出将写哪些文件
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config                                     # noqa: E402
from breastdg.data.preprocess import PreprocConfig              # noqa: E402
from breastdg.train.datasets import make_loader                 # noqa: E402
from breastdg.train.engine import to_device_uint8               # noqa: E402
from evaluate_lockbox import (find_ckpts, sha256_file, youden_threshold,  # noqa: E402
                              sens_at_threshold)
from run_e8 import build_model                                  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED_BOOTSTRAP = 20261006       # 与 evaluate_lockbox 同一冻结种子


def source_val_entries() -> list[dict]:
    """源域验证集（BUSI + BUSBRA 的 val 划分）。**不含任何锁箱队列。**

    ⚠ 关于「val 是否 in-sample」（check3 要求先答，2026-10-07 已实测）：
      · 划分文件 `splits/source_busi_busbra.json`：train 2123 / val 381；
      · `build_full_source()` **实际载入** = 2123 / 381，与声明一致；
      · train ∩ val = **0**（image 路径与病灶均无交集；患者级 1439 / 254）；
      · `train_one` 只在 engine L258 用 `evaluate(source_val)` 做早停/选模，
        无梯度、无 concat、无 refit —— **val 从未进入训练**。
    「retrain」指的是**按锁箱纪律从头重训**，不是 train+val 合并。
    故阈值冻结合法（非 in-sample）。
    另：`find_ckpts` 只收 `*/final.pt`（末步），**不收** val 选出的
    `best_erm.pt` —— 评估与冻结用的都不是 val-selected 检查点。
    """
    from retrain_lockbox import build_full_source
    _, va, _ = build_full_source()
    return va


def source_preproc(saved_cfg: dict) -> PreprocConfig:
    """源域用**训练时的**预处理（原图 → 完整 P4 + 按 cfg 的色调前端）。"""
    return PreprocConfig(size=int(saved_cfg.get("image_size", 256)),
                         tone_norm=saved_cfg.get("tone_norm", True),
                         geometry=saved_cfg.get("geometry", "p4"))


@torch.no_grad()
def source_val_probs(ckpt: Path, entries: list[dict], batch_size: int = 64):
    """在**源域验证集**上跑一遍，返回 (probs, labels)。"""
    sd = torch.load(ckpt, map_location="cpu")
    saved_cfg = sd.get("cfg")
    if not saved_cfg:
        raise RuntimeError(f"{ckpt} 缺 cfg —— 无法确定预处理口径，拒绝出数")
    model = build_model(saved_cfg)
    model.load_state_dict(sd["model"])
    pp = source_preproc(saved_cfg)
    ld = make_loader(entries, batch_size=batch_size, train=False, with_label=True,
                     cfg=pp, num_workers=6)
    model.eval().to(DEV)
    ps, ys = [], []
    for batch in ld:
        x, y, _, _, _ = batch
        logits = model(to_device_uint8(x.to(DEV)))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(ps), np.concatenate(ys)


def freeze_one(ckpt: Path, entries: list[dict], out: Path | None = None,
               guard_existing: bool = True) -> dict:
    t0 = time.time()
    # ⚠ 2026-10-07 加（check3 P2①）：正向单次护栏是 evaluate_lockbox 的
    #   `exit 3`（评估卡已存在即拒绝）。但那只在**评估端**；若在评估之后
    #   重跑本脚本，就会**覆盖 thresholds.json** —— 于是"阈值先于锁箱"这个
    #   可审计性主张只剩自觉得到保障（评估卡里记的 ckpt_sha256 会与新的阈值
    #   不匹配，或更糟：恰巧匹配而阈值已被改写）。
    #   故这里加事后保护：同目录已有 thresholds.json 且相邻评估卡已存在
    #   → 拒绝覆盖，除非显式 --force-refreeze 并写明理由。
    out_p = out or (ckpt.parent / "thresholds.json")
    if guard_existing and out is None and out_p.exists():
        from evaluate_lockbox import OUT as EVAL_OUT
        card = EVAL_OUT / "evalcard_oasbud_corrected.json"
        if card.exists():
            raise RuntimeError(
                f"拒绝覆盖：{out_p} 已存在，且评估卡已落盘（{card.name}）。\n"
                f"  『阈值先于锁箱』要求阈值在评估**之前**冻结、之后不再改动。\n"
                f"  若确需重冻（例如换了检查点），请先归档评估卡并说明原因，\n"
                f"  再加 --force-refreeze。")
    p, y = source_val_probs(ckpt, entries)
    if len(set(y.tolist())) < 2:
        raise RuntimeError(f"源域验证集只有一个类别（n={len(y)}），无法定阈值")
    rec = dict(
        ckpt=str(ckpt),
        ckpt_sha256=sha256_file(ckpt),
        n_val=int(len(y)),
        prevalence_val=float(np.mean(y)),
        # 两个运行点，**全部在源域验证集上选定**
        thr_youden=youden_threshold(y, p),
        thr_sens90=sens_at_threshold(y, p, 0.90),
        sens90_target=0.90,
        auc_val=float(__import__("sklearn.metrics", fromlist=["x"])
                      .roc_auc_score(y, p)),
        seed=SEED_BOOTSTRAP,
        frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        frozen_by="scripts/freeze_thresholds.py",
        note=("阈值由**源域验证集**选定后冻结；锁箱评估只读本文件，"
              "绝不在外部集上调阈（F7.1）"),
        minutes=round((time.time() - t0) / 60, 3),
    )
    out = out or (ckpt.parent / "thresholds.json")
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ {ckpt.parent.name}/{ckpt.name}  "
          f"youden={rec['thr_youden']:.4f} sens90={rec['thr_sens90']:.4f}  "
          f"src_val AUC={rec['auc_val']:.4f}  → {out.name}", flush=True)
    return rec


def dump_source_probs(ckpts, entries: list[dict], out_dir: Path) -> int:
    """把每个检查点在**源域验证集**上的概率落一张 CSV（P0 的配套产物）。

    用途：`ensemble_and_calibrate.py --fit-source` 的融合权重必须在源域上拟合，
    而源域概率此前**没有任何脚本会落盘** —— 于是那个脚本只能拿外部集概率去
    拟合（就是这次修的 P0）。本函数补上这条缺链。

    格式与 `evaluate_lockbox.write_arm_artifacts` 的 `probs_*.csv` **同构**，
    故 `ensemble_and_calibrate.load_probs_csv` 可直接读，无需第二套解析。
    label 列写**源域真实标签**（这是允许的：源域标签从来不是被隔离对象，
    被隔离的是**目标域/外部集**标签 —— A3）。
    """
    import csv as _csv
    from evaluate_lockbox import cluster_keys
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / "probs_source_val.csv"
    keys = cluster_keys(entries)
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["variant", "seed", "image", "prob", "label", "lesion_id"])
        for v, s, ck in ckpts:
            p, y = source_val_probs(ck, entries)
            if len(p) != len(entries):
                raise RuntimeError(
                    f"{ck} 概率数 {len(p)} ≠ 源域条目数 {len(entries)} —— 下标错位")
            for e, pr, yy, k in zip(entries, p, y, keys):
                w.writerow([v, s, Path(e["image"]).name, f"{pr:.6f}",
                            int(yy), k])
            print(f"  ✓ {v}/seed{s}  源域 n={len(y)}  AUC={_auc(y, p):.4f}",
                  flush=True)
    print(f"\n✓ 已写入 {csv_p}")
    print("  下一步（源域拟合）：python scripts/ensemble_and_calibrate.py --fit-source")
    return 0


def _auc(y, p) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(np.asarray(y).tolist())) > 1 \
        else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="只冻结单个检查点")
    ap.add_argument("--out", default=None, help="配合 --ckpt：写到指定路径")
    ap.add_argument("--dump-source-probs", action="store_true",
                    help="**不写阈值**，只把源域验证集概率落 probs_source_val.csv"
                         "（供 ensemble_and_calibrate.py --fit-source 用）")
    ap.add_argument("--force-refreeze", action="store_true",
                    help="允许在评估卡已存在时覆盖 thresholds.json"
                         "（默认拒绝；见 freeze_one 的说明）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只列出将写哪些 thresholds.json，不跑模型")
    a = ap.parse_args()

    # P2①：`--out` 必须与 `--ckpt` 配对 —— 否则 35 个检查点的阈值会
    # 全部写进同一个文件、互相覆盖，只剩最后一个（静默丢 34 份）。
    if a.out and not a.ckpt:
        ap.error("--out 必须与 --ckpt 一起用：\n"
                 "  否则每个检查点都会写同一个文件，后者覆盖前者，\n"
                 "  最后只剩一个 —— 其余 34 份阈值静默丢失。")

    if a.ckpt:
        ck = Path(a.ckpt)
        if not ck.is_absolute():
            ck = config.PROJECT_ROOT / ck
        ckpts = [("smoke", 0, ck)]
    else:
        ckpts = find_ckpts()
        if not ckpts:
            print("未找到锁箱检查点（先跑 scripts/_lockbox_driver.py）")
            return 1

    ents = source_val_entries()

    if a.dump_source_probs:
        print(f"落盘源域概率：{len(ckpts)} 个检查点")
        print("  边界：只用源域（BUSI+BUSBRA val），不接触 OASBUD / BrEaST")
        print("=" * 78)
        from evaluate_lockbox import OUT as EVAL_OUT
        return dump_source_probs(ckpts, ents, EVAL_OUT.parent / "source_val")

    print(f"源域阈值冻结：{len(ckpts)} 个检查点")
    print("  边界：只用源域（BUSI+BUSBRA val），不接触 OASBUD / BrEaST")
    if a.dry_run:
        for v, s, ck in ckpts:
            tgt = Path(a.out) if a.out else (ck.parent / "thresholds.json")
            print(f"  [dry] {v}/seed{s}  → {tgt}")
        return 0

    print(f"  源域验证集 n={len(ents)}")
    print("=" * 78)
    ok = 0
    for v, s, ck in ckpts:
        print(f"[{v}/seed{s}]", flush=True)
        try:
            freeze_one(ck, ents, Path(a.out) if a.out else None,
                       guard_existing=not a.force_refreeze)
            ok += 1
        except Exception as e:                       # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"  ✗ FAIL {type(e).__name__}: {e}", flush=True)
    print("=" * 78)
    print(f"完成 {ok}/{len(ckpts)}")
    return 0 if ok == len(ckpts) else 1


if __name__ == "__main__":
    sys.exit(main())
