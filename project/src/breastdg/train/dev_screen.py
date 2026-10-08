"""阶段 3a：开发期筛选（DomainBed 式协议）。

协议（freeze_log v2.2 / v2.8.1）
--------------------------------
- **留出的域完全不可见**（连无标签图像都不用），以保证开发期估计贴近 OASBUD 的条件
  （锁箱队列同样完全未见）。无标签辅助域恒为 BUS-UCLM。
- 主方向 = 留出 BUSBRA；次方向 = 留出 BUSI。
- 先用主方向 + 1 种子筛掉明显劣者；仅对存活方案跑满 2 方向 × 3 种子。
- 判定规则：两变体差异小于其 CI 半宽时取更简单者（simple-wins）。

用法
----
  python -m breastdg.train.dev_screen --queue tier1 --steps 10000
  python -m breastdg.train.dev_screen --config baseline --directions busbra
  python -m breastdg.train.dev_screen --summary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from .. import config
from ..data import registry
from ..data.dedup import dedupe_entries
from ..data.preprocess import PreprocConfig
from ..models.backbones import REGISTRY
from ..models.fdnet import FDNetV2
from .datasets import make_loader
from .engine import TrainConfig, evaluate, train_one

OUT = config.PROJECT_ROOT / "runs" / "dev_screen"
CSV = OUT / "results.csv"

# 固定列顺序：写入时一律按此表头，避免新旧 schema 混表导致列错位
CSV_FIELDS = ["variant", "direction", "seed", "steps", "image_size",
              "tone_norm", "render_aug", "wavelet_jitter", "scale_aug",
              "ema", "model",
              "geometry", "lambda_domain", "lambda_l", "lambda_seg",
              "src_val_auc", "best_step",
              "dev_auc_erm", "dev_auc_final", "n_dev", "minutes", "run_id"]


def _code_hash() -> str:
    """训练相关源码的内容哈希（用于检测运行期代码漂移，见 F22）。"""
    import breastdg
    root = Path(breastdg.__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


class RunLock:
    """单实例锁 + 代码漂移检测（F22）。

    动机：两次事故都源于「运行期改了 src/ 下的代码」——
    一次改 CSV 表头导致列错位，一次改 datasets.py 导致 DataLoader worker
    用新代码配旧对象而崩（worker 是新进程，会重新 import 模块）。
    本锁拒绝并发启动，并在源码哈希变化时拒绝启动。
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or (OUT / ".running.json"))

    def __enter__(self):
        if self.path.exists():
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                info = {"pid": "?", "code_hash": "?"}
            raise RuntimeError(
                f"检测到另一实例正在运行（锁文件 {self.path}）：{info}\n"
                f"若确认无实例在跑，删除该锁文件后重试。")
        OUT.mkdir(parents=True, exist_ok=True)
        self.code_hash = _code_hash()
        self.path.write_text(json.dumps(
            {"pid": os.getpid(), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
             "code_hash": self.code_hash, "cwd": str(Path.cwd())},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return self

    def __exit__(self, *exc):
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def _run_id(cfg: dict, steps: int) -> str:
    """运行唯一标识 = 配置哈希 + 步数。

    **动机（已发生的事故）**：dev_screen 原先用固定路径 `runs/dev_screen/<变体>/<方向>/seed<N>/`，
    一次 500 步的验证跑静默**覆盖**了 tier1 的 3000 步产物，导致下游诊断用错检查点
    （history.json 只剩 [250, 500]），结论一度作废。
    现改为每次运行落在独立目录，且写入前若目录已存在则拒绝执行——宁可报错，不可静默覆盖。
    """
    keep = {k: v for k, v in cfg.items()
            if k in ("model", "image_size", "tone_norm", "lambda_domain",
                     "lambda_l", "lambda_seg", "render_aug", "wavelet_jitter",
                     "scale_aug", "ema", "geometry", "batch_size", "lr",
                     # ⚠ 2026-10-07 加：**必须**含 mixstyle。
                     # 否则公平基线 DeepAll 与 MixStyle 的配置在其余键上完全相同
                     # （同为 tone_norm=False、lambda_domain=0），会算出**同一个
                     # run_id** → 第二个落盘时撞上"目录已存在且非空，拒绝覆盖"
                     # 的护栏而死。这正是该护栏设计要拦的那类事故。
                     "mixstyle", "mixstyle_crossdomain")}
    keep["steps"] = steps
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str)
                          .encode()).hexdigest()[:10]


def _append_csv(row: dict) -> None:
    """按固定表头追加；缺列补空，多列忽略。"""
    OUT.mkdir(parents=True, exist_ok=True)
    exists = CSV.exists()
    with open(CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})

# ---------------------------------------------------------------------------
# 变体定义：每条只改一个（或一组）设计选项，其余取基线
# ---------------------------------------------------------------------------
# 留一域下源域仅数百张图：bs32 时一批 ≈ 1/17 epoch。
# 3000 步 ≈ 176 epoch（约 24 min）；1 万步等于 ~590 个 epoch，是荒谬的过拟合量。
BASE = dict(model="resnet50_cbam", steps=3_000, batch_size=32, num_workers=12,
            image_size=256, tone_norm=True, lambda_domain=1.0, lambda_l=0.0,
            lambda_seg=0.0, render_aug=False, wavelet_jitter=False, ema=False,
            gate_lr=1e-2, class_weight=(1.0, 1.0), amp=True, eval_every=500)

VARIANTS: dict[str, dict] = {
    "baseline":            {},
    "no_tonenorm":         {"tone_norm": False},
    "no_domainhead":       {"lambda_domain": 0.0},
    "lam_0.3":             {"lambda_domain": 0.3},
    "lam_3.0":             {"lambda_domain": 3.0},
    "render_aug":          {"render_aug": True},
    "ema":                 {"ema": True},
    "render_aug_ema":      {"render_aug": True, "ema": True},
    # 裁定 1（2026-10-01）：scale 是归因表最强单项（单 cue AUC 0.967/0.955），
    # 几何上不可消除，故在增广层去相关。作为 dev 候选臂进 simple-wins。
    "scale_aug":           {"scale_aug": True},
    "scale_render_aug":    {"scale_aug": True, "render_aug": True},
    "fdnet_gate_hf":       {"model": "fdnet_v2", "hl_only": True,  "lambda_l": 0.0},
    "fdnet_gate_dual":     {"model": "fdnet_v2", "hl_only": False, "lambda_l": 0.1},
    # ------------------------------------------------------------------
    # 异构集成的骨干臂（2026-10-05 新增，裁定 R-3）
    # 背景：论文定位声明为「五骨干异构集成」，E1 消融矩阵写「5 骨干单独（±DANN）」，
    # 审稿意见 4 直接点名要求真消融。但 dev_screen 此前只跑过 resnet50_cbam 与
    # fdnet_v2 两个模型，swin_t / densenet121_se / efficientnet_v2_s 三个骨干的类
    # 虽在 backbones.py 中存在且能前向，却**从未被训练过**。
    # 故补齐：3 骨干 × 2 方向 × 3 种子 = 18 次 ≈ 5 h（单次实测中位 17 min）。
    # 注意：-lambda_domain 0.0 的对照臂留在消融层，本组只跑主配置配方（λ=1.0）。
    "swin_t":              {"model": "swin_t"},
    "densenet121_se":      {"model": "densenet121_se"},
    "efficientnet_v2_s":   {"model": "efficientnet_v2_s"},
}

TIER1 = ["baseline", "no_tonenorm", "no_domainhead", "lam_0.3", "lam_3.0", "render_aug"]
# 裁定 1 的尺度臂并入 tier2（待 tier1 三种子定案后跑）
TIER2 = ["ema", "render_aug_ema", "scale_aug", "scale_render_aug",
         "fdnet_gate_hf", "fdnet_gate_dual"]
# 裁定 R-3（2026-10-05）：补齐异构集成的骨干臂
TIER4 = ["swin_t", "densenet121_se", "efficientnet_v2_s"]


def _load_domains():
    """返回 {'busi': [...], 'busbra': [...], 'busuclm': [...]}（源域已去重）。"""
    busi, _ = dedupe_entries(registry.busi_entries())
    return {"busi": busi, "busbra": registry.busbra_entries(),
            "busuclm": registry.busuclm_entries()}


def _filter(entries, names: set[str] | None) -> list[dict]:
    if not names:
        return entries
    out = [e for e in entries if Path(e["image"]).name in names]
    if not out:
        raise RuntimeError("划分文件与数据不匹配")
    return out


def build_run(direction: str, cfg: dict, seed: int):
    """按 LODO 划分构造数据加载器。direction = 留出的域名。"""
    dom = _load_domains()
    path = config.SPLITS_DIR / f"lodo_holdout_{direction}.json"
    sp = json.loads(path.read_text(encoding="utf-8"))
    train_names = set(sp["train"]); val_names = set(sp["val"])

    src_all = [e for k, v in dom.items() if k != direction and k != "busuclm"
               for e in v]
    tr = _filter(src_all, train_names)
    va = _filter(src_all, val_names)
    dev_ext = dom[direction]                     # 完全不可见，仅评估
    tgt = dom["busuclm"]                         # 无标签辅助域（恒为 BUS-UCLM）
    return tr, va, dev_ext, tgt


def make_model(cfg: dict):
    if cfg["model"] == "fdnet_v2":
        if cfg.get("mixstyle"):
            raise ValueError(
                "MixStyle 只对空间骨干定义（见 models/mixstyle.with_mixstyle）；"
                "FDNet 是频域双通路，不支持。公平基线用 resnet50_cbam。")
        return FDNetV2(n_domains=3, seg_head=cfg["lambda_seg"] > 0,
                       lambda_l=cfg["lambda_l"], hl_only=cfg.get("hl_only", True))
    m = REGISTRY[cfg["model"]](n_domains=3)
    if cfg.get("mixstyle"):
        # 公平基线 MixStyle（ICLR 2021）：在骨干阶段边界插入统计量交换层。
        # 只在**训练态**生效（MixStyle.forward 自带 self.training 判断），
        # 故 evaluate()/推理完全不受影响 —— 无需在评估路径做任何特判。
        from ..models.mixstyle import with_mixstyle
        kind = _MIXSTYLE_KIND[cfg["model"]]     # 显式映射，勿用字符串切分
        m.backbone = with_mixstyle(m.backbone, kind,
                                   p=cfg.get("mixstyle_p", 0.5),
                                   alpha=cfg.get("mixstyle_alpha", 0.1))
    return m


# MixStyle 的逐骨干类型（显式表，不靠字符串切分——切分会把
# "resnet50_cbam" 误映成 "resnet50"）。Swin 不支持，见 mixstyle.with_mixstyle。
_MIXSTYLE_KIND = {
    "resnet50_cbam":     "resnet",
    "densenet121_se":    "densenet",
    "efficientnet_v2_s": "efficientnet",
}


def _make_loaders(tr, va, dev_ext, tgt, tc, seed):
    """构建四个加载器。

    worker 分配：训练 loader 吃满（它是唯一持续供数的），
    三个评估 loader 只在偶发评估时取数，给少量 worker 以免 4×N 个进程抢 32 线程。
    """
    pp = PreprocConfig(size=tc.image_size, tone_norm=tc.tone_norm,
                       geometry=tc.geometry)
    nw = tc.num_workers
    ev = max(2, nw // 3)
    return (
        make_loader(tr, batch_size=tc.batch_size, train=True,
                    with_mask=tc.lambda_seg > 0, cfg=pp, num_workers=nw,
                    seed=seed, render_aug=tc.render_aug,
                    wavelet_jitter=tc.wavelet_jitter,
                    scale_aug=tc.scale_aug),
        make_loader(va, batch_size=tc.batch_size, train=False, cfg=pp, num_workers=ev),
        make_loader(dev_ext, batch_size=tc.batch_size, train=False, cfg=pp,
                    num_workers=ev),
        make_loader(tgt, batch_size=tc.batch_size, train=True, with_label=False,
                    cfg=pp, num_workers=nw),
    )


def run_one(variant: str, direction: str, seed: int, steps: int,
            eval_every: int) -> dict:
    cfg = {**BASE, **VARIANTS[variant], "steps": steps, "eval_every": eval_every,
           "seed": seed}
    if cfg["model"] == "fdnet_v2" and cfg["image_size"] == 256:
        cfg["image_size"] = 512                  # v2.8.2：512 输入 → 子带 256
    tc = TrainConfig(**cfg)

    tr, va, dev_ext, tgt = build_run(direction, cfg, seed)
    # 运行唯一标识 + 防覆盖护栏（见 _run_id 的说明）
    rid = _run_id(cfg, steps)
    out_dir = OUT / variant / direction / f"seed{seed}" / rid
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"运行目录已存在且非空，拒绝覆盖：{out_dir}\n"
            f"（这正是不允许发生的事故：固定路径会静默覆盖既有结果。"
            f"如需重跑，请改变配置/步数，或先手动归档该目录。）")
    out_dir.mkdir(parents=True, exist_ok=True)

    ld_tr, ld_va, ld_ext, ld_tg = _make_loaders(tr, va, dev_ext, tgt, tc, seed)
    model = make_model(cfg)
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    res = train_one(tc, source_train=ld_tr, source_val=ld_va, target_train=ld_tg,
                    model=model, out_dir=out_dir, log=log)

    # 两个检查点各评一次：
    #   best_erm（按源域 AUC 选）与 final（末步，已充分对抗）
    # 二者的差距直接回答"对抗训练到底帮不帮忙"，也是 Stage A/B 的测量对象。
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_aucs: dict[str, float] = {}
    for kind, fname in (("erm", "best_erm.pt"), ("final", "final.pt")):
        p = out_dir / fname
        if not p.exists():
            continue
        m = make_model(cfg)
        ck = torch.load(p, map_location="cpu")
        m.load_state_dict(ck["model"])
        dev_aucs[kind] = float(evaluate(m.to(dev), ld_ext, dev)["auc"])
        del m

    row = dict(variant=variant, direction=direction, seed=seed,
               model=cfg["model"], steps=steps, image_size=cfg["image_size"],
               tone_norm=cfg["tone_norm"], lambda_domain=cfg["lambda_domain"],
               lambda_l=cfg["lambda_l"], lambda_seg=cfg["lambda_seg"],
               render_aug=cfg["render_aug"],
               scale_aug=cfg.get("scale_aug", False),
               src_val_auc=res["best"]["auc"],
               best_step=res["best"]["step"],
               n_dev=len(dev_ext),
               dev_auc_erm=dev_aucs.get("erm", float("nan")),
               dev_auc_final=dev_aucs.get("final", float("nan")),
               minutes=round((time.time() - t0) / 60, 2))
    r = {"auc": dev_aucs.get("final", float("nan"))}
    _append_csv(row)
    log(f"  => dev AUC   ERM {row['dev_auc_erm']:.4f} / FINAL {row['dev_auc_final']:.4f}"
        f"   (best_erm @ step {row['best_step']})   {row['minutes']} min")
    return row


def summarize(min_steps: int | None = None) -> None:
    """汇总。默认只统计**正式跑**（步数 = 该变体的最大步数），排除冒烟/标定跑。

    此前不设过滤，导致 baseline 的 300 步冒烟被混入 3000 步结果（n=2）。
    """
    if not CSV.exists():
        print("尚无结果"); return
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    rows = [r for r in rows if r.get("steps", "").strip()]
    if not rows:
        print("尚无结果"); return
    if min_steps is None:
        min_steps = max(int(r["steps"]) for r in rows)
    dropped = len([r for r in rows if int(r["steps"]) < min_steps])
    rows = [r for r in rows if int(r["steps"]) >= min_steps]

    agg: dict[tuple, dict[str, list[float]]] = {}
    for r in rows:
        key = (r["variant"], r["direction"])
        d = agg.setdefault(key, {"erm": [], "final": []})
        for k in ("erm", "final"):
            v = r.get(f"dev_auc_{k}")
            if v not in (None, "", "nan"):
                d[k].append(float(v))
    print(f"仅统计步数 ≥ {min_steps} 的正式跑"
          + (f"（忽略 {dropped} 条标定/冒烟记录）" if dropped else ""))
    print("主配置 = FINAL（固定 horizon，零选择，F16）；ERM = Stage A 参照行")
    print(f"{'variant':<18}{'dir':<8}{'n':>3}{'FINAL 均值':>11}{'±SE':>8}"
          f"{'ERM 均值':>10}{'±SE':>8}")
    for (v, d), m in sorted(agg.items()):
        e = np.array(m["erm"]) if m["erm"] else np.array([np.nan])
        f = np.array(m["final"]) if m["final"] else np.array([np.nan])
        se = lambda a: (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")
        print(f"{v:<18}{d:<8}{len(f):>3}{f.mean():>11.4f}{se(f):>8.4f}"
              f"{e.mean():>10.4f}{se(e):>8.4f}")

    # 跨方向汇总 + simple-wins（F17）
    by_v: dict[str, list[float]] = {}
    for (v, d), m in agg.items():
        by_v.setdefault(v, []).extend(m["final"])
    if by_v:
        print("\n跨方向（FINAL，2 方向 × 3 种子合并）")
        print(f"{'variant':<18}{'n':>3}{'均值':>9}{'CI95 半宽':>11}")
        stats = {}
        for v, vals in sorted(by_v.items()):
            a = np.array(vals)
            hw = 1.96 * a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else float("nan")
            stats[v] = (a.mean(), hw, len(a))
            print(f"{v:<18}{len(a):>3}{a.mean():>9.4f}{hw:>11.4f}")
        best = max(stats, key=lambda k: stats[k][0])
        tied = [k for k in stats if stats[best][0] - stats[k][0] < stats[best][1]]
        print(f"\n  均值最高：{best}；落在其 CI 半宽内的（simple-wins 候选）：{tied}")
        print("  规则：候选中取最简单者（λ=0 < 单一选项 < 组合选项）。")
    print("\n（Δ = FINAL − ERM；Δ<0 说明对抗训练损害了开发集性能）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", choices=["tier1", "tier2", "tier4", "all"], default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--directions", default="busbra",
                    help="逗号分隔：busbra,busi")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    if a.summary:
        summarize(); return 0

    if a.queue:
        variants = {"tier1": TIER1, "tier2": TIER2, "tier4": TIER4,
                    "all": TIER1 + TIER2 + TIER4}[a.queue]
    elif a.config:
        variants = [a.config]
    else:
        ap.error("需给 --queue 或 --config")

    dirs = [d.strip() for d in a.directions.split(",") if d.strip()]
    seeds = [int(s) for s in a.seeds.split(",")]
    # 已完成的 (变体, 方向, 种子, 步数) 直接跳过——续跑安全，且不触发防覆盖护栏
    done: set[tuple] = set()
    if CSV.exists():
        for r in csv.DictReader(open(CSV, encoding="utf-8")):
            if r.get("steps", "").strip():
                done.add((r["variant"], r["direction"], int(r["seed"]), int(r["steps"])))
    total = len(variants) * len(dirs) * len(seeds)
    print(f"队列 {variants}\n方向 {dirs}  种子 {seeds}  共 {total} 次运行"
          f"（其中已完成 {sum((v,d,s,a.steps) in done for v in variants for d in dirs for s in seeds)} 次将跳过）")
    print("=" * 78)
    with RunLock():                      # F22：单实例锁 + 代码漂移检测
        for i, v in enumerate(variants, 1):
            for d in dirs:
                for s in seeds:
                    if (v, d, s, a.steps) in done:
                        print(f"\n[{i}/{len(variants)}] {v} / {d} / seed{s}  —— 已完成，跳过")
                        continue
                    print(f"\n[{i}/{len(variants)}] {v} / {d} / seed{s}")
                    try:
                        run_one(v, d, s, a.steps, a.eval_every)
                    except Exception as e:
                        import traceback; traceback.print_exc()
                        print(f"  FAIL: {type(e).__name__}: {e}")
    print("\n" + "=" * 78)
    summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
