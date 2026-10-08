"""锁箱期全源重训：用全部源域（BUSI + BUSBRA）训练 5 骨干 × 3 种子。

与 `dev_screen.py` 的区别
-------------------------
`dev_screen.build_run` 只支持 **LODO**（留一域）划分——留出一个源域当开发用
外部集，这是**开发期**的协议。锁箱期配方已冻结，需要的是**全源重训**：
BUSI + BUSBRA 全部参与训练（用 `splits/source_busi_busra.json` 的
train/val 划分），BUS-UCLM 仍作无标签目标域参与对抗，OASBUD / BrEaST
**不接触**。

本脚本即为此补齐。配方 = v14 定稿主配置：
  P4 几何 + 色调前端 + λ_domain=1.0 + FDNet 用 gate_dual(hl_only=False,
  lambda_l=0.1) + FINAL 检查点（F16）+ 3000 步。

产物
----
`runs/lockbox/<variant>/seed<N>/<run_id>/{final.pt,best_erm.pt,config.json,history.json}`
其中 variant ∈ {resnet50_cbam, swin_t, densenet121_se, efficientnet_v2_s,
fdnet_gate_dual}，即论文声称的**五骨干异构集成**。

用法
----
  python scripts/retrain_lockbox.py --variants all --seeds 42,123,2026
  python scripts/retrain_lockbox.py --variants fdnet_gate_dual --seeds 42

边界
----
- **不碰 OASBUD / BrEaST**（锁箱量，须等全源重训完成、配方冻结后才单次评估）
- **不碰任何私有队列**（F9 / 记忆硬规则）
- 单实例锁：同一时间只允许一个重训进程（复用 dev_screen 的 RunLock）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.train.engine import TrainConfig, train_one
from breastdg.train.dev_screen import (
    BASE, VARIANTS, RunLock, make_model, _make_loaders, _run_id, _filter,
)
from breastdg.data.preprocess import PreprocConfig

# ---- 锁箱配方（v14 定稿，a priori 冻结，禁止按结果调整）----
LOCKBOX_VARIANTS = {
    "resnet50_cbam":     {},                                     # 主配置（spatial）
    "swin_t":            {"model": "swin_t"},
    "densenet121_se":    {"model": "densenet121_se"},
    "efficientnet_v2_s": {"model": "efficientnet_v2_s"},
    "fdnet_gate_dual":   {"model": "fdnet_v2", "hl_only": False, "lambda_l": 0.1},
    # ---- 公平基线（回应拒稿意见 5：「proposed method 只和两个基线比，
    #      且基线怎么训的没写」）----
    # 规格 a priori 冻结（见 note0.txt）：
    #   · P4 几何统一 —— 几何混杂对所有方法一视同仁，这才是公平；
    #   · **不带我们的三个组件**（色调前端 / 对抗 / 集成），它们正是被消融的对象；
    #   · 骨干与 dev baseline 同为 resnet50_cbam；3 种子；评估卡同模板。
    # ⚠ 待你确认的口径取舍（已写进总结文档）：基线关掉色调前端
    #   （tone_norm=False）会让基线**偏弱**（dev 实测 no_tonenorm = −0.0282）。
    #   保守起见建议**同时**留一个 tone_norm=True 的补充臂，锁箱后补跑。
    "deepall":           {"tone_norm": False, "lambda_domain": 0.0},
    "mixstyle":          {"tone_norm": False, "lambda_domain": 0.0,
                          "mixstyle": True},
}

# 锁箱变体名 → dev_screen.VARIANTS 的键。
#
# ⚠ 必须显式映射：**"resnet50_cbam" 在 VARIANTS 里没有同名键** ——
#   主配置在 VARIANTS 里叫 "baseline"（值 = {}，模型由 BASE 的
#   model="resnet50_cbam" 提供）。若直接写 `VARIANTS[variant]`，
#   跑 resnet50_cbam 时会抛 KeyError（干跑时已实际踩到，见工作日志 5.8）。
#   锁箱侧保留 "resnet50_cbam" 这个对论文可读的名字，故在边界处做映射。
_VARIANT_KEY = {
    "resnet50_cbam":      "baseline",
    "swin_t":             "swin_t",
    "densenet121_se":     "densenet121_se",
    "efficientnet_v2_s":  "efficientnet_v2_s",
    "fdnet_gate_dual":    "fdnet_gate_dual",
    # 基线走 baseline 键，差异全部由 LOCKBOX_VARIANTS 的覆盖提供
    "deepall":            "baseline",
    "mixstyle":           "baseline",
}

OUT = config.RUNS_DIR / "lockbox"


def build_full_source():
    """全源：BUSI + BUSBRA 按持久化划分切 train/val；BUS-UCLM 作无标签目标域。"""
    busi, _ = dedupe_entries(registry.busi_entries())
    dom = {"busi": busi, "busbra": registry.busbra_entries()}

    sp = json.loads((config.SPLITS_DIR / "source_busi_busbra.json")
                    .read_text(encoding="utf-8"))
    train_names = set(sp["train"])
    val_names = set(sp["val"])

    src_all = [e for v in dom.values() for e in v]
    tr = _filter(src_all, train_names)
    va = _filter(src_all, val_names)
    tgt = registry.busuclm_entries()            # 无标签辅助域（A3 隔离）
    return tr, va, tgt


def run_one(variant: str, seed: int, steps: int, eval_every: int) -> dict:
    key = _VARIANT_KEY[variant]        # 见 _VARIANT_KEY 的说明（必须映射）
    cfg = {**BASE, **VARIANTS[key], **LOCKBOX_VARIANTS[variant],
           "steps": steps, "eval_every": eval_every, "seed": seed}
    if cfg["model"] == "fdnet_v2" and cfg["image_size"] == 256:
        cfg["image_size"] = 512                  # 与 dev_screen 同口径（v2.8.2）
    tc = TrainConfig(**cfg)

    tr, va, tgt = build_full_source()
    rid = _run_id(cfg, steps)
    out_dir = OUT / variant / f"seed{seed}" / rid
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"运行目录已存在且非空，拒绝覆盖：{out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  源域训练 {len(tr)} / 验证 {len(va)} / 无标签目标域 {len(tgt)}",
          flush=True)

    # dev_ext 传空：锁箱重训**不**评估任何留出域（那是开发期的对象）。
    # _make_loaders 需要一个 loader，给验证集占位即可，结果不读取。
    ld_tr, ld_va, _, ld_tg = _make_loaders(tr, va, va, tgt, tc, seed)
    model = make_model(cfg)
    t0 = time.time()
    res = train_one(tc, source_train=ld_tr, source_val=ld_va, target_train=ld_tg,
                    model=model, out_dir=out_dir, log=lambda s: print(s, flush=True))
    return dict(variant=variant, seed=seed, run_id=rid,
                n_train=len(tr), n_val=len(va),
                src_val_auc=res["best"]["auc"], best_step=res["best"]["step"],
                minutes=round((time.time() - t0) / 60, 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="all",
                    help="逗号分隔，或 all（=五骨干）")
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    a = ap.parse_args()

    names = (list(LOCKBOX_VARIANTS) if a.variants == "all"
             else [v.strip() for v in a.variants.split(",") if v.strip()])
    bad = [n for n in names if n not in LOCKBOX_VARIANTS]
    if bad:
        ap.error(f"未知变体 {bad}；可用 {list(LOCKBOX_VARIANTS)}")
    seeds = [int(s) for s in a.seeds.split(",")]
    total = len(names) * len(seeds)
    print(f"锁箱全源重训：{names}\n种子 {seeds}  共 {total} 次")
    print("=" * 78)

    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    with RunLock():                      # F22：单实例 + 代码漂移检测
        for v in names:
            for s in seeds:
                print(f"\n[{v} / seed{s}]", flush=True)
                try:
                    rows.append(run_one(v, s, a.steps, a.eval_every))
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"  FAIL: {type(e).__name__}: {e}", flush=True)

    p = OUT / "summary.json"
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                 encoding="utf-8")
    print(f"\n已写入 {p}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
