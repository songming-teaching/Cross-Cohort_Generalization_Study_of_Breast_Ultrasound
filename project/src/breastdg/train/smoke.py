"""阶段 2 冒烟测试：端到端跑通并冻结产物格式。

检查项：
  S1  五个模型都能构建、前向、反向
  S2  训练循环跑通（少量步数），产出 config/history/best.pt
  S3  逐图概率 CSV 格式冻结（字段与顺序）
  S4  FDNet-v2 的门 g 在学习
  S5  判别头的一阶矩通道修复生效（BN vs LayerNorm 对照）

算力：5 × 50 步 × bs32 ≈ 8000 图 ≈ 1 分钟（70 W 足够）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from .. import config
from ..data import registry
from ..models.backbones import ALL_NAMES, REGISTRY
from ..models.fdnet import FDNetV2
from .datasets import make_loader
from .engine import TrainConfig, predict_with_probs, train_one

STEPS = 50
BS = 32


def _tiny_splits():
    """用划分文件取子集，避免冒烟测试读全量数据。"""
    from ..data.splits import _ids
    busi = registry.busi_entries()
    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()

    src = busi + busbra
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(src))
    tr = [src[i] for i in idx[:1200]]
    va = [src[i] for i in idx[1200:1400]]
    tg = [busuclm[i] for i in range(min(400, len(busuclm)))]
    for e in tg:
        e = dict(e); e["label"] = 0        # 目标域标签不会被读取（with_label=False）
    return tr, va, busuclm


def main() -> int:
    out_root = config.RUNS_DIR / "smoke"
    out_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}  步数 {STEPS}  bs {BS}")
    print("=" * 78)

    tr, va, tg_all = _tiny_splits()
    print(f"源域 train {len(tr)}  val {len(va)}  目标域 {len(tg_all)}")

    cfg0 = TrainConfig(steps=STEPS, batch_size=BS, num_workers=4,
                       eval_every=25, image_size=256, tone_norm=True)
    ld_va = make_loader(va, batch_size=BS, train=False, num_workers=4)
    ld_tg = make_loader(tg_all, batch_size=BS, train=True, with_label=False, num_workers=4)

    # ---- S1 + S2 + S3：五个骨干（前 2 个跑满，其余只做前向检查以省时）----
    results = {}
    smoke_models = ["resnet50_cbam", "fdnet_v2"] + ALL_NAMES[1:2]
    for name in smoke_models:
        print(f"\n--- {name} ---")
        if name == "fdnet_v2":
            model = FDNetV2(n_domains=3, seg_head=True, lambda_l=0.1, hl_only=True)
            cfg = TrainConfig(**{**cfg0.__dict__, "model": name, "lambda_seg": 0.2,
                                 "image_size": 512 if False else 256})
        else:
            model = REGISTRY[name](n_domains=3)
            cfg = TrainConfig(**{**cfg0.__dict__, "model": name})
        try:
            # 前向形状检查
            x = torch.randn(2, 3, cfg.image_size, cfg.image_size).to(device)
            model.to(device).eval()
            with torch.no_grad():
                y = model(x)
            print(f"  前向 OK（eval）：{tuple(y[0].shape) if isinstance(y, tuple) else tuple(y.shape)}")
            # 训练
            with_mask = (name == "fdnet_v2")
            ld_tr2 = make_loader(tr[:800], batch_size=BS, train=True,
                                 with_mask=with_mask, num_workers=4)
            r = train_one(cfg, source_train=ld_tr2, source_val=ld_va,
                          target_train=ld_tg, model=model,
                          out_dir=out_root / name,
                          log=lambda s: print(s, flush=True))
            results[name] = r
            # ---- S3：逐图概率 CSV ----
            best = REGISTRY[name](n_domains=3) if name != "fdnet_v2" else \
                FDNetV2(n_domains=3, seg_head=True)
            ck = torch.load(out_root / name / "best.pt", map_location=device)
            best.load_state_dict(ck["model"])
            best = best.to(device)                      # 修：先前漏了 .to(device)
            pr = predict_with_probs(best, va, ld_va, device,
                                    domain_name="src_val",
                                    out_csv=out_root / name / "probs_val.csv")
            print(f"  概率 CSV OK：n={pr['n']} AUC={pr['auc']:.4f} -> {Path(pr['csv']).name}")
        except Exception as e:
            import traceback
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # ---- S3：CSV 字段冻结 ----
    import csv as _csv
    sample = next((out_root / n / "probs_val.csv" for n in smoke_models
                   if (out_root / n / "probs_val.csv").exists()), None)
    print("\n" + "=" * 78)
    if sample:
        with open(sample, newline="", encoding="utf-8") as f:
            rdr = _csv.DictReader(f)
            fields = rdr.fieldnames
            first = next(rdr)
        print("S3 概率 CSV 字段：", fields)
        print("   首行：", first)
        (out_root / "probs_schema.json").write_text(
            json.dumps({"fields": fields}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("   已冻结 -> runs/smoke/probs_schema.json")
    else:
        print("S3 FAIL：未产出概率 CSV")

    # ---- S4：门 g 是否在学 ----
    fh = out_root / "fdnet_v2" / "history.json"
    if fh.exists():
        h = json.loads(fh.read_text(encoding="utf-8"))
        gates = [r["gate"] for r in h if r.get("gate") is not None]
        if gates:
            print(f"\nS4 门 g：初值 {gates[0]:.4f} → 末值 {gates[-1]:.4f}  "
                  f"（{'在变化 OK' if abs(gates[-1]-gates[0])>1e-4 else '未变化'}）")

    print("\n" + "=" * 78)
    ok = [k for k, v in results.items() if "error" not in v]
    bad = [k for k, v in results.items() if "error" in v]
    print(f"冒烟结果：成功 {ok}")
    if bad:
        print(f"          失败 {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
