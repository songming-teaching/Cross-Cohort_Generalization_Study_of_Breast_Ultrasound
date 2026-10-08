"""判定性诊断：GRL 是"没东西可抹"还是"抹错了轴"。

背景（《decision.txt》第一节）
-----------------------------
判别器收敛到 ln2 有两种读法，仅看判别器损失无法区分：

  读法 1（无差可抹）：ERM 特征里 BUSI<->BUS-UCLM 本就不可分
      → GRL 从第一步起就在注入无目标的梯度噪声 → 负收益是纯扰动代价
  读法 2（抹错了轴）：ERM 特征里两域原本可分，被 GRL 抹平了
      → 负收益是对齐的代价（抹掉的方向含对 BUSBRA 有用的信息），只是对齐错了轴

本脚本用**域探针**区分二者，并加一个决定性臂：

  探针 A：ERM 特征      上  BUSI ↔ BUS-UCLM   （GRL 对齐的那一对）
  探针 B：对抗后特征    上  BUSI ↔ BUS-UCLM   （看 GRL 是否真的抹平了）
  探针 C：ERM 特征      上  BUSI ↔ BUSBRA     （holdout-BUSBRA 真正要泛化的那对）

判读：
  A ≈ 0.5                       → 读法 1
  A 显著 > 0.5 且 B 回落到 ≈0.5 → 读法 2（GRL 成功，但抹错了轴）
  C > A                         → 佐证"对齐错了轴"：真正要跨的那对更可分

冻结纪律
--------
**只用 BUSI / BUSBRA / BUS-UCLM，不用任何类标签，不含 OASBUD 与 BrEaST。**
OASBUD/BrEaST 的 d_A、对齐余弦、θ̂ 一律锁箱后由 E8 统一出数（《decision.txt》第二节）。

方法
----
- 特征：冻结模型的倒数第二层（penultimate），不做任何微调；
- 域探针：**分组 5 折 CV** 的逻辑回归，fold 内先 PCA 降到 k=100 再拟合
  （p=2048 > n，不降维必然记忆；先前一次"随机特征探针"曾因记忆收敛到 loss=0，
  该陷阱在此避免）；
- 报告两种量：交叉验证 AUC（经验 d_A = 2(1−2ε)）与
  Ledoit–Wolf 收缩协方差下的 Mahalanobis 间距 Δ̂（预测 d_A = 4Φ(Δ/2) − 2），
  后者即理论 S8(6)(7) 的口径。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src")

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.models.backbones import REGISTRY
from breastdg.train.datasets import make_loader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BS = 64
N_FOLDS = 5
PCA_K = 100


# ---------------------------------------------------------------------------
# 特征导出（冻结模型，eval 模式，无 GRL）
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(model, entries, *, size: int = 256, tone_norm: bool = True) -> np.ndarray:
    model.eval().to(DEV)
    pp = PreprocConfig(size=size, tone_norm=tone_norm)
    ld = make_loader(entries, batch_size=BS, train=False, with_label=False,
                     cfg=pp, num_workers=8)
    feats = []
    for batch in ld:
        x = batch[0]
        x = (x.to(torch.float32) / 255.0 - 0.5) / 0.5 if x.dtype == torch.uint8 else x
        f = model.forward_features(x.to(DEV, non_blocking=True))
        if isinstance(f, dict):
            f = model.fuse(f["f_l"], f["f_h"])
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats).astype(np.float32)


# ---------------------------------------------------------------------------
# 域探针
# ---------------------------------------------------------------------------
def _phi(z: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def domain_probe(Xa: np.ndarray, Xb: np.ndarray, groups,
                 n_folds: int = N_FOLDS, pca_k: int = PCA_K,
                 seed: int = 0) -> dict:
    """BUSI↔另一域 的域探针。返回 CV AUC、经验 d_A、LW-Δ̂ 预测 d_A。"""
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa), int), np.ones(len(Xb), int)]
    n, p = X.shape

    # --- 经验：分组 5 折 CV（fold 内 PCA + 逻辑回归）---
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in sgkf.split(X, y, groups):
        k = min(pca_k, len(tr) - 1, p)
        pca = PCA(n_components=k, random_state=seed).fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(pca.transform(X[tr]), y[tr])
        s = clf.decision_function(pca.transform(X[te]))
        if len(set(y[te].tolist())) > 1:
            aucs.append(roc_auc_score(y[te], s))
    auc = float(np.mean(aucs))
    dA_emp = 2.0 * (1.0 - 2.0 * max(0.0, min(0.5, 1.0 - auc)))

    # --- 理论口径：Ledoit–Wolf 收缩的合并协方差 → Δ̂ → 预测 d_A ---
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    Xc = np.vstack([Xa - mu_a, Xb - mu_b])
    lw = LedoitWolf(assume_centered=True).fit(Xc)
    Sig = lw.covariance_
    diff = mu_b - mu_a
    try:
        delta2 = float(diff @ np.linalg.solve(Sig, diff))
        delta = float(np.sqrt(max(delta2, 0.0)))
    except np.linalg.LinAlgError:
        delta = float("nan")
    dA_pred = 4.0 * _phi(delta / 2.0) - 2.0 if np.isfinite(delta) else float("nan")

    return dict(n_a=len(Xa), n_b=len(Xb), p=p,
                n_groups=len(set(groups)), cv_auc=auc, cv_auc_folds=aucs,
                dA_empirical=dA_emp, delta_hat=delta, dA_predicted=dA_pred,
                shrinkage=lw.shrinkage_)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--erm", default="runs/dev_screen/no_domainhead/busbra/seed42/best.pt",
                    help="纯 ERM 检查点（λ=0，从未被 GRL 触碰）")
    ap.add_argument("--adv", default="runs/dev_screen/baseline/busbra/seed42/final.pt",
                    help="对抗训练后的检查点（末步，已充分对抗）")
    ap.add_argument("--model", default="resnet50_cbam")
    ap.add_argument("--out", default="runs/diag/domain_probe.json")
    a = ap.parse_args()

    print(f"设备 {DEV}", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)      # 先建目录（曾漏，导致 savez 失败）
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()
    print(f"BUSI {len(busi)}  BUSBRA {len(busbra)}  BUS-UCLM {len(busuclm)}", flush=True)

    results: dict = {"pairs": {}, "checkpoints": {"erm": a.erm, "adv": a.adv}}

    for tag, ckpt in (("ERM", a.erm), ("ADV", a.adv)):
        p = Path(ckpt)
        if not p.exists():
            print(f"[skip] {tag}: 找不到 {p}", flush=True)
            continue
        m = REGISTRY[a.model](n_domains=3)
        sd = torch.load(p, map_location="cpu")
        m.load_state_dict(sd["model"])
        print(f"\n--- {tag} ({p.parent.parent.name}, step={sd.get('step','?')}) ---",
              flush=True)

        F = {"busi": extract(m, busi), "busbra": extract(m, busbra),
             "busuclm": extract(m, busuclm)}
        for k, v in F.items():
            print(f"  {k:8s} {v.shape}  值域[{v.min():+.2f},{v.max():+.2f}]", flush=True)
        np.savez_compressed(out.parent / f"feats_{tag}.npz",
                            busi=F["busi"], busbra=F["busbra"], busuclm=F["busuclm"])

        # 分组：按病灶/病例，避免同一病人跨折
        for name, other, key in (("BUSI_vs_BUSUCLM", "busuclm", "a"),
                                 ("BUSI_vs_BUSBRA", "busbra", "b")):
            g = ([e["patient_id"] for e in busi] +
                 [e["patient_id"] for e in (busuclm if other == "busuclm" else busbra)])
            r = domain_probe(F["busi"], F[other], g)
            results["pairs"][f"{tag}:{name}"] = r
            print(f"  [{name}]  CV AUC={r['cv_auc']:.4f}  "
                  f"经验d_A={r['dA_empirical']:+.3f}  "
                  f"Delta_hat={r['delta_hat']:.3f}  预测d_A={r['dA_predicted']:+.3f}  "
                  f"(LW 收缩={r['shrinkage']:.4f})", flush=True)
        del m
        torch.cuda.empty_cache()

    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 判读 ----
    print("\n" + "=" * 76)
    key_a = "ERM:BUSI_vs_BUSUCLM"
    key_b = "ADV:BUSI_vs_BUSUCLM"
    key_c = "ERM:BUSI_vs_BUSBRA"
    if key_a in results["pairs"]:
        a_auc = results["pairs"][key_a]["cv_auc"]
        b_auc = results["pairs"].get(key_b, {}).get("cv_auc", float("nan"))
        c_auc = results["pairs"].get(key_c, {}).get("cv_auc", float("nan"))
        print(f"  A  ERM 特征  BUSI<->BUS-UCLM  (GRL 对齐的那对)   AUC = {a_auc:.4f}")
        if np.isfinite(b_auc):
            print(f"  B  对抗后特征 BUSI<->BUS-UCLM  (GRL 是否抹平)     AUC = {b_auc:.4f}")
        if np.isfinite(c_auc):
            print(f"  C  ERM 特征  BUSI<->BUSBRA    (真正要泛化的那对) AUC = {c_auc:.4f}")
        print()
        if a_auc < 0.55:
            print("  → 判读：读法 1（**无差可抹**）。ERM 特征里两域本就不可分，")
            print("    GRL 注入的是无目标梯度噪声，负收益=纯扰动代价。")
        elif np.isfinite(b_auc) and b_auc < a_auc - 0.05:
            print("  → 判读：读法 2（**抹错了轴**）。ERM 特征里两域可分，")
            print(f"    GRL 把它从 {a_auc:.3f} 压到 {b_auc:.3f}——对齐确实生效，但轴选错了。")
        else:
            print("  → 判读不确定：A 可分但 B 未明显回落，需检查对抗是否充分。")
        if np.isfinite(c_auc) and c_auc > a_auc + 0.05:
            print(f"    佐证：真正要跨的那对（C={c_auc:.3f}）比被对齐的那对（A={a_auc:.3f}）更可分。")
    print("=" * 76)
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
