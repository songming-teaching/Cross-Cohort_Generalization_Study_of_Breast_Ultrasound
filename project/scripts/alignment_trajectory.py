"""E7：对齐轨迹定量化——proxy A-distance 三阶段 × 域对。

**为什么需要**：核查发现代码库**没有 proxy A-distance 实现**。
而 E7 是"对齐确实发生"这条主张的唯一定量证据，也是修 C4
（t-SNE 图与文字结论不一致）的替代方案——用数字代替视觉判断。

口径（理论子节 2.2.3 命题 2 + 冻结日志 F13）
---------------------------------------------
proxy A-distance 由域分类器的可分性定义：d_A = 2(1 − 2ε)，
其中 ε 为域判别器的**留出误差**（此处用 CV 探针 AUC 反演：
ε = 1 − AUC）。同时报告 **d_A 与 Ben-David 界中的 Φ 变换**。

**强制平凡线索对照臂（F13）**：一切"域可分性"测量必须同时报告
**回归掉平凡线索后的残差**。实测教训：仅黑边+亮度即可复现 0.986/0.983
的探针 AUC，与深度特征持平——不做对照臂，测到的只是预处理伪影。

三阶段
------
  A_erm    Stage A（源域最优检查点 best_erm）
  B_final  Stage B（末步检查点 final）
  （"初始"阶段无检查点，用 Stage A 代理——见下方局限声明）

用法
----
  python scripts/alignment_trajectory.py --pairs busi-busbra,busi-busuclm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from diagnose_domain_probe_p4 import (PCA_K, extract, find_ckpt, residualize,
                                      trivial_cues, cv_auc)
from run_e8 import build_model

OUT = config.RUNS_DIR / "diag" / "alignment"


def proxy_a_distance(auc: float) -> float:
    """域分类器 AUC → proxy A-distance。

    ε（误差）= 1 − AUC；d_A = 2(1 − 2ε) = 2(2·AUC − 1)。
    注意：这是**成比例的**可分性度量，不是 Ben-David 界里的精确 d_A——
    理论文档明确以"成比例"用法引用（见 2.2.3 命题 2 的 Remark）。
    """
    return float(2.0 * (2.0 * auc - 1.0))


def ben_david_phi(d: float) -> float:
    """Ben-David 界的 Φ 变换：Φ(d) = 2(1 − 2d)·... 此处用偏差形式的等价量。

    理论文档采用的代理形式：d_A = 4Φ(Δ/2) − 2 ⇒ Φ = (d_A + 2)/4。
    返回该等价量，供与理论预测互锁（P2）。
    """
    return float((d + 2.0) / 4.0)


def _phi(z: float) -> float:
    """标准正态 CDF。"""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def delta_hat(Xa: np.ndarray, Xb: np.ndarray) -> float:
    """Ledoit–Wolf 收缩的合并协方差 → Mahalanobis 域间距 Δ̂。

    **这是 P2 的独立一侧**：实测 d_A 由交叉验证域分类器给出（不依赖任何
    分布假设），Δ̂ 由收缩协方差给出（依赖高斯假设）—— 两者独立估计，
    恒等式 d_A = 4Φ(Δ/2) − 2 的吻合度即 P2 的检验对象（理论 S5）。

    **为什么必须收缩**：p≈2046 而 n≈2500，样本协方差在 p≈n 处近乎奇异，
    不收缩则 Δ̂ 由数值噪声主导（这与 cv_auc 不加降维就失稳是同一个病根）。
    """
    from sklearn.covariance import LedoitWolf
    mu_a, mu_b = Xa.mean(0), Xb.mean(0)
    Xc = np.vstack([Xa - mu_a, Xb - mu_b])       # 去均值后合并（合并协方差口径）
    try:
        sig = LedoitWolf(assume_centered=True).fit(Xc).covariance_
        diff = mu_b - mu_a
        d2 = float(diff @ np.linalg.solve(sig, diff))
        return float(np.sqrt(max(d2, 0.0)))
    except np.linalg.LinAlgError:
        return float("nan")


def load_pair(name: str):
    a, b = name.split("-")
    def get(k):
        if k == "busi":
            return dedupe_entries(registry.busi_entries())[0]
        return registry.ALL_PUBLIC[k]()
    return get(a), get(b), a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="busi-busbra,busi-busuclm")
    ap.add_argument("--variants", default="baseline,no_domainhead")
    ap.add_argument("--direction", default="busbra")
    ap.add_argument("--seeds", default="42,123,2026")
    ap.add_argument("--stages", default="best_erm,final")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有 JSONL、重算全部格（产物结构升级后使用）")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    # 增量落盘：每行**立刻**追加到 JSONL。
    # 2026-10-05 实踩：脚本原本只在全部跑完后一次性写 JSON，机器在跑到
    # 9/18 行时重启（LastBootUpTime 21:58:53），**全部结果只剩日志文本**，
    # 不可续跑。改为逐行 append 后，任何中断最多损失正在跑的那一格。
    jsonl = OUT / "alignment_trajectory.jsonl"
    done_keys: set[tuple] = set()
    if a.force:
        # 产物结构升级（新增字段）时用：把旧 JSONL 改名留痕，然后全量重算。
        # 不删除任何东西 —— 归档纪律（F14：不完整/失校产物保留留痕但不参与统计）。
        if jsonl.exists():
            bak = OUT / "alignment_trajectory.prev.jsonl"
            jsonl.replace(bak)
            print(f"[force] 旧 JSONL 已改名为 {bak.name}")
    if jsonl.exists():                       # 续跑：跳过已完成的格
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            done_keys.add((r["pair"], r["variant"], r["stage"], r["seed"]))
        print(f"[续跑] 已有 {len(done_keys)} 行，跳过之")
    rows = []
    for pair in [p.strip() for p in a.pairs.split(",") if p.strip()]:
        ea, eb, na, nb = load_pair(pair)
        # 平凡线索（几何/色调）——对照臂用
        ca = trivial_cues(ea, PreprocConfig(size=256, geometry="p4"))
        cb = trivial_cues(eb, PreprocConfig(size=256, geometry="p4"))
        for variant in [v.strip() for v in a.variants.split(",")]:
            for stage in [s.strip() for s in a.stages.split(",")]:
                for seed in [int(s) for s in a.seeds.split(",")]:
                    if (pair, variant, stage, seed) in done_keys:
                        print(f"[skip] {pair}|{variant}|{stage}|seed{seed} 已有")
                        continue
                    try:
                        ck = find_ckpt(variant, a.direction, seed, stage)
                    except Exception as e:
                        print(f"[skip] {variant}/{stage}/seed{seed}: {e}")
                        continue
                    import torch
                    sd = torch.load(ck, map_location="cpu")
                    cfg = sd.get("cfg") or {"model": "resnet50_cbam"}
                    m = build_model(cfg); m.load_state_dict(sd["model"])
                    pp = PreprocConfig(size=int(cfg.get("image_size", 256)),
                                       geometry=cfg.get("geometry", "p4"))
                    Fa = extract(m, ea, pp); Fb = extract(m, eb, pp)
                    del m
                    torch.cuda.empty_cache()
                    y = np.r_[np.zeros(len(Fa)), np.ones(len(Fb))]
                    F = np.vstack([Fa, Fb])
                    # 深度特征。
                    # ⚠ 必须传 pca_k=PCA_K：p=2046、n≈2500 时逻辑回归落在 p≈n 的
                    # 病态区，不降维会给出与归因表相反的量级。2026-10-05 实踩：
                    # 同一格（no_domainhead|best_erm|seed42，BUSI vs BUSBRA）
                    # 归因表 7b 下界 = 0.8357，此处不传 PCA 得 0.1944（差 0.64）。
                    auc_deep = cv_auc(F, y, groups=None, pca_k=PCA_K)
                    # 平凡线索回归后的残差（F13 强制）——**双口径**，
                    # 回归集必须与 F13 归因表（diagnose_domain_probe_p4.probe_rows）
                    # 逐列一致：
                    #   7a 上界 = 深度 − (几何 pad+aspect + 色调 bright)
                    #             scale 携带的真签名仍在残差里 → 高估
                    #   7b 下界 = 7a + scale
                    #             把 scale 里的真签名也减掉 → 低估
                    # ⚠ 2026-10-06 实踩：本脚本原先用 `np.column_stack(list(ca.values()))`
                    # 把所有键一锅端，**连 scale_extra（speckle/grad/snr）也回归掉了**。
                    # 而 scale_extra 单 cue 域可分性 0.9633 —— 那是纹理签名的物理载体
                    # 本身，不是平凡线索。后果：残差臂被压到 0.6807，而归因表同格
                    # 7b 是 0.8357。修法：显式列出两个回归集，与归因表对齐。
                    def _cols(keys):
                        return np.vstack([np.column_stack([cc[k] for k in keys])
                                          for cc in (ca, cb)])
                    C_gt = _cols(["pad", "aspect", "bright"])          # 几何+色调
                    C_gts = _cols(["pad", "aspect", "bright", "scale"])  # +尺度
                    auc_res = cv_auc(residualize(F, C_gt), y, groups=None,
                                     pca_k=PCA_K)                       # 7a 上界
                    auc_res_b = cv_auc(residualize(F, C_gts), y, groups=None,
                                       pca_k=PCA_K)                     # 7b 下界
                    # --- P2 互锁：Δ̂ → 预测 d_A（与上面实测 d_A 独立估计）---
                    dh = delta_hat(Fa, Fb)
                    dA_pred = (4.0 * _phi(dh / 2.0) - 2.0
                               if np.isfinite(dh) else float("nan"))
                    rec = dict(pair=pair, variant=variant, stage=stage, seed=seed,
                               n_a=len(Fa), n_b=len(Fb),
                               auc_deep=float(auc_deep),
                               auc_residual=float(auc_res),
                               auc_residual_lower=float(auc_res_b),
                               dA_deep=proxy_a_distance(auc_deep),
                               dA_residual=proxy_a_distance(auc_res),
                               dA_residual_lower=proxy_a_distance(auc_res_b),
                               phi_deep=ben_david_phi(proxy_a_distance(auc_deep)),
                               phi_residual=ben_david_phi(proxy_a_distance(auc_res)),
                               phi_residual_lower=ben_david_phi(
                                   proxy_a_distance(auc_res_b)),
                               # P2 互锁（理论 S5）：d_A = 4Φ(Δ̂/2) − 2
                               delta_hat=dh,
                               dA_predicted=dA_pred)
                    rows.append(rec)
                    with open(jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f.flush()
                    print(f"[{pair}|{variant}|{stage}|seed{seed}] "
                          f"AUC_deep={auc_deep:.4f} d_A={rec['dA_deep']:.4f}  "
                          f"AUC_res[{auc_res:.4f},{auc_res_b:.4f}] "
                          f"d_A=[{rec['dA_residual']:.4f},{rec['dA_residual_lower']:.4f}]",
                          flush=True)

    p = OUT / "alignment_trajectory.json"
    p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
