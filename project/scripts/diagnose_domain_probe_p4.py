"""域探针诊断（P4 几何版）：区分"没东西可抹"与"抹错了轴"，并出 F13 五行归因表。

冻结依据：docs/freeze_log.md F13。

为什么必须带平凡线索对照臂
--------------------------
letterbox 下，域探针在冻结特征上给出 AUC 0.987，但**仅用平凡线索**（黑边、宽高比、
亮度统计）就能复现 0.9862 —— 探针测到的"域差"几乎全部由预处理引入的几何/色调线索
解释，不是高频纹理签名。P4 消掉了几何线索，但**色调线索仍在**（tone_norm 在几何之前
施加，几何改动管不到它）。

因此本脚本固定输出五行归因表：

  1 仅黑边项（P4 下 = 几何填充占比）
  2 仅长宽比项（内容物理宽高比）
  3 仅亮度统计项（均值/标准差/分位数）
  4 全部平凡线索
  5 深度特征

读法：
  · P4 下第 1 行与第 2 行应近似相等 —— 验证"残留只由长宽比承载"；
  · **第 5 行与第 4 行之差，才是纹理签名证据的上限**（F13）。

并附回归对照臂：把平凡线索从深度特征中线性回归掉后，再测残差的域可分性。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.geometry import plan, pad_decomposition
from breastdg.data.preprocess import PreprocConfig, preprocess_image
from breastdg.models.backbones import REGISTRY
from breastdg.train.datasets import make_loader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BS = 64
N_FOLDS = 5
PCA_K = 100
SIZE = 256


# ---------------------------------------------------------------------------
def find_ckpt(variant: str, direction: str, seed: int, which: str = "best_erm") -> Path:
    """按 run_id 定位检查点（取最新一次运行）。

    ⚠ 2026-10-08 加：**锁箱布局支持**。
      dev_screen : runs/dev_screen/<variant>/<direction>/seed<N>/<run_id>/<file>
      lockbox    : runs/lockbox/<variant>/seed<N>/<run_id>/<file>   ← **无 direction**
    为什么需要：E7（本模块 = 域探针）要出**锁箱臂**结果，而锁箱 49 格用的是
      run_e8 那套布局。`run_e8.find_ckpt` 已加 `set_runs_root`，但**本函数是
      另一份独立定义**（E7 自己持有的），不受其影响 —— 这正是"改一处漏一处"
      的典型，故在此单独补齐。
    为什么用环境变量而非 setter：本函数被 `alignment_trajectory` /
      `_chk_e7_arm` 等 `from ... import find_ckpt` **按名引入**，加参数要改所有
      调用点；且 E7 主程序根本不 import run_e8，装 setter 也够不着。环境变量
      对"按名引入"是透明的，且默认缺省 = 旧行为完全不变。
    取值：BREASTDG_CKPT_ROOT 指向锁箱根（如 runs/lockbox）。
    """
    import os
    _root = os.environ.get("BREASTDG_CKPT_ROOT")
    if _root:
        base = Path(_root) / variant / f"seed{seed}"      # 锁箱：无 direction 层
    else:
        base = (config.PROJECT_ROOT / "runs" / "dev_screen" / variant / direction
                / f"seed{seed}")
    cands = sorted(base.glob(f"*/{which}.pt"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(
            f"找不到 {base}/*/{which}.pt"
            + ("（BREASTDG_CKPT_ROOT 已设，走锁箱布局）" if _root else ""))
    if len(cands) > 1 and _root:
        # 锁箱布局下出处必须唯一：多个 run 目录含同名文件 → 拒绝静默挑最新
        names = ("\n" + "    ").join(str(c.parent.name) for c in cands)
        raise RuntimeError(
            f"出处歧义：{base} 下有 {len(cands)} 个 {which}.pt：\n    {names}\n"
            "  拒绝静默挑最新（见 evaluate_lockbox 的同类护栏）。")
    return cands[-1]


# ---------------------------------------------------------------------------
@torch.no_grad()
def extract(model, entries, cfg: PreprocConfig) -> np.ndarray:
    model.eval().to(DEV)
    ld = make_loader(entries, batch_size=BS, train=False, with_label=False,
                     cfg=cfg, num_workers=8)
    out = []
    for batch in ld:
        x = batch[0]
        x = (x.to(torch.float32) / 255.0 - 0.5) / 0.5
        f = model.forward_features(x.to(DEV, non_blocking=True))
        if isinstance(f, dict):
            f = model.fuse(f["f_l"], f["f_h"])
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def trivial_cues(entries, cfg: PreprocConfig) -> dict[str, np.ndarray]:
    """P4 管线下的平凡线索（与模型无关），分三族（裁定 1/2，2026-10-01）。

      几何族：pad（几何填充）、aspect（内容长宽比）—— 二者同源，AUC 应近似相等
      色调族：bright（均值/标准差/分位数）
      尺度族：scale（内容缩放比）—— **几何上不可消除**，只能测量与披露；
              单 cue 域可分性 0.967/0.955，是归因表最强单项。
              附 speckle / grad / snr 作佐证（均无标签，可用于 BUS-UCLM 对）。

    P4 下 pad 与 aspect 同源（pad = 1 − 0.88²/max(r,1/r)），故第 1 行与第 2 行
    应给出近似相等的 AUC —— 这正是 F13 要求验证的"残留只由长宽比承载"。
    """
    pad, asp, br, sc, extra = [], [], [], [], []
    for e in entries:
        pi = preprocess_image(e["image"], cfg, as_rgb=False)
        g = _gray_h(e["image"])
        p = plan(g, size=cfg.size)
        pad.append(pad_decomposition(pi, p)["pad_geom"])
        asp.append(_aspect(g))
        c = pi.astype(np.float64)
        # 内容区（去填充），避免填充主导统计
        nh = max(1, int(round((p.bbox[1] - p.bbox[0]) * p.scale)))
        nw = max(1, int(round((p.bbox[3] - p.bbox[2]) * p.scale)))
        cc = c[p.pad_h:p.pad_h + nh, p.pad_w:p.pad_w + nw]
        if cc.size == 0:
            cc = c
        br.append([float(pi.mean()), float(pi.std()),
                   float(np.percentile(pi, 5)), float(np.percentile(pi, 50)),
                   float(np.percentile(pi, 95))])
        sc.append(float(p.scale))
        lap = np.zeros_like(cc)
        lap[1:-1, 1:-1] = (4 * cc[1:-1, 1:-1] - cc[:-2, 1:-1] - cc[2:, 1:-1]
                           - cc[1:-1, :-2] - cc[1:-1, 2:])
        gy, gx = np.gradient(cc)
        extra.append([float(lap.var()), float(np.sqrt(gx ** 2 + gy ** 2).mean()),
                      float(cc.mean() / (cc.std() + 1e-6))])
    return dict(pad=np.array(pad, float), aspect=np.array(asp, float),
                bright=np.array(br, float), scale=np.array(sc, float),
                scale_extra=np.array(extra, float))


SCALE_EXTRA_NAMES = ["speckle(Laplacian var)", "grad", "snr"]


def _gray_h(path: str) -> np.ndarray:
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"))
    return np.rot90(a, 1) if a.shape[0] > a.shape[1] else a


def _aspect(g: np.ndarray) -> float:
    from breastdg.data.geometry import content_bbox
    r0, r1, c0, c1 = content_bbox(g)
    return float((c1 - c0) / max(r1 - r0, 1))


# ---------------------------------------------------------------------------
def cv_auc(X: np.ndarray, y: np.ndarray, groups, seed: int = 0,
           pca_k: int | None = None) -> float:
    """分层 K 折 AUC。

    ⚠ groups=None 时必须退回 **StratifiedKFold**：
      `StratifiedGroupKFold` 收到 groups=None 会把**全部样本当作同一个组**
      （内部用 zeros 填充），于是 n_groups=1 < n_splits=5 直接抛
      `ValueError: Cannot have number of splits n_splits=5 greater than
      the number of groups: 1.`
      2026-10-05 实际踩到 —— E7 首次运行时在此崩溃。
      本分支此前必然崩溃，故不存在"依赖该行为的历史结果"，
      改动是严格增补的。
    """
    if groups is None:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=seed)
        folds = splitter.split(X, y)
    else:
        splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                        random_state=seed)
        folds = splitter.split(X, y, groups)
    aucs = []
    for tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        if pca_k is not None:
            k = min(pca_k, len(tr) - 1, X.shape[1])
            pc = PCA(n_components=k, random_state=seed).fit(Xtr)
            Xtr, Xte = pc.transform(Xtr), pc.transform(Xte)
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), y[tr])
        if len(set(y[te].tolist())) > 1:
            aucs.append(roc_auc_score(y[te], clf.decision_function(sc.transform(Xte))))
    return float(np.mean(aucs))


def residualize(F: np.ndarray, C: np.ndarray, seed: int = 0) -> np.ndarray:
    """把平凡线索从深度特征中线性回归掉，返回残差。"""
    sc = StandardScaler().fit(C)
    Cs = sc.transform(C)
    R = F - Ridge(alpha=1.0).fit(Cs, F).predict(Cs)
    return R


def probe_rows(Ca: dict, Cb: dict, Fa: np.ndarray, Fb: np.ndarray,
               ia: np.ndarray, ib: np.ndarray, ga: list, gb: list) -> dict:
    """在掩码 ia/ib 选出的子集上出**七行归因表 + 残差双口径**（裁定 2）。

    行定义：
      1 仅黑边项 / 2 仅长宽比项  —— 几何族（同源，AUC 应近似相等）
      3 仅亮度统计项             —— 色调族
      4 仅尺度项                 —— 尺度族（不可消除，最强单项）
      5  平凡(几何+色调)          —— 上界口径的对照基线
      5s 平凡(几何+色调+尺度)      —— 下界口径的对照基线
      6 深度特征
      7a 深度−(几何+色调)          —— **证据上界**：尺度携带的真签名仍在残差里 → 高估
      7b 深度−(几何+色调+尺度)     —— **证据下界**：把尺度里的真签名也减掉 → 低估
    真值在 7a 与 7b 之间，照实报区间（裁定 2）。
    另有"尺度族佐证"若干行（speckle/grad/snr），均无标签，故可用于 BUS-UCLM 对。
    """
    y = np.r_[np.zeros(int(ia.sum()), int), np.ones(int(ib.sum()), int)]
    g = np.array([p for p, k in zip(ga, ia) if k] + [p for p, k in zip(gb, ib) if k])

    def cat(key: str) -> np.ndarray:
        # concatenate(axis=0)：对 1-D（pad/aspect/scale）与 2-D（bright/scale_extra）都正确
        return np.concatenate([Ca[key][ia], Cb[key][ib]], axis=0)

    GEO = np.hstack([cat("pad")[:, None], cat("aspect")[:, None]])
    TONE = cat("bright")
    SC = cat("scale")[:, None]
    geom_tone = np.hstack([GEO, TONE])
    geom_tone_scale = np.hstack([geom_tone, SC])

    rows = {}
    for name, X in (("1 仅黑边项(pad)", cat("pad")[:, None]),
                    ("2 仅长宽比项", cat("aspect")[:, None]),
                    ("3 仅亮度统计项", TONE),
                    ("4 仅尺度项", SC)):
        rows[name] = cv_auc(X, y, g)
    rows["5 平凡(几何+色调)"] = cv_auc(geom_tone, y, g)
    rows["5s 平凡(几何+色调+尺度)"] = cv_auc(geom_tone_scale, y, g)

    Fd = np.concatenate([Fa[ia], Fb[ib]], axis=0)
    rows["6 深度特征"] = cv_auc(Fd, y, g, pca_k=PCA_K)
    rows["7a 深度−(几何+色调) [上界]"] = cv_auc(
        residualize(Fd, geom_tone), y, g, pca_k=PCA_K)
    rows["7b 深度−(几何+色调+尺度) [下界]"] = cv_auc(
        residualize(Fd, geom_tone_scale), y, g, pca_k=PCA_K)

    # 尺度族佐证（无标签）
    for i, nm in enumerate(SCALE_EXTRA_NAMES):
        rows[f"·佐证 {nm}"] = cv_auc(cat("scale_extra")[:, i:i + 1], y, g)
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="no_domainhead",
                    help="纯 ERM 检查点（λ=0，从未被 GRL 触碰）")
    ap.add_argument("--adv-variant", default="baseline")
    ap.add_argument("--direction", default="busbra")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="resnet50_cbam")
    ap.add_argument("--out", default="runs/diag/domain_probe_p4.json")
    a = ap.parse_args()

    cfg = PreprocConfig(size=SIZE, geometry="p4")
    busi, _ = dedupe_entries(registry.busi_entries())
    busbra = registry.busbra_entries()
    busuclm = registry.busuclm_entries()
    print(f"设备 {DEV}  几何 p4")
    print(f"BUSI {len(busi)}  BUSBRA {len(busbra)}  BUS-UCLM {len(busuclm)}\n")

    # 平凡线索（与模型无关，只算一次）
    C = {"BUSI": trivial_cues(busi, cfg), "BUSBRA": trivial_cues(busbra, cfg),
         "BUS-UCLM": trivial_cues(busuclm, cfg)}

    results: dict = {"config": {"variant": a.variant, "direction": a.direction,
                                "seed": a.seed, "geometry": "p4"}, "tables": {}}

    for tag, variant, which in (("ERM", a.variant, "best_erm"),
                                ("ADV", a.adv_variant, "final")):
        try:
            ck = find_ckpt(variant, a.direction, a.seed, which)
        except FileNotFoundError as e:
            print(f"[skip] {tag}: {e}"); continue
        print(f"--- {tag}: {variant}/{which}  ({ck.relative_to(config.PROJECT_ROOT)}) ---")
        m = REGISTRY[a.model](n_domains=3)
        sd = torch.load(ck, map_location="cpu")
        m.load_state_dict(sd["model"])
        F = {k: extract(m, v, cfg) for k, v in
             (("BUSI", busi), ("BUSBRA", busbra), ("BUS-UCLM", busuclm))}
        del m
        torch.cuda.empty_cache()

        # 分层规则：只对两侧都是**有标签源域**的域对分层。
        # BUS-UCLM 的良恶性标签按 §2.3 冻结为不可用于报告/设计决策，故该对不分层。
        STRATA_OK = {"BUSI_vs_BUSBRA"}
        lab = {"BUSI": np.array([int(e["label"]) for e in busi]),
               "BUSBRA": np.array([int(e["label"]) for e in busbra])}

        for pair, other in (("BUSI_vs_BUS-UCLM", "BUS-UCLM"),
                            ("BUSI_vs_BUSBRA", "BUSBRA")):
            other_ent = busuclm if other == "BUS-UCLM" else busbra
            strata = [("all", None)]
            if pair in STRATA_OK:
                strata += [("benign", 0), ("malignant", 1)]
            for sname, cls in strata:
                ia = np.ones(len(busi), bool) if cls is None else (lab["BUSI"] == cls)
                ib = (np.ones(len(other_ent), bool) if cls is None
                      else (lab[other] == cls))
                rows = probe_rows(
                    C["BUSI"], C[other], F["BUSI"], F[other], ia, ib,
                    [e["patient_id"] for e in busi],
                    [e["patient_id"] for e in other_ent])
                key = f"{tag}:{pair}" + ("" if sname == "all" else f":{sname}")
                results["tables"][key] = rows
                print(f"\n  [{pair} | {sname}]  n={int(ia.sum())}+{int(ib.sum())}")
                for k, v in rows.items():
                    print(f"    {k:34s} AUC = {v:.4f}")
                print(f"    → 证据区间 = [{rows['7b 深度−(几何+色调+尺度) [下界]']:.4f}, "
                      f"{rows['7a 深度−(几何+色调) [上界]']:.4f}]"
                      f"   （下界=减掉尺度里的真签名；上界=保留）")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
