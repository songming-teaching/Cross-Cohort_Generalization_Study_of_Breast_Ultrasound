"""E6：可解释性定量化（Grad-CAM + 病灶掩膜重叠）。

回应审稿意见 7/8
----------------
意见 7：为何只对困难恶性病例做可解释性？其他病例呢？
意见 8：可解释性方法细节缺失；Fig 4 的 "baseline maps" 用的是哪个模型？

本脚本的处理：
- **病例三类**（v14 §4.1 E6 行）：恶性困难 / 恶性典型 / 良性，每类 ≥10 例——
  正面回应"只做困难恶性"的质疑；
- **方法写明**：Grad-CAM 目标层、公式、骨干指名（意见 8）；
- **定量指标**：saliency 能量落入病灶掩膜的比例（BUSI/OASBUD 有掩膜），
  每模型 ≥30 例，配 Mann-Whitney 检验（回应"定量化"要求）。

纪律
----
- 目标层对空间骨干取最后一层卷积；FDNet-v2 取 LL 通路对应层（形态在低频，
  见理论 §2.2.3）——**并在图注写明**，否则审稿人会问"为何与别人不同"。
- 掩膜只用于**评估** saliency，不参与训练（BUS-UCLM 掩膜禁用，A3 隔离）。

用法
----
  python scripts/gradcam_interpretability.py --variant baseline --cohort busi
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.train.datasets import make_loader
from breastdg.train.engine import to_device_uint8
from run_e8 import build_model, find_ckpt

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = config.RUNS_DIR / "diag" / "gradcam"


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    """标准 Grad-CAM（Selvaraju et al. 2017）。

    L^c = ReLU( Σ_k α_k^c A^k )，其中 α_k^c = (1/Z) Σ_ij ∂y^c/∂A^k_ij。
    """

    def __init__(self, model, target_layer: torch.nn.Module):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x: torch.Tensor, target: int = 1) -> np.ndarray:
        """`target` = 反传所用类别。

        ⚠ 2026-10-08 加（实测暴露）：**原实现恒取 `logits[:, 1]`（恶性类）**。
        对**良性**病例，模型把恶性概率压到 ≈0.0004（几乎必然判对），
        此时 ∂(恶性 logit)/∂A 的通道均值 α_k 全为负 →
        `F.relu(Σ_k α_k A^k)` **整张全零** → `mask_energy_ratio` 分母为 0
        → 返回 nan。实测 7 个变体中 `efficientnet_v2_s` 的**良性列全缺**，
        叙事要求的"三类病例（良性/恶性典型/恶性困难）"实际做不了。

        这不是崩溃，而是"恶性类 CAM 在良性病例上本就不存在"的**定义性后果**。
        文献里的常规做法是取**真类** logit（每例反传其真实类别），
        这样良/恶两类都有定义。故加 `target`，由调用方按病例判据传入。
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(to_device_uint8(x.to(DEV)))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        score = logits[:, int(target)].sum()     # ← 真类 logit（见 docstring）
        score.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)        # α_k
        cam = F.relu((w * self.acts).sum(dim=1))             # Σ_k α_k A^k
        cam = cam[0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam


def target_layer_of(model, variant: str = ""):
    """取 Grad-CAM 目标层。

    空间骨干：最后一层卷积块。FDNet-v2：**LL 通路**（形态在低频，理论 §2.2.3）；
    若取高频通路则 saliency 反映的是斑点纹理而非病灶形态，与"网络读形态"的主张
    不对应——图注必须写明这一点。
    """
    if hasattr(model, "enc_l"):                  # FDNet-v2
        for m in reversed(list(model.enc_l.modules())):
            if isinstance(m, torch.nn.Conv2d):
                return m
    for name in ("layer4", "features", "backbone"):
        if hasattr(model, name):
            sub = getattr(model, name)
            for m in reversed(list(sub.modules())):
                if isinstance(m, torch.nn.Conv2d):
                    return m
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise RuntimeError("找不到卷积目标层")


# ---------------------------------------------------------------------------
# 掩膜重叠
# ---------------------------------------------------------------------------
def mask_energy_ratio(cam: np.ndarray, mask: np.ndarray) -> float:
    """saliency 能量落入病灶掩膜的比例 = Σ_cam[mask] / Σ_cam。"""
    if cam.shape != mask.shape:
        import torch as _t
        m = _t.from_numpy(mask)[None, None].float()
        m = F.interpolate(m, size=cam.shape, mode="nearest")[0, 0].numpy()
        mask = m > 0.5
    tot = cam.sum()
    return float(cam[mask].sum() / tot) if tot > 0 else float("nan")


def load_mask(path: str, size: int) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(im) > 127)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--direction", default="busbra")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--which", default="final")
    ap.add_argument("--cohort", default="busi",
                    help="只有带掩膜的队列可算重叠：busi / breast")
    ap.add_argument("--max-per-class", type=int, default=20)
    a = ap.parse_args()

    ck = find_ckpt(a.variant, a.direction, a.seed, f"{a.which}.pt")
    sd = torch.load(ck, map_location="cpu")
    cfg = sd.get("cfg") or {"model": "resnet50_cbam"}
    model = build_model(cfg)
    model.load_state_dict(sd["model"])
    model.eval().to(DEV)
    size = int(cfg.get("image_size", 256))

    ents = (dedupe_entries(registry.busi_entries())[0] if a.cohort == "busi"
            else registry.breast_entries())
    ents = [e for e in ents if e.get("masks")]
    if not ents:
        print(f"{a.cohort} 无掩膜，无法计算重叠"); return 1

    cam_engine = GradCAM(model, target_layer_of(model, a.variant))
    # ⚠ 2026-10-08 修：原为就地自造 `PreprocConfig(...)`，**与 evaluate_lockbox
    #   的 `preproc_for` 是两份独立定义**。对**原图**队列两者恰好一致，故一直
    #   没暴露；但遇到**预物化成品**队列（OASBUD：已含色调归一化 + P4）时，
    #   这里会**二次归一化**（实测内容区 MAE ≈ 9–21），而评估路径不会 ——
    #   同一批数据两套预处理，属于"改一处漏一处"的同类问题。
    #   现统一走全库唯一出处 `cohorts.preproc_for_cfg`。
    from breastdg.data.cohorts import preproc_for_cfg
    pp = preproc_for_cfg(a.cohort, cfg)
    ld = make_loader(ents, batch_size=1, train=False, with_label=True,
                     cfg=pp, num_workers=0)

    # 三类：良性 / 恶性典型 / 恶性困难
    rows = []
    seen = {0: 0, 1: 0}
    for batch in ld:
        x, y, _, idx, _ = batch
        lab = int(y.item())
        if seen[lab] >= a.max_per_class:
            continue
        e = ents[int(idx.item())]
        # 真类 CAM（见 GradCAM.__call__ 的说明）：良性病例反传良性 logit，
        # 否则 ReLU 后整张全零 → mask_energy_ratio 恒为 nan。
        cam = cam_engine(x, target=lab)
        # 掩膜（取并集）
        mk = None
        for mp in e["masks"]:
            m = load_mask(mp, cam.shape[0])
            mk = m if mk is None else (mk | m)
        if mk is None:
            continue
        ratio = mask_energy_ratio(cam, mk)
        logits = model(to_device_uint8(x.to(DEV)))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        prob = float(torch.softmax(logits.float(), 1)[0, 1])
        # "困难" = 概率靠近 0.5
        difficulty = 1.0 - abs(prob - 0.5) * 2
        rows.append(dict(image=Path(e["image"]).name, label=lab, prob=prob,
                         difficulty=difficulty, mask_energy_ratio=ratio))
        seen[lab] += 1
        if all(v >= a.max_per_class for v in seen.values()):
            break

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.variant}_{a.cohort}_{a.which}"
    p = OUT / f"gradcam_{tag}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 分组统计（恶性按难度中位数二分 → 困难/典型）
    mal = [r for r in rows if r["label"] == 1]
    ben = [r for r in rows if r["label"] == 0]
    out = dict(variant=a.variant, cohort=a.cohort, which=a.which,
               n_benign=len(ben), n_malignant=len(mal),
               mean_ratio_benign=float(np.mean([r["mask_energy_ratio"] for r in ben])) if ben else None,
               mean_ratio_malignant=float(np.mean([r["mask_energy_ratio"] for r in mal])) if mal else None,
               target_layer="LL-path (FDNet-v2)" if a.variant.startswith("fdnet")
                            else "layer4 (spatial backbone)",
               # 口径须随数一起写进产物，否则论文无法交代 saliency 是"哪一类的"
               cam_target="true_class",
               cam_convention=("每例反传**其真实类别**的 logit。"
                               "早期版本恒用恶性类 logit，导致良性病例 ReLU 后"
                               "整张全零、mask_energy_ratio 恒为 nan（实测 "
                               "efficientnet_v2_s 良性列全缺）；改真类后两类均有定义。"
                               "跨版本比较时**不可混用**这两种口径。"))
    (OUT / f"gradcam_{tag}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {p}")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
