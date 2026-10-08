"""训练引擎：按步数调度 + 单次前向的 DANN。

对旧实现的四项修复
------------------
A1  旧代码 `len_dataloader = min(源, 目标)` → 每 epoch 只跑 13 个 batch，
    25 epoch 共 325 步，严重训练不足。现改为**按源域遍历、目标域循环复用**，
    总步数显式指定，α 的 sigmoid ramp 挂在**步数**上。
A2  旧代码源/目标分两次前向且判别头含 BatchNorm1d → BN 抹平域间一阶矩差。
    现改为：特征提取一次；**判别头看拼接后的源+目标**（同一 batch，修好一阶矩通道）；
    分类头只看源域特征（避免目标域影响分类头的 BN 统计）。
A3  目标域 loader 不返回标签；早停与选模一律用**源域验证集**。
D   显式种子 + 确定性开关 + AMP + 多 worker + 每次运行独立目录 + 逐图概率 CSV。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score

from ..models.layers import dann_alpha


@dataclass
class TrainConfig:
    model: str = "resnet50_cbam"
    steps: int = 10_000
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    lambda_domain: float = 1.0
    lambda_l: float = 0.0          # FDNet-v2 的 LL 通路对抗强度
    lambda_seg: float = 0.0        # 辅助分割损失权重
    gate_lr: float = 1e-2          # 门 g 的独立学习率（标量参数，需更大步长）
    class_weight: tuple = (1.0, 1.0)   # 中性配置（筛查优先改由阈值实现，见 v2 §3）
    amp: bool = True
    seed: int = 42
    num_workers: int = 8
    eval_every: int = 500
    image_size: int = 256
    tone_norm: bool = True
    geometry: str = "p4"           # "p4"（统一管线）| "letterbox"（仅附录消融臂）
    n_domains: int = 3
    hl_only: bool = True
    render_aug: bool = False       # 声学渲染随机化增广（第三层杠杆）
    wavelet_jitter: bool = False   # 小波域高频幅度扰动（纹理签名随机化）
    scale_aug: bool = False        # 尺度随机化近似（裁定 1：去相关 scale 与域标签）
    ema: bool = False              # 权重指数滑动平均
    mixstyle: bool = False         # 公平基线：MixStyle（ICLR 2021，见 models/mixstyle）
    mixstyle_crossdomain: bool = True   # 见下（上游语义要求；基线公平性关键）


def set_seed(seed: int, deterministic: bool = True) -> None:
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def to_device_uint8(x: torch.Tensor) -> torch.Tensor:
    """uint8 主机张量 → GPU 上的 float32 [-1,1]，通道顺序不变。"""
    return (x.to(torch.float32).div_(255.0).sub_(0.5)).div_(0.5)


class EMA:
    """指数滑动平均（DomainBed 上最稳的低成本提升之一）。

    语义：每步把滑动副本朝当前权重靠拢；评估用滑动副本。
    衰减 0.999（10k 步下的标准选择）。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                s.copy_(v)

    def copy_to(self, model: nn.Module) -> None:
        """把滑动权重写回模型（用于评估；调用前请自行备份原权重）。"""
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, labels, ids = [], [], []
    for batch in loader:
        x, y, _, idx, _ = batch          # 固定位置五元组
        logits = model(to_device_uint8(x.to(device, non_blocking=True)))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy()); ids.append(idx.numpy())
    p = np.concatenate(probs); y = np.concatenate(labels); i = np.concatenate(ids)
    out = {"n": int(len(y)), "probs": p, "labels": y, "indices": i}
    if len(set(y.tolist())) > 1:
        out["auc"] = float(roc_auc_score(y, p))
    else:
        out["auc"] = float("nan")
    return out


def train_one(cfg: TrainConfig, *, source_train, source_val, target_train,
              model, out_dir: Path, log=print) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")

    set_seed(cfg.seed, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 门 g 是标量参数，与主干共用 lr 时位移会被淹没（实测 50 步仅动 1e-4）。
    # 单独给它一个更大的学习率（V2.8.6）。
    gate_p = [n for n, _ in model.named_parameters() if n == "gate_logit"]
    base_p = [p for n, p in model.named_parameters() if n != "gate_logit"]
    groups = [{"params": base_p, "lr": cfg.lr}]
    if gate_p:
        groups.append({"params": [dict(model.named_parameters())[n] for n in gate_p],
                       "lr": cfg.gate_lr})
    opt = AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    cw = torch.tensor(cfg.class_weight, device=device)
    ce_cls = nn.CrossEntropyLoss(weight=cw)
    ce_dom = nn.CrossEntropyLoss()
    bce_seg = nn.BCEWithLogitsLoss()

    src_it = _cycle(source_train)
    tgt_it = _cycle(target_train)

    hist, best = [], {"auc": -1.0, "step": -1}
    ema = EMA(model) if cfg.ema else None
    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        s = next(src_it)
        t = next(tgt_it)
        xs, ys, s_dom, _, s_mask = s                # 固定位置五元组
        xt, _, t_dom, _, _ = t
        # uint8 → GPU 后归一化（避免 4 倍共享内存带宽）
        xs = to_device_uint8(xs.to(device, non_blocking=True))
        xt = to_device_uint8(xt.to(device, non_blocking=True))
        ys = ys.to(device, non_blocking=True)
        s_dom = s_dom.to(device, non_blocking=True)
        t_dom = t_dom.to(device, non_blocking=True)

        alpha = dann_alpha(step, cfg.steps)
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
            # --- 特征一次提取 ---
            #
            # ⚠ MixStyle 的**上游语义**是"同一 batch 内跨样本交换统计量"。
            #   本引擎默认分两次前向（源、目标各一次），若照此调用 MixStyle，
            #   混的是**同域内**的样本 —— 那就退化成普通正则化，
            #   而非上游的"跨域虚拟域"。故 mixstyle 开启时**必须把源+目标
            #   拼成一个 batch 前向一次**，这才是公平基线的正确档位。
            #   （FDNet 不支持 MixStyle，见 make_model；故此处只需处理非 dict。）
            if cfg.mixstyle and cfg.mixstyle_crossdomain:
                x_cat = torch.cat([xs, xt], 0)
                f_cat = model.forward_features(x_cat)
                nb = xs.size(0)
                fs, ft = f_cat[:nb], f_cat[nb:]
            else:
                fs = model.forward_features(xs)
                ft = model.forward_features(xt)

            # --- 分类：只用源域特征（保分类头 BN 统计纯净）---
            if isinstance(fs, dict):      # FDNet-v2
                zs = model.fuse(fs["f_l"], fs["f_h"])
                zt = model.fuse(ft["f_l"], ft["f_h"])
            else:
                zs, zt = fs, ft
            loss_cls = ce_cls(model.class_head(zs), ys)

            # --- 域判别：源+目标**拼成同一 batch**（修 A2 的一阶矩通道）---
            dom_labels = torch.cat([s_dom, t_dom], 0)
            loss_dom = 0.0
            if isinstance(fs, dict):
                # FDNet-v2：高频通路接 GRL（强度 alpha），LL 通路按 lambda_l 弱对抗。
                #
                # ⚠ **此处曾传 alpha=None，使 DomainHead.forward 跳过梯度反转**，
                #   即 FDNet 的对抗训练被**静默禁用**：dom loss 一路降到 1e-4
                #   （而 ResNet 骨干在对抗均衡时稳定在 ≈ln2=0.693），却无任何报错。
                #   DomainHead.forward 的约定是 "alpha=None ⇒ 不做 GRL"，
                #   故**必须显式传 alpha**。见 F26。
                dom_h = model.domain_head_h(
                    torch.cat([fs["f_h"], ft["f_h"]], 0), alpha)
                loss_dom = loss_dom + ce_dom(dom_h, dom_labels)
                if getattr(model, "domain_head_l", None) is not None and cfg.lambda_l > 0:
                    a_l = alpha * cfg.lambda_l
                    dom_l = model.domain_head_l(
                        torch.cat([fs["f_l"], ft["f_l"]], 0), a_l)
                    loss_dom = loss_dom + ce_dom(dom_l, dom_labels)
            else:
                dom_repr = model.domain_head(torch.cat([fs, ft], 0), alpha)
                loss_dom = ce_dom(dom_repr, dom_labels)

            # --- 辅助分割（仅源域掩膜，仅训练期）---
            loss_seg = 0.0
            if cfg.lambda_seg > 0 and isinstance(fs, dict) and model.seg_head is not None:
                ms = s_mask.to(device, non_blocking=True).unsqueeze(1).float()
                if ms.sum() > 0:
                    seg = model.seg_head(fs["fl_map"])
                    if seg.shape[-2:] != ms.shape[-2:]:
                        seg = torch.nn.functional.interpolate(
                            seg, size=ms.shape[-2:], mode="bilinear", align_corners=False)
                    loss_seg = bce_seg(seg.float(), ms.float())

            loss = (loss_cls + cfg.lambda_domain * loss_dom
                    + cfg.lambda_seg * loss_seg)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt); scaler.update(); sched.step()
        if ema is not None:
            ema.update(model)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            # EMA 生效时用滑动权重评估；评估后**必须还原**训练权重，
            # 否则后续训练会从 EMA 副本继续（权重被原地覆盖）。
            backup = None
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
            # 早停/选模：**源域验证集**（修 A3）
            r = evaluate(model, source_val, device)
            auc = r["auc"]
            if np.isfinite(auc) and auc > best["auc"]:
                best = {"auc": auc, "step": step}
                torch.save({"model": model.state_dict(), "step": step,
                            "cfg": asdict(cfg)}, out_dir / "best_erm.pt")
            if backup is not None:
                model.load_state_dict(backup)
            model.train()          # 修：evaluate() 会切到 eval 模式且不还原
            gate = float(model.gate()) if hasattr(model, "gate") else None
            hist.append({"step": step, "loss": float(loss), "loss_cls": float(loss_cls),
                         "loss_dom": float(loss_dom), "alpha": alpha,
                         "val_auc": auc, "gate": gate})
            log(f"  step {step:6d}/{cfg.steps}  loss {float(loss):.4f}  "
                f"cls {float(loss_cls):.4f}  dom {float(loss_dom):.4f}  "
                f"alpha {alpha:.3f}  src_val AUC {auc:.4f}"
                + (f"  g {gate:.3f}" if gate is not None else ""))

    # ---- 第二类检查点：末步权重 ----
    # 动机（实测发现的选模冲突）：对抗压力随 alpha 升到 1.0 后，源域 AUC 反而下降，
    # 于是按源域 AUC 选出的 best_erm.pt 系统性地落在**对抗不充分的早期步**
    # （实测 500–1000 / 3000），把它当作"对抗后"状态去测 P3 是错的。
    # 因此另存末步权重，作为 Stage B（真正对抗后）的测量对象：
    #   best_erm.pt —— 按源域 AUC 选，代表**最优源域性能**（P1 与性能主张用）
    #   final.pt    —— 末步，代表**已充分对抗**（P3 谱测量用）
    if ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        torch.save({"model": model.state_dict(), "step": cfg.steps,
                    "cfg": asdict(cfg), "kind": "final_ema"}, out_dir / "final.pt")
        model.load_state_dict(backup)
        model.eval()
    else:
        torch.save({"model": model.state_dict(), "step": cfg.steps,
                    "cfg": asdict(cfg), "kind": "final"}, out_dir / "final.pt")

    (out_dir / "history.json").write_text(
        json.dumps(hist, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"  完成：best_erm 源域 val AUC {best['auc']:.4f} @ step {best['step']}"
        f"（末步 {cfg.steps} 另存 final.pt）  耗时 {(time.time()-t0)/60:.1f} min")
    return {"best": best, "history": hist, "out_dir": str(out_dir)}


@torch.no_grad()
def predict_with_probs(model, entries, loader, device, *, domain_name: str,
                       out_csv: Path, lesion_ids: list | None = None) -> dict:
    """推理并存逐图概率 CSV（供 CI / DeLong / 校准 / 谱分析复用）。"""
    import csv
    model.eval()
    rows = []
    for batch in loader:
        x, _, _, idx, _ = batch           # 固定位置五元组
        logits = model(to_device_uint8(x.to(device)))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        p = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        for k, i in enumerate(idx.numpy()):
            e = entries[int(i)]
            rows.append({"image_id": Path(e["image"]).name, "prob": float(p[k]),
                         "label": int(e["label"]),
                         "patient_id": e["patient_id"], "lesion_id": e["lesion_id"],
                         "dataset": e["dataset"], "domain": domain_name})
    out_csv = Path(out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("predict_with_probs: 未产生任何预测行")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    y = np.array([r["label"] for r in rows]); pr = np.array([r["prob"] for r in rows])
    auc = float(roc_auc_score(y, pr)) if len(set(y.tolist())) > 1 else float("nan")
    return {"n": len(rows), "auc": auc, "csv": str(out_csv)}
