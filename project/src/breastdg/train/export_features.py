"""特征导出：E8 谱分析的输入接口。

输出格式（阶段 2 冻结）
-----------------------
    features/{model}/{stage}/{branch}/{cohort}/{seed}.npz
      ├─ feats      float32 [n, p]
      ├─ labels     int64   [n]
      ├─ domains    int64   [n]
      ├─ lesion_id  int64   [n]    # 新增：cluster bootstrap / 置换校准必需
      ├─ image_id   unicode [n]    # 新增：与逐图概率 CSV 可 join
      └─ meta       json（p, n, c, 层名, 标准化来源, 配置哈希）

为何必须有 lesion_id
--------------------
F5 冻结：OASBUD 同一病灶的两个 RF 切面**不可拆**，所有 CI 用病灶级 cluster
bootstrap。没有 lesion_id，E8 在 OASBUD 上的 bootstrap 与置换校准都会做错
（把两切面当独立样本 → CI 偏窄、置换 p 值偏小）。无病例标识的队列退化为
image 级（lesion_id ≡ image 序号），并在 meta 中标注。

c = p/n 的上限（重要）
----------------------
BBP 分析要求 c < 1（c>1 时谱在 0 处出现原子，尖峰被淹没；实测 c=2, θ̃=6 时
不可分辨）。OASBUD 目标域 n=200，而各骨干 p 为 512–2048，故 **c 全部 > 1**。
本模块提供两条降 c 途径，均在 meta 中记录：
  1. `p_max`：按**事先固定**的规则降维（见下），保证 c ≤ 目标值；
  2. 层选择：改用更浅层的特征（由 `branch` 参数指定）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .. import config
from ..data.preprocess import PreprocConfig
from .datasets import make_loader

FEAT_ROOT = config.PROJECT_ROOT / "features"

# 目标 c 上限（事先固定；见 freeze_log F10）
C_TARGET_MAX = 0.8


def _cfg_hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()[:16]


def _lesion_ids(entries: list[dict]) -> tuple[np.ndarray, str]:
    """把 lesion_id 字符串映射为整数；无病灶信息的队列退化为 image 级。"""
    ids = [str(e.get("lesion_id") or e.get("patient_id") or e["image"]) for e in entries]
    uniq = {v: i for i, v in enumerate(dict.fromkeys(ids))}
    level = "lesion" if all(e.get("lesion_id") for e in entries) else "image"
    return np.array([uniq[v] for v in ids], dtype=np.int64), level


def _reduce_dims(F: np.ndarray, p_max: int, seed: int) -> tuple[np.ndarray, dict]:
    """把特征维降到 p_max 以下。

    规则（事先固定，禁止按结果回调）：取**前 p_max 个**坐标（不做随机子采样，
    以免结果依赖随机选择）。若 p 已 ≤ p_max 则原样返回。
    """
    n, p = F.shape
    if p <= p_max:
        return F, {"reduced": False, "p_orig": p, "rule": None}
    return F[:, :p_max], {"reduced": True, "p_orig": p,
                          "rule": f"取前 {p_max} 维（确定性，非随机子采样）"}


COMBINED_COHORT = "source_combined"


@torch.no_grad()
def export_combined(model, parts: dict[str, list[dict]], device, *,
                    model_name: str, stage: str, branch: str,
                    out_root: Path | None = None, seed: int = 42,
                    p_max: int | None = None, standardize: str = "source") -> Path:
    """多域**合并**导出（I1 修正）。

    为什么需要它：P1 要求测 |⟨û₁, ŵ_dom⟩| 的方向对齐，而 ŵ 需要**同一次导出里有 ≥2 个域**。
    按队列分文件的导出（`{...}/{cohort}/{seed}.npz`）中，单队列的 `domains` 是常数，
    置换检验无定义。故训练域三队列合并为 `source_combined` 单独导出一份。

    合并顺序固定（busi → busbra → busuclm），与 `DOMAIN_ID` 一致，便于复现。
    外部队列（BrEaST / OASBUD）**不参与合并**——它们只在锁箱期单独评估。
    """
    out_root = Path(out_root or FEAT_ROOT)
    order = [k for k in ("busi", "busbra", "busuclm") if k in parts]
    if len(order) < 2:
        raise ValueError(f"合并导出至少需要 2 个域，收到 {order}")

    Fs, ys, ds, idxs, entries_all = [], [], [], [], []
    off = 0
    for name in order:
        ents = parts[name]
        ld = make_loader(ents, batch_size=64, train=False, with_label=False,
                         cfg=PreprocConfig(size=256, geometry="p4"), num_workers=6)
        fs, ys_, ds_, ix_ = [], [], [], []
        for batch in ld:
            x, y, dom, idx, _ = batch
            x = x.to(device, non_blocking=True)
            if branch in ("f_L", "f_H"):
                f = model.forward_features(x)[branch]
            elif branch == "penultimate":
                f = model.forward_features(x)
                if isinstance(f, dict):
                    f = model.fuse(f["f_l"], f["f_h"])
            else:
                raise ValueError(f"未知 branch {branch!r}")
            fs.append(f.float().cpu().numpy())
            ys_.append(y.numpy()); ds_.append(dom.numpy()); ix_.append(idx.numpy())
        Fs.append(np.concatenate(fs)); ys.append(np.concatenate(ys_))
        ds.append(np.concatenate(ds_))
        idxs.append(np.concatenate(ix_) + off)
        entries_all.extend(ents)
        off += len(ents)

    F = np.concatenate(Fs).astype(np.float32)
    y = np.concatenate(ys); dom = np.concatenate(ds); sup = entries_all

    n, p_orig = F.shape
    p_max_eff = p_max if p_max is not None else int(np.floor(C_TARGET_MAX * n))
    F, red = _reduce_dims(F, max(1, p_max_eff), seed)
    p = F.shape[1]; c = p / n

    mu, sd = F.mean(0, keepdims=True), F.std(0, keepdims=True) + 1e-8
    if standardize == "source":
        F = (F - mu) / sd

    lesion, lesion_level = _lesion_ids(sup)
    image_ids = np.array([Path(e["image"]).name for e in sup], dtype=object)
    meta = {"model": model_name, "stage": stage, "branch": branch,
            "cohort": COMBINED_COHORT, "domains_merged": order,
            "seed": int(seed), "n": int(n), "p": int(p), "c": float(c),
            "c_target_max": C_TARGET_MAX, "lesion_level": lesion_level,
            "standardize": standardize,
            "n_domain_values": int(len(set(dom.tolist()))),
            **red}
    meta["config_hash"] = _cfg_hash({k: v for k, v in meta.items()})

    out_dir = out_root / model_name / stage / branch / COMBINED_COHORT
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seed}.npz"
    np.savez_compressed(out_path, feats=F, labels=y.astype(np.int64),
                        domains=dom.astype(np.int64), lesion_id=lesion,
                        image_id=image_ids.astype(str),
                        meta=json.dumps(meta, ensure_ascii=False))
    return out_path


@torch.no_grad()
def export(model, entries: list[dict], loader, device, *,
           cohort: str, model_name: str, stage: str, branch: str,
           out_root: Path | None = None, seed: int = 42,
           p_max: int | None = None, standardize: str = "source") -> Path:
    """导出某（模型 × 阶段 × 分支 × 队列 × 种子）的特征。

    branch 取值：
      空间骨干 → "penultimate"
      FDNet-v2 → "f_L" / "f_H"（逐通路；这两条正是理论 2.2.3 的可测对象）
    standardize：
      "source" = 用源域统计量（均值/标准差）标准化后**只保存**，统计量来源写入 meta；
      "none"   = 不标准化（对照臂）。
    """
    model.eval()
    out_root = Path(out_root or FEAT_ROOT)

    feats, labels, domains, idxs = [], [], [], []
    for batch in loader:
        x, y, dom, idx, _ = batch
        x = x.to(device, non_blocking=True)
        if branch in ("f_L", "f_H"):
            d = model.forward_features(x)
            f = d[branch]
        elif branch == "penultimate":
            f = model.forward_features(x)
            if isinstance(f, dict):
                f = model.fuse(f["f_l"], f["f_h"])
        else:
            raise ValueError(f"未知 branch {branch!r}")
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy()); domains.append(dom.numpy()); idxs.append(idx.numpy())

    F = np.concatenate(feats).astype(np.float32)
    y = np.concatenate(labels); dom = np.concatenate(domains); idx = np.concatenate(idxs)

    n, p_orig = F.shape
    # ---- 降 c（事先固定规则）----
    p_max_eff = p_max if p_max is not None else int(np.floor(C_TARGET_MAX * n))
    F, red = _reduce_dims(F, max(1, p_max_eff), seed)
    p = F.shape[1]
    c = p / n

    # ---- 标准化 ----
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-8
    if standardize == "source":
        F = (F - mu) / sd
    stats_src = "source_domain_self" if standardize == "source" else "none"

    # ---- 标识 ----
    sup = [entries[int(i)] for i in idx]
    lesion, lesion_level = _lesion_ids(sup)
    image_ids = np.array([Path(e["image"]).name for e in sup], dtype=object)

    meta = {
        "model": model_name, "stage": stage, "branch": branch, "cohort": cohort,
        "seed": int(seed), "n": int(n), "p": int(p), "c": float(c),
        "c_target_max": C_TARGET_MAX,
        "lesion_level": lesion_level,
        "standardize": stats_src,
        "standardize_mu": mu.ravel()[:p].tolist() if standardize == "source" else None,
        "standardize_sd": sd.ravel()[:p].tolist() if standardize == "source" else None,
        **red,
    }
    meta["config_hash"] = _cfg_hash({k: v for k, v in meta.items()
                                     if k not in ("standardize_mu", "standardize_sd")})

    out_dir = out_root / model_name / stage / branch / cohort
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{seed}.npz"
    np.savez_compressed(
        out_path,
        feats=F, labels=y.astype(np.int64), domains=dom.astype(np.int64),
        lesion_id=lesion, image_id=image_ids.astype(str),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    return out_path


def load_features(path: str | Path) -> dict:
    """读回 npz；meta 自动反序列化。"""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in ("feats", "labels", "domains", "lesion_id", "image_id")}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def sanity_report(path: str | Path) -> str:
    d = load_features(path)
    m = d["meta"]
    f = d["feats"]
    lines = [
        f"{Path(path).name}: n={m['n']} p={m['p']} c={m['c']:.3f} "
        f"(原 p={m.get('p_orig', m['p'])})",
        f"  branch={m['branch']}  队列={m['cohort']}  阶段={m['stage']}  "
        f"lesion 级={'是' if m['lesion_level']=='lesion' else '否（退化为 image 级）'}",
        f"  lesion 数={len(set(d['lesion_id'].tolist()))}  "
        f"标签 {int((d['labels']==0).sum())}:{int((d['labels']==1).sum())}  "
        f"域 {sorted(set(d['domains'].tolist()))}",
        f"  有限值比例={float(np.isfinite(f).mean()):.4f}  "
        f"标准化={m['standardize']}",
    ]
    if m["c"] > 1.0:
        lines.append("  ⚠ c>1：谱含零原子，尖峰不可分辨（见 freeze_log F10）")
    return "\n".join(lines)
