"""E8 谱分析驱动：把训练好的检查点 + 物化的 OASBUD 接到 e8_rmt_analysis_v2.py。

流程
----
1. 从 `runs/dev_screen/<变体>/<方向>/seed<N>/<run_id>/` 取检查点（F16：主配置 = final.pt）
2. 在若干队列上取**倒数第二层特征**（`export_features.py` 的口径）
3. 调 `analyze()` 出七行归因表所需的统计量，写入 Table S2 行

冻结依据
--------
- F10：主统计量 R；θ̂ 为效应量；NaN→0；一切阈值由校准产生
- F13/F21：域探针**必须**带平凡线索回归对照臂，残差出 [上界, 下界] 区间
- F19：分层只对两侧都是带标签源域的域对做；BUS-UCLM 因标签隔离不分层
- I1：P1 的方向对齐需要**多域合并**导出（单队列无置换定义）
- I2：PCA 敏感性臂只用于 B 模式队列；OASBUD 用原生 p（c>1 允许）

⚠ 本脚本只读检查点与数据，不写 `src/`。
⚠ OASBUD 需先 `scripts/materialize_oasbud.py` 物化，且 `registry.py` 需接线
  （待 F22 解除后应用 `patch_registry_oasbud.md` 中的补丁）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.models.backbones import REGISTRY
from breastdg.models.fdnet import FDNetV2
from breastdg.train.datasets import make_loader

# e8_rmt_analysis_v2.py 位于仓库外的工作文档层（meliorate/），即 PROJECT_ROOT 的父目录。
# 注意：不要再拼 "/ meliorate"，否则得到 meliorate/meliorate（曾犯此错）。
E8_DIR = config.PROJECT_ROOT.parent
if str(E8_DIR) not in sys.path:
    sys.path.insert(0, str(E8_DIR))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 检查点根：默认 dev_screen（旧行为，不变）；可由 CLI 切到锁箱布局。
#
# ⚠ 2026-10-08 加：**锁箱布局与 dev_screen 不同**
#     dev_screen : runs/dev_screen/<variant>/<direction>/seed<N>/<run_id>/<file>
#     lockbox    : runs/lockbox/<variant>/seed<N>/<run_id>/<file>   ← **无 direction**
#   之所以要支持：锁箱 49 格跑完后，E8/E6 也要在**锁箱检查点**上跑
#   （论文第②⑤幕的归因表/可解释性是锁箱量）。此前 find_ckpt 只会拼
#   dev_screen，在锁箱检查点上直接 FileNotFoundError。
#
#   实现取舍：用**模块级变量 + setter**，而不是给 find_ckpt 加参数 ——
#   因为它有 6 个调用点（alignment_trajectory / gradcam / run_e8 /
#   run_e8_adv_contrast / run_e8_batch / diagnose_domain_probe_p4），
#   逐个加参要改 6 处，且容易漏（本仓已因"改一处漏一处"栽过多次）。
#   默认 None = 旧行为，故**未显式设置时所有旧调用点行为完全不变**。
# ---------------------------------------------------------------------------
_RUNS_ROOT: Path | None = None


def set_runs_root(root: str | Path | None) -> None:
    """切换检查点根。None 或空 → 恢复默认 runs/dev_screen。

    ⚠ 2026-10-08 修：必须 **resolve() 成绝对路径**。
      传相对路径（如 `--runs-root runs/lockbox`）时，返回的检查点也是相对的，
      而 `main` 里 `ck.relative_to(config.PROJECT_ROOT)` 对相对路径直接抛
      `ValueError: ... is not in the subpath of ...`。
      实测踩到：`run_e8.py --runs-root runs/lockbox` 在**打印检查点那一步**
      就崩，连特征都还没提。归一成绝对路径后，下游 `relative_to` 才成立。
    """
    global _RUNS_ROOT
    _RUNS_ROOT = Path(root).resolve() if root else None
    if _RUNS_ROOT is not None:
        if not _RUNS_ROOT.is_dir():
            raise FileNotFoundError(f"--runs-root 不存在：{_RUNS_ROOT}")


def find_ckpt(variant: str, direction: str, seed: int, which: str) -> Path:
    # ⚠ 2026-10-08 加：**环境变量兜底**。`set_runs_root` 是「同进程内先调用」
     #   才生效；而 `gradcam_interpretability.py` / `run_e8_adv_contrast.py` /
     #   `run_e8_batch.py` 都是 `from run_e8 import find_ckpt` 后**直接调用**，
     #   从不调 setter → 它们**够不到锁箱布局**（实测：E6 只能解析 dev_screen）。
     #   环境变量对"按名引入"透明，故作为兜底：设了 `BREASTDG_CKPT_ROOT` 即走
     #   锁箱布局，未设则行为与历史**逐位相同**。
     #   优先级：显式 setter > 环境变量 > 默认 dev_screen。
    import os
    _root = _RUNS_ROOT or (
        Path(os.environ["BREASTDG_CKPT_ROOT"])
        if os.environ.get("BREASTDG_CKPT_ROOT") else None)
    if _root is not None:
        # 锁箱布局：无 direction 层
        base = _root / variant / f"seed{seed}"
    else:
        base = config.PROJECT_ROOT / "runs" / "dev_screen" / variant / direction / f"seed{seed}"
    cands = sorted(base.glob(f"*/{which}"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"找不到 {base}/*/{which}")
    if len(cands) > 1:
        # 出处歧义：同 seed 下多个 run 目录都含该文件 → 不静默挑
        # （见 evaluate_lockbox 的同类护栏）
        names = "\n    ".join(str(c.parent.name) for c in cands)
        raise RuntimeError(
            f"出处歧义：{base} 下有 {len(cands)} 个 {which}：\n    {names}\n"
            f"  拒绝静默挑最新。请归档多余目录，或显式指定。")
    return cands[0]


def build_model(cfg: dict):
    """从训练时的 cfg 重建模型。

    FDNet-v2 **不在** REGISTRY（REGISTRY 只有四个空间骨干），且
    其构造参数必须与训练时一致，hl_only 决定有无判别头，
    否则 load_state_dict 报键不匹配而失败。
    """
    if cfg.get("model") == "fdnet_v2":
        return FDNetV2(n_domains=3,
                       seg_head=float(cfg.get("lambda_seg", 0.0)) > 0,
                       lambda_l=float(cfg.get("lambda_l", 0.0)),
                       hl_only=bool(cfg.get("hl_only", True)))
    m = REGISTRY[cfg["model"]](n_domains=3)
    # ⚠ 2026-10-08 修（本轮实测暴露，**会打穿锁箱评估**）：
    #   `dev_screen.make_model` 在 `cfg["mixstyle"]` 为真时用
    #   `with_mixstyle` **插入** MixStyle 层。插入会**移动 Sequential 下标**
    #   （ResNet50 原 [4]layer1 [5]layer2 [6]layer3 [7]layer4，在 [4][5][6]
    #   之后各插一层 → layer4 由 7 变 10），故 mixstyle 检查点的键是
    #   `backbone.8.* / backbone.10.*`，而这里不带包装重建出的是
    #   `backbone.5/6/7.*` → `load_state_dict` 既缺键又**形状不符**。
    #   后果（实测）：49 格冻结时 mixstyle 7 格**全数失败**（42/49）；
    #   而 `evaluate_lockbox` 走同一个 `build_model`，一旦开跑锁箱评估
    #   （单次、不可逆），会在 mixstyle 格上直接崩 —— 届时已消耗掉
    #   唯一一次评估机会。
    #   为何此处加而**不动** dev_screen：训练已完成（49 格产物已在手），
    #   改训练侧只会使 `_code_hash()` 偏离训练时记录的值（那份哈希已写进
    #   49 个检查点，用于 F22 代码漂移检测），且不影响推理结果。
    #   为何是**恒等**的：MixStyle.forward 首行 `if not self.training: return x`
    #   —— 评估态原样返回、无参数、无 buffer，故对**所有**变体的评估路径
    #   逐比特无影响；插入只为让 state_dict 的键与下标对齐。
    if cfg.get("mixstyle"):
        from breastdg.models.mixstyle import with_mixstyle
        _KIND = {"resnet50_cbam": "resnet", "densenet121_se": "densenet",
                 "efficientnet_v2_s": "efficientnet"}
        _kind = _KIND.get(cfg["model"])
        if _kind is None:
            raise ValueError(
                f"cfg 声明 mixstyle=True 但 {cfg['model']!r} 无逐骨干映射。"
                f"MixStyle 只对空间骨干定义（Swin 的通道在最后一维，不适用），"
                f"见 models/mixstyle.with_mixstyle 的说明。")
        m.backbone = with_mixstyle(m.backbone, _kind,
                                   p=cfg.get("mixstyle_p", 0.5),
                                   alpha=cfg.get("mixstyle_alpha", 0.1))
    return m


@torch.no_grad()
def extract(model, entries, cfg: PreprocConfig, branch: str = "penultimate",
            bs: int = 64) -> np.ndarray:
    """提倒数第二层特征。

    ⚠ cfg（含 size）**必须**与训练时一致——FDNet-v2 单列配置训练时是
    512 输入 → DWT 子带 256×256；若喂 256 则子带只有 128×128、ResNet18
    末层特征图由 8×8 掉到 4×4（fdnet.py docstring 自己警告过这个不匹配）。
    实测后果（BUSI 子集 n=200）：PR 8.20 → 11.83（差 44%）、λ₁ 163.4 → 119.8。
    故调用方应传 `PreprocConfig(size=saved_cfg["image_size"], geometry="p4")`。
    """
    model.eval().to(DEV)
    ld = make_loader(entries, batch_size=bs, train=False, with_label=False,
                     cfg=cfg, num_workers=8)
    out = []
    for batch in ld:
        x = batch[0]
        x = (x.to(torch.float32) / 255.0 - 0.5) / 0.5
        f = model.forward_features(x.to(DEV, non_blocking=True))
        if isinstance(f, dict):
            # ⚠ FDNet.forward_features 返回的键是 **f_l / f_h（小写）**，
            # 见 src/breastdg/models/fdnet.py:131。
            # 2026-10-06 核查发现此处原写作 f_L / f_H（大写）—— 只要
            # --branch f_L|f_H 就必抛 KeyError，**P3 的逐子带检验从未能跑**。
            # 实测：branch=f_L -> KeyError；branch=f_l -> OK shape=(n,512)。
            # 为避免歧义，两种写法都接受。
            f = f[branch.lower()] if branch.lower() in ("f_l", "f_h") \
                else model.fuse(f["f_l"], f["f_h"])
        out.append(f.float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def standardize(F: np.ndarray, stats_from: np.ndarray | None = None
                ) -> np.ndarray:
    """按 S8(1) 标准化：逐特征维用**源域统计量**（缺失时用自身，即逐域口径）。

    为什么必需（曾犯此错）：`e8.analyze()` 的文档明说
    "feats 假定已在导出侧完成协议规定的标准化，此处仅居中"。
    若直接喂**原始**特征，各维尺度差异（实测标准差相差可逾一个量级）
    会主导协方差谱，把 λ₁ 撑到荒谬量级——实测 R=791.56（应 ≈1）、θ̂=6221。

    两种口径（F13 要求并报）：
      stats_from=None        → 逐域标准化（自用自身统计量）
      stats_from=<源域特征>   → 源域统计量（协议主口径）
    """
    src = F if stats_from is None else stats_from
    mu = src.mean(axis=0, keepdims=True)
    sd = src.std(axis=0, keepdims=True) + 1e-8
    return ((F - mu) / sd).astype(np.float32)


def _lesion_codes(entries: list[dict]) -> tuple[np.ndarray, str]:
    """把 lesion_id 字符串映射为连续整数（cluster bootstrap 用）。

    用**确定性的**映射（按首次出现顺序），不用 `hash()`——后者受 PYTHONHASHSEED
    影响，跨进程不稳定，会让结果不可复现。
    """
    ids = [str(e.get("lesion_id") or e.get("patient_id") or e["image"])
           for e in entries]
    uniq = {v: i for i, v in enumerate(dict.fromkeys(ids))}
    level = "lesion" if all(e.get("lesion_id") for e in entries) else "image"
    return np.array([uniq[v] for v in ids], dtype=np.int64), level


def load_oasbud(cohort: str = "corrected", size: int | None = None) -> list[dict]:
    """读物化后的 OASBUD manifest。

    ⚠ 2026-10-08 修两处（都会**静默出错**，故不是风格问题）：
    1. `image=r["path"]` 直接用 manifest 原值。manifest 的 path 列是
       **Windows 反斜杠**（`data\\OASBUD_p4\\...`），Linux 上 `Path()` 会把它
       当成**单个文件名** → 读图 FileNotFound。 evaluate_lockbox 的同类代码
       早已修正（取 basename 拼已知根），此处漏了 —— 又一次"改一处漏一处"。
    2. 根目录硬编码 `config.OASBUD_P4_DIR`（=256 那份）→ **无 512 支持**，
       `fdnet_gate_dual`（训练于 512）会拿到 256 的图。现按 `size` 选目录。
    """
    import csv as _csv
    from breastdg.data.cohorts import oasbud_materialize_dir
    base = oasbud_materialize_dir(int(size or 256))
    man = base / cohort / "manifest.csv"
    if not man.exists():
        raise FileNotFoundError(f"未物化：{man}（先跑 scripts/materialize_oasbud.py）")
    img_root = base / cohort / "images"
    out = []
    for r in _csv.DictReader(open(man, encoding="utf-8")):
        fn = Path(str(r["path"]).replace("\\", "/")).name    # 见上 ⚠1
        p = img_root / fn
        if not p.exists():
            raise FileNotFoundError(f"manifest 指向的图不存在：{p}")
        out.append(dict(dataset="oasbud", image=str(p), label=int(r["label"]),
                        patient_id=f"oasbud-{r['lesion_id']}",
                        lesion_id=f"oasbud-{r['lesion_id']}", masks=[],
                        birads=r["birads"], view=int(r["view"]),
                        src_id=r["src_id"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline")
    ap.add_argument("--direction", default="busbra")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--which", default="final", choices=["final", "best_erm"],
                    help="F16：主配置 = final；best_erm = Stage A 参照行")
    ap.add_argument("--model", default="resnet50_cbam")
    ap.add_argument("--branch", default="penultimate")
    ap.add_argument("--cohorts", default="busi,busbra,busuclm",
                    help="逗号分隔；oasbud_corrected / oasbud_uncorrected 需先物化")
    ap.add_argument("--conservative-n", type=int, default=None,
                    help="校准用保守 n（OASBUD 传病灶数 100）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--runs-root", default=None,
                    help="检查点根目录。默认 runs/dev_screen；"
                         "锁箱检查点请传 runs/lockbox（**无 direction 层**）")
    a = ap.parse_args()

    if a.runs_root:
        set_runs_root(a.runs_root)
        print(f"检查点根：{a.runs_root}（锁箱布局，无 direction 层）")

    ck = find_ckpt(a.variant, a.direction, a.seed, f"{a.which}.pt")
    print(f"检查点 {ck.relative_to(config.PROJECT_ROOT)}")
    sd = torch.load(ck, map_location="cpu")
    saved_cfg = sd.get("cfg") or {}
    print(f"  step={sd.get('step')}  kind={sd.get('kind', '-')}  "
          f"model={saved_cfg.get('model', '?')}  "
          f"geometry={saved_cfg.get('geometry', '?')}  "
          f"image_size={saved_cfg.get('image_size', '?')}")

    # 从检查点自带的 cfg 重建模型——**不要**用 CLI 的 --model。
    # 理由：FDNet-v2 不在 REGISTRY，且其构造参数（lambda_l / hl_only /
    # seg_head）必须与训练时一致，否则 load_state_dict 报键不匹配。
    m = build_model(saved_cfg or {"model": a.model})
    m.load_state_dict(sd["model"])

    # 尺寸同样取自检查点——训练/测试预处理必须一致（K7 类纪律）。
    # FDNet-v2 训练口径为 512 输入 → 子带 256；喂 256 会使子带降到 128、
    # ResNet18 末层特征图 8×8 → 4×4，实测 PR 差 44%（见 extract 的说明）。
    _size = int(saved_cfg.get("image_size", 256))
    print(f"  骨干分辨率 image_size={_size}")

    # ⚠ 2026-10-08 修：预处理配置改为**按队列**取，且只有一个出处。
    #   原实现在此处**一次算好** `PreprocConfig` 并给所有队列共用，
    #   且**没指定 tone_norm** → 取默认 `True`。
    #   对原图队列（busi/busbra/busuclm）那是对的；但 OASBUD 物化图**已是成品**
    #   （materialize_oasbud 已施加 tone_normalize(g,5,95) + P4），
    #   再走一遍色调归一化 = **二次归一化**，实测内容区 MAE ≈ 9–21。
    #   这条路径此前从未在 OASBUD 上跑过（run_e8_batch 明说只跑 dev 侧队列），
    #   所以"留到锁箱后统一出数"时才暴露 —— 正是它现在要跑的时机。
    #   现统一走 `cohorts.preproc_for_cfg`（全库唯一出处，与 evaluate_lockbox
    #   和 gradcam 同一份逻辑），杜绝第三份口径。
    from breastdg.data.cohorts import preproc_for_cfg, oasbud_arm

    import e8_rmt_analysis_v2 as E8

    rows = []
    for name in [c.strip() for c in a.cohorts.split(",") if c.strip()]:
        if name.startswith("oasbud"):
            # ⚠ 2026-10-08 修：原为 `"corrected" in name`（**子串**判定），
            #   而 `"corrected" in "oasbud_uncorrected"` → True，
            #   于是"未校正"臂被解析成 corrected → **两臂读同一目录**。
            #   两臂都合法、都能出数、都不报错，只是结果**逐位相同**
            #   （实测 27/27 格 R 与 theta 完全相等）→ 静默污染。
            #   现由 cohorts.oasbud_arm 做**精确**解析（唯一出处）。
            arm = oasbud_arm(name)
            ents = load_oasbud(arm, size=_size)
            n_cons = a.conservative_n or len({e["lesion_id"] for e in ents})
            # 预处理口径按**队列**（oasbud）而非按臂 —— 两臂同为预物化成品
            cohort_key = "oasbud"
        else:
            fn = registry.ALL_PUBLIC.get(name)
            if fn is None:
                print(f"[skip] 未知队列 {name}"); continue
            ents = fn()
            if name == "busi":
                ents, _ = dedupe_entries(ents)
            n_cons = a.conservative_n
            cohort_key = name

        # 按**队列**取预处理口径（OASBUD=identity，其余=P4+色调），
        # 而非全队列共用一个 —— 见上面 2026-10-08 的说明。
        cfg_q = preproc_for_cfg(cohort_key, saved_cfg)
        print(f"  [{name}] 预处理 size={cfg_q.size} tone_norm={cfg_q.tone_norm} "
              f"geometry={cfg_q.geometry}")

        F_raw = extract(m, ents, cfg_q, a.branch)
        # S8(1) 要求 feats 在**导出侧**完成标准化（analyze 只居中）。
        # 不标准化会因各维尺度差异主导谱，产生荒谬的 R/θ̂（曾实测 R=791.56）。
        F = standardize(F_raw)                 # 逐域口径（默认）
        y = np.array([int(e["label"]) for e in ents])
        # 单队列导出时 domains 为常数（P1 的方向对齐需用 export_features.export_combined
        # 的 source_combined 队列，那里才有 ≥2 个域；见 I1）。
        dom = np.full(len(ents), -1)
        lesion, _ = _lesion_codes(ents)
        meta = {"model": a.model, "stage": a.which, "branch": a.branch,
                "cohort": name, "seed": a.seed,
                "std": "per_domain", "projection": "native"}
        row = E8.analyze(F, y, dom, lesion, meta, conservative_n=n_cons)
        rows.append(row)
        print(f"  特征 {F_raw.shape}  逐维标准差比 {F_raw.std(0).max()/max(F_raw.std(0).min(),1e-9):.1f}×"
              f"  → 标准化后 {F.std(0).mean():.3f}±{F.std(0).std():.4f}")
        print(f"\n[{name}] n={len(ents)} p={F.shape[1]} c={F.shape[1]/len(ents):.3f} "
              f"校准 n={n_cons}")
        print(f"  R={row.R:.4f} (阈值 {row.R_thr:.4f})  检出={row.detected}")
        print(f"  theta={row.theta:.3f} [{row.theta_ci_lo:.3f}, {row.theta_ci_hi:.3f}]  "
              f"M={row.M:.5f}")
        print(f"  PR={row.PR:.2f} (null {row.PR_null:.1f}, frac {row.PR_frac:.3f})  "
              f"er={row.er:.2f} (frac {row.er_frac:.3f})  "
              f"tk16={row.tk16:.3f}  K={row.K_above}")
        print(f"  √c 参考线={row.sqrt_c_ref:.3f}")

    if a.out:
        p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([r.__dict__ for r in rows], indent=2,
                                ensure_ascii=False, default=str), encoding="utf-8")
        print(f"\n已写入 {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
