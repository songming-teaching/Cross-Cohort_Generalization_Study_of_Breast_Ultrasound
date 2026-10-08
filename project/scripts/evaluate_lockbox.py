"""锁箱单次评估：冻结模型 × OASBUD 双管线 / BrEaST（一次性、不可逆）。

规范（v14 §5 + 补遗 + 冻结日志）
--------------------------------
- **只评估一次**：外部集不调阈值、不选模型、不选变体（DomainBed 式锁箱纪律）。
- **阈值冻结在源域**：运行点（0.5 / 源域 Youden / 预指敏感度 0.90）由源域验证集
  选定后**冻结在 `thresholds.json`**，本脚本只**读取**，绝不在外部集上调阈（F7.1）。
- **cluster bootstrap**：OASBUD 同一病灶两个切面不可拆，簇键=病灶
  （F23：.mat 唯一 id 实为 99 个，'30nh' 重复，故不能用 id 字段）；
  BrEaST 按 case/病灶。
- **三运行点指标族**：sens/spec/PPV/NPV/accuracy + prevalence（F6）。
- **报告口径**：AUC 同时报有符号值与 |AUC−0.5|（E3 要求）；一切阈值无关指标
  才可跨队列比较（§4.1）。

用法
----
  # 正式（一次出全卡；无 --out-tag 即为正式执行，有单次护栏）
  python scripts/evaluate_lockbox.py --full-card

  # 冒烟（不碰锁箱、不限次数、可指定 dev 检查点）
  python scripts/evaluate_lockbox.py --cohort busi \
      --ckpt runs/dev_screen/baseline/busbra/seed42/<rid>/final.pt --out-tag _smoke

边界
----
- 只读 `runs/lockbox/` 的检查点与公开数据；不写 `src/`。
- **不碰任何私有队列**（F9）。

修订记录（2026-10-07，依 check2.txt 复核意见）
------------------------------------------------
P0-1 **阈值在外部集上现算** —— 锁箱纪律里唯一不能犯的错。原代码
     `thr_youden = youden_threshold(y, p)`（y,p 是**外部集**的标签与预测），
     即 Youden/sens90 两个运行点是在外部集自己身上算的。审稿人一旦发现，
     **整张卡作废**。
     修：改为读**同目录** `thresholds.json`（由 `scripts/freeze_thresholds.py`
     在源域验证集上生成并冻结）。**缺失即拒绝执行（exit 4）**，不静默回退。
     外部集上连 oracle 列都不保留。
P0-2 **无单次执行护栏** —— 可无限次重跑并静默覆盖评估卡。
     修：正式执行（无 `--out-tag`）前，若目标评估卡已存在 → 打印拒绝信息并
     **exit 3**；正式执行全程持 `RunLock`。冒烟（带 `--out-tag`）不受限。
P0-3 **评估卡被切成三次执行** —— 原子单位模糊，半张卡落入灰色地带。
     修：加 `--full-card`，**一次进程内**跑完 OASBUD 双臂 + BrEaST，
     产出一张完整评估卡 + 一份汇总 CSV。该卡才是登记对象。
P0-4 **审计字段缺失** —— 只记了检查点相对路径字符串。
     修：卡内写入 ①每个 `final.pt` 的 SHA256；②代码指纹（无 git 仓库时
     退回 `dev_screen._code_hash`，dirty 情形标注）；③环境锁定
     （torch/numpy/sklearn 版本 + `requirements.lock.txt` 的 SHA256）。
     卡写盘后自身 SHA256 一并写入。
P1-a 自举种子 `seed=0` → 改为冻结种子 **20261006**（与叙事 CI 同源，避免双版本 CI）。
P1-b `cands[-1]` 按 mtime 静默挑最新 → 同 seed 下多个 run 目录时**报错**（出处歧义）。
P1-c `sd.get("cfg") or {...}` 静默默认 resnet50_cbam → 非冒烟模式下缺失即拒绝。
P1-d BrEaST 预检：实测确认为**原图**（485×537 RGBA，位于 DATA_ROOT/BrEaST/images），
     非物化目录 —— 故 P4 + 色调在评估时施加是正确的。
P1-e 簇数打印走与 `lesion_clusters` **同一逻辑**（原来是另一套，会把无 lesion_id 的
     队列全归为 1 簇而误报）。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Windows 控制台默认 GBK；本文件 print 里含 `✗`(U+2717) 等 GBK 编不了的字符，
# 直接跑会 UnicodeEncodeError 打死进程。子进程有 PYTHONIOENCODING 兜底，
# 但本脚本会被手动直接跑（上云前冒烟），故入口自护。且**必须早于任何
# 风险 print**（preflight 坑1 按行号判时序）。2026-10-06 preflight 报出。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breastdg import config
from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.train.datasets import make_loader
from breastdg.train.engine import to_device_uint8
from run_e8 import build_model

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCKBOX = config.RUNS_DIR / "lockbox"
OUT = config.RUNS_DIR / "lockbox_eval"

# 冻结自举种子（P1-a）。与 verify_narrative_numbers.py 同源，避免双版本 CI。
SEED_BOOTSTRAP = 20261006

# 退出码约定（P0-1/P0-2；护栏刻意不用 1，以便与普通异常区分）
EXIT_OK = 0
EXIT_ERR = 1
EXIT_ALREADY_EXISTS = 3      # 评估卡已存在 → 拒绝重跑（锁箱单次纪律）
EXIT_NO_THRESHOLDS = 4       # 缺源域冻结阈值 → 拒绝执行（F7.1）

# OASBUD 物化目录（含 P4 几何 **与色调归一化** 的成品图）。
# ⚠ 2026-10-06 实测（本机 + 云上都要成立，故早于锁箱发现是本脚本的幸运）：
#   `materialize_oasbud.py` 写图前已施加 `tone_normalize(g, 5.0, 95.0)`
#   与 P4 几何 → 产物是 **已预处理成品**。
#   本脚本若再用默认 `PreprocConfig`（tone_norm=True, geometry="p4"）
#   读它，就是**二次归一化**。实测分解（同一张物化图，逐项对 raw 比 MAE）：
#     ① _load_gray_horizontal vs raw        MAE = 0.000   ← 恒等
#     ② 仅 tone_normalize    vs raw        MAE = 9.345   ← 误差**全在这里**
#     ③ G.plan/apply(ratio=1) vs raw        MAE = 0.000   ← 几何其实恒等
#   故修正为 **identity 预处理**：`geometry="letterbox"`（对正方形即恒等）
#   + `tone_norm=False`，实测 MAE = 0.000（逐位相同）。
#   （曾误判为"P4 几何重复施加"；实测证伪，根因是色调。）
# 若换 512 输入的骨干（FDNet-v2 单列配置）会 upsample 非恒等——故此处
# **限定 size 必须等于物化 size**，不一致则直接报错而非静默出数。
OASBUD_DIR = config.OASBUD_P4_DIR    # 见 config：可用 BREASTDG_OASBUD_P4 覆盖
OASBUD_MATERIALIZE_SIZE = 256        # **默认**尺寸（勿改语义：多处按此名取默认）

# ⚠ 2026-10-08 加：**双尺寸支持**（阻塞 ⑫ 的那条）。
#
#   问题：`fdnet_gate_dual` 训练于 **512**，而 OASBUD 只物化过 256 →
#   `preproc_for` 的护栏直接抛错，`--full-card` 无法完成。
#   （实测：resnet(256) 冒烟 AUC=0.8818 ✅；fdnet(512) 冒烟直接 ValueError ❌）
#
#   为什么不能把 OASBUD_MATERIALIZE_SIZE 改成 512：它是**单一常量**，
#   被 6 个 256 变体共用 → 改了会让那 6 个全部报错。
#   为什么不能"排除 fdnet"：`cmd_outer_eval` 要求模型集合与冻结权重严格一致，
#   而权重是在 **49 个模型**上拟合的 → 少 7 个 fdnet 会让 --outer-eval ×3
#   全部退 1。故必须让 fdnet **真的**按 512 评出来。
#
#   做法：物化到**独立目录** `OASBUD_p4_<size>`（256 那份**原封不动**，
#   故"重新物化导致与 dev 数不可比"的风险不发生），按 size 选根目录。
def oasbud_root(size: int | None = None) -> Path:
    """按物化尺寸给出 OASBUD 根目录。缺该尺寸的物化即**报错**（不静默回退）。"""
    size = int(size or OASBUD_MATERIALIZE_SIZE)
    if size == OASBUD_MATERIALIZE_SIZE:
        return OASBUD_DIR
    cand = OASBUD_DIR.parent / f"OASBUD_p4_{size}"
    if not (cand / "corrected" / "manifest.csv").exists():
        raise FileNotFoundError(
            f"骨干训练分辨率为 {size}，但找不到该尺寸的 OASBUD 物化：{cand}\n"
            f"  须先物化（**非破坏性**，不覆盖 {OASBUD_MATERIALIZE_SIZE} 那份）：\n"
            f"    python scripts/materialize_oasbud.py --size {size} --out {cand}\n"
            f"  ——拒绝静默重采样（那是训练/测试分辨率错配）。")
    return cand

# 正式评估的三个臂（P0-3：一次进程内全部跑完，这才是一次"单次评估"）
FULL_CARD_ARMS = [("oasbud", "corrected"), ("oasbud", "uncorrected"),
                  ("breast", "")]


# ---------------------------------------------------------------------------
# 审计（P0-4）
# ---------------------------------------------------------------------------
def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def code_fingerprint() -> dict:
    """代码指纹。本项目**不是 git 仓库**（实测），故退回 `_code_hash`。

    仍然记录 `git` 字段（若将来入了版本控制，无需改脚本即自动生效），
    并显式标注 dirty 状态 —— 与 F22「运行期代码不可变」的检测同源。
    """
    import subprocess
    out = {"git": None, "code_hash": None, "dirty": None}
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            head = r.stdout.strip()
            st = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, timeout=20)
            out["git"] = head
            out["dirty"] = bool(st.stdout.strip())
    except Exception:
        pass
    try:
        from breastdg.train.dev_screen import _code_hash
        out["code_hash"] = _code_hash()
    except Exception:
        pass
    return out


def env_fingerprint() -> dict:
    """环境锁定（P0-4）：库版本 + requirements.lock.txt 的 SHA256。"""
    env = {"python": sys.version.split()[0]}
    for mod in ("torch", "torchvision", "numpy", "sklearn", "scipy", "PIL"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:
            env[mod] = "(缺失)"
    lock = Path(__file__).resolve().parents[1] / "env" / "requirements.lock.txt"
    env["requirements_lock_sha256"] = sha256_file(lock) if lock.exists() else None
    try:
        env["cuda"] = str(torch.version.cuda)
        env["cudnn"] = str(torch.backends.cudnn.version())
    except Exception:
        pass
    return env


# ---------------------------------------------------------------------------
# 数据：OASBUD 双管线从物化 manifest 读；BrEaST 走 registry
# ---------------------------------------------------------------------------
def load_oasbud(arm: str = "corrected", size: int | None = None) -> list[dict]:
    """读物化 manifest → entries。`size` 指定物化尺寸（默认 256 那份）。

    ⚠ 两处上云必炸（2026-10-06 修）：
    1. 原用**相对路径** `Path("data/OASBUD_p4")` —— 只在 cwd=project/ 时成立。
       云上 cwd=/app 但数据挂在 /data，必须走 `PROJECT_ROOT`/`DATA_ROOT`。
       （物化产物当前位于 `project/data/`，故此处锚到 `OASBUD_DIR`。）
    2. manifest 的 `path` 列是 **Windows 反斜杠**（`data\\OASBUD_p4\\...`），
       Linux 上 `Path()` 会把它当成**单个文件名**，读图直接 FileNotFound。
       故改为「取 basename → 拼到已知根」。

    ⚠ 2026-10-08：`size` 参数为双尺寸支持所加。两个尺寸的**文件名与
      除 path 外的全部元数据逐项相同**（已实测：corrected 清单 sha256 相同、
      lesion_id/label/view/birads/src_id 全等），故簇键与标签口径不随尺寸变。
    """
    root = oasbud_root(size)
    man = root / arm / "manifest.csv"
    if not man.exists():
        raise FileNotFoundError(f"未物化：{man}（先跑 scripts/materialize_oasbud.py）")
    img_root = root / arm / "images"
    out = []
    for r in csv.DictReader(open(man, encoding="utf-8")):
        fn = Path(str(r["path"]).replace("\\", "/")).name   # 见上 ⚠2
        p = img_root / fn
        if not p.exists():
            raise FileNotFoundError(f"manifest 指向的图不存在：{p}")
        # lesion_id 含 rec_idx 前缀（`000_1mm`），已正确区分 OASBUD 中
        # 复用同一 id 字符串的两个病灶（F23：'30nh'）——直接用，不再另造簇键。
        out.append(dict(dataset="oasbud", image=str(p), label=int(r["label"]),
                        patient_id=f"oasbud-{r['lesion_id']}",
                        lesion_id=f"oasbud-{r['lesion_id']}", masks=[],
                        birads=r["birads"], view=int(r["view"]),
                        src_id=r["src_id"]))
    return out


def load_cohort(cohort: str, arm: str = "") -> list[dict]:
    if cohort == "oasbud":
        return load_oasbud(arm or "corrected")
    if cohort == "breast":
        return registry.breast_entries()
    if cohort == "busi":
        return dedupe_entries(registry.busi_entries())[0]
    raise ValueError(f"未知队列 {cohort}")


# ---------------------------------------------------------------------------
# cluster bootstrap AUC CI（簇键=病灶，F23）
# ---------------------------------------------------------------------------
def cluster_bootstrap_auc(y, p, clusters, B=2000, seed=SEED_BOOTSTRAP,
                          alpha=0.05):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p); cl = np.asarray(clusters)
    uniq = np.unique(cl)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(cl == d) for d in draw])
        yy, pp = y[idx], p[idx]
        if len(set(yy.tolist())) < 2:
            continue
        vals.append(roc_auc_score(yy, pp))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def point_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(int); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / len(y) if len(y) else float("nan")
    return dict(thr=thr, sens=sens, spec=spec, ppv=ppv, npv=npv, acc=acc,
                tp=tp, fp=fp, tn=tn, fn=fn)


# ⚠ youden_threshold / sens_at_threshold **仅供 freeze_thresholds.py 在源域上调用**。
#   本文件（评估）**不再调用它们** —— 见文件头 P0-1。保留定义只为复用实现，
#   并由下面的 assert 保证评估路径不会误用。
def youden_threshold(y, p) -> float:
    """Youden 最优阈值。**只允许在源域验证集上调用**（F7.1）。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def sens_at_threshold(y, p, target: float = 0.90) -> float:
    """达到目标敏感度的最保守阈值。**只允许在源域验证集上调用**（F7.1）。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    ok = tpr >= target
    if not ok.any():
        return float(thr[0])
    return float(thr[np.argmax(ok)])


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def find_ckpts(strict: bool = True) -> list[tuple[str, int, Path]]:
    """找锁箱检查点。`strict=True` 时，同 seed 下多个 run 目录**报错**（P1-b）。

    原实现 `cands[-1]` 按 mtime 静默挑最新 —— 那是**出处歧义**：同一 seed
    存在两个 run（例如配方改过又重跑）时，评估卡无法交代用了哪一个。
    宁可报错让人来定，不可静默挑。
    """
    out = []
    if not LOCKBOX.exists():
        return out
    for vdir in sorted(LOCKBOX.iterdir()):
        if not vdir.is_dir() or vdir.name.startswith("_"):
            continue
        for sdir in sorted(vdir.glob("seed*")):
            seed = int(sdir.name.replace("seed", ""))
            cands = sorted(sdir.glob("*/final.pt"))
            if len(cands) > 1 and strict:
                raise RuntimeError(
                    f"出处歧义：{vdir.name}/seed{seed} 下有 {len(cands)} 个 run 目录"
                    f"都含 final.pt：\n    "
                    + "\n    ".join(str(c.parent.name) for c in cands)
                    + "\n  **拒绝静默挑**（原实现按 mtime 挑最新）。请人工确认："
                      "归档多余目录，或用 --ckpt 显式指定。")
            if cands:
                out.append((vdir.name, seed, cands[0]))
    return out


def load_thresholds(ckpt: Path, required: bool = True) -> dict | None:
    """读**同目录**的源域冻结阈值（P0-1）。缺失即拒绝执行。"""
    p = ckpt.parent / "thresholds.json"
    if not p.exists():
        if required:
            print(f"\n✗ 缺源域冻结阈值：{p}\n"
                  f"  F7.1 要求阈值在**源域验证集**选定后冻结，"
                  f"绝不在外部集上调阈。\n"
                  f"  先跑：python scripts/freeze_thresholds.py --ckpt {ckpt}\n"
                  f"  —— 拒绝执行（不静默回退到在外部集上现算）。", file=sys.stderr)
            sys.exit(EXIT_NO_THRESHOLDS)
        return None
    t = json.loads(p.read_text(encoding="utf-8"))
    # 阈值必须与检查点配对（防止换了检查点忘了重冻）
    cur = sha256_file(ckpt)
    if t.get("ckpt_sha256") and t["ckpt_sha256"] != cur:
        print(f"\n✗ 阈值与检查点不匹配：{p}\n"
              f"  thresholds.ckpt_sha256 = {t['ckpt_sha256'][:16]}…\n"
              f"  实际检查点 sha256      = {cur[:16]}…\n"
              f"  说明该阈值不是为这个检查点冻的 → 拒绝执行。", file=sys.stderr)
        sys.exit(EXIT_NO_THRESHOLDS)
    return t


def preproc_for(cohort: str, saved_cfg: dict) -> PreprocConfig:
    """按队列给出**正确**的预处理配置。

    ⚠ 2026-10-08：**实现已上移到 `breastdg.data.cohorts.preproc_for_cfg`**
    （全库唯一出处）。本函数保留为薄封装，只为不动既有调用点与冻结契约。
    上移的理由：口径此前**定义了两遍** —— 评估路径用本函数，
    而 `gradcam_interpretability.py` 自己另造 `PreprocConfig`。两者对**原图**
    队列恰好一致，故一直没暴露；但遇到**预物化成品**队列（OASBUD）就会分叉
    （评估路径知道要 identity，gradcam 那条会二次归一化）。
    现在两边都调 `preproc_for_cfg`，不可能再漂移。

    尺寸口径（回答"要不要为每个数据集改一次"）：
      **尺寸由检查点决定，与队列无关。** 原图队列直接在原像素上重采样到
      `saved_cfg["image_size"]`，没有"物化尺寸"这回事 → 512 的骨干喂 BrEaST
      天然正确（实测 `fdnet(512) × BrEaST` 零改动跑通，AUC=0.8229）。
      只有预物化队列才有"物化尺寸必须等于骨干尺寸"的约束，目前仅 OASBUD。
    """
    from breastdg.data.cohorts import preproc_for_cfg
    return preproc_for_cfg(cohort, saved_cfg)


def cluster_keys(entries: list[dict]) -> list[str]:
    """每个条目的**病灶级簇键**（字符串，可直接落 CSV）。

    回退链 `lesion_id → patient_id → image` 是**唯一**的簇键定义。
    下游凡是要"按病灶归组"的地方都必须走这里，不得另写一套 ——
    否则同一批数据会有两套簇口径（P1-e 就是这么来的）。

    ⚠ 2026-10-07 增（P1② 的加固版）：`write_arm_artifacts` 原先直接写
    `e.get("lesion_id","")`，即**原始字段**而不是**解析后的簇键**。
    举一个会真的坏掉的场景：若某队列的条目只有 `patient_id` 而无
    `lesion_id`，CSV 会写出一列空串 → `ensemble_and_calibrate.load_probs`
    拿空串当簇键 → **全队列归一簇 → CI 宽度为 0 的假精确**。
    （实测 BrEaST 当前 252 条**都有** `lesion_id`，故这个具体故障**没有发生**；
    但空串会静默产出"看起来更显著"的 CI，代价不对称，故照样加固。）
    """
    return [str(e.get("lesion_id") or e.get("patient_id") or e["image"])
            for e in entries]


def lesion_clusters(entries: list[dict]) -> np.ndarray:
    """**病灶级**簇键的整数编码（F23）。这是 cluster bootstrap 的正确簇单位。

    ⚠ 2026-10-06 修（原为**图像级**，会低估 CI）：
    原实现直接拿 DataLoader 的 `idx`（= 条目下标 0…n-1）当簇键。
    对 OASBUD 那是 200 个**图像**簇，而同一病灶的两个射频切面**不可拆**
    （F23）—— 正确簇数是 100（病灶）。用图像级簇会把簇内相关当独立样本，
    **系统性缩小** CI 宽度，即把"不够显著"当"显著"。
    现按簇键归组，与 entries 顺序一一对应（loader shuffle=False，故下标对齐）。
    """
    ids = cluster_keys(entries)
    uniq = {v: i for i, v in enumerate(dict.fromkeys(ids))}
    return np.array([uniq[v] for v in ids], dtype=np.int64)


def n_clusters(entries: list[dict]) -> int:
    """与 `lesion_clusters` **同一逻辑**的簇数（P1-e）。

    原实现另写了一套 `len(set(lesion_id))`，对**没有 lesion_id 键**的队列
    （如某些 registry 返回）会把全部归为 1 簇而**误报**。现直接复用簇键。
    """
    return int(lesion_clusters(entries).max()) + 1 if entries else 0


def eval_one(ckpt: Path, entries: list[dict], cohort: str, arm: str,
             thresholds: dict | None, smoke: bool = False) -> dict:
    sd = torch.load(ckpt, map_location="cpu")
    saved_cfg = sd.get("cfg")
    if not saved_cfg:
        # P1-c：静默默认 resnet50_cbam 是**出处黑洞**。非冒烟拒绝执行。
        if not smoke:
            raise RuntimeError(
                f"{ckpt} 缺 cfg —— 无法确定模型与预处理口径。"
                f"拒绝静默默认 resnet50_cbam（出处黑洞）。")
        print(f"    ⚠ 冒烟模式：{ckpt.name} 缺 cfg，按 resnet50_cbam/256 处理")
        saved_cfg = {"model": "resnet50_cbam", "image_size": 256}
    model = build_model(saved_cfg)
    model.load_state_dict(sd["model"])
    pp = preproc_for(cohort, saved_cfg)

    ld = make_loader(entries, batch_size=64, train=False, with_label=True,
                     cfg=pp, num_workers=6)
    model.eval().to(DEV)
    probs, labels, idxs = [], [], []
    with torch.no_grad():
        for batch in ld:
            x, y, _, idx, _ = batch
            logits = model(to_device_uint8(x.to(DEV)))
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            labels.append(y.numpy()); idxs.append(idx.numpy())
    p = np.concatenate(probs); y = np.concatenate(labels); ix = np.concatenate(idxs)
    del model
    torch.cuda.empty_cache()

    # 按 loader 下标归位（shuffle=False，故顺序恒等；显式排序以防万一）
    order = np.argsort(ix)
    p, y = p[order], y[order]

    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")
    # 簇键 = **病灶**（不是图像），见 lesion_clusters 的说明
    cl = lesion_clusters(entries)
    lo, hi = cluster_bootstrap_auc(y, p, cl)

    # ---- 三个运行点（F6 / F7.1）----
    # ⚠ P0-1：**只有 fixed_0.5 允许在外部集上算**（它就是个常数）。
    #   另两个必须来自源域冻结的 thresholds.json。
    runs = {"fixed_0.5": point_metrics(y, p, 0.5)}
    thr_src = None
    if thresholds is not None:
        thr_src = {"youden": float(thresholds["thr_youden"]),
                   "sens90": float(thresholds["thr_sens90"])}
        for nm, t in thr_src.items():
            runs[nm] = point_metrics(y, p, t)
    return dict(cohort=cohort, arm=arm, n=int(len(y)),
                n_clusters=n_clusters(entries),
                prevalence=float(np.mean(y)),
                auc=auc, auc_abs=abs(auc - 0.5) + 0.5,
                auc_ci_lo=lo, auc_ci_hi=hi,
                auc_ci_seed=SEED_BOOTSTRAP,
                runs=runs,
                thr_source=("源域冻结 thresholds.json" if thr_src
                            else "（冒烟：未读阈值，仅 fixed_0.5）"),
                thr_frozen=thr_src,
                ckpt=str(ckpt),
                ckpt_sha256=sha256_file(ckpt),
                cfg=saved_cfg,
                probs=p.tolist(), labels=y.tolist())


# ---------------------------------------------------------------------------
def run_arm(cohort: str, arm: str, ckpts, smoke: bool) -> list[dict]:
    """跑一个臂（一个队列 × 一条管线），返回逐检查点结果。"""
    ents = load_cohort(cohort, arm)
    print(f"\n[{cohort}{'/' + arm if arm else ''}]  图 {len(ents)}  "
          f"病灶簇 {n_clusters(ents)}  检查点 {len(ckpts)} 个", flush=True)
    rows = []
    for variant, seed, ck in ckpts:
        thr = load_thresholds(ck, required=not smoke)
        print(f"  [{variant}/seed{seed}]", flush=True)
        r = eval_one(ck, ents, cohort, arm, thr, smoke=smoke)
        r["variant"] = variant; r["seed"] = seed
        rows.append(r)
        extra = (f"  youden={r['runs']['youden']['sens']:.3f}/{r['runs']['youden']['spec']:.3f}"
                 f"  sens90={r['runs']['sens90']['sens']:.3f}/{r['runs']['sens90']['spec']:.3f}"
                 if "youden" in r["runs"] else "  （仅 fixed_0.5）")
        print(f"    AUC={r['auc']:.4f} [{r['auc_ci_lo']:.4f},{r['auc_ci_hi']:.4f}]"
              f"  prev={r['prevalence']:.3f}{extra}", flush=True)
    return rows


def write_arm_artifacts(rows: list[dict], ents: list[dict], tag: str) -> Path:
    """写逐图概率 CSV（冻结格式）+ 评估卡（含审计字段，P0-4）。"""
    csv_p = OUT / f"probs_{tag}.csv"
    # 簇键走与 cluster bootstrap **同一**回退链（见 cluster_keys），
    # 不写原始字段 —— 否则下游按空串归组会得到 CI 宽度为 0 的假精确。
    ck = cluster_keys(ents)
    assert len(ck) == len(ents)
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "seed", "image", "prob", "label", "lesion_id"])
        for r in rows:
            assert len(r["probs"]) == len(ents), (
                f"{r['variant']}/seed{r['seed']} 概率数 {len(r['probs'])} "
                f"≠ 条目数 {len(ents)} —— 下标错位，拒绝写出可能错配的 CSV")
            for e, pr, c in zip(ents, r["probs"], ck):
                w.writerow([r["variant"], r["seed"], Path(e["image"]).name,
                            f"{pr:.6f}", int(e["label"]), c])
    # 概率列不进 JSON（体积大且 CSV 已有）；其余全留
    slim = [{k: v for k, v in r.items() if k not in ("probs", "labels")}
            for r in rows]
    card = {
        "tag": tag,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit": {                                   # P0-4 三件套
            "code": code_fingerprint(),
            "env": env_fingerprint(),
            "checkpoints": {f"{r['variant']}/seed{r['seed']}":
                            {"path": r["ckpt"], "sha256": r["ckpt_sha256"]}
                            for r in rows},
        },
        "thresholds_policy": ("youden / sens90 全部来自**源域**冻结的 "
                              "thresholds.json；外部集上只跑 fixed_0.5（F7.1）"),
        "bootstrap_seed": SEED_BOOTSTRAP,
        "rows": slim,
    }
    json_p = OUT / f"evalcard_{tag}.json"
    json_p.write_text(json.dumps(card, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    # 卡自身 SHA256 写进旁路文件（不进卡内，避免自指）
    (OUT / f"evalcard_{tag}.sha256").write_text(
        sha256_file(json_p) + "\n", encoding="utf-8")
    print(f"\n  已写入 {csv_p.name} / {json_p.name} / "
          f"evalcard_{tag}.sha256（{len(rows)} 行）")
    return json_p


def write_summary(all_rows: dict) -> Path:
    """汇总 CSV（P0-3）：一次评估的全部臂 × 全部检查点，一行一格。"""
    p = OUT / "summary_full_card.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "variant", "seed", "n", "n_clusters", "prevalence",
                    "auc", "auc_abs", "auc_ci_lo", "auc_ci_hi",
                    "sens@0.5", "spec@0.5",
                    "sens@youden", "spec@youden",
                    "sens@sens90", "spec@sens90"])
        for arm, rows in all_rows.items():
            for r in rows:
                g = lambda k, q: r["runs"].get(k, {}).get(q, "")
                w.writerow([arm, r["variant"], r["seed"], r["n"], r["n_clusters"],
                            f"{r['prevalence']:.4f}", f"{r['auc']:.4f}",
                            f"{r['auc_abs']:.4f}", f"{r['auc_ci_lo']:.4f}",
                            f"{r['auc_ci_hi']:.4f}",
                            f"{g('fixed_0.5','sens'):.4f}" if g('fixed_0.5','sens') != "" else "",
                            f"{g('fixed_0.5','spec'):.4f}" if g('fixed_0.5','spec') != "" else "",
                            f"{g('youden','sens'):.4f}" if g('youden','sens') != "" else "",
                            f"{g('youden','spec'):.4f}" if g('youden','spec') != "" else "",
                            f"{g('sens90','sens'):.4f}" if g('sens90','sens') != "" else "",
                            f"{g('sens90','spec'):.4f}" if g('sens90','spec') != "" else ""])
    print(f"  已写入 {p.name}")
    return p


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None, choices=["oasbud", "breast", "busi"],
                    help="单臂模式；正式评估请用 --full-card")
    ap.add_argument("--arm", default="", help="仅 OASBUD 用：corrected / uncorrected")
    ap.add_argument("--full-card", action="store_true",
                    help="**正式单次评估**：一次进程内跑完 OASBUD 双臂 + BrEaST，"
                         "出完整评估卡 + 汇总 CSV（P0-3）")
    ap.add_argument("--ckpt", default=None,
                    help="只评这一个检查点（**冒烟用**：可指向 dev 检查点，"
                         "从而在不触碰锁箱的前提下验证整条管线）。"
                         "给了它即强制冒烟语义，**必须**同时给 --out-tag")
    ap.add_argument("--out-tag", default=None,
                    help="产物标签；**给了它即为冒烟**（不受单次护栏限制、"
                         "不要求 thresholds.json）")
    a = ap.parse_args()

    # ⚠ 2026-10-07 修（P1⑤）：`--ckpt` 必须强制冒烟语义。
    #   原判定是 `smoke = bool(a.out_tag)`，于是
    #     `--ckpt <dev检查点> --cohort oasbud`（忘了加 tag）
    #   会按**正式**执行：写出正式文件名的 `evalcard_oasbud_corrected.json`。
    #   后果是双重的 —— ① 用 dev 检查点**烧掉唯一一次执行名额**；
    #   ② 真正的 `--full-card` 从此被 exit 3 永久挡住。
    #   故这里把「用了 --ckpt」与「冒烟」绑死，且要求 tag 显式存在，
    #   使产物名一定带冒烟标记、绝不会占用正式名额。
    if a.ckpt and not a.out_tag:
        ap.error("给了 --ckpt 就必须同时给 --out-tag：\n"
                 "  --ckpt 指到 dev 检查点即为冒烟，必须让产物名带冒烟标记，\n"
                 "  否则会写出**正式文件名**、烧掉唯一一次锁箱执行名额\n"
                 "  （并把真正的 --full-card 挡在 exit 3 外）。\n"
                 "  例：--ckpt runs/dev_screen/.../final.pt --out-tag _smoke")
    smoke = bool(a.out_tag) or bool(a.ckpt)
    if a.ckpt:
        ck = Path(a.ckpt)
        if not ck.is_absolute():
            ck = config.PROJECT_ROOT / ck
        if not ck.exists():
            print(f"✗ 检查点不存在：{ck}"); return EXIT_ERR
        ckpts = [("smoke", 0, ck)]
    else:
        ckpts = find_ckpts()          # 出处歧义时在此抛错（P1-b）
        if not ckpts:
            print("未找到锁箱检查点（先跑 scripts/_lockbox_driver.py）")
            return EXIT_ERR

    # ---- 决定要跑哪些臂 ----
    # ⚠ 2026-10-07 修（check3 P2②，与 P1⑤ 同源）：
    #   正式（非冒烟）单臂执行会**消耗单次名额** —— `--cohort oasbud`（无 tag）
    #   写出正式名 `evalcard_oasbud_corrected.json` 之后，真正的
    #   `--full-card` 就被 exit 3 **永久挡住**了，而锁箱评估不可重做。
    #   故正式模式**只接受 --full-card**；单臂仅限冒烟（必须带 --out-tag）。
    if a.full_card:
        arms = FULL_CARD_ARMS
    elif a.cohort:
        if not smoke:
            ap.error(
                "正式模式只接受 --full-card。\n"
                "  单臂 --cohort 会写出**正式文件名**的评估卡、消耗掉唯一一次\n"
                "  执行名额，从而把 --full-card 永久挡在 exit 3 之外。\n"
                "  单臂仅供冒烟：请加 --out-tag <标记>（产物名会带该标记，\n"
                "  不占用正式名额）。例：\n"
                "    --cohort oasbud --arm corrected --out-tag _smoke")
        arms = [(a.cohort, a.arm)]
    else:
        ap.error("必须给 --full-card（正式）或 --cohort（仅冒烟，需 --out-tag）")

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- P0-2：正式执行的两道护栏 ----
    def tag_for(cohort: str, arm: str) -> str:
        """臂 → 产物标签。

        ⚠ 2026-10-07 修：原为 `a.out_tag or 默认名`，于是
        `--full-card --out-tag X` 会让**三个臂共用同一个 tag X**，
        `probs_X.csv` 被后者覆盖前者 —— 静默丢两个臂的逐图概率，
        而汇总 CSV 看起来"跑完了"。现改为**后缀**：`--out-tag` 只在末尾附加，
        默认名恒为 `oasbud_corrected` / `oasbud_uncorrected` / `breast`。
        默认名**必须**保持不变 —— `e3_polarity_analysis.py` 与
        `ensemble_and_calibrate.py` 都按 `probs_oasbud_corrected.csv` 读；
        加前缀会同时改动正式产物名，那是冻结契约。
        """
        base = f"{cohort}_{arm}" if arm else cohort
        return f"{base}{a.out_tag}" if a.out_tag else base

    if not smoke and not a.ckpt:
        existing = [OUT / f"evalcard_{tag_for(c, m)}.json" for c, m in arms]
        hit = [p for p in existing if p.exists()]
        if hit:
            print("\n✗ 已存在评估卡，拒绝重跑（锁箱**单次**纪律）：\n    "
                  + "\n    ".join(str(p) for p in hit)
                  + "\n  锁箱评估是不可逆的一次性操作。若确实要重做，"
                    "请先人工归档或删除既有卡（并记录原因）。\n"
                    "  ——拒绝执行。", file=sys.stderr)
            return EXIT_ALREADY_EXISTS

    print("=" * 78)
    print(f"锁箱评估  {'**正式单次（--full-card）**' if a.full_card and not smoke else '冒烟'}")
    print(f"  臂：{arms}")
    print(f"  检查点 {len(ckpts)} 个  自举种子 {SEED_BOOTSTRAP}")
    print(f"  阈值策略：youden / sens90 读源域冻结 thresholds.json"
          f"{'（冒烟：不读、只跑 fixed_0.5）' if smoke else ''}")
    print("=" * 78)

    # 正式执行全程持锁（P0-2）；与训练共用同一 RunLock（F22）
    from breastdg.train.dev_screen import RunLock
    all_rows: dict[str, list[dict]] = {}
    lock_ctx = RunLock() if (not smoke) else None
    if lock_ctx is not None:
        lock_ctx.__enter__()
    try:
        for cohort, arm in arms:
            tag = tag_for(cohort, arm)
            rows = run_arm(cohort, arm, ckpts, smoke)
            all_rows[tag] = rows
            ents = load_cohort(cohort, arm)
            write_arm_artifacts(rows, ents, tag)
        if a.full_card:
            write_summary(all_rows)
    finally:
        if lock_ctx is not None:
            lock_ctx.__exit__(None, None, None)

    print("\n完成。" + ("" if smoke else
                       " 本次即为**登记对象**；卡文件 SHA256 见 evalcard_*.sha256。"))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
