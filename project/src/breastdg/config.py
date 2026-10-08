"""集中式路径与常量。

数据根目录默认指向本机位置，可用环境变量 BREASTDG_DATA 覆盖。
本包只登记公开数据集；任何非公开队列不得出现在代码、配置、日志或路径中
（边界见 docs/freeze_log.md 的 F9 条）。
"""
from __future__ import annotations

import os
from pathlib import Path


def _data_root() -> Path:
    env = os.environ.get("BREASTDG_DATA")
    if env:
        return Path(env)
    return Path(r"C:/Users/15315/Desktop/paper1st/data")


DATA_ROOT: Path = _data_root()
PKG_ROOT: Path = Path(__file__).resolve().parents[1]   # src/breastdg
PROJECT_ROOT: Path = PKG_ROOT.parents[0]               # project/  (含代码与全部产物)

# ---- 公开数据集 ----
BUSI_DIR: Path = DATA_ROOT / "busi" / "Dataset_BUSI_with_GT"
BUSBRA_DIR: Path = DATA_ROOT / "BUSBRA"
BUSUCLM_DIR: Path = DATA_ROOT / "BUSI-UCLM"
OASBUD_MAT: Path = DATA_ROOT / "OASBUD.mat"
BREAST_DIR: Path = DATA_ROOT / "BrEaST"

def _runs_dir() -> Path:
    """产物根目录。默认 `PROJECT_ROOT/runs`，可用 `BREASTDG_RUNS` 覆盖。

    **为什么需要这个开关（2026-10-06 加）**：云上要求「数据与产物只落数据盘，
    系统盘只放镜像」——AutoDL 的自定义镜像**只打包系统盘**，故只要产物写在
    数据盘，就**物理上**保证镜像不含数据/结果。而 `PROJECT_ROOT` 由代码位置
    推导（云上是 `/app`，属系统盘），**无法用 `BREASTDG_DATA` 影响** ——
    原实现会把整批检查点写进系统盘，实例一释放即丢，且镜像一打包就带出去。
    故此处给产物根一个独立的覆盖点。
    """
    env = os.environ.get("BREASTDG_RUNS")
    return Path(env) if env else (PROJECT_ROOT / "runs")


# ---- 输出 ----
DOCS_DIR: Path = PROJECT_ROOT / "docs"
SPLITS_DIR: Path = PROJECT_ROOT / "splits"
RUNS_DIR: Path = _runs_dir()
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

# OASBUD 物化产物目录（含 P4 几何与色调归一化的成品图）。
# 物化是**派生数据**，可由 scripts/materialize_oasbud.py 重建，故不属于
# "数据盘必须有的原始数据"；但云上同样不该落系统盘。
# 解析顺序：显式环境变量 → 数据盘 → 本机历史位置（保持本机行为不变）。
def _oasbud_p4_dir() -> Path:
    env = os.environ.get("BREASTDG_OASBUD_P4")
    if env:
        return Path(env)
    on_data = DATA_ROOT / "OASBUD_p4"
    if on_data.exists():
        return on_data
    return PROJECT_ROOT / "data" / "OASBUD_p4"


OASBUD_P4_DIR: Path = _oasbud_p4_dir()

# ---- 常量 ----
SEEDS: tuple[int, ...] = (42, 123, 2026)
IMAGE_SIZE: int = 256
LABEL_BENIGN, LABEL_MALIGNANT = 0, 1

# 任务定义：来源 BUS-UCLM 的 419 张 normal 图（无病灶）不参与，见 docs/freeze_log.md
BUSUCLM_USE_LESIONS_ONLY: bool = True

for _d in (DOCS_DIR, SPLITS_DIR, RUNS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
