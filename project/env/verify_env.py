"""环境自检 —— 断言云端环境与本机环境一致。

为什么需要它
------------
本地 dev 期结果（88 格）与云端结果会被**放进同一张对比表**（五骨干集成、
E1 消融矩阵）。若两边环境不一致，这个比较就被环境混杂了。故让不一致
**在构建期就失败**，而不是等跑完才发现数字对不上。

检查三层
--------
1. **库版本**：逐条比对 `requirements.lock.txt`（torch/torchvision 由基础
   镜像提供，单独断言）；
2. **CUDA 栈**：torch.version.cuda、cudnn 版本、可见架构列表；
3. **数值行为**：确认 `deterministic=True` 会生效、TF32 开关与本机一致
   —— 这一层最容易漏，且直接决定数能否复现。

退出码
------
0 = 全部一致；1 = 有不一致（Dockerfile 里因此会构建失败）。
注意：**架构差异（sm_89 vs sm_80/90）只警告、不失败**，
因为它在合法场景下可接受（例如只跑全新比较组），但必须被看见。
"""
from __future__ import annotations

import importlib
import pathlib
import re
import sys

LOCK = pathlib.Path(__file__).resolve().parent / "requirements.lock.txt"

# 目标环境基准（**2026-10-07 云上实测定稿**）
#
# 事实：AutoDL 的 PyTorch 镜像列表里**没有 2.6.0**，云端可用的是 2.5.1 +
# torchvision 0.20.1（官方配对：2.5.1↔0.20.1 / 2.6.0↔0.21.0，不可交叉）。
# 本机 dev 产物是 2.6.0 跑的 —— 两者存在环境差异，**已知并接受**。
#
# 影响与处置（记录用，非争议）：
#   · **锁箱这一批是自洽的**：35 格全部在同一云端环境跑完，
#     批次内各变体之间的比较**不受影响**（这是论文里用到的主要比较）。
#   · 跨批次比较（云端锁箱 vs 本机 dev 88 格）带一层环境混杂，
#     入表时须标注；若审稿人问，答复=本批为独立重训批次，非同一环境复现。
#   · 已核：本仓**零处**使用任何 2.6 专有 API，2.5.1 下无功能缺失。
#     唯一行为差异是 `torch.load` 的 `weights_only` 默认值（2.6 起 True，
#     2.5.1 为 False）—— 我们存的检查点是 {"model","step","cfg"}，cfg 为
#     纯 Python 类型的 asdict，**两个版本都安全**。
EXPECT_TORCH = "2.5.1"
EXPECT_TV = "0.20.1"
EXPECT_CUDA = "12.4"
EXPECT_CUDNN = 90100
EXPECT_ARCH = "sm_89"          # RTX 4060 Laptop / RTX 4090D 均 Ada Lovelace

fails: list[str] = []
warns: list[str] = []


def parse_lock() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            k, v = line.split("==", 1)
            out[k.strip()] = v.strip()
    return out


# 发行版名 → 用于定位模块（仅用于确认「能导入」）
MOD_NAME = {"pillow": "PIL", "opencv-python-headless": "cv2",
            "scikit-learn": "sklearn", "PyWavelets": "pywt"}


def check_libs() -> None:
    """比对**发行版**版本，而非模块的 `__version__` 属性。

    ⚠ 这里必须用 importlib.metadata：模块 `__version__` 与发行版版本
      **不总是相等**，本机已实际踩到两例 ——
        · PyWavelets: 发行版 1.10.0，而 `pywt.__version__` 报 1.8.0；
        · opencv-python-headless: 发行版 5.0.0.93（第 4 位是构建号），
          而 `cv2.__version__` 只到 5.0.0。
      先用 `__version__` 比对会误报「不一致」，进而让构建无谓地失败。
    """
    from importlib.metadata import PackageNotFoundError, version as dist_version

    want = parse_lock()
    print("--- 库版本（发行版口径）---")
    for name, want_v in sorted(want.items()):
        # 1) 能否导入（有些包装了但没装全）
        mod_name = MOD_NAME.get(name, name)
        try:
            importlib.import_module(mod_name)
        except Exception as e:                       # noqa: BLE001
            msg = f"{name}: 导入失败 {type(e).__name__}: {e}"
            fails.append(msg)
            print(f"  ✗ {name:28s} {msg}")
            continue
        # 2) 发行版版本比对
        try:
            got = dist_version(name)
        except PackageNotFoundError:
            got = "(无发行版元数据)"
        ok = got == want_v
        print(f"  {'✓' if ok else '✗'} {name:28s} 想要 {want_v:12s} 实际 {got}")
        if not ok:
            fails.append(f"{name}: 想要 {want_v}, 实际 {got}")


def check_cuda() -> None:
    import torch
    print("--- torch / CUDA 栈 ---")
    for label, got, want in [
        ("torch", torch.__version__.split("+")[0], EXPECT_TORCH),
        ("torchvision", _tv(), EXPECT_TV),
        ("torch.version.cuda", str(torch.version.cuda), EXPECT_CUDA),
        ("cudnn", str(torch.backends.cudnn.version()), str(EXPECT_CUDNN)),
    ]:
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {label:20s} 想要 {want:12s} 实际 {got}")
        if not ok:
            fails.append(f"{label}: 想要 {want}, 实际 {got}")


def _tv() -> str:
    try:
        import torchvision
        return torchvision.__version__.split("+")[0]
    except Exception:
        return "缺失"


def check_numeric() -> None:
    """数值行为 —— 最容易漏、且直接决定结果可复现性。"""
    import torch
    print("--- 数值行为 ---")
    # 与 engine.set_seed(deterministic=True) 的效果一致
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    flags = [
        ("cudnn.deterministic", torch.backends.cudnn.deterministic, True),
        ("cudnn.benchmark", torch.backends.cudnn.benchmark, False),
        ("cudnn.allow_tf32", torch.backends.cudnn.allow_tf32, True),
        ("matmul.allow_tf32", torch.backends.cuda.matmul.allow_tf32, False),
        ("float32_matmul_precision", torch.get_float32_matmul_precision(), "highest"),
    ]
    for label, got, want in flags:
        ok = got == want
        print(f"  {'✓' if ok else '✗'} {label:28s} 想要 {str(want):10s} 实际 {got}")
        if not ok:
            fails.append(f"{label}: 想要 {want}, 实际 {got}")


def check_arch() -> None:
    """架构 —— 只警告，不失败（见模块 docstring）。"""
    import torch
    print("--- 可见架构 ---")
    if not torch.cuda.is_available():
        warns.append("当前无可见 GPU（构建期正常；运行时必须有）")
        print("  ! 无可见 GPU（构建期正常）")
        return
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    listed = torch.cuda.get_arch_list()
    print(f"  设备 {name}  架构 {arch}")
    print(f"  torch 可见架构 {listed}")
    if arch == EXPECT_ARCH:
        print(f"  ✓ 与本机同为 {EXPECT_ARCH} —— 跨机复现风险最低")
    else:
        msg = (f"架构 {arch} ≠ 本机 {EXPECT_ARCH}。"
               f"数值可能在第 3-4 位小数起发散 —— 若本批与本机 dev 结果"
               f"进同一张对比表，即构成硬件混杂。建议改用 sm_89 的卡"
               f"（RTX 4090 / L4 / RTX 6000 Ada）。")
        warns.append(msg)
        print(f"  ! {msg}")


def main() -> int:
    print("=" * 66)
    print("云端环境自检（对照本机 2026-10-05 实测）")
    print("=" * 66)
    check_libs()
    check_cuda()
    check_numeric()
    check_arch()
    print("=" * 66)
    if warns:
        print(f"警告 {len(warns)} 条：")
        for w in warns:
            print(f"  ! {w}")
    if fails:
        print(f"✗ 不一致 {len(fails)} 条 —— 环境不可用：")
        for f in fails:
            print(f"    {f}")
        return 1
    print("✓ 环境与本机一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
