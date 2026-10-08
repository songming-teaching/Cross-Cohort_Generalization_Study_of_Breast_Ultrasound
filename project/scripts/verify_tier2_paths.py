"""tier2 新代码路径的语义验证（不是"能否跑通"，而是"是否真的生效"）。

动机（F26 教训）：FDNet 的冒烟测试**跑通了**，但 GRL 从未生效——
症状只在诊断量上可见（dom loss 趋 0 而非停在 ln2）。
故对新路径做**语义**验证，而非仅形状/异常检查：

  ema       —— 滑动权重是否真的偏离原始权重？final.pt 是否存的是 EMA 权重？
  scale_aug —— 是否真的改变图像、且尺度在各域内随机（去相关的必要条件）？
  render_aug + scale_aug —— 两者是否都生效（不是相互覆盖）？
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "src")

from breastdg.data import registry
from breastdg.data.dedup import dedupe_entries
from breastdg.data.preprocess import PreprocConfig
from breastdg.train.datasets import BreastDataset
from breastdg.train.engine import EMA

FAIL = []


def check(name, cond, detail=""):
    ok = bool(cond)
    if not ok:
        FAIL.append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ---------------------------------------------------------------------------
print("=== 1) EMA：滑动权重是否真的生效 ===")
m = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8))
init = {k: v.clone() for k, v in m.state_dict().items()}
ema = EMA(m, decay=0.9)
# 模拟"训练"：把权重推向一个已知的偏移
for step in range(50):
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.1)
    ema.update(m)
shadow = ema.shadow
diff_raw = max(float((m.state_dict()[k] - init[k]).abs().max())
               for k in init if init[k].dtype.is_floating_point)
diff_ema = max(float((shadow[k] - init[k]).abs().max())
               for k in init if init[k].dtype.is_floating_point)
print(f"    原始权重已移动 {diff_raw:.3f}；EMA 影子移动 {diff_ema:.3f}")
check("EMA 影子偏离初始权重", diff_ema > 0.01, f"({diff_ema:.3f})")
check("EMA 影子滞后于原始权重", diff_ema < diff_raw, f"({diff_ema:.3f} < {diff_raw:.3f})")

# copy_to 是否真的改写模型
before = m.state_dict()[list(init)[0]].clone()
ema.copy_to(m)
after = m.state_dict()[list(init)[0]]
check("copy_to 改写模型权重", not torch.allclose(before, after))

# ---------------------------------------------------------------------------
print("\n=== 2) scale_aug：是否真的改变图像、且尺度随机 ===")
busi, _ = dedupe_entries(registry.busi_entries())
ents = busi[:40]
cfg = PreprocConfig(size=256, geometry="p4")


def hf_energy(x: np.ndarray) -> float:
    """高频能量（Laplacian 方差）——尺度的直接代理。"""
    c = x.astype(np.float64)
    lap = np.zeros_like(c)
    lap[1:-1, 1:-1] = (4 * c[1:-1, 1:-1] - c[:-2, 1:-1] - c[2:, 1:-1]
                       - c[1:-1, :-2] - c[1:-1, 2:])
    return float(lap.var())


ds_off = BreastDataset(ents, cfg, train=True, scale_aug=False)
ds_on = BreastDataset(ents, cfg, train=True, scale_aug=True)
e_off = np.array([hf_energy(ds_off[i][0].numpy()[0]) for i in range(20)])
e_on = np.array([hf_energy(ds_on[i][0].numpy()[0]) for i in range(20)])
print(f"    高频能量  关闭 {e_off.mean():8.1f}±{e_off.std():7.1f}   "
      f"开启 {e_on.mean():8.1f}±{e_on.std():7.1f}")
check("scale_aug 改变了图像", not np.allclose(e_off, e_on),
      f"(均值比 {e_on.mean()/max(e_off.mean(),1e-9):.3f})")
check("尺度在同域内随机（去相关的必要条件）", e_on.std() > 1.0,
      f"(SD {e_on.std():.1f})")

# ---------------------------------------------------------------------------
print("\n=== 3) render_aug + scale_aug 组合：两者是否都生效 ===")
ds_r = BreastDataset(ents, cfg, train=True, render_aug=True)
ds_rs = BreastDataset(ents, cfg, train=True, render_aug=True, scale_aug=True)
x_r = ds_r[0][0].numpy()[0].astype(np.int32)
x_rs = ds_rs[0][0].numpy()[0].astype(np.int32)
check("组合与单独 render_aug 不同", not np.array_equal(x_r, x_rs),
      f"(逐像素最大差 {int(np.abs(x_r-x_rs).max())})")

# ---------------------------------------------------------------------------
print("\n=== 4) 每条新路径的诊断量契约（供长跑后核对）===")
print("    · ema：final.pt 应存 EMA 权重，且 meta 里 kind='final_ema'")
print("    · scale_aug：训练损失不应异常（尺度扰动会略微提高 loss）")
print("    · 组合：两者独立叠加，不互相覆盖")

print("\n" + "=" * 60)
print(f"结果：{'全部通过' if not FAIL else '存在失败项 → ' + str(FAIL)}")
sys.exit(1 if FAIL else 0)
