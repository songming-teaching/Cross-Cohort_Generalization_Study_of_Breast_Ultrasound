# 归档：E7 首跑（残差臂口径错误），2026-10-06

**为什么不作废重来而要留档**：这份产物本身没错，错的是**残差臂的回归集**。

## 错误

`alignment_trajectory.py` 用 `np.column_stack(list(ca.values()))` 把
`trivial_cues` 返回的**全部**键作为回归集，这**包含了 `scale_extra`**
（speckle/grad/snr）。而 `scale_extra` 单 cue 域可分性 **0.9633** ——
它是**纹理签名的物理载体本身**，不是应被回归掉的平凡线索。

## 量化（`scripts/_chk_e7_arm.py`，no_domainhead/busbra/seed42/best_erm）

| 回归集 | AUC |
|---|---|
| 深度特征（不回归） | 0.9967 |
| 7a 上界 = geom+tone | 0.9670 |
| 7b 下界 = 7a+scale | 0.8403 |
| **全部键（含 scale_extra）← 首跑用的** | **0.6807** |
| 仅 scale_extra（佐证） | 0.9633 |

对照 F13 归因表同格：7a = 0.9685、7b = 0.8357 → 上两行与归因表**吻合**，
末行**偏离 0.155**。故首跑的 `auc_residual` 系统性偏低。

## 修法

改用与 `diagnose_domain_probe_p4.probe_rows` **逐列一致**的双口径回归集
（7a 上界 / 7b 下界），并把两个都写进产物。

**教训**：跨脚本复用同一统计量时，**回归集必须逐列对齐**，
不能靠"传入同一函数"就假定口径一致。
