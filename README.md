# meliorate — 乳腺超声跨队列泛化研究（代码与可复现产物）

本仓库只保留 **程序** 与 **可复现论文数字的产物**。
研究叙事见论文正文；逐项检查结果见程序里的自检与 `verify_*` 脚本。

---

## 1. 目录

| 路径 | 内容 |
|---|---|
| `project/scripts/` | 全部实验与核验脚本（唯一实现处） |
| `project/src/breastdg/` | 数据、模型、训练库 |
| `project/env/` | 环境锁定、云端/本地驱动脚本 |
| `project/splits/` | 数据划分（训练/评估只读此处） |
| `project/figures/` | 论文图件（300 dpi PNG + 同名数据 CSV） |
| `project/runs/diag/` | 判读产物（E3 / E8 / Q-5 / 两级落差 / 归因表） |
| `project/runs/lockbox_eval/` | 锁箱评估：主表 + 逐图概率 + 审计卡 |
| `project/runs/ensemble/` | 集成三口径 + 融合权重 |
| `project/runs/source_val/` | 源域概率（融合拟合的输入） |
| `e8_rmt_analysis_v2.py` | E8 谱分析模块（被 `run_e8*.py` 导入） |

## 2. 数据（不入库）

公开队列：**BUSI / BUSBRA / BUS-UCLM / OASBUD / BrEaST**。
默认根目录 `../data`，可用环境变量 `BREASTDG_DATA` 覆盖。

DANN 口径：源域 = BUSI + BUSBRA，**BUS-UCLM 恒为无标签目标域且从不进源域**；
它亦不作为外部评估集（外部评估只有 OASBUD 与 BrEaST）。

## 3. 复现

```bash
cd project
python scripts/preflight.py                 # 机械体检（先跑这个）
python scripts/verify_transfer_gap.py       # 落差 级1（dev）
python scripts/verify_lockbox_gap.py        # 落差 级2（锁箱）
python scripts/aggregate_e8_lockbox.py      # E8 谱量聚合
python scripts/q5_second_order_row.py       # Q-5 归因表第八行
python scripts/make_figures.py              # 重画全部图件
```

## 4. 边界（硬规则）

- 本仓库**不包含任何私有队列的名称、数字或统计量**。
  实验室临床队列在授权前一律当不存在。
- 阈值仅在**源域**冻结后应用（F7.1）；bootstrap 种子全库统一 `20261006`；
  簇自举分组单位是**病灶**而非图像（F23）。
- Q-5 判据在**锁箱评估之前**冻结（预登记文件见 `Q-5_归因表第八行_锁箱前预登记.md`）。

## 5. 未包含（可重跑，故意不入库）

训练中间产物（`project/runs/{dev_screen,lockbox,cloud}` 约 35 G，含 8.1 G 检查点）、
虚拟环境、原始数据。按 `env/README.md` 重跑即可再生。
