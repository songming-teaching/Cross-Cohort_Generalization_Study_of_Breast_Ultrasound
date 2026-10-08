# project —— 乳腺超声零样本域泛化（重构版）

所有数字由脚本产生；训练与评估只读 `splits/` 中的划分文件。

## 环境

```bash
# 项目 venv（torch 2.6.0+cu124）
../.venv/Scripts/python.exe
```

## 常用命令

```bash
cd meliorate/project
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"

# 1) 数据集账目（论文数字唯一来源）
../.venv/Scripts/python.exe -m breastdg.reports.make_table1 --json

# 2) 重建划分（一次性；已生成，勿随意重跑）
../.venv/Scripts/python.exe -m breastdg.data.splits
```

## 布局

```
src/breastdg/
  config.py              路径与常量（只登记公开数据集）
  data/
    registry.py          五个数据集的统一登记
    dedup.py             两级近重复检测（pHASH ≤2 + NCC ≥0.98）
    splits.py            patient-level 分层划分
    preprocess.py        letterbox + 逐图低自由度色调标准化
    oasbud_rf.py         OASBUD 双重建管线
  reports/
    make_table1.py       账目脚本
docs/
  freeze_log.md          冻结日志 F1–F9（决策在评估前固定）
  labels_audit.md        标签审计
splits/                  4 个划分文件（含内容哈希）
reports/                 账目输出与图
```

## 纪律

1. 论文中任何样本量必须来自 `make_table1.py` 输出。
2. 标签、划分、阈值三类决策已冻结，见 `docs/freeze_log.md`；改动须另写补遗。
3. 不得出现任何非公开队列的路径或标识（见 `docs/freeze_log.md` F9）。
4. 阈值只在源域验证集上选定；外部集不调阈值、不选模型、不选变体。
