# 云算力迁移层

**状态：就绪但未启用。** 本目录是纯新增文件，不改 `src/`、不影响本机正在跑的训练。
决定是否上云、何时上云，由作者掌握。

---

## 一、先看结论：什么该上云，什么不该

### ✅ 该上云：锁箱全源重训（15 格，全新建）

| 理由 | 说明 |
|---|---|
| **零混杂** | 15 格全部重新训练，不存在"新旧混用" |
| **收益最大** | 本机 5.7 h → 云上约 1 h；费用粗算 **$2-5** |
| **顺带修性能 bug** | 数据管道优化（见 `memory/gpu-duty-cycle-bubble.md`）正好在同一批实施 |

### ❌ 不该上云：TIER4 剩余格

TIER4 这 18 格要跟**已完成的 78 格**（13 变体 × 2 方向 × 3 种子）做横向比较
（五骨干集成、E1 消融矩阵，审稿意见 4 的核心）。**只要有一格换机器，
比较就被硬件混杂。** 迁移的单位必须是"一个完整的比较组"。

---

## 二、跨机复现：精确的风险边界

本机 `engine.set_seed(deterministic=True)` → `cudnn.deterministic = True`，
`cudnn.benchmark = False`。这意味着：

- **同架构内 ↔ 完全可复现**：同架构不同卡（哪怕功率不同）跑出一模一样的数；
- **跨架构 ↗ 有风险**：`deterministic=True` 只保证"在**本架构可用的**确定性
  算法里选一个"，不保证不同架构选同一个。

本机是 **RTX 4060 Laptop = sm_89（Ada Lovelace）**。所以：

| 卡的架构 | 型号 | 建议 |
|---|---|---|
| **sm_89** | **RTX 4090 / L4 / RTX 6000 Ada** | ✅ **首选** —— 同架构 + 强 10 倍 + 便宜 |
| sm_90 | H100 | ⚠️ 架构不同，第 3-4 位小数可能不同 |
| sm_80 | A100 | ⚠️ 同上 |

**额外数值风险**：本机 `cudnn.allow_tf32 = True`。若云端反向默认（部分云镜像
通过 `NVIDIA_TF32_OVERRIDE=0` 关闭 TF32），数会不同。`env/verify_env.py`
会显式断言这一项。

---

## 三、时间与费用（粗算，价格不敏感）

| 项 | 本机 | 云（单张 4090） |
|---|---|---|
| 剩余总工作量 | 10.9 h | ~2-3.5 h |
| 仅锁箱 15 格 | 5.7 h | ~1 h |
| 费用 | 电费 ≈ ¥1.4 | **$2-25**（视档位） |

**今后 100 h 量级实验的可行性**：

$$\text{费用} \approx \frac{100\text{ h}}{3\sim6} \times \$1\sim2/\text{h} \approx \$20\sim60$$

对比本机电费约 $2 —— 云贵 15-30 倍，但 **wall-clock 的价值远高于此**（论文返修期尤甚）。
**规模越大越划算**：固定成本（容器、入口脚本、调试）只付一次。

---

## 四、迁移需要做什么（分工）

| 项 | 状态 |
|---|---|
| 依赖钉版 | ✅ `env/requirements.lock.txt`（上游此前**完全没有**依赖声明） |
| 镜像定义 | ✅ `env/Dockerfile`（基础镜像钉死 cu124/cudnn9） |
| 环境自检 | ✅ `env/verify_env.py`（三层断言，构建期失败而非事后发现） |
| Linux 启动入口 | ✅ `env/run_cloud.sh`（POSIX 版，替代 Windows 驱动） |
| 数据路径可移植 | ✅ 已就绪 —— `config.py` 本就有 `BREASTDG_DATA` 环境变量 |
| `src/` 可移植性 | ✅ 已确认无 Windows 专有调用（`subprocess` 只出现在 `scripts/`） |
| **账号 / API key** | ⬜ **需要作者** |
| **数据上传授权** | ⬜ **需要作者**（见下） |
| 云端执行与盯梢 | ⬜ 授权后由我承担 |

### ⚠ 数据上传必须作者明确授权

`data/` 为 2.1 GB，全部公开队列（BUSI / BUSBRA / BUSI-UCLM / BrEaST / OASBUD）。
但把研究数据传到第三方云是**对外动作**——可能被缓存或索引，即使之后删除。
且医学影像即便公开，不少机构仍要求先过数据合规审查（论文含伦理声明）。
**此步不由我代决。**

---

## 五、启用步骤

```bash
# 1) 构建（会跑环境自检；不一致则构建失败）
docker build -f env/Dockerfile -t breastdg:cu124 .

# 2) 冒烟（约 3 min，验证端到端；不碰锁箱）
docker run --gpus all --rm \
  -v /path/to/data:/data:ro \
  -v "$PWD/runs:/app/runs" \
  breastdg:cu124 bash /app/env/run_cloud.sh smoke

# 3) 锁箱重训（15 格）
docker run --gpus all --rm \
  -v /path/to/data:/data:ro \
  -v "$PWD/runs:/app/runs" \
  breastdg:cu124 bash /app/env/run_cloud.sh lockbox
```

**注意 `-v "$PWD/runs:/app/runs"`**：不挂出来，产物随容器消失。

---

## 六、刻意不做的事

- **不含锁箱评估**（`evaluate_lockbox.py` 跑 OASBUD/BrEaST）：不可逆的一次性
  锁箱量，须经作者确认后单次执行。`run_cloud.sh` 的 `stage` 里没有这一项。
- **不碰任何私有队列**（F9 + 记忆硬规则）。
- **不改 `src/`**：本目录全是新增文件。数据管道优化留到锁箱批次一并实施
  （那时零混杂，见裁决 R-5）。
