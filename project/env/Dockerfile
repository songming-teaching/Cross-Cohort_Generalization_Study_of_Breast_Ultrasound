# 云算力镜像 —— 与本机环境精确对齐的可复现训练环境。
#
# 设计原则（重要，请先读）
# ----------------------
# 1) **基础镜像钉死在 cu124 / cudnn9**。不要换成 cu126/cu128 的镜像 ——
#    那会改变卷积库行为（且 cu124 根本不支持 Blackwell，见下）。
#
#    ⚠ 2026-10-07 版本**回滚**：2.5.1 → **2.6.0**。
#    经过：用户以为基础镜像里没有 2.6.0，我据此降到了 2.5.1。随后查证
#    Docker Hub —— `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` **存在**
#    （digest 77f17f843507，3.09 GB；兄弟 tag 还有 cuda12.6/cuda11.8 两条）。
#    用户看到"没有"的，多半是**平台自带的镜像下拉列表**，不是 Hub 本身。
#
#    为什么必须回到 2.6.0 —— 这是 `verify_env.py` 的**契约**问题：
#    它的职责是断言「云端 == 本机」。本机实测（2026-10-07）是
#      torch 2.6.0+cu124 / torchvision 0.21.0+cu124 / cudnn 90100
#    若云端装 2.5.1，那 verify_env 就变成"断言云端 == 一个本机**没有**的版本"，
#    等于把"环境一致性"这条声明悄悄改成"环境不同"。降版本会带来真实的
#    数值口径差（尤其 cudnn kernel 选择），而本批云结果要本机 dev 结果
#    进同一张对比表 —— 不一致即构成环境混杂。
#
#    另核（无论 2.5.1 还是 2.6.0）：本仓**未使用任何 2.6 专有 API** ——
#      `torch.compile` / `_dynamo` / `torch.export` / `torch.func` /
#      `scaled_dot_product_attention` 全为 **0 处**。
#    唯一行为差异是 `torch.load` 的 `weights_only` 默认值（2.6 起 True，
#    2.5.1 为 False）—— 我们存的检查点是 {"model","step","cfg"}，cfg 为
#    dataclass 的 asdict（纯 Python 类型），**两个版本都安全**。
#    **torch 与 torchvision 必须配对**：2.6.0 ↔ 0.21.0，均为 +cu124。
#
# 2) **torch/torchvision 由镜像提供，不重新安装**（见 requirements.lock.txt
#    顶部说明）。用 scripts/verify_env.py 断言一致性 —— 换版本时**必须**
#    同步改它的 EXPECT_* 常量，否则构建期断言失败、镜像根本建不起来。
#    ⚠ 特别注意：EXPECT_* 的**唯一**依据是**本机实测值**，不是平台偏好、
#      不是"更新更好"。verify_env 的全部意义就是"云端 == 本机"。
# 3) 本镜像**不含任何数据**。数据在运行时挂载（见 run_cloud.sh），
#    且只允许公开队列（BUSI / BUSBRA / BUSI-UCLM / BrEaST / OASBUD）。
#    私有队列不得出现在本目录、镜像、日志或路径中（F9 + 记忆硬规则）。
#
# ⚠ **卡型限制（重要）**：本镜像为 cu124，只支持 sm_50~sm_90。
#    ✅ 可用：RTX 4090 / 4090D / L4 / A100 / H100 / **RTX 6000 Ada**（均 sm_89~sm_90）
#    ❌ 不可用：任何 **Blackwell** 卡（RTX PRO 6000 / RTX 6000D / B200，
#       均 sm_120）—— sm_120 需要 CUDA ≥12.8，cu124 时代码里根本没有该目标，
#       运行期会直接报 "no kernel image is available for execution"。
#    这与显存/价格无关，是**架构**问题：换 Blackwell 必须同时换 cu128 镜像
#    与 torch ≥2.7，那会改变环境指纹，使云结果与本机 dev 结果不可比。
#
# ⚠ 拉取：docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
#   若平台下拉列表里没有，可在实例上**手动**执行上面这条 pull（自定义镜像）。

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # 数据根目录：运行时由 -v 挂载覆盖（config.py 读这个变量）
    BREASTDG_DATA=/data

WORKDIR /app

# 1) 依赖（不含 torch/torchvision，理由见上）
COPY env/requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt

# 2) 代码（只拷需要的；runs/ 与 data/ 一律不进镜像）
COPY src/    /app/src/
COPY scripts/ /app/scripts/
COPY splits/  /app/splits/

ENV PYTHONPATH=/app/src

# 2b) **预下载预训练权重**（2026-10-07 加，上云硬需求）
#     四个骨干都用 `torchvision...Weights.DEFAULT`，首次训练会**在线拉权重**。
#     AutoDL 访问 download.pytorch.org 可能很慢或不通 —— 那时训练会在
#     第 1 步就卡住（不是报错，是静默等待），按小时计费白烧。
#     在**构建期**下载（构建机通常能通外网），权重随镜像固化，运行期零网络依赖。
#     `--mount=type=cache` 不可用时退化为普通下载；已用 `-c` 断言每个文件都真的
#     落地了 —— 下载失败必须让**构建**失败，而不是留个空缓存到运行期才炸。
RUN python -c "\
import torchvision.models as M; \
[ f(weights=w) for f, w in [ \
    (M.resnet50, M.ResNet50_Weights.DEFAULT), \
    (M.resnet18, M.ResNet18_Weights.DEFAULT), \
    (M.densenet121, M.DenseNet121_Weights.DEFAULT), \
    (M.efficientnet_v2_s, M.EfficientNet_V2_S_Weights.DEFAULT), \
    (M.swin_t, M.Swin_T_Weights.DEFAULT), \
] ]" \
 && python -c "\
import os, pathlib; \
d = pathlib.Path(os.environ.get('TORCH_HOME', '/root/.cache/torch')) / 'hub' / 'checkpoints'; \
fs = sorted(p.name for p in d.glob('*.pth')); \
print('预训练权重已固化：', len(fs), '个'); \
[print('  ', x) for x in fs]; \
assert len(fs) >= 5, f'权重数不足（{len(fs)} < 5），构建失败以暴露下载问题'"

# 3) 环境自检：版本不符就**在构建期失败**，而不是跑出可疑结果才发现
COPY env/verify_env.py /app/env/verify_env.py
RUN python /app/env/verify_env.py

# 3b) 入口脚本必须进镜像（2026-10-07 修，check3 P1③）
#     ⚠ 原镜像只 COPY 了 verify_env.py，`/app/env/` 下**没有** run_cloud.sh，
#     而下面 CMD 调的正是它 —— 容器一起来就是
#       bash: /app/env/run_cloud.sh: No such file or directory
#     即"构建成功、运行即死"。这类缺口不会在构建期暴露，只在租卡后暴露。
#     验：`docker run --rm <img> bash -n /app/env/run_cloud.sh` 应无输出。
COPY env/run_cloud.sh /app/env/run_cloud.sh
RUN bash -n /app/env/run_cloud.sh && chmod +x /app/env/run_cloud.sh

# 默认入口：跑剩余队列（幂等、可重入）
CMD ["bash", "/app/env/run_cloud.sh"]
