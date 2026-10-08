"""方向效应表刷新：源域样本量混杂的量化（裁决：主动交出的防御性证据）。

为什么要它
----------
dev 期为留一域（LODO）协议，`direction` = 留出的域，故两方向是：

    direction=busbra  → 源域仅 BUSI   → 训练 534 张 → 留出 BUSBRA (1875)
    direction=busi    → 源域仅 BUSBRA → 训练 1589 张 → 留出 BUSI  (629)

源域规模相差 2.98×，而两方向的 AUC 水平随之系统性地差 ~0.15。
这不是方法差异，是**训练数据量**的混杂。审稿人必问
「你们是不是只是训练数据变少了？」——故主动量化并披露。

产物：`reports/direction_effect.md`（TIER4 补齐后可刷新为 16 变体 × 2 方向）。

口径
----
- 只取 `steps` 达到该变体最大值的行（排除冒烟行）；
- 每格用 `dev_auc_final`（留出域最终 AUC），与既有报告口径一致；
- 每方向每变体取 3 种子均值；
- 归一化：`ratio = 大源方向 / 小源方向`，并附中位差。

用法
----
  python scripts/report_direction_effect.py
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "runs" / "dev_screen" / "results.csv"
OUT = ROOT / "reports" / "direction_effect.md"

# 源域样本量（由 splits 决定；534 = BUSI 减去验证集，1589 = BUSBRA 减去验证集）
SRC_N = {"busbra": 534, "busi": 1589}
HELDOUT = {"busbra": "BUSBRA (1875)", "busi": "BUSI (629)"}


def load() -> list[dict]:
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    rows = [r for r in rows if str(r.get("steps", "")).strip()]
    mx = max(int(r["steps"]) for r in rows)
    return [r for r in rows if int(r["steps"]) >= mx]


def main() -> int:
    rows = load()
    agg: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        agg.setdefault(r["variant"], {}).setdefault(r["direction"], []).append(
            float(r["dev_auc_final"]))

    table = []
    for v, dd in sorted(agg.items()):
        if "busbra" not in dd or "busi" not in dd:
            print(f"[warn] {v} 缺方向：{sorted(dd)}")
            continue
        small = statistics.mean(dd["busbra"])   # 小源（534）
        large = statistics.mean(dd["busi"])     # 大源（1589）
        table.append((v, small, large, large - small, len(dd["busbra"]), len(dd["busi"])))

    if not table:
        print("无数据")
        return 1

    diffs = [t[3] for t in table]
    same_sign = sum(1 for d in diffs if d > 0)
    n_tot = len(table)
    med = statistics.median(diffs)
    lo, hi = min(diffs), max(diffs)
    n_ratio = SRC_N["busi"] / SRC_N["busbra"]

    # 源域样本量比 ↔ 中位 AUC 差的相关（仅两点，故只作定性陈述）
    lines = [
        "# 方向效应（源域样本量）分析 —— 全部来自公开队列 dev 产物",
        "",
        "> 本文件由 `scripts/report_direction_effect.py` 自动生成，请勿手改。",
        f"> 生成口径：`results.csv` 中 steps 达最大值的行，`dev_auc_final`，"
        f"每方向 3 种子均值。",
        "",
        "## 背景",
        "",
        "dev 期为留一域（LODO）协议，`direction` = **留出的域**，故两个方向是：",
        "",
        "| direction | 训练源 | 源域样本量 | 留出域（评估，完全不可见） |",
        "|---|---|---|---|",
        "| `busbra` | BUSI | **534** | BUSBRA (1875) |",
        "| `busi`   | BUSBRA | **1589** | BUSI (629) |",
        "",
        f"源域规模相差 **{n_ratio:.2f}×**。这条链是真实的、且我们主动先量化它。",
        "",
        f"## 结果：{same_sign}/{n_tot} 变体完全同向",
        "",
        "| 变体 | busbra 方向（小源 534） | busi 方向（大源 1589） | 差 | n |",
        "|---|---|---|---|---|",
    ]
    for v, s, l, d, ns, nl in table:
        flag = "" if d > 0 else "  ⚠"
        lines.append(f"| {v} | {s:.4f} | {l:.4f} | **{d:+.4f}**{flag} | {ns}/{nl} |")

    lines += [
        "",
        "## 汇总",
        "",
        f"- 同向 **{same_sign}/{n_tot}**（正值 = 大源方向 AUC 更高）；",
        f"- 中位差 **{med:+.4f}**；范围 [{lo:+.4f}, {hi:+.4f}]；",
        f"- 源域样本量比 **{n_ratio:.2f}×** ↔ 中位 AUC 差 **{med:+.4f}**。",
        "",
        "## 建议的写法（三层回答）",
        "",
        "1. **这条链真实，我们自己先量化** —— 不装作不存在；",
        "2. **但它不是跨库落差的驱动** —— 用**对称性**论证，不靠量级对比：",
        "   把每一边的「同库原生性能」与其「异库迁移性能」比（均为**单检查点**），",
        "   两库的降幅几乎相等（见 `scripts/verify_transfer_gap.py`）：",
        "   BUSI **+0.1685** ↔ BUSBRA **+0.1549**，而**源域样本量差 2.98×**。",
        "   **若降幅由源量驱动，二者应差近 3 倍；实测几乎相同。**",
        "3. **两方向都报全量结果，不择优** —— 方向效应作为**已知并量化的混杂**",
        "   入表，而不是藏在附录里。",
        "",
        "> ⚠ **勿用「方向内落差」论证此事**（本表右侧两列**不能**这么相减）：",
        "> 那是把「两库**固有难度**不同」混进「迁移落差」，会造出 0.318 vs 0.006",
        "> 的**假不对称**。正确切法是**按队列**。详见 `verify_transfer_gap.py`",
        "> 的反例对照段（该脚本刻意保留错误切法作防回归）。",
        "",
        "> **叙事价值**：主动交出不利于自己的量，是**可信度信号**。它预先拆掉",
        "> 审稿人必然提出的混淆。建议放在 Discussion 的 Limitations 之前一节，",
        "> 标题类似 *\"A quantified confound: source-cohort size\"*。",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT}")
    print(f"同向 {same_sign}/{n_tot}  中位差 {med:+.4f}  范围 [{lo:+.4f}, {hi:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
