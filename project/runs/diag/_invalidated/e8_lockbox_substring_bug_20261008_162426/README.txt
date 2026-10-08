本目录下 28 个 JSON 全部作废（2026-10-08）。

缺陷：run_e8.py 队列分派用**子串**判定
    arm = "corrected" if "corrected" in name else "uncorrected"
而 "corrected" in "oasbud_uncorrected" → True，
故 oasbud_uncorrected 臂实际读了 corrected 目录 → **两臂同源**。

识别特征：每格内 oasbud_corrected 与 oasbud_uncorrected 的 R 与 theta
**逐位相同**（28/28 格命中，已逐格核验）。若某格两臂不同，说明不是本缺陷所致。

影响范围：仅 E8 谱量。锁箱评估卡（runs/lockbox_eval/probs_*.csv）**无恙** ——
其用显式元组 FULL_CARD_ARMS 分派，实测两臂 9800 键中仅 1 个 prob 相同、
max|Δ|=0.9999（AUC 0.7696 vs 0.5813）。

修法：cohorts.oasbud_arm 做精确解析（唯一出处），并对未知写法**拒绝而非猜测**。
