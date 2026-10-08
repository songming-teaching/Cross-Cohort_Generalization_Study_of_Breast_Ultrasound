"""论文图表生成（dev 期数据）。只读既有产物，不训练、不碰锁箱量。

产出（`figures/` 下，300 dpi PNG + 数据 CSV）：
  fig_e8_stage_ab    E8 谱量 Stage A→B 变化（零模型自由量，F34 口径）
  fig_e8_spectrum    实测特征值谱（log-log）+ 零模型参考
  fig_ablation       消融热图（13 变体 × 2 方向 ΔAUC）
  fig_attribution    七行归因表（平凡线索 vs 深度特征，两域对）
  fig_trivial_cues   平凡线索单 cue 可分性

配色：Okabe-Ito 色盲安全配色（不用默认 viridis 循环）。
所有图同时导出其数据为 CSV，便于论文数字溯源（验收定义第 1 条）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "runs" / "diag"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

# Okabe-Ito 色盲安全配色
OI = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
      "#E69F00", "#56B4E9", "#F0E442", "#000000"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.autolayout": True,
})


def _save(fig, name: str):
    p = FIGS / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  写出 {p}")


def _csv(name: str, header, rows):
    p = FIGS / f"{name}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"  写出 {p}")


# ---------------------------------------------------------------------------
# 1. E8：Stage A → B 的零模型自由量变化（F34 口径，核心机制证据）
# ---------------------------------------------------------------------------
def fig_e8_stage_ab():
    rows = json.load(open(DIAG / "e8_dev" / "e8_dev_results.json", encoding="utf-8"))
    idx = {(r["variant"], r["stage"], r["cohort"]): r for r in rows}
    cohorts = ["busi", "busbra", "busuclm"]
    disp = {"busi": "BUSI", "busbra": "BUSBRA", "busuclm": "BUS-UCLM"}

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (key, title, ylab) in zip(axes, [
            ("PR_frac", "Participation ratio (normalised)", r"$\mathrm{PR}/\mathrm{PR}_{\rm null}$"),
            ("er_frac", "Effective rank (normalised)", r"$r_{\rm eff}/r_{\rm eff,null}$"),
            ("tk16", "Top-16 trace fraction",
             r"$\sum_{i\leq 16}\lambda_i \,/\, \sum_i\lambda_i$")]):
        x = np.arange(len(cohorts)); w = 0.36
        A = [idx[("baseline", "best_erm", c)][key] for c in cohorts]
        B = [idx[("baseline", "final", c)][key] for c in cohorts]
        ax.bar(x - w/2, A, w, label="Stage A (ERM-best)", color=OI[0])
        ax.bar(x + w/2, B, w, label="Stage B (adversarial)", color=OI[1])
        for xi, (a, b) in enumerate(zip(A, B)):
            ax.annotate("", xy=(xi + w/2, b), xytext=(xi - w/2, a),
                        arrowprops=dict(arrowstyle="->", color=OI[7], lw=1.1,
                                        shrinkA=1, shrinkB=1))
        ax.set_xticks(x); ax.set_xticklabels([disp[c] for c in cohorts])
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab, fontsize=8)
        if key == "PR_frac":
            ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.suptitle("Stage A $\\rightarrow$ B: low-rank signature attenuates under adversarial "
                 "alignment\n(zero-model-free quantities; ResNet50-CBAM, dev hold-out protocol)",
                 fontsize=10)
    _save(fig, "fig_e8_stage_ab")
    _csv("fig_e8_stage_ab",
         ["cohort", "PR_frac_A", "PR_frac_B", "er_frac_A", "er_frac_B", "tk16_A", "tk16_B"],
         [[c,
           idx[("baseline","best_erm",c)]["PR_frac"], idx[("baseline","final",c)]["PR_frac"],
           idx[("baseline","best_erm",c)]["er_frac"],  idx[("baseline","final",c)]["er_frac"],
           idx[("baseline","best_erm",c)]["tk16"],     idx[("baseline","final",c)]["tk16"]]
          for c in cohorts])


# ---------------------------------------------------------------------------
# 2. E8：实测谱 vs 零模型（F34：体部非 MP）
# ---------------------------------------------------------------------------
def fig_e8_spectrum():
    d = json.load(open(DIAG / "f33_p4_recheck.json", encoding="utf-8"))["p4"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, coh in zip(axes, ["busi", "busbra"]):
        rec = d[coh]
        top = np.array(rec["top10_over_lam1"], dtype=float)
        k = np.arange(1, len(top) + 1)
        ax.plot(k, top, "o-", color=OI[0], ms=4, lw=1.4, label="observed (ranked)")
        ax.axhline(1.0, color=OI[1], ls="--", lw=1.2, label="Marčenko–Pastur edge")
        ax.set_yscale("log")
        ax.set_xlabel("eigenvalue rank $i$")
        ax.set_ylabel(r"$\lambda_i/\lambda_{\rm edge}$")
        ax.set_title(f"{coh.upper()}  (p={rec['p']}, c={rec['c']:.2f})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        ax.text(0.97, 0.94,
                f"$K_>$ = {rec['K_above']}\nPR = {rec['PR']:.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
    fig.suptitle("Development-cohort eigenspectra: dominant low-rank head on a "
                 "non-MP bulk (F34)\n$K_>$ counts exceedances of the MP edge "
                 "under the white-bulk null — descriptive only",
                 fontsize=10)
    _save(fig, "fig_e8_spectrum")
    _csv("fig_e8_spectrum",
         ["cohort", "p", "c", "lam1", "lam_plus_edge", "K_above", "PR",
          "lam1_share", "top30_share"],
         [[c, d[c]["p"], d[c]["c"], d[c]["lam1"], d[c]["lam_plus"],
           d[c]["K_above"], d[c]["PR"], d[c]["lam1_share"], d[c]["top30_share"]]
          for c in ("busi", "busbra")])


# ---------------------------------------------------------------------------
# 3. 消融热图（13 变体 × 2 方向）
# ---------------------------------------------------------------------------
def fig_ablation():
    rows = [r for r in csv.DictReader(
        open(ROOT / "runs" / "dev_screen" / "results.csv", encoding="utf-8"))
        if r["steps"] == "3000"]
    by = {}
    for r in rows:
        by.setdefault((r["variant"], r["direction"]), {})[int(r["seed"])] = \
            float(r["dev_auc_final"])
    dirs = ["busbra", "busi"]
    base = {d: by[("baseline", d)] for d in dirs if ("baseline", d) in by}
    # 只保留在**每个方向**都有完整 3 种子的变体——训练补齐过程中
    # 会出现部分方向的中间态（如 swin_t 先跑完 busbra、busi 未跑），
    # 若不过滤，跨变体比较会因缺失被静默拉偏。
    def _complete(v):
        for d in dirs:
            if (v, d) not in by:
                return False
            if len([s for s in by[(v, d)] if s in base.get(d, {})]) < 3:
                return False
        return True

    variants = sorted({v for v, _ in by if v != "baseline" and _complete(v)},
                      key=lambda v: -np.mean(
                          [by[(v, d)][s] - base[d][s]
                           for d in dirs for s in by[(v, d)] if s in base[d]]))
    dropped = sorted({v for v, _ in by
                      if v != "baseline" and not _complete(v)})
    if dropped:
        print(f"  [消融图] 跳过未跑齐的变体：{dropped}")
    M = np.full((len(variants), len(dirs)), np.nan)
    for i, v in enumerate(variants):
        for j, d in enumerate(dirs):
            ds = [by[(v, d)][s] - base[d][s] for s in by.get((v, d), {}) if s in base[d]]
            if ds:
                M[i, j] = np.mean(ds)

    fig, ax = plt.subplots(figsize=(5.3, 0.36 * len(variants) + 1.6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(dirs)))
    ax.set_xticklabels([f"hold out\n{d.upper()}" for d in dirs], fontsize=8)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=8)
    for i in range(len(variants)):
        for j in range(len(dirs)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]*1000:+.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    ax.set_title(r"$\Delta$AUC vs baseline ($\times 10^{-3}$), 3 seeds, paired",
                 fontsize=9)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label(r"$\Delta$AUC", fontsize=8)
    _save(fig, "fig_ablation")
    _csv("fig_ablation", ["variant"] + [f"delta_{d}" for d in dirs],
         [[v] + [M[i, j] for j in range(len(dirs))] for i, v in enumerate(variants)])


# ---------------------------------------------------------------------------
# 4. 七行归因表（意见 2/9 的正面回应）
# ---------------------------------------------------------------------------
def fig_attribution():
    t = json.load(open(DIAG / "domain_probe_p4_v2.json", encoding="utf-8"))["tables"]
    pairs = [("ERM:BUSI_vs_BUSBRA", "BUSI vs BUSBRA"),
             ("ERM:BUSI_vs_BUS-UCLM", "BUSI vs BUS-UCLM")]
    keys = ["1 仅黑边项(pad)", "2 仅长宽比项", "3 仅亮度统计项", "4 仅尺度项",
            "5 平凡(几何+色调)", "6 深度特征",
            "7a 深度−(几何+色调) [上界]", "7b 深度−(几何+色调+尺度) [下界]"]
    labels = ["pad only", "aspect only", "tone only", "scale only",
              "all trivial", "deep features", "deep − trivial (upper)",
              "deep − trivial − scale (lower)"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(keys)); w = 0.38
    for j, (pk, pl) in enumerate(pairs):
        if pk not in t:
            continue
        vals = [t[pk].get(k, np.nan) for k in keys]
        ax.bar(x + (j - 0.5) * w, vals, w, label=pl,
               color=[OI[0], OI[1]][j], alpha=0.9)
    ax.axhline(0.5, color=OI[7], ls=":", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("probe AUC"); ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7.5, frameon=False, ncol=1, loc="lower right")
    ax.set_title("Seven-row attribution: which trivial cue carries the domain signal?",
                 fontsize=9.5)
    _save(fig, "fig_attribution")
    _csv("fig_attribution", ["row"] + [p[1] for p in pairs],
         [[lb] + [t.get(pk, {}).get(k, "") for pk, _ in pairs]
          for k, lb in zip(keys, labels)])


# ---------------------------------------------------------------------------
# 5. 平凡线索单 cue
# ---------------------------------------------------------------------------
def fig_trivial_cues():
    d = json.load(open(DIAG / "trivial_cues_tier2.json", encoding="utf-8"))
    cues = ["scale", "speckle", "grad", "snr", "entropy", "ALL_2nd"]
    label = {"scale": "scale", "speckle": "speckle (Lap. var)", "grad": "gradient",
             "snr": "SNR", "entropy": "entropy", "ALL_2nd": "all (2nd order)"}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(cues)); w = 0.38
    for j, (pk, pl) in enumerate(d.items()):
        ax.bar(x + (j - 0.5) * w, [d[pk].get(c, np.nan) for c in cues], w,
               label=str(pk).replace("_", " vs "), color=[OI[0], OI[1]][j],
               alpha=0.9)
    ax.axhline(0.5, color=OI[7], ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([label[c] for c in cues], rotation=20,
                                         ha="right", fontsize=8)
    ax.set_ylabel("single-cue probe AUC"); ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("Trivial cues alone approach deep-feature separability", fontsize=9.5)
    _save(fig, "fig_trivial_cues")
    _csv("fig_trivial_cues", ["cue"] + list(d), [[c] + [d[p].get(c, "") for p in d]
                                                for c in cues])


# ---------------------------------------------------------------------------
# 6. E8：对抗开/关的谱量对比（P1 的正确对照，18 行）
# ---------------------------------------------------------------------------
def fig_e8_adv_contrast():
    """no_domainhead(λ=0, 对抗关) vs baseline(λ=1.0, 对抗开)。

    ⚠ 判读要点：**PR 是尖峰强度的减函数**（数值标定：纯 bulk PR_frac=0.997、
    单尖峰 θ=200 时降到 0.102）。故"对抗开使 PR 上升"= 尖峰被削弱 = 支持 P1。
    同时必须显示**不显著**这一事实（符号检验 6/9，p=0.25）。
    """
    rows = json.load(open(DIAG / "e8_adv_contrast" / "results.json",
                          encoding="utf-8"))
    idx = {(r["variant"], r["seed_cfg"], r["cohort"]): r for r in rows}
    seeds = ["42", "123", "2026"]
    cohorts = ["busi", "busbra", "busuclm"]
    disp = {"busi": "BUSI", "busbra": "BUSBRA", "busuclm": "BUS-UCLM"}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    # 左：逐格 PR_frac（连线示意配对）
    ax = axes[0]
    xs, off_v, on_v, cols = [], [], [], []
    for i, c in enumerate(cohorts):
        for j, s in enumerate(seeds):
            o = idx[("no_domainhead", s, c)]["PR_frac"]
            n = idx[("baseline", s, c)]["PR_frac"]
            x = i * 4 + j
            xs.append(x); off_v.append(o); on_v.append(n)
            cols.append(OI[j % len(OI)])
    ax.scatter(xs, off_v, marker="o", s=46, facecolor="white",
               edgecolor=OI[0], linewidth=1.6, label="adversarial OFF (λ=0)", zorder=3)
    ax.scatter(xs, on_v, marker="s", s=42, color=OI[1],
               label="adversarial ON (λ=1)", zorder=3)
    for x, o, n in zip(xs, off_v, on_v):
        ax.plot([x, x], [o, n], color="0.55", lw=0.9, zorder=1)
    ax.set_xticks([1, 5, 9])
    ax.set_xticklabels([disp[c] for c in cohorts])
    ax.set_ylabel(r"$\mathrm{PR}/\mathrm{PR}_{\rm null}$")
    ax.set_title("Participation ratio by seed", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    # 右：Δ 分布 + 符号检验
    ax = axes[1]
    d = [idx[("baseline", s, c)]["PR_frac"] - idx[("no_domainhead", s, c)]["PR_frac"]
         for c in cohorts for s in seeds]
    ax.barh(range(len(d)), d, color=[OI[2] if v > 0 else OI[1] for v in d], height=0.62)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"{disp[c][:6]}/{s}" for c in cohorts for s in seeds],
                       fontsize=6.5)
    ax.set_xlabel(r"$\Delta\,\mathrm{PR_{frac}}$  (ON $-$ OFF)")
    k = sum(1 for v in d if v > 0)
    ax.set_title(f"Δ per cell — {k}/{len(d)} positive, sign test $p$=0.25 (n.s.)",
                 fontsize=9)
    fig.suptitle("P1 mechanism check: stronger $\\mathrm{PR_{frac}}$ under "
                 "adversarial alignment\n"
                 "(direction consistent with spike attenuation, but NOT "
                 "significant at this sample size)", fontsize=10)
    _save(fig, "fig_e8_adv_contrast")
    _csv("fig_e8_adv_contrast",
         ["cohort", "seed", "PR_frac_off", "PR_frac_on", "delta", "tk16_off", "tk16_on"],
         [[c, s, idx[("no_domainhead", s, c)]["PR_frac"],
           idx[("baseline", s, c)]["PR_frac"],
           idx[("baseline", s, c)]["PR_frac"] - idx[("no_domainhead", s, c)]["PR_frac"],
           idx[("no_domainhead", s, c)]["tk16"], idx[("baseline", s, c)]["tk16"]]
          for c in cohorts for s in seeds])


def fig_e8_lockbox_spectrum():
    """锁箱 49 检查点的 E8 谱量：逐变体 × 逐队列，两臂配对对照。

    数据源 = `runs/diag/e8_lockbox_summary.json`（由 aggregate_e8_lockbox.py 生成）。
    ⚠ **纵轴只在同一变体内可比** —— R 随架构的特征维数/尺度变化
    （swin_t 的 R 比 resnet50_cbam 高约 73×），故按变体取对数纵轴，
    并在图上写明该口径限制，避免被读成"swin_t 的签名更强"。
    """
    d = json.load(open(DIAG / "e8_lockbox_summary.json", encoding="utf-8"))
    pv = d["per_variant"]
    cohorts = d["cohorts"]
    variants = sorted(pv, key=lambda v: -pv[v]["oasbud_corrected"]["R"]["mean"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    y = np.arange(len(variants))
    h = 0.26

    # 左：三队列 R（对数横轴，因跨变体量级差 ~73×）
    ax = axes[0]
    for i, (c, lab) in enumerate(zip(cohorts, ["OASBUD corrected",
                                               "OASBUD uncorrected", "BrEaST"])):
        m = [pv[v][c]["R"]["mean"] for v in variants]
        s = [pv[v][c]["R"]["sd"] for v in variants]
        ax.barh(y + (1 - i) * h, m, height=h, xerr=s, color=OI[i],
                label=lab, error_kw=dict(lw=0.7, capsize=2))
    ax.set_yticks(y); ax.set_yticklabels(variants, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel(r"spectral statistic $R$  (log; std over 7 seeds)")
    ax.set_title("E8 lockbox: $R$ per cohort (49 checkpoints)", fontsize=9)
    # 图例置于**轴外**：轴内任一角落都会被条形压住（swin_t 的条横跨全宽）
    ax.legend(fontsize=7, frameon=False, loc="upper left",
              bbox_to_anchor=(0.0, -0.22), ncol=3)
    # 口径警示**必须保留**（否则右列排序会被读成"swin_t 签名更强"）
    ax.text(0.0, -0.38, r"⚠ $R$ scales with architecture (vars differ $\sim$73$\times$): "
                        r"compare cohorts WITHIN a variant only",
            transform=ax.transAxes, fontsize=6.5, va="top", color=OI[1])

    # 右：两臂配对差（corrected − uncorrected），同一检查点、同一批图
    ax = axes[1]
    dR = []
    for v in variants:
        a = pv[v]["oasbud_corrected"]["R"]["mean"]
        b = pv[v]["oasbud_uncorrected"]["R"]["mean"]
        dR.append(a - b)
    col = [OI[2] if x > 0 else OI[1] for x in dR]
    ax.barh(y, dR, height=0.5, color=col)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel(r"$\Delta R = R_{\rm corrected} - R_{\rm uncorrected}$  (paired)")
    ax.set_title("Two-render-arm contrast (same 200 images)", fontsize=9)
    ax.text(0.97, 0.04, "43/49 cells: corrected > uncorrected\nmedian ratio 1.74×",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom")

    fig.suptitle("E8 lockbox arm: spectral quantities on 49 checkpoints "
                 "× 3 cohorts (147/147 detected)", fontsize=10)
    _save(fig, "fig_e8_lockbox_spectrum")
    _csv("fig_e8_lockbox_spectrum",
         ["variant", "cohort", "R_mean", "R_sd", "theta_mean", "PR_frac",
          "er_frac", "tk16", "n_seed"],
         [[v, c, pv[v][c]["R"]["mean"], pv[v][c]["R"]["sd"],
           pv[v][c]["theta"]["mean"], pv[v][c]["PR_frac"]["mean"],
           pv[v][c]["er_frac"]["mean"], pv[v][c]["tk16"]["mean"],
           pv[v][c]["R"]["n_seed"]]
          for v in variants for c in cohorts])


if __name__ == "__main__":
    print("生成图表 ...")
    for fn in (fig_e8_adv_contrast, fig_e8_spectrum, fig_e8_lockbox_spectrum,
               fig_ablation,
               fig_attribution, fig_trivial_cues):
        try:
            fn()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  {fn.__name__} FAIL: {e}")
    print("完成。")
