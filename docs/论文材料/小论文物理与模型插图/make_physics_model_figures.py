# -*- coding: utf-8 -*-
"""小论文物理与模型插图生成脚本（Fig. 1—6）。

从 Run/ch4_sp_analysis_dev、Run/ch4_sp_surrogate_dev、Run/ch4_sp_blind_B 读取
规范分析产物，生成论文正文六张图：
  fig1_method_schematic.png        方法与复频响定义示意（模型+G_h分解+重构链）；
  fig2_case_matrix_pipeline.png    工况矩阵与验证流水线；
  fig3_fields_homo_vs_layered.png  均质(H004)与层状(P061)幅值—相位—群时延场对比；
  fig4_sp_h1_nonmonotonicity.png   SP-H1：峰值幅值非单调切片与主峰频率迁移；
  fig5_sp_h2_chi_scaling.png       SP-H2：覆盖层修正量对 χ 的标度；
  fig6_cv_blind_error_by_rv.png    CV 与盲测逐例误差按 rv 分层。
输出与本脚本同目录；运行：python make_physics_model_figures.py
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ANALYSIS = os.path.join(REPO_ROOT, "Run", "ch4_sp_analysis_dev")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

SLOPES = [15.0, 30.0, 45.0, 60.0]
RVS = [0.30, 0.45, 0.60, 0.75]
DHS = [0.20, 0.60, 1.00, 1.40]
# B 池参数（计划 §4.8），用于盲测分层
B_PARAMS = {
    "B001": (22.5, 0.40, 0.375), "B002": (22.5, 0.40, 0.675),
    "B003": (22.5, 0.80, 0.525), "B004": (22.5, 1.20, 0.375),
    "B005": (37.5, 0.40, 0.525), "B006": (37.5, 0.80, 0.375),
    "B007": (37.5, 0.80, 0.675), "B008": (37.5, 1.20, 0.525),
    "B009": (52.5, 0.40, 0.675), "B010": (52.5, 0.80, 0.525),
    "B011": (52.5, 1.20, 0.375), "B012": (52.5, 1.20, 0.675),
}


def read_csv(path):
    with open(path, encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fig1_method_schematic():
    """方法与复频响定义示意图。"""
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.18)

    # (a) 模型示意：成层坡地 + 斜入射 + 地表观测与一维参考
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    # 基岩与覆盖层（坡角示意 60°，坡高 h）
    rock = Polygon([(0, 0), (10, 0), (10, 2.2), (4.3, 2.2), (5.6, 4.4), (0, 4.4)],
                   closed=True, facecolor="0.82", edgecolor="k", lw=1.0)
    cover = Polygon([(0, 4.4), (5.6, 4.4), (6.8, 6.4), (0, 6.4)],
                    closed=True, facecolor="0.62", edgecolor="k", lw=1.0)
    ax.add_patch(rock)
    ax.add_patch(cover)
    ax.text(7.6, 1.0, "bedrock\n$V_{s,b}$", ha="center", fontsize=9)
    ax.text(2.6, 5.5, "cover layer\n$V_{s,c}$, thickness $d$", ha="center", fontsize=9)
    ax.text(6.35, 5.55, "$h$", fontsize=10)
    ax.annotate("", xy=(6.82, 6.42), xytext=(5.62, 4.42),
                arrowprops=dict(arrowstyle="-", color="k", lw=0.8))
    # 坡角标注
    ax.plot([5.6, 6.6], [4.4, 4.4], "k:", lw=0.8)
    ax.text(6.35, 4.55, "$i$", fontsize=10)
    # 斜入射 SV 波
    arrow = FancyArrowPatch((1.1, 0.35), (2.6, 2.6), arrowstyle="-|>", mutation_scale=14,
                            color="tab:blue", lw=1.6)
    ax.add_patch(arrow)
    ax.text(1.0, 1.9, "SV, $15^\\circ$\nfrom vertical", fontsize=8, color="tab:blue")
    # 地表观测点与归一化坐标
    for sx, lab in [(1.2, "$s<0$"), (6.2, "$s=0$"), (8.6, "$s>0$")]:
        yy = 4.4 if sx < 5.6 else (4.4 + (sx - 5.6) * (2.0 / 1.2) if sx <= 6.8 else 6.4)
        ax.plot(sx, yy, "ko", ms=3.5)
    ax.plot(6.8, 6.4, "ko", ms=5, mfc="tab:red")
    ax.text(6.95, 6.25, "crest ($s=0$)", fontsize=8)
    ax.text(8.7, 6.28, "surface receivers, $s\\in[-4,4]$", fontsize=8, ha="center")
    # 一维自由场参考柱
    ax.add_patch(Rectangle((0.25, 0.3), 0.55, 4.0, fill=False, edgecolor="tab:green",
                           ls="--", lw=1.2))
    ax.text(0.52, 0.12, "1-D free field\n(same side)", fontsize=7.5, ha="center", color="tab:green")
    ax.set_title("(a) Two-layered slope model and observation layout", fontsize=10)

    # (b) 复频响分解与重构链
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, fc="0.92"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    facecolor=fc, edgecolor="k", lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.0))

    box(0.2, 8.2, 3.2, 1.2, "broadband multi-sine\n$g(t)$, 0.1g, 0.5-12 Hz", fc="tab:blue")
    ax.text(1.8, 9.05, "", fontsize=8)
    box(0.2, 5.6, 3.2, 1.2, "2-D FE slope model\n(Abaqus, CPE4R)")
    box(4.4, 5.6, 3.2, 1.2, "surface spectra\n$\\widehat A_h^{2D}(f,s)$")
    box(0.2, 3.0, 3.2, 1.2, "1-D free field\n$\\widehat A_h^{1D}(f)$", fc="tab:green")
    box(4.4, 3.0, 3.2, 1.2, "complex FRF\n$G_h(f,s)=\\widehat A^{2D}/\\widehat A^{1D}$", fc="tab:orange")
    box(8.4, 3.9, 3.3, 1.0, "amplitude $A_h=|G_h|$")
    box(8.4, 2.5, 3.3, 1.0, "phase $\\Phi_h$, $\\tau_g$, $k_{app}$")
    box(4.4, 0.4, 3.2, 1.2, "surrogate\nPOD-GPR: [$i$, $d/h$, $r_v$] $\\to$ $G_h$", fc="tab:purple")
    box(8.4, 0.4, 3.3, 1.2, "real-wave reconstruction\n$\\widehat A^{2D}_g=G_h\\,\\widehat A^{1D}_g$\n→ PGA, TAF, spectra", fc="tab:red")
    arrow((1.8, 8.2), (1.8, 6.8))
    arrow((3.4, 6.2), (4.4, 6.2))
    arrow((1.8, 5.6), (1.8, 4.2))
    arrow((3.4, 3.6), (4.4, 3.6))
    arrow((7.6, 3.9), (8.4, 4.35))
    arrow((7.6, 3.3), (8.4, 3.0))
    arrow((6.0, 3.0), (6.0, 1.6))
    arrow((7.6, 1.0), (8.4, 1.0))
    ax.set_title("(b) Complex FRF: definition, decomposition, surrogate and reconstruction",
                 fontsize=10)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig1_method_schematic.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out)


def fig2_case_matrix_pipeline():
    """工况矩阵与验证流水线。"""
    d = np.load(os.path.join(ANALYSIS, "complex_frf_dataset.npz"), allow_pickle=False)
    X = d["X"]
    ids = [str(s) for s in d["case_ids"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.2),
                                   gridspec_kw={"height_ratios": [1.15, 1.0]})
    # (a) 参数矩阵：P 网格 + H 基线 + B 盲测点
    cmap = plt.get_cmap("viridis")
    # 同一(d/h,rv)网格点上叠放四个坡角，按坡角水平错开避免重叠
    dodge = {slope: (k - 1.5) * 0.035 for k, slope in enumerate(SLOPES)}
    for k, slope in enumerate(SLOPES):
        sel = [j for j, cid in enumerate(ids)
               if cid.startswith("P") and abs(X[j, 0] - slope) < 1e-6]
        ax1.scatter(X[sel, 1] + dodge[slope], X[sel, 2], s=64, color=cmap(k / 3.0),
                    edgecolors="k", linewidths=0.6, label="P, $i$=%d$^\\circ$" % slope)
    # H 基线（无覆盖层，画在底部轴线上示意）
    hs = [j for j, cid in enumerate(ids) if cid.startswith("H")]
    ax1.scatter([X[j, 1] for j in hs], [-0.055] * len(hs), marker="^", s=70,
                color="none", edgecolors="k", linewidths=1.2,
                label="H homogeneous baselines")
    # B 盲测点
    bx = [B_PARAMS[c][1] for c in sorted(B_PARAMS)]
    br = [B_PARAMS[c][2] for c in sorted(B_PARAMS)]
    ax1.scatter(bx, br, marker="*", s=190, color="tab:red", edgecolors="k",
                linewidths=0.7, label="B blind test (unseen)")
    ax1.set_xlabel("thickness ratio $d/h$")
    ax1.set_ylabel("velocity ratio $r_v$")
    ax1.set_xlim(-0.08, 1.55)
    ax1.set_ylim(-0.12, 0.85)
    ax1.legend(fontsize=8, ncol=3, frameon=False)
    ax1.set_title("(a) Parameter matrix: 64 development cases (P), 4 homogeneous baselines (H), "
                  "12 blind cases (B)", fontsize=10)

    # (b) 验证与建模流水线
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6)
    ax2.axis("off")

    def stage(x, y, w, h, title, detail, fc):
        ax2.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                     facecolor=fc, edgecolor="k", lw=0.9))
        ax2.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
                 fontsize=8.5, weight="bold")
        ax2.text(x + w / 2, y + h - 0.55, detail, ha="center", va="top", fontsize=7.2)

    def link(p, q, label=None):
        ax2.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, lw=1.1))
        if label:
            ax2.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.16, label,
                     ha="center", fontsize=7.2, color="0.25")

    stage(0.15, 4.1, 2.5, 1.5, "V series (5)", "mesh / tail / domain\nsingle-factor checks", "tab:blue")
    stage(3.25, 4.1, 2.5, 1.5, "X series (5)", "Abaqus vs SPECFEM2D\ncross-software gates", "tab:blue")
    stage(6.35, 4.1, 2.5, 1.5, "H + P (68)", "broadband identification\namplitude-phase analysis", "tab:green")
    stage(9.45, 4.1, 2.4, 1.5, "Model lock", "5-fold CV, POD-GPR\nhash recorded", "tab:orange")
    stage(3.25, 0.7, 2.5, 1.5, "B blind (12)", "one-shot test on unseen\nintermediate combos", "tab:red")
    stage(6.35, 0.7, 2.5, 1.5, "C loop (10)", "EQ01-03 reconstruction\nvs direct FEM truth", "tab:red")
    link((2.65, 4.85), (3.25, 4.85), "gate")
    link((5.75, 4.85), (6.35, 4.85), "gate")
    link((8.85, 4.85), (9.45, 4.85))
    ax2.add_patch(FancyArrowPatch((10.65, 4.1), (4.5, 2.2), arrowstyle="-|>",
                                  mutation_scale=12, lw=1.1,
                                  connectionstyle="arc3,rad=0.25"))
    ax2.text(8.2, 3.0, "locked model,\nno re-tuning", fontsize=7.2, color="0.25", ha="center")
    link((5.75, 1.45), (6.35, 1.45), "locked model")
    ax2.set_title("(b) Validation and modelling pipeline (gates per master plan §8)", fontsize=10)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig2_case_matrix_pipeline.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out)


def fig3_fields_homo_vs_layered():
    """H004 均质 vs P061 层状：幅值—相位—群时延场。"""
    d = np.load(os.path.join(ANALYSIS, "complex_frf_dataset.npz"), allow_pickle=False)
    ids = [str(s) for s in d["case_ids"]]
    freq = d["frequency_hz"]
    s = d["s"]
    F, S = np.meshgrid(freq, s, indexing="ij")
    rows = [("H004", "Homogeneous H004 (60$^\\circ$ bedrock)"),
            ("P061", "Layered P061 (60$^\\circ$, d/h=1.40, $r_v$=0.30)")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for r, (cid, title) in enumerate(rows):
        k = ids.index(cid)
        valid = d["valid_mask"][k]
        amp = np.where(valid, np.log(d["amplitude"][k]), np.nan)
        phase = np.where(valid, np.degrees(d["phase_unwrapped_rad"][k]), np.nan)
        delay = np.where(valid, d["group_delay_s"][k], np.nan)
        panels = [
            (amp, "$\\ln|G_h|$", "viridis", None),
            (phase, "$\\Phi_h$ (deg)", "twilight", (-200, 200)),
            (delay, "$\\tau_g$ (s)", "coolwarm", None),
        ]
        for c, (z, label, cmap, clim) in enumerate(panels):
            ax = axes[r, c]
            pcm = ax.pcolormesh(S, F, z, cmap=cmap, shading="nearest")
            if clim:
                pcm.set_clim(*clim)
            ax.set_xlabel("$s$")
            ax.set_ylabel("$f$ (Hz)")
            fig.colorbar(pcm, ax=ax, label=label)
            if r == 0:
                ax.set_title("(%s) %s" % ("abc"[c], label), fontsize=9.5)
        axes[r, 0].text(-3.6, 10.6, title, fontsize=9.5, weight="bold")
    fig.suptitle("Amplitude-phase-group-delay joint fields: homogeneous vs layered slope "
                 "(masked where invalid)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = os.path.join(OUT_DIR, "fig3_fields_homo_vs_layered.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out)


def fig4_sp_h1():
    """SP-H1：峰值幅值随 d/h 的非单调切片 + rv=0.30 主峰频率迁移。"""
    rows = read_csv(os.path.join(ANALYSIS, "case_metrics.csv"))
    grid_amp, grid_hz = {}, {}
    for r in rows:
        if r["group"] != "P":
            continue
        key = (float(r["slope_angle_deg"]), float(r["thickness_ratio"]), float(r["velocity_ratio"]))
        grid_amp[key] = float(r["peak_amplitude"])
        grid_hz[key] = float(r["peak_frequency_hz"])
    cmap = plt.get_cmap("plasma")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for k, slope in enumerate(SLOPES):
        for j, rv in enumerate(RVS):
            seq = [grid_amp[(slope, dh, rv)] for dh in DHS]
            ls = "-" if rv in (0.30, 0.75) else "--"
            ax1.plot(DHS, seq, ls, color=cmap(k / 3.0), marker="o", ms=4, lw=1.1,
                     alpha=0.55 + 0.45 * (j in (0, 3)))
    # 图例：坡角用色、rv 用线型标注示意
    for k, slope in enumerate(SLOPES):
        ax1.plot([], [], color=cmap(k / 3.0), lw=2, label="$i$=%d$^\\circ$" % slope)
    ax1.text(0.22, 1.985, "$r_v$: solid=0.30, dashed=0.45,\ndashed=0.60, solid=0.75 "
             "(opacity marks extremes)", fontsize=7, color="0.3")
    ax1.set_xlabel("$d/h$")
    ax1.set_ylabel("Peak amplitude of $G_h$")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("(a) Peak amplitude vs $d/h$: 14/16 sequences non-monotonic", fontsize=10)

    for k, slope in enumerate(SLOPES):
        seq = [grid_hz[(slope, dh, 0.30)] for dh in DHS]
        ax2.plot(DHS, seq, marker="s", ms=5, color=cmap(k / 3.0), lw=1.4,
                 label="$i$=%d$^\\circ$" % slope)
        offset = 0.35 if k % 2 == 0 else -0.55
        ax2.text(1.42, seq[-1] + offset, "%.1f Hz" % seq[-1], fontsize=8,
                 va="center", color=cmap(k / 3.0))
    ax2.set_xlabel("$d/h$")
    ax2.set_ylabel("Dominant peak frequency (Hz)")
    ax2.set_xlim(0.1, 1.75)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title("(b) Peak-frequency migration, softest cover ($r_v$=0.30)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig4_sp_h1_nonmonotonicity.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out)


def fig5_sp_h2():
    """SP-H2：覆盖层修正量对 χ 的标度（逐例中位修正量）。"""
    from scipy.stats import spearmanr
    rows = read_csv(os.path.join(ANALYSIS, "layer_correction_metrics.csv"))
    chi, da, dp, dg = [], [], [], []
    for r in rows:
        dh = float(r["thickness_ratio"])
        rv = float(r["velocity_ratio"])
        chi.append(rv / (4 * dh))
        da.append(abs(float(r["median_delta_log_amplitude"])))
        dp.append(float(r["median_abs_delta_phase_deg"]))
        dg.append(float(r["median_abs_delta_group_delay_s"]))
    chi = np.array(chi)
    panels = [
        (np.array(da), "$|\\Delta_A|$ median (ln units)", "(a) Amplitude correction"),
        (np.array(dp), "$|\\Delta_\\Phi|$ median (deg)", "(b) Phase correction"),
        (np.array(dg), "$|\\Delta\\tau_g|$ median (s)", "(c) Group-delay correction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9))
    for ax, (y, ylab, title) in zip(axes, panels):
        ax.scatter(chi, y, s=30, color="tab:blue", edgecolors="k", linewidths=0.4, alpha=0.85)
        rho, p = spearmanr(chi, y)
        # 低次拟合趋势线仅作视觉引导
        z = np.polyfit(chi, y, 2)
        xx = np.linspace(chi.min(), chi.max(), 100)
        ax.plot(xx, np.polyval(z, xx), "tab:red", lw=1.2, ls="--")
        ax.text(0.97, 0.95, "Spearman $\\rho$=%.2f\n$p$=%.1e" % (rho, p),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="w", alpha=0.8, edgecolor="none"))
        ax.set_xlabel("$\\chi=r_v/(4\\,d/h)$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig5_sp_h2_chi_scaling.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out)


def fig6_cv_blind():
    """CV 与盲测逐例复数误差按 rv 分层箱线图（含参考线）。"""
    cv = read_csv(os.path.join(REPO_ROOT, "Run", "ch4_sp_surrogate_dev", "cv_case_metrics.csv"))
    cm = read_csv(os.path.join(ANALYSIS, "case_metrics.csv"))
    rv_of_case = {r["case_id"]: float(r["velocity_ratio"]) for r in cm if r["group"] == "P"}
    cv_groups = {rv: [] for rv in RVS}
    for r in cv:
        if r["model"] != "pod_gpr":
            continue
        cv_groups[rv_of_case[r["case_id"]]].append(float(r["E_complex_w"]) * 100)
    blind = read_csv(os.path.join(REPO_ROOT, "Run", "ch4_sp_blind_B",
                                  "unseen_combination_metrics.csv"))
    b_rvs = [0.375, 0.525, 0.675]
    b_groups = {rv: [] for rv in b_rvs}
    for r in blind:
        rv = B_PARAMS[r["case_id"]][2]
        b_groups[rv].append(float(r["E_complex_w"]) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    pos = np.arange(len(RVS))
    ax1.boxplot([cv_groups[rv] for rv in RVS], positions=pos, widths=0.5,
                patch_artist=True, boxprops=dict(facecolor="tab:blue", alpha=0.55),
                medianprops=dict(color="k", lw=1.4))
    for xi, rv in zip(pos, RVS):
        pts = cv_groups[rv]
        ax1.scatter(np.full(len(pts), xi) + np.linspace(-0.08, 0.08, len(pts)), pts,
                    s=16, color="0.2", zorder=3)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(["%.2f" % rv for rv in RVS])
    ax1.set_xlabel("$r_v$ (development grid)")
    ax1.set_ylabel("$E_{complex,w}$ per case (%)")
    ax1.axhline(10, color="tab:red", ls="--", lw=1.0, label="reference: median $\\leq$10%")
    ax1.axhline(20, color="tab:red", ls=":", lw=1.2, label="reference: P90 $\\leq$20%")
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("(a) Five-fold CV, POD-GPR (64 P cases)", fontsize=10)

    pos = np.arange(len(b_rvs))
    ax2.boxplot([b_groups[rv] for rv in b_rvs], positions=pos, widths=0.5,
                patch_artist=True, boxprops=dict(facecolor="tab:red", alpha=0.5),
                medianprops=dict(color="k", lw=1.4))
    for xi, rv in zip(pos, b_rvs):
        pts = b_groups[rv]
        ax2.scatter(np.full(len(pts), xi) + np.linspace(-0.08, 0.08, len(pts)), pts,
                    s=22, color="0.2", zorder=3)
    ax2.set_xticks(pos)
    ax2.set_xticklabels(["%.3f" % rv for rv in b_rvs])
    ax2.set_xlabel("$r_v$ (blind combinations)")
    ax2.axhline(10, color="tab:red", ls="--", lw=1.0)
    ax2.axhline(20, color="tab:red", ls=":", lw=1.2)
    ax2.set_title("(b) One-shot blind test (12 unseen combinations)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "fig6_cv_blind_error_by_rv.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已生成:", out)


if __name__ == "__main__":
    fig1_method_schematic()
    fig2_case_matrix_pipeline()
    fig3_fields_homo_vs_layered()
    fig4_sp_h1()
    fig5_sp_h2()
    fig6_cv_blind()
