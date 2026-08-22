# -*- coding: utf-8 -*-
"""Figure generation script for English draft (Figs. 1–6).

Reads standardized data products from:
  Run/ch4_sp_analysis_dev
  Run/ch4_sp_surrogate_dev
  Run/ch4_sp_blind_B

Generates 6 figures for the paper:
  fig1_method_schematic.png        Method overview (model + decomposition + surrogate + reconstruction chain)
  fig2_case_matrix_pipeline.png    Parameter matrix and staged validation pipeline
  fig3_fields_homo_vs_layered.png  Joint amplitude-phase-delay fields (homogeneous vs layered)
  fig4_sp_h1_nonmonotonicity.png   SP-H1: Non-monotonicity and peak-frequency migration
  fig5_sp_h2_chi_scaling.png       SP-H2: Scaling of cover-layer correction with chi
  fig6_cv_blind_error_by_rv.png    CV and blind test errors stratified by rv

Output directory: ../images/
Usage: python make_physics_model_figures.py
"""

import csv
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts -> 小论文英文初稿 -> 论文材料 -> docs -> REPO_ROOT
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
ANALYSIS = os.path.join(REPO_ROOT, "Run", "ch4_sp_analysis_dev")
OUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "images")

SLOPES = [15.0, 30.0, 45.0, 60.0]
RVS = [0.30, 0.45, 0.60, 0.75]
DHS = [0.20, 0.60, 1.00, 1.40]
# B-pool parameters (Plan §4.8)
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
    """Method overview and definition schematic."""
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.18)

    # (a) Model schematic
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    # Bedrock and cover
    rock = Polygon([(0, 0), (10, 0), (10, 2.2), (4.3, 2.2), (5.6, 4.4), (0, 4.4)],
                   closed=True, facecolor="0.82", edgecolor="k", lw=1.0)
    cover = Polygon([(0, 4.4), (5.6, 4.4), (6.8, 6.4), (0, 6.4)],
                    closed=True, facecolor="0.62", edgecolor="k", lw=1.0)
    ax.add_patch(rock)
    ax.add_patch(cover)
    ax.text(7.6, 1.0, "Bedrock\n$V_{s,b}$", ha="center", fontsize=9.5)
    ax.text(2.6, 5.5, "Cover layer\n$V_{s,c}$, thickness $d$", ha="center", fontsize=9.5)
    ax.text(6.35, 5.55, "$h$", fontsize=10.5)
    ax.annotate("", xy=(6.82, 6.42), xytext=(5.62, 4.42),
                arrowprops=dict(arrowstyle="-", color="k", lw=0.8))
    # Slope angle
    ax.plot([5.6, 6.6], [4.4, 4.4], "k:", lw=0.8)
    ax.text(6.35, 4.55, "$i$", fontsize=10.5)
    # Inclined SV wave
    arrow = FancyArrowPatch((1.1, 0.35), (2.6, 2.6), arrowstyle="-|>", mutation_scale=14,
                            color="tab:blue", lw=1.6)
    ax.add_patch(arrow)
    ax.text(1.0, 1.9, "SV wave, 15$^\circ$", fontsize=8.5, color="tab:blue")
    # Surface receivers
    for sx, lab in [(1.2, "$s<0$"), (6.2, "$s=0$"), (8.6, "$s>0$")]:
        yy = 4.4 if sx < 5.6 else (4.4 + (sx - 5.6) * (2.0 / 1.2) if sx <= 6.8 else 6.4)
        ax.plot(sx, yy, "ko", ms=3.5)
    ax.plot(6.8, 6.4, "ko", ms=5, mfc="tab:red")
    ax.text(6.95, 6.25, "Crest ($s=0$)", fontsize=8.5)
    ax.text(8.7, 6.28, "Surface array, $s\in[-4,4]$", fontsize=8.5, ha="center")
    # 1-D free-field reference column
    ax.add_patch(Rectangle((0.25, 0.3), 0.55, 4.0, fill=False, edgecolor="tab:green",
                           ls="--", lw=1.2))
    ax.text(0.52, 0.12, "Same-side 1-D\nfree-field column", fontsize=8, ha="center", color="tab:green")
    ax.set_title("(a) Two-layered slope under inclined SV incidence", fontsize=10.5)

    # (b) Complex FRF chain
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 12)
    ax.set_ylim(10, 0)
    ax.axis("off")

    def box(xy, w, h, text, fc="0.95", ec="0.3", fs=8.5, sub=""):
        p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.25",
                           facecolor=fc, edgecolor=ec, lw=1.0)
        ax.add_patch(p)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        if sub:
            ax.text(cx, cy - 0.3, text, ha="center", va="center", fontsize=fs, weight="bold")
            ax.text(cx, cy + 0.38, sub, ha="center", va="center", fontsize=fs - 1.2, color="0.3")
        else:
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arrow(xy1, xy2, label=""):
        p = FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=11, color="0.3", lw=1.2)
        ax.add_patch(p)
        if label:
            mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
            ax.text(mx, my - 0.22, label, ha="center", fontsize=7.5, color="0.25")

    box((0.2, 0.5), 3.2, 1.8, "Broadband excitation\n$G_{1b}$ (0.5–12 Hz multi-sine)",
        fc="#e8f4f8", ec="#2b7bba", fs=8.5)
    box((4.4, 0.5), 3.2, 1.8, "Abaqus/Standard\n2-D layered FE slope",
        fc="#e8f4f8", ec="#2b7bba", fs=8.5)
    arrow((3.4, 1.4), (4.4, 1.4))

    box((8.6, 0.5), 3.2, 1.8, "Complex FRF $G_h(f,s)$\n$=\\hat{A}_{2D}(f,s)/\\hat{A}_{1D}(f,s)$",
        fc="#fff2e6", ec="#d96b27", fs=9)
    arrow((7.6, 1.4), (8.6, 1.4), "Surface / 1D")

    box((4.4, 3.6), 3.2, 1.6, "Amplitude $|G_h|$\n$\\ln|G_h|$\\to topographic amp.",
        fc="#f0f7e6", ec="#5a8a27", fs=8)
    box((8.6, 3.6), 3.2, 1.6, "Phase $\\Phi_h$\nUnwrapped phase & $\\tau_g$\\to delay",
        fc="#f0f7e6", ec="#5a8a27", fs=8)
    arrow((9.4, 2.3), (6.0, 3.6))
    arrow((10.2, 2.3), (10.2, 3.6))

    box((4.4, 6.6), 3.2, 1.4, "POD-GPR surrogate\n$(i, d/h, r_v) \\to \\hat{G}_h(f,s)$",
        fc="#f3e8f8", ec="#8a3ab9", fs=8.5)
    arrow((6.0, 5.2), (6.0, 6.6))
    arrow((10.2, 5.2), (6.8, 6.6))

    box((8.6, 8.2), 3.2, 1.6, "Real-record reconstruction\n$\\hat{a}_{2D}(t,s)=\\mathcal{F}^{-1}[\\hat{G}_h\\cdot\\mathcal{F}(a_{1D})]$",
        fc="#fce8e6", ec="#c93b2b", fs=8)
    box((4.4, 8.4), 3.2, 1.0, "Site bedrock ground\nmotion $a_0(t)$",
        fc="#f5f5f5", ec="0.5", fs=8)
    arrow((6.0, 8.0), (6.0, 8.4))
    arrow((6.0, 6.6), (6.0, 8.0))
    arrow((7.6, 8.9), (8.4, 8.9))
    ax.set_title("(b) Definition, decomposition, surrogate, and reconstruction",
                 fontsize=10.5)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig1_method_schematic.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", out)


def fig2_case_matrix_pipeline():
    """Case matrix and staged pipeline."""
    d = np.load(os.path.join(ANALYSIS, "complex_frf_dataset.npz"), allow_pickle=False)
    X = d["X"]
    ids = [str(s) for s in d["case_ids"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.2),
                                   gridspec_kw={"height_ratios": [1.15, 1.0]})
    cmap = plt.get_cmap("viridis")
    slope_colors = {15.0: cmap(0.1), 30.0: cmap(0.38), 45.0: cmap(0.68), 60.0: cmap(0.92)}
    dodge = {15.0: -0.015, 30.0: -0.005, 45.0: 0.005, 60.0: 0.015}

    for i_case, cid in enumerate(ids):
        if cid.startswith("P"):
            sl, dh, rv = X[i_case, 0], X[i_case, 1], X[i_case, 2]
            ax1.scatter(dh + dodge[sl], rv, color=slope_colors[sl],
                        s=52, edgecolors="k", lw=0.6, zorder=3)
    for sl in SLOPES:
        ax1.scatter(0.0 + dodge[sl], 1.0, color=slope_colors[sl], marker="s",
                    s=60, edgecolors="k", lw=0.8, zorder=3)
    for bname, (sl, dh, rv) in B_PARAMS.items():
        ax1.scatter(dh, rv, marker="^", s=70, facecolor="tab:red",
                    edgecolors="k", lw=0.9, zorder=4)
        ax1.text(dh + 0.02, rv + 0.012, bname, fontsize=7, color="tab:red", weight="bold")

    handles = [
        plt.Line2D([], [], marker="o", color="w", mfc=slope_colors[sl], ms=8,
                   mec="k", label="$i=%.0f^\\circ$" % sl)
        for sl in SLOPES
    ]
    handles.append(plt.Line2D([], [], marker="s", color="w", mfc="0.6", ms=8,
                              mec="k", label="Homogeneous baseline H ($d/h=0$)"))
    handles.append(plt.Line2D([], [], marker="^", color="w", mfc="tab:red", ms=8,
                              mec="k", label="Blind combination B (12 cases)"))
    ax1.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
    ax1.set_xlabel("Cover-layer thickness ratio $d/h$", fontsize=9.5)
    ax1.set_ylabel("Velocity ratio $r_v = V_{s,c}/V_{s,b}$", fontsize=9.5)
    ax1.set_xlim(-0.08, 1.52)
    ax1.set_ylim(0.24, 1.08)
    ax1.grid(True, ls=":", alpha=0.5)
    ax1.set_title("(a) Parameter matrix: 64 development cases P ($4\\times 4\\times 4$) "
                  "+ 4 homogeneous baselines H + 12 blind cases B", fontsize=10.5)

    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 6.5)
    ax2.axis("off")

    def pbox(xy, w, h, title, sub="", fc="#f2f5f9", ec="#3b6998"):
        p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.2", facecolor=fc, edgecolor=ec, lw=1.0)
        ax2.add_patch(p)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        if sub:
            ax2.text(cx, cy + 0.22, title, ha="center", va="center", fontsize=8, weight="bold")
            ax2.text(cx, cy - 0.25, sub, ha="center", va="center", fontsize=7, color="0.25")
        else:
            ax2.text(cx, cy, title, ha="center", va="center", fontsize=8)

    def link(p1, p2, label=""):
        p = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10, color="0.35", lw=1.0)
        ax2.add_patch(p)
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            ax2.text(mx, my + 0.18, label, ha="center", fontsize=7, color="0.3")

    pbox((0.2, 4.0), 2.2, 1.7, "Stage 1: Config checks", "P061 baseline + V001–V004\n(mesh / tail / domain)",
         fc="#e6f2ff", ec="#1f77b4")
    pbox((2.8, 4.0), 2.2, 1.7, "Stage 2: Cross-software", "X001–X002\nAbaqus vs SPECFEM2D\n(4 Hz Ricker gates)",
         fc="#e6f2ff", ec="#1f77b4")
    link((2.4, 4.85), (2.8, 4.85), "passed")

    pbox((5.4, 4.0), 2.2, 1.7, "Stage 3: Physics laws", "64 layered cases (P)\nDual-channel + SP-H1/H2\n(non-monot. / $\\chi$ scaling)",
         fc="#fff3e6", ec="#ff7f0e")
    link((5.0, 4.85), (5.4, 4.85), "confidence")

    pbox((8.0, 4.0), 2.7, 1.7, "Stage 4: Surrogate", "POD-GPR vs Ridge/NN\n5-fold CV (SP-H3)\nlocked hyperparameters",
         fc="#f0e6ff", ec="#9467bd")
    link((7.6, 4.85), (8.0, 4.85), "dataset")

    pbox((3.0, 0.6), 2.7, 1.7, "Stage 5: Blind test", "12 unseen combos (B)\none-shot evaluation\nno retuning",
         fc="#f9e6eb", ec="#d62728")
    pbox((6.4, 0.6), 3.0, 1.7, "Stage 6: Real-wave closed loop", "10 C cases (EQ01–03)\ntime hist. / PGA / TAF / PSA\nvs direct FE (SP-H4)",
         fc="#e6f9ec", ec="#2ca02c")

    link((8.85, 4.85), (9.45, 4.85))
    ax2.add_patch(FancyArrowPatch((10.65, 4.1), (4.5, 2.2), arrowstyle="-|>",
                                  mutation_scale=12, lw=1.1,
                                  connectionstyle="arc3,rad=0.25"))
    ax2.text(8.2, 3.0, "locked model\nno retuning", fontsize=7.5, color="0.25", ha="center")
    link((5.75, 1.45), (6.35, 1.45), "locked model")
    ax2.set_title("(b) Validation, surrogate modelling, and closed-loop pipeline", fontsize=10.5)

    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig2_case_matrix_pipeline.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", out)


def fig3_fields_homo_vs_layered():
    """Homogeneous baseline H004 vs layered P061 joint fields."""
    d = np.load(os.path.join(ANALYSIS, "complex_frf_dataset.npz"), allow_pickle=False)
    ids = [str(s) for s in d["case_ids"]]
    freq = d["frequency_hz"]
    s = d["s"]
    F, S = np.meshgrid(freq, s, indexing="ij")
    rows = [("H004", "Homogeneous baseline H004 ($i=60^\\circ$)"),
            ("P061", "Layered configuration P061 ($i=60^\\circ, d/h=1.40, r_v=0.30$)")]
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
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig3_fields_homo_vs_layered.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", out)


def fig4_sp_h1():
    """SP-H1: Peak amplitude non-monotonicity and peak-frequency migration."""
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
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig4_sp_h1_nonmonotonicity.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", out)


def fig5_sp_h2():
    """SP-H2: Scaling of cover-layer corrections with chi."""
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
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig5_sp_h2_chi_scaling.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", out)


def fig6_cv_blind():
    """CV and blind per-case complex errors stratified by rv."""
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
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig6_cv_blind_error_by_rv.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", out)


if __name__ == "__main__":
    fig1_method_schematic()
    fig2_case_matrix_pipeline()
    fig3_fields_homo_vs_layered()
    fig4_sp_h1()
    fig5_sp_h2()
    fig6_cv_blind()
