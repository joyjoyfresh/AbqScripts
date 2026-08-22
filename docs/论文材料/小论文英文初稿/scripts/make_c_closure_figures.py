# -*- coding: utf-8 -*-
"""Figure generation script for C-pool real-earthquake closed-loop validation (Figs. 7, 8, S1).

Reads reconstruction data products from:
  Run/ch4_sp_reconstruction/C00X

Generates 3 figures for the paper:
  fig7_c005_representative_panels.png  Representative case C005 (P061/EQ01) 4-panel:
                                       crest time history, spatial PGA, spatial TAF, 5%-damping PSA
  fig8_ten_case_metrics_summary.png    Summary of closed-loop metrics for all 10 cases against reference lines
  figS1_crest_timehist_grid.png        Supplementary: 2x5 grid of crest time histories for 10 cases

Output directory: ../images/
Usage: python make_c_closure_figures.py [reconstruction_root]
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# scripts -> 小论文英文初稿 -> 论文材料 -> docs -> REPO_ROOT
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))))
RECON_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "Run", "ch4_sp_reconstruction")
OUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "images")

CASES = ["C%03d" % i for i in range(1, 11)]
REF_NRMSE = 0.10      # Reference threshold for time-history NRMSE (§4.1.1)
REF_CORR = 0.95       # Reference threshold for correlation
REF_PGA = 0.10        # Reference threshold for PGA / TAF relative error
REF_TPEAK = 0.02      # Reference threshold for peak-time absolute error (s)


def load_case(case):
    """Load single-case reconstruction NPZ and metrics JSON."""
    case_dir = os.path.join(RECON_ROOT, case)
    npz = np.load(os.path.join(case_dir, "reconstruction.npz"), allow_pickle=False)
    with open(os.path.join(case_dir, "reconstruction_metrics.json"), encoding="utf-8") as handle:
        metrics = json.load(handle)
    return npz, metrics


def case_label(metrics):
    """Generate short label 'system/record' (e.g. P061/EQ01)."""
    record = metrics["record"]
    eq = "EQ01" if "eq01" in record else ("EQ02" if "eq02" in record else "EQ03")
    p = metrics["parameters"]
    system = {"15.0": "P007", "45.0": "P039", "60.0": "P061"}.get(
        "%.1f" % p["slope_angle_deg"], "B007")
    return "%s/%s" % (system, eq)


def fig7_representative():
    """Case C005 (P061/EQ01, steepest-thickest-softest boundary configuration) 4-panel."""
    npz, metrics = load_case("C005")
    s = npz["s"]
    time = npz["time"]
    crest = int(np.argmin(np.abs(s)))  # s=0 crest index
    coverage = npz["predicted_G_h_valid_mask"].mean(axis=0)
    invalid = s[coverage < 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))

    ax = axes[0, 0]
    ax.plot(time, npz["direct_fe_acc_h_bandlimited"][crest], lw=0.7, color="0.35", label="Direct FEM")
    ax.plot(time, npz["reconstructed_acc_h"][crest], lw=0.7, color="tab:red", alpha=0.85, label="Surrogate reconstruction")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Crest acceleration (m/s$^2$)")
    ax.set_title("(a) Crest time history, band-limited 0.5-10 Hz")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    if len(invalid):
        ax.axvspan(invalid.min(), invalid.max(), color="0.85", zorder=0,
                   label="Surrogate-invalid $s$ region")
    ax.plot(s, npz["direct_fe_pga"], color="0.35", lw=1.4, label="Direct FEM")
    ax.plot(s, npz["pga_reconstructed"], color="tab:red", lw=1.4, ls="--", label="Reconstruction")
    ax.plot(s, npz["pga_1d_same_side"], color="tab:blue", lw=1.0, ls=":", label="1-D free field")
    ax.set_xlabel("Normalized surface coordinate $s$")
    ax.set_ylabel("PGA (m/s$^2$)")
    ax.set_title("(b) PGA spatial distribution")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    if len(invalid):
        ax.axvspan(invalid.min(), invalid.max(), color="0.85", zorder=0,
                   label="Surrogate-invalid $s$ region")
    ax.plot(s, npz["direct_fe_taf"], color="0.35", lw=1.4, label="Direct FEM")
    ax.plot(s, npz["taf_reconstructed"], color="tab:red", lw=1.4, ls="--", label="Reconstruction")
    ax.axhline(1.0, color="tab:blue", lw=1.0, ls=":", label="No amplification")
    ax.set_xlabel("Normalized surface coordinate $s$")
    ax.set_ylabel("TAF (-)")
    ax.set_title("(c) Topographic amplification factor")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    period = npz["period_s"]
    ax.semilogx(period, npz["direct_fe_psa"][crest], color="0.35", lw=1.4, label="Direct FEM")
    ax.semilogx(period, npz["psa_reconstructed"][crest], color="tab:red", lw=1.4, ls="--", label="Reconstruction")
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("5%-damping PSA (m/s$^2$)")
    ax.set_title("(d) Crest response spectrum")
    ax.legend(frameon=False, fontsize=8)

    nrmse = metrics["comparison_summary"]["time_nrmse"]["median"]
    fig.suptitle("C005: P061 (i=60$^\\circ$, d/h=1.40, $r_v$=0.30) under EQ01 El Centro 0.1g   "
                 "(time-history NRMSE %.1f%%)" % (nrmse * 100), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig7_c005_representative_panels.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Generated:", out)


def fig8_summary():
    """10-case closed-loop metric summary."""
    labels = []
    nrmse, corr, pga, tpeak = [], [], [], []
    for case in CASES:
        _, metrics = load_case(case)
        labels.append("%s\n%s" % (case, case_label(metrics)))
        c = metrics["comparison_summary"]
        nrmse.append(c["time_nrmse"]["median"])
        corr.append(c["correlation"]["median"])
        pga.append(c["pga_relative_error"]["median"])
        tpeak.append(c["peak_time_error_s"]["median"])
    x = np.arange(len(CASES))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.4))
    specs = [
        (axes[0, 0], nrmse, "Time-history NRMSE (-)", REF_NRMSE, "Reference 10%", ".1%", (0, 0.16)),
        (axes[0, 1], corr, "Correlation coefficient (-)", REF_CORR, "Reference 0.95", ".3f", (0.90, 1.005)),
        (axes[1, 0], pga, "PGA relative error (-)", REF_PGA, "Reference 10%", ".1%", (0, 0.05)),
        (axes[1, 1], tpeak, "Peak-time error (s)", REF_TPEAK, "Reference 0.02 s", ".3f", (0, 0.026)),
    ]
    for ax, values, ylabel, ref, reflabel, fmt, ylim in specs:
        bars = ax.bar(x, values, color="tab:blue", width=0.62)
        for xi, v in zip(x, values):
            ax.text(xi, v, format(v, fmt), ha="center", va="bottom", fontsize=6.5)
        ax.axhline(ref, color="tab:red", ls="--", lw=1.0, label=reflabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(*ylim)
    for ax, tag in zip(axes.ravel(), "(a)(b)(c)(d)"):
        ax.set_title(tag, loc="left", fontsize=10)
    fig.suptitle("Closed-loop reconstruction vs direct FEM: ten C-pool cases (medians over surface points)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "fig8_ten_case_metrics_summary.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Generated:", out)


def figS1_timehist_grid():
    """Supplementary: 10-case crest band-limited time history grid."""
    fig, axes = plt.subplots(2, 5, figsize=(14.0, 5.2), sharey=False)
    for ax, case in zip(axes.ravel(), CASES):
        npz, metrics = load_case(case)
        time = npz["time"]
        crest = int(np.argmin(np.abs(npz["s"])))
        ax.plot(time, npz["direct_fe_acc_h_bandlimited"][crest], lw=0.5, color="0.35")
        ax.plot(time, npz["reconstructed_acc_h"][crest], lw=0.5, color="tab:red", alpha=0.8)
        ax.set_title("%s %s" % (case, case_label(metrics)), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Time (s)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("Crest acceleration (m/s$^2$)", fontsize=8)
    handles = [
        plt.Line2D([], [], color="0.35", lw=1.2, label="Direct FEM"),
        plt.Line2D([], [], color="tab:red", lw=1.2, label="Reconstruction"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "figS1_crest_timehist_grid.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("Generated:", out)


if __name__ == "__main__":
    fig7_representative()
    fig8_summary()
    figS1_timehist_grid()
