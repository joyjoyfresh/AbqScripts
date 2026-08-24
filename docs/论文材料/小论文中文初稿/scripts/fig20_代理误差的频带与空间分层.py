# -*- coding: utf-8 -*-
"""生成图20：POD-GPR代理误差的频带与空间分层。脚本可独立运行。"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


REPO_ROOT = Path(__file__).resolve().parents[4]
SUMMARY_PATH = REPO_ROOT / "Run" / "ch4_sp_ml" / "model_summary.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig20_代理误差的频带与空间分层"
SELECTED_MODEL = "pod_gpr"

BAND_KEYS = ["low", "mid", "high"]
BAND_LABELS = ["低频\n0.5–3 Hz", "中频\n3–6 Hz", "高频\n6–10 Hz"]
BAND_COLORS = ["#56B4E9", "#009E73", "#D55E00"]
SEGMENT_KEYS = ["A", "B", "C"]
SEGMENT_LABELS = ["A 上平台\n" + r"$-4\leq s\leq0$", "B 坡面\n" + r"$0<s\leq1$", "C 下平台\n" + r"$1<s\leq4$"]
SEGMENT_COLORS = ["#0072B2", "#E69F00", "#CC79A7"]


def set_journal_style():
    """设置中文期刊绘图样式。"""
    for font_path in (
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\STSONG.TTF",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            fm.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimSun", "STSong", "SimHei", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_selected_summary():
    """读取最终选中模型的交叉验证汇总。"""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("未找到模型汇总表：%s" % SUMMARY_PATH)
    with SUMMARY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] == SELECTED_MODEL:
                return row
    raise RuntimeError("模型汇总表中没有%s" % SELECTED_MODEL)


def draw_median_p90(ax, labels, medians, p90_values, colors, ylabel, title, value_format):
    """绘制中位数柱和由中位数延伸到P90的误差线。"""
    positions = np.arange(len(labels))
    bars = ax.bar(
        positions,
        medians,
        width=0.58,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.88,
        zorder=2,
    )
    for position, median, p90_value in zip(positions, medians, p90_values):
        ax.vlines(position, median, p90_value, color="#333333", lw=1.0, zorder=3)
        ax.scatter(position, p90_value, marker="^", s=38, color="#111111", edgecolor="white", linewidth=0.45, zorder=4)
        ax.text(position, p90_value + max(p90_values) * 0.045, value_format % p90_value, ha="center", va="bottom", fontsize=7.5)
        ax.text(position, median * 0.52, value_format % median, ha="center", va="center", fontsize=8.0, color="white", fontweight="bold")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(p90_values) * 1.22)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, ls=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return bars


def save_figure(fig):
    """同时输出300 dpi PNG和矢量PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """按频带和A/B/C空间段拆解POD-GPR误差。"""
    set_journal_style()
    summary = read_selected_summary()

    amplitude_median = [float(summary["log_amplitude_rmse_%s_median" % key]) for key in BAND_KEYS]
    amplitude_p90 = [float(summary["log_amplitude_rmse_%s_p90" % key]) for key in BAND_KEYS]
    phase_median = [float(summary["circular_phase_rmse_deg_%s_median" % key]) for key in BAND_KEYS]
    phase_p90 = [float(summary["circular_phase_rmse_deg_%s_p90" % key]) for key in BAND_KEYS]
    complex_median = [float(summary["E_complex_w_%s_median" % key]) for key in SEGMENT_KEYS]
    complex_p90 = [float(summary["E_complex_w_%s_p90" % key]) for key in SEGMENT_KEYS]
    spatial_amplitude_median = [float(summary["log_amplitude_rmse_%s_median" % key]) for key in SEGMENT_KEYS]
    spatial_amplitude_p90 = [float(summary["log_amplitude_rmse_%s_p90" % key]) for key in SEGMENT_KEYS]

    fig, axes = plt.subplots(2, 2, figsize=(10.9, 7.6))
    fig.subplots_adjust(top=0.92, hspace=0.37, wspace=0.28)
    draw_median_p90(
        axes[0, 0],
        BAND_LABELS,
        amplitude_median,
        amplitude_p90,
        BAND_COLORS,
        r"$\ln|G_h|$均方根误差",
        "(a) 频带分层：幅值误差",
        "%.3f",
    )
    draw_median_p90(
        axes[0, 1],
        BAND_LABELS,
        phase_median,
        phase_p90,
        BAND_COLORS,
        "圆周相位均方根误差/(°)",
        "(b) 频带分层：相位误差",
        "%.1f",
    )
    draw_median_p90(
        axes[1, 0],
        SEGMENT_LABELS,
        complex_median,
        complex_p90,
        SEGMENT_COLORS,
        r"$E_{\mathrm{complex},w}$",
        "(c) 空间分层：复数场综合误差",
        "%.3f",
    )
    draw_median_p90(
        axes[1, 1],
        SEGMENT_LABELS,
        spatial_amplitude_median,
        spatial_amplitude_p90,
        SEGMENT_COLORS,
        r"$\ln|G_h|$均方根误差",
        "(d) 空间分层：幅值误差",
        "%.3f",
    )
    fig.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="none", label="中位数"),
            plt.Line2D([], [], marker="^", ls="none", ms=5.5, color="#111111", label="P90"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    save_figure(fig)
    plt.close(fig)

    print("POD-GPR频带幅值误差中位数：%s" % ", ".join("%.6f" % value for value in amplitude_median))
    print("POD-GPR频带相位误差中位数：%s" % ", ".join("%.6f" % value for value in phase_median))
    print("POD-GPR空间段复数误差中位数：%s" % ", ".join("%.6f" % value for value in complex_median))


if __name__ == "__main__":
    main()
