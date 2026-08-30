# -*- coding: utf-8 -*-
"""生成图10：均质基线与固定成层参数下的坡角影响对比。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_FILE = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig10_坡角影响对比"

SLOPE_ANGLES = (15.0, 30.0, 45.0, 60.0)
FIXED_THICKNESS_RATIO = 1.00
FIXED_VELOCITY_RATIO = 0.45
ANGLE_COLORS = {
    15.0: "#0072B2",
    30.0: "#E69F00",
    45.0: "#009E73",
    60.0: "#CC79A7",
}
ANGLE_LINESTYLES = {15.0: "-", 30.0: "--", 45.0: "-.", 60.0: ":"}
HOMOGENEOUS_COLOR = "#4D4D4D"
LAYERED_COLOR = "#D55E00"


def set_journal_style():
    """设置中文论文绘图样式。"""
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
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def select_case_index(features, groups, slope, thickness_ratio, velocity_ratio, group):
    """按参数组合唯一选取工况。"""
    mask = (
        (groups == group)
        & np.isclose(features[:, 0], slope)
        & np.isclose(features[:, 1], thickness_ratio)
        & np.isclose(features[:, 2], velocity_ratio)
    )
    indices = np.flatnonzero(mask)
    if indices.size != 1:
        raise RuntimeError(
            "工况无法唯一匹配：group=%s, i=%g, d/h=%g, rv=%g, indices=%s"
            % (group, slope, thickness_ratio, velocity_ratio, indices)
        )
    return int(indices[0])


def load_angle_series():
    """读取均质与固定成层参数的地表包络谱和峰值指标。"""
    result = {"H": {}, "P": {}}
    with np.load(DATA_FILE, allow_pickle=False) as data:
        frequency = data["frequency_hz"].astype(float)
        features = data["X"].astype(float)
        groups = data["case_groups"].astype(str)
        amplitude = data["amplitude"].astype(float)
        valid = data["valid_mask"].astype(bool)
        case_ids = data["case_ids"].astype(str)
        for slope in SLOPE_ANGLES:
            indices = {
                "H": select_case_index(features, groups, slope, 0.0, 1.0, "H"),
                "P": select_case_index(
                    features,
                    groups,
                    slope,
                    FIXED_THICKNESS_RATIO,
                    FIXED_VELOCITY_RATIO,
                    "P",
                ),
            }
            for group, index in indices.items():
                field = np.where(valid[index], amplitude[index], np.nan)
                envelope = np.nanmax(field, axis=1)
                peak_index = int(np.nanargmax(envelope))
                result[group][slope] = {
                    "case_id": str(case_ids[index]),
                    "envelope": envelope,
                    "peak_amplitude": float(envelope[peak_index]),
                    "peak_frequency": float(frequency[peak_index]),
                }
    return frequency, result


def add_panel_label(ax, label):
    """添加分图编号。"""
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def annotate_values(ax, x_values, y_values, frequency=False, offsets=None):
    """在汇总曲线上标注峰值，边界峰用大于等于号说明。"""
    if offsets is None:
        offsets = [6] * len(x_values)
    for x_value, y_value, offset in zip(x_values, y_values, offsets):
        if frequency and np.isclose(y_value, 10.0):
            text = "$\\geq$10.0"
        elif frequency:
            text = "%.1f" % y_value
        else:
            text = "%.2f" % y_value
        ax.annotate(
            text,
            (x_value, y_value),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
        )


def save_figure(fig):
    """保存300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / (OUTPUT_STEM + ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("已生成：%s" % png_path)


def main():
    """对比坡角改变时的全频包络与峰值迁移。"""
    set_journal_style()
    frequency, result = load_angle_series()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2))

    for ax, group, title, panel in (
        (axes[0, 0], "H", "均质边坡地表包络谱", "(a)"),
        (
            axes[0, 1],
            "P",
            "成层边坡地表包络谱（$d/h=1.00$, $r_v=0.45$）",
            "(b)",
        ),
    ):
        for slope in SLOPE_ANGLES:
            item = result[group][slope]
            ax.plot(
                frequency,
                item["envelope"],
                color=ANGLE_COLORS[slope],
                ls=ANGLE_LINESTYLES[slope],
                lw=1.45,
                label="$i=%d^\\circ$（%s）" % (slope, item["case_id"]),
            )
        ax.set_xlim(0.5, 10.0)
        ax.set_xlabel("频率 $f$ (Hz)")
        ax.set_ylabel("地表最大幅值 $\\max_s|G_h|$")
        ax.set_title(title)
        ax.legend(loc="upper left", ncol=2)
        ax.grid(True)
        add_panel_label(ax, panel)

    slopes = np.asarray(SLOPE_ANGLES)
    peak_amplitude_h = np.array([result["H"][value]["peak_amplitude"] for value in SLOPE_ANGLES])
    peak_amplitude_p = np.array([result["P"][value]["peak_amplitude"] for value in SLOPE_ANGLES])
    peak_frequency_h = np.array([result["H"][value]["peak_frequency"] for value in SLOPE_ANGLES])
    peak_frequency_p = np.array([result["P"][value]["peak_frequency"] for value in SLOPE_ANGLES])

    ax = axes[1, 0]
    ax.plot(slopes, peak_amplitude_h, color=HOMOGENEOUS_COLOR, marker="o", lw=1.5, label="均质基线")
    ax.plot(slopes, peak_amplitude_p, color=LAYERED_COLOR, marker="s", lw=1.5, label="固定成层参数")
    annotate_values(ax, slopes, peak_amplitude_h, offsets=[6, 6, 6, -14])
    annotate_values(ax, slopes, peak_amplitude_p, offsets=[6, 6, 6, 7])
    ax.set_xticks(slopes)
    ax.set_xlabel("坡角 $i$ ($^\\circ$)")
    ax.set_ylabel("全局峰值 $|G_h|_{\\max}$")
    ax.set_title("峰值幅值随坡角变化")
    ax.legend(loc="best")
    ax.grid(True)
    add_panel_label(ax, "(c)")

    ax = axes[1, 1]
    ax.plot(slopes, peak_frequency_h, color=HOMOGENEOUS_COLOR, marker="o", lw=1.5, label="均质基线")
    ax.plot(slopes, peak_frequency_p, color=LAYERED_COLOR, marker="s", lw=1.5, label="固定成层参数")
    annotate_values(ax, slopes, peak_frequency_h, frequency=True)
    annotate_values(ax, slopes, peak_frequency_p, frequency=True)
    ax.set_xticks(slopes)
    ax.set_ylim(0.0, 10.8)
    ax.set_xlabel("坡角 $i$ ($^\\circ$)")
    ax.set_ylabel("全局主峰频率 (Hz)")
    ax.set_title("主峰频率随坡角迁移")
    ax.legend(loc="best")
    ax.grid(True)
    add_panel_label(ax, "(d)")

    fig.suptitle("均质基线与成层固定参数条件下的坡角影响对比", fontsize=11, y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.96), w_pad=2.5, h_pad=2.2)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
