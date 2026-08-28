# -*- coding: utf-8 -*-
"""生成图12：各坡角下厚度比—波速比响应矩阵。脚本可独立运行。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig12_厚度波速比响应矩阵"

SLOPE_ANGLES = np.array([15.0, 30.0, 45.0, 60.0])
THICKNESS_RATIOS = np.array([0.20, 0.60, 1.00, 1.40])
VELOCITY_RATIOS = np.array([0.30, 0.45, 0.60, 0.75])


def set_journal_style():
    """设置统一的中文期刊绘图样式。"""
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
            "axes.titlesize": 9.4,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def derive_peak_metrics(amplitude, valid, frequency):
    """提取每个工况的全场峰值幅值和主峰频率。"""
    peak_amplitude = np.full(amplitude.shape[0], np.nan, dtype=float)
    peak_frequency = np.full(amplitude.shape[0], np.nan, dtype=float)
    for case_index in range(amplitude.shape[0]):
        field = np.where(valid[case_index], amplitude[case_index], np.nan)
        flat_index = int(np.nanargmax(field))
        frequency_index, _ = np.unravel_index(flat_index, field.shape)
        peak_amplitude[case_index] = field.ravel()[flat_index]
        peak_frequency[case_index] = frequency[frequency_index]
    return peak_amplitude, peak_frequency


def build_matrix(features, values, slope_angle):
    """构造给定坡角的4×4厚度比—波速比矩阵。"""
    matrix = np.full((THICKNESS_RATIOS.size, VELOCITY_RATIOS.size), np.nan)
    for row, thickness_ratio in enumerate(THICKNESS_RATIOS):
        for column, velocity_ratio in enumerate(VELOCITY_RATIOS):
            matched = np.flatnonzero(
                np.isclose(features[:, 0], slope_angle)
                & np.isclose(features[:, 1], thickness_ratio)
                & np.isclose(features[:, 2], velocity_ratio)
            )
            if matched.size != 1:
                raise RuntimeError(
                    "参数组合 i=%.0f°, d/h=%.2f, rv=%.2f 未唯一匹配"
                    % (slope_angle, thickness_ratio, velocity_ratio)
                )
            matrix[row, column] = values[int(matched[0])]
    return matrix


def annotate_matrix(ax, matrix, lower, upper, frequency_upper=None):
    """在每个单元格内标注精确值，并标出频带上界峰。"""
    span = max(upper - lower, np.finfo(float).eps)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            normalized = (value - lower) / span
            text_color = "white" if normalized < 0.58 else "#111111"
            if frequency_upper is not None and np.isclose(value, frequency_upper):
                label = "≥%.1f" % value
            else:
                label = "%.2f" % value if frequency_upper is None else "%.1f" % value
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=7.7,
                color=text_color,
                fontweight="bold" if frequency_upper is not None and np.isclose(value, frequency_upper) else "normal",
            )
            if frequency_upper is not None and np.isclose(value, frequency_upper):
                ax.scatter(
                    [column + 0.30],
                    [row + 0.29],
                    marker="^",
                    s=25,
                    facecolor="white",
                    edgecolor="#D55E00",
                    linewidth=0.9,
                    clip_on=False,
                    zorder=4,
                )


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """绘制四个坡角下峰值幅值和主峰频率的全因子矩阵。"""
    set_journal_style()
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到复频响数据集：%s" % DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=True) as package:
        selected = package["case_groups"] == "P"
        features = package["X"][selected]
        frequency = package["frequency_hz"]
        amplitude = package["amplitude"][selected]
        valid = package["valid_mask"][selected]

    peak_amplitude, peak_frequency = derive_peak_metrics(amplitude, valid, frequency)
    amplitude_limits = (float(np.nanmin(peak_amplitude)), float(np.nanmax(peak_amplitude)))
    frequency_limits = (float(np.nanmin(peak_frequency)), float(frequency.max()))

    fig = plt.figure(figsize=(12.3, 6.5))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055],
        left=0.07,
        right=0.955,
        bottom=0.11,
        top=0.90,
        wspace=0.16,
        hspace=0.24,
    )
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for column in range(4):
            axes[row, column] = fig.add_subplot(grid[row, column])
    amplitude_color_axis = fig.add_subplot(grid[0, 4])
    frequency_color_axis = fig.add_subplot(grid[1, 4])

    amplitude_image = None
    frequency_image = None
    letters = "abcdefgh"
    for column, slope_angle in enumerate(SLOPE_ANGLES):
        amplitude_matrix = build_matrix(features, peak_amplitude, slope_angle)
        frequency_matrix = build_matrix(features, peak_frequency, slope_angle)

        amplitude_image = axes[0, column].imshow(
            amplitude_matrix,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=amplitude_limits[0],
            vmax=amplitude_limits[1],
        )
        frequency_image = axes[1, column].imshow(
            frequency_matrix,
            origin="lower",
            aspect="auto",
            cmap="cividis",
            vmin=frequency_limits[0],
            vmax=frequency_limits[1],
        )
        annotate_matrix(axes[0, column], amplitude_matrix, *amplitude_limits)
        annotate_matrix(
            axes[1, column],
            frequency_matrix,
            *frequency_limits,
            frequency_upper=float(frequency.max()),
        )

        axes[0, column].set_title(
            "(%s) $i$=%.0f°" % (letters[column], slope_angle), loc="left", pad=5
        )
        axes[1, column].set_title(
            "(%s) $i$=%.0f°" % (letters[4 + column], slope_angle), loc="left", pad=5
        )
        for row in range(2):
            ax = axes[row, column]
            ax.set_xticks(np.arange(VELOCITY_RATIOS.size))
            ax.set_xticklabels(["%.2f" % value for value in VELOCITY_RATIOS])
            ax.set_yticks(np.arange(THICKNESS_RATIOS.size))
            if column == 0:
                ax.set_yticklabels(["%.2f" % value for value in THICKNESS_RATIOS])
                ax.set_ylabel("厚度比 $d/h$")
            else:
                ax.set_yticklabels([])
            ax.set_xlabel("波速比 $r_v$")
            ax.tick_params(which="minor", bottom=False, left=False)

    amplitude_colorbar = fig.colorbar(amplitude_image, cax=amplitude_color_axis)
    amplitude_colorbar.set_label(r"峰值幅值 $|G_h|_{\max}$", fontsize=9)
    amplitude_colorbar.ax.tick_params(labelsize=8)
    frequency_colorbar = fig.colorbar(frequency_image, cax=frequency_color_axis)
    frequency_colorbar.set_label("主峰频率 $f_p$ (Hz)", fontsize=9)
    frequency_colorbar.ax.tick_params(labelsize=8)

    fig.text(
        0.02,
        0.69,
        "峰值幅值",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.31,
        "主峰频率",
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    fig.suptitle("厚度比—波速比响应矩阵（64个P组全因子工况）", fontsize=11, y=0.975)
    fig.text(
        0.5,
        0.02,
        "单元格为全频—全空间峰值；“≥10.0”和空心三角表示主峰位于分析频带上界，真实峰频可能更高。",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
    )
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
