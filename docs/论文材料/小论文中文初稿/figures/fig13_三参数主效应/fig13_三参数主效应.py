# -*- coding: utf-8 -*-
"""生成图13：三个控制参数的边际分布与主效应。脚本可独立运行。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig13_三参数主效应"

PARAMETERS = [
    (0, "坡角 $i$", "坡角 $i$", "#0072B2"),
    (1, "厚度比 $d/h$", "厚度比 $d/h$", "#E69F00"),
    (2, "波速比 $r_v$", "波速比 $r_v$", "#009E73"),
]


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
            "legend.fontsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def derive_peak_metrics(amplitude, valid, frequency):
    """从每个工况的全频—全空间场提取峰值幅值和峰值频率。"""
    peak_amplitude = np.full(amplitude.shape[0], np.nan, dtype=float)
    peak_frequency = np.full(amplitude.shape[0], np.nan, dtype=float)
    for case_index in range(amplitude.shape[0]):
        field = np.where(valid[case_index], amplitude[case_index], np.nan)
        flat_index = int(np.nanargmax(field))
        frequency_index, _ = np.unravel_index(flat_index, field.shape)
        peak_amplitude[case_index] = field.ravel()[flat_index]
        peak_frequency[case_index] = frequency[frequency_index]
    return peak_amplitude, peak_frequency


def format_levels(parameter_index, levels):
    """按参数含义格式化横轴等级。"""
    if parameter_index == 0:
        return ["%.0f°" % value for value in levels]
    return ["%.2f" % value for value in levels]


def draw_distribution(ax, features, values, parameter_index, color, frequency_upper=None):
    """显示逐工况散点、全距、四分位距和中位数。"""
    levels = np.unique(features[:, parameter_index])
    for position, level in enumerate(levels):
        group_values = values[np.isclose(features[:, parameter_index], level)]
        order = np.argsort(group_values)
        sorted_values = group_values[order]
        jitter = np.linspace(-0.16, 0.16, sorted_values.size)
        ordinary = np.ones(sorted_values.size, dtype=bool)
        if frequency_upper is not None:
            ordinary = ~np.isclose(sorted_values, frequency_upper)
        ax.scatter(
            position + jitter[ordinary],
            sorted_values[ordinary],
            s=20,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.58,
            zorder=2,
        )
        if frequency_upper is not None and np.any(~ordinary):
            ax.scatter(
                position + jitter[~ordinary],
                sorted_values[~ordinary],
                s=34,
                marker="^",
                facecolor="white",
                edgecolor="#D55E00",
                linewidth=1.0,
                zorder=4,
            )

        minimum, q1, median, q3, maximum = np.percentile(
            group_values, [0, 25, 50, 75, 100]
        )
        ax.vlines(position, minimum, maximum, color="#333333", lw=0.9, zorder=3)
        ax.vlines(position, q1, q3, color="#333333", lw=4.0, zorder=3)
        ax.scatter(
            [position],
            [median],
            s=43,
            marker="D",
            facecolor="#F0E442",
            edgecolor="#1A1A1A",
            linewidth=0.7,
            zorder=5,
        )
    ax.set_xticks(np.arange(levels.size))
    ax.set_xticklabels(format_levels(parameter_index, levels))
    ax.set_xlim(-0.55, levels.size - 0.45)
    ax.grid(axis="y", color="#D9D9D9", lw=0.5, ls=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """绘制64个全因子工况在三个参数等级下的边际分布。"""
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
    metrics = [
        (peak_amplitude, r"全场峰值幅值 $|G_h|_{\max}$", None),
        (peak_frequency, "全场主峰频率 $f_p$ (Hz)", float(frequency.max())),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.6), sharey="row")
    letters = "abcdef"
    for row, (values, ylabel, upper) in enumerate(metrics):
        for column, (parameter_index, title, xlabel, color) in enumerate(PARAMETERS):
            ax = axes[row, column]
            draw_distribution(
                ax,
                features,
                values,
                parameter_index,
                color,
                frequency_upper=upper,
            )
            ax.set_title("(%s) %s" % (letters[row * 3 + column], title), loc="left")
            ax.set_xlabel(xlabel)
            if column == 0:
                ax.set_ylabel(ylabel)

    amplitude_margin = 0.06 * (np.nanmax(peak_amplitude) - np.nanmin(peak_amplitude))
    axes[0, 0].set_ylim(
        max(0.0, np.nanmin(peak_amplitude) - amplitude_margin),
        np.nanmax(peak_amplitude) + amplitude_margin,
    )
    axes[1, 0].set_ylim(float(frequency.min()) - 0.3, float(frequency.max()) + 0.45)
    for ax in axes[1, :]:
        ax.axhline(
            frequency.max(),
            color="#D55E00",
            lw=0.8,
            ls="--",
            alpha=0.85,
            zorder=1,
        )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor="#7F7F7F",
            markeredgecolor="white",
            markersize=5.5,
            label="逐工况值",
        ),
        Line2D([], [], color="#333333", lw=0.9, marker="|", markersize=10, label="最小—最大"),
        Line2D([], [], color="#333333", lw=4.0, label="四分位距"),
        Line2D(
            [],
            [],
            marker="D",
            ls="none",
            markerfacecolor="#F0E442",
            markeredgecolor="#1A1A1A",
            markersize=6,
            label="中位数",
        ),
        Line2D(
            [],
            [],
            marker="^",
            ls="none",
            markerfacecolor="white",
            markeredgecolor="#D55E00",
            markersize=6,
            label="10 Hz上界峰",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.89, wspace=0.16, hspace=0.30)
    fig.text(
        0.5,
        0.018,
        "64个P组全因子工况；每个参数等级含16例。分布为对其余两参数边际化后的原始结果，不表示显著性检验。",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
    )
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
