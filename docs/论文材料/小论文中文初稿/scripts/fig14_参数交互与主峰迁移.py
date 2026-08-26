# -*- coding: utf-8 -*-
"""生成图14：确定性参数交互分解与分区主峰迁移。脚本可独立运行。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig14_参数交互与主峰迁移"

COMPONENTS = [
    ("坡角 $i$", "#0072B2"),
    ("厚度比 $d/h$", "#E69F00"),
    ("波速比 $r_v$", "#009E73"),
    ("$i$×$d/h$", "#56B4E9"),
    ("$i$×$r_v$", "#CC79A7"),
    ("$d/h$×$r_v$", "#D55E00"),
    ("三阶交互", "#7F7F7F"),
]
SEGMENTS = [("A", "上平台"), ("B", "坡面"), ("C", "下平台")]
SELECTED_SLOPE = 45.0
SELECTED_VELOCITY_RATIO = 0.30


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
            "axes.titlesize": 9.5,
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


def extract_global_peaks(amplitude, valid, frequency, s_values):
    """提取全场峰值幅值、频率和位置。"""
    count = amplitude.shape[0]
    peak_amplitude = np.full(count, np.nan)
    peak_frequency = np.full(count, np.nan)
    peak_s = np.full(count, np.nan)
    for case_index in range(count):
        field = np.where(valid[case_index], amplitude[case_index], np.nan)
        flat_index = int(np.nanargmax(field))
        frequency_index, spatial_index = np.unravel_index(flat_index, field.shape)
        peak_amplitude[case_index] = field[frequency_index, spatial_index]
        peak_frequency[case_index] = frequency[frequency_index]
        peak_s[case_index] = s_values[spatial_index]
    return peak_amplitude, peak_frequency, peak_s


def extract_segment_peaks(amplitude, valid, frequency, s_values, segments, segment):
    """在指定地表区段内独立提取主峰。"""
    count = amplitude.shape[0]
    peak_amplitude = np.full(count, np.nan)
    peak_frequency = np.full(count, np.nan)
    peak_s = np.full(count, np.nan)
    allowed_space = segments == segment
    for case_index in range(count):
        field = np.where(
            valid[case_index] & allowed_space[None, :], amplitude[case_index], np.nan
        )
        flat_index = int(np.nanargmax(field))
        frequency_index, spatial_index = np.unravel_index(flat_index, field.shape)
        peak_amplitude[case_index] = field[frequency_index, spatial_index]
        peak_frequency[case_index] = frequency[frequency_index]
        peak_s[case_index] = s_values[spatial_index]
    return peak_amplitude, peak_frequency, peak_s


def deterministic_factorial_decomposition(features, response):
    """对平衡4×4×4全因子结果进行确定性平方和分解。"""
    grand_mean = float(np.mean(response))
    total = float(np.sum((response - grand_mean) ** 2))
    if total <= np.finfo(float).eps:
        return np.zeros(len(COMPONENTS), dtype=float)

    main_effects = {}
    sums = []
    for parameter_index in range(3):
        effects = {}
        component_sum = 0.0
        for level in np.unique(features[:, parameter_index]):
            selected = np.isclose(features[:, parameter_index], level)
            effect = float(np.mean(response[selected]) - grand_mean)
            effects[level] = effect
            component_sum += int(np.sum(selected)) * effect**2
        main_effects[parameter_index] = effects
        sums.append(component_sum)

    for first, second in ((0, 1), (0, 2), (1, 2)):
        component_sum = 0.0
        for first_level in np.unique(features[:, first]):
            for second_level in np.unique(features[:, second]):
                selected = np.isclose(features[:, first], first_level) & np.isclose(
                    features[:, second], second_level
                )
                interaction = (
                    float(np.mean(response[selected]))
                    - grand_mean
                    - main_effects[first][first_level]
                    - main_effects[second][second_level]
                )
                component_sum += int(np.sum(selected)) * interaction**2
        sums.append(component_sum)

    three_way = max(0.0, total - float(np.sum(sums)))
    sums.append(three_way)
    return 100.0 * np.asarray(sums) / total


def draw_decomposition(ax, contributions):
    """绘制两个响应量的确定性贡献率堆叠条。"""
    response_labels = ["峰值幅值", "主峰频率"]
    y_positions = np.array([1.0, 0.0])
    left = np.zeros(2, dtype=float)
    for component_index, (label, color) in enumerate(COMPONENTS):
        widths = contributions[:, component_index]
        ax.barh(
            y_positions,
            widths,
            left=left,
            height=0.52,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
        for row, width in enumerate(widths):
            if width >= 5.0:
                ax.text(
                    left[row] + width / 2.0,
                    y_positions[row],
                    "%.1f%%" % width,
                    ha="center",
                    va="center",
                    fontsize=7.4,
                    color="white" if color not in ("#E69F00", "#56B4E9") else "#111111",
                    fontweight="bold",
                )
        left += widths
    ax.set_yticks(y_positions)
    ax.set_yticklabels(response_labels)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("总平方和贡献率 (%)")
    ax.set_title("(a) 平衡全因子结果的确定性主效应与交互分解", loc="left")
    ax.grid(axis="x", color="#D9D9D9", lw=0.5, ls=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.31), ncol=4, frameon=False)


def draw_segment_migration(
    ax,
    segment,
    segment_name,
    amplitude,
    valid,
    frequency,
    s_values,
    segments,
    features,
    letter,
):
    """绘制某区段内64工况峰位—峰频云及一条严格固定变量轨迹。"""
    _, peak_frequency, peak_s = extract_segment_peaks(
        amplitude, valid, frequency, s_values, segments, segment
    )
    segment_s = s_values[segments == segment]
    lower_s = float(segment_s.min())
    upper_s = float(segment_s.max())
    at_frequency_upper = np.isclose(peak_frequency, frequency.max())
    at_spatial_endpoint = np.isclose(peak_s, lower_s) | np.isclose(peak_s, upper_s)

    ax.scatter(
        peak_s,
        peak_frequency,
        s=18,
        facecolor="#B3B3B3",
        edgecolor="white",
        linewidth=0.3,
        alpha=0.65,
        zorder=2,
    )
    if np.any(at_spatial_endpoint):
        ax.scatter(
            peak_s[at_spatial_endpoint],
            peak_frequency[at_spatial_endpoint],
            s=38,
            marker="s",
            facecolor="none",
            edgecolor="#E69F00",
            linewidth=1.0,
            zorder=3,
        )
    if np.any(at_frequency_upper):
        ax.scatter(
            peak_s[at_frequency_upper],
            peak_frequency[at_frequency_upper],
            s=42,
            marker="^",
            facecolor="none",
            edgecolor="#D55E00",
            linewidth=1.1,
            zorder=4,
        )

    selected = np.isclose(features[:, 0], SELECTED_SLOPE) & np.isclose(
        features[:, 2], SELECTED_VELOCITY_RATIO
    )
    selected_indices = np.flatnonzero(selected)
    selected_indices = selected_indices[np.argsort(features[selected_indices, 1])]
    selected_thickness = features[selected_indices, 1]
    ax.plot(
        peak_s[selected_indices],
        peak_frequency[selected_indices],
        color="#222222",
        lw=1.1,
        zorder=5,
    )
    ax.scatter(
        peak_s[selected_indices],
        peak_frequency[selected_indices],
        c=selected_thickness,
        cmap="viridis",
        vmin=0.20,
        vmax=1.40,
        s=52,
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )
    for case_index, thickness_ratio in zip(selected_indices, selected_thickness):
        ax.annotate(
            "%.1f" % thickness_ratio,
            (peak_s[case_index], peak_frequency[case_index]),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7.0,
            color="#1A1A1A",
            zorder=7,
        )

    ax.axhline(frequency.max(), color="#D55E00", lw=0.8, ls="--", zorder=1)
    for endpoint in (lower_s, upper_s):
        ax.axvline(endpoint, color="#E69F00", lw=0.7, ls=":", zorder=1)
    margin = max(0.04, 0.035 * (upper_s - lower_s))
    ax.set_xlim(lower_s - margin, upper_s + margin)
    ax.set_ylim(float(frequency.min()) - 0.25, float(frequency.max()) + 0.42)
    ax.set_xlabel("区段主峰位置 $s_p$")
    ax.set_ylabel("区段主峰频率 $f_p$ (Hz)" if segment == "A" else "")
    ax.set_title(
        "(%s) %s（$s$=%.2f～%.2f）" % (letter, segment_name, lower_s, upper_s),
        loc="left",
    )
    ax.text(
        0.03,
        0.95,
        "10 Hz上界 %d例\n空间端点 %d例"
        % (int(np.sum(at_frequency_upper)), int(np.sum(at_spatial_endpoint))),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        bbox=dict(facecolor="white", edgecolor="#B3B3B3", lw=0.5, alpha=0.90),
    )
    ax.grid(color="#E6E6E6", lw=0.45, ls=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig):
    """同时输出300 dpi PNG和矢量PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """绘制全因子贡献率与三个空间区段的主峰迁移。"""
    set_journal_style()
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到复频响数据集：%s" % DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=True) as package:
        selected = package["case_groups"] == "P"
        features = package["X"][selected]
        frequency = package["frequency_hz"]
        s_values = package["s"]
        segments = package["segments"]
        amplitude = package["amplitude"][selected]
        valid = package["valid_mask"][selected]

    peak_amplitude, peak_frequency, _ = extract_global_peaks(
        amplitude, valid, frequency, s_values
    )
    contributions = np.vstack(
        [
            deterministic_factorial_decomposition(features, peak_amplitude),
            deterministic_factorial_decomposition(features, peak_frequency),
        ]
    )

    fig = plt.figure(figsize=(12.4, 7.3))
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[0.78, 1.22],
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.86,
        wspace=0.20,
        hspace=0.38,
    )
    decomposition_axis = fig.add_subplot(grid[0, :])
    draw_decomposition(decomposition_axis, contributions)

    for column, ((segment, segment_name), letter) in enumerate(zip(SEGMENTS, "bcd")):
        ax = fig.add_subplot(grid[1, column])
        draw_segment_migration(
            ax,
            segment,
            segment_name,
            amplitude,
            valid,
            frequency,
            s_values,
            segments,
            features,
            letter,
        )

    migration_handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor="#B3B3B3",
            markeredgecolor="white",
            markersize=5,
            label="64工况分区主峰",
        ),
        Line2D([], [], color="#222222", marker="o", markersize=5, label="固定$i$=45°、$r_v$=0.30；点标签为$d/h$"),
        Line2D(
            [],
            [],
            marker="^",
            ls="none",
            markerfacecolor="none",
            markeredgecolor="#D55E00",
            markersize=6,
            label="$f_p$=10 Hz（上界截断）",
        ),
        Line2D(
            [],
            [],
            marker="s",
            ls="none",
            markerfacecolor="none",
            markeredgecolor="#E69F00",
            markersize=6,
            label="$s_p$位于该区段采样端点",
        ),
    ]
    fig.legend(
        handles=migration_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.005,
        "贡献率为64个确定性全因子结果的平方和分解，不作随机显著性推断；端点峰与10 Hz峰只表示搜索域内最大值，不外推为已解析的内部峰。",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#4D4D4D",
    )
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
