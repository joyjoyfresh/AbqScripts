# -*- coding: utf-8 -*-
"""生成图13：分空间区段与分频带参数效应。脚本可独立运行。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig13_分空间区段与分频带参数效应"

PARAMETERS = [
    (0, "坡角 $i$"),
    (1, "厚度比 $d/h$"),
    (2, "波速比 $r_v$"),
]
SEGMENTS = [("A", "上平台"), ("B", "坡面"), ("C", "下平台")]
FREQUENCY_BANDS = [
    (0.5, 3.0, False, "0.5–3 Hz"),
    (3.0, 6.0, False, "3–6 Hz"),
    (6.0, 10.0, True, "6–10 Hz"),
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
            "axes.titlesize": 9.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def band_mask(frequency, lower, upper, include_upper):
    """构造互不重叠的频带掩码。"""
    if include_upper:
        return (frequency >= lower) & (frequency <= upper)
    return (frequency >= lower) & (frequency < upper)


def regional_case_metric(log_amplitude, valid, frequency, segments, segment, band):
    """计算每个工况在指定空间区段和频带内的ln幅值中位数。"""
    lower, upper, include_upper, _ = band
    selected_frequency = band_mask(frequency, lower, upper, include_upper)
    selected_space = segments == segment
    values = log_amplitude[:, selected_frequency, :][:, :, selected_space]
    mask = valid[:, selected_frequency, :][:, :, selected_space]
    return np.nanmedian(np.where(mask, values, np.nan), axis=(1, 2))


def rank_correlation_with_levels(level_medians):
    """用四个等级的秩相关概括总体升降方向。"""
    ranks = np.argsort(np.argsort(level_medians)).astype(float)
    if np.allclose(ranks, ranks[0]):
        return 0.0
    return float(np.corrcoef(np.arange(level_medians.size, dtype=float), ranks)[0, 1])


def format_level(parameter_index, value):
    """格式化产生最大区域中位响应的参数等级。"""
    if parameter_index == 0:
        return "%.0f°" % value
    return "%.2f" % value


def calculate_effect_matrix(
    features, log_amplitude, valid, frequency, segments, parameter_index
):
    """计算参数等级中位响应的最大/最小幅值比及其最大等级。"""
    levels = np.unique(features[:, parameter_index])
    effect = np.full((len(SEGMENTS), len(FREQUENCY_BANDS)), np.nan)
    peak_levels = np.full_like(effect, np.nan)
    trends = np.empty(effect.shape, dtype=object)

    for row, (segment, _) in enumerate(SEGMENTS):
        for column, band in enumerate(FREQUENCY_BANDS):
            case_metric = regional_case_metric(
                log_amplitude, valid, frequency, segments, segment, band
            )
            level_medians = np.array(
                [
                    np.nanmedian(case_metric[np.isclose(features[:, parameter_index], level)])
                    for level in levels
                ]
            )
            effect[row, column] = np.exp(
                np.nanmax(level_medians) - np.nanmin(level_medians)
            )
            peak_levels[row, column] = levels[int(np.nanargmax(level_medians))]
            rho = rank_correlation_with_levels(level_medians)
            if rho >= 0.8:
                trends[row, column] = "总体升高"
            elif rho <= -0.8:
                trends[row, column] = "总体降低"
            else:
                trends[row, column] = "非单调"
    return effect, peak_levels, trends


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """按三段地表和三个频带汇总64工况的参数主效应。"""
    set_journal_style()
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到复频响数据集：%s" % DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=True) as package:
        selected = package["case_groups"] == "P"
        features = package["X"][selected]
        frequency = package["frequency_hz"]
        segments = package["segments"]
        log_amplitude = package["log_amplitude"][selected]
        valid = package["valid_mask"][selected]

    results = [
        calculate_effect_matrix(
            features,
            log_amplitude,
            valid,
            frequency,
            segments,
            parameter_index,
        )
        for parameter_index, _ in PARAMETERS
    ]
    common_max = max(float(np.nanmax(result[0])) for result in results)

    fig = plt.figure(figsize=(11.9, 4.25))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        left=0.075,
        right=0.95,
        bottom=0.19,
        top=0.84,
        wspace=0.25,
    )
    axes = [fig.add_subplot(grid[0, column]) for column in range(3)]
    color_axis = fig.add_subplot(grid[0, 3])

    image = None
    for column, ((parameter_index, parameter_title), result) in enumerate(
        zip(PARAMETERS, results)
    ):
        effect, peak_levels, trends = result
        ax = axes[column]
        image = ax.imshow(
            effect,
            origin="upper",
            aspect="auto",
            cmap="viridis",
            vmin=1.0,
            vmax=common_max,
        )
        for row in range(effect.shape[0]):
            for band_index in range(effect.shape[1]):
                normalized = (effect[row, band_index] - 1.0) / max(common_max - 1.0, 1e-12)
                text_color = "white" if normalized < 0.58 else "#111111"
                ax.text(
                    band_index,
                    row,
                    "×%.2f\n峰@%s\n%s"
                    % (
                        effect[row, band_index],
                        format_level(parameter_index, peak_levels[row, band_index]),
                        trends[row, band_index],
                    ),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color=text_color,
                    linespacing=1.18,
                )
        ax.set_xticks(np.arange(len(FREQUENCY_BANDS)))
        ax.set_xticklabels([band[3] for band in FREQUENCY_BANDS])
        ax.set_yticks(np.arange(len(SEGMENTS)))
        ax.set_yticklabels([segment_name for _, segment_name in SEGMENTS])
        ax.set_xlabel("频带")
        ax.set_ylabel("空间区段" if column == 0 else "")
        ax.set_title("(%s) %s" % ("abc"[column], parameter_title), loc="left", pad=6)

    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("参数主效应跨度（倍）", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    fig.suptitle("不同空间区段和频带内的参数效应强度", fontsize=11, y=0.965)
    fig.text(
        0.5,
        0.045,
        "每格先求单工况区域ln幅值中位数，再对其余两参数边际化；颜色为四等级中最大/最小中位幅值之比，文字给出最大响应等级及秩趋势。",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
    )
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
