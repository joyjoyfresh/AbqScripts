# -*- coding: utf-8 -*-
"""生成图15：覆盖层复数修正场与特征频率统计。脚本可独立运行。"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_DIR = REPO_ROOT / "Run" / "ch4_sp_analysis"
DATA_PATH = ANALYSIS_DIR / "complex_frf_dataset.npz"
METRICS_PATH = ANALYSIS_DIR / "layer_correction_metrics.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig15_覆盖层复数修正与特征频率"
REPRESENTATIVE_CASE = "P039"

SLOPE_COLORS = {
    15.0: "#0072B2",
    30.0: "#E69F00",
    45.0: "#009E73",
    60.0: "#D55E00",
}


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


def read_metrics(path):
    """读取64个覆盖层工况的复数修正汇总指标。"""
    numeric_columns = [
        "slope_angle_deg",
        "thickness_ratio",
        "velocity_ratio",
        "median_delta_log_amplitude",
        "median_abs_delta_phase_deg",
        "median_abs_delta_group_delay_s",
    ]
    columns = {name: [] for name in numeric_columns}
    case_ids = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_ids.append(row["case_id"])
            for name in numeric_columns:
                columns[name].append(float(row[name]))
    result = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    result["case_id"] = np.asarray(case_ids)
    return result


def average_ranks(values):
    """计算含并列值的平均秩。"""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and np.isclose(values[order[end]], values[order[start]]):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_rho(first, second):
    """计算Spearman秩相关系数，不依赖外部统计包。"""
    first_rank = average_ranks(first)
    second_rank = average_ranks(second)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def grouped_summary(x_values, y_values):
    """按相同特征频率比汇总中位数和全距。"""
    rounded = np.round(x_values, 12)
    unique = np.unique(rounded)
    median = np.full(unique.size, np.nan)
    minimum = np.full(unique.size, np.nan)
    maximum = np.full(unique.size, np.nan)
    for index, value in enumerate(unique):
        selected = np.isclose(rounded, value)
        median[index] = np.nanmedian(y_values[selected])
        minimum[index] = np.nanmin(y_values[selected])
        maximum[index] = np.nanmax(y_values[selected])
    return unique, median, minimum, maximum


def add_segment_labels(ax):
    """在复数修正场上标出三个地表区段。"""
    for left, right, label in [(-4.0, 0.0, "上平台"), (0.0, 1.0, "坡面"), (1.0, 4.0, "下平台")]:
        ax.text(
            (left + right) / 2.0,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.7,
            color="#222222",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.2),
        )
    for boundary in (0.0, 1.0):
        ax.axvline(boundary, color="white", lw=0.9, ls="--", alpha=0.9)


def draw_chi_statistics(ax, chi, slopes, values, ylabel, letter, title, signed=False):
    """绘制64工况散点以及同χ工况的中位数和全距。"""
    slope_levels = np.unique(slopes)
    multiplicative_offsets = np.linspace(-0.018, 0.018, slope_levels.size)
    for slope_angle, offset in zip(slope_levels, multiplicative_offsets):
        selected = np.isclose(slopes, slope_angle)
        ax.scatter(
            chi[selected] * (1.0 + offset),
            values[selected],
            s=25,
            facecolor=SLOPE_COLORS[float(slope_angle)],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.72,
            zorder=2,
        )

    unique, median, minimum, maximum = grouped_summary(chi, values)
    ax.vlines(unique, minimum, maximum, color="#4D4D4D", lw=0.7, alpha=0.75, zorder=3)
    ax.plot(unique, median, color="#111111", lw=1.2, marker="D", ms=3.7, zorder=4)
    if signed:
        ax.axhline(0.0, color="#666666", lw=0.8, ls="--", zorder=1)

    rho = spearman_rho(chi, values)
    ax.text(
        0.96,
        0.94,
        "Spearman $\\rho$=%.2f" % rho,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="#B3B3B3", lw=0.5, alpha=0.90),
    )
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.10, 0.20, 0.40, 0.80])
    ax.set_xticklabels(["0.05", "0.10", "0.20", "0.40", "0.80"])
    ax.set_xlim(0.045, 1.02)
    ax.set_xlabel("特征频率比 $\\chi=r_v/[4(d/h)]$")
    ax.set_ylabel(ylabel)
    ax.set_title("(%s) %s" % (letter, title), loc="left")
    ax.grid(color="#E0E0E0", lw=0.45, ls=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig):
    """同时输出300 dpi PNG和矢量PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """绘制代表工况修正场与64工况χ统计。"""
    set_journal_style()
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到复频响数据集：%s" % DATA_PATH)
    if not METRICS_PATH.exists():
        raise FileNotFoundError("未找到覆盖层修正汇总表：%s" % METRICS_PATH)

    with np.load(DATA_PATH, allow_pickle=True) as package:
        matched = np.flatnonzero(package["case_ids"] == REPRESENTATIVE_CASE)
        if matched.size != 1:
            raise RuntimeError("代表工况 %s 未唯一匹配" % REPRESENTATIVE_CASE)
        case_index = int(matched[0])
        representative_features = package["X"][case_index]
        frequency = package["frequency_hz"]
        s_values = package["s"]
        correction = package["layer_correction_C_G"][case_index]
        valid = package["layer_correction_valid_mask"][case_index]
        p_selected = package["case_groups"] == "P"
        p_case_ids = package["case_ids"][p_selected]
        p_features = package["X"][p_selected]

    log_correction = np.where(valid, np.log(np.maximum(np.abs(correction), 1e-12)), np.nan)
    phase_correction = np.where(valid, np.rad2deg(np.angle(correction)), np.nan)
    amplitude_limit = float(np.nanpercentile(np.abs(log_correction), 99.5))
    flat_index = int(np.nanargmax(np.abs(log_correction)))
    peak_frequency_index, peak_spatial_index = np.unravel_index(flat_index, log_correction.shape)
    correction_peak_frequency = frequency[peak_frequency_index]
    correction_peak_s = s_values[peak_spatial_index]

    metrics = read_metrics(METRICS_PATH)
    metric_position = {case_id: index for index, case_id in enumerate(metrics["case_id"])}
    metric_order = np.asarray([metric_position[case_id] for case_id in p_case_ids], dtype=int)
    slopes = p_features[:, 0]
    chi = p_features[:, 2] / (4.0 * p_features[:, 1])
    amplitude_correction_magnitude = np.abs(
        metrics["median_delta_log_amplitude"][metric_order]
    )
    statistic_panels = [
        (
            amplitude_correction_magnitude,
            "中位净幅值修正的绝对值",
            "c",
            "净幅值修正幅度 $|\\mathrm{median}(\\Delta_A)|$",
            False,
        ),
        (
            metrics["median_abs_delta_phase_deg"][metric_order],
            "中位绝对相位修正 (°)",
            "d",
            "相位修正",
            False,
        ),
        (
            1000.0 * metrics["median_abs_delta_group_delay_s"][metric_order],
            "中位绝对群时延修正 (ms)",
            "e",
            "群时延修正",
            False,
        ),
    ]

    fig = plt.figure(figsize=(12.5, 7.2))
    grid = fig.add_gridspec(
        2,
        6,
        height_ratios=[1.12, 0.88],
        left=0.065,
        right=0.985,
        bottom=0.135,
        top=0.91,
        wspace=0.46,
        hspace=0.36,
    )
    amplitude_axis = fig.add_subplot(grid[0, 0:3])
    phase_axis = fig.add_subplot(grid[0, 3:6])
    statistic_axes = [fig.add_subplot(grid[1, 2 * index : 2 * index + 2]) for index in range(3)]

    amplitude_mesh = amplitude_axis.pcolormesh(
        s_values,
        frequency,
        np.ma.masked_invalid(log_correction),
        shading="auto",
        cmap="RdBu_r",
        vmin=-amplitude_limit,
        vmax=amplitude_limit,
        rasterized=True,
    )
    phase_mesh = phase_axis.pcolormesh(
        s_values,
        frequency,
        np.ma.masked_invalid(phase_correction),
        shading="auto",
        cmap="twilight_shifted",
        vmin=-180.0,
        vmax=180.0,
        rasterized=True,
    )
    for ax in (amplitude_axis, phase_axis):
        add_segment_labels(ax)
        ax.scatter(
            [correction_peak_s],
            [correction_peak_frequency],
            marker="*",
            s=62,
            facecolor="#F0E442",
            edgecolor="black",
            linewidth=0.65,
            zorder=4,
        )
        ax.set_xlim(s_values.min(), s_values.max())
        ax.set_ylim(frequency.min(), frequency.max())
        ax.set_xticks([-4, 0, 1, 4])
        ax.set_xlabel("地表坐标 $s$")
        ax.set_ylabel("频率 $f$ (Hz)")
    amplitude_axis.set_title("(a) 幅值修正 $\\ln|C_G|$", loc="left")
    phase_axis.set_title("(b) 相位修正 $\\arg C_G$", loc="left")
    phase_axis.set_ylabel("")
    amplitude_colorbar = fig.colorbar(amplitude_mesh, ax=amplitude_axis, pad=0.02, fraction=0.045)
    amplitude_colorbar.set_label("$\\ln|C_G|$", fontsize=9)
    amplitude_colorbar.ax.tick_params(labelsize=8)
    phase_colorbar = fig.colorbar(phase_mesh, ax=phase_axis, pad=0.02, fraction=0.045)
    phase_colorbar.set_label("相位修正 (°)", fontsize=9)
    phase_colorbar.ax.tick_params(labelsize=8)

    for ax, panel in zip(statistic_axes, statistic_panels):
        values, ylabel, letter, title, signed = panel
        draw_chi_statistics(ax, chi, slopes, values, ylabel, letter, title, signed=signed)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=6,
            label="$i$=%.0f°" % slope,
        )
        for slope, color in SLOPE_COLORS.items()
    ]
    legend_handles.extend(
        [
            Line2D([], [], color="#4D4D4D", lw=0.8, marker="|", markersize=10, label="同$\\chi$全距"),
            Line2D([], [], color="#111111", lw=1.2, marker="D", markersize=4, label="同$\\chi$中位数"),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=6,
        frameon=False,
    )
    fig.suptitle(
        "%s复数修正场（$i$=%.0f°、$d/h$=%.2f、$r_v$=%.2f）与64工况特征频率统计"
        % (
            REPRESENTATIVE_CASE,
            representative_features[0],
            representative_features[1],
            representative_features[2],
        ),
        fontsize=11,
        y=0.975,
    )
    fig.text(
        0.5,
        0.009,
        "$C_G$为层状坡与对应均质坡的复频响之比；星号为代表工况最大绝对幅值修正（%.1f Hz，$s$=%.2f）。统计幅值为$|\\mathrm{median}(\\Delta_A)|$，其中$\\Delta_A=\\ln|C_G|$。"
        % (correction_peak_frequency, correction_peak_s),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#4D4D4D",
    )
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
