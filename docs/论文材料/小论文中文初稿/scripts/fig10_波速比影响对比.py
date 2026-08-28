# -*- coding: utf-8 -*-
"""生成图10：波速比影响对比。脚本可独立运行。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig10_波速比影响对比"

SLOPE_ANGLE = 45.0
THICKNESS_RATIO = 0.60
VELOCITY_RATIOS = np.array([0.30, 0.45, 0.60, 0.75])
LINE_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
SEGMENT_SPANS = [(-4.0, 0.0, "上平台"), (0.0, 1.0, "坡面"), (1.0, 4.0, "下平台")]


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
            "axes.titlesize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def select_cases(package):
    """严格固定坡角和厚度比，仅按波速比选取四个工况。"""
    groups = package["case_groups"]
    features = package["X"]
    indices = []
    for velocity_ratio in VELOCITY_RATIOS:
        matched = np.flatnonzero(
            (groups == "P")
            & np.isclose(features[:, 0], SLOPE_ANGLE)
            & np.isclose(features[:, 1], THICKNESS_RATIO)
            & np.isclose(features[:, 2], velocity_ratio)
        )
        if matched.size != 1:
            raise RuntimeError("波速比 %.2f 未唯一匹配到工况" % velocity_ratio)
        indices.append(int(matched[0]))
    return np.asarray(indices, dtype=int)


def add_segment_guides(ax, label_segments=False):
    """标出坡肩、坡脚及三个地表区段。"""
    fills = ["#EAF3F8", "#FFF2CC", "#E8F3EA"]
    for (left, right, label), color in zip(SEGMENT_SPANS, fills):
        ax.axvspan(left, right, color=color, alpha=0.30, zorder=0)
        if label_segments:
            ax.text(
                (left + right) / 2.0,
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#4D4D4D",
            )
    for boundary in (0.0, 1.0):
        ax.axvline(boundary, color="#666666", lw=0.8, ls="--", zorder=2)


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """绘制固定坡角、厚度比下的波速比单因素序列。"""
    set_journal_style()
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到复频响数据集：%s" % DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=True) as package:
        indices = select_cases(package)
        case_ids = package["case_ids"][indices]
        frequency = package["frequency_hz"]
        s_values = package["s"]
        amplitude = package["amplitude"][indices]
        valid = package["valid_mask"][indices]

    shown = np.where(valid, amplitude, np.nan)
    color_max = float(np.nanmax(shown))

    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(
        2,
        5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.12, 0.88],
        left=0.065,
        right=0.955,
        bottom=0.105,
        top=0.91,
        wspace=0.16,
        hspace=0.31,
    )
    heat_axes = [fig.add_subplot(grid[0, column]) for column in range(4)]
    color_axis = fig.add_subplot(grid[0, 4])
    envelope_axis = fig.add_subplot(grid[1, :4])

    letters = "abcd"
    mesh = None
    for order, (ax, case_id, velocity_ratio) in enumerate(
        zip(heat_axes, case_ids, VELOCITY_RATIOS)
    ):
        masked = np.ma.masked_invalid(shown[order])
        mesh = ax.pcolormesh(
            s_values,
            frequency,
            masked,
            shading="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=color_max,
            rasterized=True,
        )
        for boundary in (0.0, 1.0):
            ax.axvline(boundary, color="white", lw=0.8, ls="--", alpha=0.9)

        flat_index = int(np.nanargmax(shown[order]))
        frequency_index, spatial_index = np.unravel_index(flat_index, shown[order].shape)
        peak_value = shown[order, frequency_index, spatial_index]
        peak_frequency = frequency[frequency_index]
        peak_s = s_values[spatial_index]
        ax.scatter(
            [peak_s],
            [peak_frequency],
            marker="*",
            s=58,
            facecolor="#F0E442",
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
        ax.text(
            0.03,
            0.04,
            "峰值 %.2f\n(%.1f Hz, $s$=%.2f)" % (peak_value, peak_frequency, peak_s),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.6,
            bbox=dict(facecolor="white", edgecolor="#B3B3B3", lw=0.5, alpha=0.88),
        )
        ax.set_title(
            "(%s) $r_v$=%.2f（%s）" % (letters[order], velocity_ratio, case_id),
            pad=6,
        )
        ax.set_xlim(s_values.min(), s_values.max())
        ax.set_ylim(frequency.min(), frequency.max())
        ax.set_xticks([-4, 0, 1, 4])
        ax.set_xlabel("地表坐标 $s$")
        if order == 0:
            ax.set_ylabel("频率 $f$ (Hz)")
        else:
            ax.set_yticklabels([])

    colorbar = fig.colorbar(mesh, cax=color_axis)
    colorbar.set_label("幅值 $|G_h|$", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    add_segment_guides(envelope_axis, label_segments=True)
    for order, velocity_ratio in enumerate(VELOCITY_RATIOS):
        envelope = np.nanmax(shown[order], axis=0)
        envelope_axis.plot(
            s_values,
            envelope,
            color=LINE_COLORS[order],
            lw=1.7,
            label="$r_v$=%.2f" % velocity_ratio,
            zorder=3,
        )
    envelope_axis.axhline(1.0, color="#555555", lw=0.8, ls=":", zorder=1)
    envelope_axis.set_xlim(s_values.min(), s_values.max())
    envelope_axis.set_xticks(np.arange(-4, 4.1, 1.0))
    envelope_axis.set_xlabel("地表坐标 $s$")
    envelope_axis.set_ylabel(r"频带内峰值包络 $\max_f |G_h|$")
    envelope_axis.set_title("(e) 全频带空间峰值包络", loc="left", pad=6)
    envelope_axis.legend(loc="upper left", ncol=4)
    envelope_axis.grid(axis="y", color="#D9D9D9", lw=0.5, ls=":")
    envelope_axis.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "覆盖层波速比单因素对比：固定 $i$=45°、$d/h$=0.60",
        fontsize=11,
        y=0.975,
    )
    fig.text(
        0.51,
        0.018,
        "星号为各工况全频—全空间峰值；全地表统一以左侧上平台一维自由场为分母。",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#4D4D4D",
    )
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
