# -*- coding: utf-8 -*-
"""生成图8：1、3、5、7、9 Hz固定频率地表空间响应。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_FILE = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig08_固定频率地表空间响应"
TARGET_FREQUENCIES = (1.0, 3.0, 5.0, 7.0, 9.0)

COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1.5)))
REGION_COLOR = "#D9D9D9"
BOUNDARY_COLOR = "#555555"


def set_journal_style():
    """设置中文期刊图样式。"""
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


def load_profiles():
    """读取两个代表工况指定频率的幅值和相位剖面。"""
    result = {}
    with np.load(DATA_FILE, allow_pickle=False) as data:
        case_ids = [str(value) for value in data["case_ids"]]
        frequency = data["frequency_hz"].astype(float)
        s = data["s"].astype(float)
        indices = [int(np.argmin(np.abs(frequency - value))) for value in TARGET_FREQUENCIES]
        actual = [float(frequency[index]) for index in indices]
        if max(abs(a - b) for a, b in zip(actual, TARGET_FREQUENCIES)) > 1.0e-8:
            raise RuntimeError("数据频率网格不包含指定频率：%s" % actual)
        for case_id in ("H004", "P061"):
            case_index = case_ids.index(case_id)
            valid = data["valid_mask"][case_index, indices]
            amplitude = data["amplitude"][case_index, indices].astype(float)
            phase_frequency = data["phase_unwrapped_rad"][case_index, indices].astype(float)
            phase_spatial = np.unwrap(
                np.angle(data["G_h"][case_index, indices].astype(complex)),
                axis=1,
            )
            phase_offset = 2.0 * np.pi * np.round(
                (phase_frequency[:, 0] - phase_spatial[:, 0]) / (2.0 * np.pi)
            )
            phase = np.degrees(phase_spatial + phase_offset[:, None])
            result[case_id] = {
                "amplitude": np.where(valid, amplitude, np.nan),
                "phase": np.where(valid, phase, np.nan),
            }
    return s, result


def mark_regions(ax, show_names=False):
    """标注上平台、坡面与下平台。"""
    ax.axvspan(0.0, 1.0, color=REGION_COLOR, alpha=0.28, zorder=0)
    ax.axvline(0.0, color=BOUNDARY_COLOR, lw=0.75, ls="--")
    ax.axvline(1.0, color=BOUNDARY_COLOR, lw=0.75, ls="--")
    if show_names:
        y = 0.96
        ax.text(0.25, y, "上平台", transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        ax.text(0.56, y, "坡面", transform=ax.transAxes, ha="center", va="top", fontsize=7.5)
        ax.text(0.80, y, "下平台", transform=ax.transAxes, ha="center", va="top", fontsize=7.5)


def add_panel_label(ax, label):
    """添加分图编号。"""
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def save_figure(fig):
    """保存300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / (OUTPUT_STEM + ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("已生成：%s" % png_path)


def main():
    """绘制固定频率下的地表幅值与相位空间剖面。"""
    set_journal_style()
    s, profiles = load_profiles()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.0), sharex=True)
    configurations = (
        (axes[0, 0], "H004", "amplitude", "均质 H004：幅值", "幅值 $|G_h|$", "(a)"),
        (axes[0, 1], "P061", "amplitude", "成层 P061：幅值", "幅值 $|G_h|$", "(b)"),
        (axes[1, 0], "H004", "phase", "均质 H004：相位", "展开相位 $\\Phi_h$ ($^\\circ$)", "(c)"),
        (axes[1, 1], "P061", "phase", "成层 P061：相位", "展开相位 $\\Phi_h$ ($^\\circ$)", "(d)"),
    )
    for ax, case_id, field_name, title, ylabel, panel in configurations:
        for row, (frequency, color, linestyle) in enumerate(
            zip(TARGET_FREQUENCIES, COLORS, LINESTYLES)
        ):
            ax.plot(
                s,
                profiles[case_id][field_name][row],
                color=color,
                ls=linestyle,
                lw=1.35,
                label="%g Hz" % frequency,
            )
        mark_regions(ax, show_names=field_name == "amplitude")
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        add_panel_label(ax, panel)
    for ax in axes[1, :]:
        ax.set_xlabel("归一化地表坐标 $s$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle(
        "扣除斜入射水平传播相位后的地表空间响应（虚线为坡顶 $s=0$ 与坡脚 $s=1$）",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.03, 0.02, 0.995, 0.90), w_pad=2.3, h_pad=2.0)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
