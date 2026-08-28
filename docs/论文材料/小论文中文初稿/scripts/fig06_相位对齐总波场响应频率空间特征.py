# -*- coding: utf-8 -*-
"""生成图6：相位对齐总波场响应的频率—空间特征。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig06_相位对齐总波场响应频率空间特征"
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#5F5F5F"
LIGHT_GRAY = "#D9D9D9"


def set_journal_style():
    """设置中文论文图形样式。"""
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
            "legend.fontsize": 7.5,
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


def load_current_fields(case_ids):
    """读取相位对齐总波场响应数据。"""
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到总波场响应数据集：%s" % DATA_PATH)
    with np.load(DATA_PATH, allow_pickle=False) as data:
        raw_ids = data["case_ids"]
        all_ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_ids]
        frequency = data["frequency_hz"].astype(float)
        s = data["s"].astype(float)
        all_fields = data["G_h"].astype(complex)
        all_valid = data["valid_mask"].astype(bool)
    fields = {}
    for case_id in case_ids:
        if case_id not in all_ids:
            raise RuntimeError("数据集中缺少工况 %s" % case_id)
        index = all_ids.index(case_id)
        field = np.where(all_valid[index], all_fields[index], np.nan + 1j * np.nan)
        fields[case_id] = field
    return frequency, s, fields


def toe_jump(complex_field, s):
    """计算坡脚左邻点与坡脚点之间的复数对称相对跳变量。"""
    toe = int(np.argmin(np.abs(s - 1.0)))
    left_neighbor = int(np.flatnonzero(s < 1.0)[-1])
    before = complex_field[:, left_neighbor]
    after = complex_field[:, toe]
    denominator = np.abs(before) + np.abs(after)
    return 200.0 * np.abs(after - before) / denominator


def mark_surface_regions(ax):
    """标出坡顶和坡脚，并标注三个地表分区。"""
    ax.axvline(0.0, color="white", lw=0.9, ls="--")
    ax.axvline(1.0, color="white", lw=0.9, ls="--")
    ax.text(-2.0, 9.72, "上平台", ha="center", va="top", color="white", fontsize=7.5)
    ax.text(0.5, 9.72, "坡面", ha="center", va="top", color="white", fontsize=7.5)
    ax.text(2.5, 9.72, "下平台", ha="center", va="top", color="white", fontsize=7.5)


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
    """绘制相位对齐总波场响应、空间剖面和坡脚邻点变化。"""
    set_journal_style()
    case_ids = ["H004", "P061"]
    frequency, s, fields = load_current_fields(case_ids)
    log_fields = {case_id: np.log(np.abs(fields[case_id])) for case_id in case_ids}
    finite = np.concatenate(
        [values[np.isfinite(values)] for values in log_fields.values()]
    )
    color_min, color_max = np.percentile(finite, [1.0, 99.0])

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    meshes = []
    for ax, case_id, title, label in (
        (axes[0, 0], "H004", "均质坡 H004", "(a)"),
        (axes[0, 1], "P061", "成层坡 P061", "(b)"),
    ):
        mesh = ax.pcolormesh(
            s,
            frequency,
            log_fields[case_id],
            shading="nearest",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
        )
        meshes.append(mesh)
        mark_surface_regions(ax)
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(0.5, 10.0)
        ax.set_xlabel("归一化地表坐标 $s$")
        ax.set_ylabel("频率 $f$ (Hz)")
        ax.set_title(title)
        add_panel_label(ax, label)
    colorbar_axis = fig.add_axes([0.94, 0.57, 0.013, 0.29])
    colorbar = fig.colorbar(meshes[0], cax=colorbar_axis)
    colorbar.set_label(r"$\ln|G_h|$")
    colorbar.ax.tick_params(labelsize=8)

    ax = axes[1, 0]
    frequency_index = int(np.argmin(np.abs(frequency - 5.0)))
    for case_id, color, name in (
        ("H004", ORANGE, "均质 H004"),
        ("P061", BLUE, "成层 P061"),
    ):
        ax.plot(
            s,
            np.abs(fields[case_id][frequency_index]),
            color=color,
            lw=1.55,
            label=name,
        )
    ax.axvspan(0.0, 1.0, color=LIGHT_GRAY, alpha=0.28)
    ax.axvline(0.0, color=GRAY, lw=0.8, ls=":")
    ax.axvline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel("归一化地表坐标 $s$")
    ax.set_ylabel("幅值 $|G_h|$")
    ax.set_title("5 Hz总波场响应空间剖面")
    ax.legend(loc="upper left")
    ax.grid(True)
    add_panel_label(ax, "(c)")

    ax = axes[1, 1]
    for case_id, color, name in (
        ("H004", ORANGE, "均质 H004"),
        ("P061", BLUE, "成层 P061"),
    ):
        jump = toe_jump(fields[case_id], s)
        ax.plot(
            frequency,
            jump,
            color=color,
            lw=1.55,
            label=name + "（中位%.1f%%）" % np.nanmedian(jump),
        )
    ax.set_xlim(0.5, 10.0)
    ax.set_ylim(0.0, 75.0)
    ax.set_yticks([0.0, 20.0, 40.0, 60.0])
    ax.set_xlabel("频率 $f$ (Hz)")
    ax.set_ylabel("坡脚相邻点复数相对跳变量 (%)")
    ax.set_title("坡脚相邻点复数梯度")
    ax.legend(loc="upper right")
    ax.grid(True)
    add_panel_label(ax, "(d)")

    fig.suptitle(
        "相位对齐总波场响应的频率—空间特征",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(left=0.075, right=0.91, bottom=0.08, top=0.91, wspace=0.26, hspace=0.32)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
