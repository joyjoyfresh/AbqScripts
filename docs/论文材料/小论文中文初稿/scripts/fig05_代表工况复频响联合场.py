# -*- coding: utf-8 -*-
"""生成图5：代表工况复频响幅值、相位与群时延联合场。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_FILE = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig05_代表工况复频响联合场"

AMPLITUDE_CMAP = LinearSegmentedColormap.from_list(
    "amplitude_accessible",
    ["#F7FCFD", "#C7E9B4", "#7FCDBB", "#2C7FB8", "#253494"],
)
PHASE_CMAP = LinearSegmentedColormap.from_list(
    "phase_accessible",
    ["#3B4CC0", "#8DB0FE", "#F7F7F7", "#F6A385", "#B40426"],
)


def set_journal_style():
    """设置中文论文统一样式。"""
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
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_fields():
    """读取H004与P061统一左参考复频响联合场。"""
    rows = []
    with np.load(DATA_FILE, allow_pickle=False) as data:
        case_ids = [str(value) for value in data["case_ids"]]
        frequency = data["frequency_hz"].astype(float)
        s = data["s"].astype(float)
        for case_id, label in (
            ("H004", "均质 H004\n$i=60^\\circ$"),
            ("P061", r"成层 P061" + "\n" + r"$i=60^\circ,\ d/h=1.40,\ r_v=0.30$"),
        ):
            index = case_ids.index(case_id)
            valid = data["valid_mask"][index].astype(bool)
            phase_frequency = data["phase_unwrapped_rad"][index].astype(float)
            phase_spatial = np.unwrap(np.angle(data["G_h"][index].astype(complex)), axis=1)
            phase_offset = 2.0 * np.pi * np.round(
                (phase_frequency[:, 0] - phase_spatial[:, 0]) / (2.0 * np.pi)
            )
            phase_spatial = phase_spatial + phase_offset[:, None]
            rows.append(
                {
                    "case_id": case_id,
                    "label": label,
                    "valid": valid,
                    "log_amplitude": data["log_amplitude"][index].astype(float),
                    "phase_deg": np.degrees(phase_spatial),
                    "delay": data["group_delay_s"][index].astype(float),
                }
            )
    return frequency, s, rows


def finite_limits(rows, key, symmetric=False):
    """取得两工况共享的完整色阶范围。"""
    values = []
    for row in rows:
        field = row[key]
        mask = row["valid"] & np.isfinite(field)
        values.append(field[mask])
    merged = np.concatenate(values)
    if symmetric:
        limit = float(np.max(np.abs(merged)))
        return -limit, limit
    return float(np.min(merged)), float(np.max(merged))


def mark_regions(ax, top_row=False):
    """标出坡顶、坡脚及地表分区。"""
    ax.axvline(0.0, color="#222222", lw=0.7, ls="--", alpha=0.9)
    ax.axvline(1.0, color="#222222", lw=0.7, ls="--", alpha=0.9)
    if top_row:
        ax.text(-2.0, 9.70, "上平台", ha="center", va="top", fontsize=7.3)
        ax.text(0.5, 9.70, "坡面", ha="center", va="top", fontsize=7.3)
        ax.text(2.5, 9.70, "下平台", ha="center", va="top", fontsize=7.3)


def save_figure(fig):
    """保存300 dpi PNG与PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / (OUTPUT_STEM + ".png")
    pdf_path = OUTPUT_DIR / (OUTPUT_STEM + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print("已生成：%s" % png_path)
    print("已生成：%s" % pdf_path)


def main():
    """绘制两类代表工况的三指标二维联合场。"""
    set_journal_style()
    frequency, s, rows = load_fields()
    specifications = [
        ("log_amplitude", "$\\ln|G_h|$", AMPLITUDE_CMAP, finite_limits(rows, "log_amplitude")),
        ("phase_deg", "$\\Phi_h$ ($^\\circ$)", PHASE_CMAP, finite_limits(rows, "phase_deg", True)),
        ("delay", "$\\tau_g$ (s)", PHASE_CMAP, finite_limits(rows, "delay", True)),
    ]

    fig = plt.figure(figsize=(13.2, 7.2))
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=(1.0, 1.0, 0.065),
        left=0.105,
        right=0.985,
        bottom=0.105,
        top=0.90,
        wspace=0.10,
        hspace=0.17,
    )
    axes = np.empty((2, 3), dtype=object)
    colorbar_axes = []
    for row_index in range(2):
        for column_index in range(3):
            axes[row_index, column_index] = fig.add_subplot(
                grid[row_index, column_index],
                sharex=axes[0, column_index] if row_index == 1 else None,
                sharey=axes[row_index, 0] if column_index > 0 else None,
            )
    for column_index in range(3):
        colorbar_axes.append(fig.add_subplot(grid[2, column_index]))
    meshes = []
    panel_index = 0
    for row_index, row in enumerate(rows):
        for column_index, (key, title, cmap, limits) in enumerate(specifications):
            ax = axes[row_index, column_index]
            field = np.where(row["valid"] & np.isfinite(row[key]), row[key], np.nan)
            mesh = ax.pcolormesh(
                s,
                frequency,
                field,
                shading="nearest",
                cmap=cmap,
                vmin=limits[0],
                vmax=limits[1],
            )
            meshes.append(mesh)
            mark_regions(ax, top_row=row_index == 0)
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(0.5, 10.0)
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(row["label"] + "\n频率 $f$ (Hz)")
            elif row_index >= 0:
                ax.tick_params(labelleft=False)
            if row_index == 0:
                ax.tick_params(labelbottom=False)
            ax.text(
                0.018,
                0.965,
                "(%s)" % "abcdef"[panel_index],
                transform=ax.transAxes,
                va="top",
                fontsize=9.5,
                fontweight="bold",
                color="#111111",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
            )
            panel_index += 1

    for column_index, (_key, title, _cmap, _limits) in enumerate(specifications):
        colorbar = fig.colorbar(meshes[column_index], cax=colorbar_axes[column_index], orientation="horizontal")
        colorbar.set_label(title, fontsize=8.5)
        colorbar.ax.tick_params(labelsize=7.5)

    fig.suptitle("统一左侧一维自由场参考下的代表工况复频响联合场", fontsize=11, y=0.985)
    fig.supxlabel("归一化地表坐标 $s$", fontsize=9, y=0.025)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
