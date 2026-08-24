# -*- coding: utf-8 -*-
"""生成图4：Abaqus 与 SPECFEM2D 地表 PGA 跨软件验证。

脚本只读取已有跨软件 CSV/JSON 产物，不调用任何求解器。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root() -> Path:
    """从脚本位置向上查找仓库根目录。"""
    for candidate in SCRIPT_PATH.parents:
        if (candidate / "Run").is_dir() and (candidate / "docs").is_dir():
            return candidate
    raise RuntimeError("无法从脚本位置定位仓库根目录")


REPO_ROOT = find_repo_root()
X_ROOT = REPO_ROOT / "Run" / "cross_solver_X" / "abaqus"
OUT_DIR = SCRIPT_PATH.parent.parent / "images"
OUT_STEM = "fig04_Abaqus与SPECFEM2D跨软件验证"
COMPARISONS = (
    (
        "X001：均质坡 H004",
        X_ROOT / "X001-A" / "comparison" / "x001_surface_pga.csv",
        X_ROOT / "X001-A" / "comparison" / "x001_comparison_metrics.json",
    ),
    (
        "X002：成层坡 P061（标准谱元网格）",
        X_ROOT / "X002-A" / "comparison" / "x002_surface_pga.csv",
        X_ROOT / "X002-A" / "comparison" / "x002_comparison_metrics.json",
    ),
    (
        "X002-SR：成层坡 P061（加密谱元网格）",
        X_ROOT / "X002-A" / "comparison" / "x002_sr_surface_pga.csv",
        X_ROOT / "X002-A" / "comparison" / "x002_sr_comparison_metrics.json",
    ),
)


def set_journal_style() -> None:
    """设置可印刷的中文期刊图风格。"""
    for font_path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            if Path(font_path).is_file():
                fm.fontManager.addfont(font_path)
        except Exception:
            pass
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimSun", "SimHei", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
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
            "grid.linewidth": 0.55,
            "grid.alpha": 0.38,
        }
    )


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """读取跨软件地表 PGA 曲线。"""
    if not path.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % path)
    columns = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("CSV 为空：%s" % path)
    for name in rows[0]:
        columns[name] = np.asarray([float(row[name]) for row in rows], dtype=float)
    return columns


def load_json(path: Path) -> dict:
    """读取 UTF-8 JSON 文件。"""
    if not path.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure) -> None:
    """保存同名 PNG 和 PDF。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / (OUT_STEM + ".png")
    pdf_path = OUT_DIR / (OUT_STEM + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print("已生成：%s" % png_path)
    print("已生成：%s" % pdf_path)


def draw_figure() -> None:
    """绘制三组水平/竖向 PGA 空间曲线与正式门控指标。"""
    set_journal_style()
    loaded = [(label, load_csv(csv_path), load_json(json_path))
              for label, csv_path, json_path in COMPARISONS]
    abaqus_color = "#0072B2"
    specfem_color = "#D55E00"
    fig, axes = plt.subplots(3, 2, figsize=(11.3, 8.15), sharex=True)
    panel_labels = "abcdef"

    for row_index, (row_label, data, metrics) in enumerate(loaded):
        if not bool(metrics.get("formal_gates_met", False)):
            raise RuntimeError("跨软件正式门控未全部通过：%s" % row_label)
        s_values = data["s"]
        for column_index, component in enumerate(("horizontal", "vertical")):
            ax = axes[row_index, column_index]
            suffix = "h" if component == "horizontal" else "v"
            ax.axvspan(0.0, 1.0, color="#BDBDBD", alpha=0.14)
            ax.axvline(0.0, color="#777777", lw=0.65, ls="--")
            ax.axvline(1.0, color="#777777", lw=0.65, ls="--")
            ax.plot(s_values, data["abaqus_pga_" + suffix], color=abaqus_color,
                    lw=1.65, label="Abaqus")
            ax.plot(s_values, data["specfem_pga_" + suffix], color=specfem_color,
                    lw=1.35, ls="--", label="SPECFEM2D")
            gate = metrics["formal_gates"][component + "_pga_curve"]
            ax.text(
                0.018,
                0.94,
                "NRMSE = %.1f%%\n$|\\Delta s_{peak}|$ = %.2f"
                % (100.0 * float(gate["nrmse"]), float(gate["peak_location_error"])),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#888888", lw=0.55, alpha=0.9),
            )
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(bottom=0.0)
            ax.set_ylabel(("水平" if component == "horizontal" else "竖向")
                          + " PGA (m/s$^2$)")
            ax.set_title("(%s) %s" % (panel_labels[2 * row_index + column_index],
                                       "水平分量" if component == "horizontal" else "竖向分量"))
            ax.grid(True, axis="y")
            ax.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].text(
            -0.18,
            0.5,
            row_label,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    for ax in axes[-1, :]:
        ax.set_xlabel("归一化地表坐标 $s$")
        for x_value, text_value in ((-2.0, "上平台"), (0.5, "坡面"), (2.5, "下平台")):
            ax.text(
                x_value,
                0.035,
                text_value,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72),
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.57, 0.957),
               ncol=2, handlelength=2.8)
    fig.suptitle(
        "Abaqus—SPECFEM2D 地表 PGA 空间分布对比（4 Hz Ricker，$15^\\circ$ SV 斜入射）",
        fontsize=11,
        y=0.995,
    )
    fig.text(
        0.57,
        0.025,
        "正式门控：水平 NRMSE $\\leq$ 10%，竖向 NRMSE $\\leq$ 15%，"
        "$|\\Delta s_{peak}|\\leq0.10$；三组对比均满足",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.12, top=0.925, wspace=0.24, hspace=0.32)
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    """单独生成图4。"""
    draw_figure()


if __name__ == "__main__":
    main()
