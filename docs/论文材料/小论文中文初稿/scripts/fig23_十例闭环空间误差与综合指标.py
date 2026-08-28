# -*- coding: utf-8 -*-
"""生成图23：十例闭环逐位置时程误差与综合指标。脚本可独立运行。"""

import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[4]
CLOSURE_ROOT = REPO_ROOT / "Run" / "ch4_sp_reconstruction_closure"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig23_十例闭环空间误差与综合指标"
CASE_ORDER = tuple("C%03d" % index for index in range(1, 11))


def set_journal_style():
    """设置中文期刊绘图样式。"""
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
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_spatial_metrics(path):
    """读取逐位置闭环指标CSV。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in (
            "s",
            "time_nrmse",
            "correlation",
            "pga_relative_error",
            "peak_time_error_s",
            "response_spectrum_relative_error_median",
        )
    }


def load_all_cases():
    """读取10个闭环工况的逐位置结果与汇总元数据。"""
    cases = []
    for case_id in CASE_ORDER:
        case_dir = CLOSURE_ROOT / case_id
        csv_path = case_dir / "reconstruction_by_s.csv"
        json_path = case_dir / "reconstruction_metrics.json"
        if not csv_path.exists() or not json_path.exists():
            raise FileNotFoundError("闭环工况产物不完整：%s" % case_dir)
        spatial = read_spatial_metrics(csv_path)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        cases.append({"case_id": case_id, "spatial": spatial, "metadata": metadata, "path": csv_path})
    return cases


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % path)


def draw_exceedance_runs(ax, s_values, matrix, threshold):
    """逐工况标出超线的连续空间区段，避免在离散工况之间作伪插值。"""
    for row, values in enumerate(matrix):
        above = np.asarray(values > threshold, dtype=bool)
        changes = np.diff(np.r_[False, above, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1) - 1
        for start, stop in zip(starts, stops):
            ax.plot(
                [s_values[start], s_values[stop]],
                [row, row],
                color="white",
                lw=1.15,
                solid_capstyle="butt",
            )


def metric_panel(ax, cases, key, title, threshold, lower_is_better=True, auxiliary=False):
    """绘制一个逐工况中位指标面板并突出超线工况。"""
    values = np.asarray([np.median(item["spatial"][key]) for item in cases], dtype=float)
    if lower_is_better:
        fail = values > threshold
    else:
        fail = values < threshold
    x = np.arange(len(cases))
    colors = np.where(fail, "#D55E00", "#0072B2")
    ax.plot(x, values, color="#999999", lw=0.65, zorder=1)
    ax.scatter(x, values, c=colors, s=30, edgecolor="white", linewidth=0.55, zorder=3)
    line_style = ":" if auxiliary else "--"
    line_color = "#777777" if auxiliary else "#D55E00"
    ax.axhline(threshold, color=line_color, lw=0.9, ls=line_style)
    ax.set_title(title, loc="left", fontsize=8.7)
    ax.set_xticks(x)
    ax.set_xticklabels(CASE_ORDER, rotation=55, ha="right")
    ax.grid(axis="y", color="#DDDDDD", lw=0.4, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ymin = min(float(np.min(values)), threshold)
    ymax = max(float(np.max(values)), threshold)
    span = max(ymax - ymin, 1.0e-4)
    ax.set_ylim(max(0.0, ymin - 0.10 * span), ymax + 0.14 * span)
    if auxiliary:
        line_text = "10%辅助线"
    elif lower_is_better:
        line_text = "≤%.3g" % threshold
    else:
        line_text = "≥%.3g" % threshold
    ax.text(
        0.98,
        0.94,
        "%s；超线%d例" % (line_text, int(np.sum(fail))),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
        color=line_color,
    )
    return values, fail


def main():
    """绘制10×161时程NRMSE热图和五项逐工况闭环指标。"""
    set_journal_style()
    cases = load_all_cases()
    s_values = cases[0]["spatial"]["s"]
    nrmse = np.vstack([item["spatial"]["time_nrmse"] for item in cases])
    if nrmse.shape != (10, 161):
        raise ValueError("预期10×161逐位置矩阵，实际为%s。" % (nrmse.shape,))

    fig = plt.figure(figsize=(13.4, 7.7))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.22, 0.78], hspace=0.42, wspace=0.34)
    ax_heat = fig.add_subplot(grid[0, :])
    vmax = max(0.35, float(np.percentile(nrmse, 99.5)))
    image = ax_heat.imshow(
        nrmse,
        origin="upper",
        aspect="auto",
        extent=(s_values[0], s_values[-1], 9.5, -0.5),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    draw_exceedance_runs(ax_heat, s_values, nrmse, 0.10)
    ax_heat.axvline(0.0, color="#333333", lw=0.65, ls="--")
    ax_heat.axvline(1.0, color="#333333", lw=0.65, ls="--")
    ax_heat.set_xlim(-4.0, 4.0)
    ax_heat.set_yticks(np.arange(10))
    ax_heat.set_yticklabels(CASE_ORDER)
    ax_heat.set_xlabel("归一化地表坐标 $s$")
    ax_heat.set_ylabel("闭环工况")
    ax_heat.set_title("(a) 十例闭环逐位置时程NRMSE（白线标出超过10%的空间区段）", loc="left")
    ax_heat.text(-2.0, -0.22, "A 上平台", ha="center", va="bottom", fontsize=8.0, color="#333333")
    ax_heat.text(0.5, -0.22, "B 坡面", ha="center", va="bottom", fontsize=8.0, color="#333333")
    ax_heat.text(2.5, -0.22, "C 下平台", ha="center", va="bottom", fontsize=8.0, color="#333333")
    colorbar = fig.colorbar(image, ax=ax_heat, pad=0.012, fraction=0.025)
    colorbar.set_label("时程NRMSE")
    colorbar.ax.axhline(0.10, color="white", lw=1.2)

    nrmse_medians = np.asarray([np.median(item["spatial"]["time_nrmse"]) for item in cases])
    for label, value in zip(ax_heat.get_yticklabels(), nrmse_medians):
        if value > 0.10:
            label.set_color("#D55E00")
            label.set_weight("bold")

    panels = []
    panels.append(
        metric_panel(fig.add_subplot(grid[1, 0]), cases, "time_nrmse", "(b) 时程NRMSE", 0.10)
    )
    panels.append(
        metric_panel(
            fig.add_subplot(grid[1, 1]), cases, "correlation", "(c) 相关系数", 0.95,
            lower_is_better=False,
        )
    )
    panels.append(
        metric_panel(
            fig.add_subplot(grid[1, 2]), cases, "pga_relative_error", "(d) PGA相对误差", 0.10
        )
    )
    panels.append(
        metric_panel(
            fig.add_subplot(grid[1, 3]), cases, "peak_time_error_s", "(e) 峰值时刻误差/s", 0.02
        )
    )
    panels.append(
        metric_panel(
            fig.add_subplot(grid[1, 4]), cases, "response_spectrum_relative_error_median",
            "(f) 反应谱相对误差", 0.10, auxiliary=True,
        )
    )
    save_figure(fig)
    plt.close(fig)

    metric_names = ("NRMSE", "相关系数", "PGA误差", "峰时误差", "反应谱误差")
    overall = [float(np.median(values)) for values, _ in panels]
    print("数据来源：%s/C001—C010/reconstruction_by_s.csv" % CLOSURE_ROOT)
    print(
        "十例逐工况中位指标的总体中位：NRMSE %.1f%%，相关系数 %.3f，PGA误差 %.1f%%，峰时误差 %.3f s，反应谱误差 %.1f%%"
        % (100.0 * overall[0], overall[1], 100.0 * overall[2], overall[3], 100.0 * overall[4])
    )
    for name, (_, fail) in zip(metric_names, panels):
        failed_cases = [case_id for case_id, is_failed in zip(CASE_ORDER, fail) if is_failed]
        print("%s超线工况：%s" % (name, "、".join(failed_cases) if failed_cases else "无"))
    print("逐位置NRMSE超过10%%的单元占比为%.1f%%，矩阵最大值为%.3f。" % (100.0 * np.mean(nrmse > 0.10), np.max(nrmse)))


if __name__ == "__main__":
    main()
