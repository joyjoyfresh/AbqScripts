# -*- coding: utf-8 -*-
"""生成图21：12个训练外参数组合的误差排序与参数对照。脚本可独立运行。"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


REPO_ROOT = Path(__file__).resolve().parents[5]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
ML_DIR = REPO_ROOT / "Run" / "ch4_sp_ml"
UNSEEN_PATH = ML_DIR / "unseen_combination_metrics.csv"
SUMMARY_PATH = ML_DIR / "model_summary.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_STEM = "fig21_训练外组合误差排序"


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
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv_rows(path):
    """读取CSV表。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rowwise_normalize(values):
    """逐行归一化，保留各物理量内部的相对高低。"""
    values = np.asarray(values, dtype=float)
    minimum = np.nanmin(values, axis=1, keepdims=True)
    maximum = np.nanmax(values, axis=1, keepdims=True)
    return np.divide(values - minimum, maximum - minimum, out=np.zeros_like(values), where=(maximum - minimum) > 0.0)


def annotate_heatmap(ax, raw, normalized, formats):
    """在归一化热图中标注原始物理量。"""
    for row_index in range(raw.shape[0]):
        for column_index in range(raw.shape[1]):
            color = "white" if normalized[row_index, column_index] < 0.25 or normalized[row_index, column_index] > 0.74 else "#111111"
            ax.text(
                column_index,
                row_index,
                formats[row_index] % raw[row_index, column_index],
                ha="center",
                va="center",
                color=color,
                fontsize=7.2,
            )


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """按复数误差排序B组工况，并对齐参数与关键误差。"""
    set_journal_style()
    for path in (DATA_PATH, UNSEEN_PATH, SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError("未找到所需数据文件：%s" % path)

    metric_rows = read_csv_rows(UNSEEN_PATH)
    summary_rows = {row["model"]: row for row in read_csv_rows(SUMMARY_PATH)}
    with np.load(DATA_PATH, allow_pickle=False) as package:
        ids = np.asarray(package["case_ids"])
        groups = np.asarray(package["case_groups"])
        features = np.asarray(package["X"], dtype=float)
    feature_lookup = {
        str(case_id): features[index]
        for index, case_id in enumerate(ids)
        if groups[index] == "B"
    }

    ordered_rows = sorted(metric_rows, key=lambda row: float(row["E_complex_w"]))
    ordered_ids = [row["case_id"] for row in ordered_rows]
    missing = [case_id for case_id in ordered_ids if case_id not in feature_lookup]
    if missing:
        raise RuntimeError("数据集中缺少B组参数：%s" % ", ".join(missing))

    complex_error = np.asarray([float(row["E_complex_w"]) for row in ordered_rows])
    positions = np.arange(len(ordered_rows))
    cv_median = float(summary_rows["pod_gpr"]["E_complex_w_median"])
    cv_p90 = float(summary_rows["pod_gpr"]["E_complex_w_p90"])
    b_median = float(np.median(complex_error))

    fig = plt.figure(figsize=(12.7, 8.15))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.50, 0.68, 1.05], hspace=0.18)

    ax_error = fig.add_subplot(grid[0, 0])
    norm = Normalize(vmin=float(np.min(complex_error)), vmax=float(np.max(complex_error)))
    colors = plt.get_cmap("magma")(norm(complex_error))
    ax_error.bar(positions, complex_error, width=0.72, color=colors, edgecolor="white", linewidth=0.6, zorder=2)
    for position, value in zip(positions, complex_error):
        ax_error.text(position, value + 0.016, "%.3f" % value, ha="center", va="bottom", fontsize=7.1, rotation=0)
    ax_error.axhline(cv_median, color="#0072B2", lw=1.1, ls="--", label="P组CV中位 %.3f" % cv_median)
    ax_error.axhline(cv_p90, color="#D55E00", lw=1.1, ls="--", label="P组CV P90 %.3f" % cv_p90)
    ax_error.axhline(b_median, color="#222222", lw=1.0, ls=":", label="B组中位 %.3f" % b_median)
    ax_error.set_xlim(-0.58, len(positions) - 0.42)
    ax_error.set_ylim(0.0, max(complex_error) * 1.20)
    ax_error.set_xticks(positions)
    ax_error.set_xticklabels([])
    ax_error.set_ylabel(r"$E_{\mathrm{complex},w}$")
    ax_error.set_title("(a) 12个训练外参数组合的复数场误差排序", loc="left")
    ax_error.grid(axis="y", color="#DDDDDD", lw=0.45, ls=":", zorder=0)
    ax_error.spines[["top", "right"]].set_visible(False)
    ax_error.legend(ncol=3, loc="upper left", frameon=False)

    parameter_raw = np.asarray([feature_lookup[case_id] for case_id in ordered_ids], dtype=float).T
    parameter_normalized = rowwise_normalize(parameter_raw)
    ax_parameters = fig.add_subplot(grid[1, 0])
    parameter_image = ax_parameters.imshow(parameter_normalized, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    annotate_heatmap(ax_parameters, parameter_raw, parameter_normalized, ["%.1f°", "%.1f", "%.3f"])
    ax_parameters.set_yticks([0, 1, 2])
    ax_parameters.set_yticklabels(["坡角 $i$", "厚度比 $d/h$", "波速比 $r_v$"])
    ax_parameters.set_xticks(positions)
    ax_parameters.set_xticklabels([])
    ax_parameters.tick_params(length=0)
    ax_parameters.set_title("(b) 对应参数组合（逐行颜色归一化，格内为原始值）", loc="left", pad=5)
    colorbar_parameters = fig.colorbar(parameter_image, ax=ax_parameters, fraction=0.013, pad=0.012)
    colorbar_parameters.set_label("该参数内相对位置")

    metric_names = [
        "log_amplitude_rmse",
        "circular_phase_rmse_deg",
        "peak_amplitude_relative_error",
        "peak_frequency_error_hz",
        "peak_s_error",
    ]
    metric_raw = np.asarray(
        [[float(row[name]) for row in ordered_rows] for name in metric_names],
        dtype=float,
    )
    metric_raw[2] *= 100.0
    metric_normalized = rowwise_normalize(metric_raw)
    ax_metrics = fig.add_subplot(grid[2, 0])
    metric_image = ax_metrics.imshow(metric_normalized, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
    annotate_heatmap(ax_metrics, metric_raw, metric_normalized, ["%.3f", "%.1f°", "%.1f%%", "%.1f", "%.2f"])
    ax_metrics.set_yticks(np.arange(5))
    ax_metrics.set_yticklabels([
        r"$\ln|G_h|$ RMSE",
        "圆周相位RMSE",
        "峰值幅值相对误差",
        "峰值频率误差/Hz",
        r"峰值位置误差 $|\Delta s|$",
    ])
    ax_metrics.set_xticks(positions)
    ax_metrics.set_xticklabels(ordered_ids)
    ax_metrics.tick_params(length=0)
    ax_metrics.set_xlabel(r"B组训练外工况（按 $E_{\mathrm{complex},w}$ 由小到大）")
    ax_metrics.set_title("(c) 关键误差同步对照（逐行颜色归一化，格内为原始值）", loc="left", pad=5)
    colorbar_metrics = fig.colorbar(metric_image, ax=ax_metrics, fraction=0.013, pad=0.012)
    colorbar_metrics.set_label("该指标内相对误差")

    save_figure(fig)
    plt.close(fig)
    print("B组E中位=%.6f，P90=%.6f" % (b_median, float(np.percentile(complex_error, 90.0))))
    print("最小误差：%s %.6f；最大误差：%s %.6f" % (ordered_ids[0], complex_error[0], ordered_ids[-1], complex_error[-1]))


if __name__ == "__main__":
    main()
