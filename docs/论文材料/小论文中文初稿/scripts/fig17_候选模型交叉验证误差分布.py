# -*- coding: utf-8 -*-
"""生成图17：三种候选代理的交叉验证误差分布与多指标比较。脚本可独立运行。"""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[4]
ML_DIR = REPO_ROOT / "Run" / "ch4_sp_ml"
CV_PATH = ML_DIR / "cv_case_metrics.csv"
SUMMARY_PATH = ML_DIR / "model_summary.csv"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig17_候选模型交叉验证误差分布"

MODEL_ORDER = ["nearest", "pod_ridge", "pod_gpr"]
MODEL_LABELS = {"nearest": "最近邻", "pod_ridge": "POD-Ridge", "pod_gpr": "POD-GPR"}
MODEL_COLORS = {"nearest": "#999999", "pod_ridge": "#E69F00", "pod_gpr": "#0072B2"}


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
    """读取CSV并保留原始字段名。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig):
    """同时输出300 dpi PNG和矢量PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """绘制64个完整工况交叉验证结果和候选模型多指标对比。"""
    set_journal_style()
    for path in (CV_PATH, SUMMARY_PATH):
        if not path.exists():
            raise FileNotFoundError("未找到代理模型评价文件：%s" % path)

    cv_rows = read_csv_rows(CV_PATH)
    summary_rows = {row["model"]: row for row in read_csv_rows(SUMMARY_PATH)}
    case_ids = sorted({row["case_id"] for row in cv_rows})
    case_model_error = {
        (row["case_id"], row["model"]): float(row["E_complex_w"])
        for row in cv_rows
    }
    ordered_cases = sorted(case_ids, key=lambda case_id: case_model_error[(case_id, "pod_gpr")])

    fig = plt.figure(figsize=(12.5, 7.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[0.78, 1.32], hspace=0.36, wspace=0.26)

    ax_cases = fig.add_subplot(grid[0, :])
    ranks = np.arange(1, len(ordered_cases) + 1)
    for model in MODEL_ORDER:
        values = np.asarray([case_model_error[(case_id, model)] for case_id in ordered_cases])
        ax_cases.plot(
            ranks,
            values,
            color=MODEL_COLORS[model],
            lw=1.05,
            marker="o",
            ms=2.8,
            alpha=0.82,
            label=MODEL_LABELS[model],
        )
    ax_cases.set_xlim(1, len(ranks))
    ax_cases.set_ylim(bottom=0.0)
    ax_cases.set_xticks(np.arange(1, len(ranks) + 1, 8))
    ax_cases.set_xlabel("P组工况序位（按POD-GPR误差由小到大）")
    ax_cases.set_ylabel(r"加权复数相对误差 $E_{\mathrm{complex},w}$")
    ax_cases.set_title("(a) 64个完整物理工况的逐工况交叉验证误差", loc="left")
    ax_cases.grid(color="#DDDDDD", lw=0.45, ls=":")
    ax_cases.spines[["top", "right"]].set_visible(False)
    ax_cases.legend(ncol=3, loc="upper left", frameon=False)

    ax_dist = fig.add_subplot(grid[1, 0])
    rng = np.random.default_rng(20260824)
    distributions = []
    for index, model in enumerate(MODEL_ORDER, start=1):
        values = np.asarray(
            [float(row["E_complex_w"]) for row in cv_rows if row["model"] == model],
            dtype=float,
        )
        distributions.append(values)
        jitter = rng.uniform(-0.13, 0.13, len(values))
        ax_dist.scatter(
            index + jitter,
            values,
            s=14,
            facecolor=MODEL_COLORS[model],
            edgecolor="white",
            linewidth=0.3,
            alpha=0.52,
            zorder=2,
        )
        median = float(np.median(values))
        p90 = float(np.percentile(values, 90.0))
        ax_dist.hlines(median, index - 0.24, index + 0.24, color="#111111", lw=1.5, zorder=4)
        ax_dist.scatter(index, p90, marker="^", s=36, color="#D55E00", edgecolor="white", linewidth=0.45, zorder=5)
        ax_dist.text(index, p90 + 0.055, "%.3f" % p90, ha="center", va="bottom", fontsize=7.6, color="#8B2E16")
        ax_dist.text(index, median - 0.045, "中位 %.3f" % median, ha="center", va="top", fontsize=7.5)
    boxes = ax_dist.boxplot(
        distributions,
        positions=np.arange(1, 4),
        widths=0.50,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="none"),
        whiskerprops=dict(color="#555555", lw=0.8),
        capprops=dict(color="#555555", lw=0.8),
    )
    for patch, model in zip(boxes["boxes"], MODEL_ORDER):
        patch.set(facecolor=MODEL_COLORS[model], edgecolor="#555555", linewidth=0.7, alpha=0.18)
    ax_dist.set_xticks([1, 2, 3])
    ax_dist.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax_dist.set_ylabel(r"$E_{\mathrm{complex},w}$")
    ax_dist.set_ylim(0.0, max(max(values) for values in distributions) * 1.12)
    ax_dist.set_title("(b) 工况间误差分布及P90", loc="left")
    ax_dist.grid(axis="y", color="#DDDDDD", lw=0.45, ls=":")
    ax_dist.spines[["top", "right"]].set_visible(False)
    ax_dist.legend(
        handles=[
            Line2D([], [], color="#111111", lw=1.5, label="中位数"),
            Line2D([], [], marker="^", ls="none", ms=5.5, color="#D55E00", label="P90"),
        ],
        loc="upper left",
        frameon=False,
    )

    ax_metrics = fig.add_subplot(grid[1, 1])
    metric_specs = [
        ("E_complex_w", "$E_c$", "%.3f"),
        ("log_amplitude_rmse", r"$\ln|G|$", "%.3f"),
        ("circular_phase_rmse_deg", "相位/(°)", "%.1f"),
        ("group_delay_rmse_s", "群时延/s", "%.3f"),
        ("spatial_phase_gradient_rmse_rad_per_s", "空间相位梯度", "%.2f"),
    ]
    raw = np.empty((len(MODEL_ORDER), len(metric_specs) * 2), dtype=float)
    labels = []
    formats = []
    for metric_index, (metric, label, value_format) in enumerate(metric_specs):
        labels.extend([label + "\n中位", label + "\nP90"])
        formats.extend([value_format, value_format])
        for model_index, model in enumerate(MODEL_ORDER):
            raw[model_index, metric_index * 2] = float(summary_rows[model][metric + "_median"])
            raw[model_index, metric_index * 2 + 1] = float(summary_rows[model][metric + "_p90"])
    normalized = raw / np.nanmin(raw, axis=0, keepdims=True)
    image = ax_metrics.imshow(normalized, cmap="cividis", vmin=1.0, vmax=max(2.0, float(np.nanpercentile(normalized, 95.0))), aspect="auto")
    for row_index in range(raw.shape[0]):
        for column_index in range(raw.shape[1]):
            text_color = "white" if normalized[row_index, column_index] < 1.40 else "#111111"
            ax_metrics.text(
                column_index,
                row_index,
                formats[column_index] % raw[row_index, column_index],
                ha="center",
                va="center",
                fontsize=7.0,
                color=text_color,
            )
    ax_metrics.set_xticks(np.arange(len(labels)))
    ax_metrics.set_xticklabels(labels, rotation=38, ha="right")
    ax_metrics.set_yticks(np.arange(len(MODEL_ORDER)))
    ax_metrics.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax_metrics.set_title("(c) 多指标中位数与P90（格内为原始值）", loc="left")
    ax_metrics.tick_params(length=0)
    for x_position in np.arange(1.5, len(labels), 2.0):
        ax_metrics.axvline(x_position, color="white", lw=1.5)
    colorbar = fig.colorbar(image, ax=ax_metrics, fraction=0.030, pad=0.025)
    colorbar.set_label("相对该列最优值（越小越好）")

    save_figure(fig)
    plt.close(fig)

    for model in MODEL_ORDER:
        row = summary_rows[model]
        print(
            "%s：E中位=%.6f，P90=%.6f"
            % (MODEL_LABELS[model], float(row["E_complex_w_median"]), float(row["E_complex_w_p90"]))
        )


if __name__ == "__main__":
    main()
