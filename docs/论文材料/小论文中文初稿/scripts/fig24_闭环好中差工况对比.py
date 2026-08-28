# -*- coding: utf-8 -*-
"""生成图24：真实波闭环好、中、差工况的时程与全地表对比。脚本可独立运行。"""

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
OUTPUT_STEM = "fig24_闭环好中差工况对比"
CASE_ORDER = ("C003", "C008", "C001")
CASE_LEVEL = {"C003": "好", "C008": "中", "C001": "差"}
RECORD_LABELS = {
    "sp_eq01_el_centro_0p1g_dt1ms": "El Centro",
    "sp_eq02_kobe_0p1g_dt1ms": "Kobe",
    "sp_eq03_chichi_0p1g_dt1ms": "Chi-Chi",
}


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
            "axes.titlesize": 9.3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_case(case_id):
    """读取单个闭环工况的时程、逐位置指标和元数据。"""
    case_dir = CLOSURE_ROOT / case_id
    npz_path = case_dir / "reconstruction.npz"
    metrics_path = case_dir / "reconstruction_metrics.json"
    if not npz_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("闭环工况产物不完整：%s" % case_dir)
    data = np.load(npz_path, allow_pickle=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "case_id": case_id,
        "path": npz_path,
        "data": data,
        "metrics": metrics,
        "time": np.asarray(data["time"], float),
        "s": np.asarray(data["s"], float),
        "prediction": np.asarray(data["reconstructed_acc_h"], float),
        "truth": np.asarray(data["direct_fe_acc_h_bandlimited"], float),
        "taf_prediction": np.asarray(data["taf_reconstructed_left_reference"], float),
        "taf_truth": np.asarray(data["direct_fe_taf"], float),
    }


def point_metrics(truth, prediction, time):
    """逐位置计算时程NRMSE与相关系数。"""
    residual = prediction - truth
    denominator = np.sqrt(np.mean(truth ** 2, axis=1))
    nrmse = np.sqrt(np.mean(residual ** 2, axis=1)) / np.maximum(denominator, 1.0e-30)
    correlations = np.empty(truth.shape[0], dtype=float)
    for index in range(truth.shape[0]):
        left = truth[index] - np.mean(truth[index])
        right = prediction[index] - np.mean(prediction[index])
        norm = np.linalg.norm(left) * np.linalg.norm(right)
        correlations[index] = np.dot(left, right) / max(norm, 1.0e-30)
    pga_truth = np.max(np.abs(truth), axis=1)
    pga_prediction = np.max(np.abs(prediction), axis=1)
    pga_error = np.abs(pga_prediction / np.maximum(pga_truth, 1.0e-30) - 1.0)
    peak_truth = time[np.argmax(np.abs(truth), axis=1)]
    peak_prediction = time[np.argmax(np.abs(prediction), axis=1)]
    peak_time_error = np.abs(peak_prediction - peak_truth)
    return nrmse, correlations, pga_error, peak_time_error


def shade_segments(ax, labels=False):
    """标示上平台、坡面和下平台三个空间区段。"""
    spans = ((-4.0, 0.0, "A", "#0072B2"), (0.0, 1.0, "B", "#E69F00"), (1.0, 4.0, "C", "#009E73"))
    for left, right, label, color in spans:
        ax.axvspan(left, right, color=color, alpha=0.055, lw=0)
        if labels:
            ax.text((left + right) / 2.0, 0.96, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8, color=color, weight="bold")
    ax.axvline(0.0, color="#777777", lw=0.55, ls="--")
    ax.axvline(1.0, color="#777777", lw=0.55, ls="--")


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % path)


def main():
    """绘制代表位置时程、全地表NRMSE和统一左参考TAF。"""
    set_journal_style()
    cases = [load_case(case_id) for case_id in CASE_ORDER]
    for item in cases:
        item["derived"] = point_metrics(item["truth"], item["prediction"], item["time"])

    nrmse_max = max(float(np.max(item["derived"][0])) for item in cases)
    taf_max = max(float(np.max(item["taf_truth"])) for item in cases)
    taf_max = max(taf_max, max(float(np.max(item["taf_prediction"])) for item in cases))

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.2, 8.7),
        gridspec_kw={"width_ratios": [1.34, 1.0, 1.0]},
        constrained_layout=True,
    )
    for row, item in enumerate(cases):
        data = item["data"]
        time = item["time"]
        s_values = item["s"]
        crest = int(np.argmin(np.abs(s_values)))
        step = max(1, len(time) // 6500)
        axes[row, 0].plot(
            time[::step], item["truth"][crest, ::step], color="#222222", lw=0.75, label="直接有限元"
        )
        axes[row, 0].plot(
            time[::step], item["prediction"][crest, ::step], color="#D55E00", lw=0.75,
            ls="--", label="代理重构"
        )
        axes[row, 0].axhline(0.0, color="#888888", lw=0.4)
        axes[row, 0].set_xlim(float(time[0]), float(time[-1]))
        axes[row, 0].set_ylabel("$a_h$/(m·s$^{-2}$)")
        axes[row, 0].grid(color="#DDDDDD", lw=0.4, ls=":")
        axes[row, 0].spines[["top", "right"]].set_visible(False)

        nrmse, correlation, pga_error, peak_time_error = item["derived"]
        shade_segments(axes[row, 1], labels=(row == 0))
        axes[row, 1].plot(s_values, nrmse, color="#0072B2", lw=1.05)
        axes[row, 1].fill_between(s_values, 0.10, nrmse, where=nrmse > 0.10, color="#D55E00", alpha=0.24)
        axes[row, 1].axhline(0.10, color="#D55E00", lw=0.85, ls="--", label="10%参考线")
        axes[row, 1].set_xlim(-4.0, 4.0)
        axes[row, 1].set_ylim(0.0, nrmse_max * 1.07)
        axes[row, 1].set_ylabel("时程NRMSE")
        axes[row, 1].grid(axis="y", color="#DDDDDD", lw=0.4, ls=":")
        axes[row, 1].spines[["top", "right"]].set_visible(False)
        axes[row, 1].text(
            0.98,
            0.88,
            "中位 %.1f%%\n超线位置 %.0f%%" % (100.0 * np.median(nrmse), 100.0 * np.mean(nrmse > 0.10)),
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=1.5),
        )

        shade_segments(axes[row, 2], labels=(row == 0))
        axes[row, 2].plot(s_values, item["taf_truth"], color="#222222", lw=1.0, label="直接有限元")
        axes[row, 2].plot(
            s_values, item["taf_prediction"], color="#D55E00", lw=1.0, ls="--", label="代理重构"
        )
        axes[row, 2].set_xlim(-4.0, 4.0)
        axes[row, 2].set_ylim(0.0, taf_max * 1.07)
        axes[row, 2].set_ylabel("TAF（统一左参考）")
        axes[row, 2].grid(axis="y", color="#DDDDDD", lw=0.4, ls=":")
        axes[row, 2].spines[["top", "right"]].set_visible(False)
        taf_error = np.abs(item["taf_prediction"] / np.maximum(item["taf_truth"], 1.0e-30) - 1.0)
        axes[row, 2].text(
            0.98,
            0.88,
            "相对误差中位 %.1f%%" % (100.0 * np.median(taf_error)),
            transform=axes[row, 2].transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.83, pad=1.5),
        )

        metadata = item["metrics"]
        parameters = metadata["parameters"]
        record_label = RECORD_LABELS.get(metadata["record"], metadata["record"])
        axes[row, 0].text(
            0.012,
            0.93,
            "%s（%s）  %s；$i$=%.1f°，$d/h$=%.2f，$r_v$=%.3f"
            % (
                item["case_id"], CASE_LEVEL[item["case_id"]], record_label,
                parameters["slope_angle_deg"], parameters["thickness_ratio"], parameters["velocity_ratio"],
            ),
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4),
        )

    axes[0, 0].set_title("(a) 坡顶 $s=0$ 带限加速度时程", loc="left")
    axes[0, 1].set_title("(b) 全地表逐位置时程误差", loc="left")
    axes[0, 2].set_title("(c) 全地表峰值放大对比", loc="left")
    axes[0, 0].legend(loc="lower right", frameon=False, ncol=2)
    axes[0, 1].legend(loc="lower right", frameon=False)
    axes[0, 2].legend(loc="lower right", frameon=False)
    axes[-1, 0].set_xlabel("时间 $t$/s")
    axes[-1, 1].set_xlabel("归一化地表坐标 $s$")
    axes[-1, 2].set_xlabel("归一化地表坐标 $s$")
    save_figure(fig)
    plt.close(fig)

    for item in cases:
        nrmse, correlation, pga_error, peak_time_error = item["derived"]
        print("数据来源：%s" % item["path"])
        print(
            "%s：NRMSE中位 %.1f%%，相关系数中位 %.3f，PGA误差中位 %.1f%%，峰时误差中位 %.3f s"
            % (
                item["case_id"], 100.0 * np.median(nrmse), np.median(correlation),
                100.0 * np.median(pga_error), np.median(peak_time_error),
            )
        )


if __name__ == "__main__":
    main()
