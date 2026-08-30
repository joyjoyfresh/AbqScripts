# -*- coding: utf-8 -*-
"""生成图3：P061 与 V001—V004 数值设置敏感性验证。

脚本只读取已有 NPZ/CSV 产物，在内存中将全地表统一为左侧一维
自由场参考，不调用求解器，不改写评价文件。
"""

from __future__ import annotations

import csv
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
V_ROOT = REPO_ROOT / "Run" / "ch4_sp_01_V"
EVALUATION_CSV = V_ROOT / "evaluation" / "validation_comparison.csv"
OUT_DIR = SCRIPT_PATH.parent
OUT_STEM = "fig03_数值设置敏感性验证"
TARGET_FREQUENCY = np.round(np.arange(0.5, 10.0 + 0.05, 0.1), 10)
TARGET_S = np.round(np.arange(-4.0, 4.0 + 0.025, 0.05), 10)
CASES = (
    ("P061", "case-001-P061", "基准：$\\Delta=4$ m，尾段6 s，侧向1$H_s$，基底3$H_s$"),
    ("V001", "case-002-V001", "网格 1 m"),
    ("V002", "case-003-V002", "静默尾段 12 s"),
    ("V003", "case-004-V003", "侧向净空 2$H_s$"),
    ("V004", "case-005-V004", "基底深度 6$H_s$"),
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
            "legend.fontsize": 7.7,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linestyle": ":",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.38,
        }
    )


def discover_prefix(package: np.lib.npyio.NpzFile) -> str:
    """从 NPZ 键名发现复频响记录前缀。"""
    suffix = "sgrid_H_surface_h"
    matches = [key[:-len(suffix)] for key in package.files if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError("无法唯一确定复频响记录：%s" % matches)
    return matches[0]


def interpolate_complex(x_new: np.ndarray, x_old: np.ndarray,
                        values: np.ndarray) -> np.ndarray:
    """对一维复数序列的实部和虚部分别插值。"""
    valid = np.isfinite(x_old) & np.isfinite(values.real) & np.isfinite(values.imag)
    if int(np.sum(valid)) < 2:
        return np.full(x_new.shape, np.nan + 1j * np.nan, dtype=complex)
    order = np.argsort(x_old[valid])
    source_x = x_old[valid][order]
    source_values = values[valid][order]
    real = np.interp(x_new, source_x, source_values.real, left=np.nan, right=np.nan)
    imag = np.interp(x_new, source_x, source_values.imag, left=np.nan, right=np.nan)
    return real + 1j * imag


def resample_field(field: np.ndarray, frequency: np.ndarray,
                   s_values: np.ndarray) -> np.ndarray:
    """将 s×f 复数场插值到统一的 f×s 网格。"""
    frequency_stage = np.empty((field.shape[0], TARGET_FREQUENCY.size), dtype=complex)
    for s_index in range(field.shape[0]):
        frequency_stage[s_index] = interpolate_complex(
            TARGET_FREQUENCY, frequency, field[s_index]
        )
    result = np.empty((TARGET_FREQUENCY.size, TARGET_S.size), dtype=complex)
    for frequency_index in range(TARGET_FREQUENCY.size):
        result[frequency_index] = interpolate_complex(
            TARGET_S, s_values, frequency_stage[:, frequency_index]
        )
    return result


def load_uniform_left_field(case_folder: str) -> np.ndarray:
    """读取工况场，并统一除以左侧一维自由场。"""
    npz_path = V_ROOT / case_folder / "surface_results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % npz_path)
    with np.load(npz_path, allow_pickle=False) as package:
        prefix = discover_prefix(package)
        frequency = np.asarray(package[prefix + "frequency"], dtype=float)
        s_values = np.asarray(package[prefix + "sgrid_s"], dtype=float)
        explicit_left_key = prefix + "sgrid_H_surface_over_1D_left_h"
        if explicit_left_key in package.files:
            uniform_left = np.asarray(package[explicit_left_key], dtype=complex)
        else:
            total = np.asarray(package[prefix + "sgrid_H_surface_h"], dtype=complex)
            sidewise = np.asarray(package[prefix + "sgrid_H_surface_over_1D_h"], dtype=complex)
            crest_index = int(np.argmin(np.abs(s_values)))
            left_reference = total[crest_index] / sidewise[crest_index]
            valid_reference = (
                np.isfinite(left_reference.real)
                & np.isfinite(left_reference.imag)
                & (np.abs(left_reference) > 1.0e-12)
            )
            uniform_left = np.full(total.shape, np.nan + 1j * np.nan, dtype=complex)
            uniform_left[:, valid_reference] = total[:, valid_reference] / left_reference[valid_reference]
    return resample_field(uniform_left, frequency, s_values)


def load_metrics() -> dict[str, dict[str, float]]:
    """读取已独立生成的 V 系列评价指标。"""
    if not EVALUATION_CSV.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % EVALUATION_CSV)
    metrics = {}
    with EVALUATION_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            variation = row.get("variation", "")
            if variation not in {"V001", "V002", "V003", "V004"}:
                continue
            metrics[variation] = {
                "crest_frequency_error": float(row["crest_peak_frequency_error_hz"]),
                "midslope_amplitude_change": float(row["midslope_peak_amplitude_relative_change"]),
                "log_amplitude_rmse": float(row["log_amplitude_rmse"]),
            }
    missing = {"V001", "V002", "V003", "V004"} - set(metrics)
    if missing:
        raise RuntimeError("评价 CSV 缺少工况：%s" % sorted(missing))
    return metrics


def save_figure(fig: plt.Figure) -> None:
    """保存同名300 dpi PNG。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / (OUT_STEM + ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print("已生成：%s" % png_path)


def draw_figure() -> None:
    """绘制关键位置频谱、地表峰值分布和数值判据。"""
    set_journal_style()
    fields = {case_id: load_uniform_left_field(folder) for case_id, folder, _ in CASES}
    metrics = load_metrics()
    colors = {
        "P061": "#111111",
        "V001": "#0072B2",
        "V002": "#009E73",
        "V003": "#D55E00",
        "V004": "#CC79A7",
    }
    line_styles = {"P061": "-", "V001": "--", "V002": "-.", "V003": ":", "V004": (0, (5, 2))}
    short_labels = {case_id: (case_id if case_id == "P061" else "%s：%s" % (case_id, note))
                    for case_id, _folder, note in CASES}
    crest_index = int(np.argmin(np.abs(TARGET_S - 0.0)))
    midslope_index = int(np.argmin(np.abs(TARGET_S - 0.5)))

    fig, axes = plt.subplots(2, 2, figsize=(11.3, 7.1))
    ax_crest, ax_mid, ax_space, ax_gate = axes.flat
    for case_id, _folder, _note in CASES:
        width = 1.9 if case_id == "P061" else 1.25
        ax_crest.plot(
            TARGET_FREQUENCY,
            np.abs(fields[case_id][:, crest_index]),
            color=colors[case_id],
            ls=line_styles[case_id],
            lw=width,
            label=short_labels[case_id],
        )
        ax_mid.plot(
            TARGET_FREQUENCY,
            np.abs(fields[case_id][:, midslope_index]),
            color=colors[case_id],
            ls=line_styles[case_id],
            lw=width,
        )
        ax_space.plot(
            TARGET_S,
            np.nanmax(np.abs(fields[case_id]), axis=0),
            color=colors[case_id],
            ls=line_styles[case_id],
            lw=width,
        )

    ax_crest.set_xlim(0.5, 10.0)
    ax_crest.set_xlabel("频率 $f$ (Hz)")
    ax_crest.set_ylabel("$|G_h(f,s=0)|$")
    ax_crest.set_title("(a) 坡顶复频响幅值")
    ax_crest.legend(loc="upper right", ncol=2, fontsize=7.3, handlelength=2.4)
    ax_crest.grid(True, axis="y")

    ax_mid.set_xlim(0.5, 10.0)
    ax_mid.set_xlabel("频率 $f$ (Hz)")
    ax_mid.set_ylabel("$|G_h(f,s=0.5)|$")
    ax_mid.set_title("(b) 坡面中部复频响幅值")
    ax_mid.grid(True, axis="y")

    ax_space.axvspan(0.0, 1.0, color="#BDBDBD", alpha=0.14, label="坡面")
    ax_space.axvline(0.0, color="#777777", lw=0.7, ls="--")
    ax_space.axvline(1.0, color="#777777", lw=0.7, ls="--")
    ax_space.set_xlim(-4.0, 4.0)
    ax_space.set_xlabel("归一化地表坐标 $s$")
    ax_space.set_ylabel("$\\max_{0.5\\leq f\\leq10}|G_h|$")
    ax_space.set_title("(c) 全地表频带峰值的保持性")
    ax_space.grid(True, axis="y")

    variations = ("V001", "V002", "V003", "V004")
    x_position = np.arange(len(variations), dtype=float)
    bar_width = 0.24
    normalized_frequency = np.array([metrics[item]["crest_frequency_error"] / 0.2 for item in variations])
    normalized_amplitude = np.array(
        [abs(metrics[item]["midslope_amplitude_change"]) / 0.05 for item in variations]
    )
    normalized_rmse = np.array([metrics[item]["log_amplitude_rmse"] / 0.05 for item in variations])
    bar_sets = (
        (normalized_frequency, -bar_width, "#0072B2", "坡顶主峰频差 / 0.2 Hz"),
        (normalized_amplitude, 0.0, "#009E73", "坡中峰值变化 / 5%"),
        (normalized_rmse, bar_width, "#E69F00", "全场 $\\ln|G_h|$ RMSE / 0.05（辅助）"),
    )
    for values, offset, color, label in bar_sets:
        bars = ax_gate.bar(x_position + offset, values, width=bar_width * 0.9,
                           color=color, alpha=0.84, label=label)
        for bar, value in zip(bars, values):
            if value >= 0.08:
                ax_gate.text(bar.get_x() + bar.get_width() / 2.0, value + 0.08,
                             "%.2f" % value, ha="center", va="bottom", fontsize=7.2)
    ax_gate.axhline(1.0, color="#D55E00", lw=1.0, ls="--", label="参考线")
    ax_gate.set_xticks(x_position, variations)
    ax_gate.set_ylim(0.0, max(4.15, 1.12 * float(np.max(normalized_rmse))))
    ax_gate.set_ylabel("指标 / 参考线")
    ax_gate.set_title("(d) 两项主要判据均满足，V003 辅助全场指标偏大")
    ax_gate.legend(loc="upper left", fontsize=7.2, ncol=2)
    ax_gate.grid(True, axis="y")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "P061 数值设置单因素敏感性（全地表统一左侧一维自由场）",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.08, top=0.93, wspace=0.24, hspace=0.32)
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    """单独生成图3。"""
    draw_figure()


if __name__ == "__main__":
    main()
