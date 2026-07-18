# -*- coding: utf-8 -*-
"""依据第三章正式验证数据生成论文图件。"""

from __future__ import annotations

import importlib.util
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
FIG_DIR = SCRIPT_DIR / "附件" / "第3章"
FIG_DIR.mkdir(parents=True, exist_ok=True)
GEOMETRY_FIG_SCRIPT = SCRIPT_DIR / "generate_chapter3_geometry_figure_v1.py"

F1_SCRIPT = REPO_ROOT / "Run" / "Auto_ch3" / "Autorun_ch3_F1_frequency_theory_v1.py"
F1_RUNS = {"A1": "003", "A2": "004", "A3": "005", "A4": "007", "A5": "009"}
V1_REPORT = REPO_ROOT / "Run" / "ch3_V1_geometry_material" / "run-001" / "v1_validation_report.json"
ENERGY_REPORT = REPO_ROOT / "Run" / "ch3_F0_06_energy" / "run-001" / "f0_6_validation_report.json"
V3_RUN = REPO_ROOT / "Run" / "ch3_V3_slope_solution_verification" / "run-002"
V3_REPORT = V3_RUN / "v3_validation_report.json"
V4_RUN = REPO_ROOT / "Run" / "ch3_V4_external_slope_benchmark" / "run-002"
V4_CASE = V4_RUN / "B1_shen2024_step_slope"
V4_REPORT = V4_RUN / "v4_validation_report.json"
V4_REFERENCE = V4_RUN.parent / "shen2024_figure9_published_abaqus_digitized.csv"

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
THEORY_COLOR = "#202020"
GRID_COLOR = "#D9D9D9"


def _load_f1_module():
    """加载已冻结的F1参考解与工况定义。"""
    spec = importlib.util.spec_from_file_location("chapter3_f1", str(F1_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(F1_SCRIPT.parent))
    spec.loader.exec_module(module)
    return module


F1 = _load_f1_module()
CASE_MAP = {item["id"]: item for item in F1.CASES}


def apply_style():
    """设置适合中文学位论文的统一绘图样式。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.25,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def finish_axis(ax):
    """清理坐标轴并保留必要的辅助线。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5, alpha=0.7, zorder=0)


def export_figure(fig, stem):
    """同时输出600 dpi PNG和矢量PDF。"""
    fig.savefig(FIG_DIR / (stem + ".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / (stem + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax, label):
    """添加多面板编号。"""
    ax.text(-0.18, 1.07, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left")


def draw_model_geometry():
    """调用独立尺寸图脚本，保证技术文档与论文正文使用同一图源。"""
    spec = importlib.util.spec_from_file_location("chapter3_geometry_figure", str(GEOMETRY_FIG_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.draw_figure([
        FIG_DIR / "图3-1_参数化坡地几何与成层方式.png",
        FIG_DIR / "图3-1_参数化坡地几何与成层方式.pdf",
    ])


def draw_validation_framework():
    """绘制可信性论证的证据链。"""
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    items = [
        ("模型定义", "几何、材料\n斜入射、边界", "#DCEAF7"),
        ("实现核验", "12组组合\n集合与网格", "#E3F2EA"),
        ("独立理论", "A1—A5\n传递函数", "#FFF0D6"),
        ("坡地解验证", "网格、时间步\n域尺寸、退化", "#F6E2EC"),
        ("外部基准\n未闭合", "公开坡形\n四通道复算", "#FCE4D6"),
        ("数值质量", "波长、采样\n人工能量", "#E9E2F4"),
        ("结论边界", "已验目标量可信\n物理外推受限", "#E7EEF4"),
    ]
    xs = np.linspace(0.01, 0.88, len(items))
    width, height = 0.105, 0.52
    for idx, (x, item) in enumerate(zip(xs, items)):
        title, body, color = item
        box = Rectangle((x, 0.24), width, height, facecolor=color,
                        edgecolor="#4C4C4C", linewidth=0.9)
        ax.add_patch(box)
        ax.text(x + width / 2, 0.61, title, ha="center", va="center",
                fontweight="bold", fontsize=8.6)
        ax.text(x + width / 2, 0.41, body, ha="center", va="center",
                linespacing=1.35, fontsize=7.5)
        if idx < len(items) - 1:
            arrow = FancyArrowPatch((x + width + 0.006, 0.50), (xs[idx + 1] - 0.006, 0.50),
                                    arrowstyle="-|>", mutation_scale=10,
                                    color="#5C5C5C", linewidth=1.0)
            ax.add_patch(arrow)
    ax.text(0.50, 0.09, "证据逐层增加，但每一层只回答其对应问题，不以局部通过替代整体外推",
            ha="center", va="center", color="#4A4A4A", fontsize=7.8)
    export_figure(fig, "图3-2_可信性验证证据链")


def draw_v1_audit():
    """绘制12组几何材料审计的模型规模。"""
    report = json.loads(V1_REPORT.read_text(encoding="utf-8"))
    cases = report["cases"]
    codes = ["G%02d" % (index + 1) for index in range(len(cases))]
    nodes = np.array([item["node_count"] for item in cases], dtype=float)
    elements = np.array([item["element_count"] for item in cases], dtype=float)
    tags = [("H" if item["surface_geometry"] == "horizontal" else "T") + str(item["n_layers"])
            for item in cases]
    x = np.arange(len(cases))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(x - width / 2, nodes, width, label="节点数", color=OKABE_ITO[0], zorder=3)
    ax.bar(x + width / 2, elements, width, label="单元数", color=OKABE_ITO[1],
           hatch="//", edgecolor="#7A3A15", linewidth=0.4, zorder=3)
    for xpos, top, tag in zip(x, np.maximum(nodes, elements), tags):
        ax.text(xpos, top + 70, tag, ha="center", va="bottom", fontsize=7.0, color="#3A3A3A")
    ax.set_xticks(x)
    ax.set_xticklabels(codes)
    ax.set_ylabel("模型规模（个）")
    ax.set_xlabel("几何—材料审计工况编号")
    ax.set_ylim(0, 3400)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(0.99, 0.98, "H：水平界面；T：随地形界面；数字：有限层数",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.2)
    finish_axis(ax)
    export_figure(fig, "图3-3_几何材料审计模型规模")


def load_case_transfer(case_id):
    """按F1正式口径读取有限元与独立理论传递函数。"""
    case = CASE_MAP[case_id]
    run = F1_RUNS[case_id]
    case_dir = REPO_ROOT / "Run" / "ch3_F1_frequency_theory" / ("run-" + run) / case_id
    record = "ricker_wavelet_%dHz" % int(case["frequency"])
    raw = F1._load_raw(str(case_dir / "surface_results.npz"), record)
    time = raw["time"]
    dt = float(time[1] - time[0])
    input_series = raw["input"]
    if raw.get("input_time") is not None and len(raw["input_time"]) == len(raw["input"]):
        input_series = np.interp(time, raw["input_time"], raw["input"])
    nfft = 8 * len(time)
    freq = np.fft.rfftfreq(nfft, d=dt)
    base_freq = np.fft.rfftfreq(len(time), d=dt)
    theory_h_base, theory_v_base = F1._theory_transfer(base_freq, case)
    theory_h = np.interp(freq, base_freq, np.real(theory_h_base)) + 1j * np.interp(
        freq, base_freq, np.imag(theory_h_base))
    theory_v = np.interp(freq, base_freq, np.real(theory_v_base)) + 1j * np.interp(
        freq, base_freq, np.imag(theory_v_base))
    input_spectrum = np.fft.rfft(input_series - np.mean(input_series), n=nfft)
    valid = np.abs(input_spectrum) > 0.01 * max(float(np.max(np.abs(input_spectrum))), 1.0e-30)
    indices = F1._select_indices(raw["surface_x"])
    transfers_h, transfers_v = [], []
    for index in indices:
        h_spectrum = np.fft.rfft(raw["surface_h"][index] - np.mean(raw["surface_h"][index]), n=nfft)
        v_spectrum = np.fft.rfft(raw["surface_v"][index] - np.mean(raw["surface_v"][index]), n=nfft)
        transfer_h = np.full_like(h_spectrum, np.nan + 0j)
        transfer_v = np.full_like(v_spectrum, np.nan + 0j)
        transfer_h[valid] = h_spectrum[valid] / input_spectrum[valid]
        transfer_v[valid] = v_spectrum[valid] / input_spectrum[valid]
        transfers_h.append(transfer_h)
        transfers_v.append(transfer_v)
    return {
        "case": case,
        "freq": freq,
        "theory_h": theory_h,
        "theory_v": theory_v,
        "fe_h": transfers_h,
        "fe_v": transfers_v,
    }


def draw_transfer_cases(case_ids, stem):
    """绘制多个F1算例的水平和竖向传递函数。"""
    ncols = len(case_ids)
    fig, axes = plt.subplots(2, ncols, figsize=(7.2, 4.7 if ncols == 3 else 4.3), squeeze=False,
                             gridspec_kw={"hspace": 0.30, "wspace": 0.30})
    fractions = F1.F1_PROTOCOL["surface_positions"]
    panel_index = 0
    for col, case_id in enumerate(case_ids):
        data = load_case_transfer(case_id)
        case = data["case"]
        fc = float(case["frequency"])
        mask = (data["freq"] >= 0.5 * fc) & (data["freq"] <= 1.5 * fc)
        for row, component in enumerate(("h", "v")):
            ax = axes[row, col]
            theory = data["theory_" + component]
            for color, fraction, transfer in zip(OKABE_ITO[:3], fractions, data["fe_" + component]):
                ax.plot(data["freq"][mask], np.abs(transfer[mask]), color=color,
                        label="$x/L=%.2f$" % fraction, zorder=3)
            ax.plot(data["freq"][mask], np.abs(theory[mask]), color=THEORY_COLOR,
                    linestyle="--", linewidth=1.4, label="独立理论", zorder=4)
            ax.axvline(fc, color="#777777", linestyle=":", linewidth=0.9, zorder=2)
            ax.set_xlim(0.5 * fc, 1.5 * fc)
            if col == 0:
                ax.set_ylabel("$|H_x(f)|$" if row == 0 else "$|H_y(f)|$")
            if row == 1:
                ax.set_xlabel("频率（Hz）")
            component_name = "水平" if row == 0 else "竖向"
            ax.set_title("%s  %s分量" % (case_id, component_name), pad=4)
            finish_axis(ax)
            panel_label(ax, chr(ord("a") + panel_index))
            panel_index += 1
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01),
               ncol=4, frameon=False)
    export_figure(fig, stem)


def _load_metrics():
    """读取A1—A5正式验算报告并汇总最大误差。"""
    rows = []
    for case_id, run in F1_RUNS.items():
        path = REPO_ROOT / "Run" / "ch3_F1_frequency_theory" / ("run-" + run) / "f1_validation_report.json"
        case = json.loads(path.read_text(encoding="utf-8"))["cases"][0]
        metrics = case["surface_metrics"]

        def values(component, key):
            return [item[component][key] for item in metrics
                    if item[component].get("applicable", False) and key in item[component]]

        def maximum(values_):
            return 100.0 * max(values_) if values_ else np.nan

        vector = [item["vector_peak"]["output_peak_frequency_error_relative"]
                  for item in metrics if "vector_peak" in item]
        rows.append({
            "case": case_id,
            "amplitude": maximum(values("horizontal", "band_amplitude_nrmse") +
                                 values("vertical", "band_amplitude_nrmse")),
            "gain": maximum(values("horizontal", "target_gain_error") +
                            values("vertical", "target_gain_error")),
            "peak_h": maximum(values("horizontal", "output_peak_frequency_error_relative")),
            "peak_v": maximum(values("vertical", "output_peak_frequency_error_relative")),
            "peak_vector": maximum(vector),
        })
    return rows


def draw_error_summary():
    """绘制F1幅值、增益和峰频误差汇总。"""
    rows = _load_metrics()
    cases = [item["case"] for item in rows]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    width = 0.34
    amplitude = [item["amplitude"] for item in rows]
    gain = [item["gain"] for item in rows]
    ax.bar(x - width / 2, amplitude, width, color=OKABE_ITO[0], label="频带幅值NRMSE", zorder=3)
    ax.bar(x + width / 2, gain, width, color=OKABE_ITO[1], hatch="//",
           edgecolor="#7A3A15", linewidth=0.4, label="主频增益误差", zorder=3)
    ax.axhline(5.0, color=THEORY_COLOR, linestyle="--", linewidth=1.0, label="5%容限")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("误差（%）")
    ax.set_ylim(0, 5.6)
    ax.legend(frameon=False, loc="upper left")
    finish_axis(ax)
    panel_label(ax, "a")

    ax = axes[1]
    width = 0.23
    peak_h = np.array([item["peak_h"] for item in rows], dtype=float)
    peak_v = np.array([item["peak_v"] for item in rows], dtype=float)
    peak_vector = np.array([item["peak_vector"] for item in rows], dtype=float)
    ax.bar(x - width, peak_h, width, color=OKABE_ITO[0], label="水平分量", zorder=3)
    ax.bar(x, peak_v, width, color=OKABE_ITO[3], label="竖向分量（A5未通过）", zorder=3)
    ax.bar(x + width, peak_vector, width, color=OKABE_ITO[2], hatch="..",
           edgecolor="#22684E", linewidth=0.4, label="合成谱主判据", zorder=3)
    ax.axhline(3.0, color=THEORY_COLOR, linestyle="--", linewidth=1.0, label="3%容限")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("峰频误差（%）")
    ax.set_ylim(0, 10.5)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.annotate("9.37%\n超过控制线", xy=(4, peak_v[4]), xytext=(3.25, 8.0),
                arrowprops=dict(arrowstyle="->", linewidth=0.8, color="#555555"),
                fontsize=7.0, ha="center")
    finish_axis(ax)
    panel_label(ax, "b")
    export_figure(fig, "图3-6_F1验算误差汇总")


def draw_numerical_quality():
    """绘制空间离散、时间采样和能量质量指标。"""
    cases = ["A1", "A2", "A3", "A4", "A5"]
    elements_per_wave = [41.7, 41.7, 41.7, 16.7, 16.7]
    points_per_period = [333.3, 333.3, 333.3, 333.3, 166.7]
    energy = json.loads(ENERGY_REPORT.read_text(encoding="utf-8"))["energy"]
    energy_values = [energy["artificial_energy_ratio"], energy["energy_residual"]]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.35})

    ax = axes[0]
    ax.bar(x, elements_per_wave, color=OKABE_ITO[0], zorder=3)
    ax.axhline(10.0, color=THEORY_COLOR, linestyle="--", linewidth=1.0, label="最低要求")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("单元数/最短波长")
    ax.set_ylim(0, 48)
    ax.legend(frameon=False, loc="upper right")
    finish_axis(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.bar(x, points_per_period, color=OKABE_ITO[2], zorder=3)
    ax.axhline(20.0, color=THEORY_COLOR, linestyle="--", linewidth=1.0, label="最低要求")
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("输出点数/最短周期")
    ax.set_ylim(0, 370)
    ax.legend(frameon=False, loc="upper right")
    finish_axis(ax)
    panel_label(ax, "b")

    ax = axes[2]
    energy_x = np.arange(2)
    ax.bar(energy_x, energy_values, color=[OKABE_ITO[4], OKABE_ITO[3]], zorder=3)
    ax.axhline(0.05, color=THEORY_COLOR, linestyle="--", linewidth=1.0, label="5%控制线")
    ax.set_yscale("log")
    ax.set_xticks(energy_x)
    ax.set_xticklabels(["人工能量占比", "能量残差"], rotation=18, ha="right")
    ax.set_ylabel("比值（对数坐标）")
    ax.set_ylim(1.0e-8, 2.0e-1)
    ax.legend(frameon=False, loc="upper right")
    finish_axis(ax)
    panel_label(ax, "c")
    export_figure(fig, "图3-7_数值离散与能量质量")


def load_v3_case(case_id):
    """读取V3固定501点坡地响应曲线。"""
    package = np.load(V3_RUN / case_id / "surface_results.npz", allow_pickle=False)
    try:
        for key in package.files:
            if not key.endswith("_header"):
                continue
            header = [item.decode("utf-8") if isinstance(item, bytes) else str(item)
                      for item in package[key]]
            if "s" in header and "TAF_h" in header:
                data = package[key[:-7] + "_data"]
                s_index = header.index("s")
                taf_index = header.index("TAF_h")
                s = np.array([float(row[s_index]) for row in data], dtype=float)
                taf = np.array([float(row[taf_index]) for row in data], dtype=float)
                return s, taf
    finally:
        package.close()
    raise RuntimeError("V3结果缺少固定s网格：%s" % case_id)


def draw_slope_solution_verification():
    """绘制坡地网格、时间步和计算域敏感性曲线。"""
    groups = [
        (["M1", "S0", "M3"], ["12 m", "8 m", "5.33 m"], "网格尺寸"),
        (["T1", "S0", "T3"], ["2.0 ms", "1.0 ms", "0.5 ms"], "输入与求解时间步"),
        (["D1", "S0", "D3"], ["4h", "6h", "8h"], "侧向净空"),
        (["B1", "S0", "B3"], ["3h", "5h", "7h"], "底部深度"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0),
                             gridspec_kw={"wspace": 0.28, "hspace": 0.34})
    styles = [(OKABE_ITO[1], "--"), (THEORY_COLOR, "-"), (OKABE_ITO[0], ":")]
    for panel, (ax, (case_ids, labels, title)) in enumerate(zip(axes.flat, groups)):
        for case_id, label, (color, linestyle) in zip(case_ids, labels, styles):
            s, taf = load_v3_case(case_id)
            ax.plot(s, taf, color=color, linestyle=linestyle, label=label, zorder=3)
        ax.axvline(0.0, color="#999999", linewidth=0.7, linestyle="-.")
        ax.axvline(1.0, color="#999999", linewidth=0.7, linestyle="-.")
        ax.set_xlabel("归一化地表坐标 $s$")
        if panel == 0:
            ax.set_ylabel("水平地形放大系数 TAF")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
        finish_axis(ax)
        panel_label(ax, chr(ord("a") + panel))
    export_figure(fig, "图3-8_坡地响应数值解收敛")

    report = json.loads(V3_REPORT.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    for case_id, label, color, linestyle in [
            ("H0", "均质无界面", THEORY_COLOR, "-"),
            ("L1", "同材人工分层", OKABE_ITO[2], "--")]:
        s, taf = load_v3_case(case_id)
        ax.plot(s, taf, color=color, linestyle=linestyle, label=label, zorder=3)
    ax.set_xlabel("归一化地表坐标 $s$")
    ax.set_ylabel("水平地形放大系数 TAF")
    ax.set_title("同材界面退化")
    ax.legend(frameon=False)
    finish_axis(ax)
    panel_label(ax, "a")

    names = ["网格", "时间步", "侧向域", "底部域", "同材退化"]
    keys = ["mesh_medium_to_fine", "time_medium_to_fine",
            "domain_baseline_to_far", "depth_baseline_to_deep",
            "same_material_degradation"]
    curve_error = [100.0 * report["comparisons"][key]["curve_l2_relative"] for key in keys]
    peak_error = [100.0 * report["comparisons"][key]["peak_relative"] for key in keys]
    x = np.arange(len(names))
    width = 0.34
    ax = axes[1]
    ax.bar(x - width / 2, curve_error, width, color=OKABE_ITO[0],
           label="整曲线L2相对差", zorder=3)
    ax.bar(x + width / 2, peak_error, width, color=OKABE_ITO[1], hatch="//",
           edgecolor="#7A3A15", linewidth=0.4, label="峰值相对差", zorder=3)
    ax.axhline(2.0, color=THEORY_COLOR, linestyle="--", linewidth=1.0, label="2%门槛")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("差异（%）")
    ax.set_title("正式验收指标")
    ax.legend(frameon=False, loc="upper left")
    finish_axis(ax)
    panel_label(ax, "b")
    export_figure(fig, "图3-9_坡地解验证与同材退化")


def load_v4_histories():
    """读取V4有限元时程与文献图9数字化曲线。"""
    package = np.load(V4_CASE / "surface_results.npz", allow_pickle=False)
    try:
        prefixes = sorted(
            key[:-4] for key in package.files
            if key.startswith("raw_") and key.endswith("time")
            and key[:-4] + "acc_h" in package.files
            and key[:-4] + "acc_v" in package.files
            and key[:-4] + "x" in package.files
            and key[:-4] + "representative_indices" in package.files
        )
        if len(prefixes) != 1:
            raise RuntimeError("V4原始记录数异常：%s" % prefixes)
        prefix = prefixes[0]
        time = np.asarray(package[prefix + "time"], dtype=float)
        x = np.asarray(package[prefix + "x"], dtype=float)
        acc_h = np.asarray(package[prefix + "acc_h"], dtype=float)
        acc_v = np.asarray(package[prefix + "acc_v"], dtype=float)
    finally:
        package.close()
    meta = json.loads((V4_CASE / "case_meta.json").read_text(encoding="utf-8"))
    index_a = int(np.argmin(np.abs(x - float(meta["geometry"]["x_crest"]))))
    index_b = int(np.argmin(np.abs(x - float(meta["geometry"]["x_toe"]))))
    fe = {
        "time": time,
        "A_horizontal": acc_h[index_a], "A_vertical": acc_v[index_a],
        "B_horizontal": acc_h[index_b], "B_vertical": acc_v[index_b],
    }
    with V4_REFERENCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference = {"time": np.asarray([float(row["time_s"]) for row in rows])}
    for channel in ("A_horizontal", "A_vertical", "B_horizontal", "B_vertical"):
        reference[channel] = np.asarray([float(row[channel + "_mps2"]) for row in rows])
    return fe, reference


def draw_external_slope_benchmark():
    """绘制Shen等（2024）台阶坡地外部基准的四通道对比。"""
    report = json.loads(V4_REPORT.read_text(encoding="utf-8"))
    fe, reference = load_v4_histories()
    lag = float(report["common_time_lag_s"])
    mask = (fe["time"] >= 0.4) & (fe["time"] <= 0.9)
    channels = [
        ("A_horizontal", "坡肩A—水平"), ("A_vertical", "坡肩A—竖向"),
        ("B_horizontal", "坡脚B—水平"), ("B_vertical", "坡脚B—竖向"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.26})
    for panel, (ax, (channel, title)) in enumerate(zip(axes.flat, channels)):
        time = fe["time"][mask]
        published = np.interp(time + lag, reference["time"], reference[channel])
        ax.plot(time, published, color=THEORY_COLOR, linestyle="--",
                label="Shen等（2024）公开曲线", zorder=2)
        ax.plot(time, fe[channel][mask], color=OKABE_ITO[0],
                label="本文有限元", zorder=3)
        metric = report["metrics"][channel]
        ax.text(0.02, 0.95,
                "$e_{L2}$=%.1f%%，$e_p$=%.1f%%" %
                (100.0 * metric["waveform_nrmse"],
                 100.0 * metric["absolute_peak_error"]),
                transform=ax.transAxes, va="top", ha="left", fontsize=7.2)
        ax.set_title(title)
        ax.set_xlabel("时间/s")
        ax.set_ylabel("加速度/(m·s$^{-2}$)")
        if panel == 0:
            ax.legend(frameon=False, loc="lower left")
        finish_axis(ax)
        panel_label(ax, chr(ord("a") + panel))
    export_figure(fig, "图3-10_Shen2024坡地外部基准")


def main():
    apply_style()
    draw_model_geometry()
    draw_validation_framework()
    draw_v1_audit()
    draw_transfer_cases(["A1", "A2", "A3"], "图3-4_均质平场传递函数对比")
    draw_transfer_cases(["A4", "A5"], "图3-5_成层平场传递函数对比")
    draw_error_summary()
    draw_numerical_quality()
    draw_slope_solution_verification()
    draw_external_slope_benchmark()
    print("第三章图件已生成：%s" % FIG_DIR)


if __name__ == "__main__":
    main()
