# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""小论文 C 池真实波闭环插图生成脚本（图 7—8 及图 S1）。

从 Run/ch4_sp_reconstruction/C00X 读取重构产物，生成三张论文图：
  fig7_c005_representative_panels.png (fig07_代表工况闭环四联图.png)  代表性工况 C005（P061/EQ01）四联图：
                                       坡顶时程、PGA 空间分布、TAF 空间分布、坡顶 5% 阻尼反应谱；
  fig8_ten_case_metrics_summary.png (fig08_十例闭环指标汇总.png)    十例闭环指标汇总（对照参考线）；
  figS1_crest_timehist_grid.png        补充材料：十例坡顶带限时程 2x5 网格。
输出到中文初稿 images 目录与英文初稿 images 目录。
中文初稿按中文期刊风格（宋体/衬线、深色低饱和色板、简洁边框、统一字号）出图。
运行：python make_c_closure_figures.py [重构结果根目录]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

REPO_ROOT = Path(__file__).resolve().parents[4]
RECON_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "Run" / "ch4_sp_reconstruction"
OUT_DIR_CN = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUT_DIR_EN = REPO_ROOT / "docs" / "论文材料" / "小论文英文初稿" / "images"

CASES = ["C%03d" % i for i in range(1, 11)]
REF_NRMSE = 0.10
REF_CORR = 0.95
REF_PGA = 0.10
REF_TPEAK = 0.02

# 中文期刊风格调色板（与 make_physics_model_figures.py 一致）
JOURNAL_NAVY = "#1f4e79"
JOURNAL_BLUE = "#2e75b6"
JOURNAL_RED = "#c00000"
JOURNAL_GREEN = "#548235"
JOURNAL_ORANGE = "#bf6a02"
JOURNAL_GRAY = "#595959"
JOURNAL_LIGHTGRAY = "#bfbfbf"
JOURNAL_BLACK = "#000000"


def set_journal_style():
    """中文期刊风格：宋体/衬线、深色低饱和、简洁边框、统一字号。"""
    try:
        fm.fontManager.addfont(r"C:\Windows\Fonts\simsun.ttc")
        fm.fontManager.addfont(r"C:\Windows\Fonts\STSONG.TTF")
        fm.fontManager.addfont(r"C:\Windows\Fonts\simhei.ttf")
    except Exception:
        pass
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["SimSun", "STSong", "SimHei", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.titlesize"] = 9.5
    plt.rcParams["figure.titlesize"] = 10
    plt.rcParams["figure.labelsize"] = 9
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.borderpad"] = 0.3
    plt.rcParams["legend.handlelength"] = 1.6
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.35


def set_font(chinese):
    """按出图语言设置字体，避免中文初稿复用英文画布。"""
    if chinese:
        set_journal_style()
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.spines.top"] = True
        plt.rcParams["axes.spines.right"] = True


def load_case(case):
    """读取单例重构 NPZ 与指标 JSON。"""
    case_dir = Path(RECON_ROOT) / case
    npz = np.load(str(case_dir / "reconstruction.npz"), allow_pickle=False)
    with (case_dir / "reconstruction_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return npz, metrics


def case_label(metrics):
    """生成"系统/记录"短标签，如 P061/EQ01。"""
    record = metrics["record"]
    eq = "EQ01" if "eq01" in record else ("EQ02" if "eq02" in record else "EQ03")
    p = metrics["parameters"]
    system = {"15.0": "P007", "45.0": "P039", "60.0": "P061"}.get(
        "%.1f" % p["slope_angle_deg"], "B007")
    return "%s/%s" % (system, eq)


def save_fig(fig, directory, name, label):
    """将指定语言画布输出到对应初稿目录。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(str(path), dpi=300)
    print("已生成 (%s): %s" % (label, str(path)))


def fig7_representative(chinese):
    """C005（P061/EQ01，最陡最厚最软边界工况）四联图。"""
    set_font(chinese)
    npz, metrics = load_case("C005")
    s = npz["s"]
    time = npz["time"]
    crest = int(np.argmin(np.abs(s)))  # s=0 坡顶索引
    coverage = npz["predicted_G_h_valid_mask"].mean(axis=0)
    invalid = s[coverage < 0.5]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))

    FEM_COLOR = JOURNAL_BLACK
    REC_COLOR = JOURNAL_RED
    REF_COLOR = JOURNAL_NAVY

    ax = axes[0, 0]
    ax.plot(time, npz["direct_fe_acc_h_bandlimited"][crest], lw=0.8, color=FEM_COLOR,
            label="直接有限元" if chinese else "Direct FEM")
    ax.plot(time, npz["reconstructed_acc_h"][crest], lw=0.8, color=REC_COLOR, ls="--",
            label="代理重构" if chinese else "Surrogate reconstruction")
    ax.set_xlabel("时间 (s)" if chinese else "Time (s)")
    ax.set_ylabel("坡顶加速度 (m/s$^2$)" if chinese else "Crest acceleration (m/s$^2$)")
    ax.set_title("(a) 坡顶时程（0.5–10 Hz）" if chinese else "(a) Crest time history, 0.5–10 Hz")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    if len(invalid):
        ax.axvspan(invalid.min(), invalid.max(), color=JOURNAL_LIGHTGRAY, alpha=0.5, zorder=0,
                   label="代理无效 $s$ 区域" if chinese else "Surrogate-invalid $s$ region")
    pga_rec = npz["pga_reconstructed"]
    taf_rec = npz["taf_reconstructed"]

    ax.plot(s, npz["direct_fe_pga"], color=FEM_COLOR, lw=1.3, label="直接有限元" if chinese else "Direct FEM")
    ax.plot(s, pga_rec, color=REC_COLOR, lw=1.3, ls="--", label="代理重构" if chinese else "Surrogate reconstruction")
    base_key = "pga_1d_crest" if "pga_1d_crest" in npz else "pga_1d_same_side"
    ax.plot(s, npz[base_key], color=REF_COLOR, lw=1.0, ls=":",
            label="坡顶一维基准" if chinese else "Crest 1-D reference")
    ax.set_xlabel("归一化地表坐标 $s$" if chinese else "Normalized surface coordinate $s$")
    ax.set_ylabel("PGA (m/s$^2$)")
    ax.set_title("(b) PGA 空间分布" if chinese else "(b) PGA spatial distribution")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    if len(invalid):
        ax.axvspan(invalid.min(), invalid.max(), color=JOURNAL_LIGHTGRAY, alpha=0.5, zorder=0,
                   label="代理无效 $s$ 区域" if chinese else "Surrogate-invalid $s$ region")
    ax.plot(s, npz["direct_fe_taf"], color=FEM_COLOR, lw=1.3, label="直接有限元" if chinese else "Direct FEM")
    ax.plot(s, taf_rec, color=REC_COLOR, lw=1.3, ls="--", label="代理重构" if chinese else "Surrogate reconstruction")
    ax.axhline(1.0, color=REF_COLOR, lw=1.0, ls=":", label="无放大" if chinese else "No amplification")
    ax.set_xlabel("归一化地表坐标 $s$" if chinese else "Normalized surface coordinate $s$")
    ax.set_ylabel("TAF (-)")
    ax.set_title("(c) TAF 空间分布" if chinese else "(c) TAF spatial distribution")
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    period = npz["period_s"]
    ax.semilogx(period, npz["direct_fe_psa"][crest], color=FEM_COLOR, lw=1.3,
                label="直接有限元" if chinese else "Direct FEM")
    ax.semilogx(period, npz["psa_reconstructed"][crest], color=REC_COLOR, lw=1.3, ls="--",
                label="代理重构" if chinese else "Surrogate reconstruction")
    ax.set_xlabel("周期 (s)" if chinese else "Period (s)")
    ax.set_ylabel("5% 阻尼 PSA (m/s$^2$)" if chinese else "5%-damping PSA (m/s$^2$)")
    ax.set_title("(d) 坡顶反应谱" if chinese else "(d) Crest response spectrum")
    ax.legend(loc="upper right")

    nrmse = metrics["comparison_summary"]["time_nrmse"]["median"]
    if chinese:
        fig.suptitle("C005：P061（$i$=60°，$d/h$=1.40，$r_v$=0.30）在 EQ01 El Centro 0.1g 作用下"
                     "（时程 NRMSE %.1f%%）" % (nrmse * 100), fontsize=10)
    else:
        fig.suptitle("C005: P061 (i=60°, d/h=1.40, $r_v$=0.30) under EQ01 El Centro 0.1g   "
                     "(time-history NRMSE %.1f%%)" % (nrmse * 100), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig07_代表工况闭环四联图.png" if chinese else "fig7_c005_representative_panels.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def fig8_summary(chinese):
    """十例闭环指标汇总柱状图（含参考线）。"""
    set_font(chinese)
    labels_top = []   # Cxxx
    labels_bot = []   # Pxxx/EQxx
    nrmse, corr, pga, tpeak = [], [], [], []
    for case in CASES:
        _, metrics = load_case(case)
        labels_top.append(case)
        labels_bot.append(case_label(metrics))
        c = metrics["comparison_summary"]
        nrmse.append(c["time_nrmse"]["median"])
        corr.append(c["correlation"]["median"])
        pga.append(c["pga_relative_error"]["median"])
        tpeak.append(c["peak_time_error_s"]["median"])
    x = np.arange(len(CASES))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.90, bottom=0.10, hspace=0.55, wspace=0.22)
    specs = [
        (axes[0, 0], nrmse, "时程 NRMSE (-)" if chinese else "Time-history NRMSE (-)",
         REF_NRMSE, "参考线 10%" if chinese else "Reference 10%", ".1%", (0, 0.50)),
        (axes[0, 1], corr, "相关系数 (-)" if chinese else "Correlation coefficient (-)",
         REF_CORR, "参考线 0.95" if chinese else "Reference 0.95", ".3f", (0.88, 1.00)),
        (axes[1, 0], pga, "PGA 相对误差 (-)" if chinese else "PGA relative error (-)",
         REF_PGA, "参考线 10%" if chinese else "Reference 10%", ".1%", (0, 0.13)),
        (axes[1, 1], tpeak, "峰值时刻误差 (s)" if chinese else "Peak-time error (s)",
         REF_TPEAK, "参考线 0.02 s" if chinese else "Reference 0.02 s", ".3f", (0, 0.35)),
    ]
    for ax, values, ylabel, ref, reflabel, fmt, ylim in specs:
        ax.bar(x, values, color=JOURNAL_NAVY, width=0.62,
               edgecolor=JOURNAL_BLACK, lw=0.4)
        for xi, v in zip(x, values):
            ax.text(xi, v, format(v, fmt), ha="center", va="bottom", fontsize=7,
                    color=JOURNAL_BLACK)
        ax.axhline(ref, color=JOURNAL_RED, ls="--", lw=1.0, label=reflabel)
        ax.set_xticks(x)
        ax.set_xticklabels(["%s\n%s" % (t, b) for t, b in zip(labels_top, labels_bot)],
                           fontsize=7)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=7.5)
        ax.set_ylim(*ylim)
    tags = ["(a)", "(b)", "(c)", "(d)"]
    for ax, tag in zip(axes.ravel(), tags):
        ax.set_title(tag, loc="left", fontsize=9.5)
    fig.suptitle("十例 C 池闭环重构与直接有限元对比（地表点中位数）" if chinese
                 else "Closed-loop reconstruction vs direct FEM: ten C-pool cases (medians over surface points)",
                 fontsize=10)
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig08_十例闭环指标汇总.png" if chinese else "fig8_ten_case_metrics_summary.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def figS1_timehist_grid(chinese):
    """补充材料：十例坡顶带限时程网格图。"""
    set_font(chinese)
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 5.0), sharey=False)
    FEM_COLOR = JOURNAL_BLACK
    REC_COLOR = JOURNAL_RED
    for ax, case in zip(axes.ravel(), CASES):
        npz, metrics = load_case(case)
        time = npz["time"]
        crest = int(np.argmin(np.abs(npz["s"])))
        ax.plot(time, npz["direct_fe_acc_h_bandlimited"][crest], lw=0.6, color=FEM_COLOR)
        ax.plot(time, npz["reconstructed_acc_h"][crest], lw=0.6, color=REC_COLOR, ls="--", alpha=0.85)
        ax.set_title("%s %s" % (case, case_label(metrics)), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("时间 (s)" if chinese else "Time (s)", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("坡顶加速度 (m/s$^2$)" if chinese else "Crest acceleration (m/s$^2$)",
                      fontsize=8)
    handles = [
        plt.Line2D([], [], color=FEM_COLOR, lw=1.2, label="直接有限元" if chinese else "Direct FEM"),
        plt.Line2D([], [], color=REC_COLOR, lw=1.2, ls="--", label="代理重构" if chinese else "Reconstruction"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN, "figS1_crest_timehist_grid.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


if __name__ == "__main__":
    for chinese in (True, False):
        fig7_representative(chinese)
        fig8_summary(chinese)
        figS1_timehist_grid(chinese)