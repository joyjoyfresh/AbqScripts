# -*- coding: utf-8 -*-
"""生成中文初稿补充图 S1：十例坡顶时程网格。"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[5]
RECON_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "Run" / "ch4_sp_reconstruction_closure"
OUT_DIR = Path(__file__).resolve().parent

CASES = ["C%03d" % i for i in range(1, 11)]
JOURNAL_RED = "#c00000"
JOURNAL_BLACK = "#000000"


def set_journal_style():
    """设置中文期刊风格。"""
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


def load_case(case):
    """读取指定工况的重构结果与指标。"""
    case_dir = RECON_ROOT / case
    npz = np.load(str(case_dir / "reconstruction.npz"), allow_pickle=False)
    with (case_dir / "reconstruction_metrics.json").open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return npz, metrics


def case_label(metrics):
    """生成系统与地震记录短标签。"""
    record = metrics["record"]
    eq = "EQ01" if "eq01" in record else ("EQ02" if "eq02" in record else "EQ03")
    parameters = metrics["parameters"]
    system = {"15.0": "P007", "45.0": "P039", "60.0": "P061"}.get(
        "%.1f" % parameters["slope_angle_deg"], "B007")
    return "%s/%s" % (system, eq)


def save_figure(fig):
    """保存补充图 S1。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "figS1_crest_timehist_grid.png"
    fig.savefig(str(path), dpi=300)
    print("已生成 (中文初稿): %s" % str(path))


def main():
    """只生成 figS1_crest_timehist_grid.png。"""
    set_journal_style()
    fig, axes = plt.subplots(2, 5, figsize=(14.5, 5.0), sharey=False)
    for ax, case in zip(axes.ravel(), CASES):
        npz, metrics = load_case(case)
        time = npz["time"]
        crest = int(np.argmin(np.abs(npz["s"])))
        ax.plot(time, npz["direct_fe_acc_h_bandlimited"][crest], lw=0.6, color=JOURNAL_BLACK)
        ax.plot(time, npz["reconstructed_acc_h"][crest], lw=0.6, color=JOURNAL_RED,
                ls="--", alpha=0.85)
        ax.set_title("%s %s" % (case, case_label(metrics)), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("时间 (s)", fontsize=7)
        npz.close()
    for ax in axes[:, 0]:
        ax.set_ylabel("坡顶加速度 (m/s$^2$)", fontsize=8)
    handles = [
        plt.Line2D([], [], color=JOURNAL_BLACK, lw=1.2, label="直接有限元"),
        plt.Line2D([], [], color=JOURNAL_RED, lw=1.2, ls="--", label="代理重构"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 1.0),
               frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
