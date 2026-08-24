# -*- coding: utf-8 -*-
"""生成图9：代表地表位置的复频响幅值、相位与群时延谱。"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_FILE = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig09_固定位置复频响谱"
CASE_ID = "P061"

POSITIONS = (
    (-2.0, "上平台 $s=-2.0$"),
    (0.0, "坡顶 $s=0$"),
    (0.5, "坡面中部 $s=0.5$"),
    (1.0, "坡脚 $s=1.0$"),
    (2.0, "下平台 $s=2.0$"),
)
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")
LINESTYLES = ("-", "--", "-.", ":", (0, (5, 1.5)))
GRAY = "#666666"


def set_journal_style():
    """设置中文论文绘图样式。"""
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
            "legend.fontsize": 7.7,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def find_case_dir():
    """定位P061工况目录。"""
    matches = sorted((REPO_ROOT / "Run").glob("ch4_sp_*/*-%s" % CASE_ID))
    full_pool = [path for path in matches if path.parent.name == "ch4_sp_03_P"]
    if len(full_pool) == 1:
        return full_pool[0]
    if len(matches) != 1:
        raise RuntimeError("无法唯一定位%s：%s" % (CASE_ID, matches))
    return matches[0]


def cover_fundamental_frequency():
    """由工况配置计算覆盖层一维四分之一波长频率。"""
    config_path = find_case_dir() / "case_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layer = config["material_cfg"]["layers"][0]
    return float(layer["vs"]) / (4.0 * float(layer["thickness"]))


def load_spectra():
    """读取P061五个代表位置的联合频谱。"""
    with np.load(DATA_FILE, allow_pickle=False) as data:
        case_ids = [str(value) for value in data["case_ids"]]
        case_index = case_ids.index(CASE_ID)
        frequency = data["frequency_hz"].astype(float)
        s = data["s"].astype(float)
        position_indices = [int(np.argmin(np.abs(s - value))) for value, _label in POSITIONS]
        actual = [float(s[index]) for index in position_indices]
        if max(abs(actual_value - expected[0]) for actual_value, expected in zip(actual, POSITIONS)) > 1.0e-8:
            raise RuntimeError("统一网格缺少目标位置：%s" % actual)
        valid = data["valid_mask"][case_index][:, position_indices].T.astype(bool)
        amplitude = data["amplitude"][case_index][:, position_indices].T.astype(float)
        phase = np.degrees(data["phase_unwrapped_rad"][case_index][:, position_indices].T.astype(float))
        delay = data["group_delay_s"][case_index][:, position_indices].T.astype(float)
    return frequency, valid, amplitude, phase, delay


def add_panel_label(ax, label):
    """添加分图编号。"""
    ax.text(-0.13, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


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
    """绘制固定位置的幅值、相位与群时延频谱。"""
    set_journal_style()
    frequency, valid, amplitude, phase, delay = load_spectra()
    fundamental = cover_fundamental_frequency()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35), sharex=True)
    fields = (
        (amplitude, "幅值 $|G_h|$", "幅值谱", "(a)"),
        (phase, "展开相位 $\\Phi_h$ ($^\\circ$)", "相位谱", "(b)"),
        (delay, "群时延 $\\tau_g$ (s)", "群时延谱", "(c)"),
    )
    for ax, (field, ylabel, title, panel) in zip(axes, fields):
        for row, ((_position, label), color, linestyle) in enumerate(
            zip(POSITIONS, COLORS, LINESTYLES)
        ):
            mask = valid[row] & np.isfinite(field[row])
            ax.plot(
                frequency[mask],
                field[row, mask],
                color=color,
                ls=linestyle,
                lw=1.35,
                label=label,
            )
        ax.axvline(fundamental, color=GRAY, lw=0.9, ls=(0, (2, 2)))
        ax.set_xlim(0.5, 10.0)
        ax.set_xlabel("频率 $f$ (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        add_panel_label(ax, panel)
    axes[0].text(
        fundamental + 0.12,
        0.95,
        "$f_{1D}=%.2f$ Hz" % fundamental,
        transform=axes[0].get_xaxis_transform(),
        va="top",
        fontsize=7.7,
        color=GRAY,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle(
        "P061工况不同地表位置的复频响谱（统一左侧一维参考）",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.87), w_pad=2.5)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
