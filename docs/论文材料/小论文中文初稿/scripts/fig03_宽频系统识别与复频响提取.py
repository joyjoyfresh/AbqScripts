# -*- coding: utf-8 -*-
"""生成图3：宽频系统识别与复频响提取。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS_FILE = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig03_宽频系统识别与复频响提取"
CASE_ID = "P061"
RECORD = "g1b_multisine_phase_a"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#666666"


def set_journal_style():
    """设置适合中文论文的统一绘图风格。"""
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
            "legend.fontsize": 7.6,
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


def find_case_dir(case_id):
    """按工况编号查找原始后处理目录。"""
    matches = sorted((REPO_ROOT / "Run").glob("ch4_sp_*/*-%s" % case_id))
    if case_id.startswith("P"):
        full_pool = [path for path in matches if path.parent.name == "ch4_sp_03_P"]
        if len(full_pool) == 1:
            return full_pool[0]
    if len(matches) != 1:
        raise RuntimeError("无法唯一定位工况 %s：%s" % (case_id, matches))
    return matches[0]


def load_data():
    """读取P061原始时程与统一左侧参考复频响数据。"""
    case_dir = find_case_dir(CASE_ID)
    surface_file = case_dir / "surface_results.npz"
    freefield_files = sorted(case_dir.glob("freefield_reference_*.npz"))
    if len(freefield_files) != 1:
        raise RuntimeError("自由场文件数量异常：%s" % freefield_files)

    prefix = "raw_%s_" % RECORD
    with np.load(surface_file, allow_pickle=False) as raw:
        time_2d = raw[prefix + "time"].astype(float)
        input_acc = raw[prefix + "input_acc"].astype(float)
        x = raw[prefix + "x"].astype(float)
        surface_acc = raw[prefix + "acc_h"].astype(float)
        crest_index = int(np.argmin(np.abs(x - 500.0)))
        crest_acc = surface_acc[crest_index]

    with np.load(freefield_files[0], allow_pickle=False) as freefield:
        time_1d = freefield["time"].astype(float)
        left_acc = freefield["one_d_left_acc_h"].astype(float)

    with np.load(ANALYSIS_FILE, allow_pickle=False) as dataset:
        case_ids = [str(value) for value in dataset["case_ids"]]
        case_index = case_ids.index(CASE_ID)
        s = dataset["s"].astype(float)
        crest_s_index = int(np.argmin(np.abs(s)))
        frequency = dataset["frequency_hz"].astype(float)
        g_h = dataset["G_h"][case_index, :, crest_s_index]
        h_2d = dataset["H_total"][case_index, :, crest_s_index]
        valid = dataset["valid_mask"][case_index, :, crest_s_index]

    h_1d_left = np.full_like(h_2d, np.nan + 1j * np.nan)
    ratio_mask = valid & np.isfinite(g_h) & (np.abs(g_h) > 0.0)
    h_1d_left[ratio_mask] = h_2d[ratio_mask] / g_h[ratio_mask]
    return (
        time_2d,
        input_acc,
        crest_acc,
        time_1d,
        left_acc,
        frequency,
        h_2d,
        h_1d_left,
        g_h,
        ratio_mask,
    )


def normalized_spectrum(signal, time):
    """计算仅用于展示激励频带的归一化单边幅值谱。"""
    dt = float(np.median(np.diff(time)))
    demeaned = signal - float(np.mean(signal))
    spectrum = np.abs(np.fft.rfft(demeaned))
    frequency = np.fft.rfftfreq(demeaned.size, d=dt)
    maximum = float(np.max(spectrum))
    if maximum > 0.0:
        spectrum = spectrum / maximum
    return frequency, spectrum


def add_panel_label(ax, label):
    """在子图左上角添加分图编号。"""
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def save_figure(fig):
    """保存300 dpi PNG与矢量PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / (OUTPUT_STEM + ".png")
    pdf_path = OUTPUT_DIR / (OUTPUT_STEM + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print("已生成：%s" % png_path)
    print("已生成：%s" % pdf_path)


def main():
    """绘制宽频输入、参考响应与复频响提取结果。"""
    set_journal_style()
    (
        time_2d,
        input_acc,
        crest_acc,
        time_1d,
        left_acc,
        frequency,
        h_2d,
        h_1d_left,
        g_h,
        valid,
    ) = load_data()
    spectrum_f, spectrum_a = normalized_spectrum(input_acc, time_2d)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.6))
    ax = axes[0, 0]
    ax.plot(time_2d, input_acc, color=GRAY, lw=0.75)
    ax.set_xlim(time_2d[0], time_2d[-1])
    ax.set_xlabel("时间 $t$ (s)")
    ax.set_ylabel("基岩输入加速度 (m/s$^2$)")
    ax.set_title("宽频多正弦输入时程")
    add_panel_label(ax, "(a)")

    ax = axes[0, 1]
    band = spectrum_f <= 12.0
    ax.plot(spectrum_f[band], spectrum_a[band], color=BLUE, lw=1.2)
    ax.axvspan(0.5, 10.0, color=ORANGE, alpha=0.13, label="识别频带 0.5–10 Hz")
    ax.set_xlim(0.0, 12.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("频率 $f$ (Hz)")
    ax.set_ylabel("归一化输入谱")
    ax.legend(loc="upper right")
    ax.set_title("输入能量覆盖识别频带")
    add_panel_label(ax, "(b)")

    ax = axes[0, 2]
    common_end = min(float(time_2d[-1]), float(time_1d[-1]))
    mask_2d = time_2d <= common_end
    mask_1d = time_1d <= common_end
    ax.plot(time_2d[mask_2d], crest_acc[mask_2d], color=BLUE, lw=0.7, label="二维坡顶")
    ax.plot(time_1d[mask_1d], left_acc[mask_1d], color=ORANGE, lw=0.7, alpha=0.85, label="左侧一维自由场")
    ax.set_xlim(0.0, common_end)
    ax.set_xlabel("时间 $t$ (s)")
    ax.set_ylabel("水平加速度 (m/s$^2$)")
    ax.legend(loc="upper right")
    ax.set_title("待识别输出与统一参考响应")
    add_panel_label(ax, "(c)")

    ax = axes[1, 0]
    ax.plot(frequency[valid], np.abs(h_2d[valid]), color=BLUE, lw=1.35, label="$|H_{2D}|$")
    ax.plot(
        frequency[valid],
        np.abs(h_1d_left[valid]),
        color=ORANGE,
        lw=1.35,
        ls="--",
        label="$|H_{1D,L}|$",
    )
    ax.set_xlim(0.5, 10.0)
    ax.set_xlabel("频率 $f$ (Hz)")
    ax.set_ylabel("相对基岩输入的传递幅值")
    ax.legend(loc="upper right")
    ax.set_title("二维与一维传递函数")
    add_panel_label(ax, "(d)")

    ax = axes[1, 1]
    ax.plot(frequency[valid], np.abs(g_h[valid]), color=GREEN, lw=1.45)
    ax.axhline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(0.5, 10.0)
    ax.set_xlabel("频率 $f$ (Hz)")
    ax.set_ylabel("幅值 $|G_h|$")
    ax.set_title("复比值幅值：$G_h=H_{2D}/H_{1D,L}$")
    add_panel_label(ax, "(e)")

    ax = axes[1, 2]
    phase = np.degrees(np.unwrap(np.angle(g_h[valid])))
    ax.plot(frequency[valid], phase, color=PURPLE, lw=1.45)
    ax.axhline(0.0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(0.5, 10.0)
    ax.set_xlabel("频率 $f$ (Hz)")
    ax.set_ylabel(r"展开相位 $\Phi_h$ ($^\circ$)")
    ax.set_title("复比值保留相位信息")
    add_panel_label(ax, "(f)")

    for axis in axes.ravel():
        axis.grid(True)
    fig.suptitle(
        r"P061工况的宽频系统识别与复频响提取（$i=60^\circ$, $d/h=1.40$, $r_v=0.30$）",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), w_pad=2.2, h_pad=2.0)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
