# -*- coding: utf-8 -*-
"""生成图21：三条真实地震波左侧一维自由场时程与能量谱。脚本可独立运行。"""

import warnings
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[4]
C_ROOT = REPO_ROOT / "Run" / "ch4_sp_05_C"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig21_真实地震波时程与能量谱"

RECORDS = (
    {
        "label": "El Centro",
        "case_dir": "case-085-C001",
        "duration": 31.2,
        "color": "#0072B2",
    },
    {
        "label": "Kobe",
        "case_dir": "case-086-C002",
        "duration": 40.9,
        "color": "#D55E00",
    },
    {
        "label": "Chi-Chi",
        "case_dir": "case-093-C009",
        "duration": 52.8,
        "color": "#009E73",
    },
)
BANDS = ((0.5, 3.0, "#0072B2"), (3.0, 6.0, "#E69F00"), (6.0, 10.0, "#D55E00"))


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


def find_reference(case_dir):
    """查找工况中唯一的一维自由场参考文件。"""
    candidates = sorted(case_dir.glob("freefield_reference_*.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError("%s 中一维参考文件数量不是1。" % case_dir)
    return candidates[0]


def moving_average(values, width):
    """用短窗平滑能量谱，仅用于提高图面可读性。"""
    width = max(1, int(width))
    if width == 1:
        return values
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, kernel, mode="same")


def spectral_energy(time, acceleration, duration):
    """计算有效持时内0.5—10 Hz归一化能量谱及三频带能量占比。"""
    use = time <= duration
    t = np.asarray(time[use], float)
    signal = np.asarray(acceleration[use], float)
    signal = signal - np.mean(signal)
    dt = float(np.median(np.diff(t)))
    window = np.hanning(len(signal))
    spectrum = np.fft.rfft(signal * window)
    frequency = np.fft.rfftfreq(len(signal), dt)
    energy = np.abs(spectrum) ** 2
    analysis = (frequency >= 0.5) & (frequency <= 10.0)
    frequency = frequency[analysis]
    energy = energy[analysis]
    total = float(np.sum(energy))
    shares = []
    for lower, upper, _ in BANDS:
        if upper == 10.0:
            selected = (frequency >= lower) & (frequency <= upper)
        else:
            selected = (frequency >= lower) & (frequency < upper)
        shares.append(float(np.sum(energy[selected]) / total))
    smooth = moving_average(energy, max(3, int(round(0.10 / np.median(np.diff(frequency))))))
    smooth = smooth / max(float(np.max(smooth)), 1.0e-30)
    return t, signal, frequency, smooth, shares


def save_figure(fig):
    """输出300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png",):
        path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % path)


def main():
    """绘制三条真实记录的统一左侧一维自由场输入及频带能量构成。"""
    set_journal_style()
    loaded = []
    for specification in RECORDS:
        reference_path = find_reference(C_ROOT / specification["case_dir"])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Reading.*created on Python 2.*")
            reference = np.load(reference_path, allow_pickle=True)
            time = np.asarray(reference["time"], float)
            acceleration = np.asarray(reference["one_d_left_acc_h"], float)
        t, signal, frequency, spectrum, shares = spectral_energy(
            time, acceleration, specification["duration"]
        )
        item = dict(specification)
        item.update(
            {
                "path": reference_path,
                "time": t,
                "signal": signal,
                "frequency": frequency,
                "spectrum": spectrum,
                "shares": shares,
            }
        )
        loaded.append(item)

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for row, item in enumerate(loaded):
        time = item["time"]
        signal = item["signal"]
        normalized = signal / max(float(np.max(np.abs(signal))), 1.0e-30)
        step = max(1, len(time) // 8000)
        axes[row, 0].plot(time[::step], normalized[::step], color=item["color"], lw=0.75)
        axes[row, 0].axhline(0.0, color="#777777", lw=0.45)
        axes[row, 0].set_xlim(0.0, item["duration"])
        axes[row, 0].set_ylim(-1.08, 1.08)
        axes[row, 0].set_ylabel("归一化加速度")
        axes[row, 0].text(
            0.015,
            0.88,
            "%s（%.1f s）" % (item["label"], item["duration"]),
            transform=axes[row, 0].transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.5),
        )
        axes[row, 0].grid(axis="x", color="#DDDDDD", lw=0.42, ls=":")
        axes[row, 0].spines[["top", "right"]].set_visible(False)

        frequency = item["frequency"]
        spectrum = item["spectrum"]
        axes[row, 1].plot(frequency, spectrum, color="#333333", lw=0.8, zorder=3)
        for lower, upper, color in BANDS:
            selected = (frequency >= lower) & (frequency <= upper)
            axes[row, 1].fill_between(
                frequency[selected], 0.0, spectrum[selected], color=color, alpha=0.43, lw=0.0
            )
        axes[row, 1].set_xlim(0.5, 10.0)
        axes[row, 1].set_ylim(0.0, 1.08)
        axes[row, 1].set_ylabel("归一化能量谱")
        shares = item["shares"]
        axes[row, 1].text(
            0.98,
            0.92,
            "0.5—3 Hz  %.1f%%\n3—6 Hz     %.1f%%\n6—10 Hz   %.1f%%"
            % tuple(100.0 * value for value in shares),
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8.1,
            linespacing=1.35,
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", lw=0.5, alpha=0.90, pad=2.2),
        )
        axes[row, 1].grid(color="#DDDDDD", lw=0.42, ls=":")
        axes[row, 1].spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_title("(a) 左侧上平台一维自由场时程", loc="left")
    axes[0, 1].set_title("(b) 0.5—10 Hz能量分布", loc="left")
    axes[-1, 0].set_xlabel("时间 $t$/s")
    axes[-1, 1].set_xlabel("频率 $f$/Hz")
    save_figure(fig)
    plt.close(fig)

    for item in loaded:
        print("数据来源：%s" % item["path"])
        print(
            "%s频带能量占比：0.5—3 Hz %.1f%%，3—6 Hz %.1f%%，6—10 Hz %.1f%%"
            % (item["label"], *(100.0 * value for value in item["shares"]))
        )


if __name__ == "__main__":
    main()
