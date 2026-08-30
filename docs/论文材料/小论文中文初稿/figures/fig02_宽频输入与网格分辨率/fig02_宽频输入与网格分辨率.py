# -*- coding: utf-8 -*-
"""生成图2：G1b 宽频输入与网格/波长分辨率。

脚本只读取 P061 与 V001 的已有 JSON/NPZ 产物，不调用求解器。
"""

from __future__ import annotations

import json
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
P061_DIR = V_ROOT / "case-001-P061"
V001_DIR = V_ROOT / "case-002-V001"
OUT_DIR = SCRIPT_PATH.parent
OUT_STEM = "fig02_宽频输入与网格分辨率"
RECORD = "g1b_multisine_phase_a"


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
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linestyle": ":",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.38,
        }
    )


def load_json(path: Path) -> dict:
    """读取 UTF-8 JSON 文件。"""
    if not path.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def decode_npz_json(package: np.lib.npyio.NpzFile, key: str) -> dict:
    """解析 NPZ 中的 JSON 标量。"""
    value = package[key].item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(value)


def load_case_meta(case_dir: Path) -> dict:
    """读取工况 NPZ 中的网格及几何元数据。"""
    npz_path = case_dir / "surface_results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % npz_path)
    with np.load(npz_path, allow_pickle=False) as package:
        return decode_npz_json(package, "case_meta_json")


def save_figure(fig: plt.Figure) -> None:
    """保存同名300 dpi PNG。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / (OUT_STEM + ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print("已生成：%s" % png_path)


def draw_figure() -> None:
    """绘制实际输入时程、频谱与网格分辨率。"""
    set_journal_style()
    config = load_json(P061_DIR / "case_config.json")
    base_meta = load_case_meta(P061_DIR)
    refined_meta = load_case_meta(V001_DIR)
    npz_path = P061_DIR / "surface_results.npz"
    with np.load(npz_path, allow_pickle=False) as package:
        time = np.asarray(package["raw_%s_time" % RECORD], dtype=float)
        acceleration = np.asarray(package["raw_%s_input_acc" % RECORD], dtype=float)
        frf_frequency = np.asarray(package["frf_%s_frequency" % RECORD], dtype=float)

    dt = float(np.median(np.diff(time)))
    active_indices = np.flatnonzero(np.abs(acceleration) > 1.0e-8 * np.max(np.abs(acceleration)))
    if active_indices.size == 0:
        raise RuntimeError("G1b 输入时程无有效样点")
    active_end_index = int(active_indices[-1])
    active_end_time = float(time[active_end_index])
    fft_count = active_end_index + 1
    fft_frequency = np.fft.rfftfreq(fft_count, dt)
    amplitude_spectrum = np.abs(np.fft.rfft(acceleration[:fft_count]))
    amplitude_spectrum /= float(np.max(amplitude_spectrum))

    material = config["material_cfg"]
    vs_cover = float(material["layers"][0]["vs"])
    vs_bedrock = float(material["bedrock"]["vs"])
    mesh_base = float(base_meta["mesh_size"])
    mesh_refined = float(refined_meta["mesh_size"])
    stored_band = (float(np.min(frf_frequency)), float(np.max(frf_frequency)))
    analysis_band = (0.5, 10.0)
    frequency_line = np.linspace(analysis_band[0], stored_band[1], 500)

    blue = "#0072B2"
    orange = "#D55E00"
    green = "#009E73"
    purple = "#CC79A7"
    gray = "#666666"

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.9))
    ax_time, ax_spectrum, ax_wavelength, ax_resolution = axes.flat

    ax_time.plot(time, acceleration, color=blue, lw=0.8)
    ax_time.axvspan(active_end_time, float(time[-1]), color="#BDBDBD", alpha=0.28,
                    label="%.0f s 静默尾段" % config["time_cfg"]["tail_seconds"])
    ax_time.axvline(active_end_time, color=gray, lw=0.8, ls="--")
    ax_time.text(active_end_time - 0.12, 0.82 * np.max(acceleration),
                 "激励结束 %.2f s" % active_end_time, ha="right", va="top", fontsize=8)
    ax_time.set_xlim(float(time[0]), float(time[-1]))
    ax_time.set_xlabel("时间 $t$ (s)")
    ax_time.set_ylabel("输入加速度 (m/s$^2$)")
    ax_time.set_title("(a) G1b 宽频多正弦输入时程")
    ax_time.legend(loc="upper right")
    ax_time.grid(True, axis="y")

    show_spectrum = fft_frequency <= 15.0
    ax_spectrum.axvspan(stored_band[0], stored_band[1], color="#56B4E9", alpha=0.15,
                       label="存储复频响 %.1f–%.1f Hz" % stored_band)
    ax_spectrum.axvspan(analysis_band[0], analysis_band[1], color="#009E73", alpha=0.14,
                       label="正文分析 %.1f–%.0f Hz" % analysis_band)
    ax_spectrum.plot(fft_frequency[show_spectrum], amplitude_spectrum[show_spectrum],
                     color=orange, lw=1.1)
    ax_spectrum.axvline(analysis_band[1], color=green, lw=0.9, ls="--")
    ax_spectrum.set_xlim(0.0, 15.0)
    ax_spectrum.set_ylim(0.0, 1.08)
    ax_spectrum.set_xlabel("频率 $f$ (Hz)")
    ax_spectrum.set_ylabel("归一化傅里叶幅值")
    ax_spectrum.set_title("(b) 输入频谱与分析频带")
    ax_spectrum.legend(loc="lower right")
    ax_spectrum.grid(True, axis="y")

    wavelength_cover = vs_cover / frequency_line
    wavelength_bedrock = vs_bedrock / frequency_line
    ax_wavelength.semilogy(frequency_line, wavelength_cover, color=orange, lw=1.6,
                           label="覆盖层 $V_s=%.0f$ m/s" % vs_cover)
    ax_wavelength.semilogy(frequency_line, wavelength_bedrock, color=blue, lw=1.6,
                           label="基岩 $V_s=%.0f$ m/s" % vs_bedrock)
    ax_wavelength.axhline(10.0 * mesh_base, color=gray, lw=1.0, ls="--",
                          label="$10\\Delta$ (基准网格 %.0f m)" % mesh_base)
    ax_wavelength.axhline(10.0 * mesh_refined, color=purple, lw=1.0, ls=":",
                          label="$10\\Delta$ (加密网格 %.0f m)" % mesh_refined)
    ax_wavelength.axvspan(analysis_band[0], analysis_band[1], color="#009E73", alpha=0.07)
    ax_wavelength.set_xlim(analysis_band[0], stored_band[1])
    ax_wavelength.set_ylim(8.0, 5000.0)
    ax_wavelength.set_xlabel("频率 $f$ (Hz)")
    ax_wavelength.set_ylabel("剪切波长 $\\lambda_s$ (m)")
    ax_wavelength.set_title("(c) 材料波长与网格尺寸")
    ax_wavelength.legend(loc="upper right", ncol=2, fontsize=7.4)
    ax_wavelength.grid(True, which="both", axis="y")

    elements_base = vs_cover / (frequency_line * mesh_base)
    elements_refined = vs_cover / (frequency_line * mesh_refined)
    ax_resolution.semilogy(frequency_line, elements_base, color=blue, lw=1.7,
                           label="P061：$\\Delta=%.0f$ m" % mesh_base)
    ax_resolution.semilogy(frequency_line, elements_refined, color=orange, lw=1.7, ls="--",
                           label="V001：$\\Delta=%.0f$ m" % mesh_refined)
    ax_resolution.axhline(10.0, color="#D55E00", lw=1.0, ls=":",
                          label="10 单元/最短波长")
    for frequency_value in (10.0, stored_band[1]):
        n_base = vs_cover / (frequency_value * mesh_base)
        ax_resolution.plot(frequency_value, n_base, marker="o", ms=4.5, color=blue)
        ax_resolution.text(frequency_value - 0.13, n_base * 1.16,
                           "%.1f" % n_base, color=blue, ha="right", va="bottom", fontsize=8)
    ax_resolution.axvspan(analysis_band[0], analysis_band[1], color="#009E73", alpha=0.07)
    ax_resolution.set_xlim(analysis_band[0], stored_band[1])
    ax_resolution.set_ylim(7.0, 1300.0)
    ax_resolution.set_xlabel("频率 $f$ (Hz)")
    ax_resolution.set_ylabel("覆盖层每波长单元数 $N_\\lambda$")
    ax_resolution.set_title("(d) 最不利覆盖层的网格分辨率")
    ax_resolution.legend(loc="upper right")
    ax_resolution.grid(True, which="both", axis="y")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "G1b 宽频输入与最短波长网格检查（P061 / V001）",
        fontsize=11,
        y=0.995,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.08, top=0.93, wspace=0.25, hspace=0.34)
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    """单独生成图2。"""
    draw_figure()


if __name__ == "__main__":
    main()
