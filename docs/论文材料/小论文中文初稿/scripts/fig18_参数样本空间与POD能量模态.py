# -*- coding: utf-8 -*-
"""生成图18：参数样本空间、POD累计能量与前三阶复数模态。脚本可独立运行。"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = REPO_ROOT / "Run" / "ch4_sp_analysis" / "complex_frf_dataset.npz"
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig18_参数样本空间与POD能量模态"
ENERGY_TARGET = 0.995
MAX_COMPONENTS = 12


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


def prepare_full_pod(field, valid_mask, minimum_valid_fraction=0.95):
    """按训练脚本的实虚拼接、逐像素标准化方式重新计算完整POD。"""
    finite = valid_mask & np.isfinite(field.real) & np.isfinite(field.imag)
    pixel_mask = np.mean(finite, axis=0) >= float(minimum_valid_fraction)
    complex_values = field[:, pixel_mask]
    values = np.concatenate([complex_values.real, complex_values.imag], axis=1)
    finite_values = np.isfinite(values)
    counts = np.sum(finite_values, axis=0)
    sums = np.nansum(values, axis=0)
    fill = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    values = np.where(finite_values, values, fill[None, :])
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < 1.0e-10] = 1.0
    standardized = (values - center) / scale
    _, singular_values, basis = np.linalg.svd(standardized, full_matrices=False)
    return singular_values, basis, scale, pixel_mask


def physical_complex_mode(vector, scale, pixel_mask):
    """把标准化空间的实虚POD向量还原为复数场扰动并归一化幅值。"""
    scaled = vector * scale
    pixel_count = int(np.sum(pixel_mask))
    complex_values = scaled[:pixel_count] + 1j * scaled[pixel_count:]
    field = np.full(pixel_mask.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    field[pixel_mask] = complex_values
    magnitude = np.abs(field)
    maximum = np.nanmax(magnitude)
    return magnitude / maximum if maximum > 0.0 else magnitude


def add_segment_marks(ax):
    """标记坡顶、坡面和坡脚的分段边界。"""
    for boundary in (0.0, 1.0):
        ax.axvline(boundary, color="white", lw=0.8, ls="--", alpha=0.95)
    for left, right, label in [(-4.0, 0.0, "上平台"), (0.0, 1.0, "坡面"), (1.0, 4.0, "下平台")]:
        ax.text(
            (left + right) / 2.0,
            0.97,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.2,
            color="#202020",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
        )


def save_figure(fig):
    """同时输出300 dpi PNG和矢量PDF。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        output_path = OUTPUT_DIR / (OUTPUT_STEM + suffix)
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print("已生成：%s" % output_path)


def main():
    """读取P/B工况并绘制样本空间、能量谱和真实POD模态。"""
    set_journal_style()
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到复频响数据集：%s" % DATA_PATH)

    with np.load(DATA_PATH, allow_pickle=False) as package:
        groups = np.asarray(package["case_groups"])
        features = np.asarray(package["X"], dtype=float)
        frequency = np.asarray(package["frequency_hz"], dtype=float)
        s_values = np.asarray(package["s"], dtype=float)
        field = np.asarray(package["G_h"], dtype=np.complex128)
        valid = np.asarray(package["valid_mask"], dtype=bool)

    p_selected = groups == "P"
    b_selected = groups == "B"
    singular_values, basis, scale, pixel_mask = prepare_full_pod(
        field[p_selected], valid[p_selected]
    )
    cumulative = np.cumsum(singular_values ** 2) / np.sum(singular_values ** 2)
    target_count = int(np.searchsorted(cumulative, ENERGY_TARGET) + 1)
    retained_energy = float(cumulative[MAX_COMPONENTS - 1])

    fig = plt.figure(figsize=(12.2, 7.25))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.31, wspace=0.34)

    ax_space = fig.add_subplot(grid[0, 0], projection="3d")
    p_features = features[p_selected]
    b_features = features[b_selected]
    ax_space.scatter(
        p_features[:, 0],
        p_features[:, 1],
        p_features[:, 2],
        s=18,
        marker="o",
        facecolor="#56B4E9",
        edgecolor="#0072B2",
        linewidth=0.45,
        alpha=0.64,
        depthshade=False,
    )
    ax_space.scatter(
        b_features[:, 0],
        b_features[:, 1],
        b_features[:, 2],
        s=39,
        marker="D",
        facecolor="#E69F00",
        edgecolor="#7A4A00",
        linewidth=0.65,
        alpha=0.95,
        depthshade=False,
    )
    ax_space.set_xlabel("坡角 $i$/(°)", labelpad=6)
    ax_space.set_ylabel("厚度比 $d/h$", labelpad=6)
    ax_space.set_zlabel("")
    ax_space.text2D(0.92, 0.29, "波速比 $r_v$", transform=ax_space.transAxes, rotation=90, ha="center", va="center", fontsize=9)
    ax_space.set_xticks([15, 30, 45, 60])
    ax_space.set_yticks([0.2, 0.6, 1.0, 1.4])
    ax_space.set_zticks([0.30, 0.45, 0.60, 0.75])
    ax_space.view_init(elev=22, azim=-55)
    ax_space.set_title("(a) 训练与训练外参数样本", loc="left", pad=8)
    ax_space.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", ms=5.0, mfc="#56B4E9", mec="#0072B2", label="P组训练/交叉验证（64例）"),
            Line2D([], [], marker="D", ls="none", ms=5.0, mfc="#E69F00", mec="#7A4A00", label="B组训练外组合（12例）"),
        ],
        loc="upper left",
        bbox_to_anchor=(-0.04, 1.02),
        frameon=False,
    )

    ax_energy = fig.add_subplot(grid[0, 1:])
    component_numbers = np.arange(1, len(cumulative) + 1)
    ax_energy.plot(component_numbers, cumulative * 100.0, color="#0072B2", lw=1.8)
    ax_energy.fill_between(component_numbers, cumulative * 100.0, color="#56B4E9", alpha=0.16)
    ax_energy.axhline(ENERGY_TARGET * 100.0, color="#009E73", lw=1.0, ls="--", label="目标能量99.5%")
    ax_energy.axvline(MAX_COMPONENTS, color="#D55E00", lw=1.0, ls="--")
    ax_energy.scatter(
        [MAX_COMPONENTS, target_count],
        [retained_energy * 100.0, cumulative[target_count - 1] * 100.0],
        s=34,
        color=["#D55E00", "#009E73"],
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
    )
    ax_energy.annotate(
        "$K=12$：%.1f%%" % (retained_energy * 100.0),
        xy=(MAX_COMPONENTS, retained_energy * 100.0),
        xytext=(MAX_COMPONENTS + 4.0, retained_energy * 100.0 - 13.0),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#D55E00"),
        color="#8B2E16",
        fontsize=8.4,
    )
    ax_energy.annotate(
        "$K=%d$：%.1f%%" % (target_count, cumulative[target_count - 1] * 100.0),
        xy=(target_count, cumulative[target_count - 1] * 100.0),
        xytext=(target_count + 4.0, 90.5),
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#009E73"),
        color="#006B4F",
        fontsize=8.4,
    )
    ax_energy.set_xlim(1, len(cumulative))
    ax_energy.set_ylim(0, 102)
    ax_energy.set_xlabel("POD模态数 $K$")
    ax_energy.set_ylabel("累计能量占比/%")
    ax_energy.set_title("(b) 完整P组输出矩阵的POD累计能量", loc="left")
    ax_energy.grid(color="#DDDDDD", lw=0.5, ls=":")
    ax_energy.spines[["top", "right"]].set_visible(False)
    ax_energy.legend(loc="lower right", frameon=False)

    mode_axes = []
    mode_image = None
    for mode_index in range(3):
        ax = fig.add_subplot(grid[1, mode_index])
        magnitude = physical_complex_mode(basis[mode_index], scale, pixel_mask)
        mode_image = ax.pcolormesh(
            s_values,
            frequency,
            magnitude,
            shading="auto",
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        add_segment_marks(ax)
        ax.set_xlim(s_values[0], s_values[-1])
        ax.set_ylim(frequency[0], frequency[-1])
        ax.set_xlabel("归一化地表坐标 $s$")
        if mode_index == 0:
            ax.set_ylabel("频率 $f$/Hz")
        else:
            ax.set_yticklabels([])
        ax.set_title("(%s) 第%d阶复数模态幅值" % (chr(ord("c") + mode_index), mode_index + 1), loc="left")
        mode_axes.append(ax)

    colorbar = fig.colorbar(mode_image, ax=mode_axes, orientation="horizontal", fraction=0.065, pad=0.17, aspect=42)
    colorbar.set_label(r"模态幅值 $|\Phi_k|/\max(|\Phi_k|)$")
    save_figure(fig)
    plt.close(fig)
    print("POD统计：K=12保留%.3f%%，达到99.5%%需K=%d。" % (retained_energy * 100.0, target_count))


if __name__ == "__main__":
    main()
