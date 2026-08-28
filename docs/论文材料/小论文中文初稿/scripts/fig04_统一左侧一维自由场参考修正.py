# -*- coding: utf-8 -*-
"""生成图4：旧同侧参考与统一左侧一维自由场参考的真实数据对比。"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUTPUT_STEM = "fig04_统一左侧一维自由场参考修正"
RECORD = "g1b_multisine_phase_a"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GRAY = "#5F5F5F"
LIGHT_GRAY = "#D9D9D9"


def set_journal_style():
    """设置中文论文图形样式。"""
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
            "legend.fontsize": 7.5,
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
    """按编号定位唯一工况目录。"""
    matches = sorted((REPO_ROOT / "Run").glob("ch4_sp_*/*-%s" % case_id))
    if case_id.startswith("P"):
        full_pool = [path for path in matches if path.parent.name == "ch4_sp_03_P"]
        if len(full_pool) == 1:
            return full_pool[0]
    if len(matches) != 1:
        raise RuntimeError("无法唯一定位工况 %s：%s" % (case_id, matches))
    return matches[0]


def decode_json_scalar(value):
    """兼容Python 2产物中的字节串JSON。"""
    scalar = value.item()
    if isinstance(scalar, bytes):
        scalar = scalar.decode("utf-8")
    return json.loads(str(scalar))


def interpolate_frequency(source_frequency, field, valid, target_frequency):
    """沿频率轴对复数场的实部和虚部分别插值。"""
    output = np.full((target_frequency.size, field.shape[0]), np.nan + 1j * np.nan)
    for column in range(field.shape[0]):
        column_valid = valid[column] & np.isfinite(field[column])
        if np.count_nonzero(column_valid) < 2:
            continue
        real = np.interp(
            target_frequency,
            source_frequency[column_valid],
            field[column, column_valid].real,
        )
        imag = np.interp(
            target_frequency,
            source_frequency[column_valid],
            field[column, column_valid].imag,
        )
        output[:, column] = real + 1j * imag
    return output


def reconstruct_reference_fields(case_id, target_frequency):
    """由原始二维传递函数重构旧同侧参考与统一左侧参考复频响。"""
    case_dir = find_case_dir(case_id)
    surface_file = case_dir / "surface_results.npz"
    prefix = "frf_%s_" % RECORD
    with np.load(surface_file, allow_pickle=False) as raw:
        semantics = decode_json_scalar(raw["metric_semantics_json"])
        definition = semantics.get("FSAF_1D", "")
        if "same_side" not in definition:
            raise RuntimeError("旧口径定义未确认，拒绝生成对比图：%s" % definition)
        source_frequency = raw[prefix + "frequency"].astype(float)
        source_s = raw[prefix + "sgrid_s"].astype(float)
        old_field = raw[prefix + "sgrid_H_surface_over_1D_h"].astype(complex)
        old_valid = raw[prefix + "sgrid_H_surface_over_1D_h_valid_mask"].astype(bool)
        h_surface = raw[prefix + "sgrid_H_surface_h"].astype(complex)
        surface_valid = raw[prefix + "sgrid_H_surface_h_valid_mask"].astype(bool)

    # 旧口径在上平台及坡面使用左侧一维参考，因此可由H_surface/G_old恢复同一左侧分母。
    left_region = source_s <= 0.0
    ratio_valid = old_valid[left_region] & surface_valid[left_region] & (np.abs(old_field[left_region]) > 0.0)
    ratio = np.where(
        ratio_valid,
        h_surface[left_region] / old_field[left_region],
        np.nan + 1j * np.nan,
    )
    h_1d_left = np.nanmedian(ratio.real, axis=0) + 1j * np.nanmedian(ratio.imag, axis=0)
    new_valid = surface_valid & np.isfinite(h_1d_left)[None, :] & (np.abs(h_1d_left)[None, :] > 0.0)
    new_field = np.where(new_valid, h_surface / h_1d_left[None, :], np.nan + 1j * np.nan)

    old_interpolated = interpolate_frequency(
        source_frequency, old_field, old_valid, target_frequency
    )
    new_interpolated = interpolate_frequency(
        source_frequency, new_field, new_valid, target_frequency
    )
    return source_s, old_interpolated, new_interpolated


def toe_jump(complex_field, s):
    """计算坡脚左邻点与坡脚点之间的复数对称相对跳变量。"""
    toe = int(np.argmin(np.abs(s - 1.0)))
    left_neighbor = int(np.flatnonzero(s < 1.0)[-1])
    before = complex_field[:, left_neighbor]
    after = complex_field[:, toe]
    denominator = np.abs(before) + np.abs(after)
    return 200.0 * np.abs(after - before) / denominator


def mark_surface_regions(ax):
    """标出坡顶和坡脚，并标注三个地表分区。"""
    ax.axvline(0.0, color="white", lw=0.9, ls="--")
    ax.axvline(1.0, color="white", lw=0.9, ls="--")
    ax.text(-2.0, 9.72, "上平台", ha="center", va="top", color="white", fontsize=7.5)
    ax.text(0.5, 9.72, "坡面", ha="center", va="top", color="white", fontsize=7.5)
    ax.text(2.5, 9.72, "下平台", ha="center", va="top", color="white", fontsize=7.5)


def add_panel_label(ax, label):
    """添加分图编号。"""
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=10, fontweight="bold")


def save_figure(fig):
    """保存300 dpi PNG。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / (OUTPUT_STEM + ".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    print("已生成：%s" % png_path)


def main():
    """绘制旧同侧参考与统一左参考的场、剖面和坡脚跳变量。"""
    set_journal_style()
    case_ids = ["H004", "P061"]
    frequency = np.round(np.arange(0.5, 10.0 + 0.05, 0.1), 10)
    old_fields = {}
    new_fields = {}
    s = None
    for case_id in case_ids:
        case_s, old_field, new_field = reconstruct_reference_fields(case_id, frequency)
        if s is not None and not np.array_equal(case_s, s):
            raise RuntimeError("H004与P061的地表网格不一致")
        s = case_s
        old_fields[case_id] = old_field
        new_fields[case_id] = new_field

    old_log = np.log(np.abs(old_fields["P061"]))
    new_log = np.log(np.abs(new_fields["P061"]))
    finite = np.concatenate((old_log[np.isfinite(old_log)], new_log[np.isfinite(new_log)]))
    color_min, color_max = np.percentile(finite, [1.0, 99.0])

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4))
    meshes = []
    for ax, field, title, label in (
        (axes[0, 0], old_log, "旧同侧一维参考：坡脚处分母切换", "(a)"),
        (axes[0, 1], new_log, "统一左侧一维参考：全地表共用分母", "(b)"),
    ):
        mesh = ax.pcolormesh(
            s,
            frequency,
            field,
            shading="nearest",
            cmap="viridis",
            vmin=color_min,
            vmax=color_max,
        )
        meshes.append(mesh)
        mark_surface_regions(ax)
        ax.set_xlim(-4.0, 4.0)
        ax.set_ylim(0.5, 10.0)
        ax.set_xlabel("归一化地表坐标 $s$")
        ax.set_ylabel("频率 $f$ (Hz)")
        ax.set_title(title)
        add_panel_label(ax, label)
    colorbar = fig.colorbar(meshes[0], ax=axes[0, :], pad=0.025, fraction=0.035)
    colorbar.set_label(r"$\ln|G_h|$")
    colorbar.ax.tick_params(labelsize=8)

    ax = axes[1, 0]
    frequency_index = int(np.argmin(np.abs(frequency - 5.0)))
    ax.plot(
        s,
        np.abs(old_fields["P061"][frequency_index]),
        color=GRAY,
        lw=1.45,
        ls="--",
        label="旧同侧参考",
    )
    ax.plot(
        s,
        np.abs(new_fields["P061"][frequency_index]),
        color=BLUE,
        lw=1.55,
        label="统一左侧参考",
    )
    ax.axvspan(0.0, 1.0, color=LIGHT_GRAY, alpha=0.28)
    ax.axvline(0.0, color=GRAY, lw=0.8, ls=":")
    ax.axvline(1.0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlim(-4.0, 4.0)
    ax.set_xlabel("归一化地表坐标 $s$")
    ax.set_ylabel("幅值 $|G_h|$")
    ax.set_title("5 Hz空间剖面：统一分母后下平台整体重标定")
    ax.legend(loc="upper left")
    ax.grid(True)
    add_panel_label(ax, "(c)")

    ax = axes[1, 1]
    for case_id, color, name in (
        ("H004", ORANGE, "均质 H004"),
        ("P061", BLUE, "成层 P061"),
    ):
        old_jump = toe_jump(old_fields[case_id], s)
        new_jump = toe_jump(new_fields[case_id], s)
        ax.plot(
            frequency,
            old_jump,
            color=color,
            lw=1.15,
            ls="--",
            label=name + "·旧（中位%.1f%%）" % np.nanmedian(old_jump),
        )
        ax.plot(
            frequency,
            new_jump,
            color=color,
            lw=1.55,
            label=name + "·新（中位%.1f%%）" % np.nanmedian(new_jump),
        )
    ax.set_xlim(0.5, 10.0)
    ax.set_xlabel("频率 $f$ (Hz)")
    ax.set_ylabel("坡脚相邻点复数相对跳变量 (%)")
    ax.set_title("分母统一后，$s=1$附近非物理分界显著减弱")
    ax.legend(ncol=2, loc="upper right")
    ax.grid(True)
    add_panel_label(ax, "(d)")

    fig.suptitle(
        "一维自由场参考口径修正的真实数据核对（热图与剖面采用P061）",
        fontsize=11,
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        r"旧口径对上平台、坡面和下平台使用分段一维分母；新口径用同一 $A_{2D}$ 除以单一左侧一维分母。",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.10, top=0.91, wspace=0.26, hspace=0.32)
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
