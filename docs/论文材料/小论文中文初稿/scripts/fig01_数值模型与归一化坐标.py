# -*- coding: utf-8 -*-
"""生成图1：数值模型、边界条件与归一化地表坐标。

脚本只读取 P061 已有 JSON/NPZ 产物，不调用求解器。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


SCRIPT_PATH = Path(__file__).resolve()


def find_repo_root() -> Path:
    """从脚本位置向上查找仓库根目录。"""
    for candidate in SCRIPT_PATH.parents:
        if (candidate / "Run").is_dir() and (candidate / "docs").is_dir():
            return candidate
    raise RuntimeError("无法从脚本位置定位仓库根目录")


REPO_ROOT = find_repo_root()
CASE_DIR = REPO_ROOT / "Run" / "ch4_sp_01_V" / "case-001-P061"
OUT_DIR = SCRIPT_PATH.parent.parent / "images"
OUT_STEM = "fig01_数值模型与归一化坐标"


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
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
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


def surface_xy(x: np.ndarray, x_crest: float, x_toe: float,
               y_upper: float, y_lower: float) -> np.ndarray:
    """返回已知坡面折线的地表高程。"""
    return np.where(
        x <= x_crest,
        y_upper,
        np.where(
            x >= x_toe,
            y_lower,
            y_upper - (x - x_crest) * (y_upper - y_lower) / (x_toe - x_crest),
        ),
    )


def s_to_xy(s_value: float, x_crest: float, x_toe: float,
            y_upper: float, y_lower: float, slope_height: float) -> tuple[float, float]:
    """将分段归一化地表坐标 s 映射到模型坐标。"""
    if s_value <= 0.0:
        return x_crest + s_value * slope_height, y_upper
    if s_value <= 1.0:
        return (
            x_crest + s_value * (x_toe - x_crest),
            y_upper - s_value * slope_height,
        )
    return x_toe + (s_value - 1.0) * slope_height, y_lower


def save_figure(fig: plt.Figure) -> None:
    """保存同名 PNG 和 PDF。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / (OUT_STEM + ".png")
    pdf_path = OUT_DIR / (OUT_STEM + ".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    print("已生成：%s" % png_path)
    print("已生成：%s" % pdf_path)


def draw_figure() -> None:
    """使用 P061 真实几何、材料和地表节点绘图。"""
    set_journal_style()
    config = load_json(CASE_DIR / "case_config.json")
    validation = load_json(CASE_DIR / "geometry_validation.json")
    npz_path = CASE_DIR / "surface_results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError("缺少数据文件：%s" % npz_path)

    with np.load(npz_path, allow_pickle=False) as package:
        meta = decode_npz_json(package, "case_meta_json")
        record = "g1b_multisine_phase_a"
        monitor_x = np.asarray(package["raw_%s_x" % record], dtype=float)
        monitor_y = np.asarray(package["raw_%s_y" % record], dtype=float)

    geometry = meta["geometry"]
    geometry_cfg = config["geometry_cfg"]
    material = config["material_cfg"]
    slope_height = float(geometry_cfg["slope_height"])
    x_crest = float(geometry["x_crest"])
    x_toe = float(geometry["x_toe"])
    total_length = float(geometry["total_L"])
    y_lower = float(validation["top_surface"]["ymin"])
    y_upper = float(validation["top_surface"]["ymax"])
    y_base = float(validation["bbox"]["ymin"])
    y_interface = float(geometry["bedrock_thickness"])
    observation_x_min = x_crest - float(geometry_cfg["crest_window"]) * slope_height
    observation_x_max = x_toe + float(geometry_cfg["toe_window"]) * slope_height

    x_line = np.linspace(0.0, total_length, 1200)
    y_surface = surface_xy(x_line, x_crest, x_toe, y_upper, y_lower)
    bedrock_color = "#B8B8B8"
    cover_color = "#E69F00"
    boundary_color = "#0072B2"
    monitor_color = "#000000"

    fig, ax_domain = plt.subplots(figsize=(11.2, 4.15))

    # 全域图：几何尺寸和材料分区均由现有产物计算。
    bedrock_polygon = np.column_stack(
        [
            np.r_[0.0, total_length, total_length, 0.0],
            np.r_[y_base, y_base, y_interface, y_interface],
        ]
    )
    cover_polygon = np.column_stack(
        [
            np.r_[x_line, x_line[::-1]],
            np.r_[y_surface, np.full_like(x_line, y_interface)],
        ]
    )
    ax_domain.add_patch(Polygon(bedrock_polygon, closed=True, facecolor=bedrock_color,
                                edgecolor="none", alpha=0.72, label="基岩"))
    ax_domain.add_patch(Polygon(cover_polygon, closed=True, facecolor=cover_color,
                                edgecolor="none", alpha=0.58, label="覆盖层"))
    ax_domain.plot(x_line, y_surface, color="black", lw=1.35, label="自由地表")
    ax_domain.plot([0.0, 0.0], [y_base, y_upper], color=boundary_color, lw=2.1)
    ax_domain.plot([total_length, total_length], [y_base, y_lower], color=boundary_color, lw=2.1)
    ax_domain.plot([0.0, total_length], [y_base, y_base], color=boundary_color, lw=2.1)
    ax_domain.plot([0.0, total_length], [y_interface, y_interface], color="#8C6D31",
                   lw=0.9, ls="--")
    ax_domain.scatter(monitor_x, monitor_y, s=7.0, c=monitor_color, zorder=5,
                      label="地表监测节点")
    ax_domain.axvspan(0.0, observation_x_min, color="#56B4E9", alpha=0.08)
    ax_domain.axvspan(observation_x_max, total_length, color="#56B4E9", alpha=0.08)

    # 在全域模型图上直接标出后文统一使用的分段归一化坐标。
    for s_value in (-4.0, 0.0, 1.0, 4.0):
        x_value, y_value = s_to_xy(
            s_value, x_crest, x_toe, y_upper, y_lower, slope_height
        )
        ax_domain.plot(
            [x_value, x_value],
            [y_value, y_value + 18.0],
            color="#009E73",
            lw=0.9,
            zorder=6,
        )
        ax_domain.text(
            x_value,
            y_value + 21.0,
            "$s$=%g" % s_value,
            color="#007A58",
            ha="center",
            va="bottom",
            fontsize=8.0,
        )
    ax_domain.text(
        (observation_x_min + x_crest) / 2.0,
        y_upper + 10.0,
        "上平台  $s<0$",
        color="#007A58",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    ax_domain.text(
        (x_crest + x_toe) / 2.0 + 8.0,
        (y_upper + y_lower) / 2.0 + 8.0,
        "坡面\n$0<s<1$",
        color="#007A58",
        ha="left",
        va="center",
        fontsize=8.2,
    )
    ax_domain.text(
        (x_toe + observation_x_max) / 2.0,
        y_lower + 10.0,
        "下平台  $s>1$",
        color="#007A58",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )

    arrow_start = (0.29 * total_length, y_base + 25.0)
    arrow_end = (arrow_start[0] + 42.0, arrow_start[1] + 157.0)
    ax_domain.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        arrowprops=dict(arrowstyle="-|>", color="#D55E00", lw=1.8, mutation_scale=14),
    )
    ax_domain.text(arrow_start[0] - 12.0, arrow_start[1] + 72.0,
                   "SV 波\n$15^\\circ$ 斜入射", color="#D55E00", ha="right", va="center", fontsize=8.5)
    ax_domain.text(total_length * 0.02, y_base + 18.0,
                   "底部与两侧：黏弹性人工边界", color=boundary_color, fontsize=8.5)
    ax_domain.text(total_length * 0.76, y_interface * 0.48,
                   "基岩\n$V_s$ = %.0f m/s，$\\rho$ = %.0f kg/m$^3$，$\\nu$ = %.2f"
                   % (material["bedrock"]["vs"], material["bedrock"]["density"],
                      material["bedrock"]["poisson_ratio"]),
                   ha="center", va="center", fontsize=8.4)
    layer = material["layers"][0]
    ax_domain.text(total_length * 0.18, y_interface + 64.0,
                   "覆盖层\n$V_s$ = %.0f m/s，$\\rho$ = %.0f kg/m$^3$，$\\nu$ = %.2f"
                   % (layer["vs"], layer["density"], layer["poisson_ratio"]),
                   ha="center", va="center", fontsize=8.4)
    ax_domain.text(
        total_length * 0.985,
        y_base + 18.0,
        "%d 节点 / %d 单元\n地表监测节点 %d 个"
        % (validation["node_count"], validation["element_count"],
           validation["top_surface"]["node_count"]),
        ha="right",
        va="bottom",
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#777777", lw=0.6, alpha=0.9),
    )
    ax_domain.set_xlim(-20.0, total_length + 20.0)
    ax_domain.set_ylim(y_base - 10.0, y_upper + 35.0)
    ax_domain.set_aspect("equal", adjustable="box")
    ax_domain.set_xlabel("水平坐标 $x$ (m)")
    ax_domain.set_ylabel("高程 $y$ (m)")
    ax_domain.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "数值模型、边界条件与归一化地表坐标（P061：$i=60^\\circ$，$d/h=1.40$）",
        fontsize=11,
        y=0.99,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.91)
    save_figure(fig)
    plt.close(fig)


def main() -> None:
    """单独生成图1。"""
    draw_figure()


if __name__ == "__main__":
    main()
