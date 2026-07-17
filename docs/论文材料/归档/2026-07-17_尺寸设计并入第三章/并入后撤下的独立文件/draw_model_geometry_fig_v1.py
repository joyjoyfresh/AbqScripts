# -*- coding: utf-8 -*-
"""生成第三章参数化坡地尺寸与成层方式示意图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = SCRIPT_DIR / "章节Markdown"
SOURCE_MEDIA_DIR = SCRIPT_DIR / "边坡地震响应数值模拟尺寸设计_media" / "media"
CHAPTER_FIG_DIR = CHAPTER_DIR / "附件" / "第3章"
FIGURE_STEM = "图3-1_参数化坡地几何与成层方式"

BLACK = "#202020"
GRAY = "#666666"
DIM = "#404040"
WINDOW = "#D55E00"
BOUNDARY = "#0072B2"
HORIZONTAL = "#0072B2"
TERRAIN = "#009E73"
SOILS = ("#F5ECD7", "#EBDDB9", "#DEC78C")
BEDROCK = "#D8D2C8"


def apply_style():
    """设置论文图统一样式。"""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 7.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def dim_h(ax, x1, x2, y, label, color=DIM, extension_y=None, fontsize=7.2):
    """绘制水平尺寸线。"""
    ax.annotate("", xy=(x1, y), xytext=(x2, y),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=0.8,
                                shrinkA=0, shrinkB=0))
    if extension_y is not None:
        ax.plot([x1, x1], [extension_y, y], color=color, linewidth=0.45,
                linestyle=(0, (2, 2)))
        ax.plot([x2, x2], [extension_y, y], color=color, linewidth=0.45,
                linestyle=(0, (2, 2)))
    ax.text((x1 + x2) / 2.0, y + 0.09, label, color=color,
            fontsize=fontsize, ha="center", va="bottom")


def dim_v(ax, y1, y2, x, label, extension_x=None, fontsize=7.2):
    """绘制竖向尺寸线。"""
    ax.annotate("", xy=(x, y1), xytext=(x, y2),
                arrowprops=dict(arrowstyle="<->", color=DIM, linewidth=0.8,
                                shrinkA=0, shrinkB=0))
    if extension_x is not None:
        ax.plot([extension_x, x], [y1, y1], color=DIM, linewidth=0.45,
                linestyle=(0, (2, 2)))
        ax.plot([extension_x, x], [y2, y2], color=DIM, linewidth=0.45,
                linestyle=(0, (2, 2)))
    ax.text(x + 0.10, (y1 + y2) / 2.0, label, fontsize=fontsize,
            rotation=90, ha="left", va="center")


def panel_label(ax, label):
    """添加多面板编号。"""
    ax.text(-0.015, 1.015, label, transform=ax.transAxes, fontsize=9.5,
            fontweight="bold", ha="left", va="bottom")


def slope_surface(x, x_crest, x_toe, y_crest, y_toe):
    """返回三段式坡地表高程。"""
    return np.piecewise(
        x,
        [x <= x_crest, (x > x_crest) & (x < x_toe), x >= x_toe],
        [y_crest,
         lambda value: y_crest - (value - x_crest) * (y_crest - y_toe) / (x_toe - x_crest),
         y_toe],
    )


def draw_domain_panel(ax):
    """绘制计算域尺寸、观测窗和人工边界。"""
    clear = 1.0
    obs_a = 3.1
    slope_width = 1.35
    obs_c = 2.65
    x_crest = clear + obs_a
    x_toe = x_crest + slope_width
    length = x_toe + obs_c + clear
    y_toe = 2.25
    y_crest = 3.25
    y_bedrock = 1.10
    layer_1 = 2.82
    layer_2 = 2.35

    x = np.linspace(0.0, length, 600)
    surface = slope_surface(x, x_crest, x_toe, y_crest, y_toe)
    ax.add_patch(Rectangle((0.0, 0.0), length, y_bedrock,
                           facecolor=BEDROCK, edgecolor="#9A938A", hatch="///",
                           linewidth=0.45, zorder=1))
    ax.fill_between(x, np.minimum(surface, layer_1), surface,
                    color=SOILS[0], zorder=2)
    ax.fill_between(x, np.minimum(surface, layer_2), np.minimum(surface, layer_1),
                    color=SOILS[1], zorder=2)
    ax.fill_between(x, y_bedrock, np.minimum(surface, layer_2),
                    color=SOILS[2], zorder=2)

    for level in (layer_1, layer_2, y_bedrock):
        mask = surface >= level
        ax.plot(x[mask], np.full(np.count_nonzero(mask), level), color=BLACK,
                linewidth=0.55 if level != y_bedrock else 0.75, zorder=4)
    ax.plot(x, surface, color=BLACK, linewidth=1.25, zorder=5)
    ax.plot([0.0, 0.0], [0.0, y_crest], color=BLACK, linewidth=0.75, zorder=5)
    ax.plot([length, length], [0.0, y_toe], color=BLACK, linewidth=0.75, zorder=5)
    ax.plot([0.0, length], [0.0, 0.0], color=BLACK, linewidth=0.75, zorder=5)

    ax.plot([clear, x_crest], [y_crest, y_crest], color=WINDOW,
            linewidth=3.0, solid_capstyle="butt", zorder=6)
    ax.plot([x_toe, x_toe + obs_c], [y_toe, y_toe], color=WINDOW,
            linewidth=3.0, solid_capstyle="butt", zorder=6)

    boundary_style = dict(color=BOUNDARY, linewidth=2.0,
                          linestyle=(0, (5, 3)), zorder=7)
    ax.plot([0.0, 0.0], [0.0, y_crest], **boundary_style)
    ax.plot([length, length], [0.0, y_toe], **boundary_style)
    ax.plot([0.0, length], [0.0, 0.0], **boundary_style)

    y_segment = 3.78
    dim_h(ax, 0.0, clear, y_segment, "$ch$\n净空", extension_y=y_crest)
    dim_h(ax, clear, x_crest, y_segment, "$Ah$（坡顶观测窗）",
          color=WINDOW, extension_y=y_crest)
    dim_h(ax, x_crest, x_toe, y_segment, "$h/\\tan i$", extension_y=y_toe)
    dim_h(ax, x_toe, x_toe + obs_c, y_segment, "$Ch$（坡脚观测窗）",
          color=WINDOW, extension_y=y_toe)
    dim_h(ax, x_toe + obs_c, length, y_segment, "$ch$\n净空", extension_y=y_toe)
    dim_h(ax, 0.0, length, 4.47,
          "$L=(A+C+2c)h+h/\\tan i$", extension_y=4.14, fontsize=8.0)

    x_dim = length + 0.38
    dim_v(ax, 0.0, y_toe, x_dim, "$sh$（坡脚以下）", extension_x=length)
    dim_v(ax, y_toe, y_crest, x_dim, "$h$（坡高）", extension_x=x_toe)

    ax.scatter([x_crest, x_toe], [y_crest, y_toe], s=9, color=BLACK, zorder=8)
    ax.text(x_crest - 0.10, y_crest + 0.14, "坡肩", ha="right", va="bottom")
    ax.text(x_toe + 0.10, y_toe - 0.14, "坡脚", ha="left", va="top")
    ax.text(1.60, 3.02, "土层1  $t_1$", ha="center", va="center")
    ax.text(1.60, 2.57, "土层2  $t_2$", ha="center", va="center")
    ax.text(2.05, 1.70, "土层 $N$  $t_N$", ha="center", va="center")
    ax.text(2.55, 0.55, "基岩：$\\sum t_k\\leq(s-1)h$",
            ha="center", va="center", color=GRAY)
    ax.text(length * 0.70, -0.20, "粘弹性人工边界＋分层自由场输入",
            color=BOUNDARY, ha="center", va="top")

    arrow = FancyArrowPatch((length * 0.43, -0.72), (length * 0.48, -0.05),
                            arrowstyle="-|>", mutation_scale=12,
                            color=BLACK, linewidth=1.0)
    ax.add_patch(arrow)
    ax.text(length * 0.50, -0.50, "SV波斜入射  $\\theta$", ha="left", va="center")
    ax.text(x_toe - 0.50, y_toe + 0.16, "$i$", fontsize=8.5)

    ax.set_xlim(-0.25, length + 0.95)
    ax.set_ylim(-0.85, 4.75)
    ax.axis("off")
    panel_label(ax, "a")


def draw_layer_panel(ax, terrain_following=False):
    """绘制水平成层或随地形成层概化。"""
    x_crest, x_toe = 3.0, 5.1
    x = np.linspace(0.0, 8.0, 400)
    surface = slope_surface(x, x_crest, x_toe, 2.2, 0.7)
    ax.fill_between(x, -1.5, surface, color="#EFE3CE", zorder=1)
    ax.plot(x, surface, color=BLACK, linewidth=1.15, zorder=3)
    if terrain_following:
        for offset, color in ((0.55, TERRAIN), (1.05, "#CC79A7")):
            ax.plot(x, surface - offset, color=color, linewidth=1.25,
                    linestyle=(0, (5, 3)), zorder=4)
        ax.text(6.25, 1.20, "近似等厚", color=TERRAIN, ha="center")
        title = "随地形成层"
    else:
        for level, color in ((0.30, HORIZONTAL), (-0.75, "#E69F00")):
            ax.plot([0.0, 8.0], [level, level], color=color,
                    linewidth=1.25, zorder=4)
        ax.text(0.25, 0.48, "固定高程", color=HORIZONTAL, ha="left")
        title = "水平成层"
    ax.text(7.55, -1.20, "基岩", color=GRAY, ha="right")
    ax.set_xlim(0.0, 8.0)
    ax.set_ylim(-1.5, 2.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, pad=1.0)


def default_output_paths():
    """返回技术文档与第三章共用的默认输出路径。"""
    return [
        SOURCE_MEDIA_DIR / "image1.png",
        SOURCE_MEDIA_DIR / "image1.pdf",
        SOURCE_MEDIA_DIR / "image1.svg",
        CHAPTER_FIG_DIR / (FIGURE_STEM + ".png"),
        CHAPTER_FIG_DIR / (FIGURE_STEM + ".pdf"),
    ]


def draw_figure(output_paths=None):
    """绘制并输出图件；可由第三章总绘图脚本调用。"""
    apply_style()
    fig = plt.figure(figsize=(7.3, 5.65))
    grid = fig.add_gridspec(2, 2, height_ratios=(2.45, 1.0),
                            hspace=0.05, wspace=0.16)
    ax_domain = fig.add_subplot(grid[0, :])
    ax_horizontal = fig.add_subplot(grid[1, 0])
    ax_terrain = fig.add_subplot(grid[1, 1])
    draw_domain_panel(ax_domain)
    draw_layer_panel(ax_horizontal, terrain_following=False)
    draw_layer_panel(ax_terrain, terrain_following=True)
    panel_label(ax_horizontal, "b")
    panel_label(ax_terrain, "c")

    outputs = [Path(path) for path in (output_paths or default_output_paths())]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if path.suffix.lower() == ".png":
            save_kwargs["dpi"] = 600
        fig.savefig(path, **save_kwargs)
    plt.close(fig)
    return outputs


def main():
    """命令行入口。"""
    for path in draw_figure():
        print("已输出: %s" % path)


if __name__ == "__main__":
    main()

