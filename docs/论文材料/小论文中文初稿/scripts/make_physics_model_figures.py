# -*- coding: utf-8 -*-
"""小论文物理与模型插图生成脚本（图 1—6）。

从 Run/ch4_sp_analysis、Run/ch4_sp_ml 读取规范分析与代理建模产物，
分别生成中文初稿（docs/论文材料/小论文中文初稿/images/fig01_*.png ~ fig06_*.png）
与英文初稿（docs/论文材料/小论文英文初稿/images/fig1_*.png ~ fig6_*.png）图件。
中英文初稿使用各自独立的文本标签与画布渲染，严禁混用。
中文初稿按中文期刊风格（宋体/衬线、深色低饱和色板、简洁边框、统一字号）出图。
运行：python make_physics_model_figures.py
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
ANALYSIS = REPO_ROOT / "Run" / "ch4_sp_analysis"
OUT_DIR_CN = REPO_ROOT / "docs" / "论文材料" / "小论文中文初稿" / "images"
OUT_DIR_EN = REPO_ROOT / "docs" / "论文材料" / "小论文英文初稿" / "images"

SLOPES = [15.0, 30.0, 45.0, 60.0]
RVS = [0.30, 0.45, 0.60, 0.75]
DHS = [0.20, 0.60, 1.00, 1.40]
B_PARAMS = {
    "B001": (22.5, 0.40, 0.375), "B002": (22.5, 0.40, 0.675),
    "B003": (22.5, 0.80, 0.525), "B004": (22.5, 1.20, 0.375),
    "B005": (37.5, 0.40, 0.525), "B006": (37.5, 0.80, 0.375),
    "B007": (37.5, 0.80, 0.675), "B008": (37.5, 1.20, 0.525),
    "B009": (52.5, 0.40, 0.675), "B010": (52.5, 0.80, 0.525),
    "B011": (52.5, 1.20, 0.375), "B012": (52.5, 1.20, 0.675),
}

# 中文期刊风格调色板（深色低饱和）
JOURNAL_NAVY = "#1f4e79"     # 主色（深蓝）
JOURNAL_BLUE = "#2e75b6"     # 次蓝
JOURNAL_LIGHTBLUE = "#9dc3e6"
JOURNAL_RED = "#c00000"      # 强调红
JOURNAL_GREEN = "#548235"    # 次绿
JOURNAL_ORANGE = "#bf6a02"   # 次橙
JOURNAL_GRAY = "#595959"     # 中性灰
JOURNAL_LIGHTGRAY = "#bfbfbf"
JOURNAL_BLACK = "#000000"

# 4 坡角用色（蓝-绿-橙-红，深低饱和）
SLOPE_COLORS = {
    15.0: JOURNAL_NAVY,
    30.0: JOURNAL_GREEN,
    45.0: JOURNAL_ORANGE,
    60.0: JOURNAL_RED,
}

# 单色蓝渐变色板（替代 viridis/twilight）
CMAP_LN_AMP = LinearSegmentedColormap.from_list(
    "journal_lnamp", ["#f7fbfd", "#c6dbef", "#6baed6", "#2171b5", "#08519c", "#08306b"]
)
CMAP_PHASE = LinearSegmentedColormap.from_list(
    "journal_phase", ["#053061", "#2166ac", "#4393c3", "#92c5de", "#f7f7f7",
                      "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"]
)
CMAP_DELAY = LinearSegmentedColormap.from_list(
    "journal_delay", ["#053061", "#4393c3", "#f7f7f7", "#f4a582", "#67001f"]
)


def set_journal_style():
    """中文期刊风格：宋体/Times 衬线、深色低饱和、简洁边框、统一字号。"""
    # 主动注册字体（绕过缓存）
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
    # 边框
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    # 刻度
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8
    plt.rcParams["xtick.major.size"] = 3.5
    plt.rcParams["ytick.major.size"] = 3.5
    # 字号
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.titlesize"] = 9.5
    plt.rcParams["figure.titlesize"] = 10
    plt.rcParams["figure.labelsize"] = 9
    # 图例
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.borderpad"] = 0.3
    plt.rcParams["legend.handlelength"] = 1.6
    # 网格
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["grid.alpha"] = 0.35


def set_font(chinese):
    """设置字体与编码。中文期刊风格由 set_journal_style 注入。"""
    if chinese:
        set_journal_style()
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["axes.spines.top"] = True
        plt.rcParams["axes.spines.right"] = True


def read_csv(path):
    with open(str(path), encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def save_fig(fig, directory, name, label):
    """将画布输出到指定目录。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(str(path), dpi=300)
    print("已生成 (%s): %s" % (label, str(path)))


def _journal_label(ax, xlabel=None, ylabel=None, title=None):
    """统一标签风格：去掉 bold，宋体 9pt。"""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


def fig1_method_schematic(chinese):
    """方法与复频响定义示意图。"""
    set_font(chinese)
    fig = plt.figure(figsize=(11.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.3], wspace=0.20,
                          left=0.04, right=0.98, top=0.92, bottom=0.06)

    # (a) 模型示意
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.set_aspect("equal")
    ax.axis("off")
    rock = Polygon([(0, 0), (10, 0), (10, 2.2), (4.3, 2.2), (5.6, 4.4), (0, 4.4)],
                   closed=True, facecolor="#dcdcdc", edgecolor=JOURNAL_BLACK, lw=0.9)
    cover = Polygon([(0, 4.4), (5.6, 4.4), (6.8, 6.4), (0, 6.4)],
                    closed=True, facecolor="#9e9e9e", edgecolor=JOURNAL_BLACK, lw=0.9)
    ax.add_patch(rock)
    ax.add_patch(cover)
    ax.text(7.6, 1.0, "基岩\n$V_{s,b}$" if chinese else "Bedrock\n$V_{s,b}$", ha="center", fontsize=9)
    ax.text(2.6, 5.55, "覆盖层 $V_{s,c}$, 厚度 $d$" if chinese else "Cover layer, $V_{s,c}$, thickness $d$",
            ha="center", fontsize=9)
    ax.text(6.45, 5.55, "$h$", fontsize=10)
    ax.annotate("", xy=(6.82, 6.42), xytext=(5.62, 4.42),
                arrowprops=dict(arrowstyle="-", color=JOURNAL_BLACK, lw=0.8))
    ax.plot([5.6, 6.6], [4.4, 4.4], color=JOURNAL_GRAY, ls=":", lw=0.8)
    ax.text(6.35, 4.55, "$i$", fontsize=10)
    arrow = FancyArrowPatch((1.1, 0.35), (2.6, 2.6), arrowstyle="-|>", mutation_scale=14,
                            color=JOURNAL_NAVY, lw=1.6)
    ax.add_patch(arrow)
    ax.text(0.4, 2.0, "SV 波\n15°斜入射" if chinese else "SV wave\n15° incidence", fontsize=8,
            color=JOURNAL_NAVY, ha="left")
    # 地表观测点（错开标注）
    for sx, lab in [(1.2, "$s<0$"), (6.2, "$s=0$"), (8.6, "$s>0$")]:
        yy = 4.4 if sx < 5.6 else (4.4 + (sx - 5.6) * (2.0 / 1.2) if sx <= 6.8 else 6.4)
        ax.plot(sx, yy, "ko", ms=3.5)
    ax.plot(6.8, 6.4, "o", color=JOURNAL_RED, ms=5.5, mec=JOURNAL_BLACK, mew=0.8)
    ax.text(7.05, 6.20, "坡顶 ($s=0$)" if chinese else "Crest ($s=0$)", fontsize=8)
    ax.text(8.7, 5.2, "地表观测阵列\n$s\\in[-4,4]$" if chinese else "Surface array\n$s\\in[-4,4]$",
            fontsize=8, ha="center")
    # 一维自由场参考柱（标注放在虚线框内下方空白处，不超出画布）
    ax.add_patch(Rectangle((0.25, 0.5), 0.55, 3.9, fill=False, edgecolor=JOURNAL_GREEN,
                           ls="--", lw=1.2))
    ax.text(0.52, 4.65, "同侧一维\n自由场参考柱" if chinese else "Same-side 1-D\nfree-field column",
            fontsize=7.5, ha="center", color=JOURNAL_GREEN)
    # 子图标题放框图下方中部
    ax.text(5.0, -0.4, "(a) 15° SV 斜入射下两层坡地模型与地表观测" if chinese
            else "(a) Two-layered slope under inclined SV incidence",
            ha="center", fontsize=9.5)

    # (b) 复频响分解与重构链
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 12)
    ax.set_ylim(9.2, -0.4)
    ax.axis("off")

    BOX_FC = "#f4f4f4"
    BOX_AMP_EC = JOURNAL_NAVY
    BOX_PHA_EC = JOURNAL_GREEN
    BOX_OUT_EC = JOURNAL_ORANGE
    BOX_PROX_EC = JOURNAL_RED
    BOX_GREY_EC = JOURNAL_GRAY

    def box(xy, w, h, text, ec=JOURNAL_BLACK, fs=8.5, sub=""):
        p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.25",
                           facecolor=BOX_FC, edgecolor=ec, lw=0.9)
        ax.add_patch(p)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        if sub:
            ax.text(cx, cy - 0.25, text, ha="center", va="center", fontsize=fs)
            ax.text(cx, cy + 0.42, sub, ha="center", va="center", fontsize=fs - 1.0, color=JOURNAL_GRAY)
        else:
            ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arrow_p(xy1, xy2, label=""):
        p = FancyArrowPatch(xy1, xy2, arrowstyle="-|>", mutation_scale=11, color=JOURNAL_GRAY, lw=1.0)
        ax.add_patch(p)
        if label:
            mx, my = (xy1[0] + xy2[0]) / 2, (xy1[1] + xy2[1]) / 2
            # 水平箭头标签放下方且错开箭头，竖直箭头标签放左/右侧
            if abs(xy2[0] - xy1[0]) > abs(xy2[1] - xy1[1]):
                ax.text(mx, my - 0.30, label, ha="center", fontsize=7.5, color=JOURNAL_GRAY)
            else:
                ax.text(mx + 0.45, my, label, ha="left", va="center", fontsize=7.5, color=JOURNAL_GRAY)

    # 顶行：输入 → FEM → 复频响
    box((0.2, 0.4), 3.2, 1.8,
        "宽频识别信号\n$G_{1b}$（0.5–12 Hz 多正弦）" if chinese else "Broadband excitation\n$G_{1b}$ (0.5–12 Hz multi-sine)",
        ec=BOX_AMP_EC, fs=8.5)
    box((4.4, 0.4), 3.2, 1.8,
        "Abaqus/Standard\n二维成层坡地有限元" if chinese else "Abaqus/Standard\n2-D layered FE slope",
        ec=BOX_AMP_EC, fs=8.5)
    arrow_p((3.4, 1.3), (4.4, 1.3))
    box((8.6, 0.4), 3.2, 1.8,
        "水平复频响 $G_h(f,s)$\n$=\\hat{A}_{2D}/\\hat{A}_{1D}$" if chinese
        else "Complex FRF $G_h(f,s)$\n$=\\hat{A}_{2D}/\\hat{A}_{1D}$",
        ec=BOX_OUT_EC, fs=8.5)
    arrow_p((7.6, 0.6), (8.6, 0.6), "地表/一维" if chinese else "Surface / 1D")

    # 中行：双通道
    box((4.4, 3.5), 3.2, 1.6,
        "幅值通道 $|G_h|$\n$\\ln|G_h|\\to$ 地形放大" if chinese else "Amplitude $|G_h|$\n$\\ln|G_h|\\to$ topographic amp.",
        ec=BOX_PHA_EC, fs=8)
    box((8.6, 3.5), 3.2, 1.6,
        "相位通道 $\\Phi_h$\n解缠相位与群时延 $\\tau_g$" if chinese else "Phase $\\Phi_h$\nUnwrapped phase & $\\tau_g$",
        ec=BOX_PHA_EC, fs=8)
    arrow_p((9.4, 2.2), (6.0, 3.5))
    arrow_p((10.2, 2.2), (10.2, 3.5))

    # 下行：代理
    box((4.4, 6.5), 3.2, 1.4,
        "POD-GPR 复数场代理\n$(i, d/h, r_v)\\to\\hat{G}_h(f,s)$" if chinese
        else "POD-GPR surrogate\n$(i, d/h, r_v)\\to\\hat{G}_h(f,s)$",
        ec=BOX_PROX_EC, fs=8)
    arrow_p((6.0, 5.1), (6.0, 6.5))
    arrow_p((10.2, 5.1), (6.8, 6.5))

    # 底行：重构
    box((8.6, 8.1), 3.2, 1.6,
        "真实地震动时程重构\n$\\hat{a}_{2D}=\\mathcal{F}^{-1}[\\hat{G}_h\\cdot\\mathcal{F}(a_{1D})]$" if chinese
        else "Real-record reconstruction\n$\\hat{a}_{2D}=\\mathcal{F}^{-1}[\\hat{G}_h\\cdot\\mathcal{F}(a_{1D})]$",
        ec=BOX_PROX_EC, fs=8)
    box((4.4, 8.3), 3.2, 1.0,
        "场地基岩地震动 $a_0(t)$" if chinese else "Site bedrock ground\nmotion $a_0(t)$",
        ec=BOX_GREY_EC, fs=8)
    arrow_p((6.0, 7.9), (6.0, 8.3))
    arrow_p((6.0, 6.5), (6.0, 7.9))
    arrow_p((7.6, 8.8), (8.4, 8.8))
    ax.text(6.0, -0.3, "(b) 复频响定义、幅相双通道、代理模型与重构链" if chinese
            else "(b) Definition, decomposition, surrogate, and reconstruction",
            ha="center", fontsize=9.5)
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig01_方法与复频响定义示意.png" if chinese else "fig1_method_schematic.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def fig2_case_matrix_pipeline(chinese):
    """工况矩阵与验证流程图。"""
    set_font(chinese)
    d = np.load(os.path.join(ANALYSIS, "complex_frf_dataset.npz"), allow_pickle=False)
    X = d["X"]
    ids = [str(s) for s in d["case_ids"]]
    fig = plt.figure(figsize=(9.5, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.32,
                          left=0.07, right=0.97, top=0.95, bottom=0.04)
    ax1 = fig.add_subplot(gs[0])
    # (a) 参数矩阵
    dodge = {15.0: -0.018, 30.0: -0.006, 45.0: 0.006, 60.0: 0.018}

    for i_case, cid in enumerate(ids):
        if cid.startswith("P"):
            sl, dh, rv = X[i_case, 0], X[i_case, 1], X[i_case, 2]
            ax1.scatter(dh + dodge[sl], rv, color=SLOPE_COLORS[sl],
                        s=46, edgecolors=JOURNAL_BLACK, lw=0.5, zorder=3)
    for sl in SLOPES:
        ax1.scatter(0.0 + dodge[sl], 1.0, color=SLOPE_COLORS[sl], marker="s",
                    s=58, edgecolors=JOURNAL_BLACK, lw=0.7, zorder=3)
    for bname, (sl, dh, rv) in B_PARAMS.items():
        ax1.scatter(dh, rv, marker="^", s=64, facecolor="none",
                    edgecolors=JOURNAL_RED, lw=1.2, zorder=4)
        ax1.text(dh + 0.022, rv + 0.014, bname, fontsize=7, color=JOURNAL_RED)

    handles = [
        plt.Line2D([], [], marker="o", color="w", mfc=SLOPE_COLORS[sl], ms=8,
                   mec=JOURNAL_BLACK, label="$i=%.0f^\\circ$" % sl)
        for sl in SLOPES
    ]
    handles.append(plt.Line2D([], [], marker="s", color="w", mfc=JOURNAL_GRAY, ms=8,
                              mec=JOURNAL_BLACK,
                              label="均质基准 H ($d/h=0$)" if chinese else "Homogeneous baseline H ($d/h=0$)"))
    handles.append(plt.Line2D([], [], marker="^", color="w", mfc="none", ms=9,
                              mec=JOURNAL_RED, mew=1.2,
                              label="盲测组合 B（12 例）" if chinese else "Blind combination B (12 cases)"))
    ax1.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95,
               edgecolor=JOURNAL_LIGHTGRAY)
    ax1.set_xlabel("覆盖层厚度比 $d/h$" if chinese else "Cover-layer thickness ratio $d/h$")
    ax1.set_ylabel("波速比 $r_v = V_{s,c}/V_{s,b}$" if chinese else "Velocity ratio $r_v = V_{s,c}/V_{s,b}$")
    ax1.set_xlim(-0.08, 1.52)
    ax1.set_ylim(0.24, 1.08)
    ax1.set_title("(a) 参数矩阵：64 个开发工况 P ($4\\times 4\\times 4$) + 4 个均质基准 H + 12 个盲测组合 B" if chinese
                  else "(a) Parameter matrix: 64 development cases P + 4 H + 12 blind B", fontsize=9.5)

    # (b) 验证与建模流程图
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, 6.5)
    ax2.axis("off")

    def pbox(xy, w, h, title, sub="", ec=JOURNAL_BLACK):
        p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.2",
                           facecolor="#fafafa", edgecolor=ec, lw=0.9)
        ax2.add_patch(p)
        cx, cy = xy[0] + w / 2, xy[1] + h / 2
        if sub:
            ax2.text(cx, cy + 0.22, title, ha="center", va="center", fontsize=8.5)
            ax2.text(cx, cy - 0.28, sub, ha="center", va="center", fontsize=7, color=JOURNAL_GRAY)
        else:
            ax2.text(cx, cy, title, ha="center", va="center", fontsize=8.5)

    def link(p1, p2, label="", curve=False):
        if curve:
            cs = "arc3,rad=0.25"
        else:
            cs = "arc3,rad=0"
        p = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10, color=JOURNAL_GRAY, lw=1.0,
                            connectionstyle=cs)
        ax2.add_patch(p)
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            if curve:
                ax2.text(mx - 0.4, my - 0.3, label, ha="center", fontsize=7.5, color=JOURNAL_GRAY)
            else:
                # 水平箭头：标签放箭头下方且不与框标题重叠
                ax2.text(mx, my - 0.40, label, ha="center", fontsize=7.5, color=JOURNAL_GRAY)

    pbox((0.2, 4.0), 2.2, 1.7,
         "第 1 阶段：配置自检" if chinese else "Stage 1: Config checks",
         "P061 基准 + V001–V004\n(网格 / 尾段 / 域宽)" if chinese else "P061 baseline + V001–V004\n(mesh / tail / domain)")
    pbox((2.8, 4.0), 2.2, 1.7,
         "第 2 阶段：跨软件检验" if chinese else "Stage 2: Cross-software",
         "X001–X002\nAbaqus vs SPECFEM2D\n(4 Hz Ricker 闸门)" if chinese else "X001–X002\nAbaqus vs SPECFEM2D\n(4 Hz Ricker gates)")
    link((2.4, 4.30), (2.8, 4.30), "通过" if chinese else "passed")

    pbox((5.4, 4.0), 2.2, 1.7,
         "第 3 阶段：物理规律" if chinese else "Stage 3: Physics laws",
         "64 形成层工况 (P)\n幅相双通道 + SP-H1/H2\n(非单调 / $\\chi$ 标度)" if chinese else "64 layered cases (P)\nDual-channel + SP-H1/H2\n(non-monot. / $\\chi$ scaling)")
    link((5.0, 4.30), (5.4, 4.30), "建立可信度" if chinese else "confidence")

    pbox((8.0, 4.0), 2.7, 1.7,
         "第 4 阶段：代理建模" if chinese else "Stage 4: Surrogate",
         "POD-GPR vs Ridge/NN\n五折交叉验证 (SP-H3)\n冻结超参数" if chinese else "POD-GPR vs Ridge/NN\n5-fold CV (SP-H3)\nlocked hyperparameters")
    link((7.6, 4.30), (8.0, 4.30), "数据集" if chinese else "dataset")

    pbox((3.0, 0.6), 2.7, 1.7,
         "第 5 阶段：一次性盲测" if chinese else "Stage 5: Blind test",
         "12 组未见参数组合 (B)\n一次性盲测评估\n严禁调参" if chinese else "12 unseen combos (B)\none-shot evaluation\nno retuning")
    pbox((6.4, 0.6), 3.0, 1.7,
         "第 6 阶段：真实波闭环" if chinese else "Stage 6: Real-wave closed loop",
         "10 例 C 池工况 (EQ01–03)\n时程 / PGA / TAF / PSA\n对比有限元 (SP-H4)" if chinese else "10 C cases (EQ01–03)\ntime hist. / PGA / TAF / PSA\nvs direct FE (SP-H4)")

    # 第 4 阶段 → 第 5 阶段的弧线（向下）
    link((8.85, 4.30), (4.5, 2.3), "冻结模型\n无微调", curve=True)
    link((5.75, 1.45), (6.35, 1.45), "冻结模型" if chinese else "locked model")
    ax2.set_title("(b) 数值验证、代理建模与真实波闭环流程" if chinese
                  else "(b) Validation, surrogate modelling, and closed-loop pipeline", fontsize=9.5)
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig02_工况矩阵与验证流程.png" if chinese else "fig2_case_matrix_pipeline.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def fig3_fields_homo_vs_layered(chinese):
    """H004 均质 vs P061 层状：幅值—相位—群时延场。"""
    set_font(chinese)
    d = np.load(os.path.join(ANALYSIS, "complex_frf_dataset.npz"), allow_pickle=False)
    ids = [str(s) for s in d["case_ids"]]
    freq = d["frequency_hz"]
    s = d["s"]
    F, S = np.meshgrid(freq, s, indexing="ij")
    rows = [
        ("H004", "均质基准 H004 ($i=60^\\circ$)" if chinese else "Homogeneous baseline H004 ($i=60^\\circ$)"),
        ("P061", "成层工况 P061 ($i=60^\\circ, d/h=1.40, r_v=0.30$)" if chinese else "Layered configuration P061 ($i=60^\\circ, d/h=1.40, r_v=0.30$)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.0))
    panels = [
        (np.log(d["amplitude"]), "$\\ln|G_h|$", CMAP_LN_AMP, None),
        (np.degrees(d["phase_unwrapped_rad"]), "$\\Phi_h$ ($^\\circ$)" if chinese else "$\\Phi_h$ (deg)",
         CMAP_PHASE, (-200, 200)),
        (d["group_delay_s"], "$\\tau_g$ (s)", CMAP_DELAY, None),
    ]
    valid_mask = d["valid_mask"]
    for r, (cid, rowlab) in enumerate(rows):
        k = ids.index(cid)
        for c, (z_full, label, cmap, clim) in enumerate(panels):
            ax = axes[r, c]
            z = np.where(valid_mask[k], z_full[k], np.nan)
            pcm = ax.pcolormesh(S, F, z, cmap=cmap, shading="nearest")
            if clim:
                pcm.set_clim(*clim)
            ax.set_xlabel("归一化地表坐标 $s$" if chinese else "Normalized coordinate $s$")
            if c == 0:
                ax.set_ylabel("频率 $f$ (Hz)" if chinese else "$f$ (Hz)")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            cbar = fig.colorbar(pcm, ax=ax)
            cbar.set_label(label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            if r == 0:
                ax.set_title("(%s) %s" % ("abc"[c], label), fontsize=9)
        # 行标签放在左侧外
        axes[r, 0].text(-6.2, 11.6, rowlab, fontsize=9)
    fig.suptitle("均质与成层坡幅值—相位—群时延联合场（无效区已掩码）" if chinese
                 else "Amplitude-phase-group-delay joint fields: homogeneous vs layered slope (masked where invalid)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig03_均质与成层坡幅相群时延联合场.png" if chinese else "fig3_fields_homo_vs_layered.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def fig4_sp_h1(chinese):
    """SP-H1：峰值幅值随 d/h 的非单调切片 + rv=0.30 主峰频率迁移。"""
    set_font(chinese)
    rows = read_csv(os.path.join(ANALYSIS, "case_metrics.csv"))
    grid_amp, grid_hz = {}, {}
    for r in rows:
        if r["group"] != "P":
            continue
        key = (float(r["slope_angle_deg"]), float(r["thickness_ratio"]), float(r["velocity_ratio"]))
        grid_amp[key] = float(r["peak_amplitude"])
        grid_hz[key] = float(r["peak_frequency_hz"])
    fig = plt.figure(figsize=(12.0, 4.6))
    gs = fig.add_gridspec(1, 2, wspace=0.22, left=0.07, right=0.97, top=0.90, bottom=0.13)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # (a) 16 条曲线：4 坡角 × 4 r_v；r_v 用线型区分，slope 用颜色区分
    ls_map = {0.30: "-", 0.45: "--", 0.60: ":", 0.75: "-."}
    for k, slope in enumerate(SLOPES):
        for j, rv in enumerate(RVS):
            seq = [grid_amp[(slope, dh, rv)] for dh in DHS]
            is_extreme = rv in (0.30, 0.75)
            lw = 1.6 if is_extreme else 0.9
            alpha = 0.95 if is_extreme else 0.45
            ax1.plot(DHS, seq, ls_map[rv], color=SLOPE_COLORS[slope],
                     marker="o", ms=4.5, lw=lw, alpha=alpha)
    handles_slope = [
        plt.Line2D([], [], color=SLOPE_COLORS[sl], lw=1.6, label="$i$=%d$^\\circ$" % sl)
        for sl in SLOPES
    ]
    handles_rv = [
        plt.Line2D([], [], color=JOURNAL_BLACK, ls=ls_map[rv], lw=1.3,
                   label="$r_v$=%.2f" % rv) for rv in RVS
    ]
    # 坡角图例：放在右外侧（避免与曲线交叠）
    leg1 = ax1.legend(handles=handles_slope, loc="upper left",
                      bbox_to_anchor=(1.02, 1.02), fontsize=8, title="坡角 $i$",
                      title_fontsize=8.5, frameon=False)
    ax1.add_artist(leg1)
    # 波速比图例：放在右外侧下
    ax1.legend(handles=handles_rv, loc="lower left",
               bbox_to_anchor=(1.02, -0.05), fontsize=8, title="波速比 $r_v$",
               title_fontsize=8.5, frameon=False)
    ax1.set_xlabel("厚度比 $d/h$" if chinese else "Thickness ratio $d/h$")
    ax1.set_ylabel("复频响峰值幅值 $|G_h|_{\\max}$" if chinese else "Peak amplitude of $G_h$")
    ax1.set_title("(a) 峰值幅值随 $d/h$ 变化：16 条曲线中 14 条非单调" if chinese
                  else "(a) Peak amplitude vs $d/h$: 14/16 sequences non-monotonic", fontsize=9.5)

    # (b) rv=0.30 主峰频率
    for k, slope in enumerate(SLOPES):
        seq = [grid_hz[(slope, dh, 0.30)] for dh in DHS]
        ax2.plot(DHS, seq, marker="s", ms=5.5, color=SLOPE_COLORS[slope], lw=1.5,
                 label="$i$=%d$^\\circ$" % slope)
        offset = 0.32 if k % 2 == 0 else -0.50
        ax2.text(1.45, seq[-1] + offset, "%.1f Hz" % seq[-1], fontsize=8.5,
                 va="center", color=SLOPE_COLORS[slope])
    ax2.set_xlabel("厚度比 $d/h$" if chinese else "Thickness ratio $d/h$")
    ax2.set_ylabel("主峰频率 (Hz)" if chinese else "Dominant peak frequency (Hz)")
    ax2.set_xlim(0.1, 1.85)
    ax2.legend(fontsize=8, frameon=False, loc="upper right")
    ax2.set_title("(b) 最软覆盖层（$r_v$=0.30）主峰频率迁移" if chinese
                  else "(b) Peak-frequency migration, softest cover ($r_v$=0.30)", fontsize=9.5)
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig04_峰值幅值非单调性与主峰迁移.png" if chinese else "fig4_sp_h1_nonmonotonicity.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def fig5_sp_h2(chinese):
    """SP-H2：覆盖层修正量对 χ 的标度（逐例中位修正量）。"""
    set_font(chinese)
    from scipy.stats import spearmanr
    rows = read_csv(os.path.join(ANALYSIS, "layer_correction_metrics.csv"))
    chi, da, dp, dg = [], [], [], []
    for r in rows:
        dh = float(r["thickness_ratio"])
        rv = float(r["velocity_ratio"])
        chi.append(rv / (4 * dh))
        da.append(abs(float(r["median_delta_log_amplitude"])))
        dp.append(float(r["median_abs_delta_phase_deg"]))
        dg.append(float(r["median_abs_delta_group_delay_s"]))
    chi = np.array(chi)
    panels = [
        (np.array(da), "$|\\Delta_A|$ 中位数 (ln 单位)" if chinese else "$|\\Delta_A|$ median (ln units)",
         "(a) 幅值修正量" if chinese else "(a) Amplitude correction"),
        (np.array(dp), "$|\\Delta_\\Phi|$ 中位数 ($^\\circ$)" if chinese else "$|\\Delta_\\Phi|$ median (deg)",
         "(b) 相位修正量" if chinese else "(b) Phase correction"),
        (np.array(dg), "$|\\Delta\\tau_g|$ 中位数 (s)" if chinese else "$|\\Delta\\tau_g|$ median (s)",
         "(c) 群时延修正量" if chinese else "(c) Group-delay correction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    for ax, (y, ylab, title) in zip(axes, panels):
        ax.scatter(chi, y, s=34, color=JOURNAL_NAVY, edgecolors=JOURNAL_BLACK, linewidths=0.4, alpha=0.85)
        rho, p = spearmanr(chi, y)
        z = np.polyfit(chi, y, 2)
        xx = np.linspace(chi.min(), chi.max(), 100)
        ax.plot(xx, np.polyval(z, xx), color=JOURNAL_RED, lw=1.3, ls="-")
        ax.text(0.97, 0.95, "Spearman $\\rho$=%.2f\n$p$=%.1e" % (rho, p),
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="white", alpha=0.9, edgecolor=JOURNAL_LIGHTGRAY, lw=0.5,
                          boxstyle="round,pad=0.3"))
        ax.set_xlabel("$\\chi=r_v/(4\\,d/h)$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig05_覆盖层修正量与特征频率关系.png" if chinese else "fig5_sp_h2_chi_scaling.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


def fig6_cv_blind(chinese):
    """CV 与盲测逐例复数误差按 rv 分层箱线图（含参考线）。"""
    set_font(chinese)
    cv = read_csv(os.path.join(REPO_ROOT, "Run", "ch4_sp_ml", "cv_case_metrics.csv"))
    cm = read_csv(os.path.join(ANALYSIS, "case_metrics.csv"))
    rv_of_case = {r["case_id"]: float(r["velocity_ratio"]) for r in cm if r["group"] == "P"}
    cv_groups = {rv: [] for rv in RVS}
    for r in cv:
        if r["model"] != "pod_gpr":
            continue
        cv_groups[rv_of_case[r["case_id"]]].append(float(r["E_complex_w"]) * 100)
    blind = read_csv(os.path.join(REPO_ROOT, "Run", "ch4_sp_ml",
                                  "unseen_combination_metrics.csv"))
    b_rvs = [0.375, 0.525, 0.675]
    b_groups = {rv: [] for rv in b_rvs}
    for r in blind:
        rv = B_PARAMS[r["case_id"]][2]
        b_groups[rv].append(float(r["E_complex_w"]) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    pos = np.arange(len(RVS))
    bp1 = ax1.boxplot([cv_groups[rv] for rv in RVS], positions=pos, widths=0.5,
                      patch_artist=True,
                      boxprops=dict(facecolor="#c6dbef", edgecolor=JOURNAL_BLACK, lw=0.8),
                      whiskerprops=dict(color=JOURNAL_BLACK, lw=0.8),
                      capprops=dict(color=JOURNAL_BLACK, lw=0.8),
                      medianprops=dict(color=JOURNAL_RED, lw=1.5))
    for xi, rv in zip(pos, RVS):
        pts = cv_groups[rv]
        ax1.scatter(np.full(len(pts), xi) + np.linspace(-0.10, 0.10, len(pts)), pts,
                    s=18, color=JOURNAL_BLACK, alpha=0.75, zorder=3)
    ax1.set_xticks(pos)
    ax1.set_xticklabels(["%.2f" % rv for rv in RVS])
    ax1.set_xlabel("波速比 $r_v$（开发集网格）" if chinese else "$r_v$ (development grid)")
    ax1.set_ylabel("逐例加权复数误差 $E_{\\mathrm{complex},w}$ (%)" if chinese else "$E_{complex,w}$ per case (%)")
    ax1.axhline(10, color=JOURNAL_RED, ls="--", lw=1.0, label="参考线：中位数 $\\leq$10%" if chinese else "median $\\leq$10%")
    ax1.axhline(20, color=JOURNAL_RED, ls=":", lw=1.0, label="参考线：P90 $\\leq$20%" if chinese else "P90 $\\leq$20%")
    ax1.legend(fontsize=8, frameon=False, loc="upper right")
    ax1.set_title("(a) 五折交叉验证 POD-GPR（64 例 P 工况）" if chinese
                  else "(a) Five-fold CV, POD-GPR (64 P cases)", fontsize=9.5)

    pos = np.arange(len(b_rvs))
    bp2 = ax2.boxplot([b_groups[rv] for rv in b_rvs], positions=pos, widths=0.5,
                      patch_artist=True,
                      boxprops=dict(facecolor="#fddbc7", edgecolor=JOURNAL_BLACK, lw=0.8),
                      whiskerprops=dict(color=JOURNAL_BLACK, lw=0.8),
                      capprops=dict(color=JOURNAL_BLACK, lw=0.8),
                      medianprops=dict(color=JOURNAL_RED, lw=1.5))
    for xi, rv in zip(pos, b_rvs):
        pts = b_groups[rv]
        ax2.scatter(np.full(len(pts), xi) + np.linspace(-0.10, 0.10, len(pts)), pts,
                    s=24, color=JOURNAL_BLACK, alpha=0.75, zorder=3)
    ax2.set_xticks(pos)
    ax2.set_xticklabels(["%.3f" % rv for rv in b_rvs])
    ax2.set_xlabel("波速比 $r_v$（盲测参数）" if chinese else "$r_v$ (blind combinations)")
    ax2.axhline(10, color=JOURNAL_RED, ls="--", lw=1.0)
    ax2.axhline(20, color=JOURNAL_RED, ls=":", lw=1.0)
    ax2.set_title("(b) 一次性盲测评估（12 组未见组合）" if chinese
                  else "(b) One-shot blind test (12 unseen combinations)", fontsize=9.5)
    fig.tight_layout()
    save_fig(fig, OUT_DIR_CN if chinese else OUT_DIR_EN,
             "fig06_交叉验证与盲测误差分层.png" if chinese else "fig6_cv_blind_error_by_rv.png",
             "中文初稿" if chinese else "英文初稿")
    plt.close(fig)


if __name__ == "__main__":
    for chinese in (True, False):
        fig1_method_schematic(chinese)
        fig2_case_matrix_pipeline(chinese)
        fig3_fields_homo_vs_layered(chinese)
        fig4_sp_h1(chinese)
        fig5_sp_h2(chinese)
        fig6_cv_blind(chinese)