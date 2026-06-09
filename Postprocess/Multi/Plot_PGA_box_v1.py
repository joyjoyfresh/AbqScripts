# -*- coding: utf-8 -*-
"""复现论文【图20】的最大 PGA 放大因子统计箱线图 v1。

图20 布局：
  (a) 按研究变量分组的水平向/垂直向最大放大因子箱线图
      分组：Slope angle（集A，i变）、Depth ratio（集B，h/H变）、
            Impedance ratio（集C，Vr/Vs变）、Surface layer（集D，双层）。
  (b) 按入射角（0° vs 15°）分组的水平向/垂直向箱线图。

数据来源：results/index.csv + results/TAF-*.csv（对每工况取各 TAF 列最大值）。

运行：
  python Postprocess/Multi/Plot_PGA_box_v1.py  <工况根目录或 results 目录>
"""

import os  # 导入系统接口
import sys  # 导入系统参数
import re  # 导入正则（解析 Hz）
import json  # 导入 JSON（解析 layers_json）
import numpy as np  # 导入数值计算
import pandas as pd  # 导入数据分析
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.font_manager as fm  # 导入字体管理器


# ==============================================================================
#  出版级样式
# ==============================================================================
CB_PALETTE = {
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7',
}
CJK_SERIF_PRIORITY = ['Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC',
                       'Source Han Serif CN', 'SimSun', 'NSimSun', 'STSong', 'Songti SC']


def _detect_cjk_serif():
    avail = {f.name for f in fm.fontManager.ttflist}
    for n in CJK_SERIF_PRIORITY:
        if n in avail: return n
    for n in avail:
        if any(k in n.lower() for k in ('song', 'serif cjk', 'serif sc', 'songti')): return n
    return None


def setup_journal_style():
    try:
        import scienceplots; plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    cjk = _detect_cjk_serif()
    if cjk is None: raise RuntimeError('未找到中文衬线字体。')
    sl = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']
    plt.rcParams.update({
        'font.family': sl, 'font.serif': sl, 'mathtext.fontset': 'stix',
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'lines.linewidth': 0.8, 'axes.linewidth': 0.7,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'figure.dpi': 150, 'savefig.dpi': 300,
    })
    return {'cjk_serif': cjk}


def style_axes(ax):
    ax.set_facecolor('white')
    ax.tick_params(direction='in', which='both', top=True, right=True)
    for sp in ax.spines.values(): sp.set_color('black'); sp.set_linewidth(1.0)
    ax.grid(True, which='major', axis='y', linestyle='-', color='#c8c8c8', linewidth=0.5)


def export_fig(fig, path, dpi=300):
    base = os.path.splitext(path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base)) or '.', exist_ok=True)
    for fmt in ('pdf', 'png'):
        fig.savefig('{}.{}'.format(base, fmt), bbox_inches='tight', pad_inches=0.05,
                    **({'dpi': dpi} if fmt == 'png' else {}))
    print('  输出：{}.pdf / {}.png'.format(base, base))


# ==============================================================================
#  数据工具
# ==============================================================================
def resolve_results_dir(arg):
    base = os.path.abspath(arg) if arg else os.getcwd()
    if os.path.isfile(os.path.join(base, 'index.csv')): return base
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')): return os.path.join(base, 'results')
    return None


def _num(row, col):
    if col not in row or pd.isna(row.get(col)): return None
    try: return float(row[col])
    except: return None


def compute_a0(row):  # a0 = fc * a0_base
    a0_base = _num(row, 'a0_base')
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(row.get('record', '')), re.IGNORECASE)
    if a0_base is None or m is None: return None
    return round(float(m.group(1)) * a0_base, 1)


def _classify_study_var(row):  # 把工况分类到研究变量组
    """返回 'A: slope angle'/'B: depth ratio'/'C: impedance ratio'/'D: surface layer' 或 None。

    判断依据：n_layers_total（层数）+ slope_i/h_over_H/vr_over_vs2。
    集 C 单独从集 A 中区分：n_layers=2 且 i=45 且 Vr/Vs≠1.25 → 集 C。
    集 D：n_layers=3（或 n_finite=2，有表层）。
    集 B：n_layers=2，i=30°，h/H≠0.5。
    集 A：剩余双层工况（i变，h/H=0.5，Vr/Vs=1.25）。
    """
    n_finite = _num(row, 'n_finite_layers')  # 有限层数
    i = _num(row, 'slope_i')  # 坡角
    hH = _num(row, 'h_over_H')  # 厚度比
    vr = _num(row, 'vr_over_vs2')  # 阻抗比

    if n_finite is None: return None

    # 集 D：有 2 个有限层（表层 + 覆盖层）
    if int(round(n_finite)) >= 2:
        return 'D: surface layer'

    # 以下均为单有限层（双层模型，n_finite=1）
    if i is not None and abs(i - 45.0) < 1.0 and vr is not None and abs(vr - 1.25) > 0.1:
        return 'C: impedance ratio'  # 集 C：i=45°，Vr/Vs≠1.25

    if i is not None and abs(i - 30.0) < 1.0 and hH is not None and abs(hH - 0.5) > 0.05:
        return 'B: depth ratio'  # 集 B：i=30°，h/H≠0.5

    return 'A: slope angle'  # 其余双层工况归集 A


def collect_max_taf(results_dir):  # 从 TAF CSV 收集最大放大因子
    """读取 index.csv 中全部 TAF 行，对每个文件取 max(TAF_h)/max(TAF_v)，返回 DataFrame。"""
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    taf_rows = idx[idx['type'].astype(str).str.upper() == 'TAF'].copy()  # 筛选 TAF 行
    records = []  # 结果列表
    for _, row in taf_rows.iterrows():  # 遍历每条 TAF 记录
        fpath = os.path.join(results_dir, str(row['collected_file']))  # 文件完整路径
        if not os.path.isfile(fpath): continue  # 文件不存在则跳过
        try:
            df = pd.read_csv(fpath)  # 读取 TAF 数据
            max_h = float(df['TAF_h'].max())  # 水平向峰值 TAF
            max_v = float(df['TAF_v'].max())  # 垂直向峰值 TAF
        except Exception:
            continue  # 读取失败则跳过
        study_var = _classify_study_var(row)  # 研究变量分类
        records.append({
            'max_taf_h': max_h, 'max_taf_v': max_v,  # 峰值数据
            'incident_angle': _num(row, 'incident_angle'),  # 入射角
            'study_var': study_var,  # 研究变量组
            'source_folder': str(row.get('source_folder', '')),  # 来源文件夹
        })
    return pd.DataFrame(records)  # 返回结果 DataFrame


# ==============================================================================
#  箱线图绘制
# ==============================================================================
def _boxplot_groups(ax, groups, data_col, group_order, color_h='#4472C4', color_v='#ED7D31', title=''):
    """在 ax 上对 groups 中的数据按 group_order 顺序绘制分组箱线图（H/V 两色）。

    groups : {group_label: df}（每组含 max_taf_h/max_taf_v）；group_order：组顺序列表。
    """
    positions_h, positions_v, data_h, data_v = [], [], [], []  # 位置与数据
    xtick_pos, xtick_labels = [], []  # x 刻度

    n_groups = len(group_order)  # 组数
    width = 0.3  # 箱宽
    gap = 1.0  # 组间距

    for gi, grp in enumerate(group_order):  # 遍历各组
        df = groups.get(grp, pd.DataFrame())  # 该组数据
        ph = gi * gap + 0.0  # 水平向箱位置
        pv = gi * gap + width + 0.05  # 垂直向箱位置
        positions_h.append(ph); data_h.append(df['max_taf_h'].dropna().tolist())  # 水平向数据
        positions_v.append(pv); data_v.append(df['max_taf_v'].dropna().tolist())  # 垂直向数据
        xtick_pos.append((ph + pv) / 2.0)  # 刻度居中位置
        xtick_labels.append(grp.split(':')[-1].strip())  # 去掉集标签，只保留描述

    bp_h = ax.boxplot(data_h, positions=positions_h, widths=width, patch_artist=True,
                      medianprops=dict(color='white', linewidth=1.5))  # 水平向箱图
    bp_v = ax.boxplot(data_v, positions=positions_v, widths=width, patch_artist=True,
                      medianprops=dict(color='white', linewidth=1.5))  # 垂直向箱图
    for patch in bp_h['boxes']:  # 水平向箱着色
        patch.set_facecolor(color_h); patch.set_alpha(0.8)
    for patch in bp_v['boxes']:  # 垂直向箱着色
        patch.set_facecolor(color_v); patch.set_alpha(0.8)
    ax.set_xticks(xtick_pos)  # 设置刻度位置
    ax.set_xticklabels(xtick_labels, fontsize=7, rotation=15, ha='right')  # 刻度标签
    ax.set_ylabel('最大放大因子')  # 纵轴标签
    ax.set_title(title, pad=3)  # 子图标题
    style_axes(ax)  # 套用外观
    # 图例
    from matplotlib.patches import Patch
    legend_h = Patch(facecolor=color_h, alpha=0.8, label='水平向 TAF')  # 水平向图例
    legend_v = Patch(facecolor=color_v, alpha=0.8, label='垂直向 TAF')  # 垂直向图例
    ax.legend(handles=[legend_h, legend_v], loc='upper right', frameon=True,
              facecolor='white', edgecolor='black', framealpha=0.9)  # 图例


# ==============================================================================
#  图20：统计箱线图
# ==============================================================================
def plot_fig20(df, out_dir):  # 绘制图20
    """(a) 按研究变量分组；(b) 按入射角分组。"""
    print('>>> 绘制图20：最大 PGA 放大因子统计箱线图...')
    if df.empty:
        print('  警告：无有效工况数据，跳过图20。'); return

    # ---- (a) 按研究变量 ----
    STUDY_ORDER = ['A: slope angle', 'B: depth ratio', 'C: impedance ratio', 'D: surface layer']  # 组顺序
    study_groups = {g: df[df['study_var'] == g] for g in STUDY_ORDER}  # 按研究变量分组

    # ---- (b) 按入射角 ----
    ANGLE_ORDER = ['θs=0°', 'θs=15°']  # 入射角组顺序
    df['angle_grp'] = df['incident_angle'].apply(lambda v: 'θs=0°' if v is not None and abs(float(v)) < 1.0 else 'θs=15°')
    angle_groups = {g: df[df['angle_grp'] == g] for g in ANGLE_ORDER}  # 按入射角分组

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.3 * 1.1, 3.2))  # 创建 1×2 画布（稍宽）
    _boxplot_groups(ax_a, study_groups, 'max_taf_h', STUDY_ORDER,
                    title=u'(a) 按研究变量分组')  # 子图 (a)
    _boxplot_groups(ax_b, angle_groups, 'max_taf_h', ANGLE_ORDER,
                    title=u'(b) 按入射角分组')  # 子图 (b)
    fig.suptitle(u'图20  最大 PGA 放大因子统计汇总（全部工况）', fontsize=9, y=1.02)  # 总标题
    fig.tight_layout(pad=0.5, w_pad=1.0)  # 紧凑布局
    out_path = os.path.join(out_dir, 'Fig20_PGA_box', 'Fig20')  # 输出路径基名
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 建目录
    export_fig(fig, out_path)  # 导出图
    plt.close(fig)  # 释放画布


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动 Plot_PGA_box_v1（图20）...')
    info = setup_journal_style()  # 应用出版级样式
    print('>>> 中文字体: {}'.format(info['cjk_serif']))

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数
    results_dir = resolve_results_dir(arg)  # 定位 results/ 目录
    if results_dir is None:
        print('错误：未找到 index.csv。请先运行 Collect_results_v2.py。'); return

    print('>>> 读取 index.csv 并收集各工况最大 TAF...')
    df = collect_max_taf(results_dir)  # 收集所有工况的最大 TAF
    print('>>> 共读取 {} 个工况的峰值数据'.format(len(df)))

    # 各组工况数汇总
    for g, cnt in df['study_var'].value_counts().items():
        print('  {}: {} 个工况'.format(g, cnt))

    plot_fig20(df, results_dir)  # 绘制图20
    print('>>> 完成。')


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
