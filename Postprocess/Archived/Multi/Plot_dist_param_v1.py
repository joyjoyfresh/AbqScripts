# -*- coding: utf-8 -*-
"""复现论文【图11 与图13】的 TAF 沿坡分布参数化对比图 v1。

图11（集 B，覆盖层厚度比 h/H 的影响）：
  固定 i=30°，Vr/Vs=1.25；每列对应无量纲频率 a0={0.5,1.0,1.5,2.0}；
  每行对应入射角 θs∈{0°,15°}；每子图内 3 条曲线对应 h/H∈{0.25,0.50,0.75}。
  (a) θs=0°：TAF_h；(b) θs=15°：TAF_h。共 2 行×4 列=8 子图。

图13（集 C，阻抗比 Vr/Vs 的影响）：
  固定 i=45°，h/H=0.5，a0=2.0；2 行×2 列（行=θs，列=H/V）；
  每子图内 3 条曲线对应 Vr/Vs∈{1.25,2.50,5.00}。

运行：
  python Postprocess/Multi/Plot_dist_param_v1.py  <工况根目录或 results 目录>
  不传参数则取当前目录。

输出到 results/Fig11_dist_hH/ 与 results/Fig13_dist_impedance/。
"""

import os  # 导入系统接口
import sys  # 导入系统参数
import re  # 导入正则（解析 Hz）
import json  # 导入 JSON（解析 layers_json）
import glob  # 导入文件匹配
import numpy as np  # 导入数值计算
import pandas as pd  # 导入数据分析
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.ticker as mticker  # 导入刻度工具
import matplotlib.font_manager as fm  # 导入字体管理器


# ==============================================================================
#  出版级样式（从 Plot_Multi_TAF_v2 内联复制，保持一致）
# ==============================================================================
CB_PALETTE = {  # Okabe-Ito 色盲安全配色
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'purple': '#CC79A7',
}
CJK_SERIF_PRIORITY = [  # 中文衬线字体优先级
    'Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC', 'Source Han Serif CN',
    'SimSun', 'NSimSun', 'STSong', 'Songti SC',
]


def _detect_cjk_serif():  # 检测可用中文衬线字体
    available = {f.name for f in fm.fontManager.ttflist}  # 所有可用字体名
    for name in CJK_SERIF_PRIORITY:  # 按优先级查找
        if name in available:  # 命中
            return name  # 返回
    for name in available:  # 关键词兜底
        if any(k in name.lower() for k in ('song', 'serif cjk', 'serif sc', 'songti')):
            return name  # 返回命中字体
    return None  # 未找到


def setup_journal_style():  # 应用出版级样式
    """应用中文核心期刊 rcParams；返回 {'cjk_serif': 字体名}。"""
    try:
        import scienceplots  # noqa: F401  # 可选：叠加 SciencePlots 基础样式
        plt.style.use(['science', 'no-latex'])
    except Exception:
        pass  # 未安装则跳过
    cjk = _detect_cjk_serif()  # 检测中文字体
    if cjk is None:
        raise RuntimeError('未找到中文衬线字体，请安装 SimSun 或 Source Han Serif。')
    serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # 字体回退链
    plt.rcParams.update({
        'font.family': serif_list, 'font.serif': serif_list,
        'mathtext.fontset': 'stix', 'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'lines.linewidth': 0.8, 'axes.linewidth': 0.7,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'figure.dpi': 150, 'savefig.dpi': 300,
    })
    return {'cjk_serif': cjk}  # 返回选用字体


def style_axes(ax):  # 统一坐标轴外观
    """白底、四面朝内刻度、细密网格。"""
    ax.set_facecolor('white')
    ax.tick_params(direction='in', which='both', top=True, right=True)
    for spine in ax.spines.values():
        spine.set_color('black'); spine.set_linewidth(1.0)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)


def export_fig(fig, path, dpi=300):  # 导出图片（PDF + PNG）
    """存 PDF 和 PNG；自动建父目录。"""
    base = os.path.splitext(path)[0]  # 去扩展名
    os.makedirs(os.path.dirname(os.path.abspath(base)) or '.', exist_ok=True)  # 建父目录
    for fmt in ('pdf', 'png'):  # 先矢量后栅格
        fig.savefig('{}.{}'.format(base, fmt), bbox_inches='tight', pad_inches=0.05,
                    **({'dpi': dpi} if fmt == 'png' else {}))
    print('  输出：{}.pdf / {}.png'.format(base, base))  # 打印路径


# ==============================================================================
#  数据工具
# ==============================================================================
def resolve_results_dir(arg):  # 定位 results/ 目录（含 index.csv）
    """返回含 index.csv 的目录；arg 为 results/ 本身或其上层均可；None 时取当前目录。"""
    base = os.path.abspath(arg) if arg else os.getcwd()  # 基准目录
    if os.path.isfile(os.path.join(base, 'index.csv')):  # 本身是 results/
        return base
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')):  # 上层目录
        return os.path.join(base, 'results')
    return None  # 未找到


def _num(row, col):  # 安全取数值
    if col not in row or pd.isna(row.get(col)):
        return None
    try:
        return float(row[col])
    except (TypeError, ValueError):
        return None


def compute_a0(row):  # 计算无量纲频率 a0 = fc * a0_base
    a0_base = _num(row, 'a0_base')  # 该工况 a0 换算基数
    record = row.get('record')  # 输入波记录名
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(record), re.IGNORECASE) if record else None
    if a0_base is None or m is None:
        return None
    return round(float(m.group(1)) * a0_base, 1)  # a0 = fc(Hz) × a0_base


def load_taf_index(results_dir):  # 读取 index.csv 中的 TAF 行
    """返回 DataFrame：仅 TAF 类型行，附加 a0 列。"""
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    taf = idx[idx['type'].astype(str).str.upper() == 'TAF'].copy()  # 筛选 TAF
    taf['a0'] = taf.apply(compute_a0, axis=1)  # 附加 a0 列
    return taf  # 返回 TAF 清单


def read_taf_curve(results_dir, fname):  # 读取单条 TAF 曲线
    """返回 (x, TAF_h, TAF_v) 数组；失败返回 (None, None, None)。"""
    path = os.path.join(results_dir, fname)  # 完整路径
    if not os.path.isfile(path):  # 文件不存在
        return None, None, None
    try:
        df = pd.read_csv(path)  # 读取
        return df['x'].to_numpy(float), df['TAF_h'].to_numpy(float), df['TAF_v'].to_numpy(float)
    except Exception as e:
        print('  读取 {} 失败: {}'.format(fname, e))
        return None, None, None


def get_crest_toe(results_dir):  # 从坡地 PGA 表获取坡顶/坡脚坐标
    """返回 (x_crest, x_toe)；找不到返回 (None, None)。"""
    for p in sorted(glob.glob(os.path.join(results_dir, 'PGA-*-slope.csv'))):
        try:
            df = pd.read_csv(p)
            ymax, ymin = df['y'].max(), df['y'].min()
            return float(df[df['y'] >= ymax - 1e-3]['x'].max()), float(df[df['y'] <= ymin + 1e-3]['x'].min())
        except Exception:
            continue
    return None, None


# ==============================================================================
#  参数化 TAF 分布图的通用绘图函数
# ==============================================================================
def _draw_dist_panel(ax, rows_filt, results_dir, curve_col, curve_label_fmt,
                     ylabel, x_crest=None, x_toe=None, title=None):
    """在 ax 上绘制参数化 TAF 分布曲线（行 = 一组工况，曲线 = 某参数取值）。

    rows_filt  : 已按(θs, a0等)筛好的 DataFrame 行，含 collected_file / curve_col 列；
    curve_col  : 用于区分曲线的 index 列名（如 'h_over_H' / 'vr_over_vs2'）；
    curve_label_fmt: 图例格式化模板（用 % 格式，接受 float 值）；
    ylabel     : 纵轴标签；x_crest/x_toe: 坡顶坡脚竖线位置。
    """
    style_axes(ax)  # 套用统一外观
    # 按 curve_col 取值分组画曲线（升序排列）
    param_vals = sorted(rows_filt[curve_col].dropna().unique())  # 该参数的取值列表（升序）
    COLORS = [CB_PALETTE['blue'], CB_PALETTE['orange'], CB_PALETTE['vermillion'],
              CB_PALETTE['green'], CB_PALETTE['purple']]  # 颜色序列
    LINESTYLES = ['-', '--', '-.', ':']  # 线型序列
    for ki, pval in enumerate(param_vals):  # 遍历参数取值
        subset = rows_filt[abs(rows_filt[curve_col] - pval) < 1e-3]  # 该参数值的行
        if subset.empty:  # 无数据
            continue
        x, th, tv = read_taf_curve(results_dir, str(subset.iloc[0]['collected_file']))  # 读取 TAF 曲线
        if x is None:  # 读取失败
            continue
        y = th if ylabel.startswith('水平') or 'TAF_h' in ylabel else tv  # 选水平或垂直分量
        label = curve_label_fmt % pval  # 图例文本
        ax.plot(x, y, color=COLORS[ki % len(COLORS)], linestyle=LINESTYLES[ki % len(LINESTYLES)],
                linewidth=1.0, label=label)  # 画曲线
    # 坡顶/坡脚标注
    if x_crest is not None:
        ylo, yhi = ax.get_ylim()  # 获取当前纵轴范围
        ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=0.8)  # 坡顶竖线
        ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=0.8)  # 坡脚竖线
        ax.text(x_crest * 0.98, yhi * 0.92, '#1', fontsize=6.5, ha='right', va='top')  # 标注 #1
        ax.text(x_toe * 0.98, yhi * 0.92, '#2', fontsize=6.5, ha='right', va='top')  # 标注 #2
    if title:  # 面板标题
        ax.set_title(title, pad=3)
    ax.set_ylabel(ylabel)  # 纵轴标签
    ax.set_xlabel('地表观测点位置 (m)')  # 横轴标签
    if param_vals:  # 有曲线才加图例
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', framealpha=0.9)


# ==============================================================================
#  图11：h/H 对 TAF_h 沿坡分布的影响（B 集）
# ==============================================================================
def plot_fig11(taf, results_dir, out_dir):  # 绘制图11
    """布局：行=θs(0°/15°)，列=a0(0.5/1.0/1.5/2.0)；曲线=h/H。"""
    print('>>> 绘制图11：覆盖层厚度比 h/H 对 TAF_h 分布的影响（i=30°, Vr/Vs=1.25）...')
    # 筛选：i=30°, Vr/Vs2≈1.25（双层均质覆盖层）
    filt = taf[
        (abs(taf['slope_i'].fillna(-1) - 30.0) < 1.0) &
        (taf['vr_over_vs2'].fillna(-1).between(1.20, 1.30)) &
        taf['a0'].notna() & taf['h_over_H'].notna()
    ].copy()
    if filt.empty:
        print('  警告：未找到图11 所需工况（i=30°, Vr/Vs=1.25）。'); return

    thetas = sorted(filt['incident_angle'].dropna().unique())  # 入射角列表
    a0s = sorted(filt['a0'].dropna().unique())  # a0 列表
    n_rows, n_cols = len(thetas), len(a0s)  # 行列数
    x_crest, x_toe = get_crest_toe(results_dir)  # 坡顶坡脚

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * n_rows), squeeze=False)  # 画布
    for ri, theta in enumerate(thetas):  # 遍历入射角行
        for ci, a0 in enumerate(a0s):  # 遍历 a0 列
            subset = filt[
                (abs(filt['incident_angle'] - theta) < 1.0) &
                (abs(filt['a0'] - a0) < 0.05)
            ]  # 筛选该(θs, a0)工况集
            ax = axes[ri][ci]  # 当前轴
            title = r'$a_0={:.1f},\ \theta_s={:g}^\circ$'.format(a0, theta)  # 子图标题
            _draw_dist_panel(ax, subset, results_dir, 'h_over_H',
                             r'$h/H = %.2f$', '水平向 TAF',
                             x_crest=x_crest, x_toe=x_toe, title=title)  # 绘制面板
    fig.suptitle(u'图11  不同 $h/H$ 下坡面 TAF 分布（$i=30°$, $V_R/V_s=1.25$）',
                 fontsize=9, y=1.01)  # 总标题
    fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.6)  # 紧凑布局
    out_path = os.path.join(out_dir, 'Fig11_dist_hH', 'Fig11')  # 输出路径基名
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 建目录
    export_fig(fig, out_path)  # 导出图片
    plt.close(fig)  # 释放画布


# ==============================================================================
#  图13：Vr/Vs 阻抗比对 TAF 沿坡分布的影响（C 集）
# ==============================================================================
def plot_fig13(taf, results_dir, out_dir):  # 绘制图13
    """布局：行=θs(0°/15°)，列=H/V；固定 i=45°, h/H=0.5, a0=2.0；曲线=Vr/Vs。"""
    print('>>> 绘制图13：阻抗比 Vr/Vs 对 TAF 分布的影响（i=45°, h/H=0.5, a0=2.0）...')
    # 筛选：i=45°, h/H=0.5, a0=2.0
    filt = taf[
        (abs(taf['slope_i'].fillna(-1) - 45.0) < 1.0) &
        (abs(taf['h_over_H'].fillna(-1) - 0.5) < 0.05) &
        (abs(taf['a0'].fillna(-1) - 2.0) < 0.1) &
        taf['vr_over_vs2'].notna()
    ].copy()
    if filt.empty:
        print('  警告：未找到图13 所需工况（i=45°, h/H=0.5, a0=2.0）。'); return

    thetas = sorted(filt['incident_angle'].dropna().unique())  # 入射角列表
    COMPS = [('水平向 TAF', 'TAF_h'), ('垂直向 TAF', 'TAF_v')]  # 分量列表
    n_rows, n_cols = len(thetas), len(COMPS)  # 行列数
    x_crest, x_toe = get_crest_toe(results_dir)  # 坡顶坡脚

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.15 * n_cols, 2.6 * n_rows), squeeze=False)  # 画布
    for ri, theta in enumerate(thetas):  # 遍历入射角行
        subset = filt[abs(filt['incident_angle'] - theta) < 1.0]  # 该入射角工况集
        for ci, (ylabel, comp) in enumerate(COMPS):  # 遍历分量列
            ax = axes[ri][ci]  # 当前轴
            title = r'$\theta_s={:g}^\circ$,  {}  TAF'.format(theta, '水平' if comp == 'TAF_h' else '垂直')  # 标题
            _draw_dist_panel(ax, subset, results_dir, 'vr_over_vs2',
                             r'$V_R/V_s = %.2g$', ylabel,
                             x_crest=x_crest, x_toe=x_toe, title=title)  # 绘制面板
    fig.suptitle(u'图13  不同 $V_R/V_s$ 下坡面 TAF 分布（$i=45°$, $h/H=0.5$, $a_0=2.0$）',
                 fontsize=9, y=1.01)  # 总标题
    fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.6)  # 紧凑布局
    out_path = os.path.join(out_dir, 'Fig13_dist_impedance', 'Fig13')  # 输出路径基名
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 建目录
    export_fig(fig, out_path)  # 导出图片
    plt.close(fig)  # 释放画布


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动 Plot_dist_param_v1（图11 + 图13）...')
    info = setup_journal_style()  # 应用出版级样式
    print('>>> 中文字体: {}'.format(info['cjk_serif']))

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数
    results_dir = resolve_results_dir(arg)  # 定位 results/ 目录
    if results_dir is None:
        print('错误：未找到 index.csv。请先运行 Collect_results_v2.py 生成 results/index.csv，'
              '然后传入工况根目录或 results 目录。'); return

    print('>>> 读取 index.csv：{}'.format(results_dir))
    taf = load_taf_index(results_dir)  # 读取 TAF 清单
    print('>>> 共读取 {} 条 TAF 记录'.format(len(taf)))

    out_dir = results_dir  # 图输出到 results/ 同级子目录
    plot_fig11(taf, results_dir, out_dir)  # 绘制图11
    plot_fig13(taf, results_dir, out_dir)  # 绘制图13
    print('>>> 完成。')


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
