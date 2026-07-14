# -*- coding: utf-8 -*-
"""复现论文【图9、图12、图16】的峰值 TAF 参数化曲线图 v1。

图9（集 A，坡角 i 的影响）：
  横轴 a0={0.5,1.0,1.5,2.0}；布局 2×2（行=θs，列=H/V）；
  每子图 3 条曲线对应坡角 i=30°/45°/60°。

图12（集 B，覆盖层厚度比 h/H 的影响）：
  横轴 a0={0.5,1.0,1.5,2.0}；布局 2×2（行=θs，列=H/V）；
  每子图 3 条曲线对应 h/H=0.25/0.50/0.75。

图16（集 D，地表层相对厚度 h1/(H-h) 的影响）：
  横轴 h1/(H-h)={0.25,0.50,0.75,1.00}；布局 2×2（行=θs，列=H/V）；
  每子图 3 条曲线对应 Vs1/Vs2=0.50/0.75/2.00。

运行：
  python Postprocess/Multi/Plot_peakTAF_v1.py  <工况根目录或 results 目录>
  不传参数则取当前目录。

输出到 results/Fig9_peak_angle/、results/Fig12_peak_hH/、results/Fig16_peak_surface/。
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
import matplotlib.ticker as mticker  # 导入刻度工具
import matplotlib.font_manager as fm  # 导入字体管理器


# ==============================================================================
#  出版级样式（与 Plot_Multi_TAF_v2 一致）
# ==============================================================================
CB_PALETTE = {
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'blue': '#0072B2', 'vermillion': '#D55E00', 'purple': '#CC79A7',
}
CJK_SERIF_PRIORITY = ['Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC',
                       'Source Han Serif CN', 'SimSun', 'NSimSun', 'STSong', 'Songti SC']


def _detect_cjk_serif():  # 检测中文衬线字体
    avail = {f.name for f in fm.fontManager.ttflist}
    for n in CJK_SERIF_PRIORITY:
        if n in avail: return n
    for n in avail:
        if any(k in n.lower() for k in ('song', 'serif cjk', 'serif sc', 'songti')): return n
    return None


def setup_journal_style():  # 应用出版级样式
    try:
        import scienceplots; plt.style.use(['science', 'no-latex'])
    except Exception:
        pass
    cjk = _detect_cjk_serif()
    if cjk is None: raise RuntimeError('未找到中文衬线字体（SimSun/Source Han Serif）。')
    sl = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']
    plt.rcParams.update({
        'font.family': sl, 'font.serif': sl, 'mathtext.fontset': 'stix',
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'lines.linewidth': 0.9, 'axes.linewidth': 0.7,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'figure.dpi': 150, 'savefig.dpi': 300,
    })
    return {'cjk_serif': cjk}


def style_axes(ax):  # 统一轴外观
    ax.set_facecolor('white')
    ax.tick_params(direction='in', which='both', top=True, right=True)
    for sp in ax.spines.values(): sp.set_color('black'); sp.set_linewidth(1.0)
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.3)


def export_fig(fig, path, dpi=300):  # 导出 PDF + PNG
    base = os.path.splitext(path)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base)) or '.', exist_ok=True)
    for fmt in ('pdf', 'png'):
        fig.savefig('{}.{}'.format(base, fmt), bbox_inches='tight', pad_inches=0.05,
                    **({'dpi': dpi} if fmt == 'png' else {}))
    print('  输出：{}.pdf / {}.png'.format(base, base))


# ==============================================================================
#  数据工具
# ==============================================================================
def resolve_results_dir(arg):  # 定位 results/ 目录
    base = os.path.abspath(arg) if arg else os.getcwd()
    if os.path.isfile(os.path.join(base, 'index.csv')): return base
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')): return os.path.join(base, 'results')
    return None


def _num(row, col):  # 安全取数值
    if col not in row or pd.isna(row.get(col)): return None
    try: return float(row[col])
    except: return None


def compute_a0(row):  # 计算 a0 = fc × a0_base
    a0_base = _num(row, 'a0_base')
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(row.get('record', '')), re.IGNORECASE)
    if a0_base is None or m is None: return None
    return round(float(m.group(1)) * a0_base, 1)


def _h1_over(row):  # 从 layers_json 推算 h1/(H-h)
    """表层厚度 h1 = layers_json[0]['thickness']；H-h = H_minus_h（index 列）。"""
    hmh = _num(row, 'H_minus_h')  # H-h（坡高）
    if hmh is None or hmh <= 0: return None
    try:
        layers = json.loads(str(row.get('layers_json', '[]')))  # 解析层列表
        if layers and layers[0].get('thickness') is not None:  # 有表层厚度
            return round(float(layers[0]['thickness']) / hmh, 3)  # h1/(H-h)
    except Exception:
        pass
    return None  # 无法推算


def load_taf_extended(results_dir):  # 读取 TAF 清单并追加 a0 和 h1_over 列
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    taf = idx[idx['type'].astype(str).str.upper() == 'TAF'].copy()  # 筛选 TAF
    taf['a0'] = taf.apply(compute_a0, axis=1)  # 追加 a0 列
    taf['h1_over'] = taf.apply(_h1_over, axis=1)  # 追加 h1/(H-h) 列（双层模型有值）
    return taf


def peak_taf(results_dir, fname, comp):  # 读取 TAF 曲线并取峰值
    """comp='h' 取 TAF_h 峰值，comp='v' 取 TAF_v 峰值；失败返回 NaN。"""
    path = os.path.join(results_dir, fname)
    if not os.path.isfile(path): return float('nan')
    try:
        df = pd.read_csv(path)
        col = 'TAF_h' if comp == 'h' else 'TAF_v'  # 选分量
        return float(df[col].max())  # 取峰值
    except Exception:
        return float('nan')


# ==============================================================================
#  通用 2×2 峰值 TAF 绘图（行=θs，列=H/V）
# ==============================================================================
def _plot_2x2_peak(fig_name, suptitle, xlabel, x_vals, x_label_fmt,
                   curve_col, curve_label_fmt, curve_vals,
                   filtered_taf, results_dir, out_dir, x_tick_step=None):
    """通用 2×2 出图：行=入射角 θs，列=H/V 分量；x 轴为 x_vals；曲线按 curve_col 区分。

    filtered_taf : 已筛好的 DataFrame；x_vals：横轴离散点；
    curve_col   : 用于区分曲线的列（如 'slope_i' / 'h_over_H' / 'vs1_over_vs2'）；
    x_label_fmt : 横轴刻度格式函数 -> 字符串（用于 xticks）。
    """
    thetas = sorted(filtered_taf['incident_angle'].dropna().unique())  # 入射角列表
    COMPS = [('h', '水平向峰值 TAF'), ('v', '垂直向峰值 TAF')]  # 分量
    COLORS = [CB_PALETTE['blue'], CB_PALETTE['orange'], CB_PALETTE['vermillion'],
              CB_PALETTE['green'], CB_PALETTE['purple']]  # 颜色序列
    LSTS = ['-', '--', '-.']  # 线型

    fig, axes = plt.subplots(len(thetas), 2, figsize=(6.3, 2.6 * len(thetas)), squeeze=False)  # 画布
    for ri, theta in enumerate(thetas):  # 遍历入射角行
        theta_rows = filtered_taf[abs(filtered_taf['incident_angle'] - theta) < 1.0]  # 该入射角工况
        for ci, (comp, ylabel) in enumerate(COMPS):  # 遍历分量列
            ax = axes[ri][ci]  # 当前轴
            style_axes(ax)  # 套用外观
            for ki, cval in enumerate(curve_vals):  # 遍历曲线参数值
                curve_rows = theta_rows[abs(theta_rows[curve_col].fillna(-999) - cval) < 1e-3]  # 该曲线工况
                xp, yp = [], []  # 横纵坐标列表
                for xv in x_vals:  # 遍历横轴点
                    # 匹配该横轴点对应的工况行（x_tol=0.15，允许 a0/h1_over 小舍入误差）
                    row_match = curve_rows[abs(curve_rows[xlabel] - xv) < 0.15]
                    if row_match.empty:  # 无匹配
                        xp.append(xv); yp.append(float('nan'))  # 用 NaN 占位
                    else:
                        pv = peak_taf(results_dir, str(row_match.iloc[0]['collected_file']), comp)
                        xp.append(xv); yp.append(pv)  # 添加峰值
                label = curve_label_fmt % cval  # 图例文本
                ax.plot(xp, yp, color=COLORS[ki % len(COLORS)], linestyle=LSTS[ki % len(LSTS)],
                        marker='o', markersize=3, linewidth=0.9, label=label)  # 绘制曲线
            ax.set_xlabel(xlabel.replace('a0', r'$a_0$').replace('h1_over', r'$h_1/(H-h)$'))  # 横轴标签
            ax.set_ylabel(ylabel)  # 纵轴标签
            ax.set_title(r'$\theta_s={:g}^\circ$,  {}  TAF'.format(theta, '水平' if comp == 'h' else '垂直'), pad=3)  # 子图标题
            ax.set_xticks(x_vals)  # 设置 x 刻度
            ax.set_xticklabels([x_label_fmt % v for v in x_vals], fontsize=7)  # 刻度标签
            ax.legend(loc='upper left', frameon=True, facecolor='white',
                      edgecolor='black', framealpha=0.9)  # 图例
    fig.suptitle(suptitle, fontsize=9, y=1.01)  # 总标题
    fig.tight_layout(pad=0.5, h_pad=0.6, w_pad=0.6)  # 紧凑布局
    out_path = os.path.join(out_dir, fig_name, fig_name)  # 输出路径基名
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 建目录
    export_fig(fig, out_path)  # 导出图
    plt.close(fig)  # 释放画布


# ==============================================================================
#  图9：坡角 i 对峰值 TAF 的影响（集 A，横轴 a0）
# ==============================================================================
def plot_fig9(taf, results_dir, out_dir):  # 绘制图9
    print('>>> 绘制图9：坡角 i 对峰值 TAF 影响（Vr/Vs=1.25, h/H=0.5）...')
    filt = taf[
        (taf['vr_over_vs2'].fillna(-1).between(1.20, 1.30)) &
        (abs(taf['h_over_H'].fillna(-1) - 0.5) < 0.05) &
        taf['slope_i'].notna() & taf['a0'].notna()
    ].copy()  # 筛选集 A（双层，h/H=0.5，Vr/Vs=1.25）
    if filt.empty:
        print('  警告：未找到图9 所需工况。'); return
    i_vals = sorted(filt['slope_i'].dropna().unique())  # 坡角取值列表
    a0_vals = sorted(filt['a0'].dropna().unique())  # a0 取值列表
    _plot_2x2_peak(
        fig_name='Fig9_peak_angle',
        suptitle=u'图9  不同坡角 $i$ 下峰值 TAF 随无量纲频率 $a_0$ 的变化',
        xlabel='a0', x_vals=a0_vals, x_label_fmt='%.1f',
        curve_col='slope_i', curve_label_fmt=r'$i=%g^\circ$', curve_vals=i_vals,
        filtered_taf=filt, results_dir=results_dir, out_dir=out_dir,
    )


# ==============================================================================
#  图12：覆盖层厚度比 h/H 对峰值 TAF 的影响（集 B，横轴 a0）
# ==============================================================================
def plot_fig12(taf, results_dir, out_dir):  # 绘制图12
    print('>>> 绘制图12：h/H 对峰值 TAF 影响（i=30°, Vr/Vs=1.25）...')
    filt = taf[
        (abs(taf['slope_i'].fillna(-1) - 30.0) < 1.0) &
        (taf['vr_over_vs2'].fillna(-1).between(1.20, 1.30)) &
        taf['h_over_H'].notna() & taf['a0'].notna()
    ].copy()  # 筛选集 B（i=30°，双层，Vr/Vs=1.25）
    if filt.empty:
        print('  警告：未找到图12 所需工况。'); return
    hH_vals = sorted(filt['h_over_H'].dropna().unique())  # h/H 取值列表
    a0_vals = sorted(filt['a0'].dropna().unique())  # a0 取值列表
    _plot_2x2_peak(
        fig_name='Fig12_peak_hH',
        suptitle=u'图12  不同 $h/H$ 下峰值 TAF 随无量纲频率 $a_0$ 的变化（$i=30°$）',
        xlabel='a0', x_vals=a0_vals, x_label_fmt='%.1f',
        curve_col='h_over_H', curve_label_fmt=r'$h/H=%.2f$', curve_vals=hH_vals,
        filtered_taf=filt, results_dir=results_dir, out_dir=out_dir,
    )


# ==============================================================================
#  图16：地表层相对厚度 h1/(H-h) 对峰值 TAF 的影响（集 D，横轴 h1/(H-h)）
# ==============================================================================
def plot_fig16(taf, results_dir, out_dir):  # 绘制图16
    print('>>> 绘制图16：h1/(H-h) 对峰值 TAF 影响（i=45°, h/H=0.5, Vr/Vs2=2.5）...')
    filt = taf[
        (abs(taf['slope_i'].fillna(-1) - 45.0) < 1.0) &
        (abs(taf['h_over_H'].fillna(-1) - 0.5) < 0.05) &
        (taf['vr_over_vs2'].fillna(-1).between(2.45, 2.55)) &
        taf['vs1_over_vs2'].notna() & taf['h1_over'].notna()
    ].copy()  # 筛选集 D（i=45°，h/H=0.5，覆盖层 Vr/Vs2=2.5，含表层）
    if filt.empty:
        print('  警告：未找到图16 所需工况（检查 layers_json 是否包含表层厚度）。'); return
    vs1vs2_vals = sorted(filt['vs1_over_vs2'].dropna().unique())  # Vs1/Vs2 取值列表
    h1_vals = sorted(filt['h1_over'].dropna().unique())  # h1/(H-h) 取值列表
    _plot_2x2_peak(
        fig_name='Fig16_peak_surface',
        suptitle=u'图16  不同 $h_1/(H-h)$ 下峰值 TAF 随地表层相对厚度的变化（$i=45°$）',
        xlabel='h1_over', x_vals=h1_vals, x_label_fmt='%.2f',
        curve_col='vs1_over_vs2', curve_label_fmt=r'$V_{s1}/V_{s2}=%.2g$', curve_vals=vs1vs2_vals,
        filtered_taf=filt, results_dir=results_dir, out_dir=out_dir,
    )


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动 Plot_peakTAF_v1（图9 + 图12 + 图16）...')
    info = setup_journal_style()  # 应用出版级样式
    print('>>> 中文字体: {}'.format(info['cjk_serif']))

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数
    results_dir = resolve_results_dir(arg)  # 定位 results/ 目录
    if results_dir is None:
        print('错误：未找到 index.csv。请先运行 Collect_results_v2.py。'); return

    print('>>> 读取 index.csv：{}'.format(results_dir))
    taf = load_taf_extended(results_dir)  # 读取含 a0/h1_over 扩展列的 TAF 清单
    print('>>> 共读取 {} 条 TAF 记录'.format(len(taf)))

    out_dir = results_dir  # 输出到 results/ 同级子目录
    plot_fig9(taf, results_dir, out_dir)  # 绘制图9
    plot_fig12(taf, results_dir, out_dir)  # 绘制图12
    plot_fig16(taf, results_dir, out_dir)  # 绘制图16
    print('>>> 完成。')


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
