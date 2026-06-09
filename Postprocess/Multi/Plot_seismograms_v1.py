# -*- coding: utf-8 -*-
"""复现论文【图10c / 图14c】的坡面合成地震图（地表波形沿坡面偏移排布）v1。

图10c：集A，i=45°，h/H=0.5，Vr/Vs=1.25，a0=2.0，θs=0°——垂直入射代表性工况。
图14c：集A，i=45°，h/H=0.5，Vr/Vs=1.25，a0=2.0，θs=15°——斜入射代表性工况。

每幅图布局：
  行 (a)/(b) = 水平向/垂直向加速度；
  各接收点的时程波形按坡面位置（x/h）做"瀑布图"偏移排布。
  波形按 peak_acc 归一化后再乘偏移比例，相邻波形间隔固定。

数据来源：results/TIMESERIES-*.csv（记录坡地 slope 各接收节点时程）。
工况识别：index.csv → incident_angle, slope_i, h_over_H, vr_over_vs2, vs1_over_vs2, record(含 Hz)。
识别 a0≈2.0：a0 = fc * a0_base，a0_base 来自 index.csv；record 包含 fc 信息。

运行：
  python Postprocess/Multi/Plot_seismograms_v1.py  <工况根目录或 results 目录>
输出：results/Fig10c_14c_seismograms/Fig10c.pdf/.png 和 Fig14c.pdf/.png
"""

import os  # 导入系统接口
import sys  # 导入系统参数
import re  # 导入正则
import json  # 导入 JSON
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
        'lines.linewidth': 0.6, 'figure.dpi': 150, 'savefig.dpi': 300,
    })
    return {'cjk_serif': cjk}


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


def _compute_a0(row):  # a0 = fc * a0_base
    a0_base = _num(row, 'a0_base')
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(row.get('record', '')), re.IGNORECASE)
    if a0_base is None or m is None: return None
    return round(float(m.group(1)) * a0_base, 2)


def find_timeseries_case(results_dir, theta_s, slope_i, hH, vr_vs, a0_target, a0_tol=0.2):
    """定位满足条件的 TIMESERIES slope 行（集A，特定 a0≈2.0）。

    theta_s : 入射角；slope_i : 坡角；hH : h/H；vr_vs : Vr/Vs；
    a0_target : 目标 a0；a0_tol : a0 匹配容差。
    返回匹配的第一行（Series），未找到返回 None。
    """
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    ts = idx[
        (idx['type'].astype(str).str.upper() == 'TIMESERIES') &  # 筛选 TIMESERIES
        (idx['scene'].astype(str).str.lower() == 'slope')  # 仅取坡地时程
    ].copy()
    ts['a0_val'] = ts.apply(_compute_a0, axis=1)  # 计算 a0 列

    match = ts[
        (abs(ts['incident_angle'].fillna(-1) - theta_s) < 1.0) &  # 入射角匹配
        (abs(ts['slope_i'].fillna(-1) - slope_i) < 2.0) &  # 坡角匹配
        (abs(ts['h_over_H'].fillna(-1) - hH) < 0.05) &  # h/H 匹配
        (abs(ts['vr_over_vs2'].fillna(-1) - vr_vs) < 0.15) &  # 阻抗比匹配
        (abs(ts['a0_val'].fillna(-1) - a0_target) < a0_tol) &  # a0 匹配
        (ts['vs1_over_vs2'].isna() | (ts['vs1_over_vs2'].fillna(-1) < 0))  # 单层（无 vs1_over_vs2）
    ]

    if match.empty:  # 未找到
        return None
    return match.iloc[0]  # 返回第一条匹配记录


def load_timeseries(ts_path):
    """读取 TIMESERIES CSV，返回 (time, node_ids, accel_h_matrix, accel_v_matrix, x_order)。

    time          : shape (N,)，时间数组；
    node_ids      : shape (M,)，节点 ID 列表（按列名顺序）；
    accel_h_mat   : shape (N, M)，水平向加速度矩阵（列=接收点）；
    accel_v_mat   : shape (N, M)，垂直向加速度矩阵；
    sort_idx      : shape (M,) 整数数组，按节点编号升序的排列索引。
    """
    df = pd.read_csv(ts_path)  # 读取时程数据
    time = df['Time'].to_numpy(float)  # 时间列

    # 提取水平向和垂直向加速度列
    h_cols = [c for c in df.columns if re.match(r'Node_(\d+)_Accel_h', c, re.IGNORECASE)]  # 水平向列
    v_cols = [c for c in df.columns if re.match(r'Node_(\d+)_Accel_v', c, re.IGNORECASE)]  # 垂直向列

    if not h_cols:  # 无水平加速度列
        raise ValueError('TIMESERIES 缺少水平向加速度列：{}'.format(ts_path))

    # 提取节点编号用于排序（节点编号隐含位置顺序）
    node_ids_h = [int(re.search(r'Node_(\d+)_Accel_h', c, re.IGNORECASE).group(1)) for c in h_cols]
    sort_idx = np.argsort(node_ids_h)  # 按节点编号排序索引

    accel_h = df[h_cols].to_numpy(float)[:, sort_idx]  # 水平向矩阵，按位置排序
    if v_cols:  # 有垂直向列
        node_ids_v = [int(re.search(r'Node_(\d+)_Accel_v', c, re.IGNORECASE).group(1)) for c in v_cols]
        sort_v = np.argsort(node_ids_v)  # 垂直向排序索引
        accel_v = df[v_cols].to_numpy(float)[:, sort_v]  # 垂直向矩阵
    else:
        accel_v = np.zeros_like(accel_h)  # 无垂直向数据时填零

    return time, np.array(node_ids_h)[sort_idx], accel_h, accel_v


def _waterfall_panel(ax, time, accel_mat, track_spacing=1.5, color='black', label_unit=''):
    """绘制瀑布式偏移地震图（每道波形按接收点序号依次偏移）。

    accel_mat : shape (N_time, N_receivers)；各道先归一化至 [-1,1]，再乘 track_spacing。
    track_spacing : 相邻道间距（无量纲，已归一化幅值单位）。
    """
    N, M = accel_mat.shape  # 时间点数 × 接收点数
    for j in range(M):  # 遍历各接收点
        amp = accel_mat[:, j]  # 该道时程
        peak = np.abs(amp).max()  # 峰值
        if peak > 1e-15:  # 避免除零
            amp_norm = amp / peak  # 归一化到 [-1,1]
        else:
            amp_norm = amp  # 零信号直接用
        offset = j * track_spacing  # 偏移量（道序 × 间距）
        ax.plot(time, amp_norm + offset, color=color, linewidth=0.5, alpha=0.85)  # 绘制偏移波形
    ax.set_yticks(np.arange(M) * track_spacing)  # 纵轴刻度对应接收点
    ax.set_yticklabels(['P{}'.format(j + 1) for j in range(M)], fontsize=5.5)  # 接收点标签
    ax.set_xlabel('时间 (s)')  # 横轴标签
    ax.set_ylabel('接收点（沿坡面位置）{}'.format(label_unit))  # 纵轴标签
    for sp in ax.spines.values():  # 边框设置
        sp.set_color('black'); sp.set_linewidth(0.8)


# ==============================================================================
#  图10c / 图14c：合成地震图
# ==============================================================================
def plot_seismograms(results_dir, out_dir, theta_s, fig_tag):  # 绘制合成地震图
    """绘制集A，i=45°，h/H=0.5，Vr/Vs=1.25，a0=2.0，θs=theta_s° 的坡面合成地震图。

    fig_tag : '10c' 或 '14c'，用于输出文件名。
    """
    print('>>> 绘制图{}：合成地震图（θs={}°）...'.format(fig_tag, theta_s))

    row = find_timeseries_case(results_dir, theta_s=theta_s,
                                slope_i=45.0, hH=0.5, vr_vs=1.25, a0_target=2.0)  # 定位工况
    if row is None:  # 未找到
        print('  警告：未找到 a0≈2.0 工况 (θs={}°，i=45°)，跳过图{}。'.format(theta_s, fig_tag))
        return

    fname = str(row['collected_file'])  # 文件名
    ts_path = os.path.join(results_dir, fname)  # 完整路径
    print('  加载时程文件：{}'.format(os.path.basename(ts_path)))

    try:
        time, node_ids, accel_h, accel_v = load_timeseries(ts_path)  # 加载时程数据
    except Exception as e:
        print('  加载失败 ({}): {}'.format(fname, e)); return

    fig, (ax_h, ax_v) = plt.subplots(2, 1, figsize=(4.5, 6.0), sharex=True)  # 2×1 子图

    _waterfall_panel(ax_h, time, accel_h,
                     track_spacing=1.5, color=CB_PALETTE['blue'])  # (a) 水平向地震图
    ax_h.set_title('(a) 水平向加速度', pad=3)  # 子图标题

    _waterfall_panel(ax_v, time, accel_v,
                     track_spacing=1.5, color=CB_PALETTE['vermillion'])  # (b) 垂直向地震图
    ax_v.set_title('(b) 垂直向加速度', pad=3)  # 子图标题

    # 计算并显示工况参数
    a0_val = _compute_a0(row)  # 工况 a0 值
    suptitle = u'图{}  坡面合成地震图（$i=45°$，$h/H=0.5$，$V_r/V_s=1.25$，$a_0={:.1f}$，$\\theta_s={}°$）'.format(
        fig_tag, a0_val if a0_val is not None else 2.0, theta_s)
    fig.suptitle(suptitle, fontsize=9, y=1.01)  # 总标题

    fig.tight_layout(pad=0.5, h_pad=0.8)  # 紧凑布局
    out_path = os.path.join(out_dir, 'Fig{}_seismogram'.format(fig_tag),
                            'Fig{}'.format(fig_tag))  # 输出路径基名
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 建目录
    export_fig(fig, out_path)  # 导出图
    plt.close(fig)  # 释放画布


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动 Plot_seismograms_v1（图10c / 图14c）...')
    info = setup_journal_style()  # 应用出版级样式
    print('>>> 中文字体: {}'.format(info['cjk_serif']))

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数
    results_dir = resolve_results_dir(arg)  # 定位 results/ 目录
    if results_dir is None:
        print('错误：未找到 index.csv。请先运行 Collect_results_v2.py。'); return

    print('>>> 数据目录：{}'.format(results_dir))
    plot_seismograms(results_dir, results_dir, theta_s=0.0, fig_tag='10c')  # 图10c：垂直入射
    plot_seismograms(results_dir, results_dir, theta_s=15.0, fig_tag='14c')  # 图14c：斜入射
    print('>>> 完成。')


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
