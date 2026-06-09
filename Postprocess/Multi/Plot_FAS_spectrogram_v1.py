# -*- coding: utf-8 -*-
"""复现论文【图17】的坡面傅里叶振幅谱热图（Fourier Amplitude Spectrogram）v1。

图17：对 D07（软表层 Vs1/Vs2=0.5，h1/(H-h)=0.75，θs=15°）和 D23（硬表层 Vs1/Vs2=2.0，
同几何，θs=15°）两个工况，绘制"接收点位置（横轴）× 频率（纵轴）= 归一化傅里叶幅值（颜色）"热图。

数据来源：results/TIMESERIES-*.csv
  每行 = 一个时刻；列 = Time, Node_XXXXX_Accel_h, Node_XXXXX_Accel_v, ...（及 x/h 列）。
  每个接收节点单独做 FFT → 水平向幅值谱；再按接收点 x 坐标排列，得到 2D 矩阵。

工况识别（基于 index.csv 的 source_folder + vs1_over_vs2 / h1_over / incident_angle）：
  D07：Vs1/Vs2≈0.5，h1/(H-h)≈0.75，θs=15°；
  D23：Vs1/Vs2≈2.0，h1/(H-h)≈0.75，θs=15°。

运行：
  python Postprocess/Multi/Plot_FAS_spectrogram_v1.py  <工况根目录或 results 目录>
输出：results/Fig17_FAS/Fig17.pdf + .png
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
import matplotlib.colors as mcolors  # 导入颜色工具
import matplotlib.font_manager as fm  # 导入字体管理器


# ==============================================================================
#  出版级样式
# ==============================================================================
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
        'figure.dpi': 150, 'savefig.dpi': 300,
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


def _h1_over(row):  # 从 layers_json 推算 h1/(H-h)
    hmh = _num(row, 'H_minus_h')
    if hmh is None or hmh <= 0: return None
    try:
        layers = json.loads(str(row.get('layers_json', '[]')))
        if layers and layers[0].get('thickness') is not None:
            return round(float(layers[0]['thickness']) / hmh, 3)
    except Exception:
        pass
    return None


def find_timeseries_rows(results_dir, vs1vs2_target, h1_target, theta_target, tol=0.05):
    """在 index.csv 中按 vs1_over_vs2/h1_over/incident_angle 定位 TIMESERIES 记录。

    vs1vs2_target : Vs1/Vs2 目标值；h1_target : h1/(H-h) 目标值；theta_target : θs 目标值；
    tol : 匹配容差（用于浮点比较）。返回匹配的行 DataFrame（可能多行，取第一行）。
    """
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    ts = idx[idx['type'].astype(str).str.upper() == 'TIMESERIES'].copy()  # 筛选 TIMESERIES
    ts['h1_over'] = ts.apply(_h1_over, axis=1)  # 追加 h1/(H-h) 列
    match = ts[
        (abs(ts['vs1_over_vs2'].fillna(-1) - vs1vs2_target) < tol) &
        (abs(ts['h1_over'].fillna(-1) - h1_target) < tol) &
        (abs(ts['incident_angle'].fillna(-1) - theta_target) < tol) &
        (ts['scene'].astype(str).str.lower() == 'slope')  # 仅取坡地时程
    ]  # 匹配目标工况
    return match  # 返回匹配行


def compute_fas_matrix(ts_path, f_max=10.0):
    """读取 TIMESERIES CSV，对每个接收节点做 FFT，返回 (x_positions, freqs, amp_matrix)。

    x_positions : shape (n_receivers,)，接收点 x 坐标；
    freqs       : shape (n_freqs,)，频率数组（0~f_max Hz）；
    amp_matrix  : shape (n_freqs, n_receivers)，归一化傅里叶幅值矩阵（各接收点各自归一化）。
    """
    print('  读取 TIMESERIES：{}...'.format(os.path.basename(ts_path)))
    df = pd.read_csv(ts_path)  # 读取时程数据（可能较大）
    # 识别时间列和节点加速度列
    time_col = 'Time'  # 时间列名（按 Postprocess_PGA_v2 规范）
    if time_col not in df.columns:  # 时间列不存在
        raise ValueError('TIMESERIES 文件缺少 Time 列：{}'.format(ts_path))
    time = df[time_col].to_numpy(float)  # 时间数组
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0  # 采样步长
    N = len(time)  # 时间点数
    freqs = np.fft.rfftfreq(N, d=dt)  # 频率数组

    # 筛选频率范围 0~f_max Hz
    f_mask = freqs <= f_max  # 频率掩码
    freqs_plot = freqs[f_mask]  # 绘图用频率数组

    # 识别水平加速度列（列名形如 Node_XXXXX_Accel_h）
    h_cols = [c for c in df.columns if re.match(r'Node_\d+_Accel_h', c, re.IGNORECASE)]  # 水平向列
    if not h_cols:  # 无水平加速度列
        raise ValueError('TIMESERIES 文件中未找到 Node_*_Accel_h 列：{}'.format(ts_path))

    # 尝试从 x/h 列或其他方式获取 x 坐标（若无则用列序号）
    xh_col = 'x/h' if 'x/h' in df.columns else None  # x/h 列（若存在）

    x_positions = []  # 接收点 x 坐标（近似，用于排序）
    amp_list = []  # 各节点傅里叶幅值列表

    for col in h_cols:  # 遍历各节点水平加速度列
        acc = df[col].to_numpy(float)  # 加速度时程
        amp_full = np.abs(np.fft.rfft(acc))  # 傅里叶幅值（实数 FFT）
        amp = amp_full[f_mask]  # 截取 0~f_max Hz
        # 归一化：各节点自身幅值谱最大值归一（使不同位置的振幅差异显现）
        amp_max = amp.max()  # 该节点最大幅值
        if amp_max > 1e-15:  # 避免除零
            amp = amp / amp_max  # 归一化
        amp_list.append(amp)  # 收集

        # 尝试从列名提取节点编号（用于后续与 PGA 数据匹配 x 坐标）
        # 此处用列序号作为 x 位置的粗略代理（若无坐标信息）
        node_match = re.search(r'Node_(\d+)_Accel_h', col, re.IGNORECASE)
        x_approx = int(node_match.group(1)) if node_match else len(x_positions)  # 节点编号或序号
        x_positions.append(x_approx)  # 追加

    # 将列表转为数组并按 x 位置排序
    x_arr = np.array(x_positions, dtype=float)  # x 坐标数组
    sort_idx = np.argsort(x_arr)  # 排序索引
    x_sorted = x_arr[sort_idx]  # 排序后 x
    amp_matrix = np.column_stack(amp_list)[:, sort_idx]  # shape: (n_freqs, n_receivers)，按 x 排序

    return x_sorted, freqs_plot, amp_matrix  # 返回坐标、频率、幅值矩阵


# ==============================================================================
#  图17：两工况傅里叶振幅谱热图对比
# ==============================================================================
def plot_fig17(results_dir, out_dir):  # 绘制图17
    """(a) 软地表层 D07；(b) 硬地表层 D23。"""
    print('>>> 绘制图17：坡面傅里叶振幅谱热图...')

    # 定义两工况的目标参数
    CASES = [
        {'label': '(a)  $V_{s1}/V_{s2}=0.5$（软地表层）', 'vs1vs2': 0.5, 'h1': 0.75, 'theta': 15.0},
        {'label': '(b)  $V_{s1}/V_{s2}=2.0$（硬地表层）', 'vs1vs2': 2.0, 'h1': 0.75, 'theta': 15.0},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.5), squeeze=False)  # 1×2 画布
    n_plotted = 0  # 成功绘制的子图数

    for ci, case in enumerate(CASES):  # 遍历两工况
        ax = axes[0][ci]  # 当前轴
        rows = find_timeseries_rows(results_dir, case['vs1vs2'], case['h1'], case['theta'])  # 定位记录
        if rows.empty:  # 未找到
            print('  警告：未找到工况 {} (Vs1/Vs2={:.1f}, h1/(H-h)={:.2f}, θs={:g}°)'.format(
                ci + 1, case['vs1vs2'], case['h1'], case['theta']))
            ax.text(0.5, 0.5, '无数据', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(case['label'], pad=3)
            continue

        fname = str(rows.iloc[0]['collected_file'])  # 取第一条匹配记录
        ts_path = os.path.join(results_dir, fname)  # TIMESERIES 文件路径
        try:
            x_pos, freqs, amp_mat = compute_fas_matrix(ts_path, f_max=10.0)  # 计算幅值矩阵
        except Exception as e:
            print('  计算 FAS 失败 ({}): {}'.format(fname, e)); continue

        # pcolormesh 热图（横轴=接收点位置，纵轴=频率，颜色=归一化幅值）
        # amp_mat shape: (n_freqs, n_receivers)；pcolormesh 需要 x 和 y 网格
        im = ax.pcolormesh(x_pos, freqs, amp_mat,
                           cmap='hot_r', vmin=0.0, vmax=1.0, shading='auto')  # 绘制热图
        plt.colorbar(im, ax=ax, label='归一化傅里叶幅值', pad=0.02)  # 颜色条
        ax.set_xlabel('地表观测点位置')  # 横轴标签
        ax.set_ylabel('频率 (Hz)')  # 纵轴标签
        ax.set_ylim(0, 10.0)  # 纵轴 0~10 Hz
        ax.set_title(case['label'], pad=3)  # 子图标题
        for sp in ax.spines.values():  # 边框设置
            sp.set_color('black'); sp.set_linewidth(0.8)
        n_plotted += 1  # 成功计数

    if n_plotted == 0:  # 无成功绘制
        print('  警告：两个工况均无法绘制，图17 跳过。')
        plt.close(fig); return

    fig.suptitle(u'图17  软/硬地表层斜坡坡面傅里叶振幅谱对比（$h_1/(H-h)=0.75$，$\\theta_s=15°$）',
                 fontsize=9, y=1.01)  # 总标题
    fig.tight_layout(pad=0.5, w_pad=1.0)  # 紧凑布局
    out_path = os.path.join(out_dir, 'Fig17_FAS', 'Fig17')  # 输出路径基名
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # 建目录
    export_fig(fig, out_path)  # 导出图
    plt.close(fig)  # 释放画布


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动 Plot_FAS_spectrogram_v1（图17）...')
    info = setup_journal_style()  # 应用出版级样式
    print('>>> 中文字体: {}'.format(info['cjk_serif']))

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数
    results_dir = resolve_results_dir(arg)  # 定位 results/ 目录
    if results_dir is None:
        print('错误：未找到 index.csv。请先运行 Collect_results_v2.py。'); return

    print('>>> 数据目录：{}'.format(results_dir))
    plot_fig17(results_dir, results_dir)  # 绘制图17
    print('>>> 完成。')


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
