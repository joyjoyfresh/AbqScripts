# -*- coding: utf-8 -*-
"""
真实坐标 PGA 曲线绘制脚本
从已生成的 PGA-*.csv 文件中读取数据并绘制对比图。

主要特点:
1. 仅保留从 PGA-*.csv 文件读取并绘图的功能，不再进行 PGA 提取计算。
2. 横坐标为节点的真实绝对坐标 x。
3. 坡顶与坡脚的真实坐标 (#1 与 #2) 自动从 PGA-*-slope.csv 中的 y 坐标分布提取，若不存在则使用预设默认值。
4. 自动根据文件名解析波名与频率，在同一张图内绘制同系列不同频率的对比曲线。
5. 绘图风格：白底、四周黑边框、刻度朝内且四周显示、细密点状网格线、左上角带黑边框的白色图例。
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面模式
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ==============================================================================
#  配置与常量
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
PGA_GLOB_PATTERN = 'PGA-*.csv'                           # PGA 数据表文件模式
VS_OVERLYING = 1600.0  # 覆盖层剪切波速 (m/s)，用于由 Hz 估算 a0

# 默认坡体几何参数（当目录下不存在 PGA-*-slope.csv 文件时作为兜底使用）
DEFAULT_X_CREST = 800.0   # 默认坡顶 x 坐标
DEFAULT_X_TOE   = 1000.0  # 默认坡脚 x 坐标
DEFAULT_H_VAL   = 200.0   # 默认坡高 h

# 曲线样式映射（按频率/a0 升序循环使用）
CURVE_STYLES = [
    {'color': '#1f77b4', 'linestyle': '-',  'linewidth': 1.5},  # 蓝色实线
    {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 1.5},  # 橙色虚线
    {'color': '#2ca02c', 'linestyle': '-',  'linewidth': 1.5},  # 绿色实线
    {'color': '#d62728', 'linestyle': '--', 'linewidth': 1.5},  # 红色虚线
    {'color': '#9467bd', 'linestyle': '-',  'linewidth': 1.5},  # 紫色实线
    {'color': '#8c564b', 'linestyle': '--', 'linewidth': 1.5},  # 棕色虚线
    {'color': '#e377c2', 'linestyle': '-',  'linewidth': 1.5},  # 粉色实线
    {'color': '#7f7f7f', 'linestyle': '--', 'linewidth': 1.5},  # 灰色虚线
]

# ==============================================================================
#  中英文默认字体加载配置
# ==============================================================================
def build_font_properties():
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']
    cn_font = fm.FontProperties()
    en_font = fm.FontProperties()
    for name in cn_candidates:
        try:
            font_path = fm.findfont(name, fallback_to_default=False)
            cn_font = fm.FontProperties(fname=font_path)
            break
        except Exception:
            continue
    for name in en_candidates:
        try:
            font_path = fm.findfont(name, fallback_to_default=False)
            en_font = fm.FontProperties(fname=font_path)
            break
        except Exception:
            continue
    return cn_font, en_font

CN_FONT, EN_FONT = build_font_properties()

# ==============================================================================
#  辅助函数
# ==============================================================================
def get_incident_angle(cwd):
    """
    自适应获取入射角 theta_s。
    1. 优先尝试从同目录下的第一个 .cae 文件名末尾的 a{angle} 字段解析。
    2. 其次尝试从当前目录名中的 angle{angle} 或 a{angle} 字段解析。
    3. 兜底返回 0.0。
    """
    cae_files = sorted(glob.glob(os.path.join(cwd, '*.cae')))
    if cae_files:
        cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]
        match = re.search(r'a(-?\d+(?:\.\d+)?)$', cae_name)
        if match:
            return float(match.group(1))

    folder_name = os.path.basename(cwd)
    match_angle = re.search(r'angle(-?\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    if match_angle:
        return float(match_angle.group(1))
    match_a = re.search(r'a(-?\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)
    if match_a:
        return float(match_a.group(1))

    return 0.0


def parse_wave_info(core_name, h_val):
    """
    从 CSV 文件名的核心主干中解析出基础波名、a0 数值（排序键）及图例展示标签（即完整波名）。
    """
    # 1. 优先解析是否显式包含 a0
    match_a0 = re.search(r'[_-]a0[_-](?P<val>\d+(?:\.\d+)?)', core_name, re.IGNORECASE)
    if match_a0:
        a0_val = float(match_a0.group('val'))
        base_name = core_name[:match_a0.start()].strip('_-')
        return base_name, a0_val, core_name

    # 2. 尝试解析是否包含 Hz
    match_hz = re.search(r'[_-](?P<val>\d+(?:\.\d+)?)Hz', core_name, re.IGNORECASE)
    if match_hz:
        freq = float(match_hz.group('val'))
        base_name = core_name[:match_hz.start()].strip('_-')
        if h_val is not None and h_val > 1.0:
            a0_val = (2.0 * freq * h_val) / VS_OVERLYING
        else:
            a0_val = freq / 4.0  # 兜底假设 h = 200m
        return base_name, a0_val, core_name

    # 3. 兜底返回原名
    return core_name, 0.0, core_name


def style_axes(ax):
    """设置刻度朝内、四周框线、四周刻度及点线网格。"""
    ax.set_facecolor('white')
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.grid(True, which='both', linestyle=':', color='#b0b0b0', linewidth=0.5)


def get_slope_geometry(base_key, cwd):
    """
    从对应的 PGA-*-slope.csv 文件中提取坡顶/坡脚 x 坐标及坡高 h。
    返回 (x_crest, x_toe, h_val)，失败时返回 (None, None, None)。
    """
    # 优先寻找与当前波形对应的 PGA-*-slope.csv
    pga_slope_pattern = os.path.join(cwd, 'PGA-{}-slope.csv'.format(base_key))
    pga_slope_files = glob.glob(pga_slope_pattern)

    # 若找不到对应的，则尝试目录下任意 PGA-*-slope.csv
    if not pga_slope_files:
        pga_slope_files = glob.glob(os.path.join(cwd, 'PGA-*-slope.csv'))

    if pga_slope_files:
        try:
            slope_df = pd.read_csv(pga_slope_files[0])
            if 'x' in slope_df.columns and 'y' in slope_df.columns:
                y_max = slope_df['y'].max()
                y_min = slope_df['y'].min()
                h_val = y_max - y_min
                x_crest = slope_df[slope_df['y'] >= y_max - 1e-3]['x'].max()
                x_toe   = slope_df[slope_df['y'] <= y_min + 1e-3]['x'].min()
                print('  已通过 {} 识别几何参数: h={:.2f}m, 坡顶x={:.2f}m, 坡脚x={:.2f}m'.format(
                    os.path.basename(pga_slope_files[0]), h_val, x_crest, x_toe))
                return x_crest, x_toe, h_val
        except Exception as ex:
            print('  读取 PGA-slope 辅助文件获取几何信息失败: {}'.format(ex))

    return None, None, None

# ==============================================================================
#  主程序逻辑
# ==============================================================================
def main():
    print('>>> 启动 PGA.csv 绘图程序...')

    # 1. 扫描已有的 PGA-*.csv 文件（排除 *-slope.csv 辅助文件）
    all_pga_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, PGA_GLOB_PATTERN)))
    pga_files = [f for f in all_pga_files if not os.path.basename(f).lower().endswith('-slope.csv')]

    if not pga_files:
        print('错误: 目录 {} 下未找到匹配 {} 的 PGA 数据文件（已排除 *-slope.csv）。'.format(
            SCRIPT_DIR, PGA_GLOB_PATTERN))
        return

    print('共发现 {} 个 PGA 数据文件。'.format(len(pga_files)))

    # 2. 读取入射角 (theta_s)
    incident_angle = get_incident_angle(SCRIPT_DIR)
    print('识别到入射角度 theta_s = {}°'.format(incident_angle))

    # 3. 载入 PGA 数据并尝试关联坡体几何参数
    pga_records = []
    for filepath in pga_files:
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]

        # 提取 base_key (即 PGA- 之后的部分)
        if stem.upper().startswith('PGA-'):
            base_key = stem[4:]
        else:
            base_key = stem

        print('正在载入: {}...'.format(filename))
        try:
            pga_df = pd.read_csv(filepath)

            # 校验必需的列
            required_cols = {'x', 'PGA_h', 'PGA_v'}
            missing = required_cols - set(pga_df.columns)
            if missing:
                print('  警告: 文件 {} 缺失关键列 {}，跳过该文件。'.format(
                    filename, sorted(missing)))
                continue

            x_vals  = pga_df['x'].to_numpy()
            pga_h   = pga_df['PGA_h'].to_numpy()
            pga_v   = pga_df['PGA_v'].to_numpy()
            total_L = x_vals.max()

            # 尝试从 PGA-*-slope.csv 获取坡体几何信息
            x_crest, x_toe, h_val = get_slope_geometry(base_key, SCRIPT_DIR)

            # 兜底使用默认几何参数
            if x_crest is None or x_toe is None or h_val is None:
                x_crest = DEFAULT_X_CREST
                x_toe   = DEFAULT_X_TOE
                h_val   = DEFAULT_H_VAL
                print('  未找到 PGA-slope 辅助文件，将使用预设几何参数: '
                      'h={:.2f}m, 坡顶x={:.2f}m, 坡脚x={:.2f}m'.format(h_val, x_crest, x_toe))

            # 解析波形参数 (频率/a0 与波名图例)
            base_motion, a0_val, legend_label = parse_wave_info(base_key, h_val)

            pga_records.append({
                'base_motion': base_motion.lower(),
                'a0':          a0_val,
                'legend':      legend_label,
                'x':           x_vals,
                'pga_h':       pga_h,
                'pga_v':       pga_v,
                'x_crest':     x_crest,
                'x_toe':       x_toe,
                'total_L':     total_L,
            })

        except Exception as e:
            print('  处理 PGA 文件 {} 时出错: {}'.format(filename, e))

    if not pga_records:
        print('未加载到任何有效 PGA 数据记录，绘图终止。')
        return

    # 4. 按 base_motion 分组，在同一图表中绘制不同频率的对比图
    df_records = pd.DataFrame(pga_records)
    grouped = df_records.groupby('base_motion')

    # 支持 LaTeX 公式渲染
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    for motion_name, group_df in grouped:
        print('\n>>> 正在为波形 [{}] 绘制对比图...'.format(motion_name))

        # 按 a0 排序键（反映频率）排序曲线
        sorted_group = group_df.sort_values(by='a0').reset_index(drop=True)

        # 获取第一条曲线的斜坡几何定位
        x_crest = sorted_group.loc[0, 'x_crest']
        x_toe   = sorted_group.loc[0, 'x_toe']
        total_L = sorted_group.loc[0, 'total_L']

        # 循环绘制两个分量: 水平 (PGA_h) 与竖向 (PGA_v)
        for direction in ['horizontal', 'vertical']:
            fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=300)
            style_axes(ax)

            # 绘制不同频率的 PGA 曲线
            for idx, row in sorted_group.iterrows():
                style_cfg = CURVE_STYLES[idx % len(CURVE_STYLES)]
                y_data = row['pga_h'] if direction == 'horizontal' else row['pga_v']
                ax.plot(row['x'], y_data,
                        color=style_cfg['color'],
                        linestyle=style_cfg['linestyle'],
                        linewidth=style_cfg['linewidth'],
                        label=row['legend'])

            # 设定横轴范围与刻度
            ax.set_xlim(0, total_L)
            tick_step = 600.0 if abs(total_L - 1800.0) < 50.0 else (total_L / 3.0)
            ax.set_xticks(np.arange(0, total_L + 1.0, tick_step))

            # 获取当前所有 PGA 数据的最大最小值，保证 y 轴显示美观
            all_y = np.concatenate(
                [r['pga_h'] if direction == 'horizontal' else r['pga_v']
                 for _, r in sorted_group.iterrows()]
            )
            min_y = all_y.min()
            max_y = all_y.max()

            # 自适应 y 轴刻度：根据数据范围自动选取合适步长（目标 4~8 个刻度区间）
            TARGET_INTERVALS = 6  # 期望划分的刻度区间数
            raw_step = (max_y - min_y) / TARGET_INTERVALS if max_y > min_y else 1.0
            # 候选"整洁"步长列表（量级自动扩展）
            magnitude = 10.0 ** np.floor(np.log10(raw_step))
            nice_steps = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]
            y_step = magnitude * min(nice_steps, key=lambda s: abs(s * magnitude - raw_step))
            ylim_min = max(0.0, np.floor(min_y / y_step) * y_step)
            # 曲线最多占纵向总高度的 4/5，顶部留 1/5 空白给图例
            # 推导: (max_y - ylim_min) / (ylim_max - ylim_min) <= 4/5
            #   => ylim_max >= ylim_min + (max_y - ylim_min) * 5/4
            ylim_max_needed = ylim_min + (max_y - ylim_min) * (5.0 / 4.0)
            ylim_max = np.ceil(ylim_max_needed / y_step) * y_step
            # 再次确认上限严格大于 max_y（应对浮点误差）
            if ylim_max <= max_y:
                ylim_max += y_step
            ax.set_ylim(ylim_min, ylim_max)
            ax.set_yticks(np.arange(ylim_min, ylim_max + 1e-9, y_step))

            # 绘制坡顶与坡脚竖直虚线并标注 #1 与 #2
            ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)
            ax.axvline(x=x_toe,   color='black', linestyle='--', linewidth=1.0)

            # 文字标注位置
            text_y   = ylim_min + 0.92 * (ylim_max - ylim_min)
            offset_x = 0.02 * total_L
            ax.text(x_crest - offset_x, text_y, '#1',
                    fontsize=11, fontproperties=EN_FONT, va='top', ha='right')
            ax.text(x_toe - offset_x, text_y, '#2',
                    fontsize=11, fontproperties=EN_FONT, va='top', ha='right')

            # 轴标签与标题
            ax.set_xlabel('Surface Receiver Location (m)', fontsize=13, fontproperties=EN_FONT)
            dir_label = 'Horizontal' if direction == 'horizontal' else 'Vertical'
            ax.set_ylabel('{} PGA (m/s$^2$)'.format(dir_label), fontsize=13, fontproperties=EN_FONT)
            ax.set_title(r'$\theta_s = %g^\circ$' % incident_angle,
                         fontsize=14, fontproperties=EN_FONT, pad=10)

            # 图例：左上角，白色背景细黑边框
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black',
                      framealpha=1.0, prop=EN_FONT, fontsize=10.5)

            plt.tight_layout()

            # 保存对比图片
            out_img_name = 'PGA-comparison-{}-{}.png'.format(motion_name, direction)
            out_img_path = os.path.join(SCRIPT_DIR, out_img_name)
            plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print('  已成功输出对比图表: {}'.format(out_img_name))

    print('\n>>> PGA.csv 绘图流程全部完成。')


if __name__ == '__main__':
    main()
