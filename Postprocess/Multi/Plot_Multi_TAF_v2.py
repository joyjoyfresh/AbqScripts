# -*- coding: utf-8 -*-
"""
真实坐标 TAF 曲线绘制脚本
从已生成的 TAF-*.csv 文件中读取数据并绘制对比图，或由其他脚本导入调用。
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
import matplotlib.ticker as mticker  # 导入刻度定位与格式化模块

# ==============================================================================
#  配置与常量
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
TAF_GLOB_PATTERN = 'TAF-*.csv'  # TAF 数据表文件模式
CUSTOM_YLIM = {'horizontal': (0, 2), 'vertical': (0, 1)}  # 自定义纵轴显示范围。可为 None (自适应)，元组 (min, max)，或字典分别设定 {'horizontal': (min, max), 'vertical': (min, max)}
#CUSTOM_YLIM = None
TARGET_INTERVALS = 5  # 纵轴刻度区间划分数量（即把上下边界均划分为多少个小刻度区间）

# 默认坡体几何参数（当目录下不存在 PGA-*-slope.csv 文件时作为兜底使用）
DEFAULT_X_CREST = 800.0  # 默认坡顶 x 坐标
DEFAULT_X_TOE = 1000.0   # 默认坡脚 x 坐标
DEFAULT_H_VAL = 200.0     # 默认坡高 h

# 曲线样式映射（按频率升序循环使用）
CURVE_STYLES = [
    {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 1.5},  # 第一条曲线: 绿色虚线
    {'color': '#1f77b4', 'linestyle': '-.', 'linewidth': 1.5},  # 第二条曲线: 蓝色点划线
    {'color': '#d62728', 'linestyle': '-', 'linewidth': 1.5},   # 第三条曲线: 红色实线
    {'color': '#9467bd', 'linestyle': ':', 'linewidth': 1.5},   # 第四条曲线: 紫色点线
    {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.5},   # 第五条曲线: 橙色实线
    {'color': '#8c564b', 'linestyle': '--', 'linewidth': 1.5},  # 第六条曲线: 棕色虚线
    {'color': '#e377c2', 'linestyle': '-.', 'linewidth': 1.5},  # 第七条曲线: 粉色点划线
    {'color': '#7f7f7f', 'linestyle': ':', 'linewidth': 1.5},   # 第八条曲线: 灰色点线
    {'color': '#17becf', 'linestyle': '-', 'linewidth': 1.5},   # 第九条曲线: 青色实线
    {'color': '#bcbd22', 'linestyle': '--', 'linewidth': 1.5},  # 第十条曲线: 黄绿色虚线
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
    1. 优先尝试从同目录下的第一个 .cae 文件名末尾 of a{angle} 字段解析。
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


def parse_wave_info(core_name):  # 解析文件名波形参数以供分组与排序的辅助函数
    """
    从文件名中提取基础波名（用于同一图表内分组绘制）和频率数值（用于排序）。
    """
    match_hz = re.search(r'[_-](?P<val>\d+(?:\.\d+)?)Hz', core_name, re.IGNORECASE)  # 正则匹配文件名中的频率字段
    if match_hz:  # 若提取成功
        freq = float(match_hz.group('val'))  # 提取频率值作为排序键
        base_name = core_name[:match_hz.start()].strip('_-')  # 提取不含频率的基础波名作为分组键
        return base_name, freq  # 返回分组波名与频率排序键
    return core_name, 0.0  # 兜底返回完整名称与默认排序键 0.0


def style_axes(ax):
    """设置刻度朝内、四周框线、四周刻度及点线网格。"""
    ax.set_facecolor('white')
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    ax.grid(True, which='both', linestyle=':', color='#b0b0b0', linewidth=0.5)

# ==============================================================================
#  核心绘图函数
# ==============================================================================
def plot_taf_curves(taf_records, incident_angle, output_dir):
    """
    根据传入的 TAF 记录和入射角绘制对比图。
    """
    if not taf_records:
        print("无有效 TAF 数据记录，绘图终止。")
        return

    df_records = pd.DataFrame(taf_records)
    grouped = df_records.groupby('base_motion')

    # 支持 LaTeX 公式渲染设置
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False

    for motion_name, group_df in grouped:
        print(f"\n>>> 正在为波形 [{motion_name}] 绘制对比图...")
        
        # 按频率排序键对曲线进行升序排序
        sorted_group = group_df.sort_values(by='freq').reset_index(drop=True)  # 根据频率字段对分组记录进行升序排序
        
        # 获取第一条曲线的斜坡几何定位（所有曲线几何应一致）
        x_crest = sorted_group.loc[0, 'x_crest']
        x_toe = sorted_group.loc[0, 'x_toe']
        total_L = sorted_group.loc[0, 'total_L']

        # 循环绘制两个分量: 水平 (TAF_h) 与 竖向 (TAF_v)
        for direction in ['horizontal', 'vertical']:
            fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=300)
            style_axes(ax)
            
            # 绘制不同频率的 TAF 曲线
            for idx, row in sorted_group.iterrows():
                style_cfg = CURVE_STYLES[idx % len(CURVE_STYLES)]
                y_data = row['taf_h'] if direction == 'horizontal' else row['taf_v']
                ax.plot(row['x'], y_data, 
                        color=style_cfg['color'], 
                        linestyle=style_cfg['linestyle'], 
                        linewidth=style_cfg['linewidth'], 
                        label=row['legend'])
            
            # 设定横轴范围与刻度以适配 1800m 模型（适配绝对坐标 x）
            ax.set_xlim(0, total_L)
            # 自适应设定横轴步长，若是 1800m 则按 600m 一个大刻度
            tick_step = 600.0 if abs(total_L - 1800.0) < 50.0 else (total_L / 3.0)
            ax.set_xticks(np.arange(0, total_L + 1.0, tick_step))
            
            # 优先检查并读取自定义纵轴范围配置
            custom_lim = None  # 初始化自定义纵轴限制变量
            if isinstance(CUSTOM_YLIM, tuple) and len(CUSTOM_YLIM) == 2:  # 判断是否为通用的二元元组配置
                custom_lim = CUSTOM_YLIM  # 直接采用通用纵轴范围配置
            elif isinstance(CUSTOM_YLIM, dict):  # 判断是否为字典分量配置方式
                custom_lim = CUSTOM_YLIM.get(direction)  # 根据当前绘制方向提取对应的配置
                
            if custom_lim is not None:  # 若设定了有效的自定义纵轴范围
                ylim_min, ylim_max = custom_lim  # 解包自定义的纵轴下限与上限
                y_span = ylim_max - ylim_min  # 计算纵轴数值的跨度范围
                y_step = y_span / TARGET_INTERVALS if y_span > 1e-6 else 1.0  # 直接计算均分步长以精准匹配配置的区间数
            else:  # 若未设定自定义范围，则启用自适应估算逻辑
                # 获取当前所有 TAF 数据的最大最小值，保证 y 轴显示美观
                all_y_values = np.concatenate([r['taf_h'] if direction == 'horizontal' else r['taf_v'] for _, r in sorted_group.iterrows()])  # 拼接各个频率曲线的 y 数据点
                min_y = all_y_values.min()  # 取全部曲线的最小值
                max_y = all_y_values.max()  # 取全部曲线的最大值
                
                # 自适应 y 轴刻度：根据数据范围自动选取合适步长（目标 4~8 个刻度区间）
                y_span = max_y - min_y  # 计算数据的实际变化跨度
                raw_step = y_span / TARGET_INTERVALS if y_span > 1e-6 else 1.0  # 使用全局配置计算未圆整的基准步长
                # 候选"整洁"步长列表（量级自动扩展）
                magnitude = 10.0 ** np.floor(np.log10(raw_step))  # 确定当前步长的数量级
                nice_steps = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]  # 候选的美观刻度因子
                y_step = magnitude * min(nice_steps, key=lambda s: abs(s * magnitude - raw_step))  # 精准匹配最合适的圆整刻度步长
                
                ylim_min = max(0.0, np.floor(min_y / y_step) * y_step)  # 计算自适应的纵轴显示下限
                # 曲线最多占纵向总高度的 4/5，顶部留 1/5 空白给图例
                # 推导: (max_y - ylim_min) / (ylim_max - ylim_min) <= 4/5
                #   => ylim_max >= ylim_min + (max_y - ylim_min) * 5/4
                ylim_max_needed = ylim_min + (max_y - ylim_min) * (5.0 / 4.0)  # 估算所需的纵轴上限空间
                ylim_max = np.ceil(ylim_max_needed / y_step) * y_step  # 圆整得到自适应的纵轴显示上限
                # 再次确认上限严格大于 max_y（应对浮点误差）
                if ylim_max <= max_y:  # 如果圆整后的上限仍没有完全包含最大值
                    ylim_max += y_step  # 额外增加一个步长的高度以确保美观
                
            ax.set_ylim(ylim_min, ylim_max)
            ax.set_yticks(np.arange(ylim_min, ylim_max + 1e-9, y_step))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))  # 设置纵轴刻度标签显示为1位小数
            
            # 绘制坡顶与坡脚虚线并在侧上角标注 #1 与 #2
            ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)
            ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=1.0)
            
            # 文字位置：放在虚线左侧 2% total_L 处，高度在 y 轴顶部 92% 处
            text_y = ylim_min + 0.92 * (ylim_max - ylim_min)
            offset_x = 0.02 * total_L
            ax.text(x_crest - offset_x, text_y, '#1', fontsize=11, fontproperties=EN_FONT, va='top', ha='right')
            ax.text(x_toe - offset_x, text_y, '#2', fontsize=11, fontproperties=EN_FONT, va='top', ha='right')

            # 设置轴标签与标题 (中文字体支持)
            ax.set_xlabel('地表测点位置 (m)', fontsize=13, fontproperties=CN_FONT)  # 设置横坐标标签为中文“地表测点位置 (m)”
            dir_label = '水平向' if direction == 'horizontal' else '竖向'  # 根据绘制方向确定中文前缀
            ax.set_ylabel(f'{dir_label} TAF', fontsize=13, fontproperties=CN_FONT)  # 设置纵坐标标签为中文“水平向/竖向 TAF”
            ax.set_title(r'入射角 $\theta_s = %g^\circ$' % incident_angle, fontsize=14, fontproperties=CN_FONT, pad=10)  # 设置标题为中文“入射角”并结合 LaTeX 数学公式形式
            
            # 放置图例：左上角，带有细黑边框的白色背景
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', 
                      framealpha=1.0, prop=EN_FONT, fontsize=10.5)
            
            plt.tight_layout()
            
            # 保存对比图片
            out_img_name = f"TAF-comparison-{motion_name}-{direction}.png"
            out_img_path = os.path.join(output_dir, out_img_name)
            plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已成功输出对比图表: {out_img_name}")

# ==============================================================================
#  主处理逻辑
# ==============================================================================
def main():
    print(">>> 启动 TAF.csv 绘图程序 (Plot_Multi_TAF)...")
    
    # 1. 扫描已有的 TAF-*.csv 文件
    taf_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, TAF_GLOB_PATTERN)))
    
    if not taf_files:
        print(f"错误: 目录 {SCRIPT_DIR} 下未找到匹配 {TAF_GLOB_PATTERN} 的 TAF 数据文件。")
        return
        
    print(f"共发现 {len(taf_files)} 个 TAF 数据文件。")
    
    # 2. 读取入射角 (theta_s)
    incident_angle = get_incident_angle(SCRIPT_DIR)
    print(f"识别到入射角度 theta_s = {incident_angle}°")

    # 3. 载入 TAF 数据并尝试关联坡体几何参数
    taf_records = []
    for filepath in taf_files:
        filename = os.path.basename(filepath)
        stem = os.path.splitext(filename)[0]
        
        # 提取 base_key (即 TAF- 之后的部分)
        if stem.startswith('TAF-'):
            base_key = stem[4:]
        else:
            base_key = stem
            
        print(f"正在载入: {filename}...")
        try:
            taf_df = pd.read_csv(filepath)
            
            # 校验必需的列
            required_cols = {'x', 'TAF_h', 'TAF_v'}
            missing = required_cols - set(taf_df.columns)
            if missing:
                print(f"  警告: 文件 {filename} 缺失关键列 {sorted(missing)}，跳过该文件。")
                continue
                
            slope_x = taf_df['x'].to_numpy()
            taf_h = taf_df['TAF_h'].to_numpy()
            taf_v = taf_df['TAF_v'].to_numpy()
            total_L = slope_x.max()
            
            # 尝试从对应的 PGA 原始文件中读取坡顶坡脚坐标
            x_crest, x_toe, h_val = None, None, None
            
            # 优先寻找与当前波形对应的 PGA-*-slope.csv 原始文件
            pga_slope_pattern = os.path.join(SCRIPT_DIR, f"PGA-{base_key}-slope.csv")
            pga_slope_files = glob.glob(pga_slope_pattern)
            
            # 如果没找到特定的，尝试找本目录下任意 PGA-*-slope.csv
            if not pga_slope_files:
                pga_slope_files = glob.glob(os.path.join(SCRIPT_DIR, "PGA-*-slope.csv"))
                
            if pga_slope_files:
                try:
                    slope_filepath = pga_slope_files[0]
                    slope_df = pd.read_csv(slope_filepath)
                    if 'x' in slope_df.columns and 'y' in slope_df.columns:
                        y_max = slope_df['y'].max()
                        y_min = slope_df['y'].min()
                        h_val = y_max - y_min
                        x_crest = slope_df[slope_df['y'] >= y_max - 1e-3]['x'].max()
                        x_toe = slope_df[slope_df['y'] <= y_min + 1e-3]['x'].min()
                        print(f"  已通过 {os.path.basename(slope_filepath)} 识别几何参数: h={h_val:.2f}m, 坡顶x={x_crest:.2f}m, 坡脚x={x_toe:.2f}m")
                except Exception as ex:
                    print(f"  读取 PGA 辅助文件获取几何信息失败: {ex}")
            
            # 兜底使用默认几何参数
            if x_crest is None or x_toe is None or h_val is None:
                x_crest = DEFAULT_X_CREST
                x_toe = DEFAULT_X_TOE
                h_val = DEFAULT_H_VAL
                print(f"  未找到 PGA 原始文件，将使用预设几何参数: h={h_val:.2f}m, 坡顶x={x_crest:.2f}m, 坡脚x={x_toe:.2f}m")
                
            # 解析波形参数 (以不含频率的波名为分组键，以完整波名为图例)
            base_motion, freq_val = parse_wave_info(base_key)  # 解析获取用于分组的基础波名和用于排序的频率值
            legend_label = base_key  # 图例名称直接使用包含频率的完整波名
            
            taf_records.append({  # 将记录追加到列表中
                'base_motion': base_motion.lower(),  # 转为小写的基础波名
                'freq': freq_val,  # 频率数值作为后续排序键
                'legend': legend_label,  # 图例展示标签
                'x': slope_x,  # 坡面节点水平绝对坐标 x
                'taf_h': taf_h,  # 水平向地形放大因子
                'taf_v': taf_v,  # 竖向地形放大因子
                'x_crest': x_crest,  # 坡顶绝对坐标
                'x_toe': x_toe,  # 坡脚绝对坐标
                'total_L': total_L  # 坡面总长度
            })
            
        except Exception as e:
            print(f"  处理 TAF 文件 {filename} 时出错: {e}")
            
    plot_taf_curves(taf_records, incident_angle, SCRIPT_DIR)
    print("\n>>> TAF.csv 绘图流程全部完成。")

if __name__ == '__main__':
    main()
