# -*- coding: utf-8 -*-
"""
真实坐标 TAF 后处理与对比图绘制脚本 v2
适用于 VAB_oblique_TAF_double_v2.py 建模与 Postprocess_PGA_v7.py 导出的 PGA 数据

主要特点:
1. 横坐标为节点的真实绝对坐标 x，不再进行归一化 (x/h)。
2. TAF 计算方法与原版一致：坡地 PGA 分量除以对应平地 (flat) 基准的 PGA 分量。
3. 由于坡地与平地网格节点水平位置可能微小错位，将平地 PGA 插值到坡地节点的真实 x 坐标上后进行逐点相除。
4. 坡顶与坡脚的真实坐标 (#1 与 #2) 依据坡地 top 节点的高度变化动态识别，并在图中标注垂直虚线与 `#1`、`#2`。
5. 自动根据文件名解析波名与频率，在同一张图内绘制同系列不同频率的对比曲线。
6. 绘图风格严格贴合参考图：白底、四周黑边框、刻度朝内且四周显示、细密点状网格线、左上角带黑边框的白色图例。
"""

import os
import re
import glob
import math
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
PGA_GLOB_PATTERN = 'PGA-*.csv'  # PGA 峰值表文件模式
SAFE_DIVIDE_EPS = 1e-12  # 安全除法阈值
VS_OVERLYING = 1600.0  # 覆盖层剪切波速 (m/s)，用于由 Hz 估算 a0

# 曲线样式映射（按频率/a0升序循环使用）
CURVE_STYLES = [
    {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 1.5},  # 第一条曲线: 绿色虚线
    {'color': '#1f77b4', 'linestyle': '-.', 'linewidth': 1.5},  # 第二条曲线: 蓝色点划线
    {'color': '#d62728', 'linestyle': '-', 'linewidth': 1.5},   # 第三条曲线: 红色实线
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
#  核心逻辑辅助函数
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
    支持:
    - a0 格式: Ricker_a0_1.0 -> Ricker, 1.0, "Ricker_a0_1.0"
    - Hz 格式: ricker_wavelet_4Hz -> ricker_wavelet, 1.0, "ricker_wavelet_4Hz" (按 a0 = 2*f*h/Vs 转换)
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
        # 根据 a0 = 2 * f * h / Vs 转换，其中 Vs 默认为 1600.0 m/s
        if h_val is not None and h_val > 1.0:
            a0_val = (2.0 * freq * h_val) / VS_OVERLYING
        else:
            a0_val = freq / 4.0  # 兜底假设 h = 200m，即 a0 = freq / 4.0
        return base_name, a0_val, core_name

    # 3. 兜底返回原名
    return core_name, 0.0, core_name

def parse_pair_key(csv_stem):
    """从 PGA 文件名主干中解析出配对键，并区分 slope 与 flat。"""
    cleaned = re.sub(r'^PGA[_-]*', '', csv_stem, flags=re.IGNORECASE)
    is_slope = cleaned.endswith('-slope')
    is_flat = cleaned.endswith('-flat')
    if is_slope:
        base_key = cleaned[:-6]
    elif is_flat:
        base_key = cleaned[:-5]
    else:
        base_key = cleaned
        is_slope = True  # 默认视为 slope
    return base_key, is_flat, is_slope

def load_pga_file(filepath):
    """读取单份 PGA CSV 文件并按 x 绝对坐标升序排序。"""
    df = pd.read_csv(filepath)
    required = {'node_label', 'x', 'y', 'PGA_h', 'PGA_v'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"文件 {os.path.basename(filepath)} 缺失关键列: {sorted(missing)}")
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x', 'y']).sort_values(by='x').reset_index(drop=True)
    return df

def normalize_pga_dataframe(df, step=1.0):
    """
    对 PGA 数据进行归一化处理，x 坐标间隔固定为 1，其他值按照近似插值（线性插值）处理。
    """
    # 按照 x 坐标排序并求均值，防止重复的 x 导致插值问题
    grouped_df = df.groupby('x', as_index=False).mean(numeric_only=True)
    x_src = grouped_df['x'].to_numpy(dtype=float)
    x_start = float(np.min(x_src))
    x_end = float(np.max(x_src))
    
    # 生成固定步长的 x 坐标，确保两端对齐
    x_norm = np.round(np.arange(x_start, x_end + step * 0.5, step), 10)
    
    norm_dict = {'x': x_norm}
    
    # 对其他数值列进行插值
    for col in df.columns:
        if col == 'x':
            continue
        y_src = grouped_df[col].to_numpy(dtype=float)
        y_norm = np.interp(x_norm, x_src, y_src)
        if col == 'node_label':
            # 节点标签用最近邻或四舍五入转为整型
            norm_dict[col] = np.round(y_norm).astype(int)
        else:
            norm_dict[col] = y_norm
            
    res_df = pd.DataFrame(norm_dict)
    # 按原 df 的列顺序重新排序列，只保留原 df 中存在的列
    cols = [col for col in df.columns if col in res_df.columns]
    return res_df[cols]


# ==============================================================================
#  绘图细节配置
# ==============================================================================
def style_axes(ax):
    """设置刻度朝内、四周框线、四周刻度及点线网格。"""
    ax.set_facecolor('white')
    # 设置刻度朝内并显示在四周
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, labelsize=11)
    # 设置框线为黑色且细线
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    # 细密点状网格线
    ax.grid(True, which='both', linestyle=':', color='#b0b0b0', linewidth=0.5)

# ==============================================================================
#  主处理逻辑
# ==============================================================================
def main():
    print(">>> 启动真实绝对坐标 TAF 后处理计算程序...")
    
    # 1. 扫描 PGA-*.csv 文件并排除归一化临时文件
    all_csvs = sorted(glob.glob(os.path.join(SCRIPT_DIR, PGA_GLOB_PATTERN)))
    pga_files = [f for f in all_csvs if '-normalized' not in os.path.splitext(os.path.basename(f))[0]]
    
    if not pga_files:
        print(f"错误: 目录 {SCRIPT_DIR} 下未找到匹配 {PGA_GLOB_PATTERN} 的 PGA 峰值表。")
        return
        
    print(f"共发现 {len(pga_files)} 个 PGA 原始数据文件。")

    # 2. 对 normal (slope) 和 flat 进行同波配对
    pairs = {}
    for filepath in pga_files:
        stem = os.path.splitext(os.path.basename(filepath))[0]
        base_key, is_flat, is_slope = parse_pair_key(stem)
        if base_key not in pairs:
            pairs[base_key] = {'slope': None, 'flat': None}
        if is_flat:
            pairs[base_key]['flat'] = filepath
        elif is_slope:
            pairs[base_key]['slope'] = filepath

    valid_pairs = {k: v for k, v in pairs.items() if v['slope'] and v['flat']}
    print(f"成功匹配 slope/flat 算例组共 {len(valid_pairs)} 对。")

    if not valid_pairs:
        print("未找到完整配对的 slope/flat 数据对，请确保同波形包含 *-slope.csv 与 *-flat.csv 文件。")
        return

    # 3. 读取入射角 (theta_s)
    incident_angle = get_incident_angle(SCRIPT_DIR)
    print(f"识别到入射角度 theta_s = {incident_angle}°")

    # 4. 逐组进行 TAF 计算 (用插值对齐绝对坐标)
    taf_records = []  # 用于保存内存记录以供多曲线图绘制
    
    for base_key, paths in sorted(valid_pairs.items()):
        print(f"--- 正在处理: {base_key} ---")
        try:
            slope_df = load_pga_file(paths['slope'])
            flat_df = load_pga_file(paths['flat'])
            
            # 计算斜坡高度 h
            y_max = slope_df['y'].max()
            y_min = slope_df['y'].min()
            h_val = y_max - y_min
            
            # 动态获取斜坡的关键地质坐标：坡顶和坡脚
            # 坡顶：顶部高平台上最右侧节点
            x_crest = slope_df[slope_df['y'] >= y_max - 1e-3]['x'].max()
            # 坡脚：底部低平台上最左侧节点
            x_toe = slope_df[slope_df['y'] <= y_min + 1e-3]['x'].min()
            
            print(f"  坡高 h = {h_val:.2f} m | 坡顶 x = {x_crest:.2f} m | 坡脚 x = {x_toe:.2f} m")

            # 归一化 PGA
            slope_df_norm = normalize_pga_dataframe(slope_df, step=1.0)
            flat_df_norm = normalize_pga_dataframe(flat_df, step=1.0)

            # 保存归一化后的 PGA
            slope_stem = os.path.splitext(os.path.basename(paths['slope']))[0]
            flat_stem = os.path.splitext(os.path.basename(paths['flat']))[0]
            slope_norm_path = os.path.join(SCRIPT_DIR, f"{slope_stem}-normalized.csv")
            flat_norm_path = os.path.join(SCRIPT_DIR, f"{flat_stem}-normalized.csv")
            slope_df_norm.to_csv(slope_norm_path, index=False, encoding='utf-8-sig')
            flat_df_norm.to_csv(flat_norm_path, index=False, encoding='utf-8-sig')
            print(f"  已保存归一化 PGA: {os.path.basename(slope_norm_path)} 和 {os.path.basename(flat_norm_path)}")

            # 将 flat field 结果插值到 slope 真实的 x 坐标上
            slope_x = slope_df_norm['x'].to_numpy()
            flat_x = flat_df_norm['x'].to_numpy()
            
            flat_pga_h_interp = np.interp(slope_x, flat_x, flat_df_norm['PGA_h'].to_numpy())
            flat_pga_v_interp = np.interp(slope_x, flat_x, flat_df_norm['PGA_v'].to_numpy())
            
            # TAF = slope / flat
            taf_h = np.zeros_like(slope_x)
            taf_v = np.zeros_like(slope_x)
            
            mask_h = np.abs(flat_pga_h_interp) > SAFE_DIVIDE_EPS
            mask_v = np.abs(flat_pga_v_interp) > SAFE_DIVIDE_EPS
            
            taf_h[mask_h] = slope_df_norm['PGA_h'].to_numpy()[mask_h] / flat_pga_h_interp[mask_h]
            taf_v[mask_v] = slope_df_norm['PGA_v'].to_numpy()[mask_v] / flat_pga_v_interp[mask_v]
            
            # 生成 TAF 数据表并保存 (仅保留 x 坐标与 TAF 分量)
            taf_df = pd.DataFrame({
                'x': slope_x,
                'TAF_h': taf_h,
                'TAF_v': taf_v
            })
            
            taf_out_name = f"TAF-{base_key}.csv"
            taf_out_path = os.path.join(SCRIPT_DIR, taf_out_name)
            taf_df.to_csv(taf_out_path, index=False, encoding='utf-8-sig')
            print(f"  已保存 TAF 数值表: {taf_out_name}")
            
            # 解析波形参数 (频率/a0与波名图例)
            base_motion, a0_val, legend_label = parse_wave_info(base_key, h_val)
            
            # 记录用于多波形对比图的数据
            base_motion_lower = base_motion.lower()
            taf_records.append({
                'base_motion': base_motion_lower,
                'a0': a0_val,
                'legend': legend_label,
                'x': slope_x,
                'y': slope_df_norm['y'].to_numpy(),
                'taf_h': taf_h,
                'taf_v': taf_v,
                'x_crest': x_crest,
                'x_toe': x_toe,
                'total_L': slope_x.max()
            })
            
        except Exception as e:
            print(f"  处理算例 {base_key} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    # 5. 按 base_motion 进行分组，并在同一图表中绘制不同波形的对比图
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
        
        # 按 a0 排序键(反映频率)排序曲线
        sorted_group = group_df.sort_values(by='a0').reset_index(drop=True)
        
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
            
            # 获取当前所有 TAF 数据的最大最小值，保证 y 轴显示美观
            all_y_values = np.concatenate([r['taf_h'] if direction == 'horizontal' else r['taf_v'] for _, r in sorted_group.iterrows()])
            min_y = all_y_values.min()
            max_y = all_y_values.max()
            
            # y 轴范围向上/向下按 0.5 圆整
            ylim_min = min(0.5, np.floor(min_y / 0.5) * 0.5)
            ylim_max = max(2.0, np.ceil(max_y / 0.5) * 0.5)
            ax.set_ylim(ylim_min, ylim_max)
            ax.set_yticks(np.arange(ylim_min, ylim_max + 0.05, 0.5))
            
            # 绘制坡顶与坡脚虚线并在侧上角标注 #1 与 #2
            ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)
            ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=1.0)
            
            # 文字位置：放在虚线左侧 2% total_L 处，高度在 y 轴顶部 92% 处
            text_y = ylim_min + 0.92 * (ylim_max - ylim_min)
            offset_x = 0.02 * total_L
            ax.text(x_crest - offset_x, text_y, '#1', fontsize=11, fontproperties=EN_FONT, va='top', ha='right')
            ax.text(x_toe - offset_x, text_y, '#2', fontsize=11, fontproperties=EN_FONT, va='top', ha='right')

            # 设置轴标签与标题 (Times New Roman Font)
            ax.set_xlabel('Surface Receiver Location(m)', fontsize=13, fontproperties=EN_FONT)
            dir_label = 'Horizontal' if direction == 'horizontal' else 'Vertical'
            ax.set_ylabel(f'{dir_label} TAF', fontsize=13, fontproperties=EN_FONT)
            ax.set_title(r'$\theta_s = %g^\circ$' % incident_angle, fontsize=14, fontproperties=EN_FONT, pad=10)
            
            # 放置图例：左上角，带有细黑边框的白色背景
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', 
                      framealpha=1.0, prop=EN_FONT, fontsize=10.5)
            
            plt.tight_layout()
            
            # 保存对比图片
            out_img_name = f"TAF-comparison-{motion_name}-{direction}.png"
            out_img_path = os.path.join(SCRIPT_DIR, out_img_name)
            plt.savefig(out_img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  已成功输出对比图表: {out_img_name}")

    print("\n>>> 后处理计算及绘图流程全部完成。")

if __name__ == '__main__':
    main()
