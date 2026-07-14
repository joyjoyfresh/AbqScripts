# -*- coding: utf-8 -*-
"""
真实坐标 TAF 后处理数据处理脚本 v3
适用于 VAB_oblique_TAF_double_v2.py 建模与 Postprocess_PGA_v7.py 导出的 PGA 数据

主要特点:
1. 仅保留 PGA 归一化和 TAF 数据计算与保存功能，移除了所有绘图相关逻辑。
2. 横坐标为节点的真实绝对坐标 x，不再进行归一化 (x/h)。
3. TAF 计算方法与原版一致：坡地 PGA 分量除以对应平地 (flat) 基准的 PGA 分量。
4. 由于坡地与平地网格节点水平位置可能微小错位，将平地 PGA 插值到坡地节点的真实 x 坐标上后进行逐点相除。
"""

import os
import re
import glob
import numpy as np
import pandas as pd

# ==============================================================================
#  配置与常量
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
PGA_GLOB_PATTERN = 'PGA-*.csv'  # PGA 峰值表文件模式
SAFE_DIVIDE_EPS = 1e-12  # 安全除法阈值

# ==============================================================================
#  核心逻辑辅助函数
# ==============================================================================
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
#  主处理逻辑
# ==============================================================================
def main():
    print(">>> 启动真实绝对坐标 TAF 数据处理计算程序...")
    
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

    # 3. 逐组进行 TAF 计算 (用插值对齐绝对坐标)
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
            
            # TAF = slope / flat_h (both horizontal and vertical divided by horizontal free-field)
            taf_h = np.zeros_like(slope_x)
            taf_v = np.zeros_like(slope_x)
            
            mask_h = np.abs(flat_pga_h_interp) > SAFE_DIVIDE_EPS
            
            taf_h[mask_h] = slope_df_norm['PGA_h'].to_numpy()[mask_h] / flat_pga_h_interp[mask_h]
            taf_v[mask_h] = slope_df_norm['PGA_v'].to_numpy()[mask_h] / flat_pga_h_interp[mask_h]
            
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
            
        except Exception as e:
            print(f"  处理算例 {base_key} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n>>> 后处理数据计算全部完成。")

if __name__ == '__main__':
    main()
