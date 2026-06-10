# -*- coding: utf-8 -*-
"""真实坐标 TAF 后处理数据处理脚本 v2（解析自由场分母，对齐 Shen 等 2025 论文式(5)）。

【与 v1 的核心区别 —— TAF 分母口径】
  v1：TAF = 坡地 PGA(x) ÷ 同地层平坦模型水平 PGA(x)。
      该口径把软表层的一维场地放大在分子分母中抵消掉，得到的只是"纯地形附加放大"，
      因而结果显著小于论文（论文图15 三层软表层差 2.5~5 倍）。
  v2：TAF_h = PGA_h(x) / PGA_ff_h；TAF_v = PGA_v(x) / PGA_ff_h（论文式(5) 口径）。
      分母 PGA_ff_h = factor_h × max|输入加速度| 为【基岩半空间自由地表运动】解析值：
        factor_h = (1−A1)·cosα + A2·sinβ（垂直入射时 = 2.0，15° 时 ≈ 1.92），
      其中 A1/A2 为基岩自由面 SV→SV 反射 / SV→P 转换系数。
      证据：论文图13/图15 远离坡体的平台处 TAF 不回到 1，而等于该柱一维放大倍数，
      说明论文分母不含覆盖层（不是同地层平坦自由场）。

【分母参数来源（按优先级）】
  ① case_meta.json 的 ff_normalization.factor_h（v6 建模脚本写出）；
  ② case_meta.json 的 incident_angle + bedrock.cs/vv 现场解析计算（兼容 v3~v5 旧工况，
     旧结果无需重算 Abaqus 即可用本脚本重新归一化）。

【输出（与 v1 同名同格式，下游 Collect/Plot 不受影响）】
  TAF-<记录>.csv：x, TAF_h, TAF_v（论文口径）
    + QA 附加列（若有 flat 数据）：TAF_h_flatref / TAF_v_flatref（v1 旧口径，仅对照）、
      AMP1D_h（平坦模型一维放大 = flat PGA_h / PGA_ff_h，应与论文图15 远场平台值一致）；
  FFNORM-<记录>.json：分母明细（factor_h、PGA 输入、PGA_ff_h、入射角、A1/A2）。

【运行】在工况文件夹内执行（Autorun 会把本脚本拷入各工况文件夹）：
  python Compute_TAF_v2.py [工况文件夹路径]
  （flat 数据现在是可选项：没有 *-flat.csv 也能算 TAF，仅缺 QA 列。）
"""

import os
import re
import sys
import math
import glob
import json
import numpy as np
import pandas as pd

# ==============================================================================
#  配置与常量
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录（被拷入工况文件夹后即工况目录）
PGA_GLOB_PATTERN = 'PGA-*.csv'  # PGA 峰值表文件模式
SAFE_DIVIDE_EPS = 1e-12  # 安全除法阈值

# ==============================================================================
#  解析自由场分母（自包含实现，与建模脚本 _compute_free_surface_sv_coeff 同公式）
# ==============================================================================
def _safe_arcsin(value):  # 安全反正弦（截断到 [-1,1]）
    """对 arcsin 输入做截断，避免浮点超界。"""
    return math.asin(max(-1.0, min(1.0, value)))  # 截断后求反正弦


def free_surface_sv_coeffs(alpha_rad, cp, cs):  # 基岩自由面 SV 反射/转换系数
    """返回 (A1, A2, beta)：SV→SV 反射系数、SV→P 转换系数、P 波角（弧度）。"""
    beta = _safe_arcsin(cp * math.sin(alpha_rad) / cs)  # 自由面 P 波转换角
    num = cs ** 2 * math.sin(2 * alpha_rad) * math.sin(2 * beta) - cp ** 2 * math.cos(2 * alpha_rad) ** 2  # A1 分子
    den = cs ** 2 * math.sin(2 * alpha_rad) * math.sin(2 * beta) + cp ** 2 * math.cos(2 * alpha_rad) ** 2  # 公共分母
    if abs(den) < 1e-12:  # 防除零
        den = 1e-12
    a1 = num / den  # SV→SV 反射系数
    a2 = (2 * cp * cs * math.sin(2 * alpha_rad) * math.cos(2 * alpha_rad)) / den  # SV→P 转换系数
    return a1, a2, beta  # 返回系数组


def ff_factor_h(angle_deg, cs, cp):  # 解析自由场水平放大系数
    """基岩半空间自由地表水平运动 / 入射波幅值；垂直入射=2.0、15°≈1.92。"""
    a = math.radians(angle_deg if abs(angle_deg) > 1e-12 else 1e-10)  # 入射角弧度（零角用极小值）
    a1, a2, beta = free_surface_sv_coeffs(a, cp, cs)  # 自由面系数
    fh = (1.0 - a1) * math.cos(a) + a2 * math.sin(beta)  # 水平分量合成系数
    return fh, a1, a2, beta  # 返回系数与中间量


def resolve_ff_denominator(meta_dir):  # 从 case_meta.json 解析分母系数
    """返回 dict {factor_h, A1, A2, beta_deg, angle, source}；找不到 meta 时抛异常。

    优先用 v6 写出的 ff_normalization 块；否则用 incident_angle + bedrock cs/vv 现场计算
    （兼容 v3~v5 旧工况：旧结果不必重算 Abaqus 即可重新归一化）。
    """
    meta_path = os.path.join(meta_dir, 'case_meta.json')  # 元数据路径
    if not os.path.isfile(meta_path):  # 缺失元数据
        raise IOError('未找到 %s（需要其中的入射角与基岩参数来构造解析分母）' % meta_path)
    with open(meta_path, 'r', encoding='utf-8') as f:  # 读取元数据
        meta = json.load(f)
    ffn = meta.get('ff_normalization') or {}  # v6 解析分母块
    if ffn.get('factor_h'):  # ① v6 建模脚本已写出
        return {'factor_h': float(ffn['factor_h']), 'A1': ffn.get('A1'), 'A2': ffn.get('A2'),
                'beta_deg': ffn.get('beta_deg'), 'angle': meta.get('incident_angle'),
                'source': 'case_meta.ff_normalization'}
    angle = float(meta.get('incident_angle') or 0.0)  # ② 旧工况：现场计算
    bed = meta.get('bedrock') or {}  # 基岩材料块
    cs = float(bed['cs'])  # 基岩剪切波速
    vv = float(bed.get('vv', 0.3))  # 基岩泊松比
    cp = cs * math.sqrt(2.0 * (1.0 - vv) / (1.0 - 2.0 * vv))  # 由 ν 换算纵波波速
    fh, a1, a2, beta = ff_factor_h(angle, cs, cp)  # 解析系数
    return {'factor_h': fh, 'A1': a1, 'A2': a2, 'beta_deg': math.degrees(beta),
            'angle': angle, 'source': 'computed_from_meta(bedrock+angle)'}


def find_input_record(folder, base_key):  # 查找该记录对应的输入加速度 txt
    """按 base_key 匹配输入波文件；找不到时若目录内恰有一个 txt 则使用之。"""
    cand = [os.path.join(folder, base_key + '.txt')]  # 首选：记录名.txt
    if base_key.lower().startswith('job-'):  # 兼容带 job- 前缀的命名
        cand.append(os.path.join(folder, base_key[4:] + '.txt'))
    for c in cand:  # 逐个候选检查
        if os.path.isfile(c):
            return c
    txts = sorted(glob.glob(os.path.join(folder, '*.txt')))  # 兜底：目录内全部 txt
    if len(txts) == 1:  # 恰好一个则使用
        return txts[0]
    raise IOError('无法确定记录 %s 的输入波 txt（候选: %s；目录 txt 数=%d）' % (base_key, cand, len(txts)))


# ==============================================================================
#  核心逻辑辅助函数（与 v1 一致）
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
        raise ValueError('文件 %s 缺失关键列: %s' % (os.path.basename(filepath), sorted(missing)))
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['x', 'y']).sort_values(by='x').reset_index(drop=True)
    return df


def normalize_pga_dataframe(df, step=1.0):
    """对 PGA 数据进行归一化处理，x 坐标间隔固定为 step，其余列线性插值。"""
    grouped_df = df.groupby('x', as_index=False).mean(numeric_only=True)  # 重复 x 取均值
    x_src = grouped_df['x'].to_numpy(dtype=float)
    x_norm = np.round(np.arange(float(np.min(x_src)), float(np.max(x_src)) + step * 0.5, step), 10)  # 等距网格
    norm_dict = {'x': x_norm}
    for col in df.columns:  # 逐列插值
        if col == 'x':
            continue
        y_src = grouped_df[col].to_numpy(dtype=float)
        y_norm = np.interp(x_norm, x_src, y_src)
        norm_dict[col] = np.round(y_norm).astype(int) if col == 'node_label' else y_norm
    res_df = pd.DataFrame(norm_dict)
    cols = [col for col in df.columns if col in res_df.columns]
    return res_df[cols]


# ==============================================================================
#  主处理逻辑
# ==============================================================================
def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else SCRIPT_DIR  # 工况文件夹（参数优先，默认脚本所在目录）
    folder = os.path.abspath(folder)
    print('>>> 启动 TAF 计算 v2（解析自由场分母，论文式(5) 口径）...')
    print('工况目录: %s' % folder)

    # 0. 解析分母系数（来自 case_meta.json；兼容旧工况现场计算）
    ffinfo = resolve_ff_denominator(folder)
    print('分母系数 factor_h = %.6f（入射角 %s°，来源: %s）' % (
        ffinfo['factor_h'], ffinfo['angle'], ffinfo['source']))

    # 1. 扫描 PGA-*.csv 文件并排除归一化临时文件
    all_csvs = sorted(glob.glob(os.path.join(folder, PGA_GLOB_PATTERN)))
    pga_files = [f for f in all_csvs if '-normalized' not in os.path.splitext(os.path.basename(f))[0]]
    if not pga_files:
        print('错误: 目录 %s 下未找到匹配 %s 的 PGA 峰值表。' % (folder, PGA_GLOB_PATTERN))
        return
    print('共发现 %d 个 PGA 原始数据文件。' % len(pga_files))

    # 2. slope/flat 同波配对（v2：flat 为可选 QA 项，不再是必需）
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
    valid = {k: v for k, v in pairs.items() if v['slope']}  # 仅要求 slope 存在
    print('找到 slope 数据 %d 组（其中含 flat QA 数据 %d 组）。' % (
        len(valid), sum(1 for v in valid.values() if v['flat'])))
    if not valid:
        print('未找到任何 *-slope.csv 数据，退出。')
        return

    # 3. 逐组计算 TAF（解析分母）
    for base_key, paths in sorted(valid.items()):
        print('--- 正在处理: %s ---' % base_key)
        try:
            # 3.1 分母：PGA_ff_h = factor_h × max|输入加速度|
            rec_path = find_input_record(folder, base_key)  # 输入波 txt
            rec = np.loadtxt(rec_path)  # 读取 [time, acc]
            pga_in = float(np.max(np.abs(rec[:, 1])))  # 输入加速度峰值（入射波幅值 E）
            denom = ffinfo['factor_h'] * pga_in  # 解析自由场水平 PGA（分母）
            print('  输入波: %s | PGA_in=%.6g | PGA_ff_h=%.6g' % (
                os.path.basename(rec_path), pga_in, denom))

            # 3.2 读取并归一化 slope PGA
            slope_df = load_pga_file(paths['slope'])
            slope_df_norm = normalize_pga_dataframe(slope_df, step=1.0)
            slope_stem = os.path.splitext(os.path.basename(paths['slope']))[0]
            slope_df_norm.to_csv(os.path.join(folder, '%s-normalized.csv' % slope_stem),
                                 index=False, encoding='utf-8-sig')
            slope_x = slope_df_norm['x'].to_numpy()

            # 3.3 论文口径 TAF（水平/竖向均除以水平解析分母）
            taf_h = slope_df_norm['PGA_h'].to_numpy() / max(denom, SAFE_DIVIDE_EPS)
            taf_v = slope_df_norm['PGA_v'].to_numpy() / max(denom, SAFE_DIVIDE_EPS)
            out = {'x': slope_x, 'TAF_h': taf_h, 'TAF_v': taf_v}

            # 3.4 QA 附加列（仅当 flat 数据存在）：旧口径对照 + 平坦一维放大
            if paths['flat']:
                flat_df = load_pga_file(paths['flat'])
                flat_df_norm = normalize_pga_dataframe(flat_df, step=1.0)
                flat_stem = os.path.splitext(os.path.basename(paths['flat']))[0]
                flat_df_norm.to_csv(os.path.join(folder, '%s-normalized.csv' % flat_stem),
                                    index=False, encoding='utf-8-sig')
                flat_h = np.interp(slope_x, flat_df_norm['x'].to_numpy(),
                                   flat_df_norm['PGA_h'].to_numpy())  # 插值到坡地 x
                mask = np.abs(flat_h) > SAFE_DIVIDE_EPS  # 安全除法掩码
                ref_h = np.zeros_like(slope_x); ref_v = np.zeros_like(slope_x)
                ref_h[mask] = slope_df_norm['PGA_h'].to_numpy()[mask] / flat_h[mask]  # v1 旧口径
                ref_v[mask] = slope_df_norm['PGA_v'].to_numpy()[mask] / flat_h[mask]
                out['TAF_h_flatref'] = ref_h  # 旧口径水平 TAF（仅对照）
                out['TAF_v_flatref'] = ref_v  # 旧口径竖向 TAF（仅对照）
                out['AMP1D_h'] = flat_h / max(denom, SAFE_DIVIDE_EPS)  # 平坦一维放大（QA：≈论文远场平台值）
                mid = (slope_x > slope_x.min() + (slope_x.max() - slope_x.min()) * 0.2) & \
                      (slope_x < slope_x.min() + (slope_x.max() - slope_x.min()) * 0.45)  # 上平台中段
                if np.any(mid):
                    print('  QA: 平坦模型一维放大(上平台中段中位数) AMP1D_h=%.3f（应≈论文图15 远场平台 TAF）'
                          % float(np.median(out['AMP1D_h'][mid])))

            # 3.5 保存 TAF 数据表（列序与 v1 兼容：x, TAF_h, TAF_v 在前）
            taf_df = pd.DataFrame(out)
            taf_out_name = 'TAF-%s.csv' % base_key
            taf_df.to_csv(os.path.join(folder, taf_out_name), index=False, encoding='utf-8-sig')
            print('  已保存 TAF 数值表: %s（峰值 TAF_h=%.3f, TAF_v=%.3f）' % (
                taf_out_name, float(np.nanmax(taf_h)), float(np.nanmax(taf_v))))

            # 3.6 保存分母明细（追溯用）
            ffnorm = dict(ffinfo)
            ffnorm.update({'record': base_key, 'record_file': os.path.basename(rec_path),
                           'pga_input': pga_in, 'pga_ff_h': denom,
                           'taf_definition': 'TAF_h=PGA_h/PGA_ff_h; TAF_v=PGA_v/PGA_ff_h (Shen2025 Eq.5)'})
            with open(os.path.join(folder, 'FFNORM-%s.json' % base_key), 'w', encoding='utf-8') as f:
                json.dump(ffnorm, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print('  处理算例 %s 时出错: %s' % (base_key, str(e)))
            import traceback
            traceback.print_exc()

    print('\n>>> TAF v2 后处理全部完成（解析自由场分母）。')


if __name__ == '__main__':
    main()
