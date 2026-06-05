# -*- coding: utf-8 -*-
"""
网格收敛性分析后处理脚本
功能:
  1) 自动扫描指定目录下 fuke-10-* 文件夹，读取 PGA CSV 数据
  2) 在 x/h = 0,1,2,...,8 处提取最近节点的 PGA_h、PGA_v
  3) 以最细网格 (mesh_size=1) 为基准，计算各网格尺寸的相对误差
  4) 输出汇总 CSV 和误差对比 PNG 图表
"""

import matplotlib
matplotlib.use('Agg')  # 不弹出图表窗口

import os
import csv
import re
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
#  配置参数
# ============================================================
ROOT_DIR = r'D:\Abaqus'                    # 模型根目录
FOLDER_PATTERN = re.compile(r'^fuke-10-(\d+)$')  # 文件夹命名规则
WAVE_NAMES = [                              # 3 个地震波对应的 CSV 文件名
    'PGA_job-El_Centro_scaled.csv',
    'PGA_job-Loma_Prieta_scaled.csv',
    'PGA_job-Northridge_scaled.csv',
]
WAVE_SHORT = ['El_Centro', 'Loma_Prieta', 'Northridge']  # 简称
TARGET_XH = [0, 1, 2, 3, 4, 5, 6, 7, 8]   # 目标 x/h 值
BASELINE_MESH = 1                           # 基准网格尺寸
OUTPUT_DIR = ROOT_DIR                       # 输出目录


# ============================================================
#  工具函数
# ============================================================
def read_csv_data(csv_path):
    """读取 PGA CSV 文件，返回 [(x/h, PGA_h, PGA_v), ...]"""
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xh = float(row['x/h'])
            pga_h = float(row['PGA_h'])
            pga_v = float(row['PGA_v'])
            data.append((xh, pga_h, pga_v))
    return data


def find_nearest(data, target_xh):
    """
    在 data 中找到 x/h 最接近 target_xh 的行。
    返回 (actual_xh, PGA_h, PGA_v)
    """
    best = None
    best_dist = float('inf')
    for xh, pga_h, pga_v in data:
        dist = abs(xh - target_xh)
        if dist < best_dist:
            best_dist = dist
            best = (xh, pga_h, pga_v)
    return best


def extract_at_targets(data, targets):
    """
    对每个目标 x/h，提取最近点数据。
    返回 {target_xh: (actual_xh, PGA_h, PGA_v)}
    """
    result = {}
    for t in targets:
        result[t] = find_nearest(data, t)
    return result


# ============================================================
#  主流程
# ============================================================
def main():
    # ---- 1. 扫描文件夹 ----
    folders = {}  # {mesh_size: folder_path}
    for name in os.listdir(ROOT_DIR):
        m = FOLDER_PATTERN.match(name)
        if m:
            mesh_size = int(m.group(1))
            folder_path = os.path.join(ROOT_DIR, name)
            # 校验是否有 CSV 文件
            if all(os.path.isfile(os.path.join(folder_path, w)) for w in WAVE_NAMES):
                folders[mesh_size] = folder_path

    if not folders:
        print('[ERROR] 未找到有效的 fuke-10-* 文件夹（包含所有 CSV）')
        return

    mesh_sizes = sorted(folders.keys())
    print('找到 %d 个模型文件夹，网格尺寸: %s' % (len(mesh_sizes), mesh_sizes))

    if BASELINE_MESH not in mesh_sizes:
        print('[ERROR] 基准网格尺寸 %d 不在扫描结果中' % BASELINE_MESH)
        return

    # ---- 2. 读取数据并提取目标点 ----
    # all_data[mesh_size][wave_idx] = {target_xh: (actual_xh, PGA_h, PGA_v)}
    all_data = {}
    for ms in mesh_sizes:
        all_data[ms] = {}
        for wi, wname in enumerate(WAVE_NAMES):
            csv_path = os.path.join(folders[ms], wname)
            raw = read_csv_data(csv_path)
            all_data[ms][wi] = extract_at_targets(raw, TARGET_XH)
        print('  mesh_size=%d: 已读取 %d 个地震波 CSV' % (ms, len(WAVE_NAMES)))

    # ---- 3. 计算误差 ----
    baseline = all_data[BASELINE_MESH]

    # errors[mesh_size][wave_idx][target_xh] = (err_h%, err_v%)
    # errors_avg[mesh_size][target_xh] = (avg_err_h%, avg_err_v%)
    errors = {}
    errors_avg = {}
    for ms in mesh_sizes:
        errors[ms] = {}
        errors_avg[ms] = {}
        for wi in range(len(WAVE_NAMES)):
            errors[ms][wi] = {}
            for t in TARGET_XH:
                _, base_h, base_v = baseline[wi][t]
                _, cur_h, cur_v = all_data[ms][wi][t]
                err_h = (cur_h - base_h) / base_h * 100.0 if base_h != 0 else 0.0
                err_v = (cur_v - base_v) / base_v * 100.0 if base_v != 0 else 0.0
                errors[ms][wi][t] = (err_h, err_v)
        # 三波平均
        for t in TARGET_XH:
            avg_eh = np.mean([errors[ms][wi][t][0] for wi in range(len(WAVE_NAMES))])
            avg_ev = np.mean([errors[ms][wi][t][1] for wi in range(len(WAVE_NAMES))])
            errors_avg[ms][t] = (avg_eh, avg_ev)

    # ---- 4. 输出汇总 CSV ----
    csv_out = os.path.join(OUTPUT_DIR, 'mesh_convergence_summary.csv')
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头
        header = ['mesh_size', 'wave', 'x/h_target', 'x/h_actual', 'PGA_h', 'PGA_v',
                  'err_PGA_h(%)', 'err_PGA_v(%)']
        writer.writerow(header)
        for ms in mesh_sizes:
            # 各地震波
            for wi, wshort in enumerate(WAVE_SHORT):
                for t in TARGET_XH:
                    actual_xh, pga_h, pga_v = all_data[ms][wi][t]
                    err_h, err_v = errors[ms][wi][t]
                    writer.writerow([
                        ms, wshort, t,
                        '%.6f' % actual_xh,
                        '%.6f' % pga_h,
                        '%.6f' % pga_v,
                        '%.4f' % err_h,
                        '%.4f' % err_v,
                    ])
            # 三波平均行
            for t in TARGET_XH:
                # 平均 PGA
                avg_pga_h = np.mean([all_data[ms][wi][t][1] for wi in range(len(WAVE_NAMES))])
                avg_pga_v = np.mean([all_data[ms][wi][t][2] for wi in range(len(WAVE_NAMES))])
                avg_err_h, avg_err_v = errors_avg[ms][t]
                writer.writerow([
                    ms, 'Average', t,
                    '-',
                    '%.6f' % avg_pga_h,
                    '%.6f' % avg_pga_v,
                    '%.4f' % avg_err_h,
                    '%.4f' % avg_err_v,
                ])
    print('汇总 CSV 已保存: %s' % csv_out)

    # ---- 5. 绘图 ----
    # 排除基准自身（误差为 0）用于绘图
    plot_ms = [ms for ms in mesh_sizes if ms != BASELINE_MESH]

    # 颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_XH)))

    # ----- 图1: 各地震波 PGA_h 误差 -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PGA_h Relative Error vs Mesh Size', fontsize=14, fontweight='bold')
    for wi, wshort in enumerate(WAVE_SHORT):
        ax = axes[wi // 2][wi % 2]
        for ci, t in enumerate(TARGET_XH):
            y_vals = [errors[ms][wi][t][0] for ms in plot_ms]
            ax.plot(plot_ms, y_vals, 'o-', color=colors[ci], label='x/h=%d' % t, markersize=4)
        ax.set_title(wshort, fontsize=11)
        ax.set_xlabel('Mesh Size (m)')
        ax.set_ylabel('Relative Error (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
    # 第4子图：三波平均
    ax = axes[1][1]
    for ci, t in enumerate(TARGET_XH):
        y_vals = [errors_avg[ms][t][0] for ms in plot_ms]
        ax.plot(plot_ms, y_vals, 'o-', color=colors[ci], label='x/h=%d' % t, markersize=4)
    ax.set_title('Three-wave Average', fontsize=11)
    ax.set_xlabel('Mesh Size (m)')
    ax.set_ylabel('Relative Error (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png1 = os.path.join(OUTPUT_DIR, 'mesh_convergence_PGA_h.png')
    fig.savefig(png1, dpi=150)
    plt.close(fig)
    print('PGA_h 误差图已保存: %s' % png1)

    # ----- 图2: 各地震波 PGA_v 误差 -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PGA_v Relative Error vs Mesh Size', fontsize=14, fontweight='bold')
    for wi, wshort in enumerate(WAVE_SHORT):
        ax = axes[wi // 2][wi % 2]
        for ci, t in enumerate(TARGET_XH):
            y_vals = [errors[ms][wi][t][1] for ms in plot_ms]
            ax.plot(plot_ms, y_vals, 'o-', color=colors[ci], label='x/h=%d' % t, markersize=4)
        ax.set_title(wshort, fontsize=11)
        ax.set_xlabel('Mesh Size (m)')
        ax.set_ylabel('Relative Error (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
    ax = axes[1][1]
    for ci, t in enumerate(TARGET_XH):
        y_vals = [errors_avg[ms][t][1] for ms in plot_ms]
        ax.plot(plot_ms, y_vals, 'o-', color=colors[ci], label='x/h=%d' % t, markersize=4)
    ax.set_title('Three-wave Average', fontsize=11)
    ax.set_xlabel('Mesh Size (m)')
    ax.set_ylabel('Relative Error (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png2 = os.path.join(OUTPUT_DIR, 'mesh_convergence_PGA_v.png')
    fig.savefig(png2, dpi=150)
    plt.close(fig)
    print('PGA_v 误差图已保存: %s' % png2)

    # ----- 图3: 综合平均绝对误差 -----
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Mean Absolute Error across x/h=0~8 vs Mesh Size',
                 fontsize=14, fontweight='bold')
    # 各波 + 平均，分 PGA_h 和 PGA_v
    for wi, wshort in enumerate(WAVE_SHORT):
        mae_h = [np.mean([abs(errors[ms][wi][t][0]) for t in TARGET_XH]) for ms in plot_ms]
        mae_v = [np.mean([abs(errors[ms][wi][t][1]) for t in TARGET_XH]) for ms in plot_ms]
        ax.plot(plot_ms, mae_h, 's-', label='%s PGA_h' % wshort, markersize=5)
        ax.plot(plot_ms, mae_v, '^--', label='%s PGA_v' % wshort, markersize=5)
    # 三波平均
    mae_h_avg = [np.mean([abs(errors_avg[ms][t][0]) for t in TARGET_XH]) for ms in plot_ms]
    mae_v_avg = [np.mean([abs(errors_avg[ms][t][1]) for t in TARGET_XH]) for ms in plot_ms]
    ax.plot(plot_ms, mae_h_avg, 'o-', color='black', linewidth=2,
            label='Average PGA_h', markersize=6)
    ax.plot(plot_ms, mae_v_avg, 'D--', color='black', linewidth=2,
            label='Average PGA_v', markersize=6)
    ax.set_xlabel('Mesh Size (m)', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (%)', fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png3 = os.path.join(OUTPUT_DIR, 'mesh_convergence_MAE.png')
    fig.savefig(png3, dpi=150)
    plt.close(fig)
    print('综合误差图已保存: %s' % png3)

    print('\n全部完成！')


if __name__ == '__main__':
    main()
