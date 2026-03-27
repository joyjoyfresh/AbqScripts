# -*- coding: utf-8 -*-
"""
网格收敛性分析后处理脚本
功能：
  1. 自动扫描指定目录下 fuke-10-* 文件夹，读取 PGA CSV 数据
  2. 在 x/h = 0,1,2,...,8 处提取 PGA_h、PGA_v（取最近点）
  3. 以最小网格尺寸为基准，计算各模型的相对误差
  4. 输出汇总 CSV 表格和误差对比图表
"""

import os
import csv
import re
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120  # 设置交互显示分辨率（即使不弹窗也保持统一）
matplotlib.rcParams['savefig.dpi'] = 300  # 设置导出图片分辨率以满足论文插图质量
matplotlib.rcParams['font.size'] = 11  # 设置全局基础字号提升可读性
matplotlib.rcParams['axes.titlesize'] = 13  # 设置子图标题字号
matplotlib.rcParams['axes.labelsize'] = 12  # 设置坐标轴标签字号
matplotlib.rcParams['legend.fontsize'] = 9  # 设置图例字号
matplotlib.rcParams['xtick.labelsize'] = 10  # 设置横轴刻度字号
matplotlib.rcParams['ytick.labelsize'] = 10  # 设置纵轴刻度字号
matplotlib.rcParams['axes.linewidth'] = 1.0  # 设置坐标轴边框线宽
matplotlib.rcParams['lines.linewidth'] = 1.8  # 设置默认曲线线宽
matplotlib.rcParams['lines.markersize'] = 5  # 设置默认标记点大小


def style_axis(ax):
    """统一坐标轴样式，使图表更符合期刊论文风格"""
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.35)  # 添加主网格并控制透明度
    ax.minorticks_on()  # 开启次刻度增强读图精细度
    ax.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.2)  # 添加次网格用于辅助阅读
    ax.tick_params(axis='both', which='major', direction='in', length=5, width=0.9)  # 主刻度向内显示
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.7)  # 次刻度向内显示
    for spine in ax.spines.values():  # 统一四条边框线条样式
        spine.set_linewidth(1.0)  # 设置边框线宽
        spine.set_alpha(0.9)  # 设置边框透明度

# ============================================================
#  配置
# ============================================================
ROOT_DIR = r'D:\Abaqus'                       # 模型根目录
FOLDER_PATTERN = re.compile(r'^fuke-10-(\d+)$')  # 文件夹名匹配
WAVE_NAMES = [                                  # 3 个地震波对应的 CSV 文件名
    'PGA_job-El_Centro_scaled.csv',
    'PGA_job-Loma_Prieta_scaled.csv',
    'PGA_job-Northridge_scaled.csv',
]
WAVE_SHORT = ['El_Centro', 'Loma_Prieta', 'Northridge']  # 简称
TARGET_XH = [0, 1, 2, 3, 4, 5, 6, 7, 8]            # 目标 x/h 值


# ============================================================
#  工具函数
# ============================================================
def scan_folders(root_dir):
    """扫描 root_dir 下的 fuke-10-* 文件夹，返回 [(mesh_size, folder_path), ...] 按 mesh_size 排序"""
    results = []
    for name in os.listdir(root_dir):
        m = FOLDER_PATTERN.match(name)
        if m:
            mesh_size = int(m.group(1))
            folder_path = os.path.join(root_dir, name)
            if os.path.isdir(folder_path):
                results.append((mesh_size, folder_path))
    results.sort(key=lambda x: x[0])
    return results


def read_csv_data(csv_path):
    """
    读取 PGA CSV 文件，返回 [(x_over_h, pga_h, pga_v), ...] 列表
    """
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # 找到列索引
        col_xh = header.index('x/h')
        col_pga_h = header.index('PGA_h')
        col_pga_v = header.index('PGA_v')
        for row in reader:
            if not row or not row[0].strip():
                continue
            xh = float(row[col_xh])
            pga_h = float(row[col_pga_h])
            pga_v = float(row[col_pga_v])
            data.append((xh, pga_h, pga_v))
    return data


def find_nearest(data, target_xh):
    """
    在 data 中找到 x/h 最接近 target_xh 的点，返回 (actual_xh, pga_h, pga_v)
    """
    best = None
    best_diff = float('inf')
    for xh, pga_h, pga_v in data:
        diff = abs(xh - target_xh)
        if diff < best_diff:
            best_diff = diff
            best = (xh, pga_h, pga_v)
    return best


def extract_target_points(data, targets):
    """
    对 targets 中每个目标 x/h 值，提取最近点
    返回 {target_xh: (actual_xh, pga_h, pga_v), ...}
    """
    result = {}
    for t in targets:
        nearest = find_nearest(data, t)
        if nearest:
            result[t] = nearest
    return result


# ============================================================
#  主流程
# ============================================================
def main():
    # 1. 扫描文件夹
    folders = scan_folders(ROOT_DIR)
    if not folders:
        print('错误: 在 {} 下未找到 fuke-10-* 文件夹'.format(ROOT_DIR))
        sys.exit(1)

    print('找到 {} 个模型文件夹:'.format(len(folders)))
    for ms, fp in folders:
        print('  mesh_size={:>2d}  -> {}'.format(ms, fp))

    # 2. 读取所有数据
    # all_data[mesh_size][wave_index] = {target_xh: (actual_xh, pga_h, pga_v)}
    all_data = {}
    for mesh_size, folder_path in folders:
        all_data[mesh_size] = {}
        for wi, wname in enumerate(WAVE_NAMES):
            csv_path = os.path.join(folder_path, wname)
            if not os.path.isfile(csv_path):
                print('警告: {} 不存在，跳过'.format(csv_path))
                continue
            raw = read_csv_data(csv_path)
            extracted = extract_target_points(raw, TARGET_XH)
            all_data[mesh_size][wi] = extracted
        print('  mesh_size={:>2d}: 成功读取 {} 个地震波'.format(mesh_size, len(all_data[mesh_size])))

    # 3. 确定基准（最小 mesh_size）
    mesh_sizes = sorted(all_data.keys())
    baseline = mesh_sizes[0]
    print('\n基准模型: mesh_size={}'.format(baseline))

    # 4. 计算误差并输出 CSV
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_out = os.path.join(output_dir, 'mesh_convergence_summary.csv')

    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        # 表头
        header_row = ['mesh_size', 'wave', 'x/h_target', 'x/h_actual', 'PGA_h', 'PGA_v',
                       'PGA_h_ref', 'PGA_v_ref', 'error_PGA_h(%)', 'error_PGA_v(%)']
        writer.writerow(header_row)

        for ms in mesh_sizes:
            for wi in range(len(WAVE_NAMES)):
                if wi not in all_data[ms] or wi not in all_data[baseline]:
                    continue
                for t in TARGET_XH:
                    if t not in all_data[ms][wi] or t not in all_data[baseline][wi]:
                        continue
                    xh_act, pga_h, pga_v = all_data[ms][wi][t]
                    _, pga_h_ref, pga_v_ref = all_data[baseline][wi][t]
                    err_h = (pga_h - pga_h_ref) / pga_h_ref * 100 if pga_h_ref != 0 else 0
                    err_v = (pga_v - pga_v_ref) / pga_v_ref * 100 if pga_v_ref != 0 else 0
                    writer.writerow([
                        ms, WAVE_SHORT[wi], t, '{:.6f}'.format(xh_act),
                        '{:.6f}'.format(pga_h), '{:.6f}'.format(pga_v),
                        '{:.6f}'.format(pga_h_ref), '{:.6f}'.format(pga_v_ref),
                        '{:.4f}'.format(err_h), '{:.4f}'.format(err_v),
                    ])

    print('汇总 CSV 已输出: {}'.format(csv_out))

    # 5. 汇总误差数据用于绘图
    # errors[wave_index][target_xh] = { mesh_size: (err_h, err_v) }
    errors = {}
    for wi in range(len(WAVE_NAMES)):
        errors[wi] = {}
        for t in TARGET_XH:
            errors[wi][t] = {}
            for ms in mesh_sizes:
                if wi not in all_data[ms] or wi not in all_data[baseline]: continue
                if t not in all_data[ms][wi] or t not in all_data[baseline][wi]: continue
                _, pga_h, pga_v = all_data[ms][wi][t]
                _, pga_h_ref, pga_v_ref = all_data[baseline][wi][t]
                err_h = (pga_h - pga_h_ref) / pga_h_ref * 100 if pga_h_ref != 0 else 0
                err_v = (pga_v - pga_v_ref) / pga_v_ref * 100 if pga_v_ref != 0 else 0
                errors[wi][t][ms] = (err_h, err_v)

    # 三波平均误差
    avg_errors = {}  # avg_errors[target_xh] = { mesh_size: (avg_err_h, avg_err_v) }
    for t in TARGET_XH:
        avg_errors[t] = {}
        for ms in mesh_sizes:
            err_h_list = []
            err_v_list = []
            for wi in range(len(WAVE_NAMES)):
                if t in errors[wi] and ms in errors[wi][t]:
                    eh, ev = errors[wi][t][ms]
                    err_h_list.append(eh)
                    err_v_list.append(ev)
            if err_h_list:
                avg_errors[t][ms] = (sum(err_h_list) / len(err_h_list),
                                     sum(err_v_list) / len(err_v_list))

    # 6. 绘图
    plt.ioff()  # 关闭交互模式，确保脚本运行时不弹出绘图窗口
    colors = plt.cm.tab10(np.linspace(0, 1, len(TARGET_XH)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    # --- 图1: 各地震波 PGA_h 误差 ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('PGA_h Relative Error vs Mesh Size', fontsize=16, fontweight='bold')
    for wi in range(len(WAVE_NAMES)):
        ax = axes1[wi // 2][wi % 2]
        for ti, t in enumerate(TARGET_XH):
            if t in errors[wi]:
                ms_list = sorted(errors[wi][t].keys())
                err_list = [errors[wi][t][ms][0] for ms in ms_list]
                ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                        label='x/h={}'.format(t), linewidth=1.5, markersize=5)
        ax.set_title(WAVE_SHORT[wi], fontsize=13)
        ax.set_xlabel('Mesh Size (m)')
        ax.set_ylabel('Relative Error (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
        style_axis(ax)  # 应用统一坐标轴样式
    # 第4个子图: 三波平均
    ax = axes1[1][1]
    for ti, t in enumerate(TARGET_XH):
        if t in avg_errors:
            ms_list = sorted(avg_errors[t].keys())
            err_list = [avg_errors[t][ms][0] for ms in ms_list]
            ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                    label='x/h={}'.format(t), linewidth=1.5, markersize=5)
    ax.set_title('Average (3 waves)', fontsize=13)
    ax.set_xlabel('Mesh Size (m)')
    ax.set_ylabel('Relative Error (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
    style_axis(ax)  # 应用统一坐标轴样式
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_path = os.path.join(output_dir, 'mesh_convergence_PGA_h.png')
    fig1.savefig(fig1_path, dpi=200, bbox_inches='tight')
    print('PGA_h 误差图已保存: {}'.format(fig1_path))

    # --- 图2: 各地震波 PGA_v 误差 ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('PGA_v Relative Error vs Mesh Size', fontsize=16, fontweight='bold')
    for wi in range(len(WAVE_NAMES)):
        ax = axes2[wi // 2][wi % 2]
        for ti, t in enumerate(TARGET_XH):
            if t in errors[wi]:
                ms_list = sorted(errors[wi][t].keys())
                err_list = [errors[wi][t][ms][1] for ms in ms_list]
                ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                        label='x/h={}'.format(t), linewidth=1.5, markersize=5)
        ax.set_title(WAVE_SHORT[wi], fontsize=13)
        ax.set_xlabel('Mesh Size (m)')
        ax.set_ylabel('Relative Error (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
        style_axis(ax)  # 应用统一坐标轴样式
    ax = axes2[1][1]
    for ti, t in enumerate(TARGET_XH):
        if t in avg_errors:
            ms_list = sorted(avg_errors[t].keys())
            err_list = [avg_errors[t][ms][1] for ms in ms_list]
            ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                    label='x/h={}'.format(t), linewidth=1.5, markersize=5)
    ax.set_title('Average (3 waves)', fontsize=13)
    ax.set_xlabel('Mesh Size (m)')
    ax.set_ylabel('Relative Error (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
    style_axis(ax)  # 应用统一坐标轴样式
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(output_dir, 'mesh_convergence_PGA_v.png')
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    print('PGA_v 误差图已保存: {}'.format(fig2_path))

    # --- 图3: 综合平均绝对误差 ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.suptitle('Mean Absolute Error vs Mesh Size (All x/h Points Average)',
                  fontsize=14, fontweight='bold')
    # 各地震波 + 平均
    for wi in range(len(WAVE_NAMES)):
        mae_h = []
        mae_v = []
        ms_plot = []
        for ms in mesh_sizes:
            abs_err_h = []
            abs_err_v = []
            for t in TARGET_XH:
                if t in errors[wi] and ms in errors[wi][t]:
                    eh, ev = errors[wi][t][ms]
                    abs_err_h.append(abs(eh))
                    abs_err_v.append(abs(ev))
            if abs_err_h:
                ms_plot.append(ms)
                mae_h.append(sum(abs_err_h) / len(abs_err_h))
                mae_v.append(sum(abs_err_v) / len(abs_err_v))
        ax3.plot(ms_plot, mae_h, marker='o', linestyle='-',
                 label='{} PGA_h'.format(WAVE_SHORT[wi]), linewidth=1.5, markersize=5)
        ax3.plot(ms_plot, mae_v, marker='s', linestyle='--',
                 label='{} PGA_v'.format(WAVE_SHORT[wi]), linewidth=1.5, markersize=5)

    # 三波总平均
    mae_h_avg = []
    mae_v_avg = []
    ms_plot_avg = []
    for ms in mesh_sizes:
        abs_err_h_all = []
        abs_err_v_all = []
        for t in TARGET_XH:
            if t in avg_errors and ms in avg_errors[t]:
                eh, ev = avg_errors[t][ms]
                abs_err_h_all.append(abs(eh))
                abs_err_v_all.append(abs(ev))
        if abs_err_h_all:
            ms_plot_avg.append(ms)
            mae_h_avg.append(sum(abs_err_h_all) / len(abs_err_h_all))
            mae_v_avg.append(sum(abs_err_v_all) / len(abs_err_v_all))
    ax3.plot(ms_plot_avg, mae_h_avg, marker='D', linestyle='-', color='black',
             linewidth=2.5, markersize=7, label='Avg PGA_h')
    ax3.plot(ms_plot_avg, mae_v_avg, marker='D', linestyle='--', color='black',
             linewidth=2.5, markersize=7, label='Avg PGA_v')

    ax3.set_xlabel('Mesh Size (m)', fontsize=12)
    ax3.set_ylabel('Mean Absolute Error (%)', fontsize=12)
    ax3.axhline(y=5, color='red', linestyle=':', linewidth=1.2, label='5% threshold')
    ax3.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
    style_axis(ax3)  # 应用统一坐标轴样式
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3_path = os.path.join(output_dir, 'mesh_convergence_MAE.png')
    fig3.savefig(fig3_path, dpi=200, bbox_inches='tight')
    print('综合误差图已保存: {}'.format(fig3_path))

    plt.close('all')  # 关闭所有图窗对象，避免任何图形界面弹出
    print('\n完成！')


if __name__ == '__main__':
    main()
