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

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']  # 添加宋体
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['figure.dpi'] = 120  # 设置交互显示分辨率（即使不弹窗也保持统一）
matplotlib.rcParams['savefig.dpi'] = 300  # 设置导出图片分辨率以满足论文插图质量
matplotlib.rcParams['font.size'] = 12  # 稍微增大基础字号
matplotlib.rcParams['axes.titlesize'] = 14  # 设置子图标题字号
matplotlib.rcParams['axes.labelsize'] = 12  # 设置坐标轴标签字号
matplotlib.rcParams['legend.fontsize'] = 10  # 设置图例字号
matplotlib.rcParams['xtick.labelsize'] = 11  # 设置横轴刻度字号
matplotlib.rcParams['ytick.labelsize'] = 11  # 设置纵轴刻度字号
matplotlib.rcParams['axes.linewidth'] = 1.2  # 增加坐标轴边框线宽
matplotlib.rcParams['lines.linewidth'] = 2.0  # 增加默认曲线线宽
matplotlib.rcParams['lines.markersize'] = 6  # 增加默认标记点大小


def style_axis(ax):
    """统一坐标轴样式，使图表更符合中文期刊论文风格"""
    ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.4)  # 添加主网格并控制透明度
    ax.minorticks_on()  # 开启次刻度增强读图精细度
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.25)  # 添加次网格用于辅助阅读
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.0)  # 主刻度向内显示，增加长度和宽度
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=0.7)  # 次刻度向内显示
    for spine in ax.spines.values():  # 统一四条边框线条样式
        spine.set_linewidth(1.2)  # 增加边框线宽
        spine.set_alpha(1.0)  # 设置边框完全不透明

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
LOG_FILENAME = 'VAB_oblique_noGUI_v13.log'         # 运行日志文件名


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


def read_last_nonempty_line(file_path):
    """读取文本文件最后一个非空行，若不存在则返回空字符串"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  # 以 UTF-8 打开日志并忽略异常字符
        lines = f.readlines()  # 读取全部行用于倒序查找
    for line in reversed(lines):  # 从后向前查找最后一个非空行
        stripped = line.strip()  # 去除首尾空白字符
        if stripped:  # 若当前行为非空
            return stripped  # 返回最后一个非空行内容
    return ''  # 若文件全为空行则返回空字符串


def parse_runtime_to_seconds(text):
    """从文本中解析运行时间并换算为秒，无法解析时返回 None"""
    total_match = re.search(r'总耗时\s*=\s*(\d+(?:\.\d+)?)\s*s\b', text, re.IGNORECASE)  # 优先匹配“总耗时=13314.86s”格式
    if total_match:  # 若匹配到总耗时字段
        return float(total_match.group(1))  # 直接返回总耗时秒数

    bracket_match = re.search(r'\[(\d+(?:\.\d+)?)\s*s\]', text, re.IGNORECASE)  # 匹配“[13314.857s]”格式
    if bracket_match:  # 若匹配到方括号秒数字段
        return float(bracket_match.group(1))  # 返回方括号中的秒数

    number_unit_pairs = re.findall(r'(\d+(?:\.\d+)?)\s*(h|hr|hour|hours|min|m|sec|s|ms|小时|分钟|分|秒|毫秒)', text.lower())  # 匹配数值+单位
    if number_unit_pairs:  # 若匹配到带单位的时间片段
        total_seconds = 0.0  # 初始化秒数累加器
        for value_str, unit in number_unit_pairs:  # 逐个处理匹配结果
            value = float(value_str)  # 转换当前数值为浮点数
            if unit in ('h', 'hr', 'hour', 'hours', '小时'):  # 小时单位
                total_seconds += value * 3600.0  # 小时换算为秒
            elif unit in ('min', 'm', '分钟', '分'):  # 分钟单位
                total_seconds += value * 60.0  # 分钟换算为秒
            elif unit in ('sec', 's', '秒'):  # 秒单位
                total_seconds += value  # 秒直接累加
            elif unit in ('ms', '毫秒'):  # 毫秒单位
                total_seconds += value / 1000.0  # 毫秒换算为秒
        return total_seconds  # 返回累计秒数

    hms_match = re.search(r'\b(\d+):(\d+):(\d+(?:\.\d+)?)\b', text)  # 兜底匹配 HH:MM:SS(.s) 格式，避免优先误取时间戳
    if hms_match:  # 若匹配到时分秒格式
        hours = float(hms_match.group(1))  # 读取小时数
        minutes = float(hms_match.group(2))  # 读取分钟数
        seconds = float(hms_match.group(3))  # 读取秒数
        return hours * 3600.0 + minutes * 60.0 + seconds  # 转换为总秒数

    fallback_number = re.search(r'(\d+(?:\.\d+)?)', text)  # 兜底匹配任意第一个数字
    if fallback_number:  # 若存在数字
        return float(fallback_number.group(1))  # 默认按秒处理返回
    return None  # 无法解析时返回 None


def collect_runtime_by_mesh(folders):
    """收集各 mesh_size 对应日志最后一行总耗时（秒）"""
    runtime_by_mesh = {}  # 初始化运行时间字典
    for mesh_size, folder_path in folders:  # 遍历所有模型目录
        log_path = os.path.join(folder_path, LOG_FILENAME)  # 拼接日志文件路径
        if not os.path.isfile(log_path):  # 若日志文件不存在
            print('警告: {} 不存在，跳过耗时统计'.format(log_path))  # 输出缺失日志警告
            continue  # 跳过当前模型
        last_line = read_last_nonempty_line(log_path)  # 读取最后一个非空行
        if not last_line:  # 若最后行为空
            print('警告: {} 为空，跳过耗时统计'.format(log_path))  # 输出空日志警告
            continue  # 跳过当前模型
        runtime_seconds = parse_runtime_to_seconds(last_line)  # 解析总耗时秒数
        if runtime_seconds is None:  # 若解析失败
            print('警告: {} 最后一行无法解析耗时: {}'.format(log_path, last_line))  # 输出解析失败警告
            continue  # 跳过当前模型
        runtime_by_mesh[mesh_size] = runtime_seconds  # 保存当前模型耗时
    return runtime_by_mesh  # 返回耗时字典


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

    # 1.1 读取日志耗时数据
    runtime_by_mesh = collect_runtime_by_mesh(folders)
    if runtime_by_mesh:
        print('已读取 {} 个模型的日志耗时数据'.format(len(runtime_by_mesh)))
    else:
        print('警告: 未读取到任何有效日志耗时数据，将跳过耗时对比图')

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
    fig1.suptitle('PGA_h 相对误差与网格尺寸的关系', fontsize=16, fontweight='bold')
    for wi in range(len(WAVE_NAMES)):
        ax = axes1[wi // 2][wi % 2]
        for ti, t in enumerate(TARGET_XH):
            if t in errors[wi]:
                ms_list = sorted(errors[wi][t].keys())
                err_list = [errors[wi][t][ms][0] for ms in ms_list]
                ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                        label='x/h={}'.format(t), linewidth=1.5, markersize=5)
        ax.set_title(WAVE_SHORT[wi], fontsize=13)
        ax.set_xlabel('网格尺寸 (m)')
        ax.set_ylabel('相对误差 (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(mesh_sizes)
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
    ax.set_title('三波平均', fontsize=13)
    ax.set_xlabel('网格尺寸 (m)')
    ax.set_ylabel('相对误差 (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(mesh_sizes)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
    style_axis(ax)  # 应用统一坐标轴样式
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    fig1_path = os.path.join(output_dir, 'mesh_convergence_PGA_h.png')
    fig1.savefig(fig1_path, dpi=200, bbox_inches='tight')
    print('PGA_h 误差图已保存: {}'.format(fig1_path))

    # --- 图2: 各地震波 PGA_v 误差 ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('PGA_v 相对误差与网格尺寸的关系', fontsize=16, fontweight='bold')
    for wi in range(len(WAVE_NAMES)):
        ax = axes2[wi // 2][wi % 2]
        for ti, t in enumerate(TARGET_XH):
            if t in errors[wi]:
                ms_list = sorted(errors[wi][t].keys())
                err_list = [errors[wi][t][ms][1] for ms in ms_list]
                ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                        label='x/h={}'.format(t), linewidth=1.5, markersize=5)
        ax.set_title(WAVE_SHORT[wi], fontsize=13)
        ax.set_xlabel('网格尺寸 (m)')
        ax.set_ylabel('相对误差 (%)')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(mesh_sizes)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
        style_axis(ax)  # 应用统一坐标轴样式
    ax = axes2[1][1]
    for ti, t in enumerate(TARGET_XH):
        if t in avg_errors:
            ms_list = sorted(avg_errors[t].keys())
            err_list = [avg_errors[t][ms][1] for ms in ms_list]
            ax.plot(ms_list, err_list, marker=markers[ti % len(markers)], color=colors[ti],
                    label='x/h={}'.format(t), linewidth=1.5, markersize=5)
    ax.set_title('三波平均', fontsize=13)
    ax.set_xlabel('网格尺寸 (m)')
    ax.set_ylabel('相对误差 (%)')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(mesh_sizes)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
    style_axis(ax)  # 应用统一坐标轴样式
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(output_dir, 'mesh_convergence_PGA_v.png')
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    print('PGA_v 误差图已保存: {}'.format(fig2_path))

    # --- 图3: 综合平均绝对误差 ---
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    fig3.suptitle('平均绝对误差与网格尺寸的关系（所有x/h点平均）',
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
             linewidth=2.5, markersize=7, label='平均 PGA_h')
    ax3.plot(ms_plot_avg, mae_v_avg, marker='D', linestyle='--', color='black',
             linewidth=2.5, markersize=7, label='平均 PGA_v')

    ax3.set_xlabel('网格尺寸 (m)', fontsize=12)
    ax3.set_ylabel('平均绝对误差 (%)', fontsize=12)
    ax3.axhline(y=5, color='red', linestyle=':', linewidth=1.2, label='5% 阈值')
    ax3.set_xticks(mesh_sizes)
    ax3.legend(frameon=True, fancybox=False, edgecolor='black', ncol=2)  # 使用方角图例边框增强学术风格
    style_axis(ax3)  # 应用统一坐标轴样式
    fig3.tight_layout(rect=[0, 0, 1, 0.95])
    fig3_path = os.path.join(output_dir, 'mesh_convergence_MAE.png')
    fig3.savefig(fig3_path, dpi=200, bbox_inches='tight')
    print('综合误差图已保存: {}'.format(fig3_path))

    # --- 图4: 各模型脚本总运行时间对比 ---
    if runtime_by_mesh:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        fig4.suptitle('各网格精度模型脚本总运行时间对比', fontsize=14, fontweight='bold')
        ms_runtime = sorted(runtime_by_mesh.keys())
        rt_runtime = [runtime_by_mesh[ms] for ms in ms_runtime]
        bars = ax4.bar(ms_runtime, rt_runtime, color='#4C72B0', edgecolor='black', linewidth=0.9)
        ax4.set_xlabel('网格尺寸 (m)', fontsize=12)
        ax4.set_ylabel('脚本总运行时间 (s)', fontsize=12)
        ax4.set_xticks(ms_runtime)
        style_axis(ax4)
        for bar, value in zip(bars, rt_runtime):
            ax4.text(bar.get_x() + bar.get_width() / 2.0, value, '{:.2f}'.format(value),
                     ha='center', va='bottom', fontsize=10)
        fig4.tight_layout(rect=[0, 0, 1, 0.95])
        fig4_path = os.path.join(output_dir, 'mesh_convergence_runtime.png')
        fig4.savefig(fig4_path, dpi=200, bbox_inches='tight')
        print('运行时间对比图已保存: {}'.format(fig4_path))

    plt.close('all')  # 关闭所有图窗对象，避免任何图形界面弹出
    print('\n完成！')


if __name__ == '__main__':
    main()
