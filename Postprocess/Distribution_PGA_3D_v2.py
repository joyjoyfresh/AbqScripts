# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
import os  # 导入 os 模块用于路径与目录操作
import re  # 导入 re 模块用于正则解析文件夹名
import sys  # 导入 sys 模块用于读取命令行参数
import math  # 导入 math 模块用于数值刻度计算
import numpy as np  # 导入 numpy 用于数值计算与数组处理
import pandas as pd  # 导入 pandas 用于表格数据读取与汇总
import matplotlib  # 导入 matplotlib 主模块用于后端设置
matplotlib.use('Agg')  # 使用无界面后端确保脚本在批处理环境可运行
import matplotlib.pyplot as plt  # 导入 pyplot 用于绘图
from matplotlib.lines import Line2D  # 导入 Line2D 用于自定义图例句柄
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 导入 3D 坐标轴以启用 projection='3d'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 记录当前脚本所在目录
DEFAULT_BATCH_ROOT = SCRIPT_DIR  # 设置默认批处理数据根目录
OUTPUT_FIGURE_NAME = 'PGA_h_max_3D_grid.png'  # 设置输出总图文件名
OUTPUT_SINGLE_PREFIX = 'PGA_h_max_3D'  # 设置单图文件名前缀
OUTPUT_SUMMARY_NAME = 'PGA_h_max_summary.csv'  # 设置输出汇总表文件名
PGA_COLUMN = 'PGA_h'  # 设置用于统计的目标分量列
FOLDER_PATTERN = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_angle(?P<angle>-?\d+(?:\.\d+)?)')  # 定义参数文件夹名解析正则
TARGET_ANGLES = [0.0, 30.0]  # 定义目标绘图入射角顺序
ANGLE_TOLERANCE = 1e-6  # 定义角度分组时的容差
FACE_COLOR = "#ffffff"  # 定义图像背景色
SURFACE_ALPHA = 0.72  # 定义曲面透明度
SURFACE_CMAP = 'bwr'  # 定义曲面配色映射
MARKER_COLOR = '#ff1a1a'  # 定义观测点标记颜色
MARKER_EDGE_COLOR = '#000000'  # 定义观测点边线颜色
FIXED_X_TICKS = [0.0, 200.0, 400.0]  # 定义固定 X 轴刻度列表
FIXED_Y_TICKS = [15.0, 45.0, 75.0]  # 定义固定 Y 轴刻度列表
FIXED_Z_TICKS = [0.5, 1.0, 1.5]  # 定义固定 Z 轴刻度列表
MOTION_CONFIGS = [  # 定义三种地震动配置
    {'key': 'el_centro', 'token': 'elcentro', 'display': 'El Centro', 'marker': 'o'},  # 定义 El Centro 配置
    {'key': 'loma_prieta', 'token': 'lomaprieta', 'display': 'Loma Prieta', 'marker': '^'},  # 定义 Loma Prieta 配置
    {'key': 'northridge', 'token': 'northridge', 'display': 'Northridge', 'marker': 's'},  # 定义 Northridge 配置
]  # 结束地震动配置定义


def format_number_text(value):  # 定义数值显示格式函数
    return str(int(value)) if float(value).is_integer() else f'{value:g}'  # 将整数显示为无小数文本并压缩浮点尾零


def canonical_text(text):  # 定义文本规范化函数
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())  # 仅保留小写字母与数字便于关键词匹配


def parse_case_folder(folder_name):  # 定义参数文件夹名解析函数
    match = FOLDER_PATTERN.search(folder_name)  # 在文件夹名中匹配 h、i、angle 参数
    if match is None:  # 判断匹配是否失败
        return None  # 返回空值表示当前目录不是目标参数目录
    h_value = float(match.group('h'))  # 提取并转换 h 参数
    i_value = float(match.group('i'))  # 提取并转换 i 参数
    angle_value = float(match.group('angle'))  # 提取并转换 angle 参数
    return h_value, i_value, angle_value  # 返回解析得到的参数元组


def resolve_angle_group(angle_value):  # 定义将角度映射到目标角分组的函数
    for target_angle in TARGET_ANGLES:  # 遍历目标角列表
        if abs(angle_value - target_angle) <= ANGLE_TOLERANCE:  # 判断当前角是否落入目标角容差范围
            return target_angle  # 返回命中的目标角分组
    return None  # 未命中任何目标角时返回空值


def is_flat_file(stem_text):  # 定义判断文件是否为 flat 组的函数
    return re.search(r'(^|[-_])flat($|[-_])', stem_text.lower()) is not None  # 使用连字符边界匹配 flat 标记


def detect_motion(stem_text):  # 定义按文件名识别地震动类型的函数
    normalized = canonical_text(stem_text)  # 先将文件名规范化为连续小写字母数字
    for motion_cfg in MOTION_CONFIGS:  # 遍历预定义地震动配置
        if motion_cfg['token'] in normalized:  # 判断当前文件名是否包含地震动关键词
            return motion_cfg  # 返回识别到的地震动配置
    return None  # 未识别到目标地震动时返回空值


def collect_records(batch_root):  # 定义扫描批处理目录并提取记录的函数
    records = []  # 初始化记录列表
    if not os.path.isdir(batch_root):  # 判断批处理目录是否存在
        raise NotADirectoryError(f'批处理目录不存在: {batch_root}')  # 抛出目录不存在异常
    for entry_name in sorted(os.listdir(batch_root)):  # 遍历批处理根目录下全部条目
        folder_path = os.path.join(batch_root, entry_name)  # 组装当前条目的完整路径
        if not os.path.isdir(folder_path):  # 判断当前条目是否为目录
            continue  # 跳过非目录条目
        parsed = parse_case_folder(entry_name)  # 解析当前目录名中的参数
        if parsed is None:  # 判断目录名是否符合参数命名规则
            continue  # 跳过非目标参数目录
        h_value, i_value, angle_value = parsed  # 解包目录参数
        angle_group = resolve_angle_group(angle_value)  # 将 angle 映射到目标角分组
        if angle_group is None:  # 判断是否属于目标角集合
            continue  # 跳过非目标角目录
        for file_name in sorted(os.listdir(folder_path)):  # 遍历参数目录中的文件
            if not file_name.lower().endswith('.csv'):  # 判断当前文件是否为 CSV
                continue  # 跳过非 CSV 文件
            if not file_name.startswith('PGA-'):  # 判断文件名是否符合 PGA 输出前缀
                continue  # 跳过非 PGA 文件
            csv_stem = os.path.splitext(file_name)[0]  # 取不含扩展名的主文件名
            if '-normalized' in csv_stem.lower():  # 判断是否为归一化中间文件
                continue  # 跳过中间文件避免重复统计
            if is_flat_file(csv_stem):  # 判断是否为 flat 基准文件
                continue  # 跳过 flat 文件仅保留坡地结果
            motion_cfg = detect_motion(csv_stem)  # 根据文件名识别地震动类型
            if motion_cfg is None:  # 判断文件是否属于目标三种地震动
                continue  # 跳过未识别地震动的文件
            csv_path = os.path.join(folder_path, file_name)  # 组装当前 CSV 的完整路径
            try:  # 尝试读取并统计当前 CSV
                df = pd.read_csv(csv_path)  # 读取 CSV 数据表
            except Exception as exc:  # 捕获读取异常
                print(f'警告：读取失败，已跳过 {csv_path} -> {exc}')  # 输出读取失败提示
                continue  # 跳过读取失败文件
            if PGA_COLUMN not in df.columns:  # 判断目标分量列是否存在
                print(f'警告：缺少列 {PGA_COLUMN}，已跳过 {csv_path}')  # 输出缺列提示
                continue  # 跳过列不完整文件
            pga_series = pd.to_numeric(df[PGA_COLUMN], errors='coerce')  # 将目标列安全转换为数值类型
            pga_series = pga_series[np.isfinite(pga_series)]  # 过滤非有限值避免统计污染
            if pga_series.empty:  # 判断过滤后是否还有有效数值
                print(f'警告：无有效 {PGA_COLUMN} 数据，已跳过 {csv_path}')  # 输出无有效数据提示
                continue  # 跳过空有效数据文件
            pga_max = float(pga_series.max())  # 计算当前工况的 PGA_h 最大值
            records.append({'motion': motion_cfg['key'], 'motion_display': motion_cfg['display'], 'marker': motion_cfg['marker'], 'h': h_value, 'i': i_value, 'angle': angle_group, 'pga_max': pga_max, 'folder': folder_path, 'file': file_name})  # 追加结构化记录
    if not records:  # 判断是否未收集到任何有效记录
        raise FileNotFoundError('未找到有效 PGA 记录，请先运行 Batch/Autorun_TAF_v2.py 并确认每个算例目录已输出 PGA-*.csv。')  # 抛出无数据异常
    return records  # 返回记录列表


def build_summary_dataframe(records):  # 定义构建汇总表的函数
    df = pd.DataFrame(records)  # 将记录列表转换为数据表
    grouped = df.groupby(['motion', 'motion_display', 'marker', 'h', 'i', 'angle'], as_index=False)['pga_max'].mean()  # 对重复工况取均值以去重
    grouped = grouped.sort_values(by=['motion', 'angle', 'i', 'h']).reset_index(drop=True)  # 按地震动与参数排序便于检查
    return grouped  # 返回汇总后的数据表


def choose_ticks(values, max_tick_count):  # 定义自适应刻度选择函数
    unique_vals = sorted({float(v) for v in values})  # 获取去重升序后的浮点值
    if not unique_vals:  # 判断是否为空值列表
        return []  # 返回空刻度
    if len(unique_vals) <= max_tick_count:  # 判断唯一值数量是否不超过上限
        return unique_vals  # 直接返回全部唯一值作为刻度
    raw_indices = np.linspace(0, len(unique_vals) - 1, max_tick_count)  # 在索引空间均匀取样
    index_list = sorted({int(round(idx)) for idx in raw_indices})  # 将浮点索引映射为整数并去重排序
    return [unique_vals[idx] for idx in index_list]  # 返回抽样后的刻度值列表


def compute_row_z_limits(summary_df, motion_key):  # 定义按地震动计算行内统一 Z 轴范围的函数
    row_values = summary_df.loc[summary_df['motion'] == motion_key, 'pga_max'].to_numpy(dtype=float)  # 提取当前地震动行全部 Z 值
    row_values = row_values[np.isfinite(row_values)]  # 过滤非有限值
    if row_values.size == 0:  # 判断是否没有有效 Z 值
        return 0.0, 1.0  # 回退默认范围
    z_min = float(np.min(row_values))  # 计算最小值
    z_max = float(np.max(row_values))  # 计算最大值
    if abs(z_max - z_min) < 1e-9:  # 判断范围是否过窄
        z_min = max(0.0, z_min - 0.1)  # 在近似常值时向下扩展范围
        z_max = z_max + 0.1  # 在近似常值时向上扩展范围
    else:  # 当前范围正常时执行比例扩展
        span = z_max - z_min  # 计算范围跨度
        z_min = max(0.0, z_min - 0.12 * span)  # 将下界向下放宽并限制为非负
        z_max = z_max + 0.20 * span  # 将上界向上放宽留出可视空间
    return z_min, z_max  # 返回当前行统一 Z 范围


def style_3d_axes(ax, angle_value, h_values, i_values, z_limits):  # 定义 3D 坐标轴样式设置函数
    ax.set_title(r'$\theta_s = ' + format_number_text(angle_value) + r'^\circ$', fontsize=11, y=0.98)  # 设置子图标题为入射角
    ax.set_xlabel(r'$h\,(m)$', labelpad=2, fontsize=10)  # 设置 X 轴标签
    ax.set_ylabel(r'$i\,(^\circ)$', labelpad=2, fontsize=10)  # 设置 Y 轴标签
    ax.set_zlabel('PGA_h,max (g)', labelpad=4, fontsize=10)  # 设置 Z 轴标签为普通文本避免公式渲染顺序异常
    ax.set_xlim(FIXED_X_TICKS[0], FIXED_X_TICKS[-1])  # 按要求固定 X 轴范围
    ax.set_ylim(FIXED_Y_TICKS[-1], FIXED_Y_TICKS[0])  # 按要求固定 Y 轴反向范围
    ax.set_zlim(FIXED_Z_TICKS[0], FIXED_Z_TICKS[-1])  # 按要求固定 Z 轴范围
    ax.set_xticks(FIXED_X_TICKS)  # 按要求写入固定 X 轴刻度
    ax.set_yticks(FIXED_Y_TICKS)  # 按要求写入固定 Y 轴刻度
    ax.set_zticks(FIXED_Z_TICKS)  # 按要求写入固定 Z 轴刻度
    ax.set_xticklabels([format_number_text(v) for v in FIXED_X_TICKS], fontsize=9)  # 设置固定 X 轴刻度文本
    ax.set_yticklabels([format_number_text(v) for v in FIXED_Y_TICKS], fontsize=9)  # 设置固定 Y 轴刻度文本
    ax.set_zticklabels([f'{v:.1f}' for v in FIXED_Z_TICKS], fontsize=9)  # 设置固定 Z 轴刻度文本
    ax.view_init(elev=23, azim=-126)  # 设置 3D 视角与示例图相近
    ax.set_box_aspect((1.45, 1.05, 0.38))  # 设置 3D 盒体宽高比提升横向展开感
    ax.grid(True, linestyle='-', linewidth=0.5, color='#bcbcbc', alpha=0.65)  # 设置网格样式
    ax.set_facecolor(FACE_COLOR)  # 设置子图背景色


def draw_panel(ax, panel_df, marker_symbol):  # 定义单个子图绘制函数
    if panel_df.empty:  # 判断当前子图是否没有数据
        return None  # 返回空值表示无需绘制曲面
    x_arr = panel_df['h'].to_numpy(dtype=float)  # 提取 h 数组
    y_arr = panel_df['i'].to_numpy(dtype=float)  # 提取 i 数组
    z_arr = panel_df['pga_max'].to_numpy(dtype=float)  # 提取 PGA_h_max 数组
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr) & np.isfinite(z_arr)  # 构造三列全有限值掩码
    x_arr = x_arr[finite_mask]  # 过滤后的 h 数组
    y_arr = y_arr[finite_mask]  # 过滤后的 i 数组
    z_arr = z_arr[finite_mask]  # 过滤后的 PGA_h_max 数组
    if x_arr.size == 0:  # 判断过滤后是否为空
        return None  # 返回空值表示无可绘制点
    if x_arr.size >= 3:  # 判断点数是否足以构建三角曲面
        ax.plot_trisurf(x_arr, y_arr, z_arr, cmap=SURFACE_CMAP, linewidth=0.7, edgecolor='k', alpha=SURFACE_ALPHA, antialiased=True)  # 绘制带网格边线的三角曲面
    ax.scatter(x_arr, y_arr, z_arr, c=MARKER_COLOR, marker=marker_symbol, s=24, edgecolors=MARKER_EDGE_COLOR, linewidths=0.35, depthshade=False)  # 叠加红色观测点标记
    return (x_arr, y_arr, z_arr)  # 返回用于设置坐标轴的点数组


def create_figure(summary_df, output_path):  # 定义总图绘制函数
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
    fig = plt.figure(figsize=(10.5, 9.8), facecolor=FACE_COLOR)  # 创建总图画布
    row_count = len(MOTION_CONFIGS)  # 计算行数
    col_count = len(TARGET_ANGLES)  # 计算列数
    for row_index, motion_cfg in enumerate(MOTION_CONFIGS):  # 按地震动顺序遍历每一行
        z_limits = compute_row_z_limits(summary_df, motion_cfg['key'])  # 计算当前行统一 Z 轴范围
        for col_index, angle_value in enumerate(TARGET_ANGLES):  # 按目标入射角顺序遍历每一列
            subplot_index = row_index * col_count + col_index + 1  # 计算子图序号
            ax = fig.add_subplot(row_count, col_count, subplot_index, projection='3d')  # 创建当前 3D 子图
            panel_df = summary_df[(summary_df['motion'] == motion_cfg['key']) & (summary_df['angle'] == angle_value)]  # 提取当前子图数据
            panel_points = draw_panel(ax, panel_df, motion_cfg['marker'])  # 绘制当前子图曲面与观测点
            if panel_points is None:  # 判断当前子图是否无有效点
                style_3d_axes(ax, angle_value, [0.0, 1.0], [0.0, 1.0], z_limits)  # 使用兜底坐标样式初始化空子图
                ax.text2D(0.35, 0.50, 'No Data', transform=ax.transAxes, fontsize=11, color='#444444')  # 在空子图中央标记无数据
                continue  # 跳过后续样式设置
            x_arr, y_arr, _ = panel_points  # 解包当前子图有效点坐标
            style_3d_axes(ax, angle_value, x_arr, y_arr, z_limits)  # 应用当前子图坐标轴样式
    legend_handles = [Line2D([0], [0], marker=cfg['marker'], linestyle='None', color='none', markerfacecolor=MARKER_COLOR, markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=0.35, markersize=5.0, label=cfg['display']) for cfg in MOTION_CONFIGS]  # 构造底部图例句柄
    fig.legend(handles=legend_handles, loc='lower center', ncol=3, frameon=True, edgecolor='#666666', framealpha=1.0, fontsize=11, bbox_to_anchor=(0.5, 0.035), columnspacing=1.2, handletextpad=0.4)  # 绘制底部共享图例
    plt.subplots_adjust(left=0.02, right=0.99, top=0.985, bottom=0.10, wspace=0.02, hspace=0.02)  # 调整子图边距贴近示例布局
    fig.savefig(output_path, dpi=350)  # 保存总图到目标路径
    plt.close(fig)  # 关闭图对象释放内存


def build_single_figure_name(motion_key, angle_value):  # 定义单图文件名构造函数
    angle_text = format_number_text(angle_value)  # 将角度格式化为紧凑文本
    return '{}-{}-angle{}.png'.format(OUTPUT_SINGLE_PREFIX, motion_key, angle_text)  # 返回单图文件名


def create_single_figures(summary_df, output_dir):  # 定义逐子图单独导出函数
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学公式字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
    output_paths = []  # 初始化单图输出路径列表
    for motion_cfg in MOTION_CONFIGS:  # 按地震动顺序遍历每一行
        z_limits = compute_row_z_limits(summary_df, motion_cfg['key'])  # 计算当前地震动统一 Z 轴范围
        for angle_value in TARGET_ANGLES:  # 按目标入射角顺序遍历每一列
            fig = plt.figure(figsize=(6.2, 4.5), facecolor=FACE_COLOR)  # 创建单图画布
            ax = fig.add_subplot(1, 1, 1, projection='3d')  # 创建单个 3D 子图
            panel_df = summary_df[(summary_df['motion'] == motion_cfg['key']) & (summary_df['angle'] == angle_value)]  # 提取当前子图数据
            panel_points = draw_panel(ax, panel_df, motion_cfg['marker'])  # 绘制当前子图曲面与观测点
            if panel_points is None:  # 判断当前子图是否无有效点
                style_3d_axes(ax, angle_value, [0.0, 1.0], [0.0, 1.0], z_limits)  # 使用兜底坐标样式初始化空子图
                ax.text2D(0.38, 0.52, 'No Data', transform=ax.transAxes, fontsize=11, color='#444444')  # 在空子图中央标记无数据
            else:  # 当前子图存在有效点时继续设置样式
                x_arr, y_arr, _ = panel_points  # 解包当前子图有效点坐标
                style_3d_axes(ax, angle_value, x_arr, y_arr, z_limits)  # 应用当前子图坐标轴样式
            legend_handle = Line2D([0], [0], marker=motion_cfg['marker'], linestyle='None', color='none', markerfacecolor=MARKER_COLOR, markeredgecolor=MARKER_EDGE_COLOR, markeredgewidth=0.35, markersize=5.0, label=motion_cfg['display'])  # 构造单图图例句柄
            fig.legend(handles=[legend_handle], loc='lower center', ncol=1, frameon=True, edgecolor='#666666', framealpha=1.0, fontsize=10, bbox_to_anchor=(0.5, 0.02), handletextpad=0.4)  # 绘制单图底部图例
            plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.14, wspace=0.00, hspace=0.00)  # 调整单图边距保证图例不遮挡
            single_name = build_single_figure_name(motion_cfg['key'], angle_value)  # 生成当前单图文件名
            single_path = os.path.join(output_dir, single_name)  # 组装当前单图输出路径
            fig.savefig(single_path, dpi=350)  # 保存当前单图
            plt.close(fig)  # 关闭当前单图对象释放内存
            output_paths.append(single_path)  # 记录当前单图输出路径
    return output_paths  # 返回全部单图输出路径


def main():  # 定义主函数
    if len(sys.argv) >= 2:  # 判断是否传入批处理根目录参数
        batch_root = os.path.abspath(sys.argv[1])  # 使用命令行参数覆盖默认批处理目录
    else:  # 未传入参数时走默认路径
        batch_root = DEFAULT_BATCH_ROOT  # 使用项目下 Batch 目录作为默认数据根目录
    if len(sys.argv) >= 3:  # 判断是否传入输出图片路径参数
        output_figure = os.path.abspath(sys.argv[2])  # 使用命令行参数覆盖默认输出文件
    else:  # 未传入输出参数时使用默认文件名
        output_figure = os.path.join(batch_root, OUTPUT_FIGURE_NAME)  # 将图片默认输出到批处理目录
    output_dir = os.path.dirname(output_figure)  # 获取输出图片所在目录路径
    if output_dir and (not os.path.isdir(output_dir)):  # 判断输出目录是否需要创建
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录并允许已存在
    records = collect_records(batch_root)  # 扫描批处理目录并收集全部有效记录
    summary_df = build_summary_dataframe(records)  # 构建去重后的统计汇总表
    output_summary = os.path.join(output_dir, OUTPUT_SUMMARY_NAME)  # 组装汇总 CSV 输出路径
    summary_df.to_csv(output_summary, index=False, encoding='utf-8-sig')  # 保存汇总表便于后续复核
    create_figure(summary_df, output_figure)  # 生成并保存 3D 子图总图
    single_paths = create_single_figures(summary_df, output_dir)  # 生成并保存每个子图对应单图
    print('批处理目录: {}'.format(batch_root))  # 输出批处理目录信息
    print('汇总数据: {}'.format(output_summary))  # 输出汇总表路径信息
    print('输出图片: {}'.format(output_figure))  # 输出图片路径信息
    print('单图数量: {}'.format(len(single_paths)))  # 输出单图数量信息
    for single_path in single_paths:  # 遍历单图输出路径
        print('单图输出: {}'.format(single_path))  # 输出当前单图路径信息


if __name__ == '__main__':  # 判断脚本是否以主程序方式运行
    main()  # 调用主函数执行流程