# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""根据反应谱数据生成按斜坡高度 h 分面的最大谱加速度 Sa 图表"""  # 说明脚本用途
import os  # 导入 os 模块用于路径与目录操作
import re  # 导入 re 模块用于正则解析目录与文件名
import sys  # 导入 sys 模块用于读取命令行参数
import numpy as np  # 导入 numpy 用于数值计算
import pandas as pd  # 导入 pandas 用于表格读取与统计聚合
import matplotlib  # 导入 matplotlib 主模块用于后端设置
matplotlib.use('Agg')  # 设置无界面后端以支持批处理运行
import matplotlib.pyplot as plt  # 导入 pyplot 用于绘图
from matplotlib.lines import Line2D  # 导入 Line2D 用于自定义图例句柄

# ==========================================
# 配置参数
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
DEFAULT_BATCH_ROOT = SCRIPT_DIR  # 设置默认数据根目录为脚本目录
OUTPUT_FIGURE_NAME = 'Sa_distribution_by_h.png'  # 定义总图输出文件名
OUTPUT_SUMMARY_CSV = 'Sa_distribution_summary.csv'  # 定义汇总 CSV 文件名
OUTPUT_IMAGE_DIR_NAME = '5-FIGURES'  # 定义图像统一输出子目录名称
SPECTRUM_DIR = '2-RESPONSE_SPECTRUM'  # 定义反应谱输入目录名
SA_GLOB_PREFIX = 'sa_surface'  # 定义反应谱文件名前缀（小写匹配）
TARGET_PERIOD = None  # 定义目标周期（None 表示取全周期的最大值，也可设为具体值如 1.0）
PREFERRED_H_VALUES = [50.0, 100.0, 200.0, 400.0]  # 定义优先 h 显示顺序
PREFERRED_I_VALUES = [30.0, 45.0, 60.0, 75.0]  # 定义优先 i 显示顺序
MATCH_TOL = 1e-6  # 定义浮点匹配容差

# 定义标记点与颜色映射
MARKER_STYLES = {  # 定义标记风格映射字典
    30.0: ('s', '#1f77b4'),  # i=30° 使用方形标记与蓝色
    45.0: ('o', '#ff7f0e'),  # i=45° 使用圆形标记与橙色
    60.0: ('v', '#2ca02c'),  # i=60° 使用向下三角标记与绿色
    75.0: ('D', '#d62728'),  # i=75° 使用菱形标记与红色
}  # 结束标记风格定义

MOTION_CONFIGS = [  # 定义三种地震动配置
    {'key': 'el_centro', 'token': 'elcentro', 'display': 'El-Centro'},  # 定义 El-Centro 配置
    {'key': 'northridge', 'token': 'northridge', 'display': 'Northridge'},  # 定义 Northridge 配置
    {'key': 'loma_prieta', 'token': 'lomaprieta', 'display': 'Loma-Prieta'},  # 定义 Loma-Prieta 配置
]  # 结束地震动配置定义

CASE_PATTERN = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_angle(?P<angle>-?\d+(?:\.\d+)?)')  # 定义参数目录名解析正则


def canonical_text(text):  # 定义文本规范化函数
    """规范化文本用于关键词匹配"""  # 函数说明
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())  # 仅保留小写字母与数字用于关键词匹配


def parse_case_folder(folder_name):  # 定义参数目录名解析函数
    """从目录名解析 h、i、angle 参数"""  # 函数说明
    match = CASE_PATTERN.search(folder_name)  # 在目录名中匹配 h、i、angle 参数
    if match is None:  # 判断是否匹配失败
        return None  # 返回空值表示非目标参数目录
    h_value = float(match.group('h'))  # 提取并转换 h 参数
    i_value = float(match.group('i'))  # 提取并转换 i 参数
    angle_value = float(match.group('angle'))  # 提取并转换 angle 参数
    return h_value, i_value, angle_value  # 返回参数元组


def detect_motion(file_stem):  # 定义地震动类型识别函数
    """根据文件名识别地震动类型"""  # 函数说明
    normalized = canonical_text(file_stem)  # 规范化文件主干文本
    for motion_cfg in MOTION_CONFIGS:  # 遍历预定义地震动配置
        if motion_cfg['token'] in normalized:  # 判断当前文件是否包含地震动关键词
            return motion_cfg  # 返回命中的地震动配置
    return None  # 未命中时返回空值


def extract_max_sa_from_spectrum(spectrum_df, target_period=None):  # 定义从反应谱数据提取最大 Sa 的函数
    """
    从反应谱 CSV 提取最大谱加速度
    
    参数:
        spectrum_df: 反应谱数据表（行=节点，列=周期对应的 Sa 值）  # 参数说明
        target_period: 目标周期；若为 None 则取全周期的最大值  # 参数说明
    
    返回:
        sa_max: 最大谱加速度（g）  # 返回值说明
    """
    if spectrum_df.empty:  # 判断输入是否为空
        return None  # 返回空值表示无效数据
    
    # 过滤掉前两列（Node_Label 和 x/h），只保留 Sa 值列
    sa_columns = [col for col in spectrum_df.columns if col not in ['Node_Label', 'x/h']]  # 提取 Sa 值列
    
    if not sa_columns:  # 判断是否无 Sa 列
        return None  # 返回空值表示无有效数据
    
    # 提取 Sa 值矩阵
    sa_matrix = spectrum_df[sa_columns].to_numpy(dtype=float)  # 转换为浮点矩阵
    
    if target_period is not None:  # 判断是否指定了目标周期
        # 尝试查找目标周期对应的列
        target_col_name = None  # 初始化目标列名为空
        for col in sa_columns:  # 遍历每个 Sa 列
            try:  # 尝试解析列名中的周期值
                col_period = float(col.replace('T_', ''))  # 从列名中提取周期值
                if abs(col_period - target_period) < 1e-6:  # 判断是否匹配目标周期
                    target_col_name = col  # 记录目标列名
                    break  # 命中后结束循环
            except ValueError:  # 列名不包含周期信息时抛出异常
                continue  # 跳过无效列
        
        if target_col_name is not None:  # 判断是否找到目标周期列
            sa_values = spectrum_df[target_col_name].to_numpy(dtype=float)  # 提取目标周期的 Sa 值
        else:  # 未找到目标周期时使用全周期最大值
            sa_values = np.nanmax(sa_matrix, axis=1)  # 对每行节点求全周期最大值
    else:  # 未指定目标周期时使用全周期最大值
        sa_values = np.nanmax(sa_matrix, axis=1)  # 对每行节点求全周期最大值
    
    # 取所有节点的最大值作为该工况的代表值
    sa_max = float(np.nanmax(sa_values))  # 计算最大谱加速度
    
    return sa_max  # 返回最大谱加速度


def collect_records(batch_root):  # 定义扫描目录并收集记录的函数
    """扫描工况目录并从反应谱文件采集记录"""  # 函数说明
    records = []  # 初始化记录列表
    if not os.path.isdir(batch_root):  # 判断根目录是否存在
        raise NotADirectoryError('数据目录不存在: {}'.format(batch_root))  # 抛出目录不存在异常
    
    for entry_name in sorted(os.listdir(batch_root)):  # 遍历根目录下全部条目
        folder_path = os.path.join(batch_root, entry_name)  # 组装当前条目完整路径
        if not os.path.isdir(folder_path):  # 判断当前条目是否为目录
            continue  # 跳过非目录条目
        
        parsed = parse_case_folder(entry_name)  # 尝试解析参数目录名
        if parsed is None:  # 判断是否为有效参数目录
            continue  # 跳过无效目录
        h_value, i_value, angle_value = parsed  # 解包参数值
        
        # 查找反应谱文件
        spectrum_dir = os.path.join(folder_path, SPECTRUM_DIR)  # 组装反应谱目录路径
        if not os.path.isdir(spectrum_dir):  # 判断反应谱目录是否存在
            print('警告: 反应谱目录不存在 -> {}'.format(spectrum_dir))  # 输出警告
            continue  # 跳过无反应谱目录的工况
        
        spectrum_files = []  # 初始化反应谱文件列表
        for file_name in sorted(os.listdir(spectrum_dir)):  # 遍历反应谱目录内文件名
            lower_name = file_name.lower()  # 转为小写便于判断
            if (not lower_name.endswith('.csv')) or (not lower_name.startswith(SA_GLOB_PREFIX)):  # 判断是否为 Sa CSV
                continue  # 跳过非反应谱 CSV
            spectrum_files.append(os.path.join(spectrum_dir, file_name))  # 记录反应谱文件路径
        
        if not spectrum_files:  # 判断是否无反应谱文件
            print('警告: 反应谱目录中无 Sa CSV 文件 -> {}'.format(spectrum_dir))  # 输出警告
            continue  # 跳过无反应谱文件的工况
        
        # 处理每个反应谱文件
        for spectrum_path in spectrum_files:  # 遍历每个反应谱文件
            file_name = os.path.basename(spectrum_path)  # 读取文件名
            file_stem = os.path.splitext(file_name)[0]  # 读取文件主干
            motion_cfg = detect_motion(file_stem)  # 识别地震动类型
            if motion_cfg is None:  # 判断是否为目标地震动
                continue  # 跳过未识别波名文件
            
            try:  # 尝试读取反应谱数据
                spectrum_df = pd.read_csv(spectrum_path)  # 读取反应谱 CSV
                sa_max = extract_max_sa_from_spectrum(spectrum_df, TARGET_PERIOD)  # 提取最大 Sa
                if sa_max is None or sa_max <= 0:  # 判断 Sa 是否有效
                    print('警告: 反应谱文件数据无效 -> {}'.format(spectrum_path))  # 输出警告
                    continue  # 跳过无效数据
                
                records.append({  # 追加结构化记录
                    'motion': motion_cfg['key'],  # 地震动类型键
                    'motion_display': motion_cfg['display'],  # 地震动显示名
                    'h': float(h_value),  # 斜坡高度
                    'i': float(i_value),  # 斜坡角度
                    'angle': float(angle_value),  # 入射角
                    'sa_max': float(sa_max),  # 最大谱加速度
                    'file': spectrum_path  # 数据文件路径
                })  # 结束记录追加
            except Exception as exc:  # 捕获处理异常
                print('警告：已跳过 {} -> {}'.format(spectrum_path, str(exc)))  # 输出跳过信息
                continue  # 跳过当前异常文件
    
    if not records:  # 判断是否采集到有效记录
        raise FileNotFoundError('未采集到有效反应谱数据，请检查是否已运行 Compute_Response_Spectrum_v1.py')  # 抛出无数据异常
    return records  # 返回记录列表


def create_sa_by_h_figure(summary_df, output_path, motion_key='el_centro'):  # 定义创建按 h 分面图的函数
    """为指定地震动生成按 h 分面的 Sa 分布图（2×2 网格）"""  # 函数说明
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
    
    # 过滤当前地震动的数据
    motion_df = summary_df[summary_df['motion'] == motion_key].copy()  # 提取当前地震动数据
    if motion_df.empty:  # 判断是否无数据
        print('警告：地震动 {} 无数据'.format(motion_key))  # 输出警告
        return  # 直接返回
    
    # 创建 2×2 网格图
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), facecolor='white')  # 创建 2×2 子图画布
    fig.suptitle('Maximum Spectral Acceleration Distribution', fontsize=16, y=0.995)  # 添加总标题
    
    axes_flat = axes.flatten()  # 展平坐标轴数组便于迭代
    
    # 为每个 h 值生成一个子图
    for h_idx, h_value in enumerate(PREFERRED_H_VALUES):  # 遍历每个优先 h 值
        ax = axes_flat[h_idx]  # 获取当前子图坐标轴
        
        # 过滤当前 h 的数据
        h_data = motion_df[np.abs(motion_df['h'] - h_value) < MATCH_TOL].copy()  # 提取当前 h 的数据
        
        if h_data.empty:  # 判断当前 h 是否无数据
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=12)  # 标注无数据
            ax.set_title('$h = {}$ m'.format(int(h_value)), fontsize=13)  # 设置子图标题
            ax.set_xlabel(r'$\theta_s\,\mathrm{(^{\circ})}$', fontsize=12)  # 设置横轴标签
            ax.set_ylabel(r'$S_a^{\max}\,\mathrm{(g)}$', fontsize=12)  # 设置纵轴标签
            continue  # 跳过当前无数据子图
        
        # 为每个 i 值绘制曲线
        plotted_i_values = set()  # 初始化已绘制 i 值集合用于图例
        
        for i_value in PREFERRED_I_VALUES:  # 遍历每个优先 i 值
            i_data = h_data[np.abs(h_data['i'] - i_value) < MATCH_TOL].copy()  # 提取当前 i 的数据
            
            if i_data.empty:  # 判断当前 i 是否无数据
                continue  # 跳过无数据的 i 值
            
            # 按 angle 排序以便绘制连接线
            i_data = i_data.sort_values(by='angle').reset_index(drop=True)  # 按入射角排序
            
            if i_value in MARKER_STYLES:  # 判断当前 i 值是否在标记映射中
                marker, color = MARKER_STYLES[i_value]  # 获取对应的标记与颜色
            else:  # 当前 i 值不在映射中时使用默认样式
                marker = 'o'  # 默认使用圆形标记
                color = '#999999'  # 默认使用灰色
            
            # 绘制曲线与散点
            ax.plot(i_data['angle'], i_data['sa_max'], marker=marker, color=color, markersize=7, linewidth=1.5, linestyle='-', alpha=0.75, label='$i={:.0f}°$'.format(i_value))  # 绘制数据曲线
            plotted_i_values.add(i_value)  # 记录当前 i 值已绘制
        
        # 设置子图样式
        ax.set_facecolor('#f5f5f5')  # 设置子图背景色
        ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')  # 添加网格线
        for spine in ax.spines.values():  # 遍历坐标轴边框
            spine.set_linewidth(1.0)  # 设置边框线宽
            spine.set_color('#333333')  # 设置边框颜色
        
        ax.tick_params(direction='in', top=True, right=True, labelsize=11)  # 设置刻度样式
        
        # 设置坐标轴标签与范围
        ax.set_xlabel(r'$\theta_s\,\mathrm{(^{\circ})}$', fontsize=12)  # 设置横轴标签
        if h_idx % 2 == 0:  # 判断是否为左列子图
            ax.set_ylabel(r'$S_a^{\max}\,\mathrm{(g)}$', fontsize=12)  # 左列设置纵轴标签
        
        ax.set_xlim(-2, 32)  # 设置横轴范围
        ax.set_xticks(np.arange(0, 31, 5))  # 设置横轴刻度
        
        ax.set_title('$h = {}$ m'.format(int(h_value)), fontsize=13)  # 设置子图标题
        
        # 添加图例（仅在第一个子图）
        if h_idx == 0 and plotted_i_values:  # 判断是否为第一个子图且有数据
            ax.legend(loc='upper left', fontsize=11, frameon=True, fancybox=False, edgecolor='#888888')  # 添加图例
    
    plt.tight_layout()  # 自动调整布局避免遮挡
    fig.savefig(output_path, dpi=350, bbox_inches='tight')  # 保存图表
    plt.close(fig)  # 关闭图对象释放内存
    print('图表已保存: {}'.format(output_path))  # 输出保存成功信息


def main():  # 定义主函数
    """主程序入口"""  # 函数说明
    if len(sys.argv) >= 2:  # 判断是否传入数据根目录参数
        batch_root = os.path.abspath(sys.argv[1])  # 使用命令行参数作为数据目录
    else:  # 未传入参数时使用默认目录
        batch_root = DEFAULT_BATCH_ROOT  # 使用脚本目录作为默认数据目录
    
    if len(sys.argv) >= 3:  # 判断是否传入输出目录参数
        output_dir = os.path.abspath(sys.argv[2])  # 使用命令行参数作为输出目录
    else:  # 未传入输出参数时使用数据目录
        output_dir = batch_root  # 默认输出到数据目录
    
    if not os.path.isdir(output_dir):  # 判断输出目录是否存在
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    
    output_image_dir = os.path.join(output_dir, OUTPUT_IMAGE_DIR_NAME)  # 组装图像统一输出目录路径
    if not os.path.isdir(output_image_dir):  # 判断图像输出目录是否存在
        os.makedirs(output_image_dir, exist_ok=True)  # 创建图像输出目录
    
    print('数据目录: {}'.format(batch_root))  # 输出数据目录信息
    print('输出目录: {}'.format(output_image_dir))  # 输出输出目录信息
    print('目标周期: {} (None=全周期最大值)'.format(TARGET_PERIOD))  # 输出目标周期设置
    print('')  # 输出空行
    
    # 采集记录
    try:  # 尝试采集记录
        records = collect_records(batch_root)  # 扫描并采集全部记录
        print('采集记录数: {}'.format(len(records)))  # 输出采集的记录数
    except FileNotFoundError as exc:  # 捕获无数据异常
        print('错误: {}'.format(str(exc)))  # 输出错误信息
        print('请先运行: Compute_Response_Spectrum_v1.py 生成反应谱数据')  # 输出提示信息
        return  # 直接返回
    
    # 转换为数据表
    summary_df = pd.DataFrame(records)  # 将记录列表转换为数据表
    
    # 保存汇总 CSV
    output_csv = os.path.join(output_image_dir, OUTPUT_SUMMARY_CSV)  # 组装汇总 CSV 路径
    summary_df.to_csv(output_csv, index=False, encoding='utf-8-sig')  # 保存汇总 CSV
    print('汇总 CSV: {}'.format(output_csv))  # 输出汇总 CSV 路径
    print('')  # 输出空行
    
    # 为每种地震动生成图表
    for motion_cfg in MOTION_CONFIGS:  # 遍历每种地震动配置
        motion_key = motion_cfg['key']  # 获取地震动键
        motion_display = motion_cfg['display']  # 获取地震动显示名
        
        output_figure = os.path.join(output_image_dir, 'Sa_by_h_{}.png'.format(motion_key))  # 组装输出图路径
        
        try:  # 尝试生成当前地震动图表
            create_sa_by_h_figure(summary_df, output_figure, motion_key)  # 生成按 h 分面图
            print('图表已生成: {} ({})'.format(output_figure, motion_display))  # 输出成功信息
        except Exception as exc:  # 捕获生成异常
            print('警告：生成 {} 图表失败: {}'.format(motion_display, str(exc)))  # 输出错误信息
    
    print('\n生成完成！')  # 输出完成信息


if __name__ == '__main__':  # 判断脚本是否以主程序方式运行
    main()  # 调用主函数执行流程
