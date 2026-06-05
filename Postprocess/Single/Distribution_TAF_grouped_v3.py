# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""按相同 h 与 i 聚合不同 angle 的 TAF-El_Centro 曲线并绘制对比图。"""  # 说明脚本用途
import os  # 导入 os 用于路径处理与目录遍历
import re  # 导入 re 用于解析文件夹名中的参数
import math  # 导入 math 用于计算坡脚归一化坐标
import pandas as pd  # 导入 pandas 用于读取 CSV 数据
import numpy as np  # 导入 numpy 用于数值与刻度生成
import matplotlib  # 导入 matplotlib 主模块用于设置后端
matplotlib.use('Agg')  # 设置无界面后端避免脚本运行时弹窗
import matplotlib.pyplot as plt  # 导入 pyplot 用于绘图
import matplotlib.font_manager as fm  # 导入字体管理器用于中英文字体配置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 记录当前脚本目录
WORK_ROOT = SCRIPT_DIR  # 定义输入与输出统一使用脚本当前目录
TARGET_FILENAME = 'TAF-El_Centro.csv'  # 定义每个算例目录中目标 CSV 文件名
OUTPUT_FIG_DIR_NAME = '1-TAF_grouped'  # 定义分组图统一输出子目录名
LOC_CREST = 3.0  # 定义坡顶归一化位置
DEFAULT_TOE = 4.0  # 定义坡脚默认归一化位置
ANGLE_STYLES = {  # 定义 angle 与颜色线型映射字典
    0: {'color': '#800080', 'linestyle': ':', 'linewidth': 1.5, 'label': '0°'},  # 定义 0° 为紫色点线
    10: {'color': '#2ca02c', 'linestyle': (0, (4, 2)), 'linewidth': 1.5, 'label': '10°'},  # 定义 10° 为绿色短虚线
    20: {'color': '#1f77b4', 'linestyle': (0, (6, 3)), 'linewidth': 1.5, 'label': '20°'},  # 定义 20° 为蓝色长虚线
    30: {'color': '#d62728', 'linestyle': '-', 'linewidth': 1.5, 'label': '30°'},  # 定义 30° 为红色实线
}  # 结束 angle 样式映射定义
CASE_PATTERN = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_angle(?P<angle>-?\d+(?:\.\d+)?)')  # 定义参数文件夹名解析正则
H_PATTERN = re.compile(r'(^|[^0-9A-Za-z])h(?P<value>-?\d+(?:\.\d+)?)(?=$|[^0-9A-Za-z])', re.IGNORECASE)  # 定义 h 参数独立匹配正则
I_PATTERN = re.compile(r'(^|[^0-9A-Za-z])i(?P<value>-?\d+(?:\.\d+)?)(?=$|[^0-9A-Za-z])', re.IGNORECASE)  # 定义 i 参数独立匹配正则
ANGLE_PATTERN = re.compile(r'(^|[^0-9A-Za-z])angle(?P<value>-?\d+(?:\.\d+)?)(?=$|[^0-9A-Za-z])', re.IGNORECASE)  # 定义 angle 参数独立匹配正则
def build_font_properties():  # 定义构建中英文字体属性的函数
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']  # 定义中文字体候选列表
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']  # 定义英文字体候选列表
    cn_font = fm.FontProperties()  # 初始化中文字体属性
    en_font = fm.FontProperties()  # 初始化英文字体属性
    for name in cn_candidates:  # 遍历中文字体候选
        try:  # 尝试查找当前中文字体
            font_path = fm.findfont(name, fallback_to_default=False)  # 严格查找字体文件路径
            cn_font = fm.FontProperties(fname=font_path)  # 使用查找到的字体路径创建中文字体属性
            break  # 找到可用字体后结束循环
        except Exception:  # 捕获不可用字体异常
            continue  # 继续尝试下一个中文字体
    for name in en_candidates:  # 遍历英文字体候选
        try:  # 尝试查找当前英文字体
            font_path = fm.findfont(name, fallback_to_default=False)  # 严格查找字体文件路径
            en_font = fm.FontProperties(fname=font_path)  # 使用查找到的字体路径创建英文字体属性
            break  # 找到可用字体后结束循环
        except Exception:  # 捕获不可用字体异常
            continue  # 继续尝试下一个英文字体
    return cn_font, en_font  # 返回中英文字体属性
CN_FONT, EN_FONT = build_font_properties()  # 构建并缓存字体属性
def format_value_text(value):  # 定义数值格式化函数
    return str(int(value)) if float(value).is_integer() else f'{value:g}'  # 将整数格式化为无小数文本并压缩浮点尾零
def compute_toe_location_from_angle(slope_angle_deg):  # 定义根据坡角计算坡脚位置的函数
    tan_value = math.tan(math.radians(slope_angle_deg))  # 将坡角转弧度后计算正切值
    if abs(tan_value) <= 1e-12:  # 判断正切是否接近零以避免除零
        return DEFAULT_TOE  # 返回默认坡脚位置
    return LOC_CREST + 1.0 / tan_value  # 按几何关系计算坡脚归一化坐标
def parse_case_folder(folder_name):  # 定义解析参数文件夹名的函数
    h_match = H_PATTERN.search(folder_name)  # 在文件夹名中独立匹配 h 参数
    i_match = I_PATTERN.search(folder_name)  # 在文件夹名中独立匹配 i 参数
    angle_match = ANGLE_PATTERN.search(folder_name)  # 在文件夹名中独立匹配 angle 参数
    if not (h_match and i_match and angle_match):  # 判断是否存在任一参数匹配失败
        return None  # 返回空值表示不是目标参数文件夹
    h_value = float(h_match.group('value'))  # 读取 h 参数并转换为浮点数
    i_value = float(i_match.group('value'))  # 读取 i 参数并转换为浮点数
    angle_value = float(angle_match.group('value'))  # 读取 angle 参数并转换为浮点数
    return h_value, i_value, angle_value  # 返回解析得到的参数元组
def collect_taf_records(batch_root):  # 定义收集 TAF-El_Centro 数据记录的函数
    records = []  # 初始化记录列表
    for entry_name in sorted(os.listdir(batch_root)):  # 遍历批处理目录下的条目
        folder_path = os.path.join(batch_root, entry_name)  # 组装当前条目的完整路径
        if not os.path.isdir(folder_path):  # 判断当前条目是否为目录
            continue  # 跳过非目录条目
        parsed = parse_case_folder(entry_name)  # 尝试从目录名解析 h、i、angle 参数
        if parsed is None:  # 判断目录名是否不符合参数命名规则
            continue  # 跳过无效目录
        h_value, i_value, angle_value = parsed  # 解包解析得到的参数值
        csv_path = os.path.join(folder_path, TARGET_FILENAME)  # 组装目标 CSV 文件完整路径
        if not os.path.isfile(csv_path):  # 判断目标 CSV 是否存在
            continue  # 跳过缺失目标 CSV 的目录
        records.append({'h': h_value, 'i': i_value, 'angle': angle_value, 'csv_path': csv_path, 'folder': folder_path})  # 记录有效算例信息
    if not records:  # 判断是否未收集到任何有效记录
        raise FileNotFoundError(f'未在 {batch_root} 中找到包含 {TARGET_FILENAME} 的参数文件夹。')  # 抛出缺失数据错误
    return records  # 返回收集到的记录列表
def group_records_by_hi(records):  # 定义按 h 与 i 分组记录的函数
    grouped = {}  # 初始化分组字典
    for item in records:  # 遍历所有记录
        key = (item['h'], item['i'])  # 构建当前记录的分组键
        if key not in grouped:  # 判断分组键是否首次出现
            grouped[key] = []  # 初始化当前分组列表
        grouped[key].append(item)  # 将记录加入对应分组
    for key in grouped:  # 遍历每个分组键
        grouped[key].sort(key=lambda x: x['angle'])  # 按 angle 升序排序分组记录
    return grouped  # 返回按 h 与 i 分组后的结果
def load_taf_dataframe(csv_path):  # 定义读取单个 TAF CSV 的函数
    df = pd.read_csv(csv_path)  # 读取 CSV 为数据表
    required_cols = {'x/h', 'TAF_h'}  # 定义必须存在的列集合
    missing_cols = required_cols - set(df.columns)  # 计算缺失列集合
    if missing_cols:  # 判断是否存在缺失列
        raise ValueError(f'文件 {csv_path} 缺少列: {sorted(missing_cols)}')  # 抛出缺失列错误
    df = df[['x/h', 'TAF_h']].copy()  # 保留绘图必需列并复制数据
    df = df.sort_values(by='x/h')  # 按 x/h 升序排序以保证曲线顺序正确
    return df  # 返回清洗后的数据表
def plot_group_curves(h_value, i_value, group_items, output_dir):  # 定义绘制同 h、i 下多角度曲线的函数
    # 重用绘图配置并在传入的单独画布上绘制（由 draw_group_on_axis 实现）
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0))  # 创建单图画布
    draw_group_on_axis(ax, h_value, i_value, group_items)  # 在当前轴上绘制曲线
    title_h = format_value_text(h_value)  # 格式化标题中的 h 文本
    title_i = format_value_text(i_value)  # 格式化标题中的 i 文本
    output_name = f'TAF-El_Centro-h{title_h}-i{title_i}-angles.png'  # 组装当前分组输出图片文件名
    output_path = os.path.join(output_dir, output_name)  # 组装当前分组输出图片完整路径
    fig.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存图像到输出目录
    plt.close(fig)  # 关闭图对象释放内存
    print(f'已输出: {output_path}')  # 输出当前图像保存结果


def draw_group_on_axis(ax, h_value, i_value, group_items):  # 在指定轴上绘制单组曲线的函数
    ax.set_facecolor('#f2f2f2')  # 设置子图背景为浅灰色
    plotted_count = 0  # 初始化已绘制曲线计数器
    for item in group_items:  # 遍历当前分组中的各 angle 记录
        angle_raw = item['angle']  # 读取当前记录 angle 原始值
        angle_int = int(round(angle_raw))  # 将 angle 四舍五入为整数键值
        style = ANGLE_STYLES.get(angle_int)  # 从样式映射中读取当前 angle 样式
        if style is None:  # 判断当前 angle 是否不在预设映射中
            continue  # 跳过未定义样式的 angle
        taf_df = load_taf_dataframe(item['csv_path'])  # 读取当前 angle 对应的 TAF 数据
        x_values = taf_df['x/h'].to_numpy(dtype=float)  # 提取横坐标数组
        y_values = taf_df['TAF_h'].to_numpy(dtype=float)  # 提取 TAF_h 数组
        ax.plot(x_values, y_values, color=style['color'], linestyle=style['linestyle'], linewidth=style['linewidth'], label=style['label'])  # 按指定颜色与线型绘制曲线
        plotted_count += 1  # 累加已绘制曲线数量
    if plotted_count == 0:  # 判断当前分组是否没有可绘制曲线
        raise ValueError(f'h={h_value}, i={i_value} 组没有可用的 angle 曲线。')  # 抛出无可绘制曲线错误
    toe_x = compute_toe_location_from_angle(i_value)  # 根据坡角计算坡脚位置
    title_h = format_value_text(h_value)  # 格式化标题中的 h 文本
    title_i = format_value_text(i_value)  # 格式化标题中的 i 文本
    ax.set_title(f'h = {title_h}m, i = {title_i}°', fontsize=12, fontproperties=EN_FONT, pad=4)  # 设置标题样式与文本（较小字号以适配汇总图）
    ax.set_xlabel('x/h', fontsize=8, fontproperties=EN_FONT, fontstyle='italic')  # 设置横轴标签样式
    ax.set_ylabel('TAF', fontsize=8, fontproperties=EN_FONT)  # 设置纵轴标签样式
    ax.set_xlim(0, 8)  # 固定横轴范围为 0 到 8
    ax.set_xticks(np.arange(0, 9, 1))  # 设置横轴整数刻度
    ax.set_ylim(0.5, 3.0)  # 固定纵轴范围为 0.5 到 3.0
    ax.set_yticks(np.arange(0.5, 3.1, 0.5))  # 设置纵轴主刻度间距为 0.5
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # 设置纵轴数字显示为一位小数
    ax.axvline(x=LOC_CREST, color='black', linestyle='--', linewidth=1.0)  # 绘制坡顶位置虚线
    ax.axvline(x=toe_x, color='black', linestyle='--', linewidth=1.0)  # 绘制坡脚位置虚线
    text_y = 2.92  # 设置中文标注位于图上部的统一高度
    ax.text(LOC_CREST - 0.06, text_y, '坡顶', fontsize=9, fontproperties=CN_FONT, va='top', ha='right')  # 在坡顶虚线左侧上部添加“坡顶”标注
    ax.text(toe_x + 0.06, text_y, '坡脚', fontsize=9, fontproperties=CN_FONT, va='top', ha='left')  # 在坡脚虚线右侧上部添加“坡脚”标注
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.6, color='#d0d0d0')  # 设置浅灰短虚线网格
    ax.tick_params(direction='in', top=True, right=True, labelsize=8)  # 设置刻度朝内并显示上右刻度
    ax.legend(loc='upper right', frameon=True, edgecolor='#666666', prop=EN_FONT, fontsize=8)  # 设置角度图例位置与样式


def plot_summary_figure(grouped, output_dir, cols=4, include_groups=None):  # 定义生成汇总多子图的函数
    # include_groups: 可选列表，按顺序包含要绘制的 (h, i) 组合（例如 [(50,30),(50,45),...]）
    items_all = sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1]))  # 按 h 与 i 排序
    if include_groups:  # 若传入包含列表，则按该顺序筛选存在的分组
        items = []
        for h_i in include_groups:  # 按用户指定顺序遍历
            key = (float(h_i[0]), float(h_i[1]))
            if key in grouped:
                items.append((key, grouped[key]))
            else:
                print(f'警告：未找到分组 h={key[0]}, i={key[1]}，已跳过。')
    else:
        items = items_all
    n = len(items)  # 子图数量
    if n == 0:  # 若无子图则直接返回
        return
    rows = math.ceil(n / cols)  # 计算行数
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0))  # 创建子图网格
    ax_list = np.array(axes).reshape(-1)  # 扁平化轴列表
    for idx, ((h_value, i_value), group_items) in enumerate(items):  # 遍历每个分组并绘制至对应子图
        ax = ax_list[idx]  # 选择对应轴
        try:
            draw_group_on_axis(ax, h_value, i_value, group_items)  # 在轴上绘制
        except Exception as exc:  # 若单个子图出错则在该轴写明错误信息
            ax.text(0.5, 0.5, f'错误: {exc}', ha='center', va='center', fontsize=8)
            ax.set_axis_off()
    # 隐藏多余轴
    for j in range(n, len(ax_list)):
        ax_list[j].set_visible(False)
    plt.tight_layout()  # 自动布局
    summary_name = 'TAF-El_Centro-summary.png'  # 汇总图文件名
    summary_path = os.path.join(output_dir, summary_name)  # 汇总图完整路径
    fig.savefig(summary_path, dpi=300, bbox_inches='tight')  # 保存汇总图
    plt.close(fig)  # 关闭图对象释放内存
    print(f'已输出汇总图: {summary_path}')  # 输出汇总图保存结果
    return summary_path


def save_aggregated_csv(grouped, output_dir, filename='TAF-El_Centro-all.csv'):
    """将用于绘图的所有 TAF 数据汇总为单个 CSV 并保存到输出目录。
    只包含脚本实际绘制的角度（由 ANGLE_STYLES 指定）。
    """
    rows = []  # 用于收集每个子表的列表
    for (h_value, i_value), items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for item in items:
            angle_int = int(round(item['angle']))
            if angle_int not in ANGLE_STYLES:  # 仅收集将被绘制的角度
                continue
            try:
                df = load_taf_dataframe(item['csv_path'])  # 读取 CSV
            except Exception as exc:
                print(f'警告：读取 {item["csv_path"]} 失败，已跳过 -> {exc}')
                continue
            df2 = df.copy()
            df2.insert(0, 'angle', angle_int)
            df2.insert(0, 'i', float(i_value))
            df2.insert(0, 'h', float(h_value))
            df2['folder'] = os.path.basename(item['folder'])  # 记录所属文件夹名
            df2['csv_path'] = item['csv_path']  # 记录原始 CSV 路径
            rows.append(df2)
    if not rows:
        raise FileNotFoundError('未找到任何要汇总的 TAF 数据（请确认目录与角度设置）。')
    all_df = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(output_dir, filename)
    all_df.to_csv(out_path, index=False)
    print(f'已保存汇总CSV: {out_path}')
    return out_path
def main():  # 定义主函数
    batch_root = WORK_ROOT  # 设置批处理根目录为脚本当前目录
    if not os.path.isdir(batch_root):  # 判断当前目录是否存在
        raise NotADirectoryError(f'当前目录不存在: {batch_root}')  # 抛出目录不存在错误
    output_dir = os.path.join(batch_root, OUTPUT_FIG_DIR_NAME)  # 组装统一图片输出目录路径
    if not os.path.isdir(output_dir):  # 判断统一图片输出目录是否已存在
        os.makedirs(output_dir, exist_ok=True)  # 创建统一图片输出目录
    records = collect_taf_records(batch_root)  # 从当前目录收集所有有效算例记录
    grouped = group_records_by_hi(records)  # 按 h 与 i 对记录进行分组
    # 先保存汇总CSV，便于后续分析（只包含将被绘制的角度）
    try:
        save_aggregated_csv(grouped, output_dir)
    except Exception as exc:
        print(f'警告：保存汇总CSV失败 -> {exc}')
    for (h_value, i_value), group_items in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):  # 按 h 与 i 升序遍历分组
        plot_group_curves(h_value, i_value, group_items, output_dir)  # 对当前分组绘图并保存到统一图片输出目录
    # 生成一张汇总图，按若干列排列所有子图并保存
    try:
        # 仅生成与示例图相同的八个组合（按示例顺序）
        include = [(50, 30), (50, 45), (200, 45), (200, 60), (100, 30), (100, 45), (400, 45), (400, 60)]
        plot_summary_figure(grouped, output_dir, cols=4, include_groups=include)  # 生成并保存汇总图，仅包含指定组合
    except Exception as exc:  # 捕获汇总图生成错误但不影响已输出的单图
        print(f'警告：生成汇总图失败 -> {exc}')  # 输出警告信息
if __name__ == '__main__':  # 判断是否作为主脚本直接运行
    main()  # 调用主函数执行流程