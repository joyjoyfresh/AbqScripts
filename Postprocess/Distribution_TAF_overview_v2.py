# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
import os  # 导入 os 模块用于路径与目录操作
import re  # 导入 re 模块用于正则解析目录与文件名
import sys  # 导入 sys 模块用于读取命令行参数
import numpy as np  # 导入 numpy 用于数值计算与随机抖动
import pandas as pd  # 导入 pandas 用于表格读取与统计聚合
import matplotlib  # 导入 matplotlib 主模块用于后端设置
matplotlib.use('Agg')  # 设置无界面后端以支持批处理运行
import matplotlib.pyplot as plt  # 导入 pyplot 用于绘图
from matplotlib.lines import Line2D  # 导入 Line2D 用于自定义图例句柄
from matplotlib.patches import Patch  # 导入 Patch 用于柱状图图例句柄

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
DEFAULT_BATCH_ROOT = SCRIPT_DIR  # 设置默认数据根目录为脚本目录
OUTPUT_FIGURE_NAME = 'TAF_overview_grid.png'  # 定义总图输出文件名
OUTPUT_RAW_SUMMARY_NAME = 'TAF_overview_raw_summary.csv'  # 定义去重后原始汇总 CSV 文件名
OUTPUT_PANEL_SUMMARY_NAME = 'TAF_overview_panel_summary.csv'  # 定义分面统计汇总 CSV 文件名
OUTPUT_IMAGE_DIR_NAME = 'TAF_overview_figures'  # 定义图像统一输出子目录名称
PGA_GLOB_PREFIX = 'pga'  # 定义 PGA 文件名前缀（小写匹配）
TAF_GLOB_PREFIX = 'taf'  # 定义 TAF 文件名前缀（小写匹配）
TARGET_COLUMNS = ['PGA_h', 'PGA_v']  # 定义读取 PGA 时需要的列
RESAMPLE_STEP = 0.05  # 定义插值重采样步长
SAFE_DIVIDE_EPS = 1e-12  # 定义安全除法阈值防止分母趋近零
RANDOM_SEED = 20260426  # 定义散点抖动的随机种子以保证可复现
TREND_LINE_COLOR = '#ff2d2d'  # 定义平均值趋势线颜色
BACKGROUND_COLOR = '#efefef'  # 定义子图背景色
CASE_PATTERN = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_angle(?P<angle>-?\d+(?:\.\d+)?)')  # 定义参数目录名解析正则
MOTION_CONFIGS = [  # 定义三种地震动配置
    {'key': 'el_centro', 'token': 'elcentro', 'display': 'El-Centro'},  # 定义 El-Centro 配置
    {'key': 'northridge', 'token': 'northridge', 'display': 'Northridge'},  # 定义 Northridge 配置
    {'key': 'loma_prieta', 'token': 'lomaprieta', 'display': 'Loma-Prieta'},  # 定义 Loma-Prieta 配置
]  # 结束地震动配置定义
PREFERRED_ANGLE_VALUES = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # 定义优先角度显示顺序
PREFERRED_H_VALUES = [10.0, 50.0, 100.0, 200.0, 400.0]  # 定义优先 h 显示顺序
PREFERRED_I_VALUES = [15.0, 30.0, 45.0, 60.0, 75.0]  # 定义优先 i 显示顺序
MATCH_TOL = 1e-6  # 定义浮点匹配容差


def canonical_text(text):  # 定义文本规范化函数
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())  # 仅保留小写字母与数字用于关键词匹配


def parse_case_folder(folder_name):  # 定义参数目录名解析函数
    match = CASE_PATTERN.search(folder_name)  # 在目录名中匹配 h、i、angle 参数
    if match is None:  # 判断是否匹配失败
        return None  # 返回空值表示非目标参数目录
    h_value = float(match.group('h'))  # 提取并转换 h 参数
    i_value = float(match.group('i'))  # 提取并转换 i 参数
    angle_value = float(match.group('angle'))  # 提取并转换 angle 参数
    return h_value, i_value, angle_value  # 返回参数元组


def detect_motion(file_stem):  # 定义地震动类型识别函数
    normalized = canonical_text(file_stem)  # 规范化文件主干文本
    for motion_cfg in MOTION_CONFIGS:  # 遍历预定义地震动配置
        if motion_cfg['token'] in normalized:  # 判断当前文件是否包含地震动关键词
            return motion_cfg  # 返回命中的地震动配置
    return None  # 未命中时返回空值


def extract_taf_max_from_csv(csv_path):  # 定义从 TAF 文件提取 TAF 最大值的函数
    df = pd.read_csv(csv_path)  # 读取 CSV 数据表
    candidate_cols = ['TAF_h', 'TAF', 'TAF_max', 'TAFmax']  # 定义候选列优先级
    target_col = None  # 初始化目标列名为空
    for col_name in candidate_cols:  # 按优先级遍历候选列
        if col_name in df.columns:  # 判断当前候选列是否存在
            target_col = col_name  # 记录命中的目标列
            break  # 命中后结束循环
    if target_col is None:  # 判断是否未找到可用列
        raise ValueError('文件缺少 TAF 列: {}'.format(csv_path))  # 抛出缺列异常
    values = pd.to_numeric(df[target_col], errors='coerce').to_numpy(dtype=float)  # 将目标列安全转换为浮点数组
    values = values[np.isfinite(values)]  # 过滤非有限值
    if values.size == 0:  # 判断是否存在有效数值
        raise ValueError('文件无有效 TAF 数值: {}'.format(csv_path))  # 抛出无有效值异常
    return float(np.max(values))  # 返回当前文件的 TAF 最大值


def load_pga_dataframe(filepath, target_cols):  # 定义读取 PGA 文件并校验列的函数
    df = pd.read_csv(filepath)  # 读取 CSV 数据表
    required_cols = {'x/h'} | set(target_cols)  # 组装必须列集合
    missing_cols = required_cols - set(df.columns)  # 计算缺失列集合
    if missing_cols:  # 判断是否存在缺失列
        raise ValueError('文件缺少列 {} -> {}'.format(sorted(missing_cols), filepath))  # 抛出缺列异常
    df = df[['x/h'] + list(target_cols)].copy()  # 保留后续计算所需列
    df = df.sort_values(by='x/h')  # 按 x/h 升序排序
    return df  # 返回清洗后的数据表


def normalize_dataframe(df, target_cols, step):  # 定义按固定步长归一化插值函数
    grouped_df = df.groupby('x/h', as_index=False).mean(numeric_only=True)  # 对重复 x/h 节点取均值
    x_src = grouped_df['x/h'].to_numpy(dtype=float)  # 提取原始 x/h 数组
    x_start = float(np.min(x_src))  # 读取最小 x/h
    x_end = float(np.max(x_src))  # 读取最大 x/h
    x_norm = np.round(np.arange(x_start, x_end + step * 0.5, step), 10)  # 生成目标归一化横坐标
    norm_dict = {'x/h': x_norm}  # 初始化结果字典并写入 x/h
    for col in target_cols:  # 遍历目标分量列
        y_src = grouped_df[col].to_numpy(dtype=float)  # 提取当前分量原始值
        y_norm = np.interp(x_norm, x_src, y_src)  # 对当前分量执行线性插值
        norm_dict[col] = y_norm  # 写回当前分量插值结果
    return pd.DataFrame(norm_dict)  # 返回归一化后的数据表


def parse_pair_key(csv_stem):  # 定义解析 normal/flat 配对键的函数
    is_slope = csv_stem.endswith('-slope')  # 判断是否为 slope 命名
    is_flat = csv_stem.endswith('-flat')  # 判断是否为 flat 命名
    if is_slope:  # 判断是否命中 slope 后缀
        base_key = csv_stem[:-6]  # 去掉 slope 后缀得到配对键
    elif is_flat:  # 判断是否命中 flat 后缀
        base_key = csv_stem[:-5]  # 去掉 flat 后缀得到配对键
    else:  # 当前未命中后缀时执行兼容逻辑
        base_key = csv_stem  # 直接使用主干作为配对键
        is_slope = True  # 将无后缀文件视为 slope 文件
    return base_key, is_flat, is_slope  # 返回配对键与类型标识


def extract_motion_name(csv_stem):  # 定义从文件主干提取地震动名的函数
    cleaned_name = re.sub(r'^PGA[_-]*', '', csv_stem, flags=re.IGNORECASE)  # 去除 PGA 前缀
    cleaned_name = re.sub(r'^job-', '', cleaned_name, flags=re.IGNORECASE)  # 去除 job- 前缀
    cleaned_name = re.sub(r'-(?:slope|flat)$', '', cleaned_name, flags=re.IGNORECASE)  # 去除 slope/flat 后缀
    cleaned_name = re.sub(r'_(?:scaled|veled)$', '', cleaned_name, flags=re.IGNORECASE)  # 去除下划线版 scaled/veled 后缀
    cleaned_name = re.sub(r'-(?:scaled|veled)$', '', cleaned_name, flags=re.IGNORECASE)  # 去除连字符版 scaled/veled 后缀
    return cleaned_name  # 返回清洗后的波名文本


def build_pairs(pga_csv_paths):  # 定义构建 normal/flat 完整配对的函数
    pairs = {}  # 初始化配对字典
    for filepath in pga_csv_paths:  # 遍历 PGA 文件路径
        stem = os.path.splitext(os.path.basename(filepath))[0]  # 提取文件主干
        base_key, is_flat, is_slope = parse_pair_key(stem)  # 解析配对键与类型
        if base_key not in pairs:  # 判断当前配对键是否首次出现
            pairs[base_key] = {'normal': None, 'flat': None, 'motion': extract_motion_name(base_key)}  # 初始化配对槽位
        if is_flat:  # 判断是否为 flat 文件
            pairs[base_key]['flat'] = filepath  # 写入 flat 文件路径
        elif is_slope:  # 判断是否为 slope 文件
            pairs[base_key]['normal'] = filepath  # 写入 normal 文件路径
    valid_pairs = {}  # 初始化完整配对字典
    for base_key, item in pairs.items():  # 遍历全部配对槽位
        if item['normal'] is None or item['flat'] is None:  # 判断是否缺少 normal 或 flat
            continue  # 跳过不完整配对
        valid_pairs[base_key] = item  # 收集完整配对
    return valid_pairs  # 返回完整配对字典


def compute_taf_dataframe(normalized_normal_df, normalized_flat_df):  # 定义计算 TAF_h 数据表的函数
    x_normal = normalized_normal_df['x/h'].to_numpy(dtype=float)  # 读取 normal 归一化横坐标
    x_flat = normalized_flat_df['x/h'].to_numpy(dtype=float)  # 读取 flat 归一化横坐标
    if len(x_normal) != len(x_flat) or (not np.allclose(x_normal, x_flat)):  # 校验横坐标一致性
        raise ValueError('normal 与 flat 的归一化 x/h 不一致，无法计算 TAF')  # 抛出坐标不一致异常
    numerator = normalized_normal_df['PGA_h'].to_numpy(dtype=float)  # 读取 normal 水平向分子
    denominator = normalized_flat_df['PGA_h'].to_numpy(dtype=float)  # 读取 flat 水平向分母
    valid_mask = np.abs(denominator) > SAFE_DIVIDE_EPS  # 构造有效除法掩码
    taf_values = np.full_like(numerator, np.nan, dtype=float)  # 初始化 TAF 结果数组为 NaN
    taf_values[valid_mask] = numerator[valid_mask] / denominator[valid_mask]  # 在有效位置执行逐点相除
    return pd.DataFrame({'x/h': x_normal, 'TAF_h': taf_values})  # 返回 TAF_h 数据表


def ensure_taf_csvs_in_folder(folder_path):  # 定义确保目录内存在 TAF 文件的函数
    taf_paths = []  # 初始化 TAF 文件路径列表
    for file_name in sorted(os.listdir(folder_path)):  # 遍历目录内文件名
        lower_name = file_name.lower()  # 转为小写便于判断
        if (not lower_name.endswith('.csv')) or (not lower_name.startswith(TAF_GLOB_PREFIX)):  # 判断是否为 TAF CSV
            continue  # 跳过非 TAF CSV
        if 'normalized' in lower_name:  # 判断是否为归一化中间文件
            continue  # 跳过中间文件
        taf_paths.append(os.path.join(folder_path, file_name))  # 记录现有 TAF 文件路径
    if taf_paths:  # 判断是否已存在 TAF 文件
        return taf_paths  # 直接返回现有 TAF 路径列表
    pga_paths = []  # 初始化 PGA 文件路径列表
    for file_name in sorted(os.listdir(folder_path)):  # 遍历目录内文件名
        lower_name = file_name.lower()  # 转为小写便于判断
        if (not lower_name.endswith('.csv')) or (not lower_name.startswith(PGA_GLOB_PREFIX)):  # 判断是否为 PGA CSV
            continue  # 跳过非 PGA CSV
        if 'normalized' in lower_name:  # 判断是否为归一化中间文件
            continue  # 跳过中间文件
        pga_paths.append(os.path.join(folder_path, file_name))  # 记录 PGA 文件路径
    if not pga_paths:  # 判断是否不存在 PGA 文件
        return []  # 返回空列表表示无法补算
    pairs = build_pairs(pga_paths)  # 根据 PGA 文件构建完整配对
    if not pairs:  # 判断是否存在完整配对
        return []  # 返回空列表表示无法补算
    generated_taf_paths = []  # 初始化新生成 TAF 路径列表
    for _, pair_info in sorted(pairs.items()):  # 遍历完整配对并按键排序
        normal_df = load_pga_dataframe(pair_info['normal'], TARGET_COLUMNS)  # 读取 normal 数据
        flat_df = load_pga_dataframe(pair_info['flat'], TARGET_COLUMNS)  # 读取 flat 数据
        normal_norm_df = normalize_dataframe(normal_df, TARGET_COLUMNS, RESAMPLE_STEP)  # 归一化 normal 数据
        flat_norm_df = normalize_dataframe(flat_df, TARGET_COLUMNS, RESAMPLE_STEP)  # 归一化 flat 数据
        taf_df = compute_taf_dataframe(normal_norm_df, flat_norm_df)  # 计算当前配对 TAF_h
        out_name = 'TAF-{}.csv'.format(pair_info['motion'])  # 组装输出 TAF 文件名
        out_path = os.path.join(folder_path, out_name)  # 组装输出路径
        taf_df.to_csv(out_path, index=False, encoding='utf-8-sig')  # 保存 TAF 文件
        generated_taf_paths.append(out_path)  # 记录新生成文件路径
    return generated_taf_paths  # 返回新生成 TAF 文件路径列表


def collect_records(batch_root):  # 定义扫描目录并收集记录的函数
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
        taf_paths = ensure_taf_csvs_in_folder(folder_path)  # 确保目录内存在可用 TAF 文件
        for csv_path in taf_paths:  # 遍历目录内 TAF 文件
            file_name = os.path.basename(csv_path)  # 读取文件名
            file_stem = os.path.splitext(file_name)[0]  # 读取文件主干
            motion_cfg = detect_motion(file_stem)  # 识别地震动类型
            if motion_cfg is None:  # 判断是否为目标地震动
                continue  # 跳过未识别波名文件
            try:  # 尝试提取 TAF 最大值
                taf_max = extract_taf_max_from_csv(csv_path)  # 读取并计算 TAF 最大值
            except Exception as exc:  # 捕获读取异常
                print('警告：已跳过 {} -> {}'.format(csv_path, str(exc)))  # 输出跳过信息
                continue  # 跳过当前异常文件
            records.append({'motion': motion_cfg['key'], 'motion_display': motion_cfg['display'], 'h': float(h_value), 'i': float(i_value), 'angle': float(angle_value), 'taf_max': float(taf_max), 'file': csv_path})  # 追加结构化记录
    if not records:  # 判断是否采集到有效记录
        raise FileNotFoundError('未采集到有效 TAF 数据，请检查目录与文件命名。')  # 抛出无数据异常
    return records  # 返回记录列表


def sort_by_preferred(values, preferred_values):  # 定义按优先序排列数值的函数
    unique_vals = sorted({float(v) for v in values})  # 获取去重升序数值集合
    ordered = []  # 初始化排序结果列表
    used = set()  # 初始化已使用值索引集合
    for pref in preferred_values:  # 遍历优先值序列
        for idx, value in enumerate(unique_vals):  # 遍历唯一值列表
            if idx in used:  # 判断当前值是否已被使用
                continue  # 跳过已使用值
            if abs(value - float(pref)) <= MATCH_TOL:  # 判断是否命中优先值
                ordered.append(value)  # 追加匹配值到结果列表
                used.add(idx)  # 标记当前值已使用
                break  # 当前优先值命中后结束内层循环
    for idx, value in enumerate(unique_vals):  # 再次遍历唯一值列表
        if idx in used:  # 判断当前值是否已被使用
            continue  # 跳过已使用值
        ordered.append(value)  # 将剩余值按升序追加到末尾
    return ordered  # 返回最终排序结果


def build_group_stats(raw_df, group_col, preferred_values):  # 定义构建分组统计表的函数
    if raw_df.empty:  # 判断输入是否为空
        return pd.DataFrame(columns=[group_col, 'taf_avg', 'taf_min', 'taf_max', 'sample_count', 'raw_values'])  # 返回空结构表
    grouped = raw_df.groupby(group_col)['taf_max'].apply(list).reset_index(name='raw_values')  # 按分组列收集原始值列表
    grouped[group_col] = grouped[group_col].astype(float)  # 将分组列转为浮点数
    grouped['taf_avg'] = grouped['raw_values'].apply(lambda values: float(np.mean(values)))  # 计算均值列
    grouped['taf_min'] = grouped['raw_values'].apply(lambda values: float(np.min(values)))  # 计算最小值列
    grouped['taf_max'] = grouped['raw_values'].apply(lambda values: float(np.max(values)))  # 计算最大值列
    grouped['sample_count'] = grouped['raw_values'].apply(lambda values: int(len(values)))  # 计算样本数量列
    ordered_values = sort_by_preferred(grouped[group_col].tolist(), preferred_values)  # 生成分组值显示顺序
    order_map = {value: idx for idx, value in enumerate(ordered_values)}  # 构建顺序索引映射
    grouped['_order'] = grouped[group_col].apply(lambda value: order_map.get(float(value), 999999))  # 写入排序辅助列
    grouped = grouped.sort_values(by=['_order', group_col]).reset_index(drop=True)  # 按顺序列与分组列排序
    grouped = grouped.drop(columns=['_order'])  # 删除排序辅助列
    return grouped  # 返回分组统计表


def compute_plot_range(y_values, y_floor):  # 定义计算纵轴范围的函数
    finite_values = np.asarray(y_values, dtype=float)  # 将输入转为浮点数组
    finite_values = finite_values[np.isfinite(finite_values)]  # 过滤非有限值
    if finite_values.size == 0:  # 判断是否无有效数据
        return y_floor, y_floor + 1.0  # 返回默认范围
    y_max = float(np.max(finite_values))  # 读取有效数据最大值
    y_min_ref = float(y_floor)  # 读取固定下界参考值
    span = max(y_max - y_min_ref, 0.25)  # 计算跨度并设定最小跨度
    y_upper = y_max + 0.28 * span  # 计算上界并预留显示空间
    return y_min_ref, y_upper  # 返回纵轴下界与上界


def draw_stat_panel(ax, stats_df, x_col, xlabel_text, title_text, y_floor, show_ylabel=False, show_legend=False, use_equal_spacing=False, force_y_limits=None):  # 定义统计子图绘制函数
    ax.set_facecolor(BACKGROUND_COLOR)  # 设置子图背景色
    for spine in ax.spines.values():  # 遍历坐标轴边框
        spine.set_linewidth(1.0)  # 设置边框线宽
        spine.set_color('#333333')  # 设置边框颜色
    ax.tick_params(direction='in', top=True, right=True, labelsize=11, width=0.9)  # 设置刻度样式
    if stats_df.empty:  # 判断当前子图是否无数据
        ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, ha='center', va='center', fontsize=12, color='#555555')  # 在图中央标注无数据
        ax.set_title(title_text, fontsize=14, pad=6)  # 设置标题
        ax.set_xlabel(xlabel_text, fontsize=14)  # 设置横轴标签
        if show_ylabel:  # 判断是否显示纵轴标签
            ax.set_ylabel(r'$TAF_{\max}$', fontsize=14)  # 设置纵轴标签
        return  # 无数据时结束绘制
    x_values = stats_df[x_col].to_numpy(dtype=float)  # 读取横坐标值数组
    if use_equal_spacing:  # 判断是否启用等间距类别坐标
        x_plot = np.arange(len(x_values), dtype=float)  # 为当前分组生成等间距绘图坐标
    else:  # 未启用等间距时按实际数值坐标绘图
        x_plot = x_values  # 直接使用实际横坐标值
    y_avg = stats_df['taf_avg'].to_numpy(dtype=float)  # 读取均值数组
    y_min = stats_df['taf_min'].to_numpy(dtype=float)  # 读取最小值数组
    y_max = stats_df['taf_max'].to_numpy(dtype=float)  # 读取最大值数组
    sorted_x = np.sort(np.unique(x_plot))  # 计算去重升序绘图坐标
    if sorted_x.size >= 2:  # 判断横坐标数量是否至少为 2
        min_step = float(np.min(np.diff(sorted_x)))  # 计算最小步长
    else:  # 当前仅有一个横坐标时走兜底
        min_step = 1.0  # 设定默认步长
    bar_width = max(0.18 * min_step, 0.4)  # 计算柱宽并设定最小值
    jitter_span = 0.22 * min_step  # 计算散点抖动幅度
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(x_values), 3)))  # 生成颜色序列
    for x_val in x_plot:  # 遍历每个绘图横坐标
        ax.axvline(x=x_val, color='#bdbdbd', linestyle=(0, (2.5, 2.5)), linewidth=0.9, zorder=0)  # 绘制竖向虚线参考线
    for idx, x_val in enumerate(x_plot):  # 遍历每个统计点
        color = colors[idx]  # 读取当前点颜色
        ax.bar(x_val, y_avg[idx], width=bar_width, color=color, edgecolor=color, linewidth=1.4, alpha=0.30, zorder=2)  # 绘制均值柱
        yerr_lower = max(0.0, y_avg[idx] - y_min[idx])  # 计算下侧误差
        yerr_upper = max(0.0, y_max[idx] - y_avg[idx])  # 计算上侧误差
        ax.errorbar(x_val, y_avg[idx], yerr=np.array([[yerr_lower], [yerr_upper]]), fmt='none', ecolor=color, elinewidth=1.6, capsize=4.5, zorder=4)  # 绘制最小-最大误差线
    ax.plot(x_plot, y_max, color='#9a9a9a', linestyle=(0, (3, 2)), linewidth=1.2, zorder=3)  # 绘制最大值灰色虚线
    ax.plot(x_plot, y_min, color='#9a9a9a', linestyle=(0, (3, 2)), linewidth=1.2, zorder=3)  # 绘制最小值灰色虚线
    rng = np.random.RandomState(RANDOM_SEED)  # 初始化可复现随机数生成器
    for idx in range(len(stats_df)):  # 遍历每个分组行
        raw_values = np.asarray(stats_df.iloc[idx]['raw_values'], dtype=float)  # 读取当前分组原始样本
        if raw_values.size == 0:  # 判断当前分组是否无样本
            continue  # 跳过空样本分组
        jitter = (rng.rand(raw_values.size) - 0.5) * 2.0 * jitter_span  # 计算散点横向抖动
        scatter_x = np.full(raw_values.size, x_plot[idx]) + jitter  # 生成散点横坐标数组
        ax.scatter(scatter_x, raw_values, s=12, facecolors='none', edgecolors=[colors[idx]], linewidths=0.8, alpha=0.35, zorder=1)  # 绘制空心散点
    ax.plot(x_plot, y_avg, color=TREND_LINE_COLOR, linewidth=1.9, zorder=5)  # 绘制均值红色趋势线
    if force_y_limits is None:  # 判断是否传入强制纵轴范围
        y_lower, y_upper = compute_plot_range(np.concatenate((y_min, y_max)), y_floor)  # 未传入时按当前子图数据自动计算纵轴范围
    else:  # 已传入强制纵轴范围时直接使用
        y_lower, y_upper = force_y_limits  # 读取统一纵轴下界与上界
    ax.set_ylim(y_lower, y_upper)  # 设置纵轴范围
    ax.set_xticks(x_plot)  # 设置横轴刻度位置
    ax.set_xticklabels([('{:g}'.format(value)) for value in x_values], fontsize=11)  # 设置横轴刻度文本
    ax.set_title(title_text, fontsize=14, pad=5)  # 设置子图标题
    ax.set_xlabel(xlabel_text, fontsize=14)  # 设置横轴标签
    if show_ylabel:  # 判断是否显示纵轴标签
        ax.set_ylabel(r'$TAF_{\max}$', fontsize=14)  # 设置纵轴标签
    if show_legend:  # 判断是否显示图例
        legend_handles = [  # 定义图例句柄列表
            Patch(facecolor='#c9c9c9', edgecolor='#7a7a7a', alpha=0.45, label='Average'),  # 定义均值柱图例句柄
            Line2D([0], [0], color='#7a7a7a', linestyle='-', marker='|', markersize=11, linewidth=1.2, label='Min.~Max.'),  # 定义最小最大图例句柄
            Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='none', markeredgecolor='#9e9e9e', markersize=4.0, label='Data'),  # 定义散点图例句柄
        ]  # 结束图例句柄定义
        ax.legend(handles=legend_handles, loc='upper left', frameon=True, fancybox=False, edgecolor='#8a8a8a', fontsize=11)  # 绘制图例


def build_panel_summary_rows(stats_df, panel_name, x_name, motion_display='All'):  # 定义构建分面汇总行的函数
    rows = []  # 初始化汇总行列表
    for _, row in stats_df.iterrows():  # 遍历统计表每一行
        rows.append({'panel': panel_name, 'motion': motion_display, x_name: float(row[x_name]), 'taf_avg': float(row['taf_avg']), 'taf_min': float(row['taf_min']), 'taf_max': float(row['taf_max']), 'sample_count': int(row['sample_count'])})  # 追加结构化汇总行
    return rows  # 返回汇总行列表


def create_overview_figure(summary_df, output_path):  # 定义创建总图的函数
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
    fig = plt.figure(figsize=(13.2, 10.2), facecolor='white')  # 创建总图画布并加宽以减少上排遮挡
    root_grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.36)  # 定义上下两行网格
    top_grid = root_grid[0].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.26)  # 定义上排三图等宽网格并增大横向间距
    bottom_grid = root_grid[1].subgridspec(1, 2, wspace=0.15)  # 定义下排两图网格
    ax_top_1 = fig.add_subplot(top_grid[0, 0])  # 创建上排第一个子图
    ax_top_2 = fig.add_subplot(top_grid[0, 1])  # 创建上排第二个子图
    ax_top_3 = fig.add_subplot(top_grid[0, 2])  # 创建上排第三个子图
    ax_bottom_1 = fig.add_subplot(bottom_grid[0, 0])  # 创建下排第一个子图
    ax_bottom_2 = fig.add_subplot(bottom_grid[0, 1])  # 创建下排第二个子图
    top_axes = [ax_top_1, ax_top_2, ax_top_3]  # 组装上排坐标轴列表
    panel_summary_rows = []  # 初始化分面汇总行容器
    top_stats_list = []  # 初始化上排三图统计表缓存列表
    top_all_values = []  # 初始化上排三图纵轴数据汇总列表
    for motion_cfg in MOTION_CONFIGS:  # 按顺序遍历三种地震动并先缓存统计表
        panel_df = summary_df[summary_df['motion'] == motion_cfg['key']].copy()  # 提取当前地震动数据
        stats_df = build_group_stats(panel_df, 'angle', PREFERRED_ANGLE_VALUES)  # 构建按 angle 统计表
        top_stats_list.append(stats_df)  # 缓存当前子图统计表
        if not stats_df.empty:  # 判断当前统计表是否非空
            top_all_values.extend(stats_df['taf_min'].tolist())  # 汇总当前子图最小值序列
            top_all_values.extend(stats_df['taf_max'].tolist())  # 汇总当前子图最大值序列
    top_y_limits = compute_plot_range(np.asarray(top_all_values, dtype=float), 0.5)  # 基于上排全部数据计算统一纵轴范围
    for idx, motion_cfg in enumerate(MOTION_CONFIGS):  # 按顺序遍历三种地震动执行绘图
        stats_df = top_stats_list[idx]  # 读取当前地震动缓存统计表
        draw_stat_panel(top_axes[idx], stats_df, 'angle', r'$\theta_s (^{\circ})$', motion_cfg['display'], y_floor=0.5, show_ylabel=True, show_legend=True, force_y_limits=top_y_limits)  # 绘制当前上排子图并统一显示纵轴标签与图例
        panel_summary_rows.extend(build_panel_summary_rows(stats_df, panel_name='theta', x_name='angle', motion_display=motion_cfg['display']))  # 追加当前子图汇总
    h_stats_df = build_group_stats(summary_df, 'h', PREFERRED_H_VALUES)  # 构建按 h 统计表
    draw_stat_panel(ax_bottom_1, h_stats_df, 'h', r'$h\,(m)$', '', y_floor=0.0, show_ylabel=True, show_legend=True, use_equal_spacing=True)  # 绘制下排左图并启用等间距横坐标
    panel_summary_rows.extend(build_panel_summary_rows(h_stats_df, panel_name='h', x_name='h', motion_display='All'))  # 追加 h 分面汇总
    i_stats_df = build_group_stats(summary_df, 'i', PREFERRED_I_VALUES)  # 构建按 i 统计表
    draw_stat_panel(ax_bottom_2, i_stats_df, 'i', r'$i\,(^{\circ})$', '', y_floor=0.0, show_ylabel=True, show_legend=True)  # 绘制下排右图
    panel_summary_rows.extend(build_panel_summary_rows(i_stats_df, panel_name='i', x_name='i', motion_display='All'))  # 追加 i 分面汇总
    fig.text(0.50, 0.51, '(a)', ha='center', va='center', fontsize=17)  # 添加上排标注
    fig.text(0.29, 0.03, '(b)', ha='center', va='center', fontsize=17)  # 添加下排左图标注
    fig.text(0.74, 0.03, '(c)', ha='center', va='center', fontsize=17)  # 添加下排右图标注
    fig.savefig(output_path, dpi=350)  # 保存总图到输出路径
    plt.close(fig)  # 关闭图对象释放内存
    return panel_summary_rows  # 返回分面汇总行数据


def create_single_panel_figures(summary_df, output_dir):  # 定义导出每个子图为单图的函数
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
    os.makedirs(output_dir, exist_ok=True)  # 创建单图输出目录并允许已存在
    output_paths = []  # 初始化输出路径列表
    top_stats_list = []  # 初始化 theta 单图统计表缓存列表
    top_all_values = []  # 初始化 theta 单图纵轴数据汇总列表
    for motion_cfg in MOTION_CONFIGS:  # 遍历三种地震动并先缓存统计表
        panel_df = summary_df[summary_df['motion'] == motion_cfg['key']].copy()  # 提取当前地震动数据
        stats_df = build_group_stats(panel_df, 'angle', PREFERRED_ANGLE_VALUES)  # 构建按 angle 统计表
        top_stats_list.append(stats_df)  # 缓存当前统计表
        if not stats_df.empty:  # 判断当前统计表是否非空
            top_all_values.extend(stats_df['taf_min'].tolist())  # 汇总当前统计表最小值序列
            top_all_values.extend(stats_df['taf_max'].tolist())  # 汇总当前统计表最大值序列
    top_y_limits = compute_plot_range(np.asarray(top_all_values, dtype=float), 0.5)  # 计算与(a)图一致的统一纵轴范围
    for idx, motion_cfg in enumerate(MOTION_CONFIGS):  # 遍历三种地震动并逐一导出 theta 子图
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2), facecolor='white')  # 创建单图画布
        stats_df = top_stats_list[idx]  # 读取当前地震动缓存统计表
        draw_stat_panel(ax, stats_df, 'angle', r'$\theta_s (^{\circ})$', motion_cfg['display'], y_floor=0.5, show_ylabel=True, show_legend=True, force_y_limits=top_y_limits)  # 绘制当前地震动单图并统一纵轴范围
        fig.tight_layout()  # 自动调整布局避免遮挡
        output_name = 'TAF_overview_theta_{}.png'.format(motion_cfg['key'])  # 组装当前 theta 单图文件名
        output_path = os.path.join(output_dir, output_name)  # 组装当前 theta 单图路径
        fig.savefig(output_path, dpi=350)  # 保存当前 theta 单图
        plt.close(fig)  # 关闭当前图对象释放内存
        output_paths.append(output_path)  # 记录当前输出路径
    fig_h, ax_h = plt.subplots(1, 1, figsize=(5.0, 4.2), facecolor='white')  # 创建 h 单图画布
    h_stats_df = build_group_stats(summary_df, 'h', PREFERRED_H_VALUES)  # 构建按 h 统计表
    draw_stat_panel(ax_h, h_stats_df, 'h', r'$h\,(m)$', '', y_floor=0.0, show_ylabel=True, show_legend=True, use_equal_spacing=True)  # 绘制 h 单图并启用等间距横坐标
    fig_h.tight_layout()  # 自动调整 h 图布局
    output_h_path = os.path.join(output_dir, 'TAF_overview_h_v1.png')  # 组装 h 单图路径
    fig_h.savefig(output_h_path, dpi=350)  # 保存 h 单图
    plt.close(fig_h)  # 关闭 h 图对象释放内存
    output_paths.append(output_h_path)  # 记录 h 图路径
    fig_i, ax_i = plt.subplots(1, 1, figsize=(5.0, 4.2), facecolor='white')  # 创建 i 单图画布
    i_stats_df = build_group_stats(summary_df, 'i', PREFERRED_I_VALUES)  # 构建按 i 统计表
    draw_stat_panel(ax_i, i_stats_df, 'i', r'$i\,(^{\circ})$', '', y_floor=0.0, show_ylabel=True, show_legend=True)  # 绘制 i 单图
    fig_i.tight_layout()  # 自动调整 i 图布局
    output_i_path = os.path.join(output_dir, 'TAF_overview_i_v1.png')  # 组装 i 单图路径
    fig_i.savefig(output_i_path, dpi=350)  # 保存 i 单图
    plt.close(fig_i)  # 关闭 i 图对象释放内存
    output_paths.append(output_i_path)  # 记录 i 图路径
    return output_paths  # 返回全部单图输出路径列表


def main():  # 定义主函数
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
    records = collect_records(batch_root)  # 扫描并采集全部记录
    raw_df = pd.DataFrame(records)  # 将记录列表转换为数据表
    summary_df = raw_df.groupby(['motion', 'motion_display', 'h', 'i', 'angle'], as_index=False)['taf_max'].mean()  # 对重复工况取均值去重
    summary_df = summary_df.sort_values(by=['motion', 'angle', 'h', 'i']).reset_index(drop=True)  # 对汇总结果排序
    output_raw_csv = os.path.join(output_dir, OUTPUT_RAW_SUMMARY_NAME)  # 组装原始汇总 CSV 路径
    summary_df.to_csv(output_raw_csv, index=False, encoding='utf-8-sig')  # 保存原始汇总 CSV
    output_figure = os.path.join(output_image_dir, OUTPUT_FIGURE_NAME)  # 组装总图输出路径
    panel_rows = create_overview_figure(summary_df, output_figure)  # 生成总图并获取分面汇总行
    output_single_dir = output_image_dir  # 将单图输出目录设置为与总图相同目录
    single_paths = create_single_panel_figures(summary_df, output_single_dir)  # 导出每个子图的单图文件
    panel_summary_df = pd.DataFrame(panel_rows)  # 将分面汇总行转为数据表
    output_panel_csv = os.path.join(output_dir, OUTPUT_PANEL_SUMMARY_NAME)  # 组装分面汇总 CSV 路径
    panel_summary_df.to_csv(output_panel_csv, index=False, encoding='utf-8-sig')  # 保存分面汇总 CSV
    print('数据目录: {}'.format(batch_root))  # 输出数据目录信息
    print('原始汇总: {}'.format(output_raw_csv))  # 输出原始汇总 CSV 路径
    print('分面汇总: {}'.format(output_panel_csv))  # 输出分面汇总 CSV 路径
    print('图片目录: {}'.format(output_image_dir))  # 输出图片目录信息
    print('总图输出: {}'.format(output_figure))  # 输出总图路径
    print('单图目录: {}'.format(output_single_dir))  # 输出单图目录路径
    print('单图数量: {}'.format(len(single_paths)))  # 输出单图数量统计
    for single_path in single_paths:  # 遍历单图路径列表
        print('单图输出: {}'.format(single_path))  # 输出当前单图路径


if __name__ == '__main__':  # 判断脚本是否以主程序方式运行
    main()  # 调用主函数执行流程