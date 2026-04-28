# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
import os  # 导入 os 模块用于路径与目录操作
import re  # 导入 re 模块用于参数目录名解析
import sys  # 导入 sys 模块用于读取命令行参数
import numpy as np  # 导入 numpy 用于数值与缺失值处理
import pandas as pd  # 导入 pandas 用于 CSV 读取与汇总
import matplotlib  # 导入 matplotlib 主模块用于设置后端
matplotlib.use('Agg')  # 设置无界面后端以支持批处理环境
import matplotlib.pyplot as plt  # 导入 pyplot 用于绘图

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本目录
DEFAULT_BATCH_ROOT = SCRIPT_DIR  # 设置默认数据根目录为脚本所在目录
OUTPUT_CSV_NAME = 'TAF_max_vs_theta_summary.csv'  # 定义输出汇总 CSV 文件名
OUTPUT_GRID_PNG_NAME = 'TAF_max_vs_theta_grid.png'  # 定义输出汇总大图文件名
OUTPUT_FIG_DIR_NAME = '3-TAF_theta'  # 定义单图输出目录名
OUTPUT_FIG_PREFIX = 'TAF_max_theta'  # 定义单图输出文件名前缀
PGA_GLOB_PREFIX = 'pga'  # 定义 PGA 文件名前缀（小写）
TAF_GLOB_PREFIX = 'taf'  # 定义 TAF 文件名前缀（小写）
TARGET_COLUMNS = ['PGA_h', 'PGA_v']  # 定义 PGA 读取分量列
RESAMPLE_STEP = 0.05  # 定义归一化插值步长
SAFE_DIVIDE_EPS = 1e-12  # 定义安全除法阈值
TARGET_H_VALUES = [50.0, 100.0]  # 定义图中两行对应的目标 h 值
TARGET_I_VALUES = [30.0, 45.0, 60.0, 75.0]  # 定义图中四条曲线对应的目标 i 值
TARGET_ANGLES = [0.0, 10.0, 20.0, 30.0]  # 定义横轴目标入射角序列
ANGLE_TOL = 1e-6  # 定义角度匹配容差
HI_TOL = 1e-6  # 定义 h 与 i 匹配容差
FOLDER_PATTERN = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_angle(?P<angle>-?\d+(?:\.\d+)?)')  # 定义参数目录名解析正则
MOTION_CONFIGS = [  # 定义三列地震波配置
    {'key': 'el_centro', 'token': 'elcentro', 'display': 'El Centro'},  # 定义 El Centro 配置
    {'key': 'loma_prieta', 'token': 'lomaprieta', 'display': 'Loma Prieta'},  # 定义 Loma Prieta 配置
    {'key': 'northridge', 'token': 'northridge', 'display': 'Northridge'},  # 定义 Northridge 配置
]  # 结束地震波配置列表
I_STYLE = {  # 定义不同坡角曲线样式
    30.0: {'color': '#984ea3', 'marker': '^', 'label': r'$i$ = 30°'},  # 定义 i=30 的样式
    45.0: {'color': '#4daf4a', 'marker': 's', 'label': r'$i$ = 45°'},  # 定义 i=45 的样式
    60.0: {'color': '#377eb8', 'marker': 'o', 'label': r'$i$ = 60°'},  # 定义 i=60 的样式
    75.0: {'color': '#e41a1c', 'marker': 'v', 'label': r'$i$ = 75°'},  # 定义 i=75 的样式
}  # 结束样式字典


def canonical_text(text):  # 定义文本规范化函数
    return re.sub(r'[^a-z0-9]+', '', str(text).lower())  # 清洗为小写字母数字串便于匹配


def parse_case_folder(folder_name):  # 定义参数目录解析函数
    match = FOLDER_PATTERN.search(folder_name)  # 在目录名中匹配 h、i、angle
    if match is None:  # 判断是否匹配失败
        return None  # 返回空值表示非目标参数目录
    h_value = float(match.group('h'))  # 提取并转换 h 参数
    i_value = float(match.group('i'))  # 提取并转换 i 参数
    angle_value = float(match.group('angle'))  # 提取并转换 angle 参数
    return h_value, i_value, angle_value  # 返回解析结果元组


def detect_motion(file_stem):  # 定义地震波类型识别函数
    normalized = canonical_text(file_stem)  # 将文件名主干规范化用于关键词匹配
    for motion_cfg in MOTION_CONFIGS:  # 遍历地震波配置列表
        if motion_cfg['token'] in normalized:  # 判断是否包含当前地震波关键词
            return motion_cfg  # 返回命中的地震波配置
    return None  # 未命中时返回空值


def extract_taf_max_from_csv(csv_path):  # 定义从单个 TAF 文件提取 TAF_max 的函数
    df = pd.read_csv(csv_path)  # 读取 CSV 数据表
    candidate_cols = ['TAF_h', 'TAF', 'TAF_max', 'TAFmax']  # 定义候选 TAF 列优先级
    target_col = None  # 初始化目标列名为空
    for col_name in candidate_cols:  # 按优先级遍历候选列
        if col_name in df.columns:  # 判断当前候选列是否存在
            target_col = col_name  # 记录命中的目标列
            break  # 命中后结束遍历
    if target_col is None:  # 判断是否未找到可用列
        raise ValueError('文件缺少 TAF 列: {}'.format(csv_path))  # 抛出缺列异常
    values = pd.to_numeric(df[target_col], errors='coerce').to_numpy(dtype=float)  # 将目标列安全转换为浮点数组
    values = values[np.isfinite(values)]  # 过滤非有限值
    if values.size == 0:  # 判断有效数值是否为空
        raise ValueError('文件无有效 TAF 数值: {}'.format(csv_path))  # 抛出无有效值异常
    return float(np.max(values))  # 返回该文件的 TAF 最大值


def load_pga_dataframe(filepath, target_cols):  # 定义读取单个 PGA 文件的函数
    df = pd.read_csv(filepath)  # 读取 CSV 为数据表
    required_cols = {'x/h'} | set(target_cols)  # 组装必须存在的列集合
    missing_cols = required_cols - set(df.columns)  # 计算缺失列集合
    if missing_cols:  # 判断是否存在缺失列
        raise ValueError('文件缺少列 {} -> {}'.format(sorted(missing_cols), filepath))  # 抛出缺列异常
    df = df[['x/h'] + list(target_cols)].copy()  # 保留所需列并复制数据
    df = df.sort_values(by='x/h')  # 按 x/h 升序排序
    return df  # 返回清洗后的数据表


def normalize_dataframe(df, target_cols, step):  # 定义按固定步长归一化插值函数
    grouped_df = df.groupby('x/h', as_index=False).mean(numeric_only=True)  # 对重复 x/h 节点取均值
    x_src = grouped_df['x/h'].to_numpy(dtype=float)  # 提取原始 x/h 数组
    x_start = float(np.min(x_src))  # 读取最小 x/h
    x_end = float(np.max(x_src))  # 读取最大 x/h
    x_norm = np.round(np.arange(x_start, x_end + step * 0.5, step), 10)  # 生成固定步长目标坐标
    norm_dict = {'x/h': x_norm}  # 初始化归一化结果字典
    for col in target_cols:  # 遍历每个目标分量列
        y_src = grouped_df[col].to_numpy(dtype=float)  # 提取当前分量原始数组
        y_norm = np.interp(x_norm, x_src, y_src)  # 对当前分量执行线性插值
        norm_dict[col] = y_norm  # 写回插值结果
    return pd.DataFrame(norm_dict)  # 返回归一化数据表


def parse_pair_key(csv_stem):  # 定义解析 normal/flat 配对键的函数
    is_slope = csv_stem.endswith('-slope')  # 判断是否为 slope 命名
    is_flat = csv_stem.endswith('-flat')  # 判断是否为 flat 命名
    if is_slope:  # 判断是否命中 slope 后缀
        base_key = csv_stem[:-6]  # 去掉 slope 后缀得到配对键
    elif is_flat:  # 判断是否命中 flat 后缀
        base_key = csv_stem[:-5]  # 去掉 flat 后缀得到配对键
    else:  # 当前未命中后缀时回退兼容规则
        base_key = csv_stem  # 直接使用主干作为配对键
        is_slope = True  # 将该文件视为 slope 文件
    return base_key, is_flat, is_slope  # 返回配对键与类型标识


def extract_motion_name(csv_stem):  # 定义从主干提取地震波名称的函数
    cleaned_name = re.sub(r'^PGA[_-]*', '', csv_stem, flags=re.IGNORECASE)  # 去除 PGA 前缀
    cleaned_name = re.sub(r'^job-', '', cleaned_name, flags=re.IGNORECASE)  # 去除 job- 前缀
    cleaned_name = re.sub(r'-(?:slope|flat)$', '', cleaned_name, flags=re.IGNORECASE)  # 去除 slope/flat 后缀
    cleaned_name = re.sub(r'_(?:scaled|veled)$', '', cleaned_name, flags=re.IGNORECASE)  # 去除下划线后缀
    cleaned_name = re.sub(r'-(?:scaled|veled)$', '', cleaned_name, flags=re.IGNORECASE)  # 去除连字符后缀
    return cleaned_name  # 返回清洗后的波名


def build_pairs(pga_csv_paths):  # 定义构建 normal/flat 完整配对的函数
    pairs = {}  # 初始化配对字典
    for filepath in pga_csv_paths:  # 遍历每个 PGA 文件路径
        stem = os.path.splitext(os.path.basename(filepath))[0]  # 提取文件主干
        base_key, is_flat, is_slope = parse_pair_key(stem)  # 解析配对键与类型标识
        if base_key not in pairs:  # 判断当前配对键是否首次出现
            pairs[base_key] = {'normal': None, 'flat': None, 'motion': extract_motion_name(base_key)}  # 初始化当前配对槽位
        if is_flat:  # 判断当前文件是否为 flat
            pairs[base_key]['flat'] = filepath  # 写入 flat 路径
        elif is_slope:  # 判断当前文件是否为 slope
            pairs[base_key]['normal'] = filepath  # 写入 normal 路径
    valid_pairs = {}  # 初始化完整配对字典
    for base_key, item in pairs.items():  # 遍历全部配对槽位
        if item['normal'] is None or item['flat'] is None:  # 判断是否缺少 normal 或 flat
            continue  # 跳过不完整配对
        valid_pairs[base_key] = item  # 收集完整配对
    return valid_pairs  # 返回完整配对结果


def compute_taf_dataframe(normalized_normal_df, normalized_flat_df):  # 定义计算 TAF_h 数据表的函数
    x_normal = normalized_normal_df['x/h'].to_numpy(dtype=float)  # 读取 normal 归一化横坐标
    x_flat = normalized_flat_df['x/h'].to_numpy(dtype=float)  # 读取 flat 归一化横坐标
    if len(x_normal) != len(x_flat) or (not np.allclose(x_normal, x_flat)):  # 校验两组横坐标是否一致
        raise ValueError('normal 与 flat 的归一化 x/h 不一致，无法计算 TAF')  # 抛出坐标不一致异常
    denominator = normalized_flat_df['PGA_h'].to_numpy(dtype=float)  # 读取 flat 水平分量作为分母
    numerator = normalized_normal_df['PGA_h'].to_numpy(dtype=float)  # 读取 normal 水平分量作为分子
    valid_mask = np.abs(denominator) > SAFE_DIVIDE_EPS  # 构造分母非零掩码
    taf_values = np.full_like(numerator, np.nan, dtype=float)  # 初始化 TAF_h 数组为 NaN
    taf_values[valid_mask] = numerator[valid_mask] / denominator[valid_mask]  # 在有效位置执行相除
    return pd.DataFrame({'x/h': x_normal, 'TAF_h': taf_values})  # 返回 TAF_h 数据表


def ensure_taf_csvs_in_folder(folder_path):  # 定义确保目录内存在 TAF 文件的函数
    taf_paths = []  # 初始化 TAF 文件路径列表
    for file_name in sorted(os.listdir(folder_path)):  # 遍历目录内全部文件名
        lower_name = file_name.lower()  # 获取小写文件名用于判断
        if (not lower_name.endswith('.csv')) or (not lower_name.startswith(TAF_GLOB_PREFIX)):  # 判断是否为 TAF CSV
            continue  # 跳过非 TAF CSV 文件
        if 'normalized' in lower_name:  # 判断是否为归一化中间文件
            continue  # 跳过中间文件
        taf_paths.append(os.path.join(folder_path, file_name))  # 记录现有 TAF 路径
    if taf_paths:  # 判断是否已存在 TAF 文件
        return taf_paths  # 直接返回现有 TAF 列表
    pga_paths = []  # 初始化 PGA 文件路径列表
    for file_name in sorted(os.listdir(folder_path)):  # 遍历目录内全部文件名
        lower_name = file_name.lower()  # 获取小写文件名用于判断
        if (not lower_name.endswith('.csv')) or (not lower_name.startswith(PGA_GLOB_PREFIX)):  # 判断是否为 PGA CSV
            continue  # 跳过非 PGA CSV
        if 'normalized' in lower_name:  # 判断是否为归一化中间文件
            continue  # 跳过中间文件
        pga_paths.append(os.path.join(folder_path, file_name))  # 记录 PGA 路径
    if not pga_paths:  # 判断是否不存在 PGA 文件
        return []  # 返回空列表表示无法补算
    pairs = build_pairs(pga_paths)  # 构建 normal/flat 完整配对
    if not pairs:  # 判断是否不存在完整配对
        return []  # 返回空列表表示无法补算
    generated_taf_paths = []  # 初始化新生成 TAF 路径列表
    for _, pair_info in sorted(pairs.items()):  # 遍历完整配对并按键排序
        normal_df = load_pga_dataframe(pair_info['normal'], TARGET_COLUMNS)  # 读取 normal PGA 数据
        flat_df = load_pga_dataframe(pair_info['flat'], TARGET_COLUMNS)  # 读取 flat PGA 数据
        normal_norm_df = normalize_dataframe(normal_df, TARGET_COLUMNS, RESAMPLE_STEP)  # 归一化 normal 数据
        flat_norm_df = normalize_dataframe(flat_df, TARGET_COLUMNS, RESAMPLE_STEP)  # 归一化 flat 数据
        taf_df = compute_taf_dataframe(normal_norm_df, flat_norm_df)  # 计算当前配对 TAF_h 数据
        out_name = 'TAF-{}.csv'.format(pair_info['motion'])  # 组装输出 TAF 文件名
        out_path = os.path.join(folder_path, out_name)  # 组装输出 TAF 文件路径
        taf_df.to_csv(out_path, index=False, encoding='utf-8-sig')  # 保存新生成 TAF 文件
        generated_taf_paths.append(out_path)  # 记录生成文件路径
    return generated_taf_paths  # 返回新生成 TAF 列表


def collect_records(batch_root):  # 定义数据扫描与采集函数
    records = []  # 初始化记录列表
    if not os.path.isdir(batch_root):  # 判断根目录是否存在
        raise NotADirectoryError('数据目录不存在: {}'.format(batch_root))  # 抛出目录不存在异常
    for entry_name in sorted(os.listdir(batch_root)):  # 遍历根目录下所有条目
        folder_path = os.path.join(batch_root, entry_name)  # 拼接当前条目的完整路径
        if not os.path.isdir(folder_path):  # 判断当前条目是否为目录
            continue  # 跳过非目录条目
        parsed = parse_case_folder(entry_name)  # 解析参数目录名
        if parsed is None:  # 判断目录名是否不符合规则
            continue  # 跳过非参数目录
        h_value, i_value, angle_value = parsed  # 解包 h、i、angle 参数
        taf_paths = ensure_taf_csvs_in_folder(folder_path)  # 确保目录中存在 TAF 文件并在必要时自动补算
        for csv_path in taf_paths:  # 遍历目录内可用 TAF 文件路径
            file_name = os.path.basename(csv_path)  # 获取当前 TAF 文件名
            file_stem = os.path.splitext(file_name)[0]  # 获取文件名主干
            motion_cfg = detect_motion(file_stem)  # 识别地震波类型
            if motion_cfg is None:  # 判断是否为目标三种地震波
                continue  # 跳过未识别波名的文件
            try:  # 尝试提取 TAF 最大值
                taf_max = extract_taf_max_from_csv(csv_path)  # 读取并计算当前文件 TAF_max
            except Exception as exc:  # 捕获读取或计算异常
                print('警告：已跳过 {} -> {}'.format(csv_path, str(exc)))  # 输出跳过信息便于排查
                continue  # 跳过当前异常文件
            records.append({'h': h_value, 'i': i_value, 'angle': angle_value, 'motion': motion_cfg['key'], 'motion_display': motion_cfg['display'], 'taf_max': taf_max, 'file': csv_path})  # 追加结构化记录
    if not records:  # 判断是否未采集到任何记录
        raise FileNotFoundError('未采集到有效 TAF 数据，请检查目录与文件命名。')  # 抛出无数据异常
    return records  # 返回记录列表


def snap_to_target(value, targets, tol):  # 定义将实数吸附到目标序列的函数
    for target in targets:  # 遍历目标序列
        if abs(value - target) <= tol:  # 判断当前值是否落入容差
            return target  # 返回匹配到的目标值
    return None  # 未匹配到目标时返回空值


def prepare_summary(records):  # 定义汇总整理函数
    df = pd.DataFrame(records)  # 将记录列表转换为数据表
    df['h_snap'] = df['h'].apply(lambda v: snap_to_target(float(v), TARGET_H_VALUES, HI_TOL))  # 将 h 吸附到目标值
    df['i_snap'] = df['i'].apply(lambda v: snap_to_target(float(v), TARGET_I_VALUES, HI_TOL))  # 将 i 吸附到目标值
    df['angle_snap'] = df['angle'].apply(lambda v: snap_to_target(float(v), TARGET_ANGLES, ANGLE_TOL))  # 将 angle 吸附到目标值
    df = df[df['h_snap'].notna() & df['i_snap'].notna() & df['angle_snap'].notna()].copy()  # 过滤非目标网格点
    if df.empty:  # 判断过滤后是否为空
        raise ValueError('目标网格 (h,i,angle) 没有可用数据。')  # 抛出空网格异常
    grouped = df.groupby(['motion', 'motion_display', 'h_snap', 'i_snap', 'angle_snap'], as_index=False)['taf_max'].mean()  # 对重复样本取均值
    grouped = grouped.sort_values(by=['h_snap', 'motion', 'i_snap', 'angle_snap']).reset_index(drop=True)  # 对汇总结果排序
    return grouped  # 返回汇总数据


def configure_axes_style(ax):  # 定义坐标轴样式函数
    ax.set_facecolor('#e9e9e9')  # 设置子图背景色为浅灰
    for spine in ax.spines.values():  # 遍历四条边框线
        spine.set_linewidth(1.0)  # 设置边框线宽
        spine.set_color('#444444')  # 设置边框颜色
    ax.tick_params(direction='in', top=True, right=True, length=3.0, width=0.9, color='#444444', labelsize=12)  # 设置刻度样式
    ax.grid(False)  # 关闭网格线


def safe_slug(text):  # 定义安全文件名片段函数
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_')  # 转换为仅含小写字母数字下划线


def build_panel_filename(h_value, motion_key):  # 定义单图文件名构建函数
    h_token = str(int(round(float(h_value))))  # 构建 h 数值片段
    motion_token = safe_slug(motion_key)  # 构建地震波片段
    return '{}_h{}_{}.png'.format(OUTPUT_FIG_PREFIX, h_token, motion_token)  # 返回单图文件名


def plot_grid(summary_df, output_grid_path):  # 定义绘制 2x3 汇总大图函数
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 8.8), facecolor='white')  # 创建 2x3 子图画布
    y_limits_map = {50.0: (1.0, 2.25), 100.0: (1.0, 2.25)}  # 定义两行对应的纵轴范围
    y_ticks_map = {50.0: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25], 100.0: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]}  # 定义两行对应的纵轴刻度
    for row_idx, h_value in enumerate(TARGET_H_VALUES):  # 遍历两行 h 值
        for col_idx, motion_cfg in enumerate(MOTION_CONFIGS):  # 遍历三列地震波
            ax = axes[row_idx, col_idx]  # 获取当前子图对象
            configure_axes_style(ax)  # 应用统一坐标轴样式
            panel = summary_df[(summary_df['h_snap'] == h_value) & (summary_df['motion'] == motion_cfg['key'])]  # 提取当前子图数据
            for i_value in TARGET_I_VALUES:  # 遍历四条 i 曲线
                style = I_STYLE[i_value]  # 读取当前 i 对应样式
                series = panel[panel['i_snap'] == i_value].copy()  # 提取当前 i 的序列
                series = series.sort_values(by='angle_snap')  # 按角度升序排序
                x_values = series['angle_snap'].to_numpy(dtype=float)  # 读取横轴角度数组
                y_values = series['taf_max'].to_numpy(dtype=float)  # 读取纵轴 TAF_max 数组
                if x_values.size == 0:  # 判断当前曲线是否无数据
                    continue  # 跳过空曲线
                ax.plot(x_values, y_values, color=style['color'], marker=style['marker'], linestyle='-', linewidth=1.8, markersize=6.0, markeredgecolor='none', label=style['label'])  # 绘制当前 i 曲线
            ax.set_xlim(min(TARGET_ANGLES), max(TARGET_ANGLES))  # 固定横轴范围为 0 到 30
            ax.set_xticks(TARGET_ANGLES)  # 固定横轴刻度为 0/10/20/30
            ax.set_ylim(*y_limits_map[h_value])  # 设置当前行纵轴范围
            ax.set_yticks(y_ticks_map[h_value])  # 设置当前行纵轴刻度
            ax.set_title(motion_cfg['display'], fontsize=20, pad=6)  # 设置子图标题
            if col_idx == 0:  # 判断是否为每行第一列
                ax.set_ylabel(r'$TAF_{\max}$', fontsize=20)  # 设置纵轴标签
            if col_idx == 1:  # 判断是否为每行中间列
                ax.set_xlabel(r'$\theta_s$ (°)', fontsize=20)  # 设置横轴标签
            ax.legend(loc='upper left', frameon=False, fontsize=10, handlelength=2.2, handletextpad=0.3, borderaxespad=0.2)  # 绘制图例
    fig.text(0.5, 0.48, r'(a) $h$ = 50 m', ha='center', va='center', fontsize=24)  # 添加第一行行注释
    fig.text(0.5, 0.03, r'(b) $h$ = 100 m', ha='center', va='center', fontsize=24)  # 添加第二行行注释
    plt.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.12, wspace=0.33, hspace=0.62)  # 调整布局匹配目标版式
    output_parent = os.path.dirname(output_grid_path)  # 提取汇总大图输出父目录
    if output_parent:  # 判断父目录字符串是否有效
        os.makedirs(output_parent, exist_ok=True)  # 确保汇总大图输出目录已创建
    fig.savefig(output_grid_path, dpi=350)  # 保存输出汇总图像
    plt.close(fig)  # 关闭图对象释放内存


def plot_single_panels(summary_df, output_fig_dir):  # 定义逐子图输出函数
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体为 STIX
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示
    y_limits_map = {50.0: (1.0, 2.25), 100.0: (1.0, 2.25)}  # 定义不同 h 的纵轴范围
    y_ticks_map = {50.0: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25], 100.0: [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]}  # 定义不同 h 的纵轴刻度
    output_paths = []  # 初始化输出路径列表
    os.makedirs(output_fig_dir, exist_ok=True)  # 创建单图输出目录
    for h_value in TARGET_H_VALUES:  # 遍历目标 h 值
        for motion_cfg in MOTION_CONFIGS:  # 遍历三种地震波
            fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4), facecolor='white')  # 创建单图画布
            configure_axes_style(ax)  # 应用坐标轴样式
            panel = summary_df[(summary_df['h_snap'] == h_value) & (summary_df['motion'] == motion_cfg['key'])]  # 提取当前子图数据
            for i_value in TARGET_I_VALUES:  # 遍历四条 i 曲线
                style = I_STYLE[i_value]  # 读取当前 i 对应样式
                series = panel[panel['i_snap'] == i_value].copy()  # 提取当前 i 的序列
                series = series.sort_values(by='angle_snap')  # 按角度升序排序
                x_values = series['angle_snap'].to_numpy(dtype=float)  # 读取横轴角度数组
                y_values = series['taf_max'].to_numpy(dtype=float)  # 读取纵轴 TAF_max 数组
                if x_values.size == 0:  # 判断当前曲线是否无数据
                    continue  # 跳过空曲线
                ax.plot(x_values, y_values, color=style['color'], marker=style['marker'], linestyle='-', linewidth=1.8, markersize=6.0, markeredgecolor='none', label=style['label'])  # 绘制当前 i 曲线
            ax.set_xlim(min(TARGET_ANGLES), max(TARGET_ANGLES))  # 固定横轴范围为 0 到 30
            ax.set_xticks(TARGET_ANGLES)  # 固定横轴刻度为 0/10/20/30
            ax.set_ylim(*y_limits_map[h_value])  # 设置当前图纵轴范围
            ax.set_yticks(y_ticks_map[h_value])  # 设置当前图纵轴刻度
            ax.set_title(r'{} ($h$ = {} m)'.format(motion_cfg['display'], int(round(h_value))), fontsize=16, pad=8)  # 设置单图标题
            ax.set_ylabel(r'$TAF_{\max}$', fontsize=15)  # 设置纵轴标签
            ax.set_xlabel(r'$\theta_s$ (°)', fontsize=15)  # 设置横轴标签
            ax.legend(loc='upper left', frameon=False, fontsize=10, handlelength=2.2, handletextpad=0.3, borderaxespad=0.2)  # 绘制图例
            fig.tight_layout()  # 自动调整单图布局
            output_name = build_panel_filename(h_value, motion_cfg['key'])  # 生成当前单图文件名
            output_path = os.path.join(output_fig_dir, output_name)  # 生成当前单图完整路径
            fig.savefig(output_path, dpi=350)  # 保存当前单图
            plt.close(fig)  # 关闭当前图对象释放内存
            output_paths.append(output_path)  # 记录当前单图路径
    return output_paths  # 返回全部输出路径


def resolve_output_directory(batch_root, arg_output):  # 定义输出目录解析函数
    if not arg_output:  # 判断是否未提供输出参数
        return batch_root  # 使用数据目录作为默认输出目录
    abs_output = os.path.abspath(arg_output)  # 转换为绝对路径
    ext_name = os.path.splitext(abs_output)[1].lower()  # 读取参数扩展名
    if ext_name == '.png':  # 判断是否沿用旧版传入 PNG 路径习惯
        parent_dir = os.path.dirname(abs_output)  # 读取 PNG 父目录
        if parent_dir:  # 判断父目录是否非空
            return parent_dir  # 返回父目录作为输出目录
        return batch_root  # 父目录为空时回退到数据目录
    return abs_output  # 将参数作为目录路径返回


def main():  # 定义主函数
    if len(sys.argv) >= 2:  # 判断是否传入数据根目录参数
        batch_root = os.path.abspath(sys.argv[1])  # 使用命令行数据目录
    else:  # 未传入参数时使用默认目录
        batch_root = DEFAULT_BATCH_ROOT  # 使用脚本目录作为默认数据目录
    arg_output = sys.argv[2] if len(sys.argv) >= 3 else ''  # 读取可选输出参数
    output_dir = resolve_output_directory(batch_root, arg_output)  # 解析最终输出目录
    if output_dir and (not os.path.isdir(output_dir)):  # 判断输出目录是否需要创建
        os.makedirs(output_dir, exist_ok=True)  # 创建输出目录
    records = collect_records(batch_root)  # 扫描并采集所有记录
    summary_df = prepare_summary(records)  # 生成目标网格汇总数据
    output_fig_dir = os.path.join(output_dir, OUTPUT_FIG_DIR_NAME)  # 组装单图输出目录路径
    os.makedirs(output_fig_dir, exist_ok=True)  # 提前创建单图输出目录以避免汇总图保存时报错
    output_csv_path = os.path.join(output_fig_dir, OUTPUT_CSV_NAME)  # 组装汇总 CSV 输出路径
    summary_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')  # 保存汇总 CSV 便于复核
    output_grid_path = os.path.join(output_fig_dir, OUTPUT_GRID_PNG_NAME)  # 组装汇总大图输出路径
    plot_grid(summary_df, output_grid_path)  # 绘制并保存汇总大图
    output_panel_paths = plot_single_panels(summary_df, output_fig_dir)  # 逐子图绘制并保存
    print('数据目录: {}'.format(batch_root))  # 输出数据目录信息
    print('汇总表: {}'.format(output_csv_path))  # 输出汇总 CSV 路径
    print('单图目录: {}'.format(output_fig_dir))  # 输出单图目录路径
    print('汇总图: {}'.format(output_grid_path))  # 输出汇总大图路径
    print('单图数量: {}'.format(len(output_panel_paths)))  # 输出单图数量统计
    for panel_path in output_panel_paths:  # 遍历全部单图路径
        print('单图输出: {}'.format(panel_path))  # 逐行输出单图路径


if __name__ == '__main__':  # 判断脚本是否直接运行
    main()  # 调用主函数执行流程