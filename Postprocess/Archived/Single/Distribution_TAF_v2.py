import pandas as pd  # 导入 pandas 用于读取与保存 CSV 数据
import numpy as np  # 导入 numpy 用于数值计算与插值
import matplotlib  # 导入 matplotlib 主模块用于设置无界面后端
matplotlib.use('Agg')  # 设置 Agg 后端以避免脚本运行时弹窗
import matplotlib.pyplot as plt  # 导入 pyplot 用于绘图
import matplotlib.font_manager as fm  # 导入字体管理器用于中英文字体配置
import os  # 导入 os 用于路径处理
import glob  # 导入 glob 用于批量匹配文件
import re  # 导入 re 用于正则解析文件名
import math  # 导入 math 用于坡脚位置公式计算

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 记录当前脚本所在目录
CSV_GLOB_PATTERN = 'PGA*.csv'  # 定义原始 PGA 数据文件匹配模式
TARGET_COLUMNS = ['PGA_h', 'PGA_v']  # 定义需要处理的 PGA 分量列
RESAMPLE_STEP = 0.05  # 定义 x/h 归一化插值步长为 0.05
LOC_CREST = 3.0  # 定义坡顶归一化位置
LOC_TOE = 4.0  # 定义坡脚归一化位置
SAFE_DIVIDE_EPS = 1e-12  # 定义安全除法阈值避免分母接近 0


# 解析同目录 CAE 文件名末尾 a 后数字作为入射角。
def parse_incident_angle_from_cae(script_dir):
    cae_files = sorted(glob.glob(os.path.join(script_dir, '*.cae')))  # 匹配并排序当前目录下全部 CAE 文件
    if not cae_files:  # 若未找到 CAE 文件则抛出错误提示
        raise FileNotFoundError('脚本目录下未找到 .cae 文件，无法自动解析入射角度。')  # 报告缺失 CAE 文件
    cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]  # 取排序后第一个 CAE 文件名主干
    match = re.search(r'a(-?\d+(?:\.\d+)?)$', cae_name)  # 匹配文件名末尾 a 后的数字
    if not match:  # 若未命中命名规则则抛出错误
        raise ValueError(f"CAE 文件名 {cae_name}.cae 不符合要求，未找到末尾 'a' 后数字。")  # 提示命名不符合规则
    return float(match.group(1))  # 返回解析得到的入射角浮点数


INCIDENT_ANGLE = parse_incident_angle_from_cae(SCRIPT_DIR)  # 读取当前算例入射角用于标题和命名


# 解析同目录 CAE 文件名中的 h 与 i 参数用于图表标题样式。
def parse_geometry_from_cae(script_dir):
    cae_files = sorted(glob.glob(os.path.join(script_dir, '*.cae')))  # 匹配并排序当前目录下全部 CAE 文件
    if not cae_files:  # 若未找到 CAE 文件则返回空值以走兜底标题
        return None, None  # 返回空几何参数
    cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]  # 取排序后第一个 CAE 文件名主干
    match_h = re.search(r'h(-?\d+(?:\.\d+)?)', cae_name)  # 匹配文件名中的 h 后数字
    match_i = re.search(r'i(-?\d+(?:\.\d+)?)', cae_name)  # 匹配文件名中的 i 后数字
    height_value = float(match_h.group(1)) if match_h else None  # 解析边坡高度参数并在缺失时置空
    slope_angle = float(match_i.group(1)) if match_i else None  # 解析边坡角度参数并在缺失时置空
    return height_value, slope_angle  # 返回解析得到的几何参数


SLOPE_HEIGHT, SLOPE_ANGLE = parse_geometry_from_cae(SCRIPT_DIR)  # 读取当前算例 h 与 i 用于标题


# 根据坡角 i 计算坡脚归一化坐标 x/h = 3 + 1/tan(i)。
def compute_toe_location_from_angle(slope_angle_deg):
    if slope_angle_deg is None:  # 若未解析到坡角则使用默认位置
        return LOC_TOE  # 返回默认坡脚位置
    tan_value = math.tan(math.radians(slope_angle_deg))  # 将角度转弧度后计算正切值
    if abs(tan_value) <= 1e-12:  # 若正切值过小则避免除零
        return LOC_TOE  # 回退到默认坡脚位置
    return LOC_CREST + 1.0 / tan_value  # 按公式返回坡脚归一化坐标


# 构建中英文字体属性以保证中文显示稳定。
def build_font_properties():
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']  # 定义中文字体候选列表
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']  # 定义英文字体候选列表
    cn_font = fm.FontProperties()  # 初始化中文字体属性默认值
    en_font = fm.FontProperties()  # 初始化英文字体属性默认值
    for name in cn_candidates:  # 依次尝试匹配系统可用中文字体
        try:  # 捕获字体不可用异常以继续尝试下一个候选
            font_path = fm.findfont(name, fallback_to_default=False)  # 严格查找指定字体且不回退
            cn_font = fm.FontProperties(fname=font_path)  # 通过字体文件路径创建中文字体属性
            break  # 找到可用字体后结束循环
        except Exception:  # 若当前字体不可用则进入下一轮尝试
            continue  # 继续检查下一个候选字体
    for name in en_candidates:  # 依次尝试匹配系统可用英文字体
        try:  # 捕获字体不可用异常以继续尝试下一个候选
            font_path = fm.findfont(name, fallback_to_default=False)  # 严格查找指定字体且不回退
            en_font = fm.FontProperties(fname=font_path)  # 通过字体文件路径创建英文字体属性
            break  # 找到可用字体后结束循环
        except Exception:  # 若当前字体不可用则进入下一轮尝试
            continue  # 继续检查下一个候选字体
    return cn_font, en_font  # 返回中英文字体属性


CN_FONT, EN_FONT = build_font_properties()  # 构建并缓存字体属性供全流程复用


# 读取并检查单个 PGA 文件所需列。
def load_pga_dataframe(filepath, target_cols):
    df = pd.read_csv(filepath)  # 读取 CSV 为数据表
    required_cols = {'x/h'} | set(target_cols)  # 组装必须存在的列集合
    missing_cols = required_cols - set(df.columns)  # 计算缺失列集合
    if missing_cols:  # 若存在缺失列则报错
        raise ValueError(f"文件 {os.path.basename(filepath)} 缺少列: {sorted(missing_cols)}")  # 抛出缺列提示
    df = df[['x/h'] + list(target_cols)].copy()  # 仅保留后续计算所需列并复制
    df = df.sort_values(by='x/h')  # 按 x/h 升序排序保证插值输入单调
    return df  # 返回清洗后的数据表


# 将数据按固定 x/h 步长重采样并做线性插值。
def normalize_dataframe(df, target_cols, step):
    grouped_df = df.groupby('x/h', as_index=False).mean(numeric_only=True)  # 对重复 x/h 取均值防止插值报错
    x_src = grouped_df['x/h'].to_numpy(dtype=float)  # 提取原始 x/h 数组
    x_start = float(np.min(x_src))  # 读取原始最小 x/h
    x_end = float(np.max(x_src))  # 读取原始最大 x/h
    x_norm = np.round(np.arange(x_start, x_end + step * 0.5, step), 10)  # 生成步长固定为 0.05 的归一化横坐标
    norm_dict = {'x/h': x_norm}  # 初始化归一化结果字典并写入 x/h
    for col in target_cols:  # 逐列执行插值重采样
        y_src = grouped_df[col].to_numpy(dtype=float)  # 提取当前分量原始数组
        y_norm = np.interp(x_norm, x_src, y_src)  # 在线性空间对当前分量进行插值估算
        norm_dict[col] = y_norm  # 将插值结果写回结果字典
    return pd.DataFrame(norm_dict)  # 由结果字典构建并返回归一化数据表


# 收集原始 PGA 文件并排除已归一化中间文件。
def collect_original_pga_files(pattern):
    search_pattern = os.path.join(SCRIPT_DIR, pattern)  # 组装匹配路径模式
    all_files = sorted(glob.glob(search_pattern))  # 搜索并排序匹配到的 PGA 文件
    csv_files = [p for p in all_files if '-normalized' not in os.path.splitext(os.path.basename(p))[0]]  # 排除已有归一化文件避免重复处理
    if not csv_files:  # 若未找到可处理文件则报错
        raise FileNotFoundError(f'脚本目录下未找到可处理的原始 PGA 文件，匹配模式: {pattern}')  # 抛出错误提示
    return csv_files  # 返回可处理原始 PGA 文件列表


# 从文件主干解析配对键并判断是否 flat 文件。
def parse_pair_key(csv_stem):
    is_slope = csv_stem.endswith('-slope')  # 判断当前文件是否为坡地工况文件
    is_flat = csv_stem.endswith('-flat')  # 判断当前文件是否为平地基准文件
    if is_slope:  # 若命中坡地后缀则移除后缀得到配对键
        base_key = csv_stem[:-6]  # 去除 -slope 后得到同波配对键
    elif is_flat:  # 若命中平地后缀则移除后缀得到配对键
        base_key = csv_stem[:-5]  # 去除 -flat 后得到同波配对键
    else:  # 若未命中新命名后缀则保留主干并按坡地处理
        base_key = csv_stem  # 使用原主干作为配对键
        is_slope = True  # 将未标注后缀文件视为坡地文件以兼容旧数据
    return base_key, is_flat, is_slope  # 返回配对键与类型标记


# 从主干提取用于输出命名的波名。
def extract_motion_name(csv_stem):
    cleaned_name = re.sub(r'^PGA[_-]*', '', csv_stem)  # 去除 PGA 前缀并保留主体名称
    cleaned_name = re.sub(r'^job-', '', cleaned_name)  # 去除可选 job- 前缀
    cleaned_name = re.sub(r'-(?:slope|flat)$', '', cleaned_name)  # 去除新命名的 -slope 或 -flat 后缀
    cleaned_name = re.sub(r'_(?:scaled|veled)$', '', cleaned_name)  # 去除旧命名中的 _scaled 或 _veled 收尾
    cleaned_name = re.sub(r'-(?:scaled|veled)$', '', cleaned_name)  # 去除连字符命名中的 -scaled 或 -veled 收尾
    return cleaned_name  # 返回清洗后的地震波名称


# 将原始文件列表按 normal/flat 两类进行配对。
def build_pairs(csv_files):
    pairs = {}  # 初始化配对字典
    for filepath in csv_files:  # 遍历每个 PGA 文件
        stem = os.path.splitext(os.path.basename(filepath))[0]  # 获取文件名主干
        base_key, is_flat, is_slope = parse_pair_key(stem)  # 解析配对键及文件类型标志
        if base_key not in pairs:  # 若字典中尚无该配对键则初始化
            pairs[base_key] = {'normal': None, 'flat': None, 'motion': extract_motion_name(base_key)}  # 写入配对槽位和波名
        if is_flat:  # 若当前文件属于 flat 基准组
            pairs[base_key]['flat'] = filepath  # 记录 flat 文件路径
        elif is_slope:  # 若当前文件属于 slope 工况组
            pairs[base_key]['normal'] = filepath  # 记录 normal 文件路径
    valid_pairs = {}  # 初始化有效配对字典
    for base_key, item in pairs.items():  # 检查每个配对键是否完整
        if item['normal'] is None or item['flat'] is None:  # 若 normal 或 flat 任一缺失则跳过
            continue  # 忽略不完整配对
        valid_pairs[base_key] = item  # 收集完整可计算配对
    if not valid_pairs:  # 若没有任何完整配对则报错
        raise ValueError('未找到可用的 normal/flat 文件配对，请检查文件命名（flat 文件需以 _flat 结尾）。')  # 提示命名或文件缺失问题
    return valid_pairs  # 返回完整配对结果


# 计算 TAF 并生成结果数据表。
def compute_taf_dataframe(normalized_normal_df, normalized_flat_df):
    x_normal = normalized_normal_df['x/h'].to_numpy(dtype=float)  # 读取 normal 的归一化横坐标
    x_flat = normalized_flat_df['x/h'].to_numpy(dtype=float)  # 读取 flat 的归一化横坐标
    if len(x_normal) != len(x_flat) or not np.allclose(x_normal, x_flat):  # 校验两个归一化横坐标严格一致
        raise ValueError('normal 与 flat 的归一化 x/h 不一致，无法逐点相除计算 TAF。')  # 抛出坐标不一致错误
    denominator = normalized_flat_df['PGA_h'].to_numpy(dtype=float)  # 读取 flat 水平向分量作为分母
    numerator = normalized_normal_df['PGA_h'].to_numpy(dtype=float)  # 读取 normal 水平向分量作为分子
    valid_mask = np.abs(denominator) > SAFE_DIVIDE_EPS  # 构造分母非零有效掩码
    taf_values = np.full_like(numerator, np.nan, dtype=float)  # 预分配 TAF_h 数组并默认设为 NaN
    taf_values[valid_mask] = numerator[valid_mask] / denominator[valid_mask]  # 在有效位置执行逐点相除
    result = {'x/h': x_normal, 'TAF_h': taf_values}  # 仅写入水平向 TAF 结果列
    return pd.DataFrame(result)  # 返回仅含水平分量的 TAF 数据表


# 格式化数值为标题显示文本。
def format_value_text(value):
    if value is None:  # 若参数为空则返回未知占位符
        return '?'  # 返回问号用于兜底显示
    return str(int(value)) if float(value).is_integer() else f'{value:g}'  # 将整数显示为整型文本并压缩浮点尾零


# 绘制水平向 TAF 曲线并按参考图样式设置图表外观。
def plot_taf_horizontal(ax, x_h, taf_values):
    title_h = format_value_text(SLOPE_HEIGHT)  # 格式化标题中的 h 参数文本
    title_i = format_value_text(SLOPE_ANGLE)  # 格式化标题中的 i 参数文本
    toe_x = compute_toe_location_from_angle(SLOPE_ANGLE)  # 按坡角计算坡脚虚线位置
    ax.set_facecolor('#f2f2f2')  # 设置子图背景为浅灰色以接近参考图
    ax.plot(x_h, taf_values, color='#d62728', linestyle='-', linewidth=1.4)  # 以较细红色实线绘制水平向 TAF 曲线
    ax.set_title(f'h = {title_h}m, i = {title_i}°', fontsize=14, fontproperties=EN_FONT, pad=8)  # 使用参考图样式设置标题
    ax.set_xlabel('x/h', fontsize=16, fontproperties=EN_FONT, fontstyle='italic')  # 设置横轴标签为斜体样式
    ax.set_ylabel('TAF', fontsize=16, fontproperties=EN_FONT)  # 设置纵轴标签为英文 TAF
    ax.set_xlim(0, 8)  # 固定横轴范围为 0 到 8
    ax.set_xticks(np.arange(0, 9, 1))  # 固定横轴主刻度为整数刻度
    ax.set_ylim(0.5, 3.0)  # 固定纵轴范围为 0.5 到 3.0 以匹配参考图
    ax.set_yticks(np.arange(0.5, 3.1, 0.5))  # 固定纵轴刻度间距为 0.5
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))  # 设置纵轴数值格式保留一位小数
    ax.axvline(x=LOC_CREST, color='black', linestyle='--', linewidth=1.0)  # 绘制坡顶位置的黑色虚线
    ax.axvline(x=toe_x, color='black', linestyle='--', linewidth=1.0)  # 绘制按坡角计算的坡脚位置黑色虚线
    text_y = 0.56  # 设置中文标注的统一纵坐标位置
    ax.text(LOC_CREST + 0.06, text_y, '坡顶', fontsize=11, fontproperties=CN_FONT, va='bottom')  # 在坡顶虚线旁标注“坡顶”
    ax.text(toe_x + 0.06, text_y, '坡脚', fontsize=11, fontproperties=CN_FONT, va='bottom')  # 在坡脚虚线旁标注“坡脚”
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.8, color='#d0d0d0')  # 设置浅灰短虚线网格以贴近示意图
    ax.tick_params(direction='in', top=True, right=True, labelsize=12)  # 设置刻度朝内并显示上右刻度


# 将单波 TAF 水平分量曲线绘图并保存单图。
def plot_and_save_taf(taf_df, motion_name, output_dir):
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局英文默认字体为 Times 风格
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置数学字体为 STIX 以匹配 Times 风格
    plt.rcParams['axes.unicode_minus'] = False  # 修复负号显示问题
    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.6), squeeze=False)  # 创建单子图用于水平向 TAF
    ax = ax.flatten()[0]  # 提取单个坐标轴对象
    x_h = taf_df['x/h'].to_numpy(dtype=float)  # 提取横坐标数组
    taf_h = taf_df['TAF_h'].to_numpy(dtype=float)  # 提取水平向 TAF 数组
    plot_taf_horizontal(ax, x_h, taf_h)  # 按参考图样式绘制水平向 TAF 子图
    plt.tight_layout()  # 自动调整子图布局避免遮挡
    angle_text = str(int(INCIDENT_ANGLE)) if float(INCIDENT_ANGLE).is_integer() else str(INCIDENT_ANGLE)  # 生成用于命名的角度文本
    output_name = f'TAF-{motion_name}-horizontal-angle{angle_text}.png'  # 定义水平向 TAF 输出文件名
    output_path = os.path.join(output_dir, output_name)  # 组装水平向 TAF 图像完整路径
    fig.savefig(output_path, dpi=300, bbox_inches='tight')  # 保存水平向 TAF 图像并使用紧凑裁边
    plt.close(fig)  # 关闭图对象释放内存


# 主流程：归一化、计算 TAF、保存 CSV 并绘图。
def main():
    csv_files = collect_original_pga_files(CSV_GLOB_PATTERN)  # 收集原始 PGA 文件
    pairs = build_pairs(csv_files)  # 构建 normal/flat 完整配对
    for base_key, pair_info in pairs.items():  # 遍历每一个完整配对
        normal_path = pair_info['normal']  # 读取 normal 文件路径
        flat_path = pair_info['flat']  # 读取 flat 文件路径
        motion_name = pair_info['motion']  # 读取当前配对波名
        normal_df = load_pga_dataframe(normal_path, TARGET_COLUMNS)  # 加载 normal 原始数据
        flat_df = load_pga_dataframe(flat_path, TARGET_COLUMNS)  # 加载 flat 原始数据
        normal_norm_df = normalize_dataframe(normal_df, TARGET_COLUMNS, RESAMPLE_STEP)  # 对 normal 数据执行 0.05 步长归一化插值
        flat_norm_df = normalize_dataframe(flat_df, TARGET_COLUMNS, RESAMPLE_STEP)  # 对 flat 数据执行 0.05 步长归一化插值
        normal_stem = os.path.splitext(os.path.basename(normal_path))[0]  # 提取 normal 文件主干
        flat_stem = os.path.splitext(os.path.basename(flat_path))[0]  # 提取 flat 文件主干
        normal_norm_file = os.path.join(SCRIPT_DIR, f'{normal_stem}-normalized.csv')  # 生成 normal 归一化输出文件名
        flat_norm_file = os.path.join(SCRIPT_DIR, f'{flat_stem}-normalized.csv')  # 生成 flat 归一化输出文件名
        normal_norm_df.to_csv(normal_norm_file, index=False, encoding='utf-8-sig')  # 保存 normal 归一化数据为 CSV
        flat_norm_df.to_csv(flat_norm_file, index=False, encoding='utf-8-sig')  # 保存 flat 归一化数据为 CSV
        taf_df = compute_taf_dataframe(normal_norm_df, flat_norm_df)  # 计算当前波的 TAF 数据
        taf_file = os.path.join(SCRIPT_DIR, f'TAF-{motion_name}.csv')  # 生成 TAF 输出文件名并确保以 TAF 开头
        taf_df.to_csv(taf_file, index=False, encoding='utf-8-sig')  # 保存 TAF 结果到 CSV
        plot_and_save_taf(taf_df, motion_name, SCRIPT_DIR)  # 绘制并保存当前波 TAF 图像
        print(f'已完成: {base_key} -> {os.path.basename(taf_file)}')  # 输出当前配对处理完成信息


if __name__ == '__main__':  # 仅在脚本直接运行时执行主流程
    main()  # 调用主函数启动批处理流程
