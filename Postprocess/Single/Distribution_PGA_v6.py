import pandas as pd
import numpy as np
import matplotlib  # 导入 matplotlib 主模块以设置无界面后端
matplotlib.use('Agg')  # 使用 Agg 后端防止运行时弹出图形窗口
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import glob
import re
import math  # 导入 math 用于根据坡角计算坡脚归一化位置

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 配置文件路径与参数
# ==========================================
# 自动读取当前目录下“文件名前缀为 PGA”的 CSV 文件
CSV_GLOB_PATTERN = 'PGA*.csv'

# 当前绘制参数：从同目录 CAE 文件名解析 h、i、a/angle 后数字
# 例如: h100_i45_a0.cae -> h=100, i=45, incident_angle=0
def parse_case_params_from_cae(script_dir):
    cae_files = sorted(glob.glob(os.path.join(script_dir, '*.cae')))  # 搜索脚本目录下全部 CAE 文件
    if not cae_files:  # 判断是否找到 CAE 文件
        raise FileNotFoundError('脚本目录下未找到 .cae 文件，无法自动解析参数。')  # 未找到时抛出错误并提示

    cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]  # 取排序后第一个 CAE 的主文件名进行解析
    pattern_a = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_a(?P<a>-?\d+(?:\.\d+)?)$')  # 定义 h_i_a 命名模式
    pattern_angle = re.compile(r'h(?P<h>-?\d+(?:\.\d+)?)_i(?P<i>-?\d+(?:\.\d+)?)_angle(?P<a>-?\d+(?:\.\d+)?)$')  # 定义 h_i_angle 命名模式
    match = pattern_a.search(cae_name) or pattern_angle.search(cae_name)  # 依次尝试两种命名模式
    if not match:  # 判断是否命名解析失败
        raise ValueError(f"CAE 文件名 {cae_name}.cae 不符合要求，需包含 h、i 与 a/angle 参数。")  # 抛出命名不符合约定错误

    h_value = float(match.group('h'))  # 读取并转换 h 参数
    i_value = float(match.group('i'))  # 读取并转换 i 参数
    incident_angle = float(match.group('a'))  # 读取并转换入射角参数
    return h_value, i_value, incident_angle  # 返回解析得到的算例参数


def compute_toe_location_from_slope_angle(slope_angle_deg):
    tan_value = math.tan(math.radians(slope_angle_deg))  # 将坡角从角度制转换后计算正切值
    if abs(tan_value) <= 1e-12:  # 判断正切是否接近零以避免除零风险
        return 4.0  # 当坡角接近零时回退到默认坡脚位置
    return 3.0 + 1.0 / tan_value  # 按几何关系计算归一化坡脚位置


H_VALUE, SLOPE_ANGLE, INCIDENT_ANGLE = parse_case_params_from_cae(SCRIPT_DIR)  # 解析并缓存当前工况的 h、i 与入射角参数

# 要绘制的 PGA 列：
# 可选值：'PGA_h' (对应图12a), 'PGA_v' (对应图12b)
# 例如只画水平：TARGET_COLUMNS = ['PGA_h']
TARGET_COLUMNS = ['PGA_h', 'PGA_v']
PLOT_COMPONENTS_TOGETHER = False  # 控制是否将水平向与竖向绘制在同一张合并图中
INCLUDE_NORMALIZED_GROUPS = True  # 控制是否绘制分组名中包含 normalized 的组
INCLUDE_FLAT_GROUPS = True  # 控制是否绘制分组名中包含 flat 标记的组

# 各分量的默认 Y 轴范围 (请按你的数据微调)
YLIMS = {
    'PGA_h': (0.2, 1.2),  
    'PGA_v': (0.0, 0.4),
}

# 坡顶归一化位置固定为 3.0，坡脚位置由 CAE 文件名解析得到的坡角 i 动态计算
LOC_CREST = 3.0  # 定义固定坡顶归一化横坐标
LOC_TOE = compute_toe_location_from_slope_angle(SLOPE_ANGLE)  # 根据当前坡角动态计算坡脚归一化横坐标


def build_font_properties():
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']  # 定义中文字体候选列表并优先宋体
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']  # 定义英文字体候选列表并优先 Times New Roman

    cn_font = fm.FontProperties()  # 初始化中文字体属性默认值以防候选字体都不可用并避免 family 字符串解析异常
    en_font = fm.FontProperties()  # 初始化英文字体属性默认值以防候选字体都不可用并避免 family 字符串解析异常

    for name in cn_candidates:  # 遍历中文候选字体并查找系统可用字体
        try:
            font_path = fm.findfont(name, fallback_to_default=False)  # 严格查找目标字体且不回退默认字体
            cn_font = fm.FontProperties(fname=font_path)  # 若找到字体则按字体文件路径创建中文字体属性
            break
        except Exception:
            continue

    for name in en_candidates:  # 遍历英文候选字体并查找系统可用字体
        try:
            font_path = fm.findfont(name, fallback_to_default=False)  # 严格查找目标字体且不回退默认字体
            en_font = fm.FontProperties(fname=font_path)  # 若找到字体则按字体文件路径创建英文字体属性
            break
        except Exception:
            continue

    return cn_font, en_font


CN_FONT, EN_FONT = build_font_properties()  # 构建并缓存中英文字体属性供全图复用

# ==========================================
# 2. 读取并处理数据
# ==========================================
def load_pga_data(filepath, target_col):
    df = pd.read_csv(filepath)
    # 确保数据按 x/h 排序，保证画图时连线不乱
    df = df.sort_values(by='x/h')
    required_cols = {'x/h', target_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"文件 {os.path.basename(filepath)} 缺少列: {sorted(missing_cols)}")
    return df['x/h'].values, df[target_col].values


def collect_csv_files(pattern):
    search_pattern = os.path.join(SCRIPT_DIR, pattern)
    csv_files = sorted(glob.glob(search_pattern))
    if not csv_files:
        raise FileNotFoundError(f"脚本目录下未找到匹配 {pattern} 的 CSV 文件。")
    return csv_files


def extract_motion_name(csv_stem):
    match = re.search(r'job-([A-Za-z0-9_]+?)(?:_(?:scaled|veled))?$', csv_stem)  # 优先匹配 job-地震波名(_scaled/_veled) 的命名模式
    if match:  # 若命中标准命名规则则直接返回地震波名
        return match.group(1)  # 返回如 El_Centro 的地震波名称

    cleaned_name = re.sub(r'^PGA[_-]*', '', csv_stem)  # 去掉前缀 PGA_ 或 PGA-
    cleaned_name = re.sub(r'^job-', '', cleaned_name)  # 去掉可选前缀 job-
    cleaned_name = re.sub(r'_(?:scaled|veled)$', '', cleaned_name)  # 去掉可选后缀 _scaled 或 _veled
    return cleaned_name  # 返回清洗后的名称作为兜底结果


def parse_motion_and_group(csv_stem):
    parts = csv_stem.split('-', 2)  # 以最多两次分割解析形如 PGA-波名-分组后缀 的命名结构
    if len(parts) >= 3:  # 若至少存在两个连字符则按“第二个连字符后”为分组键
        motion_name = re.sub(r'_(?:scaled|veled)$', '', parts[1])  # 清理波名中的可选后处理后缀并作为图例名称
        group_suffix = parts[2]  # 将第二个连字符后的全部文本作为分组后缀
        return motion_name, group_suffix  # 返回图例波名与分组键

    return extract_motion_name(csv_stem), 'ungrouped'  # 若不满足新命名规则则回退旧解析并归入兜底分组


def group_csv_files_by_suffix(csv_files):
    grouped = {}  # 初始化分组字典用于存放“分组后缀 -> 文件列表”的映射
    for path in csv_files:  # 遍历全部 CSV 文件路径并逐个解析分组键
        stem = os.path.splitext(os.path.basename(path))[0]  # 取得不含扩展名的文件主名用于命名解析
        _, suffix = parse_motion_and_group(stem)  # 解析当前文件所属分组后缀
        grouped.setdefault(suffix, []).append(path)  # 按后缀聚合文件路径并保留同组全部样本
    return dict(sorted(grouped.items(), key=lambda item: item[0]))  # 按分组名排序后返回以保证输出顺序稳定


def sanitize_for_filename(text):
    safe = re.sub(r'[^A-Za-z0-9_-]+', '_', text).strip('_')  # 将文件名非法字符替换为下划线并去掉首尾下划线
    return safe or 'group'  # 若清理后为空则使用兜底字符串防止生成空文件名


def should_include_group(group_suffix):
    suffix_lower = group_suffix.lower()  # 将分组名统一转为小写以便做不区分大小写判断
    has_flat_tag = re.search(r'(^|-)flat($|-)', suffix_lower) is not None  # 判断分组名是否包含由连字符分隔的 flat 标记
    if not INCLUDE_FLAT_GROUPS and has_flat_tag:  # 判断是否需要过滤 flat 相关分组
        return False  # 当关闭 flat 开关且当前分组包含 flat 时返回不保留
    if INCLUDE_NORMALIZED_GROUPS:  # 判断是否允许绘制 normalized 相关分组
        return True  # 允许时直接保留当前分组
    return 'normalized' not in suffix_lower  # 不允许时过滤掉名称中包含 normalized 的分组


def build_component_series(group_files, component, group_suffix):
    x_ref = None  # 初始化参考 x/h 序列用于组内文件对齐检查
    pga_series = []  # 初始化当前分量的多文件 PGA 序列容器
    for filepath in group_files:  # 遍历组内全部文件并读取当前分量
        x_h, pga_arr = load_pga_data(filepath, component)  # 加载当前文件指定分量与对应 x/h 坐标
        if x_ref is None:  # 判断是否为首个文件
            x_ref = x_h  # 保存首个文件的 x/h 作为一致性检查基准
        else:  # 当前文件不是首个文件时执行一致性检查
            if len(x_h) != len(x_ref) or not np.allclose(x_h, x_ref):  # 判断节点数量或坐标值是否与基准一致
                raise ValueError(  # 对齐失败时抛出详细错误
                    f"分组 {group_suffix} 的 {component} 在文件 {os.path.basename(filepath)} 中的 x/h 与同组其他文件不一致，请检查节点对齐。"  # 拼接包含分组名与文件名的错误信息
                )
        pga_series.append(pga_arr)  # 将当前文件 PGA 序列加入绘图列表
    return x_ref, pga_series  # 返回当前分量的参考坐标与多条曲线数据


def plot_component(ax, x_h, pga_series, component, labels, group_suffix):
    pga_mean = np.mean(np.vstack(pga_series), axis=0)  # 计算当前分量在所有地震波下的平均 PGA 曲线

    # 绘制每条地震波曲线，并将 CSV 文件名作为图例标签
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pga_series), 3)))  # 依据地震波数量生成可区分的颜色序列
    for arr, color, label_name in zip(pga_series, colors, labels):  # 遍历每条地震波曲线及其颜色和名称
        ax.plot(x_h, arr, color=color, linestyle='--', linewidth=1.0, alpha=0.85, label=label_name)  # 绘制单条地震波曲线并标注其名称

    # 绘制平均值曲线 (红粗实线)
    ax.plot(x_h, pga_mean, color='#d62728', linestyle='-', linewidth=2.0, label='平均值')  # 叠加平均曲线并使用中文图例

    # 标题与坐标轴标签
    component_text = '水平向' if component == 'PGA_h' else '竖向'  # 将分量代码映射为中文分量名称
    ax.set_title(f'入射角 θs = {INCIDENT_ANGLE}°', fontsize=14, fontproperties=CN_FONT)  # 在标题中增加分组信息以区分不同后缀工况
    ax.set_xlabel('x/h', fontsize=12, fontproperties=EN_FONT)  # 将英文横坐标显式设为 Times 系字体

    # 根据目标列设置 Y 轴标签
    if component == 'PGA_h':  # 判断是否为水平向 PGA 分量
        ax.set_ylabel('水平向 ah,max (g)', fontsize=12, fontproperties=CN_FONT)  # 使用纯文本中文纵坐标标签以避免 mathtext 中文缺字
    else:  # 当前为竖向 PGA 分量
        ax.set_ylabel('竖向 av,max (g)', fontsize=12, fontproperties=CN_FONT)  # 使用纯文本中文纵坐标标签以避免 mathtext 中文缺字

    # 设置 X 轴范围和刻度 (固定为 0~8)
    ax.set_xlim(0, 8)
    ax.set_xticks(np.arange(0, 9, 1))

    # 设置 Y 轴范围和刻度
    ax.set_ylim(*YLIMS[component])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # 添加图例
    legend = ax.legend(loc='upper right', frameon=True, fontsize=9, prop=CN_FONT)  # 图例统一使用中文字体属性以确保“平均值”等中文可显示
    legend.get_frame().set_edgecolor('black')

    # 标记坡顶 (#1) 和坡脚 (#3)
    ax.axvline(x=LOC_CREST, color='gray', linestyle='--', linewidth=1.2)
    ax.axvline(x=LOC_TOE, color='gray', linestyle='--', linewidth=1.2)

    # 在垂直虚线旁边加上文字标注
    y_text = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(LOC_CREST - 0.06, y_text, '坡顶', fontsize=11, fontproperties=CN_FONT, va='top', ha='right')  # 在坡顶虚线左侧上部添加“坡顶”标注
    ax.text(LOC_TOE + 0.06, y_text, '坡脚', fontsize=11, fontproperties=CN_FONT, va='top', ha='left')  # 在坡脚虚线右侧上部添加“坡脚”标注

    # 开启网格线和刻度朝内
    ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True, labelsize=11)

# ==========================================
# 3. 开始绘图 (复刻论文风格)
# ==========================================
# 设置全局字体基线为英文 Times 风格，并通过字体属性单独控制中文
plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 全局默认英文采用 Times New Roman 或同类衬线字体
plt.rcParams['mathtext.fontset'] = 'stix'  # 保留数学字体为 STIX 以匹配 Times 风格
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示为方块的问题

# 收集 CSV 文件（运行时当前目录）
csv_files = collect_csv_files(CSV_GLOB_PATTERN)  # 获取脚本目录下符合前缀模式的全部 CSV 文件
csv_groups = group_csv_files_by_suffix(csv_files)  # 按“第二个连字符后缀”对 CSV 进行分组
filtered_groups = {k: v for k, v in csv_groups.items() if should_include_group(k)}  # 按 normalized 开关过滤需要绘制的分组
if not filtered_groups:  # 判断过滤后是否仍有可绘制分组
    raise ValueError('过滤后没有可绘制分组，请检查 INCLUDE_NORMALIZED_GROUPS、INCLUDE_FLAT_GROUPS 配置与文件命名。')  # 无可绘制分组时抛出提示错误

# 使用脚本当前目录作为输出目录
output_path = SCRIPT_DIR  # 将图片直接输出到脚本所在目录
angle_text = str(int(INCIDENT_ANGLE)) if float(INCIDENT_ANGLE).is_integer() else str(INCIDENT_ANGLE)  # 将角度格式化为便于命名的字符串

for group_suffix, group_files in filtered_groups.items():  # 逐个分组绘图，确保同组文件绘制在同一张图中
    csv_labels = [parse_motion_and_group(os.path.splitext(os.path.basename(p))[0])[0] for p in group_files]  # 从组内文件名解析波名作为图例标签
    safe_group_suffix = sanitize_for_filename(group_suffix)  # 将分组名转换为可安全写入文件名的字符串

    if PLOT_COMPONENTS_TOGETHER:  # 判断是否启用“水平+竖向合并出图”模式
        fig, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(6 * len(TARGET_COLUMNS), 4), squeeze=False)  # 创建横向多子图画布用于合并展示
        axes = axes.flatten()  # 拉平子图数组以便循环时统一处理
        for ax, component in zip(axes, TARGET_COLUMNS):  # 遍历目标分量并绘制到合并图各子图
            x_ref, pga_series = build_component_series(group_files, component, group_suffix)  # 构建当前分量在该分组下的曲线序列
            plot_component(ax, x_ref, pga_series, component, csv_labels, group_suffix)  # 将当前分量数据绘制到对应子图
        plt.tight_layout()  # 自动调整布局避免子图元素遮挡
        combined_file = os.path.join(output_path, f'both_{safe_group_suffix}_angle{angle_text}.png')  # 生成当前分组合并图输出路径
        fig.savefig(combined_file, dpi=300, bbox_inches='tight')  # 保存当前分组的合并图
        fig.canvas.draw()  # 渲染画布以便获取子图边界用于裁剪单图
        renderer = fig.canvas.get_renderer()  # 获取画布渲染器用于计算紧边界
        for ax, component in zip(axes, TARGET_COLUMNS):  # 逐个子图导出对应分量单图
            bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())  # 计算当前子图在英寸坐标中的紧边界
            single_file = os.path.join(output_path, f'{component}_{safe_group_suffix}_angle{angle_text}.png')  # 生成当前分量单图输出路径
            fig.savefig(single_file, dpi=300, bbox_inches=bbox)  # 按子图边界裁剪保存当前分量单图
        plt.close(fig)  # 关闭当前分组合并图对象并释放内存
    else:  # 当前选择分量分开出图模式
        for component in TARGET_COLUMNS:  # 遍历目标分量并分别生成独立图片
            x_ref, pga_series = build_component_series(group_files, component, group_suffix)  # 构建当前分量在该分组下的曲线序列
            fig, ax = plt.subplots(1, 1, figsize=(6, 4), squeeze=False)  # 创建单子图画布用于独立导出
            ax = ax.flatten()[0]  # 提取单个坐标轴对象用于绘图
            plot_component(ax, x_ref, pga_series, component, csv_labels, group_suffix)  # 将当前分量数据绘制到单图中
            plt.tight_layout()  # 自动调整布局避免标签遮挡
            single_file = os.path.join(output_path, f'{component}_{safe_group_suffix}_angle{angle_text}.png')  # 生成当前分量单图输出路径
            fig.savefig(single_file, dpi=300, bbox_inches='tight')  # 直接保存当前分量单图
            plt.close(fig)  # 关闭当前分量图对象并释放内存