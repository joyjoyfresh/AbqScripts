import pandas as pd
import numpy as np
import matplotlib  # 导入 matplotlib 主模块以设置无界面后端
matplotlib.use('Agg')  # 使用 Agg 后端防止运行时弹出图形窗口
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import glob
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 配置文件路径与参数
# ==========================================
# 自动读取当前目录下“文件名前缀为 PGA”的 CSV 文件
CSV_GLOB_PATTERN = 'PGA*.csv'

# 当前绘制的入射角度 (用于图表标题): 从同目录 CAE 文件名解析 a 后数字
# 例如: h100_i45_a0.cae -> INCIDENT_ANGLE = 0
def parse_incident_angle_from_cae(script_dir):
    cae_files = sorted(glob.glob(os.path.join(script_dir, '*.cae')))
    if not cae_files:
        raise FileNotFoundError('脚本目录下未找到 .cae 文件，无法自动解析入射角度。')

    # 若有多个 CAE，默认取排序后的第一个
    cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]
    match = re.search(r'a(-?\d+(?:\.\d+)?)$', cae_name)
    if not match:
        raise ValueError(f"CAE 文件名 {cae_name}.cae 不符合要求，未找到末尾 'a' 后数字。")
    return float(match.group(1))


INCIDENT_ANGLE = parse_incident_angle_from_cae(SCRIPT_DIR)

# 要绘制的 PGA 列：
# 可选值：'PGA_h' (对应图12a), 'PGA_v' (对应图12b)
# 例如只画水平：TARGET_COLUMNS = ['PGA_h']
TARGET_COLUMNS = ['PGA_h', 'PGA_v']

# 各分量的默认 Y 轴范围 (请按你的数据微调)
YLIMS = {
    'PGA_h': (0.2, 1.2),  
    'PGA_v': (0.0, 0.4),
}

# 坡顶和坡脚的归一化横坐标位置 (h=100, i=45° 时为 3.0 和 4.0)
LOC_CREST = 3.0
LOC_TOE = 4.0


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


def plot_component(ax, x_h, pga_series, component, labels):
    pga_mean = np.mean(np.vstack(pga_series), axis=0)  # 计算当前分量在所有地震波下的平均 PGA 曲线

    # 绘制每条地震波曲线，并将 CSV 文件名作为图例标签
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pga_series), 3)))  # 依据地震波数量生成可区分的颜色序列
    for arr, color, label_name in zip(pga_series, colors, labels):  # 遍历每条地震波曲线及其颜色和名称
        ax.plot(x_h, arr, color=color, linestyle='--', linewidth=1.0, alpha=0.85, label=label_name)  # 绘制单条地震波曲线并标注其名称

    # 绘制平均值曲线 (红粗实线)
    ax.plot(x_h, pga_mean, color='#d62728', linestyle='-', linewidth=2.0, label='平均值')  # 叠加平均曲线并使用中文图例

    # 标题与坐标轴标签
    component_text = '水平向' if component == 'PGA_h' else '竖向'  # 将分量代码映射为中文分量名称
    ax.set_title(f'入射角 θs = {INCIDENT_ANGLE}°（{component_text}）', fontsize=14, fontproperties=CN_FONT)  # 使用纯文本标题避免中文进入数学公式解析器
    ax.set_xlabel('x/h', fontsize=12, fontproperties=EN_FONT)  # 将英文横坐标显式设为 Times 系字体

    # 根据目标列设置 Y 轴标签
    if component == 'PGA_h':  # 判断是否为水平向 PGA 分量
        ax.set_ylabel('水平向峰值加速度 ah,max (g)', fontsize=12, fontproperties=CN_FONT)  # 使用纯文本中文纵坐标标签以避免 mathtext 中文缺字
    else:  # 当前为竖向 PGA 分量
        ax.set_ylabel('竖向峰值加速度 av,max (g)', fontsize=12, fontproperties=CN_FONT)  # 使用纯文本中文纵坐标标签以避免 mathtext 中文缺字

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
    ax.text(LOC_CREST + 0.1, y_text, '坡顶', fontsize=12, verticalalignment='bottom', fontproperties=CN_FONT)  # 将 #1 替换为中文地貌标注“坡顶”并绑定宋体
    ax.text(LOC_TOE + 0.1, y_text, '坡底', fontsize=12, verticalalignment='bottom', fontproperties=CN_FONT)  # 将 #3 替换为中文地貌标注“坡底”并绑定宋体

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
csv_files = collect_csv_files(CSV_GLOB_PATTERN)
csv_labels = [extract_motion_name(os.path.splitext(os.path.basename(p))[0]) for p in csv_files]  # 从 CSV 文件名中提取地震波名称用于图例

# 创建子图：可同时绘制水平与竖向 PGA
fig, axes = plt.subplots(1, len(TARGET_COLUMNS), figsize=(6 * len(TARGET_COLUMNS), 4), squeeze=False)
axes = axes.flatten()

for ax, component in zip(axes, TARGET_COLUMNS):
    # 读取所有 CSV 的同一分量数据，并检查长度与 x/h 对齐
    x_ref = None
    pga_series = []

    for filepath in csv_files:
        x_h, pga_arr = load_pga_data(filepath, component)
        if x_ref is None:
            x_ref = x_h
        else:
            if len(x_h) != len(x_ref) or not np.allclose(x_h, x_ref):
                raise ValueError(
                    f"{component} 在文件 {os.path.basename(filepath)} 中的 x/h 与其他文件不一致，请检查节点对齐。"
                )
        pga_series.append(pga_arr)

    plot_component(ax, x_ref, pga_series, component, csv_labels)

plt.tight_layout()

# 使用脚本当前目录作为输出目录
output_path = SCRIPT_DIR  # 将图片直接输出到脚本所在目录

# 保存合并图
angle_text = str(int(INCIDENT_ANGLE)) if float(INCIDENT_ANGLE).is_integer() else str(INCIDENT_ANGLE)  # 将角度格式化为便于命名的字符串
combined_file = os.path.join(output_path, f'both_angle{angle_text}.png')  # 生成合并图输出文件完整路径（不含 Fig12 前缀）
fig.savefig(combined_file, dpi=300, bbox_inches='tight')  # 以 300dpi 保存合并图并紧凑裁边

# 保存每个分量的单图（从合并图中裁剪对应子图区域）
fig.canvas.draw()  # 先渲染画布以确保可获取准确的子图边界
renderer = fig.canvas.get_renderer()  # 获取当前画布渲染器用于计算子图包围盒
for ax, component in zip(axes, TARGET_COLUMNS):
    bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())  # 计算当前子图在英寸坐标下的紧边界
    single_file = os.path.join(output_path, f'{component}_angle{angle_text}.png')  # 生成当前分量单图输出文件完整路径（不含 Fig12 前缀）
    fig.savefig(single_file, dpi=300, bbox_inches=bbox)  # 按子图边界裁剪并保存单图 PNG

# 不弹出窗口，直接关闭图形
plt.close(fig)  # 主动关闭图形对象以避免界面弹窗并释放内存