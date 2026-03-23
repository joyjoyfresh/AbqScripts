import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. 配置文件路径与参数
# ==========================================
# 自动读取当前目录下的所有 CSV 文件
CSV_GLOB_PATTERN = '*.csv'

# 当前绘制的入射角度 (用于图表标题)
INCIDENT_ANGLE = 10  

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


def plot_component(ax, x_h, pga_series, component, labels):
    pga_mean = np.mean(np.vstack(pga_series), axis=0)

    # 绘制独立输入波曲线：只给第一条加图例标签，避免图例重复
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pga_series), 3)))
    for i, (arr, color, label_name) in enumerate(zip(pga_series, colors, labels)):
        legend_label = 'individual input motion' if i == 0 else None
        ax.plot(x_h, arr, color=color, linestyle='--', linewidth=1.0, alpha=0.85, label=legend_label)

    # 绘制平均值曲线 (红粗实线)
    ax.plot(x_h, pga_mean, color='#d62728', linestyle='-', linewidth=2.0, label='mean value')

    # 标题与坐标轴标签
    ax.set_title(rf'$\theta_s = {INCIDENT_ANGLE}^\circ$ ({component})', fontsize=14)
    ax.set_xlabel('$x/h$', fontsize=12)

    # 根据目标列设置 Y 轴标签
    if component == 'PGA_h':
        ax.set_ylabel(r'$a_{h,max}\,(\mathrm{g})$', fontsize=12)
    else:
        ax.set_ylabel(r'$a_{v,max}\,(\mathrm{g})$', fontsize=12)

    # 设置 X 轴范围和刻度 (固定为 0~8)
    ax.set_xlim(0, 8)
    ax.set_xticks(np.arange(0, 9, 1))

    # 设置 Y 轴范围和刻度
    ax.set_ylim(*YLIMS[component])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    # 添加图例
    legend = ax.legend(loc='upper right', frameon=True, fontsize=11)
    legend.get_frame().set_edgecolor('black')

    # 标记坡顶 (#1) 和坡脚 (#3)
    ax.axvline(x=LOC_CREST, color='gray', linestyle='--', linewidth=1.2)
    ax.axvline(x=LOC_TOE, color='gray', linestyle='--', linewidth=1.2)

    # 在垂直虚线旁边加上文字标注
    y_text = ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(LOC_CREST + 0.1, y_text, '#1', fontsize=12, verticalalignment='bottom')
    ax.text(LOC_TOE + 0.1, y_text, '#3', fontsize=12, verticalalignment='bottom')

    # 开启网格线和刻度朝内
    ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.3)
    ax.tick_params(direction='in', top=True, right=True, labelsize=11)

# ==========================================
# 3. 开始绘图 (复刻论文风格)
# ==========================================
# 设置字体为 Times New Roman (论文常用)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# 收集 CSV 文件（运行时当前目录）
csv_files = collect_csv_files(CSV_GLOB_PATTERN)
csv_labels = [os.path.splitext(os.path.basename(p))[0] for p in csv_files]

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
# 也可以取消注释下面这行直接保存图片
# plt.savefig(f'Fig12_both_angle{INCIDENT_ANGLE}.png', dpi=300)
plt.show()