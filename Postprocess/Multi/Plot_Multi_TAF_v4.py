# -*- coding: utf-8 -*-
"""复现论文【图8】风格的跨工况 TAF 对比图（v4：集中 results/index 模式 + 图8 排版）。

论文图8：基岩-覆盖层双层模型，阻抗比 Vr/Vs2=1.25、覆盖层厚度比 h/H=0.50 的典型工况。
  整张大图竖向分为两个宏观大组：
    (a) 缓边坡 i=30°  在上；
    (b) 陡边坡 i=60°  在下。
  每个大组内部是 2×2 子图矩阵：
    行 = SV 波入射角 θs —— 第一行 θs=0°（垂直入射），第二行 θs=15°（斜入射）；
    列 = 响应分量 —— 左列 Horizontal TAF，右列 Vertical TAF；
    每个面板内画 3 条按无量纲频率 a0 区分的曲线：
      a0=1.0（虚线）、a0=1.5（点划线）、a0=2.0（实线）。
  公共要素：横轴 Surface Receiver Location (m)，细密灰色网格，坡顶/坡脚两条黑色竖虚线标注 #1/#2。

数据来源（仅集中 results/ 模式，与 Plot_Fig15_compare_v3 同口径）：
  先用 Collect_TAF_results_v1.py 把各工况 CSV 汇到 results/，本脚本读取：
    - results/index.csv：每个 TAF 文件的统一规范列（由 Collect_TAF_results_v2 据 case_meta.json 写出，
      含 slope_i 坡角、incident_angle 入射角 θs、a0_base 无量纲频率换算基数、各层 Vs、Vr/Vs2 等）；
    - results/ 下 TAF-*.csv：地表 TAF 曲线数据；
    - results/ 内任一 PGA-*-slope.csv：坡顶/坡脚(#1/#2)位置。
  坡角 i、入射角 θs 直接取自 index.csv 规范列；无量纲频率 a0 由 record 主频 fc(Hz) 与该工况 a0_base
    按 a0 = fc·a0_base 计算（a0_base=2·(H−h)/Vs2，逐工况随 case_meta 给定，无需在此猜 Vs2/几何）。

运行：
  先 `python Postprocess/Collect_TAF_results_v2.py <工况根目录>` 生成 results/index.csv，再
  `python Postprocess/Plot_Multi_TAF_v4.py <工况根目录>`（自动找其下 results/）
  或直接 `python Postprocess/Plot_Multi_TAF_v4.py <results 目录>`；不传参数取当前目录。
"""

import os  # 导入系统接口模块
import re  # 导入正则模块（从记录名解析输入波主频 f_c）
import sys  # 导入系统参数模块
import glob  # 导入文件匹配模块
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.font_manager as fm  # 导入字体管理器
import matplotlib.ticker as mticker  # 导入刻度定位器（主/次刻度、网格）


# ==============================================================================
#  配置与常量
# ==============================================================================
A0_DECIMALS = 1  # a0 数值四舍五入保留的小数位（用于分组与样式匹配）

# 各 a0 对应的曲线样式（论文图8：1.0 虚线、1.5 点划线、2.0 实线，配不同颜色便于区分）
A0_STYLES = {
    1.0: {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 1.6},  # a0=1.0：绿色虚线
    1.5: {'color': '#1f77b4', 'linestyle': '-.', 'linewidth': 1.6},  # a0=1.5：蓝色点划线
    2.0: {'color': '#d62728', 'linestyle': '-', 'linewidth': 1.6},   # a0=2.0：红色实线
}
FALLBACK_STYLES = [  # a0 不在上表时按序兜底取用的样式
    {'color': '#9467bd', 'linestyle': ':', 'linewidth': 1.6},  # 兜底1：紫色点线
    {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.6},  # 兜底2：橙色实线
    {'color': '#8c564b', 'linestyle': '--', 'linewidth': 1.6}, # 兜底3：棕色虚线
]

COLUMNS = [  # 子图列定义：(数据键, 纵轴标签, 纵轴范围或None自适应)
    ('taf_h', 'Horizontal TAF', None),  # 左列：水平向 TAF
    ('taf_v', 'Vertical TAF', None),    # 右列：竖向 TAF
]
GROUP_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f']  # 宏观大组的子标签 (a)/(b)/...

# ==============================================================================
#  字体与坐标轴样式
# ==============================================================================
def build_font_properties():  # 加载中英文字体
    """加载并返回 (中文字体, 英文字体) 属性。"""
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']  # 中文候选字体名
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']  # 英文候选字体名
    cn_font, en_font = fm.FontProperties(), fm.FontProperties()  # 初始化默认字体属性
    for name in cn_candidates:  # 遍历中文候选逐个尝试加载
        try:  # 尝试查找该中文字体路径
            cn_font = fm.FontProperties(fname=fm.findfont(name, fallback_to_default=False)); break  # 成功则实例化并跳出
        except Exception:  # 加载失败
            continue  # 尝试下一候选
    for name in en_candidates:  # 遍历英文候选逐个尝试加载
        try:  # 尝试查找该英文字体路径
            en_font = fm.FontProperties(fname=fm.findfont(name, fallback_to_default=False)); break  # 成功则实例化并跳出
        except Exception:  # 加载失败
            continue  # 尝试下一候选
    return cn_font, en_font  # 返回中英文字体属性


CN_FONT, EN_FONT = build_font_properties()  # 加载并缓存中英文字体属性


def style_axes(ax):  # 统一坐标轴外观
    """设置白底、四面朝内刻度、黑色边框、细密灰色网格（主+次）。"""
    ax.set_facecolor('white')  # 设置面板背景为白色
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True, labelsize=10)  # 主次刻度均朝内、四面显示
    for spine in ax.spines.values():  # 遍历四条边框线
        spine.set_color('black'); spine.set_linewidth(1.0)  # 设为黑色实线边框
    ax.minorticks_on()  # 打开次刻度（细密网格用）
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)  # 主网格（较明显）
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)  # 次网格（更细密）


# ==============================================================================
#  数据来源判别（仅集中 results/ 模式）
# ==============================================================================
def resolve_results_dir(arg):  # 判别集中结果 results/ 目录
    """返回含 index.csv 的 results 目录；传入目录本身即 results/ 或其下含 results/ 均可；找不到返回 None。"""
    base = os.path.abspath(arg) if arg else os.getcwd()  # 基准目录：命令行参数或当前目录
    if os.path.isfile(os.path.join(base, 'index.csv')):  # 基准目录本身即 results/（直接含清单）
        return base  # 直接使用基准目录
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')):  # 基准目录下存在 results/ 子目录
        return os.path.join(base, 'results')  # 使用其下 results/
    return None  # 未找到集中结果目录


def read_crest_toe_any(data_dir):  # 从目录内任一 PGA-*-slope.csv 读取坡顶/坡脚
    """集中模式下各工况几何相同，取任一坡地 PGA 表得 (x_crest, x_toe, total_L)。"""
    pgas = glob.glob(os.path.join(data_dir, 'PGA-*-slope.csv'))  # 该目录下全部坡地 PGA 表
    for p in pgas:  # 逐个尝试读取
        try:  # 尝试解析几何
            df = pd.read_csv(p)  # 读取 PGA 坡地表
            ymax, ymin = df['y'].max(), df['y'].min()  # 取顶/底高程
            x_crest = float(df[df['y'] >= ymax - 1e-3]['x'].max())  # 坡顶最大 x
            x_toe = float(df[df['y'] <= ymin + 1e-3]['x'].min())  # 坡脚最小 x
            return x_crest, x_toe, float(df['x'].max())  # 返回坡顶、坡脚、总长
        except Exception:  # 读取失败
            continue  # 尝试下一个文件
    return None, None, None  # 无可用几何信息


# ==============================================================================
#  坡角 i 与无量纲频率 a0 的读取（直接用 index.csv 规范列，不再解析文件夹名）
# ==============================================================================
def _num(row, col):  # 从 index 行安全取数值列
    """取 row[col] 并转 float；列缺失/空/非数值返回 None。"""
    if col not in row or pd.isna(row.get(col)):  # 列不存在或为空
        return None  # 返回空
    try:  # 尝试转换
        return float(row[col])  # 返回浮点
    except (TypeError, ValueError):  # 非数值
        return None  # 返回空


def compute_a0(row):  # 由 index 行推导无量纲频率 a0
    """用该工况规范列 a0_base 与 record 主频按 a0=fc·a0_base 计算（逐工况自洽）；失败返回 None。

    薄契约：仅依赖 index.csv 的 a0_base 与 record 两列（由 Collect 据 case_meta.json 展平而来）。
    """
    a0_base = _num(row, 'a0_base')  # 该工况的 a0 换算基数（index 规范列，来自 case_meta.json）
    record = row.get('record')  # 输入波记录名
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(record), re.IGNORECASE) if record is not None else None  # 解析主频 f_c(Hz)
    if a0_base is None or m is None:  # 缺基数或主频则无法换算
        return None  # 返回空
    a0 = float(m.group(1)) * a0_base  # a0 = f_c(Hz) × a0_base
    return round(a0, A0_DECIMALS)  # 四舍五入返回


def style_for_a0(a0, used):  # 为某 a0 取曲线样式
    """返回 (style_dict, label)；优先用预设 a0 样式，否则按序兜底。"""
    key = round(a0, A0_DECIMALS)  # 规范化 a0 键
    label = r'$a_0=%.1f$' % a0  # 图例文本（数学体）
    if key in A0_STYLES:  # 命中预设样式
        return dict(A0_STYLES[key]), label  # 返回预设样式副本与图例
    return dict(FALLBACK_STYLES[used % len(FALLBACK_STYLES)]), label  # 返回兜底样式与图例


# ==============================================================================
#  从 results/index.csv 收集工况记录
# ==============================================================================
def collect_cases_from_index(results_dir):  # 读取集中结果并整理为工况记录
    """读 index.csv + results/ 下 TAF-*.csv，返回 (工况记录列表, x_crest, x_toe, total_L)。

    每条记录含：x, taf_h, taf_v, slope_i(坡角), theta(入射角), a0(无量纲频率), folder。
    """
    index_path = os.path.join(results_dir, 'index.csv')  # 清单文件路径
    idx = pd.read_csv(index_path)  # 读取清单为 DataFrame
    taf_rows = idx[idx['type'].astype(str).str.upper() == 'TAF']  # 仅保留 TAF 类型行
    cases = []  # 工况记录列表
    for _, r in taf_rows.iterrows():  # 遍历每个 TAF 文件记录
        fname = str(r['collected_file'])  # 收集后的文件名
        fpath = os.path.join(results_dir, fname)  # 完整路径
        if not os.path.isfile(fpath):  # 文件缺失
            print('  跳过(文件不存在): %s' % fname); continue  # 提示并跳过
        slope_i = _num(r, 'slope_i')  # 坡角 i（index 规范列，来自 case_meta 几何）
        theta = _num(r, 'incident_angle')  # 入射角 θs（index 规范列）
        a0 = compute_a0(r)  # 无量纲频率 a0（a0_base × 记录主频）
        if slope_i is None or a0 is None or theta is None:  # 关键分组属性缺失
            print('  跳过(无法识别 i/a0/θs): %s (i=%s, a0=%s, θs=%s)' % (fname, slope_i, a0, theta)); continue  # 提示并跳过
        try:  # 读取 TAF 曲线数据
            df = pd.read_csv(fpath)  # 读取 TAF 表
            x = df['x'].to_numpy(float)  # 地表 x 坐标
            th = df['TAF_h'].to_numpy(float)  # 水平向 TAF
            tv = df['TAF_v'].to_numpy(float)  # 竖向 TAF
        except Exception as e:  # 读取失败
            print('  跳过(读取 TAF 失败 %s): %s' % (e, fname)); continue  # 提示并跳过
        cases.append({'folder': str(r.get('source_folder', fname)), 'x': x, 'taf_h': th, 'taf_v': tv,  # 追加工况记录
                      'slope_i': float(slope_i), 'theta': float(theta), 'a0': float(a0)})
        print('  收录 %s -> i=%g°, θs=%g°, a0=%.1f' % (fname, slope_i, theta, a0))  # 打印收录信息
    x_crest, x_toe, total_L = read_crest_toe_any(results_dir)  # 读取全局坡顶/坡脚/总长
    if total_L is None and cases:  # 几何兜底：用 TAF 的 x 最大值
        total_L = max(float(c['x'].max()) for c in cases)  # 取所有曲线 x 最大值
    return cases, x_crest, x_toe, total_L  # 返回收集结果


# ==============================================================================
#  绘图（论文图8 排版：宏观大组=坡角 i，组内 2×2：行=入射角 θs，列=H/V，曲线=a0）
# ==============================================================================
def set_x_axis(ax, total_L):  # 统一设置横轴范围与刻度
    """横轴 0–total_L；若接近 1800 用 0/600/1200/1800，否则自动取整刻度。"""
    x_max = total_L if (total_L and total_L > 1) else 1800.0  # 横轴上限（默认 1800）
    ax.set_xlim(0, x_max)  # 设置横轴范围
    if abs(x_max - 1800.0) < 50.0:  # 接近论文 1800 m 长坡
        ax.set_xticks([t for t in (0, 600, 1200, 1800) if t <= x_max + 1e-6])  # 主刻度 0/600/1200/1800
    else:  # 其他长度
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))  # 自动约 4 个整数刻度
    return x_max  # 返回横轴上限供标注使用


def auto_ylim(panels_data, fixed):  # 计算面板纵轴范围
    """fixed 非空则直接返回；否则按数据极值留 1/5 顶部空白给图例后取整。"""
    if fixed is not None:  # 已指定固定范围
        return fixed  # 直接返回
    if not panels_data:  # 无数据兜底
        return (0.0, 1.0)  # 返回默认范围
    allv = np.concatenate(panels_data)  # 拼接该列全部曲线数据
    ymin = min(0.0, float(allv.min()))  # 下限不超过 0
    ymax = float(allv.max())  # 数据最大值
    span = ymax - ymin if ymax > ymin else 1.0  # 数据跨度
    top = ymin + span * (5.0 / 4.0)  # 顶部预留 1/5 空白给图例
    step = 10.0 ** np.floor(np.log10(span / 4.0)) if span > 0 else 1.0  # 估算刻度量级
    return (np.floor(ymin / step) * step, np.ceil(top / step) * step)  # 按量级取整后返回


def plot_fig8(cases, x_crest, x_toe, total_L, out_dir):  # 绘制论文图8 风格大图
    """宏观大组按坡角 i 竖向堆叠 (a)/(b)…；每组 2×2：行=入射角 θs（升序，0° 在上），列=H/V；面板内按 a0 画曲线。"""
    slope_angles = sorted({c['slope_i'] for c in cases})  # 出现的坡角（升序，决定宏观大组顺序）
    thetas = sorted({c['theta'] for c in cases})  # 出现的入射角（升序，决定组内行顺序）
    n_groups = len(slope_angles)  # 宏观大组数量
    n_theta = max(1, len(thetas))  # 每组行数（入射角数）
    n_col = len(COLUMNS)  # 列数（H/V）
    total_rows = n_groups * n_theta  # 总行数 = 大组数 × 每组行数
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体用 STIX
    fig, axes = plt.subplots(total_rows, n_col, figsize=(5.0 * n_col, 3.6 * total_rows),  # 创建总网格画布
                             dpi=300, squeeze=False)  # 不压缩维度便于二维索引
    # 预先按列计算自适应纵轴范围（同一列所有面板共用，便于跨组对比）
    col_ylim = []  # 每列纵轴范围列表
    for key, _ylabel, fixed in COLUMNS:  # 遍历每一列
        col_data = [c[key] for c in cases]  # 该列全部曲线数据
        col_ylim.append(auto_ylim(col_data, fixed))  # 计算并记录该列范围
    for gi, i_ang in enumerate(slope_angles):  # 遍历宏观大组（坡角）
        for ti, theta in enumerate(thetas):  # 遍历组内行（入射角）
            grid_row = gi * n_theta + ti  # 当前面板所在的总行号
            for ci, (key, ylabel, _fixed) in enumerate(COLUMNS):  # 遍历列（H/V 分量）
                ax = axes[grid_row][ci]  # 当前面板坐标轴
                style_axes(ax)  # 套用统一外观（含细密网格）
                used = 0  # 兜底样式计数器
                a0_list = sorted({c['a0'] for c in cases  # 该(坡角,入射角)下出现的 a0（升序）
                                  if abs(c['slope_i'] - i_ang) < 1e-6 and abs(c['theta'] - theta) < 1e-6})
                for a0 in a0_list:  # 按 a0 逐条画曲线
                    match = [c for c in cases if abs(c['slope_i'] - i_ang) < 1e-6  # 匹配该 (i, θs, a0) 工况
                             and abs(c['theta'] - theta) < 1e-6 and abs(c['a0'] - a0) < 1e-6]
                    if not match:  # 无数据
                        continue  # 跳过
                    c = match[0]  # 取该工况
                    s, label = style_for_a0(a0, used); used += 1  # 取样式与图例
                    ax.plot(c['x'], c[key], color=s['color'], linestyle=s['linestyle'],  # 绘制 TAF 曲线
                            linewidth=s['linewidth'], label=label)
                x_max = set_x_axis(ax, total_L)  # 设置横轴范围与刻度
                ax.set_ylim(*col_ylim[ci])  # 设置该列统一纵轴范围
                if x_crest is not None:  # 标注坡顶/坡脚 #1/#2
                    ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)  # 坡顶竖虚线
                    ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=1.0)  # 坡脚竖虚线
                    y0, y1 = col_ylim[ci]  # 当前列纵轴范围
                    ty = y0 + 0.95 * (y1 - y0)  # #1/#2 文本纵位置
                    ax.text(x_crest - 0.015 * x_max, ty, '#1', fontsize=10, fontproperties=EN_FONT, va='top', ha='right')  # 标注 #1
                    ax.text(x_toe - 0.015 * x_max, ty, '#2', fontsize=10, fontproperties=EN_FONT, va='top', ha='right')  # 标注 #2
                ax.set_title(r'$\theta_s = %g^\circ$' % theta, fontsize=12, fontproperties=EN_FONT, pad=6)  # 面板入射角标题
                if grid_row == total_rows - 1:  # 仅最底行加横轴标签
                    ax.set_xlabel('Surface Receiver Location (m)', fontsize=12, fontproperties=EN_FONT)  # 横轴标签
                ax.set_ylabel(ylabel, fontsize=12, fontproperties=EN_FONT)  # 纵轴标签（按列）
                ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black',  # 面板左上角图例
                          framealpha=1.0, prop=EN_FONT, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2)  # 调整布局并为大组标签留出顶部空隙
    # 在每个宏观大组顶部左侧标注 (a) i=30° / (b) i=60° …（布局完成后按面板位置放置）
    for gi, i_ang in enumerate(slope_angles):  # 遍历宏观大组
        top_ax = axes[gi * n_theta][0]  # 该组首行左面板
        pos = top_ax.get_position()  # 取其在画布中的位置
        letter = GROUP_LETTERS[gi] if gi < len(GROUP_LETTERS) else str(gi + 1)  # 大组子标签
        fig.text(pos.x0, min(0.998, pos.y1 + 0.012),  # 标签放在该组首行面板左上方
                 r'(%s)  $i=%g^\circ$' % (letter, i_ang), fontsize=13, fontproperties=EN_FONT,
                 va='bottom', ha='left', fontweight='bold')  # 加粗显示大组标签
    out_name = 'Fig8_TAF_compare.png'  # 输出文件名
    out_path = os.path.join(out_dir, out_name)  # 输出完整路径
    fig.savefig(out_path, dpi=300, bbox_inches='tight')  # 保存图像并去除多余白边
    plt.close(fig)  # 释放画布
    print('  已输出: %s' % out_name)  # 提示输出完成


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动图8 风格 TAF 对比绘图 (Plot_Multi_TAF_v4)...')  # 启动提示
    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数（目录）
    data_dir = resolve_results_dir(arg)  # 定位集中结果 results/ 目录
    if data_dir is None:  # 未找到 index.csv
        print('错误：未找到 index.csv（集中结果）。'
              '\n请先运行 Collect_TAF_results_v1.py 生成 results/index.csv，'
              '\n再把"工况根目录"（其下含 results/）或"results 目录"作为参数传入；不传则取当前目录。')  # 错误提示
        return  # 退出
    print('>>> 集中模式：读取 %s 内的 index.csv 与 TAF-*.csv' % data_dir)  # 提示数据目录
    cases, x_crest, x_toe, total_L = collect_cases_from_index(data_dir)  # 从清单收集工况
    if not cases:  # 无可用工况
        print('错误：index.csv 中没有可识别(i/θs/a0)的 TAF 工况记录。')  # 错误提示
        return  # 退出
    n_i = len({c['slope_i'] for c in cases})  # 坡角档数
    n_t = len({c['theta'] for c in cases})  # 入射角档数
    n_a = len({c['a0'] for c in cases})  # a0 档数
    print('>>> 共 %d 个工况：坡角 %d 档、入射角 %d 档、a0 %d 档；按图8 排版出图。' % (len(cases), n_i, n_t, n_a))  # 汇总
    plot_fig8(cases, x_crest, x_toe, total_L, data_dir)  # 绘制图8 并输出到 results/ 目录
    print('>>> 图8 风格对比图完成，输出目录：%s' % data_dir)  # 完成提示


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
