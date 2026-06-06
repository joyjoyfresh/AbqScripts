# -*- coding: utf-8 -*-
"""复现 Shen 等 2025 论文【图15】风格的跨工况对比图（v3：新排版 + 仅集中模式）。

论文图15：i=45°, Vr/Vs2=2.5, h/H=0.50 的三层斜坡，地表水平/竖向 TAF 沿地表坐标的变化。
  每个厚度一张图：(a) h1/(H−h)=0.25, (b) h1/(H−h)=0.75。
  布局（2×2）：
    行 = 入射角 θs —— 第一行 θs=0°，第二行 θs=15°；
    列 = 响应分量 —— 左列 Horizontal TAF（纵轴 0–6），右列 Vertical TAF（纵轴 0–5）；
    每个面板内对比表层软/硬两条曲线（红实线 Vs1/Vs2=0.5，蓝虚线 Vs1/Vs2=2.0），左上角各放一份图例。
  公共要素：横轴 Surface Receiver Location (m) 0–1800（主刻度 0/600/1200/1800），
    细密灰色正方网格，x≈坡顶/坡脚处两条黑色竖虚线并标注 #1/#2。

数据来源（仅集中模式）：先用 Collect_TAF_results_v1.py 把各工况 CSV 汇到 results/，
  本脚本读 results/index.csv（含每个 TAF 文件的工况属性 Vs1/Vs2、h1/(H−h)、angle）+ results/ 下 TAF-*.csv；
  坡顶/坡脚(#1/#2)位置从 results/ 内任一 PGA-*-slope.csv 读取。
  index.csv 中若含多种记录（如 4/6/8Hz），仅画含 RECORD_PREFER（默认 4Hz）者。

运行：
  先 `python Postprocess/Collect_TAF_results_v1.py <工况根目录>` 生成 results/，再
  `python Postprocess/Plot_Fig15_compare_v3.py <工况根目录>`（自动找其下 results/）
  或直接 `python Postprocess/Plot_Fig15_compare_v3.py <results 目录>`；不传参数取当前目录。
"""

import os  # 导入系统接口模块
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
RECORD_PREFER = '4Hz'  # 优先选用的输入记录标识（图15 用 4 Hz Ricker）：index.csv 中含多种记录时只画含此标识者
# 不同 Vs1/Vs2（软/硬）曲线样式：键为四舍五入到 2 位的比值
RATIO_STYLES = {
    0.5: {'color': '#d62728', 'linestyle': '-', 'linewidth': 1.6, 'marker': '', 'label': r'$V_{s1}/V_{s2}=0.5$'},   # 软表层：红实线
    2.0: {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 1.6, 'marker': '', 'label': r'$V_{s1}/V_{s2}=2.0$'},  # 硬表层：蓝虚线
}
FALLBACK_STYLES = [  # 比值不在上表时按序兜底取用
    {'color': '#2ca02c', 'linestyle': '-.', 'linewidth': 1.6},
    {'color': '#9467bd', 'linestyle': ':', 'linewidth': 1.6},
    {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.6},
]


# ==============================================================================
#  字体与坐标轴样式
# ==============================================================================
def build_font_properties():  # 加载中英文字体
    """加载并返回 (中文字体, 英文字体) 属性。"""
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']  # 中文候选
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']  # 英文候选
    cn_font, en_font = fm.FontProperties(), fm.FontProperties()  # 默认字体
    for name in cn_candidates:  # 匹配中文字体
        try:
            cn_font = fm.FontProperties(fname=fm.findfont(name, fallback_to_default=False)); break
        except Exception:
            continue
    for name in en_candidates:  # 匹配英文字体
        try:
            en_font = fm.FontProperties(fname=fm.findfont(name, fallback_to_default=False)); break
        except Exception:
            continue
    return cn_font, en_font  # 返回字体属性


CN_FONT, EN_FONT = build_font_properties()  # 加载字体


def style_axes(ax):  # 统一坐标轴外观
    """设置白底、四面朝内刻度、黑色边框、细密灰色正方网格（主+次）。"""
    ax.set_facecolor('white')  # 白色背景
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True, labelsize=10)  # 主次刻度均朝内、四面显示
    for spine in ax.spines.values():  # 遍历四条边框
        spine.set_color('black'); spine.set_linewidth(1.0)  # 黑色边框
    ax.minorticks_on()  # 打开次刻度（细密网格用）
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)  # 主网格（较明显）
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)  # 次网格（细密）


# ==============================================================================
#  数据来源判别（仅集中 results/ 模式）
# ==============================================================================
def resolve_results_dir(arg):  # 判别集中结果 results/ 目录
    """返回含 index.csv 的 results 目录；传入目录本身即 results/ 或其下含 results/ 均可；找不到返回 None。"""
    base = os.path.abspath(arg) if arg else os.getcwd()  # 基准目录：参数或当前目录
    if os.path.isfile(os.path.join(base, 'index.csv')):  # 基准目录本身即 results/（含清单）
        return base  # 直接使用
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')):  # 基准目录下有 results/
        return os.path.join(base, 'results')  # 使用其下 results/
    return None  # 未找到集中结果


def read_crest_toe_any(data_dir):  # 从目录内任一 PGA-*-slope.csv 读取坡顶/坡脚
    """集中模式下各工况几何相同，取任一坡地 PGA 表得 (x_crest, x_toe, total_L)。"""
    pgas = glob.glob(os.path.join(data_dir, 'PGA-*-slope.csv'))  # 该目录所有坡地 PGA
    for p in pgas:  # 逐个尝试
        try:
            df = pd.read_csv(p)  # 读取
            ymax, ymin = df['y'].max(), df['y'].min()  # 顶/底高程
            x_crest = float(df[df['y'] >= ymax - 1e-3]['x'].max())  # 坡顶
            x_toe = float(df[df['y'] <= ymin + 1e-3]['x'].min())  # 坡脚
            return x_crest, x_toe, float(df['x'].max())  # 返回几何
        except Exception:
            continue  # 失败则试下一个
    return None, None, None  # 无可用几何


def collect_cases_from_index(results_dir):  # 从 results/index.csv 读取集中结果
    """读 index.csv + results/ 下 TAF-*.csv，返回工况记录列表与全局坡顶/坡脚/总长。"""
    index_path = os.path.join(results_dir, 'index.csv')  # 清单路径
    idx = pd.read_csv(index_path)  # 读取清单
    cases = []  # 工况记录
    taf_rows = idx[idx['type'].astype(str).str.upper() == 'TAF']  # 仅取 TAF 行
    prefer = taf_rows[taf_rows['record'].astype(str).str.contains(RECORD_PREFER, case=False, na=False)]  # 含优先记录标识者
    if len(prefer) > 0:  # 若存在优先记录（如 4Hz）则只画它，避免多频率工况在同面板重叠
        taf_rows = prefer  # 仅保留优先记录
    for _, r in taf_rows.iterrows():  # 遍历每个 TAF 文件
        fname = str(r['collected_file'])  # 收集后文件名
        fpath = os.path.join(results_dir, fname)  # 完整路径
        if not os.path.isfile(fpath):  # 文件缺失
            print('  跳过(文件不存在): %s' % fname); continue  # 提示并跳过
        try:
            vs1_vs2 = float(r['vs1_vs2']); h1_over = float(r['h1_over']); angle = float(r['angle'])  # 工况属性
        except (TypeError, ValueError):  # 属性缺失/非数值
            print('  跳过(index.csv 工况属性缺失): %s' % fname); continue  # 提示并跳过
        try:
            df = pd.read_csv(fpath)  # 读取 TAF 数据
            x = df['x'].to_numpy(float); th = df['TAF_h'].to_numpy(float); tv = df['TAF_v'].to_numpy(float)  # 取列
        except Exception as e:  # 读取失败
            print('  跳过(读取 TAF 失败 %s): %s' % (e, fname)); continue  # 提示并跳过
        cases.append({'folder': str(r.get('source_folder', fname)), 'x': x, 'taf_h': th, 'taf_v': tv,  # 追加记录
                      'vs1_vs2': vs1_vs2, 'h1_over': h1_over, 'angle': angle})
        print('  收录 %s -> Vs1/Vs2=%.2f, h1/(H-h)=%.2f, θs=%g°' % (fname, vs1_vs2, h1_over, angle))  # 打印
    x_crest, x_toe, total_L = read_crest_toe_any(results_dir)  # 全局几何（任一坡地 PGA）
    if total_L is None and cases:  # 几何兜底：用 TAF 的 x 最大值
        total_L = max(float(c['x'].max()) for c in cases)
    return cases, x_crest, x_toe, total_L  # 返回收集结果


# ==============================================================================
#  绘图（论文图15 布局：每个厚度一张图，行=水平/竖向，列=入射角，曲线=软/硬）
# ==============================================================================
def style_for_ratio(ratio, used):  # 为某 Vs1/Vs2 取曲线样式
    """返回 (style_dict, label)；优先用预设软/硬样式，否则按序兜底。"""
    key = round(ratio, 2)  # 比值键
    if key in RATIO_STYLES:  # 命中预设
        s = RATIO_STYLES[key]; return s, s['label']  # 返回预设样式与图例
    s = dict(FALLBACK_STYLES[used % len(FALLBACK_STYLES)])  # 兜底样式
    return s, r'$V_{s1}/V_{s2}=%.2f$' % ratio  # 兜底图例


def plot_one_thickness(h1_over, cases, x_crest, x_toe, total_L, out_dir, part_tag):  # 画一个厚度的图15
    """对某一 h1/(H−h) 厚度出图：行=入射角(0°上/15°下)，列=分量(左 Horizontal TAF / 右 Vertical TAF)。

    每个面板内对比软/硬表层两条曲线；横轴 Surface Receiver Location (m) 0–1800，
    纵轴左列 0–6、右列 0–5；坡顶/坡脚 #1/#2 竖虚线；每个面板左上角各放一份图例。
    """
    angles = sorted({c['angle'] for c in cases})  # 入射角（行；0°在上、15°在下）
    ratios = sorted({round(c['vs1_vs2'], 2) for c in cases})  # 出现的 Vs1/Vs2（每面板内的对比曲线）
    columns = [('taf_h', 'Horizontal TAF', (0.0, 6.0), 1.0),  # 左列：水平向，纵轴 0–6，主刻度间隔 1
               ('taf_v', 'Vertical TAF', (0.0, 5.0), 1.0)]  # 右列：竖向，纵轴 0–5，主刻度间隔 1
    x_ticks = [0, 600, 1200, 1800]  # 横轴主刻度
    x_max = total_L if (total_L and total_L > 1) else 1800.0  # 横轴上限（默认 1800）
    nrow = max(1, len(angles)); ncol = 2  # 行=角度数、列=2（H/V）
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.7 * nrow), dpi=300, squeeze=False)  # 画布
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体
    for ri, ang in enumerate(angles):  # 遍历行（入射角）
        for ci, (key, ylabel, ylim, ystep) in enumerate(columns):  # 遍历列（H/V 分量）
            ax = axes[ri][ci]  # 当前面板
            style_axes(ax)  # 设置外观（含细密网格）
            used = 0  # 兜底样式计数
            for ratio in ratios:  # 同面板内按软/硬画曲线
                match = [c for c in cases if abs(c['angle'] - ang) < 1e-6 and abs(round(c['vs1_vs2'], 2) - ratio) < 1e-6]  # 匹配工况
                if not match:  # 该(角度,比值)无数据
                    continue
                c = match[0]  # 取该工况
                s, label = style_for_ratio(ratio, used); used += 1  # 取样式与图例
                ax.plot(c['x'], c[key], color=s['color'], linestyle=s['linestyle'],  # 绘制曲线
                        linewidth=s['linewidth'], label=label)
            ax.set_xlim(0, x_max)  # 横轴范围 0–1800
            ax.set_xticks([t for t in x_ticks if t <= x_max + 1e-6])  # 主刻度 0/600/1200/1800
            ax.set_ylim(*ylim)  # 纵轴范围（左 0–6、右 0–5）
            ax.yaxis.set_major_locator(mticker.MultipleLocator(ystep))  # 纵轴主刻度间隔
            if x_crest is not None:  # 标注坡顶/坡脚 #1/#2
                ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)  # 坡顶竖线
                ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=1.0)  # 坡脚竖线
                ty = ylim[0] + 0.95 * (ylim[1] - ylim[0])  # #1/#2 文本纵位置
                ax.text(x_crest - 0.015 * x_max, ty, '#1', fontsize=10, fontproperties=EN_FONT, va='top', ha='right')  # #1
                ax.text(x_toe - 0.015 * x_max, ty, '#2', fontsize=10, fontproperties=EN_FONT, va='top', ha='right')  # #2
            ax.set_title(r'$\theta_s = %g^\circ$' % ang, fontsize=12, fontproperties=EN_FONT, pad=6)  # 每面板入射角标题
            if ri == nrow - 1:  # 底行加横轴标签
                ax.set_xlabel('Surface Receiver Location (m)', fontsize=12, fontproperties=EN_FONT)  # 横轴标签
            ax.set_ylabel(ylabel, fontsize=12, fontproperties=EN_FONT)  # 纵轴标签（每面板按列）
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black',  # 每面板左上角图例
                      framealpha=1.0, prop=EN_FONT, fontsize=9.5)
    title = u'图15(%s)  h1/(H-h) = %.2f  (i=45deg, Vr/Vs2=2.5, h/H=0.50)' % (part_tag, h1_over)  # 总标题（纯 ASCII 避免缺字）
    fig.suptitle(title, fontsize=14, fontproperties=CN_FONT, y=0.995)  # 设置总标题
    fig.tight_layout(rect=(0, 0, 1, 0.97))  # 调整布局给标题留位
    out_name = 'Fig15_compare_h%0.2f.png' % h1_over  # 输出文件名（按厚度）
    out_path = os.path.join(out_dir, out_name)  # 输出路径
    fig.savefig(out_path, dpi=300, bbox_inches='tight')  # 保存
    plt.close(fig)  # 释放
    print('  已输出: %s' % out_name)  # 提示


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数（目录）
    data_dir = resolve_results_dir(arg)  # 定位集中结果 results/ 目录
    if data_dir is None:  # 未找到 index.csv
        print('错误：未找到 index.csv（集中结果）。'
              '\n请先运行 Collect_TAF_results_v1.py 生成 results/index.csv，'
              '\n再把"工况根目录"（其下含 results/）或"results 目录"作为参数传入；不传则取当前目录。')  # 提示
        return  # 退出
    print('>>> 集中模式：读取 %s 内的 index.csv 与 TAF-*.csv' % data_dir)  # 提示
    cases, x_crest, x_toe, total_L = collect_cases_from_index(data_dir)  # 从清单收集
    if not cases:  # 无工况
        print('错误：index.csv 中没有可用的 TAF 工况记录。')  # 提示
        return  # 退出
    out_dir = data_dir  # 图片输出到 results/ 目录
    thicknesses = sorted({round(c['h1_over'], 2) for c in cases})  # 出现的厚度（每个一张图）
    print('>>> 共 %d 个工况，厚度档 = %s，将分别出图。' % (len(cases), thicknesses))  # 汇总提示
    part_letters = ['a', 'b', 'c', 'd']  # 图15(a)/(b)... 子标签
    for idx, h1_over in enumerate(thicknesses):  # 逐厚度出图
        sub = [c for c in cases if abs(round(c['h1_over'], 2) - h1_over) < 1e-6]  # 该厚度工况
        part_tag = part_letters[idx] if idx < len(part_letters) else str(idx + 1)  # 子标签
        print('--- 厚度 h1/(H-h)=%.2f：%d 个工况 ---' % (h1_over, len(sub)))  # 提示
        plot_one_thickness(h1_over, sub, x_crest, x_toe, total_L, out_dir, part_tag)  # 出图
    print('>>> 图15 风格对比图全部完成，输出目录：%s' % out_dir)  # 完成提示


if __name__ == '__main__':  # 主程序入口
    main()  # 运行
