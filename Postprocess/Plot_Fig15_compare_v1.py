# -*- coding: utf-8 -*-
"""复现 Shen 等 2025 论文【图15】风格的跨工况对比图。

论文图15：i=45°, Vr/Vs2=2.5, h/H=0.50 的三层斜坡，地表水平/竖向 PGA 放大系数沿地表坐标的变化。
  (a) h1/(H−h)=0.25, (b) h1/(H−h)=0.75；每个厚度一张图；
  子图按"入射角 θs(0°/15°)"分列、按"水平/竖向"分行；
  每个面板内对比"表层软硬"两条曲线（图例=Vs1/Vs2，软=0.5、硬=2.0）。

数据来源：批处理（如 Batch/Autorun_TAF_multilayer_v1.py）在每个工况文件夹里生成的
  TAF-<记录>.csv（列：x, TAF_h, TAF_v）。本脚本【跨文件夹】汇总这些工况画成图15 布局。
  每个工况的属性（Vs1/Vs2、h1/(H−h)、入射角）优先从该文件夹内复制的建模脚本
  VAB_oblique_TAF_multilayer_v*.py 的 material_cfg/geometry_cfg 精确解析；
  解析不到时回退按文件夹名（soft/hard、t25/t75、a0/a15）识别。
  坡顶/坡脚(#1/#2)位置从 PGA-*-slope.csv 读取。

运行：在包含各工况子文件夹的目录下运行（默认取当前工作目录；也可传入根目录）：
  cd Batch && python ../Postprocess/Plot_Fig15_compare_v1.py
  或：python Postprocess/Plot_Fig15_compare_v1.py  <存放工况文件夹的根目录>
"""

import os  # 导入系统接口模块
import re  # 导入正则表达式模块
import sys  # 导入系统参数模块
import glob  # 导入文件匹配模块
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.font_manager as fm  # 导入字体管理器

# ==============================================================================
#  配置与常量
# ==============================================================================
RECORD_PREFER = '4Hz'  # 优先选用的输入记录标识（图15 用 4 Hz Ricker；找不到则取该文件夹首个 TAF）
MODEL_SCRIPT_GLOB = 'VAB_oblique_TAF_multilayer_v*.py'  # 工况文件夹内建模脚本（用于精确解析参数）
CUSTOM_YLIM = {'horizontal': None, 'vertical': None}  # 纵轴范围：None 自适应，或 (min,max)
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
    """设置白底、四面朝内刻度、黑色边框与点状网格。"""
    ax.set_facecolor('white')  # 白色背景
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, labelsize=10)  # 刻度朝内
    for spine in ax.spines.values():  # 遍历四条边框
        spine.set_color('black'); spine.set_linewidth(1.0)  # 黑色边框
    ax.grid(True, which='both', linestyle=':', color='#b0b0b0', linewidth=0.5)  # 点状网格


# ==============================================================================
#  工况属性解析
# ==============================================================================
def parse_case_from_script(folder):  # 从文件夹内建模脚本精确解析工况属性
    """返回 dict(vs1_vs2, h1_over, angle) 或 None（无脚本或解析失败）。"""
    scripts = glob.glob(os.path.join(folder, MODEL_SCRIPT_GLOB))  # 查找建模脚本副本
    if not scripts:  # 未找到脚本
        return None  # 回退到文件夹名解析
    try:
        text = open(scripts[0], 'r', encoding='utf-8', errors='ignore').read()  # 读取脚本文本
    except Exception:
        return None  # 读取失败则回退
    vrs = re.findall(r"'velocity_ratio'\s*:\s*([0-9.]+)", text)  # 所有 velocity_ratio（首=表层，次=覆盖层）
    ths = re.findall(r"'thickness'\s*:\s*([0-9.]+)", text)  # 所有 thickness（首=表层固定厚度）
    ang = re.search(r"'angle'\s*:\s*([0-9.]+)", text)  # 入射角
    hmh = re.search(r"'H_minus_h'\s*:\s*([0-9.]+)", text)  # 斜坡高差 H−h
    if len(vrs) < 2 or not ths or not ang or not hmh:  # 不是"含表层的多层"配置
        return None  # 回退
    surf_vr, over_vr = float(vrs[0]), float(vrs[1])  # 表层、覆盖层相对基岩波速比
    vs1_vs2 = over_vr / surf_vr if surf_vr > 0 else float('nan')  # Vs1/Vs2 = (Vr/surf)/(Vr/over) = over/surf
    h1_over = float(ths[0]) / float(hmh.group(1)) if float(hmh.group(1)) > 0 else float('nan')  # h1/(H−h)
    return {'vs1_vs2': vs1_vs2, 'h1_over': h1_over, 'angle': float(ang.group(1))}  # 返回属性


def parse_case_from_name(folder):  # 从文件夹名回退解析工况属性
    """按 soft/hard、t25/t75、a0/a15 命名约定识别，返回 dict 或 None。"""
    name = os.path.basename(folder).lower()  # 文件夹名（小写）
    m_soft = 'soft' in name; m_hard = 'hard' in name  # 软/硬标志
    m_t = re.search(r't(\d+)', name)  # 厚度标签 t25/t75
    m_a = re.search(r'a(-?\d+)', name)  # 角度标签 a0/a15
    if not (m_soft or m_hard) or not m_t or not m_a:  # 关键标签缺失
        return None  # 无法识别
    vs1_vs2 = 0.5 if m_soft else 2.0  # 软=0.5、硬=2.0（论文设定）
    h1_over = int(m_t.group(1)) / 100.0  # t25→0.25, t75→0.75
    return {'vs1_vs2': vs1_vs2, 'h1_over': h1_over, 'angle': float(m_a.group(1))}  # 返回属性


def find_taf_csv(folder):  # 在工况文件夹内选取 TAF csv
    """优先返回含 RECORD_PREFER 的 TAF-*.csv，否则首个；无则 None。"""
    tafs = sorted(glob.glob(os.path.join(folder, 'TAF-*.csv')))  # 该文件夹所有 TAF 表
    if not tafs:  # 无 TAF 数据
        return None
    preferred = [f for f in tafs if RECORD_PREFER.lower() in os.path.basename(f).lower()]  # 优先记录
    return preferred[0] if preferred else tafs[0]  # 返回首选或首个


def read_crest_toe(folder):  # 从 PGA-*-slope.csv 读取坡顶/坡脚 x
    """返回 (x_crest, x_toe, total_L) 或 (None, None, None)。"""
    pgas = glob.glob(os.path.join(folder, 'PGA-*-slope.csv'))  # 坡地 PGA 表
    if not pgas:
        return None, None, None  # 无几何文件
    try:
        df = pd.read_csv(pgas[0])  # 读取坡地 PGA
        ymax, ymin = df['y'].max(), df['y'].min()  # 顶/底高程
        x_crest = df[df['y'] >= ymax - 1e-3]['x'].max()  # 坡顶：顶平台最右
        x_toe = df[df['y'] <= ymin + 1e-3]['x'].min()  # 坡脚：底平台最左
        return float(x_crest), float(x_toe), float(df['x'].max())  # 返回几何
    except Exception:
        return None, None, None  # 解析失败


# ==============================================================================
#  数据收集
# ==============================================================================
def collect_cases(root):  # 扫描根目录收集所有工况
    """遍历 root 下子文件夹，返回工况记录列表与全局坡顶/坡脚/总长。"""
    cases = []  # 工况记录
    x_crest = x_toe = total_L = None  # 全局几何（各工况几何相同）
    for entry in sorted(os.listdir(root)):  # 遍历根目录条目
        folder = os.path.join(root, entry)  # 子文件夹完整路径
        if not os.path.isdir(folder):  # 跳过非目录
            continue
        taf_csv = find_taf_csv(folder)  # 该文件夹 TAF 数据
        if not taf_csv:  # 无 TAF 则跳过
            continue
        attr = parse_case_from_script(folder) or parse_case_from_name(folder)  # 解析属性（脚本优先）
        if attr is None:  # 无法识别属性
            print('  跳过(无法识别工况属性): %s' % entry)  # 提示跳过
            continue
        try:
            df = pd.read_csv(taf_csv)  # 读取 TAF 数据
            x = df['x'].to_numpy(float); th = df['TAF_h'].to_numpy(float); tv = df['TAF_v'].to_numpy(float)  # 取列
        except Exception as e:  # 读取失败
            print('  跳过(读取 TAF 失败 %s): %s' % (e, entry)); continue  # 提示并跳过
        cc, ct, tl = read_crest_toe(folder)  # 该工况几何
        if cc is not None:  # 记录全局几何（取首个成功的）
            x_crest, x_toe, total_L = (x_crest or cc), (x_toe or ct), (total_L or tl)
        cases.append({'folder': entry, 'x': x, 'taf_h': th, 'taf_v': tv,  # 追加工况记录
                      'vs1_vs2': attr['vs1_vs2'], 'h1_over': attr['h1_over'], 'angle': attr['angle']})
        print('  收录 %s -> Vs1/Vs2=%.2f, h1/(H-h)=%.2f, θs=%g°' %  # 打印收录信息
              (entry, attr['vs1_vs2'], attr['h1_over'], attr['angle']))
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


def plot_one_thickness(h1_over, cases, x_crest, x_toe, total_L, out_dir, part_tag):  # 画一个厚度的图15 子图
    """对某一 h1/(H−h) 厚度，绘制 行=水平/竖向 × 列=入射角 的对比图。"""
    angles = sorted({c['angle'] for c in cases})  # 该厚度下出现的入射角（列）
    ratios = sorted({round(c['vs1_vs2'], 2) for c in cases})  # 出现的 Vs1/Vs2（曲线）
    directions = [('taf_h', '水平向 PGA 放大系数', 'Horizontal'),  # 行：水平
                  ('taf_v', '竖向 PGA 放大系数', 'Vertical')]  # 行：竖向
    ncol = max(1, len(angles)); nrow = 2  # 列=角度数、行=2（水平/竖向）
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), dpi=300, squeeze=False)  # 画布
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.rcParams['mathtext.fontset'] = 'stix'  # 数学字体
    for ri, (key, cn_label, en_label) in enumerate(directions):  # 遍历行（方向）
        for ci, ang in enumerate(angles):  # 遍历列（入射角）
            ax = axes[ri][ci]  # 当前面板
            style_axes(ax)  # 设置外观
            used = 0  # 兜底样式计数
            for ratio in ratios:  # 同面板内按软/硬画曲线
                match = [c for c in cases if abs(c['angle'] - ang) < 1e-6 and abs(round(c['vs1_vs2'], 2) - ratio) < 1e-6]  # 匹配工况
                if not match:  # 该(角度,比值)无数据
                    continue
                c = match[0]  # 取该工况
                s, label = style_for_ratio(ratio, used); used += 1  # 取样式与图例
                ax.plot(c['x'], c[key], color=s['color'], linestyle=s['linestyle'],  # 绘制曲线
                        linewidth=s['linewidth'], label=label)
            if total_L:  # 设定横轴范围与刻度
                ax.set_xlim(0, total_L)  # 0 到总长
                step = 600.0 if abs(total_L - 1800.0) < 50.0 else total_L / 3.0  # 刻度步长
                ax.set_xticks(np.arange(0, total_L + 1.0, step))  # 设置刻度
            lim = CUSTOM_YLIM.get('horizontal' if key == 'taf_h' else 'vertical')  # 自定义纵轴
            if isinstance(lim, tuple):  # 指定范围
                ax.set_ylim(*lim)  # 应用
            if x_crest is not None:  # 标注坡顶/坡脚 #1/#2
                ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)  # 坡顶竖线
                ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=1.0)  # 坡脚竖线
                y0, y1 = ax.get_ylim(); ty = y0 + 0.92 * (y1 - y0)  # 文本纵位置
                ax.text(x_crest - 0.02 * (total_L or 1.0), ty, '#1', fontsize=10, fontproperties=EN_FONT, va='top', ha='right')  # #1
                ax.text(x_toe - 0.02 * (total_L or 1.0), ty, '#2', fontsize=10, fontproperties=EN_FONT, va='top', ha='right')  # #2
            if ri == 0:  # 顶行加列标题（入射角）
                ax.set_title(r'$\theta_s = %g^\circ$' % ang, fontsize=13, fontproperties=CN_FONT, pad=8)  # 入射角标题
            if ri == nrow - 1:  # 底行加横轴标签
                ax.set_xlabel('地表测点位置 (m)', fontsize=12, fontproperties=CN_FONT)  # 横轴标签
            if ci == 0:  # 首列加纵轴标签（方向）
                ax.set_ylabel(cn_label, fontsize=12, fontproperties=CN_FONT)  # 纵轴标签
            if ri == 0 and ci == ncol - 1:  # 仅右上角放一次图例（软/硬）
                ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black',
                          framealpha=1.0, prop=EN_FONT, fontsize=10)  # 图例
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
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()  # 根目录：命令行参数或当前目录
    root = os.path.abspath(root)  # 转绝对路径
    print('>>> 扫描工况根目录: %s' % root)  # 提示根目录
    cases, x_crest, x_toe, total_L = collect_cases(root)  # 收集工况
    if not cases:  # 无工况
        print('错误：未在该目录下找到任何含 TAF-*.csv 的工况文件夹。'
              '\n请在存放各工况文件夹（如 Batch/）的目录下运行，或把该目录作为参数传入。')  # 提示
        return  # 退出
    out_dir = root  # 图片输出到根目录
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
