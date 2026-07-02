# -*- coding: utf-8 -*-
"""跨工况地表响应独立分图（Hybrid 专用 v1）。

读取 Collect_All_results_v2.py 收集的 results/ 内 SGRID_RESPONSE CSV，
将 Postprocess_All_surface_v2.py 的 3×2 合并图拆为 6 张独立图表分别输出。
每张图按三段归一化坐标 s 绘制（坡顶平台 A / 坡面 B / 坡脚平台 C），
样式与原 Postprocess_All_surface_v2 的子图完全一致。

数据来源：
  results/index.csv + results/SGRID_RESPONSE-*.csv（由 Collect_All_results_v2 收集）

运行：
  python Postprocess/Hybrid/Plot_Hybrid_surface_v1.py <工况根目录或 results 目录>
  不传参数则取当前目录。
"""

import os  # 导入系统接口模块
import sys  # 导入系统参数模块
import math  # 导入数学模块
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.gridspec as gridspec  # 导入网格布局（三段分轴用）
import matplotlib.ticker as mticker  # 导入刻度定位器
import matplotlib.font_manager as fm  # 导入字体管理器


# ==============================================================================
#  出版级绘图工具（内联，与 Postprocess_All_surface_v2 同口径）
# ==============================================================================
CB_PALETTE = {  # Okabe-Ito 色盲安全配色
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'purple': '#CC79A7',
}

CJK_SERIF_PRIORITY = [  # 中文衬线字体优先级
    'Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC', 'Source Han Serif CN',
    'SimSun', 'NSimSun', 'STSong', 'Songti SC',
]


def _detect_cjk_serif():  # 检测系统可用的中文衬线字体名
    """按优先级返回首个可用的中文衬线字体名；仍无则返回 None。"""
    available = {f.name for f in fm.fontManager.ttflist}  # 当前可用字体名集合
    for name in CJK_SERIF_PRIORITY:  # 按优先级查找
        if name in available:  # 命中
            return name  # 返回
    for name in available:  # 兜底模糊匹配
        if any(k in name.lower() for k in ('song', 'serif cjk', 'serif sc', 'songti')):  # 关键词匹配
            return name  # 返回
    return None  # 未找到


def setup_cn_journal_style():  # 应用中文核心期刊出版级样式
    """配置 rcParams：宋体正文 + Times New Roman 混排，返回中文字体名或 None（改用英文标签）。"""
    cjk = _detect_cjk_serif()  # 检测中文字体
    if cjk:  # 找到中文字体
        serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # 混排回退链
        plt.rcParams.update({'font.family': serif_list, 'font.serif': serif_list,
                             'mathtext.fontset': 'stix'})  # 字体链与公式字体
    plt.rcParams.update({  # 出版级通用设置
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.2,
        'axes.linewidth': 0.7, 'xtick.direction': 'in', 'ytick.direction': 'in',
        'figure.dpi': 150, 'savefig.dpi': 300,
    })
    return cjk  # 返回中文字体名


def style_axes(ax):  # 美化单轴外观（与 Postprocess_All_surface_v2 的 style_axes_local 一致）
    """白底、四面朝内刻度、黑色边框、主次网格；仅 Y 轴开次刻度（防段 B 单主刻度崩溃）。"""
    ax.set_facecolor('white')  # 白底
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)  # 朝内
    for spine in ax.spines.values():  # 四周边框
        spine.set_color('black'); spine.set_linewidth(1.0)  # 黑色 1pt
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())  # 仅 Y 轴开次刻度
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)  # 主网格
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)  # 次网格


def _ensure_parent(path):  # 确保父目录存在
    parent = os.path.dirname(os.path.abspath(path))  # 父目录
    if parent and not os.path.exists(parent):  # 不存在
        os.makedirs(parent)  # 创建


def export_figure(fig, basename, formats=('pdf', 'svg', 'png'), dpi=300):  # 多格式导出
    """把 fig 导出为多种格式，可选灰度预览。返回写出路径列表。"""
    saved = []  # 路径列表
    for fmt in formats:  # 逐格式保存
        path = '%s.%s' % (basename, fmt)  # 完整路径
        _ensure_parent(path)  # 确保目录
        kw = {'bbox_inches': 'tight', 'pad_inches': 0.05}  # 裁白边
        if fmt in ('png', 'tiff', 'tif'):  # 栅格格式
            kw['dpi'] = dpi  # 指定分辨率
        fig.savefig(path, **kw)  # 保存
        saved.append(path)  # 记录
    try:  # 尝试生成灰度预览
        from PIL import Image  # 图像处理
        png_path = basename + '.png'  # 源 PNG
        if os.path.isfile(png_path):  # PNG 存在
            gray_path = basename + '_grayscale.png'  # 灰度版路径
            Image.open(png_path).convert('L').save(gray_path)  # 转灰度保存
            saved.append(gray_path)  # 记录
    except ImportError:  # 无 Pillow
        pass  # 跳过
    return saved  # 返回所有写出路径


# ==============================================================================
#  配置：6 个面板定义（与 Postprocess_All_surface_v2 完全一致）
# ==============================================================================
DRAW_SPECS = [  # (字段键, 中文标签, 英文标签, 曲线颜色)
    ('PGA_h', '水平向 PGA (m/s²)', 'Horizontal PGA (m/s²)', CB_PALETTE['blue']),        # (a)
    ('PGA_v', '垂直向 PGA (m/s²)', 'Vertical PGA (m/s²)', CB_PALETTE['vermillion']),     # (b)
    ('AF_h', '水平向 AF', 'Horizontal AF', CB_PALETTE['blue']),                           # (c)
    ('AF_v', '垂直向 AF', 'Vertical AF', CB_PALETTE['vermillion']),                       # (d)
    ('TAF_h', '水平向 TAF', 'Horizontal TAF', CB_PALETTE['blue']),                        # (e)
    ('TAF_v', '垂直向 TAF', 'Vertical TAF', CB_PALETTE['vermillion']),                    # (f)
]
PANEL_LETTERS = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']  # 学术子图编号


# ==============================================================================
#  数据来源
# ==============================================================================
def resolve_results_dir(arg):  # 定位 results/ 目录
    """返回含 index.csv 的 results 目录；找不到返回 None。"""
    base = os.path.abspath(arg) if arg else os.getcwd()  # 基准目录
    if os.path.isfile(os.path.join(base, 'index.csv')):  # 本身即 results/
        return base  # 直接使用
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')):  # 其下含 results/
        return os.path.join(base, 'results')  # 使用子目录
    return None  # 未找到


def collect_records(results_dir):  # 从 index.csv 收集 SGRID_RESPONSE 记录
    """读取 index.csv 并收集全部 SGRID_RESPONSE 条目。

    返回列表，每项为 dict：source_folder, record, fpath（CSV 绝对路径）。
    """
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    rows = idx[idx['type'].astype(str).str.upper() == 'SGRID_RESPONSE']  # 仅保留 SGRID_RESPONSE
    records = []  # 记录列表
    for _, r in rows.iterrows():  # 遍历
        fname = str(r['collected_file'])  # 收集后文件名
        fpath = os.path.join(results_dir, fname)  # 绝对路径
        if not os.path.isfile(fpath):  # 文件缺失
            print('  跳过(文件不存在): %s' % fname); continue  # 跳过
        records.append({  # 追加记录
            'source_folder': str(r.get('source_folder', '')),  # 来源工况目录
            'record': str(r.get('record', '')),  # 输入波记录名
            'fpath': fpath,  # CSV 文件路径
        })
        print('  收录: %s' % fname)  # 提示
    return records  # 返回列表


# ==============================================================================
#  三段分轴绘制（一比一复刻 Postprocess_All_surface_v2 的子图样式）
# ==============================================================================
def draw_single_panel(fig, field, color, ylabel, df, s_all, a_max, c_max, w_b,
                      seg_titles, xlabel, panel_lbl):  # 绘制一个三段分轴面板
    """在 fig 上绘制一个字段的三段分轴图（坡顶平台 / 坡面 / 坡脚平台），
    样式与 Postprocess_All_surface_v2 的各子图完全一致。
    """
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[a_max, w_b, c_max], wspace=0.0,
                           left=0.14, right=0.98, top=0.86, bottom=0.20)  # 三段网格布局
    ax_a = fig.add_subplot(gs[0])  # 段A 坡顶平台
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)  # 段B 坡面（共 y 轴）
    ax_c = fig.add_subplot(gs[2], sharey=ax_a)  # 段C 坡脚平台（共 y 轴）
    for ax in (ax_a, ax_b, ax_c):  # 统一外观
        style_axes(ax)  # 网格和边框美化

    # 按段分组数据（拐点 s=0/1 两段共享，保证曲线触缝）
    seg_s = {'A': [], 'B': [], 'C': []}  # 各段 s 坐标
    seg_y = {'A': [], 'B': [], 'C': []}  # 各段 y 值
    y_all = []  # 全部有效值（统一 ylim 用）
    vals = df[field].to_numpy(float)  # 该字段数组
    for i in range(len(s_all)):  # 遍历每个点
        v = vals[i]  # 值
        if np.isnan(v):  # NaN 跳过
            continue
        sk = float(s_all[i])  # s 坐标
        if sk < -a_max - 1e-9 or sk > 1.0 + c_max + 1e-9:  # 超出显示范围
            continue
        y_all.append(v)  # 记录有效值
        if sk <= 1e-9:  # 段A（含坡顶棱 s=0）
            seg_s['A'].append(sk); seg_y['A'].append(v)
        if -1e-9 <= sk <= 1.0 + 1e-9:  # 段B（两端拐点都收，保证曲线触缝）
            seg_s['B'].append(sk); seg_y['B'].append(v)
        if sk >= 1.0 - 1e-9:  # 段C（含坡脚棱 s=1）
            seg_s['C'].append(sk); seg_y['C'].append(v)

    # 逐段画曲线
    for seg, ax in (('A', ax_a), ('B', ax_b), ('C', ax_c)):  # 遍历三段
        if seg_s[seg]:  # 该段有数据
            ax.plot(seg_s[seg], seg_y[seg], color=color, linestyle='-', linewidth=1.2)  # 绘图

    # 设置各段横轴范围
    ax_a.set_xlim(-a_max, 0.0)  # 段A
    ax_b.set_xlim(0.0, 1.0)  # 段B
    ax_c.set_xlim(1.0, 1.0 + c_max)  # 段C

    # 统一纵轴范围（sharey 自动同步）
    lo, hi = (min(y_all), max(y_all)) if y_all else (0.0, 1.0)  # 数据极值
    pad = 0.06 * ((hi - lo) if hi > lo else max(abs(hi), 1.0))  # 上下留白
    ax_a.set_ylim(lo - pad, hi + pad)  # 设定范围

    # 设置各段横轴刻度
    step_a = max(1, int(math.ceil(a_max / 4.0)))  # 段A 约 4 个主刻度
    ax_a.set_xticks([-float(t) for t in range(0, int(math.floor(a_max)) + 1, step_a)])  # 段A 整数刻度
    ax_b.set_xticks([0.5])  # 段B 只标中点
    step_c = max(1, int(math.ceil(c_max / 4.0)))  # 段C 约 4 个主刻度
    ax_c.set_xticks([1.0 + float(t) for t in range(0, int(math.floor(c_max)) + 1, step_c)])  # 段C 整数刻度

    # 段间边框处理（虚线 = 坡顶棱/坡脚棱标记线）
    ax_a.spines['right'].set_visible(False)  # 段A 右边框隐藏
    ax_c.spines['left'].set_visible(False)  # 段C 左边框隐藏
    for sd in ('left', 'right'):  # 段B 两侧边框改虚线
        ax_b.spines[sd].set_linestyle('--')  # 虚线
        ax_b.spines[sd].set_linewidth(0.9)  # 线宽
    plt.setp(ax_b.get_yticklabels(), visible=False)  # 段B 隐藏 y 刻度文本
    plt.setp(ax_c.get_yticklabels(), visible=False)  # 段C 隐藏 y 刻度文本

    # 三段标题
    ax_a.set_title(seg_titles[0], fontsize=7, pad=2)  # 段A 标题
    ax_b.set_title(seg_titles[1], fontsize=7, pad=2)  # 段B 标题
    ax_c.set_title(seg_titles[2], fontsize=7, pad=2)  # 段C 标题

    # 纵轴标签
    ax_a.set_ylabel(ylabel)  # 设置纵轴标签

    # 底部居中横轴标签与子图编号
    p_a = ax_a.get_position()  # 左子轴位置
    p_c = ax_c.get_position()  # 右子轴位置
    cx = (p_a.x0 + p_c.x1) / 2.0  # 居中 x
    fig.text(cx, p_a.y0 - 0.030, xlabel, ha='center', va='top', fontsize=8)  # 横轴标签
    fig.text(cx, p_a.y0 - 0.055, panel_lbl, ha='center', va='top',  # 子图编号
             fontsize=8, fontname='Times New Roman', fontweight='bold')


def plot_record(rec, results_dir, out_dir, use_cn):  # 为一条记录输出 6 张独立图
    """读取一条 SGRID_RESPONSE CSV，输出 6 张三段分轴独立图到 out_dir/<source_folder>__<record>/。"""
    df = pd.read_csv(rec['fpath'])  # 读取 CSV
    s_all = df['s'].to_numpy(float)  # 归一坐标 s 数组

    # 计算三段显示参数
    a_max = max(float(-s_all.min()), 0.5)  # 段A 显示跨度
    c_max = max(float(s_all.max()) - 1.0, 0.5)  # 段C 显示跨度
    w_b = max(1.0, 0.3 * (a_max + c_max))  # 段B 保底宽度

    # 标签（中文/英文）
    if use_cn:  # 中文标签
        seg_titles = ('坡顶平台', '坡面', '坡脚平台')  # 三段标题
        xlabel = '归一化坐标 $s$'  # 横轴标签
    else:  # 英文兜底
        seg_titles = ('Crest plateau', 'Slope', 'Toe plateau')  # 三段标题
        xlabel = 'Normalized coordinate $s$'  # 横轴标签

    # 输出子目录
    tag = '%s__%s' % (rec['source_folder'], rec['record']) if rec['record'] else rec['source_folder']  # 记录标识
    rec_dir = os.path.join(out_dir, tag)  # 记录子目录
    if not os.path.isdir(rec_dir):  # 不存在
        os.makedirs(rec_dir)  # 创建

    for pi, (field, cn_lbl, en_lbl, color) in enumerate(DRAW_SPECS):  # 遍历 6 个面板
        if field not in df.columns:  # 字段缺失
            print('    跳过(字段 %s 不在 CSV 中)' % field); continue  # 跳过
        ylabel = cn_lbl if use_cn else en_lbl  # 选择标签语言
        panel_lbl = PANEL_LETTERS[pi]  # 子图编号

        fig = plt.figure(figsize=(3.15, 2.8))  # 单栏宽画布
        draw_single_panel(fig, field, color, ylabel, df, s_all, a_max, c_max, w_b,
                          seg_titles, xlabel, panel_lbl)  # 绘制三段分轴面板

        paths = export_figure(fig, os.path.join(rec_dir, field))  # 多格式导出
        plt.close(fig)  # 释放画布
        print('    %s %s: %s' % (panel_lbl, field, '; '.join(os.path.basename(p) for p in paths)))  # 打印


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动地表响应独立分图 (Plot_Hybrid_surface_v1)...')  # 启动提示
    cjk = setup_cn_journal_style()  # 应用出版级样式
    use_cn = bool(cjk)  # 中文字体是否可用
    if cjk:  # 有中文字体
        print('>>> 出版级样式已应用：中文字体=%s' % cjk)  # 提示
    else:  # 无中文字体
        print('>>> 提示：未检测到中文字体，图内文字改用英文标签。')  # 提示

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数
    data_dir = resolve_results_dir(arg)  # 定位 results/ 目录
    if data_dir is None:  # 未找到
        print('错误：未找到 index.csv。请先运行 Collect_All_results_v2.py 生成 results/。')  # 报错
        return  # 退出

    print('>>> 读取 %s 内的 index.csv 与 SGRID_RESPONSE-*.csv' % data_dir)  # 提示
    records = collect_records(data_dir)  # 收集记录
    if not records:  # 无记录
        print('错误：index.csv 中没有 SGRID_RESPONSE 类型的记录。')  # 报错
        return  # 退出
    print('>>> 共 %d 条记录，每条输出 6 张独立三段分轴图。' % len(records))  # 概览

    out_dir = os.path.join(data_dir, 'Fig_surface_panels')  # 输出根目录
    for rec in records:  # 遍历每条记录
        print('--- %s / %s ---' % (rec['source_folder'], rec['record']))  # 打印记录信息
        try:  # 尝试绘图
            plot_record(rec, data_dir, out_dir, use_cn)  # 输出 6 张图
        except Exception as e:  # 绘图失败
            print('    错误: %s' % e)  # 打印错误

    print('>>> 独立分图全部完成，输出目录：%s' % out_dir)  # 完成提示


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
