# -*- coding: utf-8 -*-
"""跨工况地表响应分图对比（Hybrid 专用 v1）。

以 Plot_Multi_TAF_v3.py 为模板，读取 Collect_All_results_v2.py 收集的 results/ 数据，
将 Postprocess_All_surface_v2.py 中的 6 个响应面板（PGA_h/PGA_v/AF_h/AF_v/TAF_h/TAF_v）
分别绘制为独立图表，每张图叠加各工况曲线用于跨工况对比。

横轴使用三段归一化坐标 s（研究计划 §4.0 约定），坡顶棱 s=0 与坡脚棱 s=1 处标注 #1/#2。

数据来源（仅集中 results/ 模式）：
  先用 Collect_All_results_v2.py 把各工况 sgrid_response CSV 汇到 results/，本脚本读取：
    - results/index.csv：统一规范列（含 slope_i、incident_angle、a0_base 等）；
    - results/SGRID_RESPONSE-*.csv：各工况统一 s 子网格对齐的地表响应数据。
  坡角 i、入射角 θs 直接取自 index.csv 规范列；无量纲频率 a0 由 record 主频 fc(Hz)
  与该工况 a0_base 按 a0 = fc × a0_base 计算（逐工况自洽）。

运行：
  先 `python Postprocess/Hybrid/Collect_All_results_v2.py <工况根目录>` 生成 results/index.csv，再
  `python Postprocess/Hybrid/Plot_Hybrid_surface_v1.py <工况根目录>`（自动找其下 results/）
  或直接传 results/ 目录；不传参数取当前目录。
"""

import os  # 导入系统接口模块
import re  # 导入正则模块（从记录名解析输入波主频 f_c）
import sys  # 导入系统参数模块
import functools  # 导入偏函数工具（为各子图绑定绘制参数）
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.ticker as mticker  # 导入刻度定位器（主/次刻度、网格）
import matplotlib.font_manager as fm  # 导入字体管理器（中文字体检测）


# ==============================================================================
#  出版级绘图工具（中文核心期刊样式 + 多格式矢量导出）
#  说明：为方便单文件阅读与管理，这些通用绘图函数【内联在本脚本内】，不拆成独立模块。
#  Pillow(灰度预览)/SciencePlots(样式增强) 可选，缺失时优雅降级。
# ==============================================================================
CB_PALETTE = {  # Okabe-Ito 色盲安全配色（命名取色）
    'black': '#000000',   # 黑
    'orange': '#E69F00',  # 橙
    'skyblue': '#56B4E9', # 天蓝
    'green': '#009E73',   # 绿
    'yellow': '#F0E442',  # 黄
    'blue': '#0072B2',    # 蓝
    'vermillion': '#D55E00',  # 朱红
    'purple': '#CC79A7',  # 紫
}

CJK_SERIF_PRIORITY = [  # 中文衬线字体优先级（出版接受度 + 可用性）
    'Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC', 'Source Han Serif CN',  # 思源/Noto 宋体
    'SimSun', 'NSimSun', 'STSong', 'Songti SC',  # 系统宋体类
]
CJK_INSTALL_HINT = (  # 找不到中文字体时的安装提示
    '未找到任何中文衬线(宋体类)字体。请安装其一：\n'
    '  Windows: 系统自带 SimSun(宋体) 一般已存在；如缺，安装思源宋体 Source Han Serif；\n'
    '  或下载 Noto Serif CJK: https://github.com/notofonts/noto-cjk/releases\n'
    '安装后删除 matplotlib 字体缓存（~/.cache/matplotlib 或 %USERPROFILE%\\.matplotlib）再重试。'
)


def _detect_cjk_serif():  # 检测系统可用的中文衬线字体名
    """按优先级返回首个可用的中文衬线字体名；找不到则关键词兜底扫描；仍无则返回 None。"""
    available = {f.name for f in fm.fontManager.ttflist}  # 当前可用字体名集合
    for name in CJK_SERIF_PRIORITY:  # 按优先级查找
        if name in available:  # 命中
            return name  # 返回该字体名
    for name in available:  # 兜底：扫描含中文宋体关键词的字体
        low = name.lower()  # 小写副本
        if any(k in low for k in ('song', 'serif cjk', 'serif sc', 'serif cn', 'songti')):  # 关键词匹配
            return name  # 返回命中字体
    return None  # 未找到任何中文衬线字体


def setup_cn_journal_style(width='double', use_sciplots=True):  # 应用中文核心期刊出版级样式
    """配置 rcParams：宋体正文 + Times New Roman 数字/公式混排、stix 公式、嵌字、出版级字号/线宽。

    返回 dict：{'cjk_serif': 选用的中文字体名, 'width_in': 建议图宽英寸}。
    """
    if use_sciplots:  # 可选叠加 SciencePlots 基础样式
        try:  # 尝试导入并应用
            import scienceplots  # noqa: F401  # 触发样式注册
            plt.style.use(['science', 'no-latex'])  # 关 LaTeX（中文必须关）
        except Exception:  # 未装或失败
            pass  # 静默回退
    cjk = _detect_cjk_serif()  # 检测中文衬线字体
    if cjk is None:  # 找不到中文字体
        raise RuntimeError(CJK_INSTALL_HINT)  # 明确报错
    serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # Times 在前、宋体在后
    plt.rcParams.update({  # 批量设置出版级 rcParams
        'font.family': serif_list,  # 真实字体名链（逐字形回退的关键）
        'font.serif': serif_list,  # 衬线候选
        'mathtext.fontset': 'stix',  # 数学公式用 STIX
        'axes.unicode_minus': False,  # 修负号方框
        'pdf.fonttype': 42,  # PDF 嵌入 TrueType
        'ps.fonttype': 42,  # PS 同上
        'svg.fonttype': 'none',  # SVG 文本保持可编辑
        'font.size': 8,  # 基准字号
        'axes.labelsize': 8,  # 轴标签
        'axes.titlesize': 8.5,  # 面板标题
        'xtick.labelsize': 7,  # x 刻度数字
        'ytick.labelsize': 7,  # y 刻度数字
        'legend.fontsize': 7,  # 图例
        'lines.linewidth': 0.8,  # 线宽
        'axes.linewidth': 0.7,  # 轴线宽
        'xtick.direction': 'in',  # x 刻度朝内
        'ytick.direction': 'in',  # y 刻度朝内
        'figure.dpi': 150,  # 屏幕预览 dpi
        'savefig.dpi': 300,  # 栅格保存 dpi
    })
    width_in = 6.3 if width == 'double' else 3.15  # 中文核心期刊建议宽
    return {'cjk_serif': cjk, 'width_in': width_in}  # 返回选用字体与建议宽度


_VECTOR = {'pdf', 'svg', 'eps'}  # 矢量格式集合
_RASTER = {'png', 'tiff', 'tif'}  # 允许的栅格格式


def _ensure_parent(path):  # 确保目标文件的父目录存在
    """目标路径父目录不存在则创建。"""
    parent = os.path.dirname(os.path.abspath(path))  # 父目录
    if parent and not os.path.exists(parent):  # 不存在
        os.makedirs(parent)  # 创建


def _grayscale_from(fig, basename, dpi):  # 由图生成灰度预览版（色盲自检）
    """优先用 Pillow 把 PNG 转灰度；无 Pillow 则跳过并提示。返回灰度图路径或 None。"""
    try:  # 尝试导入 Pillow
        from PIL import Image  # 图像处理
    except ImportError:  # 无 Pillow
        print('[plot] 未装 Pillow，跳过灰度预览', file=sys.stderr)  # 提示
        return None  # 返回空
    png_path = basename + '.png'  # 临时/正式 PNG
    _ensure_parent(png_path)  # 确保目录
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')  # 先存 PNG
    gray_path = basename + '_grayscale.png'  # 灰度版路径
    Image.open(png_path).convert('L').save(gray_path)  # 转灰度保存
    return gray_path  # 返回灰度图路径


def export_figure(fig, basename, formats=('pdf', 'svg', 'png'), dpi=300,  # 多格式导出
                  size_inches=None, grayscale_preview=True, pad_inches=0.05):
    """把 fig 按最终物理尺寸导出为多种格式（矢量优先），可选灰度预览。返回写出路径列表。"""
    formats = [f.lower().lstrip('.') for f in formats]  # 规范化扩展名
    bad = [f for f in formats if f in ('jpg', 'jpeg') or f not in (_VECTOR | _RASTER)]  # 不支持/有损
    if bad:  # 有非法格式
        raise ValueError('不支持或不宜用于数据图的格式: %s' % bad)  # 报错
    if size_inches is not None:  # 指定最终尺寸
        fig.set_size_inches(*size_inches)  # 强制尺寸
    plt.rcParams['pdf.fonttype'] = 42  # 再次确保嵌字
    plt.rcParams['ps.fonttype'] = 42  # 同上
    plt.rcParams['svg.fonttype'] = 'none'  # SVG 文本可编辑
    saved = []  # 写出路径列表
    for fmt in formats:  # 逐格式保存
        path = '%s.%s' % (basename, fmt)  # 完整路径
        _ensure_parent(path)  # 确保目录
        kw = {'bbox_inches': 'tight', 'pad_inches': pad_inches}  # 去多余白边
        if fmt in _RASTER:  # 栅格格式指定 dpi
            kw['dpi'] = dpi  # 分辨率
        fig.savefig(path, **kw)  # 保存
        saved.append(path)  # 记录
    if grayscale_preview:  # 可选灰度预览
        g = _grayscale_from(fig, basename, dpi)  # 生成灰度版
        if g:  # 成功
            saved.append(g)  # 记录
    return saved  # 返回所有写出路径


# ==============================================================================
#  坐标轴样式
# ==============================================================================
def style_axes(ax):  # 统一坐标轴外观
    """设置白底、四面朝内刻度、黑色边框、细密灰色网格（主+次）。"""
    ax.set_facecolor('white')  # 设置面板背景为白色
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)  # 主次刻度均朝内
    for spine in ax.spines.values():  # 遍历四条边框线
        spine.set_color('black'); spine.set_linewidth(1.0)  # 黑色实线边框
    ax.minorticks_on()  # 打开次刻度
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)  # 主网格
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)  # 次网格


# ==============================================================================
#  配置与常量
# ==============================================================================
A0_DECIMALS = 1  # a0 数值四舍五入保留的小数位

# 各 a0 对应的曲线样式（与 Plot_Multi_TAF_v3 保持一致）
A0_STYLES = {  # a0 → (颜色, 线型, 线宽)
    1.0: {'color': '#55A868', 'linestyle': '--', 'linewidth': 1.2},  # a0=1.0：绿色虚线
    1.5: {'color': '#56B4E9', 'linestyle': '--', 'linewidth': 1.2},  # a0=1.5：蓝色虚线
    2.0: {'color': '#E41A1C', 'linestyle': '-', 'linewidth': 1.3},   # a0=2.0：红色实线
}
FALLBACK_STYLES = [  # a0 不在上表时按序兜底
    {'color': CB_PALETTE['orange'], 'linestyle': '-.', 'linewidth': 1.2},     # 兜底1：橙色点划线
    {'color': CB_PALETTE['purple'], 'linestyle': ':', 'linewidth': 1.2},      # 兜底2：紫色点线
    {'color': CB_PALETTE['black'], 'linestyle': '--', 'linewidth': 1.2},      # 兜底3：黑色虚线
    {'color': CB_PALETTE['green'], 'linestyle': '-', 'linewidth': 1.2},       # 兜底4：绿色实线
    {'color': CB_PALETTE['blue'], 'linestyle': '-.', 'linewidth': 1.2},       # 兜底5：蓝色点划线
    {'color': CB_PALETTE['vermillion'], 'linestyle': ':', 'linewidth': 1.2},  # 兜底6：朱红点线
]

# 用于无 a0 时按工况序号自动分配的样式
FOLDER_COLORS = [  # 工况颜色循环列表
    '#0072B2', '#D55E00', '#009E73', '#CC79A7',
    '#E69F00', '#56B4E9', '#E41A1C', '#55A868',
]
FOLDER_LSTYLES = ['-', '--', '-.', ':']  # 工况线型循环列表

# 6 个独立出图的面板定义：(字段键, 中文标签, 英文标签)
DRAW_SPECS = [
    ('PGA_h', '水平向 PGA (m/s²)', 'Horizontal PGA (m/s²)'),   # (a) 水平 PGA
    ('PGA_v', '垂直向 PGA (m/s²)', 'Vertical PGA (m/s²)'),     # (b) 垂直 PGA
    ('AF_h', '水平向 AF', 'Horizontal AF'),                     # (c) 水平 AF
    ('AF_v', '垂直向 AF', 'Vertical AF'),                       # (d) 垂直 AF
    ('TAF_h', '水平向 TAF', 'Horizontal TAF'),                  # (e) 水平 TAF
    ('TAF_v', '垂直向 TAF', 'Vertical TAF'),                    # (f) 垂直 TAF
]
PANEL_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f']  # 子标签


# ==============================================================================
#  数据来源判别（仅集中 results/ 模式）
# ==============================================================================
def resolve_results_dir(arg):  # 判别集中结果 results/ 目录
    """返回含 index.csv 的 results 目录；传入目录本身即 results/ 或其下含 results/ 均可；找不到返回 None。"""
    base = os.path.abspath(arg) if arg else os.getcwd()  # 基准目录
    if os.path.isfile(os.path.join(base, 'index.csv')):  # 基准目录本身即 results/
        return base  # 直接使用
    if os.path.isfile(os.path.join(base, 'results', 'index.csv')):  # 基准目录下存在 results/
        return os.path.join(base, 'results')  # 使用子目录
    return None  # 未找到


# ==============================================================================
#  a0 计算与样式（与 Plot_Multi_TAF_v3 同口径）
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
    """用该工况规范列 a0_base 与 record 主频按 a0=fc·a0_base 计算；失败返回 None。"""
    a0_base = _num(row, 'a0_base')  # 该工况的 a0 换算基数
    record = row.get('record')  # 输入波记录名
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(record), re.IGNORECASE) if record is not None else None  # 解析主频
    if a0_base is None or m is None:  # 缺基数或主频
        return None  # 返回空
    a0 = float(m.group(1)) * a0_base  # a0 = f_c(Hz) × a0_base
    return round(a0, A0_DECIMALS)  # 四舍五入返回


def style_for_a0(a0, used):  # 为某 a0 取曲线样式
    """返回 (style_dict, label)；优先用预设样式，否则按序兜底。"""
    key = round(a0, A0_DECIMALS)  # 规范化 a0 键
    label = r'$a_0=%.1f$' % a0  # 图例文本
    if key in A0_STYLES:  # 命中预设
        return dict(A0_STYLES[key]), label  # 返回预设样式副本
    return dict(FALLBACK_STYLES[used % len(FALLBACK_STYLES)]), label  # 返回兜底样式


def style_for_folder(idx):  # 为某工况序号取曲线样式
    """无 a0 时按工况序号自动分配颜色与线型。返回 (color, linestyle)。"""
    ci = idx % len(FOLDER_COLORS)  # 颜色循环索引
    li = (idx // len(FOLDER_COLORS)) % len(FOLDER_LSTYLES)  # 线型循环索引
    return FOLDER_COLORS[ci], FOLDER_LSTYLES[li]  # 返回颜色与线型


# ==============================================================================
#  从 results/index.csv 收集工况记录
# ==============================================================================
def collect_cases_from_index(results_dir):  # 读取集中结果并整理为工况记录
    """读 index.csv + SGRID_RESPONSE-*.csv，返回工况记录列表。

    每条记录含：s(归一坐标数组), fields(各响应字段数组字典), slope_i, theta, a0, source_folder, record。
    """
    index_path = os.path.join(results_dir, 'index.csv')  # 清单文件路径
    idx = pd.read_csv(index_path)  # 读取清单
    sgrid_rows = idx[idx['type'].astype(str).str.upper() == 'SGRID_RESPONSE']  # 仅保留 SGRID_RESPONSE 类型行
    cases = []  # 工况记录列表
    for _, r in sgrid_rows.iterrows():  # 遍历每个 SGRID_RESPONSE 文件记录
        fname = str(r['collected_file'])  # 收集后的文件名
        fpath = os.path.join(results_dir, fname)  # 完整路径
        if not os.path.isfile(fpath):  # 文件缺失
            print('  跳过(文件不存在): %s' % fname); continue  # 提示并跳过
        slope_i = _num(r, 'slope_i')  # 坡角 i
        theta = _num(r, 'incident_angle')  # 入射角 θs
        a0 = compute_a0(r)  # 无量纲频率 a0
        try:  # 读取 sgrid_response 数据
            df = pd.read_csv(fpath)  # 读取 CSV
            s = df['s'].to_numpy(float)  # 归一化坐标 s
            fields = {}  # 各字段数组
            for fkey in ('PGA_h', 'PGA_v', 'AF_h', 'AF_v', 'TAF_h', 'TAF_v'):  # 遍历 6 个响应字段
                if fkey in df.columns:  # 列存在
                    fields[fkey] = df[fkey].to_numpy(float)  # 转为数组
        except Exception as e:  # 读取失败
            print('  跳过(读取失败 %s): %s' % (e, fname)); continue  # 提示并跳过
        cases.append({  # 追加工况记录
            'source_folder': str(r.get('source_folder', fname)),  # 来源工况目录名
            'record': str(r.get('record', '')),  # 输入波记录名
            'slope_i': float(slope_i) if slope_i is not None else None,  # 坡角（可为 None）
            'theta': float(theta) if theta is not None else None,  # 入射角（可为 None）
            'a0': float(a0) if a0 is not None else None,  # 无量纲频率（可为 None）
            's': s,  # 归一坐标数组
            'fields': fields,  # 响应字段字典
        })
        print('  收录 %s -> i=%s, θs=%s, a0=%s' % (fname, slope_i, theta, a0))  # 打印收录信息
    return cases  # 返回收集结果


# ==============================================================================
#  绘图辅助
# ==============================================================================
def auto_ylim(y_arrays):  # 按数据极值计算纵轴范围
    """按全部曲线数据极值留 1/5 顶部空白（给图例），底部取 0 或数据最小值取小。"""
    if not y_arrays:  # 无数据
        return (0.0, 1.0)  # 兜底范围
    allv = np.concatenate([a[~np.isnan(a)] for a in y_arrays if len(a[~np.isnan(a)]) > 0])  # 拼接有效值
    if len(allv) == 0:  # 全为 NaN
        return (0.0, 1.0)  # 兜底
    ymin = min(0.0, float(allv.min()))  # 下限不超过 0
    ymax = float(allv.max())  # 数据最大值
    span = ymax - ymin if ymax > ymin else 1.0  # 数据跨度
    top = ymin + span * (5.0 / 4.0)  # 顶部预留 1/5 空白给图例
    step = 10.0 ** np.floor(np.log10(span / 4.0)) if span > 0 else 1.0  # 估算刻度量级
    return (np.floor(ymin / step) * step, np.ceil(top / step) * step)  # 按量级取整后返回


def set_s_axis(ax, s_min, s_max):  # 设置 s 轴范围与刻度
    """设置 s 轴范围；自动约 6 个刻度。"""
    ax.set_xlim(s_min, s_max)  # 设置横轴范围
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))  # 自动约 6 个主刻度


# ==============================================================================
#  绘图核心
# ==============================================================================
def draw_surface_panel(ax, field, ylabel, ylim, cases, title=None, show_xlabel=True):  # 绘制单个响应面板
    """在 ax 上画某字段面板：按 a0 或工况名多曲线 + 坡顶/坡脚标注 + 轴样式。

    若 a0 可用，按 a0 区分曲线样式（与 Plot_Multi_TAF_v3 一致）；
    否则按工况序号自动分配颜色与线型。
    """
    style_axes(ax)  # 套用统一外观
    a0_list = sorted({c['a0'] for c in cases if c['a0'] is not None})  # 出现的 a0 值列表
    used_fallback = 0  # 兜底样式计数器

    if a0_list:  # a0 可用：按 a0 分组画曲线
        for a0 in a0_list:  # 遍历每个 a0 值
            match = [c for c in cases if c['a0'] is not None and abs(c['a0'] - a0) < 1e-6]  # 匹配该 a0 的工况
            if not match:  # 无数据
                continue  # 跳过
            c = match[0]  # 取首个匹配工况
            y = c['fields'].get(field)  # 获取该字段数组
            if y is None:  # 字段缺失
                continue  # 跳过
            s, label = style_for_a0(a0, used_fallback)  # 获取样式与图例标签
            if round(a0, A0_DECIMALS) not in A0_STYLES:  # 非预设 a0
                used_fallback += 1  # 递增兜底计数
            ax.plot(c['s'], y, color=s['color'], linestyle=s['linestyle'],  # 绘制曲线
                    linewidth=s['linewidth'], label=label)
    else:  # a0 不可用：按工况序号分配样式
        for ci, c in enumerate(cases):  # 遍历各工况
            y = c['fields'].get(field)  # 获取该字段数组
            if y is None:  # 字段缺失
                continue  # 跳过
            color, ls = style_for_folder(ci)  # 获取自动分配样式
            label = c['source_folder']  # 用工况目录名作图例
            ax.plot(c['s'], y, color=color, linestyle=ls, linewidth=1.2, label=label)  # 绘制曲线

    # 设置 s 轴范围
    if cases:  # 有数据时从数据取范围
        all_s = np.concatenate([c['s'] for c in cases])  # 拼接全部 s 坐标
        set_s_axis(ax, float(all_s.min()), float(all_s.max()))  # 设置横轴
    else:  # 无数据兜底
        set_s_axis(ax, -3.0, 4.0)  # 默认范围

    ax.set_ylim(*ylim)  # 设置纵轴范围

    # 坡顶棱 #1 (s=0) 与坡脚棱 #2 (s=1) 标注
    ax.axvline(x=0.0, color='black', linestyle='--', linewidth=0.9)  # 坡顶竖虚线
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.9)  # 坡脚竖虚线
    ty = ylim[0] + 0.95 * (ylim[1] - ylim[0])  # 文本纵位置
    s_span = ax.get_xlim()[1] - ax.get_xlim()[0]  # s 轴跨度（用于偏移计算）
    ax.text(0.0 - 0.015 * s_span, ty, '#1', fontsize=7, va='top', ha='right')  # 标注 #1
    ax.text(1.0 - 0.015 * s_span, ty, '#2', fontsize=7, va='top', ha='right')  # 标注 #2

    if title is not None:  # 设置面板标题
        ax.set_title(title, pad=4)  # 带顶部间距
    if show_xlabel:  # 是否显示横轴标签
        ax.set_xlabel('归一化坐标 $s$')  # 横轴标签
    ax.set_ylabel(ylabel)  # 纵轴标签
    if a0_list or cases:  # 有曲线时放图例
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)  # 左上角图例


# ==============================================================================
#  分图输出（6 个字段各出一张独立图 + 灰度预览）
# ==============================================================================
def plot_surface_fields(i_ang, theta, cases, out_dir, part_tag):  # 为一组 (坡角, 入射角) 绘制 6 张独立图
    """对某一 (坡角 i, 入射角 θs) 分组绘制 6 张独立图（PGA_h/v, AF_h/v, TAF_h/v），
    每张图叠加各工况曲线按 a0 区分样式。

    输出到 out_dir/字段名_i{i}_ths{θs}/ 下。
    """
    # 构造面板标题（含坡角与入射角信息）
    title_parts = []  # 标题片段列表
    if i_ang is not None:  # 有坡角
        title_parts.append(r'$i=%g^\circ$' % i_ang)  # 坡角信息
    if theta is not None:  # 有入射角
        title_parts.append(r'$\theta_s=%g^\circ$' % theta)  # 入射角信息
    base_title = r',\ '.join(title_parts) if title_parts else ''  # 拼接基础标题

    # 构造输出文件名后缀
    suffix_parts = []  # 后缀片段列表
    if i_ang is not None:  # 有坡角
        suffix_parts.append('i%g' % i_ang)  # 坡角后缀
    if theta is not None:  # 有入射角
        suffix_parts.append('ths%g' % theta)  # 入射角后缀
    suffix = '_'.join(suffix_parts) if suffix_parts else 'all'  # 拼接后缀

    for pi, (field, cn_label, en_label) in enumerate(DRAW_SPECS):  # 遍历 6 个响应字段
        letter = PANEL_LETTERS[pi]  # 子标签
        ylabel = cn_label  # 纵轴标签（中文）
        title = base_title  # 面板标题

        # 收集该字段的全部有效曲线数据以计算统一 ylim
        y_arrays = [c['fields'][field] for c in cases if field in c['fields']]  # 有效曲线列表
        ylim = auto_ylim(y_arrays)  # 计算纵轴范围

        # 创建独立画布并绘制
        fig_size = (3.15, 2.4)  # 单栏宽度、紧凑高度
        fig, ax = plt.subplots(1, 1, figsize=fig_size)  # 创建单轴画布

        draw_fn = functools.partial(draw_surface_panel, field=field, ylabel=ylabel,  # 绑定绘制函数参数
                                     ylim=ylim, cases=cases, title=title, show_xlabel=True)
        draw_fn(ax)  # 在主轴上绘制

        # 底部居中标注子标签 (a)/(b)/.../(f)
        pos = ax.get_position()  # 轴位置
        fig.text((pos.x0 + pos.x1) / 2.0, pos.y0 - 0.04,  # 居中偏下
                 r'(%s)' % letter, fontsize=9, va='top', ha='center', fontweight='bold')  # 加粗子标签

        fig.tight_layout(rect=(0, 0.03, 1, 1), pad=0.4)  # 紧凑布局（底部留白给子标签）

        # 导出文件
        fig_name = '%s_%s' % (field, suffix)  # 图名 = 字段_后缀
        fig_dir = os.path.join(out_dir, fig_name)  # 图文件夹
        if not os.path.isdir(fig_dir):  # 不存在则建
            os.makedirs(fig_dir)  # 建目录

        paths = export_figure(fig, os.path.join(fig_dir, fig_name),  # 多格式导出
                              formats=('pdf', 'svg', 'png'), dpi=300,
                              size_inches=fig_size, grayscale_preview=True)
        plt.close(fig)  # 释放画布
        print('  (%s) %s: %s' % (letter, field, '; '.join(os.path.basename(p) for p in paths)))  # 打印输出信息


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动跨工况地表响应分图对比 (Plot_Hybrid_surface_v1)...')  # 启动提示
    info = setup_cn_journal_style()  # 应用中文核心期刊出版级样式
    print('>>> 出版级样式已应用：中文字体=%s（数字/公式用 Times New Roman）' % info['cjk_serif'])  # 提示选用字体

    arg = sys.argv[1] if len(sys.argv) >= 2 else None  # 命令行参数（目录）
    data_dir = resolve_results_dir(arg)  # 定位集中结果 results/ 目录
    if data_dir is None:  # 未找到 index.csv
        print('错误：未找到 index.csv（集中结果）。'
              '\n请先运行 Collect_All_results_v2.py 生成 results/index.csv，'
              '\n再把"工况根目录"（其下含 results/）或"results 目录"作为参数传入；不传则取当前目录。')  # 错误提示
        return  # 退出

    print('>>> 集中模式：读取 %s 内的 index.csv 与 SGRID_RESPONSE-*.csv' % data_dir)  # 提示数据目录
    cases = collect_cases_from_index(data_dir)  # 从清单收集工况
    if not cases:  # 无可用工况
        print('错误：index.csv 中没有 SGRID_RESPONSE 类型的工况记录。')  # 错误提示
        return  # 退出

    # 统计概览
    slope_angles = sorted({c['slope_i'] for c in cases if c['slope_i'] is not None})  # 出现的坡角
    theta_angles = sorted({c['theta'] for c in cases if c['theta'] is not None})  # 出现的入射角
    a0_values = sorted({c['a0'] for c in cases if c['a0'] is not None})  # 出现的 a0 值
    print('>>> 共 %d 个工况：坡角 %s、入射角 %s、a0 %s；按 (i, θs) 分组分别出 6 张图。' %
          (len(cases), slope_angles or '未知', theta_angles or '未知', a0_values or '未知'))  # 概览信息

    # 按 (slope_i, theta) 分组
    groups = {}  # 分组字典
    for c in cases:  # 遍历全部工况
        key = (c.get('slope_i'), c.get('theta'))  # 分组键
        groups.setdefault(key, []).append(c)  # 追加到对应组

    out_dir = os.path.join(data_dir, 'Fig_surface_compare')  # 统一输出根目录
    part_letters = ['a', 'b', 'c', 'd']  # 分组子标签
    for gidx, ((i_ang, theta), group_cases) in enumerate(  # 遍历各分组
            sorted(groups.items(), key=lambda x: (x[0][0] or 0, x[0][1] or 0))):
        part_tag = part_letters[gidx] if gidx < len(part_letters) else str(gidx + 1)  # 分组标签
        print('--- 坡角 i=%s°, θs=%s°：%d 个工况 ---' % (i_ang, theta, len(group_cases)))  # 打印分组信息
        plot_surface_fields(i_ang, theta, group_cases, out_dir, part_tag)  # 绘制该组 6 张独立图

    print('>>> 跨工况地表响应分图对比全部完成，输出目录：%s' % out_dir)  # 完成提示


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
