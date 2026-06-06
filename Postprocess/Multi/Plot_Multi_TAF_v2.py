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
import functools  # 导入偏函数工具（为各子图绑定绘制参数）
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.ticker as mticker  # 导入刻度定位器（主/次刻度、网格）
import matplotlib.font_manager as fm  # 导入字体管理器（中文字体检测）


# ==============================================================================
#  出版级绘图工具（中文核心期刊样式 + 多格式矢量导出 + 拆图/合成图）
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
    混排靠 matplotlib(>=3.6) 的【逐字形回退】：font.family 用【真实字体名列表】，Times 在前命中数字/西文，
    汉字 Times 无字形 → 回退到中文宋体类。
    """
    if use_sciplots:  # 可选叠加 SciencePlots 基础样式
        try:  # 尝试导入并应用
            import scienceplots  # noqa: F401  # 触发样式注册
            plt.style.use(['science', 'no-latex'])  # 关 LaTeX（中文必须关，避免缺 LaTeX 崩溃）
        except Exception:  # 未装或失败
            pass  # 静默回退到内置预设
    cjk = _detect_cjk_serif()  # 检测中文衬线字体
    if cjk is None:  # 找不到中文字体
        raise RuntimeError(CJK_INSTALL_HINT)  # 明确报错（避免画完才发现是方框）
    serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # Times New Roman 在前、宋体在后
    plt.rcParams.update({  # 批量设置出版级 rcParams
        'font.family': serif_list,  # 真实字体名链（逐字形回退的关键）
        'font.serif': serif_list,  # 衬线候选（与上一致，作冗余）
        'mathtext.fontset': 'stix',  # 数学公式用 STIX（Times 风格）
        'axes.unicode_minus': False,  # 修负号方框
        'pdf.fonttype': 42,  # PDF 嵌入 TrueType（避免 Type-3 被期刊拒）
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
    width_in = 6.3 if width == 'double' else 3.15  # 中文核心期刊建议宽：双栏≈16cm / 单栏≈8cm
    return {'cjk_serif': cjk, 'width_in': width_in}  # 返回选用字体与建议宽度


_VECTOR = {'pdf', 'svg', 'eps'}  # 矢量格式集合
_RASTER = {'png', 'tiff', 'tif'}  # 允许的栅格格式（不含 JPEG）


def _ensure_parent(path):  # 确保目标文件的父目录存在
    """目标路径父目录不存在则创建。"""
    parent = os.path.dirname(os.path.abspath(path))  # 父目录
    if parent and not os.path.exists(parent):  # 不存在
        os.makedirs(parent)  # 创建（含中间层）


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
        fig.set_size_inches(*size_inches)  # 强制尺寸（导出后勿在 Word/LaTeX 再缩放）
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


def save_composite_with_panels(out_root, fig_name, composite_fig, panel_specs,  # 合成图+子图统一输出
                               composite_size=None, panel_size=(3.15, 2.4),
                               formats=('pdf', 'svg', 'png'), panel_formats=('pdf', 'png'),
                               dpi=300, grayscale_preview=True):
    """把"合成图 + 各子图"输出到 out_root/<fig_name>/ ——本项目多子图图的统一规范。

    合成图存 <fig_name>/<fig_name>.{格式}(+灰度)；各子图按语义名存 <fig_name>/panels/<名>.{格式}。
    panel_specs : [(语义名, draw_fn), ...]；draw_fn(ax) 在给定单轴上重画该子图（与合成图共用同一函数）。
    返回 dict：{'composite': [...], 'panels': {名:[...]}, 'dir': 图文件夹}。
    """
    fig_dir = os.path.join(out_root, fig_name)  # 同名图文件夹
    panels_dir = os.path.join(fig_dir, 'panels')  # 子图子目录
    if not os.path.isdir(panels_dir):  # 不存在则建
        os.makedirs(panels_dir)  # 建目录（含父级）
    comp_paths = export_figure(composite_fig, os.path.join(fig_dir, fig_name),  # 导出合成图
                               formats=formats, dpi=dpi, size_inches=composite_size,
                               grayscale_preview=grayscale_preview)
    plt.close(composite_fig)  # 释放合成图画布
    panel_paths = {}  # 子图路径表
    for name, draw_fn in panel_specs:  # 遍历子图规格
        pf = plt.figure(figsize=panel_size)  # 单子图画布（按最终尺寸）
        ax = pf.add_subplot(1, 1, 1)  # 单坐标轴
        draw_fn(ax)  # 在该轴上重画此子图（复用同一 draw_fn 保证与合成图一致）
        pf.tight_layout(pad=0.4)  # 紧凑布局
        panel_paths[name] = export_figure(pf, os.path.join(panels_dir, name),  # 导出子图
                                          formats=panel_formats, dpi=dpi,
                                          size_inches=panel_size, grayscale_preview=False)
        plt.close(pf)  # 释放子图画布
    return {'composite': comp_paths, 'panels': panel_paths, 'dir': fig_dir}  # 返回结果汇总


# ==============================================================================
#  配置与常量
# ==============================================================================
A0_DECIMALS = 1  # a0 数值四舍五入保留的小数位（用于分组与样式匹配）

# 各 a0 对应的曲线样式（论文图8：1.0 虚线、1.5 点划线、2.0 实线，配不同颜色便于区分）
A0_STYLES = {  # 色盲安全配色(Okabe-Ito) + 区分线型，灰度下仍可分
    1.0: {'color': CB_PALETTE['blue'], 'linestyle': '--', 'linewidth': 1.2},    # a0=1.0：蓝色虚线
    1.5: {'color': CB_PALETTE['orange'], 'linestyle': '-.', 'linewidth': 1.2},  # a0=1.5：橙色点划线
    2.0: {'color': CB_PALETTE['black'], 'linestyle': '-', 'linewidth': 1.3},    # a0=2.0：黑色实线
}
FALLBACK_STYLES = [  # a0 不在上表时按序兜底取用的样式（同样色盲安全）
    {'color': CB_PALETTE['green'], 'linestyle': ':', 'linewidth': 1.2},       # 兜底1：绿色点线
    {'color': CB_PALETTE['vermillion'], 'linestyle': '-', 'linewidth': 1.2},  # 兜底2：朱红实线
    {'color': CB_PALETTE['purple'], 'linestyle': '--', 'linewidth': 1.2},     # 兜底3：紫色虚线
]

COLUMNS = [  # 子图列定义：(数据键, 纵轴标签, 纵轴范围或None自适应)
    ('taf_h', '水平向 TAF', None),  # 左列：水平向 TAF
    ('taf_v', '竖向 TAF', None),    # 右列：竖向 TAF
]
GROUP_LETTERS = ['a', 'b', 'c', 'd', 'e', 'f']  # 宏观大组的子标签 (a)/(b)/...

# ==============================================================================
#  坐标轴样式（字体由 setup_cn_journal_style 经 rcParams 统一管理）
# ==============================================================================
def style_axes(ax):  # 统一坐标轴外观
    """设置白底、四面朝内刻度、黑色边框、细密灰色网格（主+次）。"""
    ax.set_facecolor('white')  # 设置面板背景为白色
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)  # 主次刻度均朝内、四面显示（字号随 rcParams）
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


def draw_taf_panel(ax, i_ang, theta, key, ylabel, ylim, ctx, title=None, show_xlabel=True):  # 绘制单个 TAF 面板
    """在 ax 上画某 (坡角 i, 入射角 θs, 分量 key) 面板：按 a0 多曲线 + 坡顶/坡脚标注 + 轴样式。

    合成图与单独子图共用本函数，保证两者完全一致。ctx 携带 cases/x_crest/x_toe/total_L。
    title 为 None 时用 θs 标题；show_xlabel 控制是否画横轴标签（合成图仅最底行画）。
    """
    cases = ctx['cases']  # 全部工况记录
    style_axes(ax)  # 套用统一外观（含细密网格）
    used = 0  # 兜底样式计数器
    a0_list = sorted({c['a0'] for c in cases  # 该 (坡角,入射角) 下出现的 a0（升序）
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
    x_max = set_x_axis(ax, ctx['total_L'])  # 设置横轴范围与刻度
    ax.set_ylim(*ylim)  # 设置该列统一纵轴范围
    if ctx['x_crest'] is not None:  # 标注坡顶/坡脚 #1/#2
        ax.axvline(x=ctx['x_crest'], color='black', linestyle='--', linewidth=0.9)  # 坡顶竖虚线
        ax.axvline(x=ctx['x_toe'], color='black', linestyle='--', linewidth=0.9)  # 坡脚竖虚线
        ty = ylim[0] + 0.95 * (ylim[1] - ylim[0])  # #1/#2 文本纵位置
        ax.text(ctx['x_crest'] - 0.015 * x_max, ty, '#1', fontsize=7, va='top', ha='right')  # 标注 #1
        ax.text(ctx['x_toe'] - 0.015 * x_max, ty, '#2', fontsize=7, va='top', ha='right')  # 标注 #2
    ax.set_title(title if title is not None else (r'$\theta_s = %g^\circ$' % theta), pad=4)  # 面板标题
    if show_xlabel:  # 需要时画横轴标签
        ax.set_xlabel('地表观测点位置 (m)')  # 横轴标签（中文）
    ax.set_ylabel(ylabel)  # 纵轴标签（按列）
    if a0_list:  # 有曲线才放图例
        ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', framealpha=1.0)  # 面板左上角图例


def plot_fig8(cases, x_crest, x_toe, total_L, out_dir):  # 绘制论文图8 风格大图 + 拆分子图
    """宏观大组按坡角 i 竖向堆叠 (a)/(b)…；每组 行=入射角 θs（升序），列=H/V；面板内按 a0 画曲线。

    合成图与各子图统一输出到 out_dir/Fig8_TAF_compare/（子图在其下 panels/），见 save_composite_with_panels。
    """
    slope_angles = sorted({c['slope_i'] for c in cases})  # 出现的坡角（升序，决定宏观大组顺序）
    thetas = sorted({c['theta'] for c in cases})  # 出现的入射角（升序，决定组内行顺序）
    n_groups = len(slope_angles)  # 宏观大组数量
    n_theta = max(1, len(thetas))  # 每组行数（入射角数）
    n_col = len(COLUMNS)  # 列数（H/V）
    total_rows = n_groups * n_theta  # 总行数 = 大组数 × 每组行数
    ctx = {'cases': cases, 'x_crest': x_crest, 'x_toe': x_toe, 'total_L': total_L}  # 面板绘制上下文
    # 预先按列计算自适应纵轴范围（同一列所有面板共用，便于跨组对比）
    col_ylim = [auto_ylim([c[key] for c in cases], fixed) for key, _yl, fixed in COLUMNS]  # 每列纵轴范围
    # ---- 合成图：总网格 ----
    comp_size = (3.15 * n_col, max(2.6, 2.4 * total_rows))  # 中文核心期刊双栏宽 ≈16cm；高随行数
    fig, axes = plt.subplots(total_rows, n_col, figsize=comp_size, squeeze=False)  # 创建总网格画布
    for gi, i_ang in enumerate(slope_angles):  # 遍历宏观大组（坡角）
        for ti, theta in enumerate(thetas):  # 遍历组内行（入射角）
            grid_row = gi * n_theta + ti  # 当前面板所在的总行号
            for ci, (key, ylabel, _fixed) in enumerate(COLUMNS):  # 遍历列（H/V 分量）
                draw_taf_panel(axes[grid_row][ci], i_ang, theta, key, ylabel, col_ylim[ci], ctx,  # 画该面板
                               title=r'$\theta_s = %g^\circ$' % theta, show_xlabel=(grid_row == total_rows - 1))
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.8)  # 调整布局并为大组标签留出顶部空隙
    for gi, i_ang in enumerate(slope_angles):  # 在每个宏观大组顶部左侧标注 (a) i=30° …
        pos = axes[gi * n_theta][0].get_position()  # 该组首行左面板位置
        letter = GROUP_LETTERS[gi] if gi < len(GROUP_LETTERS) else str(gi + 1)  # 大组子标签
        fig.text(pos.x0, min(0.998, pos.y1 + 0.012), r'(%s)  $i=%g^\circ$' % (letter, i_ang),  # 放在首行面板左上方
                 fontsize=9, va='bottom', ha='left', fontweight='bold')  # 加粗大组标签
    # ---- 各子图：复用 draw_taf_panel，按语义命名 ----
    panel_specs = []  # [(语义名, 绘制函数), ...]
    for gi, i_ang in enumerate(slope_angles):  # 遍历坡角
        for theta in thetas:  # 遍历入射角
            for ci, (key, ylabel, _fixed) in enumerate(COLUMNS):  # 遍历分量
                comp = 'H' if key == 'taf_h' else 'V'  # 分量缩写
                name = 'i%g_ths%g_TAF-%s' % (i_ang, theta, comp)  # 语义子图名（如 i45_ths15_TAF-H）
                draw_fn = functools.partial(draw_taf_panel, i_ang=i_ang, theta=theta, key=key,  # 绑定该子图参数
                                            ylabel=ylabel, ylim=col_ylim[ci], ctx=ctx,
                                            title=r'$i=%g^\circ,\ \theta_s=%g^\circ$' % (i_ang, theta),
                                            show_xlabel=True)  # 单独子图自带横轴标签
                panel_specs.append((name, draw_fn))  # 收集子图规格
    res = save_composite_with_panels(out_dir, 'Fig8_TAF_compare', fig, panel_specs,  # 合成图+子图统一输出
                                            composite_size=comp_size, panel_size=(3.15, 2.5))
    print('  合成图: %s' % '; '.join(os.path.basename(p) for p in res['composite']))  # 提示合成图
    print('  子图 %d 张 -> %s' % (len(res['panels']), os.path.join(res['dir'], 'panels')))  # 提示子图目录


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    print('>>> 启动图8 风格 TAF 对比绘图 (Plot_Multi_TAF_v4)...')  # 启动提示
    info = setup_cn_journal_style()  # 应用中文核心期刊出版级样式（宋体+Times 混排、嵌字）
    print('>>> 出版级样式已应用：中文字体=%s（数字/公式用 Times New Roman）' % info['cjk_serif'])  # 提示选用字体
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
    if n_i * n_t <= 1:  # 仅 1 组 (i,θs)：合成图只有一行，多半是上游参数未真正扫到
        print('>>> 提示：仅识别到 1 组 (坡角,入射角)，合成图只有 %d 个面板。' % len(COLUMNS)
              + '请检查各工况 case_meta.json 的 slope_i/incident_angle 是否真随 case_config.json 变化'
              + '（曾发现 config 注入未生效导致全部塌缩到默认 i=45/θs=15）。')  # 善意提示上游问题
    plot_fig8(cases, x_crest, x_toe, total_L, data_dir)  # 绘制图8 并输出到 results/ 目录
    print('>>> 图8 风格对比图完成，输出目录：%s' % os.path.join(data_dir, 'Fig8_TAF_compare'))  # 完成提示


if __name__ == '__main__':  # 主程序入口
    main()  # 运行主流程
