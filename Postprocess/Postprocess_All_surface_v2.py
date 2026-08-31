# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import glob
import io
import json
import math
import time
import logging
import traceback

import numpy as np

DEFAULT_SCRIPT_NAME = 'Postprocess_All_surface_v2.py'  # __file__ 缺失时的兜底文件名

# 指标子图清单（每指标独立子文件夹成图；颜色沿用 Okabe-Ito 色盲安全配色）
# PGA=FE 峰值；Rin=÷注入场(background，模型实际收到的输入场)；R2D1D=÷当地一维场(local_1d，
# 二维/一维响应比)；TAF=÷左侧上平台平场(rect，规范/文献口径的地形放大系数，与论文 G_h 基准一致)
PANEL_SPECS = [
    ('PGA_h', 'blue'), ('PGA_v', 'vermillion'),
    ('Rin_h', 'skyblue'), ('Rin_v', 'orange'),
    ('R2D1D_h', 'blue'), ('R2D1D_v', 'vermillion'),
    ('TAF_h', 'green'), ('TAF_v', 'purple'),
]
SAFE_DENOM_EPS = 1e-30  # 分母安全阈值：低于此值视为无效分母，比值置 NaN
FLAT_Y_TOL = 1e-6  # 节点高程极差低于该值判为平场（无三段结构）
PLOT_SMOOTH_WINDOW = 11  # 成图用分段移动平均窗口（只影响曲线显示，不改变源数据）
FIG_FORMATS = ('png', 'pdf', 'svg')  # 每图导出格式（栅格+矢量，论文投稿口径）
FIG_ROOT_DIR = 'figs'  # 图与源数据子文件夹的根目录
GH_PROFILE_FREQUENCIES = (1.0, 3.0, 5.0, 7.0, 9.0)  # 图8式固定频率剖面的频率（Hz）
GH_REF_MASK_REL = 1e-3  # 参考谱幅值低于该值×逐节点谱峰值的频点视为无激励，置 NaN 断线
GH_LINE_COLORS = ('#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7')  # 论文图8频率配色
GH_LINE_STYLES = ('-', '--', '-.', ':', (0, (5, 1.5)))  # 论文图8频率线型


def _script_path():  # 安全获取当前脚本绝对路径（execfile 环境可能不定义 __file__）
    """返回脚本绝对路径；全局无 __file__ 时退化为当前目录下的已知脚本名。"""
    f = globals().get('__file__')
    if f:
        return os.path.abspath(f)
    return os.path.join(os.getcwd(), DEFAULT_SCRIPT_NAME)


def _script_name():
    """返回脚本文件名，不依赖 __file__。"""
    return os.path.basename(_script_path())


def _to_log_text(value):
    """把日志文本统一为 str，数值参数保持原类型。"""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return value


def _format_log_text(message, args):
    """进入 logging 前完成字符串格式化。"""
    message = _to_log_text(message)
    if not args:
        return message
    normalized = tuple(_to_log_text(item) for item in args)
    try:
        return message % normalized
    except Exception:
        return '{} {}'.format(message, ' '.join(str(i) for i in normalized))


def log_step(logger=None, message=None, *args):
    """日志函数：首次调用（或传入文件名）时初始化日志器，后续调用输出带总用时的日志。"""
    if not hasattr(log_step, '_logger'):
        if logger is not None and isinstance(logger, str):
            log_filename = logger
            logger = None
        else:
            log_filename = os.path.splitext(_script_name())[0] + '.log'  # 与脚本同名日志

        _logger = logging.getLogger('postprocess_surface')
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger.handlers = []
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        log_step._logger = _logger
        log_step._start_time = time.time()
        return _logger

    if message is not None:
        delta = time.time() - log_step._start_time
        log_step._logger.info('[%.3fs] %s' % (delta, _format_log_text(message, args)))
    return log_step._logger


def _load_json(path):  # 读 json，缺失返回 None
    if not os.path.isfile(path):
        return None
    with io.open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


# ==========================================================
#  输入 CSV 读取与工况发现
# ==========================================================


def read_surface_csv(path):
    """读取节点行格式 CSV（node_label,x,y,t=...,t=...）。

    返回 dict：labels(int 数组)/xs/ys(高程)/times(列名时间)/acc(节点×时刻矩阵)。
    """
    with io.open(path, 'r', encoding='utf-8-sig') as fh:
        cols = [c.strip() for c in fh.readline().strip().split(',')]
    if len(cols) < 4 or cols[0] != 'node_label' or cols[1] != 'x' or cols[2] != 'y':
        raise ValueError('不是节点行格式 CSV（表头应为 node_label,x,y,t=...）: %s' % path)
    times = np.array([float(c[2:]) if c.startswith('t=') else float('nan')
                      for c in cols[3:]], dtype=float)  # 时间编码在列名（t=秒）
    data = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)  # 单节点时保持二维
    if data.shape[1] != len(cols):
        raise ValueError('CSV 列数(%d)与表头(%d)不一致: %s' % (data.shape[1], len(cols), path))
    return {'labels': data[:, 0].astype(np.int64), 'xs': data[:, 1].astype(float),
            'ys': data[:, 2].astype(float), 'times': times, 'acc': data[:, 3:].astype(float)}


def discover_records(logger=None):
    """扫描工况目录内的 surface_acc_x_*.csv，并按记录名查找配套的另外 7 个 CSV。

    返回 [(record, paths)]，paths 含存在的文件键 fe_x/fe_y/bg_x/bg_y/local_x/local_y/
    rect_x/rect_y（缺失键即缺文件，对应指标置 NaN）。
    """
    records = []
    for path in sorted(glob.glob('surface_acc_x_*.csv')):
        record = os.path.basename(path)[len('surface_acc_x_'):-len('.csv')]
        paths = {'fe_x': path}
        for key, pattern in (('fe_y', 'surface_acc_y_%s.csv'),
                             ('bg_x', 'freefield_background_x_%s.csv'),
                             ('bg_y', 'freefield_background_y_%s.csv'),
                             ('local_x', 'freefield_local_1d_x_%s.csv'),
                             ('local_y', 'freefield_local_1d_y_%s.csv'),
                             ('rect_x', 'freefield_rect_x_%s.csv'),
                             ('rect_y', 'freefield_rect_y_%s.csv')):
            candidate = pattern % record
            if os.path.isfile(candidate):
                paths[key] = candidate
            elif logger:
                log_step(logger, '警告: %s 缺配套 %s，相关指标将置 NaN', record, candidate)
        records.append((record, paths))
    return records


def _align_by_label(base_labels, other):
    """把 other 的时程行按 base_labels 的节点顺序重排；other 缺该节点时置 NaN 行。"""
    n_time = other['acc'].shape[1]
    out = np.full((len(base_labels), n_time), np.nan, dtype=float)
    row_of = {int(L): i for i, L in enumerate(other['labels'])}
    for k, label in enumerate(base_labels):
        j = row_of.get(int(label))
        if j is not None:
            out[k] = other['acc'][j]
    return out


# ==========================================================
#  几何（s 坐标）与指标
# ==========================================================


def infer_slope_geometry(xs, ys):
    """由地表节点 y 高程反推坡顶/坡脚棱与坡高。

    上平台=ymax 节点（其最右者为坡顶棱），下平台=ymin 节点（其最左者为坡脚棱），
    坡高=ymax−ymin。平场（y 全等）或几何异常返回 None（调用方退化为 x 物理坐标）。
    """
    y_max, y_min = float(np.max(ys)), float(np.min(ys))
    if y_max - y_min <= FLAT_Y_TOL:  # 平场/矩形模型无三段结构
        return None
    top = ys >= y_max - FLAT_Y_TOL
    bot = ys <= y_min + FLAT_Y_TOL
    if not np.any(top) or not np.any(bot):  # 上/下平台节点缺失，几何不完整
        return None
    x_crest = float(np.max(xs[top]))  # 坡顶棱
    x_toe = float(np.min(xs[bot]))  # 坡脚棱
    if x_toe <= x_crest:  # 拓扑异常兜底
        return None
    return x_crest, x_toe, y_max - y_min


def calc_s_coords(xs, x_crest, x_toe, h_ref):
    """物理 x → 三段归一化 s 坐标（段A≤0 / 段B [0,1] / 段C≥1，拐点严格对齐）。"""
    xs = np.asarray(xs, dtype=float)
    s = np.zeros_like(xs)
    if h_ref is None or h_ref <= 0:
        h_ref = 1.0  # 兜底
    w_slope = x_toe - x_crest
    if w_slope <= 0:
        w_slope = 1.0  # 兜底
    s[xs <= x_crest + 1e-4] = (xs[xs <= x_crest + 1e-4] - x_crest) / h_ref  # 段A
    mid = (xs > x_crest + 1e-4) & (xs <= x_toe + 1e-4)
    s[mid] = (xs[mid] - x_crest) / w_slope  # 段B 线性映射至 [0,1]
    right = xs > x_toe + 1e-4
    s[right] = 1.0 + (xs[right] - x_toe) / h_ref  # 段C
    return s


def _peak(mat):
    """逐行峰值 max|·|；全 NaN 行（参考缺失节点）返回 NaN。"""
    finite = np.isfinite(mat)
    safe = np.where(finite, np.abs(mat), 0.0)
    peaks = np.max(safe, axis=1)
    peaks[~np.any(finite, axis=1)] = np.nan
    return peaks


def _safe_ratio(num, den):
    """安全逐元素比值：分母无效（NaN/过小）时置 NaN，不用小量硬造比值。"""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, np.nan, dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > SAFE_DENOM_EPS)
    out[ok] = np.asarray(num)[ok] / np.asarray(den)[ok]
    return out


def compute_metrics(a_h, a_v, refs):
    """由 FE 双分量与三组自由场参考时程计算逐节点指标。

    a_h/a_v 为 FE 水平/竖向 节点×时刻 矩阵（a_v 可为 None → 竖向指标全 NaN）；
    refs 为 {'bg_x','bg_y','local_x','local_y','rect_x','rect_y'} → 参考矩阵或 None
    （缺文件时对应比值全 NaN）。指标口径（分母均为该组逐节点峰值 max|·|）：
      PGA_h/PGA_v      FE 峰值；
      Rin_h/Rin_v      FE ÷ 注入场（背景场=三人工边界实际注入的左柱解在节点高程处，
                       即模型收到的输入场，属验证量）；
      R2D1D_h/R2D1D_v  FE ÷ 当地一维场（一维场地分析预测，二维/一维响应比）；
      TAF_h/TAF_v      FE ÷ 矩形平场（左侧上平台高程 H_upper 平场解地表响应，
                       规范/文献口径的地形放大系数，与论文 G_h 幅值基准一致）。
    分母无效（NaN 或 ≤SAFE_DENOM_EPS，如垂直入射下竖向参考恒 0）时比值置 NaN。
    """
    n_node = a_h.shape[0]
    pga_h = _peak(a_h)
    pga_v = _peak(a_v) if a_v is not None else np.full(n_node, np.nan)

    def _den(key):
        mat = refs.get(key)
        if mat is None:
            return np.full(n_node, np.nan)
        return _peak(mat)

    pbg_h, pbg_v = _den('bg_x'), _den('bg_y')
    p1d_h, p1d_v = _den('local_x'), _den('local_y')
    prt_h, prt_v = _den('rect_x'), _den('rect_y')
    return {
        'PGA_h': pga_h, 'PGA_v': pga_v,
        'Rin_h': _safe_ratio(pga_h, pbg_h), 'Rin_v': _safe_ratio(pga_v, pbg_v),
        'R2D1D_h': _safe_ratio(pga_h, p1d_h), 'R2D1D_v': _safe_ratio(pga_v, p1d_v),
        'TAF_h': _safe_ratio(pga_h, prt_h), 'TAF_v': _safe_ratio(pga_v, prt_v),
    }


# ==========================================================
#  出版级绘图基础设施
# ==========================================================

CB_PALETTE = {  # Okabe-Ito 色盲安全配色
    'black': '#000000', 'orange': '#E69F00', 'skyblue': '#56B4E9',
    'green': '#009E73', 'yellow': '#F0E442', 'blue': '#0072B2',
    'vermillion': '#D55E00', 'purple': '#CC79A7',
}

CJK_SERIF_PRIORITY = [  # 中文衬线字体优先级列表
    'Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC', 'Source Han Serif CN',
    'SimSun', 'NSimSun', 'STSong', 'Songti SC',
]


def _detect_cjk_serif_local():
    """检测可用的衬线宋体名，未找到返回 None。"""
    try:
        import matplotlib.font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        for name in CJK_SERIF_PRIORITY:
            if name in available:
                return name
        for name in available:  # 兜底模糊匹配
            low = name.lower()
            if any(k in low for k in ('song', 'serif cjk', 'serif sc', 'serif cn', 'songti')):
                return name
    except Exception:
        pass
    return None


def _rc_safe(params):
    """逐键写入 rcParams，旧版 matplotlib 缺键时跳过不崩溃。"""
    import matplotlib.pyplot as plt
    for k, v in params.items():
        try:
            plt.rcParams[k] = v
        except Exception:
            pass


def setup_cn_journal_style_local():
    """配置 matplotlib 出版级样式，返回选用的中文字体名（None=未找到，调用方改用英文标签）。"""
    import matplotlib
    cjk = _detect_cjk_serif_local()
    try:
        mpl_ver = tuple(int(p) for p in matplotlib.__version__.split('.')[:2])
    except Exception:
        mpl_ver = (0, 0)
    if cjk:
        if mpl_ver >= (3, 6):  # 新版逐字形回退：Times 在前实现"宋体正文+Times 数字"混排
            serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']
        else:  # 旧版只认首个可用字体，中文在前防止渲染成方框
            serif_list = [cjk, 'Times New Roman', 'STIXGeneral', 'DejaVu Serif']
        _rc_safe({'font.family': serif_list, 'font.serif': serif_list,
                  'mathtext.fontset': 'stix'})
    _rc_safe({
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'font.size': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'lines.linewidth': 1.2,
        'axes.linewidth': 0.7, 'xtick.direction': 'in', 'ytick.direction': 'in',
    })
    return cjk


def style_axes_local(ax):
    """白底、四面朝内刻度与细密主次网格（与独立分图脚本口径一致）。"""
    import matplotlib.ticker as ticker
    ax.set_facecolor('white')
    ax.tick_params(direction='in', which='both', top=True, right=True,
                   bottom=True, left=True)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
    # 仅 Y 轴开次要刻度：段 B 单主刻度时 X 轴 AutoMinorLocator 会崩溃
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)


def smooth_curve_local(values, window=PLOT_SMOOTH_WINDOW):
    """忽略 NaN 的居中移动平均削弱节点锯齿，不填补原始缺口。"""
    arr = np.asarray(values, dtype=float)
    if len(arr) < 3 or int(window) < 3 or not np.any(np.isfinite(arr)):
        return arr
    width = min(int(window), len(arr) if len(arr) % 2 else len(arr) - 1)
    if width < 3:
        return arr
    pad = width // 2
    valid = np.isfinite(arr)
    data = np.where(valid, arr, 0.0)
    sums = np.convolve(np.pad(data, pad, mode='edge'), np.ones(width), mode='valid')
    counts = np.convolve(np.pad(valid.astype(float), pad, mode='edge'),
                         np.ones(width), mode='valid')
    out = np.divide(sums, counts, out=np.nan * np.ones(len(arr)), where=counts > 0)
    out[~valid] = np.nan
    return out


def _field_has_finite_value(data, field):
    """检查字段是否至少包含一个有限值，避免全 NaN 列生成空白面板。"""
    for r in data:
        val = r.get(field)
        try:
            fv = float(val)
            if val is not None and not math.isnan(fv) and not math.isinf(fv):
                return True
        except Exception:
            pass
    return False


# ==========================================================
#  图与源数据输出
# ==========================================================


def _panel_labels(use_cn):
    """返回中/英面板纵轴标签、段标题、横轴标签与总标题模板。"""
    if use_cn:
        labels = {'PGA_h': u'水平向 PGA (m/s²)', 'PGA_v': u'垂直向 PGA (m/s²)',
                  'Rin_h': u'水平向 Rin（÷注入场）', 'Rin_v': u'垂直向 Rin（÷注入场）',
                  'R2D1D_h': u'水平向 R2D1D（÷一维场）', 'R2D1D_v': u'垂直向 R2D1D（÷一维场）',
                  'TAF_h': u'水平向 TAF（÷平场）', 'TAF_v': u'垂直向 TAF（÷平场）'}
        seg_titles = (u'坡顶平台', u'坡面', u'坡脚平台')
        return labels, seg_titles, u'归一化坐标 s', u'记录: %s'
    labels = {'PGA_h': u'Horizontal PGA (m/s²)', 'PGA_v': u'Vertical PGA (m/s²)',
              'Rin_h': u'Horizontal Rin (/ injected field)', 'Rin_v': u'Vertical Rin (/ injected field)',
              'R2D1D_h': u'Horizontal R2D1D (/ 1-D field)', 'R2D1D_v': u'Vertical R2D1D (/ 1-D field)',
              'TAF_h': u'Horizontal TAF (/ flat field)', 'TAF_v': u'Vertical TAF (/ flat field)'}
    return labels, (u'Crest plateau', u'Slope', u'Toe plateau'), \
        u'Normalized coordinate s', u'Record: %s'


def write_fig_data_csv(path, rows, fields):
    """把逐节点源数据（node_label,x,y,s + 指定指标列）写为图文件夹级 CSV。

    fields 为该图包含的指标名列表（单图调用时为单元素列表），行仅含当前图
    绘制范围内的节点。
    """
    cols = ['node_label', 'x', 'y', 's'] + list(fields)
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            vals = [str(int(r['node_label']))]
            for f in cols[1:]:
                v = r.get(f, float('nan'))
                vals.append('%.10g' % v if (v is not None and np.isfinite(v)) else 'nan')
            fh.write(','.join(vals) + '\n')


def plot_metric_panel(fig_dir, fig_name, field, color, record, rows, s_all,
                      flat_mode, a_max, c_max, use_cn, logger=None):
    """绘制单个指标的三段分轴剖面图并多格式导出（每个子图独立成图独立文件夹）。

    flat_mode=True（平场/矩形模型）时横轴退化为 x 物理坐标，无拐点线与段标题。
    图文件夹内的 <fig_name>.csv 仅含当前图绘制范围内（观测窗内）节点的该指标
    原始值——平滑只作用于曲线显示，CSV 不做平滑。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels, seg_titles, xlabel, sup_fmt = _panel_labels(use_cn)
    if flat_mode:
        xlabel = u'坐标 x (m)'  # 平场无三段结构，直接物理坐标

    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    style_axes_local(ax)
    values = np.array([r.get(field, float('nan')) for r in rows], dtype=float)
    if not flat_mode:  # 坡地：A/B/C 各段独立平滑，坡肩/坡脚不跨段抹平
        segments = np.where(s_all <= 0.0, 'A', np.where(s_all < 1.0, 'B', 'C'))
        for seg in ('A', 'B', 'C'):
            mask = segments == seg
            values[mask] = smooth_curve_local(values[mask])
        shown = (s_all >= -a_max - 1e-9) & (s_all <= 1.0 + c_max + 1e-9)
    else:
        values = smooth_curve_local(values)  # 平场整体平滑
        shown = np.ones(len(s_all), dtype=bool)
    # 图文件夹 CSV 仅写当前图绘制范围内的节点（rows 里的原始值，未平滑）
    write_fig_data_csv(os.path.join(fig_dir, fig_name + '.csv'),
                       [r for r, keep in zip(rows, shown) if keep], fields=[field])
    y_view = np.where(shown, values, np.nan)
    finite = np.isfinite(y_view)
    if np.any(finite):
        order = np.argsort(s_all)
        ax.plot(s_all[order], y_view[order], color=color, linestyle='-',
                linewidth=1.2, zorder=3)
    else:
        note = u'无有效数据'
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha='center',
                va='center', fontsize=7)
    if not flat_mode:
        ax.set_xlim(-a_max, 1.0 + c_max)  # 横轴严格采用观测窗实际范围
        tick_start = int(math.ceil(-a_max))
        tick_end = int(math.floor(1.0 + c_max))
        ax.set_xticks([float(t) for t in range(tick_start, tick_end + 1)])
        ax.axvline(0.0, color='#222222', linestyle='--', linewidth=1.15, zorder=10)
        ax.axvline(1.0, color='#222222', linestyle='--', linewidth=1.15, zorder=10)
        for xc, title in zip(((-a_max) / 2.0, 0.5, 1.0 + c_max / 2.0), seg_titles):
            ax.text(xc, 1.035, title, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, clip_on=False)
    else:
        ax.set_xlim(float(np.min(s_all)), float(np.max(s_all)))
    lo, hi = ((float(np.nanmin(y_view)), float(np.nanmax(y_view)))
              if np.any(finite) else (0.0, 1.0))
    pad = 0.06 * ((hi - lo) if hi > lo else max(abs(hi), 1.0))
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel(labels[field])
    ax.set_xlabel(xlabel, labelpad=6)

    fig.suptitle(sup_fmt % record, fontsize=9, fontweight='bold', y=0.97)
    fig_path = os.path.join(fig_dir, fig_name)
    for fmt in FIG_FORMATS:
        try:
            old_err = np.seterr(all='ignore')
            fig.savefig('%s.%s' % (fig_path, fmt), dpi=300, bbox_inches='tight',
                        pad_inches=0.05)
            np.seterr(**old_err)
        except Exception as e2:
            if logger:
                log_step(logger, '[plot] 导出 %s 格式失败: %s', fmt, str(e2))
    plt.close(fig)
    if logger:
        log_step(logger, '[plot] 成功生成指标剖面图: %s.{%s}',
                 fig_path, ','.join(FIG_FORMATS))


def compute_gh_profiles(a_fe, a_ref, times, target_freqs):
    """对 FE 与参考场时程做同口径 FFT，求 |G(f,s)| 在目标频率处的空间剖面。

    幅值口径 |G|=|FFT(FE)|/|FFT(参考)|，参考取 freefield_rect_x（左侧上平台平场解，
    与 TAF 分母一致；水平传播相位只影响相位不影响幅值，无需显式扣除）。参考谱幅值
    低于 GH_REF_MASK_REL×逐节点谱峰值的频点视为无激励置 NaN（曲线断线），直流分量
    剔除。返回 节点×目标频率 幅值矩阵（目标频率在有效频带外或无激励处为 NaN）；
    时长不足或时间轴无效时返回 None。
    """
    n_node = a_fe.shape[0]
    n_t = min(a_fe.shape[1], a_ref.shape[1])
    if len(times) < 2 or n_t < 8:
        return None
    dt = float(times[1] - times[0])
    if dt <= 0.0:
        return None
    nfft = 1 << int(np.ceil(np.log2(n_t * 2)))  # 2 倍补零的 2 幂长度，细化频点
    spec_fe = np.fft.rfft(a_fe[:, :n_t], n=nfft, axis=1)
    spec_ref = np.fft.rfft(a_ref[:, :n_t], n=nfft, axis=1)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    amp_ref = np.abs(spec_ref)
    with np.errstate(divide='ignore', invalid='ignore'):
        gh = np.abs(spec_fe) / amp_ref  # 谱幅值比
    ref_peak = np.max(amp_ref, axis=1)  # 逐节点参考谱峰值
    gh[:, 0] = np.nan  # 剔除直流分量
    gh[amp_ref < GH_REF_MASK_REL * ref_peak[:, None]] = np.nan  # 无激励频点断线
    out = np.full((n_node, len(target_freqs)), np.nan, dtype=float)
    for k in range(n_node):
        ok = np.isfinite(gh[k])
        if np.count_nonzero(ok) < 2:
            continue
        fk, gk = freqs[ok], gh[k][ok]
        for j, fq in enumerate(target_freqs):
            if fk[0] <= fq <= fk[-1]:
                out[k, j] = float(np.interp(fq, fk, gk))
    return out


def plot_gh_profiles(fig_root, record, freqs, gh, node_labels, xs, ys, s_all,
                     flat_mode, use_cn, logger=None):
    """绘制图8式固定频率地表空间幅值剖面（仅幅值，不含相位）并多格式导出。

    论文图8口径：横轴为全地表归一化坐标 s（平场模式退化为 x 物理坐标），灰带=坡面
    s∈[0,1]，虚线=坡顶/坡脚，五条频率曲线沿用论文配色与线型；曲线做轻量空间平滑
    （窗口 3，仅显示用，源数据不变）。纵轴标签用纯文本不用 mathtext（本样式为
    衬线+STIX 数学字体，中文与 $...$ 混排会缺字形成方框）。同时导出源数据 CSV
    （仅含实际绘制的频率列，NaN 记 'nan'）。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, _seg_titles, xlabel, sup_fmt = _panel_labels(use_cn)
    ylabel = u'幅值 |G_h|' if use_cn else u'Amplitude |G_h|'
    if flat_mode:
        xlabel = u'坐标 x (m)'
    has_data = [j for j in range(len(freqs)) if np.any(np.isfinite(gh[:, j]))]
    if not has_data:
        if logger:
            log_step(logger, '[plot] 记录 %s 固定频率幅值剖面无有效数据，跳过', record)
        return
    sub_dir = os.path.join(fig_root, 'Gh_profiles')  # 图8式剖面独立子文件夹
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    fig_name = 'Gh_profiles'
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    style_axes_local(ax)
    curve_max = 0.0
    for j in has_data:
        values = smooth_curve_local(gh[:, j].astype(float), window=3)
        if np.any(np.isfinite(values)):
            curve_max = max(curve_max, float(np.nanmax(values)))
        ax.plot(s_all, values, color=GH_LINE_COLORS[j % len(GH_LINE_COLORS)],
                linestyle=GH_LINE_STYLES[j % len(GH_LINE_STYLES)], linewidth=1.3,
                zorder=3, label='%g Hz' % freqs[j])
    x_lo, x_hi = float(np.min(s_all)), float(np.max(s_all))
    ax.set_xlim(x_lo, x_hi)
    if not flat_mode:  # 坡地：灰带坡面+坡顶坡脚虚线+分区标注（论文图8样式）
        ax.axvspan(0.0, 1.0, color='#D9D9D9', alpha=0.28, zorder=0)
        ax.axvline(0.0, color='#555555', linestyle='--', linewidth=0.8)
        ax.axvline(1.0, color='#555555', linestyle='--', linewidth=0.8)
        seg_names = (u'上平台', u'坡面', u'下平台') if use_cn else \
            (u'Upper plateau', u'Slope', u'Lower plateau')
        for xc, title in zip((x_lo / 2.0, 0.5, (1.0 + x_hi) / 2.0), seg_names):
            ax.text(xc, 1.035, title, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, clip_on=False)
    ax.set_ylim(0.0, max(curve_max, 1e-12) * 1.08)  # 幅值非负，从 0 起轴（论文图8口径）
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.legend(loc='upper right', fontsize=7, frameon=False)
    fig.suptitle(sup_fmt % record, fontsize=9, fontweight='bold', y=0.97)
    fig_path = os.path.join(sub_dir, fig_name)
    for fmt in FIG_FORMATS:
        try:
            old_err = np.seterr(all='ignore')
            fig.savefig('%s.%s' % (fig_path, fmt), dpi=300, bbox_inches='tight',
                        pad_inches=0.05)
            np.seterr(**old_err)
        except Exception as e2:
            if logger:
                log_step(logger, '[plot] 导出 %s 格式失败: %s', fmt, str(e2))
    plt.close(fig)
    # CSV 仅含当前图实际绘制的频率列（无数据频率不出现在图内也不进 CSV）
    cols = ['node_label', 'x', 'y', 's'] + ['G_%gHz' % freqs[j] for j in has_data]
    with io.open(os.path.join(sub_dir, fig_name + '.csv'), 'w', encoding='utf-8',
                 newline='') as fh:
        fh.write(','.join(cols) + '\n')
        for k in range(len(xs)):
            vals = ['%.10g' % v if np.isfinite(v) else 'nan'
                    for v in gh[k, has_data]]
            fh.write('%d,%.10g,%.10g,%.10g,%s\n' % (
                int(node_labels[k]), xs[k], ys[k], s_all[k], ','.join(vals)))
    if logger:
        log_step(logger, '[plot] 成功生成固定频率幅值剖面: %s.{%s}',
                 fig_path, ','.join(FIG_FORMATS))


# ==========================================================
#  主流程
# ==========================================================


def process_one_record(record, paths, case_cfg, use_cn, logger=None):
    """处理单条记录：读 FE 2 个 + 自由场 6 个 CSV → 对齐节点 → 指标 → 出图。

    输出结构 figs/<记录>/：每个指标一个子文件夹（<指标>/<指标>.{png,pdf,svg,csv}，
    CSV 仅含该图绘制范围内的节点数据），另加 Gh_profiles 固定频率幅值剖面。
    """
    fe_x = read_surface_csv(paths['fe_x'])
    labels, xs, ys = fe_x['labels'], fe_x['xs'], fe_x['ys']
    a_h = fe_x['acc']

    def _optional(key):
        if key not in paths:
            return None
        return _align_by_label(labels, read_surface_csv(paths[key]))

    a_v = _optional('fe_y')
    refs = {key: _optional(key) for key in
            ('bg_x', 'bg_y', 'local_x', 'local_y', 'rect_x', 'rect_y')}

    metrics = compute_metrics(a_h, a_v, refs)
    geom = infer_slope_geometry(xs, ys)
    flat_mode = geom is None
    if flat_mode:
        s_all = np.asarray(xs, dtype=float)  # 平场退化：x 物理坐标
        if logger:
            log_step(logger, '%s: 地表高程恒定（平场/矩形），横轴退化为 x 物理坐标', record)
    else:
        x_crest, x_toe, h_slope = geom
        s_all = calc_s_coords(xs, x_crest, x_toe, h_slope)
    gcfg = (case_cfg or {}).get('geometry_cfg') or {}
    crest_win = gcfg.get('crest_window')
    toe_win = gcfg.get('toe_window')
    a_max = float(crest_win) if (crest_win and float(crest_win) > 0) else \
        max(float(-np.min(s_all)) if not flat_mode else 0.0, 0.5)
    c_max = float(toe_win) if (toe_win and float(toe_win) > 0) else \
        max(float(np.max(s_all)) - (0.0 if flat_mode else 1.0), 0.5)
    if not flat_mode and not (crest_win and toe_win):
        if logger:
            log_step(logger, '%s: 未配置 crest_window/toe_window，观测范围不截断', record)

    rows = []
    for k in range(len(xs)):
        row = {'node_label': int(labels[k]), 'x': float(xs[k]), 'y': float(ys[k]),
               's': float(s_all[k])}
        for field, arr in metrics.items():
            row[field] = float(arr[k])
        rows.append(row)

    fig_root = os.path.join(FIG_ROOT_DIR, record)  # 记录级目录：各指标子图文件夹
    if not os.path.isdir(fig_root):
        os.makedirs(fig_root)
    for field, color_name in PANEL_SPECS:
        if not _field_has_finite_value(rows, field):
            if logger:  # 缺配套 CSV 或垂直入射下竖向一维场恒 0 等情形，指标全 NaN，跳过该子图
                log_step(logger, '[plot] 记录 %s 指标 %s 无有效数据，跳过该子图',
                         record, field)
            continue
        sub_dir = os.path.join(fig_root, field)  # 每个子图单独一个子文件夹
        if not os.path.isdir(sub_dir):
            os.makedirs(sub_dir)
        plot_metric_panel(sub_dir, field, field, CB_PALETTE[color_name], record,
                          rows, s_all, flat_mode, a_max, c_max, use_cn, logger)

    # 图8式固定频率地表空间幅值剖面（仅幅值；分母与 TAF 同基准=左侧上平台平场解）
    if refs.get('rect_x') is not None:
        gh = compute_gh_profiles(a_h, refs['rect_x'], fe_x['times'],
                                 GH_PROFILE_FREQUENCIES)
        if gh is not None:
            plot_gh_profiles(fig_root, record, GH_PROFILE_FREQUENCIES, gh,
                             labels, xs, ys, s_all, flat_mode, use_cn, logger)
    elif logger:
        log_step(logger, '[plot] 记录 %s 缺 freefield_rect_x，跳过固定频率幅值剖面',
                 record)

    taf_arr = metrics['TAF_h']
    ar_idx = int(np.nanargmax(taf_arr)) if np.any(~np.isnan(taf_arr)) else None
    dur = float(fe_x['times'][-1] - fe_x['times'][0]) if len(fe_x['times']) > 1 else 0.0
    if logger:
        log_step(logger, '%s: 节点=%d 时长=%.2fs TAF_h_max=%s@x=%s (输出见 %s/)',
                 record, len(xs), dur,
                 ('%.3f' % taf_arr[ar_idx]) if ar_idx is not None else 'N/A',
                 ('%.2f' % xs[ar_idx]) if ar_idx is not None else 'N/A', fig_root)
    return {'record': record, 'n_nodes': len(xs), 'duration': dur,
            'TAF_h_max': (float(taf_arr[ar_idx]) if ar_idx is not None else None),
            'TAF_h_max_x': (float(xs[ar_idx]) if ar_idx is not None else None),
            'flat_mode': flat_mode}


def main():
    """后处理脚本控制流：发现记录 → 逐记录出图（每图独立文件夹含源数据 CSV）。"""
    logger = log_step()
    log_step(logger, '脚本开始执行 (%s)', _script_name())
    case_cfg = _load_json('case_config.json') or {}
    if not case_cfg:
        log_step(logger, '提示: 无 case_config.json，观测窗等配置按缺省处理')

    records = discover_records(logger)
    if not records:
        log_step(logger, '错误: 当前目录无 surface_acc_x_*.csv，无法后处理')
        sys.exit(2)

    try:
        import matplotlib  # noqa: F401 绘图为核心功能，缺失即失败
    except ImportError:
        log_step(logger, '错误: 未检测到 matplotlib，无法出图')
        sys.exit(3)
    try:
        use_cn = bool(setup_cn_journal_style_local())
    except Exception as e:
        log_step(logger, '[plot] 应用出版级样式失败: %s，回退默认配置', str(e))
        use_cn = False
    if not use_cn:
        log_step(logger, '[plot] 提示: 未检测到中文字体，图内文字改用英文标签')

    summaries, failed = [], []
    for record, paths in records:
        try:
            summaries.append(process_one_record(record, paths, case_cfg, use_cn, logger))
        except Exception as e:
            log_step(logger, '错误: 记录 %s 处理失败: %s', record, str(e))
            log_step(logger, '错误堆栈:\n%s', traceback.format_exc())
            failed.append(record)
    log_step(logger, '完成: %d/%d 条记录出图（%s/<记录>/<指标>/ 各含图 3 格式+源数据 CSV）',
             len(summaries), len(records), FIG_ROOT_DIR)
    if failed and not summaries:
        sys.exit(1)  # 全部失败才判失败；部分成功时保留产出并如实记录


if __name__ == '__main__':
    main()
