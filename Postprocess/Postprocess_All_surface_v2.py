# -*- coding: utf-8 -*-
"""地表响应后处理：读取建模脚本输出的 4 类地表 CSV，计算逐节点指标并出图。

输入（当前工况目录，<record>=输入波记录名；均为节点行格式 node_label,x,y,t=...,t=...）:
  surface_acc_x_<record>.csv          FE 水平加速度时程（建模脚本作业完成后从 ODB 提取）
  surface_acc_y_<record>.csv          FE 竖向加速度时程
  freefield_local_1d_<record>.csv     局部一维场（节点当地竖向土柱的解析预测，水平分量）
  freefield_background_<record>.csv   背景场（三人工边界统一注入的左柱平场解，水平分量）

输出（figs/<图名>/ 子文件夹，一图一夹）:
  surface_response_<record>.png/.pdf/.svg  三段分轴多面板指标剖面大图
  surface_response_<record>.csv            该图全部面板的源数据表

指标口径（分母全部来自 CSV 自由场，不依赖 case_meta）:
  PGA_h/v/R  FE 逐节点水平/竖向/合成峰值
  PGA_1d     局部一维场峰值（坡面节点体现当地柱高变化的空间印记）
  PGA_bg     背景场峰值（模型实际收到的输入场，坡面/坡脚节点为左柱内部点取值）
  AF_h/v     = PGA/PGA_bg     相对背景输入场的总放大（场地+地形）
  TAF_h      = PGA_h/PGA_1d   相对当地一维预测的放大（纯地形效应）
  TAF_v      恒 NaN（无竖向一维自由场输出，面板文字注记）
  VTR        = PGA_v/PGA_1d   竖向响应/一维水平预测
  UTAF_R     = PGA_R/PGA_1d   合成响应统一系数
  V_over_H   = PGA_v/PGA_h    竖横峰值比

横轴为三段归一化 s 坐标（上平台 A / 坡面 B / 下平台 C，坡高归一）；坡顶/坡脚棱由
节点 y 高程反推（上平台=ymax 最右节点、下平台=ymin 最左节点）。平场/矩形模型
（y 全等）无三段结构，退化为 x 物理坐标单段轴。观测窗取 case_config.json 的
geometry_cfg.crest_window/toe_window（hs 倍数，缺省不截断）。

运行环境：普通 Python 3 + numpy + matplotlib（无需 Abaqus）。
"""

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

# 指标面板顺序（与源数据 CSV 列一致；颜色沿用 Okabe-Ito 色盲安全配色）
PANEL_SPECS = [
    ('PGA_h', 'blue'), ('PGA_v', 'vermillion'), ('PGA_R', 'black'),
    ('PGA_1d', 'skyblue'), ('PGA_bg', 'orange'),
    ('AF_h', 'blue'), ('AF_v', 'vermillion'),
    ('TAF_h', 'blue'), ('TAF_v', 'vermillion'),
    ('VTR', 'purple'), ('UTAF_R', 'green'), ('V_over_H', 'black'),
]
SAFE_DENOM_EPS = 1e-30  # 分母安全阈值：低于此值视为无效分母，比值置 NaN
FLAT_Y_TOL = 1e-6  # 节点高程极差低于该值判为平场（无三段结构）
PLOT_SMOOTH_WINDOW = 11  # 成图用分段移动平均窗口（只影响曲线显示，不改变源数据）
FIG_FORMATS = ('png', 'pdf', 'svg')  # 每图导出格式（栅格+矢量，论文投稿口径）
FIG_ROOT_DIR = 'figs'  # 图与源数据子文件夹的根目录


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
    """扫描工况目录内的 surface_acc_x_*.csv，并按记录名查找配套的另外 3 个 CSV。

    返回 [(record, paths)]，paths 含存在的文件键 fe_x/fe_y/local/bg（缺失键即缺文件）。
    """
    records = []
    for path in sorted(glob.glob('surface_acc_x_*.csv')):
        record = os.path.basename(path)[len('surface_acc_x_'):-len('.csv')]
        paths = {'fe_x': path}
        for key, pattern in (('fe_y', 'surface_acc_y_%s.csv'),
                             ('local', 'freefield_local_1d_%s.csv'),
                             ('bg', 'freefield_background_%s.csv')):
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


def compute_metrics(a_h, a_v, a_1d, a_bg):
    """由 FE 双分量与两类自由场时程计算逐节点指标（口径见模块 docstring）。

    a_h/a_v/a_1d/a_bg 均为 节点×时刻 矩阵（列数可不同，峰值独立计算）；
    a_v/a_1d/a_bg 可为 None（对应指标全 NaN）。返回字段→数组 的 dict。
    """
    n_node = a_h.shape[0]
    empty = lambda: np.full(n_node, np.nan, dtype=float)  # noqa: E731 缺失参考的占位
    a_v = a_v if a_v is not None else np.full((n_node, 1), np.nan)
    a_1d = a_1d if a_1d is not None else np.full((n_node, 1), np.nan)
    a_bg = a_bg if a_bg is not None else np.full((n_node, 1), np.nan)
    pga_h = _peak(a_h)
    pga_v = _peak(a_v)
    pga_r = _peak(np.sqrt(a_h * a_h + a_v * a_v))  # 逐时刻合成峰值
    pga_1d = _peak(a_1d)
    pga_bg = _peak(a_bg)
    return {
        'PGA_h': pga_h, 'PGA_v': pga_v, 'PGA_R': pga_r,
        'PGA_1d': pga_1d, 'PGA_bg': pga_bg,
        'AF_h': _safe_ratio(pga_h, pga_bg), 'AF_v': _safe_ratio(pga_v, pga_bg),
        'TAF_h': _safe_ratio(pga_h, pga_1d),
        'TAF_v': empty(),  # 无竖向一维自由场输出，恒不定义
        'VTR': _safe_ratio(pga_v, pga_1d),
        'UTAF_R': _safe_ratio(pga_r, pga_1d),
        'V_over_H': _safe_ratio(pga_v, pga_h),
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


def _grayscale_preview_local(png_path):
    """用 Pillow 把 PNG 转灰度另存 *_grayscale.png（色盲自检）；无 Pillow 时跳过。"""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        gray_path = png_path[:-4] + '_grayscale.png'
        Image.open(png_path).convert('L').save(gray_path)
        return gray_path
    except Exception:
        return None


# ==========================================================
#  图与源数据输出
# ==========================================================


def _panel_labels(use_cn):
    """返回中/英面板纵轴标签、段标题、横轴标签与总标题模板。"""
    if use_cn:
        labels = {'PGA_h': u'水平向 PGA (m/s²)', 'PGA_v': u'垂直向 PGA (m/s²)',
                  'PGA_R': u'合成 PGA (m/s²)',
                  'PGA_1d': u'一维预测 PGA (m/s²)', 'PGA_bg': u'背景场 PGA (m/s²)',
                  'AF_h': u'水平向 AF', 'AF_v': u'垂直向 AF',
                  'TAF_h': u'水平向 TAF', 'TAF_v': u'垂直向 TAF',
                  'VTR': u'竖向转换系数 VTR', 'UTAF_R': u'统一合成 UTAF',
                  'V_over_H': u'竖横比 V/H'}
        seg_titles = (u'坡顶平台', u'坡面', u'坡脚平台')
        return labels, seg_titles, u'归一化坐标 s', u'记录: %s'
    labels = {'PGA_h': u'Horizontal PGA (m/s²)', 'PGA_v': u'Vertical PGA (m/s²)',
              'PGA_R': u'Resultant PGA (m/s²)',
              'PGA_1d': u'1-D prediction PGA (m/s²)', 'PGA_bg': u'Background PGA (m/s²)',
              'AF_h': u'Horizontal AF', 'AF_v': u'Vertical AF',
              'TAF_h': u'Horizontal TAF', 'TAF_v': u'Vertical TAF',
              'VTR': u'Vertical conversion ratio', 'UTAF_R': u'Unified resultant TAF',
              'V_over_H': u'Vertical-to-horizontal ratio'}
    return labels, (u'Crest plateau', u'Slope', u'Toe plateau'), \
        u'Normalized coordinate s', u'Record: %s'


def write_fig_data_csv(path, rows):
    """把逐节点源数据（node_label,x,y,s + 全部指标）写为该图的配套 CSV。"""
    fields = ['node_label', 'x', 'y', 's'] + [f for f, _c in PANEL_SPECS]
    with io.open(path, 'w', encoding='utf-8', newline='') as fh:
        fh.write(','.join(fields) + '\n')
        for r in rows:
            vals = [str(int(r['node_label']))]
            for f in fields[1:]:
                v = r.get(f, float('nan'))
                vals.append('%.10g' % v if (v is not None and np.isfinite(v)) else 'nan')
            fh.write(','.join(vals) + '\n')


def plot_record(fig_dir, fig_name, record, rows, s_all, flat_mode,
                a_max, c_max, use_cn, logger=None):
    """按三段分轴布局绘制单记录的多面板指标剖面大图，并多格式导出。

    flat_mode=True（平场/矩形模型）时横轴退化为 x 物理坐标，无拐点线与段标题。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    labels, seg_titles, xlabel, sup_fmt = _panel_labels(use_cn)
    if flat_mode:
        xlabel = u'坐标 x (m)'  # 平场无三段结构，直接物理坐标
    draw_specs = [(f, CB_PALETTE[c]) for f, c in PANEL_SPECS
                  if f in rows[0] and (_field_has_finite_value(rows, f) or f == 'TAF_v')]
    if not draw_specs:
        if logger:
            log_step(logger, '[plot] 记录 %s 无可绘制字段，跳过作图', record)
        return
    n_cols = 2
    n_rows = int(math.ceil(float(len(draw_specs)) / float(n_cols)))

    fig = plt.figure(figsize=(6.3, max(8.2, 2.55 * n_rows + 0.7)))
    outer = gridspec.GridSpec(n_rows, n_cols, left=0.10, right=0.985, top=0.94,
                              bottom=0.055, hspace=0.52, wspace=0.26)

    for panel_idx, (field, color) in enumerate(draw_specs):
        ax = fig.add_subplot(outer[panel_idx // n_cols, panel_idx % n_cols])
        style_axes_local(ax)
        panel_lbl = '(%s)' % chr(ord('a') + panel_idx)  # 学术子图编号
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
        y_view = np.where(shown, values, np.nan)
        finite = np.isfinite(y_view)
        if np.any(finite):
            order = np.argsort(s_all)
            ax.plot(s_all[order], y_view[order], color=color, linestyle='-',
                    linewidth=1.2, zorder=3)
        else:
            note = (u'无竖向一维自由场输出，TAF_v 不适用' if field == 'TAF_v'
                    else u'无有效数据')
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
        ax.text(-0.16, 1.10, panel_lbl, transform=ax.transAxes, ha='left',
                va='bottom', fontsize=8, fontname='Times New Roman',
                fontweight='bold', clip_on=False)

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
    gray = _grayscale_preview_local(fig_path + '.png')
    if logger:
        log_step(logger, '[plot] 成功生成三段分轴图表: %s.{%s}%s',
                 fig_path, ','.join(FIG_FORMATS), (' + 灰度预览' if gray else ''))


# ==========================================================
#  主流程
# ==========================================================


def process_one_record(record, paths, case_cfg, use_cn, logger=None):
    """处理单条记录：读 4 个 CSV → 对齐节点 → 指标 → 图 + 源数据子文件夹。"""
    fe_x = read_surface_csv(paths['fe_x'])
    labels, xs, ys = fe_x['labels'], fe_x['xs'], fe_x['ys']
    a_h = fe_x['acc']

    def _optional(key):
        if key not in paths:
            return None
        return _align_by_label(labels, read_surface_csv(paths[key]))

    a_v = _optional('fe_y')
    a_1d = _optional('local')
    a_bg = _optional('bg')

    metrics = compute_metrics(a_h, a_v, a_1d, a_bg)
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

    fig_name = 'surface_response_%s' % record
    fig_dir = os.path.join(FIG_ROOT_DIR, fig_name)  # 一图一夹：图与源数据同目录
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    write_fig_data_csv(os.path.join(fig_dir, fig_name + '.csv'), rows)
    plot_record(fig_dir, fig_name, record, rows, s_all, flat_mode,
                a_max, c_max, use_cn, logger)

    taf_arr = metrics['TAF_h']
    ar_idx = int(np.nanargmax(taf_arr)) if np.any(~np.isnan(taf_arr)) else None
    dur = float(fe_x['times'][-1] - fe_x['times'][0]) if len(fe_x['times']) > 1 else 0.0
    if logger:
        log_step(logger, '%s: 节点=%d 时长=%.2fs TAF_h_max=%s@x=%s (源数据与图见 %s/)',
                 record, len(xs), dur,
                 ('%.3f' % taf_arr[ar_idx]) if ar_idx is not None else 'N/A',
                 ('%.2f' % xs[ar_idx]) if ar_idx is not None else 'N/A', fig_dir)
    return {'record': record, 'n_nodes': len(xs), 'duration': dur,
            'TAF_h_max': (float(taf_arr[ar_idx]) if ar_idx is not None else None),
            'TAF_h_max_x': (float(xs[ar_idx]) if ar_idx is not None else None),
            'flat_mode': flat_mode}


def main():
    """后处理脚本控制流：发现记录 → 逐记录出图与源数据 → 汇总。"""
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
    log_step(logger, '完成: %d/%d 条记录出图（图与源数据在 %s/<图名>/）',
             len(summaries), len(records), FIG_ROOT_DIR)
    if failed and not summaries:
        sys.exit(1)  # 全部失败才判失败；部分成功时保留产出并如实记录


if __name__ == '__main__':
    main()
