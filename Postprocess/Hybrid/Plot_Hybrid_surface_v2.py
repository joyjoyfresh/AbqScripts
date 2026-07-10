# -*- coding: utf-8 -*-
"""跨工况地表响应独立分图（Hybrid 专用 v2）。

读取 Collect_All_results_v2.py 收集的 results/ 内 SURFACE_RESULTS NPZ，
将 Postprocess_All_surface_v2.py 输出的可用响应/归一化指标拆为独立图表分别输出。
每张图按三段归一化坐标 s 绘制（坡顶平台 A / 坡面 B / 坡脚平台 C），
样式与原 Postprocess_All_surface_v2 的子图完全一致。

数据来源：
  results/index.csv + results/SURFACE_RESULTS-*.npz（由 Collect_All_results_v2 收集）

运行：
  python Postprocess/Hybrid/Plot_Hybrid_surface_v2.py <工况根目录或 results 目录>
  不传参数则取当前目录。
"""

import os  # 导入系统接口模块
import sys  # 导入系统参数模块
import math  # 导入数学模块
import json  # 导入 NPZ 表清单解析模块
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import matplotlib  # 导入绘图框架
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图子模块
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
    'SimSun', 'NSimSun', 'STSong', 'Songti SC', 'Microsoft YaHei', 'SimHei',
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
        serif_list = [cjk, 'Times New Roman', 'STIXGeneral', 'DejaVu Serif']  # 中文字体优先，避免中文掉字形
        plt.rcParams.update({'font.family': 'serif', 'font.serif': serif_list,
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
#  配置：独立分图字段定义（与 Postprocess_All_surface_v2 的新增指标保持一致）
# ==============================================================================
DRAW_SPECS = [  # (字段键, 中文标签, 英文标签, 曲线颜色)
    ('PGA_h', '水平向 PGA (m/s²)', 'Horizontal PGA (m/s²)', CB_PALETTE['blue']),        # (a)
    ('PGA_v', '垂直向 PGA (m/s²)', 'Vertical PGA (m/s²)', CB_PALETTE['vermillion']),     # (b)
    ('PGA_R', '合成 PGA (m/s²)', 'Resultant PGA (m/s²)', CB_PALETTE['black']),           # 合成峰值
    ('AF_h', '水平向 AF', 'Horizontal AF', CB_PALETTE['blue']),                           # (c)
    ('AF_v', '垂直向 AF', 'Vertical AF', CB_PALETTE['vermillion']),                       # (d)
    ('TAF_h', '水平向 TAF', 'Horizontal TAF', CB_PALETTE['blue']),                        # (e)
    ('TAF_v', '垂直向 TAF', 'Vertical TAF', CB_PALETTE['vermillion']),                    # (f)
    ('TAF_h_comp', '水平分量 TAF', 'Component TAF-H', CB_PALETTE['skyblue']),             # 传统分量口径别名
    ('TAF_v_comp', '竖向分量 TAF', 'Component TAF-V', CB_PALETTE['orange']),              # 传统分量口径别名
    ('VTR', '竖向转换系数 VTR', 'Vertical conversion ratio', CB_PALETTE['purple']),       # 竖向响应/水平自由场
    ('UTAF_h', '统一水平 UTAF', 'Unified TAF-H', CB_PALETTE['skyblue']),                  # 合成自由场分母
    ('UTAF_v', '统一竖向 UTAF', 'Unified TAF-V', CB_PALETTE['orange']),                   # 合成自由场分母
    ('UTAF_R', '统一合成 UTAF', 'Unified resultant TAF', CB_PALETTE['green']),            # 合成响应比
    ('TAF_R', '合成 TAF_R', 'Resultant TAF', CB_PALETTE['green']),                        # 合成响应比别名
    ('DUTAF_v', '竖向增量 ΔUTAF_v', 'Vertical increment ΔUTAF-V', CB_PALETTE['vermillion']),  # 竖向地形增量
    ('V_over_H', '竖横比 V/H', 'Vertical-to-horizontal ratio', CB_PALETTE['black']),      # 同点竖横比
]
PLOT_SMOOTH_WINDOW = 11  # 仅对成图做分段移动平均；CSV 与峰值统计保持原始值


def smooth_curve(values, window=PLOT_SMOOTH_WINDOW):  # 平滑单段空间曲线
    """用忽略 NaN 的居中移动平均削弱节点 PGA 包络锯齿，不填补原有缺失值。"""
    arr = np.asarray(values, dtype=float)  # 转为浮点数组
    if len(arr) < 3 or int(window) < 3 or not np.any(np.isfinite(arr)):  # 无需或无法平滑
        return arr.copy()  # 保持原值
    width = min(int(window) | 1, len(arr) if len(arr) % 2 else len(arr) - 1)  # 使用不超过数据长度的奇数窗
    if width < 3:  # 有效窗口不足
        return arr.copy()  # 保持原值
    pad = width // 2  # 居中窗口两侧补点数
    valid = np.isfinite(arr)  # 有效值掩码
    data = np.where(valid, arr, 0.0)  # NaN 不参与求和
    sums = np.convolve(np.pad(data, pad, mode='edge'), np.ones(width), mode='valid')  # 窗内值之和
    counts = np.convolve(np.pad(valid.astype(float), pad, mode='edge'), np.ones(width), mode='valid')  # 窗内有效数
    out = np.divide(sums, counts, out=np.full(len(arr), np.nan), where=counts > 0)  # 计算移动平均
    out[~valid] = np.nan  # 不填补原曲线缺口
    return out  # 返回平滑结果


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
    """读取 index.csv 并收集全部 NPZ 工况内的 s 网格响应记录。

    返回列表，每项为 dict：source_folder, record, fpath（CSV 绝对路径）。
    """
    idx = pd.read_csv(os.path.join(results_dir, 'index.csv'))  # 读取清单
    rows = idx[idx['type'].astype(str).str.upper() == 'SURFACE_RESULTS_NPZ']  # 仅保留单工况 NPZ 包
    records = []  # 记录列表
    for _, r in rows.iterrows():  # 遍历
        fname = str(r['collected_file'])  # 收集后文件名
        fpath = os.path.join(results_dir, fname)  # 绝对路径
        if not os.path.isfile(fpath):  # 文件缺失
            print('  跳过(文件不存在): %s' % fname); continue  # 跳过
        records.append({  # 追加记录
            'source_folder': str(r.get('source_folder', '')),  # 来源工况目录
            'record': str(r.get('record', '')),  # 输入波记录名
            'fpath': fpath,  # NPZ 文件路径
        })
        print('  收录: %s' % fname)  # 提示
    return records  # 返回列表


def _npz_text(value):  # 解析 NPZ 内的 UTF-8 标量文本
    """兼容 Py2/Py3 的 NPZ UTF-8 标量解码。"""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def read_sgrid_response_npz(path, record):  # 从单工况 NPZ 读取指定记录的 s 网格响应表
    """按 manifest_json 定位 sgrid_response_<record>.csv 并返回 DataFrame。"""
    package = np.load(path)  # NPZ 只含数值与 UTF-8 文本数组，不使用 pickle
    try:
        manifest = json.loads(_npz_text(package['manifest_json']))
        target = 'sgrid_response_%s.csv' % record
        for item in manifest:
            if item.get('name') != target:
                continue
            key = item['key']
            header = [_npz_text(v) for v in package[key + '_header']]
            values = [[_npz_text(v) for v in row] for row in package[key + '_data']]
            frame = pd.DataFrame(values, columns=header)  # 先保留 seg 文本列
            for column in header:
                if column != 'seg':  # 其余字段均为坐标或数值响应
                    frame[column] = pd.to_numeric(frame[column], errors='coerce')
            return frame
    finally:
        package.close()
    raise KeyError('NPZ 中未找到 %s' % target)


# ==============================================================================
#  连续坐标绘制（三段分割线仅作标记）
# ==============================================================================
def draw_single_panel(fig, field, color, ylabel, df, s_all, a_max, c_max, w_b,
                      seg_titles, xlabel):  # 绘制一个连续坐标面板
    """先在同一 x 轴上连续绘制曲线，再叠加坡顶/坡脚分割线，避免三段拼轴造成视觉断层。"""
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.18)  # 单轴布局
    ax = fig.add_subplot(111)  # 连续坐标轴
    style_axes(ax)  # 网格和边框美化

    # 收集连续曲线数据
    vals = df[field].to_numpy(float).copy()  # 复制字段数组，避免 pandas 返回只读视图
    segs = df['seg'].astype(str).to_numpy()  # 三段标签用于防止平滑跨越坡顶和坡脚
    for seg in ('A', 'B', 'C'):  # 各段独立平滑，保留棱角处真实突变
        mask = segs == seg  # 当前段掩码
        vals[mask] = smooth_curve(vals[mask])  # 仅改变显示曲线
    shown = (s_all >= -a_max - 1e-9) & (s_all <= 1.0 + c_max + 1e-9)  # 显示窗掩码
    y_view = np.where(shown, vals, np.nan)  # 窗外不绘制
    finite = np.isfinite(y_view)  # 有效绘图点
    if np.any(finite):  # 有有效曲线
        order = np.argsort(s_all)  # 按 s 升序连线
        ax.plot(s_all[order], y_view[order], color=color, linestyle='-', linewidth=1.2, zorder=3)  # 一条连续曲线
    else:  # 全 NaN 时明确说明物理原因，避免误判为漏画
        note = 'θ=0° 时竖向自由场为 0，分量 TAF_v 不适用' if field in ('TAF_v', 'TAF_v_comp') else '无有效数据'  # 空图说明
        ax.text(0.5, 0.5, note, transform=ax.transAxes, ha='center', va='center', fontsize=7)  # 居中标注

    # 设置坐标范围
    ax.set_xlim(-a_max, 1.0 + c_max)  # 连续横轴范围
    lo, hi = (float(np.nanmin(y_view)), float(np.nanmax(y_view))) if np.any(finite) else (0.0, 1.0)  # 数据极值
    pad = 0.06 * ((hi - lo) if hi > lo else max(abs(hi), 1.0))  # 上下留白
    ax.set_ylim(lo - pad, hi + pad)  # 设定范围

    # 横轴刻度与分割线
    tick_start = int(math.ceil(-a_max))  # 起始整数刻度
    tick_end = int(math.floor(1.0 + c_max))  # 结束整数刻度
    ax.set_xticks([float(t) for t in range(tick_start, tick_end + 1)])  # 整数刻度
    ax.axvline(0.0, color='#222222', linestyle='--', linewidth=1.15, zorder=10)  # 坡顶棱顶层分割线
    ax.axvline(1.0, color='#222222', linestyle='--', linewidth=1.15, zorder=10)  # 坡脚棱顶层分割线

    # 三段标题（放在同一连续轴上方）
    centers = [(-a_max) / 2.0, 0.5, 1.0 + c_max / 2.0]  # 三段标题中心
    for xc, title in zip(centers, seg_titles):  # 逐段标题
        ax.text(xc, 1.035, title, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=7, clip_on=False)  # 标题贴近上轴线

    ax.set_ylabel(ylabel)  # 纵轴标签
    ax.set_xlabel(xlabel, labelpad=6)  # 横轴标签


def plot_record(rec, results_dir, out_dir, use_cn):  # 为一条记录输出多张独立图
    """读取一条 NPZ 内的 SGRID_RESPONSE 表，按可用字段输出独立三段分轴图。"""
    df = read_sgrid_response_npz(rec['fpath'], rec['record'])  # 从 NPZ 读取 s 网格响应表
    s_all = df['s'].to_numpy(float)  # 归一坐标 s 数组

    # 计算三段显示参数
    a_max = max(float(-s_all.min()), 0.5)  # 段A 显示跨度
    c_max = max(float(s_all.max()) - 1.0, 0.5)  # 段C 显示跨度
    w_b = max(1.0, 0.3 * (a_max + c_max))  # 段B 保底宽度

    # 标签（中文/英文）
    if use_cn:  # 中文标签
        seg_titles = ('坡顶平台', '坡面', '坡脚平台')  # 三段标题
        xlabel = '归一化坐标 s'  # 横轴标签
    else:  # 英文兜底
        seg_titles = ('Crest plateau', 'Slope', 'Toe plateau')  # 三段标题
        xlabel = 'Normalized coordinate s'  # 横轴标签

    # 输出子目录
    tag = '%s__%s' % (rec['source_folder'], rec['record']) if rec['record'] else rec['source_folder']  # 记录标识
    rec_dir = os.path.join(out_dir, tag)  # 记录子目录
    if not os.path.isdir(rec_dir):  # 不存在
        os.makedirs(rec_dir)  # 创建

    for pi, (field, cn_lbl, en_lbl, color) in enumerate(DRAW_SPECS):  # 遍历全部可选指标
        if field not in df.columns:  # 字段缺失
            print('    跳过(字段 %s 不在 CSV 中)' % field); continue  # 跳过
        if field not in ('TAF_v', 'TAF_v_comp') and not np.any(np.isfinite(df[field].to_numpy(float))):  # 新增空列不出空白图
            print('    跳过(字段 %s 全为 NaN)' % field); continue  # 跳过
        ylabel = cn_lbl if use_cn else en_lbl  # 选择标签语言

        fig = plt.figure(figsize=(3.15, 2.8))  # 单栏宽画布
        draw_single_panel(fig, field, color, ylabel, df, s_all, a_max, c_max, w_b,
                          seg_titles, xlabel)  # 绘制三段分轴面板

        paths = export_figure(fig, os.path.join(rec_dir, field))  # 多格式导出
        plt.close(fig)  # 释放画布
        print('    %s: %s' % (field, '; '.join(os.path.basename(p) for p in paths)))  # 打印


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

    print('>>> 读取 %s 内的 index.csv 与 SURFACE_RESULTS-*.npz' % data_dir)  # 提示
    records = collect_records(data_dir)  # 收集记录
    if not records:  # 无记录
        print('错误：index.csv 中没有 SURFACE_RESULTS_NPZ 类型的记录。')  # 报错
        return  # 退出
    print('>>> 共 %d 条记录，每条按可用字段输出独立三段分轴图。' % len(records))  # 概览

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
