# -*- coding: utf-8 -*-
"""
PGA / 频域传递函数 后处理脚本 v3 —— 适用于 VAB_oblique_TAF_multilayer 系列模型

相对 v2 的改动（对应 ML/research_plan_v1.md §3.5-N1，C1 频域主线）:
1) 新增频域传递函数 H(f, x) 提取模块（本版核心改动）：
   - 读取地表 TOP_SURFACE 所有节点的加速度全时程；
   - 将不等间隔的帧时间线性重采样到均匀时间网格（FFT 前提）；
   - 对每个节点的水平/竖直加速度做单边 FFT；
   - 以"参考谱"为分母做谱比，得到传递函数 H(f, x)：
       H_h(f, x) = A_h(x, f) / Ref(f) ，H_v(f, x) = A_v(x, f) / Ref(f)
     分子分母共用同一稳定分母 Ref（默认取参考节点的水平分量），保证 H_v 也良态；
   - 用"参考谱能量阈值"剔除低能量不可靠频点（单条瞬态脉冲无法做经典相干，
     用输入谱能量门限等效实现 plan 中"相干函数筛除"的目的）；
   - 仅输出限带 [FMIN, FMAX]（默认 0.5~10 Hz，覆盖宽频脉冲有效带宽）。
   物理依据：线弹性系统地表任一点的传递函数与输入波无关，只由几何+地层+入射角决定，
   因此一次宽频脉冲即可一次性提取整条 H(f, x) 曲面，根治"三波困境"。
2) 仍保留 v2 的两项输出（默认开启，向后兼容下游绘图/统计）：
   - TIMESERIES-{name}.csv：地表节点加速度全时程；
   - PGA-{name}.csv：各节点水平/竖直 PGA 峰值表（同时给出 node_label/x/y，
     供 H-{name}.csv 的列与坐标对应，无需额外坐标文件）。
3) 动态斜坡高度 h 的计算逻辑沿用 v2（slope 取顶面起伏；flat 逐级回退）。

运行环境：Abaqus 自带 Python 2.7 + numpy（abaqus python / abaqus cae noGUI）。
本文件为自包含单文件；odbAccess 与 numpy 均做容错导入，便于在普通 Python 下单测纯数值核心。
"""

import os  # 操作系统路径工具
import csv  # CSV 读写
import time  # 计时
import logging  # 日志
import traceback  # 异常堆栈
import glob  # 通配符匹配
import re  # 正则表达式

try:
    import numpy as np  # 数值计算与 FFT（Abaqus python 自带）
    HAS_NUMPY = True  # numpy 可用标志
except ImportError:
    np = None  # 无 numpy 时占位
    HAS_NUMPY = False  # 标记 numpy 不可用（频域模块将被跳过）

try:
    from odbAccess import openOdb  # Abaqus ODB 访问接口（仅 Abaqus 环境可用）
    HAS_ODB = True  # odbAccess 可用标志
except ImportError:
    openOdb = None  # 普通 Python 下占位，便于单测纯数值函数
    HAS_ODB = False  # 标记 odbAccess 不可用

# ==============================================================================
#  配置项
# ==============================================================================
EXPORT_X_OVER_H = False  # 是否在 CSV 中导出 x/h 列（兼容旧绘图脚本时设为 True）

COMPUTE_PGA = True  # 是否输出 PGA 峰值表与 TIMESERIES 时程（向后兼容，默认开）
COMPUTE_HF = True  # 是否输出频域传递函数 H(f, x)（v3 新增核心功能，默认开）

# —— 频域 H(f) 相关配置 ——
FMIN = 0.5  # 输出频带下限（Hz）：低于此频率受输入能量不足/边界污染影响，不可靠
FMAX = 10.0  # 输出频带上限（Hz）：现有 Ricker 脉冲有效带宽上限附近
RESAMPLE = True  # FFT 前是否将不等间隔帧时间线性重采样到均匀网格（强烈建议开）
USE_WINDOW = True  # 是否对时程加汉宁窗抑制截断泄漏（同窗施于分子分母，在比值中相互抵消）
DETREND = True  # FFT 前是否去均值（消除零频直流偏置）
ENERGY_MASK_RATIO = 0.01  # 参考谱能量门限：|Ref(f)| < ratio*max|Ref| 的频点判为不可靠并置 NaN
EXPORT_PHASE = False  # 是否额外输出传递函数相位文件 H-PHASE-{name}.csv

# 参考谱（分母）来源模式：
#   'node'       —— 取某个地表节点的水平加速度谱作分母（标准谱比 SSR，自包含、默认）
#   'input_file' —— 读取外部输入加速度时程文件作分母（绝对场地+地形传递函数）
#   'none'       —— 不做谱比，仅输出各节点单边幅值谱（下游自行相除）
REFERENCE_MODE = 'node'  # 默认使用参考节点谱比，无需外部文件即可运行
REF_NODE_LABEL = None  # 参考节点标号；None 时自动取远场节点（按 x 最小，详见代码）
REF_INPUT_FILE = None  # REFERENCE_MODE='input_file' 时的输入时程文件路径（两列：time accel）


def log_step(logger=None, message=None, *args):
    """
    日志函数：首次调用初始化日志器，后续调用输出带总用时的日志。
    初始化:    logger = log_step('mylog.log')  # 传入日志文件名
               logger = log_step()            # 使用默认文件名 'logfile.log'
    记录日志:  log_step(logger, '消息 %s', val)

    logger: str 或 logging.Logger 实例；首次调用时传入日志文件名字符串
    message: str，日志消息模板（支持 % 占位符）
    *args: 消息模板对应的填充参数
    返回: logging.Logger 实例
    """
    if not hasattr(log_step, '_logger'):  # 判断是否为首次调用（尚未初始化日志器）
        if logger is not None and isinstance(logger, str):  # 若传入为字符串则作日志文件名
            log_filename = logger  # 使用传入字符串作为日志文件名
            logger = None  # 清空，避免后续误用
        else:
            log_filename = 'logfile.log'  # 未指定时使用默认文件名

        _logger = logging.getLogger('abqpy')  # 创建名为 'abqpy' 的日志器
        _logger.setLevel(logging.INFO)  # 日志级别 INFO
        _logger.propagate = False  # 禁止向父日志器传播

        _logger.handlers = []  # 清空已有处理器，防止重复添加
        formatter = logging.Formatter(  # 定义日志格式
            '%(asctime)s [%(levelname)s] %(message)s',  # 时间 [级别] 消息
            datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式
        )

        file_handler = logging.FileHandler(log_filename, mode='w')  # 以写模式打开日志文件
        file_handler.setFormatter(formatter)  # 设置格式
        _logger.addHandler(file_handler)  # 添加处理器

        log_step._logger = _logger  # 保存日志器为函数属性
        log_step._start_time = time.time()  # 记录脚本启动时间
        log_step._log_filename = log_filename  # 保存日志文件名

        return _logger  # 首次调用返回日志器

    if message is not None:  # 若传入消息则记录日志
        now = time.time()  # 当前时间戳
        delta_total = now - log_step._start_time  # 距启动的总耗时
        log_step._logger.info('[%.3fs] ' + message, delta_total, *args)  # 输出带总耗时日志

    return log_step._logger  # 返回日志器


# ==============================================================================
#  顶面节点读取（沿用 v2）
# ==============================================================================
def find_top_surface_nodes(instance):
    """
    读取实例中的 TOP_SURFACE 节点集，返回全部节点坐标。

    instance: Abaqus ODB 装配实例对象
    返回:
        top_nodes: dict, {node_label: (x, y)}
        diagnostics: dict, 诊断信息
    """
    try:
        top_nset = instance.nodeSets['TOP_SURFACE']  # 读取 TOP_SURFACE 节点集
    except KeyError:
        return {}, {'total_nodes': 0, 'selected_top_nodes': 0, 'source': 'TOP_SURFACE missing'}  # 不存在则返回空

    top_nodes = {}  # 节点字典：标号 -> (x, y)
    for node in top_nset.nodes:  # 遍历节点集中所有节点
        x = node.coordinates[0]  # x 坐标
        y = node.coordinates[1]  # y 坐标
        top_nodes[node.label] = (x, y)  # 存储坐标

    diagnostics = {  # 诊断信息
        'total_nodes': len(top_nodes),  # 总节点数
        'selected_top_nodes': len(top_nodes),  # 选中顶部节点数（全部保留）
        'source': 'TOP_SURFACE',  # 数据来源
    }
    return top_nodes, diagnostics  # 返回节点字典与诊断信息


def strip_job_prefix(name):
    """若名称以 job- 开头则去掉前缀，避免输出文件名包含 job-。

    name: str，待处理名称
    返回: str，去前缀后的名称
    """
    if name.lower().startswith('job-'):  # 判断是否以 job- 开头（忽略大小写）
        return name[4:]  # 去掉前 4 个字符
    return name  # 未匹配则原样返回


# ==============================================================================
#  动态计算斜坡高度 h 的辅助方法（沿用 v2）
# ==============================================================================
def get_slope_height_from_odb(odb_path, logger=None):
    """从指定 ODB 读取 TOP_SURFACE 的 y 坐标最大差值作为斜坡高度。

    odb_path: str，目标 ODB 完整路径
    logger: logging.Logger，可选
    返回: float 或 None
    """
    try:
        temp_odb = openOdb(path=odb_path, readOnly=True)  # 只读打开辅助 ODB
        try:
            assembly = temp_odb.rootAssembly  # 根装配体
            inst_keys = list(assembly.instances.keys())  # 实例名列表
            if not inst_keys:  # 无实例则返回 None
                return None
            instance = assembly.instances[inst_keys[0]]  # 取第一个实例
            top_nset = instance.nodeSets['TOP_SURFACE']  # 读取 TOP_SURFACE
            y_coords = [node.coordinates[1] for node in top_nset.nodes]  # 顶面 y 坐标
            if y_coords:  # 非空
                dy = max(y_coords) - min(y_coords)  # y 最大差值
                if dy > 1.0:  # 大于阈值视为有效斜坡高度
                    return dy  # 返回斜坡高度
        finally:
            temp_odb.close()  # 确保关闭 ODB
    except Exception as e:
        if logger:  # 记录错误
            log_step(logger, '读取辅助 ODB %s 失败: %s', os.path.basename(odb_path), str(e))
    return None  # 失败返回 None


def get_slope_height_from_cae(directory, logger=None):
    """
    从目录下 .cae 文件名解析斜坡高度 h（文件名形如 h200_i30_a15.cae）。

    directory: str，搜索目录
    logger: logging.Logger，可选
    返回: float 或 None
    """
    try:
        cae_files = sorted(glob.glob(os.path.join(directory, '*.cae')))  # 目录下所有 .cae
        if cae_files:  # 找到文件
            cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]  # 不含扩展名的文件名
            match = re.search(r'h(?P<h>-?\d+(?:\.\d+)?)_', cae_name)  # 正则匹配 h 参数
            if match:  # 匹配成功
                h_val = float(match.group('h'))  # 转浮点
                if h_val > 0.0:  # 正数有效
                    return h_val  # 返回 h
    except Exception as e:
        if logger:  # 记录错误
            log_step(logger, '从 CAE 文件名解析 h 失败: %s', str(e))
    return None  # 失败返回 None


def resolve_slope_height(odb_path, odb_stem, top_nodes, top_labels, logger=None):
    """
    综合判定斜坡高度 h：slope 模型取顶面起伏；flat 模型逐级回退。

    odb_path: str，当前 ODB 路径
    odb_stem: str，当前 ODB 不含扩展名文件名
    top_nodes: dict，{label: (x, y)}
    top_labels: list，按 x 排序的节点标号
    logger: logging.Logger，可选
    返回: float，斜坡高度 h
    """
    y_coords = [top_nodes[lb][1] for lb in top_labels]  # 顶面 y 坐标
    dy = max(y_coords) - min(y_coords)  # 顶面 y 差值
    odb_basename = os.path.basename(odb_path)  # 文件名（含扩展名）

    if dy > 1.0:  # 差值显著 -> slope 模型
        log_step(logger, '%s 识别为 slope 模型，h = y_max - y_min = %.6f', odb_basename, dy)  # 记录
        return dy  # 直接用顶面起伏作 h

    h = None  # flat 模型：逐级尝试获取 h
    dir_name = os.path.dirname(odb_path)  # 当前目录

    if '-flat' in odb_stem.lower():  # 当前为 flat 模型，先找对应 slope
        slope_stem = odb_stem.lower().replace('-flat', '-slope')  # 构造对应 slope 文件名
        for f in os.listdir(dir_name):  # 遍历同目录文件
            if f.lower() == slope_stem + '.odb':  # 命中对应 slope ODB
                h = get_slope_height_from_odb(os.path.join(dir_name, f), logger)  # 读取其 h
                if h is not None:  # 成功
                    log_step(logger, '%s 从对应 slope 文件 %s 获取 h = %.6f', odb_basename, f, h)  # 记录
                    break

    if h is None:  # 仍未获取，找任意 slope ODB
        for f in os.listdir(dir_name):  # 遍历同目录文件
            if f.lower().endswith('.odb') and '-slope' in f.lower():  # 命中任意 slope ODB
                h = get_slope_height_from_odb(os.path.join(dir_name, f), logger)  # 读取其 h
                if h is not None:  # 成功
                    log_step(logger, '%s 从任意 slope 文件 %s 获取 h = %.6f', odb_basename, f, h)  # 记录
                    break

    if h is None:  # 仍未获取，尝试 .cae 文件名
        h = get_slope_height_from_cae(dir_name, logger)  # 解析 h
        if h is not None:  # 成功
            log_step(logger, '%s 从 .cae 文件名解析 h = %.6f', odb_basename, h)  # 记录

    if h is None:  # 全部失败，使用兜底默认
        h = 200.0  # 默认斜坡高度
        log_step(logger, '%s 未找到 slope/CAE，使用默认 h = %.6f', odb_basename, h)  # 记录

    return h  # 返回最终 h


# ==============================================================================
#  频域核心：均匀重采样 / 单边 FFT / 传递函数（纯数值，可脱离 Abaqus 单测）
# ==============================================================================
def resample_to_uniform(times, values_2d, n_out=None):
    """
    将不等间隔时间序列线性重采样到均匀时间网格，供 FFT 使用。

    times: 1D array-like，原始帧时间（单调不减）
    values_2d: 2D array，形状 (n_series, n_t)，每行一条时程
    n_out: int 或 None，输出采样点数；None 时取原帧数
    返回:
        t_uni: 1D ndarray，均匀时间网格
        dt: float，均匀采样间隔
        out_2d: 2D ndarray，重采样后的时程，形状 (n_series, n_out)
    """
    t = np.asarray(times, dtype=float)  # 原始时间转浮点数组
    v = np.asarray(values_2d, dtype=float)  # 时程矩阵转浮点数组
    if v.ndim == 1:  # 兼容单条一维输入
        v = v.reshape(1, -1)  # 调整为二维 (1, n_t)
    n_t = t.size  # 原始采样点数
    if n_out is None:  # 未指定输出点数
        n_out = n_t  # 默认与原始点数相同
    t0 = t[0]  # 起始时间
    t1 = t[-1]  # 结束时间
    t_uni = np.linspace(t0, t1, n_out)  # 均匀时间网格
    dt = (t1 - t0) / float(n_out - 1) if n_out > 1 else 0.0  # 均匀采样间隔
    out_2d = np.empty((v.shape[0], n_out), dtype=float)  # 输出矩阵
    for i in range(v.shape[0]):  # 逐条时程线性插值
        out_2d[i, :] = np.interp(t_uni, t, v[i, :])  # numpy 线性插值
    return t_uni, dt, out_2d  # 返回均匀网格、间隔与重采样结果


def single_sided_fft(values_2d, dt, window=True, detrend=True):
    """
    对一批等长时程做单边（实数）FFT。

    values_2d: 2D array，形状 (n_series, n_t)，每行一条均匀采样时程
    dt: float，采样间隔（s）
    window: bool，是否加汉宁窗（同窗施于全部序列，在传递函数比值中相互抵消）
    detrend: bool，是否去均值
    返回:
        freqs: 1D ndarray，单边频率轴（Hz）
        spec: 2D complex ndarray，形状 (n_series, n_freq) 的复数谱
    """
    v = np.asarray(values_2d, dtype=float)  # 转浮点数组
    if v.ndim == 1:  # 兼容一维
        v = v.reshape(1, -1)  # 调整为二维
    n_t = v.shape[1]  # 每条时程长度
    work = v.copy()  # 复制以免改动原数据
    if detrend:  # 去均值
        work = work - work.mean(axis=1, keepdims=True)  # 逐行减均值
    if window:  # 加窗
        w = np.hanning(n_t)  # 汉宁窗
        work = work * w  # 逐行乘窗
    spec = np.fft.rfft(work, axis=1)  # 单边复数谱（沿时间轴）
    freqs = np.fft.rfftfreq(n_t, d=dt)  # 单边频率轴
    return freqs, spec  # 返回频率轴与复数谱


def transfer_function(num_spec, ref_spec, energy_mask_ratio=0.01):
    """
    由分子谱与参考（分母）谱计算复数传递函数，并给出可靠频点掩膜。

    num_spec: 2D complex ndarray，形状 (n_node, n_freq)，各节点分子谱
    ref_spec: 1D complex ndarray，形状 (n_freq,)，参考（分母）谱
    energy_mask_ratio: float，参考谱能量门限比例；|Ref|<ratio*max|Ref| 的频点判为不可靠
    返回:
        H: 2D complex ndarray，传递函数（不可靠频点已置 NaN）
        mask: 1D bool ndarray，True 表示该频点可靠
    """
    num = np.asarray(num_spec)  # 分子谱
    if num.ndim == 1:  # 兼容一维
        num = num.reshape(1, -1)  # 调整为二维
    ref = np.asarray(ref_spec)  # 参考谱
    ref_amp = np.abs(ref)  # 参考谱幅值
    max_amp = ref_amp.max() if ref_amp.size else 0.0  # 参考谱最大幅值
    thr = energy_mask_ratio * max_amp  # 能量门限绝对值
    mask = ref_amp >= thr  # 可靠频点掩膜（参考能量足够）
    ref_safe = np.where(mask, ref, 1.0)  # 不可靠处用 1.0 占位避免除零
    H = num / ref_safe  # 复数传递函数（逐节点除以参考谱）
    H[:, ~mask] = np.nan  # 不可靠频点置 NaN
    return H, mask  # 返回传递函数与掩膜


def band_limit(freqs, fmin, fmax):
    """
    返回频率落在 [fmin, fmax] 内的布尔索引。

    freqs: 1D ndarray，频率轴（Hz）
    fmin: float，下限
    fmax: float，上限
    返回: 1D bool ndarray
    """
    f = np.asarray(freqs, dtype=float)  # 频率轴
    return (f >= fmin) & (f <= fmax)  # 落在带宽内为 True


def read_input_time_history(path):
    """
    读取外部输入加速度时程文件（两列：time accel，支持空格/逗号/制表符分隔，# 为注释）。

    path: str，文件路径
    返回:
        times: 1D ndarray
        accel: 1D ndarray
    """
    t_list = []  # 时间列
    a_list = []  # 加速度列
    with open(path, 'r') as f:  # 打开输入文件
        for line in f:  # 逐行解析
            s = line.strip()  # 去首尾空白
            if (not s) or s.startswith('#'):  # 跳过空行与注释
                continue
            parts = re.split(r'[\s,]+', s)  # 按空白或逗号切分
            if len(parts) < 2:  # 列数不足则跳过
                continue
            try:
                t_list.append(float(parts[0]))  # 解析时间
                a_list.append(float(parts[1]))  # 解析加速度
            except ValueError:
                continue  # 非数值行跳过
    return np.asarray(t_list, dtype=float), np.asarray(a_list, dtype=float)  # 返回数组


# ==============================================================================
#  处理单个 ODB 文件
# ==============================================================================
def process_one_odb(odb_path, logger=None):
    """
    处理单个 ODB：提取地表节点加速度全时程，输出 TIMESERIES / PGA / H(f) 三类结果。

    odb_path: str，目标 ODB 完整路径
    logger: logging.Logger，可选
    """
    logger = logger or log_step()  # 默认日志器
    odb_basename = os.path.basename(odb_path)  # 文件名（含扩展名）
    odb_stem = os.path.splitext(odb_basename)[0]  # 不含扩展名文件名

    log_step(logger, '开始处理 ODB: %s', odb_basename)  # 记录开始

    odb = openOdb(path=odb_path, readOnly=True)  # 只读打开 ODB

    try:
        assembly = odb.rootAssembly  # 根装配体
        inst_keys = list(assembly.instances.keys())  # 实例名列表
        if not inst_keys:  # 无实例则跳过
            log_step(logger, '%s 无装配实例，跳过', odb_basename)
            return
        instance = assembly.instances[inst_keys[0]]  # 取第一个实例

        top_nodes, _ = find_top_surface_nodes(instance)  # 读取顶面节点
        if not top_nodes:  # 无顶面节点则跳过
            log_step(logger, '%s 未找到顶部表面节点，跳过', odb_basename)
            return

        top_labels = sorted(list(top_nodes.keys()), key=lambda lb: top_nodes[lb][0])  # 按 x 升序排列
        label_to_idx = {}  # 标号 -> 行索引
        for i, lb in enumerate(top_labels):  # 建立映射
            label_to_idx[lb] = i
        n_top = len(top_labels)  # 顶面节点数

        h = resolve_slope_height(odb_path, odb_stem, top_nodes, top_labels, logger)  # 计算斜坡高度

        step_keys = list(odb.steps.keys())  # 分析步名列表
        if not step_keys:  # 无分析步则跳过
            log_step(logger, '%s 无分析步，跳过', odb_basename)
            return
        step = odb.steps[step_keys[-1]]  # 取最后一个分析步（动力步）
        frames = step.frames  # 帧序列
        n_frames = len(frames)  # 帧数
        if n_frames == 0:  # 无帧则跳过
            log_step(logger, '%s 分析步 %s 无帧数据，跳过', odb_basename, step_keys[-1])
            return

        log_step(logger, '%s 使用分析步 %s，帧数=%d，顶部节点数=%d',
                 odb_basename, step_keys[-1], n_frames, n_top)  # 记录基本信息

        top_nset = instance.nodeSets['TOP_SURFACE']  # 顶面节点集（用于帧子集提取）

        # ==========================================
        #  一次遍历帧，把地表全时程读入内存数组
        #  （仅地表节点，内存占用可控；FFT 需要完整时程，故不再像 v2 那样仅维护running max）
        # ==========================================
        times = np.zeros(n_frames, dtype=float) if HAS_NUMPY else [0.0] * n_frames  # 帧时间数组
        if HAS_NUMPY:  # 有 numpy 时用二维数组存全时程
            acc_h = np.zeros((n_top, n_frames), dtype=float)  # 水平加速度时程矩阵
            acc_v = np.zeros((n_top, n_frames), dtype=float)  # 竖直加速度时程矩阵
        else:  # 无 numpy 时退化为列表（仅能算 PGA，不能做 FFT）
            acc_h = [[0.0] * n_frames for _ in range(n_top)]
            acc_v = [[0.0] * n_frames for _ in range(n_top)]

        log_step(logger, '%s 开始读取地表节点全时程', odb_basename)  # 记录
        for frame_idx, frame in enumerate(frames):  # 遍历所有帧
            if (frame_idx + 1) % max(1, n_frames // 10) == 0:  # 每约 10% 记录进度
                log_step(logger, '%s 读取帧进度: %d / %d', odb_basename, frame_idx + 1, n_frames)
            times[frame_idx] = getattr(frame, 'frameValue', 0.0)  # 记录帧时间
            if 'A' not in frame.fieldOutputs:  # 无加速度场则该帧留 0
                continue
            acc_values = frame.fieldOutputs['A'].getSubset(region=top_nset).values  # 顶面子集加速度
            for val in acc_values:  # 遍历各节点值
                idx = label_to_idx.get(val.nodeLabel)  # 该节点行索引
                if idx is None:  # 非顶面节点跳过
                    continue
                acc_h[idx][frame_idx] = val.data[0]  # 水平分量
                acc_v[idx][frame_idx] = val.data[1]  # 竖直分量

        csv_stem = strip_job_prefix(odb_stem)  # 去 job- 前缀的输出名

        # ==========================================
        #  输出一：TIMESERIES 全时程 + PGA 峰值表（向后兼容，默认开）
        # ==========================================
        if COMPUTE_PGA:  # 是否输出 PGA/时程
            _write_timeseries_and_pga(csv_stem, top_labels, top_nodes, label_to_idx,
                                      times, acc_h, acc_v, h, odb_basename, logger)  # 调用输出函数

        # ==========================================
        #  输出二：频域传递函数 H(f, x)（v3 新增核心，默认开）
        # ==========================================
        if COMPUTE_HF:  # 是否输出 H(f)
            if not HAS_NUMPY:  # 无 numpy 无法做 FFT
                log_step(logger, '%s 未检测到 numpy，跳过频域 H(f) 提取', odb_basename)
            else:
                _write_transfer_function(csv_stem, top_labels, top_nodes,
                                         times, acc_h, acc_v, h, odb_basename, logger)  # 调用频域输出

    except Exception as exc:
        log_step(logger, '%s 处理失败: %s', odb_basename, str(exc))  # 记录异常
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 输出堆栈
        raise  # 重新抛出
    finally:
        odb.close()  # 确保关闭 ODB


def _write_timeseries_and_pga(csv_stem, top_labels, top_nodes, label_to_idx,
                              times, acc_h, acc_v, h, odb_basename, logger):
    """
    写出 TIMESERIES 全时程表与 PGA 峰值表（行为与 v2 一致）。

    参数均由 process_one_odb 传入：节点顺序、坐标、时程矩阵、斜坡高度等。
    返回: None
    """
    n_top = len(top_labels)  # 节点数
    n_frames = len(times)  # 帧数

    timeseries_csv = 'TIMESERIES-{0}.csv'.format(csv_stem)  # 时程文件名
    pga_csv = 'PGA-{0}.csv'.format(csv_stem)  # PGA 文件名

    # —— TIMESERIES 全时程 ——
    log_step(logger, '%s 写出全时程: %s', odb_basename, timeseries_csv)  # 记录
    with open(timeseries_csv, 'w') as f_ts:  # 打开时程文件
        writer = csv.writer(f_ts, lineterminator='\n')  # CSV 写入器
        header = ['Time']  # 表头首列时间
        if EXPORT_X_OVER_H:  # 可选 x/h 列
            header.append('x/h')
        for lb in top_labels:  # 各节点水平列
            header.append('Node_{0}_Accel_h'.format(lb))
        for lb in top_labels:  # 各节点竖直列
            header.append('Node_{0}_Accel_v'.format(lb))
        writer.writerow(header)  # 写表头

        for k in range(n_frames):  # 逐帧写一行
            row = ['{0:.6e}'.format(times[k])]  # 时间列
            if EXPORT_X_OVER_H:  # 可选 x/h
                row.append('{0:.6e}'.format(top_nodes[top_labels[0]][0] / h))
            for i in range(n_top):  # 水平加速度
                row.append('{0:.6e}'.format(acc_h[i][k]))
            for i in range(n_top):  # 竖直加速度
                row.append('{0:.6e}'.format(acc_v[i][k]))
            writer.writerow(row)  # 写入该帧
    log_step(logger, '%s 全时程已保存: %s', odb_basename, timeseries_csv)  # 记录

    # —— PGA 峰值表 ——
    log_step(logger, '%s 写出 PGA 峰值表: %s', odb_basename, pga_csv)  # 记录
    results = []  # 结果列表
    for i, lb in enumerate(top_labels):  # 逐节点计算 PGA
        x_coord, y_coord = top_nodes[lb]  # 坐标
        if HAS_NUMPY:  # numpy 求峰值与峰值时刻
            ah = np.abs(acc_h[i]); av = np.abs(acc_v[i])  # 绝对值序列
            ih = int(np.argmax(ah)); iv = int(np.argmax(av))  # 峰值索引
            pga_h = float(ah[ih]); pga_v = float(av[iv])  # 峰值
            th = float(times[ih]); tv = float(times[iv])  # 峰值时刻
        else:  # 纯 Python 退化
            ah = [abs(x) for x in acc_h[i]]; av = [abs(x) for x in acc_v[i]]
            ih = ah.index(max(ah)); iv = av.index(max(av))
            pga_h = ah[ih]; pga_v = av[iv]; th = times[ih]; tv = times[iv]
        results.append({  # 记录该节点结果
            'x/h': x_coord / h, 'node_label': lb, 'x': x_coord, 'y': y_coord,
            'PGA_h': pga_h, 'PGA_v': pga_v, 'peak_h_time': th, 'peak_v_time': tv,
        })
    results.sort(key=lambda r: r['x'])  # 按 x 升序

    with open(pga_csv, 'w') as f_pga:  # 打开 PGA 文件
        writer = csv.writer(f_pga, lineterminator='\n')  # CSV 写入器
        pga_header = []  # 表头
        if EXPORT_X_OVER_H:  # 可选 x/h
            pga_header.append('x/h')
        pga_header.extend(['node_label', 'x', 'y', 'PGA_h', 'PGA_v', 'peak_h_time', 'peak_v_time'])  # 固定列
        writer.writerow(pga_header)  # 写表头
        for r in results:  # 逐节点写
            row = []
            if EXPORT_X_OVER_H:  # 可选 x/h
                row.append('%.6f' % r['x/h'])
            row.extend(['%d' % r['node_label'], '%.6f' % r['x'], '%.6f' % r['y'],
                        '%.6f' % r['PGA_h'], '%.6f' % r['PGA_v'],
                        '%.6f' % r['peak_h_time'], '%.6f' % r['peak_v_time']])  # 各列
            writer.writerow(row)  # 写入
    log_step(logger, '%s PGA 峰值表已保存: %s，节点数=%d', odb_basename, pga_csv, len(results))  # 记录


def _resolve_reference_spectrum(freqs_node, spec_h, top_labels, top_nodes, n_uniform, dt, odb_basename, logger):
    """
    根据 REFERENCE_MODE 解析传递函数的分母（参考谱）。

    freqs_node: 节点谱对应频率轴
    spec_h: 各节点水平复数谱，形状 (n_node, n_freq)
    top_labels: 按 x 排序的节点标号
    top_nodes: {label: (x, y)}
    n_uniform: 均匀采样点数（与节点 FFT 一致，供输入文件谱对齐）
    dt: 均匀采样间隔
    返回:
        ref_spec: 1D complex ndarray 或 None（None 表示 'none' 模式，不做谱比）
        ref_desc: str，参考来源说明（写入日志）
    """
    if REFERENCE_MODE == 'none':  # 不做谱比
        return None, 'none(原始幅值谱)'

    if REFERENCE_MODE == 'input_file':  # 外部输入时程作分母
        if REF_INPUT_FILE and os.path.isfile(REF_INPUT_FILE):  # 文件存在
            t_in, a_in = read_input_time_history(REF_INPUT_FILE)  # 读输入时程
            _, _, a_uni = resample_to_uniform(t_in, a_in.reshape(1, -1), n_out=n_uniform)  # 重采样到同长度
            _, ref = single_sided_fft(a_uni, dt, window=USE_WINDOW, detrend=DETREND)  # 输入谱
            return ref[0], 'input_file:%s' % os.path.basename(REF_INPUT_FILE)  # 返回输入谱
        log_step(logger, '%s REFERENCE_MODE=input_file 但文件无效，回退到 node 模式', odb_basename)  # 警告回退

    # —— node 模式（默认；input_file 失败时也回退到此）——
    ref_idx = None  # 参考节点行索引
    if REF_NODE_LABEL is not None and REF_NODE_LABEL in top_labels:  # 指定了有效参考节点
        ref_idx = top_labels.index(REF_NODE_LABEL)  # 取其索引
        ref_desc = 'node:label=%s(指定)' % str(REF_NODE_LABEL)  # 说明
    else:
        ref_idx = 0  # 默认取 x 最小的远场节点作参考（标准谱比基准）
        ref_desc = 'node:label=%s(x最小远场,自动)' % str(top_labels[0])  # 说明
    return spec_h[ref_idx], ref_desc  # 返回参考节点水平谱


def _write_transfer_function(csv_stem, top_labels, top_nodes,
                             times, acc_h, acc_v, h, odb_basename, logger):
    """
    计算并写出频域传递函数 H(f, x)：H-{name}.csv（幅值）与可选 H-PHASE-{name}.csv（相位）。

    列与 PGA-{name}.csv 的节点顺序一致；各节点的 x/y/x_over_h 坐标可从 PGA 文件对应取得。
    返回: None
    """
    n_top = len(top_labels)  # 节点数

    # —— 重采样到均匀网格 ——
    if RESAMPLE:  # 不等间隔帧 -> 均匀网格
        t_uni, dt, ah_uni = resample_to_uniform(times, np.asarray(acc_h))  # 水平重采样
        _, _, av_uni = resample_to_uniform(times, np.asarray(acc_v))  # 竖直重采样
    else:  # 假定已均匀
        t_uni = np.asarray(times, dtype=float)  # 时间轴
        dt = (t_uni[-1] - t_uni[0]) / float(t_uni.size - 1) if t_uni.size > 1 else 0.0  # 间隔
        ah_uni = np.asarray(acc_h, dtype=float)  # 水平时程
        av_uni = np.asarray(acc_v, dtype=float)  # 竖直时程
    if dt <= 0.0:  # 间隔异常则放弃
        log_step(logger, '%s 采样间隔异常(dt=%.3e)，跳过频域提取', odb_basename, dt)
        return

    # —— 单边 FFT ——
    freqs, spec_h = single_sided_fft(ah_uni, dt, window=USE_WINDOW, detrend=DETREND)  # 水平谱
    _, spec_v = single_sided_fft(av_uni, dt, window=USE_WINDOW, detrend=DETREND)  # 竖直谱
    log_step(logger, '%s FFT 完成: dt=%.4es, N=%d, df=%.4fHz, fNyq=%.2fHz',
             odb_basename, dt, t_uni.size, freqs[1] - freqs[0] if freqs.size > 1 else 0.0, freqs[-1])  # 记录

    # —— 解析参考谱（分母）——
    ref_spec, ref_desc = _resolve_reference_spectrum(
        freqs, spec_h, top_labels, top_nodes, t_uni.size, dt, odb_basename, logger)  # 取分母
    log_step(logger, '%s 传递函数分母来源: %s', odb_basename, ref_desc)  # 记录分母来源

    # —— 计算传递函数 / 或保留原始幅值谱 ——
    if ref_spec is None:  # 'none' 模式：输出单边幅值谱（已做单边幅值修正）
        n_t = t_uni.size  # 采样点数
        scale = 2.0 / n_t  # 单边幅值修正系数
        Hh = np.abs(spec_h) * scale  # 水平幅值谱
        Hv = np.abs(spec_v) * scale  # 竖直幅值谱
        mask = np.ones(freqs.size, dtype=bool)  # 全频点保留
        col_tag = 'AMP'  # 列名标识：幅值谱
    else:  # 谱比模式：H = 节点谱 / 参考谱（H_h、H_v 共用同一稳定分母）
        Hh_c, mask = transfer_function(spec_h, ref_spec, ENERGY_MASK_RATIO)  # 水平传递函数
        Hv_c, _ = transfer_function(spec_v, ref_spec, ENERGY_MASK_RATIO)  # 竖直传递函数
        Hh = np.abs(Hh_c)  # 水平幅值
        Hv = np.abs(Hv_c)  # 竖直幅值
        col_tag = 'H'  # 列名标识：传递函数

    # —— 限带 [FMIN, FMAX] ——
    band = band_limit(freqs, FMIN, FMAX)  # 带内布尔索引
    sel = band & mask if ref_spec is not None else band  # 谱比模式同时要求可靠
    f_out = freqs[band]  # 输出频率（仅按带宽截取，掩膜以 NaN 体现而非删行）
    if f_out.size == 0:  # 带内无频点
        log_step(logger, '%s 频带 [%.2f,%.2f]Hz 内无有效频点，跳过', odb_basename, FMIN, FMAX)
        return

    # 谱比模式下：带内但不可靠的频点幅值置 NaN（保持频率轴连续，便于 ML 网格对齐）
    Hh_band = Hh[:, band].copy()  # 带内水平幅值
    Hv_band = Hv[:, band].copy()  # 带内竖直幅值
    if ref_spec is not None:  # 谱比模式应用可靠性掩膜
        band_mask = mask[band]  # 带内可靠掩膜
        Hh_band[:, ~band_mask] = np.nan  # 不可靠置 NaN
        Hv_band[:, ~band_mask] = np.nan  # 不可靠置 NaN

    # —— 写出幅值文件 ——
    h_csv = 'H-{0}.csv'.format(csv_stem)  # 传递函数幅值文件名
    log_step(logger, '%s 写出频域传递函数: %s（%d 频点 × %d 节点）',
             odb_basename, h_csv, f_out.size, n_top)  # 记录
    with open(h_csv, 'w') as f_h:  # 打开文件
        writer = csv.writer(f_h, lineterminator='\n')  # CSV 写入器
        header = ['Freq_Hz']  # 表头首列频率
        for lb in top_labels:  # 各节点水平列
            header.append('Node_{0}_{1}_h'.format(lb, col_tag))
        for lb in top_labels:  # 各节点竖直列
            header.append('Node_{0}_{1}_v'.format(lb, col_tag))
        writer.writerow(header)  # 写表头
        for j in range(f_out.size):  # 逐频点写一行
            row = ['{0:.6e}'.format(f_out[j])]  # 频率列
            for i in range(n_top):  # 水平幅值
                row.append(_fmt_val(Hh_band[i, j]))
            for i in range(n_top):  # 竖直幅值
                row.append(_fmt_val(Hv_band[i, j]))
            writer.writerow(row)  # 写入
    log_step(logger, '%s 频域传递函数已保存: %s', odb_basename, h_csv)  # 记录

    # —— 可选写出相位文件 ——
    if EXPORT_PHASE and ref_spec is not None:  # 仅谱比模式输出相位
        ph_csv = 'H-PHASE-{0}.csv'.format(csv_stem)  # 相位文件名
        Hh_c_band = Hh_c[:, band]  # 带内复数（水平）
        Hv_c_band = Hv_c[:, band]  # 带内复数（竖直）
        with open(ph_csv, 'w') as f_p:  # 打开相位文件
            writer = csv.writer(f_p, lineterminator='\n')  # CSV 写入器
            header = ['Freq_Hz']  # 表头
            for lb in top_labels:  # 水平相位列
                header.append('Node_{0}_phase_h'.format(lb))
            for lb in top_labels:  # 竖直相位列
                header.append('Node_{0}_phase_v'.format(lb))
            writer.writerow(header)  # 写表头
            for j in range(f_out.size):  # 逐频点
                row = ['{0:.6e}'.format(f_out[j])]  # 频率
                for i in range(n_top):  # 水平相位（弧度）
                    row.append(_fmt_val(np.angle(Hh_c_band[i, j])))
                for i in range(n_top):  # 竖直相位（弧度）
                    row.append(_fmt_val(np.angle(Hv_c_band[i, j])))
                writer.writerow(row)  # 写入
        log_step(logger, '%s 相位文件已保存: %s', odb_basename, ph_csv)  # 记录


def _fmt_val(v):
    """将数值格式化为科学计数字符串；NaN 写为空（便于 pandas 读为缺失值）。

    v: float
    返回: str
    """
    if v != v:  # NaN 判定（NaN != NaN）
        return ''  # 空字符串表示缺失
    return '{0:.6e}'.format(float(v))  # 科学计数格式


# ==============================================================================
#  主入口
# ==============================================================================
if __name__ == '__main__':
    logger = log_step('Postprocess_PGA.log')  # 初始化日志器
    total_start = time.time()  # 记录开始时间
    try:
        log_step(logger, '脚本开始执行 (v3: PGA + 频域传递函数 H(f,x))')  # 记录启动
        if not HAS_ODB:  # 不在 Abaqus 环境
            log_step(logger, '未检测到 odbAccess，本脚本需在 Abaqus python 环境运行')  # 提示
        if not HAS_NUMPY:  # 无 numpy
            log_step(logger, '未检测到 numpy，频域 H(f) 模块将被跳过')  # 提示

        cwd = os.getcwd()  # 当前工作目录
        odb_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.odb')])  # 目录下 ODB
        if not odb_files:  # 无 ODB
            log_step(logger, '当前目录 %s 下未找到 .odb 文件', cwd)
        else:
            log_step(logger, '共找到 %d 个 ODB 文件', len(odb_files))  # 记录数量

        for odb_file in odb_files:  # 遍历处理
            process_one_odb(odb_path=os.path.join(cwd, odb_file), logger=logger)  # 处理单个 ODB

        log_step(logger, '脚本执行完成，总耗时=%.2fs', time.time() - total_start)  # 记录完成
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))  # 记录失败
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 输出堆栈
        raise  # 重新抛出
