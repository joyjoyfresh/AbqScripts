# -*- coding: utf-8 -*-
"""坡地模型地表响应统一后处理 v2（PGA + AF/TAF + 复频响/FSAF + 5% PSA/RSAF + 三段重采样）。

v2 变更（相对 v1）：新增研究计划 §4.0 第②步"两步对齐"的重采样——把逐节点曲线与 H(f,s) 曲面
按三段（坡顶平台A/坡面B/坡脚平台C）插值到统一 s 子网格（N_A=120/N_B=60/N_C=80，段A 近坡顶棱加密），
输出固定长度 N_A+N_B+N_C 的对齐矩阵给 POD/ML；AR_max 仍在重采样前的原始曲线上精确提取，
并把其所在段号与归一坐标回写 surface_summary.json。

遍历当前工况目录所有 job-*.odb，从 TOP_SURFACE 节点集全时程场输出提取地表加速度，逐波输出：
  1. PGA：每个地表节点水平(A1)/竖向(A2)/合成(R)加速度峰值。
  2. 放大系数（口径与 case_meta 完全一致）：
       AF_h  = PGA_h / (factor_h × PGA_in)      —— 相对基岩入射的总放大（含场地+地形）
       TAF_h = AF_h / taf_h(同侧一维理论台阶)   —— 纯地形放大，远场应趋于 1
       AF_v  = PGA_v / (factor_h × PGA_in)      —— 寄生竖向放大（统一水平分母，B&P2005 口径）
       TAF_v = AF_v / taf_v(同侧)               —— 仅斜入射且 taf_v>TAFV_GUARD 时计算，否则 NaN
       VTR   = PGA_v / PGA_h_ff(同侧)            —— 竖向地形转换系数，0° 入射仍有意义
       UTAF_* = PGA_* / PGA_R_ff(同侧)           —— 合成自由场统一分母下的水平/竖向/合成响应
       DUTAF_v = (PGA_v - PGA_v_ff) / PGA_R_ff   —— 地形额外诱发的竖向响应增量
       V/H   = PGA_v / PGA_h                    —— 同点竖横比
     同侧规则：x ≤ x_toe（坡顶平台+坡面）用 left 柱，x > x_toe（坡脚平台）用 right 柱。
  3. 复频响与傅里叶谱放大：规范 NPZ 保存复数 H、相位所需信息和显式 valid_mask；
     FSAF=|H| 只作为确定性派生视图；真实一维参考齐全时另算 FSAF_1D。
     同侧端点分母另记 station_ratio，禁止误称匹配均质坡 H_topo；run_cfg.frf_cfg.fmax_hz 控制输出频带。
  4. 反应谱：计算 5% 阻尼弹性伪加速度谱 PSA；只有配置了同一记录的真实 rock/1D 参考时才计算
     RSAF_rock、RSAF_1D 和 URSAF_z，缺参考或分母过小时写 NaN+valid_mask，不以 epsilon 造峰。
  5. 数据状态：只记录 ODB 提取、数组生成和文件写出是否完成；研究质量评价由独立脚本执行。

输入：job-*.odb + case_meta.json + 输入波 txt（路径优先取 case_config.json 的 run_cfg.wave_files，
      按文件名主干与记录名匹配；缺省回退工况目录下同名 .txt）。
输出：surface_results.npz                                  （单工况唯一数值包：逐节点响应、传函、s 网格、元数据与汇总）
      figs/surface_response_<record>.{png,pdf,svg}         （三段分轴出版级图，横轴为 §4.0 三段归一坐标 s）
      figs/surface_response_raw_s_<record>.{png,pdf,svg}   （原始逐节点数据仅做 x→s 换算的对照图）
      （CSV/JSON 仅在运行期间作为重采样和绘图临时文件，打包完成后自动删除）
运行：abaqus python Postprocess_All_surface_v2.py   （在含 job-*.odb 与 case_meta.json 的工况目录内）
约定：Abaqus 自带 Python 2.7 + numpy；纯数值函数不依赖 odbAccess，可在普通 Python 下单测。
"""

from __future__ import print_function

import os
import sys
import glob
import csv
import json
import math
import io
import time
import logging
import traceback
import zipfile
from xml.sax.saxutils import escape as xml_escape
import numpy as np

openOdb = None  # 占位
is_abaqus = False  # 是否成功加载 Abaqus ODB 接口

try:
    import abaqusConstants  # Abaqus 常量模块
    from odbAccess import openOdb  # Abaqus ODB 接口
    is_abaqus = True  # 成功导入即按 Abaqus 环境处理
except Exception:
    openOdb = None  # 普通 Python 环境下保留降级路径

try:
    if hasattr(sys, 'setdefaultencoding'):  # 仅在 Python 2 下执行
        eval("reload(sys)")  # 用 eval 动态执行，避开 Python 3 静态分析对未定义 reload 的报错
        sys.setdefaultencoding('utf-8')  # 设置默认编码
except Exception:
    pass

SPEC_MASK_RATIO = 0.05   # 输入谱幅值掩码比例：低于峰值 5% 的频点剔除（0/0 噪声带）
F_LO = 0.3               # 传函可靠带下限(Hz)：更低频段脉冲能量太薄
FMAX_FACTOR = 2.5        # 未显式配置 frf_fmax_hz 时，兼容旧工况的上限 = FMAX_FACTOR×fc
PAD_FACTOR = 4           # FFT 补零倍数（防卷绕，与建模脚本 fd 引擎同口径）
TAFV_GUARD = 0.05        # taf_v 低于该值视为"竖向自由场≈0"（垂直入射），TAF_v 置 NaN 防除零
SAFE_DENOM_EPS = 1e-30   # 分母安全阈值：只用于判定无效分母，不用小量强行制造比值
RSA_DENOM_RATIO = 1e-8   # 反应谱分母相对峰值阈值：低于此值的周期点置为无效
RSA_DAMPING = 0.05       # 默认反应谱阻尼比
RSA_PERIOD_MIN = 0.10    # 默认反应谱最小周期(s)，PGA 不混入 T=0
RSA_PERIOD_MAX = 2.00    # 默认反应谱最大周期(s)
RSA_PERIOD_COUNT = 40    # 默认反应谱对数等距点数
RSA_KEY_PERIODS = (0.10, 0.20, 0.50, 1.00, 2.00)  # 论文单列的关键周期(s)


_DEFAULT_SCRIPT_NAME = 'Postprocess_All_surface_v2.py'  # __file__ 缺失时的兜底文件名


def _script_path():  # 安全获取当前脚本绝对路径（Abaqus 内核可能不定义 __file__）
    """返回脚本绝对路径；Abaqus 用 execfile/kernel 执行时全局可能无 __file__，此时退化为当前目录下的已知脚本名。"""
    f = globals().get('__file__')
    if f:  # __file__ 存在时
        return os.path.abspath(f)
    return os.path.join(os.getcwd(), _DEFAULT_SCRIPT_NAME)  # 兜底：当前工作目录(工况文件夹) + 已知脚本名


def _script_name():
    """返回脚本文件名，不依赖 __file__。"""
    return os.path.basename(_script_path())


def _script_dir():
    """返回脚本所在目录；__file__ 缺失时退化为当前工作目录。"""
    return os.path.dirname(_script_path())


try:
    _TEXT_TYPE = unicode
except NameError:
    _TEXT_TYPE = str


def _to_log_text(value):
    """把日志文本统一为Unicode，数值参数保持原类型。"""
    if isinstance(value, _TEXT_TYPE):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8', 'replace')
    return value


def _format_log_text(message, args):
    """在进入logging前完成Unicode格式化，兼容Abaqus Python 2.7。"""
    message = _to_log_text(message)
    if not args:
        return message
    normalized = tuple(_to_log_text(item) for item in args)
    try:
        return message % normalized
    except Exception:
        suffix = u' '.join(_TEXT_TYPE(item) for item in normalized)
        return u'{} {}'.format(message, suffix)


def log_step(logger=None, message=None, *args):
    """日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。"""
    if not hasattr(log_step, '_logger'):
        if logger is not None and isinstance(logger, str):
            log_filename = logger
            logger = None
        else:
            script_name = _script_name()  # 获取当前脚本名
            log_filename = os.path.splitext(script_name)[0] + '.log'  # 使用与脚本同名的日志文件名

        _logger = logging.getLogger('abqpy')
        _logger.setLevel(logging.INFO)
        _logger.propagate = False  # 禁止向父日志器传播

        _logger.handlers = []
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        log_step._logger = _logger
        log_step._start_time = time.time()
        log_step._log_filename = log_filename

        return _logger

    if message is not None:
        now = time.time()
        delta_total = now - log_step._start_time
        rendered = _format_log_text(message, args)
        log_step._logger.info(u'[%.3fs] %s' % (delta_total, rendered))

    return log_step._logger


# ==========================================================
#  纯数值函数（不依赖 odbAccess，可单测）
# ==========================================================


def strip_record_name(odb_name):  # job-<记录>-slope.odb -> <记录>
    """从 odb 文件名剥离 job- 前缀与 -slope/-flat 场景后缀，返回记录名。"""
    base = os.path.basename(odb_name)
    if base.lower().endswith('.odb'):
        base = base[:-4]
    if base.lower().startswith('job-'):
        base = base[4:]
    for suf in ('-slope', '-flat'):  # 场景后缀（build_models 命名规则）
        if base.lower().endswith(suf):
            base = base[:-len(suf)]
            break
    return base


def to_uniform(t, sig_mat):
    """把帧时刻 t 与信号矩阵 sig_mat(节点×时刻) 重采样到等间隔时间轴。

    FIXED 增量下帧本就等距（原样返回）；CDP 非线性自动增量下帧距不等，
    用中位数 dt 建均匀轴并逐节点线性插值（FFT 前提是等间隔采样）。
    返回 (t_uniform, sig_uniform, dt)。
    """
    t = np.asarray(t, dtype=float)
    sig_mat = np.atleast_2d(np.asarray(sig_mat, dtype=float))
    if sig_mat.shape[1] != t.size:  # 时程列数必须与原始帧时间轴一致
        raise ValueError('时程列数(%d)与帧时间轴长度(%d)不一致，无法重采样' % (sig_mat.shape[1], t.size))
    dts = np.diff(t)
    dt = float(np.median(dts))
    if dt <= 0:
        raise ValueError('帧时间步 dt<=0，无法重采样')
    if float(np.max(dts) - np.min(dts)) > 1e-6 * dt:  # 帧距不均匀（自动增量）
        tu = np.arange(0.0, float(t[-1]) + 0.5 * dt, dt)  # 均匀时间轴
        out = np.vstack([np.interp(tu, t, sig_mat[k]) for k in range(sig_mat.shape[0])])  # 逐节点线性插值
        return tu, out, dt
    return t, sig_mat, dt


def _padded_fft_length(n):  # 统一计算补零后的 FFT 长度
    """返回不小于 ``PAD_FACTOR*n`` 的最小 2 的幂。"""
    nfft = 1
    while nfft < PAD_FACTOR * int(n):
        nfft *= 2
    return nfft


def resolve_frf_fmax_hz(fc=None, fmax_hz=None):
    """解析复频响频带上限，显式上限优先于阻尼主频派生值。"""
    if fmax_hz is not None:
        value = float(fmax_hz)
        if not np.isfinite(value) or value <= F_LO:
            raise ValueError('frf_fmax_hz 必须大于 %.3f Hz' % F_LO)
        return value
    if fc is not None and float(fc) > 0.0:
        return FMAX_FACTOR * float(fc)
    return None


def compute_complex_H(a_out_mat, a_ref, dt, fc=None, fmax_hz=None):
    """计算候选物理频带上的复频响并显式返回输入谱有效掩码。

    返回 ``(freqs, H_complex, valid_mask, A_ref)``。候选轴先按 ``F_LO`` 和可选频带上限裁剪，
    再按候选带内参考谱峰值执行幅值掩码；无效频点写复 NaN。这样既保留显式无效点，
    又避免把整个 Nyquist 高频区的 NaN 矩阵写入规范数据包。``fmax_hz`` 用于把论文频带
    与材料阻尼主频 ``fc`` 解耦；未显式提供时保留旧的 ``FMAX_FACTOR*fc`` 行为。
    """
    a_out_mat = np.atleast_2d(np.asarray(a_out_mat, dtype=float))
    a_ref = np.asarray(a_ref, dtype=float).reshape(-1)
    if a_out_mat.shape[1] < 2 or a_ref.size < 2:
        raise ValueError('复频响至少需要 2 个时程点')
    if not np.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError('复频响采样步长 dt 必须为正数')
    n = max(a_out_mat.shape[1], a_ref.size)  # 以较长者定 FFT 基准长度
    nfft = _padded_fft_length(n)  # 补零至 4 倍长度的 2 的幂（防卷绕 + 加密频轴）
    A_ref_full = np.fft.rfft(a_ref, n=nfft)  # 参考单边谱
    A_out_full = np.fft.rfft(a_out_mat, n=nfft, axis=1)  # 各节点单边谱
    freqs_full = np.fft.rfftfreq(nfft, dt)  # 完整频率轴
    candidate = freqs_full >= F_LO
    effective_fmax = resolve_frf_fmax_hz(fc=fc, fmax_hz=fmax_hz)
    if effective_fmax is not None:
        candidate &= freqs_full <= effective_fmax
    freqs = freqs_full[candidate]
    A_ref = A_ref_full[candidate]
    A_out = A_out_full[:, candidate]
    del A_ref_full, A_out_full, freqs_full  # 尽早释放完整 Nyquist 谱，降低 Abaqus 内核峰值内存
    ref_abs = np.abs(A_ref)
    ref_peak = float(np.max(ref_abs)) if ref_abs.size else 0.0
    keep = np.zeros(freqs.shape, dtype=bool)
    if np.isfinite(ref_peak) and ref_peak > SAFE_DENOM_EPS:
        keep = ref_abs >= SPEC_MASK_RATIO * ref_peak  # 参考没能量的频点不得相除
    H_complex = np.empty(A_out.shape, dtype=np.complex128)
    H_complex[:] = complex(float('nan'), float('nan'))
    if np.any(keep):
        H_complex[:, keep] = A_out[:, keep] / A_ref[keep][None, :]
    return freqs, H_complex, keep, A_ref


def compute_H(a_out_mat, a_in, dt, fc=None, fmax_hz=None):
    """兼容旧调用的 FSAF 视图：返回可靠频点及 ``|H_complex|``。"""
    freqs, H_complex, valid_mask, _unused_ref = compute_complex_H(
        a_out_mat, a_in, dt, fc=fc, fmax_hz=fmax_hz)
    return freqs[valid_mask], np.abs(H_complex[:, valid_mask])


def spectral_ratio(a_out_mat, a_ref, dt, fc=None):
    """兼容旧调用的参考台站幅值谱比，不能解释为匹配均质坡地形传函。"""
    return compute_H(a_out_mat, a_ref, dt, fc=fc)


def _safe_array_ratio(num, den, threshold_ratio=RSA_DENOM_RATIO):
    """按真实分母阈值计算数组比值，返回 ``(ratio, valid_mask)``。

    ``den`` 可按 NumPy 广播到 ``num``；阈值相对分母有限值峰值确定。无效位置使用 NaN，
    临时安全分母只用于避免运行期除零，绝不作为结果中的 epsilon 分母。
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    finite_den = np.isfinite(den)
    peak = float(np.max(np.abs(den[finite_den]))) if np.any(finite_den) else 0.0
    threshold = max(SAFE_DENOM_EPS, float(threshold_ratio) * peak)
    den_valid = np.zeros(den.shape, dtype=bool)
    if np.any(finite_den):
        den_valid[finite_den] = np.abs(den[finite_den]) > threshold  # 严格浮点模式下不对 NaN 做 abs/比较
    safe_den = np.where(den_valid, den, 1.0)
    finite_num = np.isfinite(num)
    safe_num = np.where(finite_num, num, 0.0)
    ratio = safe_num / safe_den
    valid = finite_num & den_valid
    ratio = np.where(valid, ratio, np.nan)
    return ratio, valid


def response_spectrum_periods(case_cfg=None):
    """读取反应谱配置并返回严格递增的正周期数组和阻尼比。

    配置位于 ``run_cfg.response_spectrum_cfg``；可显式给 ``periods``，否则使用
    ``period_min/period_max/period_count`` 生成对数等距网格。
    """
    run_cfg = (case_cfg or {}).get('run_cfg') or {}
    cfg = run_cfg.get('response_spectrum_cfg') or {}
    damping = float(cfg.get('damping_ratio', RSA_DAMPING))
    if not (0.0 <= damping < 1.0):
        raise ValueError('反应谱阻尼比必须满足 0<=damping_ratio<1')
    explicit = cfg.get('periods')
    if explicit is not None:
        periods = np.asarray(explicit, dtype=float).reshape(-1)
    else:
        p_min = float(cfg.get('period_min', RSA_PERIOD_MIN))
        p_max = float(cfg.get('period_max', RSA_PERIOD_MAX))
        count = int(cfg.get('period_count', RSA_PERIOD_COUNT))
        if p_min <= 0.0 or p_max <= p_min or count < 2:
            raise ValueError('反应谱周期配置必须满足 0<period_min<period_max 且 period_count>=2')
        periods = np.exp(np.linspace(math.log(p_min), math.log(p_max), count))
    periods = np.unique(periods[np.isfinite(periods) & (periods > 0.0)])
    if periods.size == 0:
        raise ValueError('反应谱周期数组没有有效正值')
    return periods, damping


def compute_psa(acc_mat, dt, periods, damping=RSA_DAMPING):
    """用 Newmark 平均加速度法计算弹性单自由度伪加速度反应谱。

    ``acc_mat`` 为 ``记录/节点×时刻``，返回 ``记录/节点×周期``；单位与输入加速度一致。
    算法采用 ``beta=1/4``、``gamma=1/2``，不混入 ``T=0`` 的 PGA。
    """
    acc_mat = np.atleast_2d(np.asarray(acc_mat, dtype=float))
    periods = np.asarray(periods, dtype=float).reshape(-1)
    dt = float(dt)
    damping = float(damping)
    if acc_mat.shape[1] < 2 or dt <= 0.0:
        raise ValueError('反应谱至少需要 2 个时程点且 dt>0')
    if periods.size == 0 or np.any(~np.isfinite(periods)) or np.any(periods <= 0.0):
        raise ValueError('反应谱周期必须为有限正数')
    if not (0.0 <= damping < 1.0):
        raise ValueError('反应谱阻尼比必须满足 0<=damping<1')
    beta = 0.25
    gamma = 0.5
    omega = (2.0 * math.pi / periods)[None, :]
    stiffness = omega * omega
    viscous = 2.0 * damping * omega
    denom = 1.0 + gamma * dt * viscous + beta * dt * dt * stiffness
    shape = (acc_mat.shape[0], periods.size)
    disp = np.zeros(shape, dtype=float)
    vel = np.zeros(shape, dtype=float)
    rel_acc = -acc_mat[:, 0][:, None] * np.ones((1, periods.size), dtype=float)
    max_abs_disp = np.abs(disp)
    for index in range(1, acc_mat.shape[1]):
        disp_pred = disp + dt * vel + dt * dt * (0.5 - beta) * rel_acc
        vel_pred = vel + dt * (1.0 - gamma) * rel_acc
        rel_acc = (-acc_mat[:, index][:, None] - viscous * vel_pred - stiffness * disp_pred) / denom
        disp = disp_pred + beta * dt * dt * rel_acc
        vel = vel_pred + gamma * dt * rel_acc
        max_abs_disp = np.maximum(max_abs_disp, np.abs(disp))
    return stiffness * max_abs_disp


def compute_side_reference_H(acc_h, xs, x_toe, ref_left, ref_right, dt, freqs,
                             excitation_valid, fc=None, fmax_hz=None):
    """计算相对同侧真实一维参考时程的复谱比及二维有效掩码。

    结果仅在入射波与对应一维参考谱同时有效时保留；缺少任一侧参考时，该侧保持复NaN。
    """
    acc_h = np.atleast_2d(np.asarray(acc_h, dtype=float))
    xs = np.asarray(xs, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    excitation_valid = np.asarray(excitation_valid, dtype=bool)
    transfer = np.empty((acc_h.shape[0], freqs.size), dtype=np.complex128)
    transfer[:] = complex(float('nan'), float('nan'))
    valid = np.zeros(transfer.shape, dtype=bool)
    left_mask = xs <= (float(x_toe) + 1e-4)
    for node_mask, reference, side in ((left_mask, ref_left, 'left'), (~left_mask, ref_right, 'right')):
        if reference is None or not np.any(node_mask):
            continue
        side_freqs, side_transfer, side_valid, _unused = compute_complex_H(
            acc_h[node_mask], reference, dt, fc=fc, fmax_hz=fmax_hz)
        if side_freqs.shape != freqs.shape or not np.allclose(side_freqs, freqs):
            raise RuntimeError('%s 一维参考谱比频轴与入射频响不一致' % side)
        shared = excitation_valid & side_valid
        for local_index, global_index in enumerate(np.where(node_mask)[0]):
            transfer[global_index, shared] = side_transfer[local_index, shared]
            valid[global_index, shared] = True
    return transfer, valid


def compute_uniform_reference_H(acc_h, reference, dt, freqs, excitation_valid,
                                fc=None, fmax_hz=None):
    """计算全地表相对同一个一维参考时程的复谱比。"""
    acc_h = np.atleast_2d(np.asarray(acc_h, dtype=float))
    freqs = np.asarray(freqs, dtype=float)
    excitation_valid = np.asarray(excitation_valid, dtype=bool)
    transfer = np.empty((acc_h.shape[0], freqs.size), dtype=np.complex128)
    transfer[:] = complex(float('nan'), float('nan'))
    valid = np.zeros(transfer.shape, dtype=bool)
    if reference is None:
        return transfer, valid
    ref_freqs, ref_transfer, ref_valid, _unused = compute_complex_H(
        acc_h, reference, dt, fc=fc, fmax_hz=fmax_hz)
    if ref_freqs.shape != freqs.shape or not np.allclose(ref_freqs, freqs):
        raise RuntimeError('统一左侧一维参考谱比频轴与入射频响不一致')
    shared = excitation_valid & ref_valid
    transfer[:, shared] = ref_transfer[:, shared]
    valid[:, shared] = True
    return transfer, valid


def _safe_ratio(num, den):  # 安全除法
    """只在分母真实有效时计算比值；无效时返回 NaN，避免 epsilon 分母制造假峰值。"""
    if den is None:  # 分母缺失
        return float('nan')  # 返回无效
    try:  # 尝试转浮点
        den = float(den)  # 分母
        num = float(num)  # 分子
    except Exception:  # 非数值
        return float('nan')  # 返回无效
    if math.isnan(den) or abs(den) <= SAFE_DENOM_EPS:  # 分母为零或 NaN
        return float('nan')  # 返回无效
    return num / den  # 正常比值


def _abs_or_none(value):  # 取绝对值或 None
    """把可选参考系数转成非负峰值系数；缺失时返回 None。"""
    if value is None:  # 缺失
        return None  # 返回空
    try:  # 尝试转换
        return abs(float(value))  # PGA 参考值使用幅值
    except Exception:  # 转换失败
        return None  # 返回空


def surface_metrics(xs, a1_mat, a2_mat, pga_in, factor_h, factor_v, taf_lr, x_toe):
    """由地表加速度矩阵计算逐节点 PGA/AF/TAF/V_H 指标。

    xs       : 节点 x 坐标（升序）；a1_mat/a2_mat：节点×时刻 水平/竖向加速度。
    pga_in   : 输入记录峰值；factor_h：斜入射自由面水平放大系数（分母 = factor_h×pga_in）。
    factor_v : 斜入射自由面竖向系数（仅作缺省竖向自由场参考，PGA 取绝对值）。
    taf_lr   : {'left': (taf_h, taf_v), 'right': (taf_h, taf_v)}，一维理论台阶（可为 None）。
    x_toe    : 坡脚 x（同侧规则分界：x≤x_toe 用 left，x>x_toe 用 right）。
    返回逐节点 dict 列表；旧口径列保留，并新增统一分母、竖向转换和调试分母列。
    """
    pga_h = np.max(np.abs(a1_mat), axis=1)  # 逐节点水平峰值
    pga_v = np.max(np.abs(a2_mat), axis=1)  # 逐节点竖向峰值
    pga_r = np.max(np.sqrt(a1_mat * a1_mat + a2_mat * a2_mat), axis=1)  # 逐节点合成峰值
    denom_h0 = factor_h * pga_in if (factor_h and pga_in) else None  # 基准自由面水平分母
    fallback_taf_v = None  # 缺少一维理论 taf_v 时的半空间竖向参考
    if denom_h0 and factor_v is not None and factor_h:  # 可由半空间系数估算
        fallback_taf_v = abs(float(factor_v)) / abs(float(factor_h))  # 转成相对水平分母的竖向比例
    rows = []
    for k in range(len(xs)):
        side = 'left' if xs[k] <= x_toe else 'right'  # 同侧规则：坡顶平台+坡面归 left，坡脚平台归 right
        taf_h_ref, taf_v_ref = (taf_lr.get(side) or (None, None)) if taf_lr else (None, None)
        taf_h_abs = _abs_or_none(taf_h_ref)  # 同侧水平一维自由场比例
        taf_v_abs = _abs_or_none(taf_v_ref)  # 同侧竖向一维自由场比例
        if taf_v_abs is None:  # 元数据缺 taf_v
            taf_v_abs = fallback_taf_v  # 回退到半空间自由面系数
        ff_h = denom_h0 * taf_h_abs if (denom_h0 and taf_h_abs is not None) else float('nan')  # 同侧自由场水平 PGA
        ff_v = denom_h0 * taf_v_abs if (denom_h0 and taf_v_abs is not None) else float('nan')  # 同侧自由场竖向 PGA
        if not math.isnan(ff_h) and not math.isnan(ff_v):  # 两个分量参考均有效
            ff_r = math.sqrt(ff_h * ff_h + ff_v * ff_v)  # 同侧自由场合成 PGA 参考
        elif not math.isnan(ff_h):  # 只有水平参考
            ff_r = abs(ff_h)  # 垂直入射时退化为水平参考
        else:  # 无参考
            ff_r = float('nan')  # 合成参考无效
        af_h = _safe_ratio(pga_h[k], denom_h0)  # 现有口径：水平响应/基准水平自由面
        af_v = _safe_ratio(pga_v[k], denom_h0)  # 现有口径：竖向响应/基准水平自由面
        taf_h = _safe_ratio(pga_h[k], ff_h)  # 传统分量口径：水平响应/同侧水平自由场
        if taf_v_abs is not None and taf_v_abs > TAFV_GUARD:  # 竖向自由场足够大才可定义传统竖向 TAF
            taf_v = _safe_ratio(pga_v[k], ff_v)  # 传统分量口径：竖向响应/同侧竖向自由场
        else:  # 垂直入射或竖向自由场过小
            taf_v = float('nan')  # 不定义，避免病态分母
        vtr = _safe_ratio(pga_v[k], ff_h)  # 竖向转换系数：竖向响应/同侧水平自由场
        utaf_h = _safe_ratio(pga_h[k], ff_r)  # 统一分母水平响应系数
        utaf_v = _safe_ratio(pga_v[k], ff_r)  # 统一分母竖向响应系数
        utaf_r = _safe_ratio(pga_r[k], ff_r)  # 统一分母合成响应系数
        dutaf_v = _safe_ratio(float(pga_v[k]) - ff_v, ff_r) if not math.isnan(ff_v) else float('nan')  # 竖向地形增量系数
        rows.append({'x': float(xs[k]), 'PGA_h': float(pga_h[k]), 'PGA_v': float(pga_v[k]),
                     'PGA_R': float(pga_r[k]), 'AF_h': af_h, 'TAF_h': taf_h,
                     'AF_v': af_v, 'TAF_v': taf_v, 'TAF_h_comp': taf_h,
                     'TAF_v_comp': taf_v, 'VTR': vtr, 'UTAF_h': utaf_h,
                     'UTAF_v': utaf_v, 'UTAF_R': utaf_r, 'TAF_R': utaf_r,
                     'DUTAF_v': dutaf_v, 'FF_PGA_h0': denom_h0 if denom_h0 else float('nan'),
                     'FF_PGA_h': ff_h, 'FF_PGA_v': ff_v, 'FF_PGA_R': ff_r,
                     'V_over_H': float(pga_v[k]) / float(pga_h[k]) if pga_h[k] > 0 else float('nan'),
                     'ff_side': side})
    return rows


def _config_bool(value, default=None):  # 把配置中的布尔值或字符串统一转换为布尔值
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, _TEXT_TYPE):
        return value.strip().lower() in ('true', '1', 'yes', 'on', 'enable', 'enabled')
    return default


# ==========================================================
#  工况文件读取（meta / config / 输入波）
# ==========================================================


def _load_json(path):  # 读 json，缺失返回 None
    if not os.path.isfile(path):
        return None
    with io.open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def find_wave_file(record, case_cfg):
    """按记录名定位输入波 txt：优先 case_config 的 run_cfg.wave_files（文件名主干匹配），回退目录同名 txt。"""
    cands = []
    wf = ((case_cfg or {}).get('run_cfg') or {}).get('wave_files') or []  # 注入的波形路径列表
    if not isinstance(wf, (list, tuple)):
        wf = [wf]
    cands.extend([str(p) for p in wf])
    cands.extend(sorted(glob.glob('*.txt')))  # 工况目录内 txt 兜底
    for p in cands:
        if os.path.splitext(os.path.basename(p))[0] == record and os.path.isfile(p):  # 主干与记录名一致
            return p
    exist = [p for p in cands if os.path.isfile(p)]  # 无精确匹配时
    return exist[0] if len(exist) == 1 else None  # 仅剩单一候选才敢兜底


def _response_spectrum_cfg(case_cfg):  # 读取嵌套反应谱配置
    """返回 ``run_cfg.response_spectrum_cfg``，避免增加生产配置顶层键。"""
    return (((case_cfg or {}).get('run_cfg') or {}).get('response_spectrum_cfg') or {})


def _frf_config(case_cfg):  # 读取复频响数据处理配置
    """返回 ``run_cfg.frf_cfg``，这里只放频带等数据处理参数。"""
    run_cfg = (case_cfg or {}).get('run_cfg') or {}
    return dict(run_cfg.get('frf_cfg') or {})


def _response_reference_spec(record, case_cfg):  # 读取逐记录真实参考文件配置
    """返回逐记录参考文件字典。

    规范键为 ``rock``、``one_d_left``、``one_d_right``；``one_d`` 可表示左右共用同一参考。
    配置位于 ``run_cfg.response_spectrum_cfg.reference_files.<record>``，并支持 ``default`` 兜底。
    """
    refs = _response_spectrum_cfg(case_cfg).get('reference_files') or {}
    if not isinstance(refs, dict):
        refs = {}
    spec = refs.get(record)
    if spec is None:
        spec = refs.get('default')
    spec = dict(spec) if isinstance(spec, dict) else {}
    auto_path = os.path.abspath('freefield_reference_%s.npz' % record)  # 建模脚本生成的同口径参考
    if os.path.isfile(auto_path):
        automatic = {
            'rock': {'path': auto_path, 'key': 'rock_acc_h'},
            'one_d_left': {'path': auto_path, 'key': 'one_d_left_acc_h'},
            'one_d_right': {'path': auto_path, 'key': 'one_d_right_acc_h'},
        }
        for name, value in automatic.items():
            if not spec.get(name):
                spec[name] = value
    return spec


def _load_reference_series(source_spec, target_time):  # 载入并对齐真实自由场参考时程
    """读取两列文本或NPZ参考时程并插值到 ODB 时间轴，返回时程与加载信息。"""
    if not source_spec:
        return None, {'status': 'not_configured'}
    key = None
    if isinstance(source_spec, dict):
        key = source_spec.get('key')
        path = source_spec.get('path')
    else:
        path = source_spec
    if not path:
        return None, {'status': 'not_configured'}
    source = os.path.abspath(str(path)) if not os.path.isabs(str(path)) else str(path)
    if not os.path.isfile(source):
        return None, {'status': 'missing', 'path': source}
    try:
        if source.lower().endswith('.npz'):
            package = np.load(source)
            if not key or key not in package or 'time' not in package:
                raise ValueError('NPZ参考文件缺少 time 或指定时程键 %s' % key)
            ref_time = np.asarray(package['time'], dtype=float)
            ref_acc = np.asarray(package[key], dtype=float)
            package.close()
        else:
            table = np.asarray(np.loadtxt(source), dtype=float)
            if table.ndim != 2 or table.shape[0] < 2 or table.shape[1] < 2:
                raise ValueError('参考文件必须至少包含两行两列 time, acceleration')
            ref_time = table[:, 0]
            ref_acc = table[:, 1]
        if np.any(~np.isfinite(ref_time)) or np.any(~np.isfinite(ref_acc)) or np.any(np.diff(ref_time) <= 0.0):
            raise ValueError('参考时程包含非有限值或时间列不严格递增')
        target_time = np.asarray(target_time, dtype=float)
        aligned = np.interp(target_time, ref_time, ref_acc, left=0.0, right=0.0)
        covered = (target_time >= ref_time[0]) & (target_time <= ref_time[-1])
        return aligned, {'status': 'loaded', 'path': source, 'key': key,
                         'source_time_range': [float(ref_time[0]), float(ref_time[-1])],
                         'target_coverage_fraction': float(np.mean(covered))}
    except Exception as exc:
        return None, {'status': 'invalid', 'path': source, 'error': str(exc)}


def load_reference_payload(record, time_axis, case_cfg):  # 加载复频响与反应谱共用的一维参考
    """返回左右一维与基岩参考时程及其加载信息，不执行反应谱计算。"""
    spec = _response_reference_spec(record, case_cfg)
    common_1d = spec.get('one_d')
    paths = {
        'rock': spec.get('rock'),
        'one_d_left': spec.get('one_d_left') or common_1d,
        'one_d_right': spec.get('one_d_right') or common_1d,
    }
    references = {}
    reference_info = {}
    for name in ('rock', 'one_d_left', 'one_d_right'):
        series, load_info = _load_reference_series(paths.get(name), time_axis)
        references[name] = series
        reference_info[name] = load_info
    return {'series': references, 'info': reference_info}


def compute_response_spectrum_payload(record, xs, x_toe, time_axis, acc_h, acc_v, dt, case_cfg,
                                      reference_payload=None):
    """计算单条记录的 PSA、RSAF/URSAF、真实参考时程及处理元数据。"""
    cfg = _response_spectrum_cfg(case_cfg)
    if _config_bool(cfg.get('enable', True), True) is False:
        return None
    periods, damping = response_spectrum_periods(case_cfg)
    key_periods = np.asarray(cfg.get('key_periods', RSA_KEY_PERIODS), dtype=float).reshape(-1)
    key_periods = np.unique(key_periods[np.isfinite(key_periods) & (key_periods > 0.0)])
    if key_periods.size == 0:
        raise ValueError('反应谱 key_periods 没有有效正值')
    denominator_ratio = float(cfg.get('denominator_ratio', RSA_DENOM_RATIO))
    if denominator_ratio < 0.0:
        raise ValueError('反应谱 denominator_ratio 不得为负数')
    xs = np.asarray(xs, dtype=float)
    all_periods = np.unique(np.concatenate([periods, key_periods]))  # 合并计算，避免重复遍历长时程
    period_indices = np.asarray([int(np.argmin(np.abs(all_periods - value))) for value in periods], dtype=int)
    key_indices = np.asarray([int(np.argmin(np.abs(all_periods - value))) for value in key_periods], dtype=int)
    all_psa_h = compute_psa(acc_h, dt, all_periods, damping=damping)
    all_psa_v = compute_psa(acc_v, dt, all_periods, damping=damping)
    psa_h, psa_v = all_psa_h[:, period_indices], all_psa_v[:, period_indices]
    key_psa_h, key_psa_v = all_psa_h[:, key_indices], all_psa_v[:, key_indices]
    nan_ref = np.nan * np.ones(periods.shape, dtype=float)
    nan_key_ref = np.nan * np.ones(key_periods.shape, dtype=float)
    nan_surface = np.nan * np.ones(psa_h.shape, dtype=float)
    nan_key_surface = np.nan * np.ones(key_psa_h.shape, dtype=float)
    false_surface = np.zeros(psa_h.shape, dtype=bool)
    false_key_surface = np.zeros(key_psa_h.shape, dtype=bool)

    reference_payload = reference_payload or load_reference_payload(record, time_axis, case_cfg)
    references = reference_payload.get('series') or {}
    reference_info = reference_payload.get('info') or {}
    reference_psa = {}
    reference_key_psa = {}
    for name in ('rock', 'one_d_left', 'one_d_right'):
        series = references.get(name)
        if series is not None:
            all_reference_psa = compute_psa(series, dt, all_periods, damping=damping)[0]
            reference_psa[name] = all_reference_psa[period_indices]
            reference_key_psa[name] = all_reference_psa[key_indices]
        else:
            reference_psa[name] = nan_ref.copy()
            reference_key_psa[name] = nan_key_ref.copy()

    if references['rock'] is not None:
        rsaf_rock_h, rsaf_rock_valid = _safe_array_ratio(psa_h, reference_psa['rock'][None, :], denominator_ratio)
        ursaf_z, ursaf_z_valid = _safe_array_ratio(psa_v, reference_psa['rock'][None, :], denominator_ratio)
    else:
        rsaf_rock_h, rsaf_rock_valid = nan_surface.copy(), false_surface.copy()
        ursaf_z, ursaf_z_valid = nan_surface.copy(), false_surface.copy()
    if references['rock'] is not None:
        key_rsaf_rock_h, key_rsaf_rock_valid = _safe_array_ratio(
            key_psa_h, reference_key_psa['rock'][None, :], denominator_ratio)
        key_ursaf_z, key_ursaf_z_valid = _safe_array_ratio(
            key_psa_v, reference_key_psa['rock'][None, :], denominator_ratio)
    else:
        key_rsaf_rock_h, key_rsaf_rock_valid = nan_key_surface.copy(), false_key_surface.copy()
        key_ursaf_z, key_ursaf_z_valid = nan_key_surface.copy(), false_key_surface.copy()

    left_mask = xs <= float(x_toe)
    side_reference = np.nan * np.ones(psa_h.shape, dtype=float)
    if references['one_d_left'] is not None:
        side_reference[left_mask, :] = reference_psa['one_d_left'][None, :]
    if references['one_d_right'] is not None and np.any(~left_mask):
        side_reference[~left_mask, :] = reference_psa['one_d_right'][None, :]
    rsaf_1d_h, rsaf_1d_valid = _safe_array_ratio(psa_h, side_reference, denominator_ratio)
    key_side_reference = np.nan * np.ones(key_psa_h.shape, dtype=float)
    if references['one_d_left'] is not None:
        key_side_reference[left_mask, :] = reference_key_psa['one_d_left'][None, :]
    if references['one_d_right'] is not None and np.any(~left_mask):
        key_side_reference[~left_mask, :] = reference_key_psa['one_d_right'][None, :]
    key_rsaf_1d_h, key_rsaf_1d_valid = _safe_array_ratio(key_psa_h, key_side_reference, denominator_ratio)

    metadata = {
        'record': record,
        'method': 'Newmark_average_acceleration_beta_0.25_gamma_0.5_PSA',
        'damping_ratio': damping,
        'period_count': int(periods.size),
        'period_range_s': [float(periods[0]), float(periods[-1])],
        'key_periods_s': [float(value) for value in key_periods],
        'denominator_ratio': denominator_ratio,
        'reference_type': 'exact_time_history_files_only',
        'reference_info': reference_info,
        'reference_available': dict((name, references.get(name) is not None)
                                    for name in ('rock', 'one_d_left', 'one_d_right')),
        'note': '缺少真实参考时不以 factor_h*input 或 epsilon 分母代替',
    }
    return {
        'period': periods, 'damping_ratio': np.asarray(damping), 'x': xs,
        'key_period': key_periods,
        'PSA_surface_h': psa_h, 'PSA_surface_v': psa_v,
        'PSA_rock_h': reference_psa['rock'],
        'PSA_1D_left_h': reference_psa['one_d_left'],
        'PSA_1D_right_h': reference_psa['one_d_right'],
        'RSAF_rock_h': rsaf_rock_h, 'RSAF_rock_valid_mask': rsaf_rock_valid,
        'RSAF_1D_h': rsaf_1d_h, 'RSAF_1D_valid_mask': rsaf_1d_valid,
        'URSAF_z': ursaf_z, 'URSAF_z_valid_mask': ursaf_z_valid,
        'key_PSA_surface_h': key_psa_h, 'key_PSA_surface_v': key_psa_v,
        'key_PSA_rock_h': reference_key_psa['rock'],
        'key_PSA_1D_left_h': reference_key_psa['one_d_left'],
        'key_PSA_1D_right_h': reference_key_psa['one_d_right'],
        'key_RSAF_rock_h': key_rsaf_rock_h, 'key_RSAF_rock_valid_mask': key_rsaf_rock_valid,
        'key_RSAF_1D_h': key_rsaf_1d_h, 'key_RSAF_1D_valid_mask': key_rsaf_1d_valid,
        'key_URSAF_z': key_ursaf_z, 'key_URSAF_z_valid_mask': key_ursaf_z_valid,
        'reference_rock_acc_h': references['rock'],
        'reference_1D_left_acc_h': references['one_d_left'],
        'reference_1D_right_acc_h': references['one_d_right'],
        'metadata_json': json.dumps(metadata, ensure_ascii=True, sort_keys=True),
        'metadata': metadata,
    }


def meta_pieces(meta):
    """从 case_meta.json 提取后处理所需字段：(factor_h, factor_v, taf_lr, x_toe, fc)。"""
    norm = (meta or {}).get('ff_normalization') or {}
    factor_h = norm.get('factor_h')  # 斜入射自由面水平放大系数（0°时=2）
    factor_v = norm.get('factor_v')  # 斜入射自由面竖向系数（0°时通常为0）
    ff = (meta or {}).get('ff_theory') or {}
    taf_lr = {}
    for side in ('left', 'right'):  # 左(上平台)/右(下平台)一维理论台阶
        blk = ff.get(side) or {}
        taf_lr[side] = (blk.get('taf_h'), blk.get('taf_v'))
    geo = (meta or {}).get('geometry') or {}
    x_toe = geo.get('x_toe')  # 坡脚 x（同侧规则分界）
    fc = ff.get('fc_used') or ((meta or {}).get('damping') or {}).get('fc')  # 主频（可靠带上限用）
    return factor_h, factor_v, taf_lr, x_toe, fc


# ==========================================================
#  ODB 提取
# ==========================================================


def _surface_nodeset(odb):  # 定位含 TOP_SURFACE 节点集的实例
    for iname in odb.rootAssembly.instances.keys():
        inst = odb.rootAssembly.instances[iname]
        try:
            if 'TOP_SURFACE' in inst.nodeSets.keys():
                return inst.nodeSets['TOP_SURFACE']
        except Exception:
            pass
    raise RuntimeError('ODB 中未找到 TOP_SURFACE 节点集（建模需 surface_only 输出）')


def extract_surface_acc(odb):
    """从 ODB 提取 TOP_SURFACE 全时程加速度，按 x 升序返回 (xs, ys, t, a1_mat, a2_mat)。"""
    nset = _surface_nodeset(odb)
    labels = np.array([n.label for n in nset.nodes], dtype=int)
    xs = np.array([n.coordinates[0] for n in nset.nodes], dtype=float)
    ys = np.array([n.coordinates[1] for n in nset.nodes], dtype=float)
    order = np.argsort(xs)  # 地表节点按 x 升序（坡顶→坡脚）
    labels, xs, ys = labels[order], xs[order], ys[order]
    step = odb.steps[odb.steps.keys()[-1]]  # 最后一个分析步（地震步）
    times, rows1, rows2 = [], [], []
    for frame in step.frames:  # 逐帧提取（帧数=增量步数，几千帧量级）
        fo = frame.fieldOutputs['A'].getSubset(region=nset)  # 地表节点子集加速度场
        try:  # bulkDataBlocks 批量接口（快）
            data = np.concatenate([np.asarray(b.data, dtype=float) for b in fo.bulkDataBlocks])
            labs = np.concatenate([np.asarray(b.nodeLabels, dtype=int) for b in fo.bulkDataBlocks])
        except Exception:  # 逐值接口兜底（慢但稳）
            data = np.array([[v.data[0], v.data[1]] for v in fo.values], dtype=float)
            labs = np.array([v.nodeLabel for v in fo.values], dtype=int)
        pos = dict(zip([int(L) for L in labs], range(len(labs))))  # 标签→行号映射（逐帧重建，防顺序漂移）
        idx = np.array([pos[int(L)] for L in labels])
        times.append(float(frame.frameValue))
        rows1.append(data[idx, 0])  # A1 水平
        rows2.append(data[idx, 1])  # A2 竖向
    a1_mat = np.array(rows1, dtype=float).T  # 转成 节点×时刻
    a2_mat = np.array(rows2, dtype=float).T
    return xs, ys, np.array(times, dtype=float), a1_mat, a2_mat


def extract_validation_underground_acc(odb):  # 提取地下验证点的双分量时程
    """返回全部 ``VALIDATION_UNDERGROUND_*`` 点的坐标和时程；旧 ODB 缺少集合时返回空字典。"""
    def first_node(node_set):  # 兼容装配级节点集返回的嵌套 OdbMeshNodeArray
        node = node_set.nodes
        while not hasattr(node, 'coordinates'):
            if len(node) <= 0:
                raise RuntimeError('地下验证点集合为空')
            node = node[0]
        return node

    point_sets = []
    assembly_sets = getattr(odb.rootAssembly, 'nodeSets', {})
    point_names = sorted([str(name) for name in assembly_sets.keys()
                          if str(name).startswith('VALIDATION_UNDERGROUND_')])
    for name in point_names:
        if name in assembly_sets:
            point_sets.append((name, assembly_sets[name]))
    for iname in odb.rootAssembly.instances.keys():
        inst = odb.rootAssembly.instances[iname]
        inst_names = sorted([str(name) for name in inst.nodeSets.keys()
                             if str(name).startswith('VALIDATION_UNDERGROUND_')])
        for name in inst_names:
            if name not in [item[0] for item in point_sets] and name in inst.nodeSets.keys():
                point_sets.append((name, inst.nodeSets[name]))
    if not point_sets:
        return {}
    step = odb.steps[odb.steps.keys()[-1]]
    times, rows_h, rows_v, xs, ys = [], [], [], [], []
    for name, nset in sorted(point_sets):
        node = first_node(nset)
        xs.append(float(node.coordinates[0]))
        ys.append(float(node.coordinates[1]))
    for frame in step.frames:
        fo = frame.fieldOutputs['A']
        frame_h, frame_v = [], []
        for _name, nset in sorted(point_sets):
            values = fo.getSubset(region=nset).values
            if not values:
                raise RuntimeError('地下验证点集合无 A 输出值')
            frame_h.append(float(values[0].data[0]))
            frame_v.append(float(values[0].data[1]))
        times.append(float(frame.frameValue))
        rows_h.append(frame_h)
        rows_v.append(frame_v)
    return {'time': np.asarray(times, dtype=float),
            'name': np.asarray([name for name, _nset in sorted(point_sets)]),
            'x': np.asarray(xs, dtype=float),
            'y': np.asarray(ys, dtype=float), 'acc_h': np.asarray(rows_h, dtype=float).T,
            'acc_v': np.asarray(rows_v, dtype=float).T}


ENERGY_HISTORY_VARIABLES = ('ALLAE', 'ALLIE', 'ALLKE', 'ALLVD', 'ALLWK', 'ALLPD', 'ALLSE', 'ETOTAL')  # freefield 能量历史变量


def extract_energy_history(odb):  # 从 ODB 整体历史输出提取能量时间序列
    """返回 {'time': 时间轴, 'values': {变量: 序列}}；缺少历史输出时返回空字典。"""
    step = odb.steps[odb.steps.keys()[-1]]  # 读取最后一个动力分析步
    selected = None
    for region_name in step.historyRegions.keys():  # 遍历 Assembly/节点历史区域
        region = step.historyRegions[region_name]
        available = set(region.historyOutputs.keys())
        if 'ALLIE' in available and ('ALLWK' in available or 'ETOTAL' in available):  # 优先整体能量区域
            selected = region
            break
    if selected is None:
        return {}
    values = {}
    reference_time = None
    for variable in ENERGY_HISTORY_VARIABLES:
        if variable not in selected.historyOutputs:
            continue
        data = np.asarray(selected.historyOutputs[variable].data, dtype=float)
        if data.ndim != 2 or data.shape[1] < 2 or data.shape[0] < 2:
            continue
        times = data[:, 0]
        series = data[:, 1]
        if reference_time is None:
            reference_time = times
            values[variable] = series
        elif times.shape == reference_time.shape and np.allclose(times, reference_time):
            values[variable] = series
        else:
            values[variable] = np.interp(reference_time, times, series)
    if reference_time is None:
        return {}
    return {'time': np.asarray(reference_time, dtype=float), 'values': values}


# ==========================================================
#  输出
# ==========================================================


def _open_csv(path):  # Py2/Py3 兼容的 csv 写句柄
    if sys.version_info[0] >= 3:
        return open(path, 'w', newline='')
    return open(path, 'wb')


RESPONSE_FIELDS = [  # 地表响应表固定列顺序
    'x', 'y', 'PGA_h', 'PGA_v', 'PGA_R',
    'AF_h', 'AF_v', 'TAF_h', 'TAF_v',
    'TAF_h_comp', 'TAF_v_comp', 'VTR',
    'UTAF_h', 'UTAF_v', 'UTAF_R', 'TAF_R', 'DUTAF_v',
    'V_over_H', 'FF_PGA_h0', 'FF_PGA_h', 'FF_PGA_v', 'FF_PGA_R',
    'ff_side'
]


def write_response_csv(path, ys, rows):
    """写逐节点指标 csv：包含现有 AF/TAF 与新增统一分母、竖向转换和调试分母列。"""
    fh = _open_csv(path)
    w = csv.writer(fh)
    w.writerow(RESPONSE_FIELDS)
    for k, r in enumerate(rows):
        out = dict(r)  # 复制一份，补入 y 坐标
        out['y'] = float(ys[k])  # 当前节点 y 坐标
        w.writerow([out.get(f, float('nan')) for f in RESPONSE_FIELDS])
    fh.close()


def write_matrix_csv(path, axis_name, axis, xs, values):
    """写频率/周期—空间矩阵 CSV：首列为坐标轴，其后每节点一列。"""
    fh = _open_csv(path)
    w = csv.writer(fh)
    w.writerow([axis_name] + ['x=%.3f' % float(x) for x in xs])
    for i in range(len(axis)):
        w.writerow([float(axis[i])] + [float(v) for v in values[:, i]])
    fh.close()


def write_H_csv(path, freqs, xs, H):
    """兼容旧调用：写频率—空间幅值矩阵 CSV。"""
    write_matrix_csv(path, 'f_Hz', freqs, xs, H)


NPZ_FILENAME = 'surface_results.npz'  # 单工况唯一数值输出文件名
XLSX_FILENAME = 'surface_results.xlsx'  # 供研究者查阅的 Excel 工作簿文件名
POSTPROCESS_STATUS_FILENAME = 'postprocess_status.json'  # 供外层批处理核验数据提取是否成功
RAW_TIMESERIES = {}  # 逐记录原始时程与能量历史，最终直接写入 NPZ
SPECTRAL_RESULTS = {}  # 逐记录复频响、FSAF 派生依据、PSA/RSAF 与有效掩码
_NPZ_TEMP_PATTERNS = (  # 打包后删除的运行期临时数值文件
    'surface_response_*.csv',
    'FSAF_inc_h_*.csv', 'FSAF_inc_v_*.csv', 'FSAF_1D_h_*.csv', 'FSAF_station_h_*.csv',
    'PSA_surface_h_*.csv', 'PSA_surface_v_*.csv', 'RSAF_rock_h_*.csv', 'RSAF_1D_h_*.csv', 'URSAF_z_*.csv',
    'sgrid_response_*.csv',
    'sgrid_FSAF_inc_h_*.csv', 'sgrid_FSAF_inc_v_*.csv', 'sgrid_FSAF_1D_h_*.csv', 'sgrid_FSAF_station_h_*.csv',
    'sgrid_PSA_surface_h_*.csv', 'sgrid_PSA_surface_v_*.csv',
    'sgrid_RSAF_rock_h_*.csv', 'sgrid_RSAF_1D_h_*.csv', 'sgrid_URSAF_z_*.csv',
    # 旧 v2 临时文件仍可被重新打包，保证已有中断工况兼容
    'H_surface_h_*.csv', 'H_surface_v_*.csv', 'H_topo_h_*.csv',
    'sgrid_H_surface_h_*.csv', 'sgrid_H_surface_v_*.csv', 'sgrid_H_topo_h_*.csv',
)


def _npz_bytes(value):  # 把文本统一编码为 UTF-8 字节数组，兼容 Py2/Py3 且不依赖 pickle
    """返回适合 np.savez 的 UTF-8 字节标量。"""
    if not isinstance(value, bytes):
        value = value.encode('utf-8')
    return np.asarray(value)


def _load_npz_no_pickle(path):  # 兼容 Abaqus NumPy 1.15/Windows 的 NPZ 路径读取缺陷
    """只读无 pickle NPZ；Python 2 下直接以路径构造 NpzFile，避开旧 np.load 的文件对象定位错误。"""
    if sys.version_info[0] < 3 and hasattr(np.lib.npyio, 'NpzFile'):
        return np.lib.npyio.NpzFile(path, allow_pickle=False)
    try:
        return np.load(path, allow_pickle=False)
    except TypeError:  # 极旧 NumPy 不接受 allow_pickle 参数
        return np.load(path)


def write_postprocess_status(summary, success, reason):  # 写外层批处理可直接读取的轻量状态
    """记录本次数据提取和规范文件写出状态，不承担研究质量评价。"""
    records = []
    for item in (summary or {}).get('records') or []:
        records.append({
            'record': item.get('record'),
            'data_status': 'failed' if item.get('error') else 'extracted',
            'error': item.get('error'),
        })
    payload = {
        'schema_version': 2,
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'success': bool(success),
        'status': 'completed' if success else 'failed',
        'reason': str(reason),
        'records': records,
    }
    with open(POSTPROCESS_STATUS_FILENAME, 'w') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def _read_csv_table_for_npz(path):  # 读取临时 CSV 为表头和文本矩阵
    """将任意临时 CSV 转为 UTF-8 文本数组，保留数值精度与字符串列。"""
    if sys.version_info[0] >= 3:
        fh = io.open(path, 'r', encoding='utf-8-sig', newline='')
    else:
        fh = open(path, 'rb')
    try:
        rows = list(csv.reader(fh))
    finally:
        fh.close()
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _npz_safe_record(record):  # 生成稳定的 NPZ 键后缀
    """把记录名转换为不含路径和空格的 NPZ 键后缀。"""
    text = str(record or 'record')
    return ''.join(ch if (ch.isalnum() or ch in ('_', '-')) else '_' for ch in text)


def _put_raw_timeseries_payload(payload, manifest, raw_timeseries):  # 写入原始时程扩展字段
    """将逐记录原始时程、地下验证点和能量历史直接写入 NPZ。"""
    for record, data in sorted((raw_timeseries or {}).items()):
        suffix = _npz_safe_record(record)
        prefix = 'raw_%s_' % suffix
        for field in ('time', 'x', 'y', 'acc_h', 'acc_v', 'input_acc',
                      'underground_name',
                      'underground_time', 'underground_x', 'underground_y',
                      'underground_acc_h', 'underground_acc_v', 'energy_time'):
            value = data.get(field)
            if value is None:
                continue
            payload[prefix + field] = np.asarray(value)
            manifest.append({'key': prefix + field, 'name': 'raw_%s_%s' % (record, field),
                             'kind': 'raw_timeseries'})
        for variable, value in sorted((data.get('energy_values') or {}).items()):
            if value is None:
                continue
            field = 'energy_%s' % str(variable)
            payload[prefix + field] = np.asarray(value)
            manifest.append({'key': prefix + field, 'name': 'raw_%s_%s' % (record, field),
                             'kind': 'energy_history'})
def _put_spectral_payload(payload, manifest, spectral_results):  # 写入规范复频响与反应谱数组
    """把逐记录复数 FRF、PSA/RSAF 和显式有效掩码写入 NPZ，不使用 pickle。"""
    for record, groups in sorted((spectral_results or {}).items()):
        suffix = _npz_safe_record(record)
        for group in ('frf', 'rsa'):
            data = (groups or {}).get(group) or {}
            for field, value in sorted(data.items()):
                if value is None or field == 'metadata':  # 内部字典不直接持久化，使用 metadata_json
                    continue
                key = '%s_%s_%s' % (group, suffix, field)
                if field.endswith('_json'):
                    payload[key] = _npz_bytes(value if isinstance(value, (str, _TEXT_TYPE))
                                              else json.dumps(value, ensure_ascii=True, sort_keys=True))
                    kind = 'metadata_json'
                else:
                    payload[key] = np.asarray(value)
                    kind = 'complex_frf' if group == 'frf' else 'response_spectrum'
                array = np.asarray(payload[key])
                manifest.append({'key': key, 'name': key, 'kind': kind,
                                 'shape': list(array.shape), 'dtype': str(array.dtype)})


def write_surface_npz(meta, case_cfg, raw_timeseries=None, spectral_results=None):  # 打包全部数值产物并删除临时 CSV/JSON
    """写单工况 NPZ 数值包；表结构由 manifest_json 描述，禁止 pickle 以保障跨环境读取。"""
    paths = []
    for pattern in _NPZ_TEMP_PATTERNS:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    semantics = {
        'complex_frf': 'H=A_surface/A_incident，保存复数值；相位由 angle(H) 派生',
        'FSAF_inc': 'FSAF_inc=abs(H)，与复频响共用 valid_mask，不是独立标签',
        'FSAF_1D': 'FSAF_1D=abs(A_surface/A_exact_same_side_1D)，同时受入射谱与一维参考谱掩码约束',
        'FSAF_station': '同侧端点参考台站幅值谱比，不等于匹配均质坡 H_topo',
        'PSA': '5% 默认阻尼弹性伪加速度谱，T=0 不在周期网格内',
        'RSAF_rock': 'PSA_surface_h/PSA_exact_rock_reference_h',
        'RSAF_1D': 'PSA_surface_h/PSA_exact_same_side_1D_reference_h',
        'URSAF_z': 'PSA_surface_v/PSA_exact_rock_reference_h，采用统一水平分母',
        'array_axis_order': 'FRF/FSAF 为 node_or_s × frequency；PSA/RSAF 为 node_or_s × period',
        'invalid_policy': '真实分母不足时 NaN+valid_mask；禁止 epsilon 分母',
    }
    payload = {
        'schema_version': np.asarray(2),
        'postprocess_version': _npz_bytes('Postprocess_All_surface_v2'),
        'metric_semantics_json': _npz_bytes(json.dumps(semantics, ensure_ascii=True, sort_keys=True)),
        'case_meta_json': _npz_bytes(json.dumps(meta or {}, ensure_ascii=True, sort_keys=True)),
        'case_config_json': _npz_bytes(json.dumps(case_cfg or {}, ensure_ascii=True, sort_keys=True)),
    }
    for name in ('surface_summary.json', 'sgrid_params.json'):
        if os.path.isfile(name):
            with open(name, 'rb') as fh:
                payload[name.replace('.', '_')] = np.asarray(fh.read())
    manifest = []
    _put_raw_timeseries_payload(payload, manifest, raw_timeseries)
    _put_spectral_payload(payload, manifest, spectral_results)
    for index, path in enumerate(paths):
        header, rows = _read_csv_table_for_npz(path)
        key = 'table_%03d' % index
        payload[key + '_header'] = np.asarray([str(v).encode('utf-8') for v in header])
        payload[key + '_data'] = np.asarray([[str(v).encode('utf-8') for v in row] for row in rows])
        manifest.append({'key': key, 'name': os.path.basename(path), 'kind': 'table'})
    payload['manifest_json'] = _npz_bytes(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    np.savez_compressed(NPZ_FILENAME, **payload)
    for path in paths + [name for name in ('surface_summary.json', 'sgrid_params.json') if os.path.isfile(name)]:
        try:
            os.remove(path)
        except OSError:
            pass
    return len(manifest)


def _xlsx_col(index):  # Excel 列序号转字母
    """把从 1 开始的列序号转换为 Excel 列名。"""
    out = ''
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def _npz_text(value):  # NPZ UTF-8 文本解码
    """兼容 Py2/Py3 的 NPZ 字节标量或数组元素。"""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


def _xlsx_cell(row, col, value, style=0):  # 写单个 OpenXML 单元格
    """将数值写为 Excel 数字，其余内容写为 inlineStr，避免共享字符串表。"""
    ref = '%s%d' % (_xlsx_col(col), row)
    text = _npz_text(value)
    try:
        number = float(text)
        if not math.isnan(number) and not math.isinf(number):  # Py2/Py3 兼容的有限数判定
            return '<c r="%s" s="%d"><v>%.15g</v></c>' % (ref, style, number)
    except Exception:
        pass
    return '<c r="%s" s="%d" t="inlineStr"><is><t>%s</t></is></c>' % (ref, style, xml_escape(text))


def _xlsx_sheet_xml(rows):  # 将二维表写为简单工作表 XML
    """首行应用表头样式，数据行保持原始数值类型。"""
    body = []
    for r_idx, row in enumerate(rows, 1):
        cells = [_xlsx_cell(r_idx, c_idx, value, 1 if r_idx == 1 else 0)
                 for c_idx, value in enumerate(row, 1)]
        body.append('<row r="%d">%s</row>' % (r_idx, ''.join(cells)))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
            'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
            '<sheetFormatPr defaultRowHeight="15"/><sheetData>%s</sheetData></worksheet>') % ''.join(body)


def write_surface_xlsx_from_npz():  # 由 NPZ 生成研究者查阅用 Excel 工作簿
    """以标准库导出 XLSX：概览页 + NPZ 内每张数值表一页，不依赖 Abaqus 外部库。"""
    package = _load_npz_no_pickle(NPZ_FILENAME)
    try:
        manifest = json.loads(_npz_text(package['manifest_json']))
        overview = [['项目', '内容'], ['schema_version', _npz_text(package['schema_version'])],
                    ['case_meta_json', _npz_text(package['case_meta_json'])],
                    ['case_config_json', _npz_text(package['case_config_json'])],
                    ['surface_summary_json', _npz_text(package['surface_summary_json'])]]
        if 'metric_semantics_json' in package:
            overview.append(['metric_semantics_json', _npz_text(package['metric_semantics_json'])])
        if 'sgrid_params_json' in package:
            overview.append(['sgrid_params_json', _npz_text(package['sgrid_params_json'])])
        sheets = [('Overview', overview)]  # Abaqus Python 2 XML 拼接使用 ASCII 工作表名以避免编码错误
        used_names = set(['Overview'])
        for index, item in enumerate(manifest, 1):
            if item.get('kind') not in (None, 'table'):
                continue  # 规范数组由 NPZ 保真保存，Excel 仅展开人读二维表
            base = item['name'].replace('.csv', '')
            name = base[:28]
            while name in used_names:
                name = (base[:24] + '_%02d' % index)[:31]
            used_names.add(name)
            key = item['key']
            header = [_npz_text(v) for v in package[key + '_header']]
            values = [[_npz_text(v) for v in row] for row in package[key + '_data']]
            sheets.append((name, [header] + values))
    finally:
        package.close()
    content_overrides = ['<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
                         '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>']
    workbook_sheets = []
    workbook_rels = []
    with zipfile.ZipFile(XLSX_FILENAME, 'w', zipfile.ZIP_DEFLATED) as archive:
        for index, (name, rows) in enumerate(sheets, 1):
            content_overrides.append('<Override PartName="/xl/worksheets/sheet%d.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' % index)
            workbook_sheets.append('<sheet name="%s" sheetId="%d" r:id="rId%d"/>' % (xml_escape(name), index, index))
            workbook_rels.append('<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet%d.xml"/>' % (index, index))
            archive.writestr('xl/worksheets/sheet%d.xml' % index, _xlsx_sheet_xml(rows).encode('utf-8'))
        archive.writestr('[Content_Types].xml', ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>%s</Types>' % ''.join(content_overrides)).encode('utf-8'))
        archive.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr('xl/workbook.xml', ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>%s</sheets></workbook>' % ''.join(workbook_sheets)).encode('utf-8'))
        archive.writestr('xl/_rels/workbook.xml.rels', ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">%s<Relationship Id="rId%d" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>' % (''.join(workbook_rels), len(sheets) + 1)).encode('utf-8'))
        archive.writestr('xl/styles.xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="10"/><name val="Calibri"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Calibri"/></font></fonts><fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs><cellXfs count="2"><xf xfId="0"/><xf xfId="0" fontId="1" fillId="1" applyFont="1" applyFill="1"/></cellXfs></styleSheet>')
    return len(sheets)


# ==========================================================
#  主流程
# ==========================================================


def process_one_odb(odb_path, meta, case_cfg, logger=None):
    """处理单条 odb：提取→指标→传函→写文件，返回汇总 dict。"""
    logger = logger or log_step()
    record = strip_record_name(odb_path)
    factor_h, factor_v, taf_lr, x_toe, fc = meta_pieces(meta)
    frf_cfg = _frf_config(case_cfg)
    explicit_fmax = frf_cfg.get('fmax_hz')
    frf_fmax_hz = resolve_frf_fmax_hz(fc=fc, fmax_hz=explicit_fmax)
    wave = find_wave_file(record, case_cfg)  # 定位输入波（PGA_in 与 H 分母）
    if wave is None:
        log_step(logger, '警告: %s 找不到匹配输入波 txt，AF/TAF/H 跳过，仅输出 PGA', record)
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        xs, ys, t, a1_mat, a2_mat = extract_surface_acc(odb)
        underground = extract_validation_underground_acc(odb)  # 读取解析验证地下点
        energy = extract_energy_history(odb)  # 读取 freefield 整体能量历史
    finally:
        odb.close()
    t_raw = t  # 保留 ODB 原始帧时间轴，两个分量必须据此同步重采样
    t, a1_mat, dt = to_uniform(t_raw, a1_mat)  # 水平分量均匀化（FIXED 下原样返回）
    t_v, a2_mat, _ = to_uniform(t_raw, a2_mat)  # 竖向分量必须使用同一原始帧时间轴
    if t_v.shape != t.shape or not np.allclose(t_v, t):  # 极端浮点差异时强制对齐到水平时间轴
        a2_mat = np.vstack([np.interp(t, t_v, a2_mat[k]) for k in range(a2_mat.shape[0])])

    pga_in = None
    a_in = None
    a_in_pad = None
    if wave:
        rec = np.loadtxt(wave)  # 输入记录 [time, acc]
        a_in = rec[:, 1]
        pga_in = float(np.max(np.abs(a_in)))
        dt_in = float(rec[1, 0] - rec[0, 0])
        if abs(dt_in - dt) > 1e-6 * dt:  # 帧步长与输入步长不一致（输出降频/自动增量）时重采样输入
            a_in = np.interp(t, rec[:, 0], a_in, left=0.0, right=0.0)
    if x_toe is None:  # meta 缺几何时退化：全部归 left
        x_toe = float(xs[-1]) + 1.0
        log_step(logger, '警告: case_meta 缺 x_toe，同侧规则退化为全 left')

    rows = surface_metrics(xs, a1_mat, a2_mat, pga_in, factor_h, factor_v, taf_lr, x_toe)  # 逐节点指标
    incident_angle = float((meta or {}).get('incident_angle') or 0.0)  # 读取斜入射传播方向
    write_response_csv('surface_response_%s.csv' % record, ys, rows)

    frf_payload = None
    frf_metadata = None
    if a_in is not None:
        n_len = a1_mat.shape[1]
        a_in_pad = np.zeros(n_len)  # 输入补零到地表时程长度（尾段静默）
        m = min(n_len, a_in.size)
        a_in_pad[:m] = a_in[:m]
        freqs, H_h_complex, input_valid, input_spectrum = compute_complex_H(
            a1_mat, a_in_pad, dt, fc=fc, fmax_hz=frf_fmax_hz)
        freqs_v, H_v_complex, input_valid_v, _unused_input_v = compute_complex_H(
            a2_mat, a_in_pad, dt, fc=fc, fmax_hz=frf_fmax_hz)
        if freqs_v.shape != freqs.shape or not np.allclose(freqs_v, freqs) or not np.array_equal(input_valid_v, input_valid):
            raise RuntimeError('水平/竖向复频响的频轴或输入有效掩码不一致')

        # 同侧端点参考台站复谱比；它是诊断量，不等同于匹配均质坡 H_topo
        left_mask = xs <= x_toe
        H_station = np.empty(H_h_complex.shape, dtype=np.complex128)
        H_station[:] = complex(float('nan'), float('nan'))
        station_valid = np.zeros(H_h_complex.shape, dtype=bool)
        f_l, H_l, valid_l, _unused_l = compute_complex_H(
            a1_mat[left_mask], a1_mat[0], dt, fc=fc, fmax_hz=frf_fmax_hz)
        if f_l.shape != freqs.shape or not np.allclose(f_l, freqs):
            raise RuntimeError('左参考台站谱比频轴与入射频响不一致')
        shared_l = input_valid & valid_l
        for local_index, global_index in enumerate(np.where(left_mask)[0]):
            H_station[global_index, shared_l] = H_l[local_index, shared_l]
            station_valid[global_index, shared_l] = True
        if np.any(~left_mask):
            f_r, H_r, valid_r, _unused_r = compute_complex_H(
                a1_mat[~left_mask], a1_mat[-1], dt, fc=fc, fmax_hz=frf_fmax_hz)
            if f_r.shape != freqs.shape or not np.allclose(f_r, freqs):
                raise RuntimeError('右参考台站谱比频轴与入射频响不一致')
            shared_r = input_valid & valid_r
            for local_index, global_index in enumerate(np.where(~left_mask)[0]):
                H_station[global_index, shared_r] = H_r[local_index, shared_r]
                station_valid[global_index, shared_r] = True

        fsaf_h = np.abs(H_h_complex[:, input_valid])
        fsaf_v = np.abs(H_v_complex[:, input_valid])
        fsaf_station = np.abs(H_station[:, input_valid])
        write_H_csv('FSAF_inc_h_%s.csv' % record, freqs[input_valid], xs, fsaf_h)
        write_H_csv('FSAF_inc_v_%s.csv' % record, freqs[input_valid], xs, fsaf_v)
        write_H_csv('FSAF_station_h_%s.csv' % record, freqs[input_valid], xs, fsaf_station)

        valid_count = int(np.sum(input_valid))
        candidate_count = int(freqs.size)
        valid_fraction = float(valid_count) / float(candidate_count) if candidate_count else 0.0
        frf_metadata = {
            'reference': 'incident_base_acceleration',
            'frequency_range_hz': [float(freqs[input_valid][0]), float(freqs[input_valid][-1])] if valid_count else None,
            'valid_bin_count': valid_count, 'candidate_bin_count': candidate_count,
            'valid_fraction': valid_fraction,
            'spectrum_mask_ratio': SPEC_MASK_RATIO, 'frequency_lower_hz': F_LO,
            'frequency_upper_hz': frf_fmax_hz,
            'frequency_upper_source': ('run_cfg.frf_cfg.fmax_hz' if explicit_fmax is not None
                                       else ('damping_fc_factor' if frf_fmax_hz is not None else 'nyquist')),
            'zero_padding_factor': PAD_FACTOR,
            'fft_length': _padded_fft_length(max(a1_mat.shape[1], a_in_pad.size)),
            'window': 'full_odb_time_history_no_taper', 'detrend': 'none',
            'station_ratio_definition': 'same_side_endpoint_reference_not_matched_homogeneous_slope',
        }
        frf_payload = {
            'frequency': freqs, 'x': xs, 'input_spectrum': input_spectrum,
            'valid_mask': input_valid, 'station_valid_mask': station_valid,
            'H_surface_h': H_h_complex, 'H_surface_v': H_v_complex,
            'H_station_h': H_station,
            'metadata_json': json.dumps(frf_metadata, ensure_ascii=True, sort_keys=True),
            'metadata': frf_metadata,
        }

    reference_payload = load_reference_payload(record, t, case_cfg)
    rsa_payload = compute_response_spectrum_payload(
        record, xs, x_toe, t, a1_mat, a2_mat, dt, case_cfg,
        reference_payload=reference_payload)
    if rsa_payload is not None:
        periods = rsa_payload['period']
        write_matrix_csv('PSA_surface_h_%s.csv' % record, 'T_s', periods, xs, rsa_payload['PSA_surface_h'])
        write_matrix_csv('PSA_surface_v_%s.csv' % record, 'T_s', periods, xs, rsa_payload['PSA_surface_v'])
        write_matrix_csv('RSAF_rock_h_%s.csv' % record, 'T_s', periods, xs, rsa_payload['RSAF_rock_h'])
        write_matrix_csv('RSAF_1D_h_%s.csv' % record, 'T_s', periods, xs, rsa_payload['RSAF_1D_h'])
        write_matrix_csv('URSAF_z_%s.csv' % record, 'T_s', periods, xs, rsa_payload['URSAF_z'])

    if frf_payload is not None:
        reference_series = reference_payload.get('series') or {}
        ref_left = reference_series.get('one_d_left')
        ref_right = reference_series.get('one_d_right')
        H_over_1d, valid_over_1d = compute_side_reference_H(
            a1_mat, xs, x_toe, ref_left, ref_right, dt, freqs, input_valid,
            fc=fc, fmax_hz=frf_fmax_hz)
        H_over_1d_left, valid_over_1d_left = compute_uniform_reference_H(
            a1_mat, ref_left, dt, freqs, input_valid,
            fc=fc, fmax_hz=frf_fmax_hz)
        frf_payload['H_surface_over_1D_h'] = H_over_1d
        frf_payload['one_d_valid_mask'] = valid_over_1d
        frf_payload['H_surface_over_1D_left_h'] = H_over_1d_left
        frf_payload['one_d_left_valid_mask'] = valid_over_1d_left
        write_H_csv('FSAF_1D_h_%s.csv' % record, freqs[input_valid], xs,
                    np.abs(H_over_1d[:, input_valid]))
        write_H_csv('FSAF_1D_left_h_%s.csv' % record, freqs[input_valid], xs,
                    np.abs(H_over_1d_left[:, input_valid]))
        frf_metadata['one_d_reference_available'] = bool(np.any(valid_over_1d))
        frf_metadata['one_d_left_global_reference_available'] = bool(np.any(valid_over_1d_left))
        possible_1d = int(len(xs) * np.sum(input_valid))
        frf_metadata['one_d_valid_fraction'] = (float(np.sum(valid_over_1d)) / float(possible_1d)
                                                if possible_1d else 0.0)
        frf_payload['metadata_json'] = json.dumps(frf_metadata, ensure_ascii=True, sort_keys=True)

    spectral_groups = {}
    if frf_payload is not None:
        spectral_groups['frf'] = frf_payload
    if rsa_payload is not None:
        spectral_groups['rsa'] = rsa_payload
    if spectral_groups:
        SPECTRAL_RESULTS[record] = spectral_groups

    RAW_TIMESERIES[record] = {
        'time': t, 'x': xs, 'y': ys, 'acc_h': a1_mat, 'acc_v': a2_mat,
        'input_acc': a_in_pad,
        'underground_name': underground.get('name'),
        'underground_time': underground.get('time'), 'underground_x': underground.get('x'),
        'underground_y': underground.get('y'), 'underground_acc_h': underground.get('acc_h'),
        'underground_acc_v': underground.get('acc_v'),
        'energy_time': np.asarray(energy.get('time', []), dtype=float),
        'energy_values': energy.get('values', {}),
    }

    taf_arr = np.array([r['TAF_h'] for r in rows], dtype=float)
    ar_idx = int(np.nanargmax(taf_arr)) if np.any(~np.isnan(taf_arr)) else None  # AR_max 在重采样前的原始曲线上取
    rsa_metadata = rsa_payload.get('metadata') if rsa_payload is not None else None
    summary = {'record': record, 'n_nodes': len(xs), 'dt': dt, 'duration': float(t[-1]),
               'wave_file': wave, 'pga_in': pga_in, 'factor_h': factor_h, 'factor_v': factor_v, 'fc': fc,
               'AR_max': (float(taf_arr[ar_idx]) if ar_idx is not None else None),  # 峰值放大（招牌标量）
               'AR_max_x': (rows[ar_idx]['x'] if ar_idx is not None else None),  # 峰值位置
               'incident_angle': incident_angle,
               'frf_valid_bin_count': (frf_metadata or {}).get('valid_bin_count', 0),
               'frf_valid_fraction': (frf_metadata or {}).get('valid_fraction', 0.0),
               'one_d_reference_available': bool((frf_metadata or {}).get('one_d_reference_available', False)),
               'response_spectrum_period_count': (rsa_metadata or {}).get('period_count', 0)}
    summary['data_status'] = 'extracted'  # 只记录数据提取状态；研究质量由独立评估脚本判定
    log_step(logger, '%s: 节点=%d PGA_in=%s AR_max=%s@x=%s FRF有效频点=%d 一维参考=%s',
             record, len(xs), str(pga_in),
             str(summary['AR_max']), str(summary['AR_max_x']),
             int(summary['frf_valid_bin_count']),
             '有' if summary['one_d_reference_available'] else '无')
    return summary


# ==========================================================
#  出版级绘图工具（三段归一化坐标 s 出图）
# ==========================================================
CB_PALETTE = {  # Okabe-Ito 色盲安全配色
    'black': '#000000',  # 黑
    'orange': '#E69F00',  # 橙
    'skyblue': '#56B4E9',  # 天蓝
    'green': '#009E73',  # 绿
    'yellow': '#F0E442',  # 黄
    'blue': '#0072B2',  # 蓝
    'vermillion': '#D55E00',  # 朱红
    'purple': '#CC79A7',  # 紫
}

CJK_SERIF_PRIORITY = [  # 中文衬线字体优先级列表
    'Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC', 'Source Han Serif CN',  # 衬线宋体类
    'SimSun', 'NSimSun', 'STSong', 'Songti SC',  # 系统宋体类
]


def _detect_cjk_serif_local():  # 检测系统可用的中文衬线字体
    """检测可用的衬线宋体名。"""
    try:
        import matplotlib.font_manager as fm  # 导入字体管理器
        available = {f.name for f in fm.fontManager.ttflist}  # 可用字体集合
        for name in CJK_SERIF_PRIORITY:  # 遍历优先级
            if name in available:  # 命中
                return name  # 返回
        for name in available:  # 兜底模糊匹配
            low = name.lower()  # 小写
            if any(k in low for k in ('song', 'serif cjk', 'serif sc', 'serif cn', 'songti')):  # 包含宋体字眼
                return name  # 返回
    except Exception:  # 异常
        pass  # 忽略
    return None  # 未找到


def _rc_safe(params):  # 逐键安全写入 rcParams
    """逐键写入 rcParams，旧版 matplotlib 缺键时跳过该键不崩溃。"""
    import matplotlib.pyplot as plt  # 导入 pyplot
    for k, v in params.items():  # 遍历配置项
        try:  # 尝试写入
            plt.rcParams[k] = v  # 写入单键
        except Exception:  # 该版本无此键
            pass  # 跳过


def setup_cn_journal_style_local():  # 自动应用中文核心期刊绘图配置
    """配置 matplotlib 出版级样式，返回选用的中文字体名（找不到返回 None，调用方应改用英文标签）。

    字体链按 matplotlib 版本适配：>=3.6 支持逐字形回退，Times 在前实现"宋体正文+Times 数字"混排；
    旧版（含 Abaqus 自带 Python 的 matplotlib）只取链中首个可用字体，必须把中文字体放最前，
    否则汉字全部渲染成方框——这正是 v1 首版在 Abaqus 环境下作图出方框的根因。
    """
    import matplotlib  # 导入顶层包（取版本号）
    cjk = _detect_cjk_serif_local()  # 检测中文字体
    try:  # 解析版本号
        mpl_ver = tuple(int(p) for p in matplotlib.__version__.split('.')[:2])  # (主, 次)
    except Exception:  # 解析失败
        mpl_ver = (0, 0)  # 按旧版处理
    if cjk:  # 找到中文字体
        if mpl_ver >= (3, 6):  # 新版：逐字形回退，Times 在前混排
            serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # 混排回退链
        else:  # 旧版：只认首个可用字体，中文在前保证不出方框
            serif_list = [cjk, 'Times New Roman', 'STIXGeneral', 'DejaVu Serif']  # 中文优先链
        _rc_safe({
            'font.family': serif_list,  # 字体系列
            'font.serif': serif_list,  # 衬线系列
            'mathtext.fontset': 'stix',  # 数学公式采用 STIX 风格
        })
    _rc_safe({
        'axes.unicode_minus': False,  # 解决负号显示为方框问题
        'pdf.fonttype': 42,  # PDF 嵌入 TrueType 字体防止报错
        'ps.fonttype': 42,  # PS 同上
        'svg.fonttype': 'none',  # SVG 文本保持可编辑
        'font.size': 8,  # 基准字号 8pt
        'axes.labelsize': 8,  # 轴标签字号 8pt
        'xtick.labelsize': 7,  # x 轴刻度字号 7pt
        'ytick.labelsize': 7,  # y 轴刻度字号 7pt
        'lines.linewidth': 1.2,  # 曲线线宽 1.2pt（与 Plot_Fig15_compare_v3 一致）
        'axes.linewidth': 0.7,  # 边框线宽 0.7pt
        'xtick.direction': 'in',  # 刻度线朝内
        'ytick.direction': 'in',  # 刻度线朝内
    })
    return cjk  # 返回选用中文字体名（None=未找到）


def style_axes_local(ax):  # 美化单轴外观
    """配置白底、四面朝内刻度与细密主次网格（口径与 Plot_Fig15_compare_v3 的 style_axes 一致）。"""
    ax.set_facecolor('white')  # 设置背景为白色
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)  # 朝内显示
    for spine in ax.spines.values():  # 遍历四周边框
        spine.set_color('black')  # 设为黑色
        spine.set_linewidth(1.0)  # 线宽 1.0（与 v3 一致）
    # 只开启 Y 轴的次要刻度，X 轴不开启，以防止段 B (只有一个主刻度 [0.5]) 导致 X 轴 AutoMinorLocator 崩溃
    import matplotlib.ticker as ticker
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)  # 主网格
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)  # 次网格


def _resolve_slope_height(geo):  # 推算坡高（三段归一化 A/C 段特征长度）
    """从 case_meta.geometry 推算坡高：H_minus_h → w_slope·tan(i) → (x_toe−x_crest)·tan(i)。

    注意 geometry['h'] 是"坡脚下土层厚"（可为负），不是坡高，绝不能用它归一化——
    这是 v1 首版 s 轴范围出错（本应 [-A_max, 1+C_max] 却画到 [-10, 8.5]）的根因。
    找不到有效坡高返回 None。
    """
    hmh = geo.get('H_minus_h')  # 坡高（建模脚本口径：H_minus_h 即斜坡特征高度）
    if hmh and float(hmh) > 0:  # 有效
        return float(hmh)  # 直接返回
    w = geo.get('w_slope')  # 坡面水平跨度
    if w is None and geo.get('x_toe') is not None and geo.get('x_crest') is not None:  # 跨度缺失时由拐点差推算
        w = float(geo['x_toe']) - float(geo['x_crest'])  # 坡脚 x − 坡顶 x
    i_deg = geo.get('i')  # 坡角(度)
    if w and i_deg:  # 跨度与坡角齐备
        h = float(w) * math.tan(math.radians(float(i_deg)))  # 坡高 = 水平跨度×tan(坡角)
        if h > 0:  # 有效
            return h  # 返回推算值
    return None  # 无法推算


def calc_s_coords(xs, x_crest, x_toe, h_ref):  # 计算三段归一化坐标 s
    """根据拓扑关系将 x 坐标计算为连续的无量纲三段归一化坐标 s（研究计划 §4.0）。

    参数:
        xs (list/ndarray): 物理 x 坐标数组
        x_crest (float): 坡顶棱 x 坐标
        x_toe (float): 坡脚棱 x 坐标
        h_ref (float): 坡高（A/C 段特征长度，用 _resolve_slope_height 取得）

    返回:
        numpy.ndarray: 三段归一化 s 坐标数组（段A ≤0 / 段B [0,1] / 段C ≥1，拐点严格对齐）
    """
    xs = np.asarray(xs, dtype=float)  # 强制转为 ndarray
    s = np.zeros_like(xs)  # 初始化 s 数组
    if h_ref is None or h_ref <= 0:  # 厚度参考无效
        h_ref = 1.0  # 兜底值
    w_slope = x_toe - x_crest  # 计算坡面水平跨度
    if w_slope <= 0:  # 无效跨度
        w_slope = 1.0  # 兜底值
    for i, x in enumerate(xs):  # 遍历每个节点
        if x <= x_crest + 1e-4:  # 属于段 A 坡顶平台
            s[i] = (x - x_crest) / h_ref  # 计算对应 s (值为负或零)
        elif x <= x_toe + 1e-4:  # 属于段 B 坡面（含坡脚棱交界点）
            s[i] = (x - x_crest) / w_slope  # 线性映射至 [0, 1] 区间
        else:  # 属于段 C 坡脚平台
            s[i] = 1.0 + (x - x_toe) / h_ref  # 计算对应 s (值大于 1)
    return s  # 返回结果


def read_response_csv_local(path):  # 无依赖读取地表响应 CSV
    """使用内置 csv 模块读取地表响应 CSV，以防无 pandas。

    参数:
        path (str): CSV 文件路径

    返回:
        list: 字典列表，每一个元素代表一行
    """
    data = []  # 初始化数据列表
    try:  # 尝试读取
        if sys.version_info[0] >= 3:  # Python 3
            f = io.open(path, 'r', encoding='utf-8-sig')  # 自动处理带 BOM 的 UTF-8
        else:  # Python 2
            f = open(path, 'rb')  # 以二进制打开
        reader = csv.reader(f)  # 创建 reader
        header = next(reader)  # 提取表头
        if header and header[0].startswith('\xef\xbb\xbf'):  # 清理可能存在的 BOM 字符
            header[0] = header[0][3:]  # 截断
        elif header and header[0].startswith('﻿'):  # 另一种 unicode BOM 标记
            header[0] = header[0][1:]  # 截断
        for row in reader:  # 循环读取每行
            if not row:  # 空行
                continue  # 跳过
            item = {}  # 行字典
            for idx, h in enumerate(header):  # 遍历表头与对应值
                val_str = row[idx]  # 值文本
                try:  # 尝试转换数值
                    val_low = str(val_str).strip().lower()  # 统一大小写并去空格
                    if val_low in ('nan', 'na', '', '-nan(ind)', 'nan(ind)', '1.#ind', '-1.#ind'):  # 空值/非法数字
                        val = float('nan')  # 置为 NaN
                    else:  # 正常数字
                        val = float(val_str)  # 转浮点数
                except ValueError:  # 非数值
                    val = val_str  # 保留字符串
                item[h] = val  # 写入字典
            data.append(item)  # 加入列表
        f.close()  # 关闭文件句柄
    except Exception as e:  # 读取异常
        print('[plot] 读取 CSV %s 失败: %s' % (path, str(e)))  # 提示用户
        return []  # 返回空
    return data  # 返回结果


def _grayscale_preview_local(png_path):  # 生成灰度预览（色盲自检，可选）
    """用 Pillow 把 PNG 转灰度另存 *_grayscale.png；无 Pillow 时静默跳过，返回路径或 None。"""
    try:  # 尝试导入 Pillow
        from PIL import Image  # 图像处理
    except ImportError:  # 未安装
        return None  # 跳过
    try:  # 尝试转换
        gray_path = png_path[:-4] + '_grayscale.png'  # 灰度版路径
        Image.open(png_path).convert('L').save(gray_path)  # 转灰度保存
        return gray_path  # 返回路径
    except Exception:  # 转换失败
        return None  # 跳过


def prepare_plot_data(csv_path, x_crest, x_toe, h_slope, raw=False):  # 准备原始或重采样绘图数据
    """读取响应表并返回记录名、逐行数据、s 坐标和输出名前缀。"""
    data = read_response_csv_local(csv_path)  # 读取响应表
    prefix = 'surface_response_' if raw else 'sgrid_response_'  # 数据源文件名前缀
    record = os.path.basename(csv_path)[len(prefix):-4]  # 提取记录名
    if not data:  # 空表
        return record, data, np.array([], dtype=float), ('raw_s_' if raw else '')  # 返回空数据
    if raw:  # 原始逐节点表只做 x→s 换算
        xs = np.array([row['x'] for row in data], dtype=float)  # 原始物理坐标
        s_all = calc_s_coords(xs, x_crest, x_toe, h_slope)  # 不插值、不补点、不重采样
    else:  # 统一网格表已有 s 列
        s_all = np.array([row['s'] for row in data], dtype=float)  # 直接读取归一坐标
    return record, data, s_all, ('raw_s_' if raw else '')  # 返回绘图所需数据


def _field_has_finite_value(data, field):  # 判断字段是否存在有效值
    """检查字段是否至少包含一个有限值；用于避免旧 CSV 的新增空列生成空白图。"""
    for r in data:  # 遍历行
        val = r.get(field)  # 读取字段
        try:  # 尝试判定有限性
            fv = float(val)  # 转浮点
            if val is not None and (not math.isnan(fv)) and (not math.isinf(fv)):  # 有有限值
                return True  # 可绘制
        except Exception:  # 非数值
            pass  # 忽略
    return False  # 无有效值


PLOT_SMOOTH_WINDOW = 11  # 仅用于成图的分段移动平均窗口，原始数值与 NPZ 不改变


def smooth_curve_local(values, window=PLOT_SMOOTH_WINDOW):  # 分段平滑空间响应曲线
    """使用忽略 NaN 的居中移动平均削弱节点锯齿，不填补原始缺口。"""
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
    counts = np.convolve(np.pad(valid.astype(float), pad, mode='edge'), np.ones(width), mode='valid')
    out = np.divide(sums, counts, out=np.nan * np.ones(len(arr)), where=counts > 0)
    out[~valid] = np.nan
    return out


def plot_results(meta, case_cfg, logger=None):  # 画图主控制函数
    """遍历 surface_response_*.csv，按【三段分轴】布局生成 3x2 出版级图表。

    每个指标面板拆成 坡顶平台(A)/坡面(B)/坡脚平台(C) 三个共 y 轴子轴：
    宽度按各段归一跨度定（B 段保底宽度，避免被挤成一条缝），
    坡顶棱 #1 / 坡脚棱 #2 严格落在轴缝虚线上——对应研究计划 §4.0 拐点对齐约定。
    观测范围取 case_config.geometry_cfg 的 crest_window/toe_window（单位 h），
    未配置时不截断（连边界净空段一起画）并提示。

    参数:
        meta (dict): 从 case_meta.json 加载的元数据字典
        case_cfg (dict): 从 case_config.json 加载的配置字典
    """
    logger = logger or log_step()
    try:  # 尝试导入绘图包
        import matplotlib  # 导入 matplotlib
        matplotlib.use('Agg')  # 无界面后端防止崩溃
        import matplotlib.pyplot as plt  # 导入 pyplot
        import matplotlib.gridspec as gridspec  # 网格布局（三段分轴用）
    except ImportError:  # 未装绘图包
        log_step(logger, '[plot] 提示: 未检测到 matplotlib 库，跳过图表自动绘制。')  # 提示
        return  # 退出

    ctx = _resolve_s_context(meta, case_cfg)  # 解析三段坐标几何上下文（拐点/坡高/观测窗口）
    if ctx is None:  # 上下文不完整
        log_step(logger, '[plot] 警告: case_meta 缺拐点或坡高（x_crest/x_toe 或 H_minus_h/w_slope·tan(i)），跳过作图。')  # 警告
        return  # 退出
    x_crest, x_toe, h_slope, a_win, c_win = ctx  # 解包几何上下文

    csvs = [(path, True) for path in sorted(glob.glob('surface_response_*.csv'))]  # 原始逐节点数据
    csvs += [(path, False) for path in sorted(glob.glob('sgrid_response_*.csv'))]  # 统一网格数据
    if not csvs:  # 两类文件均不存在
        log_step(logger, '[plot] 未发现 surface_response_*.csv 或 sgrid_response_*.csv，跳过作图。')  # 提示
        return  # 退出

    cjk = None  # 选用中文字体名
    try:  # 尝试应用中文出版级样式
        cjk = setup_cn_journal_style_local()  # 应用配置（返回中文字体名或 None）
    except Exception as e:  # 失败
        log_step(logger, '[plot] 应用出版级样式失败: %s，回退默认配置。', str(e))  # 提示
    use_cn = bool(cjk)  # 找不到中文字体时整图改用英文标签，杜绝方框
    if not use_cn:  # 无中文字体
        log_step(logger, '[plot] 提示: 未检测到中文字体，图内文字改用英文标签，避免渲染成方框。')  # 提示

    if use_cn:  # 中文标签集
        labels = {'PGA_h': u'水平向 PGA (m/s²)', 'PGA_v': u'垂直向 PGA (m/s²)',
                  'PGA_R': u'合成 PGA (m/s²)',
                  'AF_h': u'水平向 AF', 'AF_v': u'垂直向 AF',
                  'TAF_h': u'水平向 TAF', 'TAF_v': u'垂直向 TAF',
                  'TAF_h_comp': u'水平分量 TAF', 'TAF_v_comp': u'竖向分量 TAF',
                  'VTR': u'竖向转换系数 VTR',
                  'UTAF_h': u'统一水平 UTAF', 'UTAF_v': u'统一竖向 UTAF',
                  'UTAF_R': u'统一合成 UTAF', 'TAF_R': u'合成 TAF_R',
                  'DUTAF_v': u'竖向增量 ΔUTAF_v',
                  'V_over_H': u'竖横比 V/H'}  # 纵轴标签
        seg_titles = (u'坡顶平台', u'坡面', u'坡脚平台')  # 三段标题
        xlabel = u'归一化坐标 s'  # 横轴标签
        sup_fmt = u'记录: %s'  # 总标题模板
    else:  # 英文兜底标签集
        labels = {'PGA_h': u'Horizontal PGA (m/s²)', 'PGA_v': u'Vertical PGA (m/s²)',
                  'PGA_R': u'Resultant PGA (m/s²)',
                  'AF_h': u'Horizontal AF', 'AF_v': u'Vertical AF',
                  'TAF_h': u'Horizontal TAF', 'TAF_v': u'Vertical TAF',
                  'TAF_h_comp': u'Component TAF-H', 'TAF_v_comp': u'Component TAF-V',
                  'VTR': u'Vertical conversion ratio',
                  'UTAF_h': u'Unified TAF-H', 'UTAF_v': u'Unified TAF-V',
                  'UTAF_R': u'Unified resultant TAF', 'TAF_R': u'Resultant TAF',
                  'DUTAF_v': u'Vertical increment ΔUTAF-V',
                  'V_over_H': u'Vertical-to-horizontal ratio'}  # 纵轴标签
        seg_titles = (u'Crest plateau', u'Slope', u'Toe plateau')  # 三段标题
        xlabel = u'Normalized coordinate s'  # 横轴标签
        sup_fmt = u'Record: %s'  # 总标题模板

    draw_specs_all = [  # 与 Plot_Hybrid_surface_v2.py 的独立分图字段和配色严格一致
        ('PGA_h', CB_PALETTE['blue']), ('PGA_v', CB_PALETTE['vermillion']),
        ('PGA_R', CB_PALETTE['black']), ('AF_h', CB_PALETTE['blue']),
        ('AF_v', CB_PALETTE['vermillion']), ('TAF_h', CB_PALETTE['blue']),
        ('TAF_v', CB_PALETTE['vermillion']), ('TAF_h_comp', CB_PALETTE['skyblue']),
        ('TAF_v_comp', CB_PALETTE['orange']), ('VTR', CB_PALETTE['purple']),
        ('UTAF_h', CB_PALETTE['skyblue']), ('UTAF_v', CB_PALETTE['orange']),
        ('UTAF_R', CB_PALETTE['green']), ('TAF_R', CB_PALETTE['green']),
        ('DUTAF_v', CB_PALETTE['vermillion']), ('V_over_H', CB_PALETTE['black']),
    ]

    for csv_path, raw in csvs:  # 遍历原始与重采样记录
        record = os.path.basename(csv_path)  # 异常日志的默认记录名
        try:  # 尝试画图
            record, data, s_all, output_prefix = prepare_plot_data(
                csv_path, x_crest, x_toe, h_slope, raw=raw)  # 读取并统一为 s 坐标
            if not data:  # 无数据
                continue  # 跳过

            a_max = float(a_win) if (a_win and float(a_win) > 0) else max(float(-s_all.min()), 0.5)  # 段A 显示跨度
            c_max = float(c_win) if (c_win and float(c_win) > 0) else max(float(s_all.max()) - 1.0, 0.5)  # 段C 显示跨度
            if not (a_win and c_win):  # 未配置观测窗口
                log_step(logger, '[plot] 提示: geometry_cfg 未配置 crest_window/toe_window，观测范围不截断（净空段一并画出）。')  # 提示
            draw_specs = [(f, c) for f, c in draw_specs_all
                          if f in data[0] and (_field_has_finite_value(data, f) or f in ('TAF_v', 'TAF_v_comp'))]  # 仅绘制当前 CSV 有效字段
            if not draw_specs:  # 无可绘制字段
                continue  # 跳过
            n_cols = 2  # 双栏排布
            n_rows = int(math.ceil(float(len(draw_specs)) / float(n_cols)))  # 根据字段数动态确定行数

            fig = plt.figure(figsize=(6.3, max(8.2, 2.55 * n_rows + 0.7)))  # 双栏宽画布，按面板数增高
            outer = gridspec.GridSpec(n_rows, n_cols, left=0.10, right=0.985, top=0.94, bottom=0.055,
                                      hspace=0.52, wspace=0.26)  # 外层动态网格

            for panel_idx, (field, color) in enumerate(draw_specs):  # 遍历面板规格
                row_idx = panel_idx // n_cols  # 当前面板行号
                col_idx = panel_idx % n_cols  # 当前面板列号
                panel_lbl = '(%s)' % chr(ord('a') + panel_idx)  # 学术子图编号
                ax = fig.add_subplot(outer[row_idx, col_idx])  # 与独立分图脚本一致：每个面板使用单一连续坐标轴
                style_axes_local(ax)  # 网格、刻度和边框与独立分图统一
                values = np.array([r.get(field, float('nan')) for r in data], dtype=float)
                segments = np.where(s_all <= 0.0, 'A', np.where(s_all < 1.0, 'B', 'C'))
                for seg in ('A', 'B', 'C'):  # 各段独立平滑，坡肩/坡脚不跨段抹平
                    mask = segments == seg
                    values[mask] = smooth_curve_local(values[mask])
                shown = (s_all >= -a_max - 1e-9) & (s_all <= 1.0 + c_max + 1e-9)
                y_view = np.where(shown, values, np.nan)
                finite = np.isfinite(y_view)
                if np.any(finite):
                    order = np.argsort(s_all)
                    ax.plot(s_all[order], y_view[order], color=color, linestyle='-', linewidth=1.2, zorder=3)
                else:
                    note = u'θ=0° 时竖向自由场为 0，分量 TAF_v 不适用' if field in ('TAF_v', 'TAF_v_comp') else u'无有效数据'
                    ax.text(0.5, 0.5, note, transform=ax.transAxes, ha='center', va='center', fontsize=7)
                ax.set_xlim(-a_max, 1.0 + c_max)  # 连续横轴严格采用观测窗实际范围
                lo, hi = (float(np.nanmin(y_view)), float(np.nanmax(y_view))) if np.any(finite) else (0.0, 1.0)
                pad = 0.06 * ((hi - lo) if hi > lo else max(abs(hi), 1.0))
                ax.set_ylim(lo - pad, hi + pad)
                tick_start, tick_end = int(math.ceil(-a_max)), int(math.floor(1.0 + c_max))
                ax.set_xticks([float(t) for t in range(tick_start, tick_end + 1)])
                ax.axvline(0.0, color='#222222', linestyle='--', linewidth=1.15, zorder=10)
                ax.axvline(1.0, color='#222222', linestyle='--', linewidth=1.15, zorder=10)
                for xc, title in zip(((-a_max) / 2.0, 0.5, 1.0 + c_max / 2.0), seg_titles):
                    ax.text(xc, 1.035, title, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=7, clip_on=False)
                ax.set_ylabel(labels[field])
                ax.set_xlabel(xlabel, labelpad=6)
                ax.text(-0.16, 1.10, panel_lbl, transform=ax.transAxes, ha='left', va='bottom',
                        fontsize=8, fontname='Times New Roman', fontweight='bold', clip_on=False)

            fig.suptitle(sup_fmt % record, fontsize=9, fontweight='bold', y=0.97)  # 总标题

            out_dir = 'figs'  # 输出子目录
            if not os.path.exists(out_dir):  # 目录不存在
                os.makedirs(out_dir)  # 创建

            fig_path = os.path.join(out_dir, 'surface_response_%s%s' % (output_prefix, record))  # 原始对照图带 raw_s 前缀
            for fmt in ('png', 'pdf', 'svg'):  # 多格式导出（矢量+栅格，与 Plot_Fig15_compare_v3 口径）
                try:  # 逐格式保存
                    old_err = np.seterr(all='ignore')
                    fig.savefig('%s.%s' % (fig_path, fmt), dpi=300, bbox_inches='tight', pad_inches=0.05)  # 保存
                    np.seterr(**old_err)
                except Exception as e2:  # 某格式失败不影响其余
                    log_step(logger, '[plot] 导出 %s 格式失败: %s', fmt, str(e2))  # 提示
            plt.close(fig)  # 关闭释放内存
            gray = _grayscale_preview_local(fig_path + '.png')  # 灰度预览（色盲自检，可选）
            log_step(logger, '[plot] 成功生成三段分轴图表: %s.{png,pdf,svg}%s',
                     fig_path, (' + 灰度预览' if gray else ''))  # 提示用户
        except Exception as e:  # 绘图异常
            log_step(logger, '[plot] 绘制记录 %s 失败: %s', record, str(e))  # 错误提示


# ==========================================================
#  三段重采样（研究计划 §4.0 第②步：统一 s 子网格对齐输出，给 POD/ML 用）
# ==========================================================
# 统一按 s 坐标每 0.01 划分，原固定个数网格 N_A/N_B/N_C 已废弃
_SEG_EPS = 1e-9  # 拐点归段容差（s=0/1 两侧段共享）
SGRID_FIELDS = [  # 参与重采样的响应字段
    'PGA_h', 'PGA_v', 'PGA_R',
    'AF_h', 'AF_v', 'TAF_h', 'TAF_v',
    'TAF_h_comp', 'TAF_v_comp', 'VTR',
    'UTAF_h', 'UTAF_v', 'UTAF_R', 'TAF_R', 'DUTAF_v',
    'V_over_H'
]
SGRID_CONTINUOUS_FIELDS = set(['PGA_h', 'PGA_v', 'PGA_R', 'AF_h', 'AF_v', 'V_over_H'])  # 连续物理量跨段整体插值，避免坡脚前端外假平台
SGRID_MAX_GAPFILL = 0.15  # TAF 等分段量只补很短的内部缺口，避免跨大段无效区造线


def _resolve_s_context(meta, case_cfg):  # 解析三段坐标几何上下文
    """从 case_meta/case_config 解析三段坐标所需几何，绘图与重采样共用。

    返回 (x_crest, x_toe, h_slope, crest_window, toe_window)；拐点或坡高缺失返回 None。
    """
    geo = (meta or {}).get('geometry') or {}  # 几何块
    x_crest = geo.get('x_crest')  # 坡顶棱
    x_toe = geo.get('x_toe')  # 坡脚棱
    if x_crest is None or x_toe is None:  # 拐点缺失
        return None  # 无法计算
    h_slope = _resolve_slope_height(geo)  # 坡高（A/C 段特征长度）
    if h_slope is None:  # 坡高缺失
        return None  # 无法计算
    gcfg = (case_cfg or {}).get('geometry_cfg') or {}  # 无量纲几何设计配置
    crest_win = gcfg.get('crest_window')  # 坡顶观测窗（hs 倍数）
    toe_win = gcfg.get('toe_window')  # 坡脚观测窗（hs 倍数）
    if crest_win is not None:
        crest_win = float(crest_win)  # 仅保留观测窗，不把 side_clearance 的边界净空画入图中
    if toe_win is not None:
        toe_win = float(toe_win)  # 仅保留观测窗，不把 side_clearance 的边界净空画入图中
    return float(x_crest), float(x_toe), float(h_slope), crest_win, toe_win  # 上下文元组


def build_s_grid(a_max, c_max):  # 构造统一三段 s 子网格
    """构造统一子网格：不分段分个数划分，统一按归一化坐标 s 每 0.01 划分。

    范围为 [-a_max, 1.0 + c_max]。返回 (s_grid, seg_labels)。
    """
    # 构造从 -a_max 到 1.0 + c_max 步长为 0.01 的网格
    # 为防浮点精度问题，使用 100 倍整数索引计算再缩放
    start_idx = int(np.round(-float(a_max) * 100))
    end_idx = int(np.round((1.0 + float(c_max)) * 100))
    indices = np.arange(start_idx, end_idx + 1)
    s_grid = indices * 0.01

    # 段标签打法（网格点归段依据）：
    # s <= 0 的打 'A'
    # 0 < s <= 1.0 的打 'B'（坡面终点 s=1.00 严格归段 B，与以 ref_left 为分母的复频响匹配）
    # s > 1.0 的打 'C'
    seg_labels = []
    for s in s_grid:
        if s <= 0.0:
            seg_labels.append('A')
        elif s <= 1.0 + 1e-9:
            seg_labels.append('B')
        else:
            seg_labels.append('C')
    return s_grid, seg_labels  # 返回网格与标签


def _seg_mask(arr, seg):  # 数组元素归段掩码
    """返回 arr 中属于段 seg 的布尔掩码；拐点 s=0/1 由相邻两段共享（保证插值触棱）。"""
    arr = np.asarray(arr, dtype=float)  # 转数组
    if seg == 'A':  # 坡顶平台
        return arr <= _SEG_EPS  # s≤0
    if seg == 'B':  # 坡面
        return (arr >= -_SEG_EPS) & (arr <= 1.0 + _SEG_EPS)  # 0≤s≤1
    return arr >= 1.0 - _SEG_EPS  # 坡脚平台 s≥1


def resample_curve(s_nodes, y_nodes, s_grid, seg_labels):  # 单条曲线三段重采样
    """把逐节点曲线按段线性插值到统一 s 子网格；段内有效点<2 时该段输出 NaN（不跨拐点借点）。"""
    s_nodes = np.asarray(s_nodes, dtype=float)  # 节点 s 坐标（升序）
    y_nodes = np.asarray(y_nodes, dtype=float)  # 节点值
    grid = np.asarray(s_grid, dtype=float)  # 网格坐标
    lab = np.array(seg_labels)  # 网格段标签
    out = np.nan * np.ones(len(grid))  # 输出初始化为 NaN
    for seg in ('A', 'B', 'C'):  # 逐段独立插值
        nm = _seg_mask(s_nodes, seg) & ~np.isnan(y_nodes)  # 段内有效节点
        if int(np.sum(nm)) < 2:  # 有效点不足
            continue  # 该段保持 NaN
        gm = lab == seg  # 网格段掩码
        out[gm] = np.interp(grid[gm], s_nodes[nm], y_nodes[nm], left=np.nan, right=np.nan)  # 段内插值，端外不造假
    return out  # 返回对齐后曲线


def resample_curve_global(s_nodes, y_nodes, s_grid):  # 单条曲线整体重采样
    """把连续响应量按真实节点整体线性插值到统一 s 子网格；外侧无真实节点处输出 NaN。"""
    s_nodes = np.asarray(s_nodes, dtype=float)  # 节点 s 坐标
    y_nodes = np.asarray(y_nodes, dtype=float)  # 节点值
    grid = np.asarray(s_grid, dtype=float)  # 网格坐标
    ok = ~np.isnan(s_nodes) & ~np.isnan(y_nodes)  # 有效节点
    if int(np.sum(ok)) < 2:  # 有效点不足
        return np.nan * np.ones(len(grid))  # 返回空列
    order = np.argsort(s_nodes[ok])  # 保证插值节点升序
    sx = s_nodes[ok][order]  # 升序 s
    yy = y_nodes[ok][order]  # 同步排序值
    return np.interp(grid, sx, yy, left=np.nan, right=np.nan)  # 整体插值，端外不造假


def fill_short_internal_gaps(out, s_nodes, y_nodes, s_grid, max_gap=SGRID_MAX_GAPFILL):  # 补短内部缺口
    """用两侧真实有效节点线性补齐短内部缺口；不补端外，也不跨越过宽缺口。"""
    out = np.asarray(out, dtype=float).copy()  # 避免原地污染调用者
    s_nodes = np.asarray(s_nodes, dtype=float)  # 原始节点坐标
    y_nodes = np.asarray(y_nodes, dtype=float)  # 原始节点值
    grid = np.asarray(s_grid, dtype=float)  # 目标网格
    ok = ~np.isnan(s_nodes) & ~np.isnan(y_nodes)  # 有效节点
    if int(np.sum(ok)) < 2:  # 有效点不足
        return out  # 无法补
    order = np.argsort(s_nodes[ok])  # 升序排列
    sx = s_nodes[ok][order]  # 有效 s
    yy = y_nodes[ok][order]  # 有效值
    interp_all = np.interp(grid, sx, yy, left=np.nan, right=np.nan)  # 内部候选补值
    fill = np.isnan(out) & ~np.isnan(interp_all)  # 只补已有输出中的空洞
    for i in np.where(fill)[0]:  # 逐空点检查邻近真实点距离
        left = np.searchsorted(sx, grid[i], side='right') - 1  # 左侧真实点
        right = left + 1  # 右侧真实点
        if left >= 0 and right < len(sx) and (sx[right] - sx[left]) <= float(max_gap):  # 缺口足够短
            out[i] = interp_all[i]  # 接上线性补值
    return out  # 返回补齐后曲线


def fill_short_internal_gaps_matrix(out, H, s_nodes, s_grid,
                                    max_gap=SGRID_MAX_GAPFILL):  # 补曲面的短内部缺口
    """逐频补齐连续谱曲面在几何棱附近的短缺口；不补端外或宽缺口。"""
    result = np.asarray(out).copy()
    source = np.atleast_2d(np.asarray(H))
    is_complex = np.iscomplexobj(source)
    for j in range(source.shape[1]):
        if is_complex:
            real = fill_short_internal_gaps(
                result[:, j].real, s_nodes, source[:, j].real, s_grid, max_gap)
            imag = fill_short_internal_gaps(
                result[:, j].imag, s_nodes, source[:, j].imag, s_grid, max_gap)
            missing = ~np.isfinite(result[:, j].real) | ~np.isfinite(result[:, j].imag)
            fill = missing & np.isfinite(real) & np.isfinite(imag)
            result[fill, j] = real[fill] + 1j * imag[fill]
        else:
            result[:, j] = fill_short_internal_gaps(
                result[:, j], s_nodes, source[:, j], s_grid, max_gap)
    return result


def resample_H_matrix(H, s_nodes, s_grid, seg_labels,
                      fill_short_gaps=False):  # H 曲面空间维三段重采样
    """把实数或复数矩阵沿空间维逐段插值到统一 s 子网格。

    复数频响分别插值实部和虚部，避免直接插值包裹相位；每个频点只使用有限节点。
    对物理连续的地表响应可显式补齐几何棱附近不超过阈值的短内部缺口。
    """
    H = np.atleast_2d(np.asarray(H))  # 节点×频点
    is_complex = np.iscomplexobj(H)
    s_nodes = np.asarray(s_nodes, dtype=float)  # 节点 s 坐标
    grid = np.asarray(s_grid, dtype=float)  # 网格坐标
    lab = np.array(seg_labels)  # 网格段标签
    if is_complex:
        out = np.empty((len(grid), H.shape[1]), dtype=np.complex128)
        out[:] = complex(float('nan'), float('nan'))
    else:
        H = np.asarray(H, dtype=float)
        out = np.nan * np.ones((len(grid), H.shape[1]))  # 输出矩阵
    for seg in ('A', 'B', 'C'):  # 逐段
        seg_nodes = _seg_mask(s_nodes, seg)  # 段内节点
        gm = lab == seg  # 网格段掩码
        for j in range(H.shape[1]):  # 逐频点做空间插值
            if is_complex:
                finite = np.isfinite(H[:, j].real) & np.isfinite(H[:, j].imag)
            else:
                finite = np.isfinite(H[:, j])
            nm = seg_nodes & finite
            if int(np.sum(nm)) < 2:
                continue
            if is_complex:
                real = np.interp(grid[gm], s_nodes[nm], H[nm, j].real, left=np.nan, right=np.nan)
                imag = np.interp(grid[gm], s_nodes[nm], H[nm, j].imag, left=np.nan, right=np.nan)
                out[gm, j] = real + 1j * imag
            else:
                out[gm, j] = np.interp(grid[gm], s_nodes[nm], H[nm, j], left=np.nan, right=np.nan)
    if fill_short_gaps:
        out = fill_short_internal_gaps_matrix(out, H, s_nodes, s_grid)
    return out  # 返回对齐曲面


def s_to_x(s, x_crest, x_toe, h_slope):  # 三段归一坐标反算物理 x
    """把 s 反算回物理 x 坐标（核查/追溯用），映射与 calc_s_coords 严格互逆。"""
    if s <= 0.0:  # 段A
        return x_crest + s * h_slope  # 距坡顶棱 |s|·h
    if s < 1.0:  # 段B
        return x_crest + s * (x_toe - x_crest)  # 坡面按水平跨度线性映射
    return x_toe + (s - 1.0) * h_slope  # 段C：距坡脚棱 (s-1)·h


def _parse_matrix_float(value):
    """解析矩阵CSV数值，并把Windows非有限数标记统一还原为NaN。"""
    text = str(value).strip()
    normalized = text.lower().replace(' ', '')
    nonfinite_markers = ('nan', 'ind', 'qnan', 'snan', '#inf', 'infinity')
    if any(marker in normalized for marker in nonfinite_markers) or normalized in ('inf', '+inf', '-inf'):
        return float('nan')
    number = float(text)
    return number if np.isfinite(number) else float('nan')


def read_H_csv_local(path):  # 读回 write_H_csv 产物
    """解析频率/周期—空间矩阵CSV，返回(轴坐标, xs, 节点×轴点矩阵)。"""
    if not os.path.isfile(path):  # 文件不存在（如该波未生成 H）
        return None, None, None  # 返回空
    try:  # 尝试解析
        if sys.version_info[0] >= 3:  # Python 3
            fh = io.open(path, 'r', encoding='utf-8-sig')  # 文本读（去 BOM）
        else:  # Python 2
            fh = open(path, 'rb')  # 二进制读
        reader = csv.reader(fh)  # 创建 reader
        header = next(reader)  # 表头：f_Hz, x=..., ...
        xs = [float(str(c).split('=')[1]) for c in header[1:]]  # 解析列头 x 坐标
        freqs, rows = [], []  # 频点与矩阵行暂存
        for row in reader:  # 逐行（行=频点）
            if not row:  # 空行
                continue  # 跳过
            freqs.append(float(row[0]))  # 频率或周期
            rows.append([_parse_matrix_float(v) for v in row[1:]])  # 非有限频点保留为NaN，不使整张矩阵失效
        fh.close()  # 关闭句柄
        return np.array(freqs, dtype=float), np.array(xs, dtype=float), np.array(rows, dtype=float).T  # 转 节点×频点
    except Exception as e:  # 解析失败
        log_step(None, '[sgrid] 读取 %s 失败: %s', path, str(e))  # 提示
        return None, None, None  # 返回空


def write_sgrid_response_csv(path, s_grid, seg_labels, x_phys, cols):  # 写对齐响应表
    """写统一 s 子网格响应表：s,seg,x + 各重采样字段，共 N_A+N_B+N_C 行。"""
    fh = _open_csv(path)  # 打开句柄
    w = csv.writer(fh)  # 创建 writer
    w.writerow(['s', 'seg', 'x'] + SGRID_FIELDS)  # 表头
    for k in range(len(s_grid)):  # 逐网格点
        w.writerow([float(s_grid[k]), seg_labels[k], float(x_phys[k])] +
                   [float(cols[f][k]) for f in SGRID_FIELDS])  # 写一行
    fh.close()  # 关闭


def write_sgrid_matrix_csv(path, axis_name, axis, s_grid, seg_labels, values):
    """写统一 s 网格上的频率/周期—空间矩阵 CSV。"""
    fh = _open_csv(path)  # 打开句柄
    w = csv.writer(fh)  # 创建 writer
    w.writerow([axis_name] + ['s=%.5f_%s' % (float(s_grid[k]), seg_labels[k]) for k in range(len(s_grid))])
    for i in range(len(axis)):
        w.writerow([float(axis[i])] + [float(v) for v in values[:, i]])
    fh.close()  # 关闭


def write_sgrid_H_csv(path, freqs, s_grid, seg_labels, H_s):  # 兼容旧调用
    """兼容旧调用：写统一 s 网格上的频率—空间幅值矩阵。"""
    write_sgrid_matrix_csv(path, 'f_Hz', freqs, s_grid, seg_labels, H_s)


def _update_summary_ar(updates, logger=None):  # 回写 AR_max 段号/归一坐标
    """把各记录 AR_max 的段号与归一坐标（重采样前原始曲线精确值）合并进 surface_summary.json。"""
    logger = logger or log_step()
    path = 'surface_summary.json'  # 摘要路径
    if not updates or not os.path.isfile(path):  # 无更新或无摘要
        return  # 跳过
    try:  # 尝试更新
        with open(path, 'r') as fh:  # 读
            data = json.load(fh)  # 解析
        for rec in data.get('records', []):  # 遍历记录
            extra = updates.get(rec.get('record'))  # 查找该记录的更新项
            if extra:  # 有更新
                rec.update(extra)  # 合并字段
        with open(path, 'w') as fh:  # 写回
            json.dump(data, fh, indent=2)  # 保存
        log_step(logger, '[sgrid] AR_max 段号/归一坐标已回写 surface_summary.json')  # 提示
    except Exception as e:  # 失败
        log_step(logger, '[sgrid] 回写 surface_summary.json 失败: %s', str(e))  # 提示


def resample_outputs(meta, case_cfg, logger=None):  # 重采样主控制函数
    """把逐节点曲线、FSAF/PSA/RSAF 视图和规范复频响插值到统一三段 s 子网格。

    新输出采用 FSAF/RSAF 明确命名；旧 ``H_surface_*``/``H_topo_h`` 只作为中断工况读取回退。
    复频响在内存中分别插值实部与虚部，最终直接写入 ``surface_results.npz``。
    同时把 AR_max 段号/归一坐标（重采样前原始曲线上精确取值）回写 surface_summary.json。
    普通 Python 即可运行（不依赖 odbAccess），可对已有 CSV 反复重跑。
    """
    logger = logger or log_step()
    ctx = _resolve_s_context(meta, case_cfg)  # 解析几何上下文
    if ctx is None:  # 上下文不完整
        log_step(logger, '[sgrid] 警告: case_meta 缺拐点或坡高（x_crest/x_toe 或 H_minus_h/w_slope·tan(i)），跳过重采样。')  # 警告
        return  # 退出
    x_crest, x_toe, h_slope, a_win, c_win = ctx  # 解包几何上下文
    csvs = sorted(glob.glob('surface_response_*.csv'))  # 第①步响应表
    if not csvs:  # 无输入
        log_step(logger, '[sgrid] 未发现 surface_response_*.csv，跳过重采样。')  # 提示
        return  # 退出
    if not (a_win and c_win):  # 未配置观测窗口
        log_step(logger, '[sgrid] 警告: geometry_cfg 未配置 crest_window/toe_window，网格范围退化为本工况数据范围，跨工况将不可比！')  # 警告
    updates = {}  # summary 回写暂存
    grid_written = False  # 网格参数是否已写盘
    for csv_path in csvs:  # 逐记录处理
        record = csv_path[len('surface_response_'):-4]  # 提取记录名
        data = read_response_csv_local(csv_path)  # 读响应表
        if not data:  # 空表
            continue  # 跳过
        xs = np.array([row['x'] for row in data], dtype=float)  # 节点 x 坐标
        s_nodes = calc_s_coords(xs, x_crest, x_toe, h_slope)  # 节点三段归一坐标
        a_max = float(a_win) if (a_win and float(a_win) > 0) else max(float(-s_nodes.min()), 0.5)  # 段A 网格跨度
        c_max = float(c_win) if (c_win and float(c_win) > 0) else max(float(s_nodes.max()) - 1.0, 0.5)  # 段C 网格跨度
        s_grid, seg_labels = build_s_grid(a_max, c_max)  # 统一三段子网格
        cols = {}  # 重采样结果列
        for f in SGRID_FIELDS:  # 逐字段重采样
            y = np.array([row.get(f, float('nan')) for row in data], dtype=float)  # 原始逐节点列
            if f in SGRID_CONTINUOUS_FIELDS:  # 连续响应量
                cols[f] = resample_curve_global(s_nodes, y, s_grid)  # 跨真实节点整体插值，消除坡脚前假断层
            else:  # TAF 等分段分母量
                cols[f] = fill_short_internal_gaps(resample_curve(s_nodes, y, s_grid, seg_labels),
                                                   s_nodes, y, s_grid)  # 保留分段口径，并补齐拐点附近短缺口
        x_phys = np.array([s_to_x(float(sv), x_crest, x_toe, h_slope) for sv in s_grid])  # 网格点反算物理坐标
        write_sgrid_response_csv('sgrid_response_%s.csv' % record, s_grid, seg_labels, x_phys, cols)  # 落盘
        taf = np.array([row.get('TAF_h', float('nan')) for row in data], dtype=float)  # 原始 TAF_h（重采样前）
        if np.any(~np.isnan(taf)):  # 有有效值
            win = (s_nodes >= -a_max - _SEG_EPS) & (s_nodes <= 1.0 + c_max + _SEG_EPS)  # 仅观测窗内取峰值
            taf_win = np.where(win, taf, np.nan)  # 窗外点不参与 AR_max
            k = int(np.nanargmax(taf_win)) if np.any(~np.isnan(taf_win)) else int(np.nanargmax(taf))  # 窗内优先，空窗兜底
            sk = float(s_nodes[k])  # 峰值归一坐标
            seg = 'A' if sk < 0.0 else ('B' if sk <= 1.0 else 'C')  # 峰值所在段（拐点归坡面口径）
            updates[record] = {'AR_max': float(taf[k]), 'AR_max_x': float(data[k]['x']),
                               'AR_max_s': sk, 'AR_max_seg': seg,
                               'AR_window': [-float(a_max), 1.0 + float(c_max)]}  # 暂存回写项
        n_matrix = 0  # 幅值曲面成功计数
        matrix_specs = (
            (('FSAF_inc_h_%s.csv', 'H_surface_h_%s.csv'), 'sgrid_FSAF_inc_h_%s.csv', 'f_Hz', True),
            (('FSAF_inc_v_%s.csv', 'H_surface_v_%s.csv'), 'sgrid_FSAF_inc_v_%s.csv', 'f_Hz', True),
            (('FSAF_1D_h_%s.csv',), 'sgrid_FSAF_1D_h_%s.csv', 'f_Hz', False),
            (('FSAF_1D_left_h_%s.csv',), 'sgrid_FSAF_1D_left_h_%s.csv', 'f_Hz', True),
            (('FSAF_station_h_%s.csv', 'H_topo_h_%s.csv'), 'sgrid_FSAF_station_h_%s.csv', 'f_Hz', False),
            (('PSA_surface_h_%s.csv',), 'sgrid_PSA_surface_h_%s.csv', 'T_s', True),
            (('PSA_surface_v_%s.csv',), 'sgrid_PSA_surface_v_%s.csv', 'T_s', True),
            (('RSAF_rock_h_%s.csv',), 'sgrid_RSAF_rock_h_%s.csv', 'T_s', False),
            (('RSAF_1D_h_%s.csv',), 'sgrid_RSAF_1D_h_%s.csv', 'T_s', False),
            (('URSAF_z_%s.csv',), 'sgrid_URSAF_z_%s.csv', 'T_s', False),
        )
        for source_formats, dst_fmt, axis_name, fill_short_gaps in matrix_specs:
            source_path = None
            for source_format in source_formats:
                candidate = source_format % record
                if os.path.isfile(candidate):
                    source_path = candidate
                    break
            if source_path is None:
                continue  # 跳过
            matrix_axis, xs_h, H = read_H_csv_local(source_path)
            if matrix_axis is None:
                continue
            s_h = calc_s_coords(xs_h, x_crest, x_toe, h_slope)  # H 列坐标转 s
            H_s = resample_H_matrix(
                H, s_h, s_grid, seg_labels,
                fill_short_gaps=fill_short_gaps)  # 空间维对齐
            write_sgrid_matrix_csv(dst_fmt % record, axis_name, matrix_axis, s_grid, seg_labels, H_s)
            n_matrix += 1

        # 规范 NPZ 数组同步对齐；复数频响按实部/虚部插值，不经过幅值 CSV
        groups = SPECTRAL_RESULTS.get(record) or {}
        segment_array = np.asarray([str(value).encode('ascii') for value in seg_labels])
        frf = groups.get('frf') or {}
        if frf:
            frf_s = calc_s_coords(np.asarray(frf['x'], dtype=float), x_crest, x_toe, h_slope)
            frf['sgrid_s'] = np.asarray(s_grid, dtype=float)
            frf['sgrid_x'] = np.asarray(x_phys, dtype=float)
            frf['sgrid_segment'] = segment_array
            for field in ('H_surface_h', 'H_surface_v', 'H_surface_over_1D_h',
                          'H_surface_over_1D_left_h', 'H_station_h'):
                if field not in frf:
                    continue
                aligned = resample_H_matrix(
                    frf[field], frf_s, s_grid, seg_labels,
                    fill_short_gaps=field in ('H_surface_h', 'H_surface_v', 'H_surface_over_1D_h',
                                              'H_surface_over_1D_left_h', 'H_station_h'))  # 全复频响场启用短缺口填补，修复坡脚 s∈[0.95,1.10] 段界死区
                frf['sgrid_%s' % field] = aligned
                frf['sgrid_%s_valid_mask' % field] = np.isfinite(aligned.real) & np.isfinite(aligned.imag)
            frf_metadata = frf.get('metadata') or {}
            frf_metadata['sgrid_point_count'] = int(len(s_grid))
            frf_metadata['sgrid_complex_interpolation'] = 'real_imag_segmentwise_short_gap_fill_surface'
            frf['metadata_json'] = json.dumps(frf_metadata, ensure_ascii=True, sort_keys=True)
        rsa = groups.get('rsa') or {}
        if rsa:
            rsa_s = calc_s_coords(np.asarray(rsa['x'], dtype=float), x_crest, x_toe, h_slope)
            rsa['sgrid_s'] = np.asarray(s_grid, dtype=float)
            rsa['sgrid_x'] = np.asarray(x_phys, dtype=float)
            rsa['sgrid_segment'] = segment_array
            for field in ('PSA_surface_h', 'PSA_surface_v', 'RSAF_rock_h', 'RSAF_1D_h', 'URSAF_z',
                          'key_PSA_surface_h', 'key_PSA_surface_v', 'key_RSAF_rock_h',
                          'key_RSAF_1D_h', 'key_URSAF_z'):
                aligned = resample_H_matrix(
                    rsa[field], rsa_s, s_grid, seg_labels,
                    fill_short_gaps=field in (
                        'PSA_surface_h', 'PSA_surface_v',
                        'key_PSA_surface_h', 'key_PSA_surface_v'))
                rsa['sgrid_%s' % field] = aligned
                if 'RSAF' in field or 'URSAF' in field:
                    rsa['sgrid_%s_valid_mask' % field] = np.isfinite(aligned)
            rsa_metadata = rsa.get('metadata') or {}
            rsa_metadata['sgrid_point_count'] = int(len(s_grid))
            rsa['metadata_json'] = json.dumps(rsa_metadata, ensure_ascii=True, sort_keys=True)
        n_a_actual = int(np.sum(np.array(seg_labels) == 'A'))  # 实际 A 段点数
        n_b_actual = int(np.sum(np.array(seg_labels) == 'B'))  # 实际 B 段点数
        n_c_actual = int(np.sum(np.array(seg_labels) == 'C'))  # 实际 C 段点数
        if not grid_written:  # 网格参数只写一份（对本工况所有记录一致）
            with open('sgrid_params.json', 'w') as fh:  # 写参数文件
                json.dump({'schema_version': 1, 'N_A': n_a_actual, 'N_B': n_b_actual, 'N_C': n_c_actual,
                           'A_max': a_max, 'C_max': c_max, 'step': 0.01,
                           'h_slope': h_slope,
                           'note': u'统一 s 子网格；按归一化坐标 s 每 0.01 划分；仅含 crest_window/toe_window 观测窗'},
                          fh, indent=2)  # 保存（ensure_ascii 默认开，Py2 安全）
            grid_written = True  # 标记已写
        log_step(logger, '[sgrid] %s: 重采样完成 -> sgrid_response（%d 点/段A %d+段B %d+段C %d） + %d 个谱曲面',
                 record, len(s_grid), n_a_actual, n_b_actual, n_c_actual, n_matrix)  # 提示
    _update_summary_ar(updates, logger=logger)  # 回写 AR_max 段号/归一坐标


def main():  # 主入口函数
    """后处理脚本控制流。"""
    logger = log_step()  # 自动使用与脚本同名的日志文件
    RAW_TIMESERIES.clear()  # 同一解释器重复执行时不混入上一工况
    SPECTRAL_RESULTS.clear()  # 复频响与反应谱同样按本次工况重新建立
    if os.path.isfile(POSTPROCESS_STATUS_FILENAME):  # 防止上次运行的通过状态被误复用
        os.remove(POSTPROCESS_STATUS_FILENAME)
    log_step(logger, '脚本开始执行 (Postprocess_All_surface_v2)')
    meta = _load_json('case_meta.json')  # 读取元数据
    case_cfg = _load_json('case_config.json')  # 读取配置

    if openOdb is None:  # 无 Abaqus 环境
        existing_csv = glob.glob('surface_response_*.csv') + glob.glob('sgrid_response_*.csv')  # 检查运行期临时数据
        if not existing_csv:  # NPZ 是最终包，统一绘图脚本负责从中读取
            log_step(logger, '错误: 未检测到 odbAccess，且当前目录没有运行期 CSV；NPZ 最终包请用 Plot_Hybrid_surface_v2.py 重绘。')  # 明确失败原因
            sys.exit(2)  # 非零退出，防止批处理把无产出误判为成功
        log_step(logger, '提示: 未检测到 odbAccess (非 Abaqus 环境)，跳过 ODB 提取，直接重采样并重绘已有 CSV。')  # 提示
        resample_outputs(meta, case_cfg, logger=logger)  # §4.0 第②步重采样（可对已有 CSV 反复重跑）
        plot_results(meta, case_cfg, logger=logger)  # 运行绘图
        return  # 正常退出

    odbs = sorted(glob.glob('job-*.odb'))  # 搜索 odb 文件
    if not odbs:  # 无 odb
        log_step(logger, '错误: 当前目录无 job-*.odb，无法提取数据。')  # 报错
        sys.exit(1)  # 退出

    summaries = []  # 摘要列表
    failed_odbs = []  # 提取失败的 ODB，后续以非零状态返回给批处理
    for p in odbs:  # 遍历
        try:  # 尝试提取
            summaries.append(process_one_odb(p, meta, case_cfg, logger=logger))  # 处理单条 ODB
        except Exception as e:  # 异常
            log_step(logger, '错误: %s 处理失败: %s', p, str(e))  # 打印错误
            log_step(logger, '错误堆栈:\n%s', traceback.format_exc())  # 记录完整堆栈，便于定位具体字段或 ODB 问题
            summaries.append({'record': strip_record_name(p), 'error': str(e)})  # 记录错误
            failed_odbs.append(p)  # 记录失败项

    with open('surface_summary.json', 'w') as fh:  # 写摘要
        json.dump({'schema_version': 2, 'records': summaries}, fh, indent=2)  # 保存 json
    log_step(logger, '完成: %d 条 odb，汇总见 surface_summary.json', len(odbs))  # 提示完成

    if failed_odbs:  # 任一 ODB 失败时必须显式失败，避免批处理继续清理 ODB 并掩盖根因
        write_postprocess_status(
            {'schema_version': 2, 'records': summaries},
            False, 'odb_extraction_failed',
        )
        log_step(logger, '错误: %d/%d 条 ODB 后处理失败；已保留 ODB 和错误堆栈，停止重采样/绘图。', len(failed_odbs), len(odbs))  # 汇总提示
        sys.exit(1)  # 让 Autorun 正确标记本工况失败，不执行清理

    resample_outputs(meta, case_cfg, logger=logger)  # §4.0 第②步重采样（并回写 AR_max 段号/归一坐标）
    plot_results(meta, case_cfg, logger=logger)  # 提取数据后自动画图
    final_summary = _load_json('surface_summary.json') or {}  # 打包删除运行期JSON前记录提取摘要
    n_items = write_surface_npz(meta, case_cfg, RAW_TIMESERIES, SPECTRAL_RESULTS)  # 所有规范数组收敛为单一 NPZ 包
    n_sheets = write_surface_xlsx_from_npz()  # 再由 NPZ 导出研究者查阅用 Excel 工作簿
    log_step(logger, '完成: 已写入 %s（%d 个数值项）与 %s（%d 个工作表）；运行期 CSV/JSON 已清理。',
             NPZ_FILENAME, n_items, XLSX_FILENAME, n_sheets)
    write_postprocess_status(final_summary, True, 'data_products_written')


if __name__ == '__main__':  # 程序入口
    main()  # 执行主流程
