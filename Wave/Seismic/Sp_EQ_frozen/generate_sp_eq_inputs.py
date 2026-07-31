# -*- coding: utf-8 -*-
"""生成小论文 C001—C010 真实波直接闭环的冻结输入(EQ01—EQ03)并写出可追溯清单。

预处理协议与 G1r 参考闸门完全一致：6 阶零相位 Butterworth 12 Hz 低通、
整数倍 polyphase 重采样到 1 ms、g→m/s²、首尾 0.5 s 余弦锥化、
双零积分基函数速度趋势校正，最后统一缩放到 0.1g(0.981 m/s²)。
源文件为 Wave/Seismic/Original 下的 g 单位记录：
  EQ01=El_Centro.txt / EQ02=Kobe.txt / EQ03=ChiChi.txt。
"""

from __future__ import print_function

import hashlib
import json
import os

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(OUTPUT_DIR, '..', '..', '..'))
TARGET_DT = 0.001  # 目标采样间隔(s)，与 G1b 识别信号一致
TARGET_PGA = 0.1 * 9.81  # 目标 PGA(m/s²)：小论文统一 0.1g 线性识别尺度
LOWPASS_HZ = 12.0  # 低通截止(Hz)，与 G1b 通带上限一致
STOPBAND_CHECK_HZ = 15.0  # 高频泄漏检查频率(Hz)
FILTER_ORDER = 6  # Butterworth 阶数
TAPER_SECONDS = 0.50  # 首尾锥化时长(s)
VELOCITY_TREND_LIMIT = 1.0e-8  # 速度趋势验收上限
HIGH_FREQUENCY_ENERGY_LIMIT = 1.0e-4  # 高频能量占比验收上限
SOURCES = (  # (标识, 源文件绝对路径)，源文件均为 g 单位
    ('eq01_el_centro', os.path.join(REPO_ROOT, 'Wave', 'Seismic', 'Original', 'El_Centro.txt')),
    ('eq02_kobe', os.path.join(REPO_ROOT, 'Wave', 'Seismic', 'Original', 'Kobe.txt')),
    ('eq03_chichi', os.path.join(REPO_ROOT, 'Wave', 'Seismic', 'Original', 'ChiChi.txt')),
)


def _sha256(path):
    """计算文件SHA-256。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _model_velocity_trend(signal, times):
    """复现建模脚本的加速度积分与速度线性去趋势。"""
    dt = float(np.median(np.diff(times)))
    acceleration = signal - np.mean(signal)
    velocity = np.zeros_like(acceleration)
    velocity[1:] = np.cumsum((acceleration[:-1] + acceleration[1:]) * 0.5 * dt)
    trend = np.polyfit(times, velocity, 1)
    return trend, velocity - (trend[0] * times + trend[1])


def _zero_velocity_trend_correction(signal, times):
    """用两个端点为零、零积分基函数消除建模速度趋势。"""
    duration = float(times[-1])
    u = times / duration
    sine = np.sin(np.pi * u)
    cosine = np.cos(np.pi * u)
    basis_odd = (
        2.0 * np.pi * sine * cosine * (2.0 * u - 1.0) + 2.0 * sine ** 2
    ) / duration
    basis_even = 2.0 * np.pi * sine * cosine / duration
    target, _unused = _model_velocity_trend(signal, times)
    odd, _unused = _model_velocity_trend(basis_odd, times)
    even, _unused = _model_velocity_trend(basis_even, times)
    coefficients = np.linalg.solve(np.column_stack((odd, even)), -target)
    corrected = signal + coefficients[0] * basis_odd + coefficients[1] * basis_even
    corrected[0] = 0.0
    corrected[-1] = 0.0
    return corrected, coefficients


def _edge_taper(times):
    """返回首尾半余弦窗，抑制滤波和重采样端点跳变。"""
    distance = np.minimum(times, times[-1] - times)
    window = np.ones(times.shape, dtype=float)
    edge = distance < TAPER_SECONDS
    window[edge] = np.sin(
        0.5 * np.pi * np.maximum(distance[edge], 0.0) / TAPER_SECONDS
    ) ** 2
    return window


def process_source(source_path):
    """把规则采样的g单位记录转为低通、1 ms、SI单位输入。"""
    table = np.asarray(np.loadtxt(source_path), dtype=float)
    if table.ndim != 2 or table.shape[0] < 10 or table.shape[1] < 2:
        raise ValueError('真实波文件必须为至少10行两列: %s' % source_path)
    source_time = table[:, 0]
    source_acc_g = table[:, 1]
    source_steps = np.diff(source_time)
    source_dt = float(np.median(source_steps))
    if np.max(np.abs(source_steps - source_dt)) > 1.0e-8:
        raise ValueError('真实波时间步不规则: %s' % source_path)
    source_fs = 1.0 / source_dt
    if LOWPASS_HZ >= 0.5 * source_fs:
        raise ValueError('低通频率不低于源记录Nyquist频率: %s' % source_path)

    sos = butter(FILTER_ORDER, LOWPASS_HZ, btype='lowpass', fs=source_fs, output='sos')
    filtered_g = sosfiltfilt(sos, source_acc_g)
    ratio = source_dt / TARGET_DT
    up = int(round(ratio))
    if abs(ratio - up) > 1.0e-9:
        raise ValueError('当前生成器要求源时间步为1 ms的整数倍: %s' % source_dt)
    resampled_g = resample_poly(filtered_g, up, 1, padtype='line')
    target_count = int(round((source_time[-1] - source_time[0]) / TARGET_DT)) + 1
    target_time = np.arange(target_count, dtype=float) * TARGET_DT
    resampled_g = resampled_g[:target_count]
    signal = resampled_g * 9.81
    signal *= _edge_taper(target_time)
    signal, coefficients = _zero_velocity_trend_correction(signal, target_time)
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        raise ValueError('真实波处理后峰值为零: %s' % source_path)
    signal *= TARGET_PGA / peak
    return target_time, signal, source_dt, coefficients


def signal_metrics(times, signal):
    """计算单位、时间步、基线和高频泄漏验收指标。"""
    trend, corrected_velocity = _model_velocity_trend(signal, times)
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, TARGET_DT)
    energy = np.abs(spectrum) ** 2
    high_fraction = float(
        np.sum(energy[frequencies > STOPBAND_CHECK_HZ]) / max(np.sum(energy), 1.0e-30)
    )
    return {
        'sample_count': int(signal.size),
        'dt_s': TARGET_DT,
        'duration_s': float(times[-1]),
        'pga_m_s2': float(np.max(np.abs(signal))),
        'pga_g': float(np.max(np.abs(signal)) / 9.81),
        'mean_m_s2': float(np.mean(signal)),
        'velocity_trend_slope_m_s2': float(trend[0]),
        'velocity_trend_intercept_m_s': float(trend[1]),
        'corrected_velocity_endpoint_abs_max_m_s': float(max(
            abs(corrected_velocity[0]), abs(corrected_velocity[-1])
        )),
        'energy_fraction_above_15hz': high_fraction,
        'endpoint_abs_max_m_s2': float(max(abs(signal[0]), abs(signal[-1]))),
    }


def main():
    """生成三条冻结输入与清单，任一门槛不通过时返回非零状态。"""
    outputs = []
    for label, source_path in SOURCES:
        times, signal, source_dt, coefficients = process_source(source_path)
        filename = 'sp_%s_0p1g_dt1ms.txt' % label
        output_path = os.path.join(OUTPUT_DIR, filename)
        np.savetxt(output_path, np.column_stack((times, signal)), fmt=('%.9f', '%.12e'))
        metrics = signal_metrics(times, signal)
        metrics.update({
            'label': label,
            'wave_id': 'EQ' + label.split('_')[0][-2:],  # EQ01/EQ02/EQ03
            'filename': filename,
            'sha256': _sha256(output_path),
            'source_path': os.path.relpath(source_path, REPO_ROOT).replace('\\', '/'),
            'source_sha256': _sha256(source_path),
            'source_dt_s': source_dt,
            'source_acceleration_unit': 'g',
            'output_acceleration_unit': 'm/s^2',
            'velocity_trend_correction_coefficients': [float(value) for value in coefficients],
        })
        outputs.append(metrics)

    manifest = {
        'schema_version': 1,
        'purpose': '小论文 C001—C010 真实波直接闭环冻结输入(EQ01—EQ03)，0.1g 统一线性识别尺度',
        'generator': os.path.basename(__file__),
        'generation_parameters': {
            'target_dt_s': TARGET_DT,
            'target_pga_g': 0.1,
            'target_pga_m_s2': TARGET_PGA,
            'lowpass_hz': LOWPASS_HZ,
            'filter': 'sixth_order_zero_phase_butterworth_before_polyphase_resampling',
            'taper_seconds': TAPER_SECONDS,
            'velocity_trend_correction': 'two_zero_integral_endpoint_zero_basis_functions',
        },
        'acceptance': {
            'maximum_abs_velocity_trend_slope_m_s2': VELOCITY_TREND_LIMIT,
            'maximum_abs_velocity_trend_intercept_m_s': VELOCITY_TREND_LIMIT,
            'stopband_check_hz': STOPBAND_CHECK_HZ,
            'maximum_energy_fraction_above_15hz': HIGH_FREQUENCY_ENERGY_LIMIT,
        },
        'signals': outputs,
    }
    manifest['passed'] = bool(all(
        abs(item['velocity_trend_slope_m_s2']) <= VELOCITY_TREND_LIMIT
        and abs(item['velocity_trend_intercept_m_s']) <= VELOCITY_TREND_LIMIT
        and item['energy_fraction_above_15hz'] <= HIGH_FREQUENCY_ENERGY_LIMIT
        and abs(item['pga_m_s2'] - TARGET_PGA) <= 1.0e-10
        and item['endpoint_abs_max_m_s2'] <= 1.0e-12
        for item in outputs
    ))
    manifest_path = os.path.join(OUTPUT_DIR, 'sp_eq_input_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if not manifest['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
