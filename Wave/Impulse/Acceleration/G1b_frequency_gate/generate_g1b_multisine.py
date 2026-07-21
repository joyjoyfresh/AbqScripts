# -*- coding: utf-8 -*-
"""生成 G1b 复频响闸门使用的两条确定性宽频多正弦加速度输入。"""

from __future__ import print_function

import hashlib
import json
import os

import numpy as np


OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DT = 0.001
SAMPLE_COUNT = 8192
TARGET_PGA = 0.1 * 9.81
RAMP_SECONDS = 0.75
PASSBAND_HZ = (0.5, 12.0)
SHOULDER_HZ = (0.35, 12.5)
VALIDATION_BAND_HZ = (0.5, 10.0)
SPECTRUM_MASK_RATIO = 0.05
PHASE_B_SEED = 20260719
VELOCITY_TREND_SLOPE_LIMIT = 1.0e-8
VELOCITY_TREND_INTERCEPT_LIMIT = 1.0e-8
EFFECTIVE_INPUT_LOG_RMSE_LIMIT = 1.0e-6
EFFECTIVE_INPUT_PHASE_RMSE_LIMIT = 1.0e-6
PILOT_SIGNAL_SHA256 = {
    'a': '6b98cfdfd449e0d01b99ee5922e557fa856a66040af6cd193afc4bc400ec0844',
    'b': '65dcdc06fe28261b0012843aad768e31bb36b81d1fbf86295eb7c6819f71a9e6',
}


def _spectral_envelope(freqs):
    """构造带余弦肩部的单位幅值频谱包络。"""
    f0, f3 = SHOULDER_HZ
    f1, f2 = PASSBAND_HZ
    envelope = np.zeros(freqs.shape, dtype=float)
    rising = (freqs >= f0) & (freqs < f1)
    flat = (freqs >= f1) & (freqs <= f2)
    falling = (freqs > f2) & (freqs <= f3)
    envelope[rising] = 0.5 - 0.5 * np.cos(np.pi * (freqs[rising] - f0) / (f1 - f0))
    envelope[flat] = 1.0
    envelope[falling] = 0.5 + 0.5 * np.cos(np.pi * (freqs[falling] - f2) / (f3 - f2))
    return envelope


def _edge_window(times):
    """在输入首尾施加半余弦渐入渐出，避免瞬时跳变。"""
    duration = float(times[-1])
    distance = np.minimum(times, duration - times)
    window = np.ones(times.shape, dtype=float)
    edge = distance < RAMP_SECONDS
    window[edge] = np.sin(0.5 * np.pi * np.maximum(distance[edge], 0.0) / RAMP_SECONDS) ** 2
    return window


def _zero_integral_correction(signal, times):
    """用端点为零的平滑基函数消除离散积分残差。"""
    duration = float(times[-1])
    basis = np.sin(np.pi * times / duration) ** 2
    denominator = float(np.sum(basis))
    if denominator > 0.0:
        signal = signal - float(np.sum(signal)) / denominator * basis
    signal[0] = 0.0
    signal[-1] = 0.0
    return signal


def _model_velocity_trend(signal, times):
    """复现建模脚本的加速度积分与速度线性去趋势，返回趋势和校正速度。"""
    dt = float(np.median(np.diff(times)))
    acceleration = signal - np.mean(signal)
    velocity = np.zeros_like(acceleration)
    velocity[1:] = np.cumsum((acceleration[:-1] + acceleration[1:]) * 0.5 * dt)
    trend = np.polyfit(times, velocity, 1)
    corrected_velocity = velocity - (trend[0] * times + trend[1])
    return trend, corrected_velocity


def _zero_velocity_trend_correction(signal, times):
    """用两个端点为零、零积分基函数同时消除速度趋势的斜率和截距。"""
    duration = float(times[-1])
    u = times / duration
    sine = np.sin(np.pi * u)
    cosine = np.cos(np.pi * u)
    basis_odd = (
        2.0 * np.pi * sine * cosine * (2.0 * u - 1.0) + 2.0 * sine ** 2
    ) / duration
    basis_even = 2.0 * np.pi * sine * cosine / duration
    trend_signal, _unused_velocity = _model_velocity_trend(signal, times)
    trend_odd, _unused_odd = _model_velocity_trend(basis_odd, times)
    trend_even, _unused_even = _model_velocity_trend(basis_even, times)
    matrix = np.column_stack((trend_odd, trend_even))
    coefficients = np.linalg.solve(matrix, -trend_signal)
    corrected = signal + coefficients[0] * basis_odd + coefficients[1] * basis_even
    corrected[0] = 0.0
    corrected[-1] = 0.0
    return corrected, coefficients


def _make_signal(phase_mode):
    """按指定相位方案生成并归一化一条多正弦输入。"""
    times = np.arange(SAMPLE_COUNT, dtype=float) * DT
    freqs = np.fft.rfftfreq(SAMPLE_COUNT, DT)
    envelope = _spectral_envelope(freqs)
    active = envelope > 0.0
    active_indices = np.where(active)[0]
    phases = np.zeros(freqs.shape, dtype=float)
    if phase_mode == 'schroeder':
        order = np.arange(active_indices.size, dtype=float)
        phases[active] = -np.pi * order * (order - 1.0) / float(active_indices.size)
    elif phase_mode == 'random':
        rng = np.random.RandomState(PHASE_B_SEED)
        phases[active] = rng.uniform(-np.pi, np.pi, active_indices.size)
    else:
        raise ValueError('未知相位模式: %s' % phase_mode)

    spectrum = envelope * np.exp(1j * phases)
    spectrum[0] = 0.0
    signal = np.fft.irfft(spectrum, n=SAMPLE_COUNT)
    signal *= _edge_window(times)
    signal = _zero_integral_correction(signal, times)
    signal, trend_coefficients = _zero_velocity_trend_correction(signal, times)
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        raise RuntimeError('生成的输入峰值为零')
    signal *= TARGET_PGA / peak
    return times, signal, trend_coefficients


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _signal_metrics(times, signal):
    """按生产后处理的 4 倍补零规则计算输入谱有效性指标。"""
    nfft = 1
    while nfft < 4 * signal.size:
        nfft *= 2
    freqs = np.fft.rfftfreq(nfft, DT)
    amplitude = np.abs(np.fft.rfft(signal, n=nfft))
    candidate = (freqs >= VALIDATION_BAND_HZ[0]) & (freqs <= VALIDATION_BAND_HZ[1])
    band_amplitude = amplitude[candidate]
    band_peak = float(np.max(band_amplitude))
    valid = band_amplitude >= SPECTRUM_MASK_RATIO * band_peak
    valid_freqs = freqs[candidate][valid]
    rms = float(np.sqrt(np.mean(signal * signal)))
    velocity_residual = float(np.sum(0.5 * (signal[:-1] + signal[1:]) * np.diff(times)))
    velocity_trend, corrected_velocity = _model_velocity_trend(signal, times)
    effective_acceleration = signal - np.mean(signal) - velocity_trend[0]
    effective_spectrum = np.fft.rfft(effective_acceleration, n=nfft)
    nominal_spectrum = np.fft.rfft(signal, n=nfft)
    shared = candidate & (np.abs(nominal_spectrum) >= SPECTRUM_MASK_RATIO * band_peak)
    effective_ratio = effective_spectrum[shared] / nominal_spectrum[shared]
    effective_log_rmse = float(np.sqrt(np.mean(np.log(np.abs(effective_ratio)) ** 2)))
    effective_phase_rmse = float(np.sqrt(np.mean(np.angle(effective_ratio) ** 2)))
    return {
        'sample_count': int(signal.size),
        'dt_s': DT,
        'duration_s': float(times[-1]),
        'pga_m_s2': float(np.max(np.abs(signal))),
        'rms_m_s2': rms,
        'crest_factor': float(np.max(np.abs(signal)) / rms),
        'mean_m_s2': float(np.mean(signal)),
        'velocity_residual_m_s': velocity_residual,
        'velocity_trend_slope_m_s2': float(velocity_trend[0]),
        'velocity_trend_intercept_m_s': float(velocity_trend[1]),
        'corrected_velocity_endpoint_abs_max_m_s': float(max(
            abs(corrected_velocity[0]), abs(corrected_velocity[-1])
        )),
        'effective_input_log_magnitude_rmse': effective_log_rmse,
        'effective_input_phase_rmse_rad': effective_phase_rmse,
        'endpoint_abs_max_m_s2': float(max(abs(signal[0]), abs(signal[-1]))),
        'validation_band_hz': list(VALIDATION_BAND_HZ),
        'validation_candidate_bin_count': int(np.sum(candidate)),
        'validation_valid_bin_count': int(np.sum(valid)),
        'validation_valid_fraction': float(np.mean(valid)),
        'validation_valid_frequency_range_hz': (
            [float(valid_freqs[0]), float(valid_freqs[-1])] if valid_freqs.size else None
        ),
        'spectrum_mask_ratio': SPECTRUM_MASK_RATIO,
    }


def main():
    outputs = []
    signals = []
    for label, phase_mode in (('a', 'schroeder'), ('b', 'random')):
        times, signal, trend_coefficients = _make_signal(phase_mode)
        filename = 'g1b_multisine_phase_%s.txt' % label
        path = os.path.join(OUTPUT_DIR, filename)
        np.savetxt(path, np.column_stack((times, signal)), fmt=('%.9f', '%.12e'))
        metrics = _signal_metrics(times, signal)
        metrics.update({
            'label': label.upper(),
            'phase_mode': phase_mode,
            'filename': filename,
            'sha256': _sha256(path),
            'pre_normalization_velocity_trend_correction_coefficients': [
                float(value) for value in trend_coefficients
            ],
            'pilot_signal_sha256': PILOT_SIGNAL_SHA256[label],
        })
        outputs.append(metrics)
        signals.append(signal)

    nfft = 1
    while nfft < 4 * SAMPLE_COUNT:
        nfft *= 2
    freqs = np.fft.rfftfreq(nfft, DT)
    band = (freqs >= VALIDATION_BAND_HZ[0]) & (freqs <= VALIDATION_BAND_HZ[1])
    phase_a = np.angle(np.fft.rfft(signals[0], n=nfft)[band])
    phase_b = np.angle(np.fft.rfft(signals[1], n=nfft)[band])
    phase_independence = float(abs(np.mean(np.exp(1j * (phase_a - phase_b)))))

    manifest = {
        'schema_version': 2,
        'purpose': 'G1b 0.5-10 Hz complex-FRF full-band gate',
        'generator': os.path.basename(__file__),
        'generation_parameters': {
            'sample_count': SAMPLE_COUNT,
            'dt_s': DT,
            'target_pga_m_s2': TARGET_PGA,
            'ramp_seconds': RAMP_SECONDS,
            'passband_hz': list(PASSBAND_HZ),
            'shoulder_hz': list(SHOULDER_HZ),
            'phase_b_seed': PHASE_B_SEED,
            'velocity_trend_correction': 'two_zero_integral_endpoint_zero_basis_functions',
        },
        'acceptance': {
            'minimum_valid_fraction_0p5_10hz': 0.95,
            'maximum_phase_coherence': 0.35,
            'maximum_crest_factor': 5.0,
            'maximum_abs_velocity_residual_m_s': 1.0e-5,
            'maximum_abs_velocity_trend_slope_m_s2': VELOCITY_TREND_SLOPE_LIMIT,
            'maximum_abs_velocity_trend_intercept_m_s': VELOCITY_TREND_INTERCEPT_LIMIT,
            'maximum_effective_input_log_magnitude_rmse': EFFECTIVE_INPUT_LOG_RMSE_LIMIT,
            'maximum_effective_input_phase_rmse_rad': EFFECTIVE_INPUT_PHASE_RMSE_LIMIT,
        },
        'provenance_note': (
            'schema 1 inputs were used only for SYS-S2 tail convergence in '
            'Run/ch4_G1b_frequency_gate/run-001; schema 2 closes the model velocity-trend gate'
        ),
        'phase_coherence_0p5_10hz': phase_independence,
        'signals': outputs,
    }
    manifest['passed'] = bool(
        all(item['validation_valid_fraction'] >= 0.95 for item in outputs)
        and all(item['crest_factor'] <= 5.0 for item in outputs)
        and all(abs(item['velocity_residual_m_s']) <= 1.0e-5 for item in outputs)
        and all(abs(item['velocity_trend_slope_m_s2']) <= VELOCITY_TREND_SLOPE_LIMIT for item in outputs)
        and all(abs(item['velocity_trend_intercept_m_s']) <= VELOCITY_TREND_INTERCEPT_LIMIT for item in outputs)
        and all(item['effective_input_log_magnitude_rmse'] <= EFFECTIVE_INPUT_LOG_RMSE_LIMIT for item in outputs)
        and all(item['effective_input_phase_rmse_rad'] <= EFFECTIVE_INPUT_PHASE_RMSE_LIMIT for item in outputs)
        and phase_independence <= 0.35
    )
    manifest_path = os.path.join(OUTPUT_DIR, 'g1b_multisine_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    if not manifest['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
