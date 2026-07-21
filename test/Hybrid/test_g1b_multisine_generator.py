# -*- coding: utf-8 -*-
"""G1b宽频多正弦输入生成器的纯Python回归测试。"""

import importlib.util
import os
import unittest

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GENERATOR_PATH = os.path.join(
    REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'G1b_frequency_gate',
    'generate_g1b_multisine.py',
)
SPEC = importlib.util.spec_from_file_location('g1b_multisine_generator', GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def _independent_model_velocity_trend(signal, times):
    """独立复现建模脚本算法，避免直接调用生成器内部验收函数。"""
    dt = float(np.median(np.diff(times)))
    acceleration = signal - np.mean(signal)
    velocity = np.zeros_like(acceleration)
    velocity[1:] = np.cumsum((acceleration[:-1] + acceleration[1:]) * 0.5 * dt)
    return np.polyfit(times, velocity, 1)


class G1bMultisineGeneratorTests(unittest.TestCase):

    def test_signals_are_deterministic_and_pass_velocity_trend_gate(self):
        for phase_mode in ('schroeder', 'random'):
            times_1, signal_1, _coefficients_1 = GENERATOR._make_signal(phase_mode)
            times_2, signal_2, _coefficients_2 = GENERATOR._make_signal(phase_mode)
            np.testing.assert_array_equal(times_1, times_2)
            np.testing.assert_array_equal(signal_1, signal_2)
            trend = _independent_model_velocity_trend(signal_1, times_1)
            self.assertLessEqual(abs(float(trend[0])), GENERATOR.VELOCITY_TREND_SLOPE_LIMIT)
            self.assertLessEqual(abs(float(trend[1])), GENERATOR.VELOCITY_TREND_INTERCEPT_LIMIT)

    def test_signal_metrics_pass_declared_input_gates(self):
        for phase_mode in ('schroeder', 'random'):
            times, signal, _coefficients = GENERATOR._make_signal(phase_mode)
            metrics = GENERATOR._signal_metrics(times, signal)
            self.assertAlmostEqual(metrics['pga_m_s2'], GENERATOR.TARGET_PGA, places=12)
            self.assertGreaterEqual(metrics['validation_valid_fraction'], 0.95)
            self.assertLessEqual(metrics['crest_factor'], 5.0)
            self.assertLessEqual(
                metrics['effective_input_log_magnitude_rmse'],
                GENERATOR.EFFECTIVE_INPUT_LOG_RMSE_LIMIT,
            )
            self.assertLessEqual(
                metrics['effective_input_phase_rmse_rad'],
                GENERATOR.EFFECTIVE_INPUT_PHASE_RMSE_LIMIT,
            )

    def test_two_phase_realizations_remain_independent(self):
        _times_a, signal_a, _coefficients_a = GENERATOR._make_signal('schroeder')
        _times_b, signal_b, _coefficients_b = GENERATOR._make_signal('random')
        nfft = 1
        while nfft < 4 * signal_a.size:
            nfft *= 2
        freqs = np.fft.rfftfreq(nfft, GENERATOR.DT)
        band = (
            (freqs >= GENERATOR.VALIDATION_BAND_HZ[0])
            & (freqs <= GENERATOR.VALIDATION_BAND_HZ[1])
        )
        phase_a = np.angle(np.fft.rfft(signal_a, n=nfft)[band])
        phase_b = np.angle(np.fft.rfft(signal_b, n=nfft)[band])
        coherence = float(abs(np.mean(np.exp(1j * (phase_a - phase_b)))))
        self.assertLessEqual(coherence, 0.35)


if __name__ == '__main__':
    unittest.main()
