# -*- coding: utf-8 -*-
"""G1r真实波输入生成器的确定性与单位门测试。"""

from __future__ import print_function

import hashlib
import importlib.util
import os
import unittest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GENERATOR_PATH = os.path.join(
    REPO_ROOT, 'Wave', 'Seismic', 'G1r_reference_gate', 'generate_g1r_real_inputs.py',
)
SPEC = importlib.util.spec_from_file_location('generate_g1r_real_inputs', GENERATOR_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class G1rRealInputGeneratorTests(unittest.TestCase):

    def test_processed_signals_pass_declared_gates(self):
        for _label, source_path in GENERATOR.SOURCES:
            times, signal, _source_dt, _coefficients = GENERATOR.process_source(source_path)
            metrics = GENERATOR.signal_metrics(times, signal)
            self.assertAlmostEqual(metrics['dt_s'], 0.001, places=12)
            self.assertAlmostEqual(metrics['pga_m_s2'], 0.3 * 9.81, places=10)
            self.assertLessEqual(
                abs(metrics['velocity_trend_slope_m_s2']), GENERATOR.VELOCITY_TREND_LIMIT,
            )
            self.assertLessEqual(
                abs(metrics['velocity_trend_intercept_m_s']), GENERATOR.VELOCITY_TREND_LIMIT,
            )
            self.assertLessEqual(
                metrics['energy_fraction_above_15hz'], GENERATOR.HIGH_FREQUENCY_ENERGY_LIMIT,
            )

    def test_generation_is_byte_deterministic(self):
        GENERATOR.main()
        first = {}
        for label, _source_path in GENERATOR.SOURCES:
            path = os.path.join(GENERATOR.OUTPUT_DIR, 'g1r_%s_0p3g_dt1ms.txt' % label)
            first[label] = sha256(path)
        GENERATOR.main()
        for label, _source_path in GENERATOR.SOURCES:
            path = os.path.join(GENERATOR.OUTPUT_DIR, 'g1r_%s_0p3g_dt1ms.txt' % label)
            self.assertEqual(first[label], sha256(path))


if __name__ == '__main__':
    unittest.main()
