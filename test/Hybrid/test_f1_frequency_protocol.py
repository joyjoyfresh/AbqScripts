# -*- coding: utf-8 -*-
"""F1 频域传递函数指标纯 Python 回归。"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'Batch'))
import Autorun_ch3_F1_frequency_theory_v1 as f1


def main():
    dt = 0.001
    time = np.arange(0.0, 4.0 + 0.5 * dt, dt)
    input_series = np.sin(2.0 * np.pi * 2.0 * time) * np.hanning(len(time))
    spectrum = np.fft.rfft(input_series)
    theory = np.ones(len(spectrum), dtype=complex) * 2.0
    theory[0] = 0.0
    actual = np.fft.irfft(spectrum * theory, n=len(time))
    passed = f1._transfer_metrics(actual, input_series, dt, 2.0, theory)
    assert passed['passed'], passed
    failed = f1._transfer_metrics(actual * 1.20, input_series, dt, 2.0, theory)
    assert not failed['passed'], failed
    print('test_f1_frequency_protocol: 2/2 ok')


if __name__ == '__main__':
    main()
