# -*- coding: utf-8 -*-
"""F0-3 NPZ 原始时程与 QA 扩展协议测试（不依赖 Abaqus）。"""

from __future__ import print_function

import json
import os
import sys
import tempfile
sys.modules['abaqusConstants'] = None
sys.modules['odbAccess'] = None

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
POSTPROCESS_DIR = os.path.join(REPO, 'Postprocess', 'Hybrid')
if POSTPROCESS_DIR not in sys.path:
    sys.path.insert(0, POSTPROCESS_DIR)
import Postprocess_All_surface_v2 as post


def main():
    time = np.linspace(0.0, 1.0, 101)
    input_acc = np.sin(2.0 * np.pi * 4.0 * time)
    theory_h, theory_v = post._theory_series_from_input(input_acc, 2.0, 0.25)
    err, amp, ok = post._series_error(theory_h * 1.01, theory_h)
    assert ok and err.shape == time.shape and abs(amp - 0.01) < 1.0e-12
    phase = post._phase_error_deg(theory_h, theory_h, time[1] - time[0], 4.0)
    assert phase is not None and abs(phase) < 1.0e-8

    payload = {}
    manifest = []
    post._put_raw_timeseries_payload(payload, manifest, {
        'ricker-flat': {
            'time': time,
            'x': np.asarray([0.0, 1.0]),
            'y': np.asarray([10.0, 10.0]),
            'acc_h': np.vstack([theory_h, theory_h * 1.01]),
            'acc_v': np.vstack([theory_v, theory_v * 0.99]),
            'input_acc': input_acc,
            'theory_acc_h': theory_h,
            'theory_acc_v': theory_v,
            'error_acc_h': np.vstack([np.zeros_like(time), theory_h * 0.01]),
            'error_acc_v': np.vstack([np.zeros_like(time), -theory_v * 0.01]),
            'representative_indices': np.asarray([0, 1]),
            'energy_time': np.asarray([], dtype=float),
            'qa_theory_json': json.dumps({'status': 'baseline_only'}),
            'qa_reflection_json': json.dumps({'status': 'not_computed'}),
            'qa_energy_json': json.dumps({'status': 'not_available'}),
        }
    })
    assert 'raw_ricker-flat_time' in payload
    assert 'raw_ricker-flat_acc_h' in payload
    assert 'raw_ricker-flat_qa_theory_json' in payload
    assert len(manifest) == 15
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'surface_results.npz')
        np.savez_compressed(path, **payload)
        package = np.load(path)
        try:
            assert package['raw_ricker-flat_acc_h'].shape == (2, 101)
            assert json.loads(package['raw_ricker-flat_qa_reflection_json'].item())['status'] == 'not_computed'
        finally:
            package.close()

    base_summary = {
        'qa_farfield_basis': 'upstream_left',
        'qa_farfield_err_left': 0.01,
        'qa_farfield_err_right': 0.20,
        'suspect': False,
        'qa_mesh_pass': True,
        'qa_domain_pass': True,
        'qa_external_pass': True,
        'qa_time_pass': True,
    }
    raw_not_ready = {
        'qa_reflection_json': json.dumps({'status': 'not_computed'}),
        'qa_energy_json': json.dumps({'status': 'not_available'}),
    }
    all_gate_result = post.evaluate_qa_gates(base_summary, {}, raw_not_ready)
    assert all_gate_result['qa_gates']['theory'] is True
    assert all_gate_result['qa_gates']['reflection'] is False
    assert all_gate_result['overall_pass'] is False

    theory_only = post.evaluate_qa_gates(base_summary, {'qa_cfg': {'required': ['theory']}}, raw_not_ready)
    assert theory_only['overall_pass'] is True

    base_summary['suspect'] = True
    window_fail = post.evaluate_qa_gates(base_summary, {'qa_cfg': {'required': ['theory', 'time']}}, raw_not_ready)
    assert window_fail['qa_gates']['theory'] is False
    assert window_fail['qa_gates']['time'] is True
    assert window_fail['overall_pass'] is False
    print('test_npz_protocol_v2: 7/7 ok')


if __name__ == '__main__':
    main()
