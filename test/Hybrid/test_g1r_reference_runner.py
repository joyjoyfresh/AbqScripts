# -*- coding: utf-8 -*-
"""G1r参考库批处理的纯Python测试。"""

from __future__ import print_function

import importlib.util
import os
import tempfile

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNNER_PATH = os.path.join(REPO_ROOT, 'Run', 'Auto_ch4', 'Autorun_ch4_G1r.py')
SPEC = importlib.util.spec_from_file_location('autorun_ch4_g1r', RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def main():
    fd, progress_path = tempfile.mkstemp(suffix='.log')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write('作业进度: job-test，已算到 1.000 秒/共 2.000 秒\n'.encode('utf-8'))
        started_at = os.path.getmtime(progress_path)
        offset, messages = RUNNER._read_new_job_progress(progress_path, 0, started_at)
        assert offset > 0 and len(messages) == 1 and 'job-test' in messages[0]
    finally:
        os.remove(progress_path)

    cases = RUNNER.build_case_plan()
    assert len(cases) == 12
    assert len(set(item['case_id'] for item in cases)) == 12
    assert all(item['stage'] == 'broadband' for item in cases[:4])
    assert all(item['stage'] == 'real' for item in cases[4:])
    diagnostic = RUNNER.build_s2_mesh20_diagnostic_case()
    assert diagnostic['case_id'] == 'case-G1r-S2-broadband-mesh20-diagnostic'
    assert diagnostic['stage'] == 'diagnostic'
    assert diagnostic['config']['mesh_cfg']['elems_per_wavelength'] == 20
    assert next(item for item in cases if item['case_id'] == 'case-G1r-S2-broadband')['config']['mesh_cfg']['elems_per_wavelength'] == 10
    init_diagnostics = RUNNER.build_real_initialization_diagnostic_cases()
    assert len(init_diagnostics) == 2
    assert all(item['stage'] == 'diagnostic' for item in init_diagnostics)
    assert all(item['profile']['id'] == 'H1' for item in init_diagnostics)
    assert all(
        item['config']['freefield_cfg']['initial_state_mode'] == 'incremental'
        for item in init_diagnostics
    )
    assert all(
        item['config']['mesh_cfg']['elems_per_wavelength'] == 20
        for item in cases
        if item['profile']['id'] == 'S2' and item['stage'] == 'real'
    )
    assert all(
        item['config']['mesh_cfg']['elems_per_wavelength'] == 10
        for item in cases
        if not (item['profile']['id'] == 'S2' and item['stage'] == 'real')
    )
    for item in cases:
        config = item['config']
        assert config['run_cfg']['validation_geometry'] == 'flat'
        assert config['tssi_cfg'] == {'enable': False, 'scene': 'freefield'}
        assert config['run_cfg']['qa_cfg']['required'] == ['frf', 'response_spectrum']
        assert 'job_cfg' not in config
        expected_initial_state = 'incremental' if item['stage'] == 'real' else 'raw'
        assert config['freefield_cfg']['initial_state_mode'] == expected_initial_state
        RUNNER.validate_config(config)

    frequencies = np.asarray([0.5, 2.0, 4.0, 8.0, 10.0])
    h1 = next(item for item in RUNNER.PROFILES if item['id'] == 'H1')
    s2 = next(item for item in RUNNER.PROFILES if item['id'] == 'S2')
    h1_h, h1_v = RUNNER.theory_transfer(frequencies, h1)
    s2_h, s2_v = RUNNER.theory_transfer(frequencies, s2)
    assert np.all(np.isfinite(h1_h.real)) and np.all(np.isfinite(h1_h.imag))
    assert np.max(np.abs(h1_v)) < 1.0e-10
    assert np.all(np.isfinite(s2_h.real)) and np.all(np.isfinite(s2_v.real))
    assert np.max(np.abs(s2_v)) > 1.0e-6
    assert np.max(np.abs(h1_h)) < 2.01
    release_gate = RUNNER.build_real_release_gate({
        'broadband': {'cases': [
            {'profile_id': 'H1', 'passed': True},
            {'profile_id': 'W1', 'passed': True},
            {'profile_id': 'M2', 'passed': True},
            {'profile_id': 'S2', 'passed': False},
        ]},
        's2_mesh20_diagnostic': {
            'passed': True,
            'thresholds_changed': False,
            'diagnostic_conclusion': 'supported_numerical_dispersion',
        },
    })
    assert release_gate['passed']
    summary = RUNNER._metric_summary([0.01, 0.02, 0.03])
    assert summary['count'] == 3 and summary['median'] == 0.02
    print('test_g1r_reference_runner: 12/12 plan and independent transfer ok')


if __name__ == '__main__':
    main()
