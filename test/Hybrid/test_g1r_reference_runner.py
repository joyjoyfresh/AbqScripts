# -*- coding: utf-8 -*-
"""G1r参考库批处理的纯Python测试。"""

from __future__ import print_function

import importlib.util
import json
import os
import tempfile
import threading

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

    with tempfile.TemporaryDirectory() as progress_root:
        model_log = os.path.join(progress_root, 'slope_frame_ssi_full_v2.log')
        with open(model_log, 'wb') as handle:
            handle.write(
                '2026-01-01 [INFO] model-a 分析步已创建, 时长=10.00(含尾段 2.00)\n'.encode('utf-8')
            )
        RUNNER.write_json(
            os.path.join(progress_root, RUNNER.CONFIG_FILENAME),
            {'tssi_cfg': {'enable': False}},
        )
        sta_path = os.path.join(progress_root, 'job-model-a.sta')
        with open(sta_path, 'wb') as handle:
            handle.write(
                b'   1     1   1     0     1     1  1.00       1.00       0.100000\n'
                b'   1     2   1     0     1     1  2.50       2.50       0.100000\n'
            )
        started_at = os.path.getmtime(sta_path)
        messages = RUNNER._read_sta_job_progress(progress_root, started_at, model_log)
        assert len(messages) == 1
        assert 'job-model-a' in messages[0]
        assert '2.500 秒/共 10.000 秒' in messages[0]
        assert '25.0%' in messages[0]

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
    initial_gate = {
        'passed': True,
        'status': 'passed',
        'next_action': 'rerun_all_real_cases_with_incremental_initial_state',
    }
    with tempfile.TemporaryDirectory() as temp_root:
        release_root = os.path.join(temp_root, 'release')
        initial_root = os.path.join(temp_root, 'initial')
        os.makedirs(release_root)
        os.makedirs(initial_root)
        RUNNER.write_json(
            os.path.join(release_root, 'g1r_real_release_gate.json'),
            release_gate,
        )
        RUNNER.write_json(
            os.path.join(release_root, 'run_manifest.json'),
            {
                'git_commit': 'release-test',
                'stage_gates': {'real_release': release_gate},
            },
        )
        RUNNER.write_json(
            os.path.join(initial_root, 'g1r_real_init_diagnostic_gate.json'),
            initial_gate,
        )
        RUNNER.write_json(
            os.path.join(initial_root, 'run_manifest.json'),
            {
                'git_commit': 'initial-test',
                'stage_gates': {'real_init_diagnostic': initial_gate},
            },
        )
        evidence, imported_release, imported_initial = (
            RUNNER.import_real_rerun_evidence(release_root, initial_root)
        )
        verified_release, verified_initial = (
            RUNNER.verify_real_rerun_evidence(evidence)
        )
        assert imported_release == verified_release == release_gate
        assert imported_initial == verified_initial == initial_gate
        assert evidence['release']['gate_file_sha256']
        rerun_release = RUNNER.build_real_rerun_release_gate(
            imported_release, imported_initial, evidence,
        )
        assert rerun_release['passed']
        assert rerun_release['formal_initial_state_mode'] == 'incremental'
        with open(
            os.path.join(initial_root, 'run_manifest.json'),
            'r', encoding='utf-8',
        ) as handle:
            changed_manifest = json.load(handle)
        changed_manifest['updated_at'] = 'changed'
        RUNNER.write_json(
            os.path.join(initial_root, 'run_manifest.json'), changed_manifest,
        )
        try:
            RUNNER.verify_real_rerun_evidence(evidence)
        except ValueError as error:
            assert 'source_manifest_sha256' in str(error)
        else:
            raise AssertionError('证据清单变化后应拒绝正式运行')
    summary = RUNNER._metric_summary([0.01, 0.02, 0.03])
    assert summary['count'] == 3 and summary['median'] == 0.02

    with tempfile.TemporaryDirectory() as pipeline_root:
        originals = {
            'MAX_WORKERS': RUNNER.MAX_WORKERS,
            'POSTPROCESS_WORKERS': RUNNER.POSTPROCESS_WORKERS,
            'create_case_folder': RUNNER.create_case_folder,
            'run_scripts_in_folder': RUNNER.run_scripts_in_folder,
            'write_run_manifest': RUNNER.write_run_manifest,
        }
        post_started = threading.Event()
        stage_calls = []
        calls_lock = threading.Lock()

        def fake_create(case, root_dir):
            folder = os.path.join(root_dir, case['case_id'])
            os.makedirs(folder, exist_ok=True)
            return folder

        def fake_run(folder, _sequence, step_offset=0, stage_label='stage'):
            case_id = os.path.basename(folder)
            with calls_lock:
                stage_calls.append((case_id, stage_label, step_offset))
            if stage_label == 'model' and case_id == 'case-b':
                return post_started.wait(2.0)
            if stage_label == 'postprocess' and case_id == 'case-a':
                post_started.set()
            return True

        try:
            RUNNER.MAX_WORKERS = 1
            RUNNER.POSTPROCESS_WORKERS = 1
            RUNNER.create_case_folder = fake_create
            RUNNER.run_scripts_in_folder = fake_run
            RUNNER.write_run_manifest = lambda *args, **kwargs: None
            pipeline_cases = [{'case_id': 'case-a'}, {'case_id': 'case-b'}]
            pipeline_statuses = {'case-a': 'planned', 'case-b': 'planned'}
            failed = RUNNER.run_case_group(
                pipeline_root, pipeline_cases, pipeline_cases,
                pipeline_statuses, {},
            )
            assert failed == []
            assert post_started.is_set()
            assert set(pipeline_statuses.values()) == {'pipeline_passed'}
            assert ('case-a', 'postprocess', 1) in stage_calls
        finally:
            RUNNER.MAX_WORKERS = originals['MAX_WORKERS']
            RUNNER.POSTPROCESS_WORKERS = originals['POSTPROCESS_WORKERS']
            RUNNER.create_case_folder = originals['create_case_folder']
            RUNNER.run_scripts_in_folder = originals['run_scripts_in_folder']
            RUNNER.write_run_manifest = originals['write_run_manifest']

    print('test_g1r_reference_runner: plan, transfer, evidence and pipeline ok')


if __name__ == '__main__':
    main()
