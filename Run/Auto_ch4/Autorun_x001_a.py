# -*- coding: utf-8 -*-
"""运行 X001-A，并在完成后与既有 X001-S 作统一口径比较。

运行形式：
    python Run/Auto_ch4/Autorun_x001_a.py [输出根目录]

缺省输出根目录为 ``Run/cross_solver_X/abaqus``，工况目录固定为 ``X001-A``。
本入口保留 ODB、求解日志、原始时程和全场快照，不执行过程文件清理。
"""

from __future__ import absolute_import, print_function

import datetime
import hashlib
import importlib.util
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RUN_HELPER = os.path.join(REPO_ROOT, 'Run', 'Auto_ch4', 'Autorun_ch4_sp_02_H_v1.py')
PARAMETER_FILE = os.path.join(REPO_ROOT, 'Run', 'Auto_ch4', 'x_validation_parameters.json')
MODEL_SCRIPT = os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py')
SURFACE_POSTPROCESS = os.path.join(REPO_ROOT, 'Postprocess', 'Postprocess_All_surface_v2.py')
WAVEFIELD_POSTPROCESS = os.path.join(
    REPO_ROOT, 'Run', 'Auto_ch4', 'extract_x001_a_wavefield.py')
COMPARISON_SCRIPT = os.path.join(
    REPO_ROOT, 'Run', 'Auto_ch4', 'compare_x001_a_s.py')
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'cross_solver_X', 'abaqus')


def mark_evaluated(output_root):
    """比较脚本成功写出评价结果后，将批次清单状态更新为 evaluated。"""
    case_dir = os.path.join(output_root, 'X001-A')
    metrics_path = os.path.join(
        case_dir, 'comparison', 'x001_comparison_metrics.json')
    manifest_path = os.path.join(output_root, 'run_manifest.json')
    if not os.path.isfile(metrics_path):
        raise RuntimeError('缺少X001评价结果: {}'.format(metrics_path))
    if not os.path.isfile(manifest_path):
        raise RuntimeError('缺少批次运行清单: {}'.format(manifest_path))
    with open(metrics_path, 'r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    if metrics.get('status') != 'evaluated':
        raise RuntimeError('X001评价结果状态不是evaluated: {}'.format(metrics.get('status')))
    with open(manifest_path, 'r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    case_entry = manifest.get('cases', {}).get('X001-A')
    if not isinstance(case_entry, dict):
        raise RuntimeError('批次运行清单缺少X001-A条目')
    now = datetime.datetime.now().isoformat()
    case_entry['status'] = 'evaluated'
    case_entry['evaluation_result'] = (
        'formal_gates_met'
        if bool(metrics.get('formal_gates_met'))
        else 'formal_gates_not_met'
    )
    case_entry['evaluation_file'] = os.path.relpath(metrics_path, output_root).replace(os.sep, '/')
    case_entry['updated_at'] = now
    manifest['updated_at'] = now
    with open(manifest_path, 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)


def load_pipeline():
    """按绝对路径加载现有单工况批处理基础设施。"""
    spec = importlib.util.spec_from_file_location('x001_a_pipeline', RUN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_parameters():
    """读取公共参数表并校验公共脉冲哈希。"""
    with open(PARAMETER_FILE, 'r', encoding='utf-8') as handle:
        params = json.load(handle)
    case = params.get('cases', {}).get('X001-A')
    if not isinstance(case, dict):
        raise RuntimeError('公共参数表缺少X001-A定义，请先重新运行输入生成脚本')
    relative_wave = params['pulse']['abaqus_acceleration_file'].replace('/', os.sep)
    wave_path = os.path.join(REPO_ROOT, relative_wave)
    if not os.path.isfile(wave_path):
        raise RuntimeError('公共脉冲不存在: {}'.format(wave_path))
    with open(wave_path, 'rb') as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    expected = str(params['pulse']['abaqus_acceleration_sha256'])
    if digest != expected:
        raise RuntimeError('公共脉冲SHA-256不一致: 实际{}，参数表{}'.format(digest, expected))
    return params, case, wave_path


def build_config(params, case, wave_path):
    """仅从公共参数表组装 X001-A 的 Abaqus 配置。"""
    geometry = params['geometry']
    bedrock = params['materials']['bedrock']
    pulse = params['pulse']
    rock_points = params['observations']['rock_checks']
    return {
        'material_cfg': {
            'angle': float(params['physics']['incidence_angle_from_vertical_deg']),
            'surface_geometry': 'horizontal',
            'bedrock': {
                'vs': float(bedrock['vs']),
                'poisson_ratio': float(bedrock['poisson_ratio']),
                'density': float(bedrock['density']),
            },
            'layers': [],
        },
        'geometry_cfg': {
            'slope_height': float(geometry['slope_height']),
            'slope_angle': float(geometry['slope_angle_deg']),
            'crest_window': float(geometry['crest_window_h']),
            'toe_window': float(geometry['toe_window_h']),
            'side_clearance': float(geometry['side_clearance_h']),
            'base_depth': float(geometry['base_depth_h']),
        },
        'damping_cfg': {
            'enable': True,
            'method': 'rayleigh',
            'constant_xi': 0.03,
            'bedrock_xi': float(bedrock['target_xi']),
            'fc': float(pulse['f0_hz']),
            'f1_factor': 0.5,
            'f2_factor': 2.5,
            'anchor': 'perband',
        },
        'mesh_cfg': {
            'size': float(case['mesh_size']),
            'auto': bool(case['mesh_auto']),
            'elems_per_wavelength': 10,
            'fmax_factor': 2.5,
            'elem': 'CPE4R',
            'graded': bool(case['mesh_graded']),
        },
        'time_cfg': {
            'check': True,
            'min_steps_per_fmax_period': 20,
            'tail_seconds': float(pulse['abaqus_tail_seconds']),
        },
        'freefield_cfg': {
            'engine': 'fd',
            'include_damping': True,
            'reference_field_mode': 'global_upper',
            'bottom_ymax_mode': 'local',
            'initial_state_mode': 'incremental',
            'phase_origin_x': float(pulse['abaqus_phase_origin_x_m']),
        },
        'run_cfg': {
            'surface_only': True,
            'critical_angle_check': True,
            'wave_files': [os.path.basename(wave_path)],
            'validation_geometry': 'slope',
            'submit_jobs': True,
            'job_progress_interval_seconds': 300.0,
            'validation_points': [
                {'name': 'R%04d' % (index + 1), 'x': float(point[0]), 'y': float(point[1])}
                for index, point in enumerate(rock_points)
            ],
            'validation_point_tolerance': float(case['validation_point_tolerance']),
            'full_field_frequency': int(case['full_field_frequency']),
            'frf_cfg': {'fmax_hz': float(pulse['acceleration_effective_band_5pct_hz'][1])},
            'response_spectrum_cfg': {'enable': False},
        },
        'eql_cfg': {'enable': False},
        'tssi_cfg': {
            'enable': False,
            'scene': 'freefield',
            'nonlinear': False,
            'gravity': 'off',
        },
    }


def main():
    """配置并调用正式批处理基础设施。"""
    params, case, wave_path = load_parameters()
    config = build_config(params, case, wave_path)
    pipeline = load_pipeline()
    output_root = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOT
    pipeline.ROOT_DIR = output_root
    pipeline.FOLDER_PREFIX = ''
    pipeline.MAX_WORKERS = 1
    pipeline.POSTPROCESS_WORKERS = 1
    pipeline.DELETE_FILE_TYPES = []
    pipeline.TERMINAL_PROGRESS_POLL_SECONDS = 60.0
    pipeline.PARAMETER_CASES = [{'name': 'X001-A', 'config': config}]
    pipeline.MODEL_SCRIPT_SEQUENCE = [MODEL_SCRIPT]
    pipeline.CASE_POSTPROCESS_SCRIPT_SEQUENCE = [
        SURFACE_POSTPROCESS,
        WAVEFIELD_POSTPROCESS,
        COMPARISON_SCRIPT,
    ]
    pipeline.SCRIPT_SEQUENCE = (
        pipeline.MODEL_SCRIPT_SEQUENCE + pipeline.CASE_POSTPROCESS_SCRIPT_SEQUENCE +
        [wave_path, PARAMETER_FILE]
    )
    pipeline.ABAQUS_PYTHON_SCRIPTS = {
        os.path.basename(SURFACE_POSTPROCESS),
        os.path.basename(WAVEFIELD_POSTPROCESS),
    }
    pipeline.ABAQUS_SCRIPTS = pipeline.ABAQUS_CAE_SCRIPTS | pipeline.ABAQUS_PYTHON_SCRIPTS
    os.environ['ABQSCRIPTS_REPO_ROOT'] = REPO_ROOT
    pipeline.main()
    mark_evaluated(output_root)


if __name__ == '__main__':
    main()
