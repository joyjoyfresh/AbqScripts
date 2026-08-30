# -*- coding: utf-8 -*-
"""运行 X002-A（P061 两层坡），不调用跨软件比较脚本。

运行形式：
    python Run/Auto_ch4/Autorun_x002_a.py [输出根目录]

缺省输出根目录为 ``Run/cross_solver_X/abaqus``，工况目录固定为 ``X002-A``。
本入口保留 ODB、求解日志、原始时程和全场快照，不执行过程文件清理。
跨软件比较（vs X002-S）在求解完成后另行执行。
"""

from __future__ import absolute_import, print_function

import datetime
import importlib.util
import json
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
RUN_HELPER = os.path.join(REPO_ROOT, 'Run', 'Auto_ch4', 'Autorun_ch4_sp_02_H_v1.py')
PARAMETER_FILE = os.path.join(REPO_ROOT, 'Run', 'Auto_ch4', 'cross_solver', 'x_validation_parameters.json')
MODEL_SCRIPT = os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py')
SURFACE_POSTPROCESS = os.path.join(REPO_ROOT, 'Postprocess', 'Postprocess_All_surface_v2.py')
WAVEFIELD_POSTPROCESS = os.path.join(
    REPO_ROOT, 'Run', 'Auto_ch4', 'cross_solver', 'extract_x001_a_wavefield.py')
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'cross_solver_X', 'abaqus')


def load_parameters():
    """读取公共参数表。"""
    with open(PARAMETER_FILE, 'r', encoding='utf-8') as handle:
        params = json.load(handle)
    case = params.get('cases', {}).get('X002-A')
    if not isinstance(case, dict):
        raise RuntimeError('公共参数表缺少X002-A定义，请先重新运行输入生成脚本')
    return params, case


def build_config(params, case):
    """从公共参数表组装 X002-A 的 Abaqus 配置（含覆盖层）。"""
    geometry = params['geometry']
    bedrock = params['materials']['bedrock']
    cover = params['materials']['cover']
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
            'layers': [
                {
                    'name': 'cover',
                    'vs': float(cover['vs']),
                    'poisson_ratio': float(cover['poisson_ratio']),
                    'density': float(cover['density']),
                    'thickness': float(cover['thickness_below_crest']),
                },
            ],
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
            'constant_xi': float(cover['target_xi']),
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
        'run_cfg': {
            'surface_only': True,
            'critical_angle_check': True,
            'wave_files': [os.path.basename(
                os.path.join(REPO_ROOT,
                             params['pulse']['abaqus_acceleration_file'].replace('/', os.sep)))],
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


def load_pipeline():
    """按绝对路径加载现有单工况批处理基础设施。"""
    spec = importlib.util.spec_from_file_location('x002_a_pipeline', RUN_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    """配置并调用正式批处理基础设施。"""
    params, case = load_parameters()
    config = build_config(params, case)
    pipeline = load_pipeline()
    output_root = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_ROOT
    pipeline.ROOT_DIR = output_root
    pipeline.FOLDER_PREFIX = ''
    pipeline.MAX_WORKERS = 1
    pipeline.POSTPROCESS_WORKERS = 1
    pipeline.DELETE_FILE_TYPES = []
    pipeline.TERMINAL_PROGRESS_POLL_SECONDS = 60.0
    pipeline.PARAMETER_CASES = [{'name': 'X002-A', 'config': config}]
    pipeline.MODEL_SCRIPT_SEQUENCE = [MODEL_SCRIPT]
    pipeline.CASE_POSTPROCESS_SCRIPT_SEQUENCE = [
        SURFACE_POSTPROCESS,
        WAVEFIELD_POSTPROCESS,
    ]
    pipeline.SCRIPT_SEQUENCE = (
        pipeline.MODEL_SCRIPT_SEQUENCE + pipeline.CASE_POSTPROCESS_SCRIPT_SEQUENCE +
        [os.path.join(REPO_ROOT,
                      params['pulse']['abaqus_acceleration_file'].replace('/', os.sep)),
         PARAMETER_FILE]
    )
    pipeline.ABAQUS_PYTHON_SCRIPTS = {
        os.path.basename(SURFACE_POSTPROCESS),
        os.path.basename(WAVEFIELD_POSTPROCESS),
    }
    pipeline.ABAQUS_SCRIPTS = pipeline.ABAQUS_CAE_SCRIPTS | pipeline.ABAQUS_PYTHON_SCRIPTS
    os.environ['ABQSCRIPTS_REPO_ROOT'] = REPO_ROOT
    pipeline.main()


if __name__ == '__main__':
    main()
