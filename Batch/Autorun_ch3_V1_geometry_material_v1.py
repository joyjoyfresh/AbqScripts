# -*- coding: utf-8 -*-
"""V1：12 个代表性几何、材料和观测生成工况的只建模审计。"""

from __future__ import print_function

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'
MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_V1_geometry_material')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write('\n')


def next_run_dir(root):
    if not os.path.isdir(root):
        os.makedirs(root)
    numbers = [int(name[4:]) for name in os.listdir(root)
               if name.startswith('run-') and name[4:].isdigit()]
    path = os.path.join(root, 'run-%03d' % ((max(numbers) if numbers else 0) + 1))
    os.makedirs(path)
    return path


def run_model(case_dir):
    log_path = os.path.join(case_dir, 'autorun_model_only.log')
    with open(log_path, 'wb') as fh:
        fh.write(('命令：%s\n工作目录：%s\n' % (ABAQUS_CMD, case_dir)).encode('utf-8'))
        result = subprocess.run([ABAQUS_CMD, 'cae', 'noGUI=slope_frame_ssi_full_v2.py'],
                                cwd=case_dir, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=3600, check=False)
    if result.returncode != 0:
        raise RuntimeError('Abaqus 建模失败：%s' % case_dir)


def build_config(case):
    return {
        'material_cfg': {'angle': case.get('wave_angle', 0.0),
                         'surface_geometry': case['surface_geometry'], 'layers': case['layers']},
        'geometry_cfg': {'slope_height': case['slope_height'], 'slope_angle': case['slope_angle'],
                         'crest_window': 2.0, 'toe_window': 2.0, 'side_clearance': 0.2,
                         'base_depth': case.get('base_depth', 3.0)},
        'damping_cfg': {'enable': False, 'fc': 4.0},
        'mesh_cfg': {'size': 8.0, 'auto': False, 'elem': 'CPE4R', 'graded': False},
        'time_cfg': {'check': True, 'tail_seconds': 0.0},
        'freefield_cfg': {'engine': 'fd', 'include_damping': False},
        'run_cfg': {'surface_only': True, 'critical_angle_check': True,
                    'validation_geometry': 'slope', 'submit_jobs': False,
                    'wave_files': ['ricker_wavelet_4Hz.txt']},
        'tssi_cfg': {'enable': False, 'scene': 'freefield', 'nonlinear': False, 'gravity': 'off'},
    }


def audit_case(case_dir, case):
    audit_path = os.path.join(case_dir, 'geometry_validation.json')
    meta_path = os.path.join(case_dir, 'case_meta.json')
    if not os.path.isfile(audit_path) or not os.path.isfile(meta_path):
        raise RuntimeError('缺少几何审计或 case_meta：%s' % case_dir)
    with open(audit_path, encoding='utf-8') as fh:
        audit = json.load(fh)
    with open(meta_path, encoding='utf-8') as fh:
        meta = json.load(fh)
    assert audit.get('validation_geometry') == 'slope'
    assert audit['top_surface']['y_range'] > 1.0
    assert audit['node_count'] > 0 and audit['element_count'] > 0
    assert all(int(audit['boundary_node_counts'].get(name, 0)) > 0
               for name in ('Left_boundary', 'Right_boundary', 'Bottom_boundary', 'TOP_SURFACE'))
    assert abs(float(meta['geometry']['H_minus_h']) - float(case['slope_height'])) < 1.0e-8
    assert abs(float(meta['geometry']['i']) - float(case['slope_angle'])) < 1.0e-8
    assert meta.get('surface_geometry') == case['surface_geometry']
    assert int(meta['derived']['n_finite_layers']) == len(case['layers'])
    assert not [name for name in os.listdir(case_dir) if name.lower().endswith('.odb')]
    return {'name': case['name'], 'node_count': audit['node_count'], 'element_count': audit['element_count'],
            'top_y_range': audit['top_surface']['y_range'], 'n_layers': len(case['layers']),
            'surface_geometry': case['surface_geometry']}


CASES = [
    {'name': 'case-h25-i30-homogeneous', 'slope_height': 25.0, 'slope_angle': 30.0, 'surface_geometry': 'horizontal', 'layers': []},
    {'name': 'case-h50-i45-homogeneous', 'slope_height': 50.0, 'slope_angle': 45.0, 'surface_geometry': 'horizontal', 'layers': []},
    {'name': 'case-h100-i60-homogeneous', 'slope_height': 100.0, 'slope_angle': 60.0, 'surface_geometry': 'horizontal', 'layers': []},
    {'name': 'case-h25-i30-single-terrain', 'slope_height': 25.0, 'slope_angle': 30.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 800.0, 'poisson_ratio': 0.3, 'density': 2200.0, 'thickness': 10.0}]},
    {'name': 'case-h50-i45-single-terrain', 'slope_height': 50.0, 'slope_angle': 45.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 600.0, 'poisson_ratio': 0.3, 'density': 2100.0, 'thickness': 25.0}]},
    {'name': 'case-h100-i60-single-terrain', 'slope_height': 100.0, 'slope_angle': 60.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2000.0, 'thickness': 50.0}]},
    {'name': 'case-h25-i30-double-terrain', 'slope_height': 25.0, 'slope_angle': 30.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2000.0, 'thickness': 5.0}, {'name': 'overlying', 'vs': 800.0, 'poisson_ratio': 0.3, 'density': 2200.0, 'thickness': 10.0}]},
    {'name': 'case-h50-i45-double-terrain', 'slope_height': 50.0, 'slope_angle': 45.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2000.0, 'thickness': 10.0}, {'name': 'overlying', 'vs': 800.0, 'poisson_ratio': 0.3, 'density': 2200.0, 'thickness': 20.0}]},
    {'name': 'case-h100-i60-double-terrain', 'slope_height': 100.0, 'slope_angle': 60.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2000.0, 'thickness': 20.0}, {'name': 'overlying', 'vs': 800.0, 'poisson_ratio': 0.3, 'density': 2200.0, 'thickness': 40.0}]},
    {'name': 'case-h50-i45-thin-soft', 'slope_height': 50.0, 'slope_angle': 45.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 250.0, 'poisson_ratio': 0.3, 'density': 1900.0, 'thickness': 2.0}]},
    {'name': 'case-h100-i60-same-material-bands', 'slope_height': 100.0, 'slope_angle': 60.0, 'surface_geometry': 'terrain', 'layers': [{'name': 'surface', 'vs': 2000.0, 'poisson_ratio': 0.3, 'density': 2500.0, 'thickness': 20.0}, {'name': 'overlying', 'vs': 2000.0, 'poisson_ratio': 0.3, 'density': 2500.0, 'thickness': 30.0}]},
    {'name': 'case-h25-i60-horizontal-layers', 'slope_height': 25.0, 'slope_angle': 60.0, 'surface_geometry': 'horizontal', 'layers': [{'name': 'surface', 'vs': 600.0, 'poisson_ratio': 0.3, 'density': 2100.0, 'thickness': 8.0}, {'name': 'overlying', 'vs': 1000.0, 'poisson_ratio': 0.3, 'density': 2300.0, 'thickness': 12.0}]},
]


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    run_dir = next_run_dir(root)
    manifest = {'unit': 'V1', 'run_dir': run_dir, 'case_count': len(CASES),
                'cases': CASES, 'created_at': datetime.datetime.now().isoformat(),
                'source_sha256': {os.path.basename(MODEL_SOURCE): sha256(MODEL_SOURCE),
                                  os.path.basename(WAVE_SOURCE): sha256(WAVE_SOURCE)}}
    write_json(os.path.join(run_dir, 'v1_run_manifest.json'), manifest)
    reports = []
    try:
        for case in CASES:
            case_dir = os.path.join(run_dir, case['name'])
            os.makedirs(case_dir)
            shutil.copy2(MODEL_SOURCE, os.path.join(case_dir, os.path.basename(MODEL_SOURCE)))
            shutil.copy2(WAVE_SOURCE, os.path.join(case_dir, os.path.basename(WAVE_SOURCE)))
            write_json(os.path.join(case_dir, 'case_config.json'), build_config(case))
            run_model(case_dir)
            reports.append(audit_case(case_dir, case))
        report = {'status': 'passed', 'unit': 'V1', 'run_dir': run_dir,
                  'case_count': len(reports), 'cases': reports,
                  'finished_at': datetime.datetime.now().isoformat()}
    except Exception as exc:
        report = {'status': 'failed', 'unit': 'V1', 'run_dir': run_dir,
                  'case_count_completed': len(reports), 'error': str(exc),
                  'failed_at': datetime.datetime.now().isoformat()}
        write_json(os.path.join(run_dir, 'v1_validation_report.json'), report)
        raise
    write_json(os.path.join(run_dir, 'v1_validation_report.json'), report)
    print('V1 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
