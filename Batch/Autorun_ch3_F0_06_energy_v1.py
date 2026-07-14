# -*- coding: utf-8 -*-
"""F0-6：单一平坦均质控制工况的 freefield 能量历史闭环。"""

from __future__ import print_function

import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_F0_06_energy')
MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')
POST_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Postprocess_All_surface_v2.py')
COLLECT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Collect_All_results_v2.py')
PLOT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Plot_Hybrid_surface_v2.py')
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')
CASE_NAME = 'case-flat-homogeneous-energy'


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


def run_command(command, cwd, log_path, timeout=3600):
    env = os.environ.copy()
    if os.path.abspath(command[0]) == os.path.abspath(sys.executable):
        env['PYTHONIOENCODING'] = 'utf-8'
    with open(log_path, 'wb') as fh:
        fh.write(('命令：%s\n工作目录：%s\n\n' % (' '.join(command), cwd)).encode('utf-8'))
        result = subprocess.run(command, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError('步骤失败，退出码=%s，日志=%s' % (result.returncode, log_path))


def prepare_case(case_dir):
    for source in (MODEL_SOURCE, POST_SOURCE, COLLECT_SOURCE, PLOT_SOURCE, WAVE_SOURCE, ABAQUS_CMD):
        if not os.path.isfile(source):
            raise RuntimeError('缺少输入：%s' % source)
    if not os.path.isdir(case_dir):
        os.makedirs(case_dir)
    for source in (MODEL_SOURCE, POST_SOURCE, WAVE_SOURCE):
        shutil.copy2(source, os.path.join(case_dir, os.path.basename(source)))
    config = {
        'material_cfg': {'angle': 0.0, 'layers': [], 'surface_geometry': 'horizontal'},
        'geometry_cfg': {'slope_height': 50.0, 'slope_angle': 45.0, 'crest_window': 2.0,
                         'toe_window': 2.0, 'side_clearance': 2.0, 'base_depth': 3.0},
        'damping_cfg': {'enable': False, 'fc': 4.0},
        'mesh_cfg': {'size': 8.0, 'auto': False, 'elem': 'CPE4R', 'graded': False},
        'time_cfg': {'check': True, 'tail_seconds': 1.0},
        'freefield_cfg': {'engine': 'fd', 'include_damping': False},
        'run_cfg': {'surface_only': True, 'critical_angle_check': True,
                    'validation_geometry': 'flat', 'submit_jobs': True,
                    'wave_files': ['ricker_wavelet_4Hz.txt']},
        'qa_cfg': {'required': ['energy'], 'artificial_energy_ratio_tol': 0.05,
                   'energy_residual_tol': 0.05},
        'tssi_cfg': {'enable': False, 'scene': 'freefield', 'nonlinear': False, 'gravity': 'off'},
    }
    write_json(os.path.join(case_dir, 'case_config.json'), config)
    return config


def inspect_energy(case_dir):
    npz_path = os.path.join(case_dir, 'surface_results.npz')
    if not os.path.isfile(npz_path):
        raise RuntimeError('缺少 surface_results.npz')
    package = np.load(npz_path)
    try:
        summary = json.loads(package['surface_summary_json'].item().decode('utf-8'))
        record = summary['records'][0]
        raw_prefix = 'raw_%s_' % record['record'].replace('.', '_')
        qa_key = raw_prefix + 'qa_energy_json'
        if qa_key not in package.files:
            raise RuntimeError('NPZ 缺少 %s' % qa_key)
        energy = json.loads(package[qa_key].item().decode('utf-8'))
        energy_keys = [key for key in package.files if key.startswith(raw_prefix + 'energy_')]
        if energy.get('status') != 'passed' or len(energy_keys) < 5:
            raise RuntimeError('能量 QA 未通过或变量不足：%s' % json.dumps(energy, ensure_ascii=True))
        return {'record': record['record'], 'energy': energy, 'energy_key_count': len(energy_keys),
                'npz': npz_path}
    finally:
        package.close()


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    run_dir = next_run_dir(root)
    case_dir = os.path.join(run_dir, CASE_NAME)
    config = prepare_case(case_dir)
    manifest = {'unit': 'F0-6', 'run_dir': run_dir, 'case_name': CASE_NAME,
                'created_at': datetime.datetime.now().isoformat(), 'case_config': config,
                'source_sha256': {os.path.basename(path): sha256(path)
                                  for path in (MODEL_SOURCE, POST_SOURCE, COLLECT_SOURCE, PLOT_SOURCE, WAVE_SOURCE)}}
    write_json(os.path.join(run_dir, 'f0_6_run_manifest.json'), manifest)
    try:
        run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], case_dir,
                    os.path.join(case_dir, 'autorun_01_model.log'))
        run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POST_SOURCE)], case_dir,
                    os.path.join(case_dir, 'autorun_02_postprocess.log'))
        for source in (COLLECT_SOURCE, PLOT_SOURCE):
            shutil.copy2(source, os.path.join(run_dir, os.path.basename(source)))
        run_command([sys.executable, os.path.basename(COLLECT_SOURCE), run_dir], run_dir,
                    os.path.join(run_dir, 'autorun_03_collect.log'))
        run_command([sys.executable, os.path.basename(PLOT_SOURCE), run_dir], run_dir,
                    os.path.join(run_dir, 'autorun_04_plot.log'))
        report = inspect_energy(case_dir)
        report.update({'status': 'passed', 'unit': 'F0-6', 'run_dir': run_dir,
                       'finished_at': datetime.datetime.now().isoformat()})
    except Exception as exc:
        report = {'status': 'failed', 'unit': 'F0-6', 'run_dir': run_dir, 'error': str(exc),
                  'failed_at': datetime.datetime.now().isoformat()}
        write_json(os.path.join(run_dir, 'f0_6_validation_report.json'), report)
        raise
    write_json(os.path.join(run_dir, 'f0_6_validation_report.json'), report)
    print('F0-6 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
