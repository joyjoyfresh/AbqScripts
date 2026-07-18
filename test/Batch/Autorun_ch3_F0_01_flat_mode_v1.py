# -*- coding: utf-8 -*-
"""F0-1：只建模不求解的显式平场验证模式回归。

该脚本使用 Hybrid v2 主脚本生成矩形平场、材料界面和边界节点集，
通过 ``submit_jobs=False`` 禁止提交 Abaqus 求解；产物写入 ``test/Abaqus``。
"""

from __future__ import print_function

import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'
MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py')
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'test', 'Abaqus', 'ch3_F0_01_flat_mode')


def sha256(path):
    """计算文件 SHA-256，固化本次回归所用输入。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """以 UTF-8 写出审计 JSON。"""
    with io_open(path, 'w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def io_open(path, mode):
    """兼容 Python 3 的 UTF-8 文本打开。"""
    return open(path, mode, encoding='utf-8')


def next_run_dir(unit_root):
    """创建不覆盖既有结果的新 run 目录。"""
    if not os.path.isdir(unit_root):
        os.makedirs(unit_root)
    numbers = [int(name[4:]) for name in os.listdir(unit_root)
               if name.startswith('run-') and name[4:].isdigit()]
    run_dir = os.path.join(unit_root, 'run-%03d' % ((max(numbers) if numbers else 0) + 1))
    os.makedirs(run_dir)
    return run_dir


def run_command(case_dir, log_path):
    """调用 Abaqus 只建模模式并保存完整控制台日志。"""
    command = [ABAQUS_CMD, 'cae', 'noGUI=slope_frame_ssi_full_v2.py']
    with open(log_path, 'wb') as handle:
        header = ('命令：%s\n工作目录：%s\n\n' % (' '.join(command), case_dir)).encode('utf-8')
        handle.write(header)
        result = subprocess.run(command, cwd=case_dir, stdout=handle,
                                stderr=subprocess.STDOUT, check=False,
                                timeout=3600)
    if result.returncode != 0:
        raise RuntimeError('Abaqus 建模回归失败，退出码=%s，日志=%s' % (result.returncode, log_path))


def _run_variant(root_dir, variant):
    """运行一个均质或成层平场建模回归并返回审计摘要。"""
    case_name = variant['name']
    case_dir = os.path.join(root_dir, case_name)
    if not os.path.isdir(case_dir):
        os.makedirs(case_dir)
    model_name = os.path.basename(MODEL_SOURCE)
    wave_name = os.path.basename(WAVE_SOURCE)
    shutil.copy2(MODEL_SOURCE, os.path.join(case_dir, model_name))
    shutil.copy2(WAVE_SOURCE, os.path.join(case_dir, wave_name))
    config = {
        'material_cfg': {
            'angle': 0.0,
            'surface_geometry': variant['surface_geometry'],
            'layers': variant['layers'],
        },
        'geometry_cfg': {
            'slope_height': 50.0,
            'slope_angle': 45.0,
            'crest_window': 2.0,
            'toe_window': 2.0,
            'side_clearance': 0.2,
            'base_depth': variant['base_depth'],
        },
        'damping_cfg': {'enable': False, 'fc': 4.0},
        'mesh_cfg': {'size': 8.0, 'auto': False, 'elem': 'CPE4R', 'graded': False},
        'time_cfg': {'check': True, 'tail_seconds': 0.0},
        'freefield_cfg': {'engine': 'fd', 'include_damping': False},
        'run_cfg': {
            'surface_only': True,
            'critical_angle_check': True,
            'wave_files': [wave_name],
            'validation_geometry': variant.get('validation_geometry', 'flat'),
            'submit_jobs': False,
        },
        'tssi_cfg': {'enable': False, 'scene': 'freefield', 'nonlinear': False, 'gravity': 'off'},
    }
    write_json(os.path.join(case_dir, 'case_config.json'), config)
    run_command(case_dir, os.path.join(case_dir, 'autorun_model_only.log'))
    audit_path = os.path.join(case_dir, 'geometry_validation.json')
    meta_path = os.path.join(case_dir, 'case_meta.json')
    if not os.path.isfile(audit_path) or not os.path.isfile(meta_path):
        raise RuntimeError('%s 缺少 geometry_validation.json 或 case_meta.json' % case_name)
    with io_open(audit_path, 'r') as handle:
        audit = json.load(handle)
    with io_open(meta_path, 'r') as handle:
        meta = json.load(handle)
    expected_mode = variant.get('validation_geometry', 'flat')
    if audit.get('validation_geometry') != expected_mode or meta.get('validation_geometry') != expected_mode:
        raise RuntimeError('%s 的审计或 case_meta 模式错误，期望=%s' % (case_name, expected_mode))
    top_range = float(audit['top_surface']['y_range'])
    if expected_mode == 'flat' and top_range > 1.0e-5:
        raise RuntimeError('%s 平场顶面 y_range 超出阈值' % case_name)
    if expected_mode == 'slope' and top_range <= 1.0:
        raise RuntimeError('%s 默认坡地哨兵的地表高差异常小，可能误走 flat 分支' % case_name)
    for key in ('Left_boundary', 'Right_boundary', 'Bottom_boundary', 'TOP_SURFACE'):
        if int(audit['boundary_node_counts'].get(key, 0)) <= 0:
            raise RuntimeError('%s 边界节点集为空：%s' % (case_name, key))
    odbs = [name for name in os.listdir(case_dir) if name.lower().endswith('.odb')]
    if odbs:
        raise RuntimeError('%s 的 submit_jobs=False 仍产生 ODB：%s' % (case_name, odbs))
    return {'name': case_name, 'audit': audit, 'case_meta': meta}


def main():
    """准备均质、成层平场和默认坡地哨兵三类工况，运行建模回归并写出验收报告。"""
    unit_root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    root_dir = next_run_dir(unit_root)
    for path, label in ((MODEL_SOURCE, 'Hybrid v2 建模脚本'),
                        (WAVE_SOURCE, '4 Hz Ricker 输入波')):
        if not os.path.isfile(path):
            raise RuntimeError('%s不存在：%s' % (label, path))
    model_name = os.path.basename(MODEL_SOURCE)
    wave_name = os.path.basename(WAVE_SOURCE)
    variants = [
        {'name': 'case-flat-homogeneous', 'surface_geometry': 'horizontal',
         'base_depth': 1.0, 'layers': []},
        {'name': 'case-flat-layered', 'surface_geometry': 'terrain',
         'base_depth': 3.0,
         'layers': [
             {'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3,
              'density': 2000.0, 'thickness': 40.0},
             {'name': 'overlying', 'vs': 800.0, 'poisson_ratio': 0.3,
              'density': 2200.0, 'thickness': 40.0},
         ]},
        {'name': 'case-slope-default-sentinel', 'surface_geometry': 'horizontal',
         'base_depth': 1.0, 'layers': [], 'validation_geometry': 'slope'},
    ]
    manifest = {'unit': 'F0-1', 'purpose': '显式平场验证模式只建模回归，不提交求解',
                'created_at': datetime.datetime.now().isoformat(), 'run_dir': root_dir,
                'variants': variants,
                'source_sha256': {model_name: sha256(MODEL_SOURCE), wave_name: sha256(WAVE_SOURCE)}}
    write_json(os.path.join(root_dir, 'f0_1_run_manifest.json'), manifest)
    report_path = os.path.join(root_dir, 'f0_1_validation_report.json')
    try:
        reports = [_run_variant(root_dir, variant) for variant in variants]
        report = {'status': 'passed', 'unit': 'F0-1', 'run_dir': root_dir,
                  'cases': reports, 'finished_at': datetime.datetime.now().isoformat()}
        write_json(report_path, report)
        print('F0-1 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        report = {'status': 'failed', 'unit': 'F0-1', 'run_dir': root_dir,
                  'error': str(exc), 'failed_at': datetime.datetime.now().isoformat()}
        write_json(report_path, report)
        raise


if __name__ == '__main__':
    main()
