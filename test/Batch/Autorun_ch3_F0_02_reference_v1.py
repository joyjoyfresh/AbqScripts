# -*- coding: utf-8 -*-
"""F0-2：生成独立 P–SV 层状参考解的固定 NPZ 基准。

该 Autorun 不启动 Abaqus，不调用生产 FD 内核；它仅运行
``Modeling/Archived/Hybrid/reference_layered_psv_v1.py``，把多剖面、多角度、多频率
的自由表面复响应直接写入 NPZ，供后续 V2/V3 端到端对比使用。
"""

from __future__ import print_function

import datetime
import hashlib
import json
import os
import sys

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from Modeling.Archived.Hybrid import reference_layered_psv_v1 as reference


DEFAULT_ROOT = os.path.join(REPO_ROOT, 'test', 'Abaqus', 'ch3_F0_02_reference')
REFERENCE_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Archived', 'Hybrid', 'reference_layered_psv_v1.py')


def sha256(path):
    """计算参考程序哈希。"""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """写出 UTF-8 JSON。"""
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')


def next_run_dir(unit_root):
    """创建不覆盖既有结果的新 run 目录。"""
    if not os.path.isdir(unit_root):
        os.makedirs(unit_root)
    numbers = [int(name[4:]) for name in os.listdir(unit_root)
               if name.startswith('run-') and name[4:].isdigit()]
    run_dir = os.path.join(unit_root, 'run-%03d' % ((max(numbers) if numbers else 0) + 1))
    os.makedirs(run_dir)
    return run_dir


def main():
    """计算固定参考矩阵并执行基础门槛。"""
    unit_root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    run_dir = next_run_dir(unit_root)
    halfspace = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3}
    profiles = {
        'H0_homogeneous': [],
        'L1_low_contrast': [
            {'vs': 1200.0, 'rho': 2300.0, 'nu': 0.3, 'thickness': 40.0}],
        'L2_soft_layer': [
            {'vs': 400.0, 'rho': 2000.0, 'nu': 0.3, 'thickness': 40.0}],
        'L3_two_layer': [
            {'vs': 400.0, 'rho': 2000.0, 'nu': 0.3, 'thickness': 40.0},
            {'vs': 800.0, 'rho': 2200.0, 'nu': 0.3, 'thickness': 40.0}],
    }
    angles = [0.0, 15.0, 30.0]
    frequencies = [2.0, 4.0, 8.0]
    rows = []
    for profile_name, layers in profiles.items():
        for angle in angles:
            for frequency in frequencies:
                result = reference.surface_response(frequency, layers, halfspace, angle)
                rows.append({
                    'profile': profile_name,
                    'angle_deg': angle,
                    'frequency_hz': frequency,
                    'ux_real': result['ux'].real,
                    'ux_imag': result['ux'].imag,
                    'uy_real': result['uy'].real,
                    'uy_imag': result['uy'].imag,
                    'ux_abs': abs(result['ux']),
                    'uy_abs': abs(result['uy']),
                    'traction_residual': result['traction_residual'],
                    'reflected_p_abs': abs(result['reflected_p']),
                    'reflected_sv_abs': abs(result['reflected_sv']),
                })
    residuals = np.asarray([row['traction_residual'] for row in rows], dtype=float)
    finite = all(np.all(np.isfinite([row[key] for key in ('ux_abs', 'uy_abs', 'traction_residual')]))
                 for row in rows)
    h0 = [row for row in rows if row['profile'] == 'H0_homogeneous' and row['angle_deg'] == 0.0]
    h0_ratio = [row['ux_abs'] * halfspace['vs'] for row in h0]
    if not finite or np.max(residuals) > 1.0e-10:
        raise RuntimeError('参考解矩阵非有限或自由表面牵引残差超限: %.3e' % np.max(residuals))
    if max(abs(value - 2.0) for value in h0_ratio) > 1.0e-10:
        raise RuntimeError('均质半空间垂直入射退化不满足 |ux|*Vs=2: %r' % h0_ratio)

    dtype = [('profile', 'U32'), ('angle_deg', 'f8'), ('frequency_hz', 'f8'),
             ('ux_real', 'f8'), ('ux_imag', 'f8'), ('uy_real', 'f8'), ('uy_imag', 'f8'),
             ('ux_abs', 'f8'), ('uy_abs', 'f8'), ('traction_residual', 'f8'),
             ('reflected_p_abs', 'f8'), ('reflected_sv_abs', 'f8')]
    table = np.empty(len(rows), dtype=dtype)
    for idx, row in enumerate(rows):
        table[idx] = tuple(row[name] for name, _kind in dtype)
    np.savez_compressed(os.path.join(run_dir, 'reference_transfer.npz'),
                        records=table, frequencies_hz=np.asarray(frequencies),
                        angles_deg=np.asarray(angles), profile_names=np.asarray(list(profiles.keys())))
    manifest = {'unit': 'F0-2', 'purpose': '独立 P-SV 层状频域参考解固定基准',
                'created_at': datetime.datetime.now().isoformat(), 'run_dir': run_dir,
                'reference_script': REFERENCE_SOURCE, 'reference_sha256': sha256(REFERENCE_SOURCE),
                'profiles': profiles, 'angles_deg': angles, 'frequencies_hz': frequencies,
                'record_count': len(rows), 'maximum_traction_residual': float(np.max(residuals)),
                'halfspace_vertical_ratio': h0_ratio}
    write_json(os.path.join(run_dir, 'reference_manifest.json'), manifest)
    report = {'status': 'passed', 'unit': 'F0-2', 'run_dir': run_dir,
              'record_count': len(rows), 'maximum_traction_residual': float(np.max(residuals)),
              'halfspace_vertical_ratio': h0_ratio,
              'npz': os.path.join(run_dir, 'reference_transfer.npz'),
              'finished_at': datetime.datetime.now().isoformat()}
    write_json(os.path.join(run_dir, 'f0_2_validation_report.json'), report)
    print('F0-2 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
