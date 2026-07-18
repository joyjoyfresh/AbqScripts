# -*- coding: utf-8 -*-
"""第三章 U3b：复用 U3a 基准工况的时间步收敛试验。"""

from __future__ import print_function

import datetime  # 导入时间记录模块
import json  # 导入报告模块
import os  # 导入路径模块
import shutil  # 导入复制模块
import sys  # 导入命令行模块

import numpy as np  # 导入曲线收敛计算模块

import Autorun_ch3_03_mesh_convergence_v1 as base  # 复用已验证的四脚本调度与 NPZ 读取逻辑


REPO_ROOT = base.REPO_ROOT  # 仓库根目录
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'test', 'Abaqus', 'ch3_03_numerical_convergence')  # U3 测试输出根目录
BASELINE_DIR = os.path.join(DEFAULT_ROOT, 'run-001', 'case-mesh-8m')  # U3a 的1.0 ms基准工况
DT_LEVELS = (0.002, 0.0005)  # 仅新增粗、细两级；中间1.0 ms复用 U3a 结果


def case_name(dt):
    """按毫秒生成稳定工况名。"""
    return 'case-dt-%gms' % (dt * 1000.0)


def case_config(dt, wave_path):
    """保持 U3a 的8 m网格，只替换具有目标采样间隔的输入波。"""
    config = base.case_config(8.0)
    config['time_cfg'] = {'check': True, 'min_steps_per_fmax_period': 20,
                          'tail_seconds': 1.0}
    config['run_cfg']['wave_files'] = [wave_path]
    return config


def resample_wave(dt, directory):
    """按目标 dt 重采样输入波，分析步将严格继承该真实采样间隔。"""
    source = np.loadtxt(base.WAVE_SOURCE)
    time = np.arange(source[0, 0], source[-1, 0] + 0.1 * dt, dt)
    acceleration = np.interp(time, source[:, 0], source[:, 1])
    path = os.path.join(directory, 'ricker_wavelet_4Hz_dt_%gus.txt' % (dt * 1.0e6))
    np.savetxt(path, np.column_stack((time, acceleration)), fmt='%.10e')
    return path


def prepare(root_dir):
    """创建0.5 ms和2.0 ms两个新增工况。"""
    base.require_file(BASELINE_DIR, 'U3a 1.0 ms基准目录') if os.path.isfile(BASELINE_DIR) else None
    if not os.path.isdir(BASELINE_DIR):
        raise RuntimeError('U3a 1.0 ms基准目录不存在：%s' % BASELINE_DIR)
    cases = []
    for dt in DT_LEVELS:
        directory = os.path.join(root_dir, case_name(dt))
        os.makedirs(directory, exist_ok=True)
        for source in (base.MODEL_SOURCE, base.POST_SOURCE):
            shutil.copy2(source, os.path.join(directory, os.path.basename(source)))
        wave_path = resample_wave(dt, directory)
        base.write_json(os.path.join(directory, 'case_config.json'), case_config(dt, wave_path))
        cases.append((dt, directory))
    base.write_json(os.path.join(root_dir, 'u3b_run_manifest.json'), {
        'unit': 'U3b', 'created_at': datetime.datetime.now().isoformat(),
        'variable': 'max_increment', 'new_levels': list(DT_LEVELS),
        'baseline_1ms': BASELINE_DIR})
    return cases


def validate(root_dir, cases):
    """以0.5 ms为参考检查1.0 ms整曲线和峰值收敛。"""
    s_mid, y_mid, summary_mid = base.read_curve(BASELINE_DIR)
    fine_dir = dict(cases)[0.0005]
    s_ref, y_ref, summary_ref = base.read_curve(fine_dir)
    coarse_summary = base.read_curve(dict(cases)[0.002])[2]
    actual_dt = {'2.0ms': float(coarse_summary['dt']), '1.0ms': float(summary_mid['dt']),
                 '0.5ms': float(summary_ref['dt'])}
    expected_dt = {'2.0ms': 0.002, '1.0ms': 0.001, '0.5ms': 0.0005}
    for key in expected_dt:
        if abs(actual_dt[key] - expected_dt[key]) > 1.0e-6 * expected_dt[key]:
            raise RuntimeError('U3b %s 实际 dt 不符：%.9g' % (key, actual_dt[key]))
    if bool(summary_mid.get('suspect')) or bool(summary_ref.get('suspect')):
        raise RuntimeError('U3b 参与收敛比较的工况远场 QA 未通过')
    if not np.allclose(s_mid, s_ref):
        y_mid = np.interp(s_ref, s_mid, y_mid)
    mask = np.isfinite(y_mid) & np.isfinite(y_ref)
    curve_l2 = float(np.linalg.norm(y_mid[mask] - y_ref[mask]) / np.linalg.norm(y_ref[mask]))
    peak_mid = float(summary_mid['AR_max'])
    peak_ref = float(summary_ref['AR_max'])
    peak_rel = abs(peak_mid / peak_ref - 1.0)
    report = {'status': 'passed' if max(curve_l2, peak_rel) <= 0.05 else 'failed',
              'criterion': '1.0 ms 与 0.5 ms 的整曲线 L2 相对差和 AR_max 相对差均不超过5%',
              'curve_l2_relative': curve_l2, 'peak_relative': peak_rel,
              'actual_dt': actual_dt,
              'AR_max': {'2.0ms': coarse_summary['AR_max'], '1.0ms': peak_mid, '0.5ms': peak_ref},
              'finished_at': datetime.datetime.now().isoformat()}
    base.write_json(os.path.join(root_dir, 'u3b_time_validation_report.json'), report)
    if report['status'] != 'passed':
        raise RuntimeError('U3b 时间步收敛未通过：%r' % report)
    return report


def finalize(root_dir, cases):
    """后处理新增工况并执行时间步验收。"""
    for _, directory in cases:
        base.run_step([base.ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(base.POST_SOURCE)], directory,
                      os.path.join(directory, 'autorun_02_postprocess.log'))
    for source in (base.COLLECT_SOURCE, base.PLOT_SOURCE):
        shutil.copy2(source, os.path.join(root_dir, os.path.basename(source)))
    base.run_step([sys.executable, os.path.basename(base.COLLECT_SOURCE), root_dir], root_dir,
                  os.path.join(root_dir, 'autorun_03_collect.log'))
    base.run_step([sys.executable, os.path.basename(base.PLOT_SOURCE), root_dir], root_dir,
                  os.path.join(root_dir, 'autorun_04_plot.log'))
    print('U3b 通过：%s' % json.dumps(validate(root_dir, cases), ensure_ascii=False, sort_keys=True))


def main():
    """执行 U3b，支持 --run、--prepare 和 --finalize。"""
    root_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else base.next_run_dir(DEFAULT_ROOT)
    action = sys.argv[2] if len(sys.argv) >= 3 else '--run'
    if action not in ('--run', '--prepare', '--finalize'):
        raise RuntimeError('动作仅支持 --run/--prepare/--finalize')
    cases = prepare(root_dir)
    if action == '--prepare':
        print('U3b 准备完成：%s' % root_dir)
        return
    if action == '--run':
        for _, directory in cases:
            base.run_step([base.ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(base.MODEL_SOURCE)], directory,
                          os.path.join(directory, 'autorun_01_model.log'))
    finalize(root_dir, cases)


if __name__ == '__main__':
    main()
