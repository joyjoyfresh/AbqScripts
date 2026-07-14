# -*- coding: utf-8 -*-
"""F1：均质/成层平场的频域传递函数验证入口。

本脚本只在 F1 门禁授权后运行。它继续调用论文固定四脚本链，
但验收指标从长时程逐点误差改为目标频带传递函数幅值、目标频率增益
和输出主频误差；V2 的时域 pilot 仅作为诊断记录，不在此处复用为通过证据。
"""

from __future__ import print_function

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
MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')
POST_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Postprocess_All_surface_v2.py')
COLLECT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Collect_All_results_v2.py')
PLOT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Plot_Hybrid_surface_v2.py')
REFERENCE_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'reference_layered_psv_v1.py')
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_F1_frequency_theory')
WAVE_DIR = os.path.join(REPO_ROOT, 'Run', '_f1_generated_waves')
sys.path.insert(0, os.path.join(REPO_ROOT, 'Modeling', 'Hybrid'))
import reference_layered_psv_v1 as reference

HALFSPACE = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3}
F1_PROTOCOL = {
    'name': 'frequency_transfer_function_v1',
    'cases': ['A1', 'A2', 'A3', 'A4', 'A5'],
    'surface_positions': [0.25, 0.50, 0.75],
    'target_band_relative_to_fc': [0.5, 1.5],
    'metrics': ['band_amplitude_nrmse', 'target_gain_error', 'vector_peak_frequency_error'],
    'thresholds': {'band_amplitude_nrmse': 0.05, 'target_gain_error': 0.05,
                   'output_peak_frequency_error_relative': 0.03},
    'phase': '仅报告复传递函数相位和线性传播相位，不作为时域逐点通过条件',
}

CASES = [
    {'id': 'A1', 'angle': 0.0, 'frequency': 2.0, 'layers': [], 'label': '均质垂直入射'},
    {'id': 'A2', 'angle': 15.0, 'frequency': 2.0, 'layers': [], 'label': '均质斜入射'},
    {'id': 'A3', 'angle': 30.0, 'frequency': 2.0, 'layers': [], 'label': '均质近临界入射'},
    {'id': 'A4', 'angle': 0.0, 'frequency': 2.0,
     'layers': [{'name': 'surface', 'vs': 800.0, 'rho': 2000.0, 'nu': 0.3,
                 'poisson_ratio': 0.3, 'density': 2000.0, 'thickness': 20.0}],
     'label': '成层垂直入射'},
    {'id': 'A5', 'angle': 15.0, 'frequency': 4.0,
     'layers': [{'name': 'surface', 'vs': 800.0, 'rho': 2000.0, 'nu': 0.3,
                 'poisson_ratio': 0.3, 'density': 2000.0, 'thickness': 20.0}],
     'label': '成层斜入射'},
]


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


def write_ricker(path, frequency, dt=0.001, duration=4.0):
    time = np.arange(0.0, duration + 0.5 * dt, dt)
    t0 = 2.0 / float(frequency)
    arg = np.pi * float(frequency) * (time - t0)
    acceleration = (1.0 - 2.0 * arg * arg) * np.exp(-arg * arg)
    acceleration /= max(float(np.max(np.abs(acceleration))), 1.0e-30)
    np.savetxt(path, np.column_stack((time, acceleration)), fmt='%.9f %.12e')


def run_command(command, cwd, log_path, timeout=3600):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    with open(log_path, 'wb') as fh:
        fh.write(('命令：%s\n工作目录：%s\n' % (' '.join(command), cwd)).encode('utf-8'))
        result = subprocess.run(command, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError('步骤失败，退出码=%s：%s' % (result.returncode, log_path))


def make_config(case, wave_name):
    return {
        'material_cfg': {'angle': case['angle'], 'layers': case['layers'],
                         'surface_geometry': 'horizontal'},
        'geometry_cfg': {'slope_height': 25.0, 'slope_angle': 45.0,
                         'crest_window': 20.0, 'toe_window': 20.0,
                         # 2 Hz 基岩波长约1000 m，底部深度取20h以减弱底部反射
                         'side_clearance': 2.0, 'base_depth': 20.0},
        'damping_cfg': {'enable': False, 'fc': case['frequency']},
        # 16 m 网格对应2 Hz基岩波长约62.5个单元，满足F1频带离散要求并控制计算规模
        'mesh_cfg': {'size': 8.0 if case['frequency'] >= 4.0 else 16.0,
                     'auto': False, 'elem': 'CPE4R', 'graded': False},
        'time_cfg': {'check': True, 'tail_seconds': 1.0},
        'freefield_cfg': {'engine': 'fd', 'include_damping': False,
                          'phase_origin_x': 'center'},
        # 采用主脚本标注的标准Liu弹簧恢复系数，作为平场人工边界的预注册基准
        'boundary_cfg': {'dashpot_scale': 1.0, 'spring_scale': 2.0},
        'run_cfg': {'surface_only': True, 'critical_angle_check': True,
                    'validation_geometry': 'flat', 'submit_jobs': True,
                    'wave_files': [wave_name]},
        'qa_cfg': {'required': ['theory', 'energy'],
                   'artificial_energy_ratio_tol': 0.05,
                   'energy_residual_tol': 0.05},
        'f1_validation_cfg': F1_PROTOCOL,
        'tssi_cfg': {'enable': False, 'scene': 'freefield',
                     'nonlinear': False, 'gravity': 'off'},
    }


def _load_raw(npz_path, record):
    package = np.load(npz_path)
    try:
        prefix = 'raw_%s_' % ''.join(ch if (ch.isalnum() or ch in ('_', '-')) else '_'
                                     for ch in record)
        def array(name):
            key = prefix + name
            return np.asarray(package[key], dtype=float) if key in package.files else None
        return {'time': array('time'), 'input': array('input_acc'),
                'input_time': array('underground_time'),
                'surface_h': array('acc_h'), 'surface_v': array('acc_v'),
                'surface_x': array('x')}
    finally:
        package.close()


def _select_indices(xs):
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    indices = []
    for fraction in F1_PROTOCOL['surface_positions']:
        target = xmin + float(fraction) * (xmax - xmin)
        index = int(np.argmin(np.abs(xs - target)))
        if index not in indices:
            indices.append(index)
    if len(indices) != 3:
        raise RuntimeError('F1 三个内部测点不唯一: %s' % indices)
    return indices


def _theory_transfer(freqs, case):
    horizontal = np.zeros(len(freqs), dtype=complex)
    vertical = np.zeros(len(freqs), dtype=complex)
    for index, frequency in enumerate(freqs):
        if frequency <= 0.0:
            continue
        if case['layers']:
            if abs(float(case['angle'])) < 1.0e-12:
                # 垂直入射的水平分量退化为独立SH层状传递函数，避免P-SV单位SV归一化的零分量陷阱
                layer = case['layers'][0]
                kh = 2.0 * np.pi * frequency * float(layer['thickness']) / float(layer['vs'])
                impedance_ratio = (float(layer['rho']) * float(layer['vs']) /
                                   (HALFSPACE['rho'] * HALFSPACE['vs']))
                horizontal[index] = 2.0 / complex(np.cos(kh), impedance_ratio * np.sin(kh))
                vertical[index] = 0.0j
                continue
            result = reference.surface_response(frequency, case['layers'], HALFSPACE,
                                                case['angle'])
            # 独立参考程序按单位粒子位移输出，转换为与均质半空间函数一致的位移传递函数
            horizontal[index] = -HALFSPACE['vs'] * result['ux']
            vertical[index] = -HALFSPACE['vs'] * result['uy']
        else:
            result = reference.homogeneous_halfspace_transfer(frequency, 0.0,
                                                               HALFSPACE, case['angle'])
            horizontal[index] = result['ux']
            vertical[index] = result['uy']
    return horizontal, vertical


def _transfer_metrics(actual, input_series, dt, frequency, theory_transfer):
    # 传递函数幅值应对传播相位延迟不敏感；输入与输出使用同一完整记录窗，避免Hann窗对不同到时赋予不同权重
    window = np.ones(len(actual), dtype=float)
    nfft = 8 * len(actual)  # 零填充只提高峰值频率读数分辨率，不改变原始记录内容
    input_spectrum = np.fft.rfft((input_series - np.mean(input_series)) * window, n=nfft)
    actual_spectrum = np.fft.rfft((actual - np.mean(actual)) * window, n=nfft)
    frequencies = np.fft.rfftfreq(nfft, d=float(dt))
    theory_base_freq = np.fft.rfftfreq(len(actual), d=float(dt))
    if len(theory_transfer) != len(frequencies):
        theory_transfer = (np.interp(frequencies, theory_base_freq, np.real(theory_transfer)) +
                           1j * np.interp(frequencies, theory_base_freq, np.imag(theory_transfer)))
    valid = np.abs(input_spectrum) > 0.01 * max(float(np.max(np.abs(input_spectrum))), 1.0e-30)
    transfer = np.zeros_like(actual_spectrum, dtype=complex)
    transfer[valid] = actual_spectrum[valid] / input_spectrum[valid]
    band = ((frequencies >= F1_PROTOCOL['target_band_relative_to_fc'][0] * frequency) &
            (frequencies <= F1_PROTOCOL['target_band_relative_to_fc'][1] * frequency) & valid)
    if not np.any(band):
        raise RuntimeError('目标频带没有有效输入谱点')
    theory_band_level = float(np.max(np.abs(theory_transfer[band])))
    if theory_band_level < 1.0e-6:
        leakage = float(np.max(np.abs(transfer[band])))
        return {'applicable': False, 'passed': True,
                'numerical_leakage_gain': leakage,
                'reason': '理论传递函数在目标频带近似为零，仅报告数值泄漏'}
    band_nrmse = float(np.linalg.norm(np.abs(transfer[band]) - np.abs(theory_transfer[band])) /
                       max(np.linalg.norm(np.abs(theory_transfer[band])), 1.0e-30))
    target_index = int(np.argmin(np.abs(frequencies - float(frequency))))
    gain_error = float(abs(abs(transfer[target_index]) /
                           max(abs(theory_transfer[target_index]), 1.0e-30) - 1.0))
    band_indices = np.where(band)[0]
    output_peak = frequencies[int(band_indices[int(np.argmax(np.abs(actual_spectrum[band])))] )]
    theory_output = np.abs(input_spectrum) * np.abs(theory_transfer)
    theory_peak = frequencies[int(band_indices[int(np.argmax(theory_output[band]))])]
    peak_error = float(abs(output_peak - theory_peak) / max(float(frequency), 1.0e-30))
    passed = bool(band_nrmse <= F1_PROTOCOL['thresholds']['band_amplitude_nrmse'] and
                  gain_error <= F1_PROTOCOL['thresholds']['target_gain_error'] and
                  peak_error <= F1_PROTOCOL['thresholds']['output_peak_frequency_error_relative'])
    return {'applicable': True, 'band_amplitude_nrmse': band_nrmse, 'target_gain_error': gain_error,
            'output_peak_frequency_error_relative': peak_error,
            'output_peak_frequency_hz': float(output_peak),
            'theory_output_peak_frequency_hz': float(theory_peak), 'passed': passed}


def _vector_peak_metric(actual_h, actual_v, input_series, dt, frequency, theory_h, theory_v):
    """按合成P-SV传递谱评价主频，避免把单一耦合分量的局部峰误当作整体主频。"""
    nfft = 8 * len(actual_h)
    window = np.ones(len(actual_h), dtype=float)
    inp = np.fft.rfft((input_series - np.mean(input_series)) * window, n=nfft)
    ah = np.fft.rfft((actual_h - np.mean(actual_h)) * window, n=nfft) / inp
    av = np.fft.rfft((actual_v - np.mean(actual_v)) * window, n=nfft) / inp
    base_freq = np.fft.rfftfreq(len(actual_h), d=float(dt))
    freq = np.fft.rfftfreq(nfft, d=float(dt))
    th = np.interp(freq, base_freq, np.real(theory_h)) + 1j * np.interp(freq, base_freq, np.imag(theory_h))
    tv = np.interp(freq, base_freq, np.real(theory_v)) + 1j * np.interp(freq, base_freq, np.imag(theory_v))
    band = (freq >= 0.5 * frequency) & (freq <= 1.5 * frequency)
    actual_peak = freq[np.where(band)[0][int(np.argmax(np.sqrt(np.abs(ah[band]) ** 2 + np.abs(av[band]) ** 2)))]]
    theory_peak = freq[np.where(band)[0][int(np.argmax(np.sqrt(np.abs(th[band]) ** 2 + np.abs(tv[band]) ** 2)))]]
    error = float(abs(actual_peak - theory_peak) / max(float(frequency), 1.0e-30))
    return {'output_peak_frequency_hz': float(actual_peak),
            'theory_output_peak_frequency_hz': float(theory_peak),
            'output_peak_frequency_error_relative': error, 'passed': error <= 0.03}


def validate_case(case_dir, case):
    npz_path = os.path.join(case_dir, 'surface_results.npz')
    if not os.path.isfile(npz_path):
        raise RuntimeError('F1 缺少 NPZ: %s' % npz_path)
    record = 'ricker_wavelet_%dHz' % int(case['frequency'])
    raw = _load_raw(npz_path, record)
    if raw['time'] is None or raw['input'] is None or raw['surface_h'] is None:
        raise RuntimeError('F1 NPZ 缺少原始时程字段')
    dt = float(raw['time'][1] - raw['time'][0])
    input_series = raw['input']
    if raw.get('input_time') is not None and len(raw['input_time']) == len(raw['input']):
        # 输入波与ODB输出步长可能存在舍入差，先统一到地表输出时间轴再做FFT
        input_series = np.interp(raw['time'], raw['input_time'], raw['input'])
    frequencies = np.fft.rfftfreq(len(raw['time']), d=dt)
    theory_h, theory_v = _theory_transfer(frequencies, case)
    indices = _select_indices(raw['surface_x'])
    metrics = []
    for index in indices:
        horizontal = _transfer_metrics(raw['surface_h'][index], input_series, dt,
                                       case['frequency'], theory_h)
        vertical = _transfer_metrics(raw['surface_v'][index], input_series, dt,
                                     case['frequency'], theory_v)
        vector_peak = _vector_peak_metric(raw['surface_h'][index], raw['surface_v'][index],
                                          input_series, dt, case['frequency'], theory_h, theory_v)
        if vertical.get('applicable'):
            vertical['component_peak_role'] = 'auxiliary_only'
            vertical['passed'] = bool(vertical['band_amplitude_nrmse'] <= F1_PROTOCOL['thresholds']['band_amplitude_nrmse'] and
                                      vertical['target_gain_error'] <= F1_PROTOCOL['thresholds']['target_gain_error'])
        metrics.append({'index': int(index), 'x': float(raw['surface_x'][index]),
                        'horizontal': horizontal, 'vertical': vertical,
                        'vector_peak': vector_peak})
    passed = all(item['horizontal']['passed'] and item['vertical']['passed'] and
                 item['vector_peak']['passed'] for item in metrics)
    return {'case_id': case['id'], 'label': case['label'], 'angle': case['angle'],
            'frequency': case['frequency'], 'layers': case['layers'], 'dt': dt,
            'surface_metrics': metrics, 'passed': passed}


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    limit = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0
    requested_ids = [item.strip() for item in sys.argv[2].split(',') if item.strip()] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else []
    run_dir = next_run_dir(root)
    if not os.path.isdir(WAVE_DIR):
        os.makedirs(WAVE_DIR)
    selected = [case for case in CASES if case['id'] in requested_ids] if requested_ids else (CASES[:limit] if limit > 0 else CASES)
    if not selected:
        raise RuntimeError('F1 未找到请求的工况编号: %s' % requested_ids)
    waves = {}
    for frequency in sorted(set(case['frequency'] for case in selected)):
        path = os.path.join(WAVE_DIR, 'ricker_wavelet_%dHz.txt' % int(frequency))
        write_ricker(path, frequency)
        waves[frequency] = path
    manifest = {'unit': 'F1', 'run_dir': run_dir, 'protocol': F1_PROTOCOL,
                'cases': selected, 'source_sha256':
                {os.path.basename(path): sha256(path) for path in
                 (MODEL_SOURCE, POST_SOURCE, COLLECT_SOURCE, PLOT_SOURCE, REFERENCE_SOURCE)},
                'waves_sha256': {os.path.basename(path): sha256(path) for path in waves.values()},
                'created_at': datetime.datetime.now().isoformat()}
    write_json(os.path.join(run_dir, 'f1_run_manifest.json'), manifest)
    reports = []
    try:
        for case in selected:
            case_dir = os.path.join(run_dir, case['id'])
            os.makedirs(case_dir)
            wave_name = os.path.basename(waves[case['frequency']])
            for source in (MODEL_SOURCE, POST_SOURCE):
                shutil.copy2(source, os.path.join(case_dir, os.path.basename(source)))
            shutil.copy2(waves[case['frequency']], os.path.join(case_dir, wave_name))
            write_json(os.path.join(case_dir, 'case_config.json'), make_config(case, wave_name))
            run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], case_dir,
                        os.path.join(case_dir, 'autorun_01_model.log'))
            run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POST_SOURCE)], case_dir,
                        os.path.join(case_dir, 'autorun_02_postprocess.log'))
            reports.append(validate_case(case_dir, case))
        for source in (COLLECT_SOURCE, PLOT_SOURCE, REFERENCE_SOURCE):
            shutil.copy2(source, os.path.join(run_dir, os.path.basename(source)))
        run_command([sys.executable, os.path.basename(COLLECT_SOURCE), run_dir], run_dir,
                    os.path.join(run_dir, 'autorun_03_collect.log'))
        run_command([sys.executable, os.path.basename(PLOT_SOURCE), run_dir], run_dir,
                    os.path.join(run_dir, 'autorun_04_plot.log'))
        report = {'status': 'passed' if all(item['passed'] for item in reports) else 'failed',
                  'unit': 'F1', 'run_dir': run_dir, 'case_count': len(reports),
                  'cases': reports, 'finished_at': datetime.datetime.now().isoformat()}
    except Exception as exc:
        report = {'status': 'failed', 'unit': 'F1', 'run_dir': run_dir,
                  'case_count_completed': len(reports), 'error': str(exc),
                  'failed_at': datetime.datetime.now().isoformat()}
        write_json(os.path.join(run_dir, 'f1_validation_report.json'), report)
        raise
    write_json(os.path.join(run_dir, 'f1_validation_report.json'), report)
    if report['status'] != 'passed':
        raise RuntimeError('F1 频域理论门槛未全部通过，详见 f1_validation_report.json')
    print('F1 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
