# -*- coding: utf-8 -*-
"""V2：平坦均质半空间解析解—Abaqus ODB 端到端验证。

验证协议已预注册为：主峰中心 t0=2/fc，直达波窗 [t0-2Tc,t0+4Tc]；
地表取 x/L=0.25、0.50、0.75 三个内部点；在窗口内用归一化互相关自动
估计时移，再评价到时、峰值、波形和主频幅值四项指标。
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
WAVE_DIR = os.path.join(REPO_ROOT, 'Run', '_v2_generated_waves')
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_V2_flat_homogeneous')
sys.path.insert(0, os.path.join(REPO_ROOT, 'Modeling', 'Hybrid'))
import reference_layered_psv_v1 as reference

HALFSPACE = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3}
ANGLES = [-30.0, -15.0, 0.0, 15.0, 30.0]
FREQUENCIES = [2.0, 4.0, 8.0]
V2_PROTOCOL = {
    'name': 'pre_registered_direct_wave_window_v1',
    'phase_origin': 'x=模型域中点；时间参考为输入波底部参考面，传播深度=base_depth*slope_height',
    'ricker_peak_t0_cycles': 2.0,
    'wave_pre_roll': 'max_abs_horizontal_slowness*half_domain_width+0.25 s margin',
    'window_cycles': [-2.0, 4.0],
    'surface_positions': [0.25, 0.50, 0.75],
    'time_shift': 'maximum_normalized_cross_correlation_in_window',
    'thresholds': {'arrival_error_over_period': 0.02, 'peak_error': 0.05,
                   'nrmse': 0.10, 'spectrum_amplitude_error': 0.05},
    'spectrum_band': 'single_frequency_bin_at_fc',
}


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
    nums = [int(name[4:]) for name in os.listdir(root)
            if name.startswith('run-') and name[4:].isdigit()]
    path = os.path.join(root, 'run-%03d' % ((max(nums) if nums else 0) + 1))
    os.makedirs(path)
    return path


def write_ricker(path, frequency, pre_roll, dt=0.001):
    duration = float(pre_roll) + 6.0 / float(frequency) + 0.5
    time = np.arange(0.0, duration + 0.5 * dt, dt)
    t0 = float(pre_roll) + 2.0 / float(frequency)  # 预滚保证域中点相位原点两侧主峰均在记录内
    arg = np.pi * float(frequency) * (time - t0)
    acc = (1.0 - 2.0 * arg * arg) * np.exp(-arg * arg)
    acc /= max(float(np.max(np.abs(acc))), 1.0e-30)
    np.savetxt(path, np.column_stack((time, acc)), fmt='%.9f %.12e')
    return {'frequency': float(frequency), 'dt': float(dt), 'duration': float(duration),
            'pre_roll': float(pre_roll), 't0': float(t0), 'window_end': float(t0 + 4.0 / float(frequency))}


def run_command(command, cwd, log_path, timeout=3600):
    env = os.environ.copy()
    if os.path.abspath(command[0]) == os.path.abspath(sys.executable):
        env['PYTHONIOENCODING'] = 'utf-8'
    with open(log_path, 'wb') as fh:
        fh.write(('命令：%s\n工作目录：%s\n' % (' '.join(command), cwd)).encode('utf-8'))
        result = subprocess.run(command, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError('步骤失败，退出码=%s：%s' % (result.returncode, log_path))


def make_cases():
    cases = []
    for angle in ANGLES:
        for frequency in FREQUENCIES:
            cases.append({'name': 'case-flat-a%+03d-f%02d' % (int(angle), int(frequency)),
                          'angle': angle, 'frequency': frequency})
    return cases


def make_config(case, wave_name):
    return {
        'material_cfg': {'angle': case['angle'], 'layers': [], 'surface_geometry': 'horizontal'},
        # 直达波窗最长为 3 s（2 Hz），侧向净空扩大到 300h 以使三测点的侧边反射晚于比较窗。
        'geometry_cfg': {'slope_height': 25.0, 'slope_angle': 45.0, 'crest_window': 300.0,
                         'toe_window': 300.0, 'side_clearance': 2.0, 'base_depth': 3.0},
        'damping_cfg': {'enable': False, 'fc': case['frequency']},
        'mesh_cfg': {'size': 8.0, 'auto': False, 'elem': 'CPE4R', 'graded': False},
        'time_cfg': {'check': True, 'tail_seconds': 1.0},
        'freefield_cfg': {'engine': 'fd', 'include_damping': False, 'phase_origin_x': 'center'},
        'run_cfg': {'surface_only': True, 'critical_angle_check': True,
                    'validation_geometry': 'flat', 'submit_jobs': True, 'wave_files': [wave_name]},
        'qa_cfg': {'required': ['theory', 'energy'], 'artificial_energy_ratio_tol': 0.05,
                   'energy_residual_tol': 0.05},
        'v2_wave_cfg': case.get('wave_cfg', {}),
        'v2_validation_cfg': V2_PROTOCOL,
        'tssi_cfg': {'enable': False, 'scene': 'freefield', 'nonlinear': False, 'gravity': 'off'},
    }


def _load_raw(npz_path, record):
    package = np.load(npz_path)
    try:
        suffix = ''.join(ch if (ch.isalnum() or ch in ('_', '-')) else '_' for ch in record)
        prefix = 'raw_%s_' % suffix
        def arr(name):
            key = prefix + name
            if key not in package.files:
                return None
            return np.asarray(package[key], dtype=float)
        return {'time': arr('time'), 'underground_time': arr('underground_time'), 'input': arr('input_acc'), 'surface_h': arr('acc_h'),
                'surface_v': arr('acc_v'), 'surface_x': arr('x'), 'surface_y': arr('y'),
                'underground_h': arr('underground_acc_h'), 'underground_v': arr('underground_acc_v'),
                'underground_x': arr('underground_x'), 'underground_y': arr('underground_y')}
    finally:
        package.close()


def theory_series(input_acc, dt, angle, depths, xs, phase_origin_depth):
    n = len(input_acc)
    freqs = np.fft.rfftfreq(n, d=float(dt))
    spectrum = np.fft.rfft(input_acc)
    surface_h = [np.zeros_like(spectrum, dtype=complex) for _ in xs]
    surface_v = [np.zeros_like(spectrum, dtype=complex) for _ in xs]
    underground_h = [np.zeros_like(spectrum, dtype=complex) for _ in depths]
    underground_v = [np.zeros_like(spectrum, dtype=complex) for _ in depths]
    p = np.sin(np.deg2rad(float(angle))) / float(HALFSPACE['vs'])
    for index, frequency in enumerate(freqs[1:], 1):
        omega = 2.0 * np.pi * frequency
        temporal_phase = np.exp(-1j * omega * float(phase_origin_depth) /
                                (float(HALFSPACE['vs']) * np.cos(np.deg2rad(float(angle)))))
        surface = reference.homogeneous_halfspace_transfer(frequency, 0.0, HALFSPACE, angle)
        for point, x in enumerate(xs):
            horizontal_phase = reference.horizontal_phase_factor(frequency, x, HALFSPACE, angle)
            phase = temporal_phase * horizontal_phase
            surface_h[point][index] = spectrum[index] * surface['ux'] * phase
            surface_v[point][index] = spectrum[index] * surface['uy'] * phase
        for point, depth in enumerate(depths):
            response = reference.homogeneous_halfspace_transfer(frequency, depth, HALFSPACE, angle)
            phase = temporal_phase
            underground_h[point][index] = spectrum[index] * response['ux'] * phase
            underground_v[point][index] = spectrum[index] * response['uy'] * phase
    return ([np.fft.irfft(v, n=n) for v in surface_h], [np.fft.irfft(v, n=n) for v in surface_v],
            [np.fft.irfft(v, n=n) for v in underground_h],
            [np.fft.irfft(v, n=n) for v in underground_v])


def _window_bounds(frequency, t0_value=None):
    period = 1.0 / float(frequency)
    t0 = float(t0_value) if t0_value is not None else V2_PROTOCOL['ricker_peak_t0_cycles'] * period
    return t0 + V2_PROTOCOL['window_cycles'][0] * period, t0 + V2_PROTOCOL['window_cycles'][1] * period


def _estimate_lag(actual, theory):
    actual = np.asarray(actual, dtype=float)
    theory = np.asarray(theory, dtype=float)
    actual = actual - float(np.mean(actual))
    theory = theory - float(np.mean(theory))
    denom = max(float(np.linalg.norm(actual) * np.linalg.norm(theory)), 1.0e-30)
    corr = np.correlate(actual, theory, mode='full') / denom
    return int(np.argmax(corr) - (len(actual) - 1)), float(np.max(corr))


def metric(actual, theory, time, frequency, t0_value=None):
    actual = np.asarray(actual, dtype=float)
    theory = np.asarray(theory, dtype=float)
    time = np.asarray(time, dtype=float)
    dt = float(time[1] - time[0])
    t_start, t_end = _window_bounds(frequency, t0_value=t0_value)
    mask = (time >= t_start - 0.5 * dt) & (time <= t_end + 0.5 * dt)
    if int(np.sum(mask)) < 8:
        raise ValueError('V2 直达波窗内样本不足: %.6g—%.6g s' % (t_start, t_end))
    tw = time[mask]
    aw = actual[mask]
    thw = theory[mask]
    lag_samples, corr = _estimate_lag(aw, thw)
    aligned = np.interp(tw + lag_samples * dt, tw, aw, left=0.0, right=0.0)
    scale = max(float(np.linalg.norm(thw)), 1.0e-30)
    nrmse = float(np.linalg.norm(aligned - thw) / scale)
    peak = float(abs(np.max(np.abs(aligned)) / max(np.max(np.abs(thw)), 1.0e-30) - 1.0))
    freqs = np.fft.rfftfreq(len(tw), d=dt)
    idx = int(np.argmin(np.abs(freqs - float(frequency))))
    sa = np.fft.rfft(aligned - np.mean(aligned))[idx]
    st = np.fft.rfft(thw - np.mean(thw))[idx]
    spectrum = float(abs(abs(sa) / max(abs(st), 1.0e-30) - 1.0))
    phase = float((np.angle(sa / st, deg=True) + 180.0) % 360.0 - 180.0) if abs(st) > 1.0e-30 and abs(sa) > 1.0e-30 else 999.0
    arrival = abs(lag_samples) * dt
    period = 1.0 / float(frequency)
    passed = bool(arrival <= V2_PROTOCOL['thresholds']['arrival_error_over_period'] * period and
                  peak <= V2_PROTOCOL['thresholds']['peak_error'] and
                  nrmse <= V2_PROTOCOL['thresholds']['nrmse'] and
                  spectrum <= V2_PROTOCOL['thresholds']['spectrum_amplitude_error'])
    return {'window_s': [float(tw[0]), float(tw[-1])], 'lag_samples': int(lag_samples),
            'arrival_error_s': float(arrival), 'correlation': float(corr),
            'nrmse': nrmse, 'peak_error': peak, 'spectrum_amplitude_error': spectrum,
            'phase_error_deg_auxiliary': abs(phase), 'passed': passed}


def validate_case(case_dir, case):
    npz_path = os.path.join(case_dir, 'surface_results.npz')
    if not os.path.isfile(npz_path):
        raise RuntimeError('缺少 NPZ：%s' % case_dir)
    raw = _load_raw(npz_path, 'ricker_wavelet_%dHz' % int(case['frequency']))
    if raw['input'] is None or raw['surface_h'] is None or raw['underground_h'] is None:
        raise RuntimeError('NPZ 缺少 V2 地表或地下原始时程字段')
    dt = float(raw['time'][1] - raw['time'][0])
    x = np.asarray(raw['surface_x'], dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    selected = []
    for fraction in V2_PROTOCOL['surface_positions']:
        target = xmin + float(fraction) * (xmax - xmin)
        index = int(np.argmin(np.abs(x - target)))
        if index not in selected:
            selected.append(index)
    if len(selected) != len(V2_PROTOCOL['surface_positions']):
        raise RuntimeError('V2 三个内部测点无法选出互异节点: %s' % selected)
    depths = [float(raw['surface_y'][selected[0]] - value) for value in raw['underground_y']]
    case_config = {}
    config_path = os.path.join(case_dir, 'case_config.json')
    if os.path.isfile(config_path):
        with open(config_path, 'r', encoding='utf-8') as fh:
            case_config = json.load(fh)
    geom = case_config.get('geometry_cfg', {})
    phase_origin_depth = float(geom.get('base_depth', 0.0)) * float(geom.get('slope_height', 0.0))
    phase_origin_x = 0.5 * (xmin + xmax)
    theory_h, theory_v, theory_uh, theory_uv = theory_series(raw['input'], dt, case['angle'], depths,
                                                              [x[i] - phase_origin_x for i in selected], phase_origin_depth)
    surface_metrics = []
    for point, index in enumerate(selected):
        surface_metrics.append({'index': int(index), 'x': float(x[index]), 'position_fraction': V2_PROTOCOL['surface_positions'][point],
                                'horizontal': metric(raw['surface_h'][index], theory_h[point], raw['time'], case['frequency'],
                                                     t0_value=case.get('wave_cfg', {}).get('t0')),
                                'vertical': metric(raw['surface_v'][index], theory_v[point], raw['time'], case['frequency'],
                                                   t0_value=case.get('wave_cfg', {}).get('t0'))})
    underground_metrics = []
    for index in range(len(raw['underground_h'])):
        underground_metrics.append({'index': int(index), 'x': float(raw['underground_x'][index]),
                                    'horizontal': metric(raw['underground_h'][index], theory_uh[index], raw['underground_time'] if raw['underground_time'] is not None else raw['time'], case['frequency'],
                                                         t0_value=case.get('wave_cfg', {}).get('t0')),
                                    'vertical': metric(raw['underground_v'][index], theory_uv[index], raw['underground_time'] if raw['underground_time'] is not None else raw['time'], case['frequency'],
                                                       t0_value=case.get('wave_cfg', {}).get('t0'))})
    passed = all(item['horizontal']['passed'] and item['vertical']['passed'] for item in surface_metrics)
    return {'name': case['name'], 'angle': case['angle'], 'frequency': case['frequency'],
            'dt': dt, 'protocol': V2_PROTOCOL, 'phase_origin_depth': phase_origin_depth,
            'phase_origin_x': phase_origin_x,
            'surface_indices': selected, 'surface_metrics': surface_metrics, 'underground_metrics': underground_metrics,
            'passed': passed}


def main():
    root = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_dir = next_run_dir(root)
    if not os.path.isdir(WAVE_DIR):
        os.makedirs(WAVE_DIR)
    cases = make_cases()
    selected = cases[:limit] if limit > 0 else cases
    # 预滚时间按最大论文入射角和半域宽度事前计算，确保斜入射波主峰不在记录起点之前。
    hs = 25.0
    total_length = (300.0 + 2.0) * hs + hs / np.tan(np.deg2rad(45.0)) + (300.0 + 2.0) * hs
    max_horizontal_slowness = abs(np.sin(np.deg2rad(max(abs(v) for v in ANGLES)))) / float(HALFSPACE['vs'])
    pre_roll = max_horizontal_slowness * 0.5 * total_length + 0.25
    waves = {}
    wave_meta = {}
    for frequency in FREQUENCIES:
        path = os.path.join(WAVE_DIR, 'ricker_wavelet_%dHz.txt' % int(frequency))
        wave_meta[frequency] = write_ricker(path, frequency, pre_roll)
        waves[frequency] = path
    for case in selected:
        case['pre_roll'] = pre_roll
        case['wave_cfg'] = dict(wave_meta[case['frequency']])
    manifest = {'unit': 'V2', 'run_dir': run_dir, 'requested_case_count': len(selected),
                'full_matrix_case_count': len(cases), 'cases': selected,
                'protocol': V2_PROTOCOL, 'wave_meta': wave_meta, 'domain_total_length': total_length,
                'waves_sha256': {os.path.basename(path): sha256(path) for path in waves.values()},
                'source_sha256': {os.path.basename(path): sha256(path) for path in (MODEL_SOURCE, POST_SOURCE, COLLECT_SOURCE, PLOT_SOURCE, REFERENCE_SOURCE)},
                'created_at': datetime.datetime.now().isoformat()}
    write_json(os.path.join(run_dir, 'v2_run_manifest.json'), manifest)
    reports = []
    try:
        for case in selected:
            case_dir = os.path.join(run_dir, case['name'])
            os.makedirs(case_dir)
            wave_name = os.path.basename(waves[case['frequency']])
            shutil.copy2(MODEL_SOURCE, os.path.join(case_dir, os.path.basename(MODEL_SOURCE)))
            shutil.copy2(POST_SOURCE, os.path.join(case_dir, os.path.basename(POST_SOURCE)))
            shutil.copy2(waves[case['frequency']], os.path.join(case_dir, wave_name))
            write_json(os.path.join(case_dir, 'case_config.json'), make_config(case, wave_name))
            run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], case_dir,
                        os.path.join(case_dir, 'autorun_01_model.log'))
            run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POST_SOURCE)], case_dir,
                        os.path.join(case_dir, 'autorun_02_postprocess.log'))
            reports.append(validate_case(case_dir, case))
        for source in (COLLECT_SOURCE, PLOT_SOURCE):
            shutil.copy2(source, os.path.join(run_dir, os.path.basename(source)))
        shutil.copy2(REFERENCE_SOURCE, os.path.join(run_dir, os.path.basename(REFERENCE_SOURCE)))
        run_command([sys.executable, os.path.basename(COLLECT_SOURCE), run_dir], run_dir,
                    os.path.join(run_dir, 'autorun_03_collect.log'))
        run_command([sys.executable, os.path.basename(PLOT_SOURCE), run_dir], run_dir,
                    os.path.join(run_dir, 'autorun_04_plot.log'))
        report = {'status': 'passed' if all(item['passed'] for item in reports) else 'failed',
                  'unit': 'V2', 'run_dir': run_dir, 'case_count': len(reports), 'cases': reports,
                  'finished_at': datetime.datetime.now().isoformat()}
    except Exception as exc:
        report = {'status': 'failed', 'unit': 'V2', 'run_dir': run_dir,
                  'case_count_completed': len(reports), 'error': str(exc),
                  'failed_at': datetime.datetime.now().isoformat()}
        write_json(os.path.join(run_dir, 'v2_validation_report.json'), report)
        raise
    write_json(os.path.join(run_dir, 'v2_validation_report.json'), report)
    if report['status'] != 'passed':
        raise RuntimeError('V2 解析门槛未全部通过，详见 %s' % os.path.join(run_dir, 'v2_validation_report.json'))
    print('V2 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
