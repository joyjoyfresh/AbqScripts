# -*- coding: utf-8 -*-
"""从已完成的 F1 NPZ 生成频域验收报告、指标表和传递函数图。"""

from __future__ import print_function

import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'Batch'))
import Autorun_ch3_F1_frequency_theory_v1 as f1


def write_json(path, value):
    with open(path, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    if len(sys.argv) < 2:
        raise SystemExit('用法：python Generate_ch3_F1_frequency_artifacts_v1.py <run_dir> [case_id]')
    run_dir = os.path.abspath(sys.argv[1])
    case_id = sys.argv[2] if len(sys.argv) > 2 else 'A1'
    case = next(item for item in f1.CASES if item['id'] == case_id)
    case_dir = os.path.join(run_dir, case_id)
    report = f1.validate_case(case_dir, case)
    report_doc = {'status': 'passed' if report['passed'] else 'failed', 'unit': 'F1',
                  'run_dir': run_dir, 'case_count': 1, 'cases': [report]}
    write_json(os.path.join(run_dir, 'f1_validation_report.json'), report_doc)

    npz_path = os.path.join(case_dir, 'surface_results.npz')
    package = np.load(npz_path)
    try:
        prefix = 'raw_ricker_wavelet_%dHz_' % int(case['frequency'])
        time = np.asarray(package[prefix + 'time'], dtype=float)
        input_time = np.asarray(package[prefix + 'underground_time'], dtype=float)
        input_acc = np.interp(time, input_time, np.asarray(package[prefix + 'input_acc'], dtype=float))
        surface_h = np.asarray(package[prefix + 'acc_h'], dtype=float)
        surface_x = np.asarray(package[prefix + 'x'], dtype=float)
    finally:
        package.close()
    dt = float(time[1] - time[0])
    freq = np.fft.rfftfreq(len(time), d=dt)
    input_spec = np.fft.rfft(input_acc - np.mean(input_acc))
    indices = f1._select_indices(surface_x)
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ['#0072B2', '#D55E00', '#009E73']
    for color, index, fraction in zip(colors, indices, f1.F1_PROTOCOL['surface_positions']):
        transfer = np.fft.rfft(surface_h[index] - np.mean(surface_h[index])) / input_spec
        mask = (freq >= 0.5 * case['frequency']) & (freq <= 1.5 * case['frequency'])
        ax.plot(freq[mask], np.abs(transfer[mask]), color=color,
                label='x/L=%.2f' % fraction, linewidth=1.4)
        metric = report['surface_metrics'][indices.index(index)]['horizontal']
        rows.append([case_id, fraction, surface_x[index], metric['band_amplitude_nrmse'],
                     metric['target_gain_error'], metric['output_peak_frequency_error_relative'],
                     metric['passed']])
    ax.axvline(case['frequency'], color='0.35', linestyle='--', linewidth=0.9, label='$f_c$')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|H(f)|')
    ax.set_title('F1 frequency-domain transfer validation: %s' % case_id)
    ax.grid(True, color='0.9', linewidth=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'F1_%s_transfer_function.png' % case_id), dpi=300)
    fig.savefig(os.path.join(run_dir, 'F1_%s_transfer_function.pdf' % case_id))
    plt.close(fig)
    with open(os.path.join(run_dir, 'F1_%s_metrics.csv' % case_id), 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'x_over_L', 'x_m', 'band_amplitude_nrmse',
                         'target_gain_error', 'peak_frequency_error_relative', 'passed'])
        writer.writerows(rows)
    print('F1 artifacts written: %s' % run_dir)


if __name__ == '__main__':
    main()
