# -*- coding: utf-8 -*-
"""独立 P–SV 层状参考解测试（纯 Python，不导入生产建模脚本）。"""

from __future__ import print_function

import io
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from Modeling.Archived.Hybrid import reference_layered_psv_v1 as ref


HALFSPACE = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3}
LAYERS = [
    {'vs': 400.0, 'rho': 2000.0, 'nu': 0.3, 'thickness': 40.0},
    {'vs': 800.0, 'rho': 2200.0, 'nu': 0.3, 'thickness': 40.0},
]


def _sh_transfer(freqs, layers, halfspace):
    """独立 SH Haskell 递推，用于验证垂直入射 P–SV 退化。"""
    out = []
    for freq in freqs:
        omega = 2.0 * np.pi * freq
        up, down = 1.0 + 0j, 1.0 + 0j
        stack = list(layers) + [{'vs': halfspace['vs'], 'rho': halfspace['rho'], 'thickness': None}]
        for idx in range(len(layers)):
            z1 = stack[idx]['rho'] * stack[idx]['vs']
            z2 = stack[idx + 1]['rho'] * stack[idx + 1]['vs']
            ratio = z1 / z2
            phase = omega * stack[idx]['thickness'] / stack[idx]['vs']
            ep, em = np.exp(1j * phase), np.exp(-1j * phase)
            up, down = (0.5 * up * (1.0 + ratio) * ep + 0.5 * down * (1.0 - ratio) * em,
                        0.5 * up * (1.0 - ratio) * ep + 0.5 * down * (1.0 + ratio) * em)
        out.append(abs(2.0 / up))
    return np.asarray(out)


def main():
    source_path = os.path.abspath(ref.__file__)
    source = io.open(source_path, encoding='utf-8').read()
    assert '_fd_solve_column' not in source
    assert 'slope_frame_ssi_full_v2' not in source

    half = ref.surface_response(4.0, [], HALFSPACE, incident_angle_deg=0.0)
    assert abs(abs(half['ux']) * HALFSPACE['vs'] - 2.0) < 1.0e-10
    assert abs(half['uy']) < 1.0e-14
    assert half['traction_residual'] < 1.0e-12

    same = [{'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3, 'thickness': 80.0}]
    deg = ref.surface_response(4.0, same, HALFSPACE, incident_angle_deg=15.0)
    base = ref.surface_response(4.0, [], HALFSPACE, incident_angle_deg=15.0)
    assert abs(abs(deg['ux']) - abs(base['ux'])) / max(abs(base['ux']), 1.0e-30) < 1.0e-10
    assert abs(abs(deg['uy']) - abs(base['uy'])) / max(abs(base['uy']), 1.0e-30) < 1.0e-10

    freqs = np.linspace(0.5, 12.0, 32)
    psv = ref.transfer_function(freqs, LAYERS, HALFSPACE, incident_angle_deg=0.0)
    sh = _sh_transfer(freqs, LAYERS, HALFSPACE)
    psv_ratio = np.abs(psv['ux']) * HALFSPACE['vs']
    assert np.max(np.abs(psv_ratio - sh)) / np.max(sh) < 1.0e-9
    assert np.max(psv['traction_residual']) < 1.0e-10

    oblique = ref.transfer_function([2.0, 4.0, 8.0], LAYERS, HALFSPACE,
                                    incident_angle_deg=30.0)
    assert np.all(np.isfinite(np.abs(oblique['ux'])))
    assert np.all(np.isfinite(np.abs(oblique['uy'])))
    assert np.max(oblique['traction_residual']) < 1.0e-10
    depth0 = ref.homogeneous_halfspace_transfer(4.0, 0.0, HALFSPACE, incident_angle_deg=0.0)
    depth25 = ref.homogeneous_halfspace_transfer(4.0, 25.0, HALFSPACE, incident_angle_deg=0.0)
    assert abs(depth0['ux'] - 2.0) < 1.0e-10
    assert np.isfinite(depth25['ux'].real) and np.isfinite(depth25['uy'].real)
    damped_halfspace = dict(HALFSPACE, rayleigh_alpha=0.02, rayleigh_beta=0.001)
    damped_same = [dict(HALFSPACE, thickness=80.0, rayleigh_alpha=0.02, rayleigh_beta=0.001)]
    damped_layer = ref.surface_response(4.0, damped_same, damped_halfspace, incident_angle_deg=15.0)
    damped_base = ref.surface_response(4.0, [], damped_halfspace, incident_angle_deg=15.0)
    omega = 2.0 * np.pi * 4.0
    mat = ref._material(2000.0, 2500.0, 0.3, omega=omega,
                        rayleigh_alpha=0.02, rayleigh_beta=0.001)
    p = np.sin(np.radians(15.0)) / 2000.0
    qs = ref._vertical_slowness(mat['vs'], p)
    expected_decay = abs(np.exp(-1j * omega * qs * 80.0))
    actual_decay = abs(damped_layer['ux']) / abs(damped_base['ux'])
    assert expected_decay < 1.0
    assert abs(actual_decay - expected_decay) < 1.0e-10
    assert damped_layer['traction_residual'] < 1.0e-10
    print('test_reference_layered_psv_v1: 8/8 ok')


if __name__ == '__main__':
    main()
