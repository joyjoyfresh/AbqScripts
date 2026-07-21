# -*- coding: utf-8 -*-
"""验证fd自由场增量初态与Abaqus状态文件进度解析。"""

import importlib.util
import math
import os
import sys
import tempfile
import types

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py')
WAVE_PATH = os.path.join(
    REPO_ROOT, 'Wave', 'Seismic', 'G1r_reference_gate',
    'g1r_loma_prieta_0p3g_dt1ms.txt',
)


def _load_model():
    """用桩模块屏蔽Abaqus依赖并导入当前统一建模脚本。"""
    abaqus = types.ModuleType('abaqus')
    abaqus.mdb = None
    sys.modules['abaqus'] = abaqus
    for name in ('abaqusConstants', 'caeModules', 'mesh'):
        sys.modules[name] = types.ModuleType(name)
    region = types.ModuleType('regionToolset')
    region.Region = object
    sys.modules['regionToolset'] = region
    spec = importlib.util.spec_from_file_location('slope_frame_ssi_full_v2', MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.basestring = str
    return module


def _make_context(module, initial_state_mode):
    """按已完成H1-Loma工况构造不依赖Abaqus的自由场上下文。"""
    wave = np.loadtxt(WAVE_PATH)
    time = wave[:, 0]
    acceleration = wave[:, 1]
    dt = float(time[1] - time[0])
    bedrock = module.Material(2000.0, 0.3, 2500.0, None, 'bedrock')
    site = module.Site(bedrock, [], 600.0)
    geometry = module.Geometry(
        800.0, math.radians(45.0), 350.0, 100.0, None, 600.0,
        0.0, -100.0, 600.0, 600.0, 600.0, 0.001, [],
    )
    stratigraphy = module._build_stratigraphy(site, geometry)
    shear_modulus = 2500.0 * 2000.0 ** 2
    cp = 2000.0 * math.sqrt(2.0 * (1.0 - 0.3) / (1.0 - 2.0 * 0.3))
    lame_lambda = 2500.0 * (cp ** 2 - 2.0 * 2000.0 ** 2)
    damping_config = {
        'enable': True, 'method': 'rayleigh', 'constant_xi': 0.03,
        'q_bedrock': 999.0, 'fc': 4.0, 'f1_factor': 0.5,
        'f2_factor': 2.5, 'anchor': 'perband', 'harmonics_cover': 3.0,
    }
    damping = module._band_damping_terms(stratigraphy, damping_config)
    zero = np.zeros(acceleration.shape, dtype=float)
    freefield = {
        'engine': 'fd', 'include_damping': True, 'spectrum_tol': 1.0e-7,
        'fcut': 15.0, 'pad_factor': 4, 'phase_origin_x': 'center',
        'tail_seconds': 6.0, 'initial_state_mode': initial_state_mode,
    }
    return module.FreeFieldCtx(
        site, geometry, stratigraphy, 600.0, 600.0, 0.0,
        0.0, 0.0, 0.0, shear_modulus, lame_lambda, 2000.0, cp,
        np.column_stack((time, zero)), np.column_stack((time, zero)),
        dt, time, 3, acceleration, damping, freefield,
    )


def main():
    module = _load_model()
    fd, sta_path = tempfile.mkstemp(suffix='.sta')
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(b" STEP INC ATT SEVERE EQUIL TOTAL TOTAL STEP INC OF\n")
            handle.write(b" 1 1 1 0 1 1 0.010 0.010 0.010\n")
            handle.write(b" 1 2 1 0 1 1 0.025 0.025 0.015\n")
            handle.write(b" 2 1 1 0 1 1 1.025 0.100 0.100\n")
        assert module._read_sta_step_progress(sta_path, 1) == 0.025
        assert module._read_sta_step_progress(sta_path, 2) == 0.100
        assert module._read_sta_step_progress(sta_path, 3) is None
    finally:
        os.remove(sta_path)

    raw_context = _make_context(module, 'raw')
    module._FD_SOLVER_CACHE.clear()
    raw = module._fd_freefield_at_node('l', 0.0, 600.0, 600.0, raw_context)
    assert abs(raw['ux'][0]) > 0.10
    assert abs(raw['dotux'][0]) > 1.0e-3

    incremental_context = _make_context(module, 'incremental')
    module._FD_SOLVER_CACHE.clear()
    incremental = module._fd_freefield_at_node(
        'l', 0.0, 600.0, 600.0, incremental_context,
    )
    for key in ('ux', 'uy', 'dotux', 'dotuy', 'sigmax', 'sigmay'):
        assert abs(float(incremental[key][0])) <= 1.0e-10, (key, incremental[key][0])

    raw_acceleration = np.gradient(raw['dotux'], raw_context.dt)
    incremental_acceleration = np.gradient(
        incremental['dotux'], incremental_context.dt,
    )
    assert np.max(np.abs(raw_acceleration - incremental_acceleration)) <= 1.0e-10
    print('test_fd_initial_state: 初态归零、加速度不变且sta进度解析通过')


if __name__ == '__main__':
    main()
