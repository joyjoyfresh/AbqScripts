# -*- coding: utf-8 -*-
"""OpenSees 版本的建模与 PGA 提取一体化脚本。

复现了 Modeling/Multi/VAB_oblique_TAF_multilayer_v8.py 的全部参数配置、一维解析引擎、人工边界与等效力计算，
并用 OpenSees 动力有限元替代 Abaqus 进行求解，同时前置提取地表 PGA 写入 TIMESERIES 与 PGA 的 CSV 文件。
"""

import os
import io
import sys
import math
import json
import time
import shutil
import csv
import subprocess
import numpy as np
from collections import namedtuple

# ==========================================================
#  配置参数（默认值）
# ==========================================================

material_cfg = {
    'angle': 15,
    'surface_geometry': 'horizontal', # horizontal=水平, terrain=等厚沿地形
    'bedrock': {
        'elastic_modulus': 26e9,
        'poisson_ratio': 0.3,
        'density': 2500,
    },
    'layers': [
        {'name': 'overlying',
         'velocity_ratio': 1.25,
         'poisson_ratio': 0.3,
         'density': 2500},
    ],
}

geometry_cfg = {
    'H_minus_h': 200.0,
    'i': 45.0,
    'h_over_H': 0.5,
    'total_L': 1800.0,
    'left_flat': 1000.0,
    'bedrock_thickness': 200.0,
}

# 默认阻尼
damping_cfg = {
    'enable': False,
    'method': 'rayleigh',
    'constant_xi': 0.02,
    'qs_factor': 0.05,
    'q_bedrock': 999.0,
    'fc': None,
    'f1_factor': 0.5,
    'f2_factor': 2.5,
    'anchor': 'perband',
    'harmonics_cover': 3.0,
}

# 默认网格
mesh_cfg = {
    'size': 4.0,
    'auto': False,
    'elems_per_wavelength': 10,
    'fmax_factor': 2.5,
    'min_size': 0.2,
    'max_band_ratio': 4.0,
    'max_size': None,
    'resolve_harmonics': 3.0,
    'min_elems_through_thickness': 6,
}

time_cfg = {
    'check': False,
    'min_steps_per_fmax_period': 20,
    'tail_seconds': 0.0,
}

freefield_cfg = {
    'engine': 'fd',
    'include_damping': True,
    'spectrum_tol': 1e-7,
    'fcut': None,
    'pad_factor': 4,
}

run_cfg = {
    'run_flat': False,
    'surface_only': True,
    'critical_angle_check': False,
}

# ==========================================================
#  结构体与常量定义
# ==========================================================

Material = namedtuple('Material', ['cs', 'vv', 'density', 'thickness', 'name'])
Site = namedtuple('Site', ['bedrock', 'layers', 'bedrock_thickness'])
Geometry = namedtuple('Geometry', [
    'total_L', 'i', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness',
    'H', 'h', 'H_upper', 'H_lower', 'H_flat', 'w_slope',
    'layer_interfaces'])
BoundaryNode = namedtuple('BoundaryNode', ['label', 'x', 'y', 'influence', 'kn', 'cn', 'kt', 'ct'])
FreeFieldCtx = namedtuple('FreeFieldCtx', [
    'site', 'geom', 'strat', 'ymax_l', 'ymax_r', 'ymin',
    'alpha', 'beta_p', 'p_horiz',
    'GG', 'lam', 'cs', 'cp',
    'VEL', 'DIS', 'dt', 'time_arr', 'max_reflect_order',
    'acc', 'damp_terms', 'ffcfg'])

_FD_SOLVER_CACHE = {}
OPENSEES_EXE = r"C:\Users\12462\Documents\Apps\OpenSees3.8.0-x64.exe\OpenSees3.8.0\bin\OpenSees.exe"

def log_step(logger_file=None, message=None, *args):
    """写运行日志"""
    if not hasattr(log_step, '_start_time'):
        log_step._start_time = time.time()
        if logger_file:
            log_step._file = open(logger_file, 'w', encoding='utf-8')
        else:
            log_step._file = sys.stdout

    if message is not None:
        t_now = time.time() - log_step._start_time
        line = '[{:.3f}s] {}\n'.format(t_now, message % args if args else message)
        log_step._file.write(line)
        log_step._file.flush()
        if log_step._file != sys.stdout:
            sys.stdout.write(line)

# ==========================================================
#  纯 Python 物理/数学计算核心
# ==========================================================

def _compute_wave_speed_from_elastic_modulus(elastic_modulus, poisson_ratio, density):
    """计算剪切波速"""
    GG = elastic_modulus / 2.0 / (1.0 + poisson_ratio)
    return math.sqrt(GG / density)

def _compute_elastic_modulus_from_wave_speed(cs, vv, density):
    """计算弹性模量"""
    GG = density * cs * cs
    return GG * 2.0 * (1.0 + vv)

def _compute_material_params(cs, vv, density):
    """计算弹性模量与拉梅常数"""
    GG = density * cs * cs
    EE = GG * 2.0 * (1.0 + vv)
    lam = GG * 2.0 * vv / (1.0 - 2.0 * vv)
    cp = math.sqrt((lam + 2.0 * GG) / density)
    return {'cs': cs, 'vv': vv, 'density': density, 'GG': GG, 'EE': EE, 'lam': lam, 'cp': cp}

def _estimate_dominant_freq(acc, dt):
    """估计加速度谱主频"""
    n = len(acc)
    nfft = _next_pow2(n)
    spec = np.abs(np.fft.rfft(acc, n=nfft))
    freqs = np.fft.rfftfreq(nfft, dt)
    idx = np.argmax(spec[1:]) + 1 # 排除直流分量
    return freqs[idx]

def _damping_ratio_from_q(cs, is_bedrock, dcfg):
    """由品质因子换算阻尼比"""
    if is_bedrock:
        Q = float(dcfg.get('q_bedrock', 999.0))
    else:
        q_factor = float(dcfg.get('qs_factor', 0.05))
        Q = q_factor * cs
    xi = 1.0 / (2.0 * Q) if Q > 0 else 0.0
    return Q, xi

def _rayleigh_coeffs(xi, dcfg, fc, f_layer=None):
    """计算瑞利阻尼系数 alpha, beta"""
    method = dcfg.get('method', 'rayleigh')
    if method == 'stiffness':
        beta = 2.0 * xi / (2.0 * math.pi * fc)
        return 0.0, beta
    
    anchor = dcfg.get('anchor', 'input')
    if anchor == 'perband' and f_layer is not None:
        f1 = min(float(dcfg.get('f1_factor', 0.5)) * fc, f_layer)
        f2 = max(float(dcfg.get('f2_factor', 2.5)) * fc, float(dcfg.get('harmonics_cover', 3.0)) * f_layer)
    elif anchor == 'dual' and dcfg.get('f_site') is not None:
        f_site = dcfg['f_site']
        f1 = min(float(dcfg.get('f1_factor', 0.5)) * fc, f_site)
        f2 = max(float(dcfg.get('f2_factor', 2.5)) * fc, f_site)
    else:
        f1 = float(dcfg.get('f1_factor', 0.5)) * fc
        f2 = float(dcfg.get('f2_factor', 2.5)) * fc

    w1 = 2.0 * math.pi * f1
    w2 = 2.0 * math.pi * f2
    denom = w1 + w2
    alpha = 2.0 * xi * w1 * w2 / denom
    beta = 2.0 * xi / denom
    return alpha, beta

def _resolve_damping(dcfg, fc_est):
    """确定阻尼配置主频"""
    out = dict(dcfg)
    if out.get('fc') is None:
        out['fc'] = fc_est
    return out

def _band_resonance_freq(band):
    """计算分层共振基频"""
    cs = band['mat'].cs
    thk = band['y1'] - band['y0']
    return cs / (4.0 * thk) if thk > 0 else 999.0

def _material_resonance_freq(mat, site, geom):
    """计算材料层的共振基频"""
    cum = 0.0
    for L in site.layers:
        if L.thickness is not None:
            cum += L.thickness
            if L.name == mat.name:
                return L.cs / (4.0 * L.thickness)
        else:
            H = geom.H_upper - geom.bedrock_thickness
            h_bottom = H - cum
            if L.name == mat.name:
                return L.cs / (4.0 * h_bottom)
    return site.bedrock.cs / 400.0

def _compute_free_surface_sv_coeff(alpha, cp, cs):
    """计算自由面 SV 波系数"""
    beta = _safe_arcsin(cp * math.sin(alpha) / cs)
    num = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta) - cp ** 2 * math.cos(2 * alpha) ** 2
    den = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta) + cp ** 2 * math.cos(2 * alpha) ** 2
    if abs(den) < 1e-12:
        den = 1e-12
    a1 = num / den
    a2 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / den
    return {'A1': a1, 'A2': a2, 'beta': beta}

def _safe_arcsin(value):
    return math.asin(max(-1.0, min(1.0, value)))

def _integrate_acc_to_velocity(acc, dt, time_arr):
    """时程积分并做趋势项校正"""
    vel = np.zeros_like(acc)
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2.0 * dt)
    slope, intercept = np.polyfit(time_arr, vel, 1)
    vel -= (slope * time_arr + intercept)
    return vel, slope

def _surface_y_at(x, H_upper, H_lower, left_flat, w_slope):
    """计算局部地表高程"""
    w = max(w_slope, 1e-9)
    if x <= left_flat:
        return H_upper
    if x <= left_flat + w:
        return H_upper - (x - left_flat) * (H_upper - H_lower) / w
    return H_lower

def _build_stratigraphy(site, geom, ymin=0.0, surface_geometry='horizontal'):
    """构建一维柱场地分层带"""
    H_upper = geom.H_upper
    bt = geom.bedrock_thickness
    layers_td = list(site.layers)
    if not layers_td:
        return [{'name': site.bedrock.name, 'mat': site.bedrock, 'y0': ymin, 'y1': H_upper, 'fix': 'elevation'}]
    terrain = (surface_geometry == 'terrain')
    bands_td = []
    y_top = H_upper
    depth_top = 0.0
    for L in layers_td:
        if L.thickness is not None:
            band = {'name': L.name, 'mat': L, 'y0': y_top - L.thickness, 'y1': y_top}
            if terrain:
                band['fix'] = 'depth'
                band['d0'] = depth_top
                band['d1'] = depth_top + L.thickness
            else:
                band['fix'] = 'elevation'
            bands_td.append(band)
            y_top -= L.thickness
            depth_top += L.thickness
        else:
            band = {'name': L.name, 'mat': L, 'y0': bt, 'y1': y_top}
            if terrain:
                band['fix'] = 'fill'
                band['dtop'] = depth_top
            else:
                band['fix'] = 'elevation'
            bands_td.append(band)
    bedrock_band = {'name': site.bedrock.name, 'mat': site.bedrock, 'y0': ymin, 'y1': bt, 'fix': 'elevation'}
    bands_bt = list(reversed(bands_td))
    return [bedrock_band] + bands_bt

def _band_bounds_at(band, ys):
    fix = band.get('fix', 'elevation')
    if fix == 'depth':
        return ys - band['d1'], ys - band['d0']
    if fix == 'fill':
        return band['y0'], ys - band['dtop']
    return band['y0'], band['y1']

def make_geometry(total_L, H_minus_h, i, h_over_H, left_flat, bedrock_thickness, fixed_thicknesses=None):
    H = H_minus_h / (1.0 - h_over_H)
    h = H - H_minus_h
    H_upper = bedrock_thickness + H
    H_lower = bedrock_thickness + h
    H_flat = bedrock_thickness + H
    w_slope = H_minus_h / math.tan(math.radians(i))
    fixed = list(fixed_thicknesses or [])
    layer_interfaces = []
    cum = 0.0
    for t in fixed:
        cum += t
        layer_interfaces.append(H_upper - cum)
    layer_interfaces = sorted(layer_interfaces)
    return Geometry(total_L=total_L, i=i, left_flat=left_flat, H_minus_h=H_minus_h,
                    h_over_H=h_over_H, bedrock_thickness=bedrock_thickness,
                    H=H, h=h, H_upper=H_upper, H_lower=H_lower, H_flat=H_flat, w_slope=w_slope,
                    layer_interfaces=layer_interfaces)

def make_flat_geometry(geom):
    return geom._replace(H_lower=geom.H_upper, w_slope=0.001)

def _next_pow2(n):
    m = 1
    while m < n:
        m *= 2
    return m

def _band_damping_terms(strat, damping):
    terms = {}
    for idx, band in enumerate(strat):
        if damping and damping.get('enable'):
            _Q, xi = _damping_ratio_from_q(band['mat'].cs, idx == 0, damping)
            f_layer = None if idx == 0 else _band_resonance_freq(band)
            a_ray, b_ray = _rayleigh_coeffs(xi, damping, damping['fc'], f_layer)
        else:
            a_ray, b_ray = 0.0, 0.0
        terms[band['name']] = (a_ray, b_ray)
    return terms

def _fd_input_spectrum(ctx):
    cached = _FD_SOLVER_CACHE.get('_input')
    if cached is not None:
        return cached
    ffcfg = ctx.ffcfg
    acc = np.asarray(ctx.acc, dtype=float)
    dt = float(ctx.dt)
    N = acc.shape[0]
    pad = max(2, int(ffcfg.get('pad_factor', 4)))
    Nfft = _next_pow2(N * pad)
    A = np.fft.rfft(acc, n=Nfft)
    freqs = np.fft.rfftfreq(Nfft, dt)
    tol = float(ffcfg.get('spectrum_tol', 1e-7))
    amax = float(np.max(np.abs(A))) if A.size else 0.0
    mask = np.abs(A) > tol * amax
    mask[0] = False
    fcut = ffcfg.get('fcut')
    if fcut:
        mask = mask & (freqs <= float(fcut))
    idx = np.nonzero(mask)[0]
    omega = 2.0 * math.pi * freqs[idx]
    U0 = -A[idx] / (omega ** 2)
    tail = float(ffcfg.get('tail_seconds', 0.0) or 0.0)
    Nout = min(Nfft, N + int(round(tail / dt))) if tail > 0 else N
    cached = {'Nfft': Nfft, 'Nout': Nout, 'dt': dt, 'nfreq': len(freqs),
              'idx': idx, 'omega': omega, 'U0': U0}
    _FD_SOLVER_CACHE['_input'] = cached
    return cached

def _fd_layer_params(seg, omega, p, damp_terms, include_damping):
    rho = float(seg['density'])
    mu0 = rho * seg['cs'] ** 2
    lam0 = rho * (seg['cp'] ** 2 - 2.0 * seg['cs'] ** 2)
    if include_damping:
        a_ray, b_ray = damp_terms.get(seg['name'], (0.0, 0.0))
    else:
        a_ray, b_ray = 0.0, 0.0
    rhoC = rho * (1.0 - 1j * a_ray / omega)
    sfac = 1.0 + 1j * omega * b_ray
    muC = mu0 * sfac
    lamC = lam0 * sfac
    cs2 = muC / rhoC
    cp2 = (lamC + 2.0 * muC) / rhoC
    qs = np.sqrt(1.0 / cs2 - p * p)
    qp = np.sqrt(1.0 / cp2 - p * p)
    qs = np.where(qs.imag > 0.0, -qs, qs)
    qp = np.where(qp.imag > 0.0, -qp, qp)
    return {'qs': qs, 'qp': qp, 'mu': muC, 'lam': lamC,
            'csC': np.sqrt(cs2), 'cpC': np.sqrt(cp2), 'p': p}

def _fd_wave_params(seg, la, kind):
    qs, qp, csC, cpC, p = la['qs'], la['qp'], la['csC'], la['cpC'], la['p']
    if kind == 'Pu':
        return {'dx': cpC * p, 'dy': cpC * qp, 'ky': qp, 'yref': seg['y0']}
    if kind == 'Pd':
        return {'dx': cpC * p, 'dy': -cpC * qp, 'ky': -qp, 'yref': seg['y1']}
    if kind == 'Su':
        return {'dx': csC * qs, 'dy': -csC * p, 'ky': qs, 'yref': seg['y0']}
    return {'dx': -csC * qs, 'dy': -csC * p, 'ky': -qs, 'yref': seg['y1']}

def _fd_field_coeffs(wave, la, omega, p, y):
    ph = np.exp(-1j * omega * wave['ky'] * (y - wave['yref']))
    dx = wave['dx'] * ph
    dy = wave['dy'] * ph
    lam, mu = la['lam'], la['mu']
    miw = -1j * omega
    syy = miw * (lam * p * dx + (lam + 2.0 * mu) * wave['ky'] * dy)
    sxy = miw * mu * (wave['ky'] * dx + p * dy)
    sxx = miw * ((lam + 2.0 * mu) * p * dx + lam * wave['ky'] * dy)
    return dx, dy, syy, sxy, sxx

def _fd_solve_column(column, p, omega, damp_terms, include_damping):
    nseg = len(column)
    M = nseg - 1
    las = [_fd_layer_params(seg, omega, p, damp_terms, include_damping) for seg in column]
    waves = []
    waves.append([(0, _fd_wave_params(column[0], las[0], 'Pd')),
                  (1, _fd_wave_params(column[0], las[0], 'Sd'))])
    col = 2
    for m in range(1, nseg):
        wm = []
        for kind in ('Pu', 'Pd', 'Su', 'Sd'):
            wm.append((col, _fd_wave_params(column[m], las[m], kind)))
            col += 1
        waves.append(wm)
    nunk = col
    inc = _fd_wave_params(column[0], las[0], 'Su')
    nb = omega.shape[0]
    A = np.zeros((nb, nunk, nunk), dtype=complex)
    b = np.zeros((nb, nunk), dtype=complex)
    row = 0
    for j in range(M):
        Y = column[j]['y1']
        for sgn, m in ((1.0, j), (-1.0, j + 1)):
            la = las[m]
            for cidx, w in waves[m]:
                ux, uy, syy, sxy, _sxx = _fd_field_coeffs(w, la, omega, p, Y)
                A[:, row + 0, cidx] += sgn * ux
                A[:, row + 1, cidx] += sgn * uy
                A[:, row + 2, cidx] += sgn * syy
                A[:, row + 3, cidx] += sgn * sxy
            if m == 0:
                ux, uy, syy, sxy, _sxx = _fd_field_coeffs(inc, la, omega, p, Y)
                b[:, row + 0] -= sgn * ux
                b[:, row + 1] -= sgn * uy
                b[:, row + 2] -= sgn * syy
                b[:, row + 3] -= sgn * sxy
        row += 4
    Ys = column[-1]['y1']
    laT = las[-1]
    for cidx, w in waves[-1]:
        _ux, _uy, syy, sxy, _sxx = _fd_field_coeffs(w, laT, omega, p, Ys)
        A[:, row + 0, cidx] += syy
        A[:, row + 1, cidx] += sxy
    if M == 0:
        _ux, _uy, syy, sxy, _sxx = _fd_field_coeffs(inc, laT, omega, p, Ys)
        b[:, row + 0] -= syy
        b[:, row + 1] -= sxy
    try:
        amps = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError:
        amps = np.zeros((nb, nunk), dtype=complex)
        for k in range(nb):
            amps[k] = np.linalg.lstsq(A[k], b[k], rcond=-1)[0]
    return {'amps': amps, 'las': las, 'waves': waves, 'inc': inc, 'column': column}

def _fd_eval_column(sol, omega, p, y):
    column = sol['column']
    seg_idx = 0
    for k in range(len(column) - 1, -1, -1):
        if column[k]['y0'] - 1e-6 <= y <= column[k]['y1'] + 1e-6:
            seg_idx = k
            break
    la = sol['las'][seg_idx]
    amps = sol['amps']
    ux = np.zeros_like(omega, dtype=complex)
    uy = np.zeros_like(omega, dtype=complex)
    syy = np.zeros_like(omega, dtype=complex)
    sxy = np.zeros_like(omega, dtype=complex)
    sxx = np.zeros_like(omega, dtype=complex)
    for cidx, w in sol['waves'][seg_idx]:
        cux, cuy, csyy, csxy, csxx = _fd_field_coeffs(w, la, omega, p, y)
        a = amps[:, cidx]
        ux += a * cux; uy += a * cuy
        syy += a * csyy; sxy += a * csxy; sxx += a * csxx
    if seg_idx == 0:
        cux, cuy, csyy, csxy, csxx = _fd_field_coeffs(sol['inc'], la, omega, p, y)
        ux += cux; uy += cuy
        syy += csyy; sxy += csxy; sxx += csxx
    vx = 1j * omega * ux
    vy = 1j * omega * uy
    return {'ux': ux, 'uy': uy, 'vx': vx, 'vy': vy, 'sxx': sxx, 'syy': syy, 'sxy': sxy}

def _column_seg(cs, vv, density, alpha_p, y0, y1, name):
    GG = density * cs * cs
    lam = GG * 2.0 * vv / (1.0 - 2.0 * vv)
    cp = math.sqrt((lam + 2.0 * GG) / density)
    return {'cs': cs, 'vv': vv, 'density': density, 'cp': cp, 'y0': y0, 'y1': y1, 'name': name}

def _build_column(strat, ymax_col, alpha_p, ymin):
    col = []
    for band in strat:
        y0, y1 = _band_bounds_at(band, ymax_col)
        if y1 <= y0 + 1e-5:
            continue
        col.append(_column_seg(band['mat'].cs, band['mat'].vv, band['mat'].density, alpha_p, y0, y1, band['name']))
    return col

def _fd_freefield_at_node(boundary, x0, y0, ymax_col, ctx):
    inp = _fd_input_spectrum(ctx)
    key = round(ymax_col, 4)
    sol = _FD_SOLVER_CACHE.get(key)
    if sol is None:
        column = _build_column(ctx.strat, ymax_col, ctx.p_horiz, ctx.ymin)
        sol = _fd_solve_column(column, ctx.p_horiz, inp['omega'], ctx.damp_terms,
                               bool(ctx.ffcfg.get('include_damping', True)))
        _FD_SOLVER_CACHE[key] = sol
    fields = _fd_eval_column(sol, inp['omega'], ctx.p_horiz, y0)
    shift = np.exp(-1j * inp['omega'] * ctx.p_horiz * x0)
    scale = inp['U0'] * shift
    out = {}
    for name in ('ux', 'uy', 'vx', 'vy', 'sxx', 'syy', 'sxy'):
        spec = np.zeros(inp['nfreq'], dtype=complex)
        spec[inp['idx']] = fields[name] * scale
        out[name] = np.fft.irfft(spec, n=inp['Nfft'])[:inp['Nout']]
    if boundary == 'l':
        sigmax = -out['sxx']; sigmay = -out['sxy']
    elif boundary == 'r':
        sigmax = out['sxx']; sigmay = out['sxy']
    else:
        sigmax = -out['sxy']; sigmay = -out['syy']
    t_out = np.arange(inp['Nout']) * inp['dt']
    return {'time': t_out, 'ux': out['ux'], 'uy': out['uy'],
            'dotux': out['vx'], 'dotuy': out['vy'], 'sigmax': sigmax, 'sigmay': sigmay}

def _fd_engine_selfcheck(logger=None):
    p0 = 1e-15
    col1 = [_column_seg(2000.0, 0.3, 2500.0, p0, 0.0, 400.0, 'bedrock')]
    om1 = 2.0 * math.pi * np.array([1.0, 3.0, 7.0])
    sol1 = _fd_solve_column(col1, p0, om1, {'bedrock': (0.0, 0.0)}, True)
    f1 = _fd_eval_column(sol1, om1, p0, 400.0)
    err1 = float(np.max(np.abs(np.abs(f1['ux']) - 2.0))) / 2.0
    
    col2 = [_column_seg(2000.0, 0.3, 2500.0, p0, 0.0, 200.0, 'bedrock'),
            _column_seg(800.0, 0.3, 2500.0, p0, 200.0, 400.0, 'cover')]
    om2 = np.array([2.0 * math.pi * 1.0])
    sol2 = _fd_solve_column(col2, p0, om2, {'bedrock': (0.0, 0.0), 'cover': (0.0, 0.0)}, True)
    f2 = _fd_eval_column(sol2, om2, p0, 400.0)
    kh = 2.0 * math.pi * 1.0 * 200.0 / 800.0
    ana = 2.0 / abs(complex(math.cos(kh), (800.0 / 2000.0) * math.sin(kh)))
    err2 = abs(abs(f2['ux'][0]) - ana) / ana
    result = {'halfspace_err': err1, 'single_layer_err': err2}
    if logger:
        log_step(logger, 'fd 引擎自检: 半空间误差=%.2e, 单层解析误差=%.2e（阈值 1e-3）', err1, err2)
    if err1 > 1e-3 or err2 > 1e-3:
        raise RuntimeError('fd 引擎自检失败: halfspace_err=%.3e, single_layer_err=%.3e' % (err1, err2))
    return result

# ==========================================================
#  结构化网格生成器
# ==========================================================

def generate_mesh(geom, site, mesh_size, mcfg, fc):
    """生成斜坡或平地结构化网格"""
    # 1. 计算 s_soil (最软层网格尺寸)
    cs_min = site.bedrock.cs
    for L in site.layers:
        if L.cs < cs_min:
            cs_min = L.cs
            
    if mcfg.get('auto', False) and fc is not None:
        epw = float(mcfg.get('elems_per_wavelength', 10))
        fmax = float(mcfg.get('fmax_factor', 2.5)) * fc
        s_soil = cs_min / (epw * fmax)
        s_soil = min(s_soil, mesh_size)
    else:
        s_soil = mesh_size
        
    s_soil = max(s_soil, float(mcfg.get('min_size', 0.2)))
    
    # 2. 纵向分段数 Ny1, Ny2
    max_ratio = float(mcfg.get('max_band_ratio', 4.0))
    s_bedrock = s_soil * (site.bedrock.cs / cs_min)
    s_bedrock = min(s_bedrock, s_soil * max_ratio)
    
    N_y1 = max(2, int(round(geom.bedrock_thickness / s_bedrock)))
    N_y2 = max(int(mcfg.get('min_elems_through_thickness', 6)), int(round(geom.H / s_soil)))
    
    # 3. 横向分段数 Nx1, Nx2, Nx3
    N_x1 = max(2, int(round(geom.left_flat / s_soil)))
    N_x2 = max(2, int(round(geom.w_slope / s_soil)))
    right_flat = geom.total_L - geom.left_flat - geom.w_slope
    N_x3 = max(2, int(round(right_flat / s_soil)))
    
    # 4. 横向节点坐标点 x_coords
    x_coords = []
    dx1 = geom.left_flat / N_x1
    for i in range(N_x1):
        x_coords.append(i * dx1)
    dx2 = geom.w_slope / N_x2
    for i in range(N_x2):
        x_coords.append(geom.left_flat + i * dx2)
    dx3 = right_flat / N_x3
    for i in range(N_x3 + 1):
        x_coords.append(geom.left_flat + geom.w_slope + i * dx3)
        
    Nx = N_x1 + N_x2 + N_x3
    Ny = N_y1 + N_y2
    
    # 5. 生成节点坐标二维格网
    nodes_grid = []
    for i in range(Nx + 1):
        x = x_coords[i]
        ys = _surface_y_at(x, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)
        
        y_col = []
        dy_bedrock = geom.bedrock_thickness / N_y1
        for j in range(N_y1):
            y_col.append(j * dy_bedrock)
            
        h_soil = ys - geom.bedrock_thickness
        dy_soil = h_soil / N_y2
        for k in range(N_y2 + 1):
            y_col.append(geom.bedrock_thickness + k * dy_soil)
            
        nodes_grid.append(y_col)
        
    return x_coords, nodes_grid, N_x1, N_x2, N_x3, N_y1, N_y2, s_soil

# ==========================================================
#  人工边界等效荷载计算
# ==========================================================

def _build_equivalent_forces(nodes_by_boundary, ctx):
    field_data = {}
    geom = ctx.geom
    for boundary in ('l', 'r', 'b'):
        for bn in nodes_by_boundary[boundary]:
            if boundary == 'l':
                ymax_col = ctx.ymax_l
            elif boundary == 'r':
                ymax_col = ctx.ymax_r
            else:
                ymax_col = _surface_y_at(bn.x, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)

            ff = _fd_freefield_at_node(boundary, bn.x, bn.y, ymax_col, ctx)
            time = ff['time']
            ux, uy = ff['ux'], ff['uy']
            dotux, dotuy = ff['dotux'], ff['dotuy']
            sigmax, sigmay = ff['sigmax'], ff['sigmay']

            if boundary in ('l', 'r'):
                fx = bn.kn * ux + bn.cn * dotux + bn.influence * sigmax
                fy = bn.kt * uy + bn.ct * dotuy + bn.influence * sigmay
            else:
                fx = bn.kt * ux + bn.ct * dotux + bn.influence * sigmax
                fy = bn.kn * uy + bn.cn * dotuy + bn.influence * sigmay

            field_data['{}-{}-fx'.format(bn.label, boundary)] = np.column_stack((time, fx))
            field_data['{}-{}-fy'.format(bn.label, boundary)] = np.column_stack((time, fy))
    return field_data

# ==========================================================
#  写入 case_meta.json
# ==========================================================

def _meta_f(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _meta_material(name, cs, vv, density, thickness=None):
    return {'name': str(name), 'cs': _meta_f(cs), 'vv': _meta_f(vv),
            'density': _meta_f(density), 'thickness': _meta_f(thickness)}

def _damping_meta(site, damping, geom=None):
    if not (damping and damping.get('enable')):
        return {'enable': False}
    fc = damping.get('fc')
    per_layer = []
    mats = [(site.bedrock, True)] + [(L, False) for L in site.layers]
    for mat, is_bedrock in mats:
        Q, xi = _damping_ratio_from_q(mat.cs, is_bedrock, damping)
        f_layer = None if (is_bedrock or geom is None) else _material_resonance_freq(mat, site, geom)
        a_ray, b_ray = _rayleigh_coeffs(xi, damping, fc, f_layer)
        per_layer.append({'name': str(mat.name), 'cs': _meta_f(mat.cs),
                          'Q': _meta_f(Q), 'xi': _meta_f(xi),
                          'f_layer': _meta_f(f_layer),
                          'alpha': _meta_f(a_ray), 'beta': _meta_f(b_ray)})
    return {'enable': True, 'method': damping.get('method'), 'fc': _meta_f(fc),
            'anchor': damping.get('anchor', 'input'), 'f_site': _meta_f(damping.get('f_site')),
            'harmonics_cover': _meta_f(damping.get('harmonics_cover')),
            'qs_factor': _meta_f(damping.get('qs_factor')), 'q_bedrock': _meta_f(damping.get('q_bedrock')),
            'f1_factor': _meta_f(damping.get('f1_factor')), 'f2_factor': _meta_f(damping.get('f2_factor')),
            'layers': per_layer}

def _write_case_meta(material_cfg, geom, site, mesh_size, script_name, logger, damping=None, ffcfg=None,
                     sgeom='horizontal', acc_path=None, selfcheck=None):
    try:
        bedrock = _meta_material(site.bedrock.name, site.bedrock.cs, site.bedrock.vv,
                                 site.bedrock.density, site.bedrock.thickness)
        layers = [_meta_material(L.name, L.cs, L.vv, L.density, L.thickness) for L in site.layers]
        geometry = {'i': geom.i, 'total_L': geom.total_L, 'left_flat': geom.left_flat,
                    'H_minus_h': geom.H_minus_h, 'h_over_H': geom.h_over_H,
                    'bedrock_thickness': geom.bedrock_thickness, 'H': geom.H, 'h': geom.h,
                    'w_slope': geom.w_slope,
                    'x_crest': geom.left_flat, 'x_toe': geom.left_flat + geom.w_slope}
        geometry = {k: _meta_f(v) for k, v in geometry.items()}
        n_finite = len(layers)
        has_bedrock = site.bedrock is not None
        n_total = n_finite + (1 if has_bedrock else 0)
        model_type = 'single' if n_total <= 1 else ('double' if n_total == 2 else 'multilayer')
        Hmh = geometry.get('H_minus_h')
        slope_height = Hmh if Hmh is not None else geometry.get('h')
        vs_bedrock = bedrock['cs'] if has_bedrock else None
        vs_surface = layers[0]['cs'] if layers else vs_bedrock
        vs_cover = layers[-1]['cs'] if layers else vs_surface
        vs_min = min([L['cs'] for L in layers]) if layers else vs_bedrock
        
        vr_over_vs2 = (vs_bedrock / vs_cover) if (vs_bedrock and vs_cover) else None
        vs1_over_vs2 = (vs_surface / vs_cover) if (vs_surface and vs_cover) else None
        a0_base = (2.0 * slope_height / vs_cover) if (slope_height and vs_cover) else None
        fc_meta = (damping or {}).get('fc')
        a0_val = (fc_meta * a0_base) if (fc_meta and a0_base) else None
        derived = {
            'n_finite_layers': n_finite,
            'n_layers_total': n_total,
            'vs_bedrock': _meta_f(vs_bedrock),
            'vs_surface': _meta_f(vs_surface),
            'vs_cover': _meta_f(vs_cover),
            'vs_min': _meta_f(vs_min),
            'vr_over_vs2': _meta_f(vr_over_vs2),
            'vs1_over_vs2': _meta_f(vs1_over_vs2),
            'slope_height': _meta_f(slope_height),
            'a0_base': _meta_f(a0_base),
            'a0': _meta_f(a0_val),
        }
        out_dir = os.path.abspath(os.getcwd())
        meta = {
            'schema_version': 1,
            'model_type': model_type,
            'model_script': str(script_name),
            'incident_angle': _meta_f(material_cfg['angle']),
            'surface_geometry': str(sgeom),
            'mesh_size': _meta_f(mesh_size),
            'geometry': geometry,
            'bedrock': bedrock,
            'layers': layers,
            'derived': derived,
            'damping': _damping_meta(site, damping, geom),
            'record': None,
            'extra': {},
            'folder': os.path.basename(out_dir.rstrip('/\\')),
        }
        
        ang_deg = float(material_cfg['angle'])
        alpha_r = math.radians(ang_deg if abs(ang_deg) > 1e-12 else 1e-10)
        mat_b = _compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)
        fs = _compute_free_surface_sv_coeff(alpha_r, mat_b['cp'], mat_b['cs'])
        factor_h = (1.0 - fs['A1']) * math.cos(alpha_r) + fs['A2'] * math.sin(fs['beta'])
        factor_v = -((1.0 + fs['A1']) * math.sin(alpha_r) + fs['A2'] * math.cos(fs['beta']))
        meta['ff_normalization'] = {
            'method': 'bedrock_halfspace_free_surface',
            'A1': _meta_f(fs['A1']), 'A2': _meta_f(fs['A2']),
            'beta_deg': _meta_f(math.degrees(fs['beta'])),
            'factor_h': _meta_f(factor_h),
            'factor_v': _meta_f(factor_v),
            'note': 'TAF_h=PGA_h/(factor_h*PGA_in); TAF_v=PGA_v/(factor_h*PGA_in)',
        }
        meta['freefield'] = {
            'engine': (ffcfg or {}).get('engine'),
            'include_damping': (ffcfg or {}).get('include_damping'),
        }
        
        ff_theory = None
        if acc_path and os.path.isfile(acc_path):
            try:
                rec = np.loadtxt(acc_path)
                acc0 = rec[:, 1]
                dt0 = float(rec[1, 0] - rec[0, 0])
                strat_t = _build_stratigraphy(site, geom, ymin=0.0, surface_geometry=sgeom)
                damp_terms = _band_damping_terms(strat_t, damping)
                p0 = math.sin(alpha_r) / mat_b['cs']
                Nfft = _next_pow2(len(acc0) * 4)
                A0 = np.fft.rfft(acc0, n=Nfft)
                freqs0 = np.fft.rfftfreq(Nfft, dt0)
                mask0 = np.abs(A0) > 1e-7 * float(np.max(np.abs(A0)))
                mask0[0] = False
                idx0 = np.nonzero(mask0)[0]
                om0 = 2.0 * math.pi * freqs0[idx0]
                incl = bool((ffcfg or {}).get('include_damping', True))
                denom0 = factor_h * float(np.max(np.abs(acc0)))
                ff_theory = {'fc_used': _meta_f((damping or {}).get('fc')),
                             'damped': bool(damping and damping.get('enable') and incl),
                             'note': 'fd 引擎一维柱地表 PGA/(factor_h*PGA_in)；FE 远场台阶应与之一致(±5%)'}
                for tag, ys in (('left', geom.H_upper), ('right', geom.H_lower)):
                    col_t = _build_column(strat_t, ys, p0, 0.0)
                    sol_t = _fd_solve_column(col_t, p0, om0, damp_terms, incl)
                    fld = _fd_eval_column(sol_t, om0, p0, ys)
                    spec_t = np.zeros(len(freqs0), dtype=complex)
                    spec_t[idx0] = fld['ux'] * A0[idx0]
                    ax0 = np.fft.irfft(spec_t, n=Nfft)
                    spec_t = np.zeros(len(freqs0), dtype=complex)
                    spec_t[idx0] = fld['uy'] * A0[idx0]
                    ay0 = np.fft.irfft(spec_t, n=Nfft)
                    ff_theory[tag] = {'taf_h': _meta_f(float(np.max(np.abs(ax0))) / denom0),
                                      'taf_v': _meta_f(float(np.max(np.abs(ay0))) / denom0),
                                      'surface_y': _meta_f(ys),
                                      'layers': [seg['name'] for seg in col_t]}
            except Exception as _fe:
                if logger:
                    log_step(logger, 'ff_theory 计算失败(不影响建模): %s', str(_fe))
                ff_theory = None
        meta['ff_theory'] = ff_theory
        meta['selfcheck'] = selfcheck
        text = json.dumps(meta, ensure_ascii=False, indent=2, default=_meta_f)
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        path = os.path.join(out_dir, 'case_meta.json')
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(text)
        if logger:
            log_step(logger, 'case_meta.json 已写出: %s', path)
        return ff_theory
    except Exception as _e:
        if logger:
            log_step(logger, 'case_meta.json 写出失败: %s', str(_e))
        return None

# ==========================================================
#  参数深度合并与加载
# ==========================================================

def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_case_config():
    global material_cfg, geometry_cfg, damping_cfg, mesh_cfg, time_cfg, freefield_cfg, run_cfg
    path = os.path.join(os.getcwd(), 'case_config.json')
    if not os.path.isfile(path):
        log_step(None, '未发现 case_config.json，使用脚本内默认配置')
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        if isinstance(cfg.get('material_cfg'), dict):
            material_cfg = _deep_merge(material_cfg, cfg['material_cfg'])
        if isinstance(cfg.get('geometry_cfg'), dict):
            geometry_cfg = _deep_merge(geometry_cfg, cfg['geometry_cfg'])
        if isinstance(cfg.get('damping_cfg'), dict):
            damping_cfg = _deep_merge(damping_cfg, cfg['damping_cfg'])
        if isinstance(cfg.get('mesh_cfg'), dict):
            mesh_cfg = _deep_merge(mesh_cfg, cfg['mesh_cfg'])
        if cfg.get('mesh_size') is not None:
            mesh_cfg['size'] = cfg['mesh_size']
        if isinstance(cfg.get('time_cfg'), dict):
            time_cfg = _deep_merge(time_cfg, cfg['time_cfg'])
        if isinstance(cfg.get('freefield_cfg'), dict):
            freefield_cfg = _deep_merge(freefield_cfg, cfg['freefield_cfg'])
        if isinstance(cfg.get('run_cfg'), dict):
            run_cfg = _deep_merge(run_cfg, cfg['run_cfg'])
        log_step(None, '已成功加载 case_config.json 并覆盖默认配置')
    except Exception as e:
        log_step(None, '加载 case_config.json 失败: %s', str(e))

# ==========================================================
#  OpenSees 建模与求解核心
# ==========================================================

def pick_material_params(xc, yc, strat, geom, surface_geometry):
    """确定单元在坐标 (xc, yc) 处的弹性参数"""
    ys = _surface_y_at(xc, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)
    tol = 1e-4
    # 从下到上遍历各带（由 _build_stratigraphy 保证 [0] 是基岩，依次向上）
    for idx, band in enumerate(strat):
        y0, y1 = _band_bounds_at(band, ys)
        if y0 - tol <= yc < y1 + tol:
            return idx, band
    return len(strat) - 1, strat[-1] # 兜底最顶层

def build_site_objects(mat_cfg, geom_cfg):
    """由配置组装 Site 对象"""
    cs_bedrock = _compute_wave_speed_from_elastic_modulus(
        mat_cfg['bedrock']['elastic_modulus'],
        mat_cfg['bedrock']['poisson_ratio'],
        mat_cfg['bedrock']['density'])
    bedrock = Material(cs=cs_bedrock, vv=mat_cfg['bedrock']['poisson_ratio'],
                       density=mat_cfg['bedrock']['density'], thickness=None, name='Bedrock')
    
    layers_cfg = mat_cfg.get('layers', [])
    layers = []
    fixed_thicknesses = []
    nL = len(layers_cfg)
    for idx, lc in enumerate(layers_cfg):
        cs = cs_bedrock / lc['velocity_ratio']
        is_bottom = (idx == nL - 1)
        thickness = None if is_bottom else lc['thickness']
        if not is_bottom:
            fixed_thicknesses.append(lc['thickness'])
        layers.append(Material(cs=cs, vv=lc['poisson_ratio'], density=lc['density'],
                               thickness=thickness, name=lc['name']))
        
    site = Site(bedrock=bedrock, layers=layers, bedrock_thickness=geom_cfg['bedrock_thickness'])
    return site, fixed_thicknesses

def run_one_opensees_model(model_name, geom, site, strat, angle, acc_file, s_size, m_cfg, damp, ff_cfg, scene):
    """生成并运行单个 OpenSees 动力分析（slope 或 flat）"""
    log_step(None, '--- 开始建模求解 OpenSees 模型 [%s] ---', model_name)
    
    # 1. 物理常数与对拍自检
    alpha_deg = float(angle)
    if alpha_deg == 0:
        alpha_deg = 1e-10
    alpha_r = math.radians(alpha_deg)
    
    mat_bedrock = _compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)
    cs1, cp1 = mat_bedrock['cs'], mat_bedrock['cp']
    p_horiz = math.sin(alpha_r) / cs1
    beta1 = _safe_arcsin(cp1 * math.sin(alpha_r) / cs1)
    
    # 2. 读取输入地震波
    ACC = np.loadtxt(acc_file)
    time_arr = ACC[:, 0]
    acc = ACC[:, 1]
    dt = ACC[1, 0] - ACC[0, 0]
    
    # 积分速度与位移
    vel, _vel_slope = _integrate_acc_to_velocity(acc, dt, time_arr)
    dis = np.zeros_like(vel)
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2.0 * dt)
    VEL = np.column_stack((time_arr, vel))
    DIS = np.column_stack((time_arr, dis))
    
    # 3. 构造场地网格剖分
    # 若是 flat 场景，派生出平坦场地几何
    geom_used = make_flat_geometry(geom) if scene == 'flat' else geom
    x_coords, nodes_grid, N_x1, N_x2, N_x3, N_y1, N_y2, s_soil = generate_mesh(
        geom_used, site, s_size, m_cfg, damp['fc'])
    
    Nx = N_x1 + N_x2 + N_x3
    Ny = N_y1 + N_y2
    
    log_step(None, '网格剖分: Nx=%d (左段=%d, 坡段=%d, 右段=%d), Ny=%d (基岩段=%d, 土层段=%d)',
             Nx, N_x1, N_x2, N_x3, Ny, N_y1, N_y2)
    log_step(None, '总节点数=%d, 总单元数=%d', (Nx + 1) * (Ny + 1), Nx * Ny)
    
    # 4. 计算各材料带的阻尼系数与弹簧系数需要的局部材料归属
    ymax_l = geom_used.H_upper
    ymax_r = geom_used.H_lower
    ymin = 0.0
    
    # 边界节点集组装
    # 我们确定三个边界
    l_nodes = [] # 左边界节点，j=0..Ny
    r_nodes = [] # 右边界节点，j=0..Ny
    b_nodes = [] # 底边界节点，i=0..Nx
    
    # 我们为节点分配全局 nodeTag
    # nodeTag = i * (Ny + 1) + j + 1
    # 边界节点信息
    def get_node_tag(i, j):
        return i * (Ny + 1) + j + 1
    
    # 计算影响长度与弹簧阻尼系数
    def get_material_for_coord(x, y):
        idx, band = pick_material_params(x, y, strat, geom_used, material_cfg['surface_geometry'])
        return _compute_material_params(band['mat'].cs, band['mat'].vv, band['mat'].density)

    # 弹簧的参考长度 R
    ymax = max(ymax_l, ymax_r)
    
    # 4.1 左边界
    l_nodes_bn = []
    for j in range(Ny + 1):
        tag = get_node_tag(0, j)
        x = x_coords[0]
        y = nodes_grid[0][j]
        # 计算影响长度
        if j == 0:
            inf = (nodes_grid[0][1] - nodes_grid[0][0]) / 2.0
        elif j == Ny:
            inf = (nodes_grid[0][Ny] - nodes_grid[0][Ny-1]) / 2.0
        else:
            inf = (nodes_grid[0][j+1] - nodes_grid[0][j-1]) / 2.0
            
        mat = get_material_for_coord(x, y)
        kn = mat['GG'] / 2.0 / ymax * inf
        cn = mat['density'] * mat['cp'] * inf
        kt = mat['GG'] / 4.0 / ymax * inf
        ct = mat['density'] * mat['cs'] * inf
        l_nodes_bn.append(BoundaryNode(label=tag, x=x, y=y, influence=inf, kn=kn, cn=cn, kt=kt, ct=ct))
        
    # 4.2 右边界
    r_nodes_bn = []
    for j in range(Ny + 1):
        tag = get_node_tag(Nx, j)
        x = x_coords[Nx]
        y = nodes_grid[Nx][j]
        if j == 0:
            inf = (nodes_grid[Nx][1] - nodes_grid[Nx][0]) / 2.0
        elif j == Ny:
            inf = (nodes_grid[Nx][Ny] - nodes_grid[Nx][Ny-1]) / 2.0
        else:
            inf = (nodes_grid[Nx][j+1] - nodes_grid[Nx][j-1]) / 2.0
            
        mat = get_material_for_coord(x, y)
        kn = mat['GG'] / 2.0 / ymax * inf
        cn = mat['density'] * mat['cp'] * inf
        kt = mat['GG'] / 4.0 / ymax * inf
        ct = mat['density'] * mat['cs'] * inf
        r_nodes_bn.append(BoundaryNode(label=tag, x=x, y=y, influence=inf, kn=kn, cn=cn, kt=kt, ct=ct))
        
    # 4.3 底边界
    b_nodes_bn = []
    for i in range(Nx + 1):
        tag = get_node_tag(i, 0)
        x = x_coords[i]
        y = nodes_grid[i][0]
        if i == 0:
            inf = (x_coords[1] - x_coords[0]) / 2.0
        elif i == Nx:
            inf = (x_coords[Nx] - x_coords[Nx-1]) / 2.0
        else:
            inf = (x_coords[i+1] - x_coords[i-1]) / 2.0
            
        mat = get_material_for_coord(x, y)
        kn = mat['GG'] / 2.0 / ymax * inf
        cn = mat['density'] * mat['cp'] * inf
        kt = mat['GG'] / 4.0 / ymax * inf
        ct = mat['density'] * mat['cs'] * inf
        b_nodes_bn.append(BoundaryNode(label=tag, x=x, y=y, influence=inf, kn=kn, cn=cn, kt=kt, ct=ct))

    # 4.4 等效力计算
    damp_terms = _band_damping_terms(strat, damp)
    ctx = FreeFieldCtx(
        site=site, geom=geom_used, strat=strat,
        ymax_l=ymax_l, ymax_r=ymax_r, ymin=ymin,
        alpha=alpha_r, beta_p=beta1, p_horiz=p_horiz,
        GG=mat_bedrock['GG'], lam=mat_bedrock['lam'], cs=cs1, cp=cp1,
        VEL=VEL, DIS=DIS, dt=dt, time_arr=time_arr, max_reflect_order=3,
        acc=acc, damp_terms=damp_terms, ffcfg=ff_cfg)
    
    nodes_by_boundary = {'l': l_nodes_bn, 'r': r_nodes_bn, 'b': b_nodes_bn}
    field_data = _build_equivalent_forces(nodes_by_boundary, ctx)
    log_step(None, '边界节点等效荷载计算完成，共计 %d 条边界力时程', len(field_data))
    
    # 5. 生成 OpenSees Tcl 脚本
    tcl_file = 'model_{}.tcl'.format(scene)
    with open(tcl_file, 'w', encoding='utf-8') as f:
        f.write("# OpenSees Two-Dimensional Elastic Slope Model\n")
        f.write("# Scene: {}\n\n".format(scene))
        f.write("wipe\n")
        f.write("model BasicBuilder -ndm 2 -ndf 2\n\n")
        
        # 5.1 材料定义
        # 为 strat 里的每一带定义材料
        # 在 Tcl 中，用 nDMaterial ElasticIsotropic $matTag $E $v $rho
        f.write("# Materials Definition\n")
        for idx, band in enumerate(strat):
            mat = band['mat']
            params = _compute_material_params(mat.cs, mat.vv, mat.density)
            # $matTag 可以设为 idx + 1
            f.write("nDMaterial ElasticIsotropic {} {:.6e} {:.3f} {:.3f}\n".format(
                idx + 1, params['EE'], params['vv'], params['density']))
        f.write("\n")
        
        # 5.2 节点定义
        f.write("# Nodes Definition\n")
        for i in range(Nx + 1):
            for j in range(Ny + 1):
                tag = get_node_tag(i, j)
                x = x_coords[i]
                y = nodes_grid[i][j]
                f.write("node {:d} {:.6f} {:.6f}\n".format(tag, x, y))
        f.write("\n")
        
        # 5.3 单元定义与材料划分
        f.write("# Elements Definition\n")
        # 对每一个单元，通过质心落带分配材料
        elements_by_band = {idx: [] for idx in range(len(strat))}
        for i in range(Nx):
            for j in range(Ny):
                ele_tag = i * Ny + j + 1
                n1 = get_node_tag(i, j)
                n2 = get_node_tag(i+1, j)
                n3 = get_node_tag(i+1, j+1)
                n4 = get_node_tag(i, j+1)
                
                # 质心坐标
                xc = (x_coords[i] + x_coords[i+1]) / 2.0
                yc = (nodes_grid[i][j] + nodes_grid[i+1][j] + nodes_grid[i+1][j+1] + nodes_grid[i][j+1]) / 4.0
                
                band_idx, band = pick_material_params(xc, yc, strat, geom_used, material_cfg['surface_geometry'])
                elements_by_band[band_idx].append(ele_tag)
                
                # element quad $eleTag $n1 $n2 $n3 $n4 $thick PlaneStrain $matTag
                f.write("element quad {:d} {:d} {:d} {:d} {:d} 1.0 \"PlaneStrain\" {:d}\n".format(
                    ele_tag, n1, n2, n3, n4, band_idx + 1))
        f.write("\n")
        
        # 5.4 多层 Rayleigh 阻尼分配
        if damp and damp.get('enable'):
            f.write("# Rayleigh Damping Definition via Region\n")
            for idx, band in enumerate(strat):
                eles = elements_by_band[idx]
                if not eles:
                    continue
                # 计算这层的 Rayleigh 阻尼
                _Q, xi = _damping_ratio_from_q(band['mat'].cs, idx == 0, damp)
                f_layer = None if idx == 0 else _band_resonance_freq(band)
                alpha, beta = _rayleigh_coeffs(xi, damp, damp['fc'], f_layer)
                
                f.write("region {:d} -ele".format(idx + 10))
                for ele in eles:
                    f.write(" {:d}".format(ele))
                f.write(" -rayleigh {:.6e} 0.0 {:.6e} 0.0\n".format(alpha, beta))
            f.write("\n")
            
        # 5.5 边界弹簧阻尼定义 (zeroLength 并联)
        f.write("# Artificial Boundary Spring-Dashpots Definition\n")
        
        created_ground_nodes = set()
        def write_boundary_springs(boundary, dof_normal, dof_tangent, boundary_nodes):
            if boundary == 'l':
                mat_offset = 2000000
            elif boundary == 'r':
                mat_offset = 2400000
            else:
                mat_offset = 2800000

            for bn in boundary_nodes:
                bn_tag = bn.label
                ground_tag = bn_tag + 100000
                if ground_tag not in created_ground_nodes:
                    f.write("node {:d} {:.6f} {:.6f}\n".format(ground_tag, bn.x, bn.y))
                    f.write("fix {:d} 1 1\n".format(ground_tag))
                    created_ground_nodes.add(ground_tag)
                
                # 5.5.1 法向弹簧-阻尼
                mat_kn = bn_tag + mat_offset
                mat_cn = bn_tag + mat_offset + 100000
                f.write("uniaxialMaterial Elastic {:d} {:.6e}\n".format(mat_kn, bn.kn))
                f.write("uniaxialMaterial Viscous {:d} {:.6e} 1.0\n".format(mat_cn, bn.cn))
                # 建立零长度单元
                ele_tag_kn = bn_tag + 200000
                ele_tag_cn = bn_tag + 210000
                if boundary == 'r':
                    ele_tag_kn += 100000
                    ele_tag_cn += 100000
                elif boundary == 'b':
                    ele_tag_kn += 200000
                    ele_tag_cn += 200000
                    
                f.write("element zeroLength {:d} {:d} {:d} -mat {:d} -dir {:d}\n".format(
                    ele_tag_kn, ground_tag, bn_tag, mat_kn, dof_normal))
                f.write("element zeroLength {:d} {:d} {:d} -mat {:d} -dir {:d}\n".format(
                    ele_tag_cn, ground_tag, bn_tag, mat_cn, dof_normal))
                
                # 5.5.2 切向弹簧-阻尼
                mat_kt = bn_tag + mat_offset + 200000
                mat_ct = bn_tag + mat_offset + 300000
                f.write("uniaxialMaterial Elastic {:d} {:.6e}\n".format(mat_kt, bn.kt))
                f.write("uniaxialMaterial Viscous {:d} {:.6e} 1.0\n".format(mat_ct, bn.ct))
                
                ele_tag_kt = bn_tag + 220000
                ele_tag_ct = bn_tag + 230000
                if boundary == 'r':
                    ele_tag_kt += 100000
                    ele_tag_ct += 100000
                elif boundary == 'b':
                    ele_tag_kt += 200000
                    ele_tag_ct += 200000
                    
                f.write("element zeroLength {:d} {:d} {:d} -mat {:d} -dir {:d}\n".format(
                    ele_tag_kt, ground_tag, bn_tag, mat_kt, dof_tangent))
                f.write("element zeroLength {:d} {:d} {:d} -mat {:d} -dir {:d}\n".format(
                    ele_tag_ct, ground_tag, bn_tag, mat_ct, dof_tangent))

        # 左边界 (l)：法向为 1，切向为 2
        f.write("# Left Boundary Springs\n")
        write_boundary_springs('l', 1, 2, l_nodes_bn)
        # 右边界 (r)：法向为 1，切向为 2
        f.write("# Right Boundary Springs\n")
        write_boundary_springs('r', 1, 2, r_nodes_bn)
        # 底边界 (b)：法向为 2，切向为 1
        f.write("# Bottom Boundary Springs\n")
        write_boundary_springs('b', 2, 1, b_nodes_bn)
        f.write("\n")
        
        # 5.6 边界等效力时程载荷定义 (pattern Plain)
        f.write("# Boundary Equivalent Forces Definition\n")
        dt_used = float(dt)
        for boundary in ('l', 'r', 'b'):
            if boundary == 'l':
                ts_offset_x = 5000000
                ts_offset_y = 5200000
                pat_offset_x = 7000000
                pat_offset_y = 7200000
            elif boundary == 'r':
                ts_offset_x = 5400000
                ts_offset_y = 5600000
                pat_offset_x = 7400000
                pat_offset_y = 7600000
            else:
                ts_offset_x = 5800000
                ts_offset_y = 6000000
                pat_offset_x = 7800000
                pat_offset_y = 8000000

            for bn in nodes_by_boundary[boundary]:
                bn_tag = bn.label
                fx_name = '{}-{}-fx'.format(bn_tag, boundary)
                fy_name = '{}-{}-fy'.format(bn_tag, boundary)
                fx_data = field_data[fx_name][:, 1]
                fy_data = field_data[fy_name][:, 1]
                
                # 写入 X 向力时程
                ts_tag_x = bn_tag + ts_offset_x
                f.write("timeSeries Path {:d} -dt {:.6f} -values {{".format(ts_tag_x, dt_used))
                for val in fx_data:
                    f.write(" {:.6e}".format(val))
                f.write("}\n")
                
                # 写入 Y 向力时程
                ts_tag_y = bn_tag + ts_offset_y
                f.write("timeSeries Path {:d} -dt {:.6f} -values {{".format(ts_tag_y, dt_used))
                for val in fy_data:
                    f.write(" {:.6e}".format(val))
                f.write("}\n")
                
                # 定义 Plain Pattern 荷载模式
                pat_tag_x = bn_tag + pat_offset_x
                f.write("pattern Plain {:d} {:d} {{\n".format(pat_tag_x, ts_tag_x))
                f.write("    load {:d} 1.0 0.0\n".format(bn_tag))
                f.write("}\n")
                
                pat_tag_y = bn_tag + pat_offset_y
                f.write("pattern Plain {:d} {:d} {{\n".format(pat_tag_y, ts_tag_y))
                f.write("    load {:d} 0.0 1.0\n".format(bn_tag))
                f.write("}\n")

        f.write("\n")
        
        # 5.7 顶部地表节点 Recorder 记录器
        # 我们获取地表所有的节点 Tag
        top_labels = []
        for i in range(Nx + 1):
            top_labels.append(get_node_tag(i, Ny))
            
        f.write("# Top Surface Recorders\n")
        # 导出水平向和竖向加速度
        f.write("recorder Node -file surface_acc_h.out -time -node")
        for tag in top_labels:
            f.write(" {:d}".format(tag))
        f.write(" -dof 1 accel\n")
        
        f.write("recorder Node -file surface_acc_v.out -time -node")
        for tag in top_labels:
            f.write(" {:d}".format(tag))
        f.write(" -dof 2 accel\n\n")
        
        # 5.8 动力隐式 HHT 求解与动力分析步
        f.write("# Dynamics Implicit HHT Analysis\n")
        f.write("constraints Plain\n")
        f.write("numberer RCM\n")
        f.write("system BandGeneral\n")
        f.write("test NormDispIncr 1.0e-5 10 0\n")
        f.write("algorithm Newton\n")
        f.write("integrator HHT 0.95\n")
        f.write("analysis Transient\n")
        
        Nout = len(fx_data) # 输出点数（时程的长度）
        f.write("analyze {:d} {:.6f}\n".format(Nout, dt_used))
        f.write("wipe\n")
        
    # 6. 执行 OpenSees
    log_step(None, '写入 model_{}.tcl 完成，开始运行 OpenSees 动力求解...'.format(scene))
    t_start = time.time()
    try:
        proc = subprocess.run([OPENSEES_EXE, tcl_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            log_step(None, 'OpenSees 求解异常，错误信息:\n%s', proc.stderr)
            raise RuntimeError('OpenSees 求解失败！')
    except Exception as e:
        log_step(None, '无法启动 OpenSees 可执行文件 %s 或运行出错: %s', OPENSEES_EXE, str(e))
        raise
    t_end = time.time()
    log_step(None, 'OpenSees 动力分析结束，耗时: %.2fs', t_end - t_start)
    
    # 7. 读取 OpenSees 输出文件并整理为 Abaqus 兼容 CSV 格式
    # 7.1 读取水平加速度 surface_acc_h.out 与 竖向加速度 surface_acc_v.out
    # 两个文件格式都是: 第一列为时间，其后各列依次为 top_labels 中各节点的加速度
    acc_h_raw = np.loadtxt('surface_acc_h.out')
    acc_v_raw = np.loadtxt('surface_acc_v.out')
    
    time_series = acc_h_raw[:, 0]
    acc_h_nodes = acc_h_raw[:, 1:]
    acc_v_nodes = acc_v_raw[:, 1:]
    
    # 7.2 写入 TIMESERIES 和 PGA CSV 文件
    csv_stem = model_name
    if csv_stem.startswith('job-'):
        csv_stem = csv_stem[4:]
    
    timeseries_csv_name = 'TIMESERIES-{}.csv'.format(csv_stem)
    pga_csv_name = 'PGA-{}.csv'.format(csv_stem)
    
    # 7.2.1 写入时程 CSV 文件 TIMESERIES-{name}.csv
    with open(timeseries_csv_name, 'w', encoding='utf-8') as f_ts:
        writer_ts = csv.writer(f_ts, lineterminator='\n')
        header = ['Time']
        for tag in top_labels:
            header.append('Node_{}_Accel_h'.format(tag))
        for tag in top_labels:
            header.append('Node_{}_Accel_v'.format(tag))
        writer_ts.writerow(header)
        
        for k in range(len(time_series)):
            row = ['{:.6e}'.format(time_series[k])]
            for i in range(len(top_labels)):
                row.append('{:.6e}'.format(acc_h_nodes[k, i]))
            for i in range(len(top_labels)):
                row.append('{:.6e}'.format(acc_v_nodes[k, i]))
            writer_ts.writerow(row)
            
    log_step(None, '时程数据已写入: %s', timeseries_csv_name)
    
    # 7.2.2 计算 PGA 峰值并写入 PGA-{name}.csv
    results = []
    # 从中计算 PGA 峰值和发生时间
    for i, tag in enumerate(top_labels):
        x = x_coords[i]
        y = nodes_grid[i][Ny]
        
        # 水平
        pga_h_val = np.max(np.abs(acc_h_nodes[:, i]))
        idx_h = np.argmax(np.abs(acc_h_nodes[:, i]))
        pga_h_time = time_series[idx_h]
        
        # 竖向
        pga_v_val = np.max(np.abs(acc_v_nodes[:, i]))
        idx_v = np.argmax(np.abs(acc_v_nodes[:, i]))
        pga_v_time = time_series[idx_v]
        
        results.append({
            'node_label': tag,
            'x': x,
            'y': y,
            'PGA_h': pga_h_val,
            'PGA_v': pga_v_val,
            'peak_h_time': pga_h_time,
            'peak_v_time': pga_v_time
        })
        
    with open(pga_csv_name, 'w', encoding='utf-8') as f_pga:
        writer_pga = csv.writer(f_pga, lineterminator='\n')
        pga_header = ['node_label', 'x', 'y', 'PGA_h', 'PGA_v', 'peak_h_time', 'peak_v_time']
        writer_pga.writerow(pga_header)
        
        for r in results:
            row = [
                '{:d}'.format(r['node_label']),
                '{:.6f}'.format(r['x']),
                '{:.6f}'.format(r['y']),
                '{:.6f}'.format(r['PGA_h']),
                '{:.6f}'.format(r['PGA_v']),
                '{:.6f}'.format(r['peak_h_time']),
                '{:.6f}'.format(r['peak_v_time'])
            ]
            writer_pga.writerow(row)
            
    log_step(None, 'PGA 峰值数据已写入: %s，地表总节点数=%d', pga_csv_name, len(results))
    
    # 8. 清理 OpenSees 生成的临时 out 文件和 Tcl 文件
    for ext in ('surface_acc_h.out', 'surface_acc_v.out', tcl_file):
        if os.path.exists(ext):
            os.remove(ext)

# ==========================================================
#  主控制入口
# ==========================================================

def main():
    logger = 'VAB_oblique_TAF_multilayer_opensees.log'
    log_step(logger, '=== OpenSees 动力数值模拟一体化工作流开始 ===')
    t_all_start = time.time()
    
    # 1. 自动读取并覆盖 case_config.json 注入
    _load_case_config()
    
    # 2. 从当前目录自动搜索加速度记录波 *.txt
    cwd = os.getcwd()
    txt_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.txt')])
    if not txt_files:
        log_step(None, '错误: 未能在当前目录 %s 寻找到任何输入波 *.txt 文件！', cwd)
        sys.exit(1)
        
    log_step(None, '检索到输入波记录: %s', ', '.join(txt_files))
    
    # 3. 构造 site 场地和斜坡 geom 几何
    site, fixed_thicknesses = build_site_objects(material_cfg, geometry_cfg)
    geom = make_geometry(
        geometry_cfg['total_L'], geometry_cfg['H_minus_h'], geometry_cfg['i'],
        geometry_cfg['h_over_H'], geometry_cfg['left_flat'], geometry_cfg['bedrock_thickness'],
        fixed_thicknesses
    )
    
    # 4. 执行一维对拍自检
    selfcheck = _fd_engine_selfcheck()
    
    # 5. 主频确定与阻尼解析
    acc_path = os.path.join(cwd, txt_files[0])
    acc_test = np.loadtxt(acc_path)
    dt_test = acc_test[1, 0] - acc_test[0, 0]
    fc_est = _estimate_dominant_freq(acc_test[:, 1], dt_test)
    
    damp = _resolve_damping(damping_cfg, fc_est)
    
    # 6. 写出 case_meta.json 并计算一维理论解析 TAF 值
    strat = _build_stratigraphy(site, geom, surface_geometry=material_cfg['surface_geometry'])
    ff_theory = _write_case_meta(
        material_cfg=material_cfg,
        geom=geom,
        site=site,
        mesh_size=mesh_cfg['size'],
        script_name='VAB_oblique_TAF_multilayer_opensees_v1.py',
        logger=logger,
        damping=damp,
        ffcfg=freefield_cfg,
        sgeom=material_cfg['surface_geometry'],
        acc_path=acc_path,
        selfcheck=selfcheck
    )
    
    if ff_theory:
        log_step(None, '一维解析解析地表理论放大系数计算完成: 左边界水平=%.3f, 右边界水平=%.3f',
                 ff_theory['left']['taf_h'], ff_theory['right']['taf_h'])
        
    # 7. 分别建模求解 slope (和 flat, 如果 run_cfg['run_flat'] 开启)
    for f in txt_files:
        base_name = os.path.splitext(os.path.basename(f))[0]
        
        # 7.1 slope 边坡模型求解与后处理
        model_slope = 'job-{}-slope'.format(base_name)
        run_one_opensees_model(
            model_name=model_slope,
            geom=geom,
            site=site,
            strat=strat,
            angle=material_cfg['angle'],
            acc_file=f,
            s_size=mesh_cfg['size'],
            m_cfg=mesh_cfg,
            damp=damp,
            ff_cfg=freefield_cfg,
            scene='slope'
        )
        
        # 7.2 flatref 水平自由场模型求解与后处理 (平坦对照模型，可选)
        if run_cfg.get('run_flat', False):
            model_flat = 'job-{}-flat'.format(base_name)
            run_one_opensees_model(
                model_name=model_flat,
                geom=geom,
                site=site,
                strat=strat,
                angle=material_cfg['angle'],
                acc_file=f,
                s_size=mesh_cfg['size'],
                m_cfg=mesh_cfg,
                damp=damp,
                ff_cfg=freefield_cfg,
                scene='flat'
            )
            
    log_step(None, '=== OpenSees 动力数值模拟一体化工作流成功结束 (总耗时: %.2fs) ===', time.time() - t_all_start)

if __name__ == '__main__':
    main()
