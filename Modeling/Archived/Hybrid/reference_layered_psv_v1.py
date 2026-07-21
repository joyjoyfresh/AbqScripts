# -*- coding: utf-8 -*-
"""独立 P–SV 层状介质频域参考解。

本文件不导入 Abaqus，也不调用 Hybrid v2 的任何自由场函数。采用平面波状态向量
``[u_x, u_y, sigma_xy, sigma_yy]``，在有限层内用 P/SV 上、下行波传播矩阵，
在底部半空间设置一个上行 SV 入射波，在顶部施加自由表面牵引为零条件。
该程序用于 F0-2/V2 的独立参考，不承担有限元求解和工程设计系数计算。
"""

from __future__ import division

import math

import numpy as np


def _material(vs, density, poisson_ratio=0.3, omega=None,
              rayleigh_alpha=0.0, rayleigh_beta=0.0):
    """由波速、密度和泊松比计算实数或瑞利阻尼复材料参数。"""
    vs = float(vs)
    rho = float(density)
    nu = float(poisson_ratio)
    if vs <= 0.0 or rho <= 0.0 or not (-1.0 < nu < 0.5):
        raise ValueError('材料参数非法: Vs=%r rho=%r nu=%r' % (vs, rho, nu))
    mu0 = rho * vs ** 2
    lam0 = 2.0 * mu0 * nu / (1.0 - 2.0 * nu)
    cp0 = math.sqrt((lam0 + 2.0 * mu0) / rho)
    alpha = float(rayleigh_alpha or 0.0)
    beta = float(rayleigh_beta or 0.0)
    if omega is not None and (alpha != 0.0 or beta != 0.0):
        rho_used = rho * (1.0 - 1j * alpha / float(omega))
        stiffness_factor = 1.0 + 1j * float(omega) * beta
        mu = mu0 * stiffness_factor
        lam = lam0 * stiffness_factor
        vs_used = np.sqrt(mu / rho_used)
        cp_used = np.sqrt((lam + 2.0 * mu) / rho_used)
    else:
        rho_used, mu, lam = rho, mu0, lam0
        vs_used, cp_used = vs, cp0
    return {
        'vs': vs_used, 'vs0': vs, 'rho': rho_used, 'rho0': rho,
        'nu': nu, 'mu': mu, 'lam': lam, 'cp': cp_used, 'cp0': cp0,
        'rayleigh_alpha': alpha, 'rayleigh_beta': beta,
    }


def _validate_layers(layers_top_down, halfspace, omega=None):
    """校验层状介质输入并转换为统一材料字典。"""
    hs = _material(
        halfspace['vs'], halfspace['rho'], halfspace.get('nu', 0.3), omega=omega,
        rayleigh_alpha=halfspace.get('rayleigh_alpha', 0.0),
        rayleigh_beta=halfspace.get('rayleigh_beta', 0.0),
    )
    layers = []
    for idx, layer in enumerate(layers_top_down or []):
        thickness = float(layer['thickness'])
        if thickness <= 0.0:
            raise ValueError('有限层%d厚度必须>0' % idx)
        mat = _material(
            layer['vs'], layer['rho'], layer.get('nu', 0.3), omega=omega,
            rayleigh_alpha=layer.get('rayleigh_alpha', 0.0),
            rayleigh_beta=layer.get('rayleigh_beta', 0.0),
        )
        mat['thickness'] = thickness
        layers.append(mat)
    return layers, hs


def _vertical_slowness(speed, p):
    """返回实数时正向、复数时满足衰减方向的垂向慢度。"""
    value = complex(1.0 / speed ** 2 - p ** 2)
    q = np.sqrt(value + 0j)
    if q.imag > 0.0 or (abs(q.imag) <= 1.0e-15 and q.real < 0.0):
        q = -q
    return q


def _state_matrix(mat, p):
    """组装一层的 P/SV 四波状态矩阵。"""
    cp = mat['cp']
    cs = mat['vs']
    lam = mat['lam']
    mu = mat['mu']
    qp = _vertical_slowness(cp, p)
    qs = _vertical_slowness(cs, p)

    def column(kind, q):
        if kind == 'P':
            ux, uy = p, q
            sxy = 2.0 * mu * p * q
            syy = lam / cp ** 2 + 2.0 * mu * q ** 2
        else:
            ux, uy = -q, p
            sxy = mu * (p ** 2 - q ** 2)
            syy = 2.0 * mu * p * q
        return np.array([ux, uy, sxy, syy], dtype=complex)

    return np.column_stack((column('P', qp), column('S', qs),
                            column('P', -qp), column('S', -qs)))


def _layer_transfer(mat, thickness, p, omega):
    """返回该有限层从底部状态到顶部状态的传播矩阵。"""
    matrix = _state_matrix(mat, p)
    qp = _vertical_slowness(mat['cp'], p)
    qs = _vertical_slowness(mat['vs'], p)
    phase = np.diag([
        np.exp(-1j * omega * qp * thickness),
        np.exp(-1j * omega * qs * thickness),
        np.exp(1j * omega * qp * thickness),
        np.exp(1j * omega * qs * thickness),
    ])
    return np.dot(np.dot(matrix, phase), np.linalg.inv(matrix))


def surface_response(freq_hz, layers_top_down, halfspace, incident_angle_deg=0.0):
    """计算单位上行 SV 入射下的自由表面复位移响应。

    参数
    ----
    freq_hz : float
        正频率 Hz。
    layers_top_down : list[dict]
        从上到下有限层，每项含 ``vs/rho/thickness``，可选 ``nu``。
    halfspace : dict
        基岩半空间，含 ``vs/rho``，可选 ``nu``。
    incident_angle_deg : float
        半空间内 SV 入射角，角度以竖直方向为零。

    返回
    ----
    dict
        ``ux/uy`` 为自由表面复位移，``traction_residual`` 为归一化牵引残差，
        ``incident_p`` 为水平慢度，``reflected_p/reflected_sv`` 为底部反射振幅。
    """
    freq = float(freq_hz)
    angle = float(incident_angle_deg)
    if freq <= 0.0 or not (-89.9 < angle < 89.9):
        raise ValueError('频率或入射角非法: f=%r angle=%r' % (freq_hz, incident_angle_deg))
    omega = 2.0 * math.pi * freq
    layers, hs = _validate_layers(layers_top_down, halfspace, omega=omega)
    alpha = math.radians(angle)
    p = math.sin(alpha) / hs['vs0']

    propagation = np.eye(4, dtype=complex)
    for layer in reversed(layers):  # 从半空间界面向上依次穿过底层到顶层
        propagation = np.dot(_layer_transfer(layer, layer['thickness'], p, omega), propagation)

    half_matrix = _state_matrix(hs, p)
    incident = np.dot(propagation, half_matrix[:, 1])  # 单位上行 SV 入射
    reflected = np.dot(propagation, half_matrix[:, 2:4])  # 下行 P/SV 反射基底
    traction_rows = (2, 3)
    reflected_coeff = np.linalg.solve(reflected[np.ix_(traction_rows, (0, 1))],
                                      -incident[list(traction_rows)])
    amplitudes = np.array([0.0 + 0j, 1.0 + 0j,
                           reflected_coeff[0], reflected_coeff[1]], dtype=complex)
    surface = np.dot(propagation, np.dot(half_matrix, amplitudes))
    traction_scale = max(1.0, float(np.max(np.abs(surface[:2]))))
    traction_residual = float(np.max(np.abs(surface[2:])) / traction_scale)
    return {
        'ux': complex(surface[0]),
        'uy': complex(surface[1]),
        'traction_residual': traction_residual,
        'incident_p': float(p),
        'reflected_p': complex(reflected_coeff[0]),
        'reflected_sv': complex(reflected_coeff[1]),
    }


def transfer_function(freqs_hz, layers_top_down, halfspace, incident_angle_deg=0.0):
    """返回自由表面相对单位入射 SV 位移的复传递函数。"""
    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.ndim != 1 or np.any(freqs <= 0.0):
        raise ValueError('freqs_hz 必须是一维正频率数组')
    ux = np.empty(freqs.shape, dtype=complex)
    uy = np.empty(freqs.shape, dtype=complex)
    residual = np.empty(freqs.shape, dtype=float)
    for idx, freq in enumerate(freqs):
        result = surface_response(freq, layers_top_down, halfspace, incident_angle_deg)
        ux[idx] = result['ux']
        uy[idx] = result['uy']
        residual[idx] = result['traction_residual']
    return {'ux': ux, 'uy': uy, 'traction_residual': residual}


def homogeneous_halfspace_transfer(freq_hz, depth, halfspace, incident_angle_deg=0.0):
    """返回均质半空间深度点相对入射水平位移的 P-SV 复传递函数。

    `depth=0` 位于自由表面；正值向下。该函数只用于 V2 的均质半空间端到端对比，
    不调用 Abaqus 或生产自由场内核。
    """
    freq = float(freq_hz)
    depth = float(depth)
    angle = float(incident_angle_deg)
    if freq <= 0.0:
        raise ValueError('频率非法: f=%r' % freq_hz)
    omega = 2.0 * math.pi * freq
    mat = _material(
        halfspace['vs'], halfspace['rho'], halfspace.get('nu', 0.3), omega=omega,
        rayleigh_alpha=halfspace.get('rayleigh_alpha', 0.0),
        rayleigh_beta=halfspace.get('rayleigh_beta', 0.0),
    )
    p = math.sin(math.radians(angle)) / mat['vs0']
    matrix = _state_matrix(mat, p)
    incident = matrix[:, 1]
    reflected = matrix[:, 2:4]
    coeff = np.linalg.solve(reflected[np.ix_((2, 3), (0, 1))], -incident[[2, 3]])
    amplitudes = np.array([0.0 + 0j, 1.0 + 0j, coeff[0], coeff[1]], dtype=complex)
    qp = _vertical_slowness(mat['cp'], p)
    qs = _vertical_slowness(mat['vs'], p)
    phases = np.array([np.exp(1j * omega * qp * depth), np.exp(1j * omega * qs * depth),
                       np.exp(-1j * omega * qp * depth), np.exp(-1j * omega * qs * depth)], dtype=complex)
    field = np.dot(matrix, amplitudes * phases)
    denom = -1.0 / mat['vs0']  # 以入射 SV 粒子位移幅值归一化，而非仅取水平分量
    return {'ux': complex(field[0] / denom), 'uy': complex(field[1] / denom),
            'traction_residual': float(np.max(np.abs(field[2:])) / max(1.0, float(np.max(np.abs(field[:2])))))}


def horizontal_phase_factor(freq_hz, x, halfspace, incident_angle_deg=0.0):
    """返回平面斜入射波在水平坐标 x 处的传播相位因子。

    参考解采用 ``exp(iωt)`` 约定，水平相位写为
    ``exp(-iω p x)``，其中 ``p=sin(theta)/Vs``。x=0 为模型左端
    的预注册相位原点，不根据有限元结果反向调整。
    """
    freq = float(freq_hz)
    x = float(x)
    vs = float(halfspace['vs'])
    if freq < 0.0 or vs <= 0.0:
        raise ValueError('频率或 Vs 非法: f=%r Vs=%r' % (freq_hz, vs))
    p = math.sin(math.radians(float(incident_angle_deg))) / vs
    return complex(np.exp(-1j * 2.0 * math.pi * freq * p * x))


def main():
    """命令行最小示例。"""
    halfspace = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3}
    layers = [{'vs': 400.0, 'rho': 2000.0, 'nu': 0.3, 'thickness': 40.0}]
    result = surface_response(4.0, layers, halfspace, incident_angle_deg=15.0)
    print('独立 P-SV 参考解: |ux|=%.6e |uy|=%.6e traction_residual=%.3e' %
          (abs(result['ux']), abs(result['uy']), result['traction_residual']))


if __name__ == '__main__':
    main()
