# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""向量化自由场引擎与现有逐频率实现的数值等价性测试（不依赖 Abaqus）。"""

import os, sys, math, importlib, time  # 导入标准库
import numpy as np  # 导入数值库

# 屏蔽 Abaqus 依赖，强制走纯数值分支
for _m in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):
    sys.modules[_m] = None
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'Modeling', 'Multi')  # test/Multi → 仓库根 → Modeling/Multi
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
eng = importlib.import_module('VAB_oblique_TAF_double_v7')


def transfer_vec(y_target, x_target, mat_bedrock, mat_overlying,
                 h_bedrock, h_overlying, y_bottom, p, freq_arr):
    """向量化计算单节点的频域传递函数数组（与逐频率传播矩阵等价）。

    返回 (T_ux, T_uy, T_dotux, T_dotuy, T_sxx, T_syy, T_sxy)，
    使得 ux_freq = T_ux * vel_freq 等（DC 分量置零）。
    """
    omega = 2.0 * math.pi * freq_arr  # 角频率数组
    cs1 = mat_bedrock['cs']; cp1 = mat_bedrock['cp']
    cs2 = mat_overlying['cs']; cp2 = mat_overlying['cp']
    GG1 = mat_bedrock['GG']; lam1 = mat_bedrock['lam']
    GG2 = mat_overlying['GG']; lam2 = mat_overlying['lam']
    y_intf = y_bottom + h_bedrock
    y_top = y_intf + h_overlying

    def _qval(c):
        val = (1.0 / c) ** 2 - p ** 2
        if val >= 0:
            return complex(math.sqrt(val), 0)
        return complex(0, math.sqrt(-val))

    qs1 = _qval(cs1); qp1 = _qval(cp1); qs2 = _qval(cs2); qp2 = _qval(cp2)

    def coeff(q_sv, q_p, GG, lam, cs, cp, direction, wave_type):
        sgn = 1.0 if direction == 'up' else -1.0
        if wave_type == 'SV':
            q = q_sv
            ux_d = q / (1.0 / cs)
            uy_d = -sgn * p / (1.0 / cs)
        else:
            q = q_p
            ux_d = p / (1.0 / cp)
            uy_d = sgn * q / (1.0 / cp)
        sig_xx = -(lam * (p * ux_d + sgn * q * uy_d) + 2.0 * GG * p * ux_d)
        sig_yy = -(lam * (p * ux_d + sgn * q * uy_d) + 2.0 * GG * sgn * q * uy_d)
        tau_xy = -GG * (sgn * q * ux_d + p * uy_d)
        return ux_d, uy_d, sig_xx, sig_yy, tau_xy

    # 频率无关系数（一次算好）
    ux_sv1u, uy_sv1u, sxx_sv1u, syy_sv1u, sxy_sv1u = coeff(qs1, qp1, GG1, lam1, cs1, cp1, 'up', 'SV')
    ux_sv1d, uy_sv1d, sxx_sv1d, syy_sv1d, sxy_sv1d = coeff(qs1, qp1, GG1, lam1, cs1, cp1, 'down', 'SV')
    ux_p1d, uy_p1d, sxx_p1d, syy_p1d, sxy_p1d = coeff(qs1, qp1, GG1, lam1, cs1, cp1, 'down', 'P')
    ux_sv2u, uy_sv2u, sxx_sv2u, syy_sv2u, sxy_sv2u = coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'up', 'SV')
    ux_p2u, uy_p2u, sxx_p2u, syy_p2u, sxy_p2u = coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'up', 'P')
    ux_sv2d, uy_sv2d, sxx_sv2d, syy_sv2d, sxy_sv2d = coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'down', 'SV')
    ux_p2d, uy_p2d, sxx_p2d, syy_p2d, sxy_p2d = coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'down', 'P')

    # 频率相关相位（向量）
    phi_s2_up = np.exp(1j * omega * qs2 * h_overlying)
    phi_p2_up = np.exp(1j * omega * qp2 * h_overlying)
    phi_s2_dn = np.conj(phi_s2_up)
    phi_p2_dn = np.conj(phi_p2_up)

    # 自由面 2x2：解析求逆（向量）
    A00 = syy_sv2d * phi_s2_dn; A01 = syy_p2d * phi_p2_dn
    A10 = sxy_sv2d * phi_s2_dn; A11 = sxy_p2d * phi_p2_dn
    det = A00 * A11 - A01 * A10
    det = np.where(np.abs(det) < 1e-30, 1e-30, det)  # 防退化
    iA00 = A11 / det; iA01 = -A01 / det
    iA10 = -A10 / det; iA11 = A00 / det
    Bsv0 = -syy_sv2u * phi_s2_up; Bsv1 = -sxy_sv2u * phi_s2_up
    Bp0 = -syy_p2u * phi_p2_up; Bp1 = -sxy_p2u * phi_p2_up
    Fsv0 = iA00 * Bsv0 + iA01 * Bsv1
    Fsv1 = iA10 * Bsv0 + iA11 * Bsv1
    Fp0 = iA00 * Bp0 + iA01 * Bp1
    Fp1 = iA10 * Bp0 + iA11 * Bp1

    # 覆盖层界面叠加（向量）
    cov_ux_sv = ux_sv2u + Fsv0 * ux_sv2d + Fsv1 * ux_p2d
    cov_uy_sv = uy_sv2u + Fsv0 * uy_sv2d + Fsv1 * uy_p2d
    cov_syy_sv = syy_sv2u + Fsv0 * syy_sv2d + Fsv1 * syy_p2d
    cov_sxy_sv = sxy_sv2u + Fsv0 * sxy_sv2d + Fsv1 * sxy_p2d
    cov_ux_p = ux_p2u + Fp0 * ux_sv2d + Fp1 * ux_p2d
    cov_uy_p = uy_p2u + Fp0 * uy_sv2d + Fp1 * uy_p2d
    cov_syy_p = syy_p2u + Fp0 * syy_sv2d + Fp1 * syy_p2d
    cov_sxy_p = sxy_p2u + Fp0 * sxy_sv2d + Fp1 * sxy_p2d

    Nk = omega.shape[0]
    A = np.zeros((Nk, 4, 4), dtype=complex)
    A[:, 0, 0] = ux_sv1d; A[:, 1, 0] = uy_sv1d; A[:, 2, 0] = sxy_sv1d; A[:, 3, 0] = syy_sv1d
    A[:, 0, 1] = ux_p1d; A[:, 1, 1] = uy_p1d; A[:, 2, 1] = sxy_p1d; A[:, 3, 1] = syy_p1d
    A[:, 0, 2] = -cov_ux_sv; A[:, 1, 2] = -cov_uy_sv; A[:, 2, 2] = -cov_sxy_sv; A[:, 3, 2] = -cov_syy_sv
    A[:, 0, 3] = -cov_ux_p; A[:, 1, 3] = -cov_uy_p; A[:, 2, 3] = -cov_sxy_p; A[:, 3, 3] = -cov_syy_p
    B = np.empty((Nk, 4, 1), dtype=complex)
    B[:, 0, 0] = -ux_sv1u; B[:, 1, 0] = -uy_sv1u; B[:, 2, 0] = -sxy_sv1u; B[:, 3, 0] = -syy_sv1u

    X = np.linalg.solve(A, B)[:, :, 0]  # 批量求解 4x4
    Rss = X[:, 0]; Rsp = X[:, 1]; Tss = X[:, 2]; Tsp = X[:, 3]
    a_sv2 = Fsv0 * Tss + Fp0 * Tsp
    a_p2 = Fsv1 * Tss + Fp1 * Tsp

    if y_target <= y_intf + 1e-4:  # 基岩层分支
        dy = y_target - y_bottom
        phi_inc = np.exp(1j * omega * qs1 * dy)
        phi_ref = np.exp(1j * omega * qs1 * (2.0 * h_bedrock - dy))
        phi_pref = np.exp(1j * omega * qp1 * (2.0 * h_bedrock - dy))
        ux = ux_sv1u * phi_inc + Rss * ux_sv1d * phi_ref + Rsp * ux_p1d * phi_pref
        uy = uy_sv1u * phi_inc + Rss * uy_sv1d * phi_ref + Rsp * uy_p1d * phi_pref
        sxx = sxx_sv1u * phi_inc + Rss * sxx_sv1d * phi_ref + Rsp * sxx_p1d * phi_pref
        syy = syy_sv1u * phi_inc + Rss * syy_sv1d * phi_ref + Rsp * syy_p1d * phi_pref
        sxy = sxy_sv1u * phi_inc + Rss * sxy_sv1d * phi_ref + Rsp * sxy_p1d * phi_pref
    else:  # 覆盖层分支
        dyi = y_target - y_intf
        p_su = np.exp(1j * omega * qs2 * dyi); p_sd = np.exp(-1j * omega * qs2 * dyi)
        p_pu = np.exp(1j * omega * qp2 * dyi); p_pd = np.exp(-1j * omega * qp2 * dyi)
        ux = Tss * ux_sv2u * p_su + a_sv2 * ux_sv2d * p_sd + Tsp * ux_p2u * p_pu + a_p2 * ux_p2d * p_pd
        uy = Tss * uy_sv2u * p_su + a_sv2 * uy_sv2d * p_sd + Tsp * uy_p2u * p_pu + a_p2 * uy_p2d * p_pd
        sxx = Tss * sxx_sv2u * p_su + a_sv2 * sxx_sv2d * p_sd + Tsp * sxx_p2u * p_pu + a_p2 * sxx_p2d * p_pd
        syy = Tss * syy_sv2u * p_su + a_sv2 * syy_sv2d * p_sd + Tsp * syy_p2u * p_pu + a_p2 * syy_p2d * p_pd
        sxy = Tss * sxy_sv2u * p_su + a_sv2 * sxy_sv2d * p_sd + Tsp * sxy_p2u * p_pu + a_p2 * sxy_p2d * p_pd

    phase_x = np.exp(1j * omega * p * x_target)
    inv_iw = np.zeros_like(omega, dtype=complex)
    nz = np.abs(omega) >= 1e-20
    inv_iw[nz] = 1.0 / (1j * omega[nz])
    # 传递函数（DC 自动为 0：inv_iw[0]=0 使位移为 0，应力乘 phase_x 后单独清零 DC）
    T_ux = ux * phase_x * inv_iw
    T_uy = uy * phase_x * inv_iw
    T_dotux = 1j * omega * T_ux
    T_dotuy = 1j * omega * T_uy
    T_sxx = sxx * phase_x
    T_syy = syy * phase_x
    T_sxy = sxy * phase_x
    # DC 分量与 omega≈0 处清零，匹配逐频率实现的 continue 行为
    dc = ~nz
    for arr in (T_ux, T_uy, T_dotux, T_dotuy, T_sxx, T_syy, T_sxy):
        arr[dc] = 0.0
    return T_ux, T_uy, T_dotux, T_dotuy, T_sxx, T_syy, T_sxy


def compute_vec(y_target, x_target, mat_bedrock, mat_overlying,
                h_bedrock, h_overlying, y_bottom, p, vel_freq, freq_arr, N_fft):
    T = transfer_vec(y_target, x_target, mat_bedrock, mat_overlying,
                     h_bedrock, h_overlying, y_bottom, p, freq_arr)
    T_ux, T_uy, T_dotux, T_dotuy, T_sxx, T_syy, T_sxy = T
    return {
        'ux': np.fft.irfft(T_ux * vel_freq, n=N_fft),
        'uy': np.fft.irfft(T_uy * vel_freq, n=N_fft),
        'dotux': np.fft.irfft(T_dotux * vel_freq, n=N_fft),
        'dotuy': np.fft.irfft(T_dotuy * vel_freq, n=N_fft),
        'sxx': np.fft.irfft(T_sxx * vel_freq, n=N_fft),
        'syy': np.fft.irfft(T_syy * vel_freq, n=N_fft),
        'sxy': np.fft.irfft(T_sxy * vel_freq, n=N_fft),
    }


def main():
    mat = eng._compute_material_params(2000.0, 0.3, 2500.0)
    mat_ov = eng._compute_material_params(1600.0, 0.3, 2500.0)
    dt = 0.001; n = 2000
    t = np.arange(n) * dt
    v = (1.0 - 2.0 * (math.pi * 2.0 * (t - 1.0)) ** 2) * np.exp(-(math.pi * 2.0 * (t - 1.0)) ** 2)
    N_fft = 1
    while N_fft < n:
        N_fft *= 2
    N_fft *= 2
    vp = np.zeros(N_fft); vp[:n] = v
    freq = np.fft.rfftfreq(N_fft, d=dt)
    vf = np.fft.rfft(vp)

    cases = [
        ('垂直-覆盖层', 250.0, 900.0, 1e-10, 200.0),
        ('垂直-基岩',   100.0, 0.0,   1e-10, 200.0),
        ('斜入射-覆盖', 350.0, 1800.0, 15.0, 100.0),
        ('斜入射-基岩', 50.0,  1800.0, 15.0, 150.0),
        ('界面附近',    200.0, 500.0, 20.0, 200.0),
    ]
    worst = 0.0
    for name, y, x, ang, hov in cases:
        p = math.sin(math.radians(ang)) / 2000.0
        ref = eng._compute_freefield_at_node(y_target=y, x_target=x, mat_bedrock=mat,
            mat_overlying=mat_ov, h_bedrock=200.0, h_overlying=hov, y_bottom=0.0,
            p_horiz=p, vel_freq=vf, freq_arr=freq, dt=dt, N_fft=N_fft)
        new = compute_vec(y, x, mat, mat_ov, 200.0, hov, 0.0, p, vf, freq, N_fft)
        cmax = 0.0
        for k in ('ux', 'uy', 'dotux', 'dotuy', 'sxx', 'syy', 'sxy'):
            sc = np.max(np.abs(ref[k])) + 1e-30
            rel = np.max(np.abs(ref[k] - new[k])) / sc
            cmax = max(cmax, rel)
        worst = max(worst, cmax)
        print('%-12s 最大相对偏差 = %.3e' % (name, cmax))
    print('\n所有算例最大相对偏差 = %.3e  (应 < 1e-9)' % worst)

    # 计时对比
    N = 20
    p = math.sin(math.radians(15.0)) / 2000.0
    t0 = time.time()
    for _ in range(N):
        eng._compute_freefield_at_node(y_target=250.0, x_target=900.0, mat_bedrock=mat,
            mat_overlying=mat_ov, h_bedrock=200.0, h_overlying=200.0, y_bottom=0.0,
            p_horiz=p, vel_freq=vf, freq_arr=freq, dt=dt, N_fft=N_fft)
    t_old = (time.time() - t0) / N
    t0 = time.time()
    for _ in range(N):
        compute_vec(250.0, 900.0, mat, mat_ov, 200.0, 200.0, 0.0, p, vf, freq, N_fft)
    t_new = (time.time() - t0) / N
    print('\n逐频率: %.1f ms/节点   向量化: %.1f ms/节点   加速 %.1fx'
          % (t_old * 1000, t_new * 1000, t_old / t_new))


if __name__ == '__main__':
    main()
