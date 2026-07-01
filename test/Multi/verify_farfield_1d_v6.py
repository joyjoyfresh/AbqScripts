# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""远场一维台阶值定量验证脚本（针对 v6 三层图15 工况的偏差诊断）。

目的：用纯 numpy 复现 v6 的频域分层自由场解（_fd_solve_column 同一套方程），
计算远场（上/下平台）一维柱在 4Hz Ricker 入射下的地表 PGA 放大，
并按 Compute_TAF_v2 的口径归一化（TAF = PGA / (factor_h × PGA_in)），
从而判断：FE 远场台阶 ≈ 一维理论值（FE 忠实）还是明显偏低（FE 失真）。

同时做敏感性分析：阻尼开/关、Q 取法、Ricker 主频 fc、时窗截断。
不依赖 Abaqus，可直接 python 运行。
"""

import math  # 导入数学模块
import numpy as np  # 导入数值计算库

# ==========================================================
#  基本材料与几何参数（与 Autorun_TAF_multilayer_v2-testv6-2 三层工况一致）
# ==========================================================
RHO = 2500.0  # 各层密度 (kg/m^3)
NU = 0.3  # 各层泊松比
CS_BED = 2000.0  # 基岩剪切波速 (m/s)
CS_OVER = 800.0  # 覆盖层剪切波速 (m/s)，Vr/Vs2=2.5
CS_SURF_SOFT = 400.0  # 软表层剪切波速 (m/s)，Vr/Vs1=5.0 → Vs1/Vs2=0.5
CS_SURF_STIFF = 1600.0  # 硬表层剪切波速 (m/s)，Vr/Vs1=1.25 → Vs1/Vs2=2.0
BEDROCK_TOP = 200.0  # 基岩界面高程 y (m)
H_UPPER = 600.0  # 上平台地表高程 (m)：基岩200 + 覆盖层 H=400
H_LOWER = 400.0  # 下平台地表高程 (m)：基岩200 + 覆盖层 h=200
H1 = 50.0  # 表层固定厚度 (m)：h1/(H-h)=0.25 → 50

RICKER_PATH = '/sessions/admiring-pensive-carson/mnt/AbqScripts/Wave/Impulse/Acceleration/ricker_wavelet_4Hz.txt'  # 输入波路径


def cp_of(cs):  # 由剪切波速与泊松比计算纵波波速
    """cp = cs·sqrt(2(1-ν)/(1-2ν))，ν=0.3 时约 1.8708·cs。"""
    return cs * math.sqrt(2.0 * (1.0 - NU) / (1.0 - 2.0 * NU))  # 返回纵波波速


def make_column(segs):  # 构造柱层段列表（从下到上）
    """segs: [(name, cs, y0, y1), ...]，返回层段 dict 列表。"""
    return [{'name': n, 'cs': c, 'cp': cp_of(c), 'rho': RHO, 'y0': a, 'y1': b}  # 打包层段
            for (n, c, a, b) in segs]  # 遍历输入


# ==========================================================
#  阻尼模型（与 v6 _damping_ratio_from_q / _rayleigh_coeffs 同公式）
# ==========================================================
def rayleigh_for(cs, is_bedrock, fc, qs_factor=0.05, q_bedrock=999.0, f1f=0.5, f2f=2.5):  # 计算该层瑞利系数
    """Q = qs_factor·cs（基岩取 q_bedrock），ξ=1/(2Q)，双频(0.5fc, 2.5fc)拟合 α/β。"""
    Q = q_bedrock if is_bedrock else qs_factor * cs  # 该层品质因子
    xi = 1.0 / (2.0 * Q)  # 阻尼比
    w1 = 2.0 * math.pi * f1f * fc  # 拟合下限圆频率
    w2 = 2.0 * math.pi * f2f * fc  # 拟合上限圆频率
    alpha = 2.0 * xi * w1 * w2 / (w1 + w2)  # 瑞利 α
    beta = 2.0 * xi / (w1 + w2)  # 瑞利 β
    return alpha, beta  # 返回系数


# ==========================================================
#  频域分层自由场求解（复现 v6 _fd_* 系列，方程一致）
# ==========================================================
def fd_layer_params(seg, omega, p, ab):  # 单层段逐频复参数
    """ab=(alpha,beta) 瑞利系数；返回复慢度/复模量 dict（与 v6 _fd_layer_params 一致）。"""
    rho = seg['rho']  # 密度
    mu0 = rho * seg['cs'] ** 2  # 实剪切模量
    lam0 = rho * (seg['cp'] ** 2 - 2.0 * seg['cs'] ** 2)  # 实拉梅常数
    a_ray, b_ray = ab  # 瑞利系数
    rhoC = rho * (1.0 - 1j * a_ray / omega)  # 复密度
    sfac = 1.0 + 1j * omega * b_ray  # 刚度比例因子
    muC = mu0 * sfac  # 复剪切模量
    lamC = lam0 * sfac  # 复拉梅常数
    cs2 = muC / rhoC  # 复剪切波速平方
    cp2 = (lamC + 2.0 * muC) / rhoC  # 复纵波波速平方
    qs = np.sqrt(1.0 / cs2 - p * p)  # SV 垂直慢度
    qp = np.sqrt(1.0 / cp2 - p * p)  # P 垂直慢度
    qs = np.where(qs.imag > 0.0, -qs, qs)  # 强制衰减分支
    qp = np.where(qp.imag > 0.0, -qp, qp)  # 强制衰减分支
    return {'qs': qs, 'qp': qp, 'mu': muC, 'lam': lamC,  # 打包复参数
            'csC': np.sqrt(cs2), 'cpC': np.sqrt(cp2), 'p': p}  # 复波速与水平慢度


def fd_wave_params(seg, la, kind):  # 某层某类波的极化与相位参数（与 v6 一致）
    """kind ∈ {Pu,Pd,Su,Sd}；返回 {dx,dy,ky,yref}。"""
    qs, qp, csC, cpC, p = la['qs'], la['qp'], la['csC'], la['cpC'], la['p']  # 取参数
    if kind == 'Pu':  # 上行 P
        return {'dx': cpC * p, 'dy': cpC * qp, 'ky': qp, 'yref': seg['y0']}  # 极化/相位
    if kind == 'Pd':  # 下行 P
        return {'dx': cpC * p, 'dy': -cpC * qp, 'ky': -qp, 'yref': seg['y1']}  # 极化/相位
    if kind == 'Su':  # 上行 SV
        return {'dx': csC * qs, 'dy': -csC * p, 'ky': qs, 'yref': seg['y0']}  # 极化/相位
    return {'dx': -csC * qs, 'dy': -csC * p, 'ky': -qs, 'yref': seg['y1']}  # 下行 SV


def fd_field_coeffs(wave, la, omega, p, y):  # 某波在高程 y 的 5 个场量系数（与 v6 一致）
    """返回 (ux, uy, σyy, σxy, σxx) 系数。"""
    ph = np.exp(-1j * omega * wave['ky'] * (y - wave['yref']))  # 垂直相位因子
    dx = wave['dx'] * ph  # x 向位移系数
    dy = wave['dy'] * ph  # y 向位移系数
    lam, mu = la['lam'], la['mu']  # 复模量
    miw = -1j * omega  # 公共因子
    syy = miw * (lam * p * dx + (lam + 2.0 * mu) * wave['ky'] * dy)  # σyy
    sxy = miw * mu * (wave['ky'] * dx + p * dy)  # σxy
    sxx = miw * ((lam + 2.0 * mu) * p * dx + lam * wave['ky'] * dy)  # σxx
    return dx, dy, syy, sxy, sxx  # 返回系数


def fd_solve_column(column, p, omega, ab_map):  # 柱频域全局矩阵求解（与 v6 一致）
    """ab_map: {层名:(α,β)}；返回 {'amps','las','waves','inc','column'}。"""
    nseg = len(column)  # 层段数
    M = nseg - 1  # 有限层数
    las = [fd_layer_params(seg, omega, p, ab_map[seg['name']]) for seg in column]  # 各层复参数
    waves = [[(0, fd_wave_params(column[0], las[0], 'Pd')),  # 基岩反射下行 P
              (1, fd_wave_params(column[0], las[0], 'Sd'))]]  # 基岩反射下行 SV
    col = 2  # 未知量列号
    for m in range(1, nseg):  # 各有限层 4 波
        wm = []  # 该层波表
        for kind in ('Pu', 'Pd', 'Su', 'Sd'):  # 四类波
            wm.append((col, fd_wave_params(column[m], las[m], kind)))  # 记录
            col += 1  # 列号递增
        waves.append(wm)  # 追加
    nunk = col  # 未知量总数
    inc = fd_wave_params(column[0], las[0], 'Su')  # 入射上行 SV（单位幅值）
    nb = omega.shape[0]  # 频点数
    A = np.zeros((nb, nunk, nunk), dtype=complex)  # 系数矩阵
    b = np.zeros((nb, nunk), dtype=complex)  # 右端项
    row = 0  # 行号
    for j in range(M):  # 各界面连续条件
        Y = column[j]['y1']  # 界面高程
        for sgn, m in ((1.0, j), (-1.0, j + 1)):  # 界面下方(+)与上方(−)
            la = las[m]  # 该层参数
            for cidx, w in waves[m]:  # 该层各未知波
                ux, uy, syy, sxy, _ = fd_field_coeffs(w, la, omega, p, Y)  # 界面场量
                A[:, row + 0, cidx] += sgn * ux  # ux 连续
                A[:, row + 1, cidx] += sgn * uy  # uy 连续
                A[:, row + 2, cidx] += sgn * syy  # σyy 连续
                A[:, row + 3, cidx] += sgn * sxy  # σxy 连续
            if m == 0:  # 入射波移项
                ux, uy, syy, sxy, _ = fd_field_coeffs(inc, la, omega, p, Y)  # 入射场量
                b[:, row + 0] -= sgn * ux  # 移项
                b[:, row + 1] -= sgn * uy  # 移项
                b[:, row + 2] -= sgn * syy  # 移项
                b[:, row + 3] -= sgn * sxy  # 移项
        row += 4  # 行号推进
    Ys = column[-1]['y1']  # 地表高程
    laT = las[-1]  # 顶层参数
    for cidx, w in waves[-1]:  # 顶层各波参与自由面条件
        _, _, syy, sxy, _ = fd_field_coeffs(w, laT, omega, p, Ys)  # 地表应力
        A[:, row + 0, cidx] += syy  # σyy=0
        A[:, row + 1, cidx] += sxy  # σxy=0
    if M == 0:  # 半空间退化
        _, _, syy, sxy, _ = fd_field_coeffs(inc, laT, omega, p, Ys)  # 入射地表应力
        b[:, row + 0] -= syy  # 移项
        b[:, row + 1] -= sxy  # 移项
    amps = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # 批量直解
    return {'amps': amps, 'las': las, 'waves': waves, 'inc': inc, 'column': column}  # 返回柱解


def fd_surface_disp(sol, omega, p):  # 地表位移谱（单位入射）
    """返回 (ux, uy) 逐频复数组（在最顶层上界 y 处评估）。"""
    column = sol['column']  # 柱层段
    y = column[-1]['y1']  # 地表高程
    seg_idx = len(column) - 1  # 最顶层索引
    la = sol['las'][seg_idx]  # 顶层参数
    ux = np.zeros_like(omega, dtype=complex)  # x 位移谱
    uy = np.zeros_like(omega, dtype=complex)  # y 位移谱
    for cidx, w in sol['waves'][seg_idx]:  # 顶层各波
        cux, cuy, _, _, _ = fd_field_coeffs(w, la, omega, p, y)  # 场量系数
        ux += sol['amps'][:, cidx] * cux  # 叠加 x
        uy += sol['amps'][:, cidx] * cuy  # 叠加 y
    if seg_idx == 0:  # 半空间退化时叠加入射波
        cux, cuy, _, _, _ = fd_field_coeffs(sol['inc'], la, omega, p, y)  # 入射场量
        ux += cux  # 叠加
        uy += cuy  # 叠加
    return ux, uy  # 返回地表位移谱


# ==========================================================
#  解析分母 factor_h（与 Compute_TAF_v2 同公式）
# ==========================================================
def factor_h_of(angle_deg, cs, cp):  # 基岩半空间自由地表水平放大系数
    """垂直入射=2.0，15°≈1.92。"""
    a = math.radians(angle_deg if abs(angle_deg) > 1e-12 else 1e-10)  # 入射角弧度
    beta = math.asin(max(-1.0, min(1.0, cp * math.sin(a) / cs)))  # P 波角
    num = cs ** 2 * math.sin(2 * a) * math.sin(2 * beta) - cp ** 2 * math.cos(2 * a) ** 2  # A1 分子
    den = cs ** 2 * math.sin(2 * a) * math.sin(2 * beta) + cp ** 2 * math.cos(2 * a) ** 2  # 公共分母
    a1 = num / den  # SV→SV 反射系数
    a2 = (2 * cp * cs * math.sin(2 * a) * math.cos(2 * a)) / den  # SV→P 转换系数
    return (1.0 - a1) * math.cos(a) + a2 * math.sin(beta)  # 水平合成系数


# ==========================================================
#  时域 PGA 评估
# ==========================================================
def taf_plateau(column, angle_deg, acc, dt, damping=True, fc=None, qs_factor=0.05, window=None):  # 计算远场台阶 TAF
    """对给定柱与入射角，返回 (TAF_h, TAF_v)：地表 PGA / (factor_h × PGA_in)。

    damping=False 时全弹性；fc：瑞利拟合主频；window：PGA 评估时窗（秒，None=全程）。
    """
    N = len(acc)  # 记录长度
    Nfft = 1  # FFT 长度初值
    while Nfft < N * 4:  # 补零 4 倍并取 2 的幂
        Nfft *= 2  # 翻倍
    A = np.fft.rfft(acc, n=Nfft)  # 加速度谱
    freqs = np.fft.rfftfreq(Nfft, dt)  # 频率轴
    mask = np.abs(A) > 1e-7 * np.max(np.abs(A))  # 谱幅值掩码（同 v6 tol）
    mask[0] = False  # 去 DC
    idx = np.nonzero(mask)[0]  # 求解频点
    omega = 2.0 * math.pi * freqs[idx]  # 圆频率
    p = math.sin(math.radians(angle_deg if angle_deg > 0 else 1e-10)) / CS_BED  # 水平慢度
    if damping:  # 启用阻尼
        ab_map = {seg['name']: rayleigh_for(seg['cs'], seg['name'] == 'bedrock', fc, qs_factor)  # 各层瑞利系数
                  for seg in column}  # 按层名构表
    else:  # 全弹性
        ab_map = {seg['name']: (0.0, 0.0) for seg in column}  # 全零
    sol = fd_solve_column(column, p, omega, ab_map)  # 求解柱
    ux, uy = fd_surface_disp(sol, omega, p)  # 地表位移谱（单位入射）
    spec_ax = np.zeros(len(freqs), dtype=complex)  # 地表 x 加速度谱
    spec_ay = np.zeros(len(freqs), dtype=complex)  # 地表 y 加速度谱
    spec_ax[idx] = ux * A[idx]  # a = -ω²·u·U0 = u_unit·A（两次积分相消）
    spec_ay[idx] = uy * A[idx]  # 同上
    ax = np.fft.irfft(spec_ax, n=Nfft)  # x 加速度时程
    ay = np.fft.irfft(spec_ay, n=Nfft)  # y 加速度时程
    nwin = Nfft if window is None else min(Nfft, int(window / dt))  # 评估窗口长度
    pga_h = np.max(np.abs(ax[:nwin]))  # 水平 PGA
    pga_v = np.max(np.abs(ay[:nwin]))  # 垂直 PGA
    denom = factor_h_of(angle_deg, CS_BED, cp_of(CS_BED)) * np.max(np.abs(acc))  # 解析分母
    return pga_h / denom, pga_v / denom  # 返回 TAF 台阶值


def make_ricker(fc, dt=1e-3, T=4.0):  # 生成 Ricker 子波加速度
    """中心频率 fc，时移 1.1/fc，时长 T。"""
    t = np.arange(0.0, T, dt)  # 时间轴
    t0 = 1.1 / fc  # 时移
    arg = (math.pi * fc) ** 2 * (t - t0) ** 2  # 公共项
    return t, (1.0 - 2.0 * arg) * np.exp(-arg)  # 返回时间轴与子波


# ==========================================================
#  主流程
# ==========================================================
def main():  # 主入口
    """先做解析校核，再算各远场柱台阶值与敏感性。"""
    rec = np.loadtxt(RICKER_PATH)  # 读取 4Hz Ricker 记录
    t_arr, acc = rec[:, 0], rec[:, 1]  # 时间与加速度
    dt = t_arr[1] - t_arr[0]  # 步长
    spec = np.abs(np.fft.rfft(acc - acc.mean()))  # 幅值谱
    fr = np.fft.rfftfreq(len(acc), dt)  # 频率轴
    fc_est = fr[np.argmax(spec)]  # 主频估计（v6 同口径）
    print('记录: N=%d, dt=%.4fs, 时长=%.2fs, fc估计=%.3fHz, PGA_in=%.4g' %
          (len(acc), dt, t_arr[-1], fc_est, np.max(np.abs(acc))))  # 打印记录信息

    # ---- 校核 T1：弹性半空间垂直入射 → TAF_h 应=1.0（地表运动=2E，分母=2E）----
    col_half = make_column([('bedrock', CS_BED, 0.0, H_UPPER)])  # 均质基岩柱
    th, tv = taf_plateau(col_half, 0.0, acc, dt, damping=False)  # 弹性半空间
    print('[T1] 弹性半空间 0°: TAF_h=%.4f (应=1.0), TAF_v=%.4f (应≈0)' % (th, tv))  # 打印校核

    # ---- 校核 T2：单层弹性谐波传递函数 vs 解析 SH 公式 ----
    col_1L = make_column([('bedrock', CS_BED, 0.0, BEDROCK_TOP),  # 基岩
                          ('overlying', CS_OVER, BEDROCK_TOP, H_LOWER)])  # 单覆盖层 200m
    fch = 1.0  # 校核频率 (Hz)
    om = np.array([2.0 * math.pi * fch])  # 单频
    ab0 = {'bedrock': (0.0, 0.0), 'overlying': (0.0, 0.0)}  # 弹性
    sol = fd_solve_column(col_1L, 1e-15, om, ab0)  # 近垂直入射
    ux, _ = fd_surface_disp(sol, om, 1e-15)  # 地表位移
    kh = 2.0 * math.pi * fch * 200.0 / CS_OVER  # 层内相位角
    az = (RHO * CS_OVER) / (RHO * CS_BED)  # 阻抗比（层/基岩）
    ana = 2.0 / abs(complex(math.cos(kh), az * math.sin(kh)))  # 解析地表幅值（入射 E=1）
    print('[T2] 单层 %.1fHz: fd|ux|=%.4f vs 解析=%.4f' % (fch, abs(ux[0]), ana))  # 打印校核

    # ---- 远场柱定义 ----
    col_up_soft = make_column([('bedrock', CS_BED, 0.0, BEDROCK_TOP),  # 上平台柱：基岩
                               ('overlying', CS_OVER, BEDROCK_TOP, H_UPPER - H1),  # 覆盖层 350m
                               ('surface', CS_SURF_SOFT, H_UPPER - H1, H_UPPER)])  # 软表层 50m
    col_up_stiff = make_column([('bedrock', CS_BED, 0.0, BEDROCK_TOP),  # 上平台柱：基岩
                                ('overlying', CS_OVER, BEDROCK_TOP, H_UPPER - H1),  # 覆盖层 350m
                                ('surface', CS_SURF_STIFF, H_UPPER - H1, H_UPPER)])  # 硬表层 50m
    col_low = make_column([('bedrock', CS_BED, 0.0, BEDROCK_TOP),  # 下平台柱：基岩
                           ('overlying', CS_OVER, BEDROCK_TOP, H_LOWER)])  # 覆盖层 200m（无表层）

    # ---- 主结果：v6 同参（阻尼开，Qs=0.05cs，fc 自动估计）----
    print('\n==== 远场一维台阶 TAF（v6 同参：阻尼开 Qs=0.05cs）====')  # 标题
    for name, col in (('上平台 软表层', col_up_soft), ('上平台 硬表层', col_up_stiff), ('下平台', col_low)):  # 三柱
        for ang in (0.0, 15.0):  # 两角度
            th, tv = taf_plateau(col, ang, acc, dt, damping=True, fc=fc_est)  # 计算台阶
            print('  %s θs=%2g°: TAF_h=%.3f  TAF_v=%.3f' % (name, ang, th, tv))  # 打印

    # ---- 敏感性 1：阻尼关（全弹性）----
    print('\n==== 敏感性：全弹性（阻尼关）====')  # 标题
    for name, col in (('上平台 软表层', col_up_soft), ('上平台 硬表层', col_up_stiff), ('下平台', col_low)):  # 三柱
        th, tv = taf_plateau(col, 0.0, acc, dt, damping=False)  # 0° 弹性
        print('  %s θs=0°: TAF_h=%.3f  TAF_v=%.3f' % (name, th, tv))  # 打印

    # ---- 敏感性 2：时窗截断（FE 分析步=2s vs 全程）----
    print('\n==== 敏感性：PGA 评估时窗（软表层柱, 0°, 阻尼开）====')  # 标题
    for win in (1.0, 1.5, 2.0, 3.0, None):  # 各窗口
        th, tv = taf_plateau(col_up_soft, 0.0, acc, dt, damping=True, fc=fc_est, window=win)  # 计算
        print('  窗口=%s s: TAF_h=%.3f  TAF_v=%.3f' % (str(win), th, tv))  # 打印

    # ---- 敏感性 3：Ricker 主频扫描（软表层柱, 0°, 阻尼开）----
    print('\n==== 敏感性：Ricker 主频 fc 扫描（软表层柱, 0°, 阻尼开 Qs=0.05cs）====')  # 标题
    for fcx in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):  # 各主频
        _, accx = make_ricker(fcx)  # 生成子波
        th, tv = taf_plateau(col_up_soft, 0.0, accx, 1e-3, damping=True, fc=fcx)  # 计算
        th2, _ = taf_plateau(col_up_stiff, 0.0, accx, 1e-3, damping=True, fc=fcx)  # 硬表层对照
        print('  fc=%.1fHz (a0=%.2f): 软 TAF_h=%.3f | 硬 TAF_h=%.3f' %
              (fcx, fcx * 2.0 * 200.0 / CS_OVER, th, th2))  # 打印（a0=2fc(H-h)/Vs2）

    # ---- 敏感性 4：Q 取法（软表层柱, 0°）----
    print('\n==== 敏感性：Q 取法（软表层柱, 0°, 4Hz 记录）====')  # 标题
    for qf in (0.05, 0.1, 0.2):  # 各 qs_factor
        th, _ = taf_plateau(col_up_soft, 0.0, acc, dt, damping=True, fc=fc_est, qs_factor=qf)  # 计算
        print('  Qs=%.2f·cs (表层Q=%g): TAF_h=%.3f' % (qf, qf * CS_SURF_SOFT, th))  # 打印

    # ---- 频域传递曲线：软/硬表层柱 |地表ux|/2 随频率（供resonance定位）----
    print('\n==== 谐波放大 |u_surf|/2E（0°, 弹性）====')  # 标题
    fgrid = np.arange(0.5, 10.01, 0.5)  # 频率网格
    om = 2.0 * math.pi * fgrid  # 圆频率
    for name, col in (('软表层柱', col_up_soft), ('硬表层柱', col_up_stiff), ('下平台柱', col_low)):  # 三柱
        ab0 = {seg['name']: (0.0, 0.0) for seg in col}  # 弹性
        sol = fd_solve_column(col, 1e-15, om, ab0)  # 求解
        ux, _ = fd_surface_disp(sol, om, 1e-15)  # 地表位移
        amp = np.abs(ux) / 2.0  # 相对露头放大
        line = '  %s: ' % name + ' '.join('%.1fHz=%.2f' % (f, a) for f, a in zip(fgrid, amp))  # 拼行
        print(line)  # 打印


if __name__ == '__main__':  # 直接运行
    main()  # 调用主入口
