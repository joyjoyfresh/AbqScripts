# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""对比 v3（射线延迟法）与 v7（频域传播矩阵法）的自由场等效节点力（不依赖 Abaqus）。

做法：
  - 同一条输入波（6Hz Ricker，当作加速度记录）、同一斜坡几何/材料/入射角；
  - 取若干代表性边界节点，用两套自由场算法各算等效节点力 fx/fy 时程；
  - 弹簧/阻尼/影响长度系数对两者完全相同，故差异只来自自由场 u/v/σ 本身；
  - 用【循环互相关】对全缓冲区比较：自动吸收"时间零点参考差"和"FFT 环绕"，
    给出波形相似度（最佳相关）与峰值比，避免被时移/环绕误导。
说明：v3 自由场逻辑原内嵌在 VAB_oblique（含 Abaqus 依赖、无法直接 import），
      故按 v3 源码逐式重建为独立纯函数；v7 直接调用其引擎。
      重要口径：v3 水平传播用"正延迟"(+x 扫过)，v7 用 exp(+iωpx)(相当于 -x 扫过)，
      二者水平方向约定相反；循环互相关按节点各自对齐，故比较的是"波形形状"本身。
运行：python test/compare_v3_vs_v7_freefield.py
"""

import os, sys, math, importlib  # 导入标准库
import numpy as np  # 导入数值库

for _m in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 屏蔽 Abaqus
    sys.modules[_m] = None
PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Modeling', 'Multi')  # 上级目录
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
eng = importlib.import_module('VAB_oblique_TAF_double_v7')  # 导入 v7 引擎


# ============================================================
#  v3 射线延迟法：按 v3 源码逐式重建
# ============================================================
def _safe_arcsin(v):  # v3 安全反正弦
    return math.asin(max(-1.0, min(1.0, v)))  # 截断到合法域


def _interface_sv_coeff(alpha1, mat1, mat2):  # v3：界面 SV 阻抗近似（忽略 SV→P）
    z1s = mat1['density'] * mat1['cs'] * max(1e-8, math.cos(alpha1))  # 入射侧阻抗
    alpha2 = _safe_arcsin(mat2['cs'] * math.sin(alpha1) / mat1['cs'])  # 透射角
    z2s = mat2['density'] * mat2['cs'] * max(1e-8, math.cos(alpha2))  # 透射侧阻抗
    denom = z1s + z2s if abs(z1s + z2s) > 1e-12 else 1e-12  # 分母防零
    return {'Rss': (z2s - z1s) / denom, 'Tss': 2.0 * z2s / denom, 'alpha2': alpha2}  # 系数


def _free_surface_sv_coeff(alpha, cp, cs):  # v3：自由面 SV 反射系数
    beta_p = _safe_arcsin(cp * math.sin(alpha) / cs)  # 转换角
    num = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) - cp ** 2 * math.cos(2 * alpha) ** 2  # A1 分子
    den = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) + cp ** 2 * math.cos(2 * alpha) ** 2  # 分母
    den = den if abs(den) >= 1e-12 else 1e-12  # 防零
    return {'A1': num / den, 'A2': (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / den, 'beta': beta_p}


def _free_surface_p_coeff(beta, cp, cs):  # v3：自由面 P 反射系数
    alpha = _safe_arcsin(cs * math.sin(beta) / cp)  # 对应入射角
    num = cp ** 2 * math.cos(2 * alpha) ** 2 - cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # B2 分子
    den = cp ** 2 * math.cos(2 * alpha) ** 2 + cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 分母
    den = den if abs(den) >= 1e-12 else 1e-12  # 防零
    return {'B2': num / den, 'alpha': alpha}  # 返回 B2


def _delay(sig, delay_t, dt, M):  # 将 1D 信号延迟 delay_t（采样平移，长度 M）
    nd = int(np.round(delay_t / dt))  # 延迟步数
    out = np.zeros(M)  # 输出数组
    if nd < M:  # 有效平移
        m = min(len(sig), M - nd)  # 可填充长度
        out[nd:nd + m] = sig[:m]  # 平移填入
    return out  # 返回延迟信号


def v3_global_coeff(angle, mat_b, mat_o, bedrock_thickness, ymax, ymin, xmin, xmax, max_order=3):
    """重建 v3 全局反射/转换/循环系数。"""
    angle = 1e-10 if angle == 0 else round(angle, 4)  # 角度处理
    alpha1 = math.radians(angle)  # 入射 SV 角
    ic12 = _interface_sv_coeff(alpha1, mat_b, mat_o)  # 界面 1->2
    alpha2 = ic12['alpha2']  # 透射角
    beta1 = _safe_arcsin(mat_b['cp'] * math.sin(alpha1) / mat_b['cs'])  # 基岩 P 角
    beta2 = _safe_arcsin(mat_o['cp'] * math.sin(alpha2) / mat_o['cs']) if abs(math.sin(alpha2)) > 0 else 1e-10  # 覆盖层 P 角
    fsv2 = _free_surface_sv_coeff(alpha2, mat_o['cp'], mat_o['cs'])  # 覆盖层自由面 SV
    fp2 = _free_surface_p_coeff(beta2, mat_o['cp'], mat_o['cs'])  # 覆盖层自由面 P
    ic21 = _interface_sv_coeff(alpha2, mat_o, mat_b)  # 界面 2->1
    h2 = max(0.0, ymax - bedrock_thickness)  # 覆盖层厚度（v3 全局一刀切）
    cycle_sv = fsv2['A1'] * ic21['Rss']  # SV 循环系数
    cycle_p = fp2['B2'] * ic21['Rss']  # P 循环系数
    order = max(0, int(max_order))  # 阶数上限
    sum_sv = sum(cycle_sv ** k for k in range(order + 1))  # SV 几何级数
    sum_p = sum(cycle_p ** k for k in range(order + 1))  # P 几何级数
    return {  # 打包系数
        'alpha': alpha1, 'beta_p': beta1,
        'A1': ic12['Rss'] + ic12['Tss'] * fsv2['A1'] * ic21['Tss'] * sum_sv,  # 等效反射
        'A2': ic12['Tss'] * fsv2['A2'] * ic21['Tss'] * sum_p,  # 等效转换
        'cs': mat_b['cs'], 'cp': mat_b['cp'], 'GG': mat_b['GG'], 'lam': mat_b['lam'],
        'cs2': mat_o['cs'], 'cp2': mat_o['cp'], 'alpha2': alpha2, 'beta2': beta2,
        'Ly': bedrock_thickness - ymin, 'Lx': xmax - xmin,
        'cycle_sv': cycle_sv, 'cycle_p': cycle_p,
        'cycle_delay_sv': (2.0 * h2 * math.cos(alpha2) / mat_o['cs']) if h2 > 0 else 0.0,
        'cycle_delay_p': (2.0 * h2 * math.cos(beta2) / mat_o['cp']) if h2 > 0 else 0.0,
        'order_count': order}


def v3_node_force(boundary, x0, y0, ymax_col, a, sig_dis, sig_vel, dt, M, kn, cn, kt, ct, A):
    """v3：单节点等效力 fx/fy（长度 M）。"""
    alpha, beta_p = a['alpha'], a['beta_p']  # 入射/P 角
    A1, A2 = a['A1'], a['A2']  # 等效系数
    cs, cp, GG, lam = a['cs'], a['cp'], a['GG'], a['lam']  # 基岩材料
    cs2, cp2, alpha2, beta2 = a['cs2'], a['cp2'], a['alpha2'], a['beta2']  # 覆盖层
    Ly, Lx = a['Ly'], a['Lx']  # 几何
    cyc_sv, cyc_p = a['cycle_sv'], a['cycle_p']  # 循环系数
    cyc_d_sv, cyc_d_p, order = a['cycle_delay_sv'], a['cycle_delay_p'], a['order_count']  # 循环延迟/阶数

    if boundary in ('l', 'r'):  # 侧边界走时
        if y0 <= Ly:  # 基岩段
            tA = y0 * math.cos(alpha) / cs
            tB = (2 * Ly - y0) * math.cos(alpha) / cs
            tC = ((Ly - y0) / (cp * math.cos(beta_p))
                  + (Ly - (Ly - y0) * math.tan(alpha) * math.tan(beta_p)) * math.cos(alpha) / cs)
        else:  # 覆盖层段
            tA = Ly * math.cos(alpha) / cs + (y0 - Ly) * math.cos(alpha2) / cs2
            tB = Ly * math.cos(alpha) / cs + (2 * ymax_col - Ly - y0) * math.cos(alpha2) / cs2
            tC = Ly * math.cos(alpha) / cs + (y0 - Ly) * math.cos(beta2) / cp2
        if boundary == 'r':  # 右边界叠加横向延迟
            ex = Lx * math.sin(alpha) / cs
            tA += ex; tB += ex; tC += ex
    else:  # 底边界走时
        tA = x0 * math.sin(alpha) / cs
        tB = (2 * Ly + x0 * math.tan(alpha)) * math.cos(alpha) / cs
        tC = (Ly / (cp * math.cos(beta_p))
              + (Ly * math.cos(alpha) + x0 * math.sin(alpha) - Ly * math.tan(beta_p) * math.sin(alpha)) / cs)
    tA, tB, tC = (round(tt / dt) * dt for tt in (tA, tB, tC))  # 对齐步长

    def superpose(sig):  # 主+反射+转换三路径叠加
        sA = _delay(sig, tA, dt, M)
        sB = np.zeros(M); sC = np.zeros(M)
        for k in range(order + 1):
            sB += (cyc_sv ** k) * _delay(sig, tB + k * cyc_d_sv, dt, M)
            sC += (cyc_p ** k) * _delay(sig, tC + k * cyc_d_p, dt, M)
        return sA, sB, sC

    dA, dB, dC = superpose(sig_dis)  # 位移路径
    vA, vB, vC = superpose(sig_vel)  # 速度路径
    ux = dA * math.cos(alpha) - A1 * dB * math.cos(alpha) + A2 * dC * math.sin(beta_p)  # x 位移
    uy = -dA * math.sin(alpha) - A1 * dB * math.sin(alpha) - A2 * dC * math.cos(beta_p)  # y 位移
    dotux = vA * math.cos(alpha) - A1 * vB * math.cos(alpha) + A2 * vC * math.sin(beta_p)  # x 速度
    dotuy = -vA * math.sin(alpha) - A1 * vB * math.sin(alpha) - A2 * vC * math.cos(beta_p)  # y 速度

    sin2a, cos2a = math.sin(2 * alpha), math.cos(2 * alpha)  # 双角
    sin2bp = math.sin(beta_p) ** 2; sin2bp_2 = math.sin(2 * beta_p); cosbp2 = math.cos(beta_p) ** 2  # P 角项
    if boundary == 'l':  # 左应力
        sigmax = GG / cs * sin2a * (vA - A1 * vB) + A2 * (lam + 2 * GG * sin2bp) / cp * vC
        sigmay = GG / cs * cos2a * (vA + A1 * vB) - A2 * GG * sin2bp_2 / cp * vC
    elif boundary == 'r':  # 右应力
        sigmax = GG / cs * sin2a * (-vA + A1 * vB) - A2 * (lam + 2 * GG * sin2bp) / cp * vC
        sigmay = GG / cs * cos2a * (-vA - A1 * vB) + A2 * GG * sin2bp_2 / cp * vC
    else:  # 底应力
        sigmax = GG / cs * cos2a * (vA + A1 * vB) - A2 * GG * sin2bp_2 / cp * vC
        sigmay = GG / cs * sin2a * (-vA + A1 * vB) + A2 * (lam + 2 * GG * cosbp2) / cp * vC

    if boundary in ('l', 'r'):  # 侧边等效力
        fx = kn * ux + cn * dotux + A * sigmax
        fy = kt * uy + ct * dotuy + A * sigmay
    else:  # 底边等效力
        fx = kt * ux + ct * dotux + A * sigmax
        fy = kn * uy + cn * dotuy + A * sigmay
    return fx, fy  # 长度 M


# ============================================================
#  v7 频域传播矩阵法（全缓冲区，不截断）
# ============================================================
def v7_node_force(boundary, x0, y0, h_ov, mat_b, mat_o, bedrock_thickness, ymin,
                  p_horiz, vel_freq, freq_arr, dt, N_fft, kn, cn, kt, ct, A):
    """v7：单节点等效力 fx/fy（全长 N_fft）。"""
    ff = eng._compute_freefield_at_node(y_target=y0, x_target=x0, mat_bedrock=mat_b,
        mat_overlying=mat_o, h_bedrock=bedrock_thickness, h_overlying=h_ov, y_bottom=ymin,
        p_horiz=p_horiz, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=N_fft)  # 自由场全长
    ux, uy, dux, duy = ff['ux'], ff['uy'], ff['dotux'], ff['dotuy']  # 位移/速度
    sxx, syy, sxy = ff['sxx'], ff['syy'], ff['sxy']  # 应力
    if boundary == 'l':  # 左：外法向 (-1,0)
        fx = kn * ux + cn * dux + A * (-sxx); fy = kt * uy + ct * duy + A * (-sxy)
    elif boundary == 'r':  # 右：外法向 (+1,0)
        fx = kn * ux + cn * dux + A * (sxx); fy = kt * uy + ct * duy + A * (sxy)
    else:  # 底：外法向 (0,-1)
        fx = kt * ux + ct * dux + A * (-sxy); fy = kn * uy + cn * duy + A * (-syy)
    return fx, fy  # 全长 N_fft


# ============================================================
#  循环互相关对比（对时移/FFT 环绕鲁棒）
# ============================================================
def circ_compare(f3, f7, dt):  # 用循环互相关比较两条全缓冲区力时程
    M = min(len(f3), len(f7))  # 统一长度
    f3 = f3[:M]; f7 = f7[:M]  # 截到同长
    p3 = np.max(np.abs(f3)) + 1e-30  # v3 峰值（全缓冲区，环绕不丢）
    p7 = np.max(np.abs(f7)) + 1e-30  # v7 峰值
    s3, s7 = np.std(f3), np.std(f7)  # 标准差
    if s3 < 1e-30 or s7 < 1e-30:  # 退化保护
        return p7 / p3, float('nan'), float('nan'), 0.0
    a = (f3 - f3.mean()) / s3  # 标准化 v3
    b = (f7 - f7.mean()) / s7  # 标准化 v7
    xc = np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n=M) / M  # 循环互相关（各循环位移的相关系数）
    corr0 = float(xc[0])  # 零位移相关
    k = int(np.argmax(xc))  # 最佳循环位移索引
    corr_best = float(xc[k])  # 最佳相关
    lag = k if k <= M // 2 else k - M  # 转为带符号位移
    return p7 / p3, corr0, corr_best, lag * dt * 1000.0  # 峰值比/零移相关/最佳相关/最佳时移(ms)


def main():  # 对比主函数
    eng._FF_TRANSFER_CACHE.clear()  # 清空缓存
    wave = os.path.join(PARENT, '..', '..', 'Wave', 'Impulse', 'ricker_wavelet_6Hz.txt')  # 波文件
    ACC = np.loadtxt(wave)  # 读取
    dt = ACC[1, 0] - ACC[0, 0]  # 步长
    N = 4000  # 延长后长度（补零，保证走时延迟与混响尾都在缓冲区内）
    acc = np.zeros(N); acc[:len(ACC)] = ACC[:, 1]  # 补零加速度
    t_arr = np.arange(N) * dt  # 时间轴
    N_fft = 1
    while N_fft < N:
        N_fft *= 2
    N_fft *= 2  # FFT 长度
    print('输入: %s  原长=%d  延长 N=%d  N_fft=%d (%.1fs)  dt=%.4f' %
          (os.path.basename(wave), len(ACC), N, N_fft, N_fft * dt, dt))

    vv, rho = 0.3, 2500.0  # 材料
    cs_b = math.sqrt((26e9 / (2 * (1 + vv))) / rho); cs_o = cs_b / 1.25  # 波速
    mat_b = eng._compute_material_params(cs_b, vv, rho); mat_o = eng._compute_material_params(cs_o, vv, rho)
    angle = 15.0  # 入射角

    bedrock_thickness = 200.0  # 基岩厚
    H_upper, H_lower = 600.0, 400.0  # 坡顶/坡脚地表高
    left_flat, w_slope, total_L = 1000.0, 200.0, 1800.0  # 平台/坡宽/总长
    ymin, ymax_l, ymax_r = 0.0, H_upper, H_lower  # 底/左顶/右顶
    ymax = max(ymax_l, ymax_r); xmin, xmax = 0.0, total_L  # 全局最高/横向范围
    p_horiz = math.sin(math.radians(angle)) / cs_b  # 水平慢度

    vel3 = np.zeros(N); vel3[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)  # v3 速度（裸积分）
    dis3 = np.zeros(N); dis3[1:] = np.cumsum((vel3[:-1] + vel3[1:]) / 2 * dt)  # v3 位移
    vel7, _ = eng._integrate_acc_to_velocity(acc, dt, t_arr)  # v7 速度（去趋势）
    vp = np.zeros(N_fft); vp[:N] = vel7; vel_freq = np.fft.rfft(vp); freq_arr = np.fft.rfftfreq(N_fft, d=dt)  # v7 频域
    coeff3 = v3_global_coeff(angle, mat_b, mat_o, bedrock_thickness, ymax, ymin, xmin, xmax, 3)  # v3 系数
    print('v3 等效系数 A1=%.4f A2=%.4f (忽略界面 SV→P) | 入射方向: v3=+x扫过 / v7=-x扫过(相位约定相反)'
          % (coeff3['A1'], coeff3['A2']))

    nodes = [
        ('左-基岩(入)',   'l', 0.0,     100.0, ymax_l - bedrock_thickness, ymax_l),
        ('左-覆盖(入)',   'l', 0.0,     400.0, ymax_l - bedrock_thickness, ymax_l),
        ('右-覆盖(出)',   'r', total_L, 300.0, ymax_r - bedrock_thickness, ymax_r),
        ('底-坡顶下(入)', 'b', 200.0,   0.0,
         eng._surface_y_at(200.0, H_upper, H_lower, left_flat, w_slope) - bedrock_thickness, ymax),
        ('底-坡脚下(入)', 'b', 1700.0,  0.0,
         eng._surface_y_at(1700.0, H_upper, H_lower, left_flat, w_slope) - bedrock_thickness, ymax),
    ]

    print('\n%-13s %-3s | %-9s %-12s %-10s %-10s %-8s' %
          ('节点(入/出)', '分量', '峰值比v7/v3', '零移相关', '最佳相关', '最佳时移ms', ''))
    print('-' * 72)
    for name, b, x, y, h_ov, ymax_col in nodes:  # 遍历代表节点
        mat = mat_b if y < bedrock_thickness + 1e-4 else mat_o  # 弹簧/阻尼用材料
        A = 4.0  # 影响长度（两法相同）
        kn = mat['GG'] / 2.0 / ymax * A; cn = mat['density'] * mat['cp'] * A  # 法向
        kt = mat['GG'] / 4.0 / ymax * A; ct = mat['density'] * mat['cs'] * A  # 切向
        fx3, fy3 = v3_node_force(b, x, y, ymax_col, coeff3, dis3, vel3, dt, N_fft, kn, cn, kt, ct, A)  # v3
        fx7, fy7 = v7_node_force(b, x, y, h_ov, mat_b, mat_o, bedrock_thickness, ymin,
                                 p_horiz, vel_freq, freq_arr, dt, N_fft, kn, cn, kt, ct, A)  # v7
        first = True  # 控制首行打印节点名
        for comp, f3, f7 in (('fx', fx3, fx7), ('fy', fy3, fy7)):  # 分量比较
            ratio, c0, cb, lag = circ_compare(f3, f7, dt)  # 循环互相关指标
            print('%-13s %-3s | %-9.3f %-12.3f %-10.3f %-10.1f' %
                  (name if first else '', comp, ratio, c0, cb, lag))
            first = False  # 后续不再重复名

    print('\n解读：')
    print(' - 峰值比≈1 且 最佳相关≈1 → 该节点两法注入的波几乎一致。')
    print(' - 最佳相关高但峰值比偏离 1 → 波形相似、幅值不同（受界面转换/混响截断/厚度差异影响）。')
    print(' - 最佳相关偏低 → 波形本身不同，FEM/TAF 会有可见差异。')
    print(' - 标"(入)"为入射侧边界(力大、对结果影响主导)，"(出)"为出射侧(粘弹性边界吸收,力小、影响次要)。')


if __name__ == '__main__':
    main()
