# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""用 6Hz 瑞克波驱动自由场引擎的测试（不依赖 Abaqus）。

目的：在主频 6Hz 的输入下，核查向量化自由场引擎的物理一致性：
  1. 均匀半空间垂直入射：地表水平放大 ≈ 2、竖向 ≈ 0、自由面应力 ≈ 0；
  2. 斜入射 15°：自由面应力 ≈ 0 且结果全部有限；
  3. 项目双层场地（基岩 Vs=2000 / 覆盖层 Vs=1600）：报告地表放大与有限性。
运行：在含 numpy 的普通 Python 环境下 `python test/test_ricker_6hz.py`。
"""

import os, sys, math, importlib  # 导入标准库
import numpy as np  # 导入数值库

# 屏蔽 Abaqus 依赖，强制走纯数值分支
for _m in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):
    sys.modules[_m] = None
PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Modeling', 'Multi')  # 上级目录（含 v7 脚本）
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)
eng = importlib.import_module('VAB_oblique_TAF_double_v7')  # 导入 v7 引擎


def make_ricker(dt, n, f0, t0):  # 生成 Ricker 子波作为输入速度时程
    """返回 (时间轴, 6Hz 主频 Ricker 速度时程)。"""
    t = np.arange(n) * dt  # 时间轴
    arg = (math.pi * f0 * (t - t0)) ** 2  # Ricker 自变量
    return t, (1.0 - 2.0 * arg) * np.exp(-arg)  # Ricker 波形


def rfft_inputs(vel, dt):  # 输入速度补零到 2 的幂并 rfft
    """返回 (vel_freq, freq_arr, N_fft, N_orig)。"""
    n_orig = len(vel)  # 原始长度
    n_fft = 1  # 初始化 FFT 长度
    while n_fft < n_orig:  # 找不小于原长的 2 的幂
        n_fft *= 2  # 倍增
    n_fft *= 2  # 再翻倍避免时域混叠（与主程序一致）
    vp = np.zeros(n_fft); vp[:n_orig] = vel  # 补零
    return np.fft.rfft(vp), np.fft.rfftfreq(n_fft, d=dt), n_fft, n_orig  # 频域输入与长度


def main():  # 测试主函数
    eng._FF_TRANSFER_CACHE.clear()  # 清空缓存，保证干净计时与结果

    # ---- 输入：6Hz 瑞克速度子波 ----
    dt = 0.001  # 时间步长 (s)，Nyquist=500Hz 远高于 6Hz
    n = 2000  # 采样点数（2 s）
    f0 = 6.0  # Ricker 主频 (Hz)
    t0 = 0.5  # 峰值时刻 (s)
    t_arr, vel_in = make_ricker(dt, n, f0, t0)  # 生成输入速度时程
    vel_freq, freq_arr, n_fft, n_orig = rfft_inputs(vel_in, dt)  # 频域准备
    v_peak = np.max(np.abs(vel_in))  # 输入速度峰值
    print('输入: 6Hz Ricker, dt=%.3f, N=%d, 峰值=%.4f' % (dt, n, v_peak))  # 打印输入信息

    # ---- 材料 ----
    rho, vv = 2500.0, 0.3  # 密度与泊松比（项目取值）
    mat_b = eng._compute_material_params(2000.0, vv, rho)  # 基岩 Vs=2000
    mat_o = eng._compute_material_params(1600.0, vv, rho)  # 覆盖层 Vs=1600（velocity_ratio=1.25）

    # ============ 检验 1：均匀半空间垂直入射 ============
    print('\n===== 1. 均匀半空间 (Vs=2000) 垂直入射 =====')  # 分节标题
    cs = 2000.0  # 半空间剪切波速
    mat = eng._compute_material_params(cs, vv, rho)  # 半空间材料
    y_bottom, h_bed, h_ov = 0.0, 200.0, 200.0  # 底高、基岩厚、覆盖层厚
    y_surf = y_bottom + h_bed + h_ov  # 自由面 y
    p0 = math.sin(math.radians(1e-10)) / cs  # 近垂直入射水平慢度（≈0）
    ff = eng._compute_freefield_at_node(y_target=y_surf, x_target=0.0,
        mat_bedrock=mat, mat_overlying=mat, h_bedrock=h_bed, h_overlying=h_ov,
        y_bottom=y_bottom, p_horiz=p0, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)
    vx = ff['dotux'][:n_orig]; vy = ff['dotuy'][:n_orig]  # 地表水平/竖向速度
    syy = ff['syy'][:n_orig]; sxy = ff['sxy'][:n_orig]  # 地表法向/剪应力
    sstress = mat['GG'] / cs * v_peak  # 特征应力量级 ρ·cs·v
    amp = np.max(np.abs(vx)) / v_peak  # 地表水平放大比
    print('地表水平放大比 (应≈2.0):     %.4f' % amp)  # 放大比
    print('竖向/水平 (应≈0):            %.3e' % (np.max(np.abs(vy)) / np.max(np.abs(vx))))  # 竖向占比
    print('|σ_yy|/特征应力 (应≈0):      %.3e' % (np.max(np.abs(syy)) / sstress))  # 自由面法向应力
    print('|τ_xy|/特征应力 (应≈0):      %.3e' % (np.max(np.abs(sxy)) / sstress))  # 自由面剪应力

    # ============ 检验 2：斜入射 15° ============
    print('\n===== 2. 均匀半空间斜入射 15° =====')  # 分节标题
    p15 = math.sin(math.radians(15.0)) / cs  # 15° 水平慢度
    ffb = eng._compute_freefield_at_node(y_target=y_surf, x_target=0.0,
        mat_bedrock=mat, mat_overlying=mat, h_bedrock=h_bed, h_overlying=h_ov,
        y_bottom=y_bottom, p_horiz=p15, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)
    finite_ok = all(np.all(np.isfinite(ffb[k])) for k in ffb)  # 结果是否全部有限
    print('|σ_yy|/特征应力 (应≈0):      %.3e' % (np.max(np.abs(ffb['syy'][:n_orig])) / sstress))  # 自由面法向应力
    print('|τ_xy|/特征应力 (应≈0):      %.3e' % (np.max(np.abs(ffb['sxy'][:n_orig])) / sstress))  # 自由面剪应力
    print('结果全部有限 (应 True):      %s' % finite_ok)  # 有限性

    # ============ 检验 3：项目双层场地（坡顶柱，覆盖层厚 200）斜入射 15° ============
    print('\n===== 3. 双层场地 (基岩2000/覆盖层1600) 斜入射 15° =====')  # 分节标题
    p15b = math.sin(math.radians(15.0)) / 2000.0  # 双层用基岩波速定义慢度
    y_surf2 = 0.0 + 200.0 + 200.0  # 坡顶地表 y（基岩200+覆盖200）
    ff2 = eng._compute_freefield_at_node(y_target=y_surf2, x_target=0.0,
        mat_bedrock=mat_b, mat_overlying=mat_o, h_bedrock=200.0, h_overlying=200.0,
        y_bottom=0.0, p_horiz=p15b, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)
    vx2 = ff2['dotux'][:n_orig]  # 双层地表水平速度
    finite2 = all(np.all(np.isfinite(ff2[k])) for k in ff2)  # 有限性
    print('地表水平放大比 (含层共振，>2):  %.4f' % (np.max(np.abs(vx2)) / v_peak))  # 双层放大比（含覆盖层放大）
    print('结果全部有限 (应 True):      %s' % finite2)  # 有限性

    # ============ 判定 ============
    print('\n===== 判定 =====')  # 判定标题
    ok_amp = abs(amp - 2.0) < 0.05  # 放大比≈2
    ok_vy = (np.max(np.abs(vy)) / np.max(np.abs(vx))) < 1e-3  # 竖向≈0
    ok_free = (np.max(np.abs(syy)) / sstress < 1e-3) and (np.max(np.abs(sxy)) / sstress < 1e-3)  # 自由面应力≈0
    allok = ok_amp and ok_vy and ok_free and finite_ok and finite2  # 总判定
    print('1 地表放大=2:       %s' % ('通过' if ok_amp else '不通过'))  # 放大判定
    print('1 竖向≈0:          %s' % ('通过' if ok_vy else '不通过'))  # 竖向判定
    print('1 自由面应力≈0:     %s' % ('通过' if ok_free else '不通过'))  # 自由面判定
    print('2 斜入射有限:       %s' % ('通过' if finite_ok else '不通过'))  # 斜入射判定
    print('3 双层有限:         %s' % ('通过' if finite2 else '不通过'))  # 双层判定
    print('\n总判定: %s' % ('全部通过 ✓' if allok else '存在不通过 ✗'))  # 总判定


if __name__ == '__main__':  # 判断是否直接运行
    main()  # 调用测试主函数
