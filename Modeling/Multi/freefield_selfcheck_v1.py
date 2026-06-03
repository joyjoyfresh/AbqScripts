# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""
自由场引擎退化自检（不依赖 Abaqus）。

目的：验证 VAB_oblique_TAF_double_v6.py 中的双层自由场引擎
（_compute_freefield_at_node / _propagator_matrix_freefield）在
退化算例下是否符合真实地震波物理，重点核查 v5 评审中存疑的
"应力 σ 与位移/速度之间的相对标度与符号"是否自洽。

退化算例：均匀半空间（覆盖层与基岩同材料）+ 自由面。
真实物理预期：
  1. 自由面应力 σ_yy ≈ 0, τ_xy ≈ 0（自由面边界条件）。
  2. 垂直入射（angle=0）时地表水平运动 = 2 × 入射波（自由面放大）。
  3. 垂直入射时地表竖向运动 ≈ 0（SV 纯水平偏振，无 SV->P 转换）。
  4. 阻抗关系：内部任一深度的剪应力幅值与速度幅值满足 |τ_xy| ~ ρ·cs·|v|
     量级（用于核查 σ 相对 v 的绝对标度，而非仅相对比例）。
运行：在含 numpy 的普通 Python 环境下 `python freefield_selfcheck_v1.py`。
"""

import os  # 导入操作系统接口
import sys  # 导入系统模块（用于路径注入）
import math  # 导入数学模块
import importlib  # 导入动态导入模块
import numpy as np  # 导入数值计算库


def _load_engine():  # 定义加载 v6 自由场引擎的函数
    """从 v6 脚本动态导入纯数值自由场函数（屏蔽 Abaqus 依赖，强制走纯数值分支）。"""  # 说明函数用途
    # 注意：本机的 abaqus 包是"可导入"的，一旦导入就会拉起 CAE 会话；
    # 因此在导入 v6 前先把 Abaqus 相关模块在 sys.modules 中置为 None，
    # 使 v6 顶部的 `from abaqus import *` 抛出 ImportError 并落入其容错分支。
    for _m in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 遍历需屏蔽的模块
        sys.modules[_m] = None  # 置为 None 使其导入抛出 ImportError
    here = os.path.dirname(os.path.abspath(__file__))  # 取当前脚本所在目录
    if here not in sys.path:  # 判断目录是否已在搜索路径中
        sys.path.insert(0, here)  # 将目录加入模块搜索路径
    mod = importlib.import_module('VAB_oblique_TAF_double_v6')  # 动态导入 v6 模块
    return mod  # 返回模块对象


def _make_ricker(dt, n, f0, t0):  # 定义 Ricker 子波生成函数
    """生成 Ricker 子波作为输入速度时程（零均值、紧支、便于检验）。"""  # 说明函数用途
    t = np.arange(n) * dt  # 构造时间轴
    arg = (math.pi * f0 * (t - t0)) ** 2  # 计算 Ricker 自变量
    v = (1.0 - 2.0 * arg) * np.exp(-arg)  # 计算 Ricker 波形
    return t, v  # 返回时间轴与波形


def _rfft_inputs(vel, dt):  # 定义输入速度的频域准备函数
    """对速度时程补零到 2 的幂并做 rfft，返回引擎所需的频域输入。"""  # 说明函数用途
    n_orig = len(vel)  # 原始长度
    n_fft = 1  # 初始化 FFT 长度
    while n_fft < n_orig:  # 寻找不小于原长的 2 的幂
        n_fft *= 2  # 倍增
    n_fft *= 2  # 再翻倍以避免时域混叠（与主程序一致）
    vel_pad = np.zeros(n_fft)  # 创建补零数组
    vel_pad[:n_orig] = vel  # 填入原始速度
    freq_arr = np.fft.rfftfreq(n_fft, d=dt)  # 正频率数组
    vel_freq = np.fft.rfft(vel_pad)  # 速度频谱
    return vel_freq, freq_arr, n_fft, n_orig  # 返回频域输入与长度信息


def run_selfcheck():  # 定义自检主函数
    """执行均匀半空间退化算例并打印各项物理一致性判据。"""  # 说明函数用途
    eng = _load_engine()  # 加载引擎模块

    # ---- 材料：覆盖层与基岩取同一材料，构成均匀半空间 ----
    cs = 2000.0  # 剪切波速 (m/s)
    vv = 0.3  # 泊松比
    rho = 2500.0  # 密度 (kg/m^3)
    mat = eng._compute_material_params(cs, vv, rho)  # 计算材料参数字典
    cp = mat['cp']  # 读取纵波波速
    print('材料: cs=%.1f, cp=%.1f, rho=%.1f, G=%.3e' % (cs, cp, rho, mat['GG']))  # 打印材料信息

    # ---- 几何：底部 y=0，基岩厚 200，覆盖层厚 200，自由面 y=400 ----
    y_bottom = 0.0  # 模型底部 y 坐标
    h_bedrock = 200.0  # 基岩层厚度
    h_overlying = 200.0  # 覆盖层厚度
    y_surface = y_bottom + h_bedrock + h_overlying  # 自由面 y 坐标
    y_mid = y_bottom + h_bedrock  # 取界面深度作为内部检验点

    # ---- 输入：Ricker 速度子波 ----
    dt = 0.002  # 时间步长 (s)
    n = 2000  # 采样点数
    f0 = 2.0  # Ricker 主频 (Hz)
    t0 = 1.0  # Ricker 峰值时刻 (s)
    t_arr, vel_in = _make_ricker(dt, n, f0, t0)  # 生成输入速度时程
    vel_freq, freq_arr, n_fft, n_orig = _rfft_inputs(vel_in, dt)  # 频域准备
    v_in_peak = np.max(np.abs(vel_in))  # 输入速度峰值

    # ===================== 检验 A：垂直入射 =====================
    print('\n========== A. 垂直入射（angle=0）均匀半空间 ==========')  # 打印分节标题
    angle = 1e-10  # 用极小角度近似垂直入射（与主程序一致）
    alpha1 = math.radians(angle)  # 转弧度
    p_horiz = math.sin(alpha1) / cs  # 水平慢度（≈0）

    # 地表节点响应
    ff_top = eng._compute_freefield_at_node(  # 计算自由面节点自由场
        y_target=y_surface, x_target=0.0,  # 目标点坐标
        mat_bedrock=mat, mat_overlying=mat,  # 同材料
        h_bedrock=h_bedrock, h_overlying=h_overlying,  # 层厚
        y_bottom=y_bottom, p_horiz=p_horiz,  # 底部与慢度
        vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)  # 频域输入

    vx_top = ff_top['dotux'][:n_orig]  # 地表水平速度时程
    vy_top = ff_top['dotuy'][:n_orig]  # 地表竖向速度时程
    syy_top = ff_top['syy'][:n_orig]  # 地表 σ_yy 时程
    sxy_top = ff_top['sxy'][:n_orig]  # 地表 τ_xy 时程

    amp_ratio = np.max(np.abs(vx_top)) / v_in_peak  # 地表水平速度放大比
    stress_scale = mat['GG'] / cs * v_in_peak  # 应力的特征量级 ρ*cs*v = G/cs*v
    print('地表水平速度放大比 (应≈2.0):      %.4f' % amp_ratio)  # 打印放大比
    print('地表竖向速度/水平速度 (应≈0):      %.3e' % (np.max(np.abs(vy_top)) / np.max(np.abs(vx_top))))  # 打印竖向占比
    print('地表 |σ_yy| / 特征应力 (应≈0):     %.3e' % (np.max(np.abs(syy_top)) / stress_scale))  # 打印自由面法向应力
    print('地表 |τ_xy| / 特征应力 (应≈0):     %.3e' % (np.max(np.abs(sxy_top)) / stress_scale))  # 打印自由面剪应力

    # 内部点阻抗量级核查（应力标度是否与速度自洽）
    ff_mid = eng._compute_freefield_at_node(  # 计算内部点自由场
        y_target=y_mid, x_target=0.0,
        mat_bedrock=mat, mat_overlying=mat,
        h_bedrock=h_bedrock, h_overlying=h_overlying,
        y_bottom=y_bottom, p_horiz=p_horiz,
        vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)
    vx_mid = ff_mid['dotux'][:n_orig]  # 内部点水平速度
    sxy_mid = ff_mid['sxy'][:n_orig]  # 内部点剪应力
    imp_ratio = np.max(np.abs(sxy_mid)) / (rho * cs * np.max(np.abs(vx_mid)))  # 剪应力/（ρcs·速度）量级比
    print('内部点 |τ_xy| / (ρ·cs·|v_x|) 量级 (O(1) 表示标度自洽): %.4f' % imp_ratio)  # 打印阻抗量级比

    # ===================== 检验 B：斜入射 =====================
    print('\n========== B. 斜入射（angle=15）均匀半空间 ==========')  # 打印分节标题
    angle_b = 15.0  # 斜入射角
    p_b = math.sin(math.radians(angle_b)) / cs  # 斜入射水平慢度
    ff_top_b = eng._compute_freefield_at_node(  # 计算斜入射地表响应
        y_target=y_surface, x_target=0.0,
        mat_bedrock=mat, mat_overlying=mat,
        h_bedrock=h_bedrock, h_overlying=h_overlying,
        y_bottom=y_bottom, p_horiz=p_b,
        vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)
    syy_tb = ff_top_b['syy'][:n_orig]  # 斜入射地表 σ_yy
    sxy_tb = ff_top_b['sxy'][:n_orig]  # 斜入射地表 τ_xy
    finite_ok = np.all(np.isfinite(ff_top_b['dotux'])) and np.all(np.isfinite(syy_tb))  # 是否全部有限
    print('地表 |σ_yy| / 特征应力 (应≈0):     %.3e' % (np.max(np.abs(syy_tb)) / stress_scale))  # 打印自由面法向应力
    print('地表 |τ_xy| / 特征应力 (应≈0):     %.3e' % (np.max(np.abs(sxy_tb)) / stress_scale))  # 打印自由面剪应力
    print('斜入射结果全部有限 (应 True):      %s' % finite_ok)  # 打印有限性

    # ===================== 判定 =====================
    print('\n========== 判定 ==========')  # 打印判定标题
    ok_amp = abs(amp_ratio - 2.0) < 0.05  # 放大比是否接近 2
    ok_vy = (np.max(np.abs(vy_top)) / np.max(np.abs(vx_top))) < 1e-3  # 竖向是否近零
    ok_free = (np.max(np.abs(syy_top)) / stress_scale < 1e-3) and \
              (np.max(np.abs(sxy_top)) / stress_scale < 1e-3)  # 自由面应力是否近零
    print('A1 地表放大=2:        %s' % ('通过' if ok_amp else '不通过'))  # 打印放大判定
    print('A2 竖向≈0:           %s' % ('通过' if ok_vy else '不通过'))  # 打印竖向判定
    print('A3 自由面应力≈0:      %s' % ('通过' if ok_free else '不通过'))  # 打印自由面判定
    print('B  斜入射自由面/有限:  %s' % ('通过' if finite_ok else '不通过'))  # 打印斜入射判定


if __name__ == '__main__':  # 判断是否直接运行
    run_selfcheck()  # 调用自检主函数
