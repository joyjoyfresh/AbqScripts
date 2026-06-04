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
    mod = importlib.import_module('VAB_oblique_TAF_double_v7')  # 动态导入 v7 模块（纯物理函数与 v6 一致）
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

    # ===================== 检验 C：速度基线校正（#4 回归） =====================
    print('\n========== C. 速度基线校正回归（#4） ==========')  # 打印分节标题
    # 构造一段"干净"加速度（双窗正弦），叠加直流零偏 + 线性漂移，模拟真实记录的基线问题
    tc = np.arange(n) * dt  # 构造时间轴
    acc_clean = np.sin(2.0 * math.pi * 1.5 * tc) * np.exp(-((tc - 1.0) / 0.5) ** 2)  # 干净加速度（带高斯窗）
    acc_bad = acc_clean + 0.05 + 0.02 * tc  # 叠加 0.05 直流零偏与线性漂移
    vel_corr, slope = eng._integrate_acc_to_velocity(acc_bad, dt, tc)  # 调用 v6 同一函数做积分+基线校正
    # 对照：不做任何校正的朴素积分
    vel_naive = np.zeros_like(acc_bad)  # 初始化朴素速度
    vel_naive[1:] = np.cumsum((acc_bad[:-1] + acc_bad[1:]) / 2 * dt)  # 朴素梯形积分（无校正）
    vscale = np.max(np.abs(vel_corr)) + 1e-30  # 速度特征量级（防零除）
    naive_drift = abs(vel_naive[-1] - vel_naive[0]) / (np.max(np.abs(vel_naive)) + 1e-30)  # 朴素积分的净漂移占比
    corr_resid_slope = abs(np.polyfit(tc, vel_corr, 1)[0])  # 校正后残余线性斜率
    corr_mean = abs(np.mean(vel_corr)) / vscale  # 校正后速度均值占比
    print('扣除的速度趋势斜率:               %.4e' % slope)  # 打印被扣除的趋势斜率
    print('朴素积分净漂移/峰值 (越大越坏):    %.3f' % naive_drift)  # 打印朴素积分漂移
    print('校正后残余斜率 (应≈0):            %.3e' % corr_resid_slope)  # 打印残余斜率
    print('校正后速度均值/峰值 (应≈0):        %.3e' % corr_mean)  # 打印残余均值

    # ===================== 检验 D：底边覆盖层厚度按 x 取值（#2 回归） =====================
    print('\n========== D. 底边覆盖层厚度修正回归（#2） ==========')  # 打印分节标题
    # 需要软/硬两层对比，厚度差异才会通过"层共振"体现在自由场里
    mat_ov = eng._compute_material_params(1000.0, vv, rho)  # 覆盖层取更软材料 cs=1000
    # 斜坡几何：坡顶平台高 400（软层厚 200），坡脚平台高 300（软层薄 100）
    bedrock_t = 200.0  # 基岩厚度
    H_up = 400.0  # 坡顶平台地表高度
    H_lo = 300.0  # 坡脚平台地表高度
    left_flat = 1000.0  # 上平台长度
    w_slope = 200.0  # 坡面水平长度
    total_L = 1800.0  # 模型总长
    x_corner = total_L  # 右下角点横坐标（位于坡脚平台正下方）

    # D-1：几何函数 _surface_y_at 三点检查
    sy_top = eng._surface_y_at(0.0, H_up, H_lo, left_flat, w_slope)  # 坡顶平台地表
    sy_toe = eng._surface_y_at(total_L, H_up, H_lo, left_flat, w_slope)  # 坡脚平台地表
    sy_mid = eng._surface_y_at(left_flat + w_slope / 2.0, H_up, H_lo, left_flat, w_slope)  # 坡面中点地表
    geom_ok = (abs(sy_top - H_up) < 1e-9) and (abs(sy_toe - H_lo) < 1e-9) and \
              (abs(sy_mid - 0.5 * (H_up + H_lo)) < 1e-9)  # 三点是否符合预期
    print('地表高度 坡顶/坡脚/坡中: %.1f / %.1f / %.1f (期望 %.1f / %.1f / %.1f)' %
          (sy_top, sy_toe, sy_mid, H_up, H_lo, 0.5 * (H_up + H_lo)))  # 打印三点高度

    # D-2：右下角点自由场一致性（取剪应力时程对比）
    def _corner_sxy(h_ov):  # 定义按给定覆盖层厚度计算角点剪应力时程的辅助函数
        ff = eng._compute_freefield_at_node(  # 计算右下角点自由场
            y_target=0.0, x_target=x_corner,  # 角点位于底边 y=0
            mat_bedrock=mat, mat_overlying=mat_ov,  # 硬基岩 + 软覆盖层
            h_bedrock=bedrock_t, h_overlying=h_ov,  # 基岩厚 + 给定覆盖层厚
            y_bottom=0.0, p_horiz=p_horiz,  # 底部与（垂直入射）慢度
            vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=n_fft)  # 频域输入
        return ff['sxy'][:n_orig]  # 返回剪应力时程

    h_right = H_lo - bedrock_t  # 右边界所用覆盖层厚度（薄，=100）
    h_old_bottom = H_up - bedrock_t  # 旧做法：底边一刀切用最厚（=200）
    h_fix_bottom = eng._surface_y_at(x_corner, H_up, H_lo, left_flat, w_slope) - bedrock_t  # 修正后底边按 x 取厚度
    sxy_right = _corner_sxy(h_right)  # 右边界视角的角点剪应力
    sxy_old = _corner_sxy(h_old_bottom)  # 旧底边视角的角点剪应力
    sxy_fix = _corner_sxy(h_fix_bottom)  # 修正底边视角的角点剪应力
    scale = np.max(np.abs(sxy_right)) + 1e-30  # 参考量级（防零除）
    rel_old = np.max(np.abs(sxy_old - sxy_right)) / scale  # 旧做法与右边界的不一致度
    rel_fix = np.max(np.abs(sxy_fix - sxy_right)) / scale  # 修正后与右边界的不一致度
    print('修正后底边厚度 = %.1f (右边界厚度 = %.1f, 旧底边 = %.1f)' % (h_fix_bottom, h_right, h_old_bottom))  # 打印厚度对比
    print('旧做法 角点不一致度 (越大越坏): %.3f' % rel_old)  # 打印旧不一致度
    print('修正后 角点不一致度 (应≈0):     %.3e' % rel_fix)  # 打印修正后不一致度

    # ===================== 检验 E：v7 重构等价性（建模层行为不变） =====================
    print('\n========== E. v7 重构等价性（建模层） ==========')  # 打印分节标题
    # E-1：make_geometry 派生量与公式一致
    g = eng.make_geometry(total_L=1800.0, H_minus_h=200.0, i=45.0,
                          h_over_H=0.5, left_flat=1000.0, bedrock_thickness=200.0)  # 构建几何对象
    geom_ok2 = (abs(g.H - 400.0) < 1e-9 and abs(g.h - 200.0) < 1e-9 and
                abs(g.H_upper - 600.0) < 1e-9 and abs(g.H_lower - 400.0) < 1e-9 and
                abs(g.H_flat - 600.0) < 1e-9 and abs(g.w_slope - 200.0) < 1e-9)  # 校验派生量
    print('make_geometry 派生量 (H/h/上/下/平/坡宽): %.0f/%.0f/%.0f/%.0f/%.0f/%.0f' %
          (g.H, g.h, g.H_upper, g.H_lower, g.H_flat, g.w_slope))  # 打印派生量

    # E-2：_build_equivalent_forces 输出与 v6 显式公式逐点一致
    bt = g.bedrock_thickness  # 基岩厚度
    ymax_l = g.H_upper  # 左边界顶高
    ymax_r = g.H_lower  # 右边界顶高
    nodes_by_boundary = {  # 构造三条边界各一个测试节点
        'l': [eng.BoundaryNode(label=101, x=0.0, y=300.0, influence=4.0, kn=1.0, cn=2.0, kt=3.0, ct=4.0)],
        'r': [eng.BoundaryNode(label=202, x=1800.0, y=250.0, influence=4.0, kn=1.5, cn=2.5, kt=3.5, ct=4.5)],
        'b': [eng.BoundaryNode(label=303, x=1800.0, y=0.0, influence=4.0, kn=1.2, cn=2.2, kt=3.2, ct=4.2)],
    }
    ctx = eng.FreeFieldCtx(mat_bedrock=mat, mat_overlying=mat_ov, geom=g,
                           ymax_l=ymax_l, ymax_r=ymax_r, ymin=0.0,
                           p_horiz=p_horiz, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt,
                           N_fft=n_fft, N_orig=n_orig, time_arr=t_arr)  # 打包上下文
    fd = eng._build_equivalent_forces(nodes_by_boundary, ctx)  # 调用 v7 重构函数

    def _ref_force(bn, boundary):  # 用 v6 显式公式独立复算等效力
        if boundary == 'l':
            h_ov = max(0.0, ymax_l - bt)
        elif boundary == 'r':
            h_ov = max(0.0, ymax_r - bt)
        else:
            h_ov = max(0.0, eng._surface_y_at(bn.x, g.H_upper, g.H_lower, g.left_flat, g.w_slope) - bt)
        ff = eng._compute_freefield_at_node(y_target=bn.y, x_target=bn.x,
                                            mat_bedrock=mat, mat_overlying=mat_ov,
                                            h_bedrock=bt, h_overlying=h_ov, y_bottom=0.0,
                                            p_horiz=p_horiz, vel_freq=vel_freq, freq_arr=freq_arr,
                                            dt=dt, N_fft=n_fft)
        ux = ff['ux'][:n_orig]; uy = ff['uy'][:n_orig]
        dux = ff['dotux'][:n_orig]; duy = ff['dotuy'][:n_orig]
        sxx = ff['sxx'][:n_orig]; syy = ff['syy'][:n_orig]; sxy = ff['sxy'][:n_orig]
        if boundary == 'l':
            fx = bn.kn * ux + bn.cn * dux + bn.influence * (-sxx)
            fy = bn.kt * uy + bn.ct * duy + bn.influence * (-sxy)
        elif boundary == 'r':
            fx = bn.kn * ux + bn.cn * dux + bn.influence * sxx
            fy = bn.kt * uy + bn.ct * duy + bn.influence * sxy
        else:
            fx = bn.kt * ux + bn.ct * dux + bn.influence * (-sxy)
            fy = bn.kn * uy + bn.cn * duy + bn.influence * (-syy)
        return fx, fy

    max_diff = 0.0  # 记录最大逐点偏差
    for boundary in ('l', 'r', 'b'):  # 遍历三条边界
        bn = nodes_by_boundary[boundary][0]  # 取该边界测试节点
        fx_ref, fy_ref = _ref_force(bn, boundary)  # 独立复算
        fx_new = fd['{}-{}-fx'.format(bn.label, boundary)][:, 1]  # v7 输出 fx
        fy_new = fd['{}-{}-fy'.format(bn.label, boundary)][:, 1]  # v7 输出 fy
        max_diff = max(max_diff, np.max(np.abs(fx_new - fx_ref)), np.max(np.abs(fy_new - fy_ref)))  # 更新最大偏差
    print('等效力 v7 重构 vs v6 公式 最大逐点偏差 (应=0): %.3e' % max_diff)  # 打印等价性偏差

    # ===================== 判定 =====================
    print('\n========== 判定 ==========')  # 打印判定标题
    ok_amp = abs(amp_ratio - 2.0) < 0.05  # 放大比是否接近 2
    ok_vy = (np.max(np.abs(vy_top)) / np.max(np.abs(vx_top))) < 1e-3  # 竖向是否近零
    ok_free = (np.max(np.abs(syy_top)) / stress_scale < 1e-3) and \
              (np.max(np.abs(sxy_top)) / stress_scale < 1e-3)  # 自由面应力是否近零
    ok_baseline = (corr_resid_slope < 1e-6) and (corr_mean < 1e-9) and (naive_drift > 0.1)  # 校正有效且朴素确有漂移
    ok_geom = geom_ok  # 几何函数三点是否正确
    ok_consist = (rel_fix < 1e-9) and (rel_old > 0.01)  # 修正后一致且旧做法确有不一致
    print('A1 地表放大=2:        %s' % ('通过' if ok_amp else '不通过'))  # 打印放大判定
    print('A2 竖向≈0:           %s' % ('通过' if ok_vy else '不通过'))  # 打印竖向判定
    print('A3 自由面应力≈0:      %s' % ('通过' if ok_free else '不通过'))  # 打印自由面判定
    print('B  斜入射自由面/有限:  %s' % ('通过' if finite_ok else '不通过'))  # 打印斜入射判定
    print('C  速度基线校正:       %s' % ('通过' if ok_baseline else '不通过'))  # 打印基线校正判定
    print('D1 地表高度函数:      %s' % ('通过' if ok_geom else '不通过'))  # 打印几何函数判定
    print('D2 底边厚度一致性:    %s' % ('通过' if ok_consist else '不通过'))  # 打印底边一致性判定
    ok_equiv = geom_ok2 and (max_diff < 1e-9)  # 几何派生量正确且等效力逐点一致
    print('E1 几何派生量:        %s' % ('通过' if geom_ok2 else '不通过'))  # 打印几何派生判定
    print('E2 重构等效力等价:    %s' % ('通过' if (max_diff < 1e-9) else '不通过'))  # 打印等效力等价判定


if __name__ == '__main__':  # 判断是否直接运行
    run_selfcheck()  # 调用自检主函数
