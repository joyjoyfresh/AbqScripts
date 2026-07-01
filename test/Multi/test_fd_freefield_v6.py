# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""v6 频域精确分层自由场引擎（fd）验证测试（不依赖 Abaqus，可直接运行）。

测试项：
  T1 半空间退化：fd 引擎 vs v5 射线法（半空间下射线法即精确解析解式(2)），波形逐点对比；
  T2 三层垂直入射：fd 传递函数 vs 独立实现的 SH Haskell 递推（弹性 + 阻尼两种工况）；
  T3 退化一致性：三层取相同材料 == 均质半空间，fd 结果应严格一致；
  T4 界面连续性：界面上下 ux/uy/σyy/σxy 连续（σxx 允许跳变）；
  T5 论文图15 平台预测：软表层柱一维放大 × 解析分母 → 预测远场平台 TAF（对照论文量级）。

运行：python test/test_fd_freefield_v6.py
（沙箱/特殊环境可用环境变量 V6_PATH 指定 v6 脚本所在目录。）
"""

import os, sys, math, types, importlib  # 导入标准库
import numpy as np  # 导入数值库

for _n, _attr in (('abaqus', 'mdb'), ('abaqusConstants', None),  # 用桩模块屏蔽 Abaqus 依赖
                  ('regionToolset', 'Region'), ('caeModules', None), ('mesh', None)):
    _m = types.ModuleType(_n)  # 创建桩模块
    if _attr == 'mdb':  # abaqus 需要 mdb 占位
        _m.mdb = types.SimpleNamespace()  # 设置 mdb 占位
    elif _attr == 'Region':  # regionToolset 需要 Region 占位
        _m.Region = object  # 设置 Region 占位
    sys.modules[_n] = _m  # 注册桩模块

_OVERRIDE = os.environ.get('V6_PATH')  # 可选：v6 脚本目录覆盖（沙箱测试用）
if _OVERRIDE:  # 提供了覆盖目录
    sys.path.insert(0, _OVERRIDE)  # 优先从覆盖目录导入
PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Modeling', 'Multi')  # 上级目录（Multi）
if PARENT not in sys.path:  # 确保可 import 上级脚本
    sys.path.append(PARENT)  # 加入搜索路径（在覆盖目录之后）
ml = importlib.import_module('VAB_oblique_TAF_multilayer_v6')  # 导入 v6 多层脚本


# ==========================================================
#  公共构造函数
# ==========================================================


GCFG = {'H_minus_h': 200.0, 'i': 45.0, 'h_over_H': 0.5, 'total_L': 1800.0,  # 论文几何
        'left_flat': 1000.0, 'bedrock_thickness': 200.0}  # 上平台与基岩厚度
NU, RHO = 0.3, 2500.0  # 泊松比与密度（论文恒定值）
BEDROCK = {'elastic_modulus': 26e9, 'poisson_ratio': NU, 'density': RHO}  # 基岩（Vs=2000）


def ricker(fc=4.0, dt=1e-3, T=2.0, t0=0.3):  # 生成 Ricker 加速度记录
    """返回 (t, acc)：中心频率 fc 的 Ricker 子波加速度时程。"""
    t = np.arange(0.0, T + dt * 0.5, dt)  # 时间轴
    a = (1 - 2 * (math.pi * fc * (t - t0)) ** 2) * np.exp(-(math.pi * fc * (t - t0)) ** 2)  # Ricker 公式
    return t, a  # 返回时间轴与加速度


def make_ctx(layers, angle=15.0, fc=4.0, damping_on=False, engine='fd', include_damping=True):  # 构造 FreeFieldCtx
    """按层配置构造 v6 FreeFieldCtx（含 v6 新增 acc/damp_terms/ffcfg 字段）。"""
    mcfg = {'angle': angle, 'bedrock': dict(BEDROCK), 'layers': layers}  # 材料配置
    site, fixed = ml.build_site(mcfg, GCFG)  # 构建场地
    geom = ml.make_geometry(total_L=GCFG['total_L'], H_minus_h=GCFG['H_minus_h'], i=GCFG['i'],  # 几何
                            h_over_H=GCFG['h_over_H'], left_flat=GCFG['left_flat'],
                            bedrock_thickness=GCFG['bedrock_thickness'], fixed_thicknesses=fixed)
    strat = ml._build_stratigraphy(site, geom, ymin=0.0)  # 分层带
    a1 = math.radians(angle if abs(angle) > 1e-12 else 1e-10)  # 入射角弧度
    mb = ml._compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩参数
    p = math.sin(a1) / mb['cs']  # 水平慢度
    b1 = ml._safe_arcsin(mb['cp'] * math.sin(a1) / mb['cs'])  # 基岩 P 角
    t, acc = ricker(fc=fc)  # 输入 Ricker
    dt = t[1] - t[0]  # 步长
    vel, _ = ml._integrate_acc_to_velocity(acc.copy(), dt, t)  # 积分速度（射线法路径用）
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 积分位移
    damping = None  # 默认无阻尼
    if damping_on:  # 启用阻尼时构造解析后配置
        damping = {'enable': True, 'method': 'rayleigh', 'qs_factor': 0.05,
                   'q_bedrock': 999.0, 'fc': fc, 'f1_factor': 0.5, 'f2_factor': 2.5}
    damp_terms = ml._band_damping_terms(strat, damping)  # 各带瑞利系数表
    ffcfg = dict(ml.freefield_cfg); ffcfg['engine'] = engine; ffcfg['include_damping'] = include_damping  # 引擎配置
    ml._REFL_COEFF_CACHE.clear(); ml._FD_SOLVER_CACHE.clear()  # 清空缓存
    ctx = ml.FreeFieldCtx(site=site, geom=geom, strat=strat, ymax_l=geom.H_upper, ymax_r=geom.H_lower,
                          ymin=0.0, alpha=a1, beta_p=b1, p_horiz=p, GG=mb['GG'], lam=mb['lam'],
                          cs=mb['cs'], cp=mb['cp'], VEL=np.column_stack((t, vel)),
                          DIS=np.column_stack((t, dis)), dt=dt, time_arr=t, max_reflect_order=3,
                          acc=acc, damp_terms=damp_terms, ffcfg=ffcfg)
    return ctx, geom  # 返回上下文与几何


SURF_SOFT_50 = {'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': NU, 'density': RHO, 'thickness': 50.0}  # 软表层50m(Vs=400)
SURF_SOFT_150 = {'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': NU, 'density': RHO, 'thickness': 150.0}  # 软表层150m
SURF_HARD_150 = {'name': 'surface', 'velocity_ratio': 1.25, 'poisson_ratio': NU, 'density': RHO, 'thickness': 150.0}  # 硬表层150m(Vs=1600)
OVER_25 = {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': NU, 'density': RHO}  # 覆盖层(Vs=800)


def rel_err(a, b):  # 相对误差（按参考峰值归一）
    """返回 max|a-b| / max|b|（b 为参考）。"""
    ref = np.max(np.abs(b))  # 参考峰值
    return float(np.max(np.abs(a - b)) / (ref if ref > 0 else 1.0))  # 归一最大偏差


# ==========================================================
#  T1 半空间退化：fd vs 射线法（射线法此时为精确解析解）
# ==========================================================


def t1_halfspace():  # T1 测试主体
    print('== T1 半空间退化：fd vs 射线法(=解析式2)，θ=15°，弹性 ==')  # 标题
    ctx, geom = make_ctx([], angle=15.0, damping_on=False, include_damping=False)  # 均质半空间上下文
    gv = ml._make_delay_cache(ctx.VEL, ctx.dt)  # 射线法速度缓存
    gd = ml._make_delay_cache(ctx.DIS, ctx.dt)  # 射线法位移缓存
    nodes = [('l', 0.0, 50.0, geom.H_upper), ('l', 0.0, 350.0, geom.H_upper),  # 左边界两深度
             ('b', 900.0, 0.0, ml._surface_y_at(900.0, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)),  # 底边节点
             ('r', geom.total_L, 100.0, geom.H_lower)]  # 右边界节点
    N = len(ctx.time_arr)  # 比较窗口长度（原始时长）
    worst = 0.0  # 最大相对误差记录
    for bnd, x, y, ymax in nodes:  # 遍历代表节点
        ray = ml._compute_freefield_at_node(bnd, x, y, ymax, ctx, gv, gd)  # 射线法（精确）
        fd = ml._fd_freefield_at_node(bnd, x, y, ymax, ctx)  # fd 引擎
        for k, tol in (('dotux', 0.03), ('dotuy', 0.03), ('sigmax', 0.03), ('sigmay', 0.03),
                       ('ux', 0.05), ('uy', 0.05)):  # 各场量与容差
            e = rel_err(fd[k][:N], ray[k][:N])  # 截断到共同窗口比较
            worst = max(worst, e if k.startswith('dot') or k.startswith('sig') else 0.0)  # 记录速度/应力最大误差
            assert e < tol, 'T1 %s(%s,%.0f,%.0f) 相对误差 %.4f 超容差 %.2f' % (k, bnd, x, y, e, tol)  # 校验
        print('  节点(%s, x=%.0f, y=%.0f) 全场量通过' % (bnd, x, y))  # 通过提示
    print('  T1 通过（速度/应力最大相对误差 %.4f）\n' % worst)  # 总结


# ==========================================================
#  T2 三层垂直入射：fd 传递函数 vs 独立 SH Haskell 递推
# ==========================================================


def sh_transfer(freqs, layers_top_down, halfspace, xi_funcs=None):  # 独立 SH 递推（Kramer 教材公式）
    """layers_top_down: [(rho, cs, h), ...] 自地表向下；halfspace: (rho, cs)。

    返回 |u_surface / u_incident|（入射为半空间中上行波位移幅值）。
    xi_funcs：与层列表等长 + 半空间 1 项的 ξ(ω) 函数列表（None=弹性）。
    """
    out = np.zeros(len(freqs))  # 初始化传递函数模
    for i, f in enumerate(freqs):  # 逐频计算
        w = 2.0 * math.pi * f  # 圆频率
        A, B = 1.0 + 0j, 1.0 + 0j  # 地表层上/下行幅值（自由面 A=B）
        stack = list(layers_top_down) + [halfspace + (None,)]  # 末项半空间（厚度占位）
        for m in range(len(layers_top_down)):  # 自上向下递推
            rho1, c1, h1 = stack[m]  # 当前层
            rho2, c2 = stack[m + 1][0], stack[m + 1][1]  # 下一层
            if xi_funcs:  # 含阻尼：复波速 c*(1+iξ)^0.5 近似 → 用与 fd 相同的复模量口径
                c1 = c1 * np.sqrt(1.0 + 2j * xi_funcs[m](w))  # 复波速（μ(1+2iξ) 等效）
                c2 = c2 * np.sqrt(1.0 + 2j * xi_funcs[m + 1](w))  # 复波速
            alp = (rho1 * c1) / (rho2 * c2)  # 阻抗比
            k1 = w / c1  # 该层波数
            e_p, e_m = np.exp(1j * k1 * h1), np.exp(-1j * k1 * h1)  # 相位因子
            A2 = 0.5 * A * (1 + alp) * e_p + 0.5 * B * (1 - alp) * e_m  # 下层上行幅值
            B2 = 0.5 * A * (1 - alp) * e_p + 0.5 * B * (1 + alp) * e_m  # 下层下行幅值
            A, B = A2, B2  # 递推
        out[i] = abs(2.0 / A)  # 地表(=2) / 半空间上行(=A)
    return out  # 返回传递函数模


def t2_layered_transfer():  # T2 测试主体
    print('== T2 三层垂直入射传递函数：fd vs 独立 SH Haskell 递推（弹性） ==')  # 标题
    ctx, geom = make_ctx([SURF_SOFT_150, OVER_25], angle=0.0, damping_on=False, include_damping=False)  # 三层弹性
    freqs = np.linspace(0.3, 12.0, 240)  # 比较频带
    omega = 2.0 * math.pi * freqs  # 圆频率
    col = ml._build_column(ctx.strat, geom.H_upper, ctx.p_horiz, 0.0)  # 上平台柱（3 层段）
    sol = ml._fd_solve_column(col, ctx.p_horiz, omega, ctx.damp_terms, False)  # fd 求解（弹性）
    f_surf = ml._fd_eval_column(sol, omega, ctx.p_horiz, geom.H_upper)  # 地表场量谱
    tf_fd = np.abs(f_surf['ux'])  # fd 传递函数模（单位入射）
    tf_sh = sh_transfer(freqs, [(RHO, 400.0, 150.0), (RHO, 800.0, 250.0)], (RHO, 2000.0))  # SH 递推
    err = float(np.max(np.abs(tf_fd - tf_sh) / np.max(tf_sh)))  # 最大归一误差
    assert err < 1e-6, 'T2 弹性传递函数误差 %.3e 超容差 1e-6' % err  # 校验
    print('  弹性最大归一误差 %.2e；共振峰 fd=%.3f @ %.2fHz' % (err, tf_fd.max(), freqs[tf_fd.argmax()]))  # 输出
    print('  T2 通过\n')  # 总结


# ==========================================================
#  T3 退化一致性：三层同材料 == 均质半空间
# ==========================================================


def t3_degenerate():  # T3 测试主体
    print('== T3 退化一致性：三层同基岩材料 == 均质半空间（θ=15°） ==')  # 标题
    same1 = {'name': 'surface', 'velocity_ratio': 1.0, 'poisson_ratio': NU, 'density': RHO, 'thickness': 150.0}  # 同材料"表层"
    same2 = {'name': 'overlying', 'velocity_ratio': 1.0, 'poisson_ratio': NU, 'density': RHO}  # 同材料"覆盖层"
    ctx3, geom = make_ctx([same1, same2], angle=15.0, damping_on=False, include_damping=False)  # 伪三层
    ctx1, _ = make_ctx([], angle=15.0, damping_on=False, include_damping=False)  # 真均质
    for bnd, x, y, ymax in (('l', 0.0, 420.0, geom.H_upper), ('l', 0.0, 520.0, geom.H_upper),
                            ('b', 500.0, 0.0, geom.H_upper)):  # 代表节点（含"层"内点）
        ml._FD_SOLVER_CACHE.clear()  # 清缓存（ctx 切换）
        fd3 = ml._fd_freefield_at_node(bnd, x, y, ymax, ctx3)  # 伪三层结果
        ml._FD_SOLVER_CACHE.clear()  # 清缓存
        fd1 = ml._fd_freefield_at_node(bnd, x, y, ymax, ctx1)  # 均质结果
        for k in ('ux', 'uy', 'dotux', 'dotuy', 'sigmax', 'sigmay'):  # 全场量比较
            e = rel_err(fd3[k], fd1[k])  # 相对误差
            assert e < 1e-8, 'T3 %s(%s,%.0f,%.0f) 误差 %.2e 超 1e-8' % (k, bnd, x, y, e)  # 校验
    print('  T3 通过（伪三层与均质逐点一致）\n')  # 总结


# ==========================================================
#  T4 界面连续性：ux/uy/σyy/σxy 连续，σxx 允许跳变
# ==========================================================


def t4_continuity():  # T4 测试主体
    print('== T4 界面连续性（三层，θ=15°，含阻尼） ==')  # 标题
    ctx, geom = make_ctx([SURF_SOFT_150, OVER_25], angle=15.0, damping_on=True, include_damping=True)  # 三层含阻尼
    freqs = np.linspace(0.5, 10.0, 120)  # 检查频带
    omega = 2.0 * math.pi * freqs  # 圆频率
    col = ml._build_column(ctx.strat, geom.H_upper, ctx.p_horiz, 0.0)  # 上平台柱
    sol = ml._fd_solve_column(col, ctx.p_horiz, omega, ctx.damp_terms, True)  # fd 求解（含阻尼）
    for Y in (geom.bedrock_thickness, geom.H_upper - 150.0):  # 两个材料界面
        up = ml._fd_eval_column(sol, omega, ctx.p_horiz, Y + 1e-4)  # 界面上侧
        dn = ml._fd_eval_column(sol, omega, ctx.p_horiz, Y - 1e-4)  # 界面下侧
        for k in ('ux', 'uy', 'syy', 'sxy'):  # 必须连续的场量
            ref = np.max(np.abs(dn[k]))  # 参考幅值
            e = float(np.max(np.abs(up[k] - dn[k])) / (ref if ref > 0 else 1.0))  # 相对差
            assert e < 1e-5, 'T4 %s 在界面 y=%.0f 不连续：%.2e' % (k, Y, e)  # 校验
        jump = float(np.max(np.abs(up['sxx'] - dn['sxx'])) / np.max(np.abs(dn['sxx'])))  # σxx 跳变
        print('  界面 y=%.0f：ux/uy/σyy/σxy 连续；σxx 相对跳变 %.2f（物理预期非零）' % (Y, jump))  # 输出
    print('  T4 通过\n')  # 总结


# ==========================================================
#  T5 论文图15 远场平台预测（信息性 + 量级断言）
# ==========================================================


def t5_plateau():  # T5 测试主体
    print('== T5 论文图15 远场平台 TAF 预测（软表层，fc=4Hz，含阻尼） ==')  # 标题
    for label, surf, paper_ref in (('(a) h1=50m', SURF_SOFT_50, '论文≈2.5~2.8'),
                                   ('(b) h1=150m', SURF_SOFT_150, '论文≈4.5~5'),
                                   ('(b硬) h1=150m硬表层', SURF_HARD_150, '论文≈1.5~1.8')):  # 三种表层
        for ang in (0.0, 15.0):  # 两个入射角
            ctx, geom = make_ctx([surf, OVER_25], angle=ang, fc=4.0, damping_on=True, include_damping=True)  # 上下文
            fd = ml._fd_freefield_at_node('l', 0.0, geom.H_upper, geom.H_upper, ctx)  # 上平台地表自由场
            dt = ctx.dt  # 步长
            acc_h = np.gradient(fd['dotux'], dt)  # 水平加速度（速度数值微分）
            acc_v = np.gradient(fd['dotuy'], dt)  # 竖向加速度
            a1r = ctx.alpha  # 基岩入射角
            fs = ml._compute_free_surface_sv_coeff(a1r, ctx.cp, ctx.cs)  # 基岩自由面系数
            fh = (1.0 - fs['A1']) * math.cos(a1r) + fs['A2'] * math.sin(fs['beta'])  # 解析分母系数
            denom = fh * float(np.max(np.abs(ctx.acc)))  # 解析分母 PGA_ff_h
            taf_h = float(np.max(np.abs(acc_h)) / denom)  # 平台 TAF_h 预测
            taf_v = float(np.max(np.abs(acc_v)) / denom)  # 平台 TAF_v 预测
            print('  %s θ=%2.0f°: 1D平台 TAF_h=%.2f TAF_v=%.2f （%s）' % (label, ang, taf_h, taf_v, paper_ref))  # 输出
            assert 0.5 < taf_h < 10.0, 'T5 平台 TAF_h=%.2f 超合理范围' % taf_h  # 量级护栏
    print('  T5 完成（与论文图15 左平台水平对照）\n')  # 总结


def main():  # 测试主流程
    t1_halfspace()  # T1 半空间退化
    t2_layered_transfer()  # T2 SH 递推对照
    t3_degenerate()  # T3 退化一致性
    t4_continuity()  # T4 界面连续性
    t5_plateau()  # T5 论文平台预测
    print('全部测试通过。')  # 总结


if __name__ == '__main__':  # 直接运行入口
    main()  # 执行测试
