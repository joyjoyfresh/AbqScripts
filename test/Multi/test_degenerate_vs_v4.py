# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""退化一致性测试：多层脚本在 M=1（双层）时，自由场逐节点结果应与已验证的 v4 完全一致。

做法（不依赖 Abaqus）：
  - 用桩模块屏蔽 abaqus/abaqusConstants/regionToolset/caeModules/mesh 后，import 两个脚本；
  - 同一材料/几何/入射角/输入波，分别构造 v4 与 multilayer 的 FreeFieldCtx；
  - 取代表性边界节点（左/右侧的基岩与覆盖层节点、上平台与坡面下的底边节点），
    各自调用其 _compute_freefield_at_node，比较 ux/uy/dotux/dotuy/sigmax/sigmay；
  - 自由场引擎仅依赖 numpy/math，弹簧/阻尼/影响长度不参与，故差异只来自自由场本身。
期望：M=1 时 multilayer 退化为 v4 单腔，逐节点最大绝对差 ~ 机器精度（断言 < 1e-9）。
运行：python test/test_degenerate_vs_v4.py
"""

import os, sys, math, types, importlib  # 导入标准库
import numpy as np  # 导入数值库

# ============ 用桩模块屏蔽 Abaqus 依赖（自由场引擎不使用这些对象）============
_abq = types.ModuleType('abaqus')  # 创建 abaqus 桩模块
_abq.mdb = types.SimpleNamespace()  # 提供 mdb 占位（from abaqus import mdb）
sys.modules['abaqus'] = _abq  # 注册 abaqus 桩
sys.modules['abaqusConstants'] = types.ModuleType('abaqusConstants')  # 注册 abaqusConstants 桩
_rt = types.ModuleType('regionToolset')  # 创建 regionToolset 桩模块
_rt.Region = object  # 提供 Region 占位（from regionToolset import Region）
sys.modules['regionToolset'] = _rt  # 注册 regionToolset 桩
sys.modules['caeModules'] = types.ModuleType('caeModules')  # 注册 caeModules 桩
sys.modules['mesh'] = types.ModuleType('mesh')  # 注册 mesh 桩

PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Modeling', 'Multi')  # 上级目录（Multi）
if PARENT not in sys.path:  # 确保可 import 上级脚本
    sys.path.insert(0, PARENT)  # 加入搜索路径
v4 = importlib.import_module('VAB_oblique_TAF_double_v4')  # 导入已验证的 v4 脚本
ml = importlib.import_module('VAB_oblique_TAF_multilayer_v1')  # 导入新多层脚本


def _make_ricker(fp=5.0, dt=0.005, n=400):  # 构造 Ricker 子波作为加速度时程
    t = np.arange(n) * dt  # 时间轴
    t0 = 1.0 / fp  # 峰值时刻
    a = (1 - 2 * (math.pi * fp * (t - t0)) ** 2) * np.exp(-(math.pi * fp * (t - t0)) ** 2)  # Ricker 波形
    return np.column_stack((t, a))  # 返回 [t, acc]


def _build_v4_ctx(cs_b, cs_o, nu, rho, geom_cfg, angle, ACC):  # 构造 v4 的 FreeFieldCtx（复刻 v4.VAB_oblique）
    mat_b = v4._compute_material_params(cs_b, nu, rho)  # 基岩材料参数
    mat_o = v4._compute_material_params(cs_o, nu, rho)  # 覆盖层材料参数
    geom = v4.make_geometry(total_L=geom_cfg['total_L'], H_minus_h=geom_cfg['H_minus_h'],  # 斜坡几何
                            i=geom_cfg['i'], h_over_H=geom_cfg['h_over_H'],
                            left_flat=geom_cfg['left_flat'], bedrock_thickness=geom_cfg['bedrock_thickness'])
    angle = 1e-10 if angle == 0 else round(angle, 4)  # 入射角处理
    alpha1 = math.radians(angle)  # 基岩 SV 入射角
    cs1, cp1, cs2, cp2 = mat_b['cs'], mat_b['cp'], mat_o['cs'], mat_o['cp']  # 两层波速
    interface_12 = v4._compute_interface_sv_coeff(alpha1, mat_b, mat_o)  # 界面 1->2
    alpha2 = interface_12['alpha2']  # 覆盖层 SV 透射角
    beta1 = v4._safe_arcsin(cp1 * math.sin(alpha1) / cs1)  # 基岩 P 角
    beta2 = v4._safe_arcsin(cp2 * math.sin(alpha2) / cs2) if abs(math.sin(alpha2)) > 0 else 1e-10  # 覆盖层 P 角
    free_sv_2 = v4._compute_free_surface_sv_coeff(alpha2, cp2, cs2)  # 覆盖层自由面 SV 系数
    free_p_2 = v4._compute_free_surface_p_coeff(beta2, cp2, cs2)  # 覆盖层自由面 P 系数
    interface_21 = v4._compute_interface_sv_coeff(alpha2, mat_o, mat_b)  # 界面 2->1
    cycle_sv = free_sv_2['A1'] * interface_21['Rss']  # SV 混响幅值因子
    cycle_p = free_p_2['B2'] * interface_21['Rss']  # P 混响幅值因子
    oc = max(0, int(v4.MAX_REFLECT_ORDER))  # 截断阶数
    sum_sv = sum([cycle_sv ** k for k in range(oc + 1)])  # SV 级数和
    sum_p = sum([cycle_p ** k for k in range(oc + 1)])  # P 级数和
    Rss_eff = interface_12['Rss'] + interface_12['Tss'] * free_sv_2['A1'] * interface_21['Tss'] * sum_sv  # 等效反射
    Rsp_eff = interface_12['Rsp'] + interface_12['Tss'] * free_sv_2['A2'] * interface_21['Tss'] * sum_p  # 等效转换
    time_arr, acc = ACC[:, 0], ACC[:, 1]  # 时间轴与加速度
    dt = ACC[1, 0] - ACC[0, 0]  # 步长
    vel, _ = v4._integrate_acc_to_velocity(acc, dt, time_arr)  # 积分得速度
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 积分得位移
    VEL = np.column_stack((time_arr, vel)); DIS = np.column_stack((time_arr, dis))  # 组合时程
    ctx = v4.FreeFieldCtx(mat_bedrock=mat_b, mat_overlying=mat_o, geom=geom,  # 构造 v4 上下文
                          ymax_l=geom.H_upper, ymax_r=geom.H_lower, ymin=0.0,
                          alpha=alpha1, beta_p=beta1, alpha2=alpha2, beta2=beta2,
                          A1=Rss_eff, A2=Rsp_eff, cycle_sv=cycle_sv, cycle_p=cycle_p,
                          GG=mat_b['GG'], lam=mat_b['lam'], cs=cs1, cp=cp1, cs2=cs2, cp2=cp2,
                          VEL=VEL, DIS=DIS, dt=dt, time_arr=time_arr, max_reflect_order=oc)
    return ctx, geom  # 返回上下文与几何


def _build_ml_ctx(material_cfg, geom_cfg, angle, ACC):  # 构造 multilayer 的 FreeFieldCtx
    site, fixed = ml.build_site(material_cfg, geom_cfg)  # 构建场地与固定层厚
    geom = ml.make_geometry(total_L=geom_cfg['total_L'], H_minus_h=geom_cfg['H_minus_h'],  # 斜坡几何
                            i=geom_cfg['i'], h_over_H=geom_cfg['h_over_H'],
                            left_flat=geom_cfg['left_flat'], bedrock_thickness=geom_cfg['bedrock_thickness'],
                            fixed_thicknesses=fixed)
    strat = ml._build_stratigraphy(site, geom, ymin=0.0)  # 场地分层带
    angle = 1e-10 if angle == 0 else round(angle, 4)  # 入射角处理
    alpha1 = math.radians(angle)  # 基岩 SV 入射角
    mat_b = ml._compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩材料参数
    cs1, cp1 = mat_b['cs'], mat_b['cp']  # 基岩波速
    p_horiz = math.sin(alpha1) / cs1  # 水平慢度
    beta1 = ml._safe_arcsin(cp1 * math.sin(alpha1) / cs1)  # 基岩 P 角
    time_arr, acc = ACC[:, 0], ACC[:, 1]  # 时间轴与加速度
    dt = ACC[1, 0] - ACC[0, 0]  # 步长
    vel, _ = ml._integrate_acc_to_velocity(acc, dt, time_arr)  # 积分得速度
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 积分得位移
    VEL = np.column_stack((time_arr, vel)); DIS = np.column_stack((time_arr, dis))  # 组合时程
    ml._REFL_COEFF_CACHE.clear()  # 清空等效系数缓存
    ctx = ml.FreeFieldCtx(site=site, geom=geom, strat=strat,  # 构造 multilayer 上下文
                          ymax_l=geom.H_upper, ymax_r=geom.H_lower, ymin=0.0,
                          alpha=alpha1, beta_p=beta1, p_horiz=p_horiz,
                          GG=mat_b['GG'], lam=mat_b['lam'], cs=cs1, cp=cp1,
                          VEL=VEL, DIS=DIS, dt=dt, time_arr=time_arr, max_reflect_order=max(0, int(ml.MAX_REFLECT_ORDER)))
    return ctx, geom  # 返回上下文与几何


def main():  # 测试主流程
    nu, rho, vr = 0.3, 2500.0, 1.25  # 泊松比、密度、覆盖层相对波速比
    angle = 15.0  # 入射角
    geom_cfg = {'H_minus_h': 200.0, 'i': 45.0, 'h_over_H': 0.5, 'total_L': 1800.0,  # 几何配置
                'left_flat': 1000.0, 'bedrock_thickness': 200.0}
    material_cfg = {'angle': angle,  # 多层材料配置（M=1 双层）
                    'bedrock': {'elastic_modulus': 26e9, 'poisson_ratio': nu, 'density': rho},
                    'layers': [{'name': 'overlying', 'velocity_ratio': vr, 'poisson_ratio': nu, 'density': rho}]}
    cs_b = ml._compute_wave_speed_from_elastic_modulus(26e9, nu, rho)  # 基岩剪切波速
    cs_o = cs_b / vr  # 覆盖层剪切波速
    ACC = _make_ricker(fp=5.0, dt=0.005, n=400)  # 输入加速度时程

    ctx_v4, geom = _build_v4_ctx(cs_b, cs_o, nu, rho, geom_cfg, angle, ACC)  # v4 上下文
    ctx_ml, _ = _build_ml_ctx(material_cfg, geom_cfg, angle, ACC)  # multilayer 上下文

    gv4 = v4._make_delay_cache(ctx_v4.VEL, ctx_v4.dt); dv4 = v4._make_delay_cache(ctx_v4.DIS, ctx_v4.dt)  # v4 延迟缓存
    gml = ml._make_delay_cache(ctx_ml.VEL, ctx_ml.dt); dml = ml._make_delay_cache(ctx_ml.DIS, ctx_ml.dt)  # ml 延迟缓存

    Hu, Hl = geom.H_upper, geom.H_lower  # 坡顶/坡脚地表高度
    sy = lambda x: ml._surface_y_at(x, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 地表高度函数
    nodes = [  # 代表性节点：(边界, x, y, 该柱地表高度)
        ('l', 0.0, 100.0, Hu),    # 左-基岩
        ('l', 0.0, 450.0, Hu),    # 左-覆盖层
        ('r', 1800.0, 100.0, Hl), # 右-基岩
        ('r', 1800.0, 300.0, Hl), # 右-覆盖层
        ('b', 500.0, 0.0, sy(500.0)),    # 底-上平台下方
        ('b', 1100.0, 0.0, sy(1100.0)),  # 底-坡面下方
    ]
    keys = ['ux', 'uy', 'dotux', 'dotuy', 'sigmax', 'sigmay']  # 比较的场量
    worst = 0.0  # 记录全局最大绝对差
    print('节点(边界,x,y,ymax) -> 各场量最大绝对差')  # 打印表头
    for bnd, x, y, ymax in nodes:  # 遍历每个节点
        f4 = v4._compute_freefield_at_node(bnd, x, y, ymax, ctx_v4, gv4, dv4)  # v4 自由场
        fm = ml._compute_freefield_at_node(bnd, x, y, ymax, ctx_ml, gml, dml)  # ml 自由场
        diffs = {}  # 各场量差
        for k in keys:  # 遍历场量
            n = min(len(f4[k]), len(fm[k]))  # 取公共长度
            d = float(np.max(np.abs(np.asarray(f4[k][:n]) - np.asarray(fm[k][:n]))))  # 最大绝对差
            diffs[k] = d  # 记录
            worst = max(worst, d)  # 更新全局
        print('  (%s,%.0f,%.0f,%.0f): %s' % (bnd, x, y, ymax,  # 打印该节点各场量差
              ', '.join('%s=%.2e' % (k, diffs[k]) for k in keys)))
    print('\n全局最大绝对差 = %.3e' % worst)  # 打印全局最大差
    assert worst < 1e-9, '退化一致性失败：M=1 与 v4 偏差过大 (%.3e)' % worst  # 断言一致
    print('通过：M=1 时 multilayer 与 v4 自由场逐节点一致。')  # 打印通过


if __name__ == '__main__':  # 直接运行入口
    main()  # 执行测试
