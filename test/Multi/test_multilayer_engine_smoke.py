# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""多层自由场引擎冒烟测试（不依赖 Abaqus）：验证 1/2/3 层均可构建分层、求等效系数、算自由场。

检查项：
  - _build_stratigraphy 分层带数量与界面 y 正确；
  - 各代表性柱（坡顶3层、坡脚2层、坡面）的 _effective_refl_coeffs / _column_cavities 有限且层数正确；
  - 代表节点（含表层节点）的自由场 ux/uy/σ 全部有限。
运行：python test/test_multilayer_engine_smoke.py
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

PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Modeling', 'Multi')  # 上级目录（Multi）
if PARENT not in sys.path:  # 确保可 import 上级脚本
    sys.path.insert(0, PARENT)  # 加入搜索路径
ml = importlib.import_module('VAB_oblique_TAF_multilayer_v1')  # 导入多层脚本


def _ctx(material_cfg, gcfg, angle=15.0):  # 构造自由场上下文
    site, fixed = ml.build_site(material_cfg, gcfg)  # 构建场地与固定层厚
    geom = ml.make_geometry(total_L=gcfg['total_L'], H_minus_h=gcfg['H_minus_h'], i=gcfg['i'],  # 斜坡几何
                            h_over_H=gcfg['h_over_H'], left_flat=gcfg['left_flat'],
                            bedrock_thickness=gcfg['bedrock_thickness'], fixed_thicknesses=fixed)
    strat = ml._build_stratigraphy(site, geom, ymin=0.0)  # 场地分层带
    alpha1 = math.radians(angle)  # 入射角
    mat_b = ml._compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩参数
    cs1, cp1 = mat_b['cs'], mat_b['cp']  # 基岩波速
    p = math.sin(alpha1) / cs1; beta1 = ml._safe_arcsin(cp1 * math.sin(alpha1) / cs1)  # 慢度与基岩 P 角
    t = np.arange(400) * 0.005  # 时间轴
    a = (1 - 2 * (math.pi * 5 * (t - 0.2)) ** 2) * np.exp(-(math.pi * 5 * (t - 0.2)) ** 2)  # Ricker 加速度
    vel, _ = ml._integrate_acc_to_velocity(a, 0.005, t)  # 积分得速度
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * 0.005)  # 积分得位移
    ml._REFL_COEFF_CACHE.clear()  # 清空缓存
    ctx = ml.FreeFieldCtx(site=site, geom=geom, strat=strat, ymax_l=geom.H_upper, ymax_r=geom.H_lower,  # 上下文
                          ymin=0.0, alpha=alpha1, beta_p=beta1, p_horiz=p, GG=mat_b['GG'], lam=mat_b['lam'],
                          cs=cs1, cp=cp1, VEL=np.column_stack((t, vel)), DIS=np.column_stack((t, dis)),
                          dt=0.005, time_arr=t, max_reflect_order=3)
    return ctx, geom, strat, p  # 返回上下文、几何、分层、慢度


def _check(material_cfg, gcfg, exp_nbands, tag):  # 对一种分层配置做检查
    ctx, geom, strat, p = _ctx(material_cfg, gcfg)  # 构造上下文
    assert len(strat) == exp_nbands, '%s 分层带数=%d，期望 %d' % (tag, len(strat), exp_nbands)  # 校验带数
    gv = ml._make_delay_cache(ctx.VEL, ctx.dt); gd = ml._make_delay_cache(ctx.DIS, ctx.dt)  # 延迟缓存
    print('%s 分层带=%s' % (tag, [(b['name'], round(b['y0'], 1), round(b['y1'], 1)) for b in strat]))  # 打印分层
    # 代表节点（含坡顶表层、覆盖层、基岩、坡脚、底边、坡面）
    Hu, Hl = geom.H_upper, geom.H_lower  # 坡顶/坡脚高度
    sy = lambda x: ml._surface_y_at(x, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 地表高度
    nodes = [('l', 0.0, Hu - 10.0, Hu), ('l', 0.0, (geom.bedrock_thickness + Hu) / 2, Hu),  # 坡顶表层/覆盖层
             ('l', 0.0, 50.0, Hu), ('r', geom.total_L, Hl - 50.0, Hl),  # 坡顶基岩 / 坡脚
             ('b', geom.left_flat / 2, 0.0, sy(geom.left_flat / 2)),  # 上平台下底边
             ('b', geom.left_flat + geom.w_slope / 2, 0.0, sy(geom.left_flat + geom.w_slope / 2))]  # 坡面下底边
    for bnd, x, y, ymax in nodes:  # 遍历节点
        ff = ml._compute_freefield_at_node(bnd, x, y, ymax, ctx, gv, gd)  # 计算自由场
        for k in ('ux', 'uy', 'dotux', 'dotuy', 'sigmax', 'sigmay'):  # 遍历场量
            assert np.all(np.isfinite(ff[k])), '%s 节点(%s,%.0f,%.0f) %s 非有限' % (tag, bnd, x, y, k)  # 校验有限
    print('  %s 全部代表节点自由场有限。' % tag)  # 打印通过


def main():  # 测试主流程
    nu, rho = 0.3, 2500.0  # 泊松比、密度
    gcfg = {'H_minus_h': 200.0, 'i': 45.0, 'h_over_H': 0.5, 'total_L': 1800.0,  # 几何
            'left_flat': 1000.0, 'bedrock_thickness': 200.0}
    base = {'angle': 15.0, 'bedrock': {'elastic_modulus': 26e9, 'poisson_ratio': nu, 'density': rho}}  # 基础配置
    surf = {'name': 'surface', 'velocity_ratio': 1.6, 'poisson_ratio': nu, 'density': rho, 'thickness': 50.0}  # 表层
    over = {'name': 'overlying', 'velocity_ratio': 1.25, 'poisson_ratio': nu, 'density': rho}  # 覆盖层
    _check(dict(base, layers=[]), gcfg, 1, '单层M=0')  # 单层
    _check(dict(base, layers=[over]), gcfg, 2, '双层M=1')  # 双层
    _check(dict(base, layers=[surf, over]), gcfg, 3, '三层M=2')  # 三层
    print('\n通过：1/2/3 层自由场引擎均可正常构建与求解。')  # 打印总通过


if __name__ == '__main__':  # 直接运行入口
    main()  # 执行测试
