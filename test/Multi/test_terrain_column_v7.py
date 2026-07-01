# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""v7 建模脚本纯 Python 单元测试（不依赖 Abaqus 环境，可直接 python 运行）。

通过注入 Abaqus 桩模块（stub）加载 VAB_oblique_TAF_multilayer_v7.py 的纯计算部分，验证：
  T1 分层带构造 _build_stratigraphy：horizontal/terrain 两模式的 fix/d0/d1/dtop 字段；
  T2 柱构造 _build_column：terrain 模式下随地表高度（上平台/坡中/下平台）正确取层；
  T3 fd 引擎半空间退化：弹性垂直入射地表水平位移 = 2×入射幅值；
  T4 远场一维理论台阶回归：4Hz Ricker 下与 plan.md 诊断数字一致
     （horizontal 软表层左柱 1.568 / 右柱 1.328 / 硬表层左柱 0.793；terrain 软表层右柱 1.681）。
运行：python test_terrain_column_v7.py（可用环境变量 TEST_V7_PATH 覆盖脚本路径）。
"""

import os  # 导入路径模块
import sys  # 导入系统模块
import math  # 导入数学模块
import types  # 导入动态模块构造工具
import numpy as np  # 导入数值计算库


def _install_abaqus_stubs():  # 注入 Abaqus 桩模块
    """让 v7 脚本的 import 语句在普通 Python 中可通过（仅满足模块级导入，不实现功能）。"""
    for name in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 全部被导入的 Abaqus 模块
        if name in sys.modules:  # 已存在则不覆盖
            continue  # 跳过
        m = types.ModuleType(name)  # 创建空桩模块
        if name == 'abaqus':  # abaqus 模块需提供 mdb 属性
            m.mdb = None  # 占位 mdb
        if name == 'regionToolset':  # regionToolset 需提供 Region 名称
            m.Region = object  # 占位 Region
        sys.modules[name] = m  # 注册桩模块


def _load_v7():  # 加载 v7 脚本为模块对象
    """编译并执行 v7 源码（__name__ 非 __main__，不会触发 main()），返回模块对象。"""
    here = os.path.dirname(os.path.abspath(__file__))  # 本测试文件所在目录
    default = os.path.join(here, '..', '..', 'Modeling', 'Multi', 'VAB_oblique_TAF_multilayer_v7.py')  # 默认取上级目录的 v7 脚本
    path = os.environ.get('TEST_V7_PATH', default)  # 允许环境变量覆盖路径
    with open(path, 'r', encoding='utf-8') as f:  # 读取源码
        src = f.read()  # 源码文本
    mod = types.ModuleType('vab_v7')  # 创建空模块容器
    mod.__file__ = os.path.abspath(path)  # 设置 __file__ 供脚本内部使用
    exec(compile(src, path, 'exec'), mod.__dict__)  # 编译并执行（仅模块级定义）
    return mod  # 返回已加载模块


def _make_case(mod, surf_vr=5.0, surf_thick=50.0):  # 构造与批处理一致的三层工况
    """返回 (site, geom)：基岩2000 + 覆盖层800 + 表层(软400/硬1600, 厚50)，i=45 几何。"""
    mat_cfg = {  # 材料配置（与 Autorun_TAF_multilayer_v2-testv6-2 三层工况一致）
        'angle': 0,  # 垂直入射
        'bedrock': {'elastic_modulus': 26e9, 'poisson_ratio': 0.3, 'density': 2500},  # 基岩 Vs=2000
        'layers': [  # 从上到下：表层 + 覆盖层
            {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},  # 表层
            {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},  # 覆盖层 Vs=800
        ],  # 结束层列表
    }  # 结束材料配置
    geo_cfg = {'H_minus_h': 200.0, 'i': 45.0, 'h_over_H': 0.5, 'total_L': 1800.0,  # 几何配置（论文图15 口径）
               'left_flat': 1000.0, 'bedrock_thickness': 200.0}  # 上平台与基岩厚度
    site, fixed = mod.build_site(mat_cfg, geo_cfg)  # 构建场地对象
    geom = mod.make_geometry(total_L=geo_cfg['total_L'], H_minus_h=geo_cfg['H_minus_h'], i=geo_cfg['i'],  # 构建几何对象
                             h_over_H=geo_cfg['h_over_H'], left_flat=geo_cfg['left_flat'],  # 平台与深度比
                             bedrock_thickness=geo_cfg['bedrock_thickness'], fixed_thicknesses=fixed)  # 基岩厚与固定层厚
    return site, geom  # 返回场地与几何


def _column_brief(col):  # 把柱层段压缩为 (层名, 厚度) 列表
    """返回 [(name, thickness), ...]，便于断言。"""
    return [(seg['name'], round(seg['y1'] - seg['y0'], 6)) for seg in col]  # 逐段提取名称与厚度


def _taf_theory(mod, site, geom, sgeom, ys, acc, dt, damping):  # 用 v7 模块函数计算某柱理论台阶
    """复刻 _write_case_meta 的 ff_theory 流程：返回 (TAF_h, TAF_v)。"""
    strat = mod._build_stratigraphy(site, geom, ymin=0.0, surface_geometry=sgeom)  # 与建模同口径分层
    damp_terms = mod._band_damping_terms(strat, damping)  # 各带瑞利系数
    alpha_r = math.radians(1e-10)  # 垂直入射用极小角
    mat_b = mod._compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩参数
    fs = mod._compute_free_surface_sv_coeff(alpha_r, mat_b['cp'], mat_b['cs'])  # 自由面系数
    factor_h = (1.0 - fs['A1']) * math.cos(alpha_r) + fs['A2'] * math.sin(fs['beta'])  # 水平放大系数（0°≈2）
    p0 = math.sin(alpha_r) / mat_b['cs']  # 水平慢度
    Nfft = mod._next_pow2(len(acc) * 4)  # FFT 长度
    A0 = np.fft.rfft(acc, n=Nfft)  # 加速度谱
    freqs0 = np.fft.rfftfreq(Nfft, dt)  # 频率轴
    mask0 = np.abs(A0) > 1e-7 * float(np.max(np.abs(A0)))  # 谱掩码
    mask0[0] = False  # 去直流
    idx0 = np.nonzero(mask0)[0]  # 求解频点
    om0 = 2.0 * math.pi * freqs0[idx0]  # 圆频率
    col = mod._build_column(strat, ys, p0, 0.0)  # 构造该柱
    sol = mod._fd_solve_column(col, p0, om0, damp_terms, True)  # 频域求解（计入阻尼）
    fld = mod._fd_eval_column(sol, om0, p0, ys)  # 地表场量谱
    spec = np.zeros(len(freqs0), dtype=complex)  # x 谱容器
    spec[idx0] = fld['ux'] * A0[idx0]  # 加速度谱叠加
    ax = np.fft.irfft(spec, n=Nfft)  # x 加速度时程
    spec = np.zeros(len(freqs0), dtype=complex)  # y 谱容器
    spec[idx0] = fld['uy'] * A0[idx0]  # y 加速度谱
    ay = np.fft.irfft(spec, n=Nfft)  # y 加速度时程
    denom = factor_h * float(np.max(np.abs(acc)))  # 解析分母
    return float(np.max(np.abs(ax))) / denom, float(np.max(np.abs(ay))) / denom  # 返回 TAF_h/TAF_v


def main():  # 测试主入口
    """顺序执行 T1~T4，全部通过则打印 ALL TESTS PASSED。"""
    _install_abaqus_stubs()  # 注入桩模块
    mod = _load_v7()  # 加载 v7 脚本
    site, geom = _make_case(mod)  # 软表层三层工况

    # ---- T1：分层带构造（两种表层几何模式） ----
    strat_h = mod._build_stratigraphy(site, geom, surface_geometry='horizontal')  # horizontal 模式分层
    assert [b['fix'] for b in strat_h] == ['elevation'] * 3, 'T1 horizontal 模式应全部为 elevation 带'  # 校验 fix
    strat_t = mod._build_stratigraphy(site, geom, surface_geometry='terrain')  # terrain 模式分层
    assert [b['fix'] for b in strat_t] == ['elevation', 'fill', 'depth'], 'T1 terrain 模式带类型错误'  # 校验 fix 顺序
    assert strat_t[2]['d0'] == 0.0 and strat_t[2]['d1'] == 50.0, 'T1 表层埋深 d0/d1 错误'  # 校验表层埋深
    assert strat_t[1]['dtop'] == 50.0, 'T1 覆盖层 dtop 错误'  # 校验覆盖层顶埋深
    print('[T1] 分层带构造: 通过 (horizontal=3×elevation, terrain=elevation/fill/depth)')  # 打印通过

    # ---- T2：柱构造随地表高度取层（terrain） ----
    p0 = 1e-15  # 近垂直入射水平慢度
    expect = {600.0: [('Bedrock', 200.0), ('overlying', 350.0), ('surface', 50.0)],  # 上平台柱
              500.0: [('Bedrock', 200.0), ('overlying', 250.0), ('surface', 50.0)],  # 坡中柱
              400.0: [('Bedrock', 200.0), ('overlying', 150.0), ('surface', 50.0)]}  # 下平台柱
    for ys, exp in expect.items():  # 遍历三种柱
        col = mod._build_column(strat_t, ys, p0, 0.0)  # terrain 模式构柱
        assert _column_brief(col) == exp, 'T2 terrain 柱(ys=%g)取层错误: %s' % (ys, _column_brief(col))  # 校验层组成
    col_h = mod._build_column(strat_h, 400.0, p0, 0.0)  # horizontal 模式下平台柱
    assert _column_brief(col_h) == [('Bedrock', 200.0), ('overlying', 200.0)], 'T2 horizontal 下平台柱不应含表层'  # 校验无表层
    print('[T2] 柱构造取层: 通过 (terrain 三柱含表层 50m, horizontal 下平台无表层)')  # 打印通过

    # ---- T3：fd 引擎半空间退化（弹性 0° 地表水平位移 = 2E） ----
    site1, geom1 = _make_case(mod)  # 借用几何
    site1 = site1._replace(layers=[])  # 改为无有限层（均质基岩半空间）
    strat1 = mod._build_stratigraphy(site1, geom1)  # 单带分层
    col1 = mod._build_column(strat1, geom1.H_upper, p0, 0.0)  # 均质柱
    om = 2.0 * math.pi * np.array([1.0, 3.0, 5.0])  # 校核频率
    sol1 = mod._fd_solve_column(col1, p0, om, {'Bedrock': (0.0, 0.0)}, True)  # 弹性求解
    fld1 = mod._fd_eval_column(sol1, om, p0, geom1.H_upper)  # 地表场量
    assert np.allclose(np.abs(fld1['ux']), 2.0, atol=1e-6), 'T3 半空间地表水平位移应=2E'  # 校验自由面放大
    print('[T3] fd 半空间退化: 通过 (|ux_surf|=2.000)')  # 打印通过

    # ---- T4：远场一维理论台阶回归（需 4Hz Ricker 输入文件） ----
    here = os.path.dirname(os.path.abspath(__file__))  # 测试目录
    rick = os.path.normpath(os.path.join(here, '..', '..', 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt'))  # 仓库内输入波
    if not os.path.isfile(rick):  # 输入波缺失则跳过回归
        print('[T4] 跳过（未找到 %s）' % rick)  # 打印跳过
    else:  # 输入波存在时执行回归
        rec = np.loadtxt(rick)  # 读取记录
        acc, dt = rec[:, 1], float(rec[1, 0] - rec[0, 0])  # 加速度与步长
        fc_est = mod._estimate_dominant_freq(acc, dt)  # 主频估计（≈4Hz）
        damping = mod._resolve_damping(dict(mod.damping_cfg), fc_est)  # 解析阻尼配置（Qs=0.05cs）
        checks = [  # (说明, 模式, 表层vr, 柱地表, 期望TAF_h)
            ('horizontal 软表层左柱', 'horizontal', 5.0, 600.0, 1.568),  # plan.md 表值
            ('horizontal 软表层右柱', 'horizontal', 5.0, 400.0, 1.328),  # 下平台无表层
            ('horizontal 硬表层左柱', 'horizontal', 1.25, 600.0, 0.793),  # 硬表层
            ('terrain    软表层右柱', 'terrain', 5.0, 400.0, 1.681),  # 表层沿地形 → 下平台含表层
        ]  # 结束回归表
        for name, sg, vr, ys, exp in checks:  # 逐项回归
            s, g = _make_case(mod, surf_vr=vr)  # 构造对应工况
            th, _tv = _taf_theory(mod, s, g, sg, ys, acc, dt, damping)  # 计算理论台阶
            assert abs(th - exp) < 0.02, 'T4 %s TAF_h=%.3f 期望 %.3f' % (name, th, exp)  # 容差 0.02
            print('[T4] %s: TAF_h=%.3f (期望 %.3f) 通过' % (name, th, exp))  # 打印通过

    print('ALL TESTS PASSED')  # 全部通过标记


if __name__ == '__main__':  # 直接运行
    main()  # 执行测试
