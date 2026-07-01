# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""v8 新特性纯 Python 单元测试（不依赖 Abaqus 环境，可直接 python 运行）。

覆盖 v8 增量：
  T1 _fd_engine_selfcheck：建模前自检（半空间退化 + 单层 SH 解析对拍）误差应为机器精度量级；
  T2 _site_fundamental_freq：三层工况场地基频 f_site = 1/(4Σd/Vs) = 0.4444 Hz；
  T3 _rayleigh_coeffs 双控锚定：anchor='dual' 时拟合下限取 min(f1_factor·fc, f_site)；
  T4 time_cfg['tail_seconds']：fd 输入谱缓存的输出窗口 Nout 正确延长；
  T5 临界角常数：基岩(ν=0.3) SV 临界角 ≈ 32.31°（VAB_oblique 校验所依据的公式）。
运行：python test_v8_features_v1.py（可用环境变量 TEST_V8_PATH 覆盖脚本路径）。
"""

import os  # 导入路径模块
import sys  # 导入系统模块
import math  # 导入数学模块
import types  # 导入动态模块构造工具
import numpy as np  # 导入数值计算库


def _install_abaqus_stubs():  # 注入 Abaqus 桩模块
    """让 v8 脚本的 import 语句在普通 Python 中可通过（仅满足模块级导入）。"""
    for name in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 全部 Abaqus 模块
        if name in sys.modules:  # 已存在则不覆盖
            continue  # 跳过
        m = types.ModuleType(name)  # 创建空桩模块
        if name == 'abaqus':  # abaqus 模块需提供 mdb 属性
            m.mdb = None  # 占位 mdb
        if name == 'regionToolset':  # regionToolset 需提供 Region 名称
            m.Region = object  # 占位 Region
        sys.modules[name] = m  # 注册桩模块


def _load_v8():  # 加载 v8 脚本为模块对象
    """编译并执行 v8 源码（不触发 main），返回模块对象。"""
    here = os.path.dirname(os.path.abspath(__file__))  # 本测试文件所在目录
    default = os.path.join(here, '..', '..', 'Modeling', 'Multi', 'VAB_oblique_TAF_multilayer_v8.py')  # 默认取上级目录的 v8 脚本
    path = os.environ.get('TEST_V8_PATH', default)  # 允许环境变量覆盖路径
    with open(path, 'r', encoding='utf-8') as f:  # 读取源码
        src = f.read()  # 源码文本
    mod = types.ModuleType('vab_v8')  # 创建空模块容器
    mod.__file__ = os.path.abspath(path)  # 设置 __file__
    exec(compile(src, path, 'exec'), mod.__dict__)  # 编译并执行（仅模块级定义）
    return mod  # 返回已加载模块


def main():  # 测试主入口
    """顺序执行 T1~T5，全部通过则打印 ALL V8 TESTS PASSED。"""
    _install_abaqus_stubs()  # 注入桩模块
    mod = _load_v8()  # 加载 v8 脚本

    # ---- T1：fd 引擎内置自检 ----
    res = mod._fd_engine_selfcheck(None)  # 运行自检（无日志器）
    assert res['halfspace_err'] < 1e-6, 'T1 半空间自检误差过大: %g' % res['halfspace_err']  # 校验半空间
    assert res['single_layer_err'] < 1e-6, 'T1 单层自检误差过大: %g' % res['single_layer_err']  # 校验单层
    print('[T1] fd 引擎自检: 通过 (halfspace=%.1e, single_layer=%.1e)' % (res['halfspace_err'], res['single_layer_err']))  # 打印

    # ---- T2：场地基频（三层：表层 50m/Vs400 + 覆盖层 350m/Vs800） ----
    mat_cfg = {'angle': 0, 'bedrock': {'elastic_modulus': 26e9, 'poisson_ratio': 0.3, 'density': 2500},  # 材料配置
               'layers': [{'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 50.0},  # 表层
                          {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500}]}  # 覆盖层
    geo_cfg = {'H_minus_h': 200.0, 'i': 45.0, 'h_over_H': 0.5, 'total_L': 1800.0,  # 几何配置
               'left_flat': 1000.0, 'bedrock_thickness': 200.0}  # 平台与基岩厚
    site, fixed = mod.build_site(mat_cfg, geo_cfg)  # 构建场地
    geom = mod.make_geometry(total_L=1800.0, H_minus_h=200.0, i=45.0, h_over_H=0.5,  # 构建几何
                             left_flat=1000.0, bedrock_thickness=200.0, fixed_thicknesses=fixed)  # 含固定层厚
    f_site = mod._site_fundamental_freq(site, geom)  # 估算场地基频
    expect = 1.0 / (4.0 * (50.0 / 400.0 + 350.0 / 800.0))  # 期望值 = 1/(4×0.5625) ≈ 0.4444
    assert abs(f_site - expect) < 1e-9, 'T2 f_site=%g 期望 %g' % (f_site, expect)  # 校验基频
    site1 = site._replace(layers=[])  # 均质场地
    assert mod._site_fundamental_freq(site1, geom) is None, 'T2 均质场地应返回 None'  # 校验退化
    print('[T2] 场地基频: 通过 (f_site=%.4f Hz)' % f_site)  # 打印

    # ---- T3：瑞利双控锚定 ----
    fc = 4.0  # 输入主频
    base = {'method': 'rayleigh', 'f1_factor': 0.5, 'f2_factor': 2.5}  # 基础配置
    a_in, b_in = mod._rayleigh_coeffs(0.025, dict(base, anchor='input'), fc)  # input 锚定（=v7 行为）
    a_du, b_du = mod._rayleigh_coeffs(0.025, dict(base, anchor='dual', f_site=f_site), fc)  # dual 锚定
    w1d = 2.0 * math.pi * min(0.5 * fc, f_site)  # dual 期望下限圆频率（f_site < 2.0 → 取 f_site）
    w2 = 2.0 * math.pi * 2.5 * fc  # 上限圆频率
    assert abs(a_du - 2.0 * 0.025 * w1d * w2 / (w1d + w2)) < 1e-12, 'T3 dual α 不符'  # 校验 α
    assert abs(b_du - 2.0 * 0.025 / (w1d + w2)) < 1e-12, 'T3 dual β 不符'  # 校验 β
    w1i = 2.0 * math.pi * 0.5 * fc  # input 期望下限圆频率
    assert abs(a_in - 2.0 * 0.025 * w1i * w2 / (w1i + w2)) < 1e-12, 'T3 input α 改变（应=v7）'  # 校验向后兼容
    print('[T3] 瑞利双控锚定: 通过 (input α=%.4f → dual α=%.4f)' % (a_in, a_du))  # 打印

    # ---- T4：fd 输入谱输出窗口延长（tail_seconds） ----
    dt = 1e-3  # 时间步长
    t = np.arange(0.0, 2.0, dt)  # 2 秒时间轴
    arg = (math.pi * 4.0) ** 2 * (t - 0.275) ** 2  # Ricker 公共项
    acc = (1.0 - 2.0 * arg) * np.exp(-arg)  # 4Hz Ricker 子波
    ffcfg = dict(mod.freefield_cfg)  # 自由场配置副本
    ffcfg['tail_seconds'] = 1.0  # 设置 1 秒静默尾段
    ctx = mod.FreeFieldCtx(site=None, geom=None, strat=None, ymax_l=None, ymax_r=None, ymin=None,  # 构造最小上下文
                           alpha=None, beta_p=None, p_horiz=None, GG=None, lam=None, cs=None, cp=None,  # 占位
                           VEL=None, DIS=None, dt=dt, time_arr=t, max_reflect_order=3,  # 时间信息
                           acc=acc, damp_terms=None, ffcfg=ffcfg)  # 记录与配置
    mod._FD_SOLVER_CACHE.clear()  # 清空缓存（避免跨用例污染）
    inp = mod._fd_input_spectrum(ctx)  # 计算输入谱缓存
    assert inp['Nout'] == len(acc) + 1000, 'T4 Nout=%d 期望 %d' % (inp['Nout'], len(acc) + 1000)  # 校验窗口延长
    ffcfg['tail_seconds'] = 0.0  # 关闭尾段
    mod._FD_SOLVER_CACHE.clear()  # 再次清空缓存
    inp0 = mod._fd_input_spectrum(ctx)  # 重新计算
    assert inp0['Nout'] == len(acc), 'T4 tail=0 时 Nout 应等于记录长度'  # 校验向后兼容
    print('[T4] 静默尾段: 通过 (tail=1.0s → Nout=%d, tail=0 → Nout=%d)' % (inp['Nout'], inp0['Nout']))  # 打印

    # ---- T5：基岩 SV 临界角常数 ----
    mat_b = mod._compute_material_params(2000.0, 0.3, 2500.0)  # 基岩派生参数
    crit = math.degrees(math.asin(mat_b['cs'] / mat_b['cp']))  # 临界角 = asin(cs/cp)
    assert abs(crit - 32.31) < 0.05, 'T5 临界角=%.2f 期望≈32.31' % crit  # 校验常数
    print('[T5] 临界角常数: 通过 (%.2f°)' % crit)  # 打印

    print('ALL V8 TESTS PASSED')  # 全部通过标记


if __name__ == '__main__':  # 直接运行
    main()  # 执行测试
