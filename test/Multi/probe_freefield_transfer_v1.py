# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""探针脚本：在纯 Python 环境下单独运行 v7 自由场传递函数引擎，检查其频域是否有非物理尖峰。

做法：先给 Abaqus 相关模块打桩（stub），使 v7 模块可在无 Abaqus 环境导入，
随后直接调用 _freefield_transfer / _compute_freefield_at_node，对一个代表性地表节点
扫描频率轴，输出传递函数幅值统计与最大尖峰所在频率。
"""

import os  # 导入操作系统接口
import sys  # 导入系统模块
import types  # 导入类型模块用于构造桩模块
import numpy as np  # 导入数值计算库


def _install_abaqus_stubs():  # 定义安装 Abaqus 桩模块的函数
    """构造 abaqus / abaqusConstants / regionToolset / caeModules / mesh 桩，使 v7 可被导入。"""
    for name in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 遍历需打桩的模块名
        mod = types.ModuleType(name)  # 创建空桩模块
        sys.modules[name] = mod  # 注册到 sys.modules
    sys.modules['abaqus'].mdb = types.SimpleNamespace()  # abaqus 需提供 mdb 属性
    sys.modules['regionToolset'].Region = object  # regionToolset 需提供 Region 名称


def _load_v7_module():  # 定义加载 v7 模块的函数
    """以源文件路径加载 VAB_oblique_TAF_double_v7.py 模块对象。"""
    here = os.path.dirname(os.path.abspath(__file__))  # 当前 test 目录
    src = os.path.normpath(os.path.join(here, '..', '..', 'Modeling', 'Multi', 'VAB_oblique_TAF_double_v7.py'))  # 上级目录的 v7 源文件
    import importlib.util  # 导入按路径加载模块的工具
    spec = importlib.util.spec_from_file_location('v7mod', src)  # 构造模块规范
    mod = importlib.util.module_from_spec(spec)  # 从规范创建模块对象
    spec.loader.exec_module(mod)  # 执行模块代码完成加载
    return mod  # 返回模块对象


def main():  # 定义主流程
    _install_abaqus_stubs()  # 先打桩
    v7 = _load_v7_module()  # 加载 v7 模块

    # ===== 材料参数（与 v7 默认配置一致：基岩 Vs=2000，覆盖层 Vs=1600，nu=0.3，rho=2500）=====
    mat_bedrock = v7._compute_material_params(cs=2000.0, vv=0.3, density=2500.0)  # 基岩材料
    mat_overlying = v7._compute_material_params(cs=1600.0, vv=0.3, density=2500.0)  # 覆盖层材料

    import math  # 导入数学模块
    angle_deg = 15.0  # SV 波入射角（度）
    p_horiz = math.sin(math.radians(angle_deg)) / mat_bedrock['cs']  # 水平慢度（Snell）

    # ===== 几何（代表性地表柱：基岩厚 200，覆盖层厚 200，地表 y=400）=====
    y_bottom = 0.0  # 模型底部 y
    h_bedrock = 200.0  # 基岩层厚
    h_overlying = 200.0  # 覆盖层厚
    y_surface = y_bottom + h_bedrock + h_overlying  # 地表 y 坐标
    x_target = 0.0  # 取 x=0 处的柱

    # ===== 频率轴（dt=0.001，与 4/6/8Hz Ricker 一致；N_fft 取 4096 模拟约 2s 记录补零）=====
    dt = 0.001  # 时间步长（s）
    N_fft = 4096  # FFT 长度
    freq_arr = np.fft.rfftfreq(N_fft, d=dt)  # 正频率轴（Hz）

    T = v7._freefield_transfer(  # 计算地表节点频域传递函数
        y_target=y_surface, x_target=x_target,
        mat_bedrock=mat_bedrock, mat_overlying=mat_overlying,
        h_bedrock=h_bedrock, h_overlying=h_overlying, y_bottom=y_bottom,
        p_horiz=p_horiz, freq_arr=freq_arr, dt=dt, N_fft=N_fft)

    print('频率轴: 0 ~ {:.1f} Hz, 共 {} 个 bin, df={:.3f} Hz'.format(
        freq_arr[-1], len(freq_arr), freq_arr[1] - freq_arr[0]))  # 打印频率轴信息

    # ===== 关注地震频段（0~50Hz）内各传递函数幅值的统计与尖峰 =====
    band = freq_arr <= 50.0  # 0~50Hz 频段掩码
    for comp in ('dotux', 'dotuy', 'ux', 'uy'):  # 遍历速度与位移传递函数
        mag = np.abs(T[comp])  # 幅值
        mb = mag[band]  # 频段内幅值
        fb = freq_arr[band]  # 频段内频率
        imax = int(np.argmax(mb))  # 最大幅值索引
        # 统计相邻 bin 的跳变最大值，刻画"毛刺"程度
        jump = np.max(np.abs(np.diff(mb))) if len(mb) > 1 else 0.0  # 相邻 bin 最大跳变
        print('[{:6s}] 0-50Hz: max|T|={:.3e} @ {:.2f}Hz, 中位={:.3e}, 相邻最大跳变={:.3e}'.format(
            comp, mb[imax], fb[imax], np.median(mb), jump))  # 打印统计

    # ===== 全频段（含 50Hz 以上）峰值，检查高频是否异常放大 =====
    print('\n--- 全频段峰值（含高频）---')  # 分隔标题
    for comp in ('dotux', 'dotuy'):  # 遍历速度传递函数
        mag = np.abs(T[comp])  # 幅值
        imax = int(np.argmax(mag))  # 最大索引
        print('[{:6s}] 全频段 max|T|={:.3e} @ {:.2f}Hz'.format(comp, mag[imax], freq_arr[imax]))  # 打印

    # ===== 检查界面 4×4 / 自由面 2×2 是否出现近奇异（行列式极小但未被清零）=====
    # 直接复算自由面 2×2 行列式量级，定位潜在尖峰来源
    print('\n--- 直接复算诊断（自由面 2x2 行列式量级）---')  # 分隔标题
    omega = 2.0 * math.pi * freq_arr  # 角频率
    cs2, cp2 = mat_overlying['cs'], mat_overlying['cp']  # 覆盖层波速

    def qval(c):  # 计算垂直慢度
        val = (1.0 / c) ** 2 - p_horiz ** 2  # 慢度平方
        return complex(math.sqrt(val), 0) if val >= 0 else complex(0, math.sqrt(-val))  # 实或虚

    qs2, qp2 = qval(cs2), qval(cp2)  # 覆盖层 SV/P 垂直慢度
    print('覆盖层 qs2={}, qp2={}（虚部非零=倏逝波）'.format(qs2, qp2))  # 打印垂直慢度


if __name__ == '__main__':  # 脚本入口
    main()  # 运行主流程
