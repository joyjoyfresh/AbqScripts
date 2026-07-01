# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
# 测试目的：验证 v9 分层重锚定（anchor='perband'）确实修复了软薄层混响频段的瑞利高频过阻尼，
#           同时不破坏旧 'input' 锚定（向后兼容）。纯 Python（仅 math），无需进入 Abaqus 环境。
#
# 说明：被测脚本顶部依赖 Abaqus 模块，且工作区 shell 挂载对该大文件存在同步滞后，
#       故此处【逐字内联】被测的三个纯函数（与 VAB_oblique_TAF_multilayer_v8.py 中实现完全一致），
#       仅验证 v9 阻尼重锚定公式的行为。若改动了脚本中对应函数，请同步本文件以免漂移。

import sys  # 导入系统模块（退出码）
import math  # 导入数学模块（ξ(f) 计算）


# ====== 以下三个函数与被测脚本 VAB_oblique_TAF_multilayer_v8.py 逐字一致 ======

def _damping_ratio_from_q(cs, is_bedrock, dcfg):  # 由 Q 值换算阻尼比 ξ
    """根据剪切波速与是否基岩，计算品质因子 Q 与阻尼比 ξ=1/(2Q)。"""
    if is_bedrock:  # 基岩层
        Q = dcfg['q_bedrock']  # 基岩 Q≈999（近乎无衰减）
    else:  # 有限层
        Q = dcfg['qs_factor'] * cs  # Qs = qs_factor * cs（论文 coarse-grain 法）
    xi = 1.0 / (2.0 * Q)  # 阻尼比 ξ = 1/(2Q)
    return Q, xi  # 返回品质因子与阻尼比


def _rayleigh_coeffs(xi, dcfg, fc, f_layer=None):  # 由阻尼比计算瑞利阻尼系数 α, β（v9：支持逐层重锚定）
    """按指定方法将阻尼比 ξ 换算为 Abaqus 瑞利阻尼系数 (alpha, beta)。"""
    if dcfg['method'] == 'stiffness':  # 仅刚度比例阻尼
        alpha = 0.0  # 质量比例系数为零
        beta = xi / (math.pi * fc)  # β = ξ/(π·fc)，使 fc 处 ξ 精确
    else:  # 默认 rayleigh 双频拟合
        anchor = dcfg.get('anchor', 'input')  # 拟合锚定方式
        f1 = dcfg['f1_factor'] * fc  # 拟合下限频率（默认锚定输入主频）
        f2 = dcfg['f2_factor'] * fc  # 拟合上限频率（默认锚定输入主频）
        if anchor == 'perband' and f_layer and f_layer > 0:  # v9：逐层按该层共振频带重锚定
            hc = float(dcfg.get('harmonics_cover', 3.0))  # 上限覆盖到共振基频的谐波次数
            f1 = min(f1, float(f_layer))  # 下限纳入该层共振基频（薄软层 f_layer 高时下限不变）
            f2 = max(f2, hc * float(f_layer))  # 上限纳入该层共振谐波，防止混响段被高频 β 过阻尼
        elif anchor == 'dual' and dcfg.get('f_site'):  # v8：双控锚定（场地基频+输入主频）
            f1 = min(f1, float(dcfg['f_site']))  # 下限取较小者，使拟合带覆盖场地基频
        w1 = 2.0 * math.pi * f1  # 拟合下限圆频率
        w2 = 2.0 * math.pi * f2  # 拟合上限圆频率
        alpha = 2.0 * xi * w1 * w2 / (w1 + w2)  # 瑞利 α（两端 ξ 相等≈恒定 Q）
        beta = 2.0 * xi / (w1 + w2)  # 瑞利 β
    return alpha, beta  # 返回瑞利阻尼系数


def _band_resonance_freq(band):  # v9：由分层带几何估算该层一维共振基频 f=cs/(4·d)
    """返回材料带 band 的四分之一波长共振基频 (Hz)；厚度无效返回 None。"""
    d = float(band['y1']) - float(band['y0'])  # 该带标称厚度
    if d <= 0:  # 厚度无效
        return None  # 无共振频率概念
    return float(band['mat'].cs) / (4.0 * d)  # 四分之一波长共振基频


# ====== 测试逻辑 ======

class _Mat(object):  # 简易材料容器（仅供 _band_resonance_freq 取 cs）
    def __init__(self, cs):  # 构造
        self.cs = cs  # 剪切波速


def xi_of_f(alpha, beta, f):  # 由瑞利系数求某频率处的阻尼比 ξ(f)=α/(2ω)+βω/2
    """返回频率 f(Hz) 处的瑞利阻尼比。"""
    w = 2.0 * math.pi * f  # 圆频率
    return alpha / (2.0 * w) + beta * w / 2.0  # 瑞利阻尼比公式


def main():  # 测试主流程
    """构造软薄表层工况，比较 input 与 perband 两种锚定的 ξ(f) 行为并断言。"""
    fc = 2.0  # 输入波主频 (Hz)（典型 Ricker）
    cs_soft = 800.0  # 软表层剪切波速 (m/s)，对应 Vs1/Vs2=0.5（Vs2=1600）
    d_soft = 25.0  # 软表层厚度 (m)
    f_layer_expect = cs_soft / (4.0 * d_soft)  # 预期共振基频 = 800/100 = 8.0 Hz

    dcfg_common = {'method': 'rayleigh', 'qs_factor': 0.05, 'q_bedrock': 999.0,
                   'f1_factor': 0.5, 'f2_factor': 2.5}  # 公共阻尼配置
    Q, xi_target = _damping_ratio_from_q(cs_soft, False, dcfg_common)  # 软层 Q 与目标阻尼比

    band = {'y0': 0.0, 'y1': d_soft, 'mat': _Mat(cs_soft)}  # 模拟一条软层带
    f_layer = _band_resonance_freq(band)  # 共振基频

    dcfg_input = dict(dcfg_common, anchor='input')  # 旧 input 锚定
    dcfg_perband = dict(dcfg_common, anchor='perband', harmonics_cover=3.0)  # 新 perband 锚定
    a_in, b_in = _rayleigh_coeffs(xi_target, dcfg_input, fc, f_layer)  # input 系数
    a_pb, b_pb = _rayleigh_coeffs(xi_target, dcfg_perband, fc, f_layer)  # perband 系数

    f2_pb = max(dcfg_common['f2_factor'] * fc, 3.0 * f_layer)  # perband 实际上限
    print('=' * 76)
    print('软薄表层工况: cs=%.0f m/s, d=%.0f m, fc=%.1f Hz' % (cs_soft, d_soft, fc))
    print('共振基频 f_layer=%.3f Hz (期望 %.3f), 目标阻尼比 xi=%.4f (Q=%.0f)'
          % (f_layer, f_layer_expect, xi_target, Q))
    print('input  拟合带 [%.2f, %.2f] Hz' % (dcfg_common['f1_factor'] * fc, dcfg_common['f2_factor'] * fc))
    print('perband拟合带 [%.2f, %.2f] Hz' % (min(dcfg_common['f1_factor'] * fc, f_layer), f2_pb))
    print('-' * 76)
    print('%8s %12s %12s %10s %10s' % ('f(Hz)', 'xi_input', 'xi_perband', 'in/目标', 'pb/目标'))
    for f in [fc, 4.0, f_layer, 16.0, 3.0 * f_layer, 32.0]:  # 探测频率（含共振基频与谐波）
        xi_in = xi_of_f(a_in, b_in, f)
        xi_pb = xi_of_f(a_pb, b_pb, f)
        print('%8.2f %12.5f %12.5f %10.2f %10.2f' % (f, xi_in, xi_pb, xi_in / xi_target, xi_pb / xi_target))
    print('=' * 76)

    failures = []  # 收集失败项
    # 1) 共振频率口径
    if abs(f_layer - f_layer_expect) > 1e-6:
        failures.append('f_layer 计算错误: %.4f != %.4f' % (f_layer, f_layer_expect))
    # 2) input 向后兼容（与旧公式 f1=1,f2=5 一致）
    w1, w2 = 2 * math.pi * 1.0, 2 * math.pi * 5.0
    a_old = 2.0 * xi_target * w1 * w2 / (w1 + w2)
    b_old = 2.0 * xi_target / (w1 + w2)
    if abs(a_in - a_old) > 1e-9 or abs(b_in - b_old) > 1e-12:
        failures.append('input 破坏向后兼容')
    # 3) 动机成立：input 在 3 阶谐波处显著过阻尼（≥2 倍目标）
    xi_in_h3 = xi_of_f(a_in, b_in, 3.0 * f_layer)
    if xi_in_h3 < 2.0 * xi_target:
        failures.append('input 未体现高频过阻尼: xi(3f)/目标=%.2f' % (xi_in_h3 / xi_target))
    # 4) 核心修复：perband 在共振基频与 3 阶谐波处都不更阻尼，且 ≤1.2 倍目标
    for f in [f_layer, 3.0 * f_layer]:
        xi_in = xi_of_f(a_in, b_in, f)
        xi_pb = xi_of_f(a_pb, b_pb, f)
        if xi_pb > xi_in + 1e-12:
            failures.append('perband 未改善 f=%.2f' % f)
        if xi_pb > 1.2 * xi_target:
            failures.append('perband 在 f=%.2f 仍过阻尼: pb/目标=%.2f' % (f, xi_pb / xi_target))
    # 5) perband 在 f2 处约等于目标（两频点拟合性质）
    xi_pb_f2 = xi_of_f(a_pb, b_pb, f2_pb)
    if abs(xi_pb_f2 - xi_target) > 0.05 * xi_target:
        failures.append('perband 在 f2 偏离目标')

    if failures:  # 有失败项
        print('测试未通过:')
        for msg in failures:
            print('  [FAIL] ' + msg)
        sys.exit(1)
    print('全部断言通过：perband 已消除软层混响频段的过阻尼，且 input 模式向后兼容。')
    print('量化改善: 3*f_layer=%.1f Hz 处 过阻尼倍数 input=%.2f → perband=%.2f'
          % (3.0 * f_layer, xi_in_h3 / xi_target, xi_of_f(a_pb, b_pb, 3.0 * f_layer) / xi_target))
    sys.exit(0)


if __name__ == '__main__':  # 直接运行入口
    main()  # 调用测试主流程
