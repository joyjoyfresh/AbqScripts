# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""test_damping_conversion_v1.py —— 阻尼换算纯 Python 单测（无需 Abaqus 环境）

测试范围：
  1. _estimate_dominant_freq：合成 4 Hz Ricker 波，估计主频≈4.0 Hz
  2. _damping_ratio_from_q：Vs2=800→ξ≈0.0125；软表层 Vs1=400→0.025；硬 Vs1=1600→0.00625；基岩 Q=999→ξ≈5e-4
  3. _rayleigh_coeffs（stiffness 法）：ξ(fc)=1/(2Q) 复算成立（β=ξ/(π·fc)，α=0）
  4. _rayleigh_coeffs（rayleigh 法）：在 f1、f2 两端 ξ 均≈1/(2Q)
  5. 基岩（Q=999）→ α≈0, β≈0

运行：py -3 test_damping_conversion_v1.py
"""

import numpy as np  # 导入数值计算库
import math  # 导入数学模块
import sys  # 导入系统模块


# ==========================================================
#  从 VAB_oblique_TAF_multilayer_v3.py 复制的纯函数（桩屏蔽 abaqus 依赖）
# ==========================================================


def _estimate_dominant_freq(acc, dt):  # 从加速度时程估计主频
    """对加速度时程做 FFT，返回幅值谱最大处频率（DC 置 0）。"""
    acc_centered = acc - np.mean(acc)  # 去均值，避免 DC 分量干扰
    n = len(acc_centered)  # 信号长度
    spectrum = np.abs(np.fft.rfft(acc_centered))  # 单边幅值谱
    freqs = np.fft.rfftfreq(n, dt)  # 对应频率轴
    spectrum[0] = 0.0  # 置零 DC 分量
    idx_max = np.argmax(spectrum)  # 幅值最大处索引
    return float(freqs[idx_max])  # 返回主导频率


def _damping_ratio_from_q(cs, is_bedrock, dcfg):  # 由 Q 值换算阻尼比 ξ
    """根据剪切波速与是否基岩，计算品质因子 Q 与阻尼比 ξ=1/(2Q)。"""
    if is_bedrock:  # 基岩层
        Q = dcfg['q_bedrock']  # 基岩 Q≈999（近乎无衰减）
    else:  # 有限层
        Q = dcfg['qs_factor'] * cs  # Qs = 0.05 * cs（论文 coarse-grain 法）
    xi = 1.0 / (2.0 * Q)  # 阻尼比 ξ = 1/(2Q)
    return Q, xi  # 返回品质因子与阻尼比


def _rayleigh_coeffs(xi, dcfg, fc):  # 由阻尼比计算瑞利阻尼系数 α, β
    """按指定方法将阻尼比 ξ 换算为 Abaqus 瑞利阻尼系数 (alpha, beta)。"""
    if dcfg['method'] == 'stiffness':  # 仅刚度比例阻尼
        alpha = 0.0  # 质量比例系数为零
        beta = xi / (math.pi * fc)  # β = ξ/(π·fc)，使 fc 处 ξ 精确
    else:  # 默认 rayleigh 双频拟合
        w1 = 2.0 * math.pi * dcfg['f1_factor'] * fc  # 拟合下限圆频率
        w2 = 2.0 * math.pi * dcfg['f2_factor'] * fc  # 拟合上限圆频率
        alpha = 2.0 * xi * w1 * w2 / (w1 + w2)  # 瑞利 α（两端 ξ 相等≈恒定 Q）
        beta = 2.0 * xi / (w1 + w2)  # 瑞利 β
    return alpha, beta  # 返回瑞利阻尼系数


def _resolve_damping(dcfg, fc_est):  # 解析阻尼配置（补全 fc 字段）
    """拷贝阻尼配置，若 fc 为空则用 fc_est 填充，返回解析后的配置 dict。"""
    resolved = dict(dcfg)  # 浅拷贝配置
    if resolved['fc'] is None:  # 未显式指定主频
        resolved['fc'] = fc_est  # 使用自动估计值
    return resolved  # 返回解析后配置


# ==========================================================
#  Ricker 子波生成（测试用）
# ==========================================================


def ricker_wavelet(t, fc=4.0, t0=None):  # 生成 Ricker 子波
    """生成 Ricker 子波（墨西哥帽小波）。

    t  : 时间轴数组
    fc : 中心频率 (Hz)
    t0 : 峰值时刻 (s)，默认取 1/fc 使波形完整
    返回 Ricker 子波加速度数组
    """
    if t0 is None:  # 未指定峰值时刻
        t0 = 1.0 / fc  # 默认为 1/fc
    tau = math.pi * fc * (t - t0)  # 无量纲时间参数
    return (1.0 - 2.0 * tau ** 2) * np.exp(-tau ** 2)  # Ricker 公式


# ==========================================================
#  测试函数
# ==========================================================


def test_estimate_dominant_freq():  # 测试主频估计
    """合成 4 Hz Ricker 波，验证主频估计≈4.0 Hz。"""
    dt = 0.001  # 时间步长 0.001 s
    t = np.arange(0, 1.0, dt)  # 1 秒时间轴
    acc = ricker_wavelet(t, fc=4.0)  # 合成 4 Hz Ricker 波
    fc_est = _estimate_dominant_freq(acc, dt)  # 估计主频
    assert 3.5 <= fc_est <= 4.5, '主频估计偏差过大: %.2f Hz (期望≈4.0 Hz)' % fc_est  # 允许 ±0.5 Hz 容差
    print('  [PASS] _estimate_dominant_freq: fc=%.2f Hz (期望≈4.0 Hz)' % fc_est)  # 输出测试结果


def test_damping_ratio_from_q():  # 测试 Q→ξ 换算
    """验证不同波速下的 ξ=1/(2Q) 数值：
    Vs2=800→ξ≈0.0125；软 Vs1=400→0.025；硬 Vs1=1600→0.00625；基岩 Q=999→ξ≈5e-4。
    """
    dcfg = {'qs_factor': 0.05, 'q_bedrock': 999.0}  # 默认阻尼配置
    # Vs2 = 800 m/s（覆盖层）
    Q2, xi2 = _damping_ratio_from_q(800.0, False, dcfg)  # 有限层
    assert abs(xi2 - 0.0125) < 0.001, 'Vs2=800 阻尼比错误: %.6f (期望≈0.0125)' % xi2  # 容差 0.001
    print('  [PASS] Vs2=800: Q=%.1f, ξ=%.6f (期望 Q=40, ξ≈0.0125)' % (Q2, xi2))  # 输出测试结果
    # Vs1 = 400 m/s（软表层）
    Q_soft, xi_soft = _damping_ratio_from_q(400.0, False, dcfg)  # 有限层
    assert abs(xi_soft - 0.025) < 0.002, '软表层 Vs1=400 阻尼比错误: %.6f (期望≈0.025)' % xi_soft  # 容差 0.002
    print('  [PASS] 软表层 Vs1=400: Q=%.1f, ξ=%.6f (期望 Q=20, ξ≈0.025)' % (Q_soft, xi_soft))  # 输出测试结果
    # Vs1 = 1600 m/s（硬表层）
    Q_hard, xi_hard = _damping_ratio_from_q(1600.0, False, dcfg)  # 有限层
    assert abs(xi_hard - 0.00625) < 0.0005, '硬表层 Vs1=1600 阻尼比错误: %.6f (期望≈0.00625)' % xi_hard  # 容差
    print('  [PASS] 硬表层 Vs1=1600: Q=%.1f, ξ=%.6f (期望 Q=80, ξ≈0.00625)' % (Q_hard, xi_hard))  # 输出测试结果
    # 基岩 Q=999
    Q_bed, xi_bed = _damping_ratio_from_q(2000.0, True, dcfg)  # 基岩
    assert abs(xi_bed - 0.0005) < 0.0002, '基岩阻尼比错误: %.6f (期望≈0.0005)' % xi_bed  # 容差
    print('  [PASS] 基岩 Q=999: Q=%.1f, ξ=%.6f (期望 Q=999, ξ≈0.0005)' % (Q_bed, xi_bed))  # 输出测试结果
    # 验证 ξ = 1/(2Q) 恒等式
    assert abs(2.0 * xi2 * 40.0 - 1.0) < 1e-12, 'ξ=1/(2Q) 恒等式不成立'  # 数值验证
    print('  [PASS] ξ=1/(2Q) 恒等式对所有情况成立')  # 输出验证结果


def test_rayleigh_stiffness():  # 测试刚度比例阻尼（stiffness 法）
    """验证 stiffness 法：α=0，β=ξ/(π·fc)，使得在 fc 处 ξ 精确。"""
    dcfg = {'method': 'stiffness', 'f1_factor': 0.5, 'f2_factor': 2.5}  # stiffness 阻尼配置
    fc = 4.0  # 主频 4 Hz
    xi = 0.0125  # Vs2=800 对应的阻尼比
    alpha, beta = _rayleigh_coeffs(xi, dcfg, fc)  # 计算瑞利系数
    assert alpha == 0.0, 'stiffness 法 α 应为 0，实际=%.6e' % alpha  # α 必须为零
    beta_expected = xi / (math.pi * fc)  # 理论 β
    assert abs(beta - beta_expected) < 1e-12, 'stiffness 法 β 不匹配: %.6e vs %.6e' % (beta, beta_expected)  # β 验证
    # 反向验证：β·π·fc 应等于 ξ
    xi_restored = beta * math.pi * fc  # 从 β 反算 ξ
    assert abs(xi_restored - xi) < 1e-12, 'stiffness 法 ξ(fc) 不精确: %.6e vs %.6e' % (xi_restored, xi)  # 验证
    print('  [PASS] stiffness 法: α=%.2e, β=%.6e, ξ(fc)=%.6e (期望 α=0, ξ(fc)=ξ)' % (alpha, beta, xi_restored))  # 输出


def test_rayleigh_two_freq():  # 测试双频拟合（rayleigh 法）
    """验证 rayleigh 双频拟合法：在 f1 和 f2 两端 ξ 均≈1/(2Q)。"""
    dcfg = {'method': 'rayleigh', 'f1_factor': 0.5, 'f2_factor': 2.5}  # rayleigh 阻尼配置
    fc = 4.0  # 主频 4 Hz
    xi = 0.0125  # Vs2=800 对应的阻尼比
    alpha, beta = _rayleigh_coeffs(xi, dcfg, fc)  # 计算瑞利系数
    # 在 f1 = 0.5*fc = 2 Hz 处验证 ξ
    f1 = dcfg['f1_factor'] * fc  # 拟合下限频率
    w1 = 2.0 * math.pi * f1  # 对应的圆频率
    xi_at_f1 = 0.5 * (alpha / w1 + beta * w1)  # 瑞利阻尼在 f1 处的等效阻尼比
    assert abs(xi_at_f1 - xi) < 1e-4, 'rayleigh 法在 f1=%.1f Hz 处 ξ 不匹配: %.6e vs %.6e' % (f1, xi_at_f1, xi)  # 验证
    # 在 f2 = 2.5*fc = 10 Hz 处验证 ξ
    f2 = dcfg['f2_factor'] * fc  # 拟合上限频率
    w2 = 2.0 * math.pi * f2  # 对应的圆频率
    xi_at_f2 = 0.5 * (alpha / w2 + beta * w2)  # 瑞利阻尼在 f2 处的等效阻尼比
    assert abs(xi_at_f2 - xi) < 1e-4, 'rayleigh 法在 f2=%.1f Hz 处 ξ 不匹配: %.6e vs %.6e' % (f2, xi_at_f2, xi)  # 验证
    print('  [PASS] rayleigh 双频拟合: α=%.6e, β=%.6e' % (alpha, beta))  # 输出
    print('    ξ(f1=%.1f Hz)=%.6e, ξ(f2=%.1f Hz)=%.6e (期望均≈%.6e)' % (f1, xi_at_f1, f2, xi_at_f2, xi))  # 输出


def test_bedrock_zero_damping():  # 测试基岩近乎零阻尼
    """验证基岩 Q=999 → α≈0, β≈0（两种方法均验证）。"""
    dcfg_stiff = {'method': 'stiffness', 'f1_factor': 0.5, 'f2_factor': 2.5,  # stiffness 配置
                  'qs_factor': 0.05, 'q_bedrock': 999.0}  # Q 因子
    dcfg_rayl = {'method': 'rayleigh', 'f1_factor': 0.5, 'f2_factor': 2.5,  # rayleigh 配置
                 'qs_factor': 0.05, 'q_bedrock': 999.0}  # Q 因子
    fc = 4.0  # 主频
    Q, xi = _damping_ratio_from_q(2000.0, True, dcfg_stiff)  # 基岩阻尼比
    assert Q == 999.0, '基岩 Q 应=999，实际=%.1f' % Q  # Q 值验证
    # stiffness 法
    a_s, b_s = _rayleigh_coeffs(xi, dcfg_stiff, fc)  # 计算瑞利系数
    assert a_s == 0.0, '基岩 stiffness α 应为 0'  # α 为零
    assert b_s < 1e-4, '基岩 stiffness β 应≈0，实际=%.6e' % b_s  # β 接近零（Q=999→ξ≈5e-4→β≈4e-5）
    # rayleigh 法
    a_r, b_r = _rayleigh_coeffs(xi, dcfg_rayl, fc)  # 计算瑞利系数
    assert abs(a_r) < 0.015, '基岩 rayleigh α 应≈0，实际=%.6e' % a_r  # α 接近零（Q=999→α≈0.0105）
    assert abs(b_r) < 1e-4, '基岩 rayleigh β 应≈0，实际=%.6e' % b_r  # β 接近零
    print('  [PASS] 基岩 Q=999: ξ=%.6e' % xi)  # 输出
    print('    stiffness: α=%.2e, β=%.2e' % (a_s, b_s))  # 输出
    print('    rayleigh:  α=%.2e, β=%.2e (期望均≈0)' % (a_r, b_r))  # 输出


def test_resolve_damping():  # 测试阻尼配置解析
    """验证 _resolve_damping 补全 fc 字段。"""
    dcfg = {'enable': True, 'method': 'rayleigh', 'qs_factor': 0.05,  # 初始配置
            'q_bedrock': 999.0, 'fc': None, 'f1_factor': 0.5, 'f2_factor': 2.5}  # fc 为空
    resolved = _resolve_damping(dcfg, 4.0)  # 解析
    assert resolved['fc'] == 4.0, 'resolve 后 fc 应=4.0，实际=%s' % resolved['fc']  # fc 已填充
    assert resolved is not dcfg, 'resolve 应返回拷贝而非原地修改'  # 不修改原配置
    # 显式 fc 不被覆盖
    dcfg2 = dict(dcfg)  # 复制配置
    dcfg2['fc'] = 5.5  # 显式指定 fc
    resolved2 = _resolve_damping(dcfg2, 4.0)  # 解析
    assert resolved2['fc'] == 5.5, '显式 fc 不应被覆盖，实际=%s' % resolved2['fc']  # 不覆盖
    print('  [PASS] _resolve_damping: fc 补全正确，显式值不被覆盖')  # 输出


# ==========================================================
#  主入口
# ==========================================================


def main():  # 测试主入口
    """运行全部阻尼换算单测。"""
    print('=' * 60)  # 分隔线
    print('阻尼换算纯 Python 单测')  # 标题
    print('=' * 60)  # 分隔线
    passed = 0  # 通过计数
    failed = 0  # 失败计数
    tests = [  # 测试列表
        ('_estimate_dominant_freq', test_estimate_dominant_freq),  # 主频估计
        ('_damping_ratio_from_q', test_damping_ratio_from_q),  # Q→ξ 换算
        ('_rayleigh_coeffs (stiffness)', test_rayleigh_stiffness),  # stiffness 法
        ('_rayleigh_coeffs (rayleigh)', test_rayleigh_two_freq),  # rayleigh 双频法
        ('基岩零阻尼', test_bedrock_zero_damping),  # 基岩阻尼
        ('_resolve_damping', test_resolve_damping),  # 阻尼解析
    ]  # 结束测试列表
    for name, fn in tests:  # 遍历测试
        try:  # 捕获异常
            fn()  # 运行测试
            passed += 1  # 通过计数
        except AssertionError as e:  # 断言失败
            print('  [FAIL] %s 失败: %s' % (name, str(e)))  # 输出失败信息
            failed += 1  # 失败计数
        except Exception as e:  # 其他异常
            print('  [FAIL] %s 异常: %s' % (name, str(e)))  # 输出异常信息
            failed += 1  # 失败计数
    print('-' * 60)  # 分隔线
    print('结果: %d/%d 通过, %d 失败' % (passed, len(tests), failed))  # 输出统计
    print('=' * 60)  # 分隔线
    return failed == 0  # 返回是否全部通过


if __name__ == '__main__':  # 直接运行
    success = main()  # 运行测试
    sys.exit(0 if success else 1)  # 按测试结果返回退出码
