# -*- coding: utf-8 -*-
"""
Postprocess_PGA_v3.py 频域核心单元测试（纯 Python + numpy，无需 Abaqus 环境）。

验证四件事：
1) resample_to_uniform：不等间隔采样能正确线性重采样到均匀网格；
2) single_sided_fft：单一正弦的单边幅值谱峰值落在正确频率；
3) transfer_function：由 num=H_true*ref 构造的谱能精确反演出 H_true，且低能量频点被掩膜；
4) 端到端：Ricker 输入经已知 SDOF 传递函数得到输出，pipeline 反演 H 与解析解一致。

运行：python test_transfer_function_v3.py
"""

import os  # 路径
import sys  # 修改导入路径
import numpy as np  # 数值

# 把上级目录（含 Postprocess_PGA_v3.py）加入导入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Postprocess', 'General'))  # 指向 General/
import Postprocess_PGA_v3 as p3  # 被测模块


def test_resample():
    """不等间隔采样重采样到均匀网格：分段线性函数下线性插值应精确，正弦下应合理逼近。"""
    # (a) 线性函数：线性插值在数学上精确，误差应接近机器精度，直接校验实现正确性
    g_lin = lambda t: 0.5 * t + 0.3  # 线性测试函数
    t_irr = np.sort(np.r_[np.linspace(0, 2, 40), np.random.RandomState(0).rand(20) * 2])  # 不等间隔时间
    t_uni, dt, y_uni = p3.resample_to_uniform(t_irr, g_lin(t_irr).reshape(1, -1), n_out=200)  # 重采样
    err_lin = np.max(np.abs(y_uni[0] - g_lin(t_uni)))  # 与解析值偏差
    assert dt > 0 and t_uni.size == 200, 'resample 网格异常'  # 网格检查
    assert err_lin < 1e-12, '线性函数重采样误差过大: %.3e' % err_lin  # 精确性检查
    # (b) 多点正弦：线性插值有二阶误差，仅作合理性检查（点足够密时误差小）
    g_sin = lambda t: np.sin(2 * np.pi * 1.3 * t)  # 正弦测试函数
    t_dense = np.linspace(0, 2, 400)  # 较密的不等间隔基础（此处用密采样保证逼近）
    _, _, y_sin = p3.resample_to_uniform(t_dense, g_sin(t_dense).reshape(1, -1), n_out=300)  # 重采样
    _, _, t_chk = p3.resample_to_uniform(t_dense, t_dense.reshape(1, -1), n_out=300)  # 对应时间
    err_sin = np.max(np.abs(y_sin[0] - g_sin(t_chk[0])))  # 偏差
    assert err_sin < 5e-3, '正弦重采样误差过大: %.3e' % err_sin  # 合理性检查
    print('  [OK] test_resample, 线性精确误差=%.2e, 正弦逼近误差=%.2e' % (err_lin, err_sin))  # 通过


def test_fft_peak():
    """单一正弦的单边幅值谱峰值应落在该正弦频率。"""
    dt = 0.002  # 采样间隔 (500 Hz)
    N = 4096  # 采样点
    t = np.arange(N) * dt  # 时间轴
    f0 = 3.7  # 正弦频率
    y = np.sin(2 * np.pi * f0 * t)  # 测试正弦
    freqs, spec = p3.single_sided_fft(y.reshape(1, -1), dt, window=True, detrend=True)  # FFT
    f_peak = freqs[int(np.argmax(np.abs(spec[0])))]  # 峰值频率
    assert abs(f_peak - f0) < (freqs[1] - freqs[0]) * 2, '峰值频率偏离: %.3f vs %.3f' % (f_peak, f0)  # 检查
    print('  [OK] test_fft_peak, 峰值频率=%.3fHz (真值 %.3fHz)' % (f_peak, f0))  # 通过


def test_transfer_recovery():
    """num = H_true * ref，transfer_function 应精确反演 H_true，并掩膜低能量频点。"""
    nfreq = 256  # 频点数
    rng = np.random.RandomState(1)  # 随机种子
    ref = (rng.randn(nfreq) + 1j * rng.randn(nfreq))  # 参考谱（复数）
    ref[200:] *= 1e-6  # 制造高频低能量段（应被掩膜）
    H_true = (rng.randn(3, nfreq) + 1j * rng.randn(3, nfreq))  # 3 个节点的真值传递函数
    num = H_true * ref  # 构造分子谱
    H, mask = p3.transfer_function(num, ref, energy_mask_ratio=0.01)  # 反演
    rel = np.abs(H[:, mask] - H_true[:, mask]) / (np.abs(H_true[:, mask]) + 1e-12)  # 可靠频点相对误差
    assert np.nanmax(rel) < 1e-9, '传递函数反演误差过大: %.3e' % np.nanmax(rel)  # 精确性
    assert np.all(np.isnan(H[:, ~mask])), '低能量频点未被置 NaN'  # 掩膜检查
    assert mask[200:].sum() == 0, '低能量段未被判为不可靠'  # 低能量段应全部掩膜
    print('  [OK] test_transfer_recovery, 可靠频点最大相对误差=%.2e, 掩膜频点数=%d' %
          (np.nanmax(rel), int((~mask).sum())))  # 通过


def test_end_to_end_sdof():
    """Ricker 输入经已知 SDOF 传递函数得到输出，pipeline 反演 H 与解析解一致。"""
    dt = 0.002  # 采样间隔
    N = 8192  # 采样点
    t = np.arange(N) * dt  # 时间轴
    fc = 4.0  # Ricker 中心频率
    tp = 1.0 / fc  # 主周期
    ts = t - 2.0 * tp  # 平移使脉冲居中
    ref = (1 - 2 * (np.pi * fc * ts) ** 2) * np.exp(-(np.pi * fc * ts) ** 2)  # Ricker 子波（参考/输入）

    freqs = np.fft.rfftfreq(N, d=dt)  # 频率轴
    fn = 5.0  # SDOF 共振频率
    zeta = 0.05  # 阻尼比
    r = freqs / fn  # 频率比
    H_true = 1.0 / (1.0 - r ** 2 + 2j * zeta * r)  # 解析 SDOF 传递函数
    num = np.fft.irfft(H_true * np.fft.rfft(ref), n=N)  # 输出时程 = 反变换(H*REF)

    # pipeline：不加窗/不去趋势以保证可逆精确（窗仅用于抑制泄漏，会引入近似）
    fz, spec_ref = p3.single_sided_fft(ref.reshape(1, -1), dt, window=False, detrend=False)  # 参考谱
    _, spec_num = p3.single_sided_fft(num.reshape(1, -1), dt, window=False, detrend=False)  # 输出谱
    H, mask = p3.transfer_function(spec_num, spec_ref[0], energy_mask_ratio=0.02)  # 反演 H

    band = p3.band_limit(fz, 1.0, 10.0)  # 1~10Hz 带（Ricker 主能量段）
    sel = band & mask  # 带内且可靠
    rel = np.abs(np.abs(H[0, sel]) - np.abs(H_true[sel])) / (np.abs(H_true[sel]) + 1e-12)  # 幅值相对误差
    assert np.nanmax(rel) < 1e-6, '端到端反演误差过大: %.3e' % np.nanmax(rel)  # 检查
    # 峰值应出现在共振频率附近
    f_res = fz[sel][int(np.argmax(np.abs(H[0, sel])))]  # 反演的共振峰频率
    assert abs(f_res - fn) < 0.2, '共振峰频率偏离: %.2f vs %.2f' % (f_res, fn)  # 检查
    print('  [OK] test_end_to_end_sdof, 带内最大相对误差=%.2e, 共振峰=%.2fHz (真值 %.2fHz)' %
          (np.nanmax(rel), f_res, fn))  # 通过


if __name__ == '__main__':
    print('运行 Postprocess_PGA_v3 频域核心单元测试...')  # 提示
    test_resample()  # 1
    test_fft_peak()  # 2
    test_transfer_recovery()  # 3
    test_end_to_end_sdof()  # 4
    print('全部测试通过。')  # 汇总
