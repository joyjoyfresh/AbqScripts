# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""探针脚本2：用真实 6Hz Ricker 走完整自由场链路，输出地表时域响应与 PGA，并扫覆盖层厚度。

目的：判断 v7 自由场引擎产生的地表运动是否干净（无非物理高频毛刺），
以及"地表 PGA 随覆盖层厚度"是否光滑（光滑=引擎正常；erratic=引擎有 bug）。
"""

import os  # 导入操作系统接口
import sys  # 导入系统模块
import types  # 导入类型模块
import math  # 导入数学模块
import numpy as np  # 导入数值计算库
import matplotlib  # 导入绘图库
matplotlib.use('Agg')  # 使用无界面后端
import matplotlib.pyplot as plt  # 导入绘图接口


def _install_abaqus_stubs():  # 安装 Abaqus 桩模块
    for name in ('abaqus', 'abaqusConstants', 'regionToolset', 'caeModules', 'mesh'):  # 遍历模块名
        sys.modules[name] = types.ModuleType(name)  # 注册空桩
    sys.modules['abaqus'].mdb = types.SimpleNamespace()  # 提供 mdb
    sys.modules['regionToolset'].Region = object  # 提供 Region


def _load_v7_module():  # 加载 v7 模块
    here = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    src = os.path.normpath(os.path.join(here, '..', '..', 'Modeling', 'Multi', 'VAB_oblique_TAF_double_v7.py'))  # v7 源文件
    import importlib.util  # 导入加载工具
    spec = importlib.util.spec_from_file_location('v7mod', src)  # 模块规范
    mod = importlib.util.module_from_spec(spec)  # 创建模块
    spec.loader.exec_module(mod)  # 执行加载
    return mod  # 返回模块


def main():  # 主流程
    _install_abaqus_stubs()  # 打桩
    v7 = _load_v7_module()  # 加载 v7

    # ===== 材料与入射 =====
    mat_bedrock = v7._compute_material_params(cs=2000.0, vv=0.3, density=2500.0)  # 基岩
    mat_overlying = v7._compute_material_params(cs=1600.0, vv=0.3, density=2500.0)  # 覆盖层
    angle_deg = 15.0  # 入射角
    p_horiz = math.sin(math.radians(angle_deg)) / mat_bedrock['cs']  # 水平慢度

    # ===== 读取真实 6Hz Ricker 加速度并积分为速度 =====
    here = os.path.dirname(os.path.abspath(__file__))  # 当前目录
    wave = os.path.normpath(os.path.join(here, '..', '..', 'Wave', 'Impulse', 'ricker_wavelet_6Hz.txt'))  # 波文件路径
    ACC = np.loadtxt(wave)  # 读取加速度
    time_arr = ACC[:, 0]  # 时间列
    acc = ACC[:, 1]  # 加速度列
    dt = time_arr[1] - time_arr[0]  # 步长
    vel, _slope = v7._integrate_acc_to_velocity(acc, dt, time_arr)  # 积分为速度（含基线校正）

    N_orig = len(vel)  # 原始长度
    N_fft = 1  # FFT 长度
    while N_fft < N_orig:  # 找 2 的幂
        N_fft *= 2  # 倍增
    N_fft *= 2  # 再翻倍避免混叠
    vel_padded = np.zeros(N_fft)  # 补零数组
    vel_padded[:N_orig] = vel  # 填入速度
    freq_arr = np.fft.rfftfreq(N_fft, d=dt)  # 频率轴
    vel_freq = np.fft.rfft(vel_padded)  # 速度谱

    # ===== 单点时域响应（地表柱：基岩200 + 覆盖200）=====
    y_bottom, h_bedrock, h_ov = 0.0, 200.0, 200.0  # 几何
    ff = v7._compute_freefield_at_node(  # 计算时域自由场
        y_target=y_bottom + h_bedrock + h_ov, x_target=0.0,
        mat_bedrock=mat_bedrock, mat_overlying=mat_overlying,
        h_bedrock=h_bedrock, h_overlying=h_ov, y_bottom=y_bottom,
        p_horiz=p_horiz, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=N_fft)

    t = np.arange(N_fft) * dt  # 时间轴
    dotux = ff['dotux']  # x 向速度时程
    # 由速度数值微分得加速度，检查高频毛刺
    accx = np.gradient(dotux, dt)  # x 向加速度
    print('地表柱(h_ov=200): 速度PGV_x={:.4f} m/s, 加速度PGA_x={:.4f} m/s^2'.format(
        np.max(np.abs(dotux[:N_orig])), np.max(np.abs(accx[:N_orig]))))  # 打印 PGV/PGA

    # ===== 扫覆盖层厚度，计算地表 PGV_x，检查随厚度是否光滑 =====
    h_list = np.arange(2.0, 400.0 + 1e-9, 2.0)  # 覆盖层厚度序列（2~400m）
    pgv = np.zeros_like(h_list)  # 存 PGV
    for i, hh in enumerate(h_list):  # 遍历厚度
        v7._FF_TRANSFER_CACHE.clear()  # 清缓存避免键冲突
        ffi = v7._compute_freefield_at_node(  # 计算该厚度地表响应
            y_target=y_bottom + h_bedrock + hh, x_target=0.0,
            mat_bedrock=mat_bedrock, mat_overlying=mat_overlying,
            h_bedrock=h_bedrock, h_overlying=hh, y_bottom=y_bottom,
            p_horiz=p_horiz, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt, N_fft=N_fft)
        pgv[i] = np.max(np.abs(ffi['dotux'][:N_orig]))  # 记录 PGV_x

    # 相邻厚度 PGV 跳变（刻画 erratic 程度）
    rel_jump = np.max(np.abs(np.diff(pgv))) / np.median(pgv)  # 最大相对跳变
    print('PGV_x(h_ov) 扫描: 中位={:.4f}, 最小={:.4f}, 最大={:.4f}, 相邻最大相对跳变={:.2%}'.format(
        np.median(pgv), pgv.min(), pgv.max(), rel_jump))  # 打印统计

    # ===== 出图 =====
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # 三联子图
    axs[0].plot(t[:N_orig], dotux[:N_orig])  # 地表速度时程
    axs[0].set_title('Surface dotux (h_ov=200)')  # 标题
    axs[0].set_xlabel('t (s)'); axs[0].set_ylabel('v (m/s)')  # 轴标签
    axs[1].plot(freq_arr, np.abs(np.fft.rfft(dotux)))  # 地表速度时程的频谱幅值
    axs[1].set_xlim(0, 40); axs[1].set_title('|dotux spectrum|')  # 频域
    axs[1].set_xlabel('f (Hz)')  # 轴标签
    axs[2].plot(h_list, pgv)  # PGV vs 厚度
    axs[2].set_title('Surface PGV_x vs overlying thickness')  # 标题
    axs[2].set_xlabel('h_overlying (m)'); axs[2].set_ylabel('PGV_x (m/s)')  # 轴标签
    fig.tight_layout()  # 紧凑布局
    out = os.path.join(here, 'probe_timedomain_v1.png')  # 输出图路径
    fig.savefig(out, dpi=130)  # 保存图
    print('图已保存:', out)  # 打印


if __name__ == '__main__':  # 入口
    main()  # 运行
