import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

# ==========================================
# 0. 可配置参数（集中在顶部）
# ==========================================
# 可批量输入多个地震波 txt 文件
# 示例：
# FILE_PATHS = [
#     r'C:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\Northridge.txt',
#     r'C:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\Kobe.txt',
# ]
FILE_PATHS = [
    r'C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Original\El_Centro.txt',
    r'C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Original\Northridge.txt',
    r'C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Original\Loma_Prieta.txt',
]

# 统一输出目录：调幅后的 txt 与图表均保存到该目录
OUTPUT_DIR = r'C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled'

# True: 自动扫描当前脚本目录下所有 txt（自动排除 *_scaled.txt）
# False: 使用 FILE_PATHS 中手动给出的路径
AUTO_SCAN_TXT = False
TARGET_PGA = 0.30  # 目标峰值加速度，单位：g

# 颜色序列：不同 txt 文件按顺序轮换颜色，同一文件三张图颜色一致
COLOR_CYCLE = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange']

# 图1：加速度时程图边界
ACCEL_XLIM = (0, 30)
ACCEL_YLIM = (-0.30, 0.30)

# 图2：傅里叶振幅谱边界
FFT_XLIM = (0, 15)
FFT_YLIM = (0.00, 0.30)

# 图3：反应谱边界
PSA_XLIM = (1e-2, 1e1)
PSA_YLIM = (0.00, 1.00)

def load_ground_motion(file_path):
    """读取地震波数据：第一列时间，第二列加速度。"""
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Time', 'Acceleration'])
    df = df[pd.to_numeric(df['Time'], errors='coerce').notna()]
    df = df[pd.to_numeric(df['Acceleration'], errors='coerce').notna()]

    t = df['Time'].astype(float).values
    accel = df['Acceleration'].astype(float).values

    if len(t) < 2:
        raise ValueError("地震动数据点不足，至少需要2个数据点。")

    # 保证时间递增，避免采样间隔计算异常
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    accel = accel[sort_idx]

    # 去重并避免 dt 为 0
    unique_mask = np.insert(np.diff(t) > 0, 0, True)
    t = t[unique_mask]
    accel = accel[unique_mask]

    if len(t) < 2:
        raise ValueError("时间序列存在重复或无效点，无法计算采样间隔。")

    return t, accel


def get_input_files():
    """获取待处理的 txt 文件列表。"""
    if not AUTO_SCAN_TXT:
        if not FILE_PATHS:
            raise ValueError("AUTO_SCAN_TXT=False 时，FILE_PATHS 不能为空。")
        return FILE_PATHS

    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_files = []
    for name in os.listdir(script_dir):
        full_path = os.path.join(script_dir, name)
        if not os.path.isfile(full_path):
            continue

        stem, ext = os.path.splitext(name)
        if ext.lower() != '.txt':
            continue

        # 自动排除已处理后的 scaled 文件
        if '_scaled' in stem.lower():
            continue

        txt_files.append(full_path)

    txt_files.sort()
    if not txt_files:
        raise ValueError(f"目录中未找到可处理的 txt 文件：{script_dir}")

    print(f"自动扫描目录：{script_dir}")
    print(f"共找到 {len(txt_files)} 个待处理 txt 文件（已排除 _scaled）。")
    return txt_files


def lowpass_and_scale(t, accel, target_pga, cutoff_freq=15.0):
    """先低通滤波，再按目标 PGA 调幅。"""
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist

    accel_filtered = accel.copy()
    if normal_cutoff < 1.0:
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        accel_filtered = signal.filtfilt(b, a, accel_filtered)
    else:
        print(
            f"警告：奈奎斯特频率 ({nyquist:.2f} Hz) 小于等于截止频率 "
            f"({cutoff_freq} Hz)，跳过低通滤波。"
        )

    pga_original = np.max(np.abs(accel_filtered))
    if pga_original == 0:
        raise ValueError("加速度数据全为零，无法进行调幅。")

    scale_factor = target_pga / pga_original
    accel_scaled = accel_filtered * scale_factor

    # 保证正峰值为主峰值（便于统一观感）
    if np.max(accel_scaled) < np.abs(np.min(accel_scaled)):
        accel_scaled = -accel_scaled

    return accel_scaled


def calc_fft(t, accel):
    """计算单边傅里叶振幅谱。"""
    dt_raw = np.diff(t)
    dt = np.median(dt_raw)

    is_uniform = np.allclose(dt_raw, dt, rtol=1e-4, atol=1e-8)
    if is_uniform:
        t_uniform = t
        accel_uniform = accel
    else:
        t_uniform = np.arange(t[0], t[-1] + 0.5 * dt, dt)
        accel_uniform = np.interp(t_uniform, t, accel)

    n_uniform = len(t_uniform)
    # 去均值避免 0 Hz (直流分量) 出现异常大峰值
    accel_detrended = accel_uniform - np.mean(accel_uniform)
    # 计算实数 FFT
    dft_coeff = np.fft.rfft(accel_detrended)
    positive_frequencies = np.fft.rfftfreq(n_uniform, d=dt)
    # 地震工程标准计算方法，乘以时间步长 dt
    positive_amplitudes = np.abs(dft_coeff) * dt
    # //if n_uniform % 2 == 0:
    # //    positive_amplitudes[1:-1] *= 2.0
    # //else:
    # //    positive_amplitudes[1:] *= 2.0

    return dt, accel_detrended, positive_frequencies, positive_amplitudes


def calc_response_spectrum(accel_detrended, dt, damping=0.05):
    """基于 Newmark-β 法计算 5% 阻尼弹性加速度反应谱。"""
    periods = np.logspace(-2, 1, 500)
    sa = np.zeros(len(periods))

    gamma = 0.5
    beta = 0.25

    ag = accel_detrended
    n_rs = len(ag)

    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma)
    a7 = gamma * dt

    for i, period in enumerate(periods):
        if period <= dt:
            sa[i] = np.max(np.abs(ag))
            continue

        m = 1.0
        omega_n = 2.0 * np.pi / period
        k = m * omega_n**2
        c = 2.0 * damping * omega_n * m
        k_eff = k + a0 * m + a1 * c

        u = 0.0
        v = 0.0
        a_rel = -(c * v + k * u) / m - ag[0]
        max_abs_acc = abs(a_rel + ag[0])

        for j in range(n_rs - 1):
            p_eff = (
                -m * ag[j + 1]
                + m * (a0 * u + a2 * v + a3 * a_rel)
                + c * (a1 * u + a4 * v + a5 * a_rel)
            )

            u_next = p_eff / k_eff
            a_next = a0 * (u_next - u) - a2 * v - a3 * a_rel
            v_next = v + a6 * a_rel + a7 * a_next

            abs_acc = abs(a_next + ag[j + 1])
            if abs_acc > max_abs_acc:
                max_abs_acc = abs_acc

            u, v, a_rel = u_next, v_next, a_next

        sa[i] = max_abs_acc

    return periods, sa


def save_plots(t, accel, label_name, freqs, amps, periods, sa, line_color, output_dir):
    """分别保存三张图，并额外保存一张三联图，不弹出窗口。"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # 图1：加速度时程（单独成图）
    fig1, ax1 = plt.subplots(figsize=(6.0, 4.5))
    ax1.plot(t, accel, color=line_color, linewidth=1.0, label=label_name)
    ax1.set_title('Acceleration, g', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_xlim(*ACCEL_XLIM)
    ax1.set_xticks(np.arange(0, 35, 5))
    ax1.set_ylim(*ACCEL_YLIM)
    ax1.set_yticks([-0.30, -0.15, 0.00, 0.15, 0.30])
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax1.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    legend1 = ax1.legend(loc='upper right', frameon=True, fontsize=11)
    legend1.get_frame().set_edgecolor('black')
    ax1.tick_params(direction='in', top=True, right=True, labelsize=11)
    fig1.tight_layout()
    accel_fig_path = os.path.join(output_dir, f'{label_name}_acceleration.png')
    fig1.savefig(accel_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 图2：傅里叶振幅谱（单独成图）
    fig2, ax2 = plt.subplots(figsize=(6.0, 4.5))
    ax2.plot(freqs, amps, color=line_color, linewidth=1.0, label=label_name)
    ax2.set_title('Fourier Amplitude Spectrum', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Fourier Amplitude (g$\\cdot$s)', fontsize=12)
    ax2.set_xlim(*FFT_XLIM)
    ax2.set_xticks(np.arange(0, FFT_XLIM[1] + 1, 5))
    max_amp = np.max(amps[freqs <= FFT_XLIM[1]]) # 只看 15Hz 以内的最大值
    if max_amp <= 0.20:
        ax2.set_ylim(0.00, 0.20)
        ax2.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    else:
        ax2.set_ylim(0.00, 0.30)
        ax2.set_yticks([0.00, 0.10, 0.20, 0.30])
    # //ax2.set_ylim(*FFT_YLIM)
    ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
    legend2 = ax2.legend(loc='upper right', frameon=True, fontsize=11)
    legend2.get_frame().set_edgecolor('black')
    ax2.tick_params(direction='in', top=True, right=True, labelsize=11)
    fig2.tight_layout()
    fft_fig_path = os.path.join(output_dir, f'{label_name}_fft.png')
    fig2.savefig(fft_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 图3：弹性加速度反应谱（单独成图）
    fig3, ax3 = plt.subplots(figsize=(6.0, 4.5))
    ax3.plot(periods, sa, color=line_color, linewidth=1.2, label=label_name)
    ax3.set_title('Elastic Acceleration Response Spectrum', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Spectral Period T (s)', fontsize=12)
    ax3.set_ylabel('Spectral Acceleration Sa (g)', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_xlim(*PSA_XLIM)
    sa_max = np.max(sa)
    if sa_max > 1.0:
        ax3.set_ylim(0.00, 1.25)
        ax3.set_yticks([0, 0.25, 0.50, 0.75, 1.00, 1.25])
    else:
        ax3.set_ylim(*PSA_YLIM)
    ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    ax3.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.8)
    ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)
    legend3 = ax3.legend(loc='upper right', frameon=True, fontsize=11)
    legend3.get_frame().set_edgecolor('black')
    ax3.tick_params(direction='in', which='both', top=True, right=True, labelsize=11)
    fig3.tight_layout()
    psa_fig_path = os.path.join(output_dir, f'{label_name}_response_spectrum.png')
    fig3.savefig(psa_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 三联图：将三个子图放在同一张图中输出
    fig_all, axes = plt.subplots(1, 3, figsize=(18, 4.5))  # 创建 1x3 组合图画布
    ax1_all, ax2_all, ax3_all = axes  # 解包三个子图坐标轴

    ax1_all.plot(t, accel, color=line_color, linewidth=1.0, label=label_name)  # 绘制加速度时程曲线
    ax1_all.set_title('Acceleration, g', fontweight='bold', fontsize=14)  # 设置子图1标题
    ax1_all.set_xlabel('Time (s)', fontsize=12)  # 设置子图1横坐标标题
    ax1_all.set_xlim(*ACCEL_XLIM)  # 设置子图1横坐标范围
    ax1_all.set_xticks(np.arange(0, 35, 5))  # 设置子图1横坐标刻度
    ax1_all.set_ylim(*ACCEL_YLIM)  # 设置子图1纵坐标范围
    ax1_all.set_yticks([-0.30, -0.15, 0.00, 0.15, 0.30])  # 设置子图1纵坐标刻度
    ax1_all.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # 设置子图1纵坐标显示格式
    ax1_all.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)  # 设置子图1网格样式
    legend1_all = ax1_all.legend(loc='upper right', frameon=True, fontsize=11)  # 添加子图1图例
    legend1_all.get_frame().set_edgecolor('black')  # 设置子图1图例边框颜色
    ax1_all.tick_params(direction='in', top=True, right=True, labelsize=11)  # 设置子图1刻度样式

    ax2_all.plot(freqs, amps, color=line_color, linewidth=1.0, label=label_name)  # 绘制傅里叶振幅谱曲线
    ax2_all.set_title('Fourier Amplitude Spectrum', fontweight='bold', fontsize=14)  # 设置子图2标题
    ax2_all.set_xlabel('Frequency (Hz)', fontsize=12)  # 设置子图2横坐标标题
    ax2_all.set_ylabel('Fourier Amplitude (g$\\cdot$s)', fontsize=12)  # 设置子图2纵坐标标题
    ax2_all.set_xlim(*FFT_XLIM)  # 设置子图2横坐标范围
    ax2_all.set_xticks(np.arange(0, FFT_XLIM[1] + 1, 5))  # 设置子图2横坐标刻度
    max_amp_all = np.max(amps[freqs <= FFT_XLIM[1]])  # 计算 15Hz 以内的最大振幅
    if max_amp_all <= 0.20:  # 判断是否采用较小纵坐标范围
        ax2_all.set_ylim(0.00, 0.20)  # 设置子图2纵坐标范围为 0~0.20
        ax2_all.set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])  # 设置子图2纵坐标刻度为 0.05 间隔
    else:  # 否则采用较大纵坐标范围
        ax2_all.set_ylim(0.00, 0.30)  # 设置子图2纵坐标范围为 0~0.30
        ax2_all.set_yticks([0.00, 0.10, 0.20, 0.30])  # 设置子图2纵坐标刻度为 0.10 间隔
    ax2_all.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # 设置子图2纵坐标显示格式
    ax2_all.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)  # 设置子图2网格样式
    legend2_all = ax2_all.legend(loc='upper right', frameon=True, fontsize=11)  # 添加子图2图例
    legend2_all.get_frame().set_edgecolor('black')  # 设置子图2图例边框颜色
    ax2_all.tick_params(direction='in', top=True, right=True, labelsize=11)  # 设置子图2刻度样式

    ax3_all.plot(periods, sa, color=line_color, linewidth=1.2, label=label_name)  # 绘制反应谱曲线
    ax3_all.set_title('Elastic Acceleration Response Spectrum', fontweight='bold', fontsize=14)  # 设置子图3标题
    ax3_all.set_xlabel('Spectral Period T (s)', fontsize=12)  # 设置子图3横坐标标题
    ax3_all.set_ylabel('Spectral Acceleration Sa (g)', fontsize=12)  # 设置子图3纵坐标标题
    ax3_all.set_xscale('log')  # 将子图3横坐标设置为对数坐标
    ax3_all.set_xlim(*PSA_XLIM)  # 设置子图3横坐标范围
    sa_max_all = np.max(sa)  # 计算反应谱最大值
    if sa_max_all > 1.0:  # 判断是否需要扩大纵坐标范围
        ax3_all.set_ylim(0.00, 1.25)  # 设置子图3纵坐标范围为 0~1.25
        ax3_all.set_yticks([0, 0.25, 0.50, 0.75, 1.00, 1.25])  # 设置子图3纵坐标刻度
    else:  # 若最大值不超过 1.0
        ax3_all.set_ylim(*PSA_YLIM)  # 使用默认反应谱纵坐标范围
    ax3_all.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))  # 设置子图3纵坐标显示格式
    ax3_all.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.8)  # 设置子图3主网格样式
    ax3_all.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)  # 设置子图3次网格样式
    legend3_all = ax3_all.legend(loc='upper right', frameon=True, fontsize=11)  # 添加子图3图例
    legend3_all.get_frame().set_edgecolor('black')  # 设置子图3图例边框颜色
    ax3_all.tick_params(direction='in', which='both', top=True, right=True, labelsize=11)  # 设置子图3刻度样式

    fig_all.suptitle(label_name, fontsize=14, fontweight='bold')  # 设置组合图总标题
    fig_all.tight_layout(rect=(0, 0, 1, 0.95))  # 调整组合图布局并预留总标题空间
    combined_fig_path = os.path.join(output_dir, f'{label_name}_all_in_one.png')  # 生成组合图输出路径
    fig_all.savefig(combined_fig_path, dpi=300, bbox_inches='tight')  # 保存组合图到文件
    plt.close(fig_all)  # 关闭组合图以释放内存

    print(f"已保存图片：{accel_fig_path}")
    print(f"已保存图片：{fft_fig_path}")
    print(f"已保存图片：{psa_fig_path}")
    print(f"已保存图片：{combined_fig_path}")


def main():
    # 创建统一输出目录（若不存在则自动创建）
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_paths = get_input_files()

    for idx, file_path in enumerate(file_paths):
        print(f"\n开始处理：{file_path}")
        if not os.path.exists(file_path):
            print(f"跳过：找不到文件 {file_path}")
            continue

        try:
            t, accel = load_ground_motion(file_path)
            accel = lowpass_and_scale(t, accel, TARGET_PGA, cutoff_freq=15.0)
            dt, accel_detrended, freqs, amps = calc_fft(t, accel)
            periods, sa = calc_response_spectrum(accel_detrended, dt, damping=0.05)

            base_name = os.path.splitext(os.path.basename(file_path))[0]
            scaled_file_path = os.path.join(OUTPUT_DIR, base_name + '_scaled.txt')
            pd.DataFrame({'Time': t, 'Acceleration': accel}).to_csv(
                scaled_file_path,
                sep='\t',
                index=False,
                header=False,
                float_format='%.8f',
            )
            print(f"已保存调幅+滤波时程：{scaled_file_path}")

            label_name = base_name
            line_color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
            save_plots(t, accel, label_name, freqs, amps, periods, sa, line_color, OUTPUT_DIR)

        except Exception as exc:
            print(f"处理失败：{file_path} -> {exc}")

    # 仅保存图片，不弹出绘图窗口


if __name__ == '__main__':
    main()