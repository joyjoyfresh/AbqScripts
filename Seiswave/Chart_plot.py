import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. 可配置参数（集中在顶部）
# ==========================================
# *请将 'elcentro.csv' 替换为你的实际文件路径，例如 'C:/data/elcentro.txt'
FILE_PATH = r'C:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\El_Centro.txt'

# 线条颜色（可使用颜色名、十六进制、RGB元组等）
COLOR = 'red'  # 统一颜色
ACCEL_LINE_COLOR = COLOR
FFT_LINE_COLOR = COLOR
PSA_LINE_COLOR = COLOR

# 图1：加速度时程图边界
ACCEL_XLIM = (0, 30)
ACCEL_YLIM = (-0.30, 0.30)

# 图2：傅里叶振幅谱边界
FFT_XLIM = (0, 15)
FFT_YLIM = (0.00, 0.30)

# 图3：反应谱边界
PSA_XLIM = (1e-2, 1e1)
PSA_YLIM = (0.00, 1.00)

# ==========================================
# 1. 读取地震波数据
# ==========================================
file_path = FILE_PATH

try:
    # 无表头TXT：第一列时间，第二列加速度
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=['Time', 'Acceleration'])

    # 兼容可能混入的表头行（例如 Time Acceleration）
    df = df[pd.to_numeric(df['Time'], errors='coerce').notna()]
    df = df[pd.to_numeric(df['Acceleration'], errors='coerce').notna()]

    t = df['Time'].astype(float).values
    accel = df['Acceleration'].astype(float).values

except FileNotFoundError:
    print(f"找不到文件: {file_path}，请检查路径是否正确。")
except KeyError as e:
    print(f"数据列名错误: 找不到列名 {e}。")
    t, accel = [], []

if len(t) < 2:
    raise ValueError("地震动数据点不足，至少需要2个数据点。")

# 保证时间递增，避免采样间隔计算异常
sort_idx = np.argsort(t)
t = t[sort_idx]
accel = accel[sort_idx]

# ==========================================
# 2. 绘制加速度时程图表
# ==========================================
if len(t) > 0:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(t, accel, color=ACCEL_LINE_COLOR, linewidth=1.0, label='El-Centro')

    ax.set_title('Acceleration, g', fontweight='bold', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)

    ax.set_xlim(*ACCEL_XLIM)
    ax.set_xticks(np.arange(0, 35, 5))

    ax.set_ylim(*ACCEL_YLIM)
    ax.set_yticks([-0.30, -0.15, 0.00, 0.15, 0.30])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

    ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

    legend = ax.legend(loc='upper right', frameon=True, fontsize=11)
    legend.get_frame().set_edgecolor('black')

    ax.tick_params(direction='in', top=True, right=True, labelsize=11)

    plt.tight_layout()

# ==========================================
# 3. 执行快速傅里叶变换 (FFT)
# ==========================================
# 计算采样间隔 (dt) 和数据点数 (N)
dt = np.median(np.diff(t))
N = len(t)

# 去除均值可减小直流分量对频谱与动力积分的干扰
accel_detrended = accel - np.mean(accel)

# 执行单边 FFT
fft_result = np.fft.rfft(accel_detrended)
positive_frequencies = np.fft.rfftfreq(N, d=dt)

# 计算归一化单边振幅谱
positive_amplitudes = np.abs(fft_result) / N
if N % 2 == 0:
    # 偶数点：除直流和Nyquist外乘2
    if len(positive_amplitudes) > 2:
        positive_amplitudes[1:-1] *= 2
else:
    # 奇数点：除直流外乘2
    if len(positive_amplitudes) > 1:
        positive_amplitudes[1:] *= 2


# ==========================================
# 4. 绘制频域傅里叶振幅谱图
# ==========================================
# 设置全局字体为类似 Times New Roman 的衬线字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# 创建画布
fig, ax = plt.subplots(figsize=(7, 4))

# 绘制频域曲线
ax.plot(positive_frequencies, positive_amplitudes, color=FFT_LINE_COLOR, linewidth=1.0, label='El-Centro')

# 修改标题和坐标轴标签
ax.set_title('Fourier Amplitude', fontweight='bold', fontsize=14)
ax.set_xlabel('Frequency (Hz)', fontsize=12)

# 设置 X 轴的范围和刻度 (0 到 15 Hz)
ax.set_xlim(*FFT_XLIM)
ax.set_xticks(np.arange(0, 20, 5)) 

ax.set_ylim(*FFT_YLIM)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 设置网格线
ax.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)

# 设置图例
legend = ax.legend(loc='upper right', frameon=True, fontsize=11)
legend.get_frame().set_edgecolor('black')

# 刻度线样式微调：向内
ax.tick_params(direction='in', top=True, right=True, labelsize=11)

# 调整布局
plt.tight_layout()

# ==========================================
# 5. 计算加速度反应谱 (拟加速度谱 PSA)
# ==========================================
# 设置阻尼比，建筑结构默认通常为 5% (0.05)
damping = 0.05 

# 生成对数分布的周期数组，从 0.01秒 (10^-2) 到 10秒 (10^1)
periods = np.logspace(-2, 1, 500) 
psa = np.zeros(len(periods))

print("正在进行结构动力学求解（杜哈梅积分），请稍候...")

# Newmark-beta 参数（平均加速度法）
gamma = 0.5
beta = 0.25

# 反应谱计算使用去均值加速度，减少低频漂移误差
ag = accel_detrended

for i, T in enumerate(periods):
    if T < 0.01:
        # 极短周期结构，其最大加速度直接等于地面峰值加速度(PGA)
        psa[i] = np.max(np.abs(ag))
        continue

    # 单位质量体系
    m = 1.0
    omega_n = 2 * np.pi / T
    k = omega_n**2 * m
    c = 2 * damping * omega_n * m

    # 初始条件：u(0)=0, v(0)=0
    u = 0.0
    v = 0.0
    a_rel = (-ag[0] - c * v - k * u) / m
    max_u = abs(u)

    # Newmark 常数
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt * (gamma / (2.0 * beta) - 1.0)
    a6 = dt * (1.0 - gamma)
    a7 = gamma * dt

    k_eff = k + a0 * m + a1 * c

    for j in range(N - 1):
        p_next = -m * ag[j + 1]

        p_eff = (
            p_next
            + m * (a0 * u + a2 * v + a3 * a_rel)
            + c * (a1 * u + a4 * v + a5 * a_rel)
        )

        u_next = p_eff / k_eff
        a_next = a0 * (u_next - u) - a2 * v - a3 * a_rel
        v_next = v + a6 * a_rel + a7 * a_next

        u = u_next
        v = v_next
        a_rel = a_next

        if abs(u) > max_u:
            max_u = abs(u)

    # PSA = omega_n^2 * Sd
    psa[i] = omega_n**2 * max_u

print("计算完成！正在绘图...")

# ==========================================
# 6. 绘制反应谱图
# ==========================================
# 设置全局字体为类似 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, ax = plt.subplots(figsize=(7, 4))

# 绘制折线
ax.plot(periods, psa, color=PSA_LINE_COLOR, linewidth=1.2, label='El-Centro')

# 设置标题和标签
ax.set_title('Spectral Acceleration, g', fontweight='bold', fontsize=14)
ax.set_xlabel('Spectral Period (s)', fontsize=12)

# ========== 核心设置：对数坐标轴 ==========
ax.set_xscale('log')
ax.set_xlim(*PSA_XLIM)

ax.set_ylim(*PSA_YLIM)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 设置网格线（对数坐标轴需要分别设置主网格和次网格）
# 主网格线（实线）
ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='gray', alpha=0.8)
# 次网格线（虚线，对应对数刻度中间的竖线）
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

# 设置图例
legend = ax.legend(loc='upper right', frameon=True, fontsize=11)
legend.get_frame().set_edgecolor('black')

# 刻度线微调
ax.tick_params(direction='in', which='both', top=True, right=True, labelsize=11)

plt.tight_layout()
plt.show()