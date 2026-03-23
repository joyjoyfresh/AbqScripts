import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. 可配置参数（集中在顶部）
# ==========================================
# *请将 'elcentro.csv' 替换为你的实际文件路径，例如 'C:/data/elcentro.txt'
FILE_PATH = r'C:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\Northridge.txt'
TARGET_PGA = 0.30  # 目标峰值加速度，单位：g

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
label_name = file_path.split('/')[-1].split('.')[0]  # 从文件名提取标签（不含路径和扩展名）
# ==========================================
# 2. 地震波调幅处理（Scale to Target PGA）
# ==========================================
# 计算原始峰值加速度（取绝对值最大值）
pga_original = np.max(np.abs(accel))

if pga_original == 0:
    raise ValueError("加速度数据全为零，无法进行调幅。")

# 调幅系数
scale_factor = TARGET_PGA / pga_original
accel = accel * scale_factor

# 确保最大加速度为正数：若峰值（代数最大值）为负，则整体取反
if np.max(accel) < np.abs(np.min(accel)):
    accel = -accel

# 将调幅后的时程保存为新的 txt 文件（与原文件同目录）
import os
_base, _ext = os.path.splitext(FILE_PATH)
SCALED_FILE_PATH = _base + '_scaled' + _ext
scaled_df = pd.DataFrame({'Time': t, 'Acceleration': accel})
scaled_df.to_csv(SCALED_FILE_PATH, sep='\t', index=False, header=False,
                 float_format='%.8f')

# ==========================================
# 3. 绘制加速度时程图表
# ==========================================
if len(t) > 0:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(t, accel, color=ACCEL_LINE_COLOR, linewidth=1.0, label=label_name)

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
# 3. 对时程信号执行离散傅里叶变换 (DFT)
# ==========================================
# 计算采样间隔（取中位数以应对浮点误差），并确定均匀网格
dt_raw = np.diff(t)
dt = np.median(dt_raw)

# 若时间间隔并非严格等步长，先插值到均匀网格再做 DFT
is_uniform = np.allclose(dt_raw, dt, rtol=1e-4, atol=1e-8)
if is_uniform:
    t_uniform   = t
    accel_uniform = accel
else:
    t_uniform   = np.arange(t[0], t[-1] + 0.5 * dt, dt)
    accel_uniform = np.interp(t_uniform, t, accel)

N_uniform = len(t_uniform)

# 去均值（去除直流分量），避免 DC 峰值主导低频段
accel_detrended = accel_uniform - np.mean(accel_uniform)

# 利用 rfft 对实数序列做单边 DFT，得到正频率系数
dft_coeff           = np.fft.rfft(accel_detrended)
positive_frequencies = np.fft.rfftfreq(N_uniform, d=dt)  # 单位：Hz

# 将双边谱折叠为单边谱：
#   - 直流项 (index 0) 和 Nyquist 项（N 为偶数时的最后一项）不乘 2
#   - 其余正频率项乘 2，以保持能量守恒
positive_amplitudes = np.abs(dft_coeff) * dt  # 乘以 dt 使量纲为 g·s
if N_uniform % 2 == 0:
    positive_amplitudes[1:-1] *= 2.0
else:
    positive_amplitudes[1:]   *= 2.0


# ==========================================
# 4. 绘制傅里叶振幅谱（图2）
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']

fig2, ax2 = plt.subplots(figsize=(7, 4))

# 绘制频域曲线
ax2.plot(positive_frequencies, positive_amplitudes,
         color=FFT_LINE_COLOR, linewidth=1.0, label=label_name)

# 标题与坐标标签
ax2.set_title('Fourier Amplitude Spectrum', fontweight='bold', fontsize=14)
ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Fourier Amplitude (g$\cdot$s)', fontsize=12)

# X 轴范围与刻度
ax2.set_xlim(*FFT_XLIM)
ax2.set_xticks(np.arange(0, FFT_XLIM[1] + 1, 5))

# Y 轴范围与格式
ax2.set_ylim(*FFT_YLIM)
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 网格、图例、刻度
ax2.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.5)
legend2 = ax2.legend(loc='upper right', frameon=True, fontsize=11)
legend2.get_frame().set_edgecolor('black')
ax2.tick_params(direction='in', top=True, right=True, labelsize=11)

fig2.tight_layout()

# ==========================================
# 5. 计算弹性加速度反应谱 (Sa，阻尼比 5%)
# ==========================================
# 原理：将地震加速度时程 ag 作为基础激励，逐一输入自振周期为 T、
#       阻尼比为 5% 的单自由度 (SDOF) 弹性系统，
#       记录每个系统的最大绝对加速度响应作为该周期对应的谱加速度 Sa(T)。

damping = 0.05                        # 阻尼比 ξ = 5%
periods = np.logspace(-2, 1, 500)     # 自振周期范围：0.01 s ~ 10 s（对数均匀分布）
sa      = np.zeros(len(periods))      # 存储各周期对应的谱加速度

# 使用 Newmark-β 法（平均加速度法，无条件稳定）进行逐步时程积分
# γ = 0.5，β = 0.25 → 平均加速度假设，二阶精度
gamma = 0.5
beta  = 0.25

# 地震动加速度时程（已去均值），基于均匀采样网格
ag   = accel_detrended
N_rs = len(ag)

# 预计算 Newmark 常数（仅与 dt 有关，所有周期共用）
a0 = 1.0 / (beta * dt**2)
a1 = gamma / (beta * dt)
a2 = 1.0 / (beta * dt)
a3 = 1.0 / (2.0 * beta) - 1.0
a4 = gamma / beta - 1.0
a5 = dt * (gamma / (2.0 * beta) - 1.0)
a6 = dt * (1.0 - gamma)
a7 = gamma * dt

for i, T in enumerate(periods):
    # 当周期极短（趋近于刚体），Sa 趋近于 PGA
    if T <= dt:
        sa[i] = np.max(np.abs(ag))
        continue

    # SDOF 系统参数（取单位质量 m = 1）
    m       = 1.0
    omega_n = 2.0 * np.pi / T          # 自振圆频率
    k       = m * omega_n ** 2         # 刚度
    c       = 2.0 * damping * omega_n * m  # 阻尼系数

    # 有效刚度（Newmark 隐式格式）
    k_eff = k + a0 * m + a1 * c

    # 初始条件：系统从静止状态出发
    u     = 0.0   # 相对位移
    v     = 0.0   # 相对速度
    # 由运动方程求初始相对加速度
    a_rel = -(c * v + k * u) / m - ag[0]

    # 初始绝对加速度（相对加速度 + 地震动加速度）
    max_abs_acc = abs(a_rel + ag[0])

    # ---------- Newmark-β 逐步积分 ----------
    for j in range(N_rs - 1):
        # 等效荷载（下一时刻地震力 + 预测项）
        p_eff = (
            -m * ag[j + 1]
            + m * (a0 * u + a2 * v + a3 * a_rel)
            + c * (a1 * u + a4 * v + a5 * a_rel)
        )

        # 求解下一时刻相对位移、加速度、速度
        u_next = p_eff / k_eff
        a_next = a0 * (u_next - u) - a2 * v - a3 * a_rel
        v_next = v + a6 * a_rel + a7 * a_next

        # 绝对加速度 = 相对加速度 + 地震动加速度
        abs_acc = abs(a_next + ag[j + 1])
        if abs_acc > max_abs_acc:
            max_abs_acc = abs_acc

        u, v, a_rel = u_next, v_next, a_next

    # 该周期的谱加速度 = 时程内最大绝对加速度
    sa[i] = max_abs_acc


# ==========================================
# 6. 绘制弹性加速度反应谱（图3）
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman'] + plt.rcParams['font.serif']

fig3, ax3 = plt.subplots(figsize=(7, 4))

# 绘制 Sa-T 曲线
ax3.plot(periods, sa,
         color=PSA_LINE_COLOR, linewidth=1.2, label=label_name)

# 标题与坐标标签
ax3.set_title('Elastic Acceleration Response Spectrum', fontweight='bold', fontsize=14)
ax3.set_xlabel('Spectral Period T (s)', fontsize=12)
ax3.set_ylabel('Spectral Acceleration Sa (g)', fontsize=12)

# X 轴采用对数坐标（反应谱标准作法）
ax3.set_xscale('log')
ax3.set_xlim(*PSA_XLIM)

# Y 轴线性坐标
ax3.set_ylim(*PSA_YLIM)
ax3.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))

# 主网格（实线）+ 次网格（点线），适配对数坐标
ax3.grid(True, which='major', linestyle='-',  linewidth=0.5, color='gray', alpha=0.8)
ax3.grid(True, which='minor', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

# 图例与刻度
legend3 = ax3.legend(loc='upper right', frameon=True, fontsize=11)
legend3.get_frame().set_edgecolor('black')
ax3.tick_params(direction='in', which='both', top=True, right=True, labelsize=11)

fig3.tight_layout()
plt.show()