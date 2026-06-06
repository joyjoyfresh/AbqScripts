import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 设置学术风格字体
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取脚本所在目录的绝对路径
out_path = os.path.join(SCRIPT_DIR, "Acceleration")  # 构建输出子目录路径
os.makedirs(out_path, exist_ok=True)  # 确保输出目录存在，若不存在则创建


def generate_impulse_wave(P0, T, duration, dt):
    """
    生成脉冲波（Impulse Wave）位移、速度与加速度时程

    位移公式：
      P(tau) = 16*P0 * [G(tau) - 4*G(tau-1/4) + 6*G(tau-1/2)
                        - 4*G(tau-3/4) + G(tau-1)]
      其中 tau = t/T，G(tau) = tau^3 * H(tau)

    速度由位移对时间求导得到：
      v(t) = dP/dt = (1/T) * dP/dtau

    加速度由速度对时间求导得到：
      a(t) = dv/dt = (1/T²) * d²P/dtau²

    参数
    ----
    P0 : float
        位移振幅 (m)
    T : float
        脉冲持续时间 (s)
    duration : float
        总时长 (s)
    dt : float
        时间步长 (s)

    返回
    ----
    t : ndarray
        时间轴 (s)
    displacement : ndarray
        位移时程 (m)
    velocity : ndarray
        速度时程 (m/s)
    acceleration : ndarray
        加速度时程 (m/s²)
    """
    # 时间轴
    num_points = int(duration / dt) + 1
    t = np.linspace(0, duration, num_points)
    tau = t / T  # 归一化时间

    # 基函数
    def H(x):
        """Heaviside 单位阶跃函数"""
        return np.heaviside(x, 1.0)

    def G(x):
        """G(x) = x^3 * H(x)"""
        return (x ** 3) * H(x)

    def G_prime(x):
        """G'(x) = 3*x^2 * H(x)"""
        return 3 * (x ** 2) * H(x)

    def G_double_prime(x):
        """G''(x) = 6*x * H(x)"""
        return 6 * x * H(x)

    # 二项式系数与偏移量
    coeffs = [1, -4, 6, -4, 1]
    shifts = [0, 0.25, 0.50, 0.75, 1.0]

    # 位移 P(t)
    poly_P = sum(c * G(tau - s) for c, s in zip(coeffs, shifts))
    displacement = 16 * P0 * poly_P

    # 速度 v(t) = dP/dt，链式法则引入 1/T
    poly_V = sum(c * G_prime(tau - s) for c, s in zip(coeffs, shifts))
    velocity = (16 * P0 / T) * poly_V

    # 加速度 a(t) = dv/dt，链式法则引入 1/T²
    poly_A = sum(c * G_double_prime(tau - s) for c, s in zip(coeffs, shifts))
    acceleration = (16 * P0 / (T ** 2)) * poly_A

    return t, displacement, velocity, acceleration


# ============================================================
# 配置参数
# ============================================================
P0 = 0.1            # 位移振幅 (m)
T = 0.3             # 脉冲持续时间 (s)
time_duration = 1.5  # 总时长 (s)
time_step = 0.001    # 时间步长 (s)
save_plots = False   # 是否保存图片到磁盘

# 颜色配置（与 ricker 波风格统一）
curve_color_disp = "#1f77b4"  # 蓝色 — 位移
curve_color_acc = "#ff7f0e"   # 橙色 — 加速度

# ============================================================
# 生成脉冲波
# ============================================================
t, displacement, velocity, acceleration = generate_impulse_wave(P0, T, time_duration, time_step)

# ============================================================
# 保存数据（时间 — 加速度）到 TXT
# ============================================================
out_filename = os.path.join(out_path, f"impulse_wave_T{T:.2f}s.txt")  # 输出到 Acceleration 子目录
data = np.column_stack((t, acceleration))
np.savetxt(out_filename, data, fmt='%.6f', delimiter='\t', comments='')
print(f"Saved waveform to: {out_filename}")

# ============================================================
# 绘图 — 位移
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t, displacement, label=f'$T = {T:.1f}$ s', color=curve_color_disp, linewidth=1.8)

ax.set_xlim(0.0, time_duration)
ax.set_ylim(-0.2 * P0, 1.1 * P0)

ax.set_xlabel('$t$ (s)', fontsize=12)
ax.set_ylabel('$P(t)$ (m)', fontsize=12)
ax.grid(False)

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 刻度配置
if np.isclose(time_duration, 1.5):
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
if np.isclose(P0, 0.1):
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

# 刻度线内向、四边显示
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.1)
ax.tick_params(axis='both', labelsize=11)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', length=3)

# 图例
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=12)

plt.tight_layout()
out_png_disp = os.path.join(out_path, f"impulse_wave_disp_T{T:.2f}s.png")  # 输出到 Velocity 子目录
if save_plots:
    plt.savefig(out_png_disp, dpi=300)
    print(f"Saved displacement plot to: {out_png_disp}")
plt.close()

# ============================================================
# 绘图 — 加速度
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t, acceleration, label=f'$T = {T:.1f}$ s', color=curve_color_acc, linewidth=1.8)

ax.set_xlim(0.0, time_duration)
a_max = np.max(np.abs(acceleration))
ax.set_ylim(-1.1 * a_max, 1.1 * a_max)

ax.set_xlabel('$t$ (s)', fontsize=12)
ax.set_ylabel('$a(t)$ (m/s$^2$)', fontsize=12)
ax.grid(False)

# 边框加粗
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 刻度配置
if np.isclose(time_duration, 1.5):
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

# 刻度线内向、四边显示
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.1)
ax.tick_params(axis='both', labelsize=11)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', length=3)

# 图例
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=12)

plt.tight_layout()
out_png_acc = os.path.join(out_path, f"impulse_wave_acc_T{T:.2f}s.png")  # 输出到 Acceleration 子目录
if save_plots:
    plt.savefig(out_png_acc, dpi=300)
    print(f"Saved acceleration plot to: {out_png_acc}")
plt.close()
