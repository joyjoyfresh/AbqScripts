import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 脉冲波（Impulse Wave）位移与速度时程
#
# 位移公式：
#   P(tau) = 16*P0 * [G(tau) - 4*G(tau-1/4) + 6*G(tau-1/2)
#                     - 4*G(tau-3/4) + G(tau-1)]
#   其中 tau = t/T，G(tau) = tau^3 * H(tau)
#
# 速度由位移对时间求导得到：
#   v(t) = dP/dt = (1/T) * dP/dtau
# ============================================================

# 参数定义
P0 = 0.1   # 位移振幅 (m)
T  = 0.3   # 脉冲持续时间 (s)
total_duration = 1.5  # 总时长 (s)
dt             = 0.001  # 时间步长 (s)

# 时间轴：0, 0.001, 0.002, ..., 1.5（共 1501 点）
num_points = int(total_duration / dt) + 1
t   = np.linspace(0, total_duration, num_points)
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

# 二项式系数与偏移量
coeffs  = [1, -4, 6, -4, 1]
shifts  = [0, 0.25, 0.50, 0.75, 1.0]

# 位移 P(t)
poly_P = sum(c * G(tau - s) for c, s in zip(coeffs, shifts))
P = 16 * P0 * poly_P

# 速度 v(t) = dP/dt，链式法则引入 1/T
poly_V = sum(c * G_prime(tau - s) for c, s in zip(coeffs, shifts))
V = (16 * P0 / T) * poly_V

# ============================================================
# 绘图
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Impulse Wave — Displacement & Velocity', fontsize=14)

# 位移
ax1.plot(t, P, color='#1f77b4', linewidth=2, label=r'Displacement $P(t)$')
ax1.axvline(T, color='red', linestyle=':', label=f'Pulse End ($T={T}$ s)')
ax1.set_ylabel('Displacement (m)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc='upper right')

# 速度
ax2.plot(t, V, color='#ff7f0e', linewidth=2, label=r'Velocity $v(t)$')
ax2.axvline(T, color='red', linestyle=':', label=f'Pulse End ($T={T}$ s)')

# 标注速度极值
for func, sign, offset in [(np.argmax, 1, +0.1), (np.argmin, -1, -0.4)]:
    idx = func(V)
    ax2.plot(t[idx], V[idx], 'ro', markersize=4)
    label = f'{"Max" if sign > 0 else "Min"}: {V[idx]:.2f} m/s'
    ax2.text(t[idx], V[idx] + offset, label, ha='center', fontsize=9)

ax2.set_xlabel('Time $t$ (s)', fontsize=12)
ax2.set_ylabel('Velocity (m/s)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# ============================================================
# 保存数据（时间 — 速度）
# ============================================================
output_path = r'C:\Users\12462\Documents\Master\Abaqus\Scripts\Seiswave\impulse_wave_v2.txt'
data = np.column_stack((t, V))
np.savetxt(output_path, data, fmt='%.6f', delimiter='\t', comments='')
print(f'Data saved to: {output_path}')
