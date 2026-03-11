import numpy as np
import matplotlib.pyplot as plt

# 1. 参数定义
P0 = 0.1   # 位移振幅 (m)
T = 0.3    # 脉冲持续时间 (s)
total_duration = 1.5 # 总绘图时长 (s)

# 2. 时间轴
t = np.linspace(0, total_duration, 1500)
tau = t / T  # 归一化时间

# 3. 定义函数
def H(x):
    """Heaviside 阶跃函数"""
    return np.heaviside(x, 1.0)

# G(tau) 用于计算位移
def G(x):
    return (x**3) * H(x)

# G'(tau) 用于计算速度
# G'(tau) = d/dtau (tau^3 * H(tau)) = 3*tau^2 * H(tau)
def G_prime(x):
    return 3 * (x**2) * H(x)

# 4. 计算位移 P(t)
# P(tau) = 16*P0 * Sum( c_i * G(tau - shift_i) )
# 系数: 1, -4, 6, -4, 1
# 偏移: 0, 0.25, 0.5, 0.75, 1.0
poly_P = (G(tau) 
            - 4*G(tau - 0.25) 
            + 6*G(tau - 0.5) 
            - 4*G(tau - 0.75) 
            + G(tau - 1.0))
P = 16 * P0 * poly_P

# 5. 计算速度 v(t)
# v(t) = (1/T) * 16*P0 * Sum( c_i * G'(tau - shift_i) )
poly_V = (G_prime(tau) 
            - 4*G_prime(tau - 0.25) 
            + 6*G_prime(tau - 0.5) 
            - 4*G_prime(tau - 0.75) 
            + G_prime(tau - 1.0))

# 注意系数 1/T 来自链式法则 dt = T * dtau
V = (16 * P0 / T) * poly_V

# 6. 绘图
plt.figure(figsize=(10, 8))

# 子图1: 位移
plt.subplot(2, 1, 1)
plt.plot(t, P, label=r'Displacement $P(t)$', color='#1f77b4', linewidth=2)
plt.ylabel('Displacement (m)', fontsize=12)
plt.title('Displacement and Velocity of Impulse', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(T, color='red', linestyle=':', label='Pulse End ($T=0.3s$)')
plt.legend(loc='upper right')

# 子图2: 速度
plt.subplot(2, 1, 2)
plt.plot(t, V, label=r'Velocity $v(t)$', color='#ff7f0e', linewidth=2)
plt.xlabel('Time $t$ (s)', fontsize=12)
plt.ylabel('Velocity (m/s)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(T, color='red', linestyle=':', label='Pulse End ($T=0.3s$)')

# 标记速度的最大值和最小值
v_max = np.max(V)
v_min = np.min(V)
t_max = t[np.argmax(V)]
t_min = t[np.argmin(V)]

plt.plot(t_max, v_max, 'ro', markersize=4)
plt.text(t_max, v_max + 0.1, f'Max: {v_max:.2f} m/s', ha='center', fontsize=9)
plt.plot(t_min, v_min, 'ro', markersize=4)
plt.text(t_min, v_min - 0.4, f'Min: {v_min:.2f} m/s', ha='center', fontsize=9)

plt.legend(loc='upper right')

plt.tight_layout()
plt.show()


# --- Save to TXT ---
data = np.column_stack((t, V))
np.savetxt('C:\\Users\\12462\\Documents\\硕士\\2025科研\\模拟\\impulse_wave\\impulse_wave.txt', data, fmt='%.6f', delimiter='\t', comments='')
