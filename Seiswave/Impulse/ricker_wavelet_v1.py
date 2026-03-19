import numpy as np
import matplotlib.pyplot as plt

def generate_ricker_wavelet(f_m, duration, dt, amp_max=1.0, t_peak_shift=0.5):
    # Create time vector
    t = np.arange(0, duration + dt/2, dt)
    
    # Ricker Wavelet Formula
    # factor = (pi * f_m * (t - t_0))^2
    # y = A * (1 - 2 * factor) * exp(-factor)
    factor = (np.pi * f_m * (t - t_peak_shift))**2
    amp = amp_max * (1 - 2 * factor) * np.exp(-factor)
    
    return t, amp

# --- Configuration ---
f_center = 5.0       # 5 Hz
time_duration = 2.0  # 0-2 seconds
time_step = 0.001    # 0.001s step
max_acc = 1.0        # 1 m/s^2
peak_time = 0.2      # Shift peak to 0.5s so it's fully visible

# --- Execution ---
t, amplitude = generate_ricker_wavelet(f_center, time_duration, time_step, max_acc, peak_time)

# 方法2：使用NumPy的向量化方法（更高效）
def numpy_cumtrapz(y, x, initial=0):
    """使用NumPy向量化操作的累积积分"""
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    cumulative = np.cumsum(avg_y * dx)
    # 添加初始值
    return np.concatenate(([initial], cumulative))

# 计算速度（对加速度积分）- 使用向量化方法
velocity = numpy_cumtrapz(amplitude, t, initial=0)

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(t, amplitude, label='Ricker Wavelet', linewidth=2)
plt.title(f'Ricker Wavelet (fm={f_center}Hz, Max Amp={max_acc} $m/s^2$)\nPeak centered at t={peak_time}s')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude ($m/s^2$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, velocity, label='Ricker Wavelet', linewidth=2)
plt.title(f'Ricker Wavelet (fm={f_center}Hz, Max Amp={max_acc} $m/s^2$)\nPeak centered at t={peak_time}s')
plt.xlabel('Time (s)')
plt.ylabel('velocity ($m/s$)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()

# --- Save to TXT ---
data = np.column_stack((t, velocity))
np.savetxt('C:\\Users\\12462\\Documents\\硕士\\2025科研\\模拟\\ricker_wavelet\\ricker_wavelet.txt', data, fmt='%.6f', delimiter='\t', comments='')