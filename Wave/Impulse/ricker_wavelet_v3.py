import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Set font to serif to match the academic style in the image
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_ricker_wavelet(f_m, duration, dt, amp_max=1.0, t_peak_shift=None):
    if t_peak_shift is None:
        t_peak_shift = 1.0 / f_m
    
    # Create time vector
    t = np.arange(0, duration + dt/2, dt)
    
    # Ricker Wavelet Formula
    # factor = (pi * f_m * (t - t_0))^2
    # y = A * (1 - 2 * factor) * exp(-factor)
    factor = (np.pi * f_m * (t - t_peak_shift))**2
    amp = amp_max * (1 - 2 * factor) * np.exp(-factor)
    
    return t, amp

def numpy_cumtrapz(y, x, initial=0):
    """Cumulative integration using NumPy vectorized operations"""
    dx = np.diff(x)
    avg_y = (y[:-1] + y[1:]) / 2.0
    cumulative = np.cumsum(avg_y * dx)
    # Add initial value
    return np.concatenate(([initial], cumulative))

# --- Configuration ---
f_center = 6.0       # Center frequency (Hz)
time_duration = 2.0  # 0-2 seconds
time_step = 0.001    # 0.001s step
max_acc = 1.0        # Max amplitude of acceleration (m/s^2)
peak_time = 1.0 / f_center  # Shift peak to 1.0/f_center so it matches the image
save_plots = False    # Whether to save plots to disk

# Colors matching the reference image
colors_map = {
    8.0: "#e41a1c",  # Red
    4.0: "#1f78b4",  # Blue
    2.0: "#ffb200",  # Warm amber/yellow
    1.0: "#2ca02c",  # Green
}
curve_color = colors_map.get(f_center, "#1f78b4")

# --- Execution ---
t, amplitude = generate_ricker_wavelet(f_center, time_duration, time_step, max_acc, peak_time)

# Calculate velocity (integral of acceleration)
velocity = numpy_cumtrapz(amplitude, t, initial=0)

# --- Save to TXT ---
out_filename = os.path.join(SCRIPT_DIR, f"ricker_wavelet_{f_center:.0f}Hz.txt")
data = np.column_stack((t, amplitude))
np.savetxt(out_filename, data, fmt='%.6f', delimiter='\t', comments='')
print(f"Saved waveform to: {out_filename}")

# --- Plot Acceleration (Ricker Wavelet) ---
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t, amplitude, label=f'$f_c = {f_center:.0f}$ Hz', color=curve_color, linewidth=1.8)

ax.set_xlim(0.0, time_duration)
ax.set_ylim(-0.5 * max_acc, 1.0 * max_acc)

ax.set_xlabel('$t$ (s)', fontsize=12)
ax.set_ylabel('$r(t)$', fontsize=12)
ax.grid(False)

# Make spines/borders slightly thicker
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# Ticks configuration (specific to duration=2.0 and max_acc=1.0 to match the reference image)
if np.isclose(time_duration, 2.0):
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
if np.isclose(max_acc, 1.0):
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# Make major and minor ticks point inwards, visible on all sides
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.1)
ax.tick_params(axis='both', labelsize=11)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', length=3)

# Legend configuration
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=12)

plt.tight_layout()
out_png_acc = os.path.join(SCRIPT_DIR, f"ricker_wavelet_acc_{f_center:.0f}Hz.png")
if save_plots:
    plt.savefig(out_png_acc, dpi=300)
    print(f"Saved acceleration plot to: {out_png_acc}")
plt.close()

# --- Plot Velocity ---
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(t, velocity, label=f'$f_c = {f_center:.0f}$ Hz Velocity', color=curve_color, linewidth=1.8)

ax.set_xlim(0.0, time_duration)
# Autoscale y-limit with margin
v_max = np.max(np.abs(velocity))
ax.set_ylim(-1.1 * v_max, 1.1 * v_max)

ax.set_xlabel('$t$ (s)', fontsize=12)
ax.set_ylabel('$v(t)$ (m/s)', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)

# Make spines/borders slightly thicker
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# Ticks configuration
if np.isclose(time_duration, 2.0):
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))

# Make major and minor ticks point inwards, visible on all sides
ax.tick_params(direction='in', top=True, right=True, which='both', width=1.1)
ax.tick_params(axis='both', labelsize=11)
ax.tick_params(which='major', length=5)
ax.tick_params(which='minor', length=3)

# Legend configuration
ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=False, fontsize=12)

plt.tight_layout()
out_png_vel = os.path.join(SCRIPT_DIR, f"ricker_wavelet_vel_{f_center:.0f}Hz.png")
if save_plots:
    plt.savefig(out_png_vel, dpi=300)
    print(f"Saved velocity plot to: {out_png_vel}")
plt.close()
