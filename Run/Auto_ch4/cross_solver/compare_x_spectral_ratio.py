"""X 组谱比跨软件对比：按论文 G_h 定义（地表谱 / 同侧远场自由场谱）比较 Abaqus 与 SPECFEM2D。

读取 compare_x*.py 已对齐的 801 点公共地表时程（common_time / s / abaqus_acc_* / specfem_acc_*），
在坡顶(s=0)、坡面中(s=0.5)、坡脚前(s=2) 三点对水平/竖向分量计算：
  - 直接傅里叶幅值谱（两侧对比，无量纲分母）；
  - G_h 型谱比 = 地表谱 / 远场自由场谱（同侧远场端 s 处近似一维自由场，两侧同构造）。
指标在 Ricker 有效激励频带 1.7—10 Hz 计算为主，并在 0.5—10 Hz 全带补充：
  direct_nrmse / ratio_nrmse / log_amp_rmse / 主峰频率差。
输出：<comparison_dir>/x00X_spectral_ratio_metrics.json 与 x00X_spectral_compare.png。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"C:\Users\12462\Documents\Code\AbqScripts\Run\cross_solver_X\abaqus"
CASES = [
    ("x001", "X001-A", os.path.join(BASE, "X001-A", "comparison", "x001_comparison_arrays.npz")),
    ("x002", "X002-A", os.path.join(BASE, "X002-A", "comparison", "x002_comparison_arrays.npz")),
    ("x002_sr", "X002-A", os.path.join(BASE, "X002-A", "comparison", "x002_sr_comparison_arrays.npz")),
]
POINTS = {"crest": 0.0, "mid": 0.5, "toe": 2.0}
F_SHOW = (0.5, 10.0)   # 全带展示
F_EXC = (1.7, 10.0)    # Ricker 有效激励带，主指标区间
NFFT = 8192


def taper(x, alpha=0.1):
    n = len(x)
    ramp = max(1, int(n * alpha / 2))
    w = np.ones(n)
    w[:ramp] = np.linspace(0, 1, ramp)
    w[-ramp:] = np.linspace(1, 0, ramp)
    return x * w


def spectrum(acc, dt, nfft=NFFT):
    x = taper(np.asarray(acc, float))
    A = np.fft.rfft(x, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=dt)
    amp = np.abs(A) * (2.0 / len(x))   # 单边幅值，忽略 DC/Nyq 缩放误差（带内可忽略）
    return freqs, amp


def nrmse(a, b):
    denom = np.mean(np.abs((a + b) * 0.5))
    return float(np.sqrt(np.mean((a - b) ** 2)) / denom) if denom > 0 else float("nan")


def log_rmse(a, b):
    la = np.log(np.maximum(a, 1e-12))
    lb = np.log(np.maximum(b, 1e-12))
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def band_mask(freqs, lo, hi):
    return (freqs >= lo) & (freqs <= hi)


def peak_freq(freqs, amp, mask):
    m = mask & (amp > 0)
    if m.sum() == 0:
        return float("nan")
    return float(freqs[m][np.argmax(amp[m])])


def main():
    for name, label, npz in CASES:
        d = np.load(npz, allow_pickle=True)
        t = np.asarray(d["common_time"], float)
        s = np.asarray(d["s"], float)
        dt = float(t[1] - t[0])
        comp_dir = os.path.dirname(npz)
        # 远场自由场代理：远离地形（s 取两端各 0.5 归一化范围）
        s_lo, s_hi = s.min(), s.max()
        ff_mask = (s <= s_lo + 0.5) | (s >= s_hi - 0.5)
        idxs = {pt: int(np.argmin(np.abs(s - val))) for pt, val in POINTS.items()}

        results = {}
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        comps = [("h", "acc_h", "Horiz"), ("v", "acc_v", "Vert")]
        for ci, (comp, suf, cn) in enumerate(comps):
            a_all = np.asarray(d[f"abaqus_{suf}"], float)
            s_all = np.asarray(d[f"specfem_{suf}"], float)
            # 频率轴只取一次（所有点共享）
            fA_all, _ = spectrum(a_all[0], dt)
            fS_all = fA_all
            ffA = a_all[ff_mask].mean(axis=0)
            ffS = s_all[ff_mask].mean(axis=0)
            _, ampA_ff = spectrum(ffA, dt)
            _, ampS_ff = spectrum(ffS, dt)
            for pj, (pt, idx) in enumerate(idxs.items()):
                _, ampA = spectrum(a_all[idx], dt)
                _, ampS = spectrum(s_all[idx], dt)
                ratioA = ampA / np.maximum(ampA_ff, 1e-12)
                ratioS = ampS / np.maximum(ampS_ff, 1e-12)
                m_exc = band_mask(fA_all, *F_EXC)
                m_full = band_mask(fA_all, *F_SHOW)
                res = {
                    "direct_nrmse_exc": nrmse(ampA[m_exc], ampS[m_exc]),
                    "direct_nrmse_full": nrmse(ampA[m_full], ampS[m_full]),
                    "ratio_nrmse_exc": nrmse(ratioA[m_exc], ratioS[m_exc]),
                    "log_amp_rmse_exc": log_rmse(ampA[m_exc], ampS[m_exc]),
                    "peak_freq_abaqus_hz": peak_freq(fA_all, ampA, m_exc),
                    "peak_freq_specfem_hz": peak_freq(fS_all, ampS, m_exc),
                    "ratio_peak_freq_abaqus_hz": peak_freq(fA_all, ratioA, m_exc),
                    "ratio_peak_freq_specfem_hz": peak_freq(fS_all, ratioS, m_exc),
                }
                res["peak_freq_diff_hz"] = res["peak_freq_abaqus_hz"] - res["peak_freq_specfem_hz"]
                res["ratio_peak_freq_diff_hz"] = res["ratio_peak_freq_abaqus_hz"] - res["ratio_peak_freq_specfem_hz"]
                results.setdefault(pt, {})[comp] = res

                ax = axes[ci, pj]
                ax.plot(fA_all, ampA, label="Abaqus", color="C0")
                ax.plot(fS_all, ampS, label="SPECFEM2D", color="C1", ls="--")
                ax.set_xlim(*F_SHOW)
                ax.set_yscale("log")
                ax.set_title(f"{cn} {pt} (s={POINTS[pt]})")
                ax.set_xlabel("Frequency / Hz")
                ax.set_ylabel("Accel. amplitude (m/s^2/Hz)")
                ax.grid(True, which="both", alpha=0.3)
                if ci == 0 and pj == 0:
                    ax.legend(fontsize=8)
                ax2 = axes[ci, pj].inset_axes([0.45, 0.45, 0.52, 0.5])
                ax2.plot(fA_all, ratioA, color="C0")
                ax2.plot(fS_all, ratioS, color="C1", ls="--")
                ax2.set_xlim(*F_EXC)
                ax2.set_yscale("log")
                ax2.set_title("Spectral ratio (G_h-like)", fontsize=8)
                ax2.tick_params(labelsize=7)
                ax2.grid(True, which="both", alpha=0.3)

        fig.suptitle(f"{name.upper()}: Abaqus vs SPECFEM2D spectral-ratio comparison ({label})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        png = os.path.join(comp_dir, f"{name}_spectral_compare.png")
        fig.savefig(png, dpi=130)
        plt.close(fig)

        metrics = {"case": name, "label": label, "band_exc_hz": F_EXC, "band_show_hz": F_SHOW,
                   "points": POINTS, "results": results}
        jpath = os.path.join(comp_dir, f"{name}_spectral_ratio_metrics.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[{name}] 已写出 {os.path.basename(png)} 与 {os.path.basename(jpath)}")
        # 终端摘要
        for pt in POINTS:
            rh = results[pt]["h"]
            rv = results[pt]["v"]
            print(f"  {pt:6s} 水平 ratio_nrmse={rh['ratio_nrmse_exc']:.3f} 峰值差={rh['peak_freq_diff_hz']:+.2f}Hz | "
                  f"竖向 ratio_nrmse={rv['ratio_nrmse_exc']:.3f} 峰值差={rv['peak_freq_diff_hz']:+.2f}Hz")


if __name__ == "__main__":
    main()
