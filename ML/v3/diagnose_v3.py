# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8，确保中文注释正常解析
"""ML v3 — Stage 0 诊断脚本（只读分析，不训练任何模型）。

复现 ml_plan_v3.md 第 3 节全部实测数字并画 5 张诊断图，产出「为什么从深度学习
换到经典 ML」的论文论证素材。必须用 py -3（Python 3.12）运行，不能用 Abaqus 的
默认 python（Py2.7，无 f-string）。

复用来源：
  - 反应谱/FFT 算法复制自 Wave/Seismic/scale_and_plot_v2.py
  - 数据加载逻辑沿用 ML/data/dataset_v2.py（内联为纯 numpy/pandas，不引入 torch）
  - v2 深度模型成绩读自 ML/outputs/summary_metrics.json
"""

import os  # 导入操作系统路径与目录操作
import re  # 导入正则表达式（解析工况文件夹名）
import json  # 导入 JSON 读写（存诊断结果、读 v2 成绩）
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入表格读取库
import matplotlib  # 导入绘图框架
matplotlib.use("Agg")  # 使用无界面后端，支持批处理保存图片
import matplotlib.pyplot as plt  # 导入绘图子模块
import matplotlib.ticker as mticker  # 导入刻度格式化器（对数轴标签用 ASCII 减号，避免 SimHei 缺字形）

# ==========================================================================
# 配置区
# ==========================================================================
DATA_DIR = r"E:\Abaqus\fuke-ALL"  # 原始数据根目录（与 config_v2.py 一致）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录（ML/）
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_v3")  # v3 输出目录（不覆盖 v2 的 outputs/）
V2_SUMMARY = os.path.join(SCRIPT_DIR, "outputs", "summary_metrics.json")  # v2 深度模型成绩文件
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 输出目录不存在则创建

WAVES = ["El_Centro", "Loma_Prieta", "Northridge"]  # 三条地震波名称
WAVE_CN = {"El_Centro": "El-Centro", "Loma_Prieta": "Loma-Prieta", "Northridge": "Northridge"}  # 显示名映射
NUM_POINTS = 161  # 统一曲线采样点数（沿用 config_v2.NUM_CURVE_POINTS）
GRID = np.linspace(0.0, 8.0, NUM_POINTS)  # 统一 x/h 网格（0→8，步长 0.05）
PEAK_MASK = GRID <= 2.0  # 峰值区掩码（x/h ≤ 2）
FAR_MASK = GRID >= 5.0  # 远场区掩码（x/h ≥ 5）
FOLDER_PATTERN = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 工况文件夹名解析规则
SA_PERIODS_TARGET = [0.1, 0.2, 0.5, 1.0, 2.0]  # 反应谱代表周期（取 Sa 值做波特征）
WAVE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 三波配色：蓝/橙/绿

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]  # 中文黑体优先（修复方块）
plt.rcParams["axes.unicode_minus"] = False  # 修复负号显示为方块的问题


# ==========================================================================
# 通用辅助函数（内联，单文件自包含）
# ==========================================================================
def np_trapz(y, x):  # 梯形积分兼容封装
    """numpy>=2.0 用 trapezoid，旧版回退 trapz。"""
    if hasattr(np, "trapezoid"):  # 新版 numpy 提供 trapezoid
        return np.trapezoid(y, x)  # 使用新接口
    return np.trapz(y, x)  # 回退旧接口


def read_csv_robust(path):  # 稳健读取 CSV
    """统一用 utf-8-sig 读，自动剥掉 TAF 文件的 BOM（列名 \\ufeffx/h）。"""
    return pd.read_csv(path, encoding="utf-8-sig")  # 带签名编码读取，BOM 被剥离


def cumtrapz(y, x):  # 累积梯形积分（避免引入 scipy）
    """返回与 y 等长的累积积分，首元素为 0。"""
    out = np.zeros_like(y, dtype=float)  # 初始化输出数组
    out[1:] = np.cumsum((y[1:] + y[:-1]) / 2.0 * np.diff(x))  # 梯形法逐段累加
    return out  # 返回累积积分序列


def calc_fft(t, accel):  # 计算单边傅里叶振幅谱（复制自 scale_and_plot_v2）
    """返回 (dt, 去均值加速度, 正频率, 正振幅)；非均匀采样先线性重采样。"""
    dt_raw = np.diff(t)  # 原始相邻时间差
    dt = np.median(dt_raw)  # 采样间隔取中位数（稳健）
    is_uniform = np.allclose(dt_raw, dt, rtol=1e-4, atol=1e-8)  # 判断是否均匀采样
    if is_uniform:  # 均匀采样
        t_uniform = t  # 直接用原时间
        accel_uniform = accel  # 直接用原加速度
    else:  # 非均匀采样
        t_uniform = np.arange(t[0], t[-1] + 0.5 * dt, dt)  # 生成均匀时间网格
        accel_uniform = np.interp(t_uniform, t, accel)  # 线性重采样加速度
    n_uniform = len(t_uniform)  # 均匀序列长度
    accel_detrended = accel_uniform - np.mean(accel_uniform)  # 去均值（避免 0Hz 直流尖峰）
    dft_coeff = np.fft.rfft(accel_detrended)  # 实数 FFT
    positive_frequencies = np.fft.rfftfreq(n_uniform, d=dt)  # 正频率轴
    positive_amplitudes = np.abs(dft_coeff) * dt  # 地震工程标准：乘以 dt
    return dt, accel_detrended, positive_frequencies, positive_amplitudes  # 返回结果


def calc_response_spectrum(accel_detrended, dt, damping=0.05):  # Newmark-β 反应谱（复制自 scale_and_plot_v2）
    """基于 Newmark-β 法计算 5% 阻尼弹性加速度反应谱，返回 (periods, sa)。"""
    periods = np.logspace(-2, 1, 500)  # 周期网格 0.01→10 s（对数 500 点）
    sa = np.zeros(len(periods))  # 初始化谱加速度数组
    gamma = 0.5  # Newmark 参数 γ
    beta = 0.25  # Newmark 参数 β（平均加速度法）
    ag = accel_detrended  # 地面加速度序列
    n_rs = len(ag)  # 序列长度
    a0 = 1.0 / (beta * dt**2)  # 积分常数 a0
    a1 = gamma / (beta * dt)  # 积分常数 a1
    a2 = 1.0 / (beta * dt)  # 积分常数 a2
    a3 = 1.0 / (2.0 * beta) - 1.0  # 积分常数 a3
    a4 = gamma / beta - 1.0  # 积分常数 a4
    a5 = dt * (gamma / (2.0 * beta) - 1.0)  # 积分常数 a5
    a6 = dt * (1.0 - gamma)  # 积分常数 a6
    a7 = gamma * dt  # 积分常数 a7
    for i, period in enumerate(periods):  # 遍历每个自振周期
        if period <= dt:  # 周期小于采样间隔时无法积分
            sa[i] = np.max(np.abs(ag))  # 近似取 PGA
            continue  # 跳过该周期
        m = 1.0  # 单位质量
        omega_n = 2.0 * np.pi / period  # 自振圆频率
        k = m * omega_n**2  # 刚度
        c = 2.0 * damping * omega_n * m  # 阻尼系数
        k_eff = k + a0 * m + a1 * c  # 等效刚度
        u = 0.0  # 初始位移
        v = 0.0  # 初始速度
        a_rel = -(c * v + k * u) / m - ag[0]  # 初始相对加速度
        max_abs_acc = abs(a_rel + ag[0])  # 初始绝对加速度峰值
        for j in range(n_rs - 1):  # 逐时间步积分
            p_eff = (-m * ag[j + 1]
                     + m * (a0 * u + a2 * v + a3 * a_rel)
                     + c * (a1 * u + a4 * v + a5 * a_rel))  # 等效荷载
            u_next = p_eff / k_eff  # 下一步位移
            a_next = a0 * (u_next - u) - a2 * v - a3 * a_rel  # 下一步相对加速度
            v_next = v + a6 * a_rel + a7 * a_next  # 下一步速度
            abs_acc = abs(a_next + ag[j + 1])  # 绝对加速度
            if abs_acc > max_abs_acc:  # 更新峰值
                max_abs_acc = abs_acc  # 记录更大值
            u, v, a_rel = u_next, v_next, a_next  # 滚动状态
        sa[i] = max_abs_acc  # 该周期的谱加速度 = 绝对加速度峰值
    return periods, sa  # 返回周期与谱加速度


def calc_wave_features(t, accel):  # 计算单条波的物理特征
    """返回 (特征字典, periods, sa)；PGA_in 应为常数 0.300（零信息，证明须剔除）。"""
    g = 9.81  # 重力加速度（m/s²）
    n = len(t)  # 序列长度
    pga = float(np.abs(accel).max())  # 输入波 PGA（单位 g）
    arias = float(np.pi / (2.0 * g) * np_trapz(accel**2, t))  # Arias 强度
    csum = np.cumsum(accel**2)  # 能量累积
    csum = csum / csum[-1]  # 归一化到 [0,1]
    i5 = min(int(np.searchsorted(csum, 0.05)), n - 1)  # 能量达 5% 的索引（防越界）
    i95 = min(int(np.searchsorted(csum, 0.95)), n - 1)  # 能量达 95% 的索引（防越界）
    d595 = float(t[i95] - t[i5])  # 有效持时 D5-95
    dt_fft, accel_d, freq, amp = calc_fft(t, accel)  # 傅里叶谱
    fp = freq[np.argmax(amp)]  # 谱峰频率
    Tp = float(1.0 / fp) if fp > 0 else float("nan")  # 卓越周期 = 1/谱峰频率
    band = (freq >= 0.25) & (freq <= 20.0)  # Rathje 平均周期频带 0.25–20Hz
    Ci = amp[band]  # 频带内傅里叶幅值
    fi = freq[band]  # 频带内频率
    Tm = float(np.sum(Ci**2 / fi) / np.sum(Ci**2))  # 平均周期 Tm（Rathje 1998）
    vel = cumtrapz(accel * g, t)  # 速度（m/s）= 加速度积分
    pgv = float(np.abs(vel).max())  # PGV
    disp = cumtrapz(vel, t)  # 位移（m）= 速度积分
    pgd = float(np.abs(disp).max())  # PGD
    periods, sa = calc_response_spectrum(accel_d, dt_fft)  # 反应谱（用 calc_fft 返回的 dt）
    sa_at = {f"Sa_{T}": float(np.interp(T, periods, sa)) for T in SA_PERIODS_TARGET}  # 取代表周期谱值
    feat = {"PGA_in": pga, "Arias": arias, "D5_95": d595, "Tp": Tp, "Tm": Tm, "PGV": pgv, "PGD": pgd}  # 标量特征
    feat.update(sa_at)  # 合入反应谱特征
    return feat, periods, sa  # 返回


def pod_recon_error(X, n_list):  # POD/PCA 重构误差
    """X:(Nsample,161)。返回 (各N重构MAE, 99%方差成分数, 累计方差, 主成分Vt, 均值mu)。"""
    mu = X.mean(axis=0)  # 均值曲线
    Xc = X - mu  # 去均值
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # 奇异值分解
    errs = {}  # 各 N 的重构 MAE
    for nn in n_list:  # 遍历保留成分数
        Xr = (U[:, :nn] * S[:nn]) @ Vt[:nn] + mu  # 用前 nn 个成分重构
        errs[nn] = float(np.abs(X - Xr).mean())  # 重构 MAE
    var = S**2  # 各成分方差
    cum = np.cumsum(var) / np.sum(var)  # 累计方差解释率
    n99 = int(np.searchsorted(cum, 0.99) + 1)  # 达 99% 方差所需成分数
    return errs, n99, cum, Vt, mu  # 返回


def load_all_curves():  # 加载全部工况曲线到统一网格
    """遍历工况文件夹，每波插值 TAF_h/PGA_h/PGA_v(坡)/PGA_v(平) 到 161 网格。

    缺任一文件的工况整体跳过（175→174）。返回 geom(Ngeom,3) 与四个 (Ngeom,3,161) 数组。
    """
    folders = sorted(f for f in os.listdir(DATA_DIR) if FOLDER_PATTERN.match(f))  # 所有工况文件夹
    geom, TAFh, PGAh, PGAvs, PGAvf = [], [], [], [], []  # 各结果列表
    skipped = 0  # 跳过计数
    for folder in folders:  # 遍历工况
        m = FOLDER_PATTERN.match(folder)  # 解析几何
        h, i_ang, angle = float(m.group(1)), float(m.group(2)), float(m.group(3))  # 坡高/坡角/入射角
        fp = os.path.join(DATA_DIR, folder)  # 工况目录
        taf_w, pgah_w, pgavs_w, pgavf_w = [], [], [], []  # 三波缓存
        ok = True  # 文件齐全标志
        for w in WAVES:  # 遍历三波
            f_taf = os.path.join(fp, f"TAF-{w}.csv")  # TAF 文件
            f_slope = os.path.join(fp, f"PGA-{w}_scaled-slope.csv")  # 坡面 PGA 文件
            f_flat = os.path.join(fp, f"PGA-{w}_scaled-flat.csv")  # 平地 PGA 文件
            if not (os.path.exists(f_taf) and os.path.exists(f_slope) and os.path.exists(f_flat)):  # 任一缺失
                ok = False  # 标记不齐全
                break  # 跳出波循环
            d_taf = read_csv_robust(f_taf).sort_values("x/h")  # 读 TAF 并按 x/h 排序
            taf_w.append(np.interp(GRID, d_taf["x/h"].values, d_taf["TAF_h"].values))  # 插值 TAF_h
            d_s = read_csv_robust(f_slope).sort_values("x/h")  # 读坡面 PGA
            pgah_w.append(np.interp(GRID, d_s["x/h"].values, d_s["PGA_h"].values))  # 插值 PGA_h
            pgavs_w.append(np.interp(GRID, d_s["x/h"].values, d_s["PGA_v"].values))  # 插值坡面 PGA_v
            d_f = read_csv_robust(f_flat).sort_values("x/h")  # 读平地 PGA
            pgavf_w.append(np.interp(GRID, d_f["x/h"].values, d_f["PGA_v"].values))  # 插值平地 PGA_v
        if not ok or len(taf_w) != 3:  # 工况文件不齐
            skipped += 1  # 计数
            continue  # 跳过
        geom.append([h, i_ang, angle])  # 记录几何参数
        TAFh.append(taf_w)  # 记录 TAF_h
        PGAh.append(pgah_w)  # 记录 PGA_h
        PGAvs.append(pgavs_w)  # 记录坡面 PGA_v
        PGAvf.append(pgavf_w)  # 记录平地 PGA_v
    print(f"  加载工况 {len(geom)} 组（跳过 {skipped} 组文件不全）")  # 打印加载情况
    return (np.array(geom), np.array(TAFh), np.array(PGAh), np.array(PGAvs), np.array(PGAvf))  # 转数组返回


# ==========================================================================
# 诊断计算（D1–D5），每段返回标量结果 + 画图所需数组
# ==========================================================================
def diag_variance(TAFh):  # D1：地形 vs 波 变异分析
    """跨波 std（同地形3波之间）vs 跨地形 std（同波不同地形）。"""
    cross_wave = float(TAFh.std(axis=1).mean())  # 跨波 std（对3波维求 std 再平均）
    cross_terrain = float(TAFh.std(axis=0).mean())  # 跨地形 std（对工况维求 std 再平均）
    ratio = cross_terrain / cross_wave  # 地形/波 比值
    cw_xh = TAFh.std(axis=1).mean(axis=0)  # 跨波 std 沿 x/h（画图用）
    ct_xh = TAFh.std(axis=0).mean(axis=0)  # 跨地形 std 沿 x/h（画图用）
    res = {"cross_wave_std": cross_wave, "cross_terrain_std": cross_terrain, "ratio": ratio,
           "taf_h_min": float(TAFh.min()), "taf_h_max": float(TAFh.max()), "taf_h_mean": float(TAFh.mean())}  # 汇总
    return res, cw_xh, ct_xh  # 返回标量 + 逐点曲线


def diag_baseline(TAFh):  # D2：傻瓜基线 vs v2 深度模型
    """傻瓜基线 = 每地形取3波平均（完全无视波）。算全区/峰值区/远场区 MAE。"""
    pred = TAFh.mean(axis=1, keepdims=True)  # 每地形3波平均（广播回3波）
    abs_err = np.abs(TAFh - pred)  # 逐点绝对误差 (Ngeom,3,161)
    mae_all = float(abs_err.mean())  # 全区 MAE
    mae_peak = float(abs_err[:, :, PEAK_MASK].mean())  # 峰值区 MAE
    mae_far = float(abs_err[:, :, FAR_MASK].mean())  # 远场区 MAE
    v2 = {}  # v2 各模型 TAF_h MAE
    if os.path.exists(V2_SUMMARY):  # v2 成绩文件存在
        with open(V2_SUMMARY, "r", encoding="utf-8") as fh:  # 打开成绩文件
            s = json.load(fh)  # 读 JSON
        v2 = {model: s[model]["TAF_h_MAE"] for model in s}  # 取各模型 TAF_h MAE
    res = {"baseline_mae_all": mae_all, "baseline_mae_peak": mae_peak, "baseline_mae_far": mae_far,
           "v2_taf_h_mae": v2}  # 汇总
    return res  # 返回


def diag_tafv(PGAvs, PGAvf):  # D3：TAF_v 病态分析
    """平地竖向 PGA_v 近零导致 TAF_v 爆炸。统计近零比例 + 原始 TAF_v 分布。"""
    flat = PGAvf.ravel()  # 所有平地竖向 PGA_v 网格点
    n = flat.size  # 总点数
    res = {"flat_pgav_min": float(np.abs(flat).min()),  # 平地竖向 PGA_v 最小绝对值
           "frac_eq0": float((flat == 0).mean()),  # 严格等于 0 的比例
           "frac_lt_1e4": float((np.abs(flat) < 1e-4).mean()),  # 小于 1e-4 的比例
           "frac_lt_1e3": float((np.abs(flat) < 1e-3).mean()),  # 小于 1e-3 的比例
           "n_points": int(n)}  # 总点数
    safe = np.clip(np.abs(PGAvf), 1e-12, None)  # 防除零的安全分母
    tafv_raw = PGAvs / safe  # 原始 TAF_v（不截断，展示爆炸）
    res["tafv_raw_max"] = float(tafv_raw.max())  # 最大爆炸值
    res["tafv_raw_p99"] = float(np.percentile(tafv_raw, 99))  # 99 分位
    res["frac_tafv_gt10"] = float((tafv_raw > 10).mean())  # 超过 10 的比例（v2 截断阈值）
    return res, tafv_raw  # 返回 + 原始 TAF_v 供画图


def diag_pod(TAFh, PGAh, PGAvs):  # D4：POD 重构误差
    """对 TAF_h/PGA_h/PGA_v(坡) 各做 POD，看降维损失。"""
    n_list = [5, 8, 10, 15, 20]  # 保留成分数列表
    channels = {"TAF_h": TAFh, "PGA_h": PGAh, "PGA_v": PGAvs}  # 通道字典
    res = {}  # 标量结果
    pod_extra = {}  # 画图用（主成分、累计方差、均值）
    for name, arr in channels.items():  # 遍历通道
        X = arr.reshape(-1, NUM_POINTS)  # 展平为 (Nsample,161)
        errs, n99, cum, Vt, mu = pod_recon_error(X, n_list)  # 计算
        res[name] = {"recon_mae": {str(k): v for k, v in errs.items()}, "n_components_99": n99}  # 存标量
        pod_extra[name] = {"cum": cum, "Vt": Vt, "mu": mu}  # 存画图数据
    return res, pod_extra, n_list  # 返回


def diag_waves():  # D5：三波物理特征 + 反应谱
    """从任一可用工况读三条 scaled 波（各工况波相同），算特征 + 反应谱。"""
    folders = sorted(f for f in os.listdir(DATA_DIR) if FOLDER_PATTERN.match(f))  # 工况列表
    feats, rs, waveforms = {}, {}, {}  # 特征/反应谱/时程
    for w in WAVES:  # 遍历三波
        wave_path = None  # 波文件路径
        for folder in folders:  # 找第一个含该波的工况
            p = os.path.join(DATA_DIR, folder, f"{w}_scaled.txt")  # 候选路径
            if os.path.exists(p):  # 存在
                wave_path = p  # 记录
                break  # 用第一个
        if wave_path is None:  # 未找到该波
            continue  # 跳过
        data = np.loadtxt(wave_path)  # 读两列波形（时间、加速度）
        t, accel = data[:, 0], data[:, 1]  # 拆分时间与加速度
        feat, periods, sa = calc_wave_features(t, accel)  # 算特征 + 反应谱
        feats[w] = feat  # 存特征
        rs[w] = (periods, sa)  # 存反应谱
        waveforms[w] = (t, accel)  # 存时程
    return feats, rs, waveforms  # 返回


# ==========================================================================
# 绘图（matplotlib + 中文字体 + 高分辨率 PNG）
# ==========================================================================
def plot_fig1(var_res, cw_xh, ct_xh):  # 图1：地形 vs 波 变异
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1×2 子图
    vals = [var_res["cross_wave_std"], var_res["cross_terrain_std"]]  # 两根柱的值
    bars = axes[0].bar(["跨波\n(同地形3波)", "跨地形\n(同波不同地形)"], vals,
                       color=["#1f77b4", "#d62728"], width=0.5)  # 柱状图
    for b, v in zip(bars, vals):  # 在柱顶标注数值
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=12)
    axes[0].set_ylabel("TAF_h 平均标准差")  # 纵轴标签
    axes[0].set_title(f"地形影响是波的 {var_res['ratio']:.2f} 倍")  # 标题带比值
    axes[1].plot(GRID, cw_xh, color="#1f77b4", label="跨波 std", linewidth=2)  # 跨波 std 沿 x/h
    axes[1].plot(GRID, ct_xh, color="#d62728", label="跨地形 std", linewidth=2)  # 跨地形 std 沿 x/h
    axes[1].set_xlabel("x/h")  # 横轴标签
    axes[1].set_ylabel("TAF_h 标准差")  # 纵轴标签
    axes[1].set_title("沿地表的变异来源对比")  # 标题
    axes[1].legend()  # 图例
    axes[1].grid(True, alpha=0.3)  # 网格
    fig.suptitle("诊断1：地形主导（地形 vs 波 的变异分解）", fontsize=14, fontweight="bold")  # 总标题
    fig.tight_layout()  # 自动布局
    fig.savefig(os.path.join(OUTPUT_DIR, "fig1_variance_terrain_vs_wave.png"), dpi=250, bbox_inches="tight")  # 保存
    plt.close(fig)  # 释放画布


def plot_fig2(base_res):  # 图2：傻瓜基线 vs v2 深度模型
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # 1×2 子图
    labels = ["傻瓜基线"]  # 横轴标签起始
    vals = [base_res["baseline_mae_all"]]  # 值起始
    colors = ["#2ca02c"]  # 基线绿色
    for model in ["cnn", "lstm", "transformer", "deeponet"]:  # 四个深度模型
        if model in base_res["v2_taf_h_mae"]:  # 存在成绩
            labels.append(model.upper())  # 加标签
            vals.append(base_res["v2_taf_h_mae"][model])  # 加值
            colors.append("#d62728")  # 红色
    bars = axes[0].bar(labels, vals, color=colors)  # 柱状图
    for b, v in zip(bars, vals):  # 标注数值
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    axes[0].axhline(base_res["baseline_mae_all"], color="#2ca02c", linestyle="--", alpha=0.7)  # 基线参考线
    axes[0].set_ylabel("TAF_h MAE（全区）")  # 纵轴
    axes[0].set_title("傻瓜基线 打败 v2 深度模型")  # 标题
    axes[0].tick_params(axis="x", rotation=15)  # x 标签旋转
    region_vals = [base_res["baseline_mae_all"], base_res["baseline_mae_peak"], base_res["baseline_mae_far"]]  # 分区值
    rbars = axes[1].bar(["全区", "峰值区\n(x/h≤2)", "远场区\n(x/h≥5)"], region_vals,
                        color=["#2ca02c", "#ff7f0e", "#1f77b4"])  # 分区柱状
    for b, v in zip(rbars, region_vals):  # 标注数值
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    axes[1].set_ylabel("TAF_h MAE")  # 纵轴
    axes[1].set_title("傻瓜基线分区误差")  # 标题
    fig.suptitle("诊断2：傻瓜基线（每地形3波平均）即超过深度模型 = v3 成败线", fontsize=13, fontweight="bold")  # 总标题
    fig.tight_layout()  # 布局
    fig.savefig(os.path.join(OUTPUT_DIR, "fig2_baseline_vs_v2deep.png"), dpi=250, bbox_inches="tight")  # 保存
    plt.close(fig)  # 释放


def plot_fig3(tafv_res, tafv_raw, PGAvf):  # 图3：TAF_v 病态
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))  # 1×2 子图
    for k in range(min(60, PGAvf.shape[0])):  # 抽样部分工况避免过密
        for wv in range(PGAvf.shape[1]):  # 三波
            axes[0].plot(GRID, np.abs(PGAvf[k, wv]), color="gray", alpha=0.15, linewidth=0.6)  # 半透明叠加
    axes[0].axhline(1e-3, color="red", linestyle="--", label="1e-3 阈值")  # 近零阈值线
    axes[0].set_yscale("log")  # 对数纵轴看近零
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))  # g 格式(ASCII 减号)，避免对数轴 mathtext 减号缺字形
    axes[0].set_xlabel("x/h")  # 横轴
    axes[0].set_ylabel("平地竖向 |PGA_v|（对数）")  # 纵轴
    axes[0].set_title(f"平地竖向 PGA_v 近零：min={tafv_res['flat_pgav_min']:.1e}, <1e-3 占 {tafv_res['frac_lt_1e3']*100:.2f}%")  # 标题
    axes[0].legend()  # 图例
    axes[1].hist(np.clip(tafv_raw.ravel(), 0, 20), bins=100, color="#d62728", alpha=0.7)  # 原始 TAF_v 直方图（截显到20）
    axes[1].axvline(10, color="black", linestyle="--", label="v2 clip 上限=10")  # v2 截断线
    axes[1].set_xlabel("原始 TAF_v（未截断，截显到 20）")  # 横轴
    axes[1].set_ylabel("频数（对数）")  # 纵轴
    axes[1].set_yscale("log")  # 对数频数看长尾
    axes[1].set_title(f"TAF_v 爆炸：max={tafv_res['tafv_raw_max']:.0f}, >10 占 {tafv_res['frac_tafv_gt10']*100:.2f}%")  # 标题
    axes[1].legend()  # 图例
    fig.suptitle("诊断3：TAF_v 物理病态（斜入射 SV 波下平地竖向 PGA 近零）", fontsize=13, fontweight="bold")  # 总标题
    fig.tight_layout()  # 布局
    fig.savefig(os.path.join(OUTPUT_DIR, "fig3_tafv_pathology.png"), dpi=250, bbox_inches="tight")  # 保存
    plt.close(fig)  # 释放


def plot_fig4(pod_res, pod_extra, n_list):  # 图4：POD 分析
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))  # 1×3 子图
    Vt = pod_extra["TAF_h"]["Vt"]  # TAF_h 主成分
    mu = pod_extra["TAF_h"]["mu"]  # TAF_h 均值曲线
    axes[0].plot(GRID, mu, "k-", linewidth=2, label="均值曲线")  # 均值
    for nn in range(4):  # 前 4 个主成分
        axes[0].plot(GRID, Vt[nn], linewidth=1.2, label=f"主成分 {nn+1}")  # 形状
    axes[0].set_xlabel("x/h")  # 横轴
    axes[0].set_ylabel("形状")  # 纵轴
    axes[0].set_title("TAF_h 的 POD 基本形状（前4）")  # 标题
    axes[0].legend(fontsize=8)  # 图例
    axes[0].grid(True, alpha=0.3)  # 网格
    for name, color in zip(["TAF_h", "PGA_h", "PGA_v"], ["#d62728", "#1f77b4", "#2ca02c"]):  # 三通道
        errs = [pod_res[name]["recon_mae"][str(nn)] for nn in n_list]  # 各 N 误差
        axes[1].plot(n_list, errs, "o-", color=color, label=name)  # 衰减曲线
    axes[1].set_xlabel("保留主成分数 N")  # 横轴
    axes[1].set_ylabel("重构 MAE")  # 纵轴
    axes[1].set_title("POD 重构误差随 N 衰减")  # 标题
    axes[1].legend()  # 图例
    axes[1].grid(True, alpha=0.3)  # 网格
    for name, color in zip(["TAF_h", "PGA_h", "PGA_v"], ["#d62728", "#1f77b4", "#2ca02c"]):  # 三通道
        cum = pod_extra[name]["cum"]  # 累计方差
        kk = min(30, len(cum))  # 取前 30 个成分
        axes[2].plot(range(1, kk + 1), cum[:kk], "-", color=color, label=name)  # 累计方差曲线
    axes[2].axhline(0.99, color="black", linestyle="--", label="99%")  # 99% 参考线
    axes[2].set_xlabel("主成分数")  # 横轴
    axes[2].set_ylabel("累计方差解释率")  # 纵轴
    axes[2].set_title("累计方差解释率")  # 标题
    axes[2].legend()  # 图例
    axes[2].grid(True, alpha=0.3)  # 网格
    fig.suptitle("诊断4：POD 降维几乎无损（161维→15维，重构误差远小于模型误差0.05）", fontsize=13, fontweight="bold")  # 总标题
    fig.tight_layout()  # 布局
    fig.savefig(os.path.join(OUTPUT_DIR, "fig4_pod_analysis.png"), dpi=250, bbox_inches="tight")  # 保存
    plt.close(fig)  # 释放


def plot_fig5(feats, rs, waveforms):  # 图5：三波时程 + 反应谱
    fig = plt.figure(figsize=(14, 9))  # 画布
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.3])  # 上3时程下1反应谱
    for idx, w in enumerate(WAVES):  # 遍历三波画时程
        if w not in waveforms:  # 缺失
            continue
        ax = fig.add_subplot(gs[0, idx])  # 时程子图
        t, accel = waveforms[w]  # 时程数据
        ax.plot(t, accel, color=WAVE_COLORS[idx], linewidth=0.7)  # 画时程
        ax.set_xlim(0, min(40, float(t.max())))  # 横轴范围
        ax.set_title(f"{WAVE_CN[w]}  (D5-95={feats[w]['D5_95']:.1f}s, Tp={feats[w]['Tp']:.2f}s)", fontsize=10)  # 标题带特征（D5-95 用 ASCII，避免 SimHei 缺下标字形）
        ax.set_xlabel("时间 (s)")  # 横轴
        ax.grid(True, alpha=0.3)  # 网格
        if idx == 0:  # 仅首图加纵轴
            ax.set_ylabel("加速度 (g)")
    ax_rs = fig.add_subplot(gs[1, :])  # 反应谱大图（跨整行）
    for idx, w in enumerate(WAVES):  # 三波反应谱叠加
        if w not in rs:  # 缺失
            continue
        periods, sa = rs[w]  # 反应谱
        ax_rs.plot(periods, sa, color=WAVE_COLORS[idx], linewidth=1.5, label=WAVE_CN[w])  # 画谱
    for T in SA_PERIODS_TARGET:  # 标代表周期
        ax_rs.axvline(T, color="gray", linestyle=":", alpha=0.5)  # 竖虚线
    ax_rs.set_xscale("log")  # 对数周期轴
    ax_rs.set_xlim(0.01, 10)  # 周期范围
    ax_rs.set_xticks([0.01, 0.1, 1.0, 10.0])  # 主刻度
    ax_rs.set_xticklabels(["0.01", "0.1", "1", "10"])  # 普通文本标签，避免 mathtext 减号缺字形
    ax_rs.set_xlabel("周期 T (s)")  # 横轴
    ax_rs.set_ylabel("谱加速度 Sa (g)")  # 纵轴
    ax_rs.set_title("5% 阻尼弹性反应谱（三波 PGA 都=0.3，区别全在频谱形状）")  # 标题
    ax_rs.legend()  # 图例
    ax_rs.grid(True, which="both", alpha=0.3)  # 网格
    fig.suptitle("诊断5：三条波的指纹（PGA_in 是常数，靠持时/卓越周期/反应谱区分）", fontsize=13, fontweight="bold")  # 总标题
    fig.tight_layout()  # 布局
    fig.savefig(os.path.join(OUTPUT_DIR, "fig5_three_waves.png"), dpi=250, bbox_inches="tight")  # 保存
    plt.close(fig)  # 释放


# ==========================================================================
# 主流程
# ==========================================================================
def main():  # 主入口
    print("=" * 60)  # 分隔线
    print("ML v3 — Stage 0 诊断（只读分析）")  # 标题
    print("=" * 60)  # 分隔线
    print(">> 加载全部工况曲线...")  # 提示
    geom, TAFh, PGAh, PGAvs, PGAvf = load_all_curves()  # 加载数据

    all_res = {}  # 汇总结果

    print("\n[诊断1] 地形 vs 波 变异")  # 段标题
    var_res, cw_xh, ct_xh = diag_variance(TAFh)  # 计算
    print(f"  跨波 std       = {var_res['cross_wave_std']:.4f}（方案预期 0.052）")  # 打印
    print(f"  跨地形 std     = {var_res['cross_terrain_std']:.4f}（方案预期 0.129）")  # 打印
    print(f"  地形/波 比值   = {var_res['ratio']:.2f}（方案预期 2.47）")  # 打印
    print(f"  TAF_h 范围     = [{var_res['taf_h_min']:.3f}, {var_res['taf_h_max']:.3f}], 均值 {var_res['taf_h_mean']:.3f}")  # 打印
    all_res["D1_variance"] = var_res  # 存

    print("\n[诊断2] 傻瓜基线 vs v2 深度模型")  # 段标题
    base_res = diag_baseline(TAFh)  # 计算
    print(f"  傻瓜基线 MAE 全区   = {base_res['baseline_mae_all']:.4f}（方案预期 0.0469）")  # 打印
    print(f"  傻瓜基线 MAE 峰值区 = {base_res['baseline_mae_peak']:.4f}（方案预期 0.0480）")  # 打印
    print(f"  傻瓜基线 MAE 远场区 = {base_res['baseline_mae_far']:.4f}（方案预期 0.0473）")  # 打印
    for model, mae in base_res["v2_taf_h_mae"].items():  # 打印 v2 对照
        flag = "（基线胜）" if base_res["baseline_mae_all"] < mae else "（基线负）"  # 比较标记
        print(f"  v2 {model:12s} TAF_h MAE = {mae:.4f} {flag}")  # 打印
    all_res["D2_baseline"] = base_res  # 存

    print("\n[诊断3] TAF_v 物理病态")  # 段标题
    tafv_res, tafv_raw = diag_tafv(PGAvs, PGAvf)  # 计算
    print(f"  平地竖向 PGA_v min = {tafv_res['flat_pgav_min']:.2e}（方案预期 0/近零）")  # 打印
    print(f"  <1e-3 占比 = {tafv_res['frac_lt_1e3']*100:.3f}%（方案预期 ~0.2%）")  # 打印
    print(f"  <1e-4 占比 = {tafv_res['frac_lt_1e4']*100:.3f}%（方案预期 ~0.05%）")  # 打印
    print(f"  原始 TAF_v max = {tafv_res['tafv_raw_max']:.1f}, >10 占 {tafv_res['frac_tafv_gt10']*100:.2f}%")  # 打印
    all_res["D3_tafv"] = tafv_res  # 存

    print("\n[诊断4] POD 重构误差")  # 段标题
    pod_res, pod_extra, n_list = diag_pod(TAFh, PGAh, PGAvs)  # 计算
    for name in ["TAF_h", "PGA_h", "PGA_v"]:  # 三通道
        e = pod_res[name]["recon_mae"]  # 误差字典
        print(f"  {name}: N=5→{e['5']:.4f}, N=10→{e['10']:.4f}, N=15→{e['15']:.4f}; 99%方差需 {pod_res[name]['n_components_99']} 个成分")  # 打印
    all_res["D4_pod"] = pod_res  # 存

    print("\n[诊断5] 三波物理特征 + 反应谱（反应谱计算约需十几秒）")  # 段标题
    feats, rs, waveforms = diag_waves()  # 计算
    for w in WAVES:  # 三波
        if w in feats:  # 存在
            f = feats[w]  # 特征
            print(f"  {w:12s}: PGA_in={f['PGA_in']:.3f}, Arias={f['Arias']:.3f}, "
                  f"D5-95={f['D5_95']:.1f}s, Tp={f['Tp']:.2f}s, Tm={f['Tm']:.2f}s")  # 打印
    all_res["D5_waves"] = feats  # 存（仅标量特征）

    json_path = os.path.join(OUTPUT_DIR, "diagnostics_v3.json")  # JSON 路径
    with open(json_path, "w", encoding="utf-8") as fh:  # 打开
        json.dump(all_res, fh, ensure_ascii=False, indent=2)  # 写入
    print(f"\n>> 诊断数字已存: {json_path}")  # 提示

    print(">> 生成诊断图...")  # 提示
    plot_fig1(var_res, cw_xh, ct_xh)  # 图1
    plot_fig2(base_res)  # 图2
    plot_fig3(tafv_res, tafv_raw, PGAvf)  # 图3
    plot_fig4(pod_res, pod_extra, n_list)  # 图4
    plot_fig5(feats, rs, waveforms)  # 图5
    print(f">> 5 张诊断图已存到: {OUTPUT_DIR}")  # 提示
    print("=" * 60)  # 分隔
    print("Stage 0 诊断完成。")  # 完成


if __name__ == "__main__":  # 主程序入口
    main()  # 运行
