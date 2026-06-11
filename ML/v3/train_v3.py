# -*- coding: utf-8 -*-
"""ML v3 — Stage 2 主模型训练与评估。

流程（Form B — POD 压缩 + 经典 ML）：
  1. 加载数据：TAF_h 曲线 (N_geom=174, 3波, 161点) + 几何参数
  2. 构建特征矩阵 X：几何特征 [h, i, angle, log_h, tan_i] + 波物理特征（11 个标量）
     （波特征从 outputs_v3/diagnostics_v3.json 读取，已预计算；三条波被统一调幅
      到 PGA=0.3，故 PGA_in 是常数，已剔除）
  3. 5 折 GroupKFold（按几何工况分组，确保同地形的 3 条波同进退，防数据泄漏）
  4. 每折内：PCA 拟合（仅训练集）→ 预测 15 个 POD 系数 → 逆 PCA 还原曲线
  5. 模型：XGBoost（MultiOutputRegressor）& GPR（MultiOutputRegressor）
  6. 评估：MAE / RMSE / R2（全区 / 峰值区 / 远场区），与 Stage 1 基线对比
  7. 可视化：特征重要性、Parity 图、随机曲线对比、GPR 不确定性带

产出（均写入 outputs_v3/）：
  train_metrics.json           — 各模型各区指标
  fig7_feature_importance.png  — XGBoost 特征重要性
  fig8_parity.png              — Predicted vs True (Parity Plot)
  fig9_curve_compare.png       — 6 个工况曲线对比
  fig10_gpr_uncertainty.png    — GPR 不确定性带（留一波）
  fig11_v2_v3_compare.png      — v2 深度模型 vs v3 经典 ML 对照表图

运行：py -3 ML/train_v3.py
"""

import os  # 路径操作
import re  # 正则（解析文件夹名）
import json  # JSON 读写
import numpy as np  # 数值计算
import pandas as pd  # CSV 读取
from sklearn.decomposition import PCA  # POD 主成分分析（与 POD 等价）
from sklearn.preprocessing import StandardScaler  # GPR 需要标准化特征
from sklearn.multioutput import MultiOutputRegressor  # 多输出回归包装器
from sklearn.model_selection import GroupKFold  # 按组分折
from sklearn.gaussian_process import GaussianProcessRegressor  # GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # GPR 核函数
import xgboost as xgb  # XGBoost 梯度提升树
import matplotlib  # 绘图框架
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt  # 绘图子模块
import matplotlib.gridspec as gridspec  # 复杂子图布局

# ==========================================================================
# 配置区
# ==========================================================================
DATA_DIR   = r"E:\Abaqus\fuke-ALL"  # 原始数据根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录（ML/）
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_v3")  # v3 输出目录
DIAG_JSON  = os.path.join(OUTPUT_DIR, "diagnostics_v3.json")  # Stage 0 诊断结果
BASE_JSON  = os.path.join(OUTPUT_DIR, "baseline_metrics.json")  # Stage 1 基线结果
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录

WAVES   = ["El_Centro", "Loma_Prieta", "Northridge"]  # 三条地震波
N_PTS   = 161  # 统一曲线点数
GRID    = np.linspace(0.0, 8.0, N_PTS)  # x/h ∈ [0,8]
PEAK    = GRID <= 2.0  # 峰值区掩码
FAR     = GRID >= 5.0  # 远场区掩码
PAT     = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 文件夹名解析
N_FOLDS = 5  # GroupKFold 折数

POD_N   = 15  # POD 保留主成分数（TAF_h 重构 MAE~0.012，远小于模型噪声 ~0.05）

# XGBoost 超参数（max_depth 限制防过拟合，小数据铁律）
XGB_PARAMS = dict(
    n_estimators=400,       # 树的数量
    max_depth=4,            # 最大树深，限制过拟合
    learning_rate=0.05,     # 学习率（慢步细调）
    subsample=0.8,          # 行采样比例
    colsample_bytree=0.8,   # 列采样比例
    reg_lambda=1.0,         # L2 正则项
    tree_method="hist",     # 直方图算法，快且省内存
    random_state=42,        # 随机种子
    n_jobs=-1,              # 并行使用所有 CPU 核
)

plt.rcParams.update({  # 全局绘图参数
    "font.sans-serif":    ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi":         150,
})

# ==========================================================================
# 波物理特征表（从 diagnostics_v3.json 预加载；三波 PGA 均=0.3，PGA_in 已剔除）
# 若 DIAG_JSON 不存在，使用硬编码的已知数值（来自 ml_plan_v3.md 第 4.2 节）
# ==========================================================================
_WAVE_FEATS_FALLBACK = {
    "El_Centro":   dict(Arias=0.01782, D5_95=23.86, Tp=0.6638, Tm=0.5404,
                        PGV=0.3553, PGD=0.2097,
                        Sa_01=0.6346, Sa_02=0.7138, Sa_05=0.9004,
                        Sa_10=0.4468, Sa_20=0.1364),
    "Loma_Prieta": dict(Arias=0.00876, D5_95=11.45, Tp=1.5964, Tm=0.6373,
                        PGV=0.3600, PGD=0.1579,
                        Sa_01=0.6805, Sa_02=1.1044, Sa_05=0.5658,
                        Sa_10=0.3048, Sa_20=0.2410),
    "Northridge":  dict(Arias=0.00769, D5_95=9.07,  Tp=0.8141, Tm=0.5520,
                        PGV=0.2703, PGD=0.0472,
                        Sa_01=0.4166, Sa_02=0.6360, Sa_05=0.5100,
                        Sa_10=0.2786, Sa_20=0.1227),
}

def load_wave_features():
    """从 diagnostics_v3.json 读取波物理特征，剔除 PGA_in（常数零信息）。"""
    if os.path.exists(DIAG_JSON):  # 优先读 Stage 0 已计算结果
        with open(DIAG_JSON, encoding="utf-8") as f:
            diag = json.load(f)
        raw = diag.get("D5_waves", {})  # 原始键名（如 "Sa_0.1"）
        feats = {}
        for w in WAVES:
            d = raw.get(w, {})
            feats[w] = dict(
                Arias=d["Arias"], D5_95=d["D5_95"], Tp=d["Tp"], Tm=d["Tm"],
                PGV=d["PGV"],     PGD=d["PGD"],
                Sa_01=d["Sa_0.1"], Sa_02=d["Sa_0.2"], Sa_05=d["Sa_0.5"],
                Sa_10=d["Sa_1.0"], Sa_20=d["Sa_2.0"],
            )
        return feats
    print("[警告] diagnostics_v3.json 不存在，使用硬编码波特征")
    return _WAVE_FEATS_FALLBACK  # 回退到硬编码


WAVE_FEATS = load_wave_features()  # 全局波特征字典
WAVE_FEAT_NAMES = list(list(WAVE_FEATS.values())[0].keys())  # 特征名称列表（11 个）

# 几何特征名称（派生特征 log_h / tan_i 增加物理尺度线性性）
GEO_FEAT_NAMES = ["h", "i", "angle", "log_h", "tan_i"]
FEAT_NAMES = GEO_FEAT_NAMES + WAVE_FEAT_NAMES  # 总特征名（5+11=16 维）


# ==========================================================================
# 辅助函数
# ==========================================================================

def read_csv(path):  # 稳健 CSV 读取
    """utf-8-sig 读取，自动剥掉 BOM。"""
    return pd.read_csv(path, encoding="utf-8-sig")


def compute_metrics(y_true, y_pred, mask=None):
    """MAE / RMSE / R2（支持区间掩码）。"""
    if mask is not None:
        y_true, y_pred = y_true[:, mask], y_pred[:, mask]
    err   = y_true - y_pred
    mae   = float(np.abs(err).mean())
    rmse  = float(np.sqrt((err ** 2).mean()))
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def eval_all_zones(y_true, y_pred):
    """全区 / 峰值区 / 远场区 指标汇总。"""
    return {
        "all":  compute_metrics(y_true, y_pred),
        "peak": compute_metrics(y_true, y_pred, PEAK),
        "far":  compute_metrics(y_true, y_pred, FAR),
    }


# ==========================================================================
# 数据加载
# ==========================================================================

def load_dataset():
    """加载所有三波完整工况的 TAF_h 曲线，返回:
      geoms : list of (h, i, angle) tuples，N_geom 个
      tafh  : ndarray (N_geom, 3, 161)
    """
    folders = sorted([f for f in os.listdir(DATA_DIR) if PAT.match(f)])
    geoms, curves, missing = [], [], 0
    for f in folders:
        m = PAT.match(f)
        h, i, angle = float(m[1]), float(m[2]), float(m[3])
        arr = []
        for w in WAVES:
            p = os.path.join(DATA_DIR, f, f"TAF-{w}.csv")
            if not os.path.exists(p):
                break
            d = read_csv(p).sort_values("x/h")
            arr.append(np.interp(GRID, d["x/h"].values, d["TAF_h"].values))
        if len(arr) == 3:
            geoms.append((h, i, angle))
            curves.append(arr)
        else:
            missing += 1
    print(f"数据加载：{len(geoms)} 个几何工况（跳过 {missing} 个不完整）")
    return geoms, np.array(curves)  # (N_geom, 3, 161)


# ==========================================================================
# 特征矩阵构建
# ==========================================================================

def build_X_y(geoms, tafh):
    """将 (N_geom, 3波, 161点) 数据展开为样本行，构建特征矩阵 X 和目标矩阵 y。

    每行 = 一个 (几何工况, 地震波) 样本，共 N_geom × 3 = N 行。

    Returns
    -------
    X      : ndarray (N, 16)，特征矩阵
    y      : ndarray (N, 161)，TAF_h 目标曲线
    groups : ndarray (N,)，GroupKFold 分组标签（同几何工况 = 同 group）
    meta   : list of (h, i, angle, wave_name) 元信息，用于绘图标注
    """
    rows_X, rows_y, groups_list, meta = [], [], [], []
    for geom_idx, (h, i, angle) in enumerate(geoms):  # 逐几何工况
        log_h = np.log(h)  # 对数化，使等比分布的 h 值在特征空间均匀
        tan_i = np.tan(np.radians(i))  # 坡角正切（坡度物理量）
        geo_vec = [h, i, angle, log_h, tan_i]  # 5 维几何特征向量

        for w_idx, w_name in enumerate(WAVES):  # 逐波展开
            wf = WAVE_FEATS[w_name]  # 11 维波特征
            wave_vec = [wf[k] for k in WAVE_FEAT_NAMES]  # 按固定顺序排列
            rows_X.append(geo_vec + wave_vec)  # 合并为 16 维特征向量
            rows_y.append(tafh[geom_idx, w_idx])  # 对应的 161 点 TAF_h 曲线
            groups_list.append(geom_idx)  # 同一地形 → 同一 group
            meta.append((h, i, angle, w_name))  # 元信息

    return (np.array(rows_X, dtype=np.float64),
            np.array(rows_y, dtype=np.float64),
            np.array(groups_list, dtype=int),
            meta)


# ==========================================================================
# GroupKFold CV 框架（POD 在折内拟合，无泄漏）
# ==========================================================================

def run_cv(X, y, groups, model_factory, n_components=POD_N, label=""):
    """5 折 GroupKFold CV：折内拟合 PCA + 模型，返回 OOF (out-of-fold) 预测。

    每折：
      1. 在训练集 TAF_h 曲线上拟合 PCA（降到 n_components 维）
      2. 训练集目标 → POD 系数
      3. 拟合模型（预测 POD 系数）
      4. 测试集预测 POD 系数 → 逆 PCA 还原曲线

    Parameters
    ----------
    model_factory : callable() -> sklearn-like estimator（每折重新实例化）

    Returns
    -------
    y_true, y_pred_oof : ndarray (N, 161)，按原始样本顺序拼合
    fold_scalers       : list of (PCA, StandardScaler) 最后一折的对象（可供外推用）
    feat_importances   : ndarray (n_components, n_features) or None（仅 XGB 支持）
    """
    kf = GroupKFold(n_splits=N_FOLDS)
    y_pred_oof = np.full_like(y, np.nan)  # 初始化为 NaN，逐折填充
    feat_imps = []  # 收集各折特征重要性（仅 XGB）

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X, groups=groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ── POD：在训练集曲线上拟合 PCA ────────────────────────────
        pca = PCA(n_components=n_components)  # 主成分分析（POD）
        coeff_tr = pca.fit_transform(y_tr)    # (N_tr, n_comp)，训练集 POD 系数
        coeff_te_shape = (len(te_idx), n_components)  # 测试集系数形状（仅用于形状检查）

        # ── 特征标准化（XGB 不需要但无害；GPR 必须）───────────────
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)  # 用训练集均值/标准差拟合
        X_te_sc = scaler.transform(X_te)      # 测试集用训练集统计量变换（防泄漏）

        # ── 训练模型（MultiOutputRegressor 多输出包装）────────────
        model = model_factory()  # 每折重新实例化（避免跨折信息泄漏）
        model.fit(X_tr_sc, coeff_tr)          # 输入特征 → 输出 POD 系数

        # ── 预测并逆 POD 还原曲线 ─────────────────────────────────
        coeff_pred = model.predict(X_te_sc)   # (N_te, n_comp)
        y_pred_oof[te_idx] = pca.inverse_transform(coeff_pred)  # 还原到 161 点曲线

        # ── 收集 XGBoost 特征重要性（各子模型平均）─────────────────
        if hasattr(model, "estimators_"):  # MultiOutputRegressor 属性
            est = model.estimators_[0]
            if hasattr(est, "feature_importances_"):  # XGB 特有
                fi = np.stack([e.feature_importances_ for e in model.estimators_])
                feat_imps.append(fi)  # (n_comp, n_features)

        N_te = len(te_idx)
        print(f"  [{label}] fold {fold_idx+1}/{N_FOLDS}：训练 {len(tr_idx)} | 测试 {N_te} 样本")

    feat_imp_mean = np.stack(feat_imps).mean(0) if feat_imps else None  # 跨折平均特征重要性
    return y, y_pred_oof, feat_imp_mean


# ==========================================================================
# 模型工厂
# ==========================================================================

def make_xgb():
    """创建 XGBoost 多输出回归器。"""
    base = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)  # 单输出 XGB（关闭日志）
    return MultiOutputRegressor(base, n_jobs=1)  # 多输出包装（n_jobs=1 避免并行嵌套）


def make_gpr():
    """创建 GPR 多输出回归器（RBF 核 + 噪声核）。"""
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))  # 幅度核（常数因子）
              * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))  # RBF 核（长度尺度）
              + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e1)))  # 噪声核
    base = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,          # 对目标去均值，稳定拟合
        n_restarts_optimizer=3,    # 多次随机初始化优化核超参，避免陷入局部最优
        random_state=42,
    )
    return MultiOutputRegressor(base, n_jobs=-1)  # 并行拟合 15 个 GPR


# ==========================================================================
# 留一波 GPR（展示不确定性）
# ==========================================================================

def run_leave_one_wave(X, y, groups, wave_indices):
    """3 折留一波 CV（每次藏起一整条波），返回 OOF 预测和 GPR 不确定性。

    wave_indices : ndarray (N,)，每个样本对应的波索引（0/1/2）

    Returns
    -------
    y_true_loo, y_pred_loo : ndarray (N, 161)
    y_std_loo              : ndarray (N, 161)，预测标准差（GPR 不确定性）
    """
    y_pred_loo = np.full_like(y, np.nan)
    y_std_loo  = np.full_like(y, np.nan)

    for w_idx, w_name in enumerate(WAVES):  # 逐波作为测试集
        te_mask = (wave_indices == w_idx)   # 测试集掩码（当前波）
        tr_mask = ~te_mask                  # 训练集掩码（另外两条波）

        X_tr, X_te = X[tr_mask], X[te_mask]
        y_tr, y_te = y[tr_mask], y[te_mask]

        # ── 折内 POD + GPR ────────────────────────────────────────
        pca = PCA(n_components=POD_N)
        coeff_tr = pca.fit_transform(y_tr)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # 逐 POD 成分拟合 GPR，逐一收集均值和标准差
        coeff_pred_mean = np.zeros((len(X_te), POD_N))  # (N_te, POD_N) 均值
        coeff_pred_std  = np.zeros((len(X_te), POD_N))  # (N_te, POD_N) 标准差
        kernel = (ConstantKernel(1.0, (1e-3, 1e3))
                  * RBF(1.0, (1e-2, 1e2))
                  + WhiteKernel(1e-2, (1e-5, 1e1)))

        for comp in range(POD_N):  # 逐主成分拟合单输出 GPR
            gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                           n_restarts_optimizer=2, random_state=42)
            gpr.fit(X_tr_sc, coeff_tr[:, comp])
            mu, sigma = gpr.predict(X_te_sc, return_std=True)
            coeff_pred_mean[:, comp] = mu
            coeff_pred_std[:, comp]  = sigma

        # ── 还原到曲线空间 ────────────────────────────────────────
        y_pred_loo[te_mask] = pca.inverse_transform(coeff_pred_mean)

        # 不确定性传播：sigma_curve ≈ sqrt(sum_k (V_k * sigma_k)^2)，V_k 是 PCA 向量
        Vt = pca.components_  # (POD_N, 161)
        # 逐样本计算曲线标准差
        for j_te in range(coeff_pred_std.shape[0]):
            std_curve = np.sqrt(np.sum((Vt * coeff_pred_std[j_te, :, None]) ** 2, axis=0))
            y_std_loo[te_mask][j_te] = std_curve  # 写入对应行

        print(f"  [留一波] 藏起 {w_name}：训练 {tr_mask.sum()} | 测试 {te_mask.sum()} 样本")

    return y, y_pred_loo, y_std_loo


# ==========================================================================
# 绘图函数
# ==========================================================================

def plot_feature_importance(feat_imp, feat_names, outpath):
    """XGBoost 特征重要性柱状图（各 POD 成分重要性平均）。"""
    imp_mean = feat_imp.mean(axis=0)   # (n_features,)，跨 POD 成分取均值
    order    = np.argsort(imp_mean)[::-1]  # 从高到低排序

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4" if n not in GEO_FEAT_NAMES else "#d62728"
              for n in [feat_names[i] for i in order]]  # 几何特征红色，波特征蓝色
    bars = ax.bar(range(len(feat_names)), imp_mean[order], color=colors)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels([feat_names[i] for i in order], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("平均特征重要性（增益，跨 POD 成分）", fontsize=10)
    ax.set_title("XGBoost 特征重要性\n（红色=几何特征，蓝色=波物理特征）", fontsize=11)

    # 添加图例说明
    from matplotlib.patches import Patch
    handles = [Patch(color="#d62728", label="几何特征 (h/i/angle/log_h/tan_i)"),
               Patch(color="#1f77b4", label="波特征 (Arias/Sa(T)/...）")]
    ax.legend(handles=handles, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


def plot_parity(y_true, y_pred_xgb, y_pred_gpr, outpath):
    """Parity Plot（预测值 vs 真实值散点，4 通道取 TAF_h 单通道逐点）。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("TAF_h Parity Plot（预测值 vs 真实值）", fontsize=13, fontweight="bold")

    for ax, y_pred, label, color in [
        (axes[0], y_pred_xgb, "XGBoost", "#1f77b4"),
        (axes[1], y_pred_gpr, "GPR",     "#ff7f0e"),
    ]:
        y_t = y_true.ravel()   # 展开成一维（所有样本所有点）
        y_p = y_pred.ravel()
        # 随机抽 5000 点防止散点过密
        rng  = np.random.default_rng(0)
        idx  = rng.choice(len(y_t), size=min(5000, len(y_t)), replace=False)
        ax.scatter(y_t[idx], y_p[idx], s=2, alpha=0.3, color=color, rasterized=True)
        lim = [y_t.min() * 0.95, y_t.max() * 1.05]
        ax.plot(lim, lim, "k--", linewidth=1)  # 对角线（完美预测）
        m = compute_metrics(y_true, y_pred)  # 全区指标
        ax.set_title(f"{label}\nMAE={m['mae']:.4f}  R2={m['r2']:.4f}", fontsize=11)
        ax.set_xlabel("真实 TAF_h", fontsize=10)
        ax.set_ylabel("预测 TAF_h", fontsize=10)
        ax.set_xlim(lim); ax.set_ylim(lim)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


def plot_curve_compare(y_true, y_pred_xgb, y_pred_gpr, meta, outpath, n_show=6):
    """随机抽取 n_show 个工况，绘制真实 vs 预测 TAF_h 曲线对比。"""
    rng      = np.random.default_rng(7)
    sel_idx  = rng.choice(len(y_true), size=n_show, replace=False)  # 随机选样本
    n_col, n_row = 3, 2  # 布局：2行3列

    fig, axes = plt.subplots(n_row, n_col, figsize=(14, 7))
    fig.suptitle("TAF_h 曲线对比（真实 vs 预测）", fontsize=13, fontweight="bold")

    for plot_i, sample_idx in enumerate(sel_idx):
        ax = axes[plot_i // n_col, plot_i % n_col]
        h, i, angle, wave = meta[sample_idx]
        ax.plot(GRID, y_true[sample_idx],    "k-",  linewidth=1.5, label="FEM 真实")
        ax.plot(GRID, y_pred_xgb[sample_idx], "b--", linewidth=1.2, label="XGBoost")
        ax.plot(GRID, y_pred_gpr[sample_idx], "r-.", linewidth=1.2, label="GPR")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)  # TAF=1 参考线
        title = f"h={h:.0f}m  i={i:.0f}°  angle={angle:.0f}°  {wave.split('_')[0]}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x/h", fontsize=8)
        ax.set_ylabel("TAF_h", fontsize=8)
        if plot_i == 0:
            ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


def plot_gpr_uncertainty(y_true, y_pred_loo, y_std_loo, wave_indices, meta, outpath):
    """留一波 GPR 不确定性带：每条波随机抽 2 个几何工况展示。"""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle("GPR 留一波不确定性带（预测 ± 2σ）\n[注意：3 折留一波是能力上限测试，3 条波只能外推]",
                 fontsize=11, fontweight="bold")
    rng = np.random.default_rng(42)

    for wave_i, wave_name in enumerate(WAVES):  # 逐波一行
        mask  = wave_indices == wave_i  # 当前波的样本掩码
        idx_w = np.where(mask)[0]      # 当前波的样本索引
        sel   = rng.choice(idx_w, size=min(2, len(idx_w)), replace=False)  # 随机 2 个样本

        for col, sample_idx in enumerate(sel):
            ax   = axes[wave_i, col]
            mu   = y_pred_loo[sample_idx]   # GPR 预测均值
            std  = y_std_loo[sample_idx]    # GPR 预测标准差
            true = y_true[sample_idx]       # FEM 真实曲线
            h, i, angle, wn = meta[sample_idx]

            ax.plot(GRID, true, "k-",  linewidth=1.5, label="FEM 真实")
            ax.plot(GRID, mu,   "r--", linewidth=1.2, label="GPR 均值")
            ax.fill_between(GRID, mu - 2*std, mu + 2*std,
                            alpha=0.25, color="red", label="±2σ")
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
            mae_s = float(np.abs(true - mu).mean())
            ax.set_title(f"波: {wave_name.replace('_',' ')} | h={h:.0f}m i={i:.0f}° angle={angle:.0f}°\n"
                         f"MAE={mae_s:.4f}", fontsize=8)
            ax.set_xlabel("x/h", fontsize=8)
            ax.set_ylabel("TAF_h", fontsize=8)
            if wave_i == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


def plot_v2_v3_comparison(results, v2_maes, outpath):
    """v2 深度模型 vs v3 经典 ML 对照柱状图（全区 TAF_h MAE）。"""
    methods = [
        ("B1 全局均值",        results["B1"]["all"]["mae"],     "#aaaaaa"),
        ("B3 网格插值",        results["B3"]["all"]["mae"],     "#87ceeb"),
        ("v2 CNN",             v2_maes.get("cnn", np.nan),      "#ff7f0e"),
        ("v2 LSTM",            v2_maes.get("lstm", np.nan),     "#9467bd"),
        ("v2 Trans.",          v2_maes.get("transformer", np.nan), "#8c564b"),
        ("v2 DeepONet",        v2_maes.get("deeponet", np.nan), "#d62728"),
        ("v3 XGBoost",         results["XGB"]["all"]["mae"],    "#1f77b4"),
        ("v3 GPR",             results["GPR"]["all"]["mae"],    "#2ca02c"),
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    x    = np.arange(len(methods))
    vals = [m[1] for m in methods]
    cols = [m[2] for m in methods]
    bars = ax.bar(x, vals, color=cols, edgecolor="white", linewidth=0.6)

    # 标注理论下限（B2 oracle）
    b2_mae = results["B2"]["all"]["mae"]
    ax.axhline(b2_mae, color="#2ca02c", linestyle="--", linewidth=1.5,
               label=f"B2 oracle 理论下限 = {b2_mae:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels([m[0] for m in methods], rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("TAF_h MAE（全区，GroupKFold CV）", fontsize=10)
    ax.set_title("v2 深度学习 vs v3 经典 ML 全面对照", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, vals):  # 柱顶数值
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


# ==========================================================================
# 主流程
# ==========================================================================

def main():
    print("=" * 68)
    print("ML v3 — Stage 2 主模型训练（POD + XGBoost / GPR）")
    print("=" * 68)
    print(f"波特征维度: {len(WAVE_FEAT_NAMES)}  几何特征维度: {len(GEO_FEAT_NAMES)}")
    print(f"总特征维度: {len(FEAT_NAMES)}  POD 主成分数: {POD_N}")

    # ── 1. 加载数据 ───────────────────────────────────────────────
    geoms, tafh = load_dataset()

    # ── 2. 构建特征矩阵 ───────────────────────────────────────────
    X, y, groups, meta = build_X_y(geoms, tafh)
    print(f"样本矩阵 X: {X.shape}  目标矩阵 y: {y.shape}")

    # wave_indices：每个样本对应哪条波（供留一波 CV 使用）
    wave_indices = np.array([WAVES.index(m[3]) for m in meta])

    # ── 3. XGBoost GroupKFold CV ─────────────────────────────────
    print(f"\n── XGBoost（{N_FOLDS} 折 GroupKFold）")
    y_true, y_pred_xgb, feat_imp = run_cv(X, y, groups, make_xgb, label="XGB")
    res_xgb = eval_all_zones(y_true, y_pred_xgb)
    print(f"  全区  MAE={res_xgb['all']['mae']:.4f}  RMSE={res_xgb['all']['rmse']:.4f}  R2={res_xgb['all']['r2']:.4f}")
    print(f"  峰值区 MAE={res_xgb['peak']['mae']:.4f}  R2={res_xgb['peak']['r2']:.4f}")
    print(f"  远场区 MAE={res_xgb['far']['mae']:.4f}  R2={res_xgb['far']['r2']:.4f}")

    # ── 4. GPR GroupKFold CV ─────────────────────────────────────
    print(f"\n── GPR（{N_FOLDS} 折 GroupKFold，可能较慢）")
    _, y_pred_gpr, _ = run_cv(X, y, groups, make_gpr, label="GPR")
    res_gpr = eval_all_zones(y_true, y_pred_gpr)
    print(f"  全区  MAE={res_gpr['all']['mae']:.4f}  RMSE={res_gpr['all']['rmse']:.4f}  R2={res_gpr['all']['r2']:.4f}")
    print(f"  峰值区 MAE={res_gpr['peak']['mae']:.4f}  R2={res_gpr['peak']['r2']:.4f}")
    print(f"  远场区 MAE={res_gpr['far']['mae']:.4f}  R2={res_gpr['far']['r2']:.4f}")

    # ── 5. 留一波 GPR（展示不确定性，此处为能力上限测试）───────────
    print(f"\n── 留一波 GPR（3 折，每次藏一整条波）")
    _, y_pred_loo, y_std_loo = run_leave_one_wave(X, y, groups, wave_indices)
    res_loo = eval_all_zones(y_true, y_pred_loo)
    print(f"  全区  MAE={res_loo['all']['mae']:.4f}  R2={res_loo['all']['r2']:.4f}  [留一波外推]")

    # ── 6. 读取基线指标（Stage 1 产出）────────────────────────────
    baseline_results = {}
    if os.path.exists(BASE_JSON):
        with open(BASE_JSON, encoding="utf-8") as f:
            raw = json.load(f)
        baseline_results = {
            "B1": raw.get("B1_global_mean", {}),
            "B2": raw.get("B2_oracle_per_geom", {}),
            "B3": raw.get("B3_grid_interp", {}),
        }
    else:
        print("[警告] baseline_metrics.json 不存在，基线对比将缺失")

    # 读取 v2 深度模型 MAE
    v2_maes = {}
    if os.path.exists(DIAG_JSON):
        with open(DIAG_JSON, encoding="utf-8") as f:
            diag = json.load(f)
        v2_maes = diag.get("D2_baseline", {}).get("v2_taf_h_mae", {})

    # ── 7. 保存指标 JSON ─────────────────────────────────────────
    all_results = {
        "XGB": res_xgb,
        "GPR": res_gpr,
        "GPR_leave_one_wave": res_loo,
    }
    all_results.update(baseline_results)  # 合并基线指标
    out_json = os.path.join(OUTPUT_DIR, "train_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存：{out_json}")

    # ── 8. 绘图 ───────────────────────────────────────────────────
    if feat_imp is not None:  # 特征重要性（XGB 特有）
        plot_feature_importance(
            feat_imp, FEAT_NAMES,
            os.path.join(OUTPUT_DIR, "fig7_feature_importance.png"),
        )

    plot_parity(y_true, y_pred_xgb, y_pred_gpr,
                os.path.join(OUTPUT_DIR, "fig8_parity.png"))

    plot_curve_compare(y_true, y_pred_xgb, y_pred_gpr, meta,
                       os.path.join(OUTPUT_DIR, "fig9_curve_compare.png"))

    plot_gpr_uncertainty(y_true, y_pred_loo, y_std_loo, wave_indices, meta,
                         os.path.join(OUTPUT_DIR, "fig10_gpr_uncertainty.png"))

    if baseline_results:  # v2 vs v3 对照图
        plot_v2_v3_comparison(
            {**baseline_results, "XGB": res_xgb, "GPR": res_gpr},
            v2_maes,
            os.path.join(OUTPUT_DIR, "fig11_v2_v3_compare.png"),
        )

    # ── 9. 汇总对照表 ─────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("汇总对照表（TAF_h，GroupKFold OOF）")
    print("=" * 68)
    print(f"{'方法':<28} {'全区MAE':>9} {'峰值MAE':>9} {'远场MAE':>9} {'全区R2':>8}")
    print("-" * 68)

    for name, res in [
        ("v3 XGBoost (CV)",    res_xgb),
        ("v3 GPR (CV)",        res_gpr),
        ("v3 GPR 留一波",      res_loo),
    ]:
        print(f"{name:<28} {res['all']['mae']:>9.4f} {res['peak']['mae']:>9.4f} "
              f"{res['far']['mae']:>9.4f} {res['all']['r2']:>8.4f}")

    if baseline_results:
        print("-" * 68)
        for name, key in [("B1 全局均值", "B1"), ("B2 oracle下限", "B2"), ("B3 网格插值", "B3")]:
            res = baseline_results[key]
            if res:
                print(f"{name:<28} {res['all']['mae']:>9.4f} {res['peak']['mae']:>9.4f} "
                      f"{res['far']['mae']:>9.4f} {res['all']['r2']:>8.4f}")

    print("-" * 68)
    print("v2 深度模型（全区 MAE）：" + "  ".join(f"{k}={v:.4f}" for k, v in v2_maes.items()))

    # 成败判定
    b2_peak = baseline_results.get("B2", {}).get("peak", {}).get("mae", 0.0480)
    xgb_peak = res_xgb["peak"]["mae"]
    gpr_peak  = res_gpr["peak"]["mae"]
    print("\n" + "=" * 68)
    if xgb_peak < b2_peak or gpr_peak < b2_peak:
        print(f"[达标] 峰值区 MAE 达到目标（< {b2_peak:.4f}）：XGB={xgb_peak:.4f}  GPR={gpr_peak:.4f}")
    else:
        print(f"[未达标] 峰值区 MAE 未达目标 {b2_peak:.4f}：XGB={xgb_peak:.4f}  GPR={gpr_peak:.4f}")
    print("[完成] Stage 2 训练评估完成")


if __name__ == "__main__":
    main()
