# -*- coding: utf-8 -*-
"""ML v3 — Stage 3 完整评估（外推测试 + 多通道 + TAF_v 专项）。

本脚本回答三个问题（主模型固定为 Stage 2 胜出的 POD + GPR）：
  A. 多通道泛化：TAF_h / PGA_h / PGA_v / TAF_v 四条曲线分别能预测多准？
     （5 折 GroupKFold；TAF_v 用第 7.4 节稳健掩码，单独汇报）
  B. 留一高度外推：藏起某个 h 值的全部工况，预测它——测「高度外推」能力。
  C. 留一角度外推：藏起某个入射角 angle 的全部工况——测「角度外推」能力。

为什么这一步重要：Stage 2 的 GroupKFold 只保证「没见过这个具体几何」，但训练集里
仍有相邻几何（如 h=50 与 h=100）。留一高度/角度则彻底藏掉一整个参数层级，是更诚实、
更接近真实使用（外插到训练网格之外）的考验。

产出（均写入 outputs_v3/）：
  evaluate_metrics.json          — A/B/C 全部指标
  fig12_extrapolation_heatmap.png — 留一高度 + 留一角度 R2/MAE 热力图
  fig13_multichannel.png         — 四通道 Parity 散点
  fig14_tafv_special.png         — TAF_v 稳健掩码前后对比

运行：py -3 ML/evaluate_v3.py
"""

import os  # 路径操作
import re  # 正则解析文件夹名
import json  # JSON 读写
import warnings  # 抑制 GPR 收敛警告刷屏
import numpy as np  # 数值计算
import pandas as pd  # CSV 读取
from sklearn.decomposition import PCA  # POD（主成分分析）
from sklearn.preprocessing import StandardScaler  # 特征标准化（GPR 必需）
from sklearn.multioutput import MultiOutputRegressor  # 多输出回归包装
from sklearn.gaussian_process import GaussianProcessRegressor  # GPR 主模型
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # GPR 核
import matplotlib  # 绘图框架
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt  # 绘图子模块

warnings.filterwarnings("ignore")  # 关闭 GPR 超参边界收敛警告（不影响结果，只是提示）

# ==========================================================================
# 配置区（与 train_v3.py 保持一致）
# ==========================================================================
DATA_DIR   = r"E:\Abaqus\fuke-ALL"  # 原始数据根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_v3")  # v3 输出目录
DIAG_JSON  = os.path.join(OUTPUT_DIR, "diagnostics_v3.json")  # Stage 0 波特征来源
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录

WAVES   = ["El_Centro", "Loma_Prieta", "Northridge"]  # 三条地震波
N_PTS   = 161  # 曲线统一点数
GRID    = np.linspace(0.0, 8.0, N_PTS)  # x/h ∈ [0,8]
PEAK    = GRID <= 2.0  # 峰值区掩码
FAR     = GRID >= 5.0  # 远场区掩码
PAT     = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 文件夹名解析

POD_N           = 15      # POD 主成分数（与 Stage 2 一致）
TAFV_FLAT_THRES = 1e-3    # 平地竖向 PGA_v 近零阈值：低于此值的点 TAF_v 不可靠（第 7.4 节）
TAFV_CLIP_MAX   = 10.0    # TAF_v 建模前的截断上限（沿用 v2 dataset 的 clip(0,10)）

plt.rcParams.update({  # 全局绘图参数
    "font.sans-serif":    ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi":         150,
})

# ==========================================================================
# 波物理特征（从 diagnostics_v3.json 读取，剔除常数 PGA_in）
# ==========================================================================
def load_wave_features():
    """读取 Stage 0 预计算的 11 维波物理特征字典。"""
    with open(DIAG_JSON, encoding="utf-8") as f:  # 打开诊断结果
        raw = json.load(f).get("D5_waves", {})  # 取波特征段
    feats = {}  # 结果字典
    for w in WAVES:  # 逐波整理
        d = raw[w]  # 当前波原始特征
        feats[w] = dict(
            Arias=d["Arias"], D5_95=d["D5_95"], Tp=d["Tp"], Tm=d["Tm"],
            PGV=d["PGV"],     PGD=d["PGD"],
            Sa_01=d["Sa_0.1"], Sa_02=d["Sa_0.2"], Sa_05=d["Sa_0.5"],
            Sa_10=d["Sa_1.0"], Sa_20=d["Sa_2.0"],
        )
    return feats


WAVE_FEATS      = load_wave_features()  # 波特征字典
WAVE_FEAT_NAMES = list(list(WAVE_FEATS.values())[0].keys())  # 11 个波特征名
GEO_FEAT_NAMES  = ["h", "i", "angle", "log_h", "tan_i"]  # 5 个几何特征名
FEAT_NAMES      = GEO_FEAT_NAMES + WAVE_FEAT_NAMES  # 总 16 维


# ==========================================================================
# 辅助函数
# ==========================================================================
def read_csv(path):  # 稳健 CSV 读取
    """utf-8-sig 读取，自动剥 BOM。"""
    return pd.read_csv(path, encoding="utf-8-sig")


def compute_metrics(y_true, y_pred, mask=None, point_mask=None):
    """MAE / RMSE / R2。

    Parameters
    ----------
    mask       : bool (161,)，列方向区间掩码（峰值/远场）
    point_mask : bool (N,161)，逐点可靠性掩码（TAF_v 专用，剔除不可靠点）
    """
    if mask is not None:  # 区间裁剪
        y_true, y_pred = y_true[:, mask], y_pred[:, mask]
        if point_mask is not None:
            point_mask = point_mask[:, mask]
    if point_mask is not None:  # 仅保留可靠点（展平为一维）
        sel    = point_mask.astype(bool)
        y_true = y_true[sel]
        y_pred = y_pred[sel]
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
    err    = y_true - y_pred  # 逐点误差
    mae    = float(np.abs(err).mean())  # 平均绝对误差
    rmse   = float(np.sqrt((err ** 2).mean()))  # 均方根误差
    ss_res = float((err ** 2).sum())  # 残差平方和
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())  # 总平方和
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0  # 决定系数(R2)
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def eval_zones(y_true, y_pred, point_mask=None):
    """全区 / 峰值区 / 远场区 指标汇总（支持逐点可靠性掩码）。"""
    return {
        "all":  compute_metrics(y_true, y_pred, None, point_mask),
        "peak": compute_metrics(y_true, y_pred, PEAK, point_mask),
        "far":  compute_metrics(y_true, y_pred, FAR,  point_mask),
    }


def make_gpr():
    """POD + GPR 多输出回归器（Stage 2 胜出配置）。"""
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))               # 幅度核
              * RBF(1.0, (1e-2, 1e2))                         # RBF 核
              + WhiteKernel(1e-2, (1e-6, 1e1)))               # 噪声核（下界放宽到 1e-6）
    base = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                    n_restarts_optimizer=2, random_state=42)
    return MultiOutputRegressor(base, n_jobs=-1)  # 并行拟合 15 个 GPR


# ==========================================================================
# 数据加载（多通道）
# ==========================================================================
def load_dataset():
    """加载所有四波完整工况的 4 个通道曲线。

    Returns
    -------
    geom  : ndarray (N_geom, 3)，[h, i, angle]
    chans : dict，每个值 (N_geom, 3, 161)
            keys: TAF_h, PGA_h, PGA_v_slope, PGA_v_flat
    """
    folders = sorted(f for f in os.listdir(DATA_DIR) if PAT.match(f))  # 工况文件夹
    geom, TAFh, PGAh, PGAvs, PGAvf = [], [], [], [], []  # 各通道结果
    skipped = 0  # 跳过计数
    for folder in folders:  # 逐工况
        m = PAT.match(folder)  # 解析几何
        h, i_ang, angle = float(m.group(1)), float(m.group(2)), float(m.group(3))
        fp = os.path.join(DATA_DIR, folder)  # 工况目录
        taf_w, pgah_w, pgavs_w, pgavf_w = [], [], [], []  # 三波缓存
        ok = True  # 文件齐全标志
        for w in WAVES:  # 逐波
            f_taf   = os.path.join(fp, f"TAF-{w}.csv")  # TAF 文件
            f_slope = os.path.join(fp, f"PGA-{w}_scaled-slope.csv")  # 坡面 PGA
            f_flat  = os.path.join(fp, f"PGA-{w}_scaled-flat.csv")  # 平地 PGA
            if not (os.path.exists(f_taf) and os.path.exists(f_slope) and os.path.exists(f_flat)):
                ok = False  # 缺文件
                break
            d_taf = read_csv(f_taf).sort_values("x/h")  # 读 TAF
            taf_w.append(np.interp(GRID, d_taf["x/h"].values, d_taf["TAF_h"].values))  # 插值 TAF_h
            d_s = read_csv(f_slope).sort_values("x/h")  # 读坡面 PGA
            pgah_w.append(np.interp(GRID, d_s["x/h"].values, d_s["PGA_h"].values))   # 插值 PGA_h
            pgavs_w.append(np.interp(GRID, d_s["x/h"].values, d_s["PGA_v"].values))  # 插值坡面 PGA_v
            d_f = read_csv(f_flat).sort_values("x/h")  # 读平地 PGA
            pgavf_w.append(np.interp(GRID, d_f["x/h"].values, d_f["PGA_v"].values))  # 插值平地 PGA_v
        if not ok or len(taf_w) != 3:  # 文件不齐
            skipped += 1
            continue
        geom.append([h, i_ang, angle])  # 记录几何
        TAFh.append(taf_w); PGAh.append(pgah_w)  # 记录水平通道
        PGAvs.append(pgavs_w); PGAvf.append(pgavf_w)  # 记录竖向通道
    print(f"数据加载：{len(geom)} 个几何工况（跳过 {skipped} 个不完整）")
    chans = {
        "TAF_h":       np.array(TAFh),   # (N_geom, 3, 161)
        "PGA_h":       np.array(PGAh),
        "PGA_v_slope": np.array(PGAvs),
        "PGA_v_flat":  np.array(PGAvf),
    }
    return np.array(geom), chans


def build_features(geom):
    """根据几何数组构建展平后的特征矩阵 X 与索引信息。

    每行 = 一个 (几何工况, 地震波) 样本，共 N_geom × 3 行。

    Returns
    -------
    X          : ndarray (N, 16)
    geom_idx   : ndarray (N,)，所属几何工况索引（GroupKFold 用）
    h_arr      : ndarray (N,)，每行的 h 值（留一高度用）
    angle_arr  : ndarray (N,)，每行的 angle 值（留一角度用）
    wave_idx   : ndarray (N,)，每行的波索引 0/1/2
    """
    rows, gidx, harr, aarr, widx = [], [], [], [], []  # 各列表
    for gi, (h, i_ang, angle) in enumerate(geom):  # 逐几何
        log_h = np.log(h)  # 对数高度
        tan_i = np.tan(np.radians(i_ang))  # 坡角正切
        geo_vec = [h, i_ang, angle, log_h, tan_i]  # 5 维几何特征
        for wi, w in enumerate(WAVES):  # 逐波
            wf = WAVE_FEATS[w]  # 波特征
            rows.append(geo_vec + [wf[k] for k in WAVE_FEAT_NAMES])  # 16 维特征
            gidx.append(gi); harr.append(h); aarr.append(angle); widx.append(wi)  # 索引信息
    return (np.array(rows, float), np.array(gidx, int),
            np.array(harr, float), np.array(aarr, float), np.array(widx, int))


def channel_to_samples(chan_curves):
    """将通道曲线 (N_geom, 3, 161) 展平为样本矩阵 (N_geom*3, 161)，顺序与 build_features 一致。"""
    return chan_curves.reshape(-1, N_PTS)


# ==========================================================================
# 通用 POD + GPR 预测（给定 train/test 索引）
# ==========================================================================
def fit_predict(X, y, tr_idx, te_idx, n_comp=POD_N):
    """折内拟合 POD + GPR，返回测试集预测曲线（无数据泄漏）。"""
    X_tr, X_te = X[tr_idx], X[te_idx]  # 划分特征
    y_tr       = y[tr_idx]  # 训练目标曲线
    pca = PCA(n_components=n_comp)  # POD
    coeff_tr = pca.fit_transform(y_tr)  # 训练集 POD 系数（仅训练集拟合）
    scaler = StandardScaler()  # 特征标准化
    X_tr_sc = scaler.fit_transform(X_tr)  # 训练集统计量
    X_te_sc = scaler.transform(X_te)  # 测试集变换
    model = make_gpr()  # GPR
    model.fit(X_tr_sc, coeff_tr)  # 拟合
    coeff_pred = model.predict(X_te_sc)  # 预测 POD 系数
    return pca.inverse_transform(coeff_pred)  # 逆 POD 还原曲线


def run_grouped_cv(X, y, group_values, label=""):
    """按 group_values 的唯一值做留一组交叉验证（每个唯一值轮流当测试集）。

    用于留一高度（group_values=h_arr）和留一角度（group_values=angle_arr）。

    Returns
    -------
    y_pred_oof : ndarray (N, 161)，OOF 预测
    per_group  : dict，每个被留出值 -> eval_zones 指标
    """
    y_pred_oof = np.full_like(y, np.nan)  # OOF 预测容器
    per_group  = {}  # 逐组指标
    uniq = sorted(np.unique(group_values))  # 所有唯一取值（升序）
    for val in uniq:  # 逐个留出
        te_idx = np.where(group_values == val)[0]  # 测试集（当前值）
        tr_idx = np.where(group_values != val)[0]  # 训练集（其余）
        pred = fit_predict(X, y, tr_idx, te_idx)  # 折内拟合预测
        y_pred_oof[te_idx] = pred  # 写入 OOF
        per_group[float(val)] = eval_zones(y[te_idx], pred)  # 该留出值的指标
        m = per_group[float(val)]["all"]  # 全区指标
        print(f"  [{label}] 留出 {val:g}：测试 {len(te_idx)} 样本  MAE={m['mae']:.4f}  R2={m['r2']:.4f}")
    return y_pred_oof, per_group


def run_standard_groupkfold(X, y, geom_idx, n_splits=5, point_mask=None):
    """标准 5 折 GroupKFold（按几何工况分组），返回 OOF 预测与分区指标。"""
    from sklearn.model_selection import GroupKFold  # 局部导入
    kf = GroupKFold(n_splits=n_splits)  # 分折器
    y_pred_oof = np.full_like(y, np.nan)  # OOF 容器
    for tr_idx, te_idx in kf.split(X, groups=geom_idx):  # 逐折
        y_pred_oof[te_idx] = fit_predict(X, y, tr_idx, te_idx)  # 折内拟合预测
    return y_pred_oof, eval_zones(y, y_pred_oof, point_mask)


# ==========================================================================
# 绘图
# ==========================================================================
def plot_extrapolation_heatmap(loh_groups, loa_groups, outpath):
    """留一高度 + 留一角度的 R2 / MAE 热力图（行=区间，列=留出值）。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    fig.suptitle("Stage 3 外推能力热力图（TAF_h，POD+GPR）\n"
                 "[每列藏起该参数值的全部工况，预测它——比 GroupKFold 更严苛]",
                 fontsize=12, fontweight="bold")
    zones    = ["all", "peak", "far"]  # 三区间
    zone_lbl = ["全区", "峰值区", "远场区"]  # 区间标签

    def draw(ax, groups, metric, title, fmt, cmap):
        """绘制单个热力图子图。"""
        vals_x = sorted(groups.keys())  # 留出值（列）
        mat = np.array([[groups[v][z][metric] for v in vals_x] for z in zones])  # (3, n_val)
        im = ax.imshow(mat, aspect="auto", cmap=cmap)  # 热力图
        ax.set_xticks(range(len(vals_x)))
        ax.set_xticklabels([f"{v:g}" for v in vals_x], fontsize=9)
        ax.set_yticks(range(len(zones)))
        ax.set_yticklabels(zone_lbl, fontsize=9)
        ax.set_title(title, fontsize=10)
        for r in range(len(zones)):  # 单元格标注数值
            for c in range(len(vals_x)):
                ax.text(c, r, fmt.format(mat[r, c]), ha="center", va="center",
                        fontsize=8, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    draw(axes[0, 0], loh_groups, "r2",  "留一高度 R2（越高越好）", "{:.3f}", "RdYlGn")
    draw(axes[0, 1], loh_groups, "mae", "留一高度 MAE（越低越好）", "{:.4f}", "RdYlGn_r")
    draw(axes[1, 0], loa_groups, "r2",  "留一角度 R2（越高越好）", "{:.3f}", "RdYlGn")
    draw(axes[1, 1], loa_groups, "mae", "留一角度 MAE（越低越好）", "{:.4f}", "RdYlGn_r")
    axes[1, 0].set_xlabel("留出的入射角 angle (°)", fontsize=9)
    axes[1, 1].set_xlabel("留出的入射角 angle (°)", fontsize=9)
    axes[0, 0].set_xlabel("留出的坡高 h (m)", fontsize=9)
    axes[0, 1].set_xlabel("留出的坡高 h (m)", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


def plot_multichannel_parity(chan_results, outpath):
    """四通道 Parity 散点（预测 vs 真实）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    fig.suptitle("四通道 Parity Plot（5 折 GroupKFold OOF）", fontsize=13, fontweight="bold")
    rng = np.random.default_rng(0)  # 抽样随机源

    items = [
        ("TAF_h", "#d62728"), ("PGA_h", "#1f77b4"),
        ("PGA_v", "#2ca02c"), ("TAF_v", "#9467bd"),
    ]
    for ax, (ch, color) in zip(axes.ravel(), items):
        data = chan_results[ch]  # 含 y_true / y_pred / point_mask
        y_t = data["y_true"]; y_p = data["y_pred"]
        pm  = data.get("point_mask")  # TAF_v 才有逐点掩码
        if pm is not None:  # 仅取可靠点
            sel = pm.astype(bool)
            y_t = y_t[sel]; y_p = y_p[sel]
        else:
            y_t = y_t.ravel(); y_p = y_p.ravel()
        idx = rng.choice(len(y_t), size=min(4000, len(y_t)), replace=False)  # 抽样
        ax.scatter(y_t[idx], y_p[idx], s=2, alpha=0.3, color=color, rasterized=True)
        lim = [float(np.min(y_t)), float(np.max(y_t))]
        ax.plot(lim, lim, "k--", linewidth=1)  # 对角线
        met = data["metrics"]["all"]  # 全区指标
        suffix = "（仅可靠点）" if pm is not None else ""
        ax.set_title(f"{ch}{suffix}\nMAE={met['mae']:.4f}  R2={met['r2']:.4f}", fontsize=11)
        ax.set_xlabel("真实值", fontsize=9); ax.set_ylabel("预测值", fontsize=9)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


def plot_tafv_special(tafv_raw, tafv_clipped, reliable_frac_xh, outpath):
    """TAF_v 专项：原始爆炸 vs 截断 + 不可靠点沿 x/h 分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("TAF_v 稳健处理（斜入射 SV 波下平地竖向 PGA 近零导致除法爆炸）",
                 fontsize=12, fontweight="bold")

    # 左：原始 TAF_v 直方图（截显到 20，展示长尾爆炸）
    axes[0].hist(np.clip(tafv_raw.ravel(), 0, 20), bins=80, color="#d62728", alpha=0.75)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("原始 TAF_v（截显到 20）", fontsize=9)
    axes[0].set_ylabel("点数（对数）", fontsize=9)
    axes[0].set_title(f"原始：max={tafv_raw.max():.0f}（爆炸）", fontsize=10)

    # 中：截断 + 掩码后的 TAF_v 直方图（建模实际所用）
    axes[1].hist(tafv_clipped.ravel(), bins=80, color="#9467bd", alpha=0.75)
    axes[1].set_xlabel(f"建模用 TAF_v（截断到 {TAFV_CLIP_MAX:g} + 掩码近零点）", fontsize=9)
    axes[1].set_ylabel("点数", fontsize=9)
    axes[1].set_title("处理后：分布收敛，可建模", fontsize=10)

    # 右：可靠点占比沿 x/h 分布（哪些位置 TAF_v 可信）
    axes[2].plot(GRID, reliable_frac_xh * 100, "-", color="#2ca02c", linewidth=1.5)
    axes[2].fill_between(GRID, 0, reliable_frac_xh * 100, alpha=0.2, color="#2ca02c")
    axes[2].set_xlabel("x/h", fontsize=9)
    axes[2].set_ylabel("可靠点占比 (%)", fontsize=9)
    axes[2].set_ylim(0, 105)
    axes[2].set_title(f"沿坡面 TAF_v 可靠区段\n(平地|PGA_v|≥{TAFV_FLAT_THRES:g} 视为可靠)", fontsize=10)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"图表已保存：{outpath}")


# ==========================================================================
# 主流程
# ==========================================================================
def main():
    print("=" * 68)
    print("ML v3 — Stage 3 完整评估（外推 + 多通道 + TAF_v 专项）")
    print("=" * 68)

    # ── 1. 加载数据 + 构建特征 ────────────────────────────────────
    geom, chans = load_dataset()
    X, geom_idx, h_arr, angle_arr, wave_idx = build_features(geom)
    print(f"特征矩阵 X: {X.shape}  几何工况: {len(geom)}")
    print(f"坡高取值: {sorted(np.unique(h_arr).tolist())}")
    print(f"入射角取值: {sorted(np.unique(angle_arr).tolist())}")

    all_results = {}  # 汇总结果

    # ── 2. Part A：多通道标准 GroupKFold ──────────────────────────
    print("\n" + "─" * 60)
    print("Part A：多通道泛化（5 折 GroupKFold）")
    print("─" * 60)

    chan_results = {}  # 各通道绘图数据

    # 2.1 三个良态通道：TAF_h / PGA_h / PGA_v(坡面)
    for ch_name, ch_key in [("TAF_h", "TAF_h"), ("PGA_h", "PGA_h"), ("PGA_v", "PGA_v_slope")]:
        y = channel_to_samples(chans[ch_key])  # 展平为样本
        y_pred, metrics = run_standard_groupkfold(X, y, geom_idx)  # CV
        chan_results[ch_name] = {"y_true": y, "y_pred": y_pred, "metrics": metrics}
        print(f"  [{ch_name}] 全区 MAE={metrics['all']['mae']:.4f}  R2={metrics['all']['r2']:.4f}  "
              f"| 峰值 MAE={metrics['peak']['mae']:.4f}")
        all_results[ch_name] = metrics

    # 2.2 TAF_v 专项（第 7.4 节稳健掩码）
    print("\n  ── TAF_v 专项稳健处理")
    pgavs = chans["PGA_v_slope"]  # 坡面竖向 PGA
    pgavf = chans["PGA_v_flat"]   # 平地竖向 PGA
    safe  = np.clip(np.abs(pgavf), 1e-12, None)  # 防除零分母
    tafv_raw = pgavs / safe  # 原始 TAF_v（展示爆炸用）
    reliable = np.abs(pgavf) >= TAFV_FLAT_THRES  # 可靠点掩码（平地竖向 PGA 不近零）
    tafv_clipped = np.clip(tafv_raw, 0.0, TAFV_CLIP_MAX)  # 截断到 [0,10] 供建模
    frac_reliable = float(reliable.mean())  # 全局可靠点占比
    print(f"  可靠点占比：{frac_reliable*100:.2f}%（平地|PGA_v|≥{TAFV_FLAT_THRES:g}）")
    print(f"  不可靠点（近零）占比：{(1-frac_reliable)*100:.2f}%（已从训练/评估排除）")

    # 不可靠点用列中位数填补，避免污染 POD 基；评估时仅看可靠点
    y_tafv = channel_to_samples(tafv_clipped)  # (N,161)
    rel_pts = channel_to_samples(reliable)     # (N,161) 逐点可靠掩码
    col_median = np.median(y_tafv, axis=0)     # 各 x/h 列中位数
    y_tafv_imputed = y_tafv.copy()  # 填补副本
    bad = ~rel_pts.astype(bool)  # 不可靠点
    cols = np.where(bad)[1]  # 不可靠点对应列
    y_tafv_imputed[bad] = col_median[cols]  # 用列中位数填补
    y_pred_tafv, metrics_tafv = run_standard_groupkfold(
        X, y_tafv_imputed, geom_idx, point_mask=rel_pts)  # 评估仅看可靠点
    chan_results["TAF_v"] = {"y_true": y_tafv, "y_pred": y_pred_tafv,
                             "point_mask": rel_pts, "metrics": metrics_tafv}
    print(f"  [TAF_v 仅可靠点] 全区 MAE={metrics_tafv['all']['mae']:.4f}  R2={metrics_tafv['all']['r2']:.4f}  "
          f"| 峰值 MAE={metrics_tafv['peak']['mae']:.4f}")
    all_results["TAF_v_reliable_only"] = metrics_tafv
    all_results["TAF_v_meta"] = {
        "flat_threshold": TAFV_FLAT_THRES, "clip_max": TAFV_CLIP_MAX,
        "reliable_frac": frac_reliable, "tafv_raw_max": float(tafv_raw.max()),
    }

    # ── 3. Part B：留一高度外推（核心量 TAF_h）─────────────────────
    print("\n" + "─" * 60)
    print("Part B：留一高度外推（藏起某 h 的全部工况，预测它）")
    print("─" * 60)
    y_tafh = channel_to_samples(chans["TAF_h"])  # TAF_h 样本
    _, loh_groups = run_grouped_cv(X, y_tafh, h_arr, label="留一高度")
    all_results["leave_one_height"] = loh_groups

    # ── 4. Part C：留一角度外推 ───────────────────────────────────
    print("\n" + "─" * 60)
    print("Part C：留一角度外推（藏起某 angle 的全部工况，预测它）")
    print("─" * 60)
    _, loa_groups = run_grouped_cv(X, y_tafh, angle_arr, label="留一角度")
    all_results["leave_one_angle"] = loa_groups

    # ── 5. 保存指标 ───────────────────────────────────────────────
    out_json = os.path.join(OUTPUT_DIR, "evaluate_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n指标已保存：{out_json}")

    # ── 6. 绘图 ───────────────────────────────────────────────────
    plot_extrapolation_heatmap(loh_groups, loa_groups,
                               os.path.join(OUTPUT_DIR, "fig12_extrapolation_heatmap.png"))
    plot_multichannel_parity(chan_results,
                             os.path.join(OUTPUT_DIR, "fig13_multichannel.png"))
    reliable_frac_xh = reliable.reshape(-1, N_PTS).mean(axis=0)  # 沿 x/h 可靠点占比
    plot_tafv_special(tafv_raw, tafv_clipped, reliable_frac_xh,
                      os.path.join(OUTPUT_DIR, "fig14_tafv_special.png"))

    # ── 7. 汇总表 ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Stage 3 汇总")
    print("=" * 68)
    print("【多通道泛化（5 折 GroupKFold）】")
    print(f"{'通道':<20} {'全区MAE':>9} {'峰值MAE':>9} {'远场MAE':>9} {'全区R2':>8}")
    print("-" * 60)
    for ch in ["TAF_h", "PGA_h", "PGA_v", "TAF_v_reliable_only"]:
        r = all_results[ch]
        print(f"{ch:<20} {r['all']['mae']:>9.4f} {r['peak']['mae']:>9.4f} "
              f"{r['far']['mae']:>9.4f} {r['all']['r2']:>8.4f}")

    print("\n【留一高度外推（TAF_h 全区）】")
    for v in sorted(loh_groups.keys()):
        r = loh_groups[v]["all"]
        print(f"  h={v:>5g} m：MAE={r['mae']:.4f}  R2={r['r2']:.4f}")

    print("\n【留一角度外推（TAF_h 全区）】")
    for v in sorted(loa_groups.keys()):
        r = loa_groups[v]["all"]
        print(f"  angle={v:>4g}°：MAE={r['mae']:.4f}  R2={r['r2']:.4f}")

    # 外推稳健性概述
    loh_r2 = [loh_groups[v]["all"]["r2"] for v in loh_groups]
    loa_r2 = [loa_groups[v]["all"]["r2"] for v in loa_groups]
    print("\n" + "=" * 68)
    print(f"留一高度 R2 范围：[{min(loh_r2):.3f}, {max(loh_r2):.3f}]  均值 {np.mean(loh_r2):.3f}")
    print(f"留一角度 R2 范围：[{min(loa_r2):.3f}, {max(loa_r2):.3f}]  均值 {np.mean(loa_r2):.3f}")
    print("[完成] Stage 3 完整评估完成")


if __name__ == "__main__":
    main()
