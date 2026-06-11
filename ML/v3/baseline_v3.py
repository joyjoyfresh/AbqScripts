# -*- coding: utf-8 -*-
"""ML v3 — Stage 1 基线评估。

三条基线：
  B1  全局均值        — 所有训练曲线的平均值，R2 的零点定义
  B2  per-geom oracle — 每地形取 3 波平均（理论下限 ≈ 0.0469，oracle，可见测试标签）
  B3  规则网格插值    — LinearNDInterpolator 在 (log_h, i, angle) 空间三维线性插值，
                        5 折 GroupKFold（按工况分组，无数据泄漏）

产出：
  outputs_v3/baseline_metrics.json   — 各基线 MAE / RMSE / R2（全区 / 峰值区 / 远场区）
  outputs_v3/fig6_baselines.png      — 基线 vs v2 深度模型对比柱状图

运行：py -3 ML/baseline_v3.py
"""

import os  # 路径与目录操作
import re  # 正则表达式（解析文件夹名）
import json  # JSON 读写
import numpy as np  # 数值计算
import pandas as pd  # CSV 读取
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator  # 多维插值
from sklearn.model_selection import GroupKFold  # 按组分折，防数据泄漏
import matplotlib  # 绘图框架
matplotlib.use("Agg")  # 无显示器后端（批处理保存）
import matplotlib.pyplot as plt  # 绘图子模块

# ==========================================================================
# 配置区
# ==========================================================================
DATA_DIR   = r"E:\Abaqus\fuke-ALL"  # 原始数据根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 脚本所在目录（ML/）
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs_v3")  # 输出目录（不覆盖 v2）
DIAG_JSON  = os.path.join(OUTPUT_DIR, "diagnostics_v3.json")  # Stage 0 诊断结果（含 v2 MAE）
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 输出目录不存在则创建

WAVES   = ["El_Centro", "Loma_Prieta", "Northridge"]  # 三条地震波
N_PTS   = 161  # 曲线统一采样点数（x/h ∈ [0,8]，步长 0.05）
GRID    = np.linspace(0.0, 8.0, N_PTS)  # 统一 x/h 网格
PEAK    = GRID <= 2.0  # 峰值区掩码（坡面附近，物理关键区）
FAR     = GRID >= 5.0  # 远场区掩码（平坦远场）
PAT     = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 文件夹名解析正则
N_FOLDS = 5  # GroupKFold 折数

plt.rcParams.update({  # 全局绘图参数
    "font.sans-serif":   ["SimHei", "Microsoft YaHei", "DejaVu Sans"],  # 中文字体优先
    "axes.unicode_minus": False,  # 修复负号显示为方块的 bug
    "figure.dpi":         150,    # 输出分辨率
})

# ==========================================================================
# 辅助函数
# ==========================================================================

def read_csv(path):  # 稳健 CSV 读取
    """用 utf-8-sig 读取，剥掉 TAF 文件的 BOM（否则列名带 \\ufeff）。"""
    return pd.read_csv(path, encoding="utf-8-sig")  # 自动去 BOM


def load_dataset():
    """遍历 DATA_DIR，加载所有三波完整的工况，插值到统一 161 点网格。

    Returns
    -------
    geoms : list of (h, i, angle) tuples，共 N_geom 个
    tafh  : ndarray (N_geom, 3, 161)，TAF_h 曲线
    """
    folders = sorted([f for f in os.listdir(DATA_DIR) if PAT.match(f)])  # 过滤并排序工况文件夹
    geoms, curves, missing = [], [], 0  # 初始化结果列表
    for f in folders:  # 逐文件夹处理
        m = PAT.match(f)  # 解析 h / i / angle
        h, i, angle = float(m[1]), float(m[2]), float(m[3])  # 转为浮点数
        arr = []  # 当前工况的 3 波曲线
        for w in WAVES:  # 逐波读取
            p = os.path.join(DATA_DIR, f, f"TAF-{w}.csv")  # TAF 文件路径
            if not os.path.exists(p):  # 文件缺失则跳出
                break
            d = read_csv(p).sort_values("x/h")  # 读取并按 x/h 排序
            arr.append(np.interp(GRID, d["x/h"].values, d["TAF_h"].values))  # 插值到统一网格
        if len(arr) == 3:  # 三波齐全才收录
            geoms.append((h, i, angle))  # 存几何参数
            curves.append(arr)  # 存曲线（3×161）
        else:
            missing += 1  # 计数不完整工况
    print(f"加载完成：{len(geoms)} 个几何工况，跳过 {missing} 个（数据不完整）")
    return geoms, np.array(curves)  # curves: (N_geom, 3, 161)


def compute_metrics(y_true, y_pred, mask=None):
    """计算 MAE / RMSE / R2。

    Parameters
    ----------
    y_true, y_pred : ndarray (N, 161)
    mask           : bool ndarray (161,)，选取子区间（None=全区）
    """
    if mask is not None:  # 若指定区间，截取对应列
        y_true = y_true[:, mask]
        y_pred = y_pred[:, mask]
    err    = y_true - y_pred  # 逐点误差
    mae    = float(np.abs(err).mean())  # 平均绝对误差
    rmse   = float(np.sqrt((err ** 2).mean()))  # 均方根误差
    ss_res = float((err ** 2).sum())  # 残差平方和
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())  # 总平方和
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0  # 决定系数(R2)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def eval_all_zones(y_true, y_pred):
    """全区 / 峰值区 / 远场区 三套指标汇总。"""
    return {
        "all":  compute_metrics(y_true, y_pred),         # 全 x/h 区间
        "peak": compute_metrics(y_true, y_pred, PEAK),   # x/h ≤ 2
        "far":  compute_metrics(y_true, y_pred, FAR),    # x/h ≥ 5
    }


# ==========================================================================
# B2 per-geometry oracle 均值（理论下限，不做 CV）
# ==========================================================================

def baseline_B2_oracle(geoms, tafh):
    """每地形用 3 波平均当预测值（oracle，可见全部标签）。

    这是「完全无视波」时的最低可达误差（≈ 0.0469），
    剩下的误差全是 3 波之间无法消除的随机散布。
    不做交叉验证，直接用全量数据计算，明确标注为 oracle。

    Returns
    -------
    dict  eval_all_zones 结构
    """
    geom_mean = tafh.mean(axis=1)  # (N_geom, 161)，每地形 3 波取均值
    y_true = tafh.reshape(-1, N_PTS)  # (N_geom*3, 161)，展开所有样本
    y_pred = np.repeat(geom_mean, 3, axis=0)  # (N_geom*3, 161)，重复 3 次对齐
    return eval_all_zones(y_true, y_pred)


# ==========================================================================
# GroupKFold 通用包装器
# ==========================================================================

def run_groupkfold(geoms, tafh, predict_fn, label=""):
    """5 折 GroupKFold（按几何工况分组），调用 predict_fn 获取预测。

    Parameters
    ----------
    predict_fn : callable(tr_geoms, tr_tafh, te_geoms) -> ndarray (N_te_geom, 161)
                 返回测试几何工况的预测曲线（每地形一条，3 波共用同一预测）

    Returns
    -------
    y_true, y_pred : ndarray (N_total_samples, 161)，供 eval_all_zones 使用
    """
    N       = len(geoms)  # 几何工况总数
    groups  = np.arange(N)  # 每个几何工况作为独立 group
    X_dummy = np.zeros(N)  # GroupKFold 需要占位 X，值无意义
    kf      = GroupKFold(n_splits=N_FOLDS)  # 5 折按组划分

    all_true, all_pred = [], []  # 收集各折预测结果
    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_dummy, groups=groups)):
        tr_geoms  = [geoms[i] for i in tr_idx]  # 训练几何列表
        tr_tafh   = tafh[tr_idx]  # 训练曲线 (N_tr, 3, 161)
        te_geoms  = [geoms[i] for i in te_idx]  # 测试几何列表
        te_tafh   = tafh[te_idx]  # 测试曲线 (N_te, 3, 161)

        pred_per_geom = predict_fn(tr_geoms, tr_tafh, te_geoms)  # (N_te_geom, 161)

        for j in range(len(te_geoms)):  # 逐几何工况展开 3 条波
            for w in range(len(WAVES)):  # 每条波的真实曲线
                all_true.append(te_tafh[j, w])    # 真实 TAF_h 曲线
                all_pred.append(pred_per_geom[j])  # 预测曲线（3 波共用同一几何预测）

        if label:  # 若指定标签，输出折进度
            N_te = len(te_geoms)
            print(f"  [{label}] fold {fold_idx+1}/{N_FOLDS}：测试 {N_te} 个几何工况")

    return np.array(all_true), np.array(all_pred)  # 各 (N_geom*3, 161)


# ==========================================================================
# B1 全局均值
# ==========================================================================

def predict_B1_global(tr_geoms, tr_tafh, te_geoms):
    """预测 = 训练集全部曲线（所有波）的逐点均值。"""
    global_mean = tr_tafh.reshape(-1, N_PTS).mean(axis=0)  # (161,)，训练集均值曲线
    return np.tile(global_mean, (len(te_geoms), 1))  # (N_te_geom, 161)，重复铺开


# ==========================================================================
# B3 规则网格线性插值
# ==========================================================================

def predict_B3_gridinterp(tr_geoms, tr_tafh, te_geoms):
    """在 (log_h, i, angle) 三维空间做线性插值。

    插值目标 = 训练几何工况的 3 波均值曲线（每地形一条）。
    使用 log(h) 而非 h，使等比分布的 h 值（10/50/100/200/400）在插值空间均匀分布。
    边界外推点（LinearNDInterpolator 返回 NaN）用最近邻回填。
    """
    tr_geom_mean = tr_tafh.mean(axis=1)  # (N_tr, 161)，训练集每地形 3 波均值

    # 构造插值点坐标：log(h) / i / angle
    tr_pts = np.array([[np.log(g[0]), g[1], g[2]] for g in tr_geoms])  # (N_tr, 3)
    te_pts = np.array([[np.log(g[0]), g[1], g[2]] for g in te_geoms])  # (N_te, 3)

    # 线性插值（scipy 支持向量值输出，values 可为 2D）
    interp_lin = LinearNDInterpolator(tr_pts, tr_geom_mean, fill_value=np.nan)
    pred = interp_lin(te_pts)  # (N_te, 161)，边界外推点 = NaN

    # 处理 NaN（测试点落在训练凸包外）：用最近邻回填
    nan_rows = np.where(np.isnan(pred).any(axis=1))[0]  # 含 NaN 的行索引
    if len(nan_rows) > 0:  # 有外推点需要回填
        interp_nn = NearestNDInterpolator(tr_pts, tr_geom_mean)  # 最近邻插值器
        pred[nan_rows] = interp_nn(te_pts[nan_rows])  # 用最近邻填充
        print(f"    [B3] {len(nan_rows)} 个外推工况用最近邻回填")

    return pred  # (N_te, 161)


# ==========================================================================
# 绘图：基线 vs v2 深度模型对比
# ==========================================================================

def plot_baselines(results, v2_maes):
    """绘制 TAF_h MAE 对比图（基线 + v2 深度模型）。

    Parameters
    ----------
    results  : dict，含 B1 / B2 / B3 各区 MAE
    v2_maes  : dict，{"cnn": mae, "lstm": mae, ...}（全区 MAE）
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # 左：综合对比，右：分区对比
    fig.suptitle("Stage 1 基线量化（TAF_h）", fontsize=14, fontweight="bold")

    # ── 左图：所有方法的全区 MAE ─────────────────────────────────
    methods_left = [
        ("B1\n全局均值",    results["B1"]["all"]["mae"],  "#aaaaaa"),  # 灰色：最差基线
        ("B2\noracle均值",  results["B2"]["all"]["mae"],  "#2ca02c"),  # 绿色：理论下限
        ("B3\n网格插值",    results["B3"]["all"]["mae"],  "#1f77b4"),  # 蓝色：真实无 ML 下限
        ("v2 CNN",          v2_maes.get("cnn", np.nan),   "#ff7f0e"),  # 橙色：v2 深度模型
        ("v2 LSTM",         v2_maes.get("lstm", np.nan),  "#9467bd"),
        ("v2 Trans.",       v2_maes.get("transformer", np.nan), "#8c564b"),
        ("v2 DeepONet",     v2_maes.get("deeponet", np.nan),    "#d62728"),  # 红色：v2 最强
    ]
    x_l  = np.arange(len(methods_left))  # x 坐标
    bars = ax1.bar(
        x_l,
        [m[1] for m in methods_left],
        color=[m[2] for m in methods_left],
        edgecolor="white", linewidth=0.6,
    )
    ax1.axhline(  # 虚线标注理论下限（B2 oracle MAE）
        results["B2"]["all"]["mae"],
        color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"B2 oracle 下限 = {results['B2']['all']['mae']:.4f}",
    )
    ax1.set_xticks(x_l)
    ax1.set_xticklabels([m[0] for m in methods_left], fontsize=9)
    ax1.set_ylabel("TAF_h MAE（全区）", fontsize=10)
    ax1.set_title("所有方法对比", fontsize=11)
    ax1.legend(fontsize=8)
    for bar, (name, val, _) in zip(bars, methods_left):  # 柱顶标注数值
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.0005,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=8,
            )

    # ── 右图：三条基线的分区 MAE（全区 / 峰值区 / 远场区）──────────
    baselines_right = [
        ("B1 全局均值",  results["B1"], "#aaaaaa"),
        ("B2 oracle",    results["B2"], "#2ca02c"),
        ("B3 网格插值",  results["B3"], "#1f77b4"),
    ]
    n_bl    = len(baselines_right)  # 基线数量
    zones   = ["all", "peak", "far"]  # 三个区间键名
    zone_lb = ["全区", "峰值区\n(x/h≤2)", "远场区\n(x/h≥5)"]  # 区间显示标签
    width   = 0.22  # 每组内单根柱宽度
    x_r     = np.arange(len(zones))  # x 坐标（按区间分组）
    offset  = np.linspace(-(n_bl - 1) / 2 * width, (n_bl - 1) / 2 * width, n_bl)  # 组内偏移

    for bl_i, (bl_name, bl_res, bl_color) in enumerate(baselines_right):  # 逐基线绘制
        vals = [bl_res[z]["mae"] for z in zones]  # 三区 MAE
        bars2 = ax2.bar(
            x_r + offset[bl_i],
            vals,
            width=width,
            color=bl_color,
            edgecolor="white", linewidth=0.5,
            label=bl_name,
        )
        for bar, val in zip(bars2, vals):  # 柱顶数值
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.0003,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=7, rotation=0,
            )

    ax2.axhline(  # 虚线：B2 oracle 全区 MAE
        results["B2"]["all"]["mae"],
        color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7,
        label=f"B2 oracle = {results['B2']['all']['mae']:.4f}",
    )
    ax2.set_xticks(x_r)
    ax2.set_xticklabels(zone_lb, fontsize=9)
    ax2.set_ylabel("TAF_h MAE", fontsize=10)
    ax2.set_title("基线分区对比", fontsize=11)
    ax2.legend(fontsize=8)

    plt.tight_layout()  # 自动调整子图间距
    out_path = os.path.join(OUTPUT_DIR, "fig6_baselines.png")  # 输出路径
    plt.savefig(out_path, bbox_inches="tight", dpi=150)  # 保存为 PNG
    plt.close()  # 释放内存
    print(f"图表已保存：{out_path}")


# ==========================================================================
# 主流程
# ==========================================================================

def main():
    print("=" * 64)
    print("ML v3 — Stage 1 基线评估")
    print("=" * 64)

    # ── 1. 加载数据 ───────────────────────────────────────────────
    geoms, tafh = load_dataset()  # tafh: (N_geom, 3, 161)
    N_geom = len(geoms)  # 几何工况数

    # ── 2. B2 oracle（不做 CV，直接全量计算，明确标注为理论下限）──
    print("\n── B2 per-geometry oracle 均值（理论下限，oracle 不可用于 CV）")
    B2 = baseline_B2_oracle(geoms, tafh)
    print(f"  全区  MAE={B2['all']['mae']:.4f}  RMSE={B2['all']['rmse']:.4f}  R2={B2['all']['r2']:.4f}")
    print(f"  峰值区 MAE={B2['peak']['mae']:.4f}")
    print(f"  远场区 MAE={B2['far']['mae']:.4f}")

    # ── 3. B1 全局均值（GroupKFold）───────────────────────────────
    print(f"\n── B1 全局均值（{N_FOLDS} 折 GroupKFold，按几何工况分组）")
    t1, p1 = run_groupkfold(geoms, tafh, predict_B1_global, label="B1")
    B1 = eval_all_zones(t1, p1)
    print(f"  全区  MAE={B1['all']['mae']:.4f}  RMSE={B1['all']['rmse']:.4f}  R2={B1['all']['r2']:.4f}")
    print(f"  峰值区 MAE={B1['peak']['mae']:.4f}")
    print(f"  远场区 MAE={B1['far']['mae']:.4f}")

    # ── 4. B3 规则网格插值（GroupKFold）───────────────────────────
    print(f"\n── B3 规则网格线性插值（{N_FOLDS} 折 GroupKFold，(log_h, i, angle) 空间）")
    t3, p3 = run_groupkfold(geoms, tafh, predict_B3_gridinterp, label="B3")
    B3 = eval_all_zones(t3, p3)
    print(f"  全区  MAE={B3['all']['mae']:.4f}  RMSE={B3['all']['rmse']:.4f}  R2={B3['all']['r2']:.4f}")
    print(f"  峰值区 MAE={B3['peak']['mae']:.4f}")
    print(f"  远场区 MAE={B3['far']['mae']:.4f}")

    # ── 5. 保存 JSON 结果 ─────────────────────────────────────────
    results = {"B1_global_mean": B1, "B2_oracle_per_geom": B2, "B3_grid_interp": B3}
    out_json = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)  # 保存指标
    print(f"\n指标已保存：{out_json}")

    # ── 6. 读取 v2 深度模型成绩（来自 Stage 0 诊断 JSON）─────────
    v2_maes = {}  # 初始化为空
    if os.path.exists(DIAG_JSON):  # 优先读 Stage 0 已整理的结果
        with open(DIAG_JSON, encoding="utf-8") as f:
            diag = json.load(f)
        v2_maes = diag.get("D2_baseline", {}).get("v2_taf_h_mae", {})
    if not v2_maes:  # 降级：用 ml_plan_v3.md 中的已知数值
        v2_maes = {
            "cnn": 0.0527, "lstm": 0.0544,
            "transformer": 0.0572, "deeponet": 0.0506,
        }
        print("[警告] 未找到 diagnostics_v3.json，使用方案文档中的已知 v2 指标")

    # ── 7. 绘制对比图 ─────────────────────────────────────────────
    plot_baselines({"B1": B1, "B2": B2, "B3": B3}, v2_maes)

    # ── 8. 打印汇总对照表 ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("汇总对照表（TAF_h）")
    print("=" * 72)
    hdr = f"{'方法':<26} {'全区MAE':>9} {'峰值MAE':>9} {'远场MAE':>9} {'全区RMSE':>10} {'全区R2':>8}"
    print(hdr)
    print("-" * 72)

    rows = [  # 各方法行数据
        ("B1  全局均值 (CV)",      B1["all"]["mae"], B1["peak"]["mae"], B1["far"]["mae"], B1["all"]["rmse"], B1["all"]["r2"]),
        ("B2  oracle per-geom",    B2["all"]["mae"], B2["peak"]["mae"], B2["far"]["mae"], B2["all"]["rmse"], B2["all"]["r2"]),
        ("B3  网格插值 (CV)",       B3["all"]["mae"], B3["peak"]["mae"], B3["far"]["mae"], B3["all"]["rmse"], B3["all"]["r2"]),
    ]
    for row in rows:  # 逐行打印
        print(f"{row[0]:<26} {row[1]:>9.4f} {row[2]:>9.4f} {row[3]:>9.4f} {row[4]:>10.4f} {row[5]:>8.4f}")

    print("-" * 72)
    print("v2 深度模型（来自 Stage 0 诊断）")
    for model, mae in v2_maes.items():  # 打印 v2 各模型全区 MAE
        print(f"  v2 {model:<18} {mae:>9.4f}")

    print("=" * 72)
    print(f"\n[目标] v3 需达到：峰值区 MAE < {B2['peak']['mae']:.4f}（B2 oracle）且 <= 0.0506（v2 DeepONet）")
    print("[完成] Stage 1 基线评估完成")


if __name__ == "__main__":
    main()
