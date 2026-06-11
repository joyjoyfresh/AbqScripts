# -*- coding: utf-8 -*-
"""
TAF 地形放大系数 机器学习预测模型 v1
=====================================
多模型对比训练与评估脚本

数据源: E:\\Abaqus\\fuke-ALL\\1-TAF_grouped\\TAF-El_Centro-all.csv
划分策略: 按工况 (h, i, angle) 整体划分 80/20
模型: LinearRegression, Ridge, KNN, SVR, RandomForest,
      GradientBoosting, XGBoost, LightGBM, MLP
"""

import os
import sys
import io
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 修复 Windows 终端 GBK 编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from matplotlib import rcParams

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
ROOT_DIR = r"E:\Abaqus\fuke-ALL"
# 输出到脚本同目录下的 outputs 文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% 验证集
EARTHQUAKE_NAMES = ['El_Centro', 'Loma_Prieta', 'Northridge']

# 中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 地震波编码映射
EQ_CODE = {name: idx for idx, name in enumerate(EARTHQUAKE_NAMES)}


# ============================================================
# 1. 从子文件夹读取所有 TAF CSV 并拼合
# ============================================================
def load_and_prepare_data(root_dir):
    """扫描所有子文件夹, 读取 TAF-*.csv, 解析参数, 拼合为统一 DataFrame"""
    import re

    print("=" * 60)
    print("1. 从子文件夹加载所有 TAF 数据")
    print("=" * 60)

    all_dfs = []
    folder_pattern = re.compile(r'fuke-ALL-h(\d+)_i(\d+)_angle(\d+)')
    skipped = 0

    for folder_name in sorted(os.listdir(root_dir)):
        m = folder_pattern.match(folder_name)
        if not m:
            continue

        h_val = float(m.group(1))
        i_val = float(m.group(2))
        angle_val = float(m.group(3))
        folder_path = os.path.join(root_dir, folder_name)

        for eq_name in EARTHQUAKE_NAMES:
            csv_file = os.path.join(folder_path, f'TAF-{eq_name}.csv')
            if not os.path.isfile(csv_file):
                skipped += 1
                continue

            try:
                tmp = pd.read_csv(csv_file)
                if 'x/h' not in tmp.columns or 'TAF_h' not in tmp.columns:
                    skipped += 1
                    continue
                tmp = tmp[['x/h', 'TAF_h']].dropna()
                tmp['h'] = h_val
                tmp['i'] = i_val
                tmp['angle'] = angle_val
                tmp['earthquake'] = eq_name
                tmp['eq_code'] = EQ_CODE[eq_name]
                all_dfs.append(tmp)
            except Exception:
                skipped += 1

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"   读取文件数: {len(all_dfs)}, 跳过: {skipped}")
    print(f"   总数据量: {len(df)} 行")

    # 工况统计
    cases = df.groupby(['h', 'i', 'angle']).size().reset_index(name='count')
    eq_cases = df.groupby(['h', 'i', 'angle', 'earthquake']).size().reset_index(name='count')
    print(f"   几何工况数: {len(cases)}")
    print(f"   (含地震波) 工况数: {len(eq_cases)}")
    print(f"   h 取值: {sorted(df['h'].unique())}")
    print(f"   i 取值: {sorted(df['i'].unique())}")
    print(f"   angle 取值: {sorted(df['angle'].unique())}")
    print(f"   earthquake: {sorted(df['earthquake'].unique())}")
    print(f"   TAF_h 范围: [{df['TAF_h'].min():.4f}, {df['TAF_h'].max():.4f}]")

    # 特征工程
    print("\n   特征工程...")
    df['log_h'] = np.log10(df['h'])
    df['i_rad'] = np.radians(df['i'])
    df['angle_rad'] = np.radians(df['angle'])
    df['tan_i'] = np.tan(df['i_rad'])
    df['h_norm'] = df['h'] / df['h'].max()
    df['i_norm'] = df['i'] / df['i'].max()
    df['angle_norm'] = df['angle'] / max(df['angle'].max(), 1)
    df['h_tan_i'] = df['h'] * df['tan_i']
    df['angle_xh'] = df['angle'] * df['x/h']

    feature_cols = ['h', 'i', 'angle', 'x/h', 'eq_code',
                    'log_h', 'i_rad', 'angle_rad', 'tan_i',
                    'h_norm', 'i_norm', 'angle_norm',
                    'h_tan_i', 'angle_xh']
    target_col = 'TAF_h'

    print(f"   特征数量: {len(feature_cols)}")
    print(f"   特征列表: {feature_cols}")

    return df, feature_cols, target_col


# ============================================================
# 2. 按工况划分数据集
# ============================================================
def split_by_case(df, feature_cols, target_col):
    """按工况 (h, i, angle) 整体划分训练集和验证集"""
    print("\n" + "=" * 60)
    print("2. 按工况划分数据集 (策略二)")
    print("=" * 60)

    # 创建工况分组标签
    df['case_id'] = df['h'].astype(str) + '_' + df['i'].astype(str) + '_' + df['angle'].astype(str)
    groups = df['case_id']

    X = df[feature_cols].values
    y = df[target_col].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, val_idx = next(splitter.split(X, y, groups))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_cases = df.iloc[train_idx]['case_id'].nunique()
    val_cases = df.iloc[val_idx]['case_id'].nunique()

    print(f"   训练集: {len(X_train)} 行, {train_cases} 个工况")
    print(f"   验证集: {len(X_val)} 行, {val_cases} 个工况")

    # 记录验证集的工况列表 (用于后续可视化)
    val_case_info = df.iloc[val_idx][['h', 'i', 'angle', 'x/h', 'case_id']].copy()

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return (X_train, X_val, X_train_scaled, X_val_scaled,
            y_train, y_val, scaler, val_case_info, train_idx, val_idx)


# ============================================================
# 3. 定义模型
# ============================================================
def get_models():
    """返回所有待比较的模型字典"""
    models = {
        '1-LinearRegression': LinearRegression(),
        '2-Ridge': Ridge(alpha=1.0),
        '3-KNN': KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1),
        '4-RandomForest': RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1),
        '5-GradientBoosting': GradientBoostingRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE),
        '6-XGBoost': XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        '7-LightGBM': LGBMRegressor(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        '8-MLP': MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            solver='adam', max_iter=500, learning_rate='adaptive',
            early_stopping=True, random_state=RANDOM_STATE),
    }
    # 标记哪些模型需要标准化输入 (SVR removed - too slow for large datasets)
    needs_scaling = {'8-MLP', '3-KNN'}
    return models, needs_scaling


# ============================================================
# 4. 训练与评估
# ============================================================
def evaluate(y_true, y_pred):
    """计算全部评估指标"""
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MaxErr': np.max(np.abs(y_true - y_pred)),
        'MAPE(%)': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def train_all_models(X_train, X_val, X_train_s, X_val_s, y_train, y_val):
    """训练所有模型并返回结果"""
    print("\n" + "=" * 60)
    print("3. 训练与评估所有模型")
    print("=" * 60)

    models, needs_scaling = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n   >>> 训练: {name} ...", end=" ")
        t0 = time.time()

        # 选择是否使用标准化数据
        if name in needs_scaling:
            Xt, Xv = X_train_s, X_val_s
        else:
            Xt, Xv = X_train, X_val

        model.fit(Xt, y_train)
        train_time = time.time() - t0

        y_pred_train = model.predict(Xt)
        y_pred_val = model.predict(Xv)

        train_metrics = evaluate(y_train, y_pred_train)
        val_metrics = evaluate(y_val, y_pred_val)

        results[name] = {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'y_pred_val': y_pred_val,
            'y_pred_train': y_pred_train,
            'train_time': train_time,
        }

        print(f"完成 ({train_time:.1f}s)")
        print(f"         Train R2={train_metrics['R2']:.6f}  MAE={train_metrics['MAE']:.6f}")
        print(f"         Val   R2={val_metrics['R2']:.6f}  MAE={val_metrics['MAE']:.6f}  "
              f"RMSE={val_metrics['RMSE']:.6f}  MaxErr={val_metrics['MaxErr']:.6f}")

    return results


# ============================================================
# 5. 结果汇总与排名
# ============================================================
def summarize_results(results):
    """生成对比汇总表"""
    print("\n" + "=" * 60)
    print("4. 模型对比排名 (按验证集 R2 降序)")
    print("=" * 60)

    rows = []
    for name, r in results.items():
        row = {'Model': name, 'Train_Time(s)': round(r['train_time'], 2)}
        for k, v in r['train_metrics'].items():
            row[f'Train_{k}'] = v
        for k, v in r['val_metrics'].items():
            row[f'Val_{k}'] = v
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values('Val_R2', ascending=False).reset_index(drop=True)
    summary_df.index += 1  # 排名从1开始

    # 打印
    print(summary_df[['Model', 'Train_Time(s)',
                       'Train_R2', 'Val_R2', 'Val_MAE', 'Val_RMSE',
                       'Val_MaxErr', 'Val_MAPE(%)']].to_string())

    # 保存
    csv_path = os.path.join(OUTPUT_DIR, 'model_comparison.csv')
    summary_df.to_csv(csv_path, index_label='Rank')
    print(f"\n   汇总表已保存: {csv_path}")

    best_name = summary_df.iloc[0]['Model']
    print(f"\n   [BEST] Best model: {best_name} (Val R2={summary_df.iloc[0]['Val_R2']:.6f})")

    return summary_df, best_name


# ============================================================
# 6. 可视化
# ============================================================
def plot_comparison_bar(summary_df):
    """绘制模型指标对比柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多模型性能对比', fontsize=18, fontweight='bold')

    models = [m.split('-', 1)[1] for m in summary_df['Model']]
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

    # R²
    ax = axes[0, 0]
    bars = ax.bar(models, summary_df['Val_R2'], color=colors, edgecolor='gray')
    ax.set_ylabel('R²', fontsize=13)
    ax.set_title('验证集 R² (越高越好)', fontsize=14)
    ax.tick_params(axis='x', rotation=35)
    ax.set_ylim(min(0, summary_df['Val_R2'].min() - 0.05), 1.02)
    for bar, val in zip(bars, summary_df['Val_R2']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # MAE
    ax = axes[0, 1]
    bars = ax.bar(models, summary_df['Val_MAE'], color=colors, edgecolor='gray')
    ax.set_ylabel('MAE', fontsize=13)
    ax.set_title('验证集 MAE (越低越好)', fontsize=14)
    ax.tick_params(axis='x', rotation=35)
    for bar, val in zip(bars, summary_df['Val_MAE']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # RMSE
    ax = axes[1, 0]
    bars = ax.bar(models, summary_df['Val_RMSE'], color=colors, edgecolor='gray')
    ax.set_ylabel('RMSE', fontsize=13)
    ax.set_title('验证集 RMSE (越低越好)', fontsize=14)
    ax.tick_params(axis='x', rotation=35)
    for bar, val in zip(bars, summary_df['Val_RMSE']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # 训练时间
    ax = axes[1, 1]
    bars = ax.bar(models, summary_df['Train_Time(s)'], color=colors, edgecolor='gray')
    ax.set_ylabel('秒', fontsize=13)
    ax.set_title('训练时间 (秒)', fontsize=14)
    ax.tick_params(axis='x', rotation=35)
    for bar, val in zip(bars, summary_df['Train_Time(s)']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'model_comparison_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   对比柱状图已保存: {path}")


def plot_parity(results, best_name, y_val):
    """绘制最优模型的预测 vs 真实散点图"""
    y_pred = results[best_name]['y_pred_val']
    metrics = results[best_name]['val_metrics']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_val, y_pred, alpha=0.3, s=8, c='steelblue')
    lims = [min(y_val.min(), y_pred.min()) - 0.05,
            max(y_val.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, 'r--', lw=2, label='理想线 y=x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('FEM 真实值 TAF_h', fontsize=14)
    ax.set_ylabel('ML 预测值 TAF_h', fontsize=14)
    ax.set_title(f'{best_name.split("-",1)[1]} — 预测 vs 真实', fontsize=16)
    ax.legend(fontsize=12)

    text = f"R² = {metrics['R2']:.6f}\nMAE = {metrics['MAE']:.6f}\nRMSE = {metrics['RMSE']:.6f}"
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'best_model_parity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Parity 图已保存: {path}")


def plot_residuals(results, best_name, y_val):
    """绘制残差分布图"""
    y_pred = results[best_name]['y_pred_val']
    residuals = y_val - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{best_name.split("-",1)[1]} — 残差分析', fontsize=16, fontweight='bold')

    # 残差直方图
    ax = axes[0]
    ax.hist(residuals, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('残差 (真实 - 预测)', fontsize=13)
    ax.set_ylabel('频数', fontsize=13)
    ax.set_title('残差分布直方图', fontsize=14)
    ax.text(0.95, 0.95, f'Mean={np.mean(residuals):.6f}\nStd={np.std(residuals):.6f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 残差 vs 真实值
    ax = axes[1]
    ax.scatter(y_val, residuals, alpha=0.3, s=6, c='steelblue')
    ax.axhline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('FEM 真实值 TAF_h', fontsize=13)
    ax.set_ylabel('残差', fontsize=13)
    ax.set_title('残差 vs 真实值', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'best_model_residuals.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   残差图已保存: {path}")


def plot_curve_comparison(df, results, best_name, val_case_info, needs_scaling_set, scaler, feature_cols):
    """挑选若干验证集工况，绘制 TAF 曲线对比"""
    model = results[best_name]['model']
    use_scaling = best_name in needs_scaling_set

    # 挑选最多 6 个验证工况
    val_cases = val_case_info[['h', 'i', 'angle', 'case_id']].drop_duplicates()
    sample_cases = val_cases.sample(n=min(6, len(val_cases)), random_state=RANDOM_STATE)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{best_name.split("-",1)[1]} — 验证集 TAF 曲线对比 (FEM vs ML)',
                 fontsize=16, fontweight='bold')

    for idx, (_, case_row) in enumerate(sample_cases.iterrows()):
        ax = axes.flatten()[idx]
        h_val, i_val, a_val = case_row['h'], case_row['i'], case_row['angle']

        # 从原始数据取该工况
        mask = (df['h'] == h_val) & (df['i'] == i_val) & (df['angle'] == a_val)
        case_df = df[mask].sort_values('x/h')

        X_case = case_df[feature_cols].values
        if use_scaling:
            X_case = scaler.transform(X_case)
        y_pred_case = model.predict(X_case)
        y_true_case = case_df['TAF_h'].values
        xh = case_df['x/h'].values

        r2 = r2_score(y_true_case, y_pred_case)

        ax.plot(xh, y_true_case, 'b-', lw=2, label='FEM', alpha=0.8)
        ax.plot(xh, y_pred_case, 'r--', lw=2, label='ML Pred', alpha=0.8)
        ax.set_title(f'h={h_val:.0f}, i={i_val:.0f}°, angle={a_val:.0f}° (R²={r2:.4f})', fontsize=11)
        ax.set_xlabel('x/h')
        ax.set_ylabel('TAF_h')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(sample_cases), 6):
        axes.flatten()[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'best_model_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   TAF 曲线对比图已保存: {path}")


def plot_feature_importance(results, best_name, feature_cols):
    """绘制特征重要性 (仅对树模型有效)"""
    model = results[best_name]['model']
    if not hasattr(model, 'feature_importances_'):
        print("   (该模型不支持 feature_importances_, 跳过)")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(indices)), importances[indices], color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=12)
    ax.set_xlabel('重要性', fontsize=13)
    ax.set_title(f'{best_name.split("-",1)[1]} — 特征重要性', fontsize=16)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'best_model_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   特征重要性图已保存: {path}")


# ============================================================
# 7. 保存最优模型
# ============================================================
def save_best_model(results, best_name, scaler, feature_cols):
    """保存最优模型和相关信息"""
    print("\n" + "=" * 60)
    print("6. 保存最优模型")
    print("=" * 60)

    model = results[best_name]['model']
    save_data = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': best_name,
        'val_metrics': results[best_name]['val_metrics'],
    }

    path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"   模型已保存: {path}")

    # 保存元数据
    meta = {
        'model_name': best_name,
        'feature_cols': feature_cols,
        'val_metrics': {k: round(v, 6) for k, v in results[best_name]['val_metrics'].items()},
        'train_metrics': {k: round(v, 6) for k, v in results[best_name]['train_metrics'].items()},
    }
    meta_path = os.path.join(OUTPUT_DIR, 'best_model_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"   元数据已保存: {meta_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("  TAF 地形放大系数 — 多模型机器学习对比")
    print("=" * 60)

    # 1. 加载数据
    df, feature_cols, target_col = load_and_prepare_data(ROOT_DIR)

    # 2. 划分数据集
    (X_train, X_val, X_train_s, X_val_s,
     y_train, y_val, scaler, val_case_info,
     train_idx, val_idx) = split_by_case(df, feature_cols, target_col)

    # 3. 训练所有模型
    results = train_all_models(X_train, X_val, X_train_s, X_val_s, y_train, y_val)

    # 4. 汇总排名
    summary_df, best_name = summarize_results(results)

    # 5. 可视化
    print("\n" + "=" * 60)
    print("5. 生成可视化图表")
    print("=" * 60)

    _, needs_scaling = get_models()
    plot_comparison_bar(summary_df)
    plot_parity(results, best_name, y_val)
    plot_residuals(results, best_name, y_val)
    plot_curve_comparison(df, results, best_name, val_case_info,
                          needs_scaling, scaler, feature_cols)
    plot_feature_importance(results, best_name, feature_cols)

    # 6. 保存模型
    save_best_model(results, best_name, scaler, feature_cols)

    print("\n" + "=" * 60)
    print("  全部完成! 结果保存在: " + OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()
