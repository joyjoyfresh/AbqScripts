# TAF 机器学习预测模型 — 训练结果报告

## 数据概况
- **数据源**: `TAF-El_Centro-all.csv` (El Centro 地震波)
- **总数据量**: 16,100 行 (100 个工况 × 161 个观测点)
- **划分方式**: 按工况组合整体划分 (策略二)
- **训练集**: 12,880 行 (80 个工况) | **验证集**: 3,220 行 (20 个工况)

> [!NOTE]
> 当前仅使用了 4 种入射角 (0°, 10°, 20°, 30°) 的数据，共 100 个工况（非 175 个），因为汇总 CSV 中仅包含这些数据。

## 模型对比排名

| 排名 | 模型 | Val R² | Val MAE | Val RMSE | Val MaxErr | 训练耗时 |
|:---:|------|:------:|:-------:|:--------:|:----------:|:-------:|
| 🥇 | **XGBoost** | **0.8824** | **0.0363** | **0.0563** | 0.3192 | 0.9s |
| 🥈 | GradientBoosting | 0.8780 | 0.0391 | 0.0574 | 0.2960 | 4.6s |
| 🥉 | LightGBM | 0.8732 | 0.0390 | 0.0585 | **0.2903** | 0.6s |
| 4 | RandomForest | 0.8484 | 0.0399 | 0.0640 | 0.3702 | 0.8s |
| 5 | MLP | 0.6907 | 0.0639 | 0.0913 | 0.5417 | 5.0s |
| 6 | KNN | 0.3781 | 0.0746 | 0.1295 | 0.8822 | 0.0s |
| 7 | SVR | 0.2047 | 0.0886 | 0.1465 | 0.9134 | 12.8s |
| 8 | LinearRegression | 0.2045 | 0.1057 | 0.1465 | 1.0144 | 0.0s |
| 9 | Ridge | 0.2045 | 0.1057 | 0.1465 | 1.0144 | 0.0s |

## 可视化结果

### 1. 多模型性能对比
![模型对比柱状图](C:\Users\12462\.gemini\antigravity\brain\0f6cb485-df69-418c-a626-f94166216f3e\artifacts\model_comparison_bar.png)

### 2. 最优模型 (XGBoost) — 预测 vs 真实散点图
![Parity图](C:\Users\12462\.gemini\antigravity\brain\0f6cb485-df69-418c-a626-f94166216f3e\artifacts\best_model_parity.png)

### 3. 验证集 TAF 曲线对比 (FEM vs ML)
![曲线对比](C:\Users\12462\.gemini\antigravity\brain\0f6cb485-df69-418c-a626-f94166216f3e\artifacts\best_model_curves.png)

### 4. 特征重要性
![特征重要性](C:\Users\12462\.gemini\antigravity\brain\0f6cb485-df69-418c-a626-f94166216f3e\artifacts\best_model_feature_importance.png)

### 5. 残差分析
![残差分析](C:\Users\12462\.gemini\antigravity\brain\0f6cb485-df69-418c-a626-f94166216f3e\artifacts\best_model_residuals.png)

## 关键发现

1. **树模型完胜**: XGBoost、GradientBoosting、LightGBM、RandomForest 四个树模型 R² 均 > 0.84，远超其他模型
2. **XGBoost 最优**: R²=0.882, MAE=0.036, 训练仅 0.9 秒
3. **线性模型失败**: R² 仅 0.2，说明 TAF 与参数之间是高度非线性的关系
4. **最重要特征**: `h_norm` (归一化高度) > `tan_i` (坡面切线) > `i` (坡角)

> [!WARNING]
> 当前 R²=0.88 的精度可能还不足以完全替代 FEM 模拟。从曲线对比图可以看到，对于高坡角 + 大入射角的复杂工况（如 h=400, i=60°），预测的波动形态捕捉尚有不足。

## 可能的改进方向

1. **增加数据**: 纳入 Loma_Prieta 和 Northridge 地震波的数据，数据量 ×3
2. **补充缺失工况**: 当前缺少 angle=5°, 15°, 25° 的数据
3. **超参数调优**: 使用 Optuna 或 GridSearchCV 精细调参
4. **模型集成**: 将 Top-3 模型 (XGB + GB + LGBM) 进行 Stacking 集成
5. **增加物理特征**: 如坡面与波前的相对方位角等

## 文件产出

| 文件 | 路径 |
|------|------|
| 训练脚本 | [TAF_ml_train_v1.py](file:///c:/Users/12462/Documents/Code/AbqScripts/ML/TAF_ml_train_v1.py) |
| 模型对比表 | `E:\Abaqus\fuke-ALL\ML_results\model_comparison.csv` |
| 最优模型 | `E:\Abaqus\fuke-ALL\ML_results\best_model.pkl` |
| 模型元数据 | `E:\Abaqus\fuke-ALL\ML_results\best_model_meta.json` |
| 所有图表 | `E:\Abaqus\fuke-ALL\ML_results\*.png` |
