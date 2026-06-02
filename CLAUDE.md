# CLAUDE.md

本文件为在本仓库中使用 Claude Code（claude.ai/code）提供指导。

## 项目概述

研究项目，研究斜坡地形对地震地动分布的影响。使用 Abaqus 有限元软件构建二维平面应变斜坡模型，配备斜入射 SV 波和粘弹性人工边界（VAB）。后处理提取 PGA（峰值地面加速度）、TAF（地形放大因子），并生成出版级绘图。

## 项目架构

### 建模脚本（`Modeling/`）

所有建模脚本共享相同结构——一个 `main()` 函数：
1. 通过 `find_acc_txt()` 从当前工作目录读取加速度 `.txt` 文件
2. 定义 `material_cfg`、`geometry_cfg`、`job_cfg` 字典
3. 调用 `create_model()` 构建几何、材料、截面、装配、步骤
4. 调用 `build_models()` 应用斜入射波的 VAB 边界条件
5. 以指定的 CPU/内存设置提交作业

**按后缀的变体**：
- `_noGUI_` = 无头执行，无 Abaqus GUI
- `_TAF_` = 包含 TAF 计算（平地参考 + 斜坡对比）
- `_double_` = 双层土壤（基岩 + 上层不同 Vs）
- `_three_` = 三层配置
- `Single/` vs `Multi/` = 单模型 vs 批量多模型生成
- 版本号（`v1` → `v13`）表示迭代；始终使用最高版本

**关键参数**：`slope_h`（高度，米）、`i`（斜坡倾角，°）、`angle`（SV 波入射角，°）、`mesh_size`、各层弹性模量/泊松比/密度。

### 波处理（`Wave/`）

- `Seismic/scale_and_plot_v3.py` — 主要工具：读取原始地震 `.txt`（2 列：时间、加速度），应用 15Hz 低通 Butterworth 滤波器，缩放至目标 PGA，输出 `_scaled.txt` + 绘图（时间历程、傅立叶谱、反应谱）
- `Seismic/turn_to_VEL_v2.py` — 将加速度转换为速度时间序列
- `Impulse/ricker_wavelet_v3.py` — 生成 Ricker 小波信号
- `Impulse/impulse_wave_v2.py` — 脉冲波生成

### 后处理（`Postprocess/`）

- `Postprocess_PGA_v6.py`（Abaqus Python）— 使用 `TOP_SURFACE` 节点集从 ODB 提取 PGA，输出 CSV（列：x/h、节点标号、x、y、PGA_h、PGA_v、峰值时间）
- `Distribution_PGA_v5.py`（原生 Python）— 读取 `PGA*.csv` 文件，绘制 PGA vs x/h 归一化距离，从 `.cae` 文件名自动检测入射角
- `Mesh_Convergence_v5.py`（原生 Python）— 扫描 `fuke-{mesh}-*` 文件夹，提取固定 x/h 位置的 PGA，计算相对于最精细网格的误差
- `Distribution_TAF_*.py` — TAF 分布绘图变体
- `Distribution_Multi_TAF_*.py` — 多批次 TAF 聚合
- `Distribution_PGA_3D_v2.py` — 三维地表 PGA 绘图

### 机器学习（`ML/`）

- `TAF_ml_train_v1.py` — 训练脚本，比较 9 个回归模型（线性回归、Ridge、KNN、SVR、随机森林、梯度提升、XGBoost、LightGBM、MLP）以从几何参数预测 TAF
- `ml_plan_v2.md` — 升级到序列到序列深度学习代理模型的详细计划（CNN/LSTM/Transformer/DeepONet 编码器 → 4 通道 162 点输出：PGA_h、PGA_v、TAF_h、TAF_v）
- 数据位于 `E:\Abaqus\fuke-ALL\`（525 个样本：175 个几何 × 3 个地震波）

### 批处理（`Batch/`）

- `Autorun_custom_v2.py` — 通用子目录遍历运行器；将脚本复制到每个子文件夹并顺序执行
- `Autorun_meshsize_v2.py` — 网格收敛性研究自动化
- `Autorun_TAF_*.py` — TAF 相关批处理
- `.bat` 文件 — 交互式选择器（`Python_selectrun.bat`）和自动运行器（`Python_autorun.bat`）

## 编码约定

**每行必须有中文注释**说明其作用。这由 `.github/copilot-instructions.md` 强制执行。注释简洁，尽可能内联，对于结构行（空行、花括号）放在前一行。

