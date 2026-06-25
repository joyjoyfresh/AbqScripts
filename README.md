# AbqScripts：斜坡场地地震响应数值模拟与机器学习代理模型

本项目是一组面向 Abaqus 的 Python / MATLAB 脚本，用于研究**二维斜坡场地在斜入射地震动下的地形放大效应**。覆盖从波形生成、有限元建模、ODB 后处理到机器学习代理建模的完整链路。

核心物理量：

- **PGA**（Peak Ground Acceleration，峰值加速度）—— 地表某点地震动最大幅值。
- **TAF**（Topographic Amplification Factor，地形放大系数）—— 斜坡上某点 PGA ÷ 同样地震下平地 PGA，>1 表示地形把地震放大了，是本研究的核心量。

建模支持：SV 波斜入射、粘弹性人工边界（VAB）、多层土、土体等效线性非线性（EQL）；自由场用频域 Thomson–Haskell 传播矩阵法精确求解。

---

## 📑 目录

- [项目结构](#项目结构)
- [运行环境](#运行环境)
- [工作流程](#工作流程)
- [详细步骤](#详细步骤)
- [机器学习代理模型](#机器学习代理模型-ml)
- [MATLAB 后处理](#matlab-后处理)
- [批处理工具](#批处理工具-batch)
- [常见问题](#常见问题)
- [输出档案要求](#输出档案要求)

---

## 项目结构

```
AbqScripts/
├─ Batch/                 # 批处理与无人值守自动运行
├─ Modeling/              # Abaqus 建模（静默 noGUI 调用）
│  ├─ Single/             #   单工况主流程：VAB_oblique / _TAF / _noGUI
│  ├─ Multi/              #   多工况批量：TAF_double / TAF_multilayer / multilayer_nonlinear
│  └─ Nodes/              #   节点集创建与坐标读取工具
├─ Postprocess/           # ODB 后处理与结果分析
│  ├─ General/            #   通用：PGA 提取、TAF 计算、结果汇总、绘图
│  ├─ Single/             #   单工况：PGA/TAF 分布、反应谱、网格收敛
│  └─ Multi/              #   多工况：TAF 对比、FAS 时频、箱线图、地震图
├─ Wave/                  # 波形生成与预处理
│  ├─ Seismic/            #   真实地震波：调幅/滤波、加速度转速度
│  │  ├─ Original/        #     原始时程（El_Centro、Kobe、ChiChi 等 11 条）
│  │  ├─ Scaled/          #     调幅后
│  │  └─ VELed/           #     转速度后
│  └─ Impulse/            #   脉冲 / Ricker 子波生成
│     └─ Acceleration/    #     各频段脉冲加速度时程
├─ ML/                    # 机器学习代理模型（预测 PGA / TAF 曲线）
│  ├─ v1/                 #   初版基线
│  ├─ v2/                 #   深度学习（CNN/LSTM/Transformer/DeepONet）
│  └─ v3/                 #   经典 ML（XGBoost/GPR + POD），当前主线
├─ MATLAB/                # MATLAB 三维 PGA 分布后处理
├─ requirements.txt       # 系统 Python 依赖（ML / 绘图用）
└─ README.md
```

> 各脚本按版本迭代（`_v1.py`、`_v2.py` …），**无特殊说明时使用目录内最高版本号**。版本号越大越新，但物理方法可能跨版本有重大变化（见 [建模说明](#第二步建模与有限元计算)）。

---

## 运行环境

本项目脚本分属两套 Python 环境，**切勿混用**：

| 用途 | 解释器 | 调用方式 |
|------|--------|---------|
| 建模、后处理（Abaqus 脚本） | Abaqus 内置 **Python 2.7** | `abaqus cae noGUI=脚本.py` |
| ML、独立绘图、波形处理 | 系统 **Python 3**（`py -3`） | `py -3 脚本.py` |

- 终端默认 `python` 在本机指向 Abaqus 的 Py2.7，**没有 f-string**，跑 ML 脚本会报错——必须用 `py -3`。
- ML 依赖见 `requirements.txt`（numpy / scipy / pandas / scikit-learn / xgboost / lightgbm / matplotlib / scienceplots）。PyTorch 仅复跑 v2 深度模型时需要，单独安装。

---

## 工作流程

```
波形生成/预处理 ──→ 建模与计算 ──→ ODB 后处理 ──→ ML 代理建模
   Wave/            Modeling/      Postprocess/      ML/
```

单工况走 Single 路线；参数扫描（坡高/坡角/入射角/网格尺寸）走 Multi + Batch 自动批量路线。

---

## 详细步骤

### 第一步：波形生成与预处理

#### 1.1 真实地震波（`Wave/Seismic/`）

**脚本**：`scale_and_plot_v2.py`（最新）

- 扫描 `Original/` 下原始 `.txt` 时程，低通滤波（默认 15 Hz）
- 按目标 PGA 调幅，输出到 `Scaled/`
- 绘制加速度时程、傅里叶振幅谱（FFT）、5% 阻尼反应谱

**脚本**：`turn_to_VEL_v2.py`——加速度积分得速度时程，输出到 `VELed/`。

#### 1.2 脉冲波（`Wave/Impulse/`）

- `ricker_wavelet_v2.py`——生成 Ricker 子波（2/4/6/8 Hz），结果在 `Acceleration/`
- `impulse_wave_v2.py`——生成脉冲波

---

### 第二步：建模与有限元计算

所有建模脚本通过 Abaqus 静默调用运行，无需启动 GUI：

```bash
abaqus cae noGUI=Modeling/Multi/VAB_oblique_TAF_multilayer_v8.py
```

**单工况（`Modeling/Single/`）**：`VAB_oblique_v3.py` / `VAB_oblique_TAF_v2.py` / `VAB_oblique_noGUI_v3.py`。

**多工况批量（`Modeling/Multi/`）**：

| 脚本谱系 | 最新版本 | 说明 |
|---------|---------|------|
| `VAB_oblique_TAF_multilayer` | v8 | 多层土斜入射，频域传播矩阵法自由场 |
| `VAB_oblique_multilayer_nonlinear` | v2 | 在 multilayer 基础上新增**土体等效线性 EQL**，含 1D SHAKE 迭代与逐单元 2D 模式 |
| `VAB_oblique_TAF_double` / `_double_TAF` / `_double` | v3 / v4 / v3 | 双层模型系列 |

**建模核心机制**（详见 `Modeling/Multi/CHANGELOG`）：

1. 频域精确自由场（Thomson–Haskell 全局矩阵法，含临界角自检）
2. 地形等厚表层几何（表层沿地形起伏铺设）
3. 建模前自动对拍一维解析解，误差 > 1e-3 强行中止
4. 逐层重锚定瑞利阻尼（按各层共振基频重拟合频带）
5. ODB 瘦身（静默尾段 + 仅输出地表节点全时程）
6. 网格自适应（K-L 判据 + 分层非均匀 + 软层谐波加密）
7. 土体非线性 EQL（可选，`enable=False` 默认线性）

**工况配置**：默认参数在脚本内 `material_cfg` 等字典定义；若工况目录下存在 `case_config.json` 会自动加载覆盖；建模结果元数据写入 `case_meta.json`（含远场 PGA 理论值 `ff_theory`，用于精度 QA）。

**节点工具（`Modeling/Nodes/`）**：`NodeSet_create_v3.py`（创建节点集）、`NodeSet_coordinates_v2.py`（读取坐标）。

---

### 第三步：ODB 后处理

#### 3.1 通用提取（`Postprocess/General/`）

- `Postprocess_PGA_v3.py`——从 ODB 提取地表 PGA（坡顶/坡面/坡脚），按 `x/h` 无量纲坐标输出 `PGA_*.csv`
- `Compute_TAF_v2.py`——由坡面 PGA ÷ 平地 PGA 计算 TAF
- `Collect_results_v2.py`——汇总多工况结果
- `Plot_PGA_v1.py` / `Plot_TAF_v1.py`——绘图

#### 3.2 单工况分析（`Postprocess/Single/`）

- `Distribution_PGA_v2.py` / `Distribution_PGA_3D_v2.py`——PGA 沿 `x/h` 分布
- `Distribution_TAF_v2.py`（及 `_theta` / `_grouped` / `_overview`）——TAF 分布与按入射角分组
- `Distribution_Sa_by_h_v2.py`——反应谱沿坡高分布
- `Mesh_Convergence_v2.py`——网格收敛性分析（指定参考网格，算误差极值）

#### 3.3 多工况分析（`Postprocess/Multi/`）

- `Distribution_Multi_TAF_v2.py` / `Plot_Multi_TAF_v3.py`——多工况 TAF 分布对比
- `Plot_Fig15_compare_v3.py`——论文图对比
- `Plot_FAS_spectrogram_v1.py`——FAS 时频谱图
- `Plot_PGA_box_v1.py` / `Plot_peakTAF_v1.py` / `Plot_seismograms_v1.py` / `Plot_dist_param_v1.py`——箱线图、峰值、地震图、参数分布

---

## 机器学习代理模型（ML）

用快速代理模型替代昂贵的 Abaqus 仿真，输入地形几何 + 波物理特征 → 预测 4 条曲线（PGA_h / PGA_v / TAF_h / TAF_v，统一 161 点，`x/h ∈ [0,8]`）。

| 版本 | 方法 | 结论 |
|------|------|------|
| v1 | 基线 | 初步探索 |
| v2 | 深度学习（CNN/LSTM/Transformer/DeepONet） | 跑完发现被「傻瓜基线」打败——522 样本太少，深度网络没从波里学到可迁移规律 |
| v3（**当前主线**） | 经典 ML（XGBoost / 高斯过程 GPR）+ POD 降维 + 波物理特征 | **实测优于 v2 全部深度模型**（TAF_h MAE 0.0244 vs 0.0506），且 CPU 秒级 vs GPU 小时级 |

v3 核心思路：地形是主导信号（影响是波的 2.47 倍），波只需压成 ~10 个物理特征（Arias 强度、持时、卓越周期、反应谱 Sa(T) 等）。

- `train_v3.py`——主文件（波特征 + POD + 模型 + 交叉验证）
- `baseline_v3.py` / `diagnose_v3.py` / `evaluate_v3.py` / `figures_v3.py`
- 完整方案见 `ML/v3/ml_plan_v3.md`（含初学者阅读说明、实测数据、成败标准）

---

## MATLAB 后处理

`MATLAB/`——三维 PGA 分布可视化与汇总：

- `Distribution_PGA_3D.m` / `Distribution_PGA_3D_from_summary_v1.m`
- `PGA_max_summary.csv` / `PGA_h_max_summary.csv`（汇总数据）

---

## 批处理工具（Batch）

| 文件 | 功能 |
|------|------|
| `Python_selectrun.bat` | 交互式选择执行脚本 |
| `Python_autorun.bat` | 按目录顺序自动迭代执行 |
| `Autorun_TAF_*.py` | TAF 工况批量（含 `double` / `multilayer` 系列，v1–v3） |
| `Autorun_meshsize_v2.py` | 网格尺寸扫描批量 |
| `Autorun_paper_Shen2025_v3.py` | Shen2025 论文工况批量 |
| `Autorun_softsoil_nonlinear_v1.py` | 软土非线性工况批量 |
| `delete_filetype_v1.py` | 批量删除指定类型文件 |

---

## 常见问题

### 脚本运行报 `SyntaxError`（f-string 等）

终端 `python` 指向 Abaqus 的 Py2.7。ML / 绘图 / 波形脚本须用 `py -3 脚本.py`；建模后处理脚本用 `abaqus cae noGUI=脚本.py`（走 Abaqus 内置 2.7，但脚本须兼容 2.7 语法，不能用 f-string）。

### 找不到文件 / 路径错误

- 建模脚本依赖 Abaqus 内核上下文，**内核内无 `__file__`**，`open()` **无 `encoding` 参数**——路径须用绝对路径或基于固定工作目录推导。
- 确认 `.txt` 波源、ODB 文件与脚本预期路径一致。

### 工况参数不生效（配置塌缩）

若 `case_config.json` 的 per-case 参数未生效、全塌缩到默认值，检查配置注入逻辑（角度 `angle` / 入射角 `i` 等键名与脚本读取口径是否一致）。

---

## 输出档案要求

数据用于报告或文献引用时，需明确归档：

1. **脚本版本与更新时间**——所用脚本尾号（版本号）及最后更新时间。
2. **计算参数**——目标地震波、缩放策略、最高求解频率 `f_max`。
3. **网格建议阈值**——基岩面最大划分尺寸：

   ```
   mesh_size ≈ Vs / (10 × f_max)
   ```

   其中 `Vs` 为剪切波速，`f_max` 为最高频率。
4. **自由场 QA**——核对 `case_meta.json` 中 `ff_theory`（远场 PGA 理论值）与实测值的偏差。
5. **ML 结果**——附 v3 对照表（`outputs_v3/table_final.md`）与 v2 vs v3 对照图，注明 GroupKFold 划分与外推测试结论。
