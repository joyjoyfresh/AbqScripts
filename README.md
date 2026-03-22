# Abaqus Scripts: 斜坡地形对地震动分布影响研究

本项目是一组面向 Abaqus 的 Python 脚本，用于构建二维斜坡场地模型，施加斜入射地震动（SV 波）及粘弹性人工边界（VAB），并在计算后提取坡面 PGA（峰值加速度）分布，分析地形放大效应。

## 1. 项目目标

核心研究问题：

- 斜坡几何（如坡高、坡角）如何改变地表地震动空间分布。
- 不同输入地震波作用下，坡顶、坡脚及平台区域 PGA 的差异与规律。
- 斜入射条件下（非垂直入射）自由场波动与边界等效力耦合作用对响应的影响。

## 2. 目录结构

```text
Scripts/
├─ Batch/                 # 批处理工具（批量/选择执行 Python）
├─ Modeling/
│  ├─ Single/             # 单模型主流程脚本（当前建议版本在此）
│  ├─ Multi/              # 多模型相关脚本
│  └─ Deprecated/         # 旧版本保留
├─ Postprocess/           # ODB 后处理与 PGA 分布绘图
└─ Wave/
	 ├─ Seismic/            # 地震波预处理、缩放与谱分析
	 └─ Impulse/            # 脉冲/Ricker 波等工具
```

## 3. 推荐工作流

建议按以下顺序执行：

1. 地震波预处理（滤波、调幅到目标 PGA）
2. Abaqus 建模与批量分析提交
3. ODB 后处理提取坡面 PGA
4. 多工况 PGA 分布统计与作图

### 第一步：地震波预处理

可使用脚本 [Wave/Seismic/Chart_plot_v2.py](Wave/Seismic/Chart_plot_v2.py)。

主要功能：

- 自动扫描脚本目录下 txt 地震波（默认排除 `_scaled` 文件）
- 低通滤波
- 调幅到目标 PGA（例如 0.30g）
- 绘制时程、傅里叶谱、反应谱

输入建议格式（两列）：

- 第 1 列：time（s）
- 第 2 列：acceleration

### 第二步：建模与作业提交

主脚本推荐使用 [Modeling/Single/VAB_oblique_noGUI_v9.py](Modeling/Single/VAB_oblique_noGUI_v9.py)。

脚本流程概述：

1. 读取当前目录所有 txt 地震动文件
2. 建立二维平面应变斜坡模型（几何/材料/网格）
3. 自动创建边界节点集合（l/r/b）
4. 计算并施加粘弹性人工边界（弹簧-阻尼）
5. 将输入加速度时程积分为速度和位移
6. 计算自由场响应与等效节点力时程
7. 为每个地震波复制模型、创建分析步并提交作业

典型运行方式（在 Abaqus Python 环境中）：

```bash
abaqus cae noGUI=Modeling/Single/VAB_oblique_noGUI_v9.py
```

或在脚本目录下：

```bash
abaqus cae noGUI=VAB_oblique_noGUI_v9.py
```

注意：

- 输入 txt 被视为加速度时程（time, acceleration）。
- `build_models` 中调用人工边界函数时使用参数名 `acc_file`。

### 第三步：ODB 后处理（PGA 提取）

推荐脚本 [Postprocess/Postprocess_PGA_v3.py](Postprocess/Postprocess_PGA_v3.py)。

功能特点：

- 自动处理当前目录下全部 odb
- 顶部节点按 `bucket_width` 分桶，每桶仅保留 1 个代表节点
- 输出水平/竖向 PGA 及峰值对应帧号、时刻
- 可选空间平滑列（默认仅用于展示）

典型运行方式：

```bash
abaqus cae noGUI=Postprocess/Postprocess_PGA_v3.py
```

输出：

- 每个 odb 生成一个 CSV，例如 `PGA_job-XXX.csv`
- 关键字段包括：`x/h, node_label, x, y, PGA_h, PGA_v, peak_h_frame, peak_h_time, peak_v_frame, peak_v_time`

### 第四步：PGA 分布统计绘图

可使用 [Postprocess/Distribution_PGA_v2.py](Postprocess/Distribution_PGA_v2.py)。

功能：

- 自动读取脚本目录 CSV（`*.csv`）
- 绘制多输入波曲线及平均曲线
- 同时支持 `PGA_h` 与 `PGA_v` 分量
- 可在图中标记坡顶与坡脚位置

## 4. 快速开始

1. 将输入地震波 txt 放入目标运行目录（或 Wave 子目录按需预处理）。
2. 在 Abaqus 命令环境运行建模脚本。
3. 分析完成后，在包含 odb 的目录运行 PGA 后处理脚本。
4. 将生成的 CSV 放在绘图脚本目录或直接在该目录执行分布绘图脚本。

## 5. 输入输出约定

### 输入

- 地震动 txt：两列数据 `[time, acceleration]`
- 建模参数：在建模主脚本顶部集中配置（坡高、坡角、材料、网格、入射角等）

### 输出

- `.cae` 工程文件
- `job-*.odb` 分析结果
- `PGA_*.csv` 后处理结果
- 绘图窗口或导出的图片文件（按绘图脚本设置）

## 6. 关键参数建议

- 网格尺寸可结合最高目标频率估算：`mesh_size ~= Vs / (f_max * n_per_wave)`
- 常用 `n_per_wave = 8~10`
- 反应谱与滤波参数需与研究频带保持一致
- 批量输入波时建议统一采样间隔或在预处理阶段重采样

## 7. 批处理辅助工具

[Batch/Auto_runPython.bat](Batch/Auto_runPython.bat)：自动顺序执行目录下全部 py。

[Batch/Choose_runPython.bat](Batch/Choose_runPython.bat)：交互式选择单个 py 执行。

这两个脚本适用于普通 Python 工具脚本，不直接替代 Abaqus 的 `abaqus cae noGUI=...` 调用。

## 8. 版本说明

- `Deprecated/` 中为历史版本，便于回溯，不建议作为新研究起点。
- 当前建议优先使用：
	- 建模：[Modeling/Single/VAB_oblique_noGUI_v9.py](Modeling/Single/VAB_oblique_noGUI_v9.py)
	- 后处理：[Postprocess/Postprocess_PGA_v3.py](Postprocess/Postprocess_PGA_v3.py)
	- 分布绘图：[Postprocess/Distribution_PGA_v2.py](Postprocess/Distribution_PGA_v2.py)

## 9. 常见问题

1. 找不到 CSV 或 ODB

- 请确认脚本运行目录与数据目录一致。
- 推荐脚本中尽量使用基于脚本路径（`__file__`）的文件检索方式。

2. 输入 txt 读取失败

- 确保 txt 为纯数字两列，时间严格递增。

3. 计算量较大

- 可先减少输入波数量、缩短时程或降低输出频率进行调试。

## 10. 引用与扩展

如果你在论文或报告中使用本项目，建议记录以下信息以保证可复现：

- 脚本版本（文件名 + 日期）
- 斜坡几何参数
- 材料参数
- 输入地震波及缩放方式
- 网格与步长设置
- 后处理筛选参数（如 `bucket_width`）

