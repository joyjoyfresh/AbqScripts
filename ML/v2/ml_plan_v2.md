# 机器学习预测 TAF & PGA 曲线 — 实施方案 v2 (时序→曲线)

> **核心变更**: 从表格回归（标量→标量）升级为 **时序→序列** 预测（地震波加速度时程 → PGA/TAF 完整曲线）。
> **状态**: 方案已确认，可以启动实施。

---

## 1. 问题定义

### 1.1 任务描述

**输入**: 地形几何参数 + 完整地震波加速度时程 → **输出**: 沿地表的 PGA 与 TAF 曲线。

```
┌─────────────────────────────────┐
│  输入                           │
│  ├── h (边坡高度, 标量)          │
│  ├── i (坡面角度, 标量)          │
│  ├── angle (入射角, 标量)        │
│  └── wave(t) (加速度时程, 序列)  │
└────────────┬────────────────────┘
             │  代理模型 (Surrogate)
             ▼
┌─────────────────────────────────┐
│  输出 (统一 162 点, x/h ∈ [0,8]) │
│  ├── PGA_h(x/h) — 水平向 PGA 曲线   │
│  ├── PGA_v(x/h) — 竖向 PGA 曲线     │
│  ├── TAF_h(x/h) — 水平地形放大系数   │
│  └── TAF_v(x/h) — 竖向地形放大系数   │
└─────────────────────────────────┘
```

> **说明**: TAF = PGA_slope / PGA_flat，此处 ML 直接预测 TAF（而非通过 flat 计算），这样用户拿到 slope 的 PGA 和 TAF 后，可以反推 flat 的等效 PGA。

### 1.2 数据规模

| 维度 | 数量 |
|------|------|
| 几何工况 (h × i × angle) | 5×5×7 = **175 组** |
| 地震波种类 | **3 种** (El_Centro, Loma_Prieta, Northridge) |
| 总样本数 | 175 × 3 = **525 个** |
| 每个样本输入长度 | 重采样到 100Hz 后 ~3120~4000 步, 统一 padding 到 4000 |
| 每个样本输出长度 | **162 点** × 4 通道 (PGA_h, PGA_v, TAF_h, TAF_v) |

### 1.3 关键问题与解决方案

| # | 问题 | 解决方案 |
|---|------|----------|
| 1 | **数据稀缺**: 仅 525 个曲线级样本，深度学习极易过拟合 | ① 严格控制参数量 < 500K；② Dropout(0.3~0.5) + Weight Decay(1e-4)；③ 早停 (patience=50)；④ 5-Fold 分组交叉验证确保评估可靠；⑤ 必要时加入波的数据增强（小幅加噪、时间拉伸） |
| 2 | **采样率不一致**: El_Centro 为 50Hz，Loma_Prieta/Northridge 为 100Hz | 统一重采样到 100Hz：El_Centro 线性插值上采样 (1560→3120 步)，其余保持不变 |
| 3 | **变长输入**: 统一到 100Hz 后仍有 ~3120 vs ~4000 步的长度差异 | 尾部补零到 L_max≈4000，构建 `length` 张量；LSTM 用 `pack_padded_sequence`，CNN/Transformer 用 global pooling 天然处理变长 |
| 4 | **输入输出尺度悬殊**: 输入 ~4000 步 → 输出仅 162 点 | 编码器采用激进下采样：5 层 Conv1d(stride=2) 将 4000→~125→GlobalAvgPool；或 LSTM+注意力池化聚合 |
| 5 | **地震波泛化**: 仅 3 种波训练，泛化到新波的能力未知 | 不做跨波泛化假设。用留一波交叉验证评估模型在未见波上的表现，结果写入文档明确适用范围 |
| 6 | **多目标平衡**: 同时预测 PGA_h/PGA_v/TAF_h/TAF_v 四个量级不同的通道 | 对每个通道独立做 Z-score 标准化后训练，损失中可加可调权重 λ；若联合训练不稳定，改为 4 个独立输出头共享编码器 |

---

## 2. 数据流

### 2.1 原始数据结构

```
E:\Abaqus\fuke-ALL\
├── fuke-ALL-h{value}_i{value}_angle{value}/   # 175 个工况文件夹 (全部存在，无缺失)
│   ├── {wave}_scaled.txt                       # 输入: 调幅后地震波加速度时程 (每工况均含 3 波)
│   │   └── 格式: Time\tAcceleration (两列, 空格分隔)
│   ├── PGA-{wave}_scaled-slope.csv             # 输出: slope 场地 PGA
│   │   └── 格式: x/h, node_label, x, y, PGA_h, PGA_v, peak_h_time, peak_v_time
│   ├── PGA-{wave}_scaled-flat.csv              # 参考: flat 场地 PGA
│   └── TAF-{wave}.csv                          # 输出: TAF 曲线
│       └── 格式: x/h, TAF_h
└── ...
```

### 2.2 地震波文件详情

| 波名 | Scaled 文件 | 步数 | dt | 采样率 | 时长 |
|------|-------------|------|-----|--------|------|
| El_Centro | El_Centro_scaled.txt | 1560 | 0.02s | 50 Hz | 31.2s |
| Loma_Prieta | Loma_Prieta_scaled.txt | 3991 | 0.01s | 100 Hz | 39.9s |
| Northridge | Northridge_scaled.txt | 3989 | 0.01s | 100 Hz | 39.9s |

> 三条波采样率和长度不同：El_Centro 为 50Hz/1560步，Loma_Prieta 和 Northridge 为 100Hz/~4000步。预处理阶段统一重采样到 100Hz（El_Centro 线性插值上采样 1560→3120 步），再统一 padding 到 4000 步。

### 2.3 输出曲线格式

**PGA-slope.csv** 关键列:
| 列 | 含义 |
|-----|------|
| `x/h` | 归一化水平距离 |
| `PGA_h` | 水平向峰值加速度 |
| `PGA_v` | 竖向峰值加速度 |

**TAF.csv** 关键列:
| 列 | 含义 |
|-----|------|
| `x/h` | 归一化水平距离 |
| `TAF_h` | 水平地形放大系数 |

> ⚠️ 现有 TAF CSV 仅含 `TAF_h`。`TAF_v` 需从 PGA 数据计算: **TAF_v = PGA_v(slope) / PGA_v(flat)**，在数据加载阶段自动生成。

> 输出目标包含 **PGA_h、PGA_v、TAF_h、TAF_v** 四个通道。

---

## 3. 数据准备

### 4.1 数据集构建流程

```python
# 伪代码: 数据集构建
for each case_folder in fuke-ALL:
    parse h, i, angle from folder name
    
    for each wave in [El_Centro, Loma_Prieta, Northridge]:
        # 加载输入
        t, accel = load_wave(f"{case_folder}/{wave}_scaled.txt")
        
        # 加载输出
        pga_h = load_pga_h(f"{case_folder}/PGA-{wave}_scaled-slope.csv")  # (162,) PGA_h
        pga_v = load_pga_v(f"{case_folder}/PGA-{wave}_scaled-slope.csv")  # (162,) PGA_v
        taf_h = load_taf(f"{case_folder}/TAF-{wave}.csv")                  # (162,) TAF_h
        taf_v = compute_taf_v(pga_v, pga_v_flat)                            # (162,) TAF_v = PGA_v_slope/PGA_v_flat
        
        # 验证一致性
        assert len(pga_h) == len(pga_v) == len(taf_h) == len(taf_v) == 162
        
        # 构建样本
        sample = {
            'wave': accel,          # (L,) 变长加速度序列 (已重采样到 100Hz)
            'h': h,                  # 标量
            'i': i,                  # 标量
            'angle': angle,          # 标量
            'pga_h': pga_h,          # (162,) 水平 PGA 目标
            'pga_v': pga_v,          # (162,) 竖向 PGA 目标
            'taf_h': taf_h,          # (162,) 水平 TAF 目标
            'taf_v': taf_v,          # (162,) 竖向 TAF 目标
        }
```

### 4.2 时序预处理策略

```
步骤 1: 统一采样率 → 100Hz
  - Loma_Prieta & Northridge: 已是 100Hz, 保持不变
  - El_Centro: 50Hz → 100Hz 线性插值上采样 (1560→3120 步)
  - 统一后最大长度: ~4000 步

步骤 2: Padding 到统一长度
  - 统计重采样后所有波的最大长度 L_max = 4000
  - 短于 4000 的波: 尾部补零到 4000
  - 同步构建 `length` 张量记录每条波的实际长度，传给模型用于正确处理填充
```

### 4.3 数据划分

主实验按工况 `(h,i,angle)` 分组，80/20 随机划分，同一工况的 3 条波始终在同一集合中：

| 集合 | 工况数 | 样本数 |
|------|--------|--------|
| 训练集 | 140 | ~420 |
| 验证集 | 35 | ~105 |

补充评估:
- **留一波交叉验证**: 每次留 1 种波的全部数据作验证，其余 2 种波训练，共 3 折
- **留一高度测试**: 留出某个 h 值 (如 h=200) 的全部工况作验证，评估高度外推能力

### 4.4 数据标准化

| 数据 | 方法 | 说明 |
|------|------|------|
| 加速度时程 | Z-score 全局标准化 | 对所有波的所有时间步统一计算 μ 和 σ，保留波间幅值差异 |
| h, i, angle | MinMax 缩放到 [0, 1] | h∈[10,400], i∈[15,75], angle∈[0,30] |
| PGA_h, PGA_v | Z-score 全局标准化 | 对所有样本统一计算 μ 和 σ |
| TAF_h, TAF_v | Z-score 全局标准化 | 全局 μ/σ，推理时反变换还原 |

> 标准化参数 (μ, σ, min, max) 从训练集计算，持久化保存，推理时复用。

### 4.5 数据质量检查清单

| 检查项 | 方法 | 处理 |
|--------|------|------|
| x/h 范围一致 | 是否都为 0~8, 162 点 | 不一致则插值对齐 |
| TAF_h 异常值 | TAF_h < 0 或 > 5 | 标记并排查原始 FEM 结果 |
| TAF_v 除零风险 | flat 场地 PGA_v ≈ 0 导致 TAF_v 无穷大 | 标记该工况，用相邻 x/h 插值填充 |
| 波的数据质量 | NaN/Inf 检查 | 剔除该样本 |
| 波的长度 | 统计最短/最长 | 已确认: El_Centro 3120步(重采样后), 其余 ~4000步 |

---

## 4. 模型架构设计

### 5.1 架构概览

```
                     ┌──────────────────────┐
  加速度时程 wave(t) │   时序编码器          │
  (L,) ─────────────►│   1D-CNN / LSTM /     │──► 波特征向量 (D_wave,)
                     │   Transformer         │
                     └──────────────────────┘
                              │
  标量条件 (h,i,angle)       │  拼接
  (3,) ──────────────────────►
                              │
                     ┌────────▼─────────────┐
                     │   曲线解码器          │
                     │   MLP / Conv1D       │──► PGA_h (162,)
                     │                      │──► PGA_v (162,)
                     │                      │──► TAF_h (162,)
                     │                      │──► TAF_v (162,)
                     └──────────────────────┘
```

关键设计决策:
- **编码器**: 将变长时序压缩为固定维度特征向量
- **条件注入**: 标量参数 (h, i, angle) 与波特征拼接后送入解码器
- **解码器**: 从潜在向量生成固定长度 (162) 的四通道曲线

### 5.2 架构 A: 1D-CNN 编码器

```
输入: wave (L,)
  │
  ├─ Conv1d(1→32, k=15, s=2) → BN → ReLU
  ├─ Conv1d(32→64, k=15, s=2) → BN → ReLU
  ├─ Conv1d(64→128, k=15, s=2) → BN → ReLU
  ├─ Conv1d(128→256, k=15, s=2) → BN → ReLU
  ├─ Conv1d(256→256, k=15, s=2) → BN → ReLU
  │
  └─ GlobalAvgPool → FC(256→128) → 波特征 (128,)
     (5 层 stride=2 下采样: 4000 → 2000 → 1000 → 500 → 250 → 125 → pool → 256 → 128)

条件融合: concat([波特征(128), h(1), i(1), angle(1)]) → (131,)

解码器: FC(131→256) → FC(256→512) → FC(512→162×4) → reshape → (4, 162)
```

**优点**: 参数少、训练快、对时序局部模式敏感  
**缺点**: 感受野固定，可能丢失长程依赖

### 5.3 架构 B: LSTM 编码器 + MLP 解码器

```
输入: wave (L,)
  │
  ├─ Linear(1→64)  # 嵌入
  ├─ LSTM(64→128, num_layers=2, bidirectional=True, batch_first=True)
  │   └─ 输出: (L, 256)  [双向拼接]
  │
  ├─ 聚合策略: 注意力池化
  │   └─ 对 LSTM 输出做 self-attention 加权求和 → (256,)
  │
  └─ FC(256→128) → 波特征 (128,)

条件融合/解码器: 同架构 A
```

**优点**: 原生支持变长序列 (pack_padded_sequence)，擅长时序建模  
**缺点**: 串行计算，长序列较慢

### 5.4 架构 C: Transformer 编码器

```
输入: wave (L,)
  │
  ├─ 位置编码: 可学习 PositionalEncoding (L, d_model=128)
  ├─ Linear(1→128) + PosEnc
  │
  ├─ TransformerEncoder × 4 层
  │   ├─ d_model=128, nhead=8, dim_feedforward=512
  │   ├─ Dropout=0.1
  │   └─ Padding Mask: 屏蔽填充位置
  │
  └─ 全局平均池化 (仅非填充位置) → FC(128→128) → 波特征 (128,)

条件融合/解码器: 同架构 A
```

**优点**: 全局感受野，并行计算，表征能力强  
**缺点**: 数据量要求高 (525 样本偏少)，容易过拟合

### 5.5 架构 D: DeepONet 风格

```
分支 1 (Branch Net — 处理波):
  输入: wave (L,)
  ├─ 1D-CNN 编码器 (同架构 A)
  └─ 输出: (128,)

分支 2 (Trunk Net — 处理输出位置):
  输入: x/h 查询点 (162, 1)
  ├─ MLP: (1→64→128→128)
  └─ 输出: (162, 128)

融合: 逐点内积
  output[i] = dot(Branch_out, Trunk_out[i])  # (128,) · (128,) → 标量
  最终: (162,) × 4 通道 (PGA_h, PGA_v, TAF_h, TAF_v)

条件 (h,i,angle): 拼接到 Branch 输入，在 wave 嵌入前与标量特征融合
```

**优点**: 天然适合"函数→函数"映射，可泛化到任意 x/h 查询点  
**缺点**: 实现较复杂

### 5.6 统一训练配置

四种架构使用完全相同的训练配置，确保对比公平：

```python
TRAIN_CONFIG = {
    'device': 'cuda',                        # RTX 3060 Ti (8GB), 模型仅需 ~2-3GB
    'optimizer': 'AdamW',
    'learning_rate': 1e-3,
    'lr_scheduler': 'ReduceLROnPlateau',     # val_loss 连续 10 epoch 不降则 lr×0.5
    'batch_size': 16,
    'max_epochs': 500,
    'early_stopping_patience': 50,
    'loss': 'MSE',
    'val_metric': 'R²',
    'random_seed': 42,
}
```

---

## 5. 损失函数

```python
# 多任务 MSE，各通道标准化到相似量级后等权求和
loss = (MSE(PGA_h_pred, PGA_h_true)
      + MSE(PGA_v_pred, PGA_v_true)
      + MSE(TAF_h_pred, TAF_h_true)
      + MSE(TAF_v_pred, TAF_v_true))
```

> 输出数据已通过 Z-score 标准化统一量级，因此四个通道使用等权 MSE。若训练中发现某通道收敛困难，再单独调整权重。

---

## 6. 评估指标

### 6.1 全局指标 (逐点平均，162×N 个点汇总计算)

| 指标 | 目标值 |
|------|--------|
| **R²** | > 0.90 |
| **MAE** (PGA_h) | < 0.02 |
| **MAE** (PGA_v) | < 0.01 |
| **MAE** (TAF_h) | < 0.04 |
| **MAE** (TAF_v) | < 0.04 |
| **RMSE** (PGA_h) | < 0.03 |
| **Max Error** (TAF_h) | < 0.10 |

### 6.2 曲线级指标

| 指标 | 说明 |
|------|------|
| **逐工况 R²** | 每个样本 162 个点的 R² 分布 (箱线图) |
| **峰值误差** | `max(PGA_pred) - max(PGA_true)` |
| **峰值位置误差** | `argmax(PGA_pred) - argmax(PGA_true)` (单位: x/h) |
| **DTW 距离** | 动态时间规整距离，衡量曲线形状相似度 |

### 6.3 物理一致性检查

| 指标 | 期望 |
|------|------|
| TAF 峰值位置 | x/h ∈ [0, 1.5] (坡面附近, TAF_h 和 TAF_v 分别检查) |
| TAF 远场值 | $|TAF(x/h>5) - 1.0| < 0.1$ |
| PGA 趋势 | x/h 增大时 PGA_h/PGA_v 应单调趋近于远场平稳值 |

---

## 7. 防止过拟合

525 样本对深度学习而言极少，以下措施强制执行：

| 措施 | 具体配置 |
|------|----------|
| **架构轻量化** | 所有架构参数量控制在 500K 以内 |
| **Dropout** | 编码器输出后的 FC 层: Dropout=0.3；解码器 FC 层: Dropout=0.2 |
| **Weight Decay** | AdamW weight_decay=1e-4 |
| **早停** | 验证 loss 连续 50 epoch 不降则终止训练，恢复最佳权重 |
| **梯度裁剪** | torch.nn.utils.clip_grad_norm_(max_norm=1.0) |
| **5-Fold 交叉验证** | GroupKFold 按工况分组，取 5 折平均 R² 作为最终指标 |
| **数据增强** | 对每条波加高斯噪声 (σ=0.01×波的标准差)，训练集扩充 2 倍 |

---

## 8. 可视化输出

1. **模型对比**: 4 架构的 R²/MAE 柱状图
2. **训练曲线**: Loss/Epoch 对比 (4 架构叠加)
3. **Parity Plot**: 最优模型的 PGA_h/PGA_v/TAF_h/TAF_v 预测 vs 真实散点图（四个子图）
4. **曲线对比**: 随机抽取 6 个工况，四条曲线预测 vs FEM 叠加
5. **误差分布**: 残差直方图 + 误差沿 x/h 变化趋势
6. **最差工况分析**: 误差最大的 3 个工况详解
7. **外推热力图**: 留一高度/角度的 R² 热力图
8. **学习曲线**: 训练集大小 vs 验证 R²

---

## 9. 代码结构

```
AbqScripts/ML/
├── ml_plan_v2.md                  # 本文件
├── data/
│   └── dataset_v2.py              # PyTorch Dataset 与 DataLoader
├── models/
│   ├── base_model.py              # 基类 (训练循环, 评估, 保存)
│   ├── cnn_encoder.py             # 1D-CNN 编码器
│   ├── lstm_encoder.py            # LSTM 编码器
│   ├── transformer_encoder.py     # Transformer 编码器
│   ├── deeponet.py                # DeepONet 风格
│   └── decoder.py                 # 曲线解码器 (共享)
├── train_v2.py                    # 多架构训练入口
├── evaluate_v2.py                 # 评估与可视化
├── config_v2.py                   # 配置文件
└── utils_v2.py                    # 工具函数
```

---

## 10. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 525 样本不足以训练深度模型 | 中高 | 🔴 致命 | 轻量化架构优先，严格控制参数；必要时回退到波特征+表格模型 |
| 变长序列处理引入 bug | 中 | 🟡 | 充分的单元测试 + 固定长度模式兜底 |
| PGA_h/PGA_v/TAF_h/TAF_v 四个目标难以同时优化 | 中 | 🟡 | 调整 λ 权重，或分独立输出头 |
| 地震波泛化不可行 (仅 3 波训练) | 高 | 🟡 | 用留一波交叉验证诚实评估泛化上限，明确文档声明适用范围 |
| 训练不稳定 (NaN loss) | 低 | 🟡 | 梯度裁剪、降低学习率、BatchNorm |

---

## 11. 成功标准

| 项目 | 判定标准 |
|------|----------|
| 数据加载 | 525 个样本全部成功加载，无文件缺失 |
| 模型训练 | 至少 1 个架构在验证集上 R² > 0.90, MAE(TAF_h) < 0.04 |
| 泛化评估 | 留一波交叉验证 R² > 0.80 |
| 推理性能 | 单次推理 (含预处理) < 1 秒 |

---

## 12. 环境要求

| 项目 | 规格 |
|------|------|
| GPU | NVIDIA RTX 3060 Ti (8GB VRAM) |
| 深度学习框架 | PyTorch ≥ 2.0 |
| Python | ≥ 3.9 |
| 关键依赖 | numpy, scipy, pandas, matplotlib, scikit-learn |

