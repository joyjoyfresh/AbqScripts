# AbqScripts：斜坡场地地震响应数值模拟与复频响代理模型

> 当前状态以 [第四至第六章与英文期刊小论文融合研究实施总计划](docs/计划文档/毕业论文与小论文科研总计划.md) 为准。计划中的工况数量不等于已完成结果；只有规范数据和独立评价均完成后，才进入论文结论。

本项目是面向 Abaqus 的 Python 工具集，用于研究二维斜入射成层坡地的地震响应、地形放大、覆盖层修正、土-结构相互作用（TSSI）及复频响代理建模。代码覆盖波形准备、Abaqus 建模、地表响应后处理、数值敏感性评价、机器学习和真实波重构。

## 当前研究主线

- **第四章与第六章**：用宽频多正弦输入识别频率—空间复频响
  `G_h(f,s)`，联合研究幅值、相位、群时延和空间相位梯度；再用降阶代理预测完整复频响，并重建未见真实波下的时程、PGA、TAF 和反应谱。
- **第五章**：采用独立的 freefield—fixed—ssi 三胞胎路线研究坡顶结构的 TSSI，不与第四、六章代理模型混训。
- **当前主计划**：430 次主线动力提交，其中第四、六章 342 次（338 次科学工况 + 4 次数值验证），第五章 TSSI 88 次；英文小论文 90 次科学工况包含在第四、六章数据池内。

## 当前进度

| 批次 | 内容 | 状态 |
| --- | --- | --- |
| V | P061 基准、V001—V003 网格/计算域/尾段验证 | 已生成规范数据并完成独立评价；当前不重复运行 |
| V 补充 | V004 侧向净空单因素检查 | 已完成独立评价；4H 作为当前工作域候选，残余不确定性已记录 |
| H | H001—H004 均质基线 | 待运行和分析 |
| P | P001—P064 三变量开发矩阵 | 待运行和分析 |
| B | B001—B012 未见组合盲测 | 待 P 模型锁定后运行 |
| C | C001—C010 真实波直接闭环 | 待 B 评价和模型永久锁定后运行 |

当前已经完成的实现包括：统一建模与后处理流水线、完整复频响和有效掩码、幅相联合分析、整工况折内训练、最近邻/POD-Ridge/POD-GPR 比较、真实波重构，以及对应的纯 Python 回归测试。上述“实现完成”不代表 H/P/B/C 的物理结果已经完成。

## 统一数据口径

以同侧同地层一维自由场为分母：

```text
G_h(f,s) = A_h_2D(f,s) / A_h_1D(f)
```

- 公共频率网格：0.5—10 Hz，96 点；
- 公共空间网格：`s=-4.00...4.00`，步长 0.05，共 161 点；
- 代理内部预测 `Re(G_h)` 和 `Im(G_h)`，再统一派生幅值、展开相位、群时延和空间相位梯度；
- 系统识别输入：`Wave/Impulse/Acceleration/G1b_frequency_gate/g1b_multisine_phase_a.txt`；
- 真实波闭环输入：`Wave/Seismic/Sp_EQ/` 下的 EQ01—EQ03 预处理记录。

## 当前代码结构

```text
AbqScripts/
├─ Modeling/
│  ├─ slope_frame_ssi_full_v2.py       # 当前唯一主建模脚本
│  └─ Archived/                        # 旧版建模脚本
├─ Postprocess/
│  ├─ Postprocess_All_surface_v2.py    # 单工况数据提取与复频响计算
│  ├─ Collect_All_results_v2.py        # 跨工况收集规范结果
│  ├─ Plot_Hybrid_surface_v2.py        # 跨工况分图
│  └─ Archived/                        # 旧版后处理脚本
├─ Run/
│  ├─ Auto_ch4/                        # 当前小论文 V/H/P/B/C 批处理入口
│  ├─ evaluation/                      # 复频响分析与数值敏感性评价
│  └─ ch4_*                            # 本地求解输出与规范结果
├─ ML/
│  ├─ train_complex_frf_surrogate.py   # 当前复频响代理训练
│  ├─ reconstruct_real_wave.py         # 真实波重构与闭环指标
│  └─ v1/、v2/、v3/                    # 历史模型与已有结果
├─ Wave/                               # 真实波、脉冲波和系统识别输入
├─ Batch/                              # 通用批处理模板
├─ docs/                               # 计划、技术文档和论文材料
├─ test/                               # 本地测试目录，整体由 .gitignore 排除
├─ requirements.txt
└─ README.md
```

旧版本脚本集中在 `Modeling/Archived/` 和 `Postprocess/Archived/`；新流程默认使用根目录下的当前脚本，不再按旧的 `Single/Multi` 目录寻找主入口。

## 批处理入口

入口均位于 `Run/Auto_ch4/`，可用系统 Python 3 调用，并通过参数传入绝对输出根目录：

| 入口 | 用途 |
| --- | --- |
| `Autorun_ch4_sp_01_V_v1.py` | P061 与 V001—V003 数值验证；已完成，勿重复运行 |
| `Autorun_ch4_sp_01_V_domain_sensitivity.py` | V004 及补充计算域敏感性 |
| `Autorun_ch4_sp_02_H_v1.py` | H001—H004 均质基线 |
| `Autorun_ch4_sp_03_P_v1.py` | P001—P064 开发工况 |
| `Autorun_ch4_sp_04_B_v1.py` | B001—B012 锁模后的未见组合盲测 |
| `Autorun_ch4_sp_05_C_v1.py` | C001—C010 真实波直接有限元闭环 |

示例：

```powershell
py -3 Run/Auto_ch4/Autorun_ch4_sp_02_H_v1.py C:\Abaqus\AbqScripts\Run\ch4_sp_02_H
```

运行前先阅读 [小论文批处理运行说明](docs/技术文档/小论文批处理运行说明.md)。B 批次只能在模型锁定后运行，C 批次不能用于重新选模；已完成的 V 批次不要重复启动。

## 环境与常用脚本

- Abaqus 建模和依赖 ODB 的后处理使用 Abaqus 内置 Python 2.7，由批处理入口调用；
- 复频响分析、代理训练、真实波重构和纯 Python 测试使用系统 Python 3；
- 普通 Python 依赖见 [`requirements.txt`](requirements.txt)：NumPy、SciPy、Pandas、scikit-learn、绘图和相关工具。

常用分析脚本：

```powershell
py -3 Run/evaluation/analyze_complex_frf.py --help
py -3 ML/train_complex_frf_surrogate.py --help
py -3 ML/reconstruct_real_wave.py --help
```

## 数据状态与保留规则

工况状态严格区分：

```text
planned → prepared → solved → data_ready → evaluated
```

`data_ready` 只表示后处理成功，不等于物理规律或代理模型结论已经成立。成功工况在清理 ODB 前必须确认至少存在且非空：

- `postprocess_status.json`，状态为 `completed`；
- `surface_results.npz`；
- `surface_results.xlsx`；
- 需要真实波参考时，还要有对应的 `freefield_reference_*.npz`。

失败工况保留 ODB、日志和状态文件，先诊断再决定是否重算。独立评价脚本负责质量和研究判读，不能用后处理是否成功替代评价结论。

## 文档入口

- [主研究计划](docs/计划文档/毕业论文第四至第六章与英文期刊小论文融合研究实施总计划最终版.md)：工况、口径、执行顺序和章节边界的唯一当前依据；
- [小论文批处理运行说明](docs/技术文档/小论文批处理运行说明.md)：V/H/P/B/C 的运行前检查、锁模、盲测和闭环规则；
- `docs/论文材料/章节Markdown/`：当前论文各章节 Markdown；
- `docs/计划文档/归档/`、各模块 `Archived/`：历史计划和旧版脚本，不作为当前主线依据。

当前结论仅能覆盖已经实际完成、数据完整且独立评价充分的二维平面应变模型、参数域和输入波。未完成工况仍属于计划，不应在 README、报告或论文中写成已完成结果。
