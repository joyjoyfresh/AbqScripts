# TSSI Step-1：二维固定基础框架（建模 + 后处理链路验证）

TSSI 路线第一步。**脱离土体/SSI**，在隔离环境里验证"框架建模 + 抗震指标提取"两半正确，
再往上叠 SSI（step2）、混凝土损伤塑性 CDP（step3）、距坡缘 M/T 扫描（step4）。

## 文件

| 文件 | 作用 |
|------|------|
| `frame_fixedbase_v1.py` | 建模+提交：2D 梁单元框架、集中质量、模态步+隐式动力步、基底水平加速度输入 |
| `postproc_frame_v1.py`  | ODB 后处理：模态 T1、基底剪力、层间位移角、楼层加速度 + 手算锚点校核 |

## 运行

工作目录放一条加速度记录 `.txt`（两列：时间 s, 加速度 m/s²），然后：

```bash
abaqus cae noGUI=Modeling/Hybrid/frame_fixedbase_v1.py       # 建模 + 提交作业
abaqus cae noGUI=Postprocess/Hybrid/postproc_frame_v1.py     # 后处理 + 验证（或 abaqus python ...）
```

参数改 `frame_fixedbase_v1.py` 顶部配置 dict，或在工作目录放 `case_config.json` 覆盖：

```json
{
  "frame_cfg":   {"n_story": 8, "n_bay": 3, "story_height": 3.2, "floor_mass": 6.0e4},
  "damping_cfg": {"ratio": 0.05, "f1": 0.8, "f2": 4.0}
}
```

## 验证锚点（"对不对"靠这三条）

1. **模态 T1**：与经验式 0.1·N、ATC 0.075·H^0.75 同量级（0.5~2× 内合理）→ 验刚度+质量
2. **基底剪力**：ΣRF1 与 Σmᵢ·aᵢ(绝对楼层加速度) 峰值比 ≈ 1±0.1、相关系数 >0.98 → 验质量+反力提取
   （小偏差来自质量比例阻尼 αMv；想要近乎严格对拍可设 `damping_cfg.ratio=0` 单跑一次）
3. **层间位移角**：相邻层 U1 差/层高，剖面形状合理（中下层偏大）→ 验提取逻辑（基底刚体运动自动抵消）

后处理产物：`postproc_results.json`、`base_shear_check.csv`、`drift_profile.csv`。

## 实跑验证结果（2026-06-29，Abaqus 2021，ricker_wavelet_4Hz 输入，5层3跨）

链路全通，三锚点全过：

| 验证项 | 结果 | 判据 | 状态 |
|--------|------|------|------|
| T1（动力FFT反推） | 0.500s，T1/0.1N=1.00 | 0.5~2× | ✅ |
| 基底剪力 ΣRF1 vs Σmᵢaᵢ | 峰值比 1.001，相关 1.0000 | ≈1±0.1, >0.98 | ✅ |
| 层间位移角 | 剖面合理(中部最大) | 形状合理 | ✅ |
| 楼层加速度 | 顶层放大 1.83× | 单调合理 | ✅ |

**踩坑（已修，关键经验）：**
1. **纯集中质量 → 旋转 DOF 无质量 → SIM-Lanczos 特征值病态**（报 `rigid body modes / EXCEEDS expected`，作业在 Frequency 步即中止）。修：梁给微小正则化密度 `density=10`（相对集中质量 0.12% 可忽略）。
2. **B23 三次梁 / B22 二次梁会生成无质量内部节点** + `BEFORE_ANALYSIS` 触发 general-section 材料缺失 FATAL。改回 **B21 + 1单元/构件 + DURING_ANALYSIS**（线性梁不生成内部节点）。
3. **Frequency 微扰步特征值仍≈0**（DURING_ANALYSIS梁+SIM 怪癖，不影响动力步刚度）→ T1 改由**动力响应顶层相对位移 FFT 反推**（系统识别，更稳）。
4. 后处理 Py2.7：`reload(sys);setdefaultencoding('utf-8')` 解中文 print；`str(json读出)` 解 unicode 作 odb key。

## 当前版本边界（v1，刻意从简）

- **材料弹性**（C30, E=30 GPa）。step1 要手算锚点，故先弹性；CDP/纤维截面留到 step3（`BeamSection integration=DURING_ANALYSIS` 给塑性留口）。
- **集中质量模型**：`material_cfg.density=10`(正则化) + 楼层 `floor_mass` 集中质量主导。要真实构件自重则设 2500 并下调 floor_mass（此时 Σm·a 校核需含结构质量）。
- **网格须 1 单元/构件**（B21）：纯集中质量模型不能细分（会出无质量节点）；要细分须改分布密度。
- **瑞利阻尼锚点 f1/f2 可回填**：默认 1/5 Hz；实测 T1≈0.5s(f1≈2Hz)，可把 f1 设 2Hz 提升精度（非必须）。
- **2D 平面应变简化**仍在：B21 平面框架对应单榀，绝对量级定性，step1 只验链路。

## 下一步（验通过后）

- **step2**：把本框架坐到 `Modeling/Multi` 那套土体坡顶（共节点或接触），复用其粘弹性边界+斜入射；
  加"刚性地基 vs SSI"两套对比，注意两模型输入一致性（去耦法）。
- **step3**：梁柱换纤维截面（uniaxial 混凝土+钢筋）或 CDP，盯收敛（可能需转 Explicit）。
- **step4**：套 `case_config.json` 做距坡缘 M/T 扫描。
