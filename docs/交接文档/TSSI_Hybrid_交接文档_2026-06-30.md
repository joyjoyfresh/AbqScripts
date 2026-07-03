# TSSI 建模交接文档（坡顶建筑地震响应 / 开题内容(4)）

> 日期：2026-06-30 ｜ 环境：Windows 11 + Abaqus 2021 ｜ 工作区：`C:\Users\12462\Documents\Code\AbqScripts`
> 本文档为单次会话交接：把开题"主要研究内容(4) TSSI 建筑地震响应分析"落地为可跑的有限元流程，分步实现并实跑验证。

---

## 0. 一句话状态

固定基础框架(step1)、平层 SSI(step2a)、平层刚性vs SSI去耦(step2a+)、**坡顶 SSI 复用 Multi 引擎(step2b)** 四步**已写代码并在本机 Abaqus 实跑验证通过**；下一步 step2b-2（坡顶刚性vs SSI去耦）→ step3（CDP 结构非线性）→ step4（M/T 距坡缘扫描）。所有新代码在 `Modeling/Hybrid/`。

---

## 1. 背景与目标

开题报告 `docs/开题报告1205-提交版.docx` 主要研究内容(4)：**考虑地形-土-结构相互作用(TSSI)的建筑地震响应分析**——坡顶多层框架、结构非线性(塑性损伤/刚度退化)、刚性地基 vs SSI 对比、距坡缘 M/T 破坏模式。

原有脚本 `Modeling/Multi/VAB_oblique_multilayer_nonlinear_v3.py`（2941 行）是**纯自由场无结构**的坡面场地响应器（上土下岩分层 + 粘弹性吸收边界 + 斜入射 SV + EQL 土非线性 + 频域 fd 自由场引擎）。内容(4)需在其上新增**结构 + SSI + 抗震后处理**模块。

**落地策略**：隔离验证，一步只加一个新风险，每步实跑 + 手算/物理锚点验对，再往上叠。

```
step1   固定基础框架            验"建模+后处理"两半（脱离土体）
step2a  平层SSI耦合             验框架-土Tie/地震透土/周期侧边/SSI周期延长
step2a+ 平层刚性vs SSI去耦       验去耦法(两模型输入一致性)
step2b  坡顶SSI(复用Multi引擎)   验坡面集成（地形放大→建筑响应）
step2b-2 坡顶刚性vs SSI去耦       ← 下一步
step3   CDP结构非线性            开题真要的塑性损伤(最高风险,可能转Explicit)
step4   M/T距坡缘扫描            框架x_off参数化 + case_config.json
```

---

## 2. 已完成（4 步，全部实跑验证通过）

### step1 — 固定基础 2D 框架（`frame_fixedbase_v1.py` + `postproc_frame_v1.py`）
脱离土体，验证框架建模 + 抗震指标提取。B21 梁、弹性 C30、楼层集中质量、隐式动力、基底 `AccelerationBC` 水平加速度。

| 锚点 | 结果 | 判据 |
|------|------|------|
| 基底剪力 ΣRF1 vs Σmᵢaᵢ | 峰值比 **1.001**、相关 **1.0000** | ≈1 → 牛顿定律验后处理 |
| T1（动力响应 FFT） | 0.500s，T1/0.1N=1.00 | 与经验式吻合 |
| 层间位移角/楼层加速度 | 剖面合理、顶层 1.83× | 形状合理 |

### step2a — 平层均质土 SSI（`frame_ssi_v1.py` + `postproc_ssi_v1.py`）
框架坐平层土块顶(Tie)。边界=**周期侧边(Equation 左右同高 U 相等→1D 剪切)+刚性基底 AccelerationBC**（非吸收边界——平层不需要）。

| 锚点 | 结果 |
|------|------|
| 场地地表主频 | **2.50Hz = Vs/(4H) 精确** → 1D 剪切柱+土+输入全对 |
| SSI 周期延长 | T 0.500→0.667s，**1.33×** |
| 结构响应方向 | 漂移增大、基底剪力略降（SSI 正确方向） |

### step2a+ — 平层刚性 vs SSI 去耦对照（`frame_ssi_v1.py` 加 `fixed` 场景）
**去耦法关键**：刚性模型基底输入 = freefield 跑出的**自由场地表运动**（无结构，脚本自动提取），使 fixed 与 ssi 唯一差异 = SSI。

| 指标 | 刚性 | SSI | SSI/刚性 |
|------|------|-----|---------|
| 自振周期 | 0.500s | 0.667s | 1.33 |
| 基底剪力 | 1.27e6 N | 2.02e5 N | 0.16 |
| 顶层加速 | 9.47(9.5×) | 1.68(1.7×) | 0.18 |

本例刚性自振 2.0Hz 撞场地基频 2.5Hz 近共振 → SSI 周期延长**失谐离开共振** → 响应骤降（**SSI 失谐减震**，有利情形）。

### step2b — 坡顶 SSI，复用 Multi 引擎（`frame_ssi_slope_v1.py` + `postproc_ssi_slope_v1.py`）
**import 复用** Multi 的坡面+分层+粘弹性吸收边界+斜入射引擎，框架坐**坡顶**(右缘贴坡肩)Tie 耦合。缩小配置(H_upper=100/总长300/网格5m)调通，各约 5min。

| 指标 | 结果 | 物理意义 |
|------|------|---------|
| 坡顶自由场放大 | **3.89×**，主频 3.0Hz | 地形+地层放大 |
| SSI 周期延长 | 1.33× | 土柔度 |
| 坡顶结构响应 | 顶层 **4.67×**、基底剪力 4.4e5N、漂移 0.0033 | 建筑放大已放大的坡顶运动 |

**贯通研究线**：斜入射15° → 坡顶地形放大 3.89× → 坡顶建筑顶层 4.67×（开题核心机理"地形放大致坡顶建筑震害加重"端到端量化；对比 step2a 平层 1.83×/1.68×，坡顶高近 3 倍）。

---

## 3. 文件清单（均在 `Modeling/Hybrid/`）

| 文件 | 说明 |
|------|------|
| `frame_fixedbase_v1.py` / `postproc_frame_v1.py` | step1 建模 / 后处理 |
| `frame_ssi_v1.py` / `postproc_ssi_v1.py` | step2a+2a+ 建模(freefield/ssi/fixed) / 后处理(含去耦对比) |
| `frame_ssi_slope_v1.py` / `postproc_ssi_slope_v1.py` | step2b 坡顶 SSI 建模(import Multi) / 后处理 |
| `README_step1.md` / `README_step2a.md` / `README_step2b.md` | 各步详细说明+实跑结果+踩坑 |
| `test_run/` `test_ssi/` `test_slope/` | 各步参考算例(含 odb + 结果 json/csv)，可删可留 |

依赖：`Modeling/Multi/VAB_oblique_multilayer_nonlinear_v3.py`（step2b import 复用，勿动）。
测试输入：`Wave/Impulse/Acceleration/ricker_wavelet_4Hz.txt`（两列 时间,加速度）。

---

## 4. 关键技术决策与踩坑（务必读）

### 建模决策
- **B21 线性梁 + 每构件 1 单元**：纯集中质量模型下不生成无质量内部节点（B22/B23 会生成→特征值病态）。
- **梁微小正则化密度 `density=10`**：纯集中质量使旋转 DOF 无质量、SIM-Lanczos 特征值病态报 `rigid body modes`；给 10(占集中质量 0.12%)使所有 DOF 质量矩阵非奇异，Σm·a 校核仍干净。
- **T1 用动力响应 FFT 反推**，不用 Frequency 微扰步（DURING_ANALYSIS 梁 + SIM Lanczos 怪癖致微扰特征值≈0，但不影响动力步刚度）。**注意**：当结构自振≈输入主频时 FFT 读到受迫频率（step2a+ 刚性周期改用 step1 自振值）。
- **柱脚 Tie 铰接**(`tieRotations=OFF`)：整体摇摆经多柱差动竖向捕捉。
- **去耦法**：刚性模型必须输入 freefield(无结构)的自由场地表运动，否则对比无意义。

### 踩坑（Abaqus 2021 + Py2.7）
- `model.Tie` 用**旧关键字** `master=`/`slave=`（非 `main`/`secondary`）。
- assembly 级 `SetFromNodeLabels(instanceName=...)` 关键字错 → 用 `asm.Set(nodes=inst.nodes.sequenceFromLabels([lab]))`。
- Multi 内核**无 `__file__`** → step2b 用多路径+绝对路径兜底定位 `Modeling/Multi`。
- Multi 固定建 `'Model-1'` → `mdb.models.changeKey` 改场景名以建多模型。
- 后处理 Py2.7 控制台中文 print 崩 → 顶部 `reload(sys);sys.setdefaultencoding('utf-8')`；`json.load` 返回 unicode 作 odb key 崩 → `str()` 转换。
- `findAt` 用解析坐标归类边/顶点（比 `.index`/`sequenceFromLabels` 稳）。

---

## 5. 如何运行

```bash
# Abaqus CLI: /c/SIMULIA/Commands/abaqus.bat
# 进各自 test 目录(已有 ricker_wavelet_4Hz.txt)，例 step2b:
cd test/Hybrid/test_slope
"/c/SIMULIA/Commands/abaqus.bat" cae noGUI="<绝对路径>/frame_ssi_slope_v1.py"   # 建+提交
"/c/SIMULIA/Commands/abaqus.bat" python "<绝对路径>/postproc_ssi_slope_v1.py"    # 后处理
```
- 建模脚本顶部配置 dict 改参数；`job_cfg['submit']` 控制是否提交。
- 求解耗时：框架单体~4min(2000隐式增量×正则化密度)；坡面缩小配置~5min/模型。

---

## 6. 当前边界 / 简化（论文需交代）

- **2D 平面应变**：框架对应单榀，绝对量级偏定性（开题"多层框架"读者默认 3D）。
- **结构弹性**：尚未上 CDP/纤维截面（step3 才是开题要的塑性损伤/刚度退化）。
- **集中质量 + 柱脚铰接 Tie**：简化基础；要柱脚抗弯需加基础底板。
- **step2b 缩小配置**：H_upper=100/总长300/网格5m 为调通集成；论文尺度放大 `soil_geometry_cfg` 即可(变慢)。
- **频域自由场 + 结构非线性自洽**：土用 EQL(等效线性)+结构真非线性是可行组合；土要真非线性需换时域自由场。

---

## 7. 下一步（具体怎么做）

### step2b-2：坡顶刚性 vs SSI 去耦（小改，建议先做）
给 `frame_ssi_slope_v1.py` 加 `fixed` 场景，套 step2a+ 的去耦法到坡面：
1. freefield 解算后，从 `job-freefield.odb` 的 `CREST_REF` 提取坡顶地表 A1（脚本已输出该节点）。
2. 建框架单体模型，基底 `AccelerationBC` = 该坡顶自由场运动。
3. postproc 加坡顶 SSI vs 刚性对比。
（参考 `frame_ssi_v1.py` 的 `extract_freefield_surface_acc` + `build_fixed_scene`）

### step3：CDP / 纤维截面结构非线性（最高风险）
- 梁柱 `BeamSection` 已设 `integration=DURING_ANALYSIS` 留口；换纤维截面(uniaxial 混凝土+钢筋)或集中塑性铰。
- 盯收敛：隐式+塑性可能不收敛 → 可能要转 Explicit（架构级改动，斜入射等效力施加方式需重核）。
- 输出损伤变量 DAMAGEC/DAMAGET。

### step4：M/T 距坡缘扫描
- 框架 `x_off`（距坡肩距离）参数化；套 `case_config.json` 注入做多工况扫描。
- 量化"距坡缘不同相对距离 M/T"的破坏模式演化（开题(4)第三层）。

---

## 8. 关键数字速查

| 量 | step2a 平层 | step2b 坡顶 |
|----|------------|------------|
| 自由场地表/坡顶放大 | 1.83× | **3.89×** |
| 结构顶层放大 | 1.68× | **4.67×** |
| SSI 周期延长 | 1.33× | 1.33× |
| 固定基础 T1 | 0.500s（step1） | — |
| SSI 自振 T | 0.667s | 0.667s |

相关记忆：`tssi-hybrid-roadmap`、`freefield-engine-version-history`、`unified-case-meta-pipeline`、`abaqus-py27-gotchas`、`config-comment-one-line`。
