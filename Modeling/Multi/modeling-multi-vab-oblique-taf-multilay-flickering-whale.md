# VAB_oblique_TAF_multilayer_v2.py 复现论文图15 优化方案

## Context（为什么要改）

目标：让 `Modeling/Multi/VAB_oblique_TAF_multilayer_v2.py` 的 Abaqus 结果能复现 Shen 等(2025) **图15**（三层斜坡，i=45°、VR/Vs2=2.5、h/H=0.5、a0=2.0、fc=4Hz、软/硬表层 ×0.25/0.75 厚 ×0°/15°）。

读论文后定位到根因（均属"FE 求解端"，**不触碰延迟时间法/射线自由场引擎**）：

1. **缺少材料衰减（最大差距）**：论文给**每一层**都加了品质因子 Q 衰减——coarse-grain 法 `Qs=0.05·cs`、`Qp=2Qs`、**基岩 Q=999**（≈无衰减）。而本脚本（及仓库内所有建模脚本）`_create_band_materials_sections` 只建了 `Elastic+Density`，**完全没有材料阻尼**。无阻尼 → 分层共振被严重高估，TAF 峰值偏高、曲线过度振荡，对不上论文。
   - 换算：ξ=1/(2Q)。Vs2=800→ξ≈1.25%；软表层 Vs1=400→ξ≈2.5%；硬表层 Vs1=1600→ξ≈0.625%；基岩→ξ≈0.05%。
2. **数值阻尼过强**：分析步用 `application=MODERATE_DISSIPATION`（HHT α≈-0.414），对高频数值阻尼很大，会压低 PGA 峰值，掩盖物理放大。
3. 论文主结果用 **SPECFEM2D 谱元法**（Abaqus 等效节点力法只是论文 2.3 验证对照）——所以"复现"= 让 Abaqus 模型的物理设定（尤其衰减）向 SEM 对齐。

> 说明：自由场等效力由（无阻尼）射线法给出，而 FE 域加了物理阻尼，边界处存在轻微不一致；但 TAF=坡地/自由场之比，两模型材料/阻尼/输入完全相同，比值基本抵消，与论文"对所有层施加 Q、TAF 取比值"口径一致，是可接受的标准近似。

### 用户已确认的决策
- 阻尼换算：**两种方法都实现、用开关切换**（默认瑞利双频拟合）。
- 数值积分：**改 `TRANSIENT_FIDELITY`**（α≈-0.05，让物理阻尼主导）。
- 网格：**保持 4m 全局**（不改）。

本方案仅改 `VAB_oblique_TAF_multilayer_v2.py` 一个文件（+ 新增一个纯 Python 测试）。siblings（multilayer_v1 / double_* / single）有同样问题，作为可选后续，本次不动。

---

## 实施步骤（全部在 `Modeling/Multi/VAB_oblique_TAF_multilayer_v2.py`）

### 1) 新增阻尼配置块（放在 `mesh_size` 附近，~L68 后）
```python
damping_cfg = {                 # 材料阻尼配置（对齐论文 Q 衰减；可被 case_config.json 覆盖）
    'enable': True,             # 是否施加材料阻尼
    'method': 'rayleigh',       # 'rayleigh'=双频拟合(α+β) / 'stiffness'=仅刚度比例(β)
    'qs_factor': 0.05,          # Qs = qs_factor*cs（论文 coarse-grain 法，cs 单位 m/s）
    'q_bedrock': 999.0,         # 基岩品质因子(≈无衰减)
    'fc': None,                 # 输入波主频(Hz)：None=从加速度记录自动估计；可显式/注入覆盖
    'f1_factor': 0.5,           # 双频拟合下限 = f1_factor*fc
    'f2_factor': 2.5,           # 双频拟合上限 = f2_factor*fc（≈Ricker 高频边界）
}
```

### 2) 新增纯函数（物理计算区，`_compute_material_params` 附近）
- `_estimate_dominant_freq(acc, dt)`：对 `acc-mean` 做 `np.fft.rfft`，返回幅值谱最大处频率（DC 置 0），Ricker 即得 fc。
- `_damping_ratio_from_q(cs, is_bedrock, dcfg)`：bedrock 用 `q_bedrock`，否则 `Q=qs_factor*cs`；返回 `(Q, xi=1/(2Q))`。
- `_rayleigh_coeffs(xi, dcfg, fc)`：
  - `method=='stiffness'`：`alpha=0`，`beta=xi/(math.pi*fc)`（fc 处 ξ 精确）。
  - `method=='rayleigh'`：`w1=2π·f1_factor·fc`、`w2=2π·f2_factor·fc`；`alpha=2*xi*w1*w2/(w1+w2)`、`beta=2*xi/(w1+w2)`（两端 ξ 相等≈恒定 Q）。
- `_resolve_damping(dcfg, fc_est)`：拷贝 dcfg，`fc = dcfg['fc'] or fc_est`，返回解析后的 dict（供建材与 meta 复用）。

### 3) 给材料施加阻尼 —— 改 `_create_band_materials_sections`（L734）
- 签名加 `damping=None`。逐带建完 `Elastic/Density` 后，若 `damping and damping['enable']`：
  - `is_bedrock = (idx == 0)`（`_build_stratigraphy` 保证 strat[0]=基岩）。
  - 算 `(Q, xi)=_damping_ratio_from_q(mat.cs, is_bedrock, damping)`，`(a,b)=_rayleigh_coeffs(xi, damping, damping['fc'])`。
  - `m.Damping(alpha=a, beta=b)`（Abaqus 瑞利阻尼，无需额外 import）。每行加中文注释。

### 4) 转发 damping —— `create_model`(L791) 与 `create_flat_model`(L937)
- 两者签名加 `damping=None`，把它传给各自的 `_create_band_materials_sections(model, strat, damping)`。两模型同参，保证 TAF 分母一致。

### 5) 降数值阻尼 —— `build_models`(L1315)
- `ImplicitDynamicsStep(... application=MODERATE_DISSIPATION)` → `application=TRANSIENT_FIDELITY`（保留 `timeIncrementationMethod=FIXED, initialInc=inc`）。改注释。

### 6) `_load_case_config`(L1413) 支持注入 damping
- 增 `if isinstance(cfg.get('damping_cfg'), dict): damping_cfg=_deep_merge(damping_cfg, cfg['damping_cfg'])`；函数多接收/返回 `damping_cfg`（与现有 material/geometry/mesh 同模式，便于 Autorun 批量调参/关阻尼）。

### 7) `main()`(L1526) 串起来
- `global` 增 `damping_cfg`。
- `_load_case_config` 调用与返回加 `damping_cfg`。
- `acc_info = find_acc_txt(...)` 后：若 `damping_cfg['fc']` 为空，`np.loadtxt(acc_info[0][0])` 估计 `fc`（多记录同 fc 是标准用法；不同 fc 时以首条为准，写注释说明）。
- `damping = _resolve_damping(damping_cfg, fc_est)`。
- `create_model(...)`、`create_flat_model(...)` 传 `damping=damping`。
- `_write_case_meta(...)` 传 `damping`。

### 8) `_write_case_meta`(L1460) 记录阻尼（可复现）
- 加 `damping` 入参；meta 增 `'damping'` 块：method、fc、qs_factor、q_bedrock，及逐层 `[{name, cs, Q, xi, alpha, beta}]`。便于下游核对/[[unified-case-meta-pipeline]] 追溯。

---

## 复用的现有件
- `_build_stratigraphy`(L299) 返回的 strat[0] 恒为基岩 → 直接用 index 判 bedrock。
- `_compute_material_params`/`_compute_elastic_modulus_from_wave_speed` 已有；阻尼只需各带 `mat.cs`。
- `_deep_merge`(L1402) 直接用于注入 damping_cfg。
- 纯 Python 测试桩模式照搬 `test/test_multilayer_engine_smoke.py`(L14-26)。

## 验证
1. **纯 Python 单测**（新增 `Modeling/Multi/test/test_damping_conversion_v1.py`，桩屏蔽 abaqus，`py -3` 跑）：
   - `_estimate_dominant_freq` 对合成 4Hz Ricker 返回≈4.0；
   - stiffness 法：ξ(fc)=1/(2Q) 复算成立；
   - rayleigh 法：在 f1、f2 两端 ξ 均≈1/(2Q)；基岩 Q=999→ξ≈5e-4、α/β≈0；
   - Vs2=800→ξ≈0.0125、软 Vs1=400→0.025、硬 Vs1=1600→0.00625 数值核对。
2. **Abaqus 单工况**（软表层 Vs1/Vs2=0.5、h1/(H-h)=0.75、θs=15°）：跑通建模+求解，看 TAF_h 峰值落到论文图15(b) 量级(~7.6 附近、峰在坡顶后方)；同条件硬表层应显著更低（验证软/硬对比≈4.4–7.4 倍关系）。
3. 检查该工况 `case_meta.json` 写出了 `damping` 块且逐层 ξ 正确。
4. **后处理口径提醒（复现必需，但属另一脚本，本次不改）**：论文 Eq.5 `TAF_v = a^x_v,max / a^ff_h,max`——竖向 TAF 的**分母是自由场水平 PGA**，不是竖向。若后处理用"竖向/竖向"会病态发散（见 [[ml-plan-v3-classical-pivot]]）。复现图15前需确认 Compute_TAF 用水平自由场 PGA 作两个方向的分母、自由场参考=平坦模型(Model-2)地表 PGA。
