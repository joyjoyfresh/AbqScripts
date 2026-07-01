# TSSI Step-2b：框架坐坡顶 + 复用 Multi 引擎的坡面 SSI 集成

TSSI 路线第二步(b)。把 step2a 的平层土块换成真实坡面：**import 复用 Multi
(`Modeling/Multi/VAB_oblique_multilayer_nonlinear_v3.py`) 的已验证波动引擎**
(上土下岩分层 + 粘弹性吸收边界 + 斜入射 SV 等效力)，框架坐【坡顶】并 Tie 耦合。

> 为何 import 而非自包含：重写 1000+ 行已验波动物理风险太高。坡面散射才需要吸收边界
> (Multi 的价值)，平层 SSI(step2a)用自建周期侧边即可。这是有意的例外。

## 文件

| 文件 | 作用 |
|------|------|
| `frame_ssi_slope_v1.py`    | import Multi 建坡面土+边界+斜入射 → 框架坐坡顶 Tie → 建 freefield+ssi 两模型 |
| `postproc_ssi_slope_v1.py` | 读两 ODB：坡顶自由场放大、SSI 周期延长、坡顶结构响应 |

## 运行

```bash
# 工作目录放一条加速度 .txt
abaqus cae noGUI=Modeling/Hybrid/frame_ssi_slope_v1.py          # 建+提交 freefield+ssi（缩小配置各约 5 分钟）
abaqus python Postprocess/Hybrid/postproc_ssi_slope_v1.py       # 后处理
```

集成方式（脚本核心）：
```
import VAB_oblique_multilayer_nonlinear_v3 as multi   # 多路径兜底定位(内核无 __file__)
multi.build_site / make_geometry / _resolve_damping   # 构 Site/Geom/阻尼
multi.create_model(...)         # 建坡面土 Model-1(含 Left/Right/Bottom_boundary + TOP_SURFACE)
mdb.models.changeKey(...)       # Model-1 -> 场景名
add_frame_on_crest(...)         # 框架坐坡顶(右缘贴坡肩 x=left_flat) + Tie 基底到坡顶土面
ImplicitDynamicsStep + multi.VAB_oblique(...)   # 建步 + 粘弹性边界 + 斜入射 SV 等效力
```

## 实跑验证结果（2026-06-30，Abaqus 2021，ricker_4Hz，入射角 15°；缩小坡 H_upper=100/总长300/网格5m）

| 验证项 | 结果 | 物理意义 |
|--------|------|---------|
| 坡顶自由场放大 | 3.89×（坡肩 x=120），主频 3.0Hz | 地形+地层放大 |
| SSI 周期延长 | T 0.500→0.667s，1.33× | 土柔度(同 step2a) |
| 坡顶结构响应 | 基底剪力 4.4e5N，漂移 0.0033，顶层 4.67× | 建筑放大已放大的坡顶运动 |

**贯通研究线**：斜入射 SV → 坡顶地形放大 3.89× → 坡顶建筑顶层 4.67×（开题核心机理"地形放大致坡顶建筑震害加重"端到端量化）。对比 step2a 平层(坡顶 1.83×/顶层 1.68×)，坡顶建筑响应高近 3 倍。

**踩坑**：① assembly 级 `SetFromNodeLabels(instanceName=...)` 关键字错 → 用 `asm.Set(nodes=inst.nodes.sequenceFromLabels([lab]))`；② Multi 内核无 `__file__` → 多路径兜底+绝对路径定位 Multi 目录；③ Multi 固定建 'Model-1' → `changeKey` 改场景名以建多模型。

## 当前版本边界（v1，集成里程碑）

- **缩小配置**(H_upper=100/总长300/网格5m)快速调通集成；论文尺度放大 `soil_geometry_cfg` 即可(运行变慢)。
- **单覆盖层**(velocity_ratio=4, Vs≈500)；多层只需扩 `soil_material_cfg['layers']`(Multi 已支持)。
- **柱脚 Tie 铰接**(tieRotations=OFF)，同 step2a。
- **结构弹性**；CDP 留 step3。

## 下一步

- **step2b-2**：加 `fixed` 去耦对照(刚性基础,输入坡顶自由场运动)——把 step2a 的去耦法套到坡面，
  显式剥离"坡顶 SSI vs 坡顶刚性"。框架基础输入 = freefield 跑出的坡顶地表运动(脚本已输出 CREST_REF)。
- **step3**：梁柱换纤维截面/CDP 非线性。
- **step4**：套 `case_config.json` 做距坡缘 M/T 扫描(框架 x_off 参数化)。
