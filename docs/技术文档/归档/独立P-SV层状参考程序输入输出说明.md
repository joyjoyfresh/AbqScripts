# 独立 P–SV 层状参考程序输入输出说明

## 用途

`Modeling/Hybrid/reference_layered_psv_v1.py` 是第三章 F0-2 的独立频域参考程序。它不导入 Abaqus，不调用 `slope_frame_ssi_full_v2.py` 的自由场函数，也不参与有限元求解。程序以半空间中的单位上行 SV 波为输入，在有限层中传播 P/SV 上、下行波，并在最上方施加自由表面剪应力和法向应力均为零的条件。

该程序目前用于均质半空间、同材分层退化、成层场地传递函数和斜入射 P–SV 牵引条件检查。它提供的是独立参考解，不应被表述为现场验证或工程设计系数。

## 输入协议

```python
halfspace = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.3}
layers_top_down = [
    {'vs': 400.0, 'rho': 2000.0, 'nu': 0.3, 'thickness': 40.0},
]
result = surface_response(
    freq_hz=4.0,
    layers_top_down=layers_top_down,
    halfspace=halfspace,
    incident_angle_deg=15.0,
)
```

有限层按从上到下排列，`thickness` 必须为正；半空间不设置厚度。入射角以竖直方向为零，正负角由调用方分别登记。

## 输出协议

单频输出 `surface_response()` 包含：

| 字段 | 含义 |
| --- | --- |
| `ux/uy` | 自由表面水平/竖向复位移响应（相对于单位上行 SV 位移幅值） |
| `traction_residual` | 自由表面两类牵引的归一化残差，必须作为硬门槛检查 |
| `incident_p` | 水平慢度 |
| `reflected_p/reflected_sv` | 半空间底部反射 P/SV 复振幅 |

`transfer_function()` 返回多个频率的 `ux`、`uy` 和 `traction_residual` 数组。F0-2 Autorun 将 4 类剖面、3 个入射角和 3 个频率共 36 个记录直接写入 `reference_transfer.npz`，并同时写出 `reference_manifest.json` 与 `f0_2_validation_report.json`。不建立长期 CSV 中转链。

## F0-2 固定门槛

1. 所有复响应和牵引残差必须为有限值。
2. 所有频率的自由表面牵引残差不超过 (10^{-10})。
3. 均质半空间、垂直入射满足 (|u_x|V_s=2) 的自由表面退化关系。
4. 独立单元测试必须通过同材有限层退化和垂直入射 SH 递推幅值对拍；同材有限层允许出现传播相位，但幅值应保持一致。

这些门槛只证明参考程序自身的解析实现和输出协议可复现；后续 V2/V3 仍需把它与 Abaqus ODB 的时程、频谱幅值和相位逐工况对比。
