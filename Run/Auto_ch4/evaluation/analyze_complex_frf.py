# -*- coding: utf-8 -*-
"""小论文相位对齐总波场响应分析与机器学习数据集生成。

程序读取各工况 ``surface_results.npz``，将总波场相对统一左侧自由场的复谱比
扣除斜入射平面波的确定性水平传播相位，再统一到 ``0.5:0.1:10 Hz``、
``s=-4:0.05:4`` 网格，输出幅值、展开相位、群时延、空间相位梯度、
层状/均质坡复数修正量和逐工况统计表。

运行示例：
    python Run/evaluation/analyze_complex_frf.py
    python Run/evaluation/analyze_complex_frf.py --figures representative

固定左侧自由场复谱比仅作为计算中间量；研究定义中的 ``G_h`` 为其乘以
``exp(+i*2*pi*f*p*(x-x_ref))`` 后的相位对齐总波场响应。参考字段无有效值时
工况会被明确跳过，不会用入射波传函或端点传函冒充 ``G_h``。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np


warnings.filterwarnings(
    "ignore",
    message=r"Reading `.npy` or `.npz` file required additional header parsing.*",
    category=UserWarning,
)  # Abaqus/Python 2 生成的规范 NPZ 在新版 NumPy 中仅有性能提示


FREQUENCY_MIN = 0.5
FREQUENCY_MAX = 10.0
FREQUENCY_STEP = 0.1
S_MIN = -4.0
S_MAX = 4.0
S_STEP = 0.05
AMPLITUDE_EPS = 1.0e-12


def regular_grid(start: float, stop: float, step: float) -> np.ndarray:
    """生成包含终点且不受浮点累计误差影响的等距网格。"""
    count = int(round((stop - start) / step)) + 1
    return np.linspace(start, stop, count, dtype=float)


def scalar_text(value) -> str:
    """把NPZ字节标量或字符串标量统一转为文本。"""
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def case_identifier(case_dir: Path) -> str:
    """从 ``case-009-P001`` 提取 ``P001``，非标准目录保留原名。"""
    match = re.search(r"(?:^|-)\b([HPBCV]\d{3})$", case_dir.name)
    return match.group(1) if match else case_dir.name


def case_group(case_id: str) -> str:
    match = re.match(r"([A-Za-z]+)", case_id)
    return match.group(1).upper() if match else "UNKNOWN"


def load_case_config(case_dir: Path, package) -> dict:
    path = case_dir / "case_config.json"
    if path.is_file():
        return read_json(path)
    if "case_config_json" in package:
        return json.loads(scalar_text(package["case_config_json"]))
    raise ValueError("缺少 case_config.json/case_config_json")


def physical_parameters(config: dict) -> tuple[float, float, float]:
    """返回坡角、总覆盖层厚度比和最上覆盖层/基岩波速比。"""
    geometry = config.get("geometry_cfg") or {}
    material = config.get("material_cfg") or {}
    bedrock = material.get("bedrock") or {}
    layers = material.get("layers") or []
    slope_angle = float(geometry["slope_angle"])
    slope_height = float(geometry["slope_height"])
    if not layers:
        return slope_angle, 0.0, 1.0
    thickness_ratio = sum(float(layer.get("thickness", 0.0)) for layer in layers) / slope_height
    velocity_ratio = float(layers[0]["vs"]) / float(bedrock["vs"])
    return slope_angle, thickness_ratio, velocity_ratio


def horizontal_phase_parameters(config: dict) -> dict[str, float]:
    """返回斜入射水平相位修正及 ``s-x`` 映射所需参数。"""
    material = config.get("material_cfg") or {}
    bedrock = material.get("bedrock") or {}
    geometry = config.get("geometry_cfg") or {}
    angle_deg = float(material["angle"])
    bedrock_vs = float(bedrock["vs"])
    if bedrock_vs <= 0.0:
        raise ValueError("基岩剪切波速必须为正数")
    freefield = config.get("freefield_cfg") or {}
    raw_x_ref = freefield.get("phase_origin_x", 0.0)
    if isinstance(raw_x_ref, str) and raw_x_ref.lower() == "center":
        total_length = geometry.get("total_L")
        if total_length is None:
            raise ValueError("phase_origin_x=center，但工况配置缺少模型总长度")
        x_ref = 0.5 * float(total_length)
    else:
        x_ref = float(raw_x_ref or 0.0)
    return {
        "incident_angle_deg": angle_deg,
        "bedrock_vs_m_s": bedrock_vs,
        "horizontal_slowness_s_m": math.sin(math.radians(angle_deg)) / bedrock_vs,
        "phase_origin_x_m": x_ref,
        "slope_height_m": float(geometry["slope_height"]),
        "crest_window_h": float(geometry.get("crest_window", 4.0)),
        "side_clearance_h": float(geometry.get("side_clearance", 1.0)),
    }


def remove_incident_horizontal_phase(field, frequency, x_values, parameters):
    """对节点×频率复场乘单位模相位因子，扣除斜入射水平传播相位。"""
    values = np.asarray(field, dtype=np.complex128)
    frequency = np.asarray(frequency, dtype=float).reshape(-1)
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    if values.shape != (x_values.size, frequency.size):
        raise ValueError("水平传播相位修正的复场、坐标和频率维度不一致")
    p_horiz = float(parameters["horizontal_slowness_s_m"])
    x_ref = float(parameters["phase_origin_x_m"])
    correction = np.exp(
        1j * 2.0 * math.pi *
        (x_values - x_ref)[:, None] * p_horiz * frequency[None, :]
    )
    return values * correction


def discover_record(package, requested: str | None = None) -> str:
    records = []
    for key in package.files:
        match = re.match(r"^frf_(.+)_frequency$", key)
        if match:
            records.append(match.group(1))
    records = sorted(set(records))
    if requested:
        if requested not in records:
            raise ValueError("指定记录 %s 不在NPZ中，可用记录=%s" % (requested, records))
        return requested
    if len(records) != 1:
        raise ValueError("无法唯一确定频响记录，可用记录=%s；请用 --record 指定" % records)
    return records[0]


def segment_labels_from_s(s_values: np.ndarray) -> np.ndarray:
    labels = np.full(s_values.shape, "B", dtype="U1")
    labels[s_values <= 0.0] = "A"
    labels[s_values > 1.0 + 1e-9] = "C"
    return labels


def decode_segments(values: np.ndarray, s_values: np.ndarray) -> np.ndarray:
    if values is None:
        return segment_labels_from_s(s_values)
    array = np.asarray(values)
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    return array.astype("U1")


def _interp_frequency(freq_src, values, valid, freq_target):
    good = valid & np.isfinite(values.real) & np.isfinite(values.imag)
    if int(np.sum(good)) < 2:
        return np.full(freq_target.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    order = np.argsort(freq_src[good])
    x = freq_src[good][order]
    y = values[good][order]
    inside = (freq_target >= x[0]) & (freq_target <= x[-1])
    out = np.full(freq_target.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    out[inside] = np.interp(freq_target[inside], x, y.real) + 1j * np.interp(
        freq_target[inside], x, y.imag
    )
    return out


def interpolate_complex_field(
    freq_src: np.ndarray,
    s_src: np.ndarray,
    values_src: np.ndarray,
    valid_src: np.ndarray,
    segments_src: np.ndarray,
    freq_target: np.ndarray,
    s_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """先沿频率、再沿空间连续插值复数场。"""
    values_src = np.asarray(values_src, dtype=np.complex128)
    valid_src = np.asarray(valid_src, dtype=bool)
    if values_src.shape != valid_src.shape:
        raise ValueError("复数场与有效掩码形状不一致: %s vs %s" % (values_src.shape, valid_src.shape))
    if values_src.shape != (len(s_src), len(freq_src)):
        raise ValueError("复数场必须为 空间×频率，当前=%s" % (values_src.shape,))
    frequency_aligned = np.vstack(
        [_interp_frequency(freq_src, values_src[index], valid_src[index], freq_target)
         for index in range(len(s_src))]
    )
    out = np.full((len(freq_target), len(s_target)), np.nan + 1j * np.nan, dtype=np.complex128)
    for f_index in range(len(freq_target)):
        row = frequency_aligned[:, f_index]
        good = np.isfinite(row.real) & np.isfinite(row.imag)
        if int(np.sum(good)) < 2:
            continue
        out[f_index, :] = np.interp(
            s_target, s_src[good], row.real[good]
        ) + 1j * np.interp(s_target, s_src[good], row.imag[good])
    mask = np.isfinite(out.real) & np.isfinite(out.imag)
    return out, mask


def recover_left_reference_transfer(
    total_field: np.ndarray,
    same_side_field: np.ndarray,
    total_mask: np.ndarray,
    same_side_mask: np.ndarray,
    s_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """从上平台数据恢复左侧一维自由场相对入射波的复传递函数。

    已有规范包仅保存 ``A_2D/A_in`` 与分段的 ``A_2D/A_1D_same_side``。
    在 ``s<=0`` 的上平台，两者之比严格等于 ``A_1D_left/A_in``；逐频率
    对所有有效上平台点取复数分量中位数，可兼容旧数据而无需重跑 Abaqus。
    """
    total_field = np.asarray(total_field, dtype=np.complex128)
    same_side_field = np.asarray(same_side_field, dtype=np.complex128)
    total_mask = np.asarray(total_mask, dtype=bool)
    same_side_mask = np.asarray(same_side_mask, dtype=bool)
    s_values = np.asarray(s_values, dtype=float)
    if total_field.shape != same_side_field.shape:
        raise ValueError("入射参考场与同侧一维参考场形状不一致")
    if total_field.shape != total_mask.shape or same_side_field.shape != same_side_mask.shape:
        raise ValueError("复数场与有效掩码形状不一致")
    if total_field.shape[0] != len(s_values):
        raise ValueError("复数场空间维与s坐标长度不一致")

    upper = s_values <= 1.0e-9
    valid_ratio = (
        upper[:, None]
        & total_mask
        & same_side_mask
        & np.isfinite(total_field.real)
        & np.isfinite(total_field.imag)
        & np.isfinite(same_side_field.real)
        & np.isfinite(same_side_field.imag)
        & (np.abs(same_side_field) > 1.0e-12)
    )
    ratio = np.full(total_field.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ratio[valid_ratio] = total_field[valid_ratio] / same_side_field[valid_ratio]
    reference = np.full(total_field.shape[1], np.nan + 1j * np.nan, dtype=np.complex128)
    reference_mask = np.zeros(total_field.shape[1], dtype=bool)
    for f_index in range(total_field.shape[1]):
        values = ratio[valid_ratio[:, f_index], f_index]
        if len(values) == 0:
            continue
        value = complex(float(np.median(values.real)), float(np.median(values.imag)))
        if np.isfinite(value.real) and np.isfinite(value.imag) and abs(value) > 1.0e-12:
            reference[f_index] = value
            reference_mask[f_index] = True
    return reference, reference_mask


def contiguous_runs(mask: np.ndarray):
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return
    split_at = np.where(np.diff(indices) > 1)[0] + 1
    for run in np.split(indices, split_at):
        if len(run):
            yield run


def unwrap_frequency_phase(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    phase = np.full(field.shape, np.nan, dtype=float)
    wrapped = np.angle(field)
    for s_index in range(field.shape[1]):
        for run in contiguous_runs(mask[:, s_index]):
            phase[run, s_index] = np.unwrap(wrapped[run, s_index])
    return phase


def smooth_series(values: np.ndarray) -> np.ndarray:
    """短窗二次多项式平滑；SciPy不可用时退化为五点滑动平均。"""
    if len(values) < 5:
        return values.copy()
    window = min(9, len(values) if len(values) % 2 == 1 else len(values) - 1)
    if window < 5:
        return values.copy()
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(values, window_length=window, polyorder=2, mode="interp")
    except Exception:
        kernel = np.ones(5, dtype=float) / 5.0
        padded = np.pad(values, (2, 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid")


def group_delay(phase: np.ndarray, mask: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    result = np.full(phase.shape, np.nan, dtype=float)
    for s_index in range(phase.shape[1]):
        for run in contiguous_runs(mask[:, s_index] & np.isfinite(phase[:, s_index])):
            if len(run) < 3:
                continue
            smoothed = smooth_series(phase[run, s_index])
            result[run, s_index] = -np.gradient(smoothed, frequency[run]) / (2.0 * math.pi)
            result[run[[0, -1]], s_index] = np.nan
    return result


def spatial_phase_gradient(
    field: np.ndarray, mask: np.ndarray, s_values: np.ndarray, segments: np.ndarray
) -> np.ndarray:
    result = np.full(field.shape, np.nan, dtype=float)
    wrapped = np.angle(field)
    for f_index in range(field.shape[0]):
        for segment in ("A", "B", "C"):
            allowed = (segments == segment) & mask[f_index]
            for run in contiguous_runs(allowed):
                if len(run) < 3:
                    continue
                unwrapped = np.unwrap(wrapped[f_index, run])
                smoothed = smooth_series(unwrapped)
                result[f_index, run] = np.gradient(smoothed, s_values[run])
                result[f_index, run[[0, -1]]] = np.nan
    return result


def interpolate_weight(freq_src, spectrum, freq_target):
    weight = np.abs(np.asarray(spectrum, dtype=np.complex128)) ** 2
    good = np.isfinite(weight) & np.isfinite(freq_src)
    if int(np.sum(good)) < 2 or float(np.max(weight[good])) <= 0.0:
        return np.ones(freq_target.shape, dtype=float)
    result = np.interp(freq_target, freq_src[good], weight[good], left=0.0, right=0.0)
    maximum = float(np.max(result))
    return result / maximum if maximum > 0.0 else np.ones(freq_target.shape, dtype=float)


def correlation(x, y) -> float:
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3 or np.std(x[good]) <= 0.0 or np.std(y[good]) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x[good], y[good])[0, 1])


def case_metrics(case, frequency, s_values) -> dict:
    mask = case["mask"]
    amplitude = case["amplitude"]
    if not np.any(mask):
        raise ValueError("统一网格没有有效总波场响应")
    peak_flat = int(np.nanargmax(np.where(mask, amplitude, np.nan)))
    f_index, s_index = np.unravel_index(peak_flat, amplitude.shape)
    log_amplitude = case["log_amplitude"]
    phase = case["phase"]
    delay = case["group_delay"]
    gradient = case["spatial_phase_gradient"]
    near_peak = slice(max(0, f_index - 1), min(len(frequency), f_index + 2))
    peak_phase_values = phase[near_peak, s_index]
    peak_phase_values = peak_phase_values[np.isfinite(peak_phase_values)]
    phase_rotation = (
        float(np.degrees(np.max(peak_phase_values) - np.min(peak_phase_values)))
        if len(peak_phase_values) >= 2 else float("nan")
    )
    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "slope_angle_deg": case["X"][0],
        "thickness_ratio": case["X"][1],
        "velocity_ratio": case["X"][2],
        "valid_fraction": float(np.mean(mask)),
        "peak_amplitude": float(amplitude[f_index, s_index]),
        "peak_frequency_hz": float(frequency[f_index]),
        "peak_s": float(s_values[s_index]),
        "phase_at_peak_deg": float(np.degrees(case["phase"][f_index, s_index])),
        "group_delay_at_peak_s": float(delay[f_index, s_index]),
        "median_amplitude": float(np.nanmedian(np.where(mask, amplitude, np.nan))),
        "median_abs_phase_deg": float(np.nanmedian(np.abs(np.degrees(phase)))),
        "median_abs_group_delay_s": float(np.nanmedian(np.abs(delay))),
        "median_abs_spatial_phase_gradient_rad_per_s": float(np.nanmedian(np.abs(gradient))),
        "phase_rotation_near_peak_deg": phase_rotation,
        "amplitude_group_delay_correlation": correlation(
            log_amplitude[mask], np.abs(delay[mask])
        ),
    }


def load_case(case_dir, frequency, s_values, requested_record=None):
    npz_path = case_dir / "surface_results.npz"
    package = np.load(npz_path, allow_pickle=False)
    try:
        record = discover_record(package, requested_record)
        prefix = "frf_%s_" % record
        config = load_case_config(case_dir, package)
        source_frequency = np.asarray(package[prefix + "frequency"], dtype=float)
        source_s = np.asarray(package[prefix + "sgrid_s"], dtype=float)
        source_x = np.asarray(package[prefix + "sgrid_x"], dtype=float)
        if source_x.shape != source_s.shape:
            raise ValueError("s网格与物理横坐标维度不一致")
        phase_parameters = horizontal_phase_parameters(config)
        source_segments = decode_segments(
            package[prefix + "sgrid_segment"] if prefix + "sgrid_segment" in package else None,
            source_s,
        )
        total_key = prefix + "sgrid_H_surface_h"
        total_mask_key = total_key + "_valid_mask"
        if total_key not in package or total_mask_key not in package:
            raise ValueError("NPZ缺少基岩参考水平响应字段 %s" % total_key)
        source_total = np.asarray(package[total_key], dtype=np.complex128)
        source_total_mask = np.asarray(package[total_mask_key], dtype=bool)
        total_field, total_mask = interpolate_complex_field(
            source_frequency,
            source_s,
            source_total,
            source_total_mask,
            source_segments,
            frequency,
            s_values,
        )
        left_key = prefix + "sgrid_H_surface_over_1D_left_h"
        left_mask_key = left_key + "_valid_mask"
        if left_key in package and left_mask_key in package:
            source_g_fixed = np.asarray(package[left_key], dtype=np.complex128)
            source_g_fixed_mask = np.asarray(package[left_mask_key], dtype=bool)
            reference_source = "explicit_left_1d_field"
        else:
            same_side_key = prefix + "sgrid_H_surface_over_1D_h"
            same_side_mask_key = same_side_key + "_valid_mask"
            if same_side_key not in package or same_side_mask_key not in package:
                raise ValueError("NPZ缺少统一左参考字段及兼容恢复字段 %s" % same_side_key)
            left_reference, left_reference_mask = recover_left_reference_transfer(
                source_total,
                package[same_side_key],
                source_total_mask,
                package[same_side_mask_key],
                source_s,
            )
            source_g_fixed = np.full(source_total.shape, np.nan + 1j * np.nan, dtype=np.complex128)
            source_g_fixed_mask = source_total_mask & left_reference_mask[None, :]
            safe_reference = np.where(left_reference_mask, left_reference, 1.0 + 0.0j)
            source_g_fixed[source_g_fixed_mask] = (
                source_total / safe_reference[None, :]
            )[source_g_fixed_mask]
            reference_source = "derived_left_1d_from_legacy_fields"
        corrected_key = prefix + "sgrid_H_total_over_freefield_phase_aligned_h"
        corrected_mask_key = corrected_key + "_valid_mask"
        if corrected_key in package and corrected_mask_key in package:
            source_g = np.asarray(package[corrected_key], dtype=np.complex128)
            source_g_mask = np.asarray(package[corrected_mask_key], dtype=bool)
            reference_source += "+explicit_horizontal_phase_removal"
        else:
            source_g = remove_incident_horizontal_phase(
                source_g_fixed, source_frequency, source_x, phase_parameters
            )
            source_g_mask = source_g_fixed_mask.copy()
            reference_source += "+derived_horizontal_phase_removal"
        common_source_mask = source_g_mask & source_g_fixed_mask
        if np.any(common_source_mask):
            magnitude_difference = np.nanmax(
                np.abs(np.abs(source_g[common_source_mask]) - np.abs(source_g_fixed[common_source_mask]))
            )
            magnitude_scale = max(float(np.nanmax(np.abs(source_g_fixed[common_source_mask]))), 1.0)
            if magnitude_difference > 1.0e-10 * magnitude_scale:
                raise ValueError("水平相位修正改变了复响应幅值")
        g_fixed_field, g_fixed_mask = interpolate_complex_field(
            source_frequency,
            source_s,
            source_g_fixed,
            source_g_fixed_mask,
            source_segments,
            frequency,
            s_values,
        )
        physical_x = np.interp(s_values, source_s, source_x)
        g_field = remove_incident_horizontal_phase(
            g_fixed_field.T, frequency, physical_x, phase_parameters
        ).T
        g_mask = g_fixed_mask.copy()
        mask = total_mask & g_mask
        if not np.any(mask):
            raise ValueError("统一左侧自由场参考响应没有有效值；需检查自由场参考")
        g_field[~mask] = np.nan + 1j * np.nan
        g_fixed_field[~mask] = np.nan + 1j * np.nan
        total_field[~mask] = np.nan + 1j * np.nan
        phase = unwrap_frequency_phase(g_field, mask)
        segments = segment_labels_from_s(s_values)
        delay = group_delay(phase, mask, frequency)
        spatial_gradient = spatial_phase_gradient(g_field, mask, s_values, segments)
        amplitude = np.abs(g_field)
        log_amplitude = np.full(amplitude.shape, np.nan, dtype=float)
        log_amplitude[mask] = np.log(np.maximum(amplitude[mask], AMPLITUDE_EPS))
        weight = interpolate_weight(
            source_frequency, package[prefix + "input_spectrum"], frequency
        )
        case_id = case_identifier(case_dir)
        return {
            "case_id": case_id,
            "group": case_group(case_id),
            "record": record,
            "case_dir": str(case_dir.resolve()),
            "X": np.asarray(physical_parameters(config), dtype=float),
            "G": g_field,
            "G_fixed_left_reference": g_fixed_field,
            "H_total": total_field,
            "physical_x": physical_x,
            "horizontal_phase": phase_parameters,
            "mask": mask,
            "weight": weight,
            "amplitude": amplitude,
            "log_amplitude": log_amplitude,
            "phase": phase,
            "group_delay": delay,
            "spatial_phase_gradient": spatial_gradient,
            "reference_source": reference_source,
        }
    finally:
        package.close()


def write_csv(path: Path, rows: list[dict], fieldnames=None):
    if not rows:
        return
    fields = list(fieldnames or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_layer_corrections(cases, frequency, s_values):
    baselines = {}
    for case in cases:
        if case["group"] == "H":
            baselines[round(float(case["X"][0]), 8)] = case
    rows = []
    for case in cases:
        case["C_G"] = np.full(case["G"].shape, np.nan + 1j * np.nan, dtype=np.complex128)
        case["correction_mask"] = np.zeros(case["mask"].shape, dtype=bool)
        if case["group"] != "P":
            continue
        baseline = baselines.get(round(float(case["X"][0]), 8))
        if baseline is None:
            continue
        valid = case["mask"] & baseline["mask"] & (np.abs(baseline["G"]) > AMPLITUDE_EPS)
        case["C_G"][valid] = case["G"][valid] / baseline["G"][valid]
        case["correction_mask"] = valid
        if not np.any(valid):
            continue
        correction_amp = np.abs(case["C_G"])
        delta_amplitude = np.full(valid.shape, np.nan, dtype=float)
        delta_amplitude[valid] = np.log(np.maximum(correction_amp[valid], AMPLITUDE_EPS))
        delta_phase = unwrap_frequency_phase(case["C_G"], valid)
        delta_delay = group_delay(delta_phase, valid, frequency)
        delta_gradient = spatial_phase_gradient(
            case["C_G"], valid, s_values, segment_labels_from_s(s_values)
        )
        index = int(np.nanargmax(np.where(valid, np.abs(delta_amplitude), np.nan)))
        f_index, s_index = np.unravel_index(index, valid.shape)
        rows.append({
            "case_id": case["case_id"],
            "baseline_case_id": baseline["case_id"],
            "slope_angle_deg": case["X"][0],
            "thickness_ratio": case["X"][1],
            "velocity_ratio": case["X"][2],
            "median_delta_log_amplitude": float(np.nanmedian(delta_amplitude)),
            "median_abs_delta_phase_deg": float(np.nanmedian(np.abs(np.degrees(delta_phase)))),
            "median_abs_delta_group_delay_s": float(np.nanmedian(np.abs(delta_delay))),
            "median_abs_delta_spatial_phase_gradient_rad_per_s": float(
                np.nanmedian(np.abs(delta_gradient))
            ),
            "max_abs_delta_log_amplitude": float(np.nanmax(np.abs(delta_amplitude))),
            "max_correction_frequency_hz": float(frequency[f_index]),
            "max_correction_s": float(s_values[s_index]),
        })
    return rows


def parameter_effect_rows(metric_rows):
    rows = []
    paper_rows = [row for row in metric_rows if row["group"] == "P"]
    fields = (
        ("slope_angle_deg", "坡角"),
        ("thickness_ratio", "厚度比"),
        ("velocity_ratio", "波速比"),
    )
    for field, label in fields:
        grouped = defaultdict(list)
        for row in paper_rows:
            grouped[float(row[field])].append(row)
        for value in sorted(grouped):
            group_rows = grouped[value]
            rows.append({
                "parameter": field,
                "parameter_label": label,
                "value": value,
                "n_cases": len(group_rows),
                "median_peak_amplitude": float(np.nanmedian([r["peak_amplitude"] for r in group_rows])),
                "median_peak_frequency_hz": float(np.nanmedian([r["peak_frequency_hz"] for r in group_rows])),
                "median_peak_s": float(np.nanmedian([r["peak_s"] for r in group_rows])),
                "median_abs_group_delay_s": float(np.nanmedian(
                    [r["median_abs_group_delay_s"] for r in group_rows]
                )),
                "median_abs_phase_deg": float(np.nanmedian(
                    [r["median_abs_phase_deg"] for r in group_rows]
                )),
                "median_abs_spatial_phase_gradient_rad_per_s": float(np.nanmedian(
                    [r["median_abs_spatial_phase_gradient_rad_per_s"] for r in group_rows]
                )),
                "median_phase_rotation_near_peak_deg": float(np.nanmedian(
                    [r["phase_rotation_near_peak_deg"] for r in group_rows]
                )),
                "median_amplitude_group_delay_correlation": float(np.nanmedian(
                    [r["amplitude_group_delay_correlation"] for r in group_rows]
                )),
            })
    return rows


def plot_case(case, frequency, s_values, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fields = (
        (case["log_amplitude"], "ln|G_h|", "viridis"),
        (np.degrees(case["phase"]), "展开相位 (deg)", "twilight"),
        (case["group_delay"], "群时延 (s)", "coolwarm"),
    )
    for axis, (values, title, cmap) in zip(axes, fields):
        image = axis.pcolormesh(s_values, frequency, values, shading="auto", cmap=cmap)
        axis.axvline(0.0, color="k", linewidth=0.6)
        axis.axvline(1.0, color="k", linewidth=0.6)
        axis.set_xlabel("s")
        axis.set_ylabel("频率 (Hz)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.85)
    figure.suptitle(case["case_id"])
    figure.savefig(output_dir / (case["case_id"] + "_amplitude_phase.png"), dpi=180)
    plt.close(figure)


def save_dataset(path, cases, frequency, s_values, metadata):
    shape = (len(cases), len(frequency), len(s_values))
    corrections = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)
    correction_mask = np.zeros(shape, dtype=bool)
    for index, case in enumerate(cases):
        corrections[index] = case.get("C_G", corrections[index])
        correction_mask[index] = case.get("correction_mask", correction_mask[index])
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int32),
        case_ids=np.asarray([case["case_id"] for case in cases], dtype="U32"),
        case_groups=np.asarray([case["group"] for case in cases], dtype="U16"),
        case_dirs=np.asarray([case["case_dir"] for case in cases], dtype="U512"),
        records=np.asarray([case["record"] for case in cases], dtype="U128"),
        X=np.vstack([case["X"] for case in cases]),
        feature_names=np.asarray(["slope_angle_deg", "thickness_ratio", "velocity_ratio"], dtype="U32"),
        frequency_hz=frequency,
        s=s_values,
        segments=segment_labels_from_s(s_values),
        G_h=np.stack([case["G"] for case in cases]),
        G_h_fixed_left_reference=np.stack([case["G_fixed_left_reference"] for case in cases]),
        H_total=np.stack([case["H_total"] for case in cases]),
        physical_x_m=np.stack([case["physical_x"] for case in cases]),
        incident_angle_deg=np.asarray([
            case["horizontal_phase"]["incident_angle_deg"] for case in cases
        ], dtype=float),
        bedrock_vs_m_s=np.asarray([
            case["horizontal_phase"]["bedrock_vs_m_s"] for case in cases
        ], dtype=float),
        horizontal_slowness_s_m=np.asarray([
            case["horizontal_phase"]["horizontal_slowness_s_m"] for case in cases
        ], dtype=float),
        phase_origin_x_m=np.asarray([
            case["horizontal_phase"]["phase_origin_x_m"] for case in cases
        ], dtype=float),
        slope_height_m=np.asarray([
            case["horizontal_phase"]["slope_height_m"] for case in cases
        ], dtype=float),
        crest_window_h=np.asarray([
            case["horizontal_phase"]["crest_window_h"] for case in cases
        ], dtype=float),
        side_clearance_h=np.asarray([
            case["horizontal_phase"]["side_clearance_h"] for case in cases
        ], dtype=float),
        valid_mask=np.stack([case["mask"] for case in cases]),
        input_weight=np.stack([case["weight"] for case in cases]),
        amplitude=np.stack([case["amplitude"] for case in cases]),
        log_amplitude=np.stack([case["log_amplitude"] for case in cases]),
        phase_unwrapped_rad=np.stack([case["phase"] for case in cases]),
        group_delay_s=np.stack([case["group_delay"] for case in cases]),
        spatial_phase_gradient_rad_per_s=np.stack(
            [case["spatial_phase_gradient"] for case in cases]
        ),
        layer_correction_C_G=corrections,
        layer_correction_valid_mask=correction_mask,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )


def discover_case_dirs(roots):
    result = []
    for root in roots:
        if not root.exists():
            continue
        if (root / "surface_results.npz").is_file():
            result.append(root)
            continue
        result.extend(sorted(path for path in root.glob("case-*") if (path / "surface_results.npz").is_file()))
    return sorted(set(path.resolve() for path in result), key=lambda path: path.name)


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="相位对齐总波场响应幅值—相位联合分析")
    parser.add_argument(
        "--input-roots",
        nargs="+",
        type=Path,
        default=[repo_root / "Run/ch4_sp_02_H", repo_root / "Run/ch4_sp_03_P", repo_root / "Run/ch4_sp_04_B"],
        help="包含case-*目录的批次根目录",
    )
    parser.add_argument("--output", type=Path, default=repo_root / "Run/ch4_sp_analysis")
    parser.add_argument("--record", default=None, help="NPZ含多条记录时指定记录名")
    parser.add_argument("--figures", choices=("none", "representative", "all"), default="representative")
    parser.add_argument("--strict", action="store_true", help="任一工况无效时立即失败")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frequency = regular_grid(FREQUENCY_MIN, FREQUENCY_MAX, FREQUENCY_STEP)
    s_values = regular_grid(S_MIN, S_MAX, S_STEP)
    case_dirs = discover_case_dirs(args.input_roots)
    if not case_dirs:
        raise SystemExit("未发现含surface_results.npz的工况目录")
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    skipped = []
    for case_dir in case_dirs:
        try:
            cases.append(load_case(case_dir, frequency, s_values, args.record))
            print("[读取] %s" % case_dir.name)
        except Exception as exc:
            skipped.append({"case_dir": str(case_dir), "error": str(exc)})
            print("[跳过] %s: %s" % (case_dir.name, exc), file=sys.stderr)
            if args.strict:
                raise
    if not cases:
        raise SystemExit("没有可用于联合分析的有效总波场响应工况")
    cases.sort(key=lambda item: item["case_id"])
    metric_rows = [case_metrics(case, frequency, s_values) for case in cases]
    correction_rows = build_layer_corrections(cases, frequency, s_values)
    effect_rows = parameter_effect_rows(metric_rows)
    write_csv(args.output / "case_metrics.csv", metric_rows)
    write_csv(args.output / "layer_correction_metrics.csv", correction_rows)
    write_csv(args.output / "parameter_effects.csv", effect_rows)
    metadata = {
        "definition": "G_h=(A_total/A_ff_left)*exp(+i*2*pi*f*p*(x-x_ref))",
        "response_name": "total wavefield response after removing oblique-incidence horizontal propagation phase",
        "reference_scope": "left upper-platform free field with local horizontal propagation phase",
        "horizontal_phase_removal": "p=sin(theta)/Vs_bedrock; unit-modulus correction preserves amplitude",
        "reference_sources": sorted(set(case["reference_source"] for case in cases)),
        "frequency_grid_hz": [FREQUENCY_MIN, FREQUENCY_MAX, FREQUENCY_STEP],
        "s_grid": [S_MIN, S_MAX, S_STEP],
        "phase": "frequency-unwrapped only on contiguous valid runs",
        "group_delay": "-d(phi)/df/(2*pi), smoothed within contiguous valid runs",
        "spatial_phase_gradient": "segment-wise d(phi)/ds; corners are not differentiated across",
        "loaded_case_count": len(cases),
        "skipped_case_count": len(skipped),
    }
    save_dataset(args.output / "complex_frf_dataset.npz", cases, frequency, s_values, metadata)
    if args.figures != "none":
        figure_dir = args.output / "figures"
        figure_dir.mkdir(exist_ok=True)
        selected = cases
        if args.figures == "representative" and len(cases) > 3:
            selected = [cases[0], cases[len(cases) // 2], cases[-1]]
        for case in selected:
            plot_case(case, frequency, s_values, figure_dir)
    status = {
        "status": "completed",
        "loaded_cases": [case["case_id"] for case in cases],
        "skipped_cases": skipped,
        "dataset": str((args.output / "complex_frf_dataset.npz").resolve()),
    }
    with (args.output / "analysis_status.json").open("w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    print("完成：%d个工况，跳过%d个，输出=%s" % (len(cases), len(skipped), args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
