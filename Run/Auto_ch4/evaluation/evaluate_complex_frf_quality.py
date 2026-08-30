# -*- coding: utf-8 -*-
"""独立评价复频响数据完整性和V001—V004数值敏感性。

本脚本只读取已经生成的 ``surface_results.npz``，不打开ODB、不改写后处理
状态，也不影响批处理是否完成。V003先消除对应归一化地表点因坐标平移引入的
理论水平传播相位，再把幅值、相对相位形态和群时延残差写成数值不确定性诊断。
评价结果写入独立JSON/CSV，供研究者判读；旧参考线仅保留作兼容性记录，不再作为
V002的自动放行条件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from analyze_complex_frf import (
    FREQUENCY_MAX,
    FREQUENCY_MIN,
    FREQUENCY_STEP,
    S_MAX,
    S_MIN,
    S_STEP,
    discover_record,
    group_delay,
    load_case,
    regular_grid,
    segment_labels_from_s,
    spatial_phase_gradient,
    unwrap_frequency_phase,
)


def finite_fraction(values) -> float:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        valid = np.isfinite(array.real) & np.isfinite(array.imag)
    else:
        valid = np.isfinite(array)
    return float(np.mean(valid)) if valid.size else 0.0


def tail_rms_ratio(values, fraction=0.10) -> float:
    array = np.atleast_2d(np.asarray(values, dtype=float))
    count = max(1, int(math.ceil(array.shape[1] * float(fraction))))
    peaks = np.max(np.abs(array), axis=1)
    valid = np.isfinite(peaks) & (peaks > 0.0)
    if not np.any(valid):
        return float("nan")
    ratios = np.sqrt(np.mean(array[valid, -count:] ** 2, axis=1)) / peaks[valid]
    return float(np.percentile(ratios[np.isfinite(ratios)], 95.0))


def _load_case_json(case, filename):
    """读取工况旁车JSON；缺失或非法时返回空字典。"""
    path = Path(case["case_dir"]) / filename
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (IOError, OSError, ValueError):
        return {}


def _load_sgrid_x(case, s_values):
    """把NPZ中的归一化地表子网格横坐标插值到统一s网格。"""
    path = Path(case["case_dir"]) / "surface_results.npz"
    if not path.is_file():
        return None
    package = np.load(path, allow_pickle=False)
    try:
        prefix = "frf_%s_" % case["record"]
        x_key = prefix + "sgrid_x"
        s_key = prefix + "sgrid_s"
        if x_key not in package or s_key not in package:
            return None
        source_s = np.asarray(package[s_key], dtype=float)
        source_x = np.asarray(package[x_key], dtype=float)
        good = np.isfinite(source_s) & np.isfinite(source_x)
        if int(np.sum(good)) < 2:
            return None
        order = np.argsort(source_s[good])
        return np.interp(
            np.asarray(s_values, dtype=float),
            source_s[good][order],
            source_x[good][order],
        )
    finally:
        package.close()


def _phase_origin_x():
    """返回自由场相位原点（建模脚本固定为左边界 x=0）。"""
    return 0.0


def _case_incident_angle_and_bedrock_vs(case, meta):
    """返回相位校正所需的SV入射角和基岩剪切波速。"""
    config = _load_case_json(case, "case_config.json")
    material = config.get("material_cfg") or {}
    bedrock = material.get("bedrock") or {}
    angle = meta.get("incident_angle", material.get("angle"))
    vs = (meta.get("bedrock") or {}).get("cs", bedrock.get("vs"))
    try:
        angle = float(angle)
        vs = float(vs)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(angle) or not np.isfinite(vs) or vs <= 0.0:
        return None, None
    return angle, vs


def coordinate_phase_alignment(reference, candidate, frequency, s_values):
    """消除对应归一化地表点的理论水平传播相位差。

    自由场相位因子为 ``exp(-i*omega*p_x*(x-phase_origin_x))``，其中
    ``p_x=sin(theta_s)/Vs_bedrock``。因此候选工况相对于参考工况的理论
    相位差为 ``-omega*delta_t_x``，校正候选复频响时乘以
    ``exp(+i*omega*delta_t_x)``。一维参考时程的竖向时间平移不在这里重复
    校正，因为每个工况的G_h已经统一使用自己的左侧上平台一维参考。
    """
    reference_meta = _load_case_json(reference, "case_meta.json")
    candidate_meta = _load_case_json(candidate, "case_meta.json")
    reference_x = _load_sgrid_x(reference, s_values)
    candidate_x = _load_sgrid_x(candidate, s_values)
    reference_origin = _phase_origin_x()
    candidate_origin = _phase_origin_x()
    angle, vs = _case_incident_angle_and_bedrock_vs(candidate, candidate_meta)
    info = {
        "applied": False,
        "reason": None,
        "formula": "delta_t_x=(x_candidate-phase_origin_candidate-x_reference+phase_origin_reference)*sin(theta_s)/Vs_bedrock",
        "phase_origin_reference_x": reference_origin,
        "phase_origin_candidate_x": candidate_origin,
        "incident_angle_deg": angle,
        "bedrock_vs_m_per_s": vs,
    }
    if reference_x is None or candidate_x is None:
        info["reason"] = "missing_sgrid_x"
        return candidate, info
    if reference_origin is None or candidate_origin is None:
        info["reason"] = "missing_phase_origin"
        return candidate, info
    if angle is None or vs is None:
        info["reason"] = "missing_incident_angle_or_bedrock_vs"
        return candidate, info

    delta_x = candidate_x - candidate_origin - (reference_x - reference_origin)
    delay = delta_x * math.sin(math.radians(angle)) / vs
    if not np.all(np.isfinite(delay)):
        info["reason"] = "nonfinite_phase_delay"
        return candidate, info
    correction = np.exp(1j * 2.0 * math.pi * frequency[:, None] * delay[None, :])
    aligned = dict(candidate)
    aligned["G"] = candidate["G"] * correction
    aligned["phase"] = unwrap_frequency_phase(aligned["G"], candidate["mask"])
    aligned["group_delay"] = group_delay(
        aligned["phase"], candidate["mask"], frequency
    )
    aligned["spatial_phase_gradient"] = spatial_phase_gradient(
        aligned["G"], candidate["mask"], s_values, segment_labels_from_s(s_values)
    )
    info.update({
        "applied": True,
        "delta_x_m_min": float(np.min(delta_x)),
        "delta_x_m_max": float(np.max(delta_x)),
        "delta_t_x_s_min": float(np.min(delay)),
        "delta_t_x_s_max": float(np.max(delay)),
        "delta_t_x_s_median": float(np.median(delay)),
    })
    reference_y = ((reference_meta.get("ff_theory") or {}).get("left") or {}).get("surface_y")
    candidate_y = ((candidate_meta.get("ff_theory") or {}).get("left") or {}).get("surface_y")
    if reference_y is not None and candidate_y is not None:
        info["delta_y_reference_m"] = float(candidate_y) - float(reference_y)
        info["delta_t_y_reference_s"] = (
            info["delta_y_reference_m"] * math.cos(math.radians(angle)) / vs
        )
    return aligned, info


def inspect_case(case_dir: Path) -> dict:
    result = {
        "case_dir": str(case_dir.resolve()),
        "case_name": case_dir.name,
        "npz_exists": (case_dir / "surface_results.npz").is_file(),
        "xlsx_exists": (case_dir / "surface_results.xlsx").is_file(),
        "reference_exists": bool(list(case_dir.glob("freefield_reference_*.npz"))),
    }
    if not result["npz_exists"]:
        result.update({"data_complete": False, "reason": "missing_surface_results_npz"})
        return result
    package = np.load(case_dir / "surface_results.npz", allow_pickle=False)
    try:
        record = discover_record(package)
        prefix = "frf_%s_" % record
        frequency = np.asarray(package[prefix + "frequency"], dtype=float)
        total = np.asarray(package[prefix + "sgrid_H_surface_h"])
        relative = np.asarray(package[prefix + "sgrid_H_surface_over_1D_h"])
        relative_mask = np.asarray(
            package[prefix + "sgrid_H_surface_over_1D_h_valid_mask"], dtype=bool
        )
        raw = np.asarray(package["raw_%s_acc_h" % record], dtype=float)
        result.update({
            "record": record,
            "frequency_min_hz": float(np.min(frequency)),
            "frequency_max_hz": float(np.max(frequency)),
            "frequency_count": int(len(frequency)),
            "total_frf_finite_fraction": finite_fraction(total),
            "relative_frf_valid_fraction": float(np.mean(relative_mask)),
            "relative_frf_finite_fraction": finite_fraction(relative),
            "surface_tail_p95_ratio": tail_rms_ratio(raw),
        })
        result["data_complete"] = bool(
            result["xlsx_exists"]
            and result["total_frf_finite_fraction"] > 0.95
            and result["relative_frf_valid_fraction"] > 0.95
            and result["frequency_max_hz"] >= 10.0
        )
        if not result["data_complete"]:
            result["reason"] = "relative_frf_or_frequency_band_incomplete"
        return result
    except Exception as exc:
        result.update({"data_complete": False, "reason": str(exc)})
        return result
    finally:
        package.close()


def weighted_complex_error(reference, candidate, mask, frequency_weight) -> float:
    weight = np.asarray(frequency_weight, dtype=float)[:, None]
    valid = (
        mask
        & np.isfinite(reference.real) & np.isfinite(reference.imag)
        & np.isfinite(candidate.real) & np.isfinite(candidate.imag)
        & np.isfinite(weight) & (weight > 0.0)
    )
    selected_weight = np.broadcast_to(weight, valid.shape)[valid]
    denominator = np.sum(selected_weight * np.abs(reference[valid]) ** 2)
    if denominator <= 0.0:
        return float("nan")
    numerator = np.sum(selected_weight * np.abs(candidate[valid] - reference[valid]) ** 2)
    return float(np.sqrt(numerator / denominator))


def circular_phase_rmse_deg(reference, candidate, mask, frequency_weight) -> float:
    frequency_weight = np.asarray(frequency_weight, dtype=float)[:, None]
    valid = (
        mask
        & np.isfinite(reference.real) & np.isfinite(reference.imag)
        & np.isfinite(candidate.real) & np.isfinite(candidate.imag)
        & np.isfinite(frequency_weight) & (frequency_weight > 0.0)
    )
    weight = np.broadcast_to(frequency_weight, valid.shape)[valid]
    denominator = float(np.sum(weight))
    if denominator <= 0.0:
        return float("nan")
    difference = np.angle(candidate[valid] * np.conj(reference[valid]))
    return float(np.degrees(np.sqrt(np.sum(weight * difference ** 2) / denominator)))


def log_amplitude_rmse(reference, candidate, mask) -> float:
    valid = mask & (np.abs(reference) > 1.0e-12) & (np.abs(candidate) > 1.0e-12)
    if not np.any(valid):
        return float("nan")
    difference = np.log(np.abs(candidate[valid])) - np.log(np.abs(reference[valid]))
    return float(np.sqrt(np.mean(difference ** 2)))


def delay_rmse(reference, candidate, mask) -> float:
    valid = mask & np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((candidate[valid] - reference[valid]) ** 2)))


def peak_features(field, mask, frequency, s_values):
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    index = int(np.nanargmax(np.where(mask, np.abs(field), np.nan)))
    f_index, s_index = np.unravel_index(index, field.shape)
    return float(np.abs(field[f_index, s_index])), float(frequency[f_index]), float(s_values[s_index])


def scalar_rmse(reference, candidate, mask) -> float:
    """计算两个实数场在有效掩码内的RMSE。"""
    valid = mask & np.isfinite(reference) & np.isfinite(candidate)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((candidate[valid] - reference[valid]) ** 2)))


def absolute_quantile(values, mask, quantile) -> float:
    """返回有效残差绝对值的分位数。"""
    valid = mask & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.percentile(np.abs(values[valid]), quantile))


def _comparison_metrics(reference, candidate, frequency, s_values, mask, weight):
    """生成一对工况的完整残差指标，不附加放行判据。"""
    mask = reference["mask"] & candidate["mask"]
    ref_peak, ref_frequency, ref_s = peak_features(reference["G"], mask, frequency, s_values)
    cur_peak, cur_frequency, cur_s = peak_features(candidate["G"], mask, frequency, s_values)
    phase_difference = np.angle(candidate["G"] * np.conj(reference["G"]))
    gradient_difference = (
        candidate["spatial_phase_gradient"] - reference["spatial_phase_gradient"]
    )
    delay_difference = candidate["group_delay"] - reference["group_delay"]
    log_amplitude_difference = (
        np.log(np.maximum(np.abs(candidate["G"]), 1.0e-12))
        - np.log(np.maximum(np.abs(reference["G"]), 1.0e-12))
    )
    row = {
        "valid_fraction": float(np.mean(mask)),
        "weighted_complex_error": weighted_complex_error(reference["G"], candidate["G"], mask, weight),
        "log_amplitude_rmse": log_amplitude_rmse(reference["G"], candidate["G"], mask),
        "log_amplitude_median_abs": absolute_quantile(log_amplitude_difference, mask, 50.0),
        "log_amplitude_p90_abs": absolute_quantile(log_amplitude_difference, mask, 90.0),
        "circular_phase_rmse_deg": circular_phase_rmse_deg(reference["G"], candidate["G"], mask, weight),
        "phase_median_abs_deg": np.degrees(absolute_quantile(phase_difference, mask, 50.0)),
        "phase_p90_abs_deg": np.degrees(absolute_quantile(phase_difference, mask, 90.0)),
        "phase_shape_gradient_rmse_rad_per_s": scalar_rmse(
            reference["spatial_phase_gradient"],
            candidate["spatial_phase_gradient"],
            mask,
        ),
        "group_delay_rmse_s": delay_rmse(reference["group_delay"], candidate["group_delay"], mask),
        "group_delay_median_abs_s": absolute_quantile(delay_difference, mask, 50.0),
        "group_delay_p90_abs_s": absolute_quantile(delay_difference, mask, 90.0),
        "peak_amplitude_relative_error": abs(cur_peak / ref_peak - 1.0) if ref_peak > 0.0 else float("nan"),
        "peak_frequency_error_hz": abs(cur_frequency - ref_frequency),
        "peak_s_error": abs(cur_s - ref_s),
    }
    segment_labels = np.full(s_values.shape, "B", dtype="U1")
    segment_labels[s_values <= 0.0] = "A"
    segment_labels[s_values >= 1.0] = "C"
    for segment in ("A", "B", "C"):
        region = mask & (segment_labels[None, :] == segment)
        row["log_amplitude_rmse_%s" % segment] = log_amplitude_rmse(
            reference["G"], candidate["G"], region
        )
        row["log_amplitude_p90_abs_%s" % segment] = absolute_quantile(
            log_amplitude_difference, region, 90.0
        )
        row["circular_phase_rmse_deg_%s" % segment] = circular_phase_rmse_deg(
            reference["G"], candidate["G"], region, weight
        )
        row["phase_p90_abs_deg_%s" % segment] = np.degrees(
            absolute_quantile(phase_difference, region, 90.0)
        )
        row["phase_shape_gradient_rmse_rad_per_s_%s" % segment] = scalar_rmse(
            reference["spatial_phase_gradient"],
            candidate["spatial_phase_gradient"],
            region,
        )
        row["group_delay_p90_abs_s_%s" % segment] = absolute_quantile(
            delay_difference, region, 90.0
        )
    return row


def _peak_at_surface_position(case, frequency, s_values, target_s):
    """提取最接近指定地表坐标的有效幅值峰值与频率。"""
    index = int(np.argmin(np.abs(np.asarray(s_values, dtype=float) - float(target_s))))
    values = np.abs(np.asarray(case["G"], dtype=np.complex128)[:, index])
    valid = (
        np.asarray(case["mask"], dtype=bool)[:, index]
        & np.isfinite(values)
        & np.isfinite(frequency)
    )
    if not np.any(valid):
        return float("nan"), float("nan"), float(s_values[index])
    candidates = np.where(valid, values, -np.inf)
    peak_index = int(np.argmax(candidates))
    return (
        float(values[peak_index]),
        float(frequency[peak_index]),
        float(s_values[index]),
    )


def key_location_metrics(reference, candidate, frequency, s_values):
    """计算坡顶主峰频率和坡面中部峰值变化两项主要判据。"""
    crest_ref_amp, crest_ref_frequency, crest_s = _peak_at_surface_position(
        reference, frequency, s_values, 0.0
    )
    crest_candidate_amp, crest_candidate_frequency, _ = _peak_at_surface_position(
        candidate, frequency, s_values, 0.0
    )
    midslope_ref_amp, midslope_ref_frequency, midslope_s = _peak_at_surface_position(
        reference, frequency, s_values, 0.5
    )
    midslope_candidate_amp, midslope_candidate_frequency, _ = _peak_at_surface_position(
        candidate, frequency, s_values, 0.5
    )
    midslope_change = (
        midslope_candidate_amp / midslope_ref_amp - 1.0
        if np.isfinite(midslope_ref_amp) and midslope_ref_amp > 0.0
        else float("nan")
    )
    crest_frequency_error = abs(crest_candidate_frequency - crest_ref_frequency)
    primary_met = bool(
        np.isfinite(crest_frequency_error)
        and np.isfinite(midslope_change)
        and crest_frequency_error <= 0.20
        and abs(midslope_change) <= 0.05
    )
    return {
        "crest_s": crest_s,
        "crest_peak_amplitude_reference": crest_ref_amp,
        "crest_peak_amplitude_candidate": crest_candidate_amp,
        "crest_peak_frequency_reference_hz": crest_ref_frequency,
        "crest_peak_frequency_candidate_hz": crest_candidate_frequency,
        "crest_peak_frequency_error_hz": crest_frequency_error,
        "midslope_s": midslope_s,
        "midslope_peak_amplitude_reference": midslope_ref_amp,
        "midslope_peak_amplitude_candidate": midslope_candidate_amp,
        "midslope_peak_amplitude_relative_change": midslope_change,
        "midslope_peak_frequency_reference_hz": midslope_ref_frequency,
        "midslope_peak_frequency_candidate_hz": midslope_candidate_frequency,
        "primary_reference_lines_met": primary_met,
    }


COMPARISON_METRIC_FIELDS = (
    "valid_fraction", "weighted_complex_error", "log_amplitude_rmse",
    "log_amplitude_median_abs", "log_amplitude_p90_abs",
    "circular_phase_rmse_deg", "phase_median_abs_deg", "phase_p90_abs_deg",
    "phase_shape_gradient_rmse_rad_per_s", "group_delay_rmse_s",
    "group_delay_median_abs_s", "group_delay_p90_abs_s",
    "peak_amplitude_relative_error", "peak_frequency_error_hz", "peak_s_error",
    "log_amplitude_rmse_A", "log_amplitude_rmse_B", "log_amplitude_rmse_C",
    "log_amplitude_p90_abs_A", "log_amplitude_p90_abs_B", "log_amplitude_p90_abs_C",
    "circular_phase_rmse_deg_A", "circular_phase_rmse_deg_B",
    "circular_phase_rmse_deg_C", "phase_p90_abs_deg_A", "phase_p90_abs_deg_B",
    "phase_p90_abs_deg_C", "phase_shape_gradient_rmse_rad_per_s_A",
    "phase_shape_gradient_rmse_rad_per_s_B", "phase_shape_gradient_rmse_rad_per_s_C",
    "group_delay_p90_abs_s_A", "group_delay_p90_abs_s_B", "group_delay_p90_abs_s_C",
)


def legacy_reference_lines(row, variation_id, tail_ratio=float("nan")):
    """保留旧参考线结果，仅用于历史文件兼容和人工回溯。"""
    if variation_id == "V001":
        reference_lines = (
            row["log_amplitude_rmse"] <= 0.05,
            row["peak_amplitude_relative_error"] <= 0.05,
            row["peak_frequency_error_hz"] <= 0.10,
            row["circular_phase_rmse_deg"] <= 5.0,
            row["group_delay_rmse_s"] <= 0.02,
        )
    elif variation_id == "V002":
        reference_lines = (
            max(row["log_amplitude_rmse_%s" % segment] for segment in ("A", "B", "C")) <= 0.05,
            row["peak_frequency_error_hz"] <= 0.10,
            max(row["circular_phase_rmse_deg_%s" % segment] for segment in ("A", "B", "C")) <= 5.0,
            row["group_delay_rmse_s"] <= 0.02,
        )
    else:
        reference_lines = (
            tail_ratio <= 0.02,
            row["weighted_complex_error"] <= 0.05,
            row["circular_phase_rmse_deg"] <= 5.0,
            row["group_delay_rmse_s"] <= 0.02,
        )
    return bool(all(reference_lines))


def compare_cases(
    reference,
    candidate,
    frequency,
    s_values,
    variation_id,
    tail_ratio=float("nan"),
    candidate_for_metrics=None,
    alignment_info=None,
):
    """比较一对工况，V003主指标使用相位校正后的残余变化。"""
    mask = reference["mask"] & candidate["mask"]
    weight = np.minimum(reference["weight"], candidate["weight"])
    raw_metrics = _comparison_metrics(
        reference, candidate, frequency, s_values, mask, weight
    )
    metric_candidate = candidate_for_metrics or candidate
    metrics = _comparison_metrics(
        reference, metric_candidate, frequency, s_values, mask, weight
    )
    row = {
        "reference": reference["case_id"],
        "variation": variation_id,
        "surface_tail_p95_ratio": tail_ratio,
    }
    row.update(metrics)
    row.update(key_location_metrics(reference, metric_candidate, frequency, s_values))
    legacy_pass = legacy_reference_lines(raw_metrics, variation_id, tail_ratio)
    row["legacy_reference_lines_met"] = legacy_pass
    if variation_id == "V003":
        row["comparison_basis"] = (
            "phase_aligned_residual"
            if alignment_info and alignment_info.get("applied")
            else "raw_unaligned"
        )
        row["phase_alignment_applied"] = bool(
            alignment_info and alignment_info.get("applied")
        )
        row["reference_lines_met"] = None
        for field in COMPARISON_METRIC_FIELDS:
            row["raw_" + field] = raw_metrics.get(field)
        if alignment_info:
            for key in (
                "delta_x_m_min", "delta_x_m_max", "delta_t_x_s_min",
                "delta_t_x_s_max", "delta_t_x_s_median",
                "delta_y_reference_m", "delta_t_y_reference_s",
            ):
                if key in alignment_info:
                    row[key] = alignment_info[key]
    else:
        row["comparison_basis"] = "unadjusted"
        row["phase_alignment_applied"] = False
        row["reference_lines_met"] = legacy_pass
    return row


def build_v003_uncertainty(row, alignment_info):
    """生成V003相位校正后的数值不确定性摘要。"""
    residual = {field: row.get(field) for field in COMPARISON_METRIC_FIELDS}
    raw = {field: row.get("raw_" + field) for field in COMPARISON_METRIC_FIELDS}
    return {
        "schema_version": 1,
        "reference": row["reference"],
        "candidate": row["variation"],
        "scope": "P061与V003侧向净空单因素工况的域不确定性诊断",
        "comparison_basis": row.get("comparison_basis"),
        "phase_alignment": alignment_info or {},
        "residual_metrics": residual,
        "raw_metrics": raw,
        "conclusion_stability": {
            "status": "not_assessable_from_single_stress_case",
            "reason": "单个P061压力工况不能决定全体H/P参数排序；需在生产数据完成后用本残差包络复核主要结论。",
            "rule": "只有当主要效应的方向、排序和论文结论在加入该数值不确定性后仍保持时，才保留强结论；重叠时改写为不可区分或需谨慎。",
        },
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="独立复频响数据与数值敏感性评价")
    parser.add_argument("--root", type=Path, default=repo_root / "Run/ch4_sp_01_V")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or (args.root / "evaluation")
    output.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(path for path in args.root.glob("case-*") if path.is_dir())
    inspections = [inspect_case(path) for path in case_dirs]
    by_suffix = {}
    for path in case_dirs:
        for suffix in ("P061", "V001", "V002", "V003", "V004"):
            if path.name.endswith(suffix):
                by_suffix[suffix] = path
    frequency = regular_grid(FREQUENCY_MIN, FREQUENCY_MAX, FREQUENCY_STEP)
    s_values = regular_grid(S_MIN, S_MAX, S_STEP)
    loaded = {}
    load_errors = {}
    for suffix, path in by_suffix.items():
        try:
            loaded[suffix] = load_case(path, frequency, s_values)
        except Exception as exc:
            load_errors[suffix] = str(exc)
    comparisons = []
    v003_uncertainty = None
    inspection_by_suffix = {}
    for suffix in by_suffix:
        for item in inspections:
            if item["case_name"].endswith(suffix):
                inspection_by_suffix[suffix] = item
                break
    if "P061" in loaded:
        for suffix in ("V001", "V002", "V003", "V004"):
            if suffix in loaded:
                tail_ratio = float(inspection_by_suffix.get(suffix, {}).get(
                    "surface_tail_p95_ratio", float("nan")
                ))
                alignment_info = None
                candidate_for_metrics = None
                if suffix == "V003":
                    candidate_for_metrics, alignment_info = coordinate_phase_alignment(
                        loaded["P061"], loaded[suffix], frequency, s_values
                    )
                comparison = compare_cases(
                    loaded["P061"],
                    loaded[suffix],
                    frequency,
                    s_values,
                    suffix,
                    tail_ratio,
                    candidate_for_metrics=candidate_for_metrics,
                    alignment_info=alignment_info,
                )
                comparisons.append(comparison)
                if suffix == "V003":
                    v003_uncertainty = build_v003_uncertainty(
                        comparison, alignment_info
                    )
    write_csv(output / "case_data_completeness.csv", inspections)
    write_csv(output / "validation_comparison.csv", comparisons)
    all_data_complete = bool(
        inspections
        and all(item.get("data_complete", False) for item in inspections)
        and len(comparisons) == 4
    )
    all_legacy_reference_lines_met = bool(
        all_data_complete
        and all(item.get("legacy_reference_lines_met", False) for item in comparisons)
    )
    all_primary_reference_lines_met = bool(
        all_data_complete
        and all(item.get("primary_reference_lines_met", False) for item in comparisons)
    )
    payload = {
        "all_data_complete": all_data_complete,
        "all_legacy_reference_lines_met": all_legacy_reference_lines_met,
        "all_primary_reference_lines_met": all_primary_reference_lines_met,
        "all_reference_lines_met": all_primary_reference_lines_met,
        "case_data": inspections,
        "validation_comparisons": comparisons,
        "comparison_load_errors": load_errors,
        "v003_uncertainty_available": bool(v003_uncertainty),
        "note": "本评价独立于后处理执行状态；全地表统一使用左侧上平台一维自由场，主要判据为坡顶主峰频率误差不超过0.2 Hz且坡面中部峰值变化不超过5%；V003复场指标先消除理论横向坐标相位，旧参考线字段仅作兼容性记录，也不改写postprocess_status.json",
    }
    with (output / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    if v003_uncertainty is not None:
        with (output / "v003_domain_uncertainty.json").open("w", encoding="utf-8") as handle:
            json.dump(v003_uncertainty, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(
        "评价完成：数据完整=%s，主要判据=%s，旧参考线结果=%s，V003残余不确定性=%s，输出=%s"
        % (
            all_data_complete,
            all_primary_reference_lines_met,
            all_legacy_reference_lines_met,
            bool(v003_uncertainty),
            output,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
