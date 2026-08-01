# -*- coding: utf-8 -*-
"""独立评价复频响数据完整性和V001—V003数值敏感性。

本脚本只读取已经生成的 ``surface_results.npz``，不打开ODB、不改写后处理
状态，也不影响批处理是否完成。评价结果写入独立JSON/CSV，供研究者判读。
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
    load_case,
    regular_grid,
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


def compare_cases(reference, candidate, frequency, s_values, variation_id, tail_ratio=float("nan")):
    mask = reference["mask"] & candidate["mask"]
    weight = np.minimum(reference["weight"], candidate["weight"])
    ref_peak, ref_frequency, ref_s = peak_features(reference["G"], mask, frequency, s_values)
    cur_peak, cur_frequency, cur_s = peak_features(candidate["G"], mask, frequency, s_values)
    row = {
        "reference": reference["case_id"],
        "variation": variation_id,
        "valid_fraction": float(np.mean(mask)),
        "weighted_complex_error": weighted_complex_error(reference["G"], candidate["G"], mask, weight),
        "log_amplitude_rmse": log_amplitude_rmse(reference["G"], candidate["G"], mask),
        "circular_phase_rmse_deg": circular_phase_rmse_deg(reference["G"], candidate["G"], mask, weight),
        "group_delay_rmse_s": delay_rmse(reference["group_delay"], candidate["group_delay"], mask),
        "peak_amplitude_relative_error": abs(cur_peak / ref_peak - 1.0) if ref_peak > 0.0 else float("nan"),
        "peak_frequency_error_hz": abs(cur_frequency - ref_frequency),
        "peak_s_error": abs(cur_s - ref_s),
        "surface_tail_p95_ratio": tail_ratio,
    }
    segment_labels = np.full(s_values.shape, "B", dtype="U1")
    segment_labels[s_values <= 0.0] = "A"
    segment_labels[s_values >= 1.0] = "C"
    for segment in ("A", "B", "C"):
        region = mask & (segment_labels[None, :] == segment)
        row["log_amplitude_rmse_%s" % segment] = log_amplitude_rmse(
            reference["G"], candidate["G"], region
        )
        row["circular_phase_rmse_deg_%s" % segment] = circular_phase_rmse_deg(
            reference["G"], candidate["G"], region, weight
        )
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
            row["surface_tail_p95_ratio"] <= 0.02,
            row["weighted_complex_error"] <= 0.05,
            row["circular_phase_rmse_deg"] <= 5.0,
            row["group_delay_rmse_s"] <= 0.02,
        )
    row["reference_lines_met"] = bool(all(reference_lines))
    return row


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
    repo_root = Path(__file__).resolve().parents[2]
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
        for suffix in ("P061", "V001", "V002", "V003"):
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
    inspection_by_suffix = {}
    for suffix in by_suffix:
        for item in inspections:
            if item["case_name"].endswith(suffix):
                inspection_by_suffix[suffix] = item
                break
    if "P061" in loaded:
        for suffix in ("V001", "V002", "V003"):
            if suffix in loaded:
                tail_ratio = float(inspection_by_suffix.get(suffix, {}).get(
                    "surface_tail_p95_ratio", float("nan")
                ))
                comparisons.append(compare_cases(
                    loaded["P061"], loaded[suffix], frequency, s_values, suffix, tail_ratio
                ))
    write_csv(output / "case_data_completeness.csv", inspections)
    write_csv(output / "validation_comparison.csv", comparisons)
    all_reference_lines_met = bool(
        inspections
        and all(item.get("data_complete", False) for item in inspections)
        and len(comparisons) == 3
        and all(item["reference_lines_met"] for item in comparisons)
    )
    payload = {
        "all_reference_lines_met": all_reference_lines_met,
        "case_data": inspections,
        "validation_comparisons": comparisons,
        "comparison_load_errors": load_errors,
        "note": "本评价独立于后处理执行状态，参考线只供人工判读，不改写postprocess_status.json",
    }
    with (output / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print("评价完成：参考线全部满足=%s，输出=%s" % (all_reference_lines_met, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
