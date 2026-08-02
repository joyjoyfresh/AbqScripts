# -*- coding: utf-8 -*-
"""评价补充域敏感性工况的相位校正后残余差异。

脚本读取补充 autorun 生成的多个候选工况，并使用正式 P061 工况作为外部参考。
所有候选工况均先消除实际归一化地表坐标造成的理论水平传播相位，再报告幅值、
相对相位形态和群时延残差。只有带 ``V004`` 后缀的工况属于正式记录，其余结果
仅用于本次参数取值分析。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import evaluate_complex_frf_quality as quality


ALIGNMENT_FIELDS = (
    "delta_x_m_min",
    "delta_x_m_max",
    "delta_t_x_s_min",
    "delta_t_x_s_max",
    "delta_t_x_s_median",
    "delta_y_reference_m",
    "delta_t_y_reference_s",
)


def read_json(path):
    """读取 JSON 对象；文件缺失或非法时返回空字典。"""
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (IOError, OSError, ValueError):
        return {}


def case_geometry(case_dir):
    """读取工况的侧向净空和基底深度。"""
    config = read_json(case_dir / "case_config.json")
    geometry = config.get("geometry_cfg") or {}
    return {
        "side_clearance": float(geometry["side_clearance"]),
        "base_depth": float(geometry["base_depth"]),
    }


def plain_value(value):
    """把 NumPy 标量转换为可写入 JSON/CSV 的 Python 标量。"""
    if isinstance(value, np.generic):
        return value.item()
    return value


def comparison_row(reference, candidate, candidate_dir, frequency, s_values):
    """生成单个候选工况的相位校正残差和原始指标。"""
    aligned, alignment = quality.coordinate_phase_alignment(
        reference, candidate, frequency, s_values
    )
    mask = reference["mask"] & candidate["mask"]
    weight = np.minimum(reference["weight"], candidate["weight"])
    residual = quality._comparison_metrics(
        reference, aligned, frequency, s_values, mask, weight
    )
    raw = quality._comparison_metrics(
        reference, candidate, frequency, s_values, mask, weight
    )
    geometry = case_geometry(candidate_dir)
    is_formal = candidate_dir.name.endswith("V004")
    row = {
        "reference": reference["case_id"],
        "candidate": candidate["case_id"],
        "candidate_folder": candidate_dir.name,
        "variation": "V004" if is_formal else candidate_dir.name,
        "formal_case": is_formal,
        "side_clearance": geometry["side_clearance"],
        "base_depth": geometry["base_depth"],
        "comparison_basis": (
            "phase_aligned_residual" if alignment.get("applied") else "raw_unaligned"
        ),
        "phase_alignment_applied": bool(alignment.get("applied")),
        "phase_alignment_reason": alignment.get("reason"),
    }
    for field in quality.COMPARISON_METRIC_FIELDS:
        row[field] = plain_value(residual.get(field))
        row["raw_" + field] = plain_value(raw.get(field))
    for field in ALIGNMENT_FIELDS:
        row[field] = plain_value(alignment.get(field))
    return row, alignment, residual, raw, geometry


def uncertainty_payload(reference, candidate, row, alignment, residual, raw, geometry):
    """生成正式 V004 的域不确定性报告。"""
    return {
        "schema_version": 1,
        "reference": reference["case_id"],
        "candidate": candidate["case_id"],
        "scope": "P061与V004单因素侧向净空工况的域不确定性诊断",
        "geometry": geometry,
        "comparison_basis": row["comparison_basis"],
        "phase_alignment": alignment,
        "residual_metrics": {
            field: plain_value(residual.get(field))
            for field in quality.COMPARISON_METRIC_FIELDS
        },
        "raw_metrics": {
            field: plain_value(raw.get(field))
            for field in quality.COMPARISON_METRIC_FIELDS
        },
        "conclusion_stability": {
            "status": "not_assessable_from_single_stress_case",
            "reason": "V004只隔离侧向净空，仍不能单独决定全体H/P参数排序；需在生产数据完成后复核主要结论。",
            "rule": "只有主要效应在加入该数值不确定性后仍保持方向和排序时，才保留强结论；重叠时改写为不可区分或需谨慎。",
        },
    }


def parse_args(argv=None):
    """解析补充域敏感性评价参数。"""
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="评价补充计算域敏感性工况")
    parser.add_argument(
        "--root",
        type=Path,
        default=repo_root / "Run/ch4_sp_01_V_domain_sensitivity",
        help="补充工况根目录",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=repo_root / "Run/ch4_sp_01_V",
        help="正式 V 根目录，用于读取 P061 参考工况",
    )
    parser.add_argument("--reference-case", default="case-001-P061")
    parser.add_argument(
        "--candidate-case",
        action="append",
        default=None,
        help="只评价指定的 case-* 目录；可重复传入",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    """执行补充域敏感性评价并写出比较表。"""
    args = parse_args(argv)
    root = args.root.resolve()
    reference_dir = (args.reference_root / args.reference_case).resolve()
    output = (args.output or (root / "evaluation")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not reference_dir.is_dir():
        raise SystemExit("参考工况目录不存在：{}".format(reference_dir))

    if args.candidate_case:
        candidate_dirs = [
            (root / name).resolve() for name in args.candidate_case
        ]
        missing = [path for path in candidate_dirs if not path.is_dir()]
        if missing:
            raise SystemExit("指定候选工况目录不存在：{}".format(missing))
    else:
        candidate_dirs = sorted(
            path for path in root.glob("case-*") if path.is_dir()
        )
    if not candidate_dirs:
        raise SystemExit("补充工况根目录没有 case-* 目录：{}".format(root))

    frequency = quality.regular_grid(
        quality.FREQUENCY_MIN,
        quality.FREQUENCY_MAX,
        quality.FREQUENCY_STEP,
    )
    s_values = quality.regular_grid(
        quality.S_MIN,
        quality.S_MAX,
        quality.S_STEP,
    )
    inspection_rows = [quality.inspect_case(reference_dir)]
    inspection_rows.extend(quality.inspect_case(path) for path in candidate_dirs)
    quality.write_csv(output / "case_data_completeness.csv", inspection_rows)

    reference_inspection = inspection_rows[0]
    if not reference_inspection.get("data_complete"):
        raise SystemExit("P061 参考工况数据不完整：{}".format(reference_inspection))
    reference = quality.load_case(reference_dir, frequency, s_values)

    rows = []
    formal_payload = None
    for candidate_dir, inspection in zip(candidate_dirs, inspection_rows[1:]):
        if not inspection.get("data_complete"):
            rows.append({
                "candidate_folder": candidate_dir.name,
                "variation": "V004" if candidate_dir.name.endswith("V004") else candidate_dir.name,
                "formal_case": candidate_dir.name.endswith("V004"),
                "error": inspection.get("reason") or "data_incomplete",
            })
            continue
        candidate = quality.load_case(candidate_dir, frequency, s_values)
        row, alignment, residual, raw, geometry = comparison_row(
            reference, candidate, candidate_dir, frequency, s_values
        )
        rows.append(row)
        if candidate_dir.name.endswith("V004"):
            formal_payload = uncertainty_payload(
                reference, candidate, row, alignment, residual, raw, geometry
            )

    quality.write_csv(
        output / "domain_sensitivity_comparison.csv",
        rows,
    )

    summary = {
        "schema_version": 1,
        "reference_case": str(reference_dir),
        "candidate_root": str(root),
        "output": str(output),
        "all_data_complete": bool(
            reference_inspection.get("data_complete")
            and all(item.get("data_complete") for item in inspection_rows[1:])
        ),
        "candidate_count": len(candidate_dirs),
        "formal_case": next(
            (row for row in rows if row.get("formal_case")), None
        ),
        "comparison_count": len([row for row in rows if "error" not in row]),
        "note": "只有V004进入正式判读；其余候选行仅用于本次域参数取值分析。",
    }
    with (output / "domain_sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=True)
    if formal_payload is not None:
        with (output / "v004_domain_uncertainty.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(formal_payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print(
        "补充域敏感性评价完成：候选数={}，正式V004报告={}，输出={}".format(
            len(candidate_dirs), formal_payload is not None, output
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
