# -*- coding: utf-8 -*-
"""计算域敏感性的成对收敛分析。

以当前最大域为临时参考，计算相邻计算域之间的直接成对差异，
报告有符号峰值变化、收敛趋势和判定准则。

核心原则：计算域敏感性应通过"大域之间的直接收敛"证明，
而不是通过"所有工况相对于最小域的差异"证明。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

import analyze_complex_frf as frf
import evaluate_complex_frf_quality as quality


REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_ROOT = REPO_ROOT / "Run" / "ch4_sp_01_V_domain_sensitivity"
FORMAL_ROOT = REPO_ROOT / "Run" / "ch4_sp_01_V"

CASES = {
    "S1-B3": (FORMAL_ROOT, "case-001-P061", 1.0, 3.0),
    "S2-B3": (DOMAIN_ROOT, "case-002-DOM-S2-B3", 2.0, 3.0),
    "S4-B3": (DOMAIN_ROOT, "case-001-V004", 4.0, 3.0),
    "S6-B3": (DOMAIN_ROOT, "case-003-DOM-S6-B3", 6.0, 3.0),
    "S2-B6": (DOMAIN_ROOT, "case-004-DOM-S2-B6", 2.0, 6.0),
    "S4-B6": (DOMAIN_ROOT, "case-005-DOM-S4-B6", 4.0, 6.0),
    "S6-B6": (DOMAIN_ROOT, "case-006-DOM-S6-B6", 6.0, 6.0),
}

LATERAL_PAIRS = [
    ("S1-B3", "S2-B3", 3.0),
    ("S2-B3", "S4-B3", 3.0),
    ("S4-B3", "S6-B3", 3.0),
    ("S2-B6", "S4-B6", 6.0),
    ("S4-B6", "S6-B6", 6.0),
]

BOTTOM_PAIRS = [
    ("S2-B3", "S2-B6", 2.0),
    ("S4-B3", "S4-B6", 4.0),
    ("S6-B3", "S6-B6", 6.0),
]

KEY_POSITIONS = {
    "crest": 0.0,
    "mid_slope": 0.5,
    "toe": 1.0,
    "near_crest": -1.0,
    "near_toe": 2.0,
}

THRESHOLDS = {
    "peak_amplitude_signed_pct": 5.0,
    "peak_frequency_hz": 0.2,
    "log_amplitude_rmse": 0.05,
    "phase_rmse_deg": 5.0,
}

H_SLOPE = 100.0
VS_BEDROCK = 2000.0
C_MIN = 600.0
F_DOMINANT = 4.0
ANALYSIS_DURATION = 6.0


def plain(v):
    if isinstance(v, np.generic):
        return v.item()
    return v


def peak_features_signed(G, mask, frequency, s_values):
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    amp = np.where(mask, np.abs(G), np.nan)
    idx = int(np.nanargmax(amp))
    fi, si = np.unravel_index(idx, G.shape)
    return plain(np.abs(G[fi, si])), plain(frequency[fi]), plain(s_values[si])


def compare_pair(ref, cand, frequency, s_values, apply_phase=False):
    """一对工况的直接比较，报告有符号峰值变化。"""
    aligned = cand
    alignment_info = {"applied": False}
    if apply_phase:
        aligned, alignment_info = quality.coordinate_phase_alignment(
            ref, cand, frequency, s_values
        )
    mask = ref["mask"] & aligned["mask"]
    weight = np.minimum(ref["weight"], aligned["weight"])
    w = weight[:, None]

    ref_pk, ref_f, ref_s = peak_features_signed(ref["G"], mask, frequency, s_values)
    cand_pk, cand_f, cand_s = peak_features_signed(aligned["G"], mask, frequency, s_values)

    result = {
        "signed_peak_change_pct": plain(
            (cand_pk - ref_pk) / ref_pk * 100.0 if ref_pk > 0 else float("nan")
        ),
        "abs_peak_change_pct": plain(
            abs(cand_pk - ref_pk) / ref_pk * 100.0 if ref_pk > 0 else float("nan")
        ),
        "ref_peak_amplitude": plain(ref_pk),
        "cand_peak_amplitude": plain(cand_pk),
        "signed_peak_freq_change_hz": plain(cand_f - ref_f),
        "signed_peak_s_change": plain(cand_s - ref_s),
        "signed_peak_s_change_m": plain((cand_s - ref_s) * H_SLOPE),
    }

    valid = mask & np.isfinite(ref["G"].real) & np.isfinite(aligned["G"].real)
    wv = np.broadcast_to(w, valid.shape)[valid]
    denom = np.sum(wv * np.abs(ref["G"][valid]) ** 2)
    numer = np.sum(wv * np.abs(aligned["G"][valid] - ref["G"][valid]) ** 2)
    result["weighted_complex_error"] = plain(
        float(np.sqrt(numer / denom)) if denom > 0 else float("nan")
    )

    lv = valid & (np.abs(ref["G"]) > 1e-12) & (np.abs(aligned["G"]) > 1e-12)
    log_diff = np.log(np.abs(aligned["G"][lv])) - np.log(np.abs(ref["G"][lv]))
    result["log_amplitude_rmse"] = plain(float(np.sqrt(np.mean(log_diff ** 2))))
    result["log_amplitude_p90"] = plain(float(np.percentile(np.abs(log_diff), 90)))

    phase_diff = np.angle(aligned["G"][valid] * np.conj(ref["G"][valid]))
    result["phase_rmse_deg"] = plain(
        float(np.degrees(np.sqrt(np.mean(phase_diff ** 2))))
    )
    result["phase_p90_deg"] = plain(float(np.degrees(np.percentile(np.abs(phase_diff), 90))))

    dv = mask & np.isfinite(ref["group_delay"]) & np.isfinite(aligned["group_delay"])
    if np.any(dv):
        dd = aligned["group_delay"][dv] - ref["group_delay"][dv]
        result["group_delay_rmse_s"] = plain(float(np.sqrt(np.mean(dd ** 2))))
        result["group_delay_p90_s"] = plain(float(np.percentile(np.abs(dd), 90)))
    else:
        result["group_delay_rmse_s"] = float("nan")
        result["group_delay_p90_s"] = float("nan")

    seg = frf.segment_labels_from_s(s_values)
    for label in ("A", "B", "C"):
        region = mask & (seg[None, :] == label)
        rv = region & (np.abs(ref["G"]) > 1e-12) & (np.abs(aligned["G"]) > 1e-12)
        if np.any(rv):
            ld = np.log(np.abs(aligned["G"][rv])) - np.log(np.abs(ref["G"][rv]))
            result["log_amplitude_rmse_%s" % label] = plain(float(np.sqrt(np.mean(ld ** 2))))
        else:
            result["log_amplitude_rmse_%s" % label] = float("nan")

    result["phase_alignment_applied"] = bool(alignment_info.get("applied"))
    return result


def taf_at_positions(case, s_values, positions):
    """提取关键位置的 TAF（幅值和相位随频率变化）。"""
    out = {}
    for name, target_s in positions.items():
        idx = int(np.argmin(np.abs(s_values - target_s)))
        out[name] = {
            "s_actual": plain(s_values[idx]),
            "amplitude": np.abs(case["G"][:, idx]),
            "phase": np.angle(case["G"][:, idx]),
        }
    return out


def taf_summary(cases, s_values, frequency, positions):
    """所有工况在关键位置的 TAF 峰值汇总。"""
    rows = []
    for label, case in cases.items():
        for pos_name, target_s in positions.items():
            idx = int(np.argmin(np.abs(s_values - target_s)))
            amp = np.abs(case["G"][:, idx])
            valid = np.isfinite(amp)
            if not np.any(valid):
                continue
            pk_idx = int(np.nanargmax(np.where(valid, amp, np.nan)))
            rows.append({
                "case": label,
                "position": pos_name,
                "s": plain(s_values[idx]),
                "peak_amplitude": plain(amp[pk_idx]),
                "peak_frequency_hz": plain(frequency[pk_idx]),
            })
    return rows


def boundary_reflection(side_h, base_h):
    """边界反射时间估计。"""
    d_side = side_h * H_SLOPE
    d_bottom = base_h * H_SLOPE
    t_side = 2.0 * d_side / C_MIN
    t_bottom = 2.0 * d_bottom / C_MIN
    wavelengths = {}
    for f in [1.0, 2.0, 4.0, 8.0]:
        wavelengths["lambda_%.0fHz_m" % f] = plain(C_MIN / f)
        wavelengths["d_side_over_lambda_%.0fHz" % f] = plain(d_side * f / C_MIN)
    return {
        "side_clearance_H": side_h,
        "base_depth_H": base_h,
        "d_side_m": plain(d_side),
        "d_bottom_m": plain(d_bottom),
        "t_return_side_s": plain(t_side),
        "t_return_bottom_s": plain(t_bottom),
        "t_analysis_s": ANALYSIS_DURATION,
        "side_reflection_in_window": t_side < ANALYSIS_DURATION,
        "bottom_reflection_in_window": t_bottom < ANALYSIS_DURATION,
        "dominant_wavelength_m": plain(C_MIN / F_DOMINANT),
        "d_side_over_dominant_lambda": plain(d_side * F_DOMINANT / C_MIN),
        **wavelengths,
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="计算域成对收敛分析")
    parser.add_argument("--domain-root", type=Path, default=DOMAIN_ROOT)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    output = args.output or (args.domain_root / "evaluation" / "pairwise")
    output.mkdir(parents=True, exist_ok=True)

    frequency = frf.regular_grid(frf.FREQUENCY_MIN, frf.FREQUENCY_MAX, frf.FREQUENCY_STEP)
    s_values = frf.regular_grid(frf.S_MIN, frf.S_MAX, frf.S_STEP)

    # ── 加载所有工况 ──
    loaded = {}
    for label, (root, dirname, side, base) in CASES.items():
        d = (root / dirname).resolve()
        if not d.is_dir():
            print("警告：工况目录不存在：%s (%s)" % (label, d))
            continue
        try:
            loaded[label] = frf.load_case(d, frequency, s_values)
            print("已加载 %s (%.0fH/%.0fH)" % (label, side, base))
        except Exception as exc:
            print("加载失败 %s：%s" % (label, exc))

    if len(loaded) != len(CASES):
        missing = set(CASES) - set(loaded)
        print("错误：以下工况加载失败：%s" % missing)
        return 1

    meta = {label: info for label, (_, _, side, base) in CASES.items()
            for info in [{"side": side, "base": base}]}

    # ── 侧向净空成对比较（同底部深度） ──
    lateral_results = []
    for ref_label, cand_label, base in LATERAL_PAIRS:
        r = compare_pair(
            loaded[ref_label], loaded[cand_label],
            frequency, s_values, apply_phase=True,
        )
        r.update({
            "pair": "%s → %s" % (ref_label, cand_label),
            "ref_case": ref_label,
            "cand_case": cand_label,
            "ref_side_H": meta[ref_label]["side"],
            "cand_side_H": meta[cand_label]["side"],
            "base_depth_H": base,
            "comparison_type": "lateral_clearance",
        })
        lateral_results.append(r)
        print("%-20s  有符号峰值变化: %+.2f%%  |Δpeak|: %.2f%%  "
              "ln|G| RMSE: %.4f  相位 RMSE: %.2f°" % (
                  r["pair"], r["signed_peak_change_pct"],
                  r["abs_peak_change_pct"], r["log_amplitude_rmse"],
                  r["phase_rmse_deg"]))

    # ── 收敛趋势分析（侧向净空） ──
    convergence = {}
    for base in [3.0, 6.0]:
        errors = []
        for r in lateral_results:
            if r["base_depth_H"] == base:
                errors.append({
                    "pair": r["pair"],
                    "ref_side": r["ref_side_H"],
                    "cand_side": r["cand_side_H"],
                    "abs_peak_pct": r["abs_peak_change_pct"],
                    "log_amp_rmse": r["log_amplitude_rmse"],
                    "phase_rmse": r["phase_rmse_deg"],
                    "signed_peak_pct": r["signed_peak_change_pct"],
                    "group_delay_rmse": r["group_delay_rmse_s"],
                })
        if len(errors) >= 2:
            peak_dec = all(
                errors[i]["abs_peak_pct"] > errors[i + 1]["abs_peak_pct"]
                for i in range(len(errors) - 1)
            )
            logamp_dec = all(
                errors[i]["log_amp_rmse"] > errors[i + 1]["log_amp_rmse"]
                for i in range(len(errors) - 1)
            )
            convergence["B%.0f" % base] = {
                "errors": errors,
                "peak_amplitude_monotonic_decrease": peak_dec,
                "log_amplitude_rmse_monotonic_decrease": logamp_dec,
            }

    # ── 底部深度成对比较（同侧向净空） ──
    bottom_results = []
    for ref_label, cand_label, side in BOTTOM_PAIRS:
        r = compare_pair(
            loaded[ref_label], loaded[cand_label],
            frequency, s_values, apply_phase=False,
        )
        r.update({
            "pair": "%s → %s" % (ref_label, cand_label),
            "ref_case": ref_label,
            "cand_case": cand_label,
            "ref_base_H": meta[ref_label]["base"],
            "cand_base_H": meta[cand_label]["base"],
            "side_clearance_H": side,
            "comparison_type": "bottom_depth",
        })
        bottom_results.append(r)
        print("%-20s  有符号峰值变化: %+.2f%%  ln|G| RMSE: %.4f" % (
            r["pair"], r["signed_peak_change_pct"], r["log_amplitude_rmse"]))

    # ── 关键位置 TAF ──
    taf_rows = taf_summary(loaded, s_values, frequency, KEY_POSITIONS)

    # ── 收敛判定 ──
    b3 = next(r for r in lateral_results
              if r["ref_case"] == "S4-B3" and r["cand_case"] == "S6-B3")
    b6 = next(r for r in lateral_results
              if r["ref_case"] == "S4-B6" and r["cand_case"] == "S6-B6")

    def judge(r):
        return {
            "peak_amplitude_pass": abs(r["signed_peak_change_pct"]) <= THRESHOLDS["peak_amplitude_signed_pct"],
            "peak_frequency_pass": abs(r["signed_peak_freq_change_hz"]) <= THRESHOLDS["peak_frequency_hz"],
            "log_amplitude_rmse_pass": r["log_amplitude_rmse"] <= THRESHOLDS["log_amplitude_rmse"],
            "phase_rmse_pass": r["phase_rmse_deg"] <= THRESHOLDS["phase_rmse_deg"],
        }

    j3 = judge(b3)
    j6 = judge(b6)
    all_pass_3 = all(j3.values())
    all_pass_6 = all(j6.values())

    judgment = {
        "thresholds": THRESHOLDS,
        "lateral_B3": {
            "pair": b3["pair"],
            "signed_peak_change_pct": b3["signed_peak_change_pct"],
            "log_amplitude_rmse": b3["log_amplitude_rmse"],
            "phase_rmse_deg": b3["phase_rmse_deg"],
            "criteria": j3,
            "all_pass": all_pass_3,
            "conclusion": (
                "4H与6H核心指标均在阈值内，可采用4H作为生产计算域"
                if all_pass_3
                else "4H与6H的核心指标仍有显著差异，需考虑增加8H工况"
            ),
        },
        "lateral_B6": {
            "pair": b6["pair"],
            "signed_peak_change_pct": b6["signed_peak_change_pct"],
            "log_amplitude_rmse": b6["log_amplitude_rmse"],
            "phase_rmse_deg": b6["phase_rmse_deg"],
            "criteria": j6,
            "all_pass": all_pass_6,
        },
        "convergence_trend": convergence,
    }

    # ── 边界反射检查 ──
    reflections = {}
    for base in [3.0, 6.0]:
        for side in [1.0, 2.0, 4.0, 6.0]:
            reflections["S%.0f-B%.0f" % (side, base)] = boundary_reflection(side, base)

    # ── 汇总 ──
    summary = {
        "schema_version": 2,
        "analysis_method": "pairwise_convergence",
        "reference_strategy": "最大域(6H)作为临时参考，计算相邻域成对差异",
        "key_difference_from_v1": [
            "不以1H为统一参考基线",
            "有符号峰值变化取代绝对值峰值误差",
            "相邻域成对比较取代全工况对1H比较",
            "收敛趋势（E递减）作为核心判据",
        ],
        "judgment": judgment,
        "boundary_reflection": reflections,
        "lateral_pairwise_count": len(lateral_results),
        "bottom_pairwise_count": len(bottom_results),
        "key_position_taf_count": len(taf_rows),
    }

    # ── 写出 ──
    with (output / "pairwise_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=True)

    all_pairs = []
    for r in lateral_results:
        all_pairs.append({k: plain(v) for k, v in r.items()})
    for r in bottom_results:
        all_pairs.append({k: plain(v) for k, v in r.items()})
    write_csv(output / "pairwise_comparison.csv", all_pairs)
    write_csv(output / "key_position_taf.csv", taf_rows)

    refl_rows = []
    for key, v in reflections.items():
        refl_rows.append({"case": key, **{k: plain(val) for k, val in v.items()}})
    write_csv(output / "boundary_reflection_check.csv", refl_rows)

    # ── 打印判定摘要 ──
    print("\n" + "=" * 70)
    print("收敛判定摘要")
    print("=" * 70)
    for depth_key, jdg in [("B3 (3H底部深度)", judgment["lateral_B3"]),
                           ("B6 (6H底部深度)", judgment["lateral_B6"])]:
        print("\n[%s] %s" % (depth_key, jdg["pair"]))
        print("  有符号峰值变化: %+.2f%% (阈值 ≤%.0f%%) %s" % (
            jdg["signed_peak_change_pct"],
            THRESHOLDS["peak_amplitude_signed_pct"],
            "PASS" if jdg["criteria"]["peak_amplitude_pass"] else "FAIL"))
        print("  ln|G| RMSE:     %.4f (阈值 ≤%.2f) %s" % (
            jdg["log_amplitude_rmse"],
            THRESHOLDS["log_amplitude_rmse"],
            "PASS" if jdg["criteria"]["log_amplitude_rmse_pass"] else "FAIL"))
        print("  相位 RMSE:      %.2f deg (阈值 <=%.0f deg) %s" % (
            jdg["phase_rmse_deg"],
            THRESHOLDS["phase_rmse_deg"],
            "PASS" if jdg["criteria"]["phase_rmse_pass"] else "FAIL"))
        if "conclusion" in jdg:
            print("  结论: %s" % jdg["conclusion"])

    print("\n收敛趋势:")
    for key, trend in convergence.items():
        print("  [%s] 峰值单调递减: %s  ln|G| RMSE单调递减: %s" % (
            key,
            "是" if trend["peak_amplitude_monotonic_decrease"] else "否",
            "是" if trend["log_amplitude_rmse_monotonic_decrease"] else "否"))
        for e in trend["errors"]:
            print("    %s: |Δpeak|=%.2f%%  ln|G| RMSE=%.4f  相位 RMSE=%.2f°" % (
                e["pair"], e["abs_peak_pct"], e["log_amp_rmse"], e["phase_rmse"]))

    print("\n边界反射检查（关键工况）:")
    for key in ["S1-B3", "S2-B3", "S4-B3", "S6-B3"]:
        r = reflections[key]
        print("  %s: d_side=%.0fm  t_return=%.2fs  分析窗内反射: %s  "
              "d/λ(4Hz)=%.2f" % (
                  key, r["d_side_m"], r["t_return_side_s"],
                  "是" if r["side_reflection_in_window"] else "否",
                  r["d_side_over_dominant_lambda"]))

    print("\n输出目录: %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
