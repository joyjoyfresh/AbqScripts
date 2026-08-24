# -*- coding: utf-8 -*-
"""训练小论文完整复频响代理：最近邻、POD-Ridge与POD-GPR。

输入为 ``analyze_complex_frf.py`` 生成的 ``complex_frf_dataset.npz``。
训练/验证按完整物理工况划分；标准化、缺失值处理、POD和回归均在当前训练折
内部完成，频率点和空间点不会被拆成独立样本。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import warnings
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


MODEL_NAMES = ("nearest", "pod_ridge", "pod_gpr")
AMPLITUDE_EPS = 1.0e-12


def decode_strings(values):
    array = np.asarray(values)
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    return array.astype(str)


def contiguous_runs(mask):
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return
    split_at = np.where(np.diff(indices) > 1)[0] + 1
    for run in np.split(indices, split_at):
        if len(run):
            yield run


def phase_group_delay(field, mask, frequency):
    phase = np.full(field.shape, np.nan, dtype=float)
    delay = np.full(field.shape, np.nan, dtype=float)
    wrapped = np.angle(field)
    for s_index in range(field.shape[1]):
        for run in contiguous_runs(mask[:, s_index]):
            phase[run, s_index] = np.unwrap(wrapped[run, s_index])
            if len(run) >= 3:
                values = phase[run, s_index]
                if len(run) >= 5:
                    kernel = np.ones(5, dtype=float) / 5.0
                    values = np.convolve(np.pad(values, (2, 2), mode="edge"), kernel, mode="valid")
                delay[run, s_index] = -np.gradient(values, frequency[run]) / (2.0 * math.pi)
                delay[run[[0, -1]], s_index] = np.nan
    return phase, delay


def spatial_phase_gradient(field, mask, s_values, segments):
    """在A/B/C各段内展开空间相位并计算表观空间相位梯度。"""
    result = np.full(field.shape, np.nan, dtype=float)
    wrapped = np.angle(field)
    for f_index in range(field.shape[0]):
        for segment in ("A", "B", "C"):
            allowed = mask[f_index] & (segments == segment)
            for run in contiguous_runs(allowed):
                if len(run) < 3:
                    continue
                values = np.unwrap(wrapped[f_index, run])
                if len(run) >= 5:
                    kernel = np.ones(5, dtype=float) / 5.0
                    values = np.convolve(np.pad(values, (2, 2), mode="edge"), kernel, mode="valid")
                result[f_index, run] = np.gradient(values, s_values[run])
                result[f_index, run[[0, -1]]] = np.nan
    return result


def prepare_output(field, valid_mask, minimum_valid_fraction=0.95):
    """把复数场转换成实部/虚部向量，并保存训练期填补与缩放参数。"""
    finite = valid_mask & np.isfinite(field.real) & np.isfinite(field.imag)
    pixel_mask = np.mean(finite, axis=0) >= float(minimum_valid_fraction)
    if int(np.sum(pixel_mask)) < 2:
        raise ValueError("训练折共同有效的复频响网格点不足")
    complex_values = field[:, pixel_mask]
    values = np.concatenate([complex_values.real, complex_values.imag], axis=1)
    finite_values = np.isfinite(values)
    counts = np.sum(finite_values, axis=0)
    sums = np.nansum(values, axis=0)
    fill = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    values = np.where(finite_values, values, fill[None, :])
    center = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale[scale < 1.0e-10] = 1.0
    standardized = (values - center) / scale
    return standardized, {
        "pixel_mask": pixel_mask,
        "fill": fill,
        "center": center,
        "scale": scale,
        "field_shape": tuple(field.shape[1:]),
    }


def select_component_count(singular_values, energy_target, maximum):
    energy = singular_values ** 2
    if float(np.sum(energy)) <= 0.0:
        return 1
    cumulative = np.cumsum(energy) / np.sum(energy)
    count = int(np.searchsorted(cumulative, float(energy_target)) + 1)
    return max(1, min(count, int(maximum), len(singular_values)))


def fit_field_model(
    X,
    field,
    valid_mask,
    model_name,
    pod_energy=0.995,
    max_components=12,
    minimum_valid_fraction=0.95,
    seed=20260801,
):
    feature_scaler = StandardScaler().fit(X)
    X_scaled = feature_scaler.transform(X)
    if model_name == "nearest":
        return {
            "model_name": model_name,
            "feature_scaler": feature_scaler,
            "X_scaled": X_scaled,
            "field": np.asarray(field, dtype=np.complex128),
            "valid_mask": np.asarray(valid_mask, dtype=bool),
        }
    standardized, output = prepare_output(field, valid_mask, minimum_valid_fraction)
    u_matrix, singular_values, vh_matrix = np.linalg.svd(standardized, full_matrices=False)
    count = select_component_count(singular_values, pod_energy, max_components)
    basis = vh_matrix[:count]
    scores = standardized.dot(basis.T)
    if model_name == "pod_ridge":
        estimator = RidgeCV(
            alphas=np.logspace(-5, 5, 21),
            cv=min(5, max(2, len(X) - 1)),
            scoring="neg_mean_squared_error",
        )
        estimator.fit(X_scaled, scores)
    elif model_name == "pod_gpr":
        estimators = []
        for component in range(count):
            kernel = (
                ConstantKernel(1.0, (1.0e-3, 1.0e3))
                * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1.0e-2, 1.0e2), nu=2.5)
                + WhiteKernel(noise_level=1.0e-6, noise_level_bounds=(1.0e-10, 1.0e-1))
            )
            estimator_component = GaussianProcessRegressor(
                kernel=kernel,
                normalize_y=True,
                n_restarts_optimizer=0,
                random_state=seed + component,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator_component.fit(X_scaled, scores[:, component])
            estimators.append(estimator_component)
        estimator = estimators
    else:
        raise ValueError("未知模型 %s" % model_name)
    output.update({
        "model_name": model_name,
        "feature_scaler": feature_scaler,
        "basis": basis,
        "singular_values": singular_values[:count],
        "component_count": count,
        "estimator": estimator,
        "pod_energy_target": float(pod_energy),
    })
    return output


def predict_field(model, X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_scaled = model["feature_scaler"].transform(X)
    if model["model_name"] == "nearest":
        distances = np.sum((X_scaled[:, None, :] - model["X_scaled"][None, :, :]) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)
        return model["field"][indices].copy(), model["valid_mask"][indices].copy()
    estimator = model["estimator"]
    if model["model_name"] == "pod_gpr":
        scores = np.column_stack([item.predict(X_scaled) for item in estimator])
    else:
        scores = np.asarray(estimator.predict(X_scaled))
        if scores.ndim == 1:
            scores = scores[:, None]
    standardized = scores.dot(model["basis"])
    values = standardized * model["scale"] + model["center"]
    pixel_count = int(np.sum(model["pixel_mask"]))
    complex_values = values[:, :pixel_count] + 1j * values[:, pixel_count:]
    output = np.full((len(X),) + tuple(model["field_shape"]), np.nan + 1j * np.nan, dtype=np.complex128)
    output[:, model["pixel_mask"]] = complex_values
    mask = np.broadcast_to(model["pixel_mask"], output.shape).copy()
    return output, mask


def weighted_complex_error(truth, prediction, mask, weight):
    weights = np.broadcast_to(np.asarray(weight, dtype=float)[:, None], truth.shape)
    # 用布尔索引取有效像素，避免掩码外非有限值经 0×inf 产生 NaN 污染求和
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    w = weights[valid]
    t = truth[valid]
    p = prediction[valid]
    denominator = np.sum(w * np.abs(t) ** 2)
    if denominator <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum(w * np.abs(p - t) ** 2) / denominator))


def circular_phase_error(truth, prediction, mask, weight):
    weights = np.broadcast_to(np.asarray(weight, dtype=float)[:, None], truth.shape)
    # 用布尔索引取有效像素，避免掩码外非有限值经 0×inf/0×NaN 污染求和
    valid = mask & np.isfinite(truth) & np.isfinite(prediction)
    if not np.any(valid):
        return float("nan")
    w = weights[valid]
    difference = np.angle(prediction[valid] * np.conj(truth[valid]))
    denominator = np.sum(w)
    if denominator <= 0.0:
        return float("nan")
    return float(np.degrees(np.sqrt(np.sum(w * difference ** 2) / denominator)))


def log_amplitude_rmse(truth, prediction, mask):
    valid = mask & (np.abs(truth) > AMPLITUDE_EPS) & (np.abs(prediction) > AMPLITUDE_EPS)
    if not np.any(valid):
        return float("nan")
    error = np.log(np.abs(prediction[valid])) - np.log(np.abs(truth[valid]))
    return float(np.sqrt(np.mean(error ** 2)))


def peak_features(field, mask, frequency, s_values):
    if not np.any(mask):
        return (float("nan"),) * 3
    index = int(np.nanargmax(np.where(mask, np.abs(field), np.nan)))
    f_index, s_index = np.unravel_index(index, field.shape)
    return float(np.abs(field[f_index, s_index])), float(frequency[f_index]), float(s_values[s_index])


def evaluate_prediction(
    case_id, model_name, fold, truth, prediction, mask, weight,
    frequency, s_values, segments,
):
    truth_peak = peak_features(truth, mask, frequency, s_values)
    pred_peak = peak_features(prediction, mask, frequency, s_values)
    truth_phase, truth_delay = phase_group_delay(truth, mask, frequency)
    pred_phase, pred_delay = phase_group_delay(prediction, mask, frequency)
    delay_valid = np.isfinite(truth_delay) & np.isfinite(pred_delay) & mask
    delay_rmse = float(np.sqrt(np.mean((pred_delay[delay_valid] - truth_delay[delay_valid]) ** 2))) if np.any(delay_valid) else float("nan")
    truth_gradient = spatial_phase_gradient(truth, mask, s_values, segments)
    pred_gradient = spatial_phase_gradient(prediction, mask, s_values, segments)
    gradient_valid = np.isfinite(truth_gradient) & np.isfinite(pred_gradient) & mask
    gradient_rmse = (
        float(np.sqrt(np.mean((pred_gradient[gradient_valid] - truth_gradient[gradient_valid]) ** 2)))
        if np.any(gradient_valid) else float("nan")
    )
    peak_index = int(np.nanargmax(np.where(mask, np.abs(truth), np.nan)))
    peak_f_index, peak_s_index = np.unravel_index(peak_index, truth.shape)
    peak_phase_error = float(np.degrees(abs(np.angle(
        prediction[peak_f_index, peak_s_index] * np.conj(truth[peak_f_index, peak_s_index])
    ))))
    peak_delay_error = (
        float(abs(pred_delay[peak_f_index, peak_s_index] - truth_delay[peak_f_index, peak_s_index]))
        if np.isfinite(pred_delay[peak_f_index, peak_s_index])
        and np.isfinite(truth_delay[peak_f_index, peak_s_index]) else float("nan")
    )
    row = {
        "case_id": case_id,
        "model": model_name,
        "fold": int(fold),
        "valid_fraction": float(np.mean(mask)),
        "E_complex_w": weighted_complex_error(truth, prediction, mask, weight),
        "log_amplitude_rmse": log_amplitude_rmse(truth, prediction, mask),
        "circular_phase_rmse_deg": circular_phase_error(truth, prediction, mask, weight),
        "group_delay_rmse_s": delay_rmse,
        "spatial_phase_gradient_rmse_rad_per_s": gradient_rmse,
        "phase_error_at_amplitude_peak_deg": peak_phase_error,
        "group_delay_error_at_amplitude_peak_s": peak_delay_error,
        "peak_amplitude_relative_error": abs(pred_peak[0] / truth_peak[0] - 1.0) if truth_peak[0] > 0.0 else float("nan"),
        "peak_frequency_error_hz": abs(pred_peak[1] - truth_peak[1]),
        "peak_s_error": abs(pred_peak[2] - truth_peak[2]),
    }
    for label, low, high in (("low", 0.5, 3.0), ("mid", 3.0, 6.0), ("high", 6.0, 10.000001)):
        region = mask & ((frequency >= low) & (frequency < high))[:, None]
        row["log_amplitude_rmse_%s" % label] = log_amplitude_rmse(truth, prediction, region)
        row["circular_phase_rmse_deg_%s" % label] = circular_phase_error(
            truth, prediction, region, weight
        )
    for segment in ("A", "B", "C"):
        region = mask & (segments == segment)[None, :]
        row["E_complex_w_%s" % segment] = weighted_complex_error(
            truth, prediction, region, weight
        )
        row["log_amplitude_rmse_%s" % segment] = log_amplitude_rmse(
            truth, prediction, region
        )
    return row


def median_metric(rows, model_name, field="E_complex_w"):
    values = [float(row[field]) for row in rows if row["model"] == model_name and np.isfinite(float(row[field]))]
    return float(np.median(values)) if values else float("inf")


def choose_model(rows):
    errors = {name: median_metric(rows, name) for name in MODEL_NAMES}
    simple = min(("nearest", "pod_ridge"), key=lambda name: errors[name])
    if np.isfinite(errors["pod_gpr"]):
        improvement = (errors[simple] - errors["pod_gpr"]) / max(errors[simple], 1.0e-12)
        if improvement >= 0.05:
            return "pod_gpr", errors
    if simple == "pod_ridge":
        improvement = (errors["nearest"] - errors["pod_ridge"]) / max(errors["nearest"], 1.0e-12)
        if improvement < 0.05:
            return "nearest", errors
    return simple, errors


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summary_rows(cv_rows):
    result = []
    for model_name in MODEL_NAMES:
        selected = [row for row in cv_rows if row["model"] == model_name]
        row = {"model": model_name, "n_cases": len(selected)}
        for metric in (
            "E_complex_w",
            "log_amplitude_rmse",
            "circular_phase_rmse_deg",
            "group_delay_rmse_s",
            "spatial_phase_gradient_rmse_rad_per_s",
            "phase_error_at_amplitude_peak_deg",
            "group_delay_error_at_amplitude_peak_s",
            "peak_amplitude_relative_error",
            "peak_frequency_error_hz",
            "peak_s_error",
            "log_amplitude_rmse_low",
            "log_amplitude_rmse_mid",
            "log_amplitude_rmse_high",
            "circular_phase_rmse_deg_low",
            "circular_phase_rmse_deg_mid",
            "circular_phase_rmse_deg_high",
            "E_complex_w_A",
            "E_complex_w_B",
            "E_complex_w_C",
            "log_amplitude_rmse_A",
            "log_amplitude_rmse_B",
            "log_amplitude_rmse_C",
        ):
            values = np.asarray([item[metric] for item in selected], dtype=float)
            values = values[np.isfinite(values)]
            row[metric + "_median"] = float(np.median(values)) if len(values) else float("nan")
            row[metric + "_p90"] = float(np.percentile(values, 90.0)) if len(values) else float("nan")
        result.append(row)
    return result


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="训练完整复频响POD代理")
    parser.add_argument("--dataset", type=Path, default=repo_root / "Run/ch4_sp_analysis/complex_frf_dataset.npz")
    parser.add_argument("--output", type=Path, default=repo_root / "Run/ch4_sp_ml")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--pod-energy", type=float, default=0.995)
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    package = np.load(args.dataset, allow_pickle=False)
    try:
        case_ids = decode_strings(package["case_ids"])
        groups = decode_strings(package["case_groups"])
        X = np.asarray(package["X"], dtype=float)
        G = np.asarray(package["G_h"], dtype=np.complex128)
        valid = np.asarray(package["valid_mask"], dtype=bool)
        weights = np.asarray(package["input_weight"], dtype=float)
        frequency = np.asarray(package["frequency_hz"], dtype=float)
        s_values = np.asarray(package["s"], dtype=float)
        segments = decode_strings(package["segments"])
        feature_names = decode_strings(package["feature_names"])
    finally:
        package.close()
    train_indices = np.where(groups == "P")[0]
    validation_indices = np.where(groups == "B")[0]
    if len(train_indices) < max(5, args.folds):
        raise SystemExit("P开发工况不足：需要至少%d例，当前%d例" % (max(5, args.folds), len(train_indices)))
    folds = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_assignments = np.full(len(train_indices), -1, dtype=int)
    cv_rows = []
    for fold, (local_train, local_test) in enumerate(folds.split(train_indices)):
        fold_assignments[local_test] = fold
        train = train_indices[local_train]
        test = train_indices[local_test]
        for model_name in MODEL_NAMES:
            print("折%d/%d，模型=%s" % (fold + 1, args.folds, model_name))
            model = fit_field_model(
                X[train], G[train], valid[train], model_name,
                pod_energy=args.pod_energy,
                max_components=args.max_components,
                minimum_valid_fraction=args.minimum_valid_fraction,
                seed=args.seed + fold * 100,
            )
            prediction, prediction_mask = predict_field(model, X[test])
            for local_index, global_index in enumerate(test):
                metric_mask = valid[global_index] & prediction_mask[local_index]
                cv_rows.append(evaluate_prediction(
                    case_ids[global_index], model_name, fold,
                    G[global_index], prediction[local_index], metric_mask,
                    weights[global_index], frequency, s_values, segments,
                ))
    selected_model, median_errors = choose_model(cv_rows)
    print("选定模型=%s，CV中位复数误差=%s" % (selected_model, median_errors))
    primary_model = fit_field_model(
        X[train_indices], G[train_indices], valid[train_indices], selected_model,
        pod_energy=args.pod_energy,
        max_components=args.max_components,
        minimum_valid_fraction=args.minimum_valid_fraction,
        seed=args.seed,
    )
    bundle = {
        "schema_version": 2,
        "selected_model": selected_model,
        "feature_names": feature_names,
        "frequency_hz": frequency,
        "s": s_values,
        "segments": segments,
        "training_case_ids": case_ids[train_indices],
        "training_X": X[train_indices],
        "G_h_model": primary_model,
        "target_definition": "G_h=A_2D/A_1D_left_global",
        "reference_scope": "全地表统一使用左侧上平台一维自由场",
        "reconstruction_requirement": "真实波重构必须提供同工况左侧上平台一维自由场时程",
        "configuration": {
            "pod_energy": args.pod_energy,
            "max_components": args.max_components,
            "minimum_valid_fraction": args.minimum_valid_fraction,
            "seed": args.seed,
        },
    }
    model_path = args.output / "complex_frf_surrogate.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_csv(args.output / "cv_case_metrics.csv", cv_rows)
    write_csv(args.output / "model_summary.csv", summary_rows(cv_rows))
    assignment_rows = [
        {"case_id": case_ids[index], "fold": int(fold_assignments[position])}
        for position, index in enumerate(train_indices)
    ]
    write_csv(args.output / "fold_assignments.csv", assignment_rows)
    validation_rows = []
    if len(validation_indices):
        prediction, prediction_mask = predict_field(primary_model, X[validation_indices])
        for local_index, global_index in enumerate(validation_indices):
            metric_mask = valid[global_index] & prediction_mask[local_index]
            validation_rows.append(evaluate_prediction(
                case_ids[global_index], selected_model, -1,
                G[global_index], prediction[local_index], metric_mask,
                weights[global_index], frequency, s_values, segments,
            ))
        write_csv(args.output / "unseen_combination_metrics.csv", validation_rows)
    selection = {
        "selected_model": selected_model,
        "selection_rule": "GPR需比较优简单方法降低至少5%中位E_complex_w；Ridge同理对最近邻",
        "cv_median_E_complex_w": median_errors,
        "training_case_count": int(len(train_indices)),
        "validation_case_count": int(len(validation_indices)),
        "model_path": str(model_path.resolve()),
    }
    with (args.output / "model_selection.json").open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2, allow_nan=True)
    print("训练完成：%s" % model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
