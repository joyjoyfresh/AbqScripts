# -*- coding: utf-8 -*-
"""用代理完整复频响和真实波一维参考重构二维坡地地表响应。

物理链为 ``A_2D(f,s)=G_h(f,s)*A_1D(f)``。脚本不会用入射波或端点
传函代替同工况一维参考；一维参考由建模脚本生成的
``freefield_reference_<record>.npz`` 提供。给定C池直接有限元工况时，程序
同时计算时程、峰值时刻、PGA、TAF和5%阻尼反应谱误差。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from pathlib import Path

import numpy as np


def decode_strings(values):
    array = np.asarray(values)
    if array.dtype.kind == "S":
        array = np.char.decode(array, "utf-8")
    return array.astype(str)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_parameters(config: dict) -> dict[str, float]:
    geometry = config.get("geometry_cfg") or {}
    material = config.get("material_cfg") or {}
    layers = material.get("layers") or []
    bedrock = material.get("bedrock") or {}
    slope_height = float(geometry["slope_height"])
    if not layers:
        thickness_ratio = 0.0
        velocity_ratio = 1.0
    else:
        thickness_ratio = sum(float(layer.get("thickness", 0.0)) for layer in layers) / slope_height
        velocity_ratio = float(layers[0]["vs"]) / float(bedrock["vs"])
    return {
        "slope_angle_deg": float(geometry["slope_angle"]),
        "thickness_ratio": thickness_ratio,
        "velocity_ratio": velocity_ratio,
    }


def predict_field(model, X):
    """按训练脚本保存的模型字典预测复频响场。"""
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
    output = np.full(
        (len(X),) + tuple(model["field_shape"]),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    output[:, model["pixel_mask"]] = complex_values
    mask = np.broadcast_to(model["pixel_mask"], output.shape).copy()
    return output, mask


def next_power_of_two(value: int) -> int:
    result = 1
    while result < int(value):
        result *= 2
    return result


def band_taper(frequency, low, high, edge_width):
    """生成频带边缘余弦渐变，频带外严格为零。"""
    frequency = np.asarray(frequency, dtype=float)
    low = float(low)
    high = float(high)
    edge_width = max(0.0, min(float(edge_width), 0.5 * (high - low)))
    weight = np.zeros(frequency.shape, dtype=float)
    inside = (frequency >= low) & (frequency <= high)
    weight[inside] = 1.0
    if edge_width > 0.0:
        lower = (frequency >= low) & (frequency < low + edge_width)
        upper = (frequency > high - edge_width) & (frequency <= high)
        weight[lower] = np.sin(0.5 * math.pi * (frequency[lower] - low) / edge_width) ** 2
        weight[upper] = np.sin(0.5 * math.pi * (high - frequency[upper]) / edge_width) ** 2
    return weight


def interpolate_transfer(model_frequency, field, mask, fft_frequency, taper):
    """逐空间点对复数传函实部/虚部插值，并施加统一频带窗。"""
    output = np.zeros((field.shape[1], len(fft_frequency)), dtype=np.complex128)
    output_mask = np.zeros(output.shape, dtype=bool)
    for s_index in range(field.shape[1]):
        valid = (
            mask[:, s_index]
            & np.isfinite(field[:, s_index].real)
            & np.isfinite(field[:, s_index].imag)
        )
        if int(np.sum(valid)) < 2:
            continue
        source_f = model_frequency[valid]
        source_h = field[valid, s_index]
        order = np.argsort(source_f)
        source_f = source_f[order]
        source_h = source_h[order]
        inside = (fft_frequency >= source_f[0]) & (fft_frequency <= source_f[-1]) & (taper > 0.0)
        output[s_index, inside] = (
            np.interp(fft_frequency[inside], source_f, source_h.real)
            + 1j * np.interp(fft_frequency[inside], source_f, source_h.imag)
        ) * taper[inside]
        output_mask[s_index, inside] = True
    return output, output_mask


def bridge_toe_spectra(spectra, s_values, toe_s=1.0, half_width=0.10):
    """在坡脚邻域以两端复谱作平滑桥接，保持二维响应的空间连续性。

    同侧一维基准在坡脚两端来自不同的一维柱，直接切换会把参考场的
    离散跳变带入重构二维响应。这里在复谱层（而非 PGA/TAF 成图后）对
    ``[toe_s-half_width, toe_s+half_width]`` 作三次平滑桥接；两端锚点及
    邻域外预测均不改动。令 ``half_width<=0`` 可复现未约束结果。
    """
    result = np.asarray(spectra, dtype=np.complex128).copy()
    s_values = np.asarray(s_values, dtype=float)
    corrected = np.zeros(s_values.shape, dtype=bool)
    if half_width <= 0.0 or result.ndim != 2 or result.shape[0] != len(s_values):
        return result, corrected
    left_s = float(toe_s) - float(half_width)
    right_s = float(toe_s) + float(half_width)
    left = int(np.argmin(np.abs(s_values - left_s)))
    right = int(np.argmin(np.abs(s_values - right_s)))
    if right <= left or abs(s_values[left] - left_s) > 1.0e-8 or abs(s_values[right] - right_s) > 1.0e-8:
        raise ValueError("坡脚连续性桥接需要网格含 s=%.3f 与 s=%.3f" % (left_s, right_s))
    source_left = result[left].copy()
    source_right = result[right].copy()
    for index in range(left + 1, right):
        ratio = (s_values[index] - s_values[left]) / (s_values[right] - s_values[left])
        weight = ratio * ratio * (3.0 - 2.0 * ratio)  # 三次平滑步函数，端点一阶导数为零
        result[index] = (1.0 - weight) * source_left + weight * source_right
        corrected[index] = True
    return result, corrected


def reference_record(package) -> str:
    if "record" not in package:
        return "record"
    value = package["record"]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def load_reference(path: Path, tail_seconds: float, minimum_duration: float | None = None):
    package = np.load(path, allow_pickle=False)
    try:
        time = np.asarray(package["time"], dtype=float)
        rock = np.asarray(package["rock_acc_h"], dtype=float)
        left = np.asarray(package["one_d_left_acc_h"], dtype=float)
        right = np.asarray(package["one_d_right_acc_h"], dtype=float)
        record = reference_record(package)
    finally:
        package.close()
    if time.ndim != 1 or len(time) < 4 or not (rock.shape == left.shape == right.shape == time.shape):
        raise ValueError("一维参考NPZ的time/rock/left/right形状不一致")
    dt_values = np.diff(time)
    dt = float(np.median(dt_values))
    if dt <= 0.0 or np.max(np.abs(dt_values - dt)) > 1.0e-6 * dt:
        raise ValueError("一维参考时间轴必须严格等间隔")
    threshold = max(float(np.max(np.abs(rock))) * 1.0e-10, 1.0e-14)
    active = np.where(np.abs(rock) > threshold)[0]
    active_end = int(active[-1] + 1) if len(active) else len(time)
    desired_end = active_end + int(round(float(tail_seconds) / dt))
    if minimum_duration is not None:
        desired_end = max(desired_end, int(math.ceil(float(minimum_duration) / dt)) + 1)
    count = min(len(time), max(4, desired_end))
    return {
        "path": str(path.resolve()),
        "record": record,
        "time": time[:count],
        "dt": dt,
        "rock": rock[:count],
        "left": left[:count],
        "right": right[:count],
    }


def discover_npz_record(package, requested=None):
    records = sorted({
        match.group(1)
        for key in package.files
        for match in [re.match(r"^frf_(.+)_frequency$", key)]
        if match
    })
    if requested:
        if requested not in records:
            raise ValueError("直接有限元NPZ中不存在记录 %s" % requested)
        return requested
    if len(records) != 1:
        raise ValueError("直接有限元NPZ记录不唯一: %s" % records)
    return records[0]


def calc_s_from_x(x, meta):
    geometry = meta.get("geometry") or {}
    x_crest = float(geometry["x_crest"])
    x_toe = float(geometry["x_toe"])
    slope_height = geometry.get("H_minus_h")
    if slope_height is None:
        slope_height = meta.get("derived", {}).get("slope_height")
    slope_height = float(slope_height)
    x = np.asarray(x, dtype=float)
    s = np.empty(x.shape, dtype=float)
    left = x <= x_crest
    slope = (x > x_crest) & (x <= x_toe)
    right = x > x_toe
    s[left] = (x[left] - x_crest) / slope_height
    s[slope] = (x[slope] - x_crest) / (x_toe - x_crest)
    s[right] = 1.0 + (x[right] - x_toe) / slope_height
    return s


def interpolate_space_series(source_s, values, target_s):
    """对完整时程沿地表坐标线性插值，权重一次求出后作用于所有时刻。"""
    source_s = np.asarray(source_s, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(source_s)
    source_s = source_s[order]
    values = values[order]
    output = np.full((len(target_s), values.shape[1]), np.nan, dtype=float)
    for index, target in enumerate(target_s):
        if target < source_s[0] or target > source_s[-1]:
            continue
        right = int(np.searchsorted(source_s, target, side="left"))
        if right == 0:
            output[index] = values[0]
        elif right >= len(source_s):
            output[index] = values[-1]
        elif abs(source_s[right] - target) <= 1.0e-12:
            output[index] = values[right]
        else:
            left = right - 1
            fraction = (target - source_s[left]) / (source_s[right] - source_s[left])
            output[index] = (1.0 - fraction) * values[left] + fraction * values[right]
    return output


def load_truth(case_dir: Path, target_s, requested_record=None):
    package = np.load(case_dir / "surface_results.npz", allow_pickle=False)
    try:
        record = discover_npz_record(package, requested_record)
        time = np.asarray(package["raw_%s_time" % record], dtype=float)
        x = np.asarray(package["raw_%s_x" % record], dtype=float)
        acceleration = np.asarray(package["raw_%s_acc_h" % record], dtype=float)
    finally:
        package.close()
    meta = read_json(case_dir / "case_meta.json")
    source_s = calc_s_from_x(x, meta)
    return {
        "record": record,
        "time": time,
        "acc": interpolate_space_series(source_s, acceleration, target_s),
        "config": read_json(case_dir / "case_config.json"),
        "case_dir": str(case_dir.resolve()),
    }


def align_time(source_time, values, target_time):
    output = np.zeros((values.shape[0], len(target_time)), dtype=float)
    for index in range(values.shape[0]):
        finite = np.isfinite(values[index])
        if int(np.sum(finite)) < 2:
            output[index] = np.nan
            continue
        output[index] = np.interp(
            target_time, source_time[finite], values[index, finite], left=0.0, right=0.0
        )
    return output


def bandlimit(values, nfft, taper, output_count):
    spectrum = np.fft.rfft(values, n=nfft, axis=1)
    return np.fft.irfft(spectrum * taper[None, :], n=nfft, axis=1)[:, :output_count]


def compute_psa(acceleration, dt, periods, damping=0.05):
    """Newmark平均加速度法计算5%阻尼伪加速度谱。"""
    acceleration = np.atleast_2d(np.asarray(acceleration, dtype=float))
    periods = np.asarray(periods, dtype=float)
    omega = (2.0 * math.pi / periods)[None, :]
    stiffness = omega * omega
    viscous = 2.0 * float(damping) * omega
    beta = 0.25
    gamma = 0.5
    denominator = 1.0 + gamma * dt * viscous + beta * dt * dt * stiffness
    shape = (acceleration.shape[0], len(periods))
    displacement = np.zeros(shape, dtype=float)
    velocity = np.zeros(shape, dtype=float)
    relative_acc = -acceleration[:, 0][:, None] * np.ones((1, len(periods)))
    maximum_displacement = np.zeros(shape, dtype=float)
    for time_index in range(1, acceleration.shape[1]):
        predicted_displacement = (
            displacement + dt * velocity + dt * dt * (0.5 - beta) * relative_acc
        )
        predicted_velocity = velocity + dt * (1.0 - gamma) * relative_acc
        relative_acc = (
            -acceleration[:, time_index][:, None]
            - viscous * predicted_velocity
            - stiffness * predicted_displacement
        ) / denominator
        displacement = predicted_displacement + beta * dt * dt * relative_acc
        velocity = predicted_velocity + gamma * dt * relative_acc
        maximum_displacement = np.maximum(maximum_displacement, np.abs(displacement))
    return stiffness * maximum_displacement


def correlation_rows(truth, prediction):
    result = np.full(truth.shape[0], np.nan, dtype=float)
    for index in range(truth.shape[0]):
        valid = np.isfinite(truth[index]) & np.isfinite(prediction[index])
        if int(np.sum(valid)) < 3:
            continue
        if np.std(truth[index, valid]) <= 0.0 or np.std(prediction[index, valid]) <= 0.0:
            continue
        result[index] = np.corrcoef(truth[index, valid], prediction[index, valid])[0, 1]
    return result


def comparison_metrics(truth, prediction, time, psa_truth, psa_prediction):
    difference = prediction - truth
    denominator = np.sum(truth ** 2, axis=1)
    nrmse = np.sqrt(np.sum(difference ** 2, axis=1) / np.maximum(denominator, 1.0e-30))
    correlation = correlation_rows(truth, prediction)
    pga_truth = np.max(np.abs(truth), axis=1)
    pga_prediction = np.max(np.abs(prediction), axis=1)
    pga_error = np.abs(pga_prediction / np.maximum(pga_truth, 1.0e-30) - 1.0)
    peak_time_truth = time[np.argmax(np.abs(truth), axis=1)]
    peak_time_prediction = time[np.argmax(np.abs(prediction), axis=1)]
    peak_time_error = np.abs(peak_time_prediction - peak_time_truth)
    spectrum_valid = psa_truth > np.maximum(np.max(psa_truth, axis=1, keepdims=True) * 1.0e-8, 1.0e-30)
    spectrum_error = np.full(psa_truth.shape, np.nan, dtype=float)
    spectrum_error[spectrum_valid] = np.abs(
        psa_prediction[spectrum_valid] / psa_truth[spectrum_valid] - 1.0
    )
    spectrum_median = np.nanmedian(spectrum_error, axis=1)
    return {
        "time_nrmse": nrmse,
        "correlation": correlation,
        "pga_relative_error": pga_error,
        "peak_time_error_s": peak_time_error,
        "response_spectrum_relative_error_median": spectrum_median,
        "response_spectrum_relative_error": spectrum_error,
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def plot_outputs(output, time, s_values, acceleration, pga, taf, truth=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    for target_s in (-2.0, 0.5, 2.0):
        index = int(np.argmin(np.abs(s_values - target_s)))
        axes[0].plot(time, acceleration[index], label="s=%.2f" % s_values[index])
    axes[0].set_xlabel("时间 (s)")
    axes[0].set_ylabel("水平加速度")
    axes[0].legend(ncol=3)
    axes[1].plot(s_values, pga, label="重构PGA")
    axes[1].plot(s_values, taf, label="重构TAF")
    axes[1].axvline(0.0, color="k", linewidth=0.6)
    axes[1].axvline(1.0, color="k", linewidth=0.6)
    axes[1].set_xlabel("s")
    axes[1].legend()
    figure.savefig(output / "reconstruction_overview.png", dpi=180)
    plt.close(figure)

    if truth is not None:
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        for axis, target_s in zip(axes, (-2.0, 0.5, 2.0)):
            index = int(np.argmin(np.abs(s_values - target_s)))
            axis.plot(time, truth[index], color="black", linewidth=1.0, label="直接有限元")
            axis.plot(time, acceleration[index], color="#D55E00", linewidth=0.9, label="代理重构")
            axis.set_ylabel("s=%.2f" % s_values[index])
            axis.legend()
        axes[-1].set_xlabel("时间 (s)")
        figure.savefig(output / "direct_fe_comparison.png", dpi=180)
        plt.close(figure)


def case_label(path: Path | None, record: str) -> str:
    if path is None:
        return record
    match = re.search(r"([A-Za-z]+\d{3})$", path.name)
    return match.group(1) if match else path.name


def run_reconstruction(bundle, parameters, reference_path, output, args, truth_case=None):
    output.mkdir(parents=True, exist_ok=True)
    model_frequency = np.asarray(bundle["frequency_hz"], dtype=float)
    s_values = np.asarray(bundle["s"], dtype=float)
    feature_names = decode_strings(bundle["feature_names"])
    unknown = [name for name in feature_names if name not in parameters]
    if unknown:
        raise ValueError("缺少模型输入参数: %s" % unknown)
    X = np.asarray([[parameters[name] for name in feature_names]], dtype=float)
    predicted, predicted_mask = predict_field(bundle["G_h_model"], X)
    G_h = predicted[0]
    G_mask = predicted_mask[0]

    truth_preview = load_truth(truth_case, s_values, args.record) if truth_case else None
    minimum_duration = float(truth_preview["time"][-1]) if truth_preview else None
    reference = load_reference(reference_path, args.tail_seconds, minimum_duration)
    time = reference["time"]
    dt = reference["dt"]
    nfft = next_power_of_two(2 * len(time))
    fft_frequency = np.fft.rfftfreq(nfft, dt)
    taper = band_taper(
        fft_frequency, float(model_frequency[0]), float(model_frequency[-1]), args.edge_taper_hz
    )
    transfer, transfer_mask = interpolate_transfer(
        model_frequency, G_h, G_mask, fft_frequency, taper
    )
    left_spectrum = np.fft.rfft(reference["left"] - np.mean(reference["left"]), n=nfft)
    right_spectrum = np.fft.rfft(reference["right"] - np.mean(reference["right"]), n=nfft)
    response_spectrum = transfer * left_spectrum[None, :]
    acceleration = np.fft.irfft(response_spectrum, n=nfft, axis=1)[:, :len(time)]
    acceleration_raw = acceleration.copy()
    left_band = np.fft.irfft(left_spectrum * taper, n=nfft)[:len(time)]
    right_band = np.fft.irfft(right_spectrum * taper, n=nfft)[:len(time)]
    pga = np.max(np.abs(acceleration), axis=1)

    # 全场恒定高度坡顶一维自由场基准（高度 Hbase + h + d）
    pga_left = float(np.max(np.abs(left_band)))
    pga_right = float(np.max(np.abs(right_band)))
    pga_1d_crest = np.full(s_values.shape, pga_left)
    pga_1d_same_side = np.where(s_values <= 1.0, pga_left, pga_right)

    taf = pga / max(pga_left, 1.0e-30)
    taf_same_side = pga / np.maximum(pga_1d_same_side, 1.0e-30)
    periods = np.exp(np.linspace(math.log(0.10), math.log(2.00), args.period_count))
    psa = compute_psa(acceleration, dt, periods, damping=0.05)
    toe_continuity_mask = np.zeros(s_values.shape, dtype=bool)

    rows = [
        {"s": float(s_values[index]), "pga_reconstructed": float(pga[index]),
         "pga_1d_same_side": float(pga_1d_same_side[index]),
         "pga_1d_crest": float(pga_1d_crest[index]),
         "pga_1d_toe_continuous": float(pga_1d_crest[index]),
         "taf_reconstructed": float(taf[index]),
         "taf_reconstructed_same_side": float(taf_same_side[index])}
        for index in range(len(s_values))
    ]
    payload = {
        "schema_version": 1,
        "record": reference["record"],
        "parameters": {name: float(value) for name, value in parameters.items()},
        "model": str(args.model.resolve()),
        "reference": reference["path"],
        "definition": "A_2D=G_h_surrogate*A_1D_same_side",
        "reported_time_domain_band_hz": [float(model_frequency[0]), float(model_frequency[-1])],
        "edge_taper_hz": float(args.edge_taper_hz),
        "note": "时程、PGA、TAF和反应谱均按代理公共频带重构；直接有限元比较采用同一频带窗",
    }
    npz_payload = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "time": time,
        "s": s_values,
        "frequency_hz": model_frequency,
        "predicted_G_h": G_h,
        "predicted_G_h_valid_mask": G_mask,
        "fft_frequency_hz": fft_frequency,
        "fft_transfer_valid_mask": transfer_mask,
        "one_d_left_acc_h_bandlimited": left_band,
        "one_d_right_acc_h_bandlimited": right_band,
        "pga_1d_same_side": pga_1d_same_side,
        "pga_1d_crest": pga_1d_crest,
        "pga_1d_toe_continuous": pga_1d_crest,
        "reconstructed_acc_h": acceleration,
        "reconstructed_acc_h_raw": acceleration_raw,
        "pga_reconstructed": pga,
        "taf_reconstructed": taf,
        "taf_reconstructed_same_side": taf_same_side,
        "toe_continuity_mask": toe_continuity_mask,
        "period_s": periods,
        "psa_reconstructed": psa,
        "metadata_json": np.asarray(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
    }

    truth_band = None
    if truth_preview is not None:
        truth_aligned = align_time(truth_preview["time"], truth_preview["acc"], time)
        truth_band = bandlimit(truth_aligned, nfft, taper, len(time))
        truth_psa = compute_psa(truth_band, dt, periods, damping=0.05)
        comparison = comparison_metrics(truth_band, acceleration, time, truth_psa, psa)
        pga_truth = np.max(np.abs(truth_band), axis=1)
        taf_truth = pga_truth / max(pga_left, 1.0e-30)
        taf_truth_same_side = pga_truth / np.maximum(pga_1d_same_side, 1.0e-30)
        npz_payload["direct_fe_acc_h_bandlimited"] = truth_band
        npz_payload["direct_fe_pga"] = pga_truth
        npz_payload["direct_fe_taf"] = taf_truth
        npz_payload["direct_fe_psa"] = truth_psa
        for index, row in enumerate(rows):
            row.update({
                "pga_direct_fe": float(pga_truth[index]),
                "taf_direct_fe": float(taf_truth[index]),
                "taf_direct_fe_same_side": float(taf_truth_same_side[index]),
                "time_nrmse": float(comparison["time_nrmse"][index]),
                "correlation": float(comparison["correlation"][index]),
                "pga_relative_error": float(comparison["pga_relative_error"][index]),
                "taf_relative_error": float(abs(taf[index] / max(taf_truth[index], 1.0e-30) - 1.0)),
                "peak_time_error_s": float(comparison["peak_time_error_s"][index]),
                "response_spectrum_relative_error_median": float(
                    comparison["response_spectrum_relative_error_median"][index]
                ),
            })
        payload["direct_fe_case"] = truth_preview["case_dir"]
        payload["comparison_summary"] = {
            name: aggregate(comparison[name])
            for name in (
                "time_nrmse", "correlation", "pga_relative_error", "peak_time_error_s",
                "response_spectrum_relative_error_median",
            )
        }
        npz_payload.update({
            "direct_fe_acc_h_bandlimited": truth_band,
            "direct_fe_pga": pga_truth,
            "direct_fe_taf": taf_truth,
            "direct_fe_taf_same_side": taf_truth_same_side,
            "direct_fe_psa": truth_psa,
            "response_spectrum_relative_error": comparison["response_spectrum_relative_error"],
        })

    np.savez_compressed(output / "reconstruction.npz", **npz_payload)
    write_csv(output / "reconstruction_by_s.csv", rows)
    with (output / "reconstruction_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=True)
    if not args.no_figures:
        plot_outputs(output, time, s_values, acceleration, pga, taf, truth_band)
    print("完成真实波重构：%s" % output)


def discover_reference(case_dir: Path, requested_record=None) -> Path:
    if requested_record:
        exact = case_dir / ("freefield_reference_%s.npz" % requested_record)
        if exact.is_file():
            return exact
    candidates = sorted(case_dir.glob("freefield_reference_*.npz"))
    if len(candidates) != 1:
        raise ValueError("%s 中无法唯一确定一维参考NPZ" % case_dir)
    return candidates[0]


def parse_args(argv=None):
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="代理复频响驱动的真实波重构")
    parser.add_argument("--model", type=Path, default=repo_root / "Run/ch4_sp_ml/complex_frf_surrogate.pkl")
    parser.add_argument("--output", type=Path, default=repo_root / "Run/ch4_sp_reconstruction")
    parser.add_argument("--truth-case", type=Path, default=None, help="一个C池直接有限元工况目录")
    parser.add_argument("--truth-root", type=Path, default=None, help="批量扫描其下case-*直接有限元工况")
    parser.add_argument("--reference", type=Path, default=None, help="独立运行时的一维参考NPZ")
    parser.add_argument("--case-config", type=Path, default=None, help="独立运行时用于读取三个物理参数")
    parser.add_argument("--slope-angle", type=float, default=None)
    parser.add_argument("--thickness-ratio", type=float, default=None)
    parser.add_argument("--velocity-ratio", type=float, default=None)
    parser.add_argument("--record", default=None)
    parser.add_argument("--tail-seconds", type=float, default=6.0)
    parser.add_argument("--edge-taper-hz", type=float, default=0.2)
    parser.add_argument("--period-count", type=int, default=40)
    parser.add_argument("--toe-continuity-half-width", type=float, default=0.10,
                        help="坡脚复谱连续性桥接半宽度；设为0可关闭")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.period_count < 2:
        raise SystemExit("--period-count必须不少于2")
    with args.model.open("rb") as handle:
        bundle = pickle.load(handle)
    cases = []
    if args.truth_case is not None:
        cases.append(args.truth_case.resolve())
    if args.truth_root is not None:
        cases.extend(sorted(
            path.resolve() for path in args.truth_root.glob("case-*")
            if (path / "surface_results.npz").is_file()
        ))
    if cases:
        for case_dir in cases:
            config = read_json(case_dir / "case_config.json")
            parameters = infer_parameters(config)
            reference = discover_reference(case_dir, args.record)
            record = reference.stem.replace("freefield_reference_", "")
            run_reconstruction(
                bundle, parameters, reference,
                args.output / case_label(case_dir, record), args, truth_case=case_dir,
            )
        return 0

    if args.reference is None:
        raise SystemExit("独立重构必须提供--reference；或提供--truth-case/--truth-root")
    if args.case_config is not None:
        parameters = infer_parameters(read_json(args.case_config))
    else:
        values = (args.slope_angle, args.thickness_ratio, args.velocity_ratio)
        if any(value is None for value in values):
            raise SystemExit("独立重构需给--case-config，或同时给三个物理参数")
        parameters = {
            "slope_angle_deg": float(args.slope_angle),
            "thickness_ratio": float(args.thickness_ratio),
            "velocity_ratio": float(args.velocity_ratio),
        }
    record = args.reference.stem.replace("freefield_reference_", "")
    run_reconstruction(bundle, parameters, args.reference, args.output / record, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
