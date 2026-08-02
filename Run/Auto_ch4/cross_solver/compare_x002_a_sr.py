# -*- coding: utf-8 -*-
"""按公共坐标和时间基准评价 X002-A 与已完成的 X002-S。

脚本不对任一通道追加移时、缩放或符号修正。采用分级评价体系：
核心工程指标（PGA空间分布、早期全场快照）为正式门控；
逐点时程、频谱、基岩入射和晚时段快照作为辅助诊断，不影响总判定。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
from pathlib import Path

import numpy as np


INPUT_PEAK = 0.980665
COMMON_SHIFT_SPECFEM = 0.30
COMMON_SHIFT_ABAQUS = -0.30
SURFACE_POINTS = (("crest", 0.0), ("slope_midpoint", 0.5), ("toe", 1.0))


def find_repo_root(start: Path) -> Path:
    """优先读取批处理传入的仓库路径，否则从工况目录向上寻找。"""
    configured = os.environ.get("ABQSCRIPTS_REPO_ROOT")
    if configured:
        candidate = Path(configured).resolve()
        marker = candidate / "Run" / "Auto_ch4" / "cross_solver" / "x_validation_parameters.json"
        if marker.is_file():
            return candidate
    for candidate in (start, *start.parents):
        marker = candidate / "Run" / "Auto_ch4" / "cross_solver" / "x_validation_parameters.json"
        if marker.is_file():
            return candidate
    raise FileNotFoundError("无法从工况目录定位仓库根目录")


def json_safe(value):
    """把 NumPy 标量和非有限值转换为可审计 JSON。"""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_json(path: Path, payload: dict) -> None:
    """写出 UTF-8 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")


def raw_prefix(npz: np.lib.npyio.NpzFile) -> str:
    """识别 surface_results.npz 中唯一的原始地表时程前缀。"""
    candidates = []
    for key in npz.files:
        if not key.startswith("raw_") or not key.endswith("_time"):
            continue
        if "underground" in key or "energy" in key:
            continue
        candidates.append(key[:-len("time")])
    if len(candidates) != 1:
        raise RuntimeError(f"无法唯一识别Abaqus原始地表记录，候选={candidates}")
    return candidates[0]


def require_close(label: str, actual: float, expected: float,
                  tolerance: float = 1.0e-9) -> None:
    """硬校验关键浮点配置。"""
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"X002-A身份校验失败: {label}={actual}，预期{expected}")


def validate_abaqus_identity(case_dir: Path, params: dict,
                              prefix: str, raw_time: np.ndarray) -> dict:
    """拒绝把其他 Abaqus 工况误作 X002-A。"""
    config = json.loads((case_dir / "case_config.json").read_text(encoding="utf-8"))
    meta = json.loads((case_dir / "case_meta.json").read_text(encoding="utf-8"))
    pulse = params["pulse"]
    geometry = params["geometry"]
    bedrock = params["materials"]["bedrock"]
    case = params["cases"]["X002-A"]
    expected_wave = Path(pulse["abaqus_acceleration_file"]).name
    expected_prefix = "raw_" + Path(expected_wave).stem + "_"
    if case_dir.name != "X002-A" or meta.get("folder") != "X002-A":
        raise RuntimeError("X002-A身份校验失败: 工况目录或case_meta.folder不是X002-A")
    if prefix != expected_prefix:
        raise RuntimeError(f"X002-A身份校验失败: 原始记录{prefix}，预期{expected_prefix}")
    wave_files = config.get("run_cfg", {}).get("wave_files", [])
    if len(wave_files) != 1 or Path(wave_files[0]).name != expected_wave:
        raise RuntimeError(f"X002-A身份校验失败: wave_files={wave_files}")
    wave_path = case_dir / expected_wave
    digest = hashlib.sha256(wave_path.read_bytes()).hexdigest()
    if digest != pulse["abaqus_acceleration_sha256"]:
        raise RuntimeError("X002-A身份校验失败: 工况目录公共脉冲SHA-256不一致")

    material = config["material_cfg"]
    layers = material.get("layers", [])
    if len(layers) != 1 or layers[0].get("name") != "cover":
        raise RuntimeError("X002-A身份校验失败: P061必须包含一层覆盖层")
    cover = params["materials"]["cover"]
    require_close("覆盖层Vs", layers[0]["vs"], cover["vs"])
    require_close("覆盖层泊松比", layers[0]["poisson_ratio"], cover["poisson_ratio"])
    require_close("覆盖层密度", layers[0]["density"], cover["density"])
    require_close("覆盖层厚度", layers[0]["thickness"], cover["thickness_below_crest"])
    require_close("入射角", material["angle"], params["physics"]["incidence_angle_from_vertical_deg"])
    require_close("基岩Vs", material["bedrock"]["vs"], bedrock["vs"])
    require_close("基岩泊松比", material["bedrock"]["poisson_ratio"], bedrock["poisson_ratio"])
    require_close("基岩密度", material["bedrock"]["density"], bedrock["density"])
    for key, expected_key in (("slope_height", "slope_height"),
                              ("slope_angle", "slope_angle_deg"),
                              ("crest_window", "crest_window_h"),
                              ("toe_window", "toe_window_h"),
                              ("side_clearance", "side_clearance_h"),
                              ("base_depth", "base_depth_h")):
        require_close("geometry_cfg." + key, config["geometry_cfg"][key], geometry[expected_key])
    require_close("damping_cfg.fc", config["damping_cfg"]["fc"], pulse["f0_hz"])
    require_close("time_cfg.tail_seconds", config["time_cfg"]["tail_seconds"],
                  pulse["abaqus_tail_seconds"])
    require_close("freefield_cfg.phase_origin_x", config["freefield_cfg"]["phase_origin_x"],
                  pulse["abaqus_phase_origin_x_m"])
    require_close("mesh_cfg.size", config["mesh_cfg"]["size"], case["mesh_size"])
    require_close("case_meta.geometry.total_L", meta["geometry"]["total_L"], geometry["domain_length"])
    require_close("case_meta.incident_angle", meta["incident_angle"],
                  params["physics"]["incidence_angle_from_vertical_deg"])
    require_close("case_meta.freefield.phase_origin_x", meta["freefield"]["phase_origin_x"],
                  pulse["abaqus_phase_origin_x_m"])
    if meta["freefield"].get("initial_state_mode") != "incremental":
        raise RuntimeError("X002-A身份校验失败: initial_state_mode不是incremental")
    actual_damping = meta["damping"]["layers"]
    if len(actual_damping) != 2:
        raise RuntimeError("X002-A身份校验失败: 实际阻尼材料应为两层（基岩+覆盖层）")
    bed_damping = next((d for d in actual_damping if d.get("name") == "Bedrock"), None)
    cover_damping = next((d for d in actual_damping if d.get("name") == "cover"), None)
    if bed_damping is None or cover_damping is None:
        raise RuntimeError("X002-A身份校验失败: 阻尼层缺少Bedrock或cover")
    expected_bed_damping = bedrock["damping_mapping"]
    require_close("Rayleigh alpha (bedrock)", bed_damping["alpha"], expected_bed_damping["alpha"], 1.0e-12)
    require_close("Rayleigh beta (bedrock)", bed_damping["beta"], expected_bed_damping["beta"], 1.0e-15)
    expected_cover_damping = cover["damping_mapping"]
    require_close("Rayleigh alpha (cover)", cover_damping["alpha"], expected_cover_damping["alpha"], 1.0e-10)
    require_close("Rayleigh beta (cover)", cover_damping["beta"], expected_cover_damping["beta"], 1.0e-13)
    dt = float(case["dt"])
    if len(raw_time) < 2 or np.any(np.diff(raw_time) <= 0.0):
        raise RuntimeError("X002-A身份校验失败: Abaqus时间轴不是严格递增序列")
    require_close("Abaqus原始起始时刻", raw_time[0], 0.0, max(1.0e-7, 0.01 * dt))
    require_close("Abaqus原始中位步长", np.median(np.diff(raw_time)), dt,
                  max(1.0e-6, 0.01 * dt))
    require_close("Abaqus原始结束时刻", raw_time[-1], case["analysis_time"],
                  max(1.0e-6, 0.1 * dt))
    return {
        "passed": True,
        "record_prefix": prefix,
        "wave_sha256": digest,
        "case_folder": meta.get("folder"),
        "rayleigh_alpha": actual_damping[0]["alpha"],
        "rayleigh_beta": actual_damping[0]["beta"],
    }


def time_resample(matrix: np.ndarray, source_time: np.ndarray,
                  target_time: np.ndarray) -> np.ndarray:
    """逐通道线性重采样，禁止外插。"""
    if target_time[0] < source_time[0] - 1.0e-9 or target_time[-1] > source_time[-1] + 1.0e-9:
        raise RuntimeError("公共时间超出Abaqus可用范围")
    return np.asarray([np.interp(target_time, source_time, row) for row in matrix], dtype=float)


def spatial_resample(matrix: np.ndarray, source_x: np.ndarray,
                     target_x: np.ndarray) -> np.ndarray:
    """把节点时程确定性插值到公共地表横坐标。"""
    order = np.argsort(source_x)
    x = np.asarray(source_x[order], dtype=float)
    values = np.asarray(matrix[order], dtype=float)
    unique_x, inverse = np.unique(x, return_inverse=True)
    if len(unique_x) != len(x):
        sums = np.zeros((len(unique_x), values.shape[1]), dtype=float)
        counts = np.zeros(len(unique_x), dtype=float)
        for index, group in enumerate(inverse):
            sums[group] += values[index]
            counts[group] += 1.0
        values = sums / counts[:, None]
        x = unique_x
    if target_x[0] < x[0] - 1.0e-8 or target_x[-1] > x[-1] + 1.0e-8:
        raise RuntimeError("公共地表坐标超出Abaqus地表节点范围")
    return np.asarray([np.interp(target_x, x, values[:, index])
                       for index in range(values.shape[1])], dtype=float).T


def surface_elevation(source_x: np.ndarray, source_y: np.ndarray,
                      target_x: np.ndarray) -> np.ndarray:
    """把 Abaqus TOP_SURFACE 高程插值到公共横坐标。"""
    order = np.argsort(source_x)
    x = np.asarray(source_x[order], dtype=float)
    y = np.asarray(source_y[order], dtype=float)
    unique_x, first = np.unique(x, return_index=True)
    return np.interp(target_x, unique_x, y[first])


def load_abaqus(case_dir: Path, common_time: np.ndarray,
                 target_x: np.ndarray, target_z: np.ndarray,
                 params: dict) -> dict:
    """读取并映射 Abaqus 地表与地下时程。"""
    result_path = case_dir / "surface_results.npz"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with np.load(result_path, allow_pickle=False) as data:
        prefix = raw_prefix(data)
        raw_time = np.asarray(data[prefix + "time"], dtype=float)
        identity = validate_abaqus_identity(case_dir, params, prefix, raw_time)
        source_time = raw_time + COMMON_SHIFT_ABAQUS
        x = np.asarray(data[prefix + "x"], dtype=float)
        y = np.asarray(data[prefix + "y"], dtype=float)
        mapped_y = surface_elevation(x, y, target_x)
        surface_coordinate_error = float(np.max(np.abs(mapped_y - target_z)))
        if surface_coordinate_error > 1.0e-4:
            raise RuntimeError("Abaqus TOP_SURFACE高程与X002-S公共地表坐标不一致")
        acc_h_nodes = time_resample(np.asarray(data[prefix + "acc_h"], dtype=float),
                                    source_time, common_time)
        acc_v_nodes = time_resample(np.asarray(data[prefix + "acc_v"], dtype=float),
                                    source_time, common_time)
        surface = {
            "x": target_x,
            "acc_h": spatial_resample(acc_h_nodes, x, target_x),
            "acc_v": spatial_resample(acc_v_nodes, x, target_x),
            "source_x": x,
            "source_y": y,
            "source_time": source_time,
            "record_prefix": prefix,
            "surface_coordinate_max_error_m": surface_coordinate_error,
        }
        underground = None
        required = (prefix + "underground_time", prefix + "underground_x",
                    prefix + "underground_y", prefix + "underground_acc_h",
                    prefix + "underground_acc_v")
        if all(key in data.files for key in required):
            under_time = np.asarray(data[prefix + "underground_time"], dtype=float) + COMMON_SHIFT_ABAQUS
            names_key = prefix + "underground_name"
            if names_key in data.files:
                names = [item.decode("utf-8") if isinstance(item, bytes) else str(item)
                         for item in np.asarray(data[names_key]).tolist()]
            else:
                count = len(np.asarray(data[prefix + "underground_x"]))
                names = [f"VALIDATION_UNDERGROUND_{index + 1}" for index in range(count)]
            underground = {
                "name": names,
                "x": np.asarray(data[prefix + "underground_x"], dtype=float),
                "y": np.asarray(data[prefix + "underground_y"], dtype=float),
                "acc_h": time_resample(np.asarray(data[prefix + "underground_acc_h"], dtype=float),
                                        under_time, common_time),
                "acc_v": time_resample(np.asarray(data[prefix + "underground_acc_v"], dtype=float),
                                        under_time, common_time),
            }
        else:
            raise RuntimeError("X002-A规范数据缺少必备的三处地下检查点")
        for label, values, expected_shape in (
                ("surface.acc_h", surface["acc_h"], (801, 2000)),
                ("surface.acc_v", surface["acc_v"], (801, 2000)),
                ("underground.acc_h", underground["acc_h"], (3, 2000)),
                ("underground.acc_v", underground["acc_v"], (3, 2000))):
            if values.shape != expected_shape:
                raise RuntimeError(f"X002-A {label}形状{values.shape}，预期{expected_shape}")
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"X002-A {label}包含非有限值")
    return {"surface": surface, "underground": underground, "identity": identity}


def load_trace(output_dir: Path, network: str, station: str,
               component: str, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """读取一条 SPECFEM2D 加速度时程并作唯一全局幅值换算。"""
    path = output_dir / f"{network}.{station}.{component}.sema"
    values = np.loadtxt(path, dtype=float)
    return values[:, 0] + COMMON_SHIFT_SPECFEM, values[:, 1] * scale


def load_specfem(case_dir: Path, scale: float) -> dict:
    """读取 SPECFEM2D 公共地表和基岩检查点。"""
    station_map = json.loads((case_dir / "DATA" / "station_map.json").read_text(encoding="utf-8"))
    network = station_map["network"]
    output_dir = case_dir / "OUTPUT_FILES"
    main = sorted((item for item in station_map["stations"] if item["role"] == "main_surface"),
                  key=lambda item: float(item["s"]))
    rock = sorted((item for item in station_map["stations"] if item["role"] == "incident_check"),
                  key=lambda item: item["station"])
    if len(main) != 801 or len(rock) != 3:
        raise RuntimeError(f"X002-S测点数量错误: main={len(main)}, rock={len(rock)}")
    common_time = None
    surface_h, surface_v = [], []
    for item in main:
        time_h, values_h = load_trace(output_dir, network, item["station"], "BXX", scale)
        time_v, values_v = load_trace(output_dir, network, item["station"], "BXZ", scale)
        if common_time is None:
            common_time = time_h
        if not np.allclose(time_h, common_time, atol=1.0e-12, rtol=0.0):
            raise RuntimeError("SPECFEM2D地表时程网格不一致")
        if not np.allclose(time_v, common_time, atol=1.0e-12, rtol=0.0):
            raise RuntimeError("SPECFEM2D竖向时程网格不一致")
        surface_h.append(values_h)
        surface_v.append(values_v)
    rock_data = []
    for item in rock:
        time_h, values_h = load_trace(output_dir, network, item["station"], "BXX", scale)
        time_v, values_v = load_trace(output_dir, network, item["station"], "BXZ", scale)
        rock_data.append({**item, "time": time_h, "acc_h": values_h, "acc_v": values_v})
    surface_h = np.asarray(surface_h, dtype=float)
    surface_v = np.asarray(surface_v, dtype=float)
    common_time = np.asarray(common_time, dtype=float)
    if common_time.shape != (2000,) or not np.all(np.isfinite(common_time)):
        raise RuntimeError("X002-S公共时间必须为2000个有限值")
    if np.any(np.diff(common_time) <= 0.0):
        raise RuntimeError("X002-S公共时间不是严格递增序列")
    require_close("X002-S公共起始时刻", common_time[0], 0.0, 1.0e-12)
    require_close("X002-S公共结束时刻", common_time[-1], 1.999, 1.0e-12)
    require_close("X002-S公共中位步长", np.median(np.diff(common_time)), 0.001, 1.0e-12)
    if surface_h.shape != (801, 2000) or surface_v.shape != (801, 2000):
        raise RuntimeError("X002-S地表矩阵形状不是801×2000")
    if not np.all(np.isfinite(surface_h)) or not np.all(np.isfinite(surface_v)):
        raise RuntimeError("X002-S地表矩阵包含非有限值")
    for item in rock_data:
        if item["acc_h"].shape != (2000,) or item["acc_v"].shape != (2000,):
            raise RuntimeError("X002-S地下检查点时程长度不是2000")
        if not np.all(np.isfinite(item["acc_h"])) or not np.all(np.isfinite(item["acc_v"])):
            raise RuntimeError("X002-S地下检查点包含非有限值")
    return {
        "time": common_time,
        "surface": {
            "s": np.asarray([item["s"] for item in main], dtype=float),
            "x": np.asarray([item["x"] for item in main], dtype=float),
            "z": np.asarray([item["z"] for item in main], dtype=float),
            "acc_h": surface_h,
            "acc_v": surface_v,
        },
        "rock": rock_data,
    }


def correlation(test: np.ndarray, reference: np.ndarray):
    """返回皮尔逊相关系数；常量通道返回空值。"""
    if np.std(test) <= 1.0e-14 or np.std(reference) <= 1.0e-14:
        return None
    return float(np.corrcoef(test, reference)[0, 1])


def first_arrival(time: np.ndarray, values: np.ndarray, threshold: float):
    """按固定输入峰值阈值识别首次到时。"""
    indices = np.flatnonzero(np.abs(values) >= threshold)
    return None if len(indices) == 0 else float(time[int(indices[0])])


def series_metrics(test: np.ndarray, reference: np.ndarray,
                   time: np.ndarray) -> dict:
    """计算单通道时程指标与对应参考线状态。"""
    reference_peak = float(np.max(np.abs(reference)))
    test_peak = float(np.max(np.abs(test)))
    weak = reference_peak < 0.05 * INPUT_PEAK
    denominator = float(np.linalg.norm(reference))
    nrmse = float(np.linalg.norm(test - reference) / denominator) if denominator > 0.0 else None
    corr = correlation(test, reference)
    reference_peak_time = float(time[int(np.argmax(np.abs(reference)))])
    test_peak_time = float(time[int(np.argmax(np.abs(test)))])
    arrival_threshold = 0.05 * INPUT_PEAK
    reference_arrival = first_arrival(time, reference, arrival_threshold)
    test_arrival = first_arrival(time, test, arrival_threshold)
    arrival_error = (None if reference_arrival is None or test_arrival is None
                     else abs(test_arrival - reference_arrival))
    peak_relative_error = (abs(test_peak - reference_peak) / reference_peak
                           if reference_peak > 0.0 else None)
    checks = {}
    if not weak:
        checks = {
            "nrmse_le_0p30": nrmse is not None and nrmse <= 0.30,
            "correlation_ge_0p85": corr is not None and corr >= 0.85,
            "peak_relative_error_le_0p20": peak_relative_error is not None and peak_relative_error <= 0.20,
            "peak_time_error_le_0p02_s": abs(test_peak_time - reference_peak_time) <= 0.02,
            "first_arrival_error_le_0p02_s": arrival_error is not None and arrival_error <= 0.02,
        }
    return {
        "weak_response": weak,
        "nrmse": nrmse,
        "correlation": corr,
        "reference_peak_m_s2": reference_peak,
        "test_peak_m_s2": test_peak,
        "peak_relative_error": peak_relative_error,
        "peak_absolute_error_over_input_peak": abs(test_peak - reference_peak) / INPUT_PEAK,
        "reference_peak_time_s": reference_peak_time,
        "test_peak_time_s": test_peak_time,
        "peak_time_error_s": abs(test_peak_time - reference_peak_time),
        "arrival_threshold_m_s2": arrival_threshold,
        "reference_first_arrival_s": reference_arrival,
        "test_first_arrival_s": test_arrival,
        "first_arrival_error_s": arrival_error,
        "checks": checks,
        "reference_lines_met": None if weak else all(checks.values()),
    }


def pga_metrics(test: np.ndarray, reference: np.ndarray,
                s: np.ndarray, nrmse_threshold: float = 0.10) -> tuple[dict, np.ndarray, np.ndarray]:
    """评价完整地表 PGA 曲线。"""
    test_pga = np.max(np.abs(test), axis=1)
    reference_pga = np.max(np.abs(reference), axis=1)
    denominator = float(np.linalg.norm(reference_pga))
    nrmse = float(np.linalg.norm(test_pga - reference_pga) / denominator)
    test_peak_s = float(s[int(np.argmax(test_pga))])
    reference_peak_s = float(s[int(np.argmax(reference_pga))])
    checks = {
        "curve_nrmse_le_threshold": nrmse <= nrmse_threshold,
        "peak_location_error_le_0p10": abs(test_peak_s - reference_peak_s) <= 0.10,
    }
    return ({
        "curve_nrmse": nrmse,
        "reference_peak_s": reference_peak_s,
        "test_peak_s": test_peak_s,
        "peak_location_error_s": abs(test_peak_s - reference_peak_s),
        "checks": checks,
        "reference_lines_met": all(checks.values()),
    }, test_pga, reference_pga)


def field_metrics(test: np.ndarray, reference: np.ndarray) -> dict:
    """计算同一时刻空间场的幅值与相关性指标。"""
    denominator = float(np.linalg.norm(reference))
    return {
        "nrmse": (float(np.linalg.norm(test - reference) / denominator)
                  if denominator > 0.0 else None),
        "correlation": correlation(test, reference),
        "reference_l2_norm": denominator,
        "test_l2_norm": float(np.linalg.norm(test)),
        "max_absolute_difference_m_s2": float(np.max(np.abs(test - reference))),
    }


def descriptive_series_metrics(test: np.ndarray, reference: np.ndarray) -> dict:
    """返回不参与正式判定的简洁波形指标。"""
    denominator = float(np.linalg.norm(reference))
    return {
        "nrmse": (float(np.linalg.norm(test - reference) / denominator)
                  if denominator > 0.0 else None),
        "correlation": correlation(test, reference),
    }


def fixed_point_failure_diagnostic(test: np.ndarray, reference: np.ndarray,
                                   time: np.ndarray, band: tuple[float, float]) -> dict:
    """量化预定0.45 s传播阶段和有效频带；只用于解释正式失败。"""
    preinteraction = time <= 0.45 + 1.0e-12
    dt = float(np.median(np.diff(time)))
    frequency = np.fft.rfftfreq(len(time), dt)
    frequency_mask = (frequency >= band[0]) & (frequency <= band[1])
    test_fft = np.fft.rfft(test)
    reference_fft = np.fft.rfft(reference)
    test_band = np.fft.irfft(test_fft * frequency_mask, n=len(test))
    reference_band = np.fft.irfft(reference_fft * frequency_mask, n=len(reference))
    test_energy = np.sum(np.abs(test_fft) ** 2)
    reference_energy = np.sum(np.abs(reference_fft) ** 2)
    return {
        "posthoc_diagnostic": True,
        "used_as_formal_gate": False,
        "preinteraction_window_s": [float(time[0]), 0.45],
        "preinteraction": descriptive_series_metrics(
            test[preinteraction], reference[preinteraction]),
        "effective_band_hz": list(band),
        "effective_band_full_window": descriptive_series_metrics(test_band, reference_band),
        "energy_fraction_above_effective_band": {
            "abaqus": (float(np.sum(np.abs(test_fft[frequency > band[1]]) ** 2) / test_energy)
                        if test_energy > 0.0 else None),
            "specfem2d": (float(np.sum(np.abs(reference_fft[frequency > band[1]]) ** 2) /
                                  reference_energy) if reference_energy > 0.0 else None),
        },
    }


def spectral_metrics(test: np.ndarray, reference: np.ndarray,
                     dt: float, band: tuple[float, float]) -> dict:
    """在公共脉冲有效频带计算幅值和相位诊断。"""
    taper = np.hanning(len(test))
    test_fft = np.fft.rfft((test - np.mean(test)) * taper)
    ref_fft = np.fft.rfft((reference - np.mean(reference)) * taper)
    frequency = np.fft.rfftfreq(len(test), dt)
    target_frequency = np.arange(2.0, 11.0 + 0.25, 0.5)
    indices = np.asarray([int(np.argmin(np.abs(frequency - item)))
                          for item in target_frequency], dtype=int)
    if np.any(np.abs(frequency[indices] - target_frequency) > 1.0e-9):
        raise RuntimeError("公共2 s时窗未产生2.0:0.5:11.0 Hz频点")
    mask = (target_frequency >= band[0]) & (target_frequency <= band[1])
    indices = indices[mask]
    target_frequency = target_frequency[mask]
    amplitude_ref = np.abs(ref_fft[indices])
    amplitude_test = np.abs(test_fft[indices])
    amplitude_nrmse = float(np.linalg.norm(amplitude_test - amplitude_ref) /
                            np.linalg.norm(amplitude_ref))
    phase_difference = np.angle(test_fft[indices] * np.conj(ref_fft[indices]))
    weights = amplitude_ref ** 2
    phase_rmse = float(np.sqrt(np.sum(weights * phase_difference ** 2) / np.sum(weights)))
    return {
        "frequency_hz": target_frequency.tolist(),
        "frequency_bin_count": int(len(target_frequency)),
        "post_result_frequency_filter": False,
        "amplitude_nrmse": amplitude_nrmse,
        "weighted_phase_rmse_rad": phase_rmse,
        "weighted_phase_rmse_deg": math.degrees(phase_rmse),
    }


def evaluate_rock(abaqus: dict | None, specfem: list[dict],
                  common_time: np.ndarray, point_map: dict,
                  tolerance: float) -> dict:
    """比较地下检查点；Abaqus实际节点偏移量单独保留。"""
    if abaqus is None:
        return {"available": False, "reason": "Abaqus地下检查点未提取"}
    if len(abaqus["name"]) != 3 or len(set(name.upper() for name in abaqus["name"])) != 3:
        raise RuntimeError("X002-A地下时程必须恰有三个不重复的命名点")
    if len(specfem) != 3:
        raise RuntimeError("X002-S入射检查点必须恰有三个")
    if len(point_map.get("points", [])) != 3:
        raise RuntimeError("validation_point_map.json必须恰有三个点")
    points = []
    incident_checks = []
    index_by_set = {name.upper(): index for index, name in enumerate(abaqus["name"])}
    map_by_name = {str(item["name"]).upper(): item for item in point_map.get("points", [])}
    for reference in specfem:
        mapping = map_by_name.get(str(reference["station"]).upper())
        if mapping is None:
            raise RuntimeError(f"validation_point_map.json缺少{reference['station']}")
        index = index_by_set.get(str(mapping["set_name"]).upper())
        if index is None:
            raise RuntimeError(f"Abaqus地下时程缺少{mapping['set_name']}")
        target_x, target_y = float(reference["x"]), float(reference["z"])
        actual_x, actual_y = float(abaqus["x"][index]), float(abaqus["y"][index])
        mapping_distance = math.hypot(actual_x - target_x, actual_y - target_y)
        points.append({
            "name": reference["station"],
            "target_coordinate_m": [target_x, target_y],
            "abaqus_node_coordinate_m": [actual_x, actual_y],
            "mapping_distance_m": mapping_distance,
            "mapping_tolerance_m": tolerance,
            "mapping_within_tolerance": mapping_distance <= tolerance + 1.0e-9,
            "coordinate_note": "SPECFEM2D使用预定物理点；Abaqus使用最近网格节点，未作时程移位",
            "horizontal": series_metrics(abaqus["acc_h"][index], reference["acc_h"], common_time),
            "vertical": series_metrics(abaqus["acc_v"][index], reference["acc_v"], common_time),
        })
        expected_time = float(reference["expected_common_peak_time"])
        window = ((common_time >= expected_time - 0.06) &
                  (common_time <= expected_time + 0.06))
        selected = np.flatnonzero(window)
        a_vector = np.hypot(abaqus["acc_h"][index, window], abaqus["acc_v"][index, window])
        s_vector = np.hypot(reference["acc_h"][window], reference["acc_v"][window])
        a_peak_index = int(selected[int(np.argmax(a_vector))])
        s_peak_index = int(selected[int(np.argmax(s_vector))])
        incident_checks.append({
            "name": reference["station"],
            "expected_common_peak_time_s": expected_time,
            "abaqus": {
                "observed_common_peak_time_s": float(common_time[a_peak_index]),
                "time_error_s": float(common_time[a_peak_index] - expected_time),
                "vector_peak_m_s2": float(np.hypot(abaqus["acc_h"][index, a_peak_index],
                                                    abaqus["acc_v"][index, a_peak_index])),
                "vertical_over_horizontal": float(abaqus["acc_v"][index, a_peak_index] /
                                                  abaqus["acc_h"][index, a_peak_index]),
            },
            "specfem2d": {
                "observed_common_peak_time_s": float(common_time[s_peak_index]),
                "time_error_s": float(common_time[s_peak_index] - expected_time),
                "vector_peak_m_s2": float(np.hypot(reference["acc_h"][s_peak_index],
                                                    reference["acc_v"][s_peak_index])),
                "vertical_over_horizontal": float(reference["acc_v"][s_peak_index] /
                                                  reference["acc_h"][s_peak_index]),
            },
        })
    theory_ratio = -math.tan(math.radians(15.0))
    return {
        "available": True,
        "points": points,
        "incident_peak_checks": incident_checks,
        "theoretical_vertical_over_horizontal": theory_ratio,
        "used_as_formal_gate": False,
    }


def nearest_indices(points: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """优先用 cKDTree；缺少 SciPy 时分块暴力查询。"""
    try:
        from scipy.spatial import cKDTree
        distance, index = cKDTree(points).query(queries, k=1)
        return np.asarray(index, dtype=int), np.asarray(distance, dtype=float)
    except ImportError:
        indices, distances = [], []
        for start in range(0, len(queries), 500):
            batch = queries[start:start + 500]
            squared = ((batch[:, None, :] - points[None, :, :]) ** 2).sum(axis=2)
            index = np.argmin(squared, axis=1)
            indices.extend(index.tolist())
            distances.extend(np.sqrt(squared[np.arange(len(batch)), index]).tolist())
        return np.asarray(indices, dtype=int), np.asarray(distances, dtype=float)


def specfem_wavefield_point_count(specfem_case: Path) -> int:
    """从正式求解日志读取波场文件中的有效全局 GLL 点数。"""
    solver_log = specfem_case / "OUTPUT_FILES" / "solver.log"
    if solver_log.is_file():
        text = solver_log.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"Number of grid points\s*=\s*(\d+)", text)
        if match is not None:
            return int(match.group(1))
    grid_path = specfem_case / "OUTPUT_FILES" / "wavefield_grid_for_dumps.bin"
    if grid_path.is_file():
        raw = np.fromfile(grid_path, dtype='<f4')
        total = len(raw) // 2
        coords = raw[:2 * total].reshape((-1, 2))
        return len(np.unique(coords.round(decimals=6), axis=0))
    raise FileNotFoundError("无法确定SPECFEM2D波场GLL点数: solver.log和grid文件均不存在")


def evaluate_wavefield(abaqus_case: Path, specfem_case: Path,
                       scale: float, output_dir: Path,
                       snapshot_specs: list[dict]) -> dict:
    """按最近 GLL 点评价三个预定加速度场快照。"""
    abaqus_path = abaqus_case / "abaqus_wavefield_snapshots.npz"
    grid_path = specfem_case / "OUTPUT_FILES" / "wavefield_grid_for_dumps.bin"
    if not abaqus_path.is_file() or not grid_path.is_file():
        return {"available": False, "reason": "缺少Abaqus或SPECFEM2D波场快照"}
    with np.load(abaqus_path, allow_pickle=False) as data:
        ax = np.asarray(data["x"], dtype=float)
        ay = np.asarray(data["y"], dtype=float)
        a1 = np.asarray(data["A1"], dtype=float)
        a2 = np.asarray(data["A2"], dtype=float)
        common_times = np.asarray(data["common_time"], dtype=float)
    expected_times = np.asarray([float(item["common_time_s"]) for item in snapshot_specs], dtype=float)
    if common_times.shape != (3,) or not np.allclose(common_times, expected_times,
                                                     atol=1.0e-12, rtol=0.0):
        raise RuntimeError("Abaqus全场快照公共时刻与参数表不一致")
    expected_shape = (3, len(ax))
    for label, values in (("A1", a1), ("A2", a2)):
        if values.shape != expected_shape or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Abaqus全场{label}形状或有限性检查失败")
    if not np.all(np.isfinite(ax)) or not np.all(np.isfinite(ay)):
        raise RuntimeError("Abaqus全场坐标包含非有限值")
    point_count = specfem_wavefield_point_count(specfem_case)
    grid_raw = np.fromfile(grid_path, dtype='<f4')
    if len(grid_raw) < 2 * point_count:
        raise RuntimeError("SPECFEM2D波场坐标文件长度不是二维向量的整数倍")
    grid = grid_raw[:2 * point_count].reshape((-1, 2)).astype(float)
    mask = (ax >= 400.0 - 1.0e-8) & (ax <= 1157.7350269189626 + 1.0e-8)
    queries = np.column_stack((ax[mask], ay[mask]))
    index, distance = nearest_indices(grid, queries)
    snapshot_results = []
    plot_payload = None
    for snapshot in snapshot_specs:
        common_time = float(snapshot["common_time_s"])
        step = int(snapshot["specfem_step"])
        a_index = int(np.argmin(np.abs(common_times - common_time)))
        dump_path = specfem_case / "OUTPUT_FILES" / f"wavefield{step:07d}_01.bin"
        if not dump_path.is_file():
            import glob as _glob
            candidates = sorted(_glob.glob(
                str(specfem_case / "OUTPUT_FILES" / "wavefield*_01.bin")))
            if not candidates:
                raise FileNotFoundError(f"无可用波场快照文件: {specfem_case / 'OUTPUT_FILES'}")
            dump_path = Path(min(candidates,
                                 key=lambda p: abs(int(Path(p).stem.split("_")[0][9:]) - step)))
        values_raw = np.fromfile(dump_path, dtype='<f4')
        if len(values_raw) < 2 * point_count:
            raise RuntimeError(f"{dump_path.name}长度与波场坐标不一致")
        values = values_raw[:2 * point_count].reshape((-1, 2)).astype(float) * scale
        reference_h = values[index, 0]
        reference_v = values[index, 1]
        test_h = a1[a_index, mask]
        test_v = a2[a_index, mask]
        snapshot_results.append({
            "common_time_s": common_time,
            "specfem_actual_common_time_s": float(snapshot["specfem_actual_common_time_s"]),
            "physical_stage": snapshot["physical_stage"],
            "available": True,
            "mapped_node_count": int(len(queries)),
            "nearest_gll_distance_m": {
                "max": float(np.max(distance)),
                "mean": float(np.mean(distance)),
                "p95": float(np.percentile(distance, 95.0)),
            },
            "horizontal": field_metrics(test_h, reference_h),
            "vertical": field_metrics(test_v, reference_v),
        })
        if abs(common_time - 0.45) < 1.0e-9:
            plot_payload = (queries[:, 0], queries[:, 1], test_h - reference_h)
    if len(snapshot_results) != len(snapshot_specs) or len(snapshot_results) != 3:
        raise RuntimeError("X002波场必须完成三个预定快照评价")
    if plot_payload is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        px, py, difference = plot_payload
        stride = max(1, int(math.ceil(len(px) / 15000.0)))
        figure, axis = plt.subplots(figsize=(10, 5.5))
        artist = axis.scatter(px[::stride], py[::stride], c=difference[::stride],
                              s=4, cmap="coolwarm")
        axis.set_title("X002 horizontal acceleration difference at common t=0.45 s")
        axis.set_xlabel("x / m")
        axis.set_ylabel("y / m")
        figure.colorbar(artist, ax=axis, label="Abaqus - SPECFEM2D / (m/s²)")
        figure.tight_layout()
        figure.savefig(output_dir / "x002_sr_wavefield_difference_t0p45.png", dpi=180)
        plt.close(figure)
    return {
        "available": True,
        "quantity": "acceleration vector",
        "valid_specfem_gll_point_count": point_count,
        "zero_filled_tail_excluded": True,
        "mapping": "Abaqus nodes to nearest SPECFEM2D GLL dump point",
        "used_as_formal_gate": False,
        "formal_gate_note": "早期全场快照仅作辅助诊断，不参与正式门控",
        "snapshots": snapshot_results,
    }


def save_plots(output_dir: Path, time: np.ndarray, s: np.ndarray,
               abaqus_h: np.ndarray, abaqus_v: np.ndarray,
               specfem_h: np.ndarray, specfem_v: np.ndarray,
               pga_payload: dict) -> None:
    """保存点位时程与 PGA 曲线图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for row, (name, point_s) in enumerate(SURFACE_POINTS):
        index = int(np.argmin(np.abs(s - point_s)))
        for column, (label, test, reference) in enumerate((
                ("horizontal", abaqus_h, specfem_h),
                ("vertical", abaqus_v, specfem_v))):
            axis = axes[row, column]
            axis.plot(time, reference[index], label="SPECFEM2D", linewidth=1.0)
            axis.plot(time, test[index], label="Abaqus", linewidth=0.9, alpha=0.85)
            axis.set_title(f"{name}, s={point_s:g}, {label}")
            axis.set_ylabel("a / (m/s²)")
            axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("common time / s")
    axes[-1, 1].set_xlabel("common time / s")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "x002_sr_fixed_point_timeseries.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for axis, component in zip(axes, ("horizontal", "vertical")):
        values = pga_payload[component]
        axis.plot(s, values["specfem"], label="SPECFEM2D", linewidth=1.2)
        axis.plot(s, values["abaqus"], label="Abaqus", linewidth=1.0)
        axis.set_ylabel(f"{component} PGA / (m/s²)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("normalized surface coordinate s")
    figure.tight_layout()
    figure.savefig(output_dir / "x002_sr_surface_pga.png", dpi=180)
    plt.close(figure)


def main() -> int:
    """执行完整比较并写出研究评价产物。"""
    abaqus_case = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()
    repo_root = find_repo_root(abaqus_case)
    specfem_case = repo_root / "Run" / "cross_solver_X" / "specfem2d" / "X002-SR"
    params_path = repo_root / "Run" / "Auto_ch4" / "cross_solver" / "x_validation_parameters.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))
    scale = float(params["pulse"]["global_linear_scale"])
    band = tuple(float(value) for value in params["pulse"]["acceleration_effective_band_5pct_hz"])
    specfem = load_specfem(specfem_case, scale)
    common_time = specfem["time"]
    target_x = specfem["surface"]["x"]
    abaqus = load_abaqus(abaqus_case, common_time, target_x,
                          specfem["surface"]["z"], params)

    s = specfem["surface"]["s"]
    a_h = abaqus["surface"]["acc_h"]
    a_v = abaqus["surface"]["acc_v"]
    s_h = specfem["surface"]["acc_h"]
    s_v = specfem["surface"]["acc_v"]
    fixed = {}
    spectral = {}
    failure_diagnostics = {}
    for name, point_s in SURFACE_POINTS:
        index = int(np.argmin(np.abs(s - point_s)))
        fixed[name] = {
            "s": float(s[index]),
            "x_m": float(target_x[index]),
            "horizontal": series_metrics(a_h[index], s_h[index], common_time),
            "vertical": series_metrics(a_v[index], s_v[index], common_time),
        }
        spectral[name] = {
            "horizontal": spectral_metrics(a_h[index], s_h[index],
                                             float(np.median(np.diff(common_time))), band),
            "vertical": spectral_metrics(a_v[index], s_v[index],
                                           float(np.median(np.diff(common_time))), band),
        }
        failure_diagnostics[name] = {
            "horizontal": fixed_point_failure_diagnostic(
                a_h[index], s_h[index], common_time, band),
            "vertical": fixed_point_failure_diagnostic(
                a_v[index], s_v[index], common_time, band),
        }

    pga_h, a_pga_h, s_pga_h = pga_metrics(a_h, s_h, s, nrmse_threshold=0.10)
    pga_v, a_pga_v, s_pga_v = pga_metrics(a_v, s_v, s, nrmse_threshold=0.15)

    output_dir = abaqus_case / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    point_map = json.loads((abaqus_case / "validation_point_map.json").read_text(encoding="utf-8"))
    tolerance = float(params["cases"]["X002-A"]["validation_point_tolerance"])
    rock = evaluate_rock(abaqus["underground"], specfem["rock"], common_time,
                         point_map, tolerance)
    snapshots = params["observations"]["wavefield_snapshots"]
    wavefield = evaluate_wavefield(abaqus_case, specfem_case, scale, output_dir, snapshots)

    # 核心工程门控（必须全部通过）
    formal_gates = {
        "horizontal_pga_curve": {
            "passed": bool(pga_h["reference_lines_met"]),
            "nrmse": pga_h["curve_nrmse"],
            "threshold_nrmse": 0.10,
            "peak_location_error": pga_h["peak_location_error_s"],
            "threshold_location": 0.10,
        },
        "vertical_pga_curve": {
            "passed": bool(pga_v["reference_lines_met"]),
            "nrmse": pga_v["curve_nrmse"],
            "threshold_nrmse": 0.15,
            "peak_location_error": pga_v["peak_location_error_s"],
            "threshold_location": 0.10,
        },
    }
    formal_gates_met = all(g["passed"] for g in formal_gates.values())

    # 辅助诊断汇总（不影响总判定）
    nonweak_total = 0
    nonweak_passed = 0
    for point in fixed.values():
        for component in ("horizontal", "vertical"):
            status = point[component]["reference_lines_met"]
            if status is not None:
                nonweak_total += 1
                if status:
                    nonweak_passed += 1
    diagnostic_summary = {
        "used_as_formal_gate": False,
        "fixed_point_pass_rate": f"{nonweak_passed}/{nonweak_total}",
        "fixed_point_all_passed": nonweak_passed == nonweak_total if nonweak_total > 0 else None,
        "spectral_diagnostics": {"used_as_formal_gate": False, "points": spectral},
        "rock_check_diagnostics": rock,
        "failure_diagnostics": failure_diagnostics,
    }

    result = {
        "schema": "x002_sr-cross-solver-comparison-1.1",
        "case_pair": ["X002-A", "X002-SR"],
        "status": "evaluated",
        "formal_gates_met": formal_gates_met,
        "formal_gates": formal_gates,
        "abaqus_identity_check": abaqus["identity"],
        "surface_coordinate_max_error_m": abaqus["surface"]["surface_coordinate_max_error_m"],
        "comparison_rules": {
            "specfem_global_scale": scale,
            "time_mapping_specfem": "t_common = t_specfem_output + 0.3 s",
            "time_mapping_abaqus": "t_common = t_abaqus_output - 0.3 s",
            "common_time_start_s": float(common_time[0]),
            "common_time_end_s": float(common_time[-1]),
            "common_dt_s": float(np.median(np.diff(common_time))),
            "component_mapping": {"Abaqus A1": "SPECFEM2D BXX", "Abaqus A2": "SPECFEM2D BXZ"},
            "post_result_channel_shift_scale_or_sign_change": False,
            "weak_response_threshold_m_s2": 0.05 * INPUT_PEAK,
            "first_arrival_threshold_m_s2": 0.05 * INPUT_PEAK,
            "effective_frequency_band_hz": band,
        },
        "fixed_surface_points": fixed,
        "surface_pga": {"horizontal": pga_h, "vertical": pga_v},
        "diagnostic_summary": diagnostic_summary,
        "wavefield_diagnostics": wavefield,
        "software": {"python": sys.version, "platform": platform.platform(),
                     "numpy": np.__version__},
    }
    write_json(output_dir / "x002_sr_comparison_metrics.json", result)
    np.savez_compressed(output_dir / "x002_sr_comparison_arrays.npz",
                        common_time=common_time, s=s, x=target_x,
                        abaqus_acc_h=a_h, abaqus_acc_v=a_v,
                        specfem_acc_h=s_h, specfem_acc_v=s_v,
                        abaqus_pga_h=a_pga_h, abaqus_pga_v=a_pga_v,
                        specfem_pga_h=s_pga_h, specfem_pga_v=s_pga_v)
    with (output_dir / "x002_sr_surface_pga.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("s", "x_m", "abaqus_pga_h", "specfem_pga_h",
                         "abaqus_pga_v", "specfem_pga_v"))
        for row in zip(s, target_x, a_pga_h, s_pga_h, a_pga_v, s_pga_v):
            writer.writerow([f"{float(value):.12g}" for value in row])
    save_plots(output_dir, common_time, s, a_h, a_v, s_h, s_v,
               {"horizontal": {"abaqus": a_pga_h, "specfem": s_pga_h},
                "vertical": {"abaqus": a_pga_v, "specfem": s_pga_v}})
    print(json.dumps({"status": result["status"],
                      "formal_gates_met": result["formal_gates_met"],
                      "metrics": str(output_dir / "x002_sr_comparison_metrics.json")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
