# -*- coding: utf-8 -*-
"""生成 X001-S、X002-S 与 X002-SR 的公共参数、Gmsh 网格和 SPECFEM2D 输入。

脚本只使用 Python 标准库。Gmsh 与 SPECFEM2D 由 WSL2 环境提供，正式运行前会先完成
公共脉冲、初始包络、四边形网格、材料顺序、边界覆盖和时间步检查。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


G_STANDARD = 9.80665
F0 = 4.0
PUBLIC_DT = 0.001
TOTAL_TIME = 2.0
T0 = 1.2 / F0
INCIDENCE_DEG = 15.0
INCIDENCE_RAD = math.radians(INCIDENCE_DEG)
GLL_MIN_SPACING_RATIO = 0.1726731646460114
PULSE_Q_CENTER = 50.0
CPML_DIAGNOSTIC_WIDTH = 100.0
ABAQUS_PHASE_ORIGIN_X = PULSE_Q_CENTER / math.sin(INCIDENCE_RAD)
ABAQUS_TAIL_SECONDS = T0
WAVEFIELD_SNAPSHOTS = (
    (0.30, 3000, "incident peak plane q=650 m, before it sweeps most of the main surface"),
    (0.45, 4500, "incident peak plane q=950 m, just after it sweeps the main surface"),
    (0.60, 6000, "incident peak plane q=1250 m, post-interaction scattered-wave stage"),
)

H = 100.0
SLOPE_DEG = 60.0
CREST_WINDOW = 4.0
TOE_WINDOW = 3.0
SIDE_CLEARANCE = 4.0
BASE_DEPTH = 6.0
LEFT_FLAT = (CREST_WINDOW + SIDE_CLEARANCE) * H
SLOPE_WIDTH = H / math.tan(math.radians(SLOPE_DEG))
TOE_X = LEFT_FLAT + SLOPE_WIDTH
RIGHT_FLAT = (TOE_WINDOW + SIDE_CLEARANCE) * H
DOMAIN_LENGTH = TOE_X + RIGHT_FLAT
TOE_Y = BASE_DEPTH * H
CREST_Y = TOE_Y + H
COVER_INTERFACE_Y = CREST_Y - 140.0

BEDROCK = {
    "name": "bedrock",
    "material_id": 1,
    "density": 2500.0,
    "vs": 2000.0,
    "poisson_ratio": 0.30,
    "target_xi": 0.0005,
}
COVER = {
    "name": "cover",
    "material_id": 2,
    "density": 2125.0,
    "vs": 600.0,
    "poisson_ratio": 0.35,
    "thickness_below_crest": 140.0,
    "target_xi": 0.03,
}

CASE_CONFIG = {
    "X001-S": {
        "parameter_source": "H004",
        "has_cover": False,
        "target_edge": 15.0,
        "dt": 0.0001,
    },
    "X002-S": {
        "parameter_source": "P061",
        "has_cover": True,
        "target_edge": 15.0,
        "dt": 0.00005,
    },
    "X002-SR": {
        "parameter_source": "P061",
        "has_cover": True,
        "target_edge": 7.5,
        "dt": 0.00005,
    },
}


def vp_from_vs_nu(vs: float, nu: float) -> float:
    """由剪切波速和泊松比换算纵波波速。"""
    return vs * math.sqrt(2.0 * (1.0 - nu) / (1.0 - 2.0 * nu))


def elastic_properties(material: dict) -> dict:
    """返回各向同性线弹性材料的派生量。"""
    rho = material["density"]
    vs = material["vs"]
    nu = material["poisson_ratio"]
    shear = rho * vs * vs
    young = 2.0 * shear * (1.0 + nu)
    lame = 2.0 * shear * nu / (1.0 - 2.0 * nu)
    result = dict(material)
    result.update({"vp": vp_from_vs_nu(vs, nu), "shear_modulus": shear,
                   "young_modulus": young, "lame_lambda": lame})
    return result


def rayleigh_at_frequency(target_xi: float, f1: float, f2: float, frequency: float) -> dict:
    """按主模型的双频拟合公式计算指定频率处的实际瑞利阻尼。"""
    w1 = 2.0 * math.pi * f1
    w2 = 2.0 * math.pi * f2
    w = 2.0 * math.pi * frequency
    alpha = 2.0 * target_xi * w1 * w2 / (w1 + w2)
    beta = 2.0 * target_xi / (w1 + w2)
    xi = alpha / (2.0 * w) + beta * w / 2.0
    return {"f1": f1, "f2": f2, "alpha": alpha, "beta": beta,
            "xi_at_4hz": xi, "qmu": 1.0 / (2.0 * xi), "qkappa": 1.0 / (2.0 * xi)}


def ricker_displacement(tau: float) -> float:
    """SPECFEM2D initialfield 使用的单位位移 Ricker。"""
    a = math.pi * math.pi * F0 * F0
    return (1.0 - 2.0 * a * tau * tau) * math.exp(-a * tau * tau)


def ricker_velocity(tau: float) -> float:
    """单位位移 Ricker 的一阶时间导数。"""
    a = math.pi * math.pi * F0 * F0
    return -2.0 * a * tau * (3.0 - 2.0 * a * tau * tau) * math.exp(-a * tau * tau)


def ricker_acceleration(tau: float) -> float:
    """单位位移 Ricker 的二阶时间导数。"""
    a = math.pi * math.pi * F0 * F0
    return -2.0 * a * (3.0 - 12.0 * a * tau * tau + 4.0 * a * a * tau ** 4) * math.exp(-a * tau * tau)


def pulse_characteristics() -> dict:
    """计算公共脉冲幅值、包络阈值和加速度有效频带。"""
    a = math.pi * math.pi * F0 * F0
    peak_acceleration = 6.0 * a
    scale = 0.1 * G_STANDARD / peak_acceleration

    dt_scan = 1.0e-5
    samples = int(1.0 / dt_scan) + 1
    values = {"displacement": [], "velocity": [], "acceleration": []}
    functions = {"displacement": ricker_displacement, "velocity": ricker_velocity,
                 "acceleration": ricker_acceleration}
    for i in range(samples):
        tau = i * dt_scan
        for name, func in functions.items():
            values[name].append(abs(func(tau)))
    peaks = {name: max(series) for name, series in values.items()}
    tail_thresholds = {}
    for name, series in values.items():
        last_over = 0
        limit = 0.01 * peaks[name]
        for i, value in enumerate(series):
            if value > limit:
                last_over = i
        tail_thresholds[name] = (last_over + 1) * dt_scan

    df = 1.0e-4
    spectrum = []
    for i in range(int(20.0 / df) + 1):
        frequency = i * df
        amplitude = frequency ** 4 * math.exp(-(frequency / F0) ** 2)
        spectrum.append((frequency, amplitude))
    peak_spectrum = max(value for _, value in spectrum)
    active = [frequency for frequency, value in spectrum if value >= 0.05 * peak_spectrum]
    return {
        "definition": "unit displacement Ricker and its analytical second derivative",
        "f0_hz": F0,
        "automatic_t0_s": T0,
        "unscaled_acceleration_peak_m_s2": peak_acceleration,
        "global_linear_scale": scale,
        "scaled_sv_vector_peak_m_s2": scale * peak_acceleration,
        "scaled_sv_horizontal_peak_m_s2": scale * peak_acceleration * math.cos(INCIDENCE_RAD),
        "scaled_sv_vertical_abs_peak_m_s2": scale * peak_acceleration * math.sin(INCIDENCE_RAD),
        "one_percent_tail_time_s": tail_thresholds,
        "acceleration_spectrum_peak_hz": math.sqrt(2.0) * F0,
        "acceleration_effective_band_5pct_hz": [active[0], active[-1]],
        "time_mapping": "t_common = t_specfem_output + 0.3 s",
        "time_mapping_specfem": "t_common = t_specfem_output + 0.3 s",
        "time_mapping_abaqus": "t_common = t_abaqus_output - 0.3 s",
        "abaqus_phase_origin_x_m": ABAQUS_PHASE_ORIGIN_X,
        "abaqus_tail_seconds": ABAQUS_TAIL_SECONDS,
        "amplitude_rule": "SPECFEM2D unit initial-field outputs are multiplied once by global_linear_scale; no channel-wise scaling",
    }


def q_coordinate(x: float, z: float) -> float:
    """返回斜入射平面波沿传播法向的空间坐标。"""
    return math.sin(INCIDENCE_RAD) * x + math.cos(INCIDENCE_RAD) * z


def max_normalized_on_q_interval(q_min: float, q_max: float, peaks: dict, n: int = 20000) -> dict:
    """计算初始时刻给定法向区间内三类场量的最大归一化幅值。"""
    functions = {"displacement": ricker_displacement, "velocity": ricker_velocity,
                 "acceleration": ricker_acceleration}
    result = {name: 0.0 for name in functions}
    for i in range(n + 1):
        q = q_min + (q_max - q_min) * i / float(n)
        tau = (PULSE_Q_CENTER - q) / BEDROCK["vs"]
        for name, func in functions.items():
            result[name] = max(result[name], abs(func(tau)) / peaks[name])
    return result


def initial_envelope_report(pulse: dict) -> dict:
    """评价覆盖层初始包络，并量化原 C-PML 条件为何不成立。"""
    peaks = {
        "displacement": 1.0,
        "velocity": max(abs(ricker_velocity(i * 1.0e-5)) for i in range(100001)),
        "acceleration": pulse["unscaled_acceleration_peak_m_s2"],
    }
    cover_q = [q_coordinate(0.0, COVER_INTERFACE_Y), q_coordinate(DOMAIN_LENGTH, TOE_Y)]
    cover_metrics = max_normalized_on_q_interval(min(cover_q), max(cover_q), peaks)

    w = CPML_DIAGNOSTIC_WIDTH
    pml_intervals = {
        "left_inner": [q_coordinate(w, w), q_coordinate(w, CREST_Y)],
        "right_inner": [q_coordinate(DOMAIN_LENGTH - w, w), q_coordinate(DOMAIN_LENGTH - w, TOE_Y)],
        "bottom_inner": [q_coordinate(w, w), q_coordinate(DOMAIN_LENGTH - w, w)],
    }
    pml_metrics = {
        name: max_normalized_on_q_interval(min(interval), max(interval), peaks)
        for name, interval in pml_intervals.items()
    }
    cover_pass = max(cover_metrics.values()) < 0.01
    pml_pass = all(max(metrics.values()) < 0.01 for metrics in pml_metrics.values())

    source_x = LEFT_FLAT
    source_z = (PULSE_Q_CENTER + BEDROCK["vs"] * T0
                - math.sin(INCIDENCE_RAD) * source_x) / math.cos(INCIDENCE_RAD)
    if not (0.0 < source_z < COVER_INTERFACE_Y):
        raise RuntimeError("解析相位原点不在基岩内，需重新选择 PULSE_Q_CENTER")
    return {
        "initial_peak_plane_q_m": PULSE_Q_CENTER,
        "source_phase_origin": {"x": source_x, "z": source_z},
        "cover_q_interval_m": [min(cover_q), max(cover_q)],
        "cover_max_normalized_initial_field": cover_metrics,
        "cover_gate_lt_1pct": cover_pass,
        "cpml_diagnostic_width_m": w,
        "cpml_inner_boundary_q_intervals_m": pml_intervals,
        "cpml_inner_boundary_max_normalized_initial_field": pml_metrics,
        "cpml_gate_lt_1pct": pml_pass,
        "formal_boundary": "Stacey absorbing boundary with Bielak incident-field correction",
        "decision": "C-PML rejected for formal X cases" if not pml_pass else "manual review required",
        "reason": "the oblique infinite plane pulse is non-zero on a side/bottom absorbing layer, while v8.1.0 initializes PML memory variables to zero",
    }


def build_parameter_table() -> dict:
    """建立求解器无关的唯一 X 组参数表。"""
    bedrock = elastic_properties(BEDROCK)
    cover = elastic_properties(COVER)
    bed_damping = rayleigh_at_frequency(BEDROCK["target_xi"], 2.0, 10.0, F0)
    layer_f = COVER["vs"] / (4.0 * COVER["thickness_below_crest"])
    cover_damping = rayleigh_at_frequency(COVER["target_xi"], min(2.0, layer_f),
                                           max(10.0, 3.0 * layer_f), F0)
    bedrock["damping_mapping"] = bed_damping
    cover["damping_mapping"] = cover_damping
    pulse = pulse_characteristics()
    envelope = initial_envelope_report(pulse)
    cases = {}
    for case_name, config in CASE_CONFIG.items():
        dt = config["dt"]
        cases[case_name] = dict(config)
        cases[case_name].update({
            "nstep": int(round(TOTAL_TIME / dt)),
            "output_sample_stride": int(round(PUBLIC_DT / dt)),
            "nproc": 1,
            "ngllx": 5,
            "polynomial_degree": 4,
        })
    cases["X001-A"] = {
        "parameter_source": "H004",
        "solver": "Abaqus/Standard",
        "spatial_discretization": "CPE4R low-order finite element",
        "time_integration": "implicit TRANSIENT_FIDELITY",
        "has_cover": False,
        "mesh_size": 4.0,
        "mesh_auto": True,
        "mesh_graded": True,
        "dt": PUBLIC_DT,
        "analysis_time": TOTAL_TIME + ABAQUS_TAIL_SECONDS,
        "output_sample_stride": 1,
        "phase_origin_x": ABAQUS_PHASE_ORIGIN_X,
        "time_mapping": "t_common = t_abaqus_output - 0.3 s",
        "validation_point_tolerance": 6.0,
        "full_field_frequency": 50,
    }
    cases["X002-A"] = {
        "parameter_source": "P061",
        "solver": "Abaqus/Standard",
        "spatial_discretization": "CPE4R low-order finite element",
        "time_integration": "implicit TRANSIENT_FIDELITY",
        "has_cover": True,
        "mesh_size": 4.0,
        "mesh_auto": True,
        "mesh_graded": True,
        "dt": PUBLIC_DT,
        "analysis_time": TOTAL_TIME + ABAQUS_TAIL_SECONDS,
        "output_sample_stride": 1,
        "phase_origin_x": ABAQUS_PHASE_ORIGIN_X,
        "time_mapping": "t_common = t_abaqus_output - 0.3 s",
        "validation_point_tolerance": 6.0,
        "full_field_frequency": 50,
    }
    return {
        "schema": "x-cross-solver-parameters-1.1",
        "units": {"length": "m", "time": "s", "density": "kg/m3", "acceleration": "m/s2"},
        "physics": {"dimension": "2D", "kinematics": "plane strain", "constitutive": "linear elastic",
                    "incident_wave": "upgoing in-plane SV", "incidence_angle_from_vertical_deg": INCIDENCE_DEG},
        "geometry": {
            "slope_height": H, "slope_angle_deg": SLOPE_DEG,
            "crest_window_h": CREST_WINDOW, "toe_window_h": TOE_WINDOW,
            "side_clearance_h": SIDE_CLEARANCE, "base_depth_h": BASE_DEPTH,
            "left_flat_length": LEFT_FLAT, "slope_horizontal_width": SLOPE_WIDTH,
            "right_flat_length": RIGHT_FLAT, "domain_length": DOMAIN_LENGTH,
            "toe_surface_y": TOE_Y, "crest_surface_y": CREST_Y,
            "crest": [LEFT_FLAT, CREST_Y],
            "slope_midpoint": [(LEFT_FLAT + TOE_X) / 2.0, (CREST_Y + TOE_Y) / 2.0],
            "toe": [TOE_X, TOE_Y], "cover_interface_y": COVER_INTERFACE_Y,
            "cover_thickness_at_toe": TOE_Y - COVER_INTERFACE_Y,
        },
        "materials": {"bedrock": bedrock, "cover": cover},
        "pulse": pulse,
        "initial_envelope": envelope,
        "boundary": {
            "top": "traction-free",
            "left_right_bottom": "Stacey with Bielak correction",
            "pml_boundary_conditions": False,
            "add_bielak": {"left": True, "right": True, "bottom": True, "top": False},
            "x002_limit": "the built-in analytical correction uses the first bedrock material; X002 comparisons require a predeclared boundary-safe time window",
        },
        "observations": {
            "main_surface_grid": "s=-4.00:0.01:4.00 (801 points)",
            "s_definition": "crest platform (x-xcrest)/H; slope 0..1; toe platform 1+(x-xtoe)/H",
            "fixed_surface_points": ["crest", "slope_midpoint", "toe"],
            "rock_checks": [[500.0, 300.0], [700.0, 300.0], [500.0, 500.0]],
            "wavefield_snapshots": [
                {
                    "common_time_s": common_time,
                    "incident_peak_plane_q_m": PULSE_Q_CENTER + BEDROCK["vs"] * common_time,
                    "specfem_step": step,
                    "specfem_actual_common_time_s": (step - 1) * CASE_CONFIG["X001-S"]["dt"],
                    "abaqus_raw_time_s": common_time - (-T0),
                    "physical_stage": stage,
                }
                for common_time, step, stage in WAVEFIELD_SNAPSHOTS
            ],
            "public_sample_dt": PUBLIC_DT,
        },
        "cases": cases,
    }


def write_json(path: Path, data: dict) -> None:
    """以稳定的 UTF-8 格式写出 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_common_pulse(path: Path, pulse: dict) -> str:
    """写出 Abaqus 使用的公共标量加速度，两列依次为时间和加速度。"""
    scale = pulse["global_linear_scale"]
    lines = []
    n = int(round(TOTAL_TIME / PUBLIC_DT))
    for i in range(n + 1):
        time_value = i * PUBLIC_DT
        acceleration = scale * ricker_acceleration(time_value - T0)
        lines.append(f"{time_value:.6f}\t{acceleration:.12e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_point(lines: list[str], point_id: int, x: float, z: float, size: float) -> None:
    lines.append(f"Point({point_id}) = {{{x:.12f}, {z:.12f}, 0, {size:.12f}}};")


def geo_header() -> list[str]:
    return ["SetFactory(\"Built-in\");", "Mesh.MshFileVersion = 2.2;", "Mesh.ElementOrder = 1;",
            "Mesh.RecombineAll = 1;", "Mesh.Algorithm = 8;"]


def generate_homogeneous_geo(target_edge: float) -> str:
    """生成 H004 的三块结构化全四边形几何。"""
    slope_surface_length = H / math.sin(math.radians(SLOPE_DEG))
    nx = [int(math.ceil(LEFT_FLAT / target_edge)), int(math.ceil(slope_surface_length / target_edge)),
          int(math.ceil(RIGHT_FLAT / target_edge))]
    ny = int(math.ceil(CREST_Y / target_edge))
    xs = [0.0, LEFT_FLAT, TOE_X, DOMAIN_LENGTH]
    tops = [CREST_Y, CREST_Y, TOE_Y, TOE_Y]
    lines = geo_header()
    for i, x in enumerate(xs, 1):
        add_point(lines, i, x, 0.0, target_edge)
        add_point(lines, i + 4, x, tops[i - 1], target_edge)
    lines += [
        "Line(1) = {1,2};", "Line(2) = {2,3};", "Line(3) = {3,4};",
        "Line(4) = {1,5};", "Line(5) = {2,6};", "Line(6) = {3,7};", "Line(7) = {4,8};",
        "Line(8) = {5,6};", "Line(9) = {6,7};", "Line(10) = {7,8};",
        "Curve Loop(1) = {1,5,-8,-4};", "Plane Surface(1) = {1};",
        "Curve Loop(2) = {2,6,-9,-5};", "Plane Surface(2) = {2};",
        "Curve Loop(3) = {3,7,-10,-6};", "Plane Surface(3) = {3};",
    ]
    for curve, count in zip((1, 8, 2, 9, 3, 10), (nx[0], nx[0], nx[1], nx[1], nx[2], nx[2])):
        lines.append(f"Transfinite Curve {{{curve}}} = {count + 1};")
    lines.append(f"Transfinite Curve {{4,5,6,7}} = {ny + 1};")
    lines += ["Transfinite Surface {1,2,3};", "Recombine Surface {1,2,3};",
              "Physical Surface(\"M1\", 1) = {1,2,3};",
              "Physical Curve(\"Top\", 101) = {8,9,10};",
              "Physical Curve(\"Left\", 102) = {4};",
              "Physical Curve(\"Right\", 103) = {7};",
              "Physical Curve(\"Bottom\", 104) = {1,2,3};"]
    return "\n".join(lines) + "\n"


def generate_layered_geo(target_edge: float) -> str:
    """生成 P061 的水平界面六块结构化全四边形几何。"""
    slope_surface_length = H / math.sin(math.radians(SLOPE_DEG))
    nx = [int(math.ceil(LEFT_FLAT / target_edge)), int(math.ceil(slope_surface_length / target_edge)),
          int(math.ceil(RIGHT_FLAT / target_edge))]
    ny_bottom = int(math.ceil(COVER_INTERFACE_Y / target_edge))
    ny_top = int(math.ceil((CREST_Y - COVER_INTERFACE_Y) / target_edge))
    xs = [0.0, LEFT_FLAT, TOE_X, DOMAIN_LENGTH]
    tops = [CREST_Y, CREST_Y, TOE_Y, TOE_Y]
    lines = geo_header()
    for i, x in enumerate(xs, 1):
        add_point(lines, i, x, 0.0, target_edge)
        add_point(lines, i + 4, x, COVER_INTERFACE_Y, target_edge)
        add_point(lines, i + 8, x, tops[i - 1], target_edge)
    lines += [
        "Line(1) = {1,2};", "Line(2) = {2,3};", "Line(3) = {3,4};",
        "Line(4) = {1,5};", "Line(5) = {2,6};", "Line(6) = {3,7};", "Line(7) = {4,8};",
        "Line(8) = {5,6};", "Line(9) = {6,7};", "Line(10) = {7,8};",
        "Line(11) = {5,9};", "Line(12) = {6,10};", "Line(13) = {7,11};", "Line(14) = {8,12};",
        "Line(15) = {9,10};", "Line(16) = {10,11};", "Line(17) = {11,12};",
        "Curve Loop(1) = {1,5,-8,-4};", "Plane Surface(1) = {1};",
        "Curve Loop(2) = {2,6,-9,-5};", "Plane Surface(2) = {2};",
        "Curve Loop(3) = {3,7,-10,-6};", "Plane Surface(3) = {3};",
        "Curve Loop(4) = {8,12,-15,-11};", "Plane Surface(4) = {4};",
        "Curve Loop(5) = {9,13,-16,-12};", "Plane Surface(5) = {5};",
        "Curve Loop(6) = {10,14,-17,-13};", "Plane Surface(6) = {6};",
    ]
    for curve, count in zip((1, 8, 15, 2, 9, 16, 3, 10, 17),
                            (nx[0], nx[0], nx[0], nx[1], nx[1], nx[1], nx[2], nx[2], nx[2])):
        lines.append(f"Transfinite Curve {{{curve}}} = {count + 1};")
    lines.append(f"Transfinite Curve {{4,5,6,7}} = {ny_bottom + 1};")
    lines.append(f"Transfinite Curve {{11,12,13,14}} = {ny_top + 1};")
    lines += ["Transfinite Surface {1,2,3,4,5,6};", "Recombine Surface {1,2,3,4,5,6};",
              "Physical Surface(\"M1\", 1) = {1,2,3};",
              "Physical Surface(\"M2\", 2) = {4,5,6};",
              "Physical Curve(\"Top\", 101) = {15,16,17};",
              "Physical Curve(\"Left\", 102) = {4,11};",
              "Physical Curve(\"Right\", 103) = {7,14};",
              "Physical Curve(\"Bottom\", 104) = {1,2,3};"]
    return "\n".join(lines) + "\n"


def parse_msh2(path: Path) -> tuple[dict[int, tuple[float, float]], list[dict], list[dict]]:
    """解析 Gmsh 2.2 ASCII 网格，并按 num_tags 读取物理组。"""
    rows = path.read_text(encoding="utf-8").splitlines()
    if "$MeshFormat" not in rows or "$Nodes" not in rows or "$Elements" not in rows:
        raise ValueError(f"不是完整的 MSH2 文件: {path}")
    fmt_index = rows.index("$MeshFormat")
    if rows[fmt_index + 1].split()[0] != "2.2":
        raise ValueError("Gmsh 文件必须为 MSH 2.2")
    node_index = rows.index("$Nodes")
    node_count = int(rows[node_index + 1])
    nodes = {}
    for row in rows[node_index + 2: node_index + 2 + node_count]:
        fields = row.split()
        nodes[int(fields[0])] = (float(fields[1]), float(fields[2]))
    element_index = rows.index("$Elements")
    element_count = int(rows[element_index + 1])
    line_elements = []
    quad_elements = []
    for row in rows[element_index + 2: element_index + 2 + element_count]:
        fields = row.split()
        element_id = int(fields[0])
        element_type = int(fields[1])
        tag_count = int(fields[2])
        tags = [int(value) for value in fields[3:3 + tag_count]]
        node_ids = [int(value) for value in fields[3 + tag_count:]]
        physical = tags[0] if tags else 0
        record = {"gmsh_id": element_id, "physical": physical, "nodes": node_ids}
        if element_type == 1:
            line_elements.append(record)
        elif element_type == 3:
            quad_elements.append(record)
        elif element_type not in (15,):
            raise ValueError(f"发现非二节点线/四节点面单元，Gmsh 类型={element_type}")
    if not quad_elements:
        raise ValueError("网格中没有四边形单元")
    return nodes, line_elements, quad_elements


def polygon_area(points: list[tuple[float, float]]) -> float:
    return 0.5 * sum(points[i][0] * points[(i + 1) % 4][1]
                     - points[(i + 1) % 4][0] * points[i][1] for i in range(4))


def edge_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def scaled_jacobian(points: list[tuple[float, float]]) -> float:
    """以四个角的归一化叉积近似检查四边形 Jacobian。"""
    values = []
    for i in range(4):
        previous = points[(i - 1) % 4]
        current = points[i]
        following = points[(i + 1) % 4]
        v1 = (following[0] - current[0], following[1] - current[1])
        v2 = (previous[0] - current[0], previous[1] - current[1])
        denominator = math.hypot(*v1) * math.hypot(*v2)
        values.append((v1[0] * v2[1] - v1[1] * v2[0]) / denominator)
    return min(values)


def export_specfem_mesh(nodes: dict[int, tuple[float, float]], line_elements: list[dict],
                         quad_elements: list[dict], mesh_dir: Path, case: dict,
                         params: dict) -> dict:
    """重排编号并写出 SPECFEM2D external mesh，同时完成质量检查。"""
    mesh_dir.mkdir(parents=True, exist_ok=True)
    node_order = sorted(nodes)
    node_map = {old_id: index + 1 for index, old_id in enumerate(node_order)}
    new_nodes = {node_map[old_id]: nodes[old_id] for old_id in node_order}

    quads = []
    materials = []
    edge_to_element = {}
    all_edges = []
    areas = []
    scaled_jacobians = []
    element_suggested_steps = []
    material_vp = {1: params["materials"]["bedrock"]["vp"],
                   2: params["materials"]["cover"]["vp"]}
    for index, item in enumerate(sorted(quad_elements, key=lambda value: value["gmsh_id"]), 1):
        quad = [node_map[value] for value in item["nodes"]]
        points = [new_nodes[value] for value in quad]
        area = polygon_area(points)
        if area < 0.0:
            quad = [quad[0], quad[3], quad[2], quad[1]]
            points = [new_nodes[value] for value in quad]
            area = polygon_area(points)
        if area <= 0.0:
            raise ValueError(f"单元 {item['gmsh_id']} 面积非正")
        material = item["physical"]
        if material not in (1, 2):
            raise ValueError(f"单元 {item['gmsh_id']} 材料物理组非法: {material}")
        quads.append(quad)
        materials.append(material)
        areas.append(area)
        scaled_jacobians.append(scaled_jacobian(points))
        local_edges = []
        for edge_index in range(4):
            n1 = quad[edge_index]
            n2 = quad[(edge_index + 1) % 4]
            key = tuple(sorted((n1, n2)))
            edge_to_element[key] = index
            length = edge_length(new_nodes[n1], new_nodes[n2])
            all_edges.append(length)
            local_edges.append(length)
        element_suggested_steps.append(0.5 * min(local_edges) * GLL_MIN_SPACING_RATIO / material_vp[material])

    boundary_groups = {101: [], 102: [], 103: [], 104: []}
    for item in line_elements:
        if item["physical"] not in boundary_groups:
            continue
        mapped = [node_map[value] for value in item["nodes"]]
        key = tuple(sorted(mapped))
        if key not in edge_to_element:
            raise ValueError(f"边界线 {item['gmsh_id']} 找不到相邻四边形")
        boundary_groups[item["physical"]].append((edge_to_element[key], mapped[0], mapped[1]))

    if not all(boundary_groups.values()):
        raise ValueError("自由面或三条吸收边界存在空物理组")
    nodes_path = mesh_dir / "nodes_coords_file"
    mesh_path = mesh_dir / "mesh_file"
    material_path = mesh_dir / "materials_file"
    free_path = mesh_dir / "free_surface_file"
    absorbing_path = mesh_dir / "absorbing_surface_file"
    nodes_path.write_text(str(len(new_nodes)) + "\n" + "".join(
        f"{new_nodes[index][0]:.12e} {new_nodes[index][1]:.12e}\n" for index in range(1, len(new_nodes) + 1)), encoding="utf-8")
    mesh_path.write_text(str(len(quads)) + "\n" + "".join(
        " ".join(str(value) for value in quad) + "\n" for quad in quads), encoding="utf-8")
    material_path.write_text("".join(f"{value}\n" for value in materials), encoding="utf-8")
    free_rows = boundary_groups[101]
    free_path.write_text(str(len(free_rows)) + "\n" + "".join(
        f"{element} 2 {n1} {n2}\n" for element, n1, n2 in free_rows), encoding="utf-8")
    absorbing_rows = []
    for physical, side_code in ((104, 1), (103, 2), (102, 4)):
        absorbing_rows.extend((element, n1, n2, side_code)
                              for element, n1, n2 in boundary_groups[physical])
    absorbing_path.write_text(str(len(absorbing_rows)) + "\n" + "".join(
        f"{element} 2 {n1} {n2} {side}\n" for element, n1, n2, side in absorbing_rows), encoding="utf-8")

    suggested_dt = min(element_suggested_steps)
    fmax = params["pulse"]["acceleration_effective_band_5pct_hz"][1]
    min_vs = params["materials"]["cover"]["vs"] if case["has_cover"] else params["materials"]["bedrock"]["vs"]
    shortest_wavelength = min_vs / fmax
    material_counts = {str(value): materials.count(value) for value in sorted(set(materials))}
    boundary_lengths = {}
    for physical, name in ((101, "free"), (102, "left"), (103, "right"), (104, "bottom")):
        boundary_lengths[name] = sum(edge_length(new_nodes[n1], new_nodes[n2])
                                     for _, n1, n2 in boundary_groups[physical])
    checks = {
        "all_elements_are_quads": True,
        "first_element_is_bedrock": materials[0] == 1,
        "positive_area": min(areas) > 0.0,
        "min_scaled_jacobian_ge_0p49": min(scaled_jacobians) >= 0.49,
        "actual_dt_lt_suggested": case["dt"] < suggested_dt,
        "max_edge_within_target": max(all_edges) <= 1.001 * case["target_edge"],
        "cover_material_present_when_required": (2 in materials) == case["has_cover"],
    }
    return {
        "node_count": len(new_nodes), "element_count": len(quads),
        "material_counts": material_counts, "first_element_material": materials[0],
        "min_edge_m": min(all_edges), "max_edge_m": max(all_edges),
        "min_area_m2": min(areas), "max_area_m2": max(areas),
        "min_scaled_jacobian": min(scaled_jacobians),
        "boundary_edge_counts": {"free": len(boundary_groups[101]), "left": len(boundary_groups[102]),
                                 "right": len(boundary_groups[103]), "bottom": len(boundary_groups[104])},
        "boundary_lengths_m": boundary_lengths,
        "shortest_wavelength_at_5pct_fmax_m": shortest_wavelength,
        "spectral_elements_per_shortest_wavelength": shortest_wavelength / max(all_edges),
        "gll_intervals_per_shortest_wavelength": 4.0 * shortest_wavelength / max(all_edges),
        "suggested_dt_s_proxy": suggested_dt, "actual_dt_s": case["dt"],
        "cfl_proxy": 0.5 * case["dt"] / suggested_dt,
        "checks": checks, "passed": all(checks.values()),
    }


def surface_z(x: float) -> float:
    if x <= LEFT_FLAT:
        return CREST_Y
    if x < TOE_X:
        return CREST_Y - (x - LEFT_FLAT) * math.tan(math.radians(SLOPE_DEG))
    return TOE_Y


def s_to_x(s_value: float) -> float:
    if s_value <= 0.0:
        return LEFT_FLAT + s_value * H
    if s_value < 1.0:
        return LEFT_FLAT + s_value * SLOPE_WIDTH
    return TOE_X + (s_value - 1.0) * H


def write_stations(data_dir: Path) -> dict:
    """写出 801 点公共地表网格及基岩/边界诊断点。"""
    stations = []
    for index in range(801):
        s_value = -4.0 + 0.01 * index
        x = s_to_x(s_value)
        stations.append({"station": f"S{index + 1:04d}", "role": "main_surface", "s": s_value,
                         "x": x, "z": surface_z(x)})
    rock_points = [(500.0, 300.0), (700.0, 300.0), (500.0, 500.0)]
    for index, (x, z) in enumerate(rock_points, 1):
        stations.append({"station": f"R{index:04d}", "role": "incident_check", "x": x, "z": z,
                         "expected_common_peak_time": (q_coordinate(x, z) - PULSE_Q_CENTER) / BEDROCK["vs"]})
    boundary_points = [(20.0, 300.0), (20.0, 580.0), (DOMAIN_LENGTH - 20.0, 300.0),
                       (DOMAIN_LENGTH - 20.0, 580.0)]
    for index, (x, z) in enumerate(boundary_points, 1):
        stations.append({"station": f"B{index:04d}", "role": "boundary_diagnostic", "x": x, "z": z})
    lines = [f"{item['station']} XV {item['x']:.9f} {item['z']:.9f} 0.0 0.0" for item in stations]
    (data_dir / "STATIONS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    station_map = {"network": "XV", "count": len(stations), "stations": stations}
    write_json(data_dir / "station_map.json", station_map)
    return station_map


def set_parameter(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^\s*{re.escape(key)}\s*=.*$")
    replacement = f"{key:<32}= {value}"
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise KeyError(f"Par_file 模板中未唯一找到参数 {key}")
    return updated


def replace_material_rows(text: str, rows: list[str]) -> str:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if re.match(r"^\s*nbmodels\s*=", line))
    stop = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("# external tomography file"))
    numeric = [i for i in range(start + 1, stop) if re.match(r"^\s*\d+\s+\d+\s+[-+0-9.]", lines[i])]
    if not numeric:
        raise ValueError("Par_file 模板中未找到材料数据行")
    lines[numeric[0]:numeric[-1] + 1] = rows
    return "\n".join(lines) + "\n"


def material_line(material: dict) -> str:
    damping = material["damping_mapping"]
    return (f"{material['material_id']} 1 {material['density']:.9g} {material['vp']:.12g} "
            f"{material['vs']:.12g} 0 0 {damping['qkappa']:.12g} {damping['qmu']:.12g} 0 0 0 0 0 0")


def write_par_file(template_path: Path, data_dir: Path, case_name: str, case: dict, params: dict) -> None:
    text = template_path.read_text(encoding="utf-8")
    values = {
        "title": f"{case_name} H004-P061 cross-solver validation",
        "SIMULATION_TYPE": "1", "NPROC": str(case["nproc"]), "NSTEP": str(case["nstep"]),
        "DT": f"{case['dt']:.10g}", "time_stepping_scheme": "1", "P_SV": ".true.",
        "NGNOD": "4", "MODEL": "default", "ATTENUATION_VISCOELASTIC": ".true.",
        "N_SLS": "3", "ATTENUATION_f0_REFERENCE": f"{F0:.8g}", "READ_VELOCITIES_AT_f0": ".true.",
        "NSOURCES": "1", "initialfield": ".true.",
        "add_Bielak_conditions_bottom": ".true.", "add_Bielak_conditions_right": ".true.",
        "add_Bielak_conditions_top": ".false.", "add_Bielak_conditions_left": ".true.",
        "seismotype": "3", "NTSTEP_BETWEEN_OUTPUT_SEISMOS": "1000",
        "NTSTEP_BETWEEN_OUTPUT_SAMPLE": str(case["output_sample_stride"]), "USER_T0": "0.0d0",
        "save_ASCII_seismograms": ".true.", "save_binary_seismograms_single": ".false.",
        "use_existing_STATIONS": ".true.", "PML_BOUNDARY_CONDITIONS": ".false.",
        "STACEY_ABSORBING_CONDITIONS": ".true.", "nbmodels": "2" if case["has_cover"] else "1",
        "read_external_mesh": ".true.", "mesh_file": "./DATA/mesh/mesh_file",
        "nodes_coords_file": "./DATA/mesh/nodes_coords_file", "materials_file": "./DATA/mesh/materials_file",
        "free_surface_file": "./DATA/mesh/free_surface_file",
        "absorbing_surface_file": "./DATA/mesh/absorbing_surface_file",
        "NTSTEP_BETWEEN_OUTPUT_INFO": "500", "output_grid_Gnuplot": ".true.",
        "OUTPUT_ENERGY": ".true.", "NTSTEP_BETWEEN_OUTPUT_ENERGY": "50",
        "NTSTEP_BETWEEN_OUTPUT_IMAGES": str(int(round(0.05 / case["dt"]))),
        "output_color_image": ".false.", "output_postscript_snapshot": ".false.",
        "output_wavefield_dumps": ".true.", "imagetype_wavefield_dumps": "3",
        "use_binary_for_wavefield_dumps": ".true.", "GPU_MODE": ".false.",
    }
    for key, value in values.items():
        text = set_parameter(text, key, value)
    rows = [material_line(params["materials"]["bedrock"])]
    if case["has_cover"]:
        rows.append(material_line(params["materials"]["cover"]))
    text = replace_material_rows(text, rows)
    data_dir.joinpath("Par_file").write_text(text, encoding="utf-8")


def write_source(data_dir: Path, params: dict) -> None:
    origin = params["initial_envelope"]["source_phase_origin"]
    text = f"""## Source 1
source_surf                     = .false.
xs                              = {origin['x']:.12f}
zs                              = {origin['z']:.12f}
source_type                     = 5
time_function_type              = 1
name_of_source_file             = ""
burst_band_width                = 0.0
f0                              = {F0:.8f}
tshift                          = 0.0
anglesource                     = {INCIDENCE_DEG:.8f}
Mxx                             = 1.0
Mzz                             = 1.0
Mxz                             = 0.0
factor                          = 1.0
vx                              = 0.0
vz                              = 0.0
"""
    data_dir.joinpath("SOURCE").write_text(text, encoding="utf-8")


def command_output(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(command, cwd=str(cwd) if cwd else None, text=True,
                                       stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def write_run_script(case_dir: Path, specfem_root: Path, nproc: int) -> None:
    script = f"""#!/usr/bin/env bash
set -euo pipefail
case_dir=\"$(cd \"$(dirname \"$0\")\" && pwd)\"
cd \"$case_dir\"
mkdir -p OUTPUT_FILES DATABASES_MPI
cp DATA/Par_file DATA/SOURCE DATA/STATIONS OUTPUT_FILES/
echo \"[X] mesher start $(date --iso-8601=seconds)\" | tee OUTPUT_FILES/mesher.log
mpirun -np {nproc} \"{specfem_root}/bin/xmeshfem2D\" 2>&1 | tee -a OUTPUT_FILES/mesher.log
echo \"[X] solver start $(date --iso-8601=seconds)\" | tee OUTPUT_FILES/solver.log
mpirun -np {nproc} \"{specfem_root}/bin/xspecfem2D\" 2>&1 | tee -a OUTPUT_FILES/solver.log
echo \"[X] completed $(date --iso-8601=seconds)\" | tee OUTPUT_FILES/completed.txt
"""
    path = case_dir / "run_case.sh"
    path.write_text(script, encoding="utf-8", newline="\n")
    path.chmod(0o755)


def summarize_run(case_dir: Path, params: dict) -> dict:
    files = sorted((case_dir / "OUTPUT_FILES").glob("*.sema"))
    finite = True
    row_counts = []
    raw_peaks = []
    time_min = None
    time_max = None
    for path in files:
        rows = 0
        local_peak = 0.0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 2:
                continue
            time_value = float(fields[0])
            value = float(fields[1])
            finite = finite and math.isfinite(time_value) and math.isfinite(value)
            rows += 1
            local_peak = max(local_peak, abs(value))
            time_min = time_value if time_min is None else min(time_min, time_value)
            time_max = time_value if time_max is None else max(time_max, time_value)
        row_counts.append(rows)
        raw_peaks.append(local_peak)
    output_dir = case_dir / "OUTPUT_FILES"
    station_map = json.loads((case_dir / "DATA" / "station_map.json").read_text(encoding="utf-8"))
    case_name = case_dir.name
    case_config = params["cases"].get(case_name, {})
    has_cover = bool(case_config.get("has_cover", False))
    cover_interface_y = CREST_Y - COVER["thickness_below_crest"]
    bedrock_shortest_wavelength = BEDROCK["vs"] / params["pulse"]["acceleration_effective_band_5pct_hz"][1]
    incident_checks = []
    for station_name in ("R0001", "R0002", "R0003"):
        metadata = next(item for item in station_map["stations"] if item["station"] == station_name)
        x_rows = []
        z_rows = []
        for component, target in (("BXX", x_rows), ("BXZ", z_rows)):
            component_path = output_dir / f"XV.{station_name}.{component}.sema"
            for line in component_path.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split()
                if len(fields) >= 2:
                    target.append((float(fields[0]), float(fields[1])))
        expected_time = metadata["expected_common_peak_time"]
        candidates = []
        for (time_x, value_x), (time_z, value_z) in zip(x_rows, z_rows):
            if abs(time_x - time_z) > 1.0e-9:
                raise ValueError(f"{station_name} 两个分量的时间网格不一致")
            common_time = time_x + T0
            if abs(common_time - expected_time) <= 0.06:
                scaled_x = value_x * params["pulse"]["global_linear_scale"]
                scaled_z = value_z * params["pulse"]["global_linear_scale"]
                candidates.append((math.hypot(scaled_x, scaled_z), common_time, scaled_x, scaled_z))
        peak, observed_time, accel_x, accel_z = max(candidates)
        below_interface = metadata["z"] < cover_interface_y
        within_cover_influence = (has_cover and below_interface
                                  and (cover_interface_y - metadata["z"]) < bedrock_shortest_wavelength)
        incident_checks.append({
            "station": station_name, "expected_common_peak_time": expected_time,
            "observed_common_peak_time": observed_time, "time_error": observed_time - expected_time,
            "vector_peak_m_s2": peak, "vector_peak_relative_error": peak / (0.1 * G_STANDARD) - 1.0,
            "accel_x_m_s2": accel_x, "accel_z_m_s2": accel_z,
            "polarization_az_over_ax": accel_z / accel_x,
            "polarization_ratio_error": accel_z / accel_x + math.tan(INCIDENCE_RAD),
            "within_cover_influence": within_cover_influence,
        })
    gate_items = [item for item in incident_checks if not item["within_cover_influence"]]
    if not gate_items:
        raise RuntimeError("覆盖层影响区外的基岩入射测点为空，无法判定入射门控")
    incident_gate = {
        "max_abs_time_error_s": max(abs(item["time_error"]) for item in gate_items),
        "max_abs_peak_relative_error": max(abs(item["vector_peak_relative_error"]) for item in gate_items),
        "max_abs_polarization_ratio_error": max(abs(item["polarization_ratio_error"]) for item in gate_items),
    }
    incident_gate["passed"] = (incident_gate["max_abs_time_error_s"] <= 0.005
                               and incident_gate["max_abs_peak_relative_error"] <= 0.10
                               and incident_gate["max_abs_polarization_ratio_error"] <= 0.02)
    incident_gate["gate_stations"] = [item["station"] for item in gate_items]
    incident_gate["excluded_cover_influence_stations"] = [
        item["station"] for item in incident_checks if item["within_cover_influence"]]
    if has_cover:
        incident_gate["note"] = (f"覆盖层工况按首个材料基岩计算P061解析入射场，界面影响区"
                                 f"(距覆盖层底界面小于基岩最短波长{bedrock_shortest_wavelength:.1f} m)内测点"
                                 f"仅作辅助诊断，不参与正式入射门控")
    result = {
        "seismogram_file_count": len(files), "all_values_finite": finite,
        "row_count_min": min(row_counts) if row_counts else 0,
        "row_count_max": max(row_counts) if row_counts else 0,
        "specfem_time_range": [time_min, time_max],
        "common_time_range": [time_min + T0, time_max + T0] if time_min is not None else None,
        "max_raw_acceleration": max(raw_peaks) if raw_peaks else None,
        "max_scaled_acceleration": (max(raw_peaks) * params["pulse"]["global_linear_scale"]
                                    if raw_peaks else None),
        "wavefield_dump_count": len([path for path in output_dir.glob("wavefield*.bin")
                                      if path.name != "wavefield_grid_for_dumps.bin"]),
        "output_file_count": len([path for path in output_dir.iterdir() if path.is_file()]),
        "output_bytes": sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()),
        "incident_checks": incident_checks, "incident_gate": incident_gate,
        "completed_marker": (output_dir / "completed.txt").exists(),
    }
    result["passed"] = (bool(files) and finite and result["completed_marker"]
                        and result["row_count_min"] > 0 and incident_gate["passed"])
    write_json(case_dir / "run_summary.json", result)
    return result


def generate_case(case_name: str, params: dict, output_root: Path, specfem_root: Path) -> dict:
    case = dict(params["cases"][case_name])
    case_dir = output_root / case_name
    data_dir = case_dir / "DATA"
    mesh_dir = data_dir / "mesh"
    case_dir.joinpath("OUTPUT_FILES").mkdir(parents=True, exist_ok=True)
    case_dir.joinpath("DATABASES_MPI").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    geo_path = mesh_dir / "model.geo"
    msh_path = mesh_dir / "model.msh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    geo_text = generate_layered_geo(case["target_edge"]) if case["has_cover"] else generate_homogeneous_geo(case["target_edge"])
    geo_path.write_text(geo_text, encoding="utf-8")
    subprocess.run(["gmsh", str(geo_path), "-2", "-format", "msh22", "-order", "1", "-o", str(msh_path), "-v", "2"],
                   check=True)
    nodes, line_elements, quad_elements = parse_msh2(msh_path)
    mesh_report = export_specfem_mesh(nodes, line_elements, quad_elements, mesh_dir, case, params)
    if not mesh_report["passed"]:
        raise RuntimeError(f"{case_name} 网格检查未通过: {mesh_report['checks']}")
    template = specfem_root / "EXAMPLES" / "canyon" / "DATA" / "Par_file"
    write_par_file(template, data_dir, case_name, case, params)
    write_source(data_dir, params)
    station_map = write_stations(data_dir)
    write_run_script(case_dir, specfem_root, case["nproc"])
    report = {"case": case_name, "config": case, "mesh": mesh_report,
              "station_count": station_map["count"], "ready_to_run": True}
    write_json(case_dir / "preflight_report.json", report)
    return report


def main() -> int:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specfem-root", type=Path,
                        default=Path("/home/fuyuxuan/software/specfem2d-v8.1.0"))
    parser.add_argument("--output-root", type=Path,
                        default=repo_root / "Run" / "cross_solver_X" / "specfem2d")
    parser.add_argument("--cases", nargs="+", choices=sorted(CASE_CONFIG), default=sorted(CASE_CONFIG))
    parser.add_argument("--run", choices=sorted(CASE_CONFIG), help="生成后立即运行指定 SPECFEM2D 工况")
    args = parser.parse_args()

    if not shutil.which("gmsh"):
        raise RuntimeError("未找到 gmsh，请在已准备的 WSL2 Ubuntu 环境中运行本脚本")
    if not (args.specfem_root / "bin" / "xspecfem2D").exists():
        raise FileNotFoundError(f"SPECFEM2D 可执行文件不存在: {args.specfem_root}")
    params = build_parameter_table()
    pulse_path = repo_root / "Wave" / "Impulse" / "Acceleration" / "ricker_displacement_derived_acceleration_4Hz_0p1g.txt"
    pulse_hash = write_common_pulse(pulse_path, params["pulse"])
    params["pulse"]["abaqus_acceleration_file"] = str(pulse_path.relative_to(repo_root)).replace(os.sep, "/")
    params["pulse"]["abaqus_acceleration_sha256"] = pulse_hash
    tracked_params = script_path.parent / "x_validation_parameters.json"
    write_json(tracked_params, params)
    write_json(args.output_root.parent / "common" / "x_validation_parameters.json", params)
    environment = {
        "specfem_root": str(args.specfem_root),
        "specfem_git_commit": command_output(["git", "rev-parse", "HEAD"], args.specfem_root),
        "gmsh_version": command_output(["gmsh", "--version"]),
        "python_version": sys.version,
        "platform": command_output(["uname", "-a"]),
    }
    write_json(args.output_root.parent / "common" / "environment.json", environment)
    reports = {}
    for case_name in args.cases:
        reports[case_name] = generate_case(case_name, params, args.output_root, args.specfem_root)
        print(f"[完成] {case_name}: {reports[case_name]['mesh']['element_count']} 个谱元")
    if args.run:
        if args.run not in args.cases:
            raise ValueError("--run 指定的工况必须同时包含在 --cases 中")
        case_dir = args.output_root / args.run
        subprocess.run(["bash", str(case_dir / "run_case.sh")], check=True)
        summary = summarize_run(case_dir, params)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary["passed"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
