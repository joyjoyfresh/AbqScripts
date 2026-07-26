# -*- coding: utf-8 -*-
"""当前统一建模脚本的纯Python核心回归，不调用Abaqus求解器。"""

from __future__ import print_function

import importlib.util
import math
import os
import sys
import types

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _load_model():
    """用最小桩模块导入当前统一建模脚本。"""
    abaqus = types.ModuleType('abaqus')
    abaqus.mdb = None
    sys.modules['abaqus'] = abaqus
    for name in ('abaqusConstants', 'caeModules', 'mesh'):
        sys.modules[name] = types.ModuleType(name)
    region = types.ModuleType('regionToolset')
    region.Region = object
    sys.modules['regionToolset'] = region
    spec = importlib.util.spec_from_file_location('slope_frame_ssi_full_v2_core_test', MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.basestring = str
    return module


def _relative_complex_error(actual_x, actual_y, expected_x, expected_y):
    """返回两个复数分量的统一最大相对误差。"""
    scale = max(
        float(np.max(np.abs(expected_x))),
        float(np.max(np.abs(expected_y))),
        1.0e-30,
    )
    return max(
        float(np.max(np.abs(actual_x - expected_x))),
        float(np.max(np.abs(actual_y - expected_y))),
    ) / scale


def _compare_reference(module, reference, layers_top_down, angle, damped):
    """把生产P-SV柱解与独立全局矩阵参考解逐频对拍。"""
    freqs = np.asarray([0.5, 1.0, 2.0, 4.0, 8.0, 10.0])
    omega = 2.0 * math.pi * freqs
    halfspace = {'vs': 2000.0, 'rho': 2500.0, 'nu': 0.30}
    p_horiz = math.sin(math.radians(angle)) / halfspace['vs']
    column = [
        module._column_seg(
            halfspace['vs'], halfspace['nu'], halfspace['rho'],
            p_horiz, 0.0, 0.0, 'bedrock',
        ),
    ]
    damp_terms = {'bedrock': (0.0, 0.0)}
    reference_layers = []
    y_cursor = 0.0
    for index, layer in enumerate(reversed(layers_top_down)):
        name = layer['name']
        column.append(module._column_seg(
            layer['vs'], layer['nu'], layer['rho'], p_horiz,
            y_cursor, y_cursor + layer['thickness'], name,
        ))
        y_cursor += layer['thickness']
        alpha_ray = 0.02 + 0.005 * index if damped else 0.0
        beta_ray = 0.0005 + 0.0001 * index if damped else 0.0
        damp_terms[name] = (alpha_ray, beta_ray)
    for layer in layers_top_down:
        alpha_ray, beta_ray = damp_terms[layer['name']]
        reference_layers.append(dict(
            layer,
            rayleigh_alpha=alpha_ray,
            rayleigh_beta=beta_ray,
        ))
    if damped:
        damp_terms['bedrock'] = (0.01, 0.00002)
        halfspace.update(rayleigh_alpha=0.01, rayleigh_beta=0.00002)
    solution = module._fd_solve_column(
        column, p_horiz, omega, damp_terms, True,
    )
    production = module._fd_eval_column(
        solution, omega, p_horiz, y_cursor,
    )
    independent = reference.transfer_function(
        freqs, reference_layers, halfspace, angle,
    )
    incident_scale = np.empty(freqs.shape, dtype=complex)
    for index, omega_i in enumerate(omega):
        material = reference._material(
            halfspace['vs'], halfspace['rho'], halfspace['nu'],
            omega=omega_i,
            rayleigh_alpha=halfspace.get('rayleigh_alpha', 0.0),
            rayleigh_beta=halfspace.get('rayleigh_beta', 0.0),
        )
        incident_scale[index] = -material['vs']
    expected_x = incident_scale * independent['ux']
    expected_y = incident_scale * independent['uy']
    return _relative_complex_error(
        production['ux'], production['uy'], expected_x, expected_y,
    )


def _test_independent_psv_reference(module):
    """覆盖均质/分层、斜入射和瑞利阻尼的独立参考解。"""
    from Modeling.Archived.Hybrid import reference_layered_psv_v1 as reference

    layer_sets = [
        [],
        [
            {
                'name': 'overlying', 'vs': 800.0, 'rho': 2200.0,
                'nu': 0.28, 'thickness': 120.0,
            },
        ],
        [
            {
                'name': 'surface', 'vs': 400.0, 'rho': 2000.0,
                'nu': 0.30, 'thickness': 40.0,
            },
            {
                'name': 'overlying', 'vs': 800.0, 'rho': 2200.0,
                'nu': 0.28, 'thickness': 80.0,
            },
        ],
    ]
    errors = []
    for layers in layer_sets:
        for angle in (0.0, 15.0, 30.0):
            errors.append(_compare_reference(module, reference, layers, angle, False))
            errors.append(_compare_reference(module, reference, layers, angle, True))
    maximum = max(errors)
    assert maximum < 1.0e-10, '生产P-SV柱解与独立参考解不一致: %.3e' % maximum
    return maximum


def _test_damping_scope(module):
    """确认有限土层与基岩阻尼入口互不串用且非法值会被拦截。"""
    config = {
        'constant_xi': 0.03,
        'qs_factor': 0.05,
        'q_bedrock': 999.0,
        'bedrock_xi': None,
    }
    q_rock, xi_rock = module._damping_ratio_from_q(
        2000.0, True, config, 'bedrock',
    )
    q_soil, xi_soil = module._damping_ratio_from_q(
        800.0, False, config, 'soil',
    )
    assert abs(q_rock - 999.0) < 1.0e-12
    assert abs(xi_rock - 1.0 / 1998.0) < 1.0e-15
    assert abs(q_soil - 1.0 / 0.06) < 1.0e-12
    assert abs(xi_soil - 0.03) < 1.0e-15
    overridden = dict(config, bedrock_xi=0.02)
    q_rock, xi_rock = module._damping_ratio_from_q(
        2000.0, True, overridden, 'bedrock',
    )
    assert abs(q_rock - 25.0) < 1.0e-12
    assert abs(xi_rock - 0.02) < 1.0e-15
    failed = False
    try:
        module._damping_ratio_from_q(
            2000.0, True, dict(config, bedrock_xi=0.0), 'bedrock',
        )
    except ValueError:
        failed = True
    assert failed, '零阻尼比必须由显式关闭enable表达，不能作为非法覆盖值静默进入'
    failed = False
    try:
        module._damping_ratio_from_q(
            800.0, False, dict(config, constant_xi=0.0), 'soil',
        )
    except ValueError:
        failed = True
    assert failed, '有限土层零阻尼比也必须被显式拦截'


def _test_boundary_quadrature(module):
    """验证线性与二次边界节点影响长度积分守恒。"""
    class Node(object):
        def __init__(self, label, x, y):
            self.label = label
            self.coordinates = (x, y)

    material = {
        'GG': 1.0e10, 'density': 2500.0,
        'cp': 3500.0, 'cs': 2000.0,
    }
    pick_material = lambda _x, _y: material
    linear = module._make_boundary_nodes(
        [Node(1, 0.0, 0.0), Node(2, 1.0, 0.0), Node(3, 3.0, 0.0)],
        'x', True, pick_material, 800.0, quadratic=False,
    )
    quadratic = module._make_boundary_nodes(
        [
            Node(1, 0.0, 0.0), Node(2, 0.5, 0.0), Node(3, 1.0, 0.0),
            Node(4, 1.5, 0.0), Node(5, 2.0, 0.0),
        ],
        'x', True, pick_material, 800.0, quadratic=True,
    )
    assert abs(sum(item.influence for item in linear) - 3.0) < 1.0e-12
    assert abs(sum(item.influence for item in quadratic) - 2.0) < 1.0e-12


def _test_equivalent_force_mapping(module):
    """验证三类边界的Ku+Cv+Aσ分量映射。"""
    time = np.asarray([0.0, 1.0])
    field = {
        'time': time,
        'ux': np.asarray([1.0, 2.0]),
        'uy': np.asarray([3.0, 4.0]),
        'dotux': np.asarray([5.0, 6.0]),
        'dotuy': np.asarray([7.0, 8.0]),
        'sigmax': np.asarray([9.0, 10.0]),
        'sigmay': np.asarray([11.0, 12.0]),
    }
    original = module._fd_freefield_at_node
    module._fd_freefield_at_node = lambda *_args, **_kwargs: field
    try:
        node = module.BoundaryNode(
            label=1, x=0.0, y=0.0, influence=2.0,
            kn=3.0, cn=5.0, kt=7.0, ct=11.0,
        )
        geometry = types.SimpleNamespace(
            H_upper=10.0, H_lower=8.0, left_flat=4.0, w_slope=2.0,
        )
        context = types.SimpleNamespace(
            geom=geometry, ffcfg={
                'engine': 'fd',
                'reference_field_mode': 'global_upper',
                'bottom_ymax_mode': 'local',
            },
            ymax_l=10.0, ymax_r=8.0,
        )
        values = module._build_equivalent_forces(
            {'l': [node], 'r': [node], 'b': [node]}, context,
        )
        expected_side_x = node.kn * field['ux'] + node.cn * field['dotux'] + node.influence * field['sigmax']
        expected_side_y = node.kt * field['uy'] + node.ct * field['dotuy'] + node.influence * field['sigmay']
        expected_bottom_x = node.kt * field['ux'] + node.ct * field['dotux'] + node.influence * field['sigmax']
        expected_bottom_y = node.kn * field['uy'] + node.cn * field['dotuy'] + node.influence * field['sigmay']
        assert np.allclose(values['1-l-fx'][:, 1], expected_side_x)
        assert np.allclose(values['1-l-fy'][:, 1], expected_side_y)
        assert np.allclose(values['1-b-fx'][:, 1], expected_bottom_x)
        assert np.allclose(values['1-b-fy'][:, 1], expected_bottom_y)
    finally:
        module._fd_freefield_at_node = original


def _test_reference_field_consistency(module):
    """确认生产口径三边界只取同一个上平台平场，旧局部柱仍可追溯。"""
    geometry = types.SimpleNamespace(
        H_upper=10.0, H_lower=8.0, left_flat=4.0, w_slope=2.0,
    )
    global_context = types.SimpleNamespace(
        geom=geometry,
        ffcfg={'reference_field_mode': 'global_upper'},
        ymax_l=10.0,
        ymax_r=8.0,
    )
    values = [
        module._reference_column_surface_y('l', 0.0, 10.0, global_context),
        module._reference_column_surface_y('r', 10.0, 8.0, global_context),
        module._reference_column_surface_y('b', 5.0, 0.0, global_context),
    ]
    assert values == [10.0, 10.0, 10.0]

    local_context = types.SimpleNamespace(
        geom=geometry,
        ffcfg={
            'reference_field_mode': 'local_columns',
            'bottom_ymax_mode': 'local',
        },
        ymax_l=10.0,
        ymax_r=8.0,
    )
    local_values = [
        module._reference_column_surface_y('l', 0.0, 10.0, local_context),
        module._reference_column_surface_y('r', 10.0, 8.0, local_context),
        module._reference_column_surface_y('b', 5.0, 0.0, local_context),
    ]
    assert np.allclose(local_values, [10.0, 8.0, 9.0])

    invalid_context = types.SimpleNamespace(
        geom=geometry,
        ffcfg={'reference_field_mode': 'global_lower'},
        ymax_l=10.0,
        ymax_r=8.0,
    )
    failed = False
    try:
        module._reference_column_surface_y('l', 0.0, 10.0, invalid_context)
    except ValueError:
        failed = True
    assert failed, '下平台平场不能外推到其自由面以上的左边界节点'

    plane_geometry = types.SimpleNamespace(
        H_upper=800.0, H_lower=600.0, left_flat=400.0, w_slope=200.0,
    )
    omega = np.asarray([2.0 * math.pi])
    amplitudes = {}
    for mode in ('global_upper', 'local_columns'):
        context = types.SimpleNamespace(
            geom=plane_geometry,
            ffcfg={
                'reference_field_mode': mode,
                'bottom_ymax_mode': 'local',
            },
            ymax_l=800.0,
            ymax_r=600.0,
        )
        values = []
        for x_coord in (400.0, 500.0, 600.0):
            surface_y = module._reference_column_surface_y(
                'b', x_coord, 0.0, context,
            )
            column = [
                module._column_seg(
                    2000.0, 0.30, 2500.0, 1.0e-15,
                    0.0, surface_y, 'bedrock',
                ),
            ]
            solution = module._fd_solve_column(
                column, 1.0e-15, omega,
                {'bedrock': (0.0, 0.0)}, True,
            )
            field = module._fd_eval_column(
                solution, omega, 1.0e-15, 0.0,
            )
            values.append(float(abs(field['ux'][0])))
        amplitudes[mode] = values
    assert np.ptp(amplitudes['global_upper']) < 1.0e-12
    assert np.ptp(amplitudes['local_columns']) > 0.1


def _test_clearance_phase_invariance(module):
    """确认对称增加侧向净空不会改变坡体相对输入相位原点。"""
    positions = []
    for clearance in (0.1, 1.0, 4.0):
        resolved = module._resolve_geometry_cfg({
            'slope_height': 200.0,
            'slope_angle': 60.0,
            'crest_window': 4.0,
            'toe_window': 3.0,
            'side_clearance': clearance,
            'base_depth': 3.0,
        }, None)
        phase_origin = 0.5 * resolved['total_L']
        slope_width = 200.0 / math.tan(math.radians(60.0))
        positions.append((
            resolved['left_flat'] - phase_origin,
            resolved['left_flat'] + slope_width - phase_origin,
        ))
    assert np.allclose(positions, [positions[0]] * len(positions))


def main():
    module = _load_model()
    maximum = _test_independent_psv_reference(module)
    _test_damping_scope(module)
    _test_boundary_quadrature(module)
    _test_equivalent_force_mapping(module)
    _test_reference_field_consistency(module)
    _test_clearance_phase_invariance(module)
    print('test_modeling_core: 6/6 ok, independent_psv_max_error=%.3e' % maximum)


if __name__ == '__main__':
    main()
