# -*- coding: utf-8 -*-
"""Hybrid v2 显式平场验证模式的纯 Python 静态回归。

本测试不导入 Abaqus，也不启动求解器；只提取几何派生函数并检查主脚本中的
配置默认值、平场分支、模型场景标签和元数据字段，确保生产坡地默认路径不漂移。
"""
from __future__ import print_function

import io
import math
import os
import re
from collections import namedtuple


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
TARGET = os.path.join(REPO, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')
Geometry = namedtuple('Geometry', [
    'total_L', 'i', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness',
    'H', 'h', 'H_upper', 'H_lower', 'H_flat', 'w_slope', 'layer_interfaces'])


def _load_geometry_functions():
    """从主脚本提取不依赖 Abaqus 的几何函数。"""
    source = io.open(TARGET, encoding='utf-8').read()
    namespace = {'math': math, 'Geometry': Geometry}
    for pattern in (r'def make_geometry.*?(?=\ndef )',
                    r'def make_flat_validation_geometry.*?(?=\ndef )'):
        match = re.search(pattern, source, re.S)
        assert match, '未找到几何函数: %s' % pattern
        exec(match.group(0), namespace)
    return namespace['make_geometry'], namespace['make_flat_validation_geometry']


def main():
    make_geometry, make_flat = _load_geometry_functions()
    source = io.open(TARGET, encoding='utf-8').read()

    geom = make_geometry(total_L=2800.0, H_minus_h=200.0, i=45.0,
                         left_flat=1400.0, toe_surface_y=600.0,
                         soil_thicknesses=[50.0, 350.0])
    flat = make_flat(geom)
    assert flat.total_L == geom.total_L
    assert flat.H_upper == geom.H_upper == 800.0
    assert flat.H_lower == flat.H_upper
    assert flat.w_slope == 1.0e-3
    assert flat.H_minus_h == geom.H_minus_h
    assert flat.layer_interfaces == geom.layer_interfaces

    try:
        make_flat(geom._replace(H_upper=0.0, H_lower=0.0))
        raise AssertionError('H_upper<=0 未触发平场几何校验')
    except ValueError:
        pass

    assert "'validation_geometry': 'slope'" in source
    assert "validation_geometry = str(_run_cfg.get('validation_geometry', 'slope')).lower()" in source
    assert 'create_flat_validation_model(' in source
    assert "'submit_jobs': True" in source
    assert '_write_geometry_validation_audit(' in source
    assert 'model_scene=validation_geometry' in source
    assert "'validation_geometry': str(validation_geometry)" in source
    assert 'if validation_geometry == \'flat\':' in source
    assert 'else:\n            base_model, part_name, inst_name = create_model(' in source
    print('test_flat_validation_geometry_v2: 12/12 ok')


if __name__ == '__main__':
    main()
