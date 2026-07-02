# -*- coding: utf-8 -*-
"""几何换算与派生单元测试（纯 Python，不进 Abaqus）。

被测对象：Modeling/Hybrid/slope_frame_ssi_full_v1.py 的
  _resolve_geometry_cfg —— 无量纲设计(6键) → 引擎绝对尺寸(5键)；
  make_geometry —— 外形 + 土层厚度表 → 全部派生量（基岩顶面/界面/h_over_H 记录量）。
新口径：坡脚面以下深度 = base_depth·hs 恒定；地层划分全在 material_cfg['layers']。
运行方式：python test/Hybrid/test_resolve_geometry_v1.py
"""
import io
import math
import os
import re
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))  # 本测试文件所在目录
_REPO = os.path.dirname(os.path.dirname(_HERE))  # 仓库根（test/Hybrid 上两级）
_TARGET = os.path.join(_REPO, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v1.py')  # 被测脚本路径

Geometry = namedtuple('Geometry', [  # 与脚本同构的几何元组
    'total_L', 'i', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness',
    'H', 'h', 'H_upper', 'H_lower', 'H_flat', 'w_slope', 'layer_interfaces'])


def _load_funcs():  # 从建模脚本中提取被测函数（避免导入 abaqus 模块）
    src = io.open(_TARGET, encoding='utf-8').read()
    ns = {'math': math, 'log_step': lambda *a, **k: None, 'Geometry': Geometry}  # 注入依赖
    for pat in (r'def _resolve_geometry_cfg.*?(?=\ndef )', r'def make_geometry.*?(?=\ndef )'):
        m = re.search(pat, src, re.S)
        assert m, '未找到被测函数: %s' % pat
        exec(m.group(0), ns)
    return ns['_resolve_geometry_cfg'], ns['make_geometry']


def _design(**over):  # 生成一份默认无量纲设计，可按键覆盖
    g = {'slope_height': 200.0, 'slope_angle': 45.0, 'crest_window': 5.0,
         'toe_window': 4.0, 'side_clearance': 2.0, 'base_depth': 3.0}
    g.update(over)
    return g


def main():  # 逐用例断言换算与派生逻辑
    resolve, make_geometry = _load_funcs()

    # 用例1：默认设计（hs=200, 45°）→ total_L=14h=2800，坡脚面以下 3h=600
    g = resolve(_design())
    assert abs(g['left_flat'] - 1400.0) < 1e-9, g
    assert abs(g['total_L'] - 2800.0) < 1e-6, g
    assert abs(g['H_lower'] - 600.0) < 1e-9, g
    assert g['H_minus_h'] == 200.0 and g['i'] == 45.0  # 引擎键映射正确
    assert set(g) == {'H_minus_h', 'i', 'left_flat', 'total_L', 'H_lower'}  # 输出仅引擎键

    # 用例2：坡角浮动（30°）→ w_slope 变长，total_L 跟着浮动；H_lower 不受坡角影响
    g = resolve(_design(slope_angle=30.0))
    assert abs(g['total_L'] - (1400.0 + 200.0 / math.tan(math.radians(30.0)) + 1200.0)) < 1e-6
    assert abs(g['H_lower'] - 600.0) < 1e-9

    # 用例3：非法取值必须报 ValueError（含 base_depth<1 净空不足）
    for over in ({'slope_height': -1.0}, {'slope_angle': 95.0}, {'base_depth': 0.5},
                 {'side_clearance': -0.5}):
        try:
            resolve(_design(**over))
            raise AssertionError('非法取值未报错: %r' % (over,))
        except ValueError:
            pass

    # 用例4：make_geometry 多土层派生（50+350，坡脚面以下 600）→ 基岩顶面=400，界面=[750]
    geom = make_geometry(total_L=2800.0, H_minus_h=200.0, i=45.0, left_flat=1400.0,
                         toe_surface_y=600.0, soil_thicknesses=[50.0, 350.0])
    assert geom.H_upper == 800.0 and geom.H_lower == 600.0
    assert geom.bedrock_thickness == 400.0  # 800 − (50+350)
    assert geom.layer_interfaces == [750.0]  # 仅土层间界面（不含基岩顶面）
    assert geom.H == 400.0 and geom.h == 200.0 and abs(geom.h_over_H - 0.5) < 1e-12  # 派生记录量

    # 用例5：薄土层（总厚 100 < 坡高 200）→ 基岩顶面高于坡脚面（基岩坡面出露），h 为负
    geom = make_geometry(total_L=2800.0, H_minus_h=200.0, i=45.0, left_flat=1400.0,
                         toe_surface_y=600.0, soil_thicknesses=[100.0])
    assert geom.bedrock_thickness == 700.0 and geom.bedrock_thickness > geom.H_lower
    assert geom.h == -100.0  # 坡脚下无土层

    # 用例6：全基岩坡（无土层）→ 基岩顶面=坡顶地表，h_over_H 无定义
    geom = make_geometry(total_L=2800.0, H_minus_h=200.0, i=45.0, left_flat=1400.0,
                         toe_surface_y=600.0, soil_thicknesses=[])
    assert geom.bedrock_thickness == 800.0 and geom.H == 0.0
    assert geom.h_over_H is None and geom.layer_interfaces == []

    print('test_resolve_geometry_v1: 6/6 ok')


if __name__ == '__main__':
    main()
