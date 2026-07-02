# -*- coding: utf-8 -*-
"""build_site 单元测试（纯 Python，不进 Abaqus）。

被测对象：Modeling/Hybrid/slope_frame_ssi_full_v1.py 的 build_site——
各土层剪切波速 vs 与厚度 thickness 均显式给定，剩余深度归基岩；
基岩顶面高程 = 坡顶地表 − Σ土层厚，校验底部净空 ≥ 2·坡高。
运行方式：python test/Hybrid/test_build_site_vs_v1.py
"""
import io
import math
import os
import re
from collections import namedtuple

_HERE = os.path.dirname(os.path.abspath(__file__))  # 本测试文件所在目录
_REPO = os.path.dirname(os.path.dirname(_HERE))  # 仓库根（test/Hybrid 上两级）
_TARGET = os.path.join(_REPO, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v1.py')  # 被测脚本路径

Material = namedtuple('Material', ['cs', 'vv', 'density', 'thickness', 'name'])  # 与脚本同构的材料元组
Site = namedtuple('Site', ['bedrock', 'layers', 'bedrock_thickness'])  # 与脚本同构的场地元组


def _load_func():  # 从建模脚本中提取被测函数（避免导入 abaqus 模块）
    src = io.open(_TARGET, encoding='utf-8').read()
    m = re.search(r'def build_site.*?(?=\ndef )', src, re.S)  # 截取函数源码段
    assert m, '未在建模脚本中找到 build_site'
    ns = {'math': math, 'Material': Material, 'Site': Site}  # 注入依赖
    exec(m.group(0), ns)
    return ns['build_site']


def _mat(**over):  # 默认材料配置（基岩+表层+覆盖层），可按键覆盖
    m = {'bedrock': {'vs': 2000.0, 'poisson_ratio': 0.3, 'density': 2500},
         'layers': [
             {'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 50},
             {'name': 'overlying', 'vs': 800.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 350},
         ]}
    m.update(over)
    return m


_GEO = {'H_minus_h': 200.0, 'H_lower': 600.0}  # 引擎几何键（坡高 200，坡脚面以下 600）


def main():  # 逐用例断言土层直读与净空校验
    f = _load_func()

    # 用例1：vs/thickness 原样进 Material；基岩顶面 = 800−400 = 400
    site, soils = f(_mat(), dict(_GEO))
    assert site.bedrock.cs == 2000.0
    assert [L.cs for L in site.layers] == [400.0, 800.0]
    assert [L.thickness for L in site.layers] == [50.0, 350.0]
    assert soils == [50.0, 350.0]
    assert site.bedrock_thickness == 400.0  # 基岩顶面高程（=2h，恰好满足净空）

    # 用例2：全基岩坡（layers=[]）→ 基岩顶面 = 坡顶地表 800
    site, soils = f(_mat(layers=[]), dict(_GEO))
    assert site.layers == [] and soils == []
    assert site.bedrock_thickness == 800.0

    # 用例3：土层总厚超限（>base_depth·h−h=400）→ 净空校验报 ValueError
    over = _mat()
    over['layers'][1]['thickness'] = 351.0  # Σt=401 → 基岩顶面 399 < 2h=400
    try:
        f(over, dict(_GEO))
        raise AssertionError('净空超限未报错')
    except ValueError:
        pass

    # 用例4：缺厚度 / 非法厚度 / 非法波速 → ValueError
    for layers in (
            [{'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2500}],  # 缺 thickness
            [{'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 0.0}],  # 零厚度
            [{'name': 'surface', 'vs': -1.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 50}],  # 负波速
    ):
        try:
            f(_mat(layers=layers), dict(_GEO))
            raise AssertionError('非法配置未报错: %r' % (layers,))
        except ValueError:
            pass

    print('test_build_site_vs_v1: 4/4 ok')


if __name__ == '__main__':
    main()
