# -*- coding: utf-8 -*-
"""小论文 V 批次补充域敏感性入口。

本入口复用正式 V 批处理的建模、后处理、状态清单和失败保留逻辑，输出到独立
的 ``Run/ch4_sp_01_V_domain_sensitivity`` 根目录。``V004`` 是正式记录的单因素
工况：仅把 P061 的侧向净空由 1H 改为 4H，基底深度保持 3H。其余域组合只供
本次域参数取值分析，不进入正式论文计划、物理效应统计或机器学习数据集。

运行形式：
    python Run/Auto_ch4/Autorun_ch4_sp_01_V_domain_sensitivity.py [求解输出根目录]
"""

from __future__ import absolute_import

import sys

import Autorun_ch4_sp_01_V_v1 as pipeline


DOMAIN_ROOT = (
    r"C:\Users\12462\Documents\Code\AbqScripts\Run\ch4_sp_01_V_domain_sensitivity"
)


def build_domain_cases():
    """生成正式 V004 及内部域组合扫描工况。"""
    case_specs = (
        ("001-V004", 4.0, 3.0),
        ("002-DOM-S2-B3", 2.0, 3.0),
        ("003-DOM-S6-B3", 6.0, 3.0),
        ("004-DOM-S2-B6", 2.0, 6.0),
        ("005-DOM-S4-B6", 4.0, 6.0),
        ("006-DOM-S6-B6", 6.0, 6.0),
    )
    cases = []
    for name, side_clearance, base_depth in case_specs:
        config = pipeline.base_config(
            60.0,
            [pipeline.cover_layer(600.0, 140.0)],
            [pipeline.G1B_WAVE],
            extra={
                "geometry_cfg": {
                    "side_clearance": side_clearance,
                    "base_depth": base_depth,
                }
            },
        )
        cases.append({"name": name, "config": config})
    return cases


def main():
    """使用正式 V 流水线运行补充域工况。"""
    pipeline.ROOT_DIR = sys.argv[1] if len(sys.argv) >= 2 else DOMAIN_ROOT
    pipeline.MAX_WORKERS = 2
    pipeline.POSTPROCESS_WORKERS = 1
    pipeline.PARAMETER_CASES = build_domain_cases()
    pipeline.main()


if __name__ == "__main__":
    main()
