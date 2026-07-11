# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""P1 左右观测窗外扩诊断批处理脚本。

用途：
    在不改动主批处理脚本 `Autorun_P1_v1.py` 的前提下，只改变观测窗与边界缓冲，
    对比端部峰值是否随侧边界外移而消失或迁移。

运行：
    python C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Batch\\Autorun_P1_window_test_v1.py
    python C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Batch\\Autorun_P1_window_test_v1.py C:\\Users\\12462\\Documents\\Code\\AbqScripts\\Run\\P1_window_test
"""

import os  # 导入路径模块
import sys  # 导入系统模块以注入主批处理脚本目录


WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 工作区根目录
BATCH_DIR = os.path.join(WORKSPACE_DIR, "Batch")  # 主批处理脚本目录
if BATCH_DIR not in sys.path:  # 避免重复插入路径
    sys.path.insert(0, BATCH_DIR)  # 允许从 test/Batch 导入 Batch 内主脚本

import Autorun_P1_v1 as base  # 复用 P1 主批处理的执行逻辑


CREST_WINDOW = 5.0  # 坡顶侧观测窗：由原 4.0h 外扩到 5.0h
TOE_WINDOW = 4.0  # 坡脚侧观测窗：由原 3.0h 外扩到 4.0h，使观测范围约为 s∈[-5,5]
SIDE_CLEARANCE = 1.0  # 侧边界缓冲：由原 0.1h 加大到 1.0h，避免新观测窗紧贴人工边界

WINDOW_CASE_KEYS = [  # 仅跑 h=50 的四个代表工况，降低诊断成本
    (50.0, 15, 0),
    (50.0, 15, 20),
    (50.0, 75, 0),
    (50.0, 75, 20),
]

WAVE_FILES = [  # 同时跑高频与低频记录，覆盖端点峰值更明显的低频情况
    os.path.join(WORKSPACE_DIR, "Wave", "Impulse", "Acceleration", "ricker_wavelet_4Hz.txt"),
    os.path.join(WORKSPACE_DIR, "Wave", "Impulse", "Acceleration", "ricker_wavelet_2Hz.txt"),
]


def build_window_cases():  # 构造观测窗外扩诊断工况
    """返回用于注入 `case_config.json` 的工况列表。"""
    cases = []  # 初始化工况列表
    for h_val, i_val, th_val in WINDOW_CASE_KEYS:  # 遍历代表工况
        cases.append({
            "name": "P1win-L1-h{}_i{}_t{}".format(base._fmt_num(h_val), i_val, th_val),  # 工况名带 P1win 前缀，避免覆盖主批
            "config": {
                "material_cfg": {"angle": th_val, "layers": []},  # 全基岩均质，保持与 P1 smoke 一致
                "geometry_cfg": {
                    "slope_height": h_val,  # 坡高
                    "slope_angle": float(i_val),  # 坡角
                    "crest_window": CREST_WINDOW,  # 坡顶侧观测窗
                    "toe_window": TOE_WINDOW,  # 坡脚侧观测窗
                    "side_clearance": SIDE_CLEARANCE,  # 侧边界缓冲
                },
                "time_cfg": {"tail_seconds": 3.0},  # 保持 H(f) 后处理尾段一致
                "run_cfg": {"wave_files": WAVE_FILES},  # 指定诊断波形
            },
        })
    return cases  # 返回工况列表


def main():  # 主入口
    """复用 `Autorun_P1_v1` 的主流程，但替换为观测窗诊断参数。"""
    base.SMOKE_MODE = True  # 使用 smoke 模式
    base.SMOKE_ROOT_DIR = os.path.join(WORKSPACE_DIR, "test", "Abaqus", "P1_window_test")  # 默认输出到 test/Abaqus
    base.MAX_WORKERS = 2  # 四个代表工况并行
    base.DELETE_FILE_TYPES = [".inp", ".msg", ".prt", ".dat", ".sta", ".sim", "jnl"]  # 成功后清理中间大文件，除了".odb", 
    base.PARAMETER_CASES = build_window_cases()  # 替换工况列表
    base.main()  # 调用主批处理流程


if __name__ == "__main__":  # 命令行入口
    main()  # 执行主函数

