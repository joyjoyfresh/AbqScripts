# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""P1 右侧边界净空加宽对照批处理脚本。

用途：
    仅复跑 h=50 m、坡角 15°、入射角 20° 的 4 Hz Ricker 波工况；保持观测窗
    s∈[-5, 5] 不变，将侧向净空由 1h 扩大到 4h。用于判别右侧低幅区是地形散射
    还是人工边界影响，不覆盖既有 P1_window_test 结果。

运行：
    python C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Batch\\Autorun_P1_boundary_clearance_test_v1.py
    python C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Batch\\Autorun_P1_boundary_clearance_test_v1.py C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Abaqus\\P1_boundary_clearance_test
"""

import os  # 导入路径模块
import sys  # 导入系统模块以注入主批处理脚本目录


WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 工作区根目录
BATCH_DIR = os.path.join(WORKSPACE_DIR, "Batch")  # 主批处理脚本目录
if BATCH_DIR not in sys.path:  # 避免重复插入路径
    sys.path.insert(0, BATCH_DIR)  # 允许从 test/Batch 导入 Batch 内主脚本

import Autorun_P1_v1 as base  # 复用 P1 主批处理的执行逻辑


CREST_WINDOW = 5.0  # 坡顶侧观测窗，固定为既有窗口诊断范围
TOE_WINDOW = 4.0  # 坡脚侧观测窗，固定为既有窗口诊断范围
SIDE_CLEARANCE = 4.0  # 侧向净空由 1h 扩至 4h，端点从 s=±6 外移至 s=±9
CASE_KEY = (50.0, 15, 20)  # 仅验证已暴露右侧低幅区的代表工况 h/i/入射角
WAVE_FILES = [  # 仅保留需判别的高频记录，避免重复计算已知正常的其他组合
    os.path.join(WORKSPACE_DIR, "Wave", "Impulse", "Acceleration", "ricker_wavelet_4Hz.txt"),
]
REFERENCE_NPZ = os.path.join(WORKSPACE_DIR, "test", "Abaqus", "P1_window_test", "case-P1win-L1-h50_i15_t20", "surface_results.npz")  # 1h 净空已完成参考包


def build_boundary_clearance_cases():  # 构造单工况边界净空对照配置
    """返回注入 `case_config.json` 的单个对照工况。"""
    h_val, i_val, th_val = CASE_KEY  # 解包坡高、坡角与入射角
    return [{
        "name": "P1clear-L1-h{}_i{}_t{}".format(base._fmt_num(h_val), i_val, th_val),  # 独立前缀避免覆盖原窗口试验
        "config": {
            "material_cfg": {"angle": th_val, "layers": []},  # 全基岩均质，保持原对照条件
            "geometry_cfg": {
                "slope_height": h_val,  # 坡高
                "slope_angle": float(i_val),  # 坡角
                "crest_window": CREST_WINDOW,  # 固定坡顶观测窗
                "toe_window": TOE_WINDOW,  # 固定坡脚观测窗
                "side_clearance": SIDE_CLEARANCE,  # 仅扩展观测窗外的边界缓冲
            },
            "time_cfg": {"tail_seconds": 3.0},  # 与原窗口诊断保持相同静默尾段
            "run_cfg": {"wave_files": WAVE_FILES},  # 仅运行 4 Hz Ricker 波
            "qa_cfg": {"mode": "window_convergence", "reference_npz": REFERENCE_NPZ,
                       "field": "TAF_h_comp", "tol": 0.05, "min_points": 1000},  # 以全窗口 1001 个 s 网格点验证收敛
        },
    }]


def main():  # 主入口
    """复用 P1 主流程，运行单工况边界净空对照。"""
    base.SMOKE_MODE = True  # 使用 smoke 模式以写入独立测试目录
    base.SMOKE_ROOT_DIR = os.path.join(WORKSPACE_DIR, "test", "Abaqus", "P1_boundary_clearance_test")  # 默认测试输出目录
    base.MAX_WORKERS = 1  # 单一工况无需并行，保留完整前台日志
    base.DELETE_FILE_TYPES = [".inp", ".msg", ".prt", ".dat", ".sta", ".sim", "jnl"]  # 成功后清理中间文件并保留 ODB 与结果包
    base.PARAMETER_CASES = build_boundary_clearance_cases()  # 注入本次唯一对照工况
    base.main()  # 调用主批处理流程


if __name__ == "__main__":  # 脚本直接执行时进入主流程
    main()  # 执行对照工况
