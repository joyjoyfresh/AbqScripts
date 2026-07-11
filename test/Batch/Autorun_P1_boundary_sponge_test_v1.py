# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""P1 右侧散射尾波的阻尼海绵修复验证批处理脚本。

用途：
    复用已收敛的 4h 侧向净空，只在观测窗外的缓冲带启用阻尼海绵层，检验其能否
    吸收传至人工边界的散射尾波并使右端远场 QA 回到 5% 阈值内，同时验证
    s∈[-5, 5] 的研究响应不发生实质改变。

运行：
    python C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Batch\\Autorun_P1_boundary_sponge_test_v1.py
    python C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Batch\\Autorun_P1_boundary_sponge_test_v1.py C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Abaqus\\P1_boundary_sponge_test
"""

import os  # 导入路径模块
import sys  # 导入系统模块以注入主批处理脚本目录


WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 工作区根目录
BATCH_DIR = os.path.join(WORKSPACE_DIR, "Batch")  # 主批处理脚本目录
if BATCH_DIR not in sys.path:  # 避免重复插入路径
    sys.path.insert(0, BATCH_DIR)  # 允许从 test/Batch 导入 Batch 内主脚本

import Autorun_P1_v1 as base  # 复用 P1 主批处理的执行逻辑


CREST_WINDOW = 5.0  # 固定坡顶观测窗，保证与已收敛对照可比
TOE_WINDOW = 4.0  # 固定坡脚观测窗，保证与已收敛对照可比
SIDE_CLEARANCE = 4.0  # 保留已验证的 4h 侧向缓冲
SPONGE_WIDTH = 100.0  # 海绵宽 2h，仅覆盖 s<-7 与 s>7 的边界缓冲末端
SPONGE_GRADES = 5  # 渐变阻尼分级数
SPONGE_XI_MAX = 0.30  # 贴边界处附加阻尼比
CASE_KEY = (50.0, 15, 20)  # 仅验证暴露问题的代表工况 h/i/入射角
WAVE_FILES = [  # 仅运行触发右端 QA 超限的 4 Hz Ricker 波
    os.path.join(WORKSPACE_DIR, "Wave", "Impulse", "Acceleration", "ricker_wavelet_4Hz.txt"),
]


def build_boundary_sponge_cases():  # 构造单工况海绵修复验证配置
    """返回注入 `case_config.json` 的单个海绵修复工况。"""
    h_val, i_val, th_val = CASE_KEY  # 解包坡高、坡角与入射角
    return [{
        "name": "P1spongeLR-L1-h{}_i{}_t{}".format(base._fmt_num(h_val), i_val, th_val),  # 独立前缀避免覆盖无海绵对照
        "config": {
            "material_cfg": {"angle": th_val, "layers": []},  # 全基岩均质，保持原对照条件
            "geometry_cfg": {
                "slope_height": h_val,  # 坡高
                "slope_angle": float(i_val),  # 坡角
                "crest_window": CREST_WINDOW,  # 固定坡顶观测窗
                "toe_window": TOE_WINDOW,  # 固定坡脚观测窗
                "side_clearance": SIDE_CLEARANCE,  # 固定已验证的边界缓冲
            },
            "boundary_cfg": {
                "sponge_enable": True,  # 启用观测窗外的渐变阻尼海绵
                "sponge_sides": ["left", "right"],  # 仅处理侧边界，避免浅模型的底部海绵侵入全域
                "sponge_width": SPONGE_WIDTH,  # 海绵宽度
                "sponge_grades": SPONGE_GRADES,  # 阻尼渐变分级数
                "sponge_xi_max": SPONGE_XI_MAX,  # 贴边界附加阻尼比
            },
            "time_cfg": {"tail_seconds": 3.0},  # 与对照保持相同静默尾段
            "run_cfg": {"wave_files": WAVE_FILES},  # 仅运行 4 Hz Ricker 波
        },
    }]


def main():  # 主入口
    """复用 P1 主流程，运行单工况阻尼海绵修复验证。"""
    base.SMOKE_MODE = True  # 使用 smoke 模式以写入独立测试目录
    base.SMOKE_ROOT_DIR = os.path.join(WORKSPACE_DIR, "test", "Abaqus", "P1_boundary_lateral_sponge_test")  # 默认测试输出目录
    base.MAX_WORKERS = 1  # 单一工况无需并行，保留完整前台日志
    base.DELETE_FILE_TYPES = [".inp", ".msg", ".prt", ".dat", ".sta", ".sim", "jnl"]  # 成功后清理中间文件并保留 ODB 与结果包
    base.PARAMETER_CASES = build_boundary_sponge_cases()  # 注入本次唯一修复验证工况
    base.main()  # 调用主批处理流程


if __name__ == "__main__":  # 脚本直接执行时进入主流程
    main()  # 执行海绵修复验证
