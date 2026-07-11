# -*- coding: utf-8 -*-
"""第三章 U1：验证 Hybrid v2 的几何、分层和观测体系实现。"""

from __future__ import print_function

import csv  # 导入索引读取模块
import datetime  # 导入时间记录模块
import hashlib  # 导入源文件哈希模块
import json  # 导入配置读写模块
import math  # 导入几何公式模块
import os  # 导入路径操作模块
import re  # 导入日志解析模块
import shutil  # 导入文件复制模块
import subprocess  # 导入外部命令执行模块
import sys  # 导入解释器与命令行参数模块

import numpy as np  # 导入 NPZ 验收模块


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 定位仓库根目录
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'  # Abaqus 启动器
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_01_geometry_observation')  # 默认 U1 论文运行根目录
CASE_NAME = 'case-terrain-two-layer'  # 唯一 U1 工况名
MAX_STEP_SECONDS = 3600  # 单步最长运行时间

MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')  # 建模脚本
POSTPROCESS_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Postprocess_All_surface_v2.py')  # 单工况后处理
COLLECT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Collect_All_results_v2.py')  # 结果收集
PLOT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Plot_Hybrid_surface_v2.py')  # 统一绘图
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')  # 输入波

SLOPE_HEIGHT = 25.0  # U1 坡高（m）
SLOPE_ANGLE = 30.0  # U1 坡角（度）
CREST_WINDOW = 2.0  # 坡顶观测窗（h）
TOE_WINDOW = 2.0  # 坡脚观测窗（h）
SIDE_CLEARANCE = 1.0  # 侧向净空（h）
BASE_DEPTH = 2.0  # 坡脚面以下深度（h）
MESH_SIZE = 4.0  # U1 网格尺寸（m）
FRAME_WIDTH = 18.0  # 默认三跨框架宽度（m），freefield 坡顶参考点与 SSI 框架中心共用此口径
LAYER_NAMES = ('surface', 'base_layer')  # 预期有限土层名称

CASE_CONFIG = {
    'material_cfg': {
        'angle': 0.0,  # 垂直入射，U1 排除斜入射变量
        'surface_geometry': 'terrain',  # 验证沿地形等厚分层实现
        'layers': [
            {'name': 'surface', 'vs': 400.0, 'poisson_ratio': 0.30, 'density': 2000.0, 'thickness': 8.0},  # 表层
            {'name': 'base_layer', 'vs': 800.0, 'poisson_ratio': 0.30, 'density': 2200.0, 'thickness': 12.0},  # 下伏层
        ],
    },
    'geometry_cfg': {
        'slope_height': SLOPE_HEIGHT,  # 坡高
        'slope_angle': SLOPE_ANGLE,  # 坡角
        'crest_window': CREST_WINDOW,  # 坡顶观测窗
        'toe_window': TOE_WINDOW,  # 坡脚观测窗
        'side_clearance': SIDE_CLEARANCE,  # 侧向净空
        'base_depth': BASE_DEPTH,  # 底部净空
    },
    'damping_cfg': {
        'enable': True,  # 保持小应变线性阻尼
        'constant_xi': 0.01,  # 统一 1% 阻尼
        'fc': 4.0,  # 固定输入主频
        'anchor': 'perband',  # 覆盖分层驻波频带
    },
    'mesh_cfg': {
        'size': MESH_SIZE,  # 固定 U1 网格
        'auto': False,  # 网格收敛留给 U3
        'elem': 'CPE4R',  # 四节点减缩积分单元
        'graded': True,  # 验证分层渐变网格设置
        'min_elems_through_thickness': 4,  # 薄层至少四层单元
    },
    'time_cfg': {
        'check': True,  # 输出时间步诊断
        'tail_seconds': 0.5,  # 仅满足后处理需要的短尾段
    },
    'freefield_cfg': {
        'engine': 'fd',  # 使用频域自由场引擎
        'include_damping': True,  # 自由场与有限元阻尼一致
    },
    'run_cfg': {
        'surface_only': True,  # 输出地表全时程
        'critical_angle_check': True,  # 垂直入射必须通过临界角检查
        'wave_files': ['ricker_wavelet_4Hz.txt'],  # 使用工况目录内输入波
    },
    'tssi_cfg': {
        'enable': False,  # 不叠加框架结构
        'scene': 'freefield',  # 创建坡顶参考点而不建框架
        'nonlinear': False,  # U1 不引入结构非线性
        'gravity': 'off',  # U1 不引入重力步
    },
}


def _require(path, label):  # 校验关键文件存在
    """若关键文件缺失则中止，不启动不完整的工况。"""
    if not os.path.isfile(path):  # 检查物理文件
        raise RuntimeError('%s不存在：%s' % (label, path))  # 指出缺失来源


def _sha256(path):  # 计算源文件哈希
    """返回文件 SHA-256，用于复现实验版本冻结。"""
    digest = hashlib.sha256()  # 初始化哈希对象
    with open(path, 'rb') as handle:  # 二进制分块读取
        while True:  # 循环读取文件
            block = handle.read(1024 * 1024)  # 每块 1 MB
            if not block:  # 读完时退出
                break
            digest.update(block)  # 累加哈希
    return digest.hexdigest()  # 返回十六进制摘要


def _write_json(path, value):  # 写出 UTF-8 JSON
    """稳定写出配置、清单或验收报告。"""
    with open(path, 'w', encoding='utf-8') as handle:  # 以 UTF-8 打开文件
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)  # 输出可读 JSON
        handle.write('\n')  # 追加换行


def _run(command, cwd, log_path):  # 调用单个外部步骤
    """运行命令并保存完整日志，任一步非零退出即停止。"""
    env = None  # Abaqus 默认继承环境
    if os.path.abspath(command[0]) == os.path.abspath(sys.executable):  # 系统 Python 后处理日志强制 UTF-8
        env = os.environ.copy()  # 复制父环境
        env['PYTHONIOENCODING'] = 'utf-8'  # 固化中文日志编码
    with open(log_path, 'wb') as handle:  # 二进制记录控制台输出
        handle.write(('命令：%s\n工作目录：%s\n\n' % (' '.join(command), cwd)).encode('utf-8'))  # 记录复现命令
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=MAX_STEP_SECONDS, check=False, env=env)  # 执行命令
    if result.returncode != 0:  # 检查退出码
        raise RuntimeError('步骤失败，退出码=%s，日志：%s' % (result.returncode, log_path))  # 失败时停止链路


def _prepare(root_dir):  # 创建工况并注入配置
    """复制单工况文件、写入 case_config 和版本清单，返回工况目录。"""
    case_dir = os.path.join(root_dir, CASE_NAME)  # 生成工况目录
    for path, label in [(ABAQUS_CMD, 'Abaqus 启动器'), (MODEL_SOURCE, '建模脚本'),
                        (POSTPROCESS_SOURCE, '单工况后处理'), (COLLECT_SOURCE, '结果收集'),
                        (PLOT_SOURCE, '统一绘图'), (WAVE_SOURCE, '输入波')]:  # 全链路预检
        _require(path, label)  # 缺失即不开始
    os.makedirs(case_dir, exist_ok=True)  # 创建工况目录
    for source in (MODEL_SOURCE, POSTPROCESS_SOURCE, WAVE_SOURCE):  # 复制单工况文件
        shutil.copy2(source, os.path.join(case_dir, os.path.basename(source)))  # 保留源文件时间戳
    _write_json(os.path.join(case_dir, 'case_config.json'), CASE_CONFIG)  # 注入唯一参数来源
    manifest = {'unit': 'U1', 'purpose': '几何、分层与观测体系实现验证',
                'created_at': datetime.datetime.now().isoformat(), 'case_name': CASE_NAME,
                'case_config': CASE_CONFIG,
                'source_sha256': {os.path.basename(p): _sha256(p) for p in
                                  (MODEL_SOURCE, POSTPROCESS_SOURCE, COLLECT_SOURCE, PLOT_SOURCE, WAVE_SOURCE)}}  # 冻结脚本版本
    _write_json(os.path.join(root_dir, 'u1_run_manifest.json'), manifest)  # 写入运行清单
    return case_dir  # 返回已准备目录


def _npz_text(value):  # 解码 NPZ 内文本
    """将 Python 2 生成的 NPZ 标量转换为 UTF-8 文本。"""
    if hasattr(value, 'item'):  # NumPy 标量
        value = value.item()  # 取出标量
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)  # 返回文本


def _validate(root_dir, case_dir):  # 执行 U1 几何和观测口径验收
    """检查元数据、建模日志、NPZ 子网格和集中索引的一致性。"""
    meta_path = os.path.join(case_dir, 'case_meta.json')  # 元数据路径
    npz_path = os.path.join(case_dir, 'surface_results.npz')  # 最终数值包路径
    index_path = os.path.join(root_dir, 'results', 'index.csv')  # 集中索引路径
    for path, label in [(meta_path, 'case_meta'), (npz_path, 'surface_results.npz'), (index_path, 'results/index.csv')]:  # 逐项检查产物
        _require(path, label)  # 缺失即失败
    with open(meta_path, 'r', encoding='utf-8') as handle:  # 读取工况元数据
        meta = json.load(handle)  # 解析 JSON
    geom = meta.get('geometry', {})  # 几何派生量
    expected_left = (CREST_WINDOW + SIDE_CLEARANCE) * SLOPE_HEIGHT  # 坡顶平台长度
    expected_right = (TOE_WINDOW + SIDE_CLEARANCE) * SLOPE_HEIGHT  # 坡脚平台长度
    expected_w = SLOPE_HEIGHT / math.tan(math.radians(SLOPE_ANGLE))  # 坡面水平投影
    expected_total = expected_left + expected_w + expected_right  # 总长度
    for label, actual, expected in [('left_flat', geom.get('left_flat'), expected_left),
                                    ('w_slope', geom.get('w_slope'), expected_w),
                                    ('total_L', geom.get('total_L'), expected_total),
                                    ('x_crest', geom.get('x_crest'), expected_left),
                                    ('x_toe', geom.get('x_toe'), expected_left + expected_w)]:  # 校验几何公式
        if actual is None or abs(float(actual) - expected) > 1.0e-8 * max(1.0, abs(expected)):  # 使用数值容差比较
            raise RuntimeError('U1 几何换算不一致：%s=%r，期望=%r' % (label, actual, expected))  # 报告具体字段
    if meta.get('surface_geometry') != 'terrain':  # 验证沿地形分层开关
        raise RuntimeError('U1 表层几何未保持 terrain：%r' % meta.get('surface_geometry'))  # 阻止错误口径
    if int(meta.get('derived', {}).get('n_finite_layers', -1)) != len(LAYER_NAMES):  # 验证有限层数
        raise RuntimeError('U1 有限层数不符：%r' % meta.get('derived', {}).get('n_finite_layers'))  # 报告层数错误
    if tuple(layer.get('name') for layer in meta.get('layers', [])) != LAYER_NAMES:  # 验证层名与顺序
        raise RuntimeError('U1 材料带名称或顺序不符：%r' % meta.get('layers'))  # 报告材料错误
    with open(os.path.join(case_dir, 'slope_frame_ssi_full_v2.log'), 'r', encoding='utf-8') as handle:  # 读取建模日志
        model_log = handle.read()  # 读入完整日志
    top_match = re.search(r'TOP_SURFACE=(\d+)', model_log)  # 提取地表节点集数量
    crest_match = re.search(r'CREST_REF 已建: x=([0-9.+-Ee]+)\(目标x=([0-9.+-Ee]+)\)', model_log)  # 提取坡顶参考点坐标
    if not top_match or int(top_match.group(1)) < 5:  # 地表节点集必须存在且可支撑空间观测
        raise RuntimeError('U1 未在日志中确认足够的 TOP_SURFACE 节点')  # 报告节点集问题
    expected_crest_target = expected_left - FRAME_WIDTH / 2.0  # freefield 参考点与默认框架基础中心同口径
    if not crest_match or abs(float(crest_match.group(2)) - expected_crest_target) > 1.0e-8:  # 先校验日志目标坐标
        raise RuntimeError('U1 坡顶参考点目标坐标不符')  # 报告参考点定义错误
    if abs(float(crest_match.group(1)) - expected_crest_target) > MESH_SIZE * 1.1:  # 实际节点允许一个网格尺寸的离散偏差
        raise RuntimeError('U1 坡顶参考点未落在框架中心邻域')  # 报告坡顶点错误
    package = np.load(npz_path)  # 打开最终数值包
    try:
        params = json.loads(_npz_text(package['sgrid_params_json']))  # 读取三段子网格参数
        manifest = json.loads(_npz_text(package['manifest_json']))  # 读取数值表清单
        target = next(item for item in manifest if item['name'] == 'sgrid_response_ricker_wavelet_4Hz.csv')  # 定位统一响应表
        row_count = int(package[target['key'] + '_data'].shape[0])  # 读取统一表行数
    finally:
        package.close()  # 关闭压缩包
    expected_rows = int(params['N_A']) + int(params['N_B']) + int(params['N_C'])  # 由三段网格参数计算期望点数
    if abs(float(params['h_slope']) - SLOPE_HEIGHT) > 1.0e-10 or float(params['A_max']) != CREST_WINDOW or float(params['C_max']) != TOE_WINDOW:  # 验证坐标尺度和观测窗
        raise RuntimeError('U1 三段归一化坐标参数不符：%r' % params)  # 报告坐标口径错误
    if row_count != expected_rows:  # 验证统一响应表固定长度
        raise RuntimeError('U1 s 网格行数不符：%d，期望=%d' % (row_count, expected_rows))  # 报告重采样错误
    with open(index_path, 'r', encoding='utf-8-sig', newline='') as handle:  # 读取集中索引
        rows = list(csv.DictReader(handle))  # 读取所有索引行
    if not any(row.get('source_folder') == CASE_NAME and row.get('type') == 'SURFACE_RESULTS_NPZ' for row in rows):  # 验证收集器写入最终包
        raise RuntimeError('U1 集中索引缺少最终 NPZ 记录')  # 报告收集错误
    return {'status': 'passed', 'top_surface_nodes': int(top_match.group(1)), 'crest_ref_x': float(crest_match.group(1)),
            'expected_crest_x': expected_crest_target, 'sgrid_rows': row_count, 'sgrid_params': params,
            'geometry': {'left_flat': expected_left, 'w_slope': expected_w, 'total_L': expected_total}}  # 返回验收摘要


def _finalize(root_dir, case_dir):  # 执行后处理和 U1 验收
    """从已有 ODB 生成最终数据、收集、绘图和验证报告。"""
    _run([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POSTPROCESS_SOURCE)], case_dir,
         os.path.join(case_dir, 'autorun_02_postprocess.log'))  # ODB 后处理
    for source in (COLLECT_SOURCE, PLOT_SOURCE):  # 复制全局后处理脚本
        shutil.copy2(source, os.path.join(root_dir, os.path.basename(source)))  # 固定本次版本
    _run([sys.executable, os.path.basename(COLLECT_SOURCE), root_dir], root_dir,
         os.path.join(root_dir, 'autorun_03_collect.log'))  # 收集 NPZ
    _run([sys.executable, os.path.basename(PLOT_SOURCE), root_dir], root_dir,
         os.path.join(root_dir, 'autorun_04_plot.log'))  # 输出图件
    report = _validate(root_dir, case_dir)  # 执行 U1 验收
    report['finished_at'] = datetime.datetime.now().isoformat()  # 写入结束时间
    _write_json(os.path.join(root_dir, 'u1_validation_report.json'), report)  # 归档报告
    print('U1 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))  # 输出摘要


def next_run_dir(unit_root):  # 自动分配不可覆盖的论文运行目录
    """按 run-001、run-002 顺序创建下一次运行目录。"""
    os.makedirs(unit_root, exist_ok=True)  # 确保单元根目录存在
    used = [int(name[4:]) for name in os.listdir(unit_root)
            if name.startswith('run-') and name[4:].isdigit()]  # 收集既有编号
    return os.path.join(unit_root, 'run-%03d' % ((max(used) if used else 0) + 1))  # 返回下一编号


def main():  # 组织 U1 分阶段执行
    """支持完整运行、仅准备及已有 ODB 的后处理验收。"""
    root_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else next_run_dir(DEFAULT_ROOT)  # 默认新建独立运行目录
    action = sys.argv[2] if len(sys.argv) >= 3 else '--run'  # 默认完整运行
    if action not in ('--run', '--prepare', '--finalize'):  # 校验动作
        raise RuntimeError('动作仅支持 --run/--prepare/--finalize，当前为：%s' % action)  # 拒绝未知动作
    case_dir = _prepare(root_dir)  # 创建工况和注入配置
    if action == '--prepare':  # 准备模式
        print('U1 准备完成：%s' % case_dir)  # 输出工况目录
        return  # 不提交作业
    if action == '--run':  # 完整模式需先求解
        _run([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], case_dir,
             os.path.join(case_dir, 'autorun_01_model.log'))  # 建模并提交求解
    _finalize(root_dir, case_dir)  # 后处理、收集、绘图和验收


if __name__ == '__main__':  # 直接运行时进入主流程
    main()  # 执行 U1
