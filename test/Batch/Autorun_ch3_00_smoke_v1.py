# -*- coding: utf-8 -*-
"""第三章 U0 最小闭环：验证 Hybrid v2 建模、后处理、NPZ 收集与绘图链路。"""

from __future__ import print_function

import csv  # 导入索引读取模块
import datetime  # 导入时间记录模块
import hashlib  # 导入源文件哈希模块
import json  # 导入配置读写模块
import os  # 导入路径操作模块
import shutil  # 导入文件复制模块
import subprocess  # 导入外部命令执行模块
import sys  # 导入解释器和命令行参数模块


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # 定位仓库根目录
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'  # 优先使用环境变量指定的 Abaqus 启动器
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'test', 'Abaqus', 'ch3_00_smoke')  # 默认 U0 测试输出根目录
CASE_NAME = 'case-homogeneous-freefield'  # 最小均质自由场工况名
MAX_STEP_SECONDS = 3600  # 每个外部步骤的最长运行时间（秒）

MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'slope_frame_ssi_full_v2.py')  # 唯一建模入口
POSTPROCESS_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Postprocess_All_surface_v2.py')  # 单工况 ODB 后处理入口
COLLECT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Collect_All_results_v2.py')  # 跨工况 NPZ 收集入口
PLOT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Plot_Hybrid_surface_v2.py')  # 集中结果绘图入口
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')  # 宽频最小输入波

CASE_CONFIG = {
    'material_cfg': {
        'angle': 0.0,  # 垂直入射，降低 U0 输入和边界诊断复杂度
        'layers': [],  # 均质基岩，避免分层共振干扰执行链检查
        'surface_geometry': 'horizontal',  # 水平材料带，作为最小几何基准
    },
    'geometry_cfg': {
        'slope_height': 50.0,  # 小尺度坡高，缩短 U0 求解时间
        'slope_angle': 45.0,  # 典型中等坡角
        'crest_window': 2.0,  # 坡顶观测窗长度为 2h
        'toe_window': 2.0,  # 坡脚观测窗长度为 2h
        'side_clearance': 0.2,  # 观测窗外两侧各留 0.2h 边界净空
        'base_depth': 1.0,  # 坡脚面以下模型深度为 1h
    },
    'damping_cfg': {
        'enable': True,  # 保留小阻尼，避免自由振荡拖尾污染后处理
        'constant_xi': 0.01,  # 统一 1% 阻尼，仅作为数值稳定的最小设置
        'fc': 4.0,  # 与 Ricker 主频一致，避免主频识别的不确定性
        'anchor': 'input',  # U0 仅围绕输入主频拟合阻尼
    },
    'mesh_cfg': {
        'size': 8.0,  # 均质高速介质下的快速基准网格
        'auto': False,  # U0 固定网格，后续 U3 再验证自适应规则
        'elem': 'CPE4R',  # 与 Hybrid v2 默认单元体系一致
        'graded': False,  # 均质基准不启用分层渐变网格
    },
    'time_cfg': {
        'check': True,  # 保留时间步诊断日志
        'tail_seconds': 1.0,  # 保留 1 秒静默尾段供频域后处理检查
    },
    'freefield_cfg': {
        'engine': 'fd',  # 使用 v2 默认频域自由场引擎
        'include_damping': True,  # 自由场与有限元材料阻尼保持一致
    },
    'run_cfg': {
        'surface_only': True,  # 仅保留地表全时程输出，控制 ODB 规模
        'critical_angle_check': True,  # 垂直入射应通过临界角检查
        'wave_files': ['ricker_wavelet_4Hz.txt'],  # 只读取工况目录内明确复制的输入波
    },
    'tssi_cfg': {
        'enable': False,  # U0 验证场地链路，不叠加框架响应
        'scene': 'freefield',  # 强制纯坡地自由场场景并输出 CREST_REF
        'nonlinear': False,  # U0 不引入结构材料非线性
        'gravity': 'off',  # U0 不引入重力步
    },
}


def _require_file(path, label):  # 校验关键输入存在
    """若关键文件缺失则抛出含用途说明的错误。"""
    if not os.path.isfile(path):  # 检查文件存在性
        raise RuntimeError('%s不存在：%s' % (label, path))  # 明确指出缺失源


def _sha256(path):  # 计算运行源文件哈希
    """返回文件 SHA-256，用于冻结本次运行所用版本。"""
    digest = hashlib.sha256()  # 初始化哈希对象
    with open(path, 'rb') as handle:  # 以二进制方式读取文件
        while True:  # 分块读取避免大文件占用过多内存
            block = handle.read(1024 * 1024)  # 每次读取 1 MB
            if not block:  # 文件读完时退出循环
                break
            digest.update(block)  # 累加当前分块
    return digest.hexdigest()  # 返回十六进制哈希


def _write_json(path, value):  # 以 UTF-8 输出运行记录
    """将字典写为稳定排序的可读 JSON。"""
    with open(path, 'w', encoding='utf-8') as handle:  # 按 UTF-8 打开目标文件
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)  # 输出结构化内容
        handle.write('\n')  # 追加换行方便查看


def _run_command(command, cwd, log_path):  # 执行单一步骤并保留控制台日志
    """运行外部命令；失败或超时时保存日志后终止当前闭环。"""
    env = None  # 默认继承父进程环境
    if os.path.abspath(command[0]) == os.path.abspath(sys.executable):  # 系统 Python 后处理可明确约束输出编码
        env = os.environ.copy()  # 复制环境，避免影响 Abaqus Python 2 子进程
        env['PYTHONIOENCODING'] = 'utf-8'  # 使中文收集与绘图日志以 UTF-8 写入审计文件
    with open(log_path, 'wb') as handle:  # 二进制日志兼容 Abaqus 控制台编码
        handle.write(('命令：%s\n工作目录：%s\n\n' % (' '.join(command), cwd)).encode('utf-8'))  # 固化可复现命令
        try:
            result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=MAX_STEP_SECONDS, check=False, env=env)  # 不让 subprocess 隐式抛错
        except subprocess.TimeoutExpired:
            raise RuntimeError('步骤超时（>%ds），日志保留在：%s' % (MAX_STEP_SECONDS, log_path))  # 保留证据后停止
    if result.returncode != 0:  # 检查外部命令退出码
        raise RuntimeError('步骤失败，退出码=%s，日志保留在：%s' % (result.returncode, log_path))  # 阻止失败链路继续


def _assert_outputs(root_dir, case_dir):  # 按 U0 门槛检查最终产物
    """检查 ODB、NPZ、NPZ 索引和图件，返回可归档的验收摘要。"""
    odbs = [name for name in os.listdir(case_dir) if name.lower().startswith('job-') and name.lower().endswith('.odb')]  # 搜索求解 ODB
    npz_path = os.path.join(case_dir, 'surface_results.npz')  # 单工况最终数值包
    index_path = os.path.join(root_dir, 'results', 'index.csv')  # 跨工况索引
    missing = []  # 初始化缺失产物列表
    for key, path in [('case_meta', os.path.join(case_dir, 'case_meta.json')),
                      ('surface_npz', npz_path), ('results_index', index_path)]:  # 遍历必要文件
        if not os.path.isfile(path):  # 文件不存在即记录
            missing.append(key)
    if not odbs:  # 无 ODB 说明建模或求解未完成
        missing.append('job_odb')
    if missing:  # 任一关键产物缺失
        raise RuntimeError('U0 产物不完整：%s' % ', '.join(missing))  # 阻止误判通过
    with open(index_path, 'r', encoding='utf-8-sig', newline='') as handle:  # 读取带 BOM 的索引
        rows = list(csv.DictReader(handle))  # 读取全部记录行
    npz_rows = [row for row in rows if row.get('source_folder') == CASE_NAME and row.get('type') == 'SURFACE_RESULTS_NPZ']  # 校验收集器的新接口
    if not npz_rows:  # 没有最终数值包索引行
        raise RuntimeError('U0 收集器未在 index.csv 写入 SURFACE_RESULTS_NPZ 行')  # 给出接口级诊断
    figure_dir = os.path.join(root_dir, 'results', 'Fig_surface_panels')  # 统一绘图目录
    png_count = 0  # 初始化 PNG 计数
    for base, _, names in os.walk(figure_dir) if os.path.isdir(figure_dir) else []:  # 遍历图件目录
        png_count += len([name for name in names if name.lower().endswith('.png')])  # 统计 PNG 图件
    if png_count == 0:  # 无图件说明绘图阶段没有实质输出
        raise RuntimeError('U0 绘图阶段未输出 PNG 图件')  # 阻止假阳性通过
    return {'odb_files': sorted(odbs), 'surface_npz': npz_path,
            'index_npz_rows': len(npz_rows), 'png_count': png_count}  # 返回验收摘要


def _prepare(root_dir):  # 准备独立工况并冻结本次运行源文件
    """创建 U0 工况目录、复制必需文件、写入配置和运行清单，返回工况目录。"""
    case_dir = os.path.join(root_dir, CASE_NAME)  # 构造单一工况目录
    for path, label in [(ABAQUS_CMD, 'Abaqus 启动器'), (MODEL_SOURCE, '建模脚本'),
                        (POSTPROCESS_SOURCE, '单工况后处理脚本'), (COLLECT_SOURCE, '收集脚本'),
                        (PLOT_SOURCE, '绘图脚本'), (WAVE_SOURCE, '输入波')]:  # 全链路静态预检
        _require_file(path, label)  # 任何源文件缺失均不启动 Abaqus
    os.makedirs(case_dir, exist_ok=True)  # 创建独立工况目录
    for path in (MODEL_SOURCE, POSTPROCESS_SOURCE, WAVE_SOURCE):  # 复制单工况必需文件
        shutil.copy2(path, os.path.join(case_dir, os.path.basename(path)))  # 保留源文件时间戳
    _write_json(os.path.join(case_dir, 'case_config.json'), CASE_CONFIG)  # 写入唯一配置注入文件
    manifest = {'unit': 'U0', 'purpose': 'Hybrid v2 全流程最小闭环',
                'created_at': datetime.datetime.now().isoformat(), 'case_name': CASE_NAME,
                'case_config': CASE_CONFIG,
                'source_sha256': {os.path.basename(path): _sha256(path) for path in
                                  (MODEL_SOURCE, POSTPROCESS_SOURCE, COLLECT_SOURCE, PLOT_SOURCE, WAVE_SOURCE)}}  # 冻结源文件版本
    _write_json(os.path.join(root_dir, 'u0_run_manifest.json'), manifest)  # 写出运行清单
    return case_dir  # 返回已注入配置的工况目录


def _finalize(root_dir, case_dir):  # 完成单工况后处理、收集、绘图和验收
    """在已有 ODB 的工况上执行后处理与集中验收，并写出通过报告。"""
    _run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POSTPROCESS_SOURCE)], case_dir,
                 os.path.join(case_dir, 'autorun_02_postprocess.log'))  # 提取并打包 NPZ
    for path in (COLLECT_SOURCE, PLOT_SOURCE):  # 固定根目录全局脚本版本
        shutil.copy2(path, os.path.join(root_dir, os.path.basename(path)))  # 复制后从根目录执行
    _run_command([sys.executable, os.path.basename(COLLECT_SOURCE), root_dir], root_dir,
                 os.path.join(root_dir, 'autorun_03_collect.log'))  # 收集跨工况 NPZ
    _run_command([sys.executable, os.path.basename(PLOT_SOURCE), root_dir], root_dir,
                 os.path.join(root_dir, 'autorun_04_plot.log'))  # 生成独立图件
    report = _assert_outputs(root_dir, case_dir)  # 执行 U0 验收
    report.update({'status': 'passed', 'finished_at': datetime.datetime.now().isoformat()})  # 补充通过状态
    _write_json(os.path.join(root_dir, 'u0_validation_report.json'), report)  # 归档验收结果
    print('U0 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))  # 输出通过摘要


def next_run_dir(unit_root):  # 自动分配不可覆盖的论文运行目录
    """按 run-001、run-002 顺序创建下一次运行目录。"""
    os.makedirs(unit_root, exist_ok=True)  # 确保单元根目录存在
    used = [int(name[4:]) for name in os.listdir(unit_root)
            if name.startswith('run-') and name[4:].isdigit()]  # 收集既有编号
    return os.path.join(unit_root, 'run-%03d' % ((max(used) if used else 0) + 1))  # 返回下一编号


def main():  # 组织 U0 端到端执行
    """支持完整运行、仅准备和已有 ODB 的后处理验收三种模式。"""
    root_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else next_run_dir(DEFAULT_ROOT)  # 默认新建独立运行目录
    action = sys.argv[2] if len(sys.argv) >= 3 else '--run'  # 默认完整运行，长作业可拆为 prepare/finalize
    if action not in ('--run', '--prepare', '--finalize'):  # 校验动作参数
        raise RuntimeError('动作仅支持 --run/--prepare/--finalize，当前为：%s' % action)  # 防止误执行
    case_dir = _prepare(root_dir)  # 所有模式均先完成源文件和配置注入
    if action == '--prepare':  # 仅准备模式供外部调度器启动 Abaqus
        print('U0 准备完成：%s' % case_dir)  # 输出可启动工况目录
        return  # 不启动 Abaqus
    if action == '--run':  # 默认完整链路：建模求解后继续后处理
        _run_command([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], case_dir,
                     os.path.join(case_dir, 'autorun_01_model.log'))  # 建模并求解
    _finalize(root_dir, case_dir)  # 对完整运行或已有 ODB 的工况执行后处理和验收


if __name__ == '__main__':  # 直接运行时进入主流程
    main()  # 执行 U0
