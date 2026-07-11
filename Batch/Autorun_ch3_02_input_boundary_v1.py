# -*- coding: utf-8 -*-
"""第三章 U2：验证斜入射输入和三侧人工边界的一致性。"""

from __future__ import print_function

import csv  # 导入集中索引检查模块
import datetime  # 导入运行时间记录模块
import json  # 导入配置与报告模块
import os  # 导入路径操作模块
import shutil  # 导入文件复制模块
import subprocess  # 导入外部命令模块
import sys  # 导入命令行模块

import numpy as np  # 导入 NPZ 验收模块


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 定位仓库根目录
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'  # Abaqus 启动器
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_02_input_boundary')  # 默认 U2 论文运行根目录
CASE_NAME = 'case-oblique-homogeneous'  # U2 唯一工况名
MAX_STEP_SECONDS = 3600  # 单步最大运行时长

MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')  # 建模入口
POST_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Postprocess_All_surface_v2.py')  # ODB 后处理入口
COLLECT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Collect_All_results_v2.py')  # NPZ 收集入口
PLOT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Plot_Hybrid_surface_v2.py')  # 绘图入口
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')  # 输入波
INCIDENT_ANGLE = 15.0  # U2 验证的 SV 入射角

CASE_CONFIG = {
    'material_cfg': {'angle': INCIDENT_ANGLE, 'layers': [], 'surface_geometry': 'horizontal'},  # 均质介质便于隔离输入与边界因素
    'geometry_cfg': {'slope_height': 25.0, 'slope_angle': 30.0, 'crest_window': 2.0,
                     'toe_window': 2.0, 'side_clearance': 6.0, 'base_depth': 5.0},  # U2 回归：加大远场距离，检验坡体散射是否污染端点 QA
    'damping_cfg': {'enable': True, 'constant_xi': 0.01, 'fc': 4.0, 'anchor': 'input'},  # 固定阻尼口径
    'mesh_cfg': {'size': 8.0, 'auto': False, 'elem': 'CPE4R', 'graded': False},  # 均质快速网格
    'time_cfg': {'check': True, 'tail_seconds': 0.5},  # 为频域后处理保留短尾段
    'freefield_cfg': {'engine': 'fd', 'include_damping': True},  # 频域精确自由场
    'run_cfg': {'surface_only': True, 'critical_angle_check': True, 'wave_files': ['ricker_wavelet_4Hz.txt']},  # 显式输入波
    'tssi_cfg': {'enable': False, 'scene': 'freefield', 'nonlinear': False, 'gravity': 'off'},  # 纯场地场景
}


def require_file(path, label):  # 校验关键文件存在
    """关键输入缺失时停止，避免启动不完整工况。"""
    if not os.path.isfile(path):  # 检查文件
        raise RuntimeError('%s不存在：%s' % (label, path))  # 报告缺失项


def write_json(path, value):  # 写入 UTF-8 报告
    """写出缩进 JSON，便于后续审计。"""
    with open(path, 'w', encoding='utf-8') as handle:  # 按 UTF-8 打开
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)  # 输出内容
        handle.write('\n')  # 追加换行


def run_step(command, cwd, log_path):  # 运行并记录一个外部步骤
    """保存命令输出；非零退出即终止后续链路。"""
    env = None  # Abaqus 默认继承环境
    if os.path.abspath(command[0]) == os.path.abspath(sys.executable):  # 系统 Python 日志统一 UTF-8
        env = os.environ.copy()  # 复制环境
        env['PYTHONIOENCODING'] = 'utf-8'  # 固化中文编码
    with open(log_path, 'wb') as handle:  # 二进制记录控制台
        handle.write(('命令：%s\n工作目录：%s\n\n' % (' '.join(command), cwd)).encode('utf-8'))  # 记录复现命令
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=MAX_STEP_SECONDS, check=False, env=env)  # 执行外部程序
    if result.returncode != 0:  # 检查退出码
        raise RuntimeError('步骤失败，退出码=%s，日志：%s' % (result.returncode, log_path))  # 保留证据后终止


def prepare(root_dir):  # 建立独立 U2 工况
    """复制四脚本链所需文件并写入参数注入配置。"""
    case_dir = os.path.join(root_dir, CASE_NAME)  # 构造工况目录
    for path, label in [(ABAQUS_CMD, 'Abaqus 启动器'), (MODEL_SOURCE, '建模脚本'), (POST_SOURCE, '单工况后处理'),
                        (COLLECT_SOURCE, '收集脚本'), (PLOT_SOURCE, '绘图脚本'), (WAVE_SOURCE, '输入波')]:  # 预检全链路
        require_file(path, label)  # 缺失即停止
    os.makedirs(case_dir, exist_ok=True)  # 创建目录
    for source in (MODEL_SOURCE, POST_SOURCE, WAVE_SOURCE):  # 复制单工况文件
        shutil.copy2(source, os.path.join(case_dir, os.path.basename(source)))  # 固定本次运行版本
    write_json(os.path.join(case_dir, 'case_config.json'), CASE_CONFIG)  # 写入唯一配置来源
    write_json(os.path.join(root_dir, 'u2_run_manifest.json'), {'unit': 'U2', 'created_at': datetime.datetime.now().isoformat(),
               'case_name': CASE_NAME, 'case_config': CASE_CONFIG})  # 写入运行清单
    return case_dir  # 返回工况目录


def npz_text(value):  # 解码 Python 2 生成的 NPZ 文本
    """将 NPZ 标量统一转换为 UTF-8 文本。"""
    if hasattr(value, 'item'):  # NumPy 标量
        value = value.item()  # 提取标量
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)  # 返回文本


def validate(root_dir, case_dir):  # 验收输入和边界一致性
    """检查角度注入、FD 自检、人工边界日志和远场质量标记。"""
    meta_path = os.path.join(case_dir, 'case_meta.json')  # 工况元数据
    npz_path = os.path.join(case_dir, 'surface_results.npz')  # 最终数值包
    index_path = os.path.join(root_dir, 'results', 'index.csv')  # 集中索引
    for path, label in [(meta_path, 'case_meta'), (npz_path, 'surface_results.npz'), (index_path, 'results/index.csv')]:  # 必要产物
        require_file(path, label)  # 缺失即失败
    with open(meta_path, 'r', encoding='utf-8') as handle:  # 读取元数据
        meta = json.load(handle)  # 解析 JSON
    if abs(float(meta.get('incident_angle')) - INCIDENT_ANGLE) > 1.0e-12:  # 校验斜入射参数
        raise RuntimeError('U2 入射角未正确注入：%r' % meta.get('incident_angle'))  # 报告配置错误
    selfcheck = meta.get('selfcheck', {})  # 读取 FD 自检误差
    if float(selfcheck.get('halfspace_err', 1.0)) > 1.0e-3 or float(selfcheck.get('single_layer_err', 1.0)) > 1.0e-3:  # 解析对拍门槛
        raise RuntimeError('U2 FD 自检未通过：%r' % selfcheck)  # 报告输入引擎问题
    with open(os.path.join(case_dir, 'slope_frame_ssi_full_v2.log'), 'r', encoding='utf-8') as handle:  # 读取建模日志
        log_text = handle.read()  # 读取全部日志
    for token in ('自由场引擎=fd', '入射角=15.0000°', '边界[l]弹簧-阻尼器已创建', '边界[r]弹簧-阻尼器已创建', '边界[b]弹簧-阻尼器已创建'):  # 三侧边界与斜入射日志锚点
        if token not in log_text:  # 任一锚点缺失
            raise RuntimeError('U2 建模日志缺少：%s' % token)  # 报告边界实现错误
    package = np.load(npz_path)  # 打开最终包
    try:
        summary = json.loads(npz_text(package['surface_summary_json']))  # 读取质量摘要
    finally:
        package.close()  # 关闭压缩包
    records = summary.get('records', [])  # 获取逐记录摘要
    if len(records) != 1:  # U2 只允许一条输入记录
        raise RuntimeError('U2 摘要记录数不符：%d' % len(records))  # 报告后处理错误
    record = records[0]  # 获取唯一记录
    if bool(record.get('suspect')):  # 远场对拍必须通过
        raise RuntimeError('U2 远场 QA 失败：%r' % record)  # 提醒先做域与边界诊断
    for key in ('qa_farfield_err_left', 'qa_farfield_err_right'):  # 检查左右端误差
        if float(record.get(key, 1.0)) > 0.05:  # 5% 门槛
            raise RuntimeError('U2 %s 超过5%%：%r' % (key, record.get(key)))  # 报告误差
    with open(index_path, 'r', encoding='utf-8-sig', newline='') as handle:  # 读取集中索引
        rows = list(csv.DictReader(handle))  # 获取索引行
    if not any(row.get('source_folder') == CASE_NAME and row.get('type') == 'SURFACE_RESULTS_NPZ' and row.get('suspect') == 'False' for row in rows):  # 检查质量标记随收集器传递
        raise RuntimeError('U2 结果索引未保留通过的质量标记')  # 报告收集错误
    return {'status': 'passed', 'incident_angle': INCIDENT_ANGLE, 'fd_selfcheck': selfcheck,
            'farfield_err_left': record['qa_farfield_err_left'], 'farfield_err_right': record['qa_farfield_err_right'],
            'AR_max': record['AR_max']}  # 返回可审计摘要


def finalize(root_dir, case_dir):  # 执行后处理、收集、绘图与验收
    """从已有 ODB 生成最终数据并写出 U2 报告。"""
    run_step([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POST_SOURCE)], case_dir,
             os.path.join(case_dir, 'autorun_02_postprocess.log'))  # ODB 后处理
    for source in (COLLECT_SOURCE, PLOT_SOURCE):  # 固定全局脚本版本
        shutil.copy2(source, os.path.join(root_dir, os.path.basename(source)))  # 复制到根目录
    run_step([sys.executable, os.path.basename(COLLECT_SOURCE), root_dir], root_dir,
             os.path.join(root_dir, 'autorun_03_collect.log'))  # 收集 NPZ
    run_step([sys.executable, os.path.basename(PLOT_SOURCE), root_dir], root_dir,
             os.path.join(root_dir, 'autorun_04_plot.log'))  # 输出图件
    report = validate(root_dir, case_dir)  # 执行质量验收
    report['finished_at'] = datetime.datetime.now().isoformat()  # 记录结束时间
    write_json(os.path.join(root_dir, 'u2_validation_report.json'), report)  # 写出报告
    print('U2 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))  # 输出通过摘要


def next_run_dir(unit_root):  # 自动分配不可覆盖的论文运行目录
    """按 run-001、run-002 顺序创建下一次运行目录。"""
    os.makedirs(unit_root, exist_ok=True)  # 确保单元根目录存在
    used = [int(name[4:]) for name in os.listdir(unit_root)
            if name.startswith('run-') and name[4:].isdigit()]  # 收集既有编号
    return os.path.join(unit_root, 'run-%03d' % ((max(used) if used else 0) + 1))  # 返回下一编号


def main():  # 支持完整运行与已有 ODB 回归
    """执行 U2，支持 --run、--prepare 与 --finalize 三种模式。"""
    root_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else next_run_dir(DEFAULT_ROOT)  # 默认新建独立运行目录
    action = sys.argv[2] if len(sys.argv) >= 3 else '--run'  # 读取动作
    if action not in ('--run', '--prepare', '--finalize'):  # 校验动作
        raise RuntimeError('动作仅支持 --run/--prepare/--finalize')  # 拒绝未知动作
    case_dir = prepare(root_dir)  # 注入工况
    if action == '--prepare':  # 仅准备
        print('U2 准备完成：%s' % case_dir)  # 输出目录
        return  # 不求解
    if action == '--run':  # 完整运行先建模求解
        run_step([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], case_dir,
                 os.path.join(case_dir, 'autorun_01_model.log'))  # 建模求解
    finalize(root_dir, case_dir)  # 后处理与验收


if __name__ == '__main__':  # 直接运行入口
    main()  # 执行 U2
