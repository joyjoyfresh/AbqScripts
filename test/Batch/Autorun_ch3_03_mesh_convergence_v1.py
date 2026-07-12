# -*- coding: utf-8 -*-
"""第三章 U3a：Hybrid v2 网格收敛全流程试验。"""

from __future__ import print_function

import datetime  # 导入时间记录模块
import json  # 导入配置与报告模块
import os  # 导入路径模块
import shutil  # 导入文件复制模块
import subprocess  # 导入外部命令模块
import sys  # 导入命令行模块

import numpy as np  # 导入曲线收敛计算模块


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 定位仓库根目录
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'  # Abaqus 启动器
DEFAULT_ROOT = os.path.join(REPO_ROOT, 'Run', 'ch3_03_numerical_convergence')  # U3 论文运行根目录
MAX_STEP_SECONDS = 3600  # 每个外部步骤最长运行时间

MODEL_SOURCE = os.path.join(REPO_ROOT, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')  # 建模入口
POST_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Postprocess_All_surface_v2.py')  # ODB 后处理入口
COLLECT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Collect_All_results_v2.py')  # NPZ 收集入口
PLOT_SOURCE = os.path.join(REPO_ROOT, 'Postprocess', 'Hybrid', 'Plot_Hybrid_surface_v2.py')  # 绘图入口
WAVE_SOURCE = os.path.join(REPO_ROOT, 'Wave', 'Impulse', 'Acceleration', 'ricker_wavelet_4Hz.txt')  # 输入波
MESH_SIZES = (12.0, 8.0, 6.0)  # 粗、中、细三级网格


def require_file(path, label):
    """检查全流程依赖文件。"""
    if not os.path.isfile(path):
        raise RuntimeError('%s不存在：%s' % (label, path))


def write_json(path, data):
    """以 UTF-8 写出可审计 JSON。"""
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)


def run_step(command, cwd, log_path):
    """运行外部步骤并将输出完整保存。"""
    env = None
    if os.path.abspath(command[0]) == os.path.abspath(sys.executable):
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
    with open(log_path, 'wb') as handle:
        handle.write(('命令：%s\n工作目录：%s\n\n' % (' '.join(command), cwd)).encode('utf-8'))
        handle.flush()
        result = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT,
                                timeout=MAX_STEP_SECONDS, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError('步骤失败，退出码=%s，日志：%s' % (result.returncode, log_path))


def next_run_dir(unit_root):
    """自动创建下一次不可覆盖的 run-### 目录。"""
    os.makedirs(unit_root, exist_ok=True)
    used = [int(name[4:]) for name in os.listdir(unit_root)
            if name.startswith('run-') and name[4:].isdigit()]
    return os.path.join(unit_root, 'run-%03d' % ((max(used) if used else 0) + 1))


def case_name(size):
    """生成稳定的网格工况名称。"""
    return 'case-mesh-%sm' % ('%g' % size)


def case_config(size):
    """构造仅改变网格尺寸的 U3a 配置。"""
    return {
        'material_cfg': {'angle': 0.0, 'layers': [], 'surface_geometry': 'horizontal'},
        'geometry_cfg': {'slope_height': 25.0, 'slope_angle': 30.0, 'crest_window': 2.0,
                         'toe_window': 2.0, 'side_clearance': 6.0, 'base_depth': 5.0},
        'damping_cfg': {'enable': True, 'constant_xi': 0.01, 'fc': 4.0, 'anchor': 'input'},
        'mesh_cfg': {'size': size, 'auto': False, 'elem': 'CPE4R', 'graded': False},
        'time_cfg': {'tail_seconds': 1.0, 'steps_per_period': 100.0, 'max_increment': 0.001},
        'freefield_cfg': {'engine': 'fd', 'include_damping': True, 'bottom_ymax_mode': 'local'},
        'boundary_cfg': {'dashpot_scale': 1.0, 'spring_scale': 1.0},
        'run_cfg': {'wave_files': [WAVE_SOURCE], 'cpu_num': 8, 'output_interval': 1,
                    'surface_only': True, 'cleanup_intermediate': False},
        'tssi_cfg': {'scene': 'freefield'}
    }


def prepare(root_dir):
    """创建三级网格工况并写入参数配置。"""
    for path, label in [(ABAQUS_CMD, 'Abaqus 启动器'), (MODEL_SOURCE, '建模脚本'),
                        (POST_SOURCE, '后处理脚本'), (COLLECT_SOURCE, '收集脚本'),
                        (PLOT_SOURCE, '绘图脚本'), (WAVE_SOURCE, '输入波')]:
        require_file(path, label)
    cases = []
    for size in MESH_SIZES:
        directory = os.path.join(root_dir, case_name(size))
        os.makedirs(directory, exist_ok=True)
        for source in (MODEL_SOURCE, POST_SOURCE, WAVE_SOURCE):
            shutil.copy2(source, os.path.join(directory, os.path.basename(source)))
        write_json(os.path.join(directory, 'case_config.json'), case_config(size))
        cases.append((size, directory))
    write_json(os.path.join(root_dir, 'u3a_run_manifest.json'), {
        'unit': 'U3a', 'created_at': datetime.datetime.now().isoformat(),
        'variable': 'mesh_size', 'levels': list(MESH_SIZES)})
    return cases


def npz_text(value):
    """解码由 Abaqus Python 2 写出的 NPZ 文本。"""
    if hasattr(value, 'item'):
        value = value.item()
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def read_curve(case_dir):
    """读取固定 501 点 s 网格上的 TAF_h 与质量摘要。"""
    path = os.path.join(case_dir, 'surface_results.npz')
    require_file(path, 'surface_results.npz')
    package = np.load(path, allow_pickle=False)
    try:
        table_key = None
        for key in package.files:
            if not key.endswith('_header'):
                continue
            fields = [npz_text(item) for item in package[key]]
            if 's' in fields and ('seg' in fields or 'segment' in fields) and 'TAF_h' in fields:
                table_key = key[:-7]
                header = fields
                break
        if table_key is None:
            raise RuntimeError('NPZ 中缺少 sgrid_response 表')
        data = package[table_key + '_data']
        s_index = header.index('s')
        taf_index = header.index('TAF_h')
        s_coord = np.array([float(npz_text(row[s_index])) for row in data])
        taf = np.array([float(npz_text(row[taf_index])) for row in data])
        summary = json.loads(npz_text(package['surface_summary_json']))['records'][0]
    finally:
        package.close()
    return s_coord, taf, summary


def validate(root_dir, cases):
    """以细网格为参考检查中等网格的整曲线与峰值收敛。"""
    curves = {}
    summaries = {}
    for size, directory in cases:
        s_coord, taf, summary = read_curve(directory)
        if bool(summary.get('suspect')):
            raise RuntimeError('网格 %.1f m 的远场 QA 未通过：%r' % (size, summary))
        curves[size] = (s_coord, taf)
        summaries[size] = summary
    s_mid, y_mid = curves[8.0]
    s_ref, y_ref = curves[6.0]
    if not np.allclose(s_mid, s_ref):
        y_mid = np.interp(s_ref, s_mid, y_mid)
    mask = np.isfinite(y_mid) & np.isfinite(y_ref)
    curve_l2 = float(np.linalg.norm(y_mid[mask] - y_ref[mask]) / np.linalg.norm(y_ref[mask]))
    peak_mid = float(summaries[8.0]['AR_max'])
    peak_ref = float(summaries[6.0]['AR_max'])
    peak_rel = abs(peak_mid / peak_ref - 1.0)
    report = {'status': 'passed' if max(curve_l2, peak_rel) <= 0.05 else 'failed',
              'criterion': '8m 与 6m 的整曲线 L2 相对差和 AR_max 相对差均不超过5%',
              'curve_l2_relative': curve_l2, 'peak_relative': peak_rel,
              'AR_max': {str(size): summaries[size]['AR_max'] for size in MESH_SIZES},
              'finished_at': datetime.datetime.now().isoformat()}
    write_json(os.path.join(root_dir, 'u3a_mesh_validation_report.json'), report)
    if report['status'] != 'passed':
        raise RuntimeError('U3a 网格收敛未通过：%r' % report)
    return report


def finalize(root_dir, cases):
    """执行三级工况后处理、统一收集、绘图和收敛验收。"""
    for _, directory in cases:
        run_step([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(POST_SOURCE)], directory,
                 os.path.join(directory, 'autorun_02_postprocess.log'))
    for source in (COLLECT_SOURCE, PLOT_SOURCE):
        shutil.copy2(source, os.path.join(root_dir, os.path.basename(source)))
    run_step([sys.executable, os.path.basename(COLLECT_SOURCE), root_dir], root_dir,
             os.path.join(root_dir, 'autorun_03_collect.log'))
    run_step([sys.executable, os.path.basename(PLOT_SOURCE), root_dir], root_dir,
             os.path.join(root_dir, 'autorun_04_plot.log'))
    report = validate(root_dir, cases)
    print('U3a 通过：%s' % json.dumps(report, ensure_ascii=False, sort_keys=True))


def main():
    """执行 U3a，支持 --run、--prepare 和 --finalize。"""
    root_dir = os.path.abspath(sys.argv[1]) if len(sys.argv) >= 2 else next_run_dir(DEFAULT_ROOT)
    action = sys.argv[2] if len(sys.argv) >= 3 else '--run'
    if action not in ('--run', '--prepare', '--finalize'):
        raise RuntimeError('动作仅支持 --run/--prepare/--finalize')
    cases = prepare(root_dir)
    if action == '--prepare':
        print('U3a 准备完成：%s' % root_dir)
        return
    if action == '--run':
        for _, directory in cases:
            run_step([ABAQUS_CMD, 'cae', 'noGUI=' + os.path.basename(MODEL_SOURCE)], directory,
                     os.path.join(directory, 'autorun_01_model.log'))
    finalize(root_dir, cases)


if __name__ == '__main__':
    main()
