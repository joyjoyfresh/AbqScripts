# -*- coding: utf-8 -*-
"""
PGA 后处理脚本 —— 独立运行于 Abaqus Python 环境
改进点:
1) 顶面节点直接读取实例中的 TOP_SURFACE 节点集（全部节点）。
2) h 自动由模型总长度 L 计算：h = L / 8，无需手动输入。
3) 输出诊断列：node_label、x、y、PGA 峰值对应帧号与时刻。
"""

from odbAccess import openOdb
import os
import csv
import time
import logging
import traceback


def log_step(logger=None, message=None, *args):
    """
    日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。
    初始化:    logger = log_step('mylog.log')  # 传入日志文件名
               logger = log_step()            # 使用默认文件名 'logfile.log'
    记录日志:  log_step(logger, '消息 %s', val)
    """
    if not hasattr(log_step, '_logger'):
        if logger is not None and isinstance(logger, str):
            log_filename = logger
            logger = None
        else:
            log_filename = 'logfile.log'

        _logger = logging.getLogger('abqpy')
        _logger.setLevel(logging.INFO)
        _logger.propagate = False

        _logger.handlers = []
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        log_step._logger = _logger
        log_step._start_time = time.time()
        log_step._log_filename = log_filename

        return _logger

    if message is not None:
        now = time.time()
        delta_total = now - log_step._start_time
        log_step._logger.info('[%.3fs] ' + message, delta_total, *args)

    return log_step._logger


# ==============================================================================
#  核心：从 TOP_SURFACE 节点集读取顶部表面节点（全部保留）
# ==============================================================================
def find_top_surface_nodes(instance):
    """
    直接读取实例中的 TOP_SURFACE 节点集，返回全部节点。

    返回:
        top_nodes: dict, {node_label: (x, y)}
        diagnostics: dict, 诊断信息
    """
    try:
        top_nset = instance.nodeSets['TOP_SURFACE']
    except KeyError:
        return {}, {'total_nodes': 0, 'selected_top_nodes': 0, 'source': 'TOP_SURFACE missing'}

    top_nodes = {}  # label -> (x, y)
    for node in top_nset.nodes:
        x = node.coordinates[0]
        y = node.coordinates[1]
        top_nodes[node.label] = (x, y)

    diagnostics = {
        'total_nodes': len(top_nodes),
        'selected_top_nodes': len(top_nodes),
        'source': 'TOP_SURFACE',
    }
    return top_nodes, diagnostics


# ==============================================================================
#  处理单个 ODB 文件
# ==============================================================================
def process_one_odb(odb_path, logger=None):
    logger = logger or log_step()
    odb_basename = os.path.basename(odb_path)
    odb_stem = os.path.splitext(odb_basename)[0]

    log_step(logger, '开始处理 ODB: %s', odb_basename)

    odb = openOdb(path=odb_path, readOnly=True)

    try:
        assembly = odb.rootAssembly
        inst_keys = list(assembly.instances.keys())
        if not inst_keys:
            log_step(logger, '%s 无装配实例，跳过', odb_basename)
            return
        instance = assembly.instances[inst_keys[0]]

        x_coords = [node.coordinates[0] for node in instance.nodes]
        if not x_coords:
            log_step(logger, '%s 实例无节点，跳过', odb_basename)
            return
        total_L = max(x_coords) - min(x_coords)
        if total_L <= 0.0:
            log_step(logger, '%s 模型总长度无效 (L=%.6f)，跳过', odb_basename, total_L)
            return
        h = total_L / 8.0
        log_step(logger, '%s 自动计算尺度: L=%.6f, h=L/8=%.6f', odb_basename, total_L, h)

        top_nodes, _ = find_top_surface_nodes(instance)
        if not top_nodes:
            log_step(logger, '%s 未找到顶部表面节点，跳过', odb_basename)
            return

        top_labels = list(top_nodes.keys())
        label_to_idx = {}
        for i, lb in enumerate(top_labels):
            label_to_idx[lb] = i
        n_top = len(top_labels)

        step_keys = list(odb.steps.keys())
        if not step_keys:
            log_step(logger, '%s 无分析步，跳过', odb_basename)
            return

        step = odb.steps[step_keys[-1]]
        frames = step.frames
        n_frames = len(frames)
        if n_frames == 0:
            log_step(logger, '%s 分析步 %s 无帧数据，跳过', odb_basename, step_keys[-1])
            return

        log_step(logger, '%s 使用分析步 %s，帧数=%d，顶部节点数=%d',
                 odb_basename, step_keys[-1], n_frames, n_top)

        pga_h = [0.0] * n_top
        pga_v = [0.0] * n_top
        peak_h_time = [0.0] * n_top
        peak_v_time = [0.0] * n_top

        top_nset = instance.nodeSets['TOP_SURFACE']
        log_step(logger, '%s 使用 TOP_SURFACE 节点集的全部节点进行后处理', odb_basename)

        for frame in frames:
            if 'A' not in frame.fieldOutputs:
                continue

            acc_field = frame.fieldOutputs['A']
            t_cur = getattr(frame, 'frameValue', 0.0)

            acc_values = acc_field.getSubset(region=top_nset).values

            for val in acc_values:
                idx = label_to_idx.get(val.nodeLabel)
                if idx is None:
                    continue

                a1 = abs(val.data[0])
                a2 = abs(val.data[1])

                if a1 > pga_h[idx]:
                    pga_h[idx] = a1
                    peak_h_time[idx] = t_cur

                if a2 > pga_v[idx]:
                    pga_v[idx] = a2
                    peak_v_time[idx] = t_cur

        results = []
        for i, label in enumerate(top_labels):
            x_coord, y_coord = top_nodes[label]
            x_over_h = x_coord / h
            results.append({
                'x/h': x_over_h,
                'node_label': label,
                'x': x_coord,
                'y': y_coord,
                'PGA_h': pga_h[i],
                'PGA_v': pga_v[i],
                'peak_h_time': peak_h_time[i],
                'peak_v_time': peak_v_time[i],
            })

        results.sort(key=lambda r: r['x/h'])

        csv_name = 'PGA_{}.csv'.format(odb_stem)
        with open(csv_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([
                'x/h', 'node_label', 'x', 'y',
                'PGA_h', 'PGA_v',
                'peak_h_time',
                'peak_v_time'
            ])
            for r in results:
                writer.writerow([
                    '%.6f' % r['x/h'],
                    '%d' % r['node_label'],
                    '%.6f' % r['x'],
                    '%.6f' % r['y'],
                    '%.6f' % r['PGA_h'],
                    '%.6f' % r['PGA_v'],
                    '%.6f' % r['peak_h_time'],
                    '%.6f' % r['peak_v_time'],
                ])
        log_step(logger, '%s 处理完成，输出文件=%s，节点数=%d', odb_basename, csv_name, len(results))
    except Exception as exc:
        log_step(logger, '%s 处理失败: %s', odb_basename, str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise
    finally:
        odb.close()


# ==============================================================================
#  主入口
# ==============================================================================
if __name__ == '__main__':
    logger = log_step('Postprocess_PGA_v5.log')
    total_start = time.time()
    try:
        log_step(logger, '脚本开始执行')

        cwd = os.getcwd()
        odb_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.odb')])
        if not odb_files:
            log_step(logger, '当前目录 %s 下未找到 .odb 文件', cwd)
        else:
            log_step(logger, '共找到 %d 个 ODB 文件', len(odb_files))

        for odb_file in odb_files:
            odb_path = os.path.join(cwd, odb_file)
            process_one_odb(
                odb_path=odb_path,
                logger=logger
            )

        log_step(logger, '脚本执行完成，总耗时=%.2fs', time.time() - total_start)
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise
