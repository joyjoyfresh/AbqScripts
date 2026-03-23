# -*- coding: utf-8 -*-
"""
PGA 后处理脚本 v3 —— 独立运行于 Abaqus Python 环境
改进点:
1) 顶面节点按 X 分桶时，桶宽直接使用 bucket_width，不再二次缩小。
2) 每个桶仅保留 1 个顶部代表节点，避免同桶多点带来的局部交替锯齿。
3) 输出诊断列：node_label、x、y、PGA 峰值对应帧号与时刻。
4) 可选输出空间平滑列（仅用于展示，不替代原始 PGA）。
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
#  核心：找到顶部表面节点（每桶仅保留 1 个代表节点）
# ==============================================================================
def find_top_surface_nodes(instance, bucket_width, y_tol=1e-6):
    """
    将节点按 X 分桶（桶宽 = bucket_width），每桶取 Y 最大层中的 1 个代表节点。

    代表节点选择规则:
    1) 先筛选 Y==max_y 的候选点
    2) 选 x 最靠近桶中心的候选
    3) 若仍并列，选 label 最小者

    返回:
        top_nodes: dict, {node_label: (x, y)}
        diagnostics: dict, 诊断信息
    """
    all_nodes = {}  # label -> (x, y)
    for node in instance.nodes:
        x = node.coordinates[0]
        y = node.coordinates[1]
        all_nodes[node.label] = (x, y)

    if not all_nodes:
        return {}, {'total_nodes': 0, 'bucket_count': 0, 'multi_top_buckets': 0}

    if bucket_width <= 0.0:
        raise ValueError('bucket_width 必须大于 0。')

    buckets = {}  # bucket_index -> [(label, x, y), ...]
    for label, (x, y) in all_nodes.items():
        bkt = int(round(x / bucket_width))
        if bkt not in buckets:
            buckets[bkt] = []
        buckets[bkt].append((label, x, y))

    top_nodes = {}
    multi_top_buckets = 0

    for bkt, nodes_in_bkt in buckets.items():
        max_y = max(n[2] for n in nodes_in_bkt)
        candidates = [n for n in nodes_in_bkt if abs(n[2] - max_y) <= y_tol]

        if len(candidates) > 1:
            multi_top_buckets += 1

        center_x = bkt * bucket_width
        # (距离桶中心, label) 最小优先
        candidates.sort(key=lambda n: (abs(n[1] - center_x), n[0]))
        chosen_label, chosen_x, chosen_y = candidates[0]
        top_nodes[chosen_label] = (chosen_x, chosen_y)

    diagnostics = {
        'total_nodes': len(all_nodes),
        'bucket_count': len(buckets),
        'multi_top_buckets': multi_top_buckets,
        'selected_top_nodes': len(top_nodes),
    }
    return top_nodes, diagnostics


# ==============================================================================
#  处理单个 ODB 文件
# ==============================================================================
def process_one_odb(odb_path, h, bucket_width, logger=None):
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

        top_nodes, _ = find_top_surface_nodes(instance, bucket_width)
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
        peak_h_frame = [-1] * n_top
        peak_v_frame = [-1] * n_top
        peak_h_time = [0.0] * n_top
        peak_v_time = [0.0] * n_top

        # 尝试获取 TOP_SURFACE 节点集，用于 getSubset 加速
        top_nset = None
        try:
            top_nset = instance.nodeSets['TOP_SURFACE']
            log_step(logger, '%s 检测到 TOP_SURFACE 节点集，使用 getSubset 加速模式', odb_basename)
        except KeyError:
            log_step(logger, '%s 未检测到 TOP_SURFACE 节点集，使用全场遍历模式（较慢）', odb_basename)

        for fi, frame in enumerate(frames):
            if 'A' not in frame.fieldOutputs:
                continue

            acc_field = frame.fieldOutputs['A']
            t_cur = getattr(frame, 'frameValue', 0.0)

            # 如果有节点集，用 getSubset 在 C++ 层过滤；否则遍历全场
            if top_nset is not None:
                acc_values = acc_field.getSubset(region=top_nset).values
            else:
                acc_values = acc_field.values

            for val in acc_values:
                idx = label_to_idx.get(val.nodeLabel)
                if idx is None:
                    continue

                a1 = abs(val.data[0])
                a2 = abs(val.data[1])

                if a1 > pga_h[idx]:
                    pga_h[idx] = a1
                    peak_h_frame[idx] = fi
                    peak_h_time[idx] = t_cur

                if a2 > pga_v[idx]:
                    pga_v[idx] = a2
                    peak_v_frame[idx] = fi
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
                'peak_h_frame': peak_h_frame[i],
                'peak_h_time': peak_h_time[i],
                'peak_v_frame': peak_v_frame[i],
                'peak_v_time': peak_v_time[i],
            })

        results.sort(key=lambda r: r['x/h'])

        csv_name = 'PGA_{}.csv'.format(odb_stem)
        with open(csv_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([
                'x/h', 'node_label', 'x', 'y',
                'PGA_h', 'PGA_v',
                'peak_h_frame', 'peak_h_time',
                'peak_v_frame', 'peak_v_time'
            ])
            for r in results:
                writer.writerow([
                    '%.6f' % r['x/h'],
                    '%d' % r['node_label'],
                    '%.6f' % r['x'],
                    '%.6f' % r['y'],
                    '%.6f' % r['PGA_h'],
                    '%.6f' % r['PGA_v'],
                    '%d' % r['peak_h_frame'],
                    '%.6f' % r['peak_h_time'],
                    '%d' % r['peak_v_frame'],
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
    logger = log_step('Postprocess_PGA_v4.log')
    total_start = time.time()
    try:
        log_step(logger, '脚本开始执行')

        # ===================== 用户参数（按需修改） =====================
        h = 100.0
        mesh_size = 10.0

        # v3: 分桶宽度直接取网格尺度（不再额外 /2）
        bucket_width = mesh_size

        # ===============================================================

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
                h=h,
                bucket_width=bucket_width,
                logger=logger
            )

        log_step(logger, '脚本执行完成，总耗时=%.2fs', time.time() - total_start)
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise
