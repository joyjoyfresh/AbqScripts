# -*- coding: utf-8 -*-
"""
PGA 后处理脚本 v7 —— 适用于 VAB_oblique_TAF_double_v2.py 创建的模型
改进点:
1) 顶面节点直接读取实例中的 TOP_SURFACE 节点集（全部节点），并按 x 坐标升序排列（空间排序）。
2) 动态计算尺度 h (斜坡高度)：
   - 对 slope 模型，计算顶面节点的最大与最小 y 坐标差值。
   - 对 flat 模型，查找同目录下对应的前缀相同且包含 -slope 的 ODB，并读取其差值；
     如无，查找目录下任意 slope 模型的差值；若再无，解析同目录下 *.cae 文件名中的 h 参数；若无，兜底使用 200.0。
3) 为避免丢失整个时程数据并防范 OOM：
   - 先保存顶面节点的所有时程数据：增量式（逐帧）写入 TIMESERIES-{name}.csv 中。
   - 帧循环完成后，再在内存中整合峰值（PGA）数据并按 x/h 排序，写入 PGA-{name}.csv 中。
"""

from odbAccess import openOdb
import os
import csv
import time
import logging
import traceback
import glob
import re

# ==============================================================================
#  配置项
# ==============================================================================
EXPORT_X_OVER_H = False  # 是否在 CSV 中导出 x/h 这一列（若需要兼容旧绘图脚本，可设为 True）


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


def strip_job_prefix(name):
    """若名称以 job- 开头则去掉该前缀，避免输出文件名包含 job-。"""
    if name.lower().startswith('job-'):
        return name[4:]
    return name


# ==============================================================================
#  动态计算斜坡高度 h 的辅助方法
# ==============================================================================
def get_slope_height_from_odb(odb_path, logger=None):
    """从指定 ODB 文件中打开并读取 TOP_SURFACE 节点集的 y 坐标最大差值作为斜坡高度。"""
    try:
        temp_odb = openOdb(path=odb_path, readOnly=True)
        try:
            assembly = temp_odb.rootAssembly
            inst_keys = list(assembly.instances.keys())
            if not inst_keys:
                return None
            instance = assembly.instances[inst_keys[0]]
            top_nset = instance.nodeSets['TOP_SURFACE']
            y_coords = [node.coordinates[1] for node in top_nset.nodes]
            if y_coords:
                dy = max(y_coords) - min(y_coords)
                if dy > 1.0:
                    return dy
        finally:
            temp_odb.close()
    except Exception as e:
        if logger:
            log_step(logger, '读取辅助 ODB %s 失败: %s', os.path.basename(odb_path), str(e))
    return None


def get_slope_height_from_cae(directory, logger=None):
    """
    尝试从目录下的 .cae 文件名中解析斜坡高度 h。
    文件名格式类似于 h200_i30_a15.cae，其中 h 后的数字为斜坡高度。
    """
    try:
        cae_files = sorted(glob.glob(os.path.join(directory, '*.cae')))
        if cae_files:
            cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]
            match = re.search(r'h(?P<h>-?\d+(?:\.\d+)?)_', cae_name)
            if match:
                h_val = float(match.group('h'))
                if h_val > 0.0:
                    return h_val
    except Exception as e:
        if logger:
            log_step(logger, '从 CAE 文件名解析 h 失败: %s', str(e))
    return None


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

        top_nodes, _ = find_top_surface_nodes(instance)
        if not top_nodes:
            log_step(logger, '%s 未找到顶部表面节点，跳过', odb_basename)
            return

        # 将节点标号按 x 坐标升序排列（保证横向空间顺序）
        top_labels = sorted(list(top_nodes.keys()), key=lambda lb: top_nodes[lb][0])
        label_to_idx = {}
        for i, lb in enumerate(top_labels):
            label_to_idx[lb] = i
        n_top = len(top_labels)

        # ==========================================
        #  动态计算斜坡高度 h
        # ==========================================
        y_coords = [top_nodes[lb][1] for lb in top_labels]
        y_max = max(y_coords)
        y_min = min(y_coords)
        dy = y_max - y_min
        if dy > 1.0:
            h = dy
            log_step(logger, '%s 识别为 slope 模型，根据顶面起伏自动计算 h = y_max - y_min = %.6f', odb_basename, h)
        else:
            h = None
            dir_name = os.path.dirname(odb_path)
            # 尝试寻找对应的 slope ODB (例如 job-XXX-flat.odb -> job-XXX-slope.odb)
            if '-flat' in odb_stem.lower():
                slope_stem = odb_stem.lower().replace('-flat', '-slope')
                for f in os.listdir(dir_name):
                    if f.lower() == slope_stem + '.odb':
                        slope_path = os.path.join(dir_name, f)
                        h = get_slope_height_from_odb(slope_path, logger)
                        if h is not None:
                            log_step(logger, '%s 识别为 flat 模型，从对应 slope 文件 %s 中获取 h = %.6f', odb_basename, f, h)
                            break

            # 如果没找到，尝试寻找目录下任意一个 slope ODB
            if h is None:
                for f in os.listdir(dir_name):
                    if f.lower().endswith('.odb') and '-slope' in f.lower():
                        slope_path = os.path.join(dir_name, f)
                        h = get_slope_height_from_odb(slope_path, logger)
                        if h is not None:
                            log_step(logger, '%s 识别为 flat 模型，从目录下任意 slope 文件 %s 中获取 h = %.6f', odb_basename, f, h)
                            break

            # 如果依然没找到，尝试从当前目录下的 .cae 文件名解析
            if h is None:
                h = get_slope_height_from_cae(dir_name, logger)
                if h is not None:
                    log_step(logger, '%s 识别为 flat 模型，从 .cae 文件名中成功解析 h = %.6f', odb_basename, h)

            # 最终兜底使用默认值 200.0
            if h is None:
                h = 200.0
                log_step(logger, '%s 识别为 flat 模型，未找到 slope 或 CAE 文件，使用默认值 h = %.6f', odb_basename, h)

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

        # 准备输出文件名
        csv_stem = strip_job_prefix(odb_stem)
        timeseries_csv_name = 'TIMESERIES-{}.csv'.format(csv_stem)
        pga_csv_name = 'PGA-{}.csv'.format(csv_stem)

        # ==========================================
        #  第一步：增量式逐帧保存时程数据到 TIMESERIES CSV
        # ==========================================
        log_step(logger, '%s 开始增量保存完整时程数据: %s', odb_basename, timeseries_csv_name)
        with open(timeseries_csv_name, 'w') as f_ts:
            writer_ts = csv.writer(f_ts, lineterminator='\n')
            
            # 写入表头
            header = ['Time']
            if EXPORT_X_OVER_H:
                header.append('x/h')
            for label in top_labels:
                header.append('Node_{}_Accel_h'.format(label))
            for label in top_labels:
                header.append('Node_{}_Accel_v'.format(label))
            writer_ts.writerow(header)

            for frame_idx, frame in enumerate(frames):
                if (frame_idx + 1) % max(1, n_frames // 10) == 0:
                    log_step(logger, '%s 处理帧进度: %d / %d', odb_basename, frame_idx + 1, n_frames)

                if 'A' not in frame.fieldOutputs:
                    continue

                acc_field = frame.fieldOutputs['A']
                t_cur = getattr(frame, 'frameValue', 0.0)
                acc_values = acc_field.getSubset(region=top_nset).values

                # 构建当前帧的数据映射 {node_label: (accel_h, accel_v)}
                frame_data = {}
                for val in acc_values:
                    frame_data[val.nodeLabel] = (val.data[0], val.data[1])

                # 构造时程 data 行
                row = ['{:.6e}'.format(t_cur)]
                if EXPORT_X_OVER_H:
                    first_node_x_over_h = top_nodes[top_labels[0]][0] / h
                    row.append('{:.6e}'.format(first_node_x_over_h))

                # 添加所有节点水平加速度
                for label in top_labels:
                    accel_h = frame_data.get(label, (0.0, 0.0))[0]
                    row.append('{:.6e}'.format(accel_h))

                # 添加所有节点竖直加速度
                for label in top_labels:
                    accel_v = frame_data.get(label, (0.0, 0.0))[1]
                    row.append('{:.6e}'.format(accel_v))

                writer_ts.writerow(row)

                # 更新峰值（PGA）数据
                for label in top_labels:
                    idx = label_to_idx[label]
                    accel_h, accel_v = frame_data.get(label, (0.0, 0.0))
                    a1 = abs(accel_h)
                    a2 = abs(accel_v)

                    if a1 > pga_h[idx]:
                        pga_h[idx] = a1
                        peak_h_time[idx] = t_cur

                    if a2 > pga_v[idx]:
                        pga_v[idx] = a2
                        peak_v_time[idx] = t_cur

        log_step(logger, '%s 完整时程数据已先保存至 %s', odb_basename, timeseries_csv_name)

        # ==========================================
        #  第二步：保存 PGA 峰值表
        # ==========================================
        log_step(logger, '%s 开始保存峰值（PGA）数据: %s', odb_basename, pga_csv_name)
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

        # 按 x 坐标进行升序排序（与按 x/h 排序等效）
        results.sort(key=lambda r: r['x'])

        with open(pga_csv_name, 'w') as f_pga:
            writer_pga = csv.writer(f_pga, lineterminator='\n')
            
            pga_header = []
            if EXPORT_X_OVER_H:
                pga_header.append('x/h')
            pga_header.extend(['node_label', 'x', 'y', 'PGA_h', 'PGA_v', 'peak_h_time', 'peak_v_time'])
            writer_pga.writerow(pga_header)

            for r in results:
                row = []
                if EXPORT_X_OVER_H:
                    row.append('%.6f' % r['x/h'])
                row.extend([
                    '%d' % r['node_label'],
                    '%.6f' % r['x'],
                    '%.6f' % r['y'],
                    '%.6f' % r['PGA_h'],
                    '%.6f' % r['PGA_v'],
                    '%.6f' % r['peak_h_time'],
                    '%.6f' % r['peak_v_time'],
                ])
                writer_pga.writerow(row)
        log_step(logger, '%s PGA 峰值表已后保存至 %s，节点数=%d', odb_basename, pga_csv_name, len(results))

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
    logger = log_step('Postprocess_PGA.log')
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
