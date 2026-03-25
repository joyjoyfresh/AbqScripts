# -*- coding: utf-8 -*-
"""
PGA 后处理脚本 v3 —— 独立运行于 Abaqus Python 环境
改进点:
1) 顶面节点按 X 分桶时，桶宽直接使用 bucket_width，不再二次缩小。
2) 每个桶仅保留 1 个顶部代表节点，避免同桶多点带来的局部交替锯齿。
3) 输出诊断列：node_label、x、y、PGA 峰值对应帧号与时刻。
4) 可选输出空间平滑列（仅用于展示，不替代原始 PGA）。

使用方法:
  abaqus cae noGUI=Postprocess_PGA_v3.py
  或
  abaqus python Postprocess_PGA_v3.py
"""

from odbAccess import openOdb
import os
import csv
import time
import sys
import traceback


# ==============================================================================
#  控制台输出函数（兼容 Windows cmd 中文）
# ==============================================================================
def print_step(message, *args):
    if not hasattr(print_step, '_start_time'):
        print_step._start_time = time.time()

    text = message % args if args else message
    delta = time.time() - print_step._start_time
    line = '[%.3fs] %s' % (delta, text)

    encoding = getattr(sys.stdout, 'encoding', None) or 'gbk'
    try:
        if sys.version_info[0] < 3:
            # Abaqus 常见 Python2 环境: stdout.write 期望 byte string
            if isinstance(line, unicode):
                out = line.encode(encoding, 'replace') + '\n'
            else:
                out = str(line) + '\n'
            sys.stdout.write(out)
        else:
            sys.stdout.write(line + '\n')
    except Exception:
        fallback = None
        for enc in (encoding, 'gbk', 'utf-8', 'ascii'):
            try:
                if sys.version_info[0] < 3:
                    if isinstance(line, unicode):
                        fallback = line.encode(enc, 'replace') + '\n'
                    else:
                        fallback = str(line) + '\n'
                    sys.stdout.write(fallback)
                else:
                    fallback = (line + '\n').encode(enc, errors='replace')
                    if hasattr(sys.stdout, 'buffer'):
                        sys.stdout.buffer.write(fallback)
                    else:
                        sys.stdout.write(fallback.decode(enc, errors='replace'))
                break
            except Exception:
                continue
    sys.stdout.flush()


# ==============================================================================
#  工具函数
# ==============================================================================
def moving_average_centered(values, window_size):
    """居中滑动平均，边界处自动缩小窗口。"""
    if window_size <= 1 or len(values) <= 2:
        return list(values)

    if window_size % 2 == 0:
        window_size += 1

    half = window_size // 2
    out = []
    n = len(values)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        seg = values[start:end]
        out.append(sum(seg) / float(len(seg)))
    return out


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
def process_one_odb(odb_path, h, bucket_width, enable_spatial_smoothing=False, smooth_window=9):
    g_acc = 9.81
    odb_basename = os.path.basename(odb_path)
    odb_stem = os.path.splitext(odb_basename)[0]

    print_step('>>> 开始处理: %s', odb_basename)
    odb = openOdb(path=odb_path, readOnly=True)

    try:
        assembly = odb.rootAssembly
        inst_keys = list(assembly.instances.keys())
        if not inst_keys:
            print_step('%s 无装配实例，跳过', odb_basename)
            return
        instance = assembly.instances[inst_keys[0]]

        top_nodes, diag = find_top_surface_nodes(instance, bucket_width)
        if not top_nodes:
            print_step('%s 未找到顶部表面节点，跳过', odb_basename)
            return

        top_labels = set(top_nodes.keys())
        print_step('%s 顶部节点筛选: 总节点=%d, 分桶=%d, 多候选桶=%d, 选中=%d',
                   odb_basename,
                   diag.get('total_nodes', 0),
                   diag.get('bucket_count', 0),
                   diag.get('multi_top_buckets', 0),
                   diag.get('selected_top_nodes', 0))

        step_keys = list(odb.steps.keys())
        if not step_keys:
            print_step('%s ODB 中无分析步，跳过', odb_basename)
            return

        step = odb.steps[step_keys[-1]]
        frames = step.frames
        n_frames = len(frames)
        if n_frames == 0:
            print_step('%s 分析步 "%s" 无帧数据，跳过', odb_basename, step_keys[-1])
            return
        print_step('%s 分析步 "%s" 帧数: %d', odb_basename, step_keys[-1], n_frames)

        pga_h = {}
        pga_v = {}
        peak_h_frame = {}
        peak_v_frame = {}
        peak_h_time = {}
        peak_v_time = {}

        for label in top_labels:
            pga_h[label] = 0.0
            pga_v[label] = 0.0
            peak_h_frame[label] = -1
            peak_v_frame[label] = -1
            peak_h_time[label] = 0.0
            peak_v_time[label] = 0.0

        for fi, frame in enumerate(frames):
            if 'A' not in frame.fieldOutputs:
                continue

            acc_field = frame.fieldOutputs['A']
            t_cur = getattr(frame, 'frameValue', 0.0)

            for val in acc_field.values:
                label = val.nodeLabel
                if label not in top_labels:
                    continue

                a1 = abs(val.data[0])
                a2 = abs(val.data[1])

                if a1 > pga_h[label]:
                    pga_h[label] = a1
                    peak_h_frame[label] = fi
                    peak_h_time[label] = t_cur

                if a2 > pga_v[label]:
                    pga_v[label] = a2
                    peak_v_frame[label] = fi
                    peak_v_time[label] = t_cur

            if (fi + 1) % 500 == 0 or fi == n_frames - 1:
                print_step('%s 已处理帧 %d / %d', odb_basename, fi + 1, n_frames)

        results = []
        for label in top_labels:
            x_coord, y_coord = top_nodes[label]
            x_over_h = x_coord / h
            results.append({
                'x/h': x_over_h,
                'node_label': label,
                'x': x_coord,
                'y': y_coord,
                'PGA_h': pga_h[label] / g_acc,
                'PGA_v': pga_v[label] / g_acc,
                'peak_h_frame': peak_h_frame[label],
                'peak_h_time': peak_h_time[label],
                'peak_v_frame': peak_v_frame[label],
                'peak_v_time': peak_v_time[label],
            })

        results.sort(key=lambda r: r['x/h'])

        if enable_spatial_smoothing and len(results) > 2:
            pga_h_series = [r['PGA_h'] for r in results]
            pga_v_series = [r['PGA_v'] for r in results]
            pga_h_smooth = moving_average_centered(pga_h_series, smooth_window)
            pga_v_smooth = moving_average_centered(pga_v_series, smooth_window)
            for i in range(len(results)):
                results[i]['PGA_h_smooth'] = pga_h_smooth[i]
                results[i]['PGA_v_smooth'] = pga_v_smooth[i]
        else:
            for i in range(len(results)):
                results[i]['PGA_h_smooth'] = results[i]['PGA_h']
                results[i]['PGA_v_smooth'] = results[i]['PGA_v']

        csv_name = 'PGA_{}.csv'.format(odb_stem)
        with open(csv_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow([
                'x/h', 'node_label', 'x', 'y',
                'PGA_h', 'PGA_v',
                'PGA_h_smooth', 'PGA_v_smooth',
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
                    '%.6f' % r['PGA_h_smooth'],
                    '%.6f' % r['PGA_v_smooth'],
                    '%d' % r['peak_h_frame'],
                    '%.6f' % r['peak_h_time'],
                    '%d' % r['peak_v_frame'],
                    '%.6f' % r['peak_v_time'],
                ])

        print_step('%s PGA 结果已保存: %s (共 %d 个节点)', odb_basename, csv_name, len(results))

    except Exception as e:
        print_step('%s 处理失败: %s', odb_basename, str(e))
        print_step('异常堆栈:\n%s', traceback.format_exc())
    finally:
        odb.close()


# ==============================================================================
#  主入口
# ==============================================================================
if __name__ == '__main__':
    total_start = time.time()

    try:
        print_step('===== PGA 后处理脚本 v3 开始 =====')

        # ===================== 用户参数（按需修改） =====================
        h = 100.0
        mesh_size = 8

        # v3: 分桶宽度直接取网格尺度（不再额外 /2）
        bucket_width = mesh_size

        # 是否输出空间平滑列（建议仅用于绘图展示）
        enable_spatial_smoothing = False
        smooth_window = 9
        # ===============================================================

        cwd = os.getcwd()
        odb_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.odb')])

        if not odb_files:
            print_step('当前目录 %s 下未找到任何 .odb 文件', cwd)
        else:
            print_step('共找到 %d 个 ODB 文件: %s', len(odb_files), ', '.join(odb_files))
            for odb_file in odb_files:
                odb_path = os.path.join(cwd, odb_file)
                process_one_odb(
                    odb_path=odb_path,
                    h=h,
                    bucket_width=bucket_width,
                    enable_spatial_smoothing=enable_spatial_smoothing,
                    smooth_window=smooth_window
                )

        print_step('===== 全部完成，总耗时=%.2fs =====', time.time() - total_start)

    except Exception as exc:
        print_step('脚本失败: %s', str(exc))
        print_step('异常堆栈:\n%s', traceback.format_exc())
