# -*- coding: utf-8 -*-
"""
PGA 后处理脚本 —— 独立运行于 Abaqus Python 环境
自动扫描当前目录所有 .odb 文件，提取顶部表面节点的 PGA 数据并输出 CSV。

使用方法:
  abaqus cae noGUI=PGA_postprocess.py
  或
  abaqus python PGA_postprocess.py

需要在下方 "用户参数" 区域设置几何参数 h, i_deg, H_lower, total_L。
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
    """
    仅输出到命令行，并尽量兼容 Windows cmd 的中文编码。
    """
    if not hasattr(print_step, '_start_time'):
        print_step._start_time = time.time()

    if args:
        text = message % args
    else:
        text = message

    delta = time.time() - print_step._start_time
    line = '[%.3fs] %s' % (delta, text)

    # 优先用当前 stdout 编码；失败时回退到 gbk/utf-8，减少中文乱码风险。
    encoding = getattr(sys.stdout, 'encoding', None) or 'gbk'
    try:
        sys.stdout.write((line + '\n').encode(encoding, errors='replace').decode(encoding, errors='replace'))
    except Exception:
        try:
            sys.stdout.write((line + '\n').encode('gbk', errors='replace').decode('gbk', errors='replace'))
        except Exception:
            sys.stdout.write((line + '\n').encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    sys.stdout.flush()


# ==============================================================================
#  核心：找到顶部表面节点（按每列 X 取最大 Y 的方式，不依赖几何公式）
# ==============================================================================
def find_top_surface_nodes(instance, mesh_tol):
    """
    鲁棒地找到二维模型顶部表面节点。
    策略：将所有节点按 X 坐标分桶（桶宽 = mesh_tol/2），
    每个桶内 Y 最大的节点即为该列的顶部表面节点。

    参数:
        instance  : ODB 实例对象
        mesh_tol  : 分桶宽度（建议设为 网格尺寸 / 2 或更小值）
    返回:
        top_nodes : dict, {node_label: (x, y)}
    """
    # 1) 收集所有节点坐标
    all_nodes = {}  # label -> (x, y)
    for node in instance.nodes:
        x = node.coordinates[0]
        y = node.coordinates[1]
        all_nodes[node.label] = (x, y)

    if not all_nodes:
        return {}

    # 2) 按 X 分桶
    half_tol = mesh_tol / 2.0
    buckets = {}  # bucket_index -> [(label, x, y), ...]
    for label, (x, y) in all_nodes.items():
        bkt = int(round(x / half_tol))
        if bkt not in buckets:
            buckets[bkt] = []
        buckets[bkt].append((label, x, y))

    # 3) 每桶取 Y 最大的节点
    top_nodes = {}
    for bkt, nodes_in_bkt in buckets.items():
        max_y = max(n[2] for n in nodes_in_bkt)
        for label, x, y in nodes_in_bkt:
            if abs(y - max_y) < 1e-6:
                top_nodes[label] = (x, y)

    return top_nodes


# ==============================================================================
#  处理单个 ODB 文件
# ==============================================================================
def process_one_odb(odb_path, h, mesh_tol):
    """
    处理单个 ODB 文件：
    1. 找出顶部表面节点
    2. 对每个表面节点，遍历所有帧读取加速度 'A'
    3. 计算水平 PGA 和竖向 PGA（单位 g）
    4. 按 x/h 排序后保存为 CSV

    参数:
        odb_path (str):   ODB 文件路径
        h        (float): 斜坡高度（用于无量纲化）
        mesh_tol (float): 分桶宽度
    """
    g_acc = 9.81
    odb_basename = os.path.basename(odb_path)
    odb_stem = os.path.splitext(odb_basename)[0]

    print_step('>>> 开始处理: %s', odb_basename)
    odb = openOdb(path=odb_path, readOnly=True)

    try:
        # ---- 获取实例 ----
        assembly = odb.rootAssembly
        inst_keys = list(assembly.instances.keys())
        if not inst_keys:
            print_step('%s 无装配实例，跳过', odb_basename)
            return
        instance = assembly.instances[inst_keys[0]]

        # ---- 找顶部表面节点 ----
        top_nodes = find_top_surface_nodes(instance, mesh_tol)
        if not top_nodes:
            print_step('%s 未找到顶部表面节点，跳过', odb_basename)
            return
        top_labels = set(top_nodes.keys())
        print_step('%s 顶部表面节点数: %d', odb_basename, len(top_labels))

        # ---- 获取分析步 ----
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
        print_step('%s 分析步 "%s"  帧数: %d', odb_basename, step_keys[-1], n_frames)

        # ---- 初始化 PGA 存储 ----
        pga_h = {}
        pga_v = {}
        for label in top_labels:
            pga_h[label] = 0.0
            pga_v[label] = 0.0

        # ---- 遍历帧提取加速度 ----
        for fi, frame in enumerate(frames):
            if 'A' not in frame.fieldOutputs:
                continue
            acc_field = frame.fieldOutputs['A']
            for val in acc_field.values:
                if val.nodeLabel in top_labels:
                    a1 = abs(val.data[0])   # 水平分量
                    a2 = abs(val.data[1])   # 竖向分量
                    if a1 > pga_h[val.nodeLabel]:
                        pga_h[val.nodeLabel] = a1
                    if a2 > pga_v[val.nodeLabel]:
                        pga_v[val.nodeLabel] = a2
            # 每 500 帧打印进度
            if (fi + 1) % 500 == 0 or fi == n_frames - 1:
                print_step('%s 已处理帧 %d / %d', odb_basename, fi + 1, n_frames)

        # ---- 整理结果 ----
        results = []
        for label in top_labels:
            x_coord = top_nodes[label][0]
            x_over_h = x_coord / h
            results.append((x_over_h,
                            pga_h[label] / g_acc,
                            pga_v[label] / g_acc))

        results.sort(key=lambda r: r[0])

        # ---- 保存 CSV ----
        csv_name = 'PGA_results_{}.csv'.format(odb_stem)
        with open(csv_name, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(['x/h', 'PGA_h', 'PGA_v'])
            for row in results:
                writer.writerow(['{:.6f}'.format(row[0]),
                                 '{:.6f}'.format(row[1]),
                                 '{:.6f}'.format(row[2])])

        print_step('%s PGA 结果已保存: %s (共 %d 个节点)',
                   odb_basename, csv_name, len(results))

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
        print_step('===== PGA 后处理脚本开始 =====')

        # ===================== 用户参数（按需修改） =====================
        h = 100.0                   # *斜坡高度 (m)
        mesh_size = 10.0            # *网格尺寸 (m)，用于分桶容差
        # ================================================================

        mesh_tol = mesh_size / 2.0  # 分桶宽度

        # 扫描当前目录的所有 .odb 文件
        cwd = os.getcwd()
        odb_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.odb')])

        if not odb_files:
            print_step('当前目录 %s 下未找到任何 .odb 文件', cwd)
        else:
            print_step('共找到 %d 个 ODB 文件: %s', len(odb_files), ', '.join(odb_files))
            for odb_file in odb_files:
                odb_path = os.path.join(cwd, odb_file)
                process_one_odb(odb_path, h, mesh_tol)

        print_step('===== 全部完成，总耗时=%.2fs =====', time.time() - total_start)

    except Exception as exc:
        print_step('脚本失败: %s', str(exc))
        print_step('异常堆栈:\n%s', traceback.format_exc())
