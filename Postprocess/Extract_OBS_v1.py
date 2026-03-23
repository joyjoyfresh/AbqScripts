# -*- coding: utf-8 -*-
"""
Extract_OBS_v1.py
=================
从脚本所在目录下所有 .odb 文件中读取 U/M/D 观察点的场输出，生成 CSV 文件。

思路：
  1. 根据几何参数（h, i, H_lower, total_L 以及观察点数量/间距）计算 U/M/D 观察点的理论坐标。
  2. 打开 odb，在最终网格中按坐标（容差匹配 + 回退最近节点）找到对应节点编号。
  3. 遍历所有分析步的所有帧，提取指定场变量（默认 'A'，即加速度）在该节点的值。
  4. 将所有帧数据写入 CSV 文件，文件命名为 <odb名>_OBS.csv。

使用方法（在 Abaqus Python 环境或 abaqus python 命令行中执行）:
    abaqus python Extract_OBS_v1.py

注意：
  - 需与 .odb 文件位于同一目录，或修改 ODB_DIR 变量。
  - 若 odb 中无 U/M/D 集合节点，脚本将根据几何参数自动计算坐标并定位最近节点。
  - 支持多分量输出（如 A1/A2 对应 x/y 加速度）。
"""

import os
import sys
import math
import csv

# ============================================================
# ======== 请在此处修改参数 =====================================
# ============================================================

# 几何参数（与建模脚本保持一致）
h = 100.0               # 斜坡高度 (m)
i_angle = 45.0          # 斜坡倾角 (°)
H_lower = 2.0 * h      # 下垫面高度 (m)
total_L = 8.0 * h      # 模型总水平长度 (m)

# 观察点数量与间距（与建模脚本保持一致）
slope_obs_count = 3         # M 点数量（斜坡上，从顶到底均匀分布）
upper_obs_count = 0         # U 点数量（上平台，从坡顶向左）
upper_obs_spacing = 50.0    # U 点间距 (m)
lower_obs_count = 0         # D 点数量（下平台，从坡脚向右）
lower_obs_spacing = 50.0    # D 点间距 (m)

# 坐标匹配容差（若容差内无节点，自动选取最近节点）
COORD_TOL = 1.0             # 单位：m（可适当放宽，如网格尺寸的 0.5 倍）

# 要提取的场变量名称（Abaqus 场变量键，如 'A'=加速度, 'U'=位移, 'V'=速度）
FIELD_KEYS = ['A']

# ODB 所在目录（默认为本脚本所在目录）
ODB_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# ======== 以下为功能实现，一般不需要修改 ========================
# ============================================================

try:
    from odbAccess import openOdb
except ImportError:
    raise ImportError('请在 Abaqus Python 环境下执行此脚本（abaqus python Extract_OBS_v1.py）')


def compute_obs_coords(h, i_angle, H_lower, total_L,
                       slope_obs_count, upper_obs_count, upper_obs_spacing,
                       lower_obs_count, lower_obs_spacing):
    """
    根据几何参数计算 U/M/D 观察点的理论坐标。
    返回有序字典列表: [(name, x, y), ...]，按 U -> M -> D 顺序排列。
    """
    w_slope = h / math.tan(math.radians(i_angle))
    left_flat = 3.0 * h
    right_flat = total_L - left_flat - w_slope
    if right_flat <= 0:
        raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                         (total_L, left_flat, w_slope))
    H_upper = H_lower + h

    slope_upper = (left_flat, H_upper)
    slope_lower = (left_flat + w_slope, H_lower)

    obs_list = []  # [(name, x, y)]
    coord_used = set()

    def add(name, x, y):
        key = (round(x, 8), round(y, 8))
        if key not in coord_used:
            obs_list.append((name, x, y))
            coord_used.add(key)
        else:
            print('[WARN] 观察点 %s (%.3f, %.3f) 与已有点重合，已跳过' % (name, x, y))

    # U 点：从坡顶向左（不含坡顶本身）
    if upper_obs_count > 0 and upper_obs_spacing > 0:
        max_u = int(math.floor(slope_upper[0] / upper_obs_spacing + 1e-12))
        eff_u = min(upper_obs_count, max_u)
        for k in range(eff_u):
            x = slope_upper[0] - (k + 1) * upper_obs_spacing
            add('U%d' % (k + 1), x, H_upper)

    # M 点：斜坡上从上到下均匀分布（含端点）
    if slope_obs_count == 1:
        add('M1', slope_upper[0], slope_upper[1])
    else:
        for k in range(slope_obs_count):
            t = float(k) / float(slope_obs_count - 1)
            x = slope_upper[0] + t * (slope_lower[0] - slope_upper[0])
            y = slope_upper[1] + t * (slope_lower[1] - slope_upper[1])
            add('M%d' % (k + 1), x, y)

    # D 点：从坡脚向右（不含坡脚本身）
    if lower_obs_count > 0 and lower_obs_spacing > 0:
        right_len = total_L - slope_lower[0]
        max_d = int(math.floor(right_len / lower_obs_spacing + 1e-12))
        eff_d = min(lower_obs_count, max_d)
        for k in range(eff_d):
            x = slope_lower[0] + (k + 1) * lower_obs_spacing
            add('D%d' % (k + 1), x, H_lower)

    return obs_list


def find_node_label(odb_part_instance, tx, ty, tol=1.0):
    """
    在 odb 实例的节点集中按坐标查找节点标签。
    先在容差内精确匹配，若无匹配则回退到最近节点。
    返回 (node_label, dist)。
    """
    best_label = None
    best_dist = float('inf')
    matched_label = None
    matched_dist = float('inf')

    for node in odb_part_instance.nodes:
        coords = node.coordinates
        x = coords[0]
        y = coords[1]
        dist = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_label = node.label
        if dist < tol and dist < matched_dist:
            matched_dist = dist
            matched_label = node.label

    if matched_label is not None:
        return matched_label, matched_dist
    else:
        return best_label, best_dist


def extract_field_for_node(odb, node_label, instance_name, field_key):
    """
    遍历 odb 所有分析步的所有帧，提取指定节点的场变量值。
    返回列表: [(step_name, frame_value, frame_time, comp1, comp2, ...), ...]
    """
    rows = []
    for step_name, step in odb.steps.items():
        for frame in step.frames:
            frame_time = frame.frameValue
            if field_key not in frame.fieldOutputs:
                continue
            field = frame.fieldOutputs[field_key]
            # 按节点标签子集读取
            try:
                subset = field.getSubset(region=odb.rootAssembly.instances[instance_name].nodes)
                # 找到对应节点
                for val in subset.values:
                    if val.nodeLabel == node_label:
                        data = val.data
                        # data 可能是标量或向量（tuple）
                        if hasattr(data, '__iter__'):
                            comps = list(data)
                        else:
                            comps = [data]
                        rows.append([step_name, frame_time] + comps)
                        break
            except Exception as e:
                # 如果 getSubset 不支持，逐个遍历
                for val in field.values:
                    if val.nodeLabel == node_label:
                        data = val.data
                        if hasattr(data, '__iter__'):
                            comps = list(data)
                        else:
                            comps = [data]
                        rows.append([step_name, frame_time] + comps)
                        break
    return rows


def get_instance_name(odb):
    """获取 odb 中第一个装配体实例名称。"""
    inst_keys = list(odb.rootAssembly.instances.keys())
    if not inst_keys:
        raise RuntimeError('ODB 中未找到装配体实例')
    return inst_keys[0]


def determine_comp_headers(field_key, n_comps):
    """根据场变量键和分量数生成列标题。"""
    comp_map = {
        'A': ['A1', 'A2', 'A3'],
        'U': ['U1', 'U2', 'U3'],
        'V': ['V1', 'V2', 'V3'],
        'S': ['S11', 'S22', 'S33', 'S12', 'S13', 'S23'],
    }
    candidates = comp_map.get(field_key, ['%s_%d' % (field_key, j+1) for j in range(9)])
    return candidates[:n_comps] if n_comps <= len(candidates) else ['%s_%d' % (field_key, j+1) for j in range(n_comps)]


def process_odb(odb_path, obs_list, field_keys, coord_tol):
    """
    处理单个 odb 文件，对 obs_list 中每个观察点逐一提取场输出，合并后写入 CSV。
    CSV 文件与 odb 同名，后缀改为 _OBS.csv，保存在同一目录。
    """
    print('\n========================================')
    print('处理 ODB: %s' % odb_path)

    odb = openOdb(path=odb_path, readOnly=True)
    inst_name = get_instance_name(odb)
    instance = odb.rootAssembly.instances[inst_name]

    # 步骤1: 定位各观察点节点
    obs_nodes = []  # [(name, node_label, actual_x, actual_y, dist)]
    for (obs_name, tx, ty) in obs_list:
        label, dist = find_node_label(instance, tx, ty, tol=coord_tol)
        if label is None:
            print('[WARN] 观察点 %s 未找到任何节点，已跳过' % obs_name)
            continue
        # 取实际坐标
        node = None
        for nd in instance.nodes:
            if nd.label == label:
                node = nd
                break
        actual_x = node.coordinates[0] if node else tx
        actual_y = node.coordinates[1] if node else ty
        if dist > coord_tol:
            print('[INFO] 观察点 %s (%.3f, %.3f) -> 最近节点 %d (%.3f, %.3f), 距离=%.4f m (超出容差，已使用最近节点)'
                  % (obs_name, tx, ty, label, actual_x, actual_y, dist))
        else:
            print('[INFO] 观察点 %s (%.3f, %.3f) -> 节点 %d (%.3f, %.3f), 距离=%.4f m'
                  % (obs_name, tx, ty, label, actual_x, actual_y, dist))
        obs_nodes.append((obs_name, label, actual_x, actual_y, dist))

    if not obs_nodes:
        print('[WARN] 无有效观察点节点，跳过此 ODB')
        odb.close()
        return

    # 步骤2: 对每个场变量和观察点提取数据
    # 输出格式：每个场变量单独一个 CSV，或合并到同一 CSV
    # 这里选择：每个场变量生成一个 CSV，按行记录 [obs_name, step, time, comp1, comp2, ...]

    odb_base = os.path.splitext(os.path.basename(odb_path))[0]
    out_dir = os.path.dirname(odb_path)

    for field_key in field_keys:
        # 收集所有观察点的数据
        all_data = []  # [(obs_name, step_name, frame_time, comp1, ...), ...]
        n_comps = 0

        for (obs_name, node_label, ax, ay, dist) in obs_nodes:
            rows = extract_field_for_node(odb, node_label, inst_name, field_key)
            for row in rows:
                # row = [step_name, frame_time, comp1, ...]
                n_comps = max(n_comps, len(row) - 2)
                all_data.append([obs_name] + row)

        if not all_data:
            print('[WARN] 场变量 %s 在所有观察点均无数据，已跳过' % field_key)
            continue

        comp_headers = determine_comp_headers(field_key, n_comps)
        header = ['obs_name', 'step', 'time'] + comp_headers

        csv_name = '%s_%s_OBS.csv' % (odb_base, field_key)
        csv_path = os.path.join(out_dir, csv_name)

        with open(csv_path, 'wb') as f:  # Abaqus Python 2 兼容
            writer = csv.writer(f)
            writer.writerow(header)
            for row in all_data:
                # 补齐分量列（有些帧可能分量数不足）
                n_row_comps = len(row) - 3
                padded = row + [''] * (n_comps - n_row_comps)
                writer.writerow(padded)

        print('[OK] 已写入 CSV: %s (%d 行数据)' % (csv_path, len(all_data)))

    odb.close()
    print('ODB 已关闭: %s' % odb_path)


def main():
    # 1. 计算观察点理论坐标
    obs_list = compute_obs_coords(
        h=h, i_angle=i_angle, H_lower=H_lower, total_L=total_L,
        slope_obs_count=slope_obs_count,
        upper_obs_count=upper_obs_count, upper_obs_spacing=upper_obs_spacing,
        lower_obs_count=lower_obs_count, lower_obs_spacing=lower_obs_spacing
    )

    if not obs_list:
        print('[ERROR] 未生成任何观察点，请检查参数设置')
        sys.exit(1)

    print('观察点列表（共 %d 个）:' % len(obs_list))
    for name, x, y in obs_list:
        print('  %-6s  x=%.3f  y=%.3f' % (name, x, y))

    # 2. 扫描目录下所有 .odb 文件
    odb_files = sorted([
        os.path.join(ODB_DIR, f)
        for f in os.listdir(ODB_DIR)
        if f.lower().endswith('.odb')
    ])

    if not odb_files:
        print('[ERROR] 目录 %s 下未找到任何 .odb 文件' % ODB_DIR)
        sys.exit(1)

    print('\n找到 %d 个 ODB 文件:' % len(odb_files))
    for fp in odb_files:
        print('  %s' % fp)

    # 3. 逐一处理
    for odb_path in odb_files:
        try:
            process_odb(odb_path, obs_list, FIELD_KEYS, COORD_TOL)
        except Exception as e:
            print('[ERROR] 处理 %s 时发生异常: %s' % (odb_path, str(e)))
            import traceback
            traceback.print_exc()

    print('\n所有 ODB 处理完毕。')


if __name__ == '__main__':
    main()
