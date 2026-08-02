# -*- coding: utf-8 -*-
"""从 X001-A ODB 提取预定公共时刻的全场 U/V/A 节点快照。

本脚本由 Abaqus Python 运行，兼容 Python 2.7。输出只做数据提取，不评价
Abaqus 与 SPECFEM2D 的一致性。
"""

from __future__ import absolute_import, print_function

import glob
import io
import json
import os
import sys
import traceback

import numpy as np
from odbAccess import openOdb


OUTPUT_FILE = 'abaqus_wavefield_snapshots.npz'
STATUS_FILE = 'abaqus_wavefield_snapshot_status.json'
COMMON_TIMES = (0.30, 0.45, 0.60)
ABAQUS_TO_COMMON_SHIFT = -0.30


def _write_json(path, payload):
    """写出 UTF-8 JSON 状态。"""
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if isinstance(text, bytes):
        text = text.decode('utf-8')
    with io.open(path, 'w', encoding='utf-8') as handle:
        handle.write(text)
        handle.write(u'\n')


def _find_odb(folder):
    """返回工况目录中唯一的正式 ODB。"""
    candidates = [path for path in glob.glob(os.path.join(folder, '*.odb'))
                  if not os.path.basename(path).lower().startswith('base')]
    if len(candidates) != 1:
        raise RuntimeError('需要且只能有一个正式ODB，实际找到%d个: %s' %
                           (len(candidates), ', '.join(candidates)))
    return candidates[0]


def _select_step(odb):
    """选择包含动力帧的最后一个分析步。"""
    names = list(odb.steps.keys())
    for name in reversed(names):
        step = odb.steps[name]
        if len(step.frames) > 1 and 'A' in step.frames[-1].fieldOutputs.keys():
            return name, step
    raise RuntimeError('ODB中没有包含加速度场的动力分析步')


def _node_coordinates(odb):
    """建立实例名与节点标签到坐标的映射。"""
    coords = {}
    for instance_name in odb.rootAssembly.instances.keys():
        instance = odb.rootAssembly.instances[instance_name]
        for node in instance.nodes:
            coords[(str(instance_name), int(node.label))] = (
                float(node.coordinates[0]), float(node.coordinates[1]))
    return coords


def _field_values(frame, variable):
    """按实例名和节点标签返回二维节点场。"""
    if variable not in frame.fieldOutputs.keys():
        raise RuntimeError('帧 %.6f 缺少场变量 %s' % (float(frame.frameValue), variable))
    result = {}
    for value in frame.fieldOutputs[variable].values:
        if value.instance is None or value.nodeLabel is None:
            continue
        data = tuple(float(item) for item in value.data)
        if len(data) < 2:
            continue
        key = (str(value.instance.name), int(value.nodeLabel))
        result[key] = (data[0], data[1])
    return result


def extract(folder):
    """提取全场快照并返回状态摘要。"""
    odb_path = _find_odb(folder)
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        step_name, step = _select_step(odb)
        coordinates = _node_coordinates(odb)
        selected = []
        for common_time in COMMON_TIMES:
            raw_target = common_time - ABAQUS_TO_COMMON_SHIFT
            frame = min(step.frames,
                        key=lambda item: abs(float(item.frameValue) - raw_target))
            if abs(float(frame.frameValue) - raw_target) > 5.0e-4 + 1.0e-12:
                raise RuntimeError('公共时刻%.3f s未找到对应Abaqus帧，最近原始时刻为%.6f s' %
                                   (common_time, float(frame.frameValue)))
            selected.append((common_time, frame))

        snapshot_fields = []
        common_keys = None
        for common_time, frame in selected:
            fields = dict((name, _field_values(frame, name)) for name in ('U', 'V', 'A'))
            keys = set(fields['A'].keys())
            keys.intersection_update(fields['U'].keys())
            keys.intersection_update(fields['V'].keys())
            keys.intersection_update(coordinates.keys())
            keys = tuple(sorted(keys))
            if not keys:
                raise RuntimeError('帧 %.6f 未取得共同U/V/A节点' % float(frame.frameValue))
            if common_keys is None:
                common_keys = keys
            elif keys != common_keys:
                raise RuntimeError('不同快照的U/V/A节点集合不一致')
            snapshot_fields.append((common_time, float(frame.frameValue), fields))

        x = np.asarray([coordinates[key][0] for key in common_keys], dtype=float)
        y = np.asarray([coordinates[key][1] for key in common_keys], dtype=float)
        labels = np.asarray([key[1] for key in common_keys], dtype=np.int64)
        instances = np.asarray([key[0] for key in common_keys])
        payload = {
            'common_time': np.asarray([item[0] for item in snapshot_fields], dtype=float),
            'abaqus_time': np.asarray([item[1] for item in snapshot_fields], dtype=float),
            'x': x,
            'y': y,
            'node_label': labels,
            'instance': instances,
        }
        for variable in ('U', 'V', 'A'):
            payload[variable + '1'] = np.asarray([
                [fields[variable][key][0] for key in common_keys]
                for _common, _raw, fields in snapshot_fields], dtype=float)
            payload[variable + '2'] = np.asarray([
                [fields[variable][key][1] for key in common_keys]
                for _common, _raw, fields in snapshot_fields], dtype=float)
        geometry_path = os.path.join(folder, 'geometry_validation.json')
        with io.open(geometry_path, 'r', encoding='utf-8') as handle:
            geometry = json.load(handle)
        expected_nodes = int(geometry['node_count'])
        if len(common_keys) != expected_nodes:
            raise RuntimeError('全场节点数%d与几何审计%d不一致' %
                               (len(common_keys), expected_nodes))
        expected_shape = (len(COMMON_TIMES), expected_nodes)
        for field_name in ('U1', 'U2', 'V1', 'V2', 'A1', 'A2'):
            if payload[field_name].shape != expected_shape:
                raise RuntimeError('%s形状%s，预期%s' %
                                   (field_name, payload[field_name].shape, expected_shape))
            if not np.all(np.isfinite(payload[field_name])):
                raise RuntimeError('%s包含非有限值' % field_name)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise RuntimeError('全场节点坐标包含非有限值')
        output_path = os.path.join(folder, OUTPUT_FILE)
        np.savez_compressed(output_path, **payload)
        return {
            'success': True,
            'odb': os.path.basename(odb_path),
            'step': step_name,
            'node_count': int(len(common_keys)),
            'geometry_audit_node_count': expected_nodes,
            'all_values_finite': True,
            'common_times_s': [float(item[0]) for item in snapshot_fields],
            'abaqus_times_s': [float(item[1]) for item in snapshot_fields],
            'time_mapping': 't_common = t_abaqus_output - 0.3 s',
            'output': OUTPUT_FILE,
            'output_size_bytes': int(os.path.getsize(output_path)),
        }
    finally:
        odb.close()


def main():
    """命令行入口。"""
    folder = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.getcwd()
    status_path = os.path.join(folder, STATUS_FILE)
    try:
        status = extract(folder)
        _write_json(status_path, status)
        print('[完成] X001-A全场快照: 节点=%d, 时刻=%s' %
              (status['node_count'], status['common_times_s']))
        return 0
    except Exception as exc:
        status = {'success': False, 'reason': str(exc), 'traceback': traceback.format_exc()}
        _write_json(status_path, status)
        print('[失败] X001-A全场快照提取: %s' % str(exc))
        return 2


if __name__ == '__main__':
    sys.exit(main())
