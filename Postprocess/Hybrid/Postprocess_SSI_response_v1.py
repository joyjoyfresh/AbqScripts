# -*- coding: utf-8 -*-
"""坡顶建筑 SSI 结构响应后处理 v1（配 slope_frame_ssi_full_v1.py 的 tssi_cfg['enable']=True 输出）。

遍历当前目录所有 job-*.odb，对【含框架(Frame-1)的 SSI 模型】逐条提取坡顶建筑地震响应：
  - CREST_REF：坡顶土面参考节点（= 框架基础运动，含地形+地层放大后的坡顶运动）
  - FLOOR_k  ：框架各层 U1/A1
指标：
  - T_ssi：顶层相对坡顶基础的 FFT 主周期；与 T_fixed(step1=0.5s) 比 = SSI 周期延长
  - 基底剪力：Σ mᵢ·aᵢ（楼层惯性力峰值）
  - 最大层间位移角：相邻层 U1 差 / 层高
  - 建筑放大：顶层 PGA / 坡顶基础 PGA（建筑对坡顶运动的放大）
纯自由场(freefield, 无框架)的 odb 自动跳过。

约定同 General 后处理：Abaqus 自带 Python 2.7 + numpy；csv.writer 输出。
运行：abaqus python Postprocess_SSI_response_v1.py   （在含 job-*.odb 与 tssi_meta.json 的工况目录内）
输出：SSI-response-summary.csv（逐波一行）+ 屏幕汇总。
"""

import os
import sys
import glob
import csv
import json
import numpy as np

try:
    from odbAccess import openOdb  # Abaqus ODB 接口（仅 Abaqus 环境）
except Exception:
    openOdb = None  # 纯 Python 占位（便于单测数值函数）

try:
    reload(sys)                       # noqa: F821  Py2.7 控制台中文
    sys.setdefaultencoding('utf-8')   # noqa
except Exception:
    pass


def strip_job_prefix(name):  # job-<记录>.odb -> <记录>
    base = os.path.basename(name)
    if base.lower().endswith('.odb'):
        base = base[:-4]
    if base.lower().startswith('job-'):
        base = base[4:]
    return base


def _load_tssi_meta():  # 读框架参数(建模脚本写出)
    with open('tssi_meta.json', 'r') as fh:
        return json.load(fh)


def _resolve_inst(odb, want):  # 实例名大小写兜底
    keys = list(odb.rootAssembly.instances.keys())
    want = str(want)
    return want if want in keys else (want.upper() if want.upper() in keys else None)


def _node_hist(step, inst, label, var):  # 取单节点历史输出数组
    hr = step.historyRegions['Node %s.%d' % (inst, label)]
    return np.array(hr.historyOutputs[var].data, dtype=float)


def _crest_ref(odb):  # 坡顶参考节点 (instanceName, label)（assembly 集 CREST_REF）
    ns = odb.rootAssembly.nodeSets['CREST_REF']
    n = ns.nodes[0] if not hasattr(ns.nodes[0], '__len__') else ns.nodes[0][0]
    return n.instanceName, n.label


def _dom_period(sig, dt):  # 去均值 FFT 主频/主周期
    x = sig - np.mean(sig)
    sp = np.abs(np.fft.rfft(x)); fr = np.fft.rfftfreq(len(x), dt); sp[0] = 0.0
    f = float(fr[int(np.argmax(sp))])
    return f, (1.0 / f if f > 0 else None)


def process_one_odb(odb_path, meta):
    """处理单条 SSI odb，返回结果 dict；非 SSI(无框架)返回 None。"""
    ns = int(meta['n_story']); sh = float(meta['story_height']); fm = float(meta['floor_mass'])
    inst_frame_want = meta.get('inst_frame', 'Frame-1')
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        inst_f = _resolve_inst(odb, inst_frame_want)  # 无框架实例 -> 非 SSI, 跳过
        if inst_f is None or 'CREST_REF' not in odb.rootAssembly.nodeSets.keys():
            return None
        eq = odb.steps[list(odb.steps.keys())[0]]     # 首个分析步(对步名鲁棒)
        inst_c, lab_c = _crest_ref(odb)               # 坡顶基础运动
        u_c = _node_hist(eq, inst_c, lab_c, 'U1')
        t = u_c[:, 0]; u_base = u_c[:, 1]; dt = float(t[1] - t[0])
        a_c = _node_hist(eq, inst_c, lab_c, 'A1')[:, 1]  # 坡顶基础加速度
        floor_u = [u_base]; floor_a = []
        for k in range(1, ns + 1):
            lab = odb.rootAssembly.instances[inst_f].nodeSets['FLOOR_%d' % k].nodes[0].label
            floor_u.append(_node_hist(eq, inst_f, lab, 'U1')[:, 1])
            floor_a.append(_node_hist(eq, inst_f, lab, 'A1')[:, 1])
        f1, T_ssi = _dom_period(floor_u[ns] - floor_u[0], dt)  # 顶层相对基础
        inertia = np.zeros_like(floor_a[0])
        for k in range(ns):
            inertia = inertia + fm * floor_a[k]
        drift = [float(np.max(np.abs(floor_u[k] - floor_u[k - 1])) / sh) for k in range(1, ns + 1)]
        crest_pga = float(np.max(np.abs(a_c)))          # 坡顶基础 PGA
        roof_pga = float(np.max(np.abs(floor_a[ns - 1])))  # 顶层 PGA
        T_fixed = float(meta.get('T_fixed_step1', 0.5))
        return {
            'T_ssi': T_ssi, 'T_fixed': T_fixed,
            'period_ratio': (T_ssi / T_fixed) if T_ssi else None,
            'base_shear': float(np.max(np.abs(inertia))),
            'max_drift': max(drift), 'drift_profile': drift,
            'crest_pga': crest_pga, 'roof_pga': roof_pga,
            'building_amp': (roof_pga / crest_pga) if crest_pga else None,
        }
    finally:
        odb.close()


def main():
    if openOdb is None:
        print(u'需在 Abaqus 环境运行(abaqus python)'); return
    if not os.path.isfile('tssi_meta.json'):
        print(u'缺 tssi_meta.json（须先用 slope_frame_ssi_full 且 tssi_cfg.enable=True 跑过）'); return
    meta = _load_tssi_meta()
    odbs = sorted(glob.glob('job-*.odb'))
    if not odbs:
        print(u'当前目录无 job-*.odb'); return

    rows = []
    for op in odbs:
        rec = strip_job_prefix(op)
        try:
            r = process_one_odb(op, meta)
        except Exception as e:
            print(u'[%s] 处理失败: %s' % (rec, str(e))); continue
        if r is None:
            print(u'[%s] 非 SSI(无框架)，跳过' % rec); continue
        rows.append((rec, r))
        print('=' * 60)
        print(u'■ %s' % rec)
        print(u'  周期延长: T_ssi=%.3fs / T_fixed=%.3fs = %.2f%s'
              % (r['T_ssi'] or 0, r['T_fixed'], r['period_ratio'] or 0,
                 u' ✓' if r['period_ratio'] and r['period_ratio'] > 1.02 else u''))
        print(u'  基底剪力=%.3eN  最大层间位移角=%.4f' % (r['base_shear'], r['max_drift']))
        print(u'  坡顶基础 PGA=%.3f  顶层 PGA=%.3f  建筑放大(顶/坡顶)=%.2f×'
              % (r['crest_pga'], r['roof_pga'], r['building_amp'] or 0))

    if not rows:
        print(u'\n无 SSI 模型结果（是否 tssi_cfg.enable=True 跑的？）'); return

    with open('SSI-response-summary.csv', 'w') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['record', 'T_ssi', 'T_fixed', 'period_ratio', 'base_shear',
                    'max_drift', 'crest_pga', 'roof_pga', 'building_amp'])
        for rec, r in rows:
            w.writerow([rec, r['T_ssi'], r['T_fixed'], r['period_ratio'], r['base_shear'],
                        r['max_drift'], r['crest_pga'], r['roof_pga'], r['building_amp']])
    print(u'\n' + '=' * 60)
    print(u'SSI 后处理完成（%d 条 SSI 波）。已写 SSI-response-summary.csv' % len(rows))


if __name__ == '__main__':
    main()
