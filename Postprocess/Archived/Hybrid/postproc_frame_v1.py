# -*- coding: utf-8 -*-
"""
Step-1 后处理 + 验证：从固定基础框架 ODB 提取抗震指标并跑手算锚点校核
======================================================================
配合 frame_fixedbase_v1.py。提取三个核心指标并验"对不对"：
  1. 模态 T1     —— 与 frame_meta.json 经验锚点对比（验刚度+质量）
  2. 基底剪力     —— Σ 柱脚 RF1；与 Σ mᵢ·aᵢ(绝对楼层加速度) 对拍（验质量+反力提取）
  3. 层间位移角   —— 相邻层 U1 差/层高（基底刚体运动抵消，验提取逻辑）
  4. 楼层峰值加速度

运行：
    abaqus cae noGUI=postproc_frame_v1.py
    （或 abaqus python postproc_frame_v1.py）
工作目录需有 job-Model-1.odb 与 frame_meta.json。
输出：postproc_results.json + base_shear_check.csv + drift_profile.csv
"""

import os
import sys
import json
import glob
import numpy as np
from odbAccess import openOdb

# Abaqus Py2.7 控制台默认 ascii，print 中文会 UnicodeEncodeError；设 utf-8 兜底
try:
    reload(sys)                       # noqa: F821 (Py2 内建)
    sys.setdefaultencoding('utf-8')   # noqa
except Exception:
    pass


def _find_odb():
    cands = sorted(glob.glob('job-*.odb')) or sorted(glob.glob('*.odb'))
    if not cands:
        raise IOError(u'当前目录无 .odb 文件')
    return cands[0]


def _load_meta():
    with open('frame_meta.json', 'r') as fh:
        return json.load(fh)


def _node_hist(step, inst, label, var):
    """返回某节点某变量的历史 [time, value] (Nx2)。"""
    key = 'Node %s.%d' % (inst, label)
    hr = step.historyRegions[key]
    return np.array(hr.historyOutputs[var].data, dtype=float)


def _set_labels(odb, inst, setname):
    """返回实例节点集中各节点 label 列表。"""
    nset = odb.rootAssembly.instances[inst].nodeSets[setname]
    return [n.label for n in nset.nodes]


def main():
    odb_name = _find_odb()
    meta = _load_meta()
    ns = int(meta['frame']['n_story'])
    sh = float(meta['frame']['story_height'])
    floor_mass = float(meta['frame']['floor_mass'])

    print('=' * 60)
    print(u'后处理 ODB: %s' % odb_name)
    odb = openOdb(odb_name, readOnly=True)

    # 解析实例名：ODB 会把实例名大写（'Frame-1'->'FRAME-1'）；json 读出是 unicode，odb API 要 str
    inst_keys = list(odb.rootAssembly.instances.keys())
    inst = str(meta['inst_name'])
    if inst not in inst_keys:
        inst = inst.upper() if inst.upper() in inst_keys else str(inst_keys[0])

    results = {'odb': odb_name, 'frame': meta['frame']}

    # ---------- 1. 模态 T1 ----------
    modal = {}
    if 'Step-Modal' in odb.steps:
        mstep = odb.steps['Step-Modal']
        freqs = []
        for fr in mstep.frames:
            mode = getattr(fr, 'mode', None)
            if mode and mode > 0:
                freqs.append(float(fr.frequency))   # Hz
        if freqs:
            T = [(1.0 / f if f > 1e-6 else None) for f in freqs]
            modal = {'frequencies_Hz': freqs, 'periods_s': T}
            T1 = next((p for p in T if p), None)
            modal['T1'] = T1
            print(u'\n[1] 模态步（Frequency）：')
            if T1:
                for i in range(len(freqs)):
                    print(u'    模态%d: f=%.4f Hz, T=%.3f s' % (i + 1, freqs[i], T[i]))
                print(u'    --> T1 = %.3f s, T1/0.1N=%.2f'
                      % (T1, T1 / meta['anchor_T1_empirical_0p1N']))
            else:
                print(u'    [跳过] 微扰特征值≈0（DURING_ANALYSIS梁+SIM-Lanczos 已知怪癖，不影响动力步）；T1 见下方 [1b]')
    results['modal'] = modal

    # ---------- 时程数据（Step-EQ）----------
    eq = odb.steps['Step-EQ']

    # 基底：RF1 求和(基底剪力)、U1(基底位移)
    base_labels = _set_labels(odb, inst, 'BASE')
    base_shear = None
    time = None
    for lab in base_labels:
        d = _node_hist(eq, inst, lab, 'RF1')
        time = d[:, 0] if time is None else time
        base_shear = d[:, 1].copy() if base_shear is None else base_shear + d[:, 1]
    # 基底剪力 = Σ柱脚水平反力 RF1。整体牛顿: ΣRF1 = Σmᵢaᵢ(绝对)，故与惯性力同号
    base_u1 = _node_hist(eq, inst, base_labels[0], 'U1')[:, 1]

    # 各层参考节点：U1(绝对位移)、A1(绝对加速度)
    floor_u = [base_u1]      # 索引0=基底
    floor_a = []
    for k in range(1, ns + 1):
        lab = _set_labels(odb, inst, 'FLOOR_%d' % k)[0]
        floor_u.append(_node_hist(eq, inst, lab, 'U1')[:, 1])
        floor_a.append(_node_hist(eq, inst, lab, 'A1')[:, 1])

    # ---------- 1b. T1 系统识别（顶层相对位移 FFT 主频）----------
    # Frequency 微扰步在本配置下特征值≈0(SIM+DURING_ANALYSIS 怪癖)，故主用动力响应反推 T1
    dt = float(time[1] - time[0])
    rel_roof = floor_u[ns] - floor_u[0]           # 顶层相对基底位移
    rel = rel_roof - np.mean(rel_roof)
    sp = np.abs(np.fft.rfft(rel))
    fr = np.fft.rfftfreq(len(rel), dt)
    sp[0] = 0.0
    f1_dyn = float(fr[int(np.argmax(sp))])
    T1_dyn = (1.0 / f1_dyn) if f1_dyn > 0 else None
    results['T1_from_dynamic'] = T1_dyn
    print(u'\n[1b] 动力响应反推 T1（顶层相对位移 FFT 主频）：')
    if T1_dyn:
        print(u'    f1=%.3f Hz -> T1=%.3f s' % (f1_dyn, T1_dyn))
        print(u'    手算锚点: 0.1N=%.2fs, ATC=%.2fs；T1/0.1N=%.2f（0.5~2.0 合理）'
              % (meta['anchor_T1_empirical_0p1N'], meta['anchor_T1_atc_0p075H075'],
                 T1_dyn / meta['anchor_T1_empirical_0p1N']))

    # ---------- 2. 基底剪力 vs Σmᵢaᵢ 对拍 ----------
    inertia = np.zeros_like(base_shear)
    for k in range(ns):
        inertia = inertia + floor_mass * floor_a[k]   # 绝对加速度×楼层质量
    peak_bs = float(np.max(np.abs(base_shear)))
    peak_in = float(np.max(np.abs(inertia)))
    ratio = peak_bs / peak_in if peak_in > 0 else None
    # 相关系数（时程一致性）
    corr = float(np.corrcoef(base_shear, inertia)[0, 1]) if peak_in > 0 else None
    print(u'\n[2] 基底剪力校核：')
    print(u'    峰值 基底剪力(ΣRF1) = %.3e N' % peak_bs)
    print(u'    峰值 Σmᵢaᵢ(惯性力)  = %.3e N' % peak_in)
    print(u'    峰值比 = %.3f（应≈1.0；偏差主要来自质量比例阻尼 αMv）' % ratio)
    print(u'    时程相关系数 = %.4f（应>0.98）' % corr)
    results['base_shear'] = {'peak_RF': peak_bs, 'peak_inertia': peak_in,
                             'peak_ratio': ratio, 'correlation': corr}

    # ---------- 3. 层间位移角 ----------
    drift_peak = []
    for k in range(1, ns + 1):
        rel = floor_u[k] - floor_u[k - 1]          # 该层相对下层位移
        dr = float(np.max(np.abs(rel)) / sh)
        drift_peak.append(dr)
    print(u'\n[3] 层间位移角（峰值）：')
    for k, dr in enumerate(drift_peak, 1):
        print(u'    第%d层: %.4f (1/%.0f)' % (k, dr, (1.0 / dr if dr > 0 else 0)))
    print(u'    最大层间位移角 = %.4f' % max(drift_peak))
    results['interstory_drift_peak'] = drift_peak
    results['max_drift'] = max(drift_peak)

    # ---------- 4. 楼层峰值加速度 ----------
    floor_pa = [float(np.max(np.abs(floor_a[k]))) for k in range(ns)]
    pga = meta.get('pga')
    print(u'\n[4] 楼层峰值绝对加速度（输入 PGA=%.3f m/s²）：' % (pga or 0))
    for k, pa in enumerate(floor_pa, 1):
        amp = pa / pga if pga else None
        print(u'    第%d层: %.3f m/s² (放大 %.2f×)' % (k, pa, amp or 0))
    results['floor_peak_acc'] = floor_pa
    results['roof_amplification'] = (floor_pa[-1] / pga) if pga else None

    # ---------- 导出 ----------
    with open('postproc_results.json', 'w') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    # 基底剪力对拍时程
    np.savetxt('base_shear_check.csv',
               np.column_stack([time, base_shear, inertia]),
               delimiter=',', header='time,base_shear_RF,inertia_sum_ma', comments='')
    # 层间位移角剖面
    np.savetxt('drift_profile.csv',
               np.column_stack([np.arange(1, ns + 1), drift_peak]),
               delimiter=',', header='story,drift_ratio', comments='')

    odb.close()
    print('\n' + '=' * 60)
    print(u'后处理完成。已写: postproc_results.json / base_shear_check.csv / drift_profile.csv')
    print(u'验证判据: (1)T1量级合理 (2)基底剪力峰值比≈1±0.1且相关>0.98 (3)层间位移角剖面合理')


if __name__ == '__main__':
    main()
