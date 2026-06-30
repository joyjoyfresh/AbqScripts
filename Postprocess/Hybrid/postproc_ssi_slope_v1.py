# -*- coding: utf-8 -*-
"""
Step-2b 坡顶 SSI 后处理 + 验证
==============================
读 job-freefield.odb、job-ssi.odb、job-fixed.odb，验坡面 SSI 集成：
  1. 坡顶自由场：坡肩地表放大(地形+地层放大 TAF)、主频
  2. SSI 周期延长：T_ssi(顶层相对坡顶基础 FFT) > T_fixed(step1=0.5s)
  3. SSI 结构响应：基底剪力/层间位移角/顶层加速度
  4. (step2b-2)坡顶刚性 vs SSI 去耦对比：刚性基础输入坡顶自由场运动，SSI/刚性比量化 SSI 效应

运行：abaqus python postproc_ssi_slope_v1.py
"""

import os
import sys
import json
import numpy as np
from odbAccess import openOdb

try:
    reload(sys)                       # noqa: F821
    sys.setdefaultencoding('utf-8')   # noqa
except Exception:
    pass


def _load_meta():
    with open('ssi_slope_meta.json', 'r') as fh:
        return json.load(fh)


def _resolve_inst(odb, want):
    keys = list(odb.rootAssembly.instances.keys())
    want = str(want)
    return want if want in keys else (want.upper() if want.upper() in keys else str(keys[0]))


def _node_hist(step, inst, label, var):
    hr = step.historyRegions['Node %s.%d' % (inst, label)]
    return np.array(hr.historyOutputs[var].data, dtype=float)


def _crest_ref(odb):
    """返回坡顶参考节点的 (instanceName, label)（来自 assembly 集 CREST_REF）。"""
    ns = odb.rootAssembly.nodeSets['CREST_REF']
    n = ns.nodes[0] if not hasattr(ns.nodes[0], '__len__') else ns.nodes[0][0]
    return n.instanceName, n.label


def _dom_period(sig, dt):
    x = sig - np.mean(sig)
    sp = np.abs(np.fft.rfft(x)); fr = np.fft.rfftfreq(len(x), dt); sp[0] = 0.0
    f = float(fr[int(np.argmax(sp))])
    return f, (1.0 / f if f > 0 else None)


def main():
    meta = _load_meta()
    pga = float(meta['pga']); T_fixed = float(meta['T_fixed_step1'])
    results = {'angle': meta['angle'], 'T_fixed': T_fixed}
    print('=' * 60)

    # ---------- 1. 坡顶自由场 ----------
    if os.path.isfile('job-freefield.odb'):
        odb = openOdb('job-freefield.odb', readOnly=True)
        inst, lab = _crest_ref(odb)
        eq = odb.steps[list(odb.steps.keys())[0]]
        d = _node_hist(eq, inst, lab, 'A1')
        time = d[:, 0]; ca = d[:, 1]; dt = float(time[1] - time[0])
        amp = float(np.max(np.abs(ca)) / pga)
        f_c, T_c = _dom_period(ca, dt)
        print(u'[1] 坡顶自由场(坡肩地表 x=%.0f)：' % meta['left_flat'])
        print(u'    峰值加速度 = %.3f m/s² (输入 PGA=%.3f) -> 坡顶放大 %.2f×' % (np.max(np.abs(ca)), pga, amp))
        print(u'    响应主频 = %.2f Hz' % f_c)
        results['freefield'] = {'crest_amp': amp, 'crest_f': f_c}
        odb.close()
    else:
        print(u'[1] 缺 job-freefield.odb')

    # ---------- 2&3. 坡顶 SSI ----------
    if os.path.isfile('job-ssi.odb'):
        odb = openOdb('job-ssi.odb', readOnly=True)
        inst_f = _resolve_inst(odb, meta['inst_frame'])
        inst_c, lab_c = _crest_ref(odb)               # 坡顶土面节点 = 框架基础运动
        eq = odb.steps[list(odb.steps.keys())[0]]
        ns = int(meta['n_story']); sh = float(meta['story_height']); fm = float(meta['floor_mass'])

        u_found = _node_hist(eq, inst_c, lab_c, 'U1')
        time = u_found[:, 0]; u0 = u_found[:, 1]; dt = float(time[1] - time[0])
        floor_u = [u0]; floor_a = []
        for k in range(1, ns + 1):
            lab = odb.rootAssembly.instances[inst_f].nodeSets['FLOOR_%d' % k].nodes[0].label
            floor_u.append(_node_hist(eq, inst_f, lab, 'U1')[:, 1])
            floor_a.append(_node_hist(eq, inst_f, lab, 'A1')[:, 1])

        f1, T_ssi = _dom_period(floor_u[ns] - floor_u[0], dt)
        print(u'\n[2] 坡顶 SSI 周期延长：')
        print(u'    T_ssi(顶层相对坡顶基础 FFT) = %.3f s ; T_fixed(step1) = %.3f s' % (T_ssi or 0, T_fixed))
        if T_ssi:
            r = T_ssi / T_fixed
            print(u'    T_ssi/T_fixed = %.2f  %s' % (r, u'✓ 周期延长' if r > 1.02 else u'≈ 无明显延长'))
            results['T_ssi'] = T_ssi; results['period_ratio'] = r

        inertia = np.zeros_like(floor_a[0])
        for k in range(ns):
            inertia = inertia + fm * floor_a[k]
        drift = [float(np.max(np.abs(floor_u[k] - floor_u[k - 1])) / sh) for k in range(1, ns + 1)]
        print(u'\n[3] 坡顶 SSI 结构响应：')
        print(u'    基底剪力(Σmᵢaᵢ) = %.3e N' % float(np.max(np.abs(inertia))))
        print(u'    最大层间位移角 = %.4f' % max(drift))
        print(u'    顶层峰值加速度 = %.3f m/s² (相对输入放大 %.2f×)'
              % (np.max(np.abs(floor_a[ns - 1])), np.max(np.abs(floor_a[ns - 1])) / pga))
        results['ssi'] = {'base_shear': float(np.max(np.abs(inertia))), 'max_drift': max(drift),
                          'roof_peak_acc': float(np.max(np.abs(floor_a[ns - 1])))}
        odb.close()
    else:
        print(u'\n[2&3] 缺 job-ssi.odb')

    # ---------- 4. 坡顶刚性基础(输入坡顶自由场) vs SSI 去耦对比 (step2b-2) ----------
    if os.path.isfile('job-fixed.odb'):
        odb = openOdb('job-fixed.odb', readOnly=True)
        inst_f = _resolve_inst(odb, meta['inst_frame'])
        eq = odb.steps[list(odb.steps.keys())[0]]
        ns = int(meta['n_story']); sh = float(meta['story_height']); fm = float(meta['floor_mass'])
        lab_b = odb.rootAssembly.instances[inst_f].nodeSets['BASE'].nodes[0].label  # 柱脚(被坡顶自由场加速度驱动)
        base_u = _node_hist(eq, inst_f, lab_b, 'U1')
        timef = base_u[:, 0]; u_base = base_u[:, 1]; dtf = float(timef[1] - timef[0])
        fu = [u_base]; fa = []
        for k in range(1, ns + 1):
            lab = odb.rootAssembly.instances[inst_f].nodeSets['FLOOR_%d' % k].nodes[0].label
            fu.append(_node_hist(eq, inst_f, lab, 'U1')[:, 1])
            fa.append(_node_hist(eq, inst_f, lab, 'A1')[:, 1])
        inert = np.zeros_like(fa[0])
        for k in range(ns):
            inert = inert + fm * fa[k]
        bs_f = float(np.max(np.abs(inert)))
        dr_f = [float(np.max(np.abs(fu[k] - fu[k - 1])) / sh) for k in range(1, ns + 1)]
        roof_f = float(np.max(np.abs(fa[ns - 1])))
        T_fixed_nat = T_fixed   # 刚性自振用 step1 值(0.5s)；fixed 的 FFT 主频是受迫响应、近共振不可靠
        results['fixed'] = {'T_natural': T_fixed_nat, 'base_shear': bs_f,
                            'max_drift': max(dr_f), 'roof_peak_acc': roof_f}
        print(u'\n[4] 坡顶刚性基础(输入坡顶自由场) vs SSI 去耦对比：')
        print(u'    刚性: T自振=%.3fs(step1) 基底剪力=%.3eN 最大漂移=%.4f 顶层加速=%.3f(放大%.1f×)'
              % (T_fixed_nat, bs_f, max(dr_f), roof_f, roof_f / pga))
        if 'ssi' in results:
            s = results['ssi']; Ts = results.get('T_ssi') or 0.0
            print(u'    SSI : T自振=%.3fs           基底剪力=%.3eN 最大漂移=%.4f 顶层加速=%.3f(放大%.1f×)'
                  % (Ts, s['base_shear'], s['max_drift'], s['roof_peak_acc'], s['roof_peak_acc'] / pga))
            print(u'    SSI/刚性比: 周期=%.2f 基底剪力=%.2f 漂移=%.2f 顶层加速=%.2f'
                  % (Ts / T_fixed_nat, s['base_shear'] / bs_f,
                     s['max_drift'] / max(dr_f), s['roof_peak_acc'] / roof_f))
            cf = results.get('freefield', {}).get('crest_f')  # 坡顶自由场主频(调谐解读)
            if cf:
                print(u'    解读: 刚性 f=%.2fHz vs 坡顶自由场 f=%.2fHz → SSI 周期延长改变结构与坡顶运动的调谐关系'
                      % (1.0 / T_fixed_nat, cf))
            results['ssi_over_fixed'] = {'T': Ts / T_fixed_nat, 'base_shear': s['base_shear'] / bs_f,
                                         'drift': s['max_drift'] / max(dr_f), 'roof_acc': s['roof_peak_acc'] / roof_f}
        odb.close()
    else:
        print(u'\n[4] 缺 job-fixed.odb，跳过去耦对比(step2b-2)')

    with open('ssi_slope_results.json', 'w') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print('\n' + '=' * 60)
    print(u'后处理完成。已写 ssi_slope_results.json')


if __name__ == '__main__':
    main()
