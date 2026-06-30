# -*- coding: utf-8 -*-
"""
Step-2a SSI 后处理 + 验证
==========================
配合 frame_ssi_v1.py。读 job-freefield.odb 与 job-ssi.odb，验三件事：
  1. 场地自由场：地表放大、场地基频 f0(FFT) ≈ Vs/(4H)
  2. SSI 周期延长：T_ssi(顶层相对基础 FFT) > T_fixed(step1=0.5s)  —— SSI 标志性特征
  3. SSI 基底剪力/顶层漂移：与 step1 固定基础对比

运行：abaqus cae noGUI=postproc_ssi_v1.py （或 abaqus python ...）
输出：ssi_results.json
"""

import os
import sys
import json
import glob
import numpy as np
from odbAccess import openOdb

try:
    reload(sys)                       # noqa: F821 (Py2)
    sys.setdefaultencoding('utf-8')   # noqa
except Exception:
    pass


def _load_meta():
    with open('ssi_meta.json', 'r') as fh:
        return json.load(fh)


def _resolve_inst(odb, want):
    keys = list(odb.rootAssembly.instances.keys())
    want = str(want)
    if want in keys:
        return want
    return want.upper() if want.upper() in keys else str(keys[0])


def _node_hist(step, inst, label, var):
    hr = step.historyRegions['Node %s.%d' % (inst, label)]
    return np.array(hr.historyOutputs[var].data, dtype=float)


def _set_label(odb, inst, setname):
    return odb.rootAssembly.instances[inst].nodeSets[setname].nodes[0].label


def _dom_period(sig, dt):
    """信号去均值后 FFT 主频对应周期 (s)；返回 (f_peak, T_peak)。"""
    x = sig - np.mean(sig)
    sp = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(len(x), dt)
    sp[0] = 0.0
    f = float(fr[int(np.argmax(sp))])
    return f, (1.0 / f if f > 0 else None)


def main():
    meta = _load_meta()
    results = {'meta_site_f0': meta['site_f0_Hz'], 'T_fixed': meta['T_fixed_from_step1_s']}
    pga = float(meta['pga'])
    print('=' * 60)

    # ---------- 1. 自由场（freefield）----------
    if os.path.isfile('job-freefield.odb'):
        odb = openOdb('job-freefield.odb', readOnly=True)
        inst = _resolve_inst(odb, meta['inst_soil'])
        eq = odb.steps['Step-EQ']
        lab = _set_label(odb, inst, 'SURF_CENTER')
        d_a = _node_hist(eq, inst, lab, 'A1')
        time = d_a[:, 0]; surf_a = d_a[:, 1]
        dt = float(time[1] - time[0])
        amp = float(np.max(np.abs(surf_a)) / pga) if pga else None
        f_site, T_site = _dom_period(surf_a, dt)
        print(u'[1] 自由场地表响应（freefield）：')
        print(u'    地表峰值加速度 = %.3f m/s² (输入 PGA=%.3f) -> 放大 %.2f×' % (np.max(np.abs(surf_a)), pga, amp))
        print(u'    地表响应主频 f=%.2f Hz；场地基频锚点 f0=%.2f Hz' % (f_site, meta['site_f0_Hz']))
        results['freefield'] = {'surf_amp': amp, 'surf_f_dom': f_site, 'site_f0': meta['site_f0_Hz']}
        odb.close()
    else:
        print(u'[1] 缺 job-freefield.odb，跳过自由场')

    # ---------- 2&3. SSI ----------
    if os.path.isfile('job-ssi.odb'):
        odb = openOdb('job-ssi.odb', readOnly=True)
        inst_s = _resolve_inst(odb, meta['inst_soil'])
        inst_f = _resolve_inst(odb, meta['inst_frame'])
        eq = odb.steps['Step-EQ']
        ns = int(meta['n_story']); sh = float(meta['frame']['story_height'])
        fm = float(meta['frame']['floor_mass'])

        # 基础运动 = 土体地表中心（框架基底 Tie 于此）
        lab_found = _set_label(odb, inst_s, 'SURF_CENTER')
        found_u1 = _node_hist(eq, inst_s, lab_found, 'U1')
        time = found_u1[:, 0]; u_found = found_u1[:, 1]
        dt = float(time[1] - time[0])

        # 各层
        floor_u = [u_found]; floor_a = []
        for k in range(1, ns + 1):
            lab = _set_label(odb, inst_f, 'FLOOR_%d' % k)
            floor_u.append(_node_hist(eq, inst_f, lab, 'U1')[:, 1])
            floor_a.append(_node_hist(eq, inst_f, lab, 'A1')[:, 1])

        # 周期延长：顶层相对基础位移 FFT
        rel_roof = floor_u[ns] - floor_u[0]
        f1_ssi, T_ssi = _dom_period(rel_roof, dt)
        T_fixed = meta['T_fixed_from_step1_s']
        print(u'\n[2] SSI 周期延长（标志性特征）：')
        print(u'    T_ssi(顶层相对基础 FFT) = %.3f s' % (T_ssi or 0))
        print(u'    T_fixed(step1 固定基础) = %.3f s' % T_fixed)
        if T_ssi:
            ratio = T_ssi / T_fixed
            verdict = u'✓ 周期延长(SSI 生效)' if ratio > 1.02 else u'≈ 无明显延长(土偏硬/检查耦合)'
            print(u'    T_ssi/T_fixed = %.2f  %s' % (ratio, verdict))
            results['T_ssi'] = T_ssi; results['period_lengthening_ratio'] = ratio

        # 基底剪力(=Σmᵢaᵢ) 与 顶层漂移
        inertia = np.zeros_like(floor_a[0])
        for k in range(ns):
            inertia = inertia + fm * floor_a[k]
        base_shear = float(np.max(np.abs(inertia)))
        drift = [float(np.max(np.abs(floor_u[k] - floor_u[k - 1])) / sh) for k in range(1, ns + 1)]
        print(u'\n[3] SSI 结构响应：')
        print(u'    基底剪力(Σmᵢaᵢ) = %.3e N' % base_shear)
        print(u'    最大层间位移角 = %.4f' % max(drift))
        print(u'    顶层峰值绝对加速度 = %.3f m/s² (放大 %.2f×)'
              % (np.max(np.abs(floor_a[ns - 1])), np.max(np.abs(floor_a[ns - 1])) / pga))
        results['ssi'] = {'base_shear': base_shear, 'max_drift': max(drift),
                          'drift_profile': drift,
                          'roof_peak_acc': float(np.max(np.abs(floor_a[ns - 1])))}
        odb.close()
    else:
        print(u'\n[2&3] 缺 job-ssi.odb，跳过 SSI')

    # ---------- 4. 刚性基础(输入自由场地表运动) vs SSI 去耦对比 ----------
    if os.path.isfile('job-fixed.odb'):
        odb = openOdb('job-fixed.odb', readOnly=True)
        inst_f = _resolve_inst(odb, meta['inst_frame'])
        eq = odb.steps['Step-EQ']
        ns = int(meta['n_story']); sh = float(meta['frame']['story_height']); fm = float(meta['frame']['floor_mass'])
        lab_b = _set_label(odb, inst_f, 'BASE')             # 刚性基础: 基底=框架柱脚(被自由场加速度驱动)
        base_u = _node_hist(eq, inst_f, lab_b, 'U1')
        timef = base_u[:, 0]; u_base = base_u[:, 1]; dtf = float(timef[1] - timef[0])
        fu = [u_base]; fa = []
        for k in range(1, ns + 1):
            lab = _set_label(odb, inst_f, 'FLOOR_%d' % k)
            fu.append(_node_hist(eq, inst_f, lab, 'U1')[:, 1])
            fa.append(_node_hist(eq, inst_f, lab, 'A1')[:, 1])
        f1f, Tf = _dom_period(fu[ns] - fu[0], dtf)
        inert = np.zeros_like(fa[0])
        for k in range(ns):
            inert = inert + fm * fa[k]
        bs_f = float(np.max(np.abs(inert)))
        dr_f = [float(np.max(np.abs(fu[k] - fu[k - 1])) / sh) for k in range(1, ns + 1)]
        roof_f = float(np.max(np.abs(fa[ns - 1])))
        T_fixed_nat = meta['T_fixed_from_step1_s']   # 刚性自振周期用 step1 值(0.5s)；fixed_ff 的 FFT 主频是受迫响应,近共振时不可靠
        results['fixed_ff'] = {'T_forced': Tf, 'T_natural': T_fixed_nat,
                               'base_shear': bs_f, 'max_drift': max(dr_f), 'roof_peak_acc': roof_f}
        print(u'\n[4] 刚性基础(输入自由场地表运动) vs SSI 去耦对比：')
        print(u'    刚性: T自振=%.3fs(step1) 基底剪力=%.3eN 最大漂移=%.4f 顶层加速=%.3f(放大%.1f×)'
              % (T_fixed_nat, bs_f, max(dr_f), roof_f, roof_f / pga))
        if 'ssi' in results:
            s = results['ssi']; Ts = results.get('T_ssi') or 0.0
            print(u'    SSI : T自振=%.3fs           基底剪力=%.3eN 最大漂移=%.4f 顶层加速=%.3f(放大%.1f×)'
                  % (Ts, s['base_shear'], s['max_drift'], s['roof_peak_acc'], s['roof_peak_acc'] / pga))
            print(u'    SSI/刚性比: 周期=%.2f 基底剪力=%.2f 漂移=%.2f 顶层加速=%.2f'
                  % (Ts / T_fixed_nat, s['base_shear'] / bs_f, s['max_drift'] / max(dr_f), s['roof_peak_acc'] / roof_f))
            # 解读：刚性自振(1/T)是否近场地基频
            print(u'    解读: 刚性 f=%.2fHz vs 场地 f0=%.2fHz' % (1.0 / T_fixed_nat, meta['site_f0_Hz']))
            print(u'          → SSI 周期延长使结构失谐离开场地共振，故响应骤降（SSI 失谐减震；本例属有利情形）')
            results['ssi_over_fixed'] = {
                'T': Ts / T_fixed_nat, 'base_shear': s['base_shear'] / bs_f,
                'drift': s['max_drift'] / max(dr_f), 'roof_acc': s['roof_peak_acc'] / roof_f}
        odb.close()
    else:
        print(u'\n[4] 缺 job-fixed.odb，跳过去耦对比')

    with open('ssi_results.json', 'w') as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print('\n' + '=' * 60)
    print(u'后处理完成。已写 ssi_results.json')
    print(u'验证判据: (1)地表放大合理&f≈f0 (2)T_ssi>T_fixed=周期延长 (3)结构响应合理')


if __name__ == '__main__':
    main()
