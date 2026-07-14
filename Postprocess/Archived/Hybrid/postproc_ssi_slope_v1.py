# -*- coding: utf-8 -*-
"""
Step-2b 坡顶 SSI 后处理 + 验证（多波）
==============================
遍历当前目录所有 job-<波名>-freefield.odb（自包含脚本 slope_frame_ssi_full_v1.py 的输出），
逐条波读 freefield/ssi/fixed 三 odb，验坡面 SSI 集成：
  1. 坡顶自由场：坡肩地表放大(地形+地层放大 TAF)、主频
  2. SSI 周期延长：T_ssi(顶层相对坡顶基础 FFT) > T_fixed(step1=0.5s)
  3. SSI 结构响应：基底剪力/层间位移角/顶层加速度
  4. (step2b-2)坡顶刚性 vs SSI 去耦对比：刚性基础输入坡顶自由场运动，SSI/刚性比量化 SSI 效应
每条波的 PGA 由其输入 <波名>.txt 现算（meta 不再存 pga，因多波各异）。

运行：abaqus python postproc_ssi_slope_v1.py
"""

import os
import sys
import glob
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


def _process_wave(tag, meta, pga):
    """处理单条波(tag)的 freefield/ssi/fixed 三 odb，返回该波结果 dict。"""
    T_fixed = float(meta['T_fixed_step1'])
    ns = int(meta['n_story']); sh = float(meta['story_height']); fm = float(meta['floor_mass'])
    res = {'pga': pga, 'T_fixed': T_fixed}
    ff_odb = 'job-%s-freefield.odb' % tag; ssi_odb = 'job-%s-ssi.odb' % tag; fix_odb = 'job-%s-fixed.odb' % tag
    print('=' * 60)
    print(u'■ 波: %s (输入 PGA=%.3f m/s²)' % (tag, pga))

    # ---------- 1. 坡顶自由场 ----------
    if os.path.isfile(ff_odb):
        odb = openOdb(ff_odb, readOnly=True)
        inst, lab = _crest_ref(odb)
        eq = odb.steps[list(odb.steps.keys())[0]]
        d = _node_hist(eq, inst, lab, 'A1')
        t = d[:, 0]; ca = d[:, 1]; dt = float(t[1] - t[0])
        amp = float(np.max(np.abs(ca)) / pga) if pga else float('nan')
        f_c, T_c = _dom_period(ca, dt)
        print(u'[1] 坡顶自由场(坡肩地表 x=%.0f)：坡顶放大 %.2f× (峰=%.3f), 主频 %.2f Hz'
              % (meta['left_flat'], amp, np.max(np.abs(ca)), f_c))
        res['freefield'] = {'crest_amp': amp, 'crest_f': f_c}
        odb.close()
    else:
        print(u'[1] 缺 %s' % ff_odb)

    # ---------- 2&3. 坡顶 SSI ----------
    if os.path.isfile(ssi_odb):
        odb = openOdb(ssi_odb, readOnly=True)
        inst_f = _resolve_inst(odb, meta['inst_frame'])
        inst_c, lab_c = _crest_ref(odb)               # 坡顶土面节点 = 框架基础运动
        eq = odb.steps[list(odb.steps.keys())[0]]
        u_found = _node_hist(eq, inst_c, lab_c, 'U1')
        t = u_found[:, 0]; u0 = u_found[:, 1]; dt = float(t[1] - t[0])
        floor_u = [u0]; floor_a = []
        for k in range(1, ns + 1):
            lab = odb.rootAssembly.instances[inst_f].nodeSets['FLOOR_%d' % k].nodes[0].label
            floor_u.append(_node_hist(eq, inst_f, lab, 'U1')[:, 1])
            floor_a.append(_node_hist(eq, inst_f, lab, 'A1')[:, 1])
        f1, T_ssi = _dom_period(floor_u[ns] - floor_u[0], dt)
        inertia = np.zeros_like(floor_a[0])
        for k in range(ns):
            inertia = inertia + fm * floor_a[k]
        drift = [float(np.max(np.abs(floor_u[k] - floor_u[k - 1])) / sh) for k in range(1, ns + 1)]
        print(u'[2] SSI 周期延长: T_ssi=%.3fs / T_fixed=%.3fs = %.2f%s'
              % (T_ssi or 0, T_fixed, (T_ssi / T_fixed) if T_ssi else 0, u' ✓延长' if T_ssi and T_ssi / T_fixed > 1.02 else u''))
        print(u'[3] SSI 结构响应: 基底剪力=%.3eN, 最大漂移=%.4f, 顶层放大=%.2f×'
              % (float(np.max(np.abs(inertia))), max(drift), np.max(np.abs(floor_a[ns - 1])) / pga if pga else float('nan')))
        if T_ssi:
            res['T_ssi'] = T_ssi; res['period_ratio'] = T_ssi / T_fixed
        res['ssi'] = {'base_shear': float(np.max(np.abs(inertia))), 'max_drift': max(drift),
                      'roof_peak_acc': float(np.max(np.abs(floor_a[ns - 1])))}
        odb.close()
    else:
        print(u'[2&3] 缺 %s' % ssi_odb)

    # ---------- 4. 坡顶刚性基础(输入坡顶自由场) vs SSI 去耦对比 (step2b-2) ----------
    if os.path.isfile(fix_odb):
        odb = openOdb(fix_odb, readOnly=True)
        inst_f = _resolve_inst(odb, meta['inst_frame'])
        eq = odb.steps[list(odb.steps.keys())[0]]
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
        res['fixed'] = {'T_natural': T_fixed_nat, 'base_shear': bs_f, 'max_drift': max(dr_f), 'roof_peak_acc': roof_f}
        print(u'[4] 刚性: 基底剪力=%.3eN 漂移=%.4f 顶层放大=%.2f×' % (bs_f, max(dr_f), roof_f / pga if pga else float('nan')))
        if 'ssi' in res:
            s = res['ssi']; Ts = res.get('T_ssi') or 0.0
            print(u'    SSI/刚性比: 周期=%.2f 基底剪力=%.2f 漂移=%.2f 顶层加速=%.2f'
                  % (Ts / T_fixed_nat, s['base_shear'] / bs_f, s['max_drift'] / max(dr_f), s['roof_peak_acc'] / roof_f))
            cf = res.get('freefield', {}).get('crest_f')  # 坡顶自由场主频(调谐解读)
            if cf:
                print(u'    解读: 刚性 f=%.2fHz vs 坡顶自由场 f=%.2fHz → SSI 周期延长改变结构与坡顶运动的调谐关系'
                      % (1.0 / T_fixed_nat, cf))
            res['ssi_over_fixed'] = {'T': Ts / T_fixed_nat, 'base_shear': s['base_shear'] / bs_f,
                                     'drift': s['max_drift'] / max(dr_f), 'roof_acc': s['roof_peak_acc'] / roof_f}
        odb.close()
    else:
        print(u'[4] 缺 %s，跳过去耦对比' % fix_odb)
    return res


def main():
    meta = _load_meta()
    ff_odbs = sorted(glob.glob('job-*-freefield.odb'))  # 多波：每条波一个 freefield odb
    if not ff_odbs:
        print(u'未找到 job-*-freefield.odb，无可后处理的波（确认已用 slope_frame_ssi_full 跑过且 submit=True）')
        return
    all_res = {'angle': meta['angle'], 'waves': {}}
    for ff in ff_odbs:
        tag = os.path.basename(ff)[4:-len('-freefield.odb')]  # job-<tag>-freefield.odb -> tag
        txt = tag + '.txt'
        if os.path.isfile(txt):
            d = np.loadtxt(txt); pga = float(np.max(np.abs(d[:, 1])))  # 该波输入 PGA
        else:
            pga = 1.0
            print(u'警告: 缺输入 %s，PGA 取 1.0（放大倍数将失真）' % txt)
        all_res['waves'][tag] = _process_wave(tag, meta, pga)
    with open('ssi_slope_results.json', 'w') as fh:
        json.dump(all_res, fh, indent=2, ensure_ascii=False)
    print('\n' + '=' * 60)
    print(u'后处理完成（%d 条波）。已写 ssi_slope_results.json' % len(all_res['waves']))


if __name__ == '__main__':
    main()
