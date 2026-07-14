# -*- coding: utf-8 -*-
"""坡地模型地表响应统一后处理 v1（PGA + AF/TAF + H(f)，配 slope_frame_ssi_full_v1.py）。

遍历当前工况目录所有 job-*.odb，从 TOP_SURFACE 节点集全时程场输出提取地表加速度，逐波输出：
  1. PGA：每个地表节点水平(A1)/竖向(A2)加速度峰值。
  2. 放大系数（口径与 case_meta 完全一致）：
       AF_h  = PGA_h / (factor_h × PGA_in)      —— 相对基岩入射的总放大（含场地+地形）
       TAF_h = AF_h / taf_h(同侧一维理论台阶)   —— 纯地形放大，远场应趋于 1
       AF_v  = PGA_v / (factor_h × PGA_in)      —— 寄生竖向放大（统一水平分母，B&P2005 口径）
       TAF_v = AF_v / taf_v(同侧)               —— 仅斜入射且 taf_v>TAFV_GUARD 时计算，否则 NaN
       V/H   = PGA_v / PGA_h                    —— 同点竖横比
     同侧规则：x ≤ x_toe（坡顶平台+坡面）用 left 柱，x > x_toe（坡脚平台）用 right 柱。
  3. 频域传函（线弹性前提；tssi 非线性/EQL 开启时曲线仅供参考）：
       H_h(x,f)     = |FFT(a1_surf)| / |FFT(a_in)|            —— 相对输入的水平传函
       H_v(x,f)     = |FFT(a2_surf)| / |FFT(a_in)|            —— 竖向传函（同一水平输入分母）
       Htopo_h(x,f) = |FFT(a1_surf)| / |FFT(a1_同侧远场参考)|  —— 参考台站谱比，纯地形放大谱
     输入谱幅 < SPEC_MASK_RATIO×峰值的频点直接剔除（0/0 噪声），频带再截 [F_LO, FMAX_FACTOR×fc]。
  4. QA：模型两端远场节点 AF_h 对拍 case_meta.ff_theory ±QA_TOL，超差该波标记 suspect。

输入：job-*.odb + case_meta.json + 输入波 txt（路径优先取 case_config.json 的 run_cfg.wave_files，
      按文件名主干与记录名匹配；缺省回退工况目录下同名 .txt）。
输出：surface_response_<record>.csv                        （逐节点一行）
      H_surface_h_<record>.csv / H_surface_v_<record>.csv  （首列 f，其后每节点一列，列头=x 坐标）
      H_topo_h_<record>.csv                                （同上，参考台站谱比）
      surface_summary.json                                 （逐波 QA / AR_max / 分母口径汇总）
运行：abaqus python Postprocess_All_surface_v1.py   （在含 job-*.odb 与 case_meta.json 的工况目录内）
约定：Abaqus 自带 Python 2.7 + numpy；纯数值函数不依赖 odbAccess，可在普通 Python 下单测。
"""

from __future__ import print_function

import os
import sys
import glob
import csv
import json
import math
import numpy as np

openOdb = None  # 占位
is_abaqus = False
try:
    # 尝试导入 abaqusConstants，若成功说明处于 Abaqus 环境。该导入在普通 Python 下抛出 ImportError 且无自举风险。
    import abaqusConstants
    is_abaqus = True
except ImportError:
    pass

if is_abaqus:
    try:
        from odbAccess import openOdb  # Abaqus ODB 接口
    except Exception:
        pass

try:
    if hasattr(sys, 'setdefaultencoding'):  # 仅在 Python 2 下执行
        eval("reload(sys)")                 # 动态求值以避开 Pylance 静态检查
        sys.setdefaultencoding('utf-8')
except Exception:
    pass

SPEC_MASK_RATIO = 0.05   # 输入谱幅值掩码比例：低于峰值 5% 的频点剔除（0/0 噪声带）
F_LO = 0.3               # 传函可靠带下限(Hz)：更低频段脉冲能量太薄
FMAX_FACTOR = 2.5        # 可靠带上限 = FMAX_FACTOR×fc（网格 K-L 判据同口径），fc 缺失则不截
PAD_FACTOR = 4           # FFT 补零倍数（防卷绕，与建模脚本 fd 引擎同口径）
QA_TOL = 0.05            # 远场 AF_h 对拍一维理论台阶的相对误差阈值（±5%）
TAFV_GUARD = 0.05        # taf_v 低于该值视为"竖向自由场≈0"（垂直入射），TAF_v 置 NaN 防除零


# ==========================================================
#  纯数值函数（不依赖 odbAccess，可单测）
# ==========================================================


def strip_record_name(odb_name):  # job-<记录>-slope.odb -> <记录>
    """从 odb 文件名剥离 job- 前缀与 -slope/-flat 场景后缀，返回记录名。"""
    base = os.path.basename(odb_name)
    if base.lower().endswith('.odb'):
        base = base[:-4]
    if base.lower().startswith('job-'):
        base = base[4:]
    for suf in ('-slope', '-flat'):  # 场景后缀（build_models 命名规则）
        if base.lower().endswith(suf):
            base = base[:-len(suf)]
            break
    return base


def to_uniform(t, sig_mat):
    """把帧时刻 t 与信号矩阵 sig_mat(节点×时刻) 重采样到等间隔时间轴。

    FIXED 增量下帧本就等距（原样返回）；CDP 非线性自动增量下帧距不等，
    用中位数 dt 建均匀轴并逐节点线性插值（FFT 前提是等间隔采样）。
    返回 (t_uniform, sig_uniform, dt)。
    """
    t = np.asarray(t, dtype=float)
    sig_mat = np.atleast_2d(np.asarray(sig_mat, dtype=float))
    dts = np.diff(t)
    dt = float(np.median(dts))
    if dt <= 0:
        raise ValueError('帧时间步 dt<=0，无法重采样')
    if float(np.max(dts) - np.min(dts)) > 1e-6 * dt:  # 帧距不均匀（自动增量）
        tu = np.arange(0.0, float(t[-1]) + 0.5 * dt, dt)  # 均匀时间轴
        out = np.vstack([np.interp(tu, t, sig_mat[k]) for k in range(sig_mat.shape[0])])  # 逐节点线性插值
        return tu, out, dt
    return t, sig_mat, dt


def compute_H(a_out_mat, a_in, dt, fc=None):
    """计算幅值传函矩阵 |H(x,f)| = |FFT(a_out)| / |FFT(a_in)|，只保留可靠频带。

    a_out_mat : 节点×时刻 加速度矩阵；a_in：输入加速度（自动补零对齐长度）；dt：采样步长。
    fc        : 输入主频(Hz)，给定时上限截 FMAX_FACTOR×fc，否则只按谱掩码与 F_LO。
    返回 (freqs, H)：freqs 为保留频点，H 为 节点×频点 幅值矩阵。
    """
    a_out_mat = np.atleast_2d(np.asarray(a_out_mat, dtype=float))
    a_in = np.asarray(a_in, dtype=float)
    n = max(a_out_mat.shape[1], a_in.size)  # 以较长者定 FFT 基准长度
    nfft = 1
    while nfft < PAD_FACTOR * n:  # 补零至 4 倍长度的 2 的幂（防卷绕 + 加密频轴）
        nfft *= 2
    A_in = np.fft.rfft(a_in, n=nfft)  # 输入单边谱
    A_out = np.fft.rfft(a_out_mat, n=nfft, axis=1)  # 各节点单边谱
    freqs = np.fft.rfftfreq(nfft, dt)  # 频率轴
    keep = np.abs(A_in) >= SPEC_MASK_RATIO * float(np.max(np.abs(A_in)))  # 谱幅值掩码：输入没能量的频点剔除
    keep &= freqs >= F_LO  # 下限截断
    if fc:  # 上限截断（网格只保证到 FMAX_FACTOR×fc）
        keep &= freqs <= FMAX_FACTOR * float(fc)
    keep[0] = False  # 排除直流分量
    return freqs[keep], np.abs(A_out[:, keep]) / np.abs(A_in[keep])[None, :]


def spectral_ratio(a_out_mat, a_ref, dt, fc=None):
    """参考台站谱比 |FFT(a_out)|/|FFT(a_ref)|（分母换成远场参考节点，掩码规则同 compute_H）。"""
    return compute_H(a_out_mat, a_ref, dt, fc=fc)


def surface_metrics(xs, a1_mat, a2_mat, pga_in, factor_h, taf_lr, x_toe):
    """由地表加速度矩阵计算逐节点 PGA/AF/TAF/V_H 指标。

    xs       : 节点 x 坐标（升序）；a1_mat/a2_mat：节点×时刻 水平/竖向加速度。
    pga_in   : 输入记录峰值；factor_h：斜入射自由面水平放大系数（分母 = factor_h×pga_in）。
    taf_lr   : {'left': (taf_h, taf_v), 'right': (taf_h, taf_v)}，一维理论台阶（可为 None）。
    x_toe    : 坡脚 x（同侧规则分界：x≤x_toe 用 left，x>x_toe 用 right）。
    返回逐节点 dict 列表（keys: x/PGA_h/PGA_v/AF_h/TAF_h/AF_v/TAF_v/V_over_H/ff_side）。
    """
    pga_h = np.max(np.abs(a1_mat), axis=1)  # 逐节点水平峰值
    pga_v = np.max(np.abs(a2_mat), axis=1)  # 逐节点竖向峰值
    denom = factor_h * pga_in if (factor_h and pga_in) else None  # 统一水平分母
    rows = []
    for k in range(len(xs)):
        side = 'left' if xs[k] <= x_toe else 'right'  # 同侧规则：坡顶平台+坡面归 left，坡脚平台归 right
        taf_h_ref, taf_v_ref = (taf_lr.get(side) or (None, None)) if taf_lr else (None, None)
        af_h = float(pga_h[k]) / denom if denom else float('nan')  # 相对入射总放大
        af_v = float(pga_v[k]) / denom if denom else float('nan')  # 寄生竖向放大（水平分母）
        taf_h = af_h / taf_h_ref if (denom and taf_h_ref) else float('nan')  # 纯地形水平放大
        taf_v = af_v / taf_v_ref if (denom and taf_v_ref and taf_v_ref > TAFV_GUARD) else float('nan')  # 竖向理论台阶太小(垂直入射)不除
        rows.append({'x': float(xs[k]), 'PGA_h': float(pga_h[k]), 'PGA_v': float(pga_v[k]),
                     'AF_h': af_h, 'TAF_h': taf_h, 'AF_v': af_v, 'TAF_v': taf_v,
                     'V_over_H': float(pga_v[k]) / float(pga_h[k]) if pga_h[k] > 0 else float('nan'),
                     'ff_side': side})
    return rows


def farfield_qa(rows, taf_lr):
    """远场 QA：两端节点 AF_h 对拍一维理论台阶，返回 (err_left, err_right, suspect)。"""
    err_l = err_r = None
    if rows and taf_lr:
        tl = (taf_lr.get('left') or (None, None))[0]  # 左柱理论台阶
        tr = (taf_lr.get('right') or (None, None))[0]  # 右柱理论台阶
        if tl and not math.isnan(rows[0]['AF_h']):
            err_l = abs(rows[0]['AF_h'] / tl - 1.0)  # 最左端节点相对误差
        if tr and not math.isnan(rows[-1]['AF_h']):
            err_r = abs(rows[-1]['AF_h'] / tr - 1.0)  # 最右端节点相对误差
    suspect = bool((err_l is not None and err_l > QA_TOL) or (err_r is not None and err_r > QA_TOL))
    return err_l, err_r, suspect


# ==========================================================
#  工况文件读取（meta / config / 输入波）
# ==========================================================


def _load_json(path):  # 读 json，缺失返回 None
    if not os.path.isfile(path):
        return None
    with open(path, 'r') as fh:
        return json.load(fh)


def find_wave_file(record, case_cfg):
    """按记录名定位输入波 txt：优先 case_config 的 run_cfg.wave_files（文件名主干匹配），回退目录同名 txt。"""
    cands = []
    wf = ((case_cfg or {}).get('run_cfg') or {}).get('wave_files') or []  # 注入的波形路径列表
    if not isinstance(wf, (list, tuple)):
        wf = [wf]
    cands.extend([str(p) for p in wf])
    cands.extend(sorted(glob.glob('*.txt')))  # 工况目录内 txt 兜底
    for p in cands:
        if os.path.splitext(os.path.basename(p))[0] == record and os.path.isfile(p):  # 主干与记录名一致
            return p
    exist = [p for p in cands if os.path.isfile(p)]  # 无精确匹配时
    return exist[0] if len(exist) == 1 else None  # 仅剩单一候选才敢兜底


def meta_pieces(meta):
    """从 case_meta.json 提取后处理所需字段：(factor_h, taf_lr, x_toe, fc)。"""
    norm = (meta or {}).get('ff_normalization') or {}
    factor_h = norm.get('factor_h')  # 斜入射自由面水平放大系数（0°时=2）
    ff = (meta or {}).get('ff_theory') or {}
    taf_lr = {}
    for side in ('left', 'right'):  # 左(上平台)/右(下平台)一维理论台阶
        blk = ff.get(side) or {}
        taf_lr[side] = (blk.get('taf_h'), blk.get('taf_v'))
    geo = (meta or {}).get('geometry') or {}
    x_toe = geo.get('x_toe')  # 坡脚 x（同侧规则分界）
    fc = ff.get('fc_used') or ((meta or {}).get('damping') or {}).get('fc')  # 主频（可靠带上限用）
    return factor_h, taf_lr, x_toe, fc


# ==========================================================
#  ODB 提取
# ==========================================================


def _surface_nodeset(odb):  # 定位含 TOP_SURFACE 节点集的实例
    for iname in odb.rootAssembly.instances.keys():
        inst = odb.rootAssembly.instances[iname]
        try:
            if 'TOP_SURFACE' in inst.nodeSets.keys():
                return inst.nodeSets['TOP_SURFACE']
        except Exception:
            pass
    raise RuntimeError('ODB 中未找到 TOP_SURFACE 节点集（建模需 surface_only 输出）')


def extract_surface_acc(odb):
    """从 ODB 提取 TOP_SURFACE 全时程加速度，按 x 升序返回 (xs, ys, t, a1_mat, a2_mat)。"""
    nset = _surface_nodeset(odb)
    labels = np.array([n.label for n in nset.nodes], dtype=int)
    xs = np.array([n.coordinates[0] for n in nset.nodes], dtype=float)
    ys = np.array([n.coordinates[1] for n in nset.nodes], dtype=float)
    order = np.argsort(xs)  # 地表节点按 x 升序（坡顶→坡脚）
    labels, xs, ys = labels[order], xs[order], ys[order]
    step = odb.steps[odb.steps.keys()[-1]]  # 最后一个分析步（地震步）
    times, rows1, rows2 = [], [], []
    for frame in step.frames:  # 逐帧提取（帧数=增量步数，几千帧量级）
        fo = frame.fieldOutputs['A'].getSubset(region=nset)  # 地表节点子集加速度场
        try:  # bulkDataBlocks 批量接口（快）
            data = np.concatenate([np.asarray(b.data, dtype=float) for b in fo.bulkDataBlocks])
            labs = np.concatenate([np.asarray(b.nodeLabels, dtype=int) for b in fo.bulkDataBlocks])
        except Exception:  # 逐值接口兜底（慢但稳）
            data = np.array([[v.data[0], v.data[1]] for v in fo.values], dtype=float)
            labs = np.array([v.nodeLabel for v in fo.values], dtype=int)
        pos = dict(zip([int(L) for L in labs], range(len(labs))))  # 标签→行号映射（逐帧重建，防顺序漂移）
        idx = np.array([pos[int(L)] for L in labels])
        times.append(float(frame.frameValue))
        rows1.append(data[idx, 0])  # A1 水平
        rows2.append(data[idx, 1])  # A2 竖向
    a1_mat = np.array(rows1, dtype=float).T  # 转成 节点×时刻
    a2_mat = np.array(rows2, dtype=float).T
    return xs, ys, np.array(times, dtype=float), a1_mat, a2_mat


# ==========================================================
#  输出
# ==========================================================


def _open_csv(path):  # Py2/Py3 兼容的 csv 写句柄
    if sys.version_info[0] >= 3:
        return open(path, 'w', newline='')
    return open(path, 'wb')


def write_response_csv(path, ys, rows):
    """写逐节点指标 csv：x,y,PGA_h,PGA_v,AF_h,TAF_h,AF_v,TAF_v,V_over_H,ff_side。"""
    fh = _open_csv(path)
    w = csv.writer(fh)
    w.writerow(['x', 'y', 'PGA_h', 'PGA_v', 'AF_h', 'TAF_h', 'AF_v', 'TAF_v', 'V_over_H', 'ff_side'])
    for k, r in enumerate(rows):
        w.writerow([r['x'], float(ys[k]), r['PGA_h'], r['PGA_v'], r['AF_h'], r['TAF_h'],
                    r['AF_v'], r['TAF_v'], r['V_over_H'], r['ff_side']])
    fh.close()


def write_H_csv(path, freqs, xs, H):
    """写传函矩阵 csv：首列 f(Hz)，其后每节点一列，列头为节点 x 坐标。"""
    fh = _open_csv(path)
    w = csv.writer(fh)
    w.writerow(['f_Hz'] + ['x=%.3f' % float(x) for x in xs])
    for i in range(len(freqs)):
        w.writerow([float(freqs[i])] + [float(v) for v in H[:, i]])
    fh.close()


# ==========================================================
#  主流程
# ==========================================================


def process_one_odb(odb_path, meta, case_cfg):
    """处理单条 odb：提取→指标→传函→写文件，返回汇总 dict。"""
    record = strip_record_name(odb_path)
    factor_h, taf_lr, x_toe, fc = meta_pieces(meta)
    wave = find_wave_file(record, case_cfg)  # 定位输入波（PGA_in 与 H 分母）
    if wave is None:
        print('警告: %s 找不到匹配输入波 txt，AF/TAF/H 跳过，仅输出 PGA' % record)
    odb = openOdb(path=odb_path, readOnly=True)
    try:
        xs, ys, t, a1_mat, a2_mat = extract_surface_acc(odb)
    finally:
        odb.close()
    t, a1_mat, dt = to_uniform(t, a1_mat)  # 均匀化时间轴（FIXED 下原样返回）
    _, a2_mat, _ = to_uniform(t, a2_mat)

    pga_in = None
    a_in = None
    if wave:
        rec = np.loadtxt(wave)  # 输入记录 [time, acc]
        a_in = rec[:, 1]
        pga_in = float(np.max(np.abs(a_in)))
        dt_in = float(rec[1, 0] - rec[0, 0])
        if abs(dt_in - dt) > 1e-6 * dt:  # 帧步长与输入步长不一致（输出降频/自动增量）时重采样输入
            a_in = np.interp(t, rec[:, 0], a_in, left=0.0, right=0.0)
    if x_toe is None:  # meta 缺几何时退化：全部归 left
        x_toe = float(xs[-1]) + 1.0
        print('警告: case_meta 缺 x_toe，同侧规则退化为全 left')

    rows = surface_metrics(xs, a1_mat, a2_mat, pga_in, factor_h, taf_lr, x_toe)  # 逐节点指标
    err_l, err_r, suspect = farfield_qa(rows, taf_lr)  # 远场对拍一维理论
    write_response_csv('surface_response_%s.csv' % record, ys, rows)

    if a_in is not None:
        n_len = a1_mat.shape[1]
        a_in_pad = np.zeros(n_len)  # 输入补零到地表时程长度（尾段静默）
        m = min(n_len, a_in.size)
        a_in_pad[:m] = a_in[:m]
        freqs, H_h = compute_H(a1_mat, a_in_pad, dt, fc=fc)  # 水平传函
        _, H_v = compute_H(a2_mat, a_in_pad, dt, fc=fc)  # 竖向传函（同一水平输入分母）
        # 参考台站谱比：同侧远场端节点作分母（左端 idx=0 / 右端 idx=-1），纯地形谱
        left_mask = xs <= x_toe
        Ht = np.zeros((len(xs), len(freqs)))
        f_l, Ht_l = spectral_ratio(a1_mat[left_mask], a1_mat[0], dt, fc=fc)  # 左侧组/左端参考
        f_r, Ht_r = spectral_ratio(a1_mat[~left_mask], a1_mat[-1], dt, fc=fc)  # 右侧组/右端参考
        Ht[left_mask] = np.vstack([np.interp(freqs, f_l, Ht_l[k]) for k in range(Ht_l.shape[0])])  # 对齐到统一频轴
        if np.any(~left_mask):
            Ht[~left_mask] = np.vstack([np.interp(freqs, f_r, Ht_r[k]) for k in range(Ht_r.shape[0])])
        write_H_csv('H_surface_h_%s.csv' % record, freqs, xs, H_h)
        write_H_csv('H_surface_v_%s.csv' % record, freqs, xs, H_v)
        write_H_csv('H_topo_h_%s.csv' % record, freqs, xs, Ht)

    taf_arr = np.array([r['TAF_h'] for r in rows], dtype=float)
    ar_idx = int(np.nanargmax(taf_arr)) if np.any(~np.isnan(taf_arr)) else None  # AR_max 在重采样前的原始曲线上取
    summary = {'record': record, 'n_nodes': len(xs), 'dt': dt, 'duration': float(t[-1]),
               'wave_file': wave, 'pga_in': pga_in, 'factor_h': factor_h, 'fc': fc,
               'AR_max': (float(taf_arr[ar_idx]) if ar_idx is not None else None),  # 峰值放大（招牌标量）
               'AR_max_x': (rows[ar_idx]['x'] if ar_idx is not None else None),  # 峰值位置
               'qa_farfield_err_left': err_l, 'qa_farfield_err_right': err_r, 'suspect': suspect}
    print('%s: 节点=%d PGA_in=%s AR_max=%s@x=%s QA(左/右)=%s/%s%s' % (
        record, len(xs), pga_in,
        summary['AR_max'], summary['AR_max_x'],
        ('%.1f%%' % (err_l * 100) if err_l is not None else 'NA'),
        ('%.1f%%' % (err_r * 100) if err_r is not None else 'NA'),
        ' [SUSPECT]' if suspect else ''))
    return summary


# ==========================================================
#  出版级绘图工具（三段归一化坐标 s 出图）
# ==========================================================
CB_PALETTE = {  # Okabe-Ito 色盲安全配色
    'black': '#000000',  # 黑
    'orange': '#E69F00',  # 橙
    'skyblue': '#56B4E9',  # 天蓝
    'green': '#009E73',  # 绿
    'yellow': '#F0E442',  # 黄
    'blue': '#0072B2',  # 蓝
    'vermillion': '#D55E00',  # 朱红
    'purple': '#CC79A7',  # 紫
}

CJK_SERIF_PRIORITY = [  # 中文衬线字体优先级列表
    'Noto Serif CJK SC', 'Noto Serif SC', 'Source Han Serif SC', 'Source Han Serif CN',  # 衬线宋体类
    'SimSun', 'NSimSun', 'STSong', 'Songti SC',  # 系统宋体类
]


def _detect_cjk_serif_local():  # 检测系统可用的中文衬线字体
    """检测可用的衬线宋体名。"""
    try:
        import matplotlib.font_manager as fm  # 导入字体管理器
        available = {f.name for f in fm.fontManager.ttflist}  # 可用字体集合
        for name in CJK_SERIF_PRIORITY:  # 遍历优先级
            if name in available:  # 命中
                return name  # 返回
        for name in available:  # 兜底模糊匹配
            low = name.lower()  # 小写
            if any(k in low for k in ('song', 'serif cjk', 'serif sc', 'serif cn', 'songti')):  # 包含宋体字眼
                return name  # 返回
    except Exception:  # 异常
        pass  # 忽略
    return None  # 未找到


def setup_cn_journal_style_local():  # 自动应用中文核心期刊绘图配置
    """配置 matplotlib 绘图风格实现中西文混排与出版级尺寸。"""
    import matplotlib.pyplot as plt  # 导入 pyplot
    cjk = _detect_cjk_serif_local()  # 检测中文字体
    if cjk:  # 找到中文字体
        serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # 混排回退链
        plt.rcParams.update({  # 更新字体链
            'font.family': serif_list,  # 字体系列
            'font.serif': serif_list,  # 衬线系列
            'mathtext.fontset': 'stix',  # 数学公式采用 STIX 风格
        })
    plt.rcParams.update({  # 批量更新出版级参数
        'axes.unicode_minus': False,  # 解决负号显示为方框问题
        'pdf.fonttype': 42,  # PDF 嵌入 TrueType 字体防止报错
        'ps.fonttype': 42,  # PS 同上
        'font.size': 8,  # 基准字号 8pt
        'axes.labelsize': 8,  # 轴标签字号 8pt
        'xtick.labelsize': 7,  # x 轴刻度字号 7pt
        'ytick.labelsize': 7,  # y 轴刻度字号 7pt
        'lines.linewidth': 0.8,  # 曲线线宽 0.8pt
        'axes.linewidth': 0.7,  # 边框线宽 0.7pt
        'xtick.direction': 'in',  # 刻度线朝内
        'ytick.direction': 'in',  # 刻度线朝内
    })


def style_axes_local(ax):  # 美化单轴外观
    """配置白底、四面朝内刻度与细密主次网格。"""
    ax.set_facecolor('white')  # 设置背景为白色
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)  # 朝内显示
    for spine in ax.spines.values():  # 遍历四周边框
        spine.set_color('black')  # 设为黑色
        spine.set_linewidth(0.8)  # 线宽 0.8
    ax.minorticks_on()  # 开启次刻度
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.5)  # 主网格
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.3)  # 次网格


def calc_s_coords(xs, x_crest, x_toe, h_ref):  # 计算三段归一化坐标 s
    """根据拓扑关系将 x 坐标计算为连续的无量纲三段归一化坐标 s。

    参数:
        xs (list/ndarray): 物理 x 坐标数组
        x_crest (float): 坡顶棱 x 坐标
        x_toe (float): 坡脚棱 x 坐标
        h_ref (float): 特征参考厚度

    返回:
        numpy.ndarray: 三段归一化 s 坐标数组
    """
    xs = np.asarray(xs, dtype=float)  # 强制转为 ndarray
    s = np.zeros_like(xs)  # 初始化 s 数组
    if h_ref is None or h_ref <= 0:  # 厚度参考无效
        h_ref = 1.0  # 兜底值
    w_slope = x_toe - x_crest  # 计算坡面水平跨度
    if w_slope <= 0:  # 无效跨度
        w_slope = 1.0  # 兜底值
    for i, x in enumerate(xs):  # 遍历每个节点
        if x <= x_crest:  # 属于段 A 坡顶平台
            s[i] = (x - x_crest) / h_ref  # 计算对应 s (值为负或零)
        elif x < x_toe:  # 属于段 B 坡面
            s[i] = (x - x_crest) / w_slope  # 线性映射至 [0, 1] 区间
        else:  # 属于段 C 坡脚平台
            s[i] = 1.0 + (x - x_toe) / h_ref  # 计算对应 s (值大于或等于 1)
    return s  # 返回结果


def read_response_csv_local(path):  # 无依赖读取地表响应 CSV
    """使用内置 csv 模块读取地表响应 CSV，以防无 pandas。

    参数:
        path (str): CSV 文件路径

    返回:
        list: 字典列表，每一个元素代表一行
    """
    data = []  # 初始化数据列表
    try:  # 尝试读取
        if sys.version_info[0] >= 3:  # Python 3
            f = open(path, 'r', encoding='utf-8-sig')  # 自动处理带 BOM 的 UTF-8
        else:  # Python 2
            f = open(path, 'rb')  # 以二进制打开
        reader = csv.reader(f)  # 创建 reader
        header = next(reader)  # 提取表头
        if header and header[0].startswith('\xef\xbb\xbf'):  # 清理可能存在的 BOM 字符
            header[0] = header[0][3:]  # 截断
        elif header and header[0].startswith('\ufeff'):  # 另一种 unicode BOM 标记
            header[0] = header[0][1:]  # 截断
        for row in reader:  # 循环读取每行
            if not row:  # 空行
                continue  # 跳过
            item = {}  # 行字典
            for idx, h in enumerate(header):  # 遍历表头与对应值
                val_str = row[idx]  # 值文本
                try:  # 尝试转换数值
                    if val_str.lower() in ('nan', 'na', ''):  # 空值/非法数字
                        val = float('nan')  # 置为 NaN
                    else:  # 正常数字
                        val = float(val_str)  # 转浮点数
                except ValueError:  # 非数值
                    val = val_str  # 保留字符串
                item[h] = val  # 写入字典
            data.append(item)  # 加入列表
        f.close()  # 关闭文件句柄
    except Exception as e:  # 读取异常
        print('[plot] 读取 CSV %s 失败: %s' % (path, str(e)))  # 提示用户
        return []  # 返回空
    return data  # 返回结果


def plot_results(meta, case_cfg):  # 画图主控制函数
    """查找并遍历所有的 surface_response_*.csv 文件，生成对应的 3x2 排版图表。

    参数:
        meta (dict): 从 case_meta.json 加载的元数据字典
        case_cfg (dict): 从 case_config.json 加载的配置字典
    """
    try:  # 尝试导入绘图包
        import matplotlib  # 导入 matplotlib
        matplotlib.use('Agg')  # 无界面后端防止崩溃
        import matplotlib.pyplot as plt  # 导入 pyplot
    except ImportError:  # 未装绘图包
        print('[plot] 提示: 未检测到 matplotlib 库，跳过图表自动绘制。')  # 提示
        return  # 退出

    geo = (meta or {}).get('geometry') or {}  # 获取几何参数
    x_crest = geo.get('x_crest')  # 坡顶棱
    x_toe = geo.get('x_toe')  # 坡脚棱
    h_ref = geo.get('h')  # 参考高度 h
    if h_ref is None or h_ref <= 0:  # 高度无效
        h_ref = geo.get('H', 1.0)  # 使用 H 兜底

    if x_crest is None or x_toe is None:  # 关键点缺失
        print('[plot] 警告: 元数据缺少 x_crest/x_toe 拐点，无法计算归一化坐标 s，跳过作图。')  # 警告
        return  # 退出

    csvs = glob.glob('surface_response_*.csv')  # 搜索符合的文件
    if not csvs:  # 无文件
        print('[plot] 未发现任何已生成的 surface_response_*.csv 曲线表，跳过作图。')  # 提示
        return  # 退出

    try:  # 尝试应用中文出版级样式
        setup_cn_journal_style_local()  # 应用配置
    except Exception as e:  # 失败
        print('[plot] 无法应用中文核心期刊字体系数: %s，回退默认配置。' % str(e))  # 提示

    for csv_path in csvs:  # 遍历处理各记录
        record = csv_path[len('surface_response_'):-4]  # 提取记录名
        try:  # 尝试画图
            data = read_response_csv_local(csv_path)  # 读取指标数据
            if not data:  # 无数据
                continue  # 跳过

            xs = [row['x'] for row in data]  # 提取 x 坐标列表
            s_coords = calc_s_coords(xs, x_crest, x_toe, h_ref)  # 转换为 s 坐标
            fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.5))  # 创建 3x2 双栏物理网格画布

            # 网格绘制布局配置
            draw_specs = [
                (0, 0, 'PGA_h', '水平向 PGA (m/s²)', CB_PALETTE['blue'], '-'),  # 水平 PGA
                (0, 1, 'PGA_v', '垂直向 PGA (m/s²)', CB_PALETTE['vermillion'], '-'),  # 垂直 PGA
                (1, 0, 'AF_h', '水平向 AF', CB_PALETTE['blue'], '-'),  # 水平 AF
                (1, 1, 'AF_v', '垂直向 AF', CB_PALETTE['vermillion'], '-'),  # 垂直 AF
                (2, 0, 'TAF_h', '水平向 TAF', CB_PALETTE['blue'], '-'),  # 水平 TAF
                (2, 1, 'TAF_v', '垂直向 TAF', CB_PALETTE['vermillion'], '-'),  # 垂直 TAF
            ]

            for row_idx, col_idx, field, ylabel, color, linestyle in draw_specs:  # 遍历规格配置
                ax = axes[row_idx, col_idx]  # 当前轴
                style_axes_local(ax)  # 网格和边框美化

                y_vals = []  # y 值数组
                valid_s = []  # s 数组
                for idx, r in enumerate(data):  # 迭代每一行
                    val = r.get(field)  # 获取值
                    if val is not None and not math.isnan(val):  # 过滤非法 NaN
                        y_vals.append(val)  # 填充
                        valid_s.append(s_coords[idx])  # 填充对应的 s 坐标

                if y_vals:  # 拥有合法数值
                    ax.plot(valid_s, y_vals, color=color, linestyle=linestyle, linewidth=1.0)  # 绘图

                ax.set_xlim(s_coords.min(), s_coords.max())  # 坐标轴左右范围

                ax.axvline(x=0.0, color='black', linestyle='--', linewidth=0.8)  # 绘制坡顶垂直辅助线
                ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8)  # 绘制坡脚垂直辅助线

                y_lim = ax.get_ylim()  # 获取当前 Y 轴极限
                ty = y_lim[0] + 0.92 * (y_lim[1] - y_lim[0])  # 计算文字纵坐标
                ax.text(-0.05, ty, '#1', fontsize=7, va='top', ha='right')  # 标注 #1 坡顶棱
                ax.text(0.95, ty, '#2', fontsize=7, va='top', ha='right')  # 标注 #2 坡脚棱

                if row_idx == 2:  # 底行坐标轴
                    ax.set_xlabel('三段归一化坐标 s')  # 设置横轴中文标签
                else:  # 非底行
                    ax.set_xticklabels([])  # 隐藏 x 刻度文本

                ax.set_ylabel(ylabel)  # 设置纵轴标签

            fig.suptitle('记录: %s' % record, fontsize=9, fontweight='bold', y=0.98)  # 总标题
            fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.4, w_pad=0.4)  # 紧凑布局微调
            
            out_dir = 'figs'  # 输出子目录
            if not os.path.exists(out_dir):  # 目录不存在
                os.makedirs(out_dir)  # 创建

            fig_path = os.path.join(out_dir, 'surface_response_%s' % record)  # 文件基路径
            for fmt in ('png', 'pdf'):  # 导出多格式
                fig.savefig('%s.%s' % (fig_path, fmt), dpi=300, bbox_inches='tight')  # 保存
            plt.close(fig)  # 关闭释放内存
            print('[plot] 成功生成三段归一化图表: %s.{png,pdf}' % fig_path)  # 提示用户
        except Exception as e:  # 绘图异常
            print('[plot] 绘制记录 %s 失败: %s' % (record, str(e)))  # 错误提示


def main():  # 主入口函数
    """后处理脚本控制流。"""
    meta = _load_json('case_meta.json')  # 读取元数据
    case_cfg = _load_json('case_config.json')  # 读取配置
    
    if openOdb is None:  # 无 Abaqus 环境
        print('提示: 未检测到 odbAccess (非 Abaqus 环境)，跳过 ODB 提取，直接进行图表重绘。')  # 提示
        plot_results(meta, case_cfg)  # 运行绘图
        return  # 正常退出
        
    odbs = sorted(glob.glob('job-*.odb'))  # 搜索 odb 文件
    if not odbs:  # 无 odb
        print('错误: 当前目录无 job-*.odb，无法提取数据。')  # 报错
        sys.exit(1)  # 退出
        
    summaries = []  # 摘要列表
    for p in odbs:  # 遍历
        try:  # 尝试提取
            summaries.append(process_one_odb(p, meta, case_cfg))  # 处理单条 ODB
        except Exception as e:  # 异常
            print('错误: %s 处理失败: %s' % (p, str(e)))  # 打印错误
            summaries.append({'record': strip_record_name(p), 'error': str(e)})  # 记录错误
            
    with open('surface_summary.json', 'w') as fh:  # 写摘要
        json.dump({'schema_version': 1, 'records': summaries}, fh, indent=2)  # 保存 json
    print('完成: %d 条 odb，汇总见 surface_summary.json' % len(odbs))  # 提示完成
    
    plot_results(meta, case_cfg)  # 提取数据后自动画图


if __name__ == '__main__':  # 程序入口
    main()  # 执行主入口

