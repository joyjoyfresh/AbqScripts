# -*- coding: utf-8 -*-
"""坡地模型地表响应统一后处理 v2（PGA + AF/TAF + H(f) + 三段重采样，配 slope_frame_ssi_full_v1.py）。

v2 变更（相对 v1）：新增研究计划 §4.0 第②步"两步对齐"的重采样——把逐节点曲线与 H(f,s) 曲面
按三段（坡顶平台A/坡面B/坡脚平台C）插值到统一 s 子网格（N_A=120/N_B=60/N_C=80，段A 近坡顶棱加密），
输出固定长度 N_A+N_B+N_C 的对齐矩阵给 POD/ML；AR_max 仍在重采样前的原始曲线上精确提取，
并把其所在段号与归一坐标回写 surface_summary.json。

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
      surface_summary.json                                 （逐波 QA / AR_max(+段号/归一坐标) / 分母口径汇总）
      figs/surface_response_<record>.{png,pdf,svg}         （三段分轴出版级图，横轴为 §4.0 三段归一坐标 s）
      sgrid_response_<record>.csv                          （统一 s 子网格对齐响应表，N_A+N_B+N_C 行）
      sgrid_H_surface_h/v_<record>.csv、sgrid_H_topo_h_<record>.csv（列=统一网格点的对齐 H 曲面）
      sgrid_params.json                                    （子网格参数，跨工况一致性校验锚点）
运行：abaqus python Postprocess_All_surface_v2.py   （在含 job-*.odb 与 case_meta.json 的工况目录内；
      普通 Python 下运行则跳过 ODB 提取，对已有 CSV 重做重采样与绘图）
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
        eval("reload(sys)")  # 用 eval 动态执行，避开 Python 3 静态分析对未定义 reload 的报错
        sys.setdefaultencoding('utf-8')  # 设置默认编码
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


def _rc_safe(params):  # 逐键安全写入 rcParams
    """逐键写入 rcParams，旧版 matplotlib 缺键时跳过该键不崩溃。"""
    import matplotlib.pyplot as plt  # 导入 pyplot
    for k, v in params.items():  # 遍历配置项
        try:  # 尝试写入
            plt.rcParams[k] = v  # 写入单键
        except Exception:  # 该版本无此键
            pass  # 跳过


def setup_cn_journal_style_local():  # 自动应用中文核心期刊绘图配置
    """配置 matplotlib 出版级样式，返回选用的中文字体名（找不到返回 None，调用方应改用英文标签）。

    字体链按 matplotlib 版本适配：>=3.6 支持逐字形回退，Times 在前实现"宋体正文+Times 数字"混排；
    旧版（含 Abaqus 自带 Python 的 matplotlib）只取链中首个可用字体，必须把中文字体放最前，
    否则汉字全部渲染成方框——这正是 v1 首版在 Abaqus 环境下作图出方框的根因。
    """
    import matplotlib  # 导入顶层包（取版本号）
    cjk = _detect_cjk_serif_local()  # 检测中文字体
    try:  # 解析版本号
        mpl_ver = tuple(int(p) for p in matplotlib.__version__.split('.')[:2])  # (主, 次)
    except Exception:  # 解析失败
        mpl_ver = (0, 0)  # 按旧版处理
    if cjk:  # 找到中文字体
        if mpl_ver >= (3, 6):  # 新版：逐字形回退，Times 在前混排
            serif_list = ['Times New Roman', cjk, 'STIXGeneral', 'DejaVu Serif']  # 混排回退链
        else:  # 旧版：只认首个可用字体，中文在前保证不出方框
            serif_list = [cjk, 'Times New Roman', 'STIXGeneral', 'DejaVu Serif']  # 中文优先链
        _rc_safe({
            'font.family': serif_list,  # 字体系列
            'font.serif': serif_list,  # 衬线系列
            'mathtext.fontset': 'stix',  # 数学公式采用 STIX 风格
        })
    _rc_safe({
        'axes.unicode_minus': False,  # 解决负号显示为方框问题
        'pdf.fonttype': 42,  # PDF 嵌入 TrueType 字体防止报错
        'ps.fonttype': 42,  # PS 同上
        'svg.fonttype': 'none',  # SVG 文本保持可编辑
        'font.size': 8,  # 基准字号 8pt
        'axes.labelsize': 8,  # 轴标签字号 8pt
        'xtick.labelsize': 7,  # x 轴刻度字号 7pt
        'ytick.labelsize': 7,  # y 轴刻度字号 7pt
        'lines.linewidth': 1.2,  # 曲线线宽 1.2pt（与 Plot_Fig15_compare_v3 一致）
        'axes.linewidth': 0.7,  # 边框线宽 0.7pt
        'xtick.direction': 'in',  # 刻度线朝内
        'ytick.direction': 'in',  # 刻度线朝内
    })
    return cjk  # 返回选用中文字体名（None=未找到）


def style_axes_local(ax):  # 美化单轴外观
    """配置白底、四面朝内刻度与细密主次网格（口径与 Plot_Fig15_compare_v3 的 style_axes 一致）。"""
    ax.set_facecolor('white')  # 设置背景为白色
    ax.tick_params(direction='in', which='both', top=True, right=True, bottom=True, left=True)  # 朝内显示
    for spine in ax.spines.values():  # 遍历四周边框
        spine.set_color('black')  # 设为黑色
        spine.set_linewidth(1.0)  # 线宽 1.0（与 v3 一致）
    ax.minorticks_on()  # 开启次刻度
    ax.grid(True, which='major', linestyle='-', color='#c8c8c8', linewidth=0.6)  # 主网格
    ax.grid(True, which='minor', linestyle=':', color='#dcdcdc', linewidth=0.4)  # 次网格


def _resolve_slope_height(geo):  # 推算坡高（三段归一化 A/C 段特征长度）
    """从 case_meta.geometry 推算坡高：H_minus_h → w_slope·tan(i) → (x_toe−x_crest)·tan(i)。

    注意 geometry['h'] 是"坡脚下土层厚"（可为负），不是坡高，绝不能用它归一化——
    这是 v1 首版 s 轴范围出错（本应 [-A_max, 1+C_max] 却画到 [-10, 8.5]）的根因。
    找不到有效坡高返回 None。
    """
    hmh = geo.get('H_minus_h')  # 坡高（建模脚本口径：H_minus_h 即斜坡特征高度）
    if hmh and float(hmh) > 0:  # 有效
        return float(hmh)  # 直接返回
    w = geo.get('w_slope')  # 坡面水平跨度
    if w is None and geo.get('x_toe') is not None and geo.get('x_crest') is not None:  # 跨度缺失时由拐点差推算
        w = float(geo['x_toe']) - float(geo['x_crest'])  # 坡脚 x − 坡顶 x
    i_deg = geo.get('i')  # 坡角(度)
    if w and i_deg:  # 跨度与坡角齐备
        h = float(w) * math.tan(math.radians(float(i_deg)))  # 坡高 = 水平跨度×tan(坡角)
        if h > 0:  # 有效
            return h  # 返回推算值
    return None  # 无法推算


def calc_s_coords(xs, x_crest, x_toe, h_ref):  # 计算三段归一化坐标 s
    """根据拓扑关系将 x 坐标计算为连续的无量纲三段归一化坐标 s（研究计划 §4.0）。

    参数:
        xs (list/ndarray): 物理 x 坐标数组
        x_crest (float): 坡顶棱 x 坐标
        x_toe (float): 坡脚棱 x 坐标
        h_ref (float): 坡高（A/C 段特征长度，用 _resolve_slope_height 取得）

    返回:
        numpy.ndarray: 三段归一化 s 坐标数组（段A ≤0 / 段B [0,1] / 段C ≥1，拐点严格对齐）
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
        elif header and header[0].startswith('﻿'):  # 另一种 unicode BOM 标记
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


def _grayscale_preview_local(png_path):  # 生成灰度预览（色盲自检，可选）
    """用 Pillow 把 PNG 转灰度另存 *_grayscale.png；无 Pillow 时静默跳过，返回路径或 None。"""
    try:  # 尝试导入 Pillow
        from PIL import Image  # 图像处理
    except ImportError:  # 未安装
        return None  # 跳过
    try:  # 尝试转换
        gray_path = png_path[:-4] + '_grayscale.png'  # 灰度版路径
        Image.open(png_path).convert('L').save(gray_path)  # 转灰度保存
        return gray_path  # 返回路径
    except Exception:  # 转换失败
        return None  # 跳过


def plot_results(meta, case_cfg):  # 画图主控制函数
    """遍历 surface_response_*.csv，按【三段分轴】布局生成 3x2 出版级图表。

    每个指标面板拆成 坡顶平台(A)/坡面(B)/坡脚平台(C) 三个共 y 轴子轴：
    宽度按各段归一跨度定（B 段保底宽度，避免被挤成一条缝），
    坡顶棱 #1 / 坡脚棱 #2 严格落在轴缝虚线上——对应研究计划 §4.0 拐点对齐约定。
    观测范围取 case_config.geometry_cfg 的 crest_window/toe_window（单位 h），
    未配置时不截断（连边界净空段一起画）并提示。

    参数:
        meta (dict): 从 case_meta.json 加载的元数据字典
        case_cfg (dict): 从 case_config.json 加载的配置字典
    """
    try:  # 尝试导入绘图包
        import matplotlib  # 导入 matplotlib
        matplotlib.use('Agg')  # 无界面后端防止崩溃
        import matplotlib.pyplot as plt  # 导入 pyplot
        import matplotlib.gridspec as gridspec  # 网格布局（三段分轴用）
    except ImportError:  # 未装绘图包
        print('[plot] 提示: 未检测到 matplotlib 库，跳过图表自动绘制。')  # 提示
        return  # 退出

    ctx = _resolve_s_context(meta, case_cfg)  # 解析三段坐标几何上下文（拐点/坡高/观测窗口）
    if ctx is None:  # 上下文不完整
        print('[plot] 警告: case_meta 缺拐点或坡高（x_crest/x_toe 或 H_minus_h/w_slope·tan(i)），跳过作图。')  # 警告
        return  # 退出
    x_crest, x_toe, h_slope, a_win, c_win = ctx  # 解包几何上下文

    csvs = glob.glob('surface_response_*.csv')  # 搜索符合的文件
    if not csvs:  # 无文件
        print('[plot] 未发现任何已生成的 surface_response_*.csv 曲线表，跳过作图。')  # 提示
        return  # 退出

    cjk = None  # 选用中文字体名
    try:  # 尝试应用中文出版级样式
        cjk = setup_cn_journal_style_local()  # 应用配置（返回中文字体名或 None）
    except Exception as e:  # 失败
        print('[plot] 应用出版级样式失败: %s，回退默认配置。' % str(e))  # 提示
    use_cn = bool(cjk)  # 找不到中文字体时整图改用英文标签，杜绝方框
    if not use_cn:  # 无中文字体
        print('[plot] 提示: 未检测到中文字体，图内文字改用英文标签，避免渲染成方框。')  # 提示

    if use_cn:  # 中文标签集
        labels = {'PGA_h': u'水平向 PGA (m/s²)', 'PGA_v': u'垂直向 PGA (m/s²)',
                  'AF_h': u'水平向 AF', 'AF_v': u'垂直向 AF',
                  'TAF_h': u'水平向 TAF', 'TAF_v': u'垂直向 TAF'}  # 纵轴标签
        seg_titles = (u'坡顶平台', u'坡面', u'坡脚平台')  # 三段标题
        xlabel = u'三段归一化坐标 s'  # 横轴标签
        sup_fmt = u'记录: %s'  # 总标题模板
    else:  # 英文兜底标签集
        labels = {'PGA_h': u'Horizontal PGA (m/s²)', 'PGA_v': u'Vertical PGA (m/s²)',
                  'AF_h': u'Horizontal AF', 'AF_v': u'Vertical AF',
                  'TAF_h': u'Horizontal TAF', 'TAF_v': u'Vertical TAF'}  # 纵轴标签
        seg_titles = (u'Crest plateau', u'Slope', u'Toe plateau')  # 三段标题
        xlabel = u'Three-segment normalized coordinate s'  # 横轴标签
        sup_fmt = u'Record: %s'  # 总标题模板

    draw_specs = [  # 面板布局：(行, 列, 字段, 颜色)
        (0, 0, 'PGA_h', CB_PALETTE['blue']),  # 水平 PGA
        (0, 1, 'PGA_v', CB_PALETTE['vermillion']),  # 垂直 PGA
        (1, 0, 'AF_h', CB_PALETTE['blue']),  # 水平 AF
        (1, 1, 'AF_v', CB_PALETTE['vermillion']),  # 垂直 AF
        (2, 0, 'TAF_h', CB_PALETTE['blue']),  # 水平 TAF
        (2, 1, 'TAF_v', CB_PALETTE['vermillion']),  # 垂直 TAF
    ]

    for csv_path in csvs:  # 遍历处理各记录
        record = csv_path[len('surface_response_'):-4]  # 提取记录名
        try:  # 尝试画图
            data = read_response_csv_local(csv_path)  # 读取指标数据
            if not data:  # 无数据
                continue  # 跳过

            xs = [row['x'] for row in data]  # 提取 x 坐标列表
            s_all = calc_s_coords(xs, x_crest, x_toe, h_slope)  # 转换为 s 坐标（按坡高归一）
            a_max = float(a_win) if (a_win and float(a_win) > 0) else max(float(-s_all.min()), 0.5)  # 段A 显示跨度
            c_max = float(c_win) if (c_win and float(c_win) > 0) else max(float(s_all.max()) - 1.0, 0.5)  # 段C 显示跨度
            if not (a_win and c_win):  # 未配置观测窗口
                print('[plot] 提示: geometry_cfg 未配置 crest_window/toe_window，观测范围不截断（净空段一并画出）。')  # 提示
            w_b = max(1.0, 0.3 * (a_max + c_max))  # 段B 显示保底宽度（各段特征长度本不同，显示比例允许不同）

            fig = plt.figure(figsize=(6.3, 7.5))  # 双栏宽画布
            outer = gridspec.GridSpec(3, 2, left=0.10, right=0.985, top=0.90, bottom=0.075,
                                      hspace=0.32, wspace=0.26)  # 外层 3x2 网格
            bottom_axes = {}  # 底行两列的 (axA, axC)，用于放横轴标签

            for row_idx, col_idx, field, color in draw_specs:  # 遍历面板规格
                inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[row_idx, col_idx],
                                                         width_ratios=[a_max, w_b, c_max], wspace=0.0)  # 三段子网格
                ax_a = fig.add_subplot(inner[0])  # 段A 坡顶平台
                ax_b = fig.add_subplot(inner[1], sharey=ax_a)  # 段B 坡面（共 y 轴）
                ax_c = fig.add_subplot(inner[2], sharey=ax_a)  # 段C 坡脚平台（共 y 轴）
                for ax in (ax_a, ax_b, ax_c):  # 统一外观
                    style_axes_local(ax)  # 网格和边框美化

                seg_s = {'A': [], 'B': [], 'C': []}  # 各段 s 坐标
                seg_y = {'A': [], 'B': [], 'C': []}  # 各段 y 值
                y_all = []  # 全部有效值（统一 ylim 用）
                for idx, r in enumerate(data):  # 迭代每一行
                    val = r.get(field)  # 获取值
                    if val is None or (isinstance(val, float) and math.isnan(val)):  # 过滤非法 NaN
                        continue  # 跳过
                    sk = float(s_all[idx])  # 该点 s 坐标
                    if sk < -a_max - 1e-9 or sk > 1.0 + c_max + 1e-9:  # 截断到观测范围
                        continue  # 跳过
                    y_all.append(float(val))  # 记录有效值
                    if sk <= 1e-9:  # 段A（含坡顶棱）
                        seg_s['A'].append(sk); seg_y['A'].append(float(val))
                    if -1e-9 <= sk <= 1.0 + 1e-9:  # 段B（两端拐点都收，保证曲线触缝）
                        seg_s['B'].append(sk); seg_y['B'].append(float(val))
                    if sk >= 1.0 - 1e-9:  # 段C（含坡脚棱）
                        seg_s['C'].append(sk); seg_y['C'].append(float(val))

                for seg, ax in (('A', ax_a), ('B', ax_b), ('C', ax_c)):  # 逐段画曲线
                    if seg_s[seg]:  # 该段有数据
                        ax.plot(seg_s[seg], seg_y[seg], color=color, linestyle='-', linewidth=1.2)  # 绘图

                ax_a.set_xlim(-a_max, 0.0)  # 段A 范围
                ax_b.set_xlim(0.0, 1.0)  # 段B 范围
                ax_c.set_xlim(1.0, 1.0 + c_max)  # 段C 范围
                if y_all:  # 统一三段 ylim（sharey 自动同步）
                    lo, hi = min(y_all), max(y_all)  # 极值
                    pad = 0.06 * ((hi - lo) if hi > lo else max(abs(hi), 1.0))  # 上下留白
                    ax_a.set_ylim(lo - pad, hi + pad)  # 设定范围

                step_a = max(1, int(math.ceil(a_max / 4.0)))  # 段A 刻度步长（约 4 个主刻度）
                ax_a.set_xticks([-float(t) for t in range(0, int(math.floor(a_max)) + 1, step_a)])  # 段A 整数刻度
                ax_b.set_xticks([0.5])  # 段B 只标中点（0/1 在轴缝上）
                step_c = max(1, int(math.ceil(c_max / 4.0)))  # 段C 刻度步长
                ax_c.set_xticks([1.0 + float(t) for t in range(0, int(math.floor(c_max)) + 1, step_c)])  # 段C 整数刻度

                ax_a.spines['right'].set_visible(False)  # 隐藏段A 右边框（让位给缝上虚线）
                ax_c.spines['left'].set_visible(False)  # 隐藏段C 左边框
                for sd in ('left', 'right'):  # 段B 两侧边框改虚线 = 坡顶/坡脚棱标记线
                    ax_b.spines[sd].set_linestyle('--')  # 虚线
                    ax_b.spines[sd].set_linewidth(0.9)  # 线宽 0.9（与 v3 竖虚线一致）
                plt.setp(ax_b.get_yticklabels(), visible=False)  # 段B 隐藏 y 刻度文本
                plt.setp(ax_c.get_yticklabels(), visible=False)  # 段C 隐藏 y 刻度文本

                ax_a.text(0.97, 0.94, '#1', transform=ax_a.transAxes, fontsize=7, va='top', ha='right')  # 标注 #1 坡顶棱
                ax_c.text(0.03, 0.94, '#2', transform=ax_c.transAxes, fontsize=7, va='top', ha='left')  # 标注 #2 坡脚棱

                if row_idx == 0:  # 首行标注三段名称
                    ax_a.set_title(seg_titles[0], fontsize=7, pad=2)  # 段A 标题
                    ax_b.set_title(seg_titles[1], fontsize=7, pad=2)  # 段B 标题
                    ax_c.set_title(seg_titles[2], fontsize=7, pad=2)  # 段C 标题
                if row_idx == 2:  # 底行记录轴对象（横轴标签用）
                    bottom_axes[col_idx] = (ax_a, ax_c)  # 存左右子轴
                else:  # 非底行
                    for ax in (ax_a, ax_b, ax_c):  # 隐藏 x 刻度文本
                        plt.setp(ax.get_xticklabels(), visible=False)  # 隐藏
                ax_a.set_ylabel(labels[field])  # 设置纵轴标签

            for col_idx in bottom_axes:  # 底行两列各放一个居中横轴标签
                p_a = bottom_axes[col_idx][0].get_position()  # 左子轴位置
                p_c = bottom_axes[col_idx][1].get_position()  # 右子轴位置
                fig.text((p_a.x0 + p_c.x1) / 2.0, p_a.y0 - 0.045, xlabel,
                         ha='center', va='top', fontsize=8)  # 居中放置

            fig.suptitle(sup_fmt % record, fontsize=9, fontweight='bold', y=0.97)  # 总标题

            out_dir = 'figs'  # 输出子目录
            if not os.path.exists(out_dir):  # 目录不存在
                os.makedirs(out_dir)  # 创建

            fig_path = os.path.join(out_dir, 'surface_response_%s' % record)  # 文件基路径
            for fmt in ('png', 'pdf', 'svg'):  # 多格式导出（矢量+栅格，与 Plot_Fig15_compare_v3 口径）
                try:  # 逐格式保存
                    fig.savefig('%s.%s' % (fig_path, fmt), dpi=300, bbox_inches='tight', pad_inches=0.05)  # 保存
                except Exception as e2:  # 某格式失败不影响其余
                    print('[plot] 导出 %s 格式失败: %s' % (fmt, str(e2)))  # 提示
            plt.close(fig)  # 关闭释放内存
            gray = _grayscale_preview_local(fig_path + '.png')  # 灰度预览（色盲自检，可选）
            print('[plot] 成功生成三段分轴图表: %s.{png,pdf,svg}%s' % (
                fig_path, (' + 灰度预览' if gray else '')))  # 提示用户
        except Exception as e:  # 绘图异常
            print('[plot] 绘制记录 %s 失败: %s' % (record, str(e)))  # 错误提示


# ==========================================================
#  三段重采样（研究计划 §4.0 第②步：统一 s 子网格对齐输出，给 POD/ML 用）
# ==========================================================
N_A = 120  # 段A(坡顶平台)子网格点数（研究计划 §4.0 推荐值）
N_B = 60  # 段B(坡面)子网格点数
N_C = 80  # 段C(坡脚平台)子网格点数
CREST_REFINE_GAMMA = 2.0  # 段A 近坡顶棱加密指数（>1 越大越密，=1 退化为均匀网格）
_SEG_EPS = 1e-9  # 拐点归段容差（s=0/1 两侧段共享）
SGRID_FIELDS = ['PGA_h', 'PGA_v', 'AF_h', 'TAF_h', 'AF_v', 'TAF_v', 'V_over_H']  # 参与重采样的响应字段


def _resolve_s_context(meta, case_cfg):  # 解析三段坐标几何上下文
    """从 case_meta/case_config 解析三段坐标所需几何，绘图与重采样共用。

    返回 (x_crest, x_toe, h_slope, crest_window, toe_window)；拐点或坡高缺失返回 None。
    """
    geo = (meta or {}).get('geometry') or {}  # 几何块
    x_crest = geo.get('x_crest')  # 坡顶棱
    x_toe = geo.get('x_toe')  # 坡脚棱
    if x_crest is None or x_toe is None:  # 拐点缺失
        return None  # 无法计算
    h_slope = _resolve_slope_height(geo)  # 坡高（A/C 段特征长度）
    if h_slope is None:  # 坡高缺失
        return None  # 无法计算
    gcfg = (case_cfg or {}).get('geometry_cfg') or {}  # 无量纲几何设计配置
    return float(x_crest), float(x_toe), float(h_slope), gcfg.get('crest_window'), gcfg.get('toe_window')  # 上下文元组


def build_s_grid(a_max, c_max):  # 构造统一三段 s 子网格
    """构造统一子网格：段A [-a_max,0] 幂律加密近坡顶棱 / 段B [0,1] 均匀 / 段C [1,1+c_max] 均匀。

    拐点 s=0、s=1 在相邻两段各保留一个网格点（对齐锚点），总长恒为 N_A+N_B+N_C——
    对应研究计划"向量长度固定、拐点严格对齐"。返回 (s_grid, seg_labels)。
    """
    t = np.linspace(0.0, 1.0, N_A)  # 段A 参数坐标
    s_a = (-float(a_max) * np.power(t, CREST_REFINE_GAMMA))[::-1]  # 幂律映射后翻转为升序，近 s=0 处点距最小
    s_b = np.linspace(0.0, 1.0, N_B)  # 段B 均匀网格
    s_c = 1.0 + np.linspace(0.0, float(c_max), N_C)  # 段C 均匀网格
    seg_labels = ['A'] * N_A + ['B'] * N_B + ['C'] * N_C  # 段标签（网格点归段依据）
    return np.concatenate([s_a, s_b, s_c]), seg_labels  # 拼接网格与标签


def _seg_mask(arr, seg):  # 数组元素归段掩码
    """返回 arr 中属于段 seg 的布尔掩码；拐点 s=0/1 由相邻两段共享（保证插值触棱）。"""
    arr = np.asarray(arr, dtype=float)  # 转数组
    if seg == 'A':  # 坡顶平台
        return arr <= _SEG_EPS  # s≤0
    if seg == 'B':  # 坡面
        return (arr >= -_SEG_EPS) & (arr <= 1.0 + _SEG_EPS)  # 0≤s≤1
    return arr >= 1.0 - _SEG_EPS  # 坡脚平台 s≥1


def resample_curve(s_nodes, y_nodes, s_grid, seg_labels):  # 单条曲线三段重采样
    """把逐节点曲线按段线性插值到统一 s 子网格；段内有效点<2 时该段输出 NaN（不跨拐点借点）。"""
    s_nodes = np.asarray(s_nodes, dtype=float)  # 节点 s 坐标（升序）
    y_nodes = np.asarray(y_nodes, dtype=float)  # 节点值
    grid = np.asarray(s_grid, dtype=float)  # 网格坐标
    lab = np.array(seg_labels)  # 网格段标签
    out = np.nan * np.ones(len(grid))  # 输出初始化为 NaN
    for seg in ('A', 'B', 'C'):  # 逐段独立插值
        nm = _seg_mask(s_nodes, seg) & ~np.isnan(y_nodes)  # 段内有效节点
        if int(np.sum(nm)) < 2:  # 有效点不足
            continue  # 该段保持 NaN
        gm = lab == seg  # 网格段掩码
        out[gm] = np.interp(grid[gm], s_nodes[nm], y_nodes[nm])  # 段内线性插值（端外取端值）
    return out  # 返回对齐后曲线


def resample_H_matrix(H, s_nodes, s_grid, seg_labels):  # H 曲面空间维三段重采样
    """把 H(节点×频点) 沿空间维逐段插值到统一 s 子网格，返回 (N_A+N_B+N_C)×频点 矩阵。"""
    H = np.atleast_2d(np.asarray(H, dtype=float))  # 节点×频点
    s_nodes = np.asarray(s_nodes, dtype=float)  # 节点 s 坐标
    grid = np.asarray(s_grid, dtype=float)  # 网格坐标
    lab = np.array(seg_labels)  # 网格段标签
    out = np.nan * np.ones((len(grid), H.shape[1]))  # 输出矩阵
    for seg in ('A', 'B', 'C'):  # 逐段
        nm = _seg_mask(s_nodes, seg)  # 段内节点
        if int(np.sum(nm)) < 2:  # 节点不足
            continue  # 跳过该段
        gm = lab == seg  # 网格段掩码
        for j in range(H.shape[1]):  # 逐频点做空间插值
            out[gm, j] = np.interp(grid[gm], s_nodes[nm], H[nm, j])  # 段内线性插值
    return out  # 返回对齐曲面


def s_to_x(s, x_crest, x_toe, h_slope):  # 三段归一坐标反算物理 x
    """把 s 反算回物理 x 坐标（核查/追溯用），映射与 calc_s_coords 严格互逆。"""
    if s <= 0.0:  # 段A
        return x_crest + s * h_slope  # 距坡顶棱 |s|·h
    if s < 1.0:  # 段B
        return x_crest + s * (x_toe - x_crest)  # 坡面按水平跨度线性映射
    return x_toe + (s - 1.0) * h_slope  # 段C：距坡脚棱 (s-1)·h


def read_H_csv_local(path):  # 读回 write_H_csv 产物
    """解析 H 矩阵 csv，返回 (freqs, xs, H 节点×频点)；文件缺失或解析失败返回 (None, None, None)。"""
    if not os.path.isfile(path):  # 文件不存在（如该波未生成 H）
        return None, None, None  # 返回空
    try:  # 尝试解析
        if sys.version_info[0] >= 3:  # Python 3
            fh = open(path, 'r', encoding='utf-8-sig')  # 文本读（去 BOM）
        else:  # Python 2
            fh = open(path, 'rb')  # 二进制读
        reader = csv.reader(fh)  # 创建 reader
        header = next(reader)  # 表头：f_Hz, x=..., ...
        xs = [float(str(c).split('=')[1]) for c in header[1:]]  # 解析列头 x 坐标
        freqs, rows = [], []  # 频点与矩阵行暂存
        for row in reader:  # 逐行（行=频点）
            if not row:  # 空行
                continue  # 跳过
            freqs.append(float(row[0]))  # 频率
            rows.append([float(v) for v in row[1:]])  # 各节点值
        fh.close()  # 关闭句柄
        return np.array(freqs, dtype=float), np.array(xs, dtype=float), np.array(rows, dtype=float).T  # 转 节点×频点
    except Exception as e:  # 解析失败
        print('[sgrid] 读取 %s 失败: %s' % (path, str(e)))  # 提示
        return None, None, None  # 返回空


def write_sgrid_response_csv(path, s_grid, seg_labels, x_phys, cols):  # 写对齐响应表
    """写统一 s 子网格响应表：s,seg,x + 各重采样字段，共 N_A+N_B+N_C 行。"""
    fh = _open_csv(path)  # 打开句柄
    w = csv.writer(fh)  # 创建 writer
    w.writerow(['s', 'seg', 'x'] + SGRID_FIELDS)  # 表头
    for k in range(len(s_grid)):  # 逐网格点
        w.writerow([float(s_grid[k]), seg_labels[k], float(x_phys[k])] +
                   [float(cols[f][k]) for f in SGRID_FIELDS])  # 写一行
    fh.close()  # 关闭


def write_sgrid_H_csv(path, freqs, s_grid, seg_labels, H_s):  # 写对齐 H 曲面
    """写对齐 H 曲面 csv：首列 f(Hz)，其后每个 s 网格点一列（列头 s=值_段）。"""
    fh = _open_csv(path)  # 打开句柄
    w = csv.writer(fh)  # 创建 writer
    w.writerow(['f_Hz'] + ['s=%.5f_%s' % (float(s_grid[k]), seg_labels[k]) for k in range(len(s_grid))])  # 表头
    for i in range(len(freqs)):  # 行=频点
        w.writerow([float(freqs[i])] + [float(v) for v in H_s[:, i]])  # 写一行
    fh.close()  # 关闭


def _update_summary_ar(updates):  # 回写 AR_max 段号/归一坐标
    """把各记录 AR_max 的段号与归一坐标（重采样前原始曲线精确值）合并进 surface_summary.json。"""
    path = 'surface_summary.json'  # 摘要路径
    if not updates or not os.path.isfile(path):  # 无更新或无摘要
        return  # 跳过
    try:  # 尝试更新
        with open(path, 'r') as fh:  # 读
            data = json.load(fh)  # 解析
        for rec in data.get('records', []):  # 遍历记录
            extra = updates.get(rec.get('record'))  # 查找该记录的更新项
            if extra:  # 有更新
                rec.update(extra)  # 合并字段
        with open(path, 'w') as fh:  # 写回
            json.dump(data, fh, indent=2)  # 保存
        print('[sgrid] AR_max 段号/归一坐标已回写 surface_summary.json')  # 提示
    except Exception as e:  # 失败
        print('[sgrid] 回写 surface_summary.json 失败: %s' % str(e))  # 提示


def resample_outputs(meta, case_cfg):  # 重采样主控制函数
    """研究计划 §4.0 第②步：把逐节点曲线与 H(f,s) 曲面插值到统一三段 s 子网格并落盘。

    输入：当前目录 surface_response_<record>.csv 与 H_surface_h/v、H_topo_h_<record>.csv（第①步产物）。
    输出：sgrid_response_<record>.csv、sgrid_H_*_<record>.csv、sgrid_params.json。
    同时把 AR_max 段号/归一坐标（重采样前原始曲线上精确取值）回写 surface_summary.json。
    普通 Python 即可运行（不依赖 odbAccess），可对已有 CSV 反复重跑。
    """
    ctx = _resolve_s_context(meta, case_cfg)  # 解析几何上下文
    if ctx is None:  # 上下文不完整
        print('[sgrid] 警告: case_meta 缺拐点或坡高（x_crest/x_toe 或 H_minus_h/w_slope·tan(i)），跳过重采样。')  # 警告
        return  # 退出
    x_crest, x_toe, h_slope, a_win, c_win = ctx  # 解包几何上下文
    csvs = sorted(glob.glob('surface_response_*.csv'))  # 第①步响应表
    if not csvs:  # 无输入
        print('[sgrid] 未发现 surface_response_*.csv，跳过重采样。')  # 提示
        return  # 退出
    if not (a_win and c_win):  # 未配置观测窗口
        print('[sgrid] 警告: geometry_cfg 未配置 crest_window/toe_window，网格范围退化为本工况数据范围，跨工况将不可比！')  # 警告
    updates = {}  # summary 回写暂存
    grid_written = False  # 网格参数是否已写盘
    for csv_path in csvs:  # 逐记录处理
        record = csv_path[len('surface_response_'):-4]  # 提取记录名
        data = read_response_csv_local(csv_path)  # 读响应表
        if not data:  # 空表
            continue  # 跳过
        xs = np.array([row['x'] for row in data], dtype=float)  # 节点 x 坐标
        s_nodes = calc_s_coords(xs, x_crest, x_toe, h_slope)  # 节点三段归一坐标
        a_max = float(a_win) if (a_win and float(a_win) > 0) else max(float(-s_nodes.min()), 0.5)  # 段A 网格跨度
        c_max = float(c_win) if (c_win and float(c_win) > 0) else max(float(s_nodes.max()) - 1.0, 0.5)  # 段C 网格跨度
        s_grid, seg_labels = build_s_grid(a_max, c_max)  # 统一三段子网格
        cols = {}  # 重采样结果列
        for f in SGRID_FIELDS:  # 逐字段重采样
            y = np.array([row.get(f, float('nan')) for row in data], dtype=float)  # 原始逐节点列
            cols[f] = resample_curve(s_nodes, y, s_grid, seg_labels)  # 段内插值对齐
        x_phys = np.array([s_to_x(float(sv), x_crest, x_toe, h_slope) for sv in s_grid])  # 网格点反算物理坐标
        write_sgrid_response_csv('sgrid_response_%s.csv' % record, s_grid, seg_labels, x_phys, cols)  # 落盘
        taf = np.array([row.get('TAF_h', float('nan')) for row in data], dtype=float)  # 原始 TAF_h（重采样前）
        if np.any(~np.isnan(taf)):  # 有有效值
            k = int(np.nanargmax(taf))  # 峰值下标（原始节点，精确不挪峰）
            sk = float(s_nodes[k])  # 峰值归一坐标
            seg = 'A' if sk < 0.0 else ('B' if sk <= 1.0 else 'C')  # 峰值所在段（拐点归坡面口径）
            updates[record] = {'AR_max_s': sk, 'AR_max_seg': seg}  # 暂存回写项
        n_h = 0  # H 曲面成功计数
        for src_fmt, dst_fmt in (('H_surface_h_%s.csv', 'sgrid_H_surface_h_%s.csv'),
                                 ('H_surface_v_%s.csv', 'sgrid_H_surface_v_%s.csv'),
                                 ('H_topo_h_%s.csv', 'sgrid_H_topo_h_%s.csv')):  # 三类曲面同口径处理
            freqs, xs_h, H = read_H_csv_local(src_fmt % record)  # 读回第①步矩阵
            if freqs is None:  # 缺失（如该波无输入波文件未生成 H）
                continue  # 跳过
            s_h = calc_s_coords(xs_h, x_crest, x_toe, h_slope)  # H 列坐标转 s
            H_s = resample_H_matrix(H, s_h, s_grid, seg_labels)  # 空间维对齐
            write_sgrid_H_csv(dst_fmt % record, freqs, s_grid, seg_labels, H_s)  # 落盘
            n_h += 1  # 计数
        if not grid_written:  # 网格参数只写一份（对本工况所有记录一致）
            with open('sgrid_params.json', 'w') as fh:  # 写参数文件
                json.dump({'schema_version': 1, 'N_A': N_A, 'N_B': N_B, 'N_C': N_C,
                           'A_max': a_max, 'C_max': c_max, 'crest_refine_gamma': CREST_REFINE_GAMMA,
                           'h_slope': h_slope,
                           'note': u'研究计划§4.0第②步统一s子网格；段A幂律加密近坡顶棱；拐点s=0/1在相邻段各留一点'},
                          fh, indent=2)  # 保存（ensure_ascii 默认开，Py2 安全）
            grid_written = True  # 标记已写
        print('[sgrid] %s: 重采样完成 -> sgrid_response（%d 点/段A %d+段B %d+段C %d） + %d 个 H 曲面' % (
            record, len(s_grid), N_A, N_B, N_C, n_h))  # 提示
    _update_summary_ar(updates)  # 回写 AR_max 段号/归一坐标


def main():  # 主入口函数
    """后处理脚本控制流。"""
    meta = _load_json('case_meta.json')  # 读取元数据
    case_cfg = _load_json('case_config.json')  # 读取配置

    if openOdb is None:  # 无 Abaqus 环境
        print('提示: 未检测到 odbAccess (非 Abaqus 环境)，跳过 ODB 提取，直接重采样并重绘图表。')  # 提示
        resample_outputs(meta, case_cfg)  # §4.0 第②步重采样（可对已有 CSV 反复重跑）
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

    resample_outputs(meta, case_cfg)  # §4.0 第②步重采样（并回写 AR_max 段号/归一坐标）
    plot_results(meta, case_cfg)  # 提取数据后自动画图


if __name__ == '__main__':  # 程序入口
    main()  # 执行主流程
