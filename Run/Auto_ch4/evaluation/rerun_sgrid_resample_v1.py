# -*- coding: utf-8 -*-
"""重跑 sgrid resample：对 surface_results.npz 的复频响 sgrid 字段
启用 fill_short_gaps 重算，修复坡脚 s∈[0.95,1.10] 段界死区。

无需 Abaqus，直接从已存的节点级 H_surface_* + raw_x + case_meta 重算。
原文件备份为 surface_results.npz.bak。

根因：Postprocess v2 旧版对 H_surface_over_1D_h/H_station_h 未启用 fill_short_gaps，
坡脚棱节点 s 因网格离散略偏离 1.0，_SEG_EPS=1e-9 太小未纳入 C 段源点，
C 段源点起点 > 1.0 使 s=1.00 及右侧短缺口 np.interp(left=nan) 全 NaN。
修复：对全部 frf 复数场启用 fill_short_gaps，补 ≤0.15 的段界短缺口。

用法：
  python rerun_sgrid_resample_v1.py <工况目录>            # 单工况验证
  python rerun_sgrid_resample_v1.py --root <根目录>        # 批量遍历 case-*
  python rerun_sgrid_resample_v1.py                        # 默认批量
"""
from __future__ import print_function

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

_SEG_EPS = 1e-9  # 拐点归段容差（与 Postprocess v2 一致）
SGRID_MAX_GAPFILL = 0.15  # 短内部缺口填补上限（与 Postprocess v2 一致）


def _seg_mask(arr, seg):
    """段内节点掩码；拐点 s=0/1 由相邻两段共享。"""
    arr = np.asarray(arr, dtype=float)
    if seg == 'A':
        return arr <= _SEG_EPS
    if seg == 'B':
        return (arr >= -_SEG_EPS) & (arr <= 1.0 + _SEG_EPS)
    return arr >= 1.0 - _SEG_EPS


def calc_s_coords(xs, x_crest, x_toe, h_ref):
    """三段归一化坐标 s（与 Postprocess v2 calc_s_coords 严格一致）。"""
    xs = np.asarray(xs, dtype=float)
    s = np.zeros_like(xs)
    if h_ref is None or h_ref <= 0:
        h_ref = 1.0
    w_slope = x_toe - x_crest
    if w_slope <= 0:
        w_slope = 1.0
    a = xs <= x_crest + 1e-4
    b = (xs > x_crest + 1e-4) & (xs <= x_toe + 1e-4)
    c = xs > x_toe + 1e-4
    s[a] = (xs[a] - x_crest) / h_ref
    s[b] = (xs[b] - x_crest) / w_slope
    s[c] = 1.0 + (xs[c] - x_toe) / h_ref
    return s


def fill_short_internal_gaps(out, s_nodes, y_nodes, s_grid, max_gap=SGRID_MAX_GAPFILL):
    """用两侧真实有效节点线性补齐短内部缺口；不补端外，不跨过宽缺口。"""
    out = np.asarray(out, dtype=float).copy()
    s_nodes = np.asarray(s_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)
    grid = np.asarray(s_grid, dtype=float)
    ok = ~np.isnan(s_nodes) & ~np.isnan(y_nodes)
    if int(np.sum(ok)) < 2:
        return out
    order = np.argsort(s_nodes[ok])
    sx = s_nodes[ok][order]
    yy = y_nodes[ok][order]
    interp_all = np.interp(grid, sx, yy, left=np.nan, right=np.nan)
    fill = np.isnan(out) & ~np.isnan(interp_all)
    for i in np.where(fill)[0]:
        left = np.searchsorted(sx, grid[i], side='right') - 1
        right = left + 1
        if left >= 0 and right < len(sx) and (sx[right] - sx[left]) <= float(max_gap):
            out[i] = interp_all[i]
    return out


def fill_short_internal_gaps_matrix(out, H, s_nodes, s_grid, max_gap=SGRID_MAX_GAPFILL):
    """逐频补齐复数谱曲面在几何棱附近的短缺口；实部虚部须同时可补才填。"""
    result = np.asarray(out).copy()
    source = np.atleast_2d(np.asarray(H))
    is_complex = np.iscomplexobj(source)
    for j in range(source.shape[1]):
        if is_complex:
            real = fill_short_internal_gaps(result[:, j].real, s_nodes, source[:, j].real, s_grid, max_gap)
            imag = fill_short_internal_gaps(result[:, j].imag, s_nodes, source[:, j].imag, s_grid, max_gap)
            missing = ~np.isfinite(result[:, j].real) | ~np.isfinite(result[:, j].imag)
            fill = missing & np.isfinite(real) & np.isfinite(imag)
            result[fill, j] = real[fill] + 1j * imag[fill]
        else:
            result[:, j] = fill_short_internal_gaps(result[:, j], s_nodes, source[:, j], s_grid, max_gap)
    return result


def resample_H_matrix(H, s_nodes, s_grid, seg_labels, fill_short_gaps=False):
    """H 曲面空间维三段重采样（与 Postprocess v2 一致，fill_short_gaps 可控）。"""
    H = np.atleast_2d(np.asarray(H))
    is_complex = np.iscomplexobj(H)
    s_nodes = np.asarray(s_nodes, dtype=float)
    grid = np.asarray(s_grid, dtype=float)
    lab = np.array(seg_labels)
    if is_complex:
        out = np.empty((len(grid), H.shape[1]), dtype=np.complex128)
        out[:] = complex(float('nan'), float('nan'))
    else:
        H = np.asarray(H, dtype=float)
        out = np.nan * np.ones((len(grid), H.shape[1]))
    for seg in ('A', 'B', 'C'):
        seg_nodes = _seg_mask(s_nodes, seg)
        gm = lab == seg
        for j in range(H.shape[1]):
            if is_complex:
                finite = np.isfinite(H[:, j].real) & np.isfinite(H[:, j].imag)
            else:
                finite = np.isfinite(H[:, j])
            nm = seg_nodes & finite
            if int(np.sum(nm)) < 2:
                continue
            if is_complex:
                real = np.interp(grid[gm], s_nodes[nm], H[nm, j].real, left=np.nan, right=np.nan)
                imag = np.interp(grid[gm], s_nodes[nm], H[nm, j].imag, left=np.nan, right=np.nan)
                out[gm, j] = real + 1j * imag
            else:
                out[gm, j] = np.interp(grid[gm], s_nodes[nm], H[nm, j], left=np.nan, right=np.nan)
    if fill_short_gaps:
        out = fill_short_internal_gaps_matrix(out, H, s_nodes, s_grid)
    return out


def _decode_bytes(value):
    """npz 标量 bytes/0-d 数组统一还原为 Python 对象。"""
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return value


def discover_record(package):
    """从 keys 推断 frf_<record>_ 前缀的 record 名。"""
    for k in package.keys():
        if k.startswith('frf_') and k.endswith('_sgrid_s'):
            return k[4:-8]  # frf_<record>_sgrid_s -> <record>（去掉前后缀各 frf_ 与 _sgrid_s 共 12 字符）
    return None


def rerun_one(npz_path, dry_run=False):
    """重算单个 surface_results.npz 的 4 个 frf sgrid 字段。"""
    package = np.load(npz_path, allow_pickle=True)
    try:
        record = discover_record(package)
        if record is None:
            return {'path': str(npz_path), 'status': 'no_record'}
        prefix = 'frf_%s_' % record

        cm = _decode_bytes(package['case_meta_json'])
        cm = json.loads(cm)
        geo = cm.get('geometry', {})
        x_crest = float(geo['x_crest'])
        x_toe = float(geo['x_toe'])

        sp = _decode_bytes(package['sgrid_params_json'])
        sp = json.loads(sp)
        h_ref = float(sp.get('h_slope', 100.0))

        raw_x_key = 'raw_%s_x' % record
        if raw_x_key not in package:
            return {'path': str(npz_path), 'status': 'no_raw_x'}
        raw_x = np.asarray(package[raw_x_key], dtype=float)
        s_nodes = calc_s_coords(raw_x, x_crest, x_toe, h_ref)

        sgrid_s = np.asarray(package[prefix + 'sgrid_s'], dtype=float)
        seg_labels = []
        for sv in sgrid_s:
            if sv <= 0.0:
                seg_labels.append('A')
            elif sv <= 1.0 + 1e-9:
                seg_labels.append('B')
            else:
                seg_labels.append('C')
        seg_labels = np.array(seg_labels)

        # 修复前 s∈[0.95,1.10] 有效统计（取 H_surface_over_1D_h 为代表）
        sg_key_g = prefix + 'sgrid_H_surface_over_1D_h'
        before = {}
        if sg_key_g in package:
            before_arr = np.asarray(package[sg_key_g])
            before_mask = (sgrid_s >= 0.90) & (sgrid_s <= 1.15)
            before = {round(float(sgrid_s[i]), 2): int(np.sum(np.isfinite(before_arr[i, :].real)))
                      for i in np.where(before_mask)[0]}

        # 重算可用的 frf sgrid 字段（全启用 fill_short_gaps）
        changes = {}
        for field in ('H_surface_h', 'H_surface_v', 'H_surface_over_1D_h',
                      'H_surface_over_1D_left_h', 'H_station_h'):
            node_key = prefix + field
            if node_key not in package:
                continue
            H = np.asarray(package[node_key])
            aligned = resample_H_matrix(H, s_nodes, sgrid_s, seg_labels, fill_short_gaps=True)
            sg_key = prefix + 'sgrid_' + field
            vm_key = sg_key + '_valid_mask'
            changes[sg_key] = aligned
            if np.iscomplexobj(aligned):
                changes[vm_key] = np.isfinite(aligned.real) & np.isfinite(aligned.imag)
        changes[prefix + 'sgrid_segment'] = np.asarray([s.encode('ascii') for s in seg_labels])
        changes = {k: v for k, v in changes.items()}
        if sg_key_g in changes:
            after_arr = changes[sg_key_g]
            after_mask = (sgrid_s >= 0.90) & (sgrid_s <= 1.15)
            after = {round(float(sgrid_s[i]), 2): int(np.sum(np.isfinite(after_arr[i, :].real)))
                     for i in np.where(after_mask)[0]}

        if dry_run:
            return {'path': str(npz_path), 'record': record, 'before': before, 'after': after, 'status': 'dry_run'}

        # 读全部数组到 dict，更新目标字段，备份后写回
        data = {k: package[k] for k in package.keys()}
        data.update(changes)
        bak = npz_path.with_suffix('.npz.bak')
        if not bak.exists():
            shutil.copy2(npz_path, bak)
        np.savez(npz_path, **data)
        return {'path': str(npz_path), 'record': record, 'before': before, 'after': after, 'status': 'ok'}
    finally:
        package.close()


def find_case_dirs(roots):
    """收集所有含 surface_results.npz 的工况目录。"""
    result = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        if (root / 'surface_results.npz').is_file():
            result.append(root)
            continue
        result.extend(sorted(p for p in root.glob('case-*') if (p / 'surface_results.npz').is_file()))
    return sorted(set(Path(r).resolve() for r in result), key=lambda p: p.name)


def main(argv=None):
    parser = argparse.ArgumentParser(description='重跑 sgrid resample 修复坡脚段界死区')
    parser.add_argument('case_dir', nargs='?', help='单工况目录（验证用）')
    parser.add_argument('--root', action='append', default=None, help='批量根目录（可多次）')
    parser.add_argument('--dry-run', action='store_true', help='只统计不写回')
    args = parser.parse_args(argv)

    if args.case_dir:
        roots = [args.case_dir]
    elif args.root:
        roots = args.root
    else:
        repo = Path(__file__).resolve().parents[3]  # AbqScripts/Run/Auto_ch4/evaluation -> 仓库根
        roots = [repo / 'Run' / sub for sub in
                 ('ch4_sp_01_V', 'ch4_sp_02_H', 'ch4_sp_03_P', 'ch4_sp_04_B', 'ch4_sp_05_C')]

    case_dirs = find_case_dirs(roots)
    if not case_dirs:
        print('未发现含 surface_results.npz 的工况目录')
        return 1
    print('发现 %d 个工况' % len(case_dirs))
    fixed = 0
    for i, cd in enumerate(case_dirs, 1):
        npz = cd / 'surface_results.npz'
        try:
            r = rerun_one(npz, dry_run=args.dry_run)
        except Exception as e:
            print('[%d/%d] %s -> ERROR: %s' % (i, len(case_dirs), cd.name, e))
            continue
        if r['status'] in ('ok', 'dry_run'):
            fixed += 1
            before = r.get('before', {})
            after = r.get('after', {})
            # 只打印坡脚关键点变化
            focus = [0.95, 1.0, 1.05, 1.1]
            bf = ' '.join('%s:%d' % (s, before.get(s, -1)) for s in focus if s in before)
            af = ' '.join('%s:%d' % (s, after.get(s, -1)) for s in focus if s in after)
            print('[%d/%d] %s %s | before[%s] after[%s]' % (i, len(case_dirs), cd.name, r['status'], bf, af))
        else:
            print('[%d/%d] %s -> %s' % (i, len(case_dirs), cd.name, r['status']))
    print('完成：修复 %d/%d' % (fixed, len(case_dirs)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
