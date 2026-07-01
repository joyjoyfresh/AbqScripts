# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软土非线性·强度扫描【汇总】(纯 Python，无需 Abaqus)。

读取 multi-lin-* / multi-nlin-* 工况，按输入强度(PGA)列出：
  软层等效线性结果(γ_eff / G/Gmax / 有效Vs / ξ) + 远场TAF + 坡顶TAF + 相对线性的放大变化。
并自动判读非线性随强度的抑制/重分布趋势。

用法：python softsoil_nonlinear_report_v1.py <强度扫描工况根目录>
"""

import os  # 路径
import sys  # 参数
import csv  # 读 TAF csv
import json  # 读 case_meta
import glob  # 找工况文件夹


def _read_taf(folder):  # 读 TAF csv -> (xs, taf_h)
    hits = glob.glob(os.path.join(folder, 'TAF-*.csv'))
    if not hits:
        return None, None
    xs = []; th = []
    with open(hits[0], encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            try:
                xs.append(float(row['x'])); th.append(float(row['TAF_h']))
            except (KeyError, ValueError):
                continue
    return (xs, th) if xs else (None, None)


def _pga_from_name(name):  # multi-nlin-0p8g -> 0.8 ; lin-ref -> None
    for tok in name.split('-'):
        if tok.endswith('g') and ('p' in tok or tok[:-1].isdigit()):
            try:
                return float(tok[:-1].replace('p', '.'))
            except ValueError:
                pass
    return None


def _metrics(xs, th, x_crest):  # 远场/坡顶峰
    far = [th[i] for i in range(len(xs)) if xs[i] < 200.0]
    far = sum(far) / len(far) if far else float('nan')
    win = [th[i] for i in range(len(xs)) if (x_crest - 90.0) <= xs[i] <= (x_crest + 30.0)]
    peak = max(win) if win else float('nan')
    return far, peak


def _eql_soft(meta):  # 取软层 EQL 结果 (geff, GG, Vs, xi)；线性返回 None
    eql = meta.get('eql') or {}
    if not eql.get('enable'):
        return None
    layers = eql.get('layers') or {}
    s = layers.get('surface') or (list(layers.values())[0] if layers else None)
    return s


def main():
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()
    folders = sorted(glob.glob(os.path.join(root, 'multi-lin-*')) + glob.glob(os.path.join(root, 'multi-nlin-*')))
    L = ['================= 软土非线性·强度扫描 汇总 =================', '根目录: %s' % root]
    if not folders:
        L.append('未找到 multi-lin-*/multi-nlin-* 工况(请确认强度扫描批处理已跑完并指向本目录)。')
        _emit(L, root); return

    # 先取线性参照的坡顶 TAF
    lin_peak = None
    rows = []  # (pga, name, soft, far, peak, is_lin)
    for fd in folders:
        name = os.path.basename(fd)
        meta_p = os.path.join(fd, 'case_meta.json')
        xs, th = _read_taf(fd)
        if not os.path.isfile(meta_p) or xs is None:
            L.append('%-20s [缺 case_meta/TAF，跳过]' % name); continue
        meta = json.load(open(meta_p, encoding='utf-8'))
        x_crest = meta.get('geometry', {}).get('x_crest', 1000.0)
        far, peak = _metrics(xs, th, x_crest)
        is_lin = name.startswith('multi-lin')
        soft = _eql_soft(meta)
        pga = _pga_from_name(name)
        if is_lin:
            lin_peak = peak
        rows.append((pga if pga else -1.0, name, soft, far, peak, is_lin))

    rows.sort(key=lambda r: (not r[5], r[0]))  # 线性在前，其余按 PGA 升序
    L.append('%-12s %-7s %-9s %-8s %-8s %-7s %-8s %-8s %-8s' % (
        '工况', 'PGA', 'γ_eff%', 'G/Gmax', '有效Vs', 'ξ%', '远场TAF', '坡顶TAF', '相对线性'))
    L.append('-' * 92)
    for (pga, name, soft, far, peak, is_lin) in rows:
        tag = name.replace('multi-', '')
        pgastr = '线性' if is_lin else ('%.2fg' % pga if pga > 0 else '?')
        if soft:
            geff = '%.3f' % (soft.get('geff', 0) * 100); gg = '%.3f' % soft.get('GG', float('nan'))
            vs = '%.0f' % soft.get('Vs', float('nan')); xi = '%.1f' % (soft.get('xi', 0) * 100)
        else:
            geff = '-'; gg = '1.000'; vs = '400'; xi = '~2'
        rel = '' if is_lin else (('%.0f%%' % (peak / lin_peak * 100)) if lin_peak else '')
        L.append('%-12s %-7s %-9s %-8s %-8s %-7s %-8.3f %-8.3f %-8s' % (
            tag, pgastr, geff, gg, vs, xi, far, peak, rel))
    L.append('-' * 92)

    # 趋势判读
    nlin = [(r[0], r[4]) for r in rows if not r[5] and r[0] > 0]
    nlin.sort()
    L.append('')
    if lin_peak and len(nlin) >= 2:
        lo_pga, lo_pk = nlin[0]; hi_pga, hi_pk = nlin[-1]
        drop = (1 - hi_pk / lin_peak) * 100.0
        L.append('趋势判读（核心成果）：')
        L.append('  线性坡顶 TAF = %.3f（与强度无关，作参照）。' % lin_peak)
        L.append('  非线性：%.2fg 时坡顶 TAF=%.3f（≈线性），%.2fg 时降到 %.3f（较线性 -%.0f%%）。'
                 % (lo_pga, lo_pk, hi_pga, hi_pk, drop))
        if drop >= 8:
            L.append('  → 随强度增大，软层 G/Gmax↓、ξ↑、共振下移，坡顶放大被【非线性抑制】，且越强抑制越明显。')
            L.append('     这条"放大随强度下降"的曲线即课题核心；它独立于 2D 俘获绝对值，是论文(纯线性)未覆盖的新结果。')
        else:
            L.append('  → 抑制不明显，可能输入强度偏低/曲线偏硬：提高 PGA 上限，或换更软的模量折减曲线(vucetic_dobry)再扫。')
    _emit(L, root)


def _emit(L, root):
    text = '\n'.join(L); print(text)
    try:
        with open(os.path.join(root, 'softsoil_nonlinear_report.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        print('\n已写出 %s' % os.path.join(root, 'softsoil_nonlinear_report.txt'))
    except OSError as e:
        print('写报告失败: %s' % e)


if __name__ == '__main__':
    main()
