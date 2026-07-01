# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软楔【持续正弦(谐波)】实验汇总(纯 Python，无需 Abaqus)。

读取 multi-harm-* 工况，按频率列出 远场TAF / 坡顶峰TAF / 几何倍率(坡顶/远场)，
并与瞬态 Ricker 的坡顶峰 ~2.0 对比，自动判读：
  软楔模态频率(2.91/3.84Hz)的坡顶稳态峰若明显 > 2.0(如 >3) → 瞬态激不满共振，论文 7.6 更可疑；
  若都仍 ~2.0 → 稳态也上不去，属方法层面差异(等效力+VAB vs SPECFEM)。

用法：python softwedge_harmonic_report_v1.py <谐波工况根目录>
"""

import os  # 路径
import sys  # 参数
import csv  # 读 TAF csv
import json  # 读 case_meta
import glob  # 找工况文件夹

TRANSIENT_PEAK = 2.0  # 瞬态 Ricker 坡顶峰参照值(收敛研究结论 ~2.0)


def _read_taf(folder):  # 读 TAF csv -> (xs, taf_h)
    """找 TAF-*.csv 读地表 x 与水平 TAF；缺失返回 (None,None)。"""
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


def _freq_from_name(name):  # 从 multi-harm-sine3p84Hz 提取频率(Hz)
    """解析文件夹名里的正弦频率(p 代表小数点)；解析不到返回 None。"""
    for tok in name.split('-'):
        if tok.lower().startswith('sine'):
            s = tok[4:].replace('Hz', '').replace('hz', '').replace('p', '.')
            try:
                return float(s)
            except ValueError:
                return None
    return None


def _metrics(xs, th, x_crest):  # 远场/坡顶峰
    """远场=x<200 段均值；坡顶峰=坡顶邻域(-90~+30) TAF_h 最大。"""
    far = [th[i] for i in range(len(xs)) if xs[i] < 200.0]
    far = sum(far) / len(far) if far else float('nan')
    win = [th[i] for i in range(len(xs)) if (x_crest - 90.0) <= xs[i] <= (x_crest + 30.0)]
    peak = max(win) if win else float('nan')
    return far, peak


def main():
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()
    folders = sorted(glob.glob(os.path.join(root, 'multi-harm-*')))
    lines = ['================= 软楔 持续正弦(谐波) 实验汇总 =================', '根目录: %s' % root,
             '(参照: 瞬态 Ricker 坡顶峰 ≈ %.1f)' % TRANSIENT_PEAK]
    if not folders:
        lines.append('未找到 multi-harm-* 工况(请确认谐波批处理已跑完并指向本目录)。')
        _emit(lines, root); return
    lines.append('%-18s %-9s %-9s %-9s %-9s' % ('频率', '远场TAF', '坡顶峰', '几何倍率', '相对瞬态'))
    lines.append('-' * 60)
    peaks = []  # (freq, peak)
    for fd in folders:
        name = os.path.basename(fd)
        freq = _freq_from_name(name)
        meta_p = os.path.join(fd, 'case_meta.json')
        xs, th = _read_taf(fd)
        if not os.path.isfile(meta_p) or xs is None:
            lines.append('%-18s [缺 case_meta/TAF，跳过]' % name); continue
        meta = json.load(open(meta_p, encoding='utf-8'))
        x_crest = meta.get('geometry', {}).get('x_crest', 1000.0)
        far, peak = _metrics(xs, th, x_crest)
        ratio = peak / far if far and far == far else float('nan')
        rel = peak / TRANSIENT_PEAK if peak == peak else float('nan')
        flabel = ('%.2fHz' % freq) if freq else name.replace('multi-harm-', '')
        lines.append('%-18s %-9.3f %-9.3f %-9.3f %-9s' % (
            flabel, far, peak, ratio, ('x%.2f' % rel)))
        if peak == peak:
            peaks.append((freq if freq else 0.0, peak))
    lines.append('-' * 60)
    # 判读：看软楔模态频率(>2.5Hz)的稳态坡顶峰
    res_peaks = [pk for (f, pk) in peaks if f and f >= 2.5]  # 软楔 2D 模态频率段
    maxpk = max([pk for _, pk in peaks]) if peaks else float('nan')
    lines.append('')
    if res_peaks and max(res_peaks) >= 3.0:
        lines.append('判读：软楔模态频率上坡顶【稳态】峰达 %.2f (>3, 明显>瞬态 %.1f)' % (max(res_peaks), TRANSIENT_PEAK))
        lines.append('  → 共振是真的、只是短脉冲 Ricker 激不满；那么论文的 Ricker 也应 ~2，其 7.6 更可疑。')
    elif maxpk == maxpk:
        lines.append('判读：所有频率(含软楔模态)坡顶【稳态】峰最高仅 %.2f，与瞬态 %.1f 相当。' % (maxpk, TRANSIENT_PEAK))
        lines.append('  → 稳态也上不去，属方法层面差异(等效力+VAB vs SPECFEM)，非"瞬态激不满"。')
        lines.append('     这进一步坐实：本方法下软表层放大 ~2.0 是收敛且物理的，论文 7.6 须谨慎。')
    _emit(lines, root)


def _emit(lines, root):
    text = '\n'.join(lines); print(text)
    try:
        with open(os.path.join(root, 'softwedge_harmonic_report.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        print('\n已写出 %s' % os.path.join(root, 'softwedge_harmonic_report.txt'))
    except OSError as e:
        print('写报告失败: %s' % e)


if __name__ == '__main__':
    main()
