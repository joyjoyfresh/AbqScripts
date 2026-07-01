# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软楔研究【结果汇总】脚本(纯 Python，无需 Abaqus)。

读取收敛批处理产出的各 multi-conv-* 工况，输出一张对照表 + 两个判据，并回显模态结论：
  对照表：每工况的 网格/单元/尾段、远场 TAF、坡顶峰 TAF、几何倍率(坡顶峰/一维理论)；
  验证闸门：远场 FE TAF 是否≈一维理论(case_meta.ff_theory)，<5% 才说明(尤其 CPE8R)边界权重对、结果可信；
  关键对照：CPE4R@4m vs CPE8R@4m(同网格)、以及网格加密趋势，看坡顶峰是否被数值频散压低。

用法：python softwedge_report_v1.py <收敛工况根目录>
输出：屏幕打印 + 在根目录写 softwedge_report.txt；若同目录有 eigen_softwedge_summary.txt 则附其结论。
"""

import os  # 路径
import sys  # 命令行参数
import csv  # 读 TAF csv
import json  # 读 case_meta
import glob  # 找工况文件夹


def _read_taf(folder):  # 读该工况 TAF csv，返回 (xs, taf_h)
    """找 TAF-*.csv 读取地表 x 与水平 TAF；缺失返回 (None, None)。"""
    hits = glob.glob(os.path.join(folder, 'TAF-*.csv'))  # TAF 文件
    if not hits:
        return None, None
    xs = []; th = []
    with open(hits[0], encoding='utf-8-sig') as f:  # utf-8-sig 兼容 BOM
        for row in csv.DictReader(f):
            try:
                xs.append(float(row['x'])); th.append(float(row['TAF_h']))
            except (KeyError, ValueError):
                continue
    return (xs, th) if xs else (None, None)


def _parse_name(folder_name):  # 从文件夹名解析 网格/单元/尾段(仅用于显示)
    """multi-conv-soft-m4-CPE4R-tail4 -> ('4','CPE4R','4')；解析不到留 '?'。"""
    size = elem = tail = '?'
    for tok in folder_name.split('-'):
        if tok.startswith('m') and tok[1:].replace('.', '').isdigit():
            size = tok[1:]
        elif tok.upper().startswith('CPE'):
            elem = tok.upper()
        elif tok.startswith('tail'):
            tail = tok[4:]
    return size, elem, tail


def _metrics(xs, th, x_crest):  # 由 TAF 曲线算 远场/坡顶峰
    """远场=x<200 段均值；坡顶峰=坡顶 x_crest 邻域(-90~+30)内 TAF_h 最大值。"""
    far = [th[i] for i in range(len(xs)) if xs[i] < 200.0]  # 远场段(平台远端)
    far = sum(far) / len(far) if far else float('nan')  # 远场均值
    win = [th[i] for i in range(len(xs)) if (x_crest - 90.0) <= xs[i] <= (x_crest + 30.0)]  # 坡顶邻域
    peak = max(win) if win else float('nan')  # 坡顶峰
    return far, peak


def main():
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()  # 收敛工况根目录
    folders = sorted(glob.glob(os.path.join(root, 'multi-conv-*')))  # 各收敛工况文件夹
    lines = []  # 报告文本行
    lines.append('================= 软楔收敛研究 汇总 =================')
    lines.append('根目录: %s' % root)
    if not folders:
        lines.append('未找到 multi-conv-* 工况文件夹(请确认收敛批处理已跑完并指向本目录)。')
        _emit(lines, root); return

    lines.append('%-26s %-5s %-7s %-5s %-9s %-9s %-9s %-8s' % (
        '工况', '网格', '单元', '尾段', '远场TAF', '一维理论', '坡顶峰', '几何倍率'))
    lines.append('-' * 92)
    gate_fail = []  # 验证闸门未过的工况
    table = {}  # name -> (elem, size, peak, ratio) 供关键对照
    for fd in folders:
        name = os.path.basename(fd)
        size, elem, tail = _parse_name(name)
        meta_p = os.path.join(fd, 'case_meta.json')
        if not os.path.isfile(meta_p):
            lines.append('%-26s [缺 case_meta.json，跳过]' % name); continue
        meta = json.load(open(meta_p, encoding='utf-8'))
        x_crest = meta.get('geometry', {}).get('x_crest', 1000.0)  # 坡顶 x
        fd1d = meta.get('ff_theory', {}).get('left', {}).get('taf_h', float('nan'))  # 一维理论(上平台柱)
        xs, th = _read_taf(fd)
        if xs is None:
            lines.append('%-26s [缺 TAF csv，跳过]' % name); continue
        far, peak = _metrics(xs, th, x_crest)
        ratio = peak / fd1d if fd1d and fd1d == fd1d else float('nan')  # 几何倍率=坡顶峰/一维理论
        # 验证闸门：远场 FE 应≈一维理论
        gate = abs(far - fd1d) / fd1d * 100.0 if fd1d else float('nan')  # 偏差%
        flag = '' if (gate == gate and gate <= 5.0) else '  <- 闸门>5%!'
        if gate == gate and gate > 5.0:
            gate_fail.append((name, gate))
        lines.append('%-26s %-5s %-7s %-5s %-9.3f %-9.3f %-9.3f %-8.3f%s' % (
            name.replace('multi-conv-', ''), size, elem, tail, far, fd1d, peak, ratio, flag))
        table[name] = (elem, size, peak, ratio)

    lines.append('-' * 92)
    # 验证闸门小结
    if gate_fail:
        lines.append('!! 验证闸门未过(远场 FE 偏离一维理论>5%)的工况:')
        for nm, g in gate_fail:
            lines.append('   %s : 偏差 %.1f%%' % (nm, g))
        lines.append('   → 这些工况(尤其 CPE8R)的边界权重/网格可能有问题，其坡顶峰对照先别采信。')
    else:
        lines.append('验证闸门：全部工况远场 FE ≈ 一维理论(<5%) → 含 CPE8R 在内结果口径可信。')

    # 关键对照：CPE4R@4m vs CPE8R@4m(同网格)
    def _find(elem, size):
        for nm, (e, s, pk, r) in table.items():
            if e == elem and s == size:
                return pk, r
        return None
    a = _find('CPE4R', '4'); b = _find('CPE8R', '4')
    lines.append('')
    lines.append('关键对照(同 4m 网格，线性 vs 二次)：')
    if a and b:
        lines.append('   CPE4R@4m 坡顶峰=%.3f(倍率%.3f)  →  CPE8R@4m 坡顶峰=%.3f(倍率%.3f)' % (a[0], a[1], b[0], b[1]))
        dp = (b[0] - a[0]) / a[0] * 100.0 if a[0] else float('nan')
        if dp == dp and dp > 5.0:
            lines.append('   升阶后坡顶峰 +%.1f%% → 数值频散确在压软楔共振，加密/升阶可救(可再看 CPE8R@2m 是否还涨)。' % dp)
        else:
            lines.append('   升阶后坡顶峰几乎不变(%.1f%%) → 频散不是主因，指向方法天花板/对论文 7.6 的质疑成立。' % (dp if dp == dp else 0.0))
    else:
        lines.append('   (CPE4R@4m 或 CPE8R@4m 工况缺失，无法对照)')

    # 附模态结论
    eig = os.path.join(root, 'eigen_softwedge_summary.txt')
    if os.path.isfile(eig):
        lines.append('')
        lines.append('===== 附：软楔模态提取结论(eigen_softwedge_summary.txt) =====')
        for ln in open(eig, encoding='utf-8').read().splitlines():
            if ln.strip().startswith('结论') or '命中' in ln or '频率范围' in ln:
                lines.append('   ' + ln)

    _emit(lines, root)


def _emit(lines, root):  # 打印并写文件
    text = '\n'.join(lines)
    print(text)
    try:
        with open(os.path.join(root, 'softwedge_report.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        print('\n已写出 %s' % os.path.join(root, 'softwedge_report.txt'))
    except OSError as e:
        print('写报告失败: %s' % e)


if __name__ == '__main__':
    main()
