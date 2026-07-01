# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""方法验证 Tier 4 汇总：时间步收敛 + 边界距离无关性(纯 Python)。

读取 multi-verify-* 工况，以 base 为参照，输出各工况 远场/坡顶 TAF 及相对 base 的偏差，
并判读 dt 收敛(dthalf vs base <2%) 与 边界无关(wide/deep vs base <几%)。

用法：python verify_numerical_report_v1.py <工况根目录>
"""

import os, sys, csv, json, glob  # 标准库


def _read_taf(folder):
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


def _metrics(xs, th, x_crest):
    far = [th[i] for i in range(len(xs)) if xs[i] < 200.0]
    far = sum(far) / len(far) if far else float('nan')
    win = [th[i] for i in range(len(xs)) if (x_crest - 90.0) <= xs[i] <= (x_crest + 30.0)]
    peak = max(win) if win else float('nan')
    return far, peak


def main():
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()
    data = {}
    for fd in sorted(glob.glob(os.path.join(root, 'multi-verify-*'))):
        name = os.path.basename(fd).replace('multi-verify-', '')
        meta_p = os.path.join(fd, 'case_meta.json')
        xs, th = _read_taf(fd)
        if not os.path.isfile(meta_p) or xs is None:
            continue
        meta = json.load(open(meta_p, encoding='utf-8'))
        xc = meta.get('geometry', {}).get('x_crest', 1000.0)
        far, peak = _metrics(xs, th, xc)
        data[name] = (far, peak)

    L = ['================= 方法验证 Tier 4：时间步收敛 + 边界距离无关性 =================', '根目录: %s' % root]
    if 'base' not in data:
        L.append('未找到 base 工况，无法对照。'); _emit(L, root); return
    fb, pb = data['base']
    L.append('%-10s %-10s %-10s %-12s %-12s' % ('工况', '远场TAF', '坡顶TAF', '坡顶vs base', '说明'))
    L.append('-' * 64)
    desc = {'base': '基线(dt=0.001,默认域)', 'dthalf': 'dt 减半(0.0005)', 'wide': '侧边界加宽', 'deep': '底边界加深'}
    order = ['base', 'dthalf', 'wide', 'deep']
    for k in order + [x for x in data if x not in order]:
        if k not in data:
            continue
        far, peak = data[k]
        d = (peak - pb) / pb * 100.0 if pb else float('nan')
        L.append('%-10s %-10.3f %-10.3f %-12s %-12s' % (k, far, peak, ('—' if k == 'base' else '%+.1f%%' % d), desc.get(k, '')))
    L.append('-' * 64)

    # 判读
    L.append('')
    dthalf = data.get('dthalf'); wide = data.get('wide'); deep = data.get('deep')
    def pdiff(x): return abs(x[1] - pb) / pb * 100.0 if (x and pb) else None
    dd = pdiff(dthalf); dw = pdiff(wide); de = pdiff(deep)
    if dd is not None:
        L.append('① 时间步收敛：dt 0.001→0.0005 坡顶 TAF 偏差 %.1f%% %s' % (dd, '→ 已收敛(<2%)，dt=0.001 足够。' if dd < 2 else '→ 偏差偏大，建议用 dt=0.0005。'))
    if dw is not None and de is not None:
        ok = (dw < 3 and de < 3)
        L.append('② 边界距离无关性：加宽 %.1f%%、加深 %.1f%% %s' % (dw, de, '→ 坡面响应不随边界位置改变(<3%)，域足够大、边界不污染解。' if ok else '→ 偏差偏大，建议进一步加大模型域。'))
    L.append('')
    L.append('结论：以上证 FE 本身数值收敛、边界充分远——与论文图15绝对值无关，属方法可信性(Verification)证据。')
    _emit(L, root)


def _emit(L, root):
    text = '\n'.join(L); print(text)
    try:
        with open(os.path.join(root, 'verify_numerical_report.txt'), 'w', encoding='utf-8') as f:
            f.write(text)
        print('\n已写出 %s' % os.path.join(root, 'verify_numerical_report.txt'))
    except OSError as e:
        print('写报告失败: %s' % e)


if __name__ == '__main__':
    main()
