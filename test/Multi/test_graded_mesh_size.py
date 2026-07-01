# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
# 测试目的：验证 v9.1 分层非均匀网格尺寸 _band_graded_sizes（与脚本逐字一致）：
#   ① 波速比缩放(深部粗)；② 薄软层按共振谐波加密；③ 穿层单元数判据(优先于 min_size)；
#   并验证总单元数仍少于均匀细网格、关闭开关可退化。纯 Python，无需 Abaqus。
# 说明：_seed_graded_mesh 调用 Abaqus 建模 API，无法在此环境运行，需在 Abaqus 内首跑时核对网格质量。

import sys  # 退出码


def _band_graded_sizes(strat, mesh_used, mcfg, fc=None):  # 与脚本 v9.1 逐字一致
    """各带尺寸 = min(波速比缩放, 谐波波长判据, 穿层判据)，受 max_band_ratio/max_size 约束，
    下限 = min(min_size, 穿层判据)。resolve_harmonics=0/None 时退化为纯波速比缩放。"""
    cs_min = min([b['mat'].cs for b in strat])  # 最软层波速
    epw = float(mcfg.get('elems_per_wavelength', 10))  # 每波长单元数
    fmax_factor = float(mcfg.get('fmax_factor', 2.5))  # 输入频带 fmax 倍数
    max_size = mcfg.get('max_size')  # 绝对上限
    max_ratio = float(mcfg.get('max_band_ratio', 4.0))  # 过渡比上限
    min_size = float(mcfg.get('min_size', 0.5))  # 单元下限
    rh = mcfg.get('resolve_harmonics', 3.0)  # 解析谐波次数
    rh = float(rh) if rh else 0.0  # 规范化
    n_thk = mcfg.get('min_elems_through_thickness', 6)  # 穿层最少单元数
    n_thk = float(n_thk) if n_thk else 0.0  # 规范化
    sizes = []  # 各带尺寸
    for b in strat:  # 从下到上
        cs = float(b['mat'].cs)  # 波速
        thk = float(b['y1']) - float(b['y0'])  # 厚度
        s = mesh_used * (cs / float(cs_min))  # ① 波速比缩放
        dl_thick = (thk / n_thk) if (n_thk > 0 and thk > 0) else None  # ③ 穿层判据
        if rh > 0 and thk > 0:  # ② 谐波加密
            f_layer = cs / (4.0 * thk)  # 共振基频
            f_resolve = rh * f_layer  # 解析到若干阶谐波
            if fc:  # 并入输入频带
                f_resolve = max(f_resolve, fmax_factor * float(fc))
            dl_wave = cs / (epw * f_resolve)  # 波长判据
            s = min(s, dl_wave)
        if dl_thick:  # ③
            s = min(s, dl_thick)
        s = min(s, mesh_used * max_ratio)  # 过渡比上限
        if max_size:  # 绝对上限
            s = min(s, float(max_size))
        floor = min(min_size, dl_thick) if dl_thick else min_size  # 下限：穿层优先
        s = max(s, floor)
        sizes.append(s)
    return sizes


class _M(object):  # 材料桩
    def __init__(self, cs):
        self.cs = cs


def _build(h_soft):  # 三层(从下到上)：基岩2000/覆盖1600/软表层800；H_upper=600, 基岩厚200
    H_upper, bt = 600.0, 200.0
    y_surf0 = H_upper - h_soft
    return [
        {'name': 'bedrock', 'mat': _M(2000), 'y0': 0.0, 'y1': bt},
        {'name': 'overlying', 'mat': _M(1600), 'y0': bt, 'y1': y_surf0},
        {'name': 'surface', 'mat': _M(800), 'y0': y_surf0, 'y1': H_upper},
    ]


def main():
    W, mesh_used, fc = 1800.0, 4.0, 4.0
    mcfg = {'elems_per_wavelength': 10, 'fmax_factor': 2.5, 'max_band_ratio': 4.0,
            'min_size': 0.5, 'resolve_harmonics': 3.0, 'min_elems_through_thickness': 6}

    def ng(strat, sizes):  # 分层单元数估算(矩形近似)
        return sum((W / s) * ((b['y1'] - b['y0']) / s) for b, s in zip(strat, sizes))

    def nu(strat, sz):  # 均匀单元数估算
        return sum((W / sz) * ((b['y1'] - b['y0']) / sz) for b in strat)

    fails = []
    print('%-7s %-8s %-22s %-10s %-10s' % ('h_soft', 'f_layer', '[bed,ovl,surf]', '分层单元', '均匀4m'))
    for h in [50.0, 25.0, 12.5]:
        strat = _build(h)
        sz = _band_graded_sizes(strat, mesh_used, mcfg, fc)
        f_layer = 800.0 / (4 * h)
        print('%-7.1f %-8.2f [%5.2f,%5.2f,%5.2f] %-10.0f %-10.0f'
              % (h, f_layer, sz[0], sz[1], sz[2], ng(strat, sz), nu(strat, mesh_used)))
        f_res = max(2.5 * fc, 3 * f_layer)
        want = min(800.0 / (10 * f_res), h / 6.0, mesh_used)
        if abs(sz[2] - want) > 1e-6:
            fails.append('h=%.1f 软层应=%.3f 实=%.3f' % (h, want, sz[2]))
        if abs(sz[0] - 10.0) > 1e-6:
            fails.append('h=%.1f 基岩应=10' % h)
        if ng(strat, sz) >= nu(strat, mesh_used):
            fails.append('h=%.1f 分层应少于均匀4m' % h)

    if _band_graded_sizes(_build(12.5), mesh_used, mcfg, fc)[2] >= 4.0:
        fails.append('h=12.5 软层应细于4m')
    if abs(_band_graded_sizes(_build(50.0), mesh_used, mcfg, fc)[2] - 4.0) > 1e-6:
        fails.append('h=50 软层应=4')
    off = _band_graded_sizes(_build(12.5), mesh_used, dict(mcfg, resolve_harmonics=0, min_elems_through_thickness=0), fc)
    if abs(off[2] - 4.0) > 1e-6:
        fails.append('关闭谐波应退化软层=4')
    if abs(_band_graded_sizes(_build(1.8), mesh_used, mcfg, fc)[2] - 0.3) > 1e-3:
        fails.append('h=1.8 穿层应=0.3(优先于min_size)')

    print('=' * 60)
    if fails:
        for f in fails:
            print('[FAIL]', f)
        sys.exit(1)
    print('通过：薄软层按谐波/穿层加密(细于4m)，厚软层维持4m，深部基岩仍10m；')
    print('      总单元仍少于均匀4m；可关闭退化；穿层判据优先于 min_size。')
    sys.exit(0)


if __name__ == '__main__':
    main()
