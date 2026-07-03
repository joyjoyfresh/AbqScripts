# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""【P0#1 重力步柱底轴力校验】读 ODB，核对 Step-gravity 末帧柱底轴力 ≈ Σ楼层质量·g/柱数。

对应 TSSI 研究计划书 §5 QA 锚点「重力步柱底轴力 与手算 Σm·g/柱数偏差 <2%」。
在 Abaqus python 下运行（由 Autorun 子进程调用），cwd=工况文件夹，自动找 *.odb。
读同目录 tssi_meta.json 拿框架参数(n_story/n_bay/floor_mass)，写 gravity_axial_check.csv。

判读：
  · gravity='off' 的工况无 Step-gravity → 打印跳过（作对照，本就不应有静重力）。
  · gravity='structure' 的工况：实测均值应与手算相对误差 <2%，否则重力施加或 Tie 传力有问题。
"""
from __future__ import print_function  # Py2/Py3 print 兼容
import os  # 路径
import glob  # 找 odb
import json  # 读 tssi_meta
import csv  # 写校验结果

try:
    from odbAccess import openOdb  # Abaqus ODB 接口(仅 abaqus python 可用)
except Exception:  # 非 abaqus 环境导入失败
    openOdb = None


GRAVITY_G = 9.81  # 重力加速度(与建模脚本同口径)


def _find_odb():  # 当前目录第一个 .odb
    ff = sorted(glob.glob('*.odb'))
    return ff[0] if ff else None


def main():
    if openOdb is None:  # 非 abaqus 环境直接退出
        print('未加载 odbAccess(非 Abaqus python 环境)，跳过轴力校验')
        return
    odb_path = _find_odb()
    if not odb_path:
        print('当前目录未找到 .odb，跳过轴力校验')
        return

    meta = {}  # 框架参数
    if os.path.isfile('tssi_meta.json'):
        with open('tssi_meta.json') as f:
            meta = json.load(f)
    n_story = int(meta.get('n_story', 0))  # 层数
    n_bay = int(meta.get('n_bay', 0))  # 跨数
    floor_mass = float(meta.get('floor_mass', 0.0))  # 每层楼板集中质量(kg)
    n_col = n_bay + 1  # 柱数=跨数+1
    expected = (n_story * floor_mass * GRAVITY_G / n_col) if n_col else 0.0  # 每柱底轴力手算值(N)

    odb = openOdb(odb_path, readOnly=True)  # 只读打开
    try:
        if 'Step-gravity' not in odb.steps.keys():  # gravity='off' 无静力步
            print('%s 无 Step-gravity(gravity=off?)，跳过轴力校验(对照工况正常)' % odb_path)
            return
        step = odb.steps['Step-gravity']  # 重力静力步
        per = []  # (历史区名, 末帧SF1)
        total_axial = 0.0  # 轴力合计
        for name, hr in step.historyRegions.items():  # 遍历历史输出区
            if 'SF1' in hr.historyOutputs.keys():  # 仅 COLS_BASE 柱单元有 SF1(轴力)
                val = hr.historyOutputs['SF1'].data[-1][1]  # 重力步末帧轴力
                per.append((name, val))
                total_axial += val
        if not per:  # 没读到 SF1(H-ColBase 未生效)
            print('警告：Step-gravity 内未读到任何 SF1 历史输出，检查 H-ColBase/COLS_BASE 是否建成')
            return
        actual_mean = abs(total_axial) / len(per)  # 各柱底轴力均值(SF1 受压为负，取绝对值)
        rel = (actual_mean - expected) / expected if expected else 0.0  # 相对误差
        verdict = 'PASS(<2%)' if abs(rel) < 0.02 else 'CHECK(>=2%)'  # §5 判据
        print('柱底轴力校验[%s]: 手算 %.1f kN/柱, 实测均值 %.1f kN/柱(%d柱), 相对误差 %+.2f%% -> %s'
              % (odb_path, expected / 1.0e3, actual_mean / 1.0e3, len(per), 100.0 * rel, verdict))
        with open('gravity_axial_check.csv', 'w') as f:  # 留痕
            w = csv.writer(f)
            w.writerow(['odb', 'n_story', 'n_bay', 'floor_mass_kg',
                        'expected_N_per_col', 'actual_mean_N_per_col', 'n_col_elems', 'rel_err', 'verdict'])
            w.writerow([odb_path, n_story, n_bay, floor_mass,
                        expected, actual_mean, len(per), rel, verdict])
    finally:
        odb.close()  # 确保关闭 odb


if __name__ == '__main__':
    main()
