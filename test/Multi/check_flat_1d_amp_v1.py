# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8
"""平坦模型一维放大 QA 校验工具 v1（不依赖 Abaqus，可直接运行）。

用途（v6 修改计划第 2 步）：
  把平坦对照模型从"TAF 分母"降级为"质检通道"后，用本脚本核对：
    数值一维放大 = 平坦模型地表 PGA_h / 解析分母 PGA_ff_h
    解析一维放大 = fd 频域引擎对上平台成层柱的预测（含与 FE 同口径的瑞利阻尼）
  二者应接近（差异反映网格/时间步/数值阻尼误差）；同时它们都应
  与论文图13/图15 中远离坡体的"平台段 TAF"一致——这正是论文分母口径的证据。

输入：一个工况文件夹（含 case_meta.json、输入波 txt；可选 PGA-*-flat.csv）。
运行：python check_flat_1d_amp_v1.py <工况文件夹路径>
（沙箱/特殊环境可用环境变量 V6_PATH 指定 v6 脚本所在目录。）
"""

import os, sys, math, glob, json, types, importlib  # 导入标准库
import numpy as np  # 导入数值库

for _n, _attr in (('abaqus', 'mdb'), ('abaqusConstants', None),  # 用桩模块屏蔽 Abaqus 依赖
                  ('regionToolset', 'Region'), ('caeModules', None), ('mesh', None)):
    _m = types.ModuleType(_n)  # 创建桩模块
    if _attr == 'mdb':  # abaqus 需要 mdb 占位
        _m.mdb = types.SimpleNamespace()  # 设置 mdb 占位
    elif _attr == 'Region':  # regionToolset 需要 Region 占位
        _m.Region = object  # 设置 Region 占位
    sys.modules[_n] = _m  # 注册桩模块

_OVERRIDE = os.environ.get('V6_PATH')  # 可选：v6 脚本目录覆盖（沙箱测试用）
if _OVERRIDE:  # 提供了覆盖目录
    sys.path.insert(0, _OVERRIDE)  # 优先从覆盖目录导入
PARENT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Modeling', 'Multi')  # 上级目录（Multi）
if PARENT not in sys.path:  # 确保可 import 上级脚本
    sys.path.append(PARENT)  # 加入搜索路径
ml = importlib.import_module('VAB_oblique_TAF_multilayer_v6')  # 导入 v6 多层脚本


def load_meta(folder):  # 读取工况元数据
    """读取 folder/case_meta.json 并返回 dict。"""
    path = os.path.join(folder, 'case_meta.json')  # 元数据路径
    with open(path, 'r', encoding='utf-8') as f:  # 打开元数据
        return json.load(f)  # 解析返回


def build_from_meta(meta):  # 由元数据重建场地/几何/分层
    """返回 (site, geom, strat, damping)：与建模时一致的对象。"""
    g = meta['geometry']  # 几何块
    bed = meta['bedrock']  # 基岩块
    site_layers = []  # 有限层列表（从上到下）
    metas = meta.get('layers') or []  # 元数据层列表
    for i, L in enumerate(metas):  # 逐层重建 Material
        th = L.get('thickness') if i < len(metas) - 1 else None  # 最底覆盖层厚度由几何决定
        site_layers.append(ml.Material(cs=float(L['cs']), vv=float(L['vv']),
                                       density=float(L['density']), thickness=th, name=str(L['name'])))
    bedrock = ml.Material(cs=float(bed['cs']), vv=float(bed['vv']),
                          density=float(bed['density']), thickness=None, name=str(bed['name']))
    site = ml.Site(bedrock=bedrock, layers=site_layers, bedrock_thickness=float(g['bedrock_thickness']))
    fixed = [float(L['thickness']) for L in metas[:-1] if L.get('thickness')]  # 顶部固定层厚
    geom = ml.make_geometry(total_L=float(g['total_L']), H_minus_h=float(g['H_minus_h']),
                            i=float(g['i']), h_over_H=float(g['h_over_H']),
                            left_flat=float(g['left_flat']),
                            bedrock_thickness=float(g['bedrock_thickness']),
                            fixed_thicknesses=fixed)  # 重建几何
    strat = ml._build_stratigraphy(site, geom, ymin=0.0)  # 分层带
    dmp = meta.get('damping') or {}  # 阻尼块
    damping = None  # 默认无阻尼
    if dmp.get('enable'):  # 启用过阻尼
        damping = {'enable': True, 'method': dmp.get('method', 'rayleigh'),
                   'qs_factor': float(dmp.get('qs_factor', 0.05)),
                   'q_bedrock': float(dmp.get('q_bedrock', 999.0)),
                   'fc': float(dmp.get('fc')), 'f1_factor': float(dmp.get('f1_factor', 0.5)),
                   'f2_factor': float(dmp.get('f2_factor', 2.5))}  # 重建解析后阻尼配置
    return site, geom, strat, damping  # 返回重建对象


def predict_plateau(meta, folder, rec_path):  # 解析预测上平台一维放大
    """用 fd 引擎预测上平台柱地表 PGA 与平台 TAF，返回 dict。"""
    site, geom, strat, damping = build_from_meta(meta)  # 重建对象
    angle = float(meta.get('incident_angle') or 0.0)  # 入射角
    a1r = math.radians(angle if abs(angle) > 1e-12 else 1e-10)  # 弧度
    mb = ml._compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩参数
    p = math.sin(a1r) / mb['cs']  # 水平慢度
    b1 = ml._safe_arcsin(mb['cp'] * math.sin(a1r) / mb['cs'])  # 基岩 P 角
    rec = np.loadtxt(rec_path)  # 输入记录 [t, acc]
    t, acc = rec[:, 0], rec[:, 1]  # 拆列
    dt = t[1] - t[0]  # 步长
    damp_terms = ml._band_damping_terms(strat, damping)  # 各带瑞利系数
    ffcfg = dict(ml.freefield_cfg)  # fd 引擎默认配置
    ml._FD_SOLVER_CACHE.clear(); ml._REFL_COEFF_CACHE.clear()  # 清缓存
    zeros2 = np.column_stack((t, np.zeros_like(acc)))  # 占位 VEL/DIS（fd 路径不用）
    ctx = ml.FreeFieldCtx(site=site, geom=geom, strat=strat, ymax_l=geom.H_upper, ymax_r=geom.H_lower,
                          ymin=0.0, alpha=a1r, beta_p=b1, p_horiz=p, GG=mb['GG'], lam=mb['lam'],
                          cs=mb['cs'], cp=mb['cp'], VEL=zeros2, DIS=zeros2, dt=dt, time_arr=t,
                          max_reflect_order=3, acc=acc, damp_terms=damp_terms, ffcfg=ffcfg)  # 上下文
    fd = ml._fd_freefield_at_node('l', 0.0, geom.H_upper, geom.H_upper, ctx)  # 上平台地表自由场
    acc_h = np.gradient(fd['dotux'], dt)  # 水平加速度（速度微分）
    acc_v = np.gradient(fd['dotuy'], dt)  # 竖向加速度
    fs = ml._compute_free_surface_sv_coeff(a1r, mb['cp'], mb['cs'])  # 基岩自由面系数
    fh = (1.0 - fs['A1']) * math.cos(a1r) + fs['A2'] * math.sin(fs['beta'])  # 解析分母系数
    pga_in = float(np.max(np.abs(acc)))  # 输入峰值
    denom = fh * pga_in  # 解析分母 PGA_ff_h
    return {'taf_h': float(np.max(np.abs(acc_h)) / denom),  # 预测平台 TAF_h
            'taf_v': float(np.max(np.abs(acc_v)) / denom),  # 预测平台 TAF_v
            'pga_h_pred': float(np.max(np.abs(acc_h))),  # 预测地表水平 PGA
            'factor_h': fh, 'pga_in': pga_in, 'denom': denom}  # 分母明细


def numerical_plateau(folder, denom):  # 读取数值平坦结果的一维放大
    """从 PGA-*-flat.csv 取中段中位数 PGA_h，除以解析分母；无文件返回 None。"""
    flats = sorted(glob.glob(os.path.join(folder, 'PGA-*-flat.csv')))  # 平坦 PGA 文件
    flats = [f for f in flats if '-normalized' not in f]  # 排除归一化副本
    if not flats:  # 无平坦数据
        return None  # 返回空
    data = np.genfromtxt(flats[0], delimiter=',', names=True)  # 读取首个文件
    x = data['x']; pga_h = data['PGA_h']  # 取列
    lo = x.min() + (x.max() - x.min()) * 0.3  # 中段下界
    hi = x.min() + (x.max() - x.min()) * 0.7  # 中段上界
    mid = (x > lo) & (x < hi)  # 中段掩码（避开边界吸收影响）
    return float(np.median(pga_h[mid]) / denom)  # 数值一维放大（中位数）


def main():  # 主流程
    folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())  # 工况文件夹
    print('== 平坦一维放大 QA：%s ==' % folder)  # 标题
    meta = load_meta(folder)  # 读元数据
    txts = sorted(glob.glob(os.path.join(folder, '*.txt')))  # 输入波候选
    if not txts:  # 无输入波
        raise IOError('工况文件夹内未找到输入波 txt')  # 报错
    rec_path = txts[0]  # 取首个（每工况单一波形）
    pred = predict_plateau(meta, folder, rec_path)  # 解析预测
    print('入射角 %s° | factor_h=%.4f | PGA_in=%.4g | PGA_ff_h=%.4g' % (
        meta.get('incident_angle'), pred['factor_h'], pred['pga_in'], pred['denom']))  # 分母明细
    print('fd 解析预测: 平台 TAF_h=%.3f, TAF_v=%.3f（应≈论文图13/15 远场平台值）' % (
        pred['taf_h'], pred['taf_v']))  # 预测结果
    num = numerical_plateau(folder, pred['denom'])  # 数值结果
    if num is None:  # 无平坦数值数据
        print('未找到 PGA-*-flat.csv（run_flat 可能已关闭），仅输出解析预测。')  # 提示
    else:  # 有数值数据时对比
        diff = (num - pred['taf_h']) / pred['taf_h'] * 100.0  # 相对差异
        print('Abaqus 平坦模型数值: 平台放大=%.3f | 与解析预测差 %.1f%%' % (num, diff))  # 对比
        print('（差异主要反映网格/时间步/数值阻尼误差；>10%% 建议检查 mesh_cfg 与 dt）')  # 解释
    print('QA 完成。')  # 结束


if __name__ == '__main__':  # 直接运行入口
    main()  # 执行主流程
