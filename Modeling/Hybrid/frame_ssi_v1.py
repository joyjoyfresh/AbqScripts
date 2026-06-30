# -*- coding: utf-8 -*-
"""
Step-2a：二维框架-土相互作用(SSI) —— 平层均质土 + 框架坐顶 耦合机制验证脚本
================================================================================
TSSI 路线第二步(a)。在 step1(固定基础框架已验)基础上，验证【新增的 SSI 耦合机制】：
    框架-土 Tie 耦合、土体地震波输入、周期侧边(1D 剪切)、SSI vs 固定基础对比。
刻意先用【平层均质土 + 竖向输入】隔离 SSI 机制；坡面几何+斜入射(Multi 已验)留 step2b。

验证信号（"对不对"）：
  1. 场地自由场：地表放大合理、场地基频 f0≈Vs/(4H)
  2. SSI 周期延长：T_ssi > T_fixed(=step1 固定基础的 0.5s) —— SSI 的标志性特征
  3. 基底剪力/顶层漂移：与 step1 固定基础对比，量化 SSI 修正

建两套模型：
  - freefield : 仅土体(无框架)，给自由场地表运动基准
  - ssi       : 框架坐土体顶(Tie)，含 SSI

边界方案（平层 SSI 标准做法，非 Multi 的吸收边界——那是坡面散射才需要的，留 step2b）：
  - 基底：刚性输入(AccelerationBC 水平加速度，沿用 step1 已验机制) + 竖向约束
  - 侧边：周期(左右同高节点 U 相等，Equation) → 土块做纯 1D 剪切，干净场地响应
  - 地表：自由

运行：
    abaqus cae noGUI=frame_ssi_v1.py        # 建 freefield + ssi 并提交
    abaqus cae noGUI=postproc_ssi_v1.py     # 后处理对比

Py2.7 兼容；不用 f-string；open 不传 encoding。
"""

from abaqus import *
from abaqusConstants import *
from abaqus import mdb
from regionToolset import Region
from caeModules import *
import mesh
import numpy as np
import math
import os
import json
import sys
import time
import logging
import traceback


# ==========================================================
#  配置参数
# ==========================================================

# 土体（平层均质半空间块）
soil_cfg = {
    'Vs': 300.0,           # 剪切波速 (m/s)，软土量级
    'density': 1800.0,     # 密度 (kg/m^3)
    'nu': 0.3,             # 泊松比
    'width': 90.0,         # 土块宽度 (m)
    'depth': 30.0,         # 土块深度 (m)。场地基频 f0=Vs/(4H)=300/120=2.5Hz
    'damping_ratio': 0.05,  # 土体阻尼比
    'mesh_size': 3.0,      # 土体网格尺寸 (m)。须 <= Vs/(10*fmax) 解析最短波长
}

# 框架（与 step1 同——保证 T_fixed 可比）
frame_cfg = {
    'n_story': 5, 'n_bay': 3, 'story_height': 3.0, 'bay_width': 6.0,
    'column': {'width': 0.5, 'depth': 0.5},
    'beam':   {'width': 0.3, 'depth': 0.6},
    'floor_mass': 5.0e4,   # 每层集中质量 (kg)
}

# 框架材料（与 step1 同；density=10 为正则化质量，详见 step1 README）
frame_material_cfg = {
    'name': 'Concrete_C30', 'E': 30.0e9, 'nu': 0.2, 'density': 10.0,
    'damping_ratio': 0.05, 'f1': 1.0, 'f2': 5.0,  # 框架瑞利锚点 (Hz)
}

# 分析 / 作业
job_cfg = {
    'num_cpus': 4, 'memory_percent': 90, 'history_freq': 1, 'submit': True,
}

# 运行控制：建/提交哪些场景
run_cfg = {
    'scenes': ['freefield', 'ssi', 'fixed'],   # 'freefield'=仅土 / 'ssi'=框架+土 / 'fixed'=刚性基础(输入自由场地表运动,去耦对照)
}


# ==========================================================
#  通用工具
# ==========================================================

_DEFAULT_SCRIPT_NAME = 'frame_ssi_v1.py'


def log_step(logger=None, message=None, *args):
    """首次调用初始化日志器；后续带累计用时输出。"""
    if not hasattr(log_step, '_logger'):
        log_filename = logger if isinstance(logger, str) else 'frame_ssi.log'
        _logger = logging.getLogger('frame_ssi')
        _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger.handlers = []
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_filename, mode='w')
        fh.setFormatter(fmt)
        _logger.addHandler(fh)
        log_step._logger = _logger
        log_step._start = time.time()
        return _logger
    if message is not None:
        log_step._logger.info('[%.3fs] ' + message, time.time() - log_step._start, *args)
    return log_step._logger


def _ensure_str(obj):
    """Py2 下把 json 的 unicode 递归转 str（Abaqus C++ API 拒收 unicode）。"""
    if sys.version_info[0] >= 3:
        return obj
    if isinstance(obj, unicode):           # noqa: F821 (Py2)
        return obj.encode('utf-8')
    if isinstance(obj, dict):
        return dict((_ensure_str(k), _ensure_str(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [_ensure_str(x) for x in obj]
    return obj


def find_acc_txt(logger=None):
    """读取 cwd 下第一条两列加速度记录 .txt，返回 (文件名, t, a, dt)。"""
    cwd = os.getcwd()
    txts = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.txt')])
    if not txts:
        raise IOError(u'当前目录无 .txt 加速度记录: %s' % cwd)
    f = txts[0]
    data = np.loadtxt(f)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(u'%s 不是两列(时间,加速度)格式' % f)
    t = data[:, 0].astype(float)
    a = data[:, 1].astype(float)
    dt = float(t[1] - t[0])
    if logger:
        log_step(logger, u'加速度记录: %s, 点数=%d, dt=%.4fs, 时长=%.2fs, |a|max=%.3f m/s^2',
                 f, len(t), dt, float(t[-1]), float(np.max(np.abs(a))))
    return f, t, a, dt


def rayleigh_coeffs(xi, f1, f2):
    """两端 ξ 相等的瑞利系数 (alpha, beta)。"""
    w1 = 2.0 * math.pi * f1
    w2 = 2.0 * math.pi * f2
    return 2.0 * xi * w1 * w2 / (w1 + w2), 2.0 * xi / (w1 + w2)


# ==========================================================
#  土体 Part
# ==========================================================

def build_soil_part(model, logger):
    """建平层均质土块 Part(CPE4R)、材料(含瑞利)、网格、节点集与地表面。返回 part。"""
    W = float(soil_cfg['width']); H = float(soil_cfg['depth'])
    cs = float(soil_cfg['Vs']); rho = float(soil_cfg['density']); nu = float(soil_cfg['nu'])
    G = rho * cs ** 2
    E = 2.0 * G * (1.0 + nu)
    f0 = cs / (4.0 * H)   # 场地基频估计

    # 几何
    sk = model.ConstrainedSketch(name='__soil__', sheetSize=max(W, H) * 2.0)
    sk.rectangle(point1=(0.0, 0.0), point2=(W, H))
    part = model.Part(name='Soil', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sk)
    del model.sketches['__soil__']

    # 材料（瑞利锚定 f0 与 5*f0，覆盖场地共振带）
    mat = model.Material(name='Soil')
    mat.Elastic(table=((E, nu),))
    mat.Density(table=((rho,),))
    a_r, b_r = rayleigh_coeffs(soil_cfg['damping_ratio'], f0, 5.0 * f0)
    mat.Damping(alpha=a_r, beta=b_r)
    model.HomogeneousSolidSection(name='SoilSec', material='Soil', thickness=1.0)
    part.SectionAssignment(region=Region(faces=part.faces), sectionName='SoilSec')
    log_step(logger, u'土体: Vs=%.0f, E=%.2e Pa, ρ=%.0f, f0=%.2f Hz, 瑞利 α=%.4f β=%.6f',
             cs, E, rho, f0, a_r, b_r)

    # 网格
    ms = float(soil_cfg['mesh_size'])
    part.setMeshControls(regions=part.faces, elemShape=QUAD, technique=STRUCTURED)
    part.seedPart(size=ms, deviationFactor=0.1, minSizeFactor=0.1)
    part.setElementType(regions=(part.faces,),
                        elemTypes=(mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD),
                                   mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)))
    part.generateMesh()
    log_step(logger, u'土体网格: CPE4R, size=%.1fm, 单元=%d, 节点=%d',
             ms, len(part.elements), len(part.nodes))

    # 节点集（按坐标）
    tol = ms * 1e-3
    nd = part.nodes
    base = [n for n in nd if abs(n.coordinates[1] - 0.0) < tol]
    top = [n for n in nd if abs(n.coordinates[1] - H) < tol]
    left = [n for n in nd if abs(n.coordinates[0] - 0.0) < tol]
    right = [n for n in nd if abs(n.coordinates[0] - W) < tol]
    part.Set(nodes=nd.sequenceFromLabels([n.label for n in base]), name='SOIL_BASE')
    part.Set(nodes=nd.sequenceFromLabels([n.label for n in top]), name='SOIL_TOP')
    part.Set(nodes=nd.sequenceFromLabels([n.label for n in left]), name='SOIL_L')
    part.Set(nodes=nd.sequenceFromLabels([n.label for n in right]), name='SOIL_R')
    # 地表中心参考节点（场地响应）
    cx = W / 2.0
    cnode = min(top, key=lambda n: abs(n.coordinates[0] - cx))
    part.Set(nodes=nd.sequenceFromLabels([cnode.label]), name='SURF_CENTER')
    # 地表面（供 Tie 主面）
    top_edge = part.edges.findAt(((cx, H, 0.0),))
    part.Surface(side1Edges=top_edge, name='SOIL_TOP_SURF')
    log_step(logger, u'土体节点集: 基底=%d, 地表=%d, 左=%d, 右=%d', len(base), len(top), len(left), len(right))
    return part


# ==========================================================
#  框架 Part（局部基底 y=0；与 step1 同构造）
# ==========================================================

def build_frame_part(model, logger):
    """建框架 Part(B21 梁,1单元/构件)、材料、截面、节点集。返回 (part, floor_full, ns)。"""
    nb = int(frame_cfg['n_bay']); ns = int(frame_cfg['n_story'])
    bw = float(frame_cfg['bay_width']); sh = float(frame_cfg['story_height'])
    xs = [j * bw for j in range(nb + 1)]
    ys = [k * sh for k in range(ns + 1)]
    z = 0.0

    sk = model.ConstrainedSketch(name='__frame__', sheetSize=max(xs[-1], ys[-1]) * 2.0)
    for x in xs:
        for k in range(ns):
            sk.Line(point1=(x, ys[k]), point2=(x, ys[k + 1]))
    for k in range(1, ns + 1):
        for j in range(nb):
            sk.Line(point1=(xs[j], ys[k]), point2=(xs[j + 1], ys[k]))
    part = model.Part(name='Frame', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseWire(sketch=sk)
    del model.sketches['__frame__']

    mc = frame_material_cfg
    mat = model.Material(name=mc['name'])
    mat.Elastic(table=((mc['E'], mc['nu']),))
    mat.Density(table=((mc['density'],),))   # 正则化质量(=10)，避免无质量 DOF
    a_r, b_r = rayleigh_coeffs(mc['damping_ratio'], mc['f1'], mc['f2'])
    mat.Damping(alpha=a_r, beta=b_r)

    col = frame_cfg['column']; bm = frame_cfg['beam']
    model.RectangularProfile(name='ColProf', a=col['width'], b=col['depth'])
    model.RectangularProfile(name='BeamProf', a=bm['width'], b=bm['depth'])
    model.BeamSection(name='ColSec', profile='ColProf', material=mc['name'],
                      integration=DURING_ANALYSIS, poissonRatio=mc['nu'])
    model.BeamSection(name='BeamSec', profile='BeamProf', material=mc['name'],
                      integration=DURING_ANALYSIS, poissonRatio=mc['nu'])
    col_mids = [((x, (ys[k] + ys[k + 1]) / 2.0, z),) for x in xs for k in range(ns)]
    beam_mids = [(((xs[j] + xs[j + 1]) / 2.0, ys[k], z),) for k in range(1, ns + 1) for j in range(nb)]
    part.Set(edges=part.edges.findAt(*col_mids), name='COLS')
    part.Set(edges=part.edges.findAt(*beam_mids), name='BEAMS')
    part.SectionAssignment(region=part.sets['COLS'], sectionName='ColSec')
    part.SectionAssignment(region=part.sets['BEAMS'], sectionName='BeamSec')
    part.assignBeamSectionOrientation(region=part.Set(edges=part.edges, name='ALL_E'),
                                      method=N1_COSINES, n1=(0.0, 0.0, -1.0))

    base_pts = [((x, ys[0], z),) for x in xs]
    part.Set(vertices=part.vertices.findAt(*base_pts), name='BASE')
    for k in range(1, ns + 1):
        part.Set(vertices=part.vertices.findAt(((xs[0], ys[k], z),)), name='FLOOR_%d' % k)
    floor_full = {}
    for k in range(1, ns + 1):
        fpts = [((x, ys[k], z),) for x in xs]
        part.Set(vertices=part.vertices.findAt(*fpts), name='FLOORALL_%d' % k)
        floor_full[k] = ('FLOORALL_%d' % k, len(xs))

    part.seedEdgeByNumber(edges=part.edges, number=1, constraint=FIXED)  # 1单元/构件(B21 不生成内部节点)
    part.setElementType(regions=(part.edges,),
                        elemTypes=(mesh.ElemType(elemCode=B21, elemLibrary=STANDARD),))
    part.generateMesh()
    log_step(logger, u'框架: B21, 层=%d 跨=%d, 单元=%d, 节点=%d', ns, nb, len(part.elements), len(part.nodes))
    return part, floor_full, ns


# ==========================================================
#  周期侧边约束（左右同高节点 U 相等 → 1D 剪切）
# ==========================================================

def apply_periodic_sides(model, soil_inst_name, logger):
    """对土体左右边界同高度节点建 Equation(U1、U2 相等)，排除基底(y≈0，由基底 BC 驱动)。"""
    asm = model.rootAssembly
    inst = asm.instances[soil_inst_name]
    H = float(soil_cfg['depth']); tol = float(soil_cfg['mesh_size']) * 1e-2
    left = sorted([n for n in inst.sets['SOIL_L'].nodes], key=lambda n: n.coordinates[1])
    right = sorted([n for n in inst.sets['SOIL_R'].nodes], key=lambda n: n.coordinates[1])
    npair = 0
    for nl, nr in zip(left, right):
        y = nl.coordinates[1]
        if y < tol:           # 跳过基底节点(由 AccelerationBC 驱动，避免过约束)
            continue
        if abs(nl.coordinates[1] - nr.coordinates[1]) > tol:
            continue          # 高度不匹配(网格异常)则跳过
        sL = 'L_%d' % nl.label
        sR = 'R_%d' % nr.label
        asm.Set(nodes=inst.nodes.sequenceFromLabels([nl.label]), name=sL)
        asm.Set(nodes=inst.nodes.sequenceFromLabels([nr.label]), name=sR)
        for dof in (1, 2):    # U1、U2 周期
            model.Equation(name='PER_%d_%d' % (nl.label, dof),
                           terms=((1.0, sL, dof), (-1.0, sR, dof)))
        npair += 1
    log_step(logger, u'周期侧边: %d 对节点 × 2 DOF = %d 条 Equation', npair, npair * 2)


# ==========================================================
#  场景组装（freefield / ssi）
# ==========================================================

def build_scene(scene, acc_name, t, a, dt, logger):
    """建一个场景模型（freefield=仅土；ssi=框架+土+Tie），含步、输入、输出。返回 model_name。"""
    model_name = scene
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    H = float(soil_cfg['depth'])

    # 土体
    soil = build_soil_part(model, logger)
    soil_inst = 'Soil-1'
    asm.Instance(name=soil_inst, part=soil, dependent=ON)

    # 框架（仅 ssi）
    frame_inst = None
    floor_full = None; ns = 0
    if scene == 'ssi':
        frame, floor_full, ns = build_frame_part(model, logger)
        frame_inst = 'Frame-1'
        asm.Instance(name=frame_inst, part=frame, dependent=ON)
        # 平移框架到土体顶面中心：局部基底 y=0 -> 全局 y=H；x 居中
        fw = int(frame_cfg['n_bay']) * float(frame_cfg['bay_width'])
        x_off = float(soil_cfg['width']) / 2.0 - fw / 2.0
        asm.translate(instanceList=(frame_inst,), vector=(x_off, H, 0.0))
        # 楼层集中质量
        m_total = float(frame_cfg['floor_mass'])
        for k in range(1, ns + 1):
            nm, cnt = floor_full[k]
            asm.engineeringFeatures.PointMassInertia(
                name='Mass_%d' % k, region=asm.instances[frame_inst].sets[nm], mass=m_total / float(cnt))
        # 框架基底 Tie 到土体地表（tieRotations=OFF：柱脚铰接，整体摇摆经多柱差动竖向捕捉）
        model.Tie(name='FrameSoil', master=asm.instances[soil_inst].surfaces['SOIL_TOP_SURF'],
                  slave=asm.instances[frame_inst].sets['BASE'],
                  positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=OFF, thickness=ON)
        log_step(logger, u'框架已坐土顶(x_off=%.1f) 并 Tie 耦合; 楼层质量已加', x_off)

    # 分析步（隐式动力）
    tp = float(t[-1])
    model.ImplicitDynamicsStep(name='Step-EQ', previous='Initial',
                               timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=dt,
                               maxNumInc=1000000, nlgeom=OFF, application=TRANSIENT_FIDELITY)

    # 周期侧边
    apply_periodic_sides(model, soil_inst, logger)

    # 基底输入：水平加速度 + 竖向约束
    base = asm.instances[soil_inst].sets['SOIL_BASE']
    model.DisplacementBC(name='BaseV', createStepName='Initial', region=base, u2=0.0)
    amp_data = tuple((float(t[i]), float(a[i])) for i in range(len(t)))
    model.TabularAmplitude(name='AccAmp', data=amp_data, timeSpan=STEP)
    model.AccelerationBC(name='BaseAcc', createStepName='Step-EQ', region=base, a1=1.0, amplitude='AccAmp')

    # 历史输出
    freq = int(job_cfg['history_freq'])
    model.HistoryOutputRequest(name='H-Surf', createStepName='Step-EQ',
                               variables=('U1', 'A1'), region=asm.instances[soil_inst].sets['SURF_CENTER'],
                               frequency=freq)
    if scene == 'ssi':
        for k in range(1, ns + 1):
            model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName='Step-EQ',
                                       variables=('U1', 'V1', 'A1'),
                                       region=asm.instances[frame_inst].sets['FLOOR_%d' % k], frequency=freq)
    model.HistoryOutputRequest(name='H-Energy', createStepName='Step-EQ',
                               variables=('ALLKE', 'ALLIE', 'ALLVD'), frequency=freq * 10)
    log_step(logger, u'[%s] 步/输入/输出已配, 时长=%.2fs', scene, tp)
    return model_name


def extract_freefield_surface_acc(logger):
    """从已解算的 job-freefield.odb 提取地表中心节点绝对加速度 A1，返回 (t, a)。

    这是去耦法的关键：刚性基础(fixed)模型须输入【无结构】自由场地表运动，
    使 fixed 与 ssi 唯一差异 = SSI 效应(土柔度+惯性相互作用)。
    """
    from odbAccess import openOdb
    odb = openOdb('job-freefield.odb', readOnly=True)
    inst_keys = list(odb.rootAssembly.instances.keys())
    inst = 'Soil-1' if 'Soil-1' in inst_keys else ('SOIL-1' if 'SOIL-1' in inst_keys else inst_keys[0])
    lab = odb.rootAssembly.instances[inst].nodeSets['SURF_CENTER'].nodes[0].label
    hr = odb.steps['Step-EQ'].historyRegions['Node %s.%d' % (inst, lab)]
    data = np.array(hr.historyOutputs['A1'].data, dtype=float)
    odb.close()
    t = data[:, 0]; a = data[:, 1]
    log_step(logger, u'自由场地表加速度已提取: 点数=%d, |a|max=%.3f m/s² (去耦法刚性基础输入)',
             len(t), float(np.max(np.abs(a))))
    return t, a


def build_fixed_scene(t, a, dt, logger):
    """建刚性(固定)基础框架模型：框架单体，柱脚嵌固，基底输入【自由场地表加速度】。

    去耦法对照：与 ssi 同一框架、同一自由场地表运动输入，差异仅在有无土柔度。
    """
    model_name = 'fixed'
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)

    frame, floor_full, ns = build_frame_part(model, logger)
    frame_inst = 'Frame-1'
    asm.Instance(name=frame_inst, part=frame, dependent=ON)
    m_total = float(frame_cfg['floor_mass'])
    for k in range(1, ns + 1):
        nm, cnt = floor_full[k]
        asm.engineeringFeatures.PointMassInertia(
            name='Mass_%d' % k, region=asm.instances[frame_inst].sets[nm], mass=m_total / float(cnt))

    tp = float(t[-1])
    model.ImplicitDynamicsStep(name='Step-EQ', previous='Initial',
                               timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=dt,
                               maxNumInc=1000000, nlgeom=OFF, application=TRANSIENT_FIDELITY)
    base = asm.instances[frame_inst].sets['BASE']
    model.DisplacementBC(name='BaseVR', createStepName='Initial', region=base, u2=0.0, ur3=0.0)
    amp_data = tuple((float(t[i]), float(a[i])) for i in range(len(t)))
    model.TabularAmplitude(name='AccAmp', data=amp_data, timeSpan=STEP)
    model.AccelerationBC(name='BaseAcc', createStepName='Step-EQ', region=base, a1=1.0, amplitude='AccAmp')

    freq = int(job_cfg['history_freq'])
    model.HistoryOutputRequest(name='H-BaseRF', createStepName='Step-EQ',
                               variables=('RF1', 'U1', 'A1'), region=base, frequency=freq)
    for k in range(1, ns + 1):
        model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName='Step-EQ',
                                   variables=('U1', 'V1', 'A1'),
                                   region=asm.instances[frame_inst].sets['FLOOR_%d' % k], frequency=freq)
    log_step(logger, u'[fixed] 刚性基础框架已建, 输入=自由场地表运动, 时长=%.2fs', tp)
    return model_name


def write_meta(acc_name, t, a, dt, logger):
    """写 ssi_meta.json：几何/材料/手算锚点。"""
    H = float(soil_cfg['depth']); cs = float(soil_cfg['Vs'])
    ns = int(frame_cfg['n_story'])
    meta = {
        'script': _DEFAULT_SCRIPT_NAME, 'acc_record': acc_name, 'dt': dt,
        'duration': float(t[-1]), 'pga': float(np.max(np.abs(a))),
        'soil': dict(soil_cfg), 'frame': dict(frame_cfg), 'frame_material': dict(frame_material_cfg),
        'site_f0_Hz': cs / (4.0 * H), 'site_T0_s': 4.0 * H / cs,
        'T_fixed_from_step1_s': 0.5,   # step1 固定基础 T1（周期延长对比基准）
        'inst_soil': 'Soil-1', 'inst_frame': 'Frame-1', 'n_story': ns,
    }
    with open('ssi_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log_step(logger, u'ssi_meta.json 已写: 场地 f0=%.2fHz T0=%.2fs, 对比 T_fixed=0.5s',
             meta['site_f0_Hz'], meta['site_T0_s'])


def submit(model_name, logger):
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model_name, description='SSI step2a',
            type=ANALYSIS, memory=job_cfg['memory_percent'], memoryUnits=PERCENTAGE,
            getMemoryFromAnalysis=True, numCpus=job_cfg['num_cpus'], numDomains=job_cfg['num_cpus'],
            multiprocessingMode=DEFAULT, numGPUs=0, echoPrint=OFF, modelPrint=OFF,
            contactPrint=OFF, historyPrint=OFF)
    mdb.save()
    t0 = time.time()
    log_step(logger, u'%s 提交中...', job_name)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, u'%s 完成, 耗时=%.2fs', job_name, time.time() - t0)


def main():
    logger = log_step('frame_ssi.log')
    t0 = time.time()
    try:
        log_step(logger, u'====== Step-2a SSI 建模开始 ======')
        acc_name, t, a, dt = find_acc_txt(logger)
        mdb.saveAs(pathName='ssi_step2a.cae')
        write_meta(acc_name, t, a, dt, logger)

        # 依赖顺序：freefield 须先解算，fixed 才能提取其地表运动作输入(去耦法)
        order = [s for s in ['freefield', 'ssi', 'fixed'] if s in run_cfg['scenes']]
        do_submit = job_cfg.get('submit', True)
        for scene in order:
            log_step(logger, u'------ 场景: %s ------', scene)
            if scene == 'fixed':
                if not do_submit or not os.path.isfile('job-freefield.odb'):
                    log_step(logger, u'[fixed] 需 freefield 先解算(submit=True 且 odb 存在)，跳过')
                    continue
                ft, fa = extract_freefield_surface_acc(logger)
                nm = build_fixed_scene(ft, fa, float(ft[1] - ft[0]), logger)
            else:
                nm = build_scene(scene, acc_name, t, a, dt, logger)
            mdb.save()
            if do_submit:
                submit(nm, logger)
        log_step(logger, u'完成。后处理: abaqus cae noGUI=postproc_ssi_v1.py')
        log_step(logger, u'====== 全部完成, 总耗时=%.2fs ======', time.time() - t0)
    except Exception as exc:
        log_step(logger, u'脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
