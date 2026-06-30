# -*- coding: utf-8 -*-
"""
Step-2b：框架坐【坡顶】+ 复用 Multi 坡面/分层/粘弹性吸收边界/斜入射引擎 —— SSI 集成
================================================================================
TSSI 路线第二步(b)。在 step2a(平层 SSI 已验)基础上，把平层土块换成真实坡面：
    import Modeling/Multi/VAB_oblique_multilayer_nonlinear_v3 复用其【已验证】波动引擎
    (上土下岩分层 + 粘弹性吸收边界 + 斜入射 SV 等效力)，框架坐【坡顶】并 Tie 耦合。

与 step2a 区别：
    step2a 平层+周期侧边+刚性基底(自建)；step2b 坡面+吸收边界+斜入射(复用 Multi)。
    坡面散射才需要吸收边界——这正是 Multi 引擎的价值，故 import 复用而非重写
    (违反"自包含"惯例是有意的：重写 1000 行已验波动物理风险太高)。

场景：
    freefield : 仅坡面土(Multi 标准) → 坡顶自由场基准
    ssi       : 框架坐坡顶 Tie 耦合 → 含 SSI
    (fixed 去耦对照留 step2b-2，需 freefield 解算后提取坡顶自由场运动)

缩小配置(快速调通集成用)：坡 H~60m、总长 300m、网格 5m。调通后可放大到论文尺度。

运行：
    abaqus cae noGUI=frame_ssi_slope_v1.py
工作目录须有一条加速度记录 .txt。

Py2.7 兼容。
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
import sys
import json
import time
import logging
import traceback

# ── 导入 Multi 波动引擎（坡面+边界+斜入射）──
# Abaqus 内核常无 __file__，故多路径兜底定位 Modeling/Multi
_CANDIDATES = []
_f = globals().get('__file__')
if _f:
    _CANDIDATES.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(_f)), '..', 'Multi')))
_CANDIDATES.append(os.path.normpath(os.path.join(os.getcwd(), '..', 'Multi')))          # cwd=Hybrid 时
_CANDIDATES.append(os.path.normpath(os.path.join(os.getcwd(), '..', '..', '..', 'Modeling', 'Multi')))  # cwd=test/Hybrid/test_* 时
_CANDIDATES.append(r'C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi')           # 绝对兜底
_MULTI_DIR = next((d for d in _CANDIDATES if os.path.isdir(d)), _CANDIDATES[-1])
if _MULTI_DIR not in sys.path:
    sys.path.insert(0, _MULTI_DIR)
import VAB_oblique_multilayer_nonlinear_v3 as multi   # noqa: E402


# ==========================================================
#  配置（缩小坡面 + 框架）
# ==========================================================

# 坡面土体（Multi 口径：基岩 + 覆盖层"上土下岩"）；缩小尺寸以快速调通集成
soil_material_cfg = {
    'angle': 15,                         # SV 斜入射角(度)
    'surface_geometry': 'horizontal',
    'bedrock': {'elastic_modulus': 26e9, 'poisson_ratio': 0.3, 'density': 2500},  # Vs≈2000
    'layers': [
        {'name': 'overlying', 'velocity_ratio': 4.0, 'poisson_ratio': 0.35, 'density': 1900},  # 覆盖层 Vs≈500，厚度由几何定
    ],
}
soil_geometry_cfg = {
    'H_minus_h': 30.0, 'i': 45.0, 'h_over_H': 0.5,
    'total_L': 300.0, 'left_flat': 120.0, 'bedrock_thickness': 40.0,
}
# 推导：H=H_minus_h/(1-h_over_H)=60; H_upper=bedrock+H=100; H_lower=bedrock+h=70; w_slope=30; 坡顶平台 x∈[0,120]

mesh_cfg = {'size': 5.0, 'auto': False, 'graded': False, 'elem': 'CPE4R'}

# 框架（同 step1/2a；坐坡顶，右缘贴坡肩 x=left_flat）
frame_cfg = {
    'n_story': 5, 'n_bay': 3, 'story_height': 3.0, 'bay_width': 6.0,
    'column': {'width': 0.5, 'depth': 0.5}, 'beam': {'width': 0.3, 'depth': 0.6},
    'floor_mass': 5.0e4,
}
frame_material_cfg = {
    'name': 'Concrete_C30', 'E': 30.0e9, 'nu': 0.2, 'density': 10.0,
    'damping_ratio': 0.05, 'f1': 1.0, 'f2': 5.0,
}

job_cfg = {'num_cpus': 4, 'memory_percent': 90, 'history_freq': 1, 'submit': True}
run_cfg = {'scenes': ['freefield', 'ssi', 'fixed']}  # freefield=仅坡土 / ssi=框架坐坡顶 / fixed=坡顶刚性(输入坡顶自由场,去耦对照)


# ==========================================================
#  日志
# ==========================================================

def log_step(logger=None, message=None, *args):
    if not hasattr(log_step, '_logger'):
        log_filename = logger if isinstance(logger, str) else 'frame_ssi_slope.log'
        _logger = logging.getLogger('frame_ssi_slope')
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


def find_acc_txt(logger=None):
    cwd = os.getcwd()
    txts = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.txt')])
    if not txts:
        raise IOError(u'当前目录无 .txt 加速度记录: %s' % cwd)
    f = txts[0]
    data = np.loadtxt(f)
    t = data[:, 0].astype(float); a = data[:, 1].astype(float)
    dt = float(t[1] - t[0])
    if logger:
        log_step(logger, u'加速度记录: %s, dt=%.4fs, 时长=%.2fs, |a|max=%.3f', f, dt, float(t[-1]), float(np.max(np.abs(a))))
    return f, t, a, dt


def rayleigh_coeffs(xi, f1, f2):
    w1 = 2.0 * math.pi * f1; w2 = 2.0 * math.pi * f2
    return 2.0 * xi * w1 * w2 / (w1 + w2), 2.0 * xi / (w1 + w2)


# ==========================================================
#  框架 Part（局部基底 y=0；同 step2a）
# ==========================================================

def build_frame_part(model, logger):
    nb = int(frame_cfg['n_bay']); ns = int(frame_cfg['n_story'])
    bw = float(frame_cfg['bay_width']); sh = float(frame_cfg['story_height'])
    xs = [j * bw for j in range(nb + 1)]; ys = [k * sh for k in range(ns + 1)]; z = 0.0
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
    mat.Density(table=((mc['density'],),))
    a_r, b_r = rayleigh_coeffs(mc['damping_ratio'], mc['f1'], mc['f2'])
    mat.Damping(alpha=a_r, beta=b_r)
    col = frame_cfg['column']; bm = frame_cfg['beam']
    model.RectangularProfile(name='ColProf', a=col['width'], b=col['depth'])
    model.RectangularProfile(name='BeamProf', a=bm['width'], b=bm['depth'])
    model.BeamSection(name='ColSec', profile='ColProf', material=mc['name'], integration=DURING_ANALYSIS, poissonRatio=mc['nu'])
    model.BeamSection(name='BeamSec', profile='BeamProf', material=mc['name'], integration=DURING_ANALYSIS, poissonRatio=mc['nu'])
    col_mids = [((x, (ys[k] + ys[k + 1]) / 2.0, z),) for x in xs for k in range(ns)]
    beam_mids = [(((xs[j] + xs[j + 1]) / 2.0, ys[k], z),) for k in range(1, ns + 1) for j in range(nb)]
    part.Set(edges=part.edges.findAt(*col_mids), name='COLS')
    part.Set(edges=part.edges.findAt(*beam_mids), name='BEAMS')
    part.SectionAssignment(region=part.sets['COLS'], sectionName='ColSec')
    part.SectionAssignment(region=part.sets['BEAMS'], sectionName='BeamSec')
    part.assignBeamSectionOrientation(region=part.Set(edges=part.edges, name='ALL_E'), method=N1_COSINES, n1=(0.0, 0.0, -1.0))
    part.Set(vertices=part.vertices.findAt(*[((x, ys[0], z),) for x in xs]), name='BASE')
    floor_full = {}
    for k in range(1, ns + 1):
        part.Set(vertices=part.vertices.findAt(((xs[0], ys[k], z),)), name='FLOOR_%d' % k)
        part.Set(vertices=part.vertices.findAt(*[((x, ys[k], z),) for x in xs]), name='FLOORALL_%d' % k)
        floor_full[k] = ('FLOORALL_%d' % k, len(xs))
    part.seedEdgeByNumber(edges=part.edges, number=1, constraint=FIXED)
    part.setElementType(regions=(part.edges,), elemTypes=(mesh.ElemType(elemCode=B21, elemLibrary=STANDARD),))
    part.generateMesh()
    log_step(logger, u'框架 Part: B21, 层=%d 跨=%d, 单元=%d', ns, nb, len(part.elements))
    return part, floor_full, ns


# ==========================================================
#  用 Multi 引擎建坡面土 + 边界 + 斜入射
# ==========================================================

def build_soil_and_oblique(model_name, acc_file, t, a, dt, logger):
    """调 Multi 建坡面土(create_model) → 建步 → VAB_oblique(边界+斜入射力)。返回 (site, geom, part, inst)。"""
    # Site/Geometry（Multi 口径）
    site, fixed_th = multi.build_site(soil_material_cfg, soil_geometry_cfg)
    geom = multi.make_geometry(
        total_L=soil_geometry_cfg['total_L'], H_minus_h=soil_geometry_cfg['H_minus_h'],
        i=soil_geometry_cfg['i'], h_over_H=soil_geometry_cfg['h_over_H'],
        left_flat=soil_geometry_cfg['left_flat'], bedrock_thickness=soil_geometry_cfg['bedrock_thickness'],
        fixed_thicknesses=fixed_th)
    # 阻尼解析（fc 自动估计）
    fc_est = multi._estimate_dominant_freq(a, dt)
    damping = multi._resolve_damping(multi.damping_cfg, fc_est)
    log_step(logger, u'坡面: H_upper=%.0f w_slope=%.0f 坡顶平台[0,%.0f]; fc=%.2fHz',
             geom.H_upper, geom.w_slope, geom.left_flat, fc_est)

    # create_model 建 Model-1 土体（含边界节点集 Left/Right/Bottom_boundary + TOP_SURFACE）
    base_model, part_name, inst_name = multi.create_model(
        site=site, geom=geom, mesh_size=mesh_cfg['size'], cae_name='ssi_slope.cae', logger=logger,
        damping=damping, surface_geometry=soil_material_cfg['surface_geometry'],
        elem_name=mesh_cfg['elem'], mesh_cfg=mesh_cfg, fc=fc_est)
    # Multi 固定建在 'Model-1'，改名为目标场景名
    if base_model != model_name:
        mdb.models.changeKey(fromName=base_model, toName=model_name)
    return site, geom, part_name, inst_name, damping, fc_est


def add_step_and_oblique(model_name, site, geom, part_name, inst_name, acc_file, damping, fc_est, t, logger):
    """建隐式动力步 + VAB_oblique 施加粘弹性边界与斜入射 SV 等效力。"""
    model = mdb.models[model_name]
    tp = float(t[-1]); dt = float(t[1] - t[0])
    step_name = multi.DEFAULT_STEP_NAME
    model.ImplicitDynamicsStep(name=step_name, previous='Initial',
                               timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=dt,
                               maxNumInc=1000000, nlgeom=OFF, application=TRANSIENT_FIDELITY)
    multi.VAB_oblique(site=site, geom=geom, angle=soil_material_cfg['angle'],
                      model_name=model_name, part_name=part_name, inst_name=inst_name,
                      acc_file=acc_file, step_name=step_name, logger=logger,
                      tcfg={'check': False}, fc_used=fc_est, ffcfg=multi.freefield_cfg,
                      damping=damping, surface_geometry=soil_material_cfg['surface_geometry'],
                      critical_angle_check=False, elem_name=mesh_cfg['elem'])
    return step_name


def add_frame_on_crest(model_name, geom, soil_part_name, soil_inst_name, logger):
    """框架坐坡顶(右缘贴坡肩) + Tie 基底到坡顶土面 + 楼层集中质量。返回 (frame_inst, ns)。"""
    model = mdb.models[model_name]
    asm = model.rootAssembly
    frame, floor_full, ns = build_frame_part(model, logger)
    frame_inst = 'Frame-1'
    asm.Instance(name=frame_inst, part=frame, dependent=ON)
    fw = int(frame_cfg['n_bay']) * float(frame_cfg['bay_width'])
    x_off = float(geom.left_flat) - fw        # 右缘贴坡肩 x=left_flat
    asm.translate(instanceList=(frame_inst,), vector=(x_off, geom.H_upper, 0.0))
    m_total = float(frame_cfg['floor_mass'])
    for k in range(1, ns + 1):
        nm, cnt = floor_full[k]
        asm.engineeringFeatures.PointMassInertia(
            name='Mass_%d' % k, region=asm.instances[frame_inst].sets[nm], mass=m_total / float(cnt))
    # 坡顶平台土面 [0,left_flat] 建 Tie 主面
    soil_part = model.parts[soil_part_name]
    crest_edge = soil_part.edges.findAt(((geom.left_flat * 0.5, geom.H_upper, 0.0),))
    soil_part.Surface(side1Edges=crest_edge, name='CREST_SURF')
    model.Tie(name='FrameSoil', master=asm.instances[soil_inst_name].surfaces['CREST_SURF'],
              slave=asm.instances[frame_inst].sets['BASE'],
              positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=OFF, thickness=ON)
    log_step(logger, u'框架坐坡顶(x_off=%.1f,y=%.0f) 并 Tie 耦合', x_off, geom.H_upper)
    return frame_inst, ns


def add_outputs(model_name, step_name, soil_inst_name, frame_inst, ns, geom, logger):
    """历史输出：坡顶土面参考节点 + (ssi)框架各层。"""
    model = mdb.models[model_name]
    asm = model.rootAssembly
    freq = int(job_cfg['history_freq'])
    # 坡肩附近土面参考节点（TOP_SURFACE 中 x 最接近 left_flat 者）
    top = asm.instances[soil_inst_name].sets['TOP_SURFACE']
    crest_node = min(top.nodes, key=lambda n: abs(n.coordinates[0] - geom.left_flat))
    asm.Set(name='CREST_REF', nodes=asm.instances[soil_inst_name].nodes.sequenceFromLabels([crest_node.label]))
    model.HistoryOutputRequest(name='H-Crest', createStepName=step_name,
                               variables=('U1', 'A1'), region=asm.sets['CREST_REF'], frequency=freq)
    if frame_inst:
        for k in range(1, ns + 1):
            model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName=step_name,
                                       variables=('U1', 'V1', 'A1'),
                                       region=asm.instances[frame_inst].sets['FLOOR_%d' % k], frequency=freq)
    log_step(logger, u'[%s] 输出已配: 坡顶参考节点(x=%.0f) + %s',
             model_name, crest_node.coordinates[0], (u'%d层框架' % ns) if frame_inst else u'仅自由场')


def submit(model_name, logger):
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model_name, description='SSI slope step2b',
            type=ANALYSIS, memory=job_cfg['memory_percent'], memoryUnits=PERCENTAGE,
            getMemoryFromAnalysis=True, numCpus=job_cfg['num_cpus'], numDomains=job_cfg['num_cpus'],
            multiprocessingMode=DEFAULT, numGPUs=0, echoPrint=OFF, modelPrint=OFF, contactPrint=OFF, historyPrint=OFF)
    mdb.save()
    t0 = time.time()
    log_step(logger, u'%s 提交中...', job_name)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, u'%s 完成, 耗时=%.2fs', job_name, time.time() - t0)


def build_scene(scene, acc_file, t, a, dt, logger):
    log_step(logger, u'------ 场景: %s ------', scene)
    site, geom, pn, inn, damping, fc = build_soil_and_oblique(scene, acc_file, t, a, dt, logger)
    frame_inst, ns = (None, int(frame_cfg['n_story']))
    if scene == 'ssi':
        frame_inst, ns = add_frame_on_crest(scene, geom, pn, inn, logger)
    step_name = add_step_and_oblique(scene, site, geom, pn, inn, acc_file, damping, fc, t, logger)
    add_outputs(scene, step_name, inn, frame_inst, ns, geom, logger)
    return scene, geom


def extract_crest_freefield_acc(logger):
    """从已解算 job-freefield.odb 提取坡顶参考节点(CREST_REF)绝对加速度 A1，返回 (t,a)。

    去耦法关键：坡顶刚性(fixed)基础须输入【无结构】坡顶自由场运动，使 fixed 与 ssi 唯一差异 = SSI。
    """
    from odbAccess import openOdb
    odb = openOdb('job-freefield.odb', readOnly=True)
    nset = odb.rootAssembly.nodeSets['CREST_REF']  # 装配级集
    nd = nset.nodes[0] if not hasattr(nset.nodes[0], '__len__') else nset.nodes[0][0]  # 取首节点(兼容分组)
    eq = odb.steps[list(odb.steps.keys())[0]]  # 首个分析步(对步名鲁棒)
    hr = eq.historyRegions['Node %s.%d' % (nd.instanceName, nd.label)]
    data = np.array(hr.historyOutputs['A1'].data, dtype=float)
    odb.close()
    t = data[:, 0]; a = data[:, 1]
    log_step(logger, u'坡顶自由场加速度已提取: 点数=%d, |a|max=%.3f m/s² (去耦法刚性基础输入)',
             len(t), float(np.max(np.abs(a))))
    return t, a


def build_fixed_scene(t, a, dt, logger):
    """建坡顶刚性基础框架(fixed)：框架单体、柱脚嵌固、基底输入【坡顶自由场加速度】。

    去耦法对照：与 ssi 同框架、同坡顶自由场输入，差异仅在有无土柔度(SSI)。
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
    for k in range(1, ns + 1):  # 楼层集中质量(与 ssi 同)
        nm, cnt = floor_full[k]
        asm.engineeringFeatures.PointMassInertia(
            name='Mass_%d' % k, region=asm.instances[frame_inst].sets[nm], mass=m_total / float(cnt))
    tp = float(t[-1])
    model.ImplicitDynamicsStep(name='Step-EQ', previous='Initial',
                               timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=dt,
                               maxNumInc=1000000, nlgeom=OFF, application=TRANSIENT_FIDELITY)
    base = asm.instances[frame_inst].sets['BASE']
    model.DisplacementBC(name='BaseVR', createStepName='Initial', region=base, u2=0.0, ur3=0.0)  # 柱脚竖向/转动约束
    amp_data = tuple((float(t[i]), float(a[i])) for i in range(len(t)))
    model.TabularAmplitude(name='AccAmp', data=amp_data, timeSpan=STEP)
    model.AccelerationBC(name='BaseAcc', createStepName='Step-EQ', region=base, a1=1.0, amplitude='AccAmp')  # 基底=坡顶自由场水平加速度
    freq = int(job_cfg['history_freq'])
    model.HistoryOutputRequest(name='H-BaseRF', createStepName='Step-EQ',
                               variables=('RF1', 'U1', 'A1'), region=base, frequency=freq)
    for k in range(1, ns + 1):
        model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName='Step-EQ',
                                   variables=('U1', 'V1', 'A1'),
                                   region=asm.instances[frame_inst].sets['FLOOR_%d' % k], frequency=freq)
    log_step(logger, u'[fixed] 坡顶刚性基础框架已建, 输入=坡顶自由场运动, 时长=%.2fs', tp)
    return model_name


def write_meta(acc_name, t, a, dt, geom, logger):
    meta = {'script': 'frame_ssi_slope_v1.py', 'acc_record': acc_name, 'dt': dt,
            'duration': float(t[-1]), 'pga': float(np.max(np.abs(a))),
            'angle': soil_material_cfg['angle'], 'left_flat': geom.left_flat,
            'H_upper': geom.H_upper, 'n_story': int(frame_cfg['n_story']),
            'story_height': float(frame_cfg['story_height']), 'floor_mass': float(frame_cfg['floor_mass']),
            'inst_soil': 'Part-1-1', 'inst_frame': 'Frame-1', 'T_fixed_step1': 0.5}
    with open('ssi_slope_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)


def main():
    logger = log_step('frame_ssi_slope.log')
    t0 = time.time()
    try:
        log_step(logger, u'====== Step-2b 坡顶 SSI(复用 Multi 引擎) 开始 ======')
        acc_file, t, a, dt = find_acc_txt(logger)
        do_submit = job_cfg.get('submit', False)
        geom_last = None
        order = [s for s in ['freefield', 'ssi', 'fixed'] if s in run_cfg['scenes']]  # fixed 须在 freefield 后(去耦法取其坡顶运动)
        for scene in order:
            if scene == 'fixed':
                if not do_submit or not os.path.isfile('job-freefield.odb'):
                    log_step(logger, u'[fixed] 需 freefield 先解算(submit=True 且 job-freefield.odb 存在)，跳过')
                    continue
                ft, fa = extract_crest_freefield_acc(logger)
                build_fixed_scene(ft, fa, float(ft[1] - ft[0]), logger)
                mdb.save()
                submit('fixed', logger)
            else:
                _, geom_last = build_scene(scene, acc_file, t, a, dt, logger)
                mdb.save()
                if do_submit:
                    submit(scene, logger)
        write_meta(acc_file, t, a, dt, geom_last, logger)
        mdb.save()
        log_step(logger, u'====== 完成(submit=%s), 总耗时=%.2fs ======', do_submit, time.time() - t0)
    except Exception as exc:
        log_step(logger, u'脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
