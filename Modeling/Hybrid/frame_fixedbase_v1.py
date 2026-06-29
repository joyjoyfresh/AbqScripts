# -*- coding: utf-8 -*-
"""
Step-1：二维固定基础多层框架 —— 抗震建模 + 后处理链路验证脚本
================================================================
目的（TSSI 路线第一步）：在【脱离土体/SSI】的隔离环境里跑通
    梁柱建模 -> 模态(T1) -> 基底加速度时程 -> 抗震指标提取
并用手算锚点验证"建模 + 后处理"两半是否正确，再往上叠 SSI / CDP。

与 Modeling/Multi/VAB_oblique_multilayer_nonlinear_v3.py 的关系：
    本脚本【不复用】土体/人工边界/斜入射；只借鉴其风格（顶部配置 dict、
    log_step 日志、find_acc_txt 输入、case_config.json 可选注入）。
    后续 step2 会把这里验好的框架"坐"到那套土体模型坡顶（共节点/接触）。

运行（Abaqus CAE，无界面）：
    abaqus cae noGUI=frame_fixedbase_v1.py
工作目录须有一条加速度记录 .txt（两列：时间, 加速度 m/s^2）。

Py2.7 兼容（Abaqus 内核）：无 __file__ 兜底；不用 f-string；open 不传 encoding。
"""

from abaqus import *
from abaqusConstants import *
from abaqus import mdb
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
#  配置参数（默认值；工况目录下 case_config.json 可覆盖）
# ==========================================================

# 框架几何与质量
frame_cfg = {
    'n_story': 5,            # 层数（不含基底）
    'n_bay': 3,             # 跨数
    'story_height': 3.0,    # 层高 (m)
    'bay_width': 6.0,       # 跨度 (m)
    'column': {'width': 0.5, 'depth': 0.5},  # 柱截面：width=出平面宽, depth=平面内高(主控弯曲)
    'beam':   {'width': 0.3, 'depth': 0.6},  # 梁截面：同上
    'floor_mass': 5.0e4,    # 每层集中质量 (kg)，含恒+活折算；均分到该层各梁柱节点
}

# 结构材料（先弹性——step1 验后处理用，需有手算锚点；CDP 留到 step3）
material_cfg = {
    'name': 'Concrete_C30',
    'E': 30.0e9,            # 弹性模量 (Pa)，C30≈30 GPa
    'nu': 0.2,             # 泊松比
    'density': 10.0,       # 梁单元密度 (kg/m^3)。取微小正则化值：纯集中质量会使旋转 DOF 无质量、
                          #   SIM-Lanczos 特征值求解病态(报 rigid body modes)；10 给所有 DOF 非零质量，
                          #   而结构总自重(~310kg)相对集中质量(250t)仅 0.12%，Σm·a 基底剪力校核仍干净。
                          #   如要真实构件自重，设 2500 并相应下调 floor_mass（此时 Σm·a 校核需含结构质量）。
}

# 结构阻尼（瑞利：α 质量比例 + β 刚度比例，两端 ξ 相等≈恒定阻尼）
damping_cfg = {
    'ratio': 0.05,         # 目标阻尼比 ξ（混凝土结构常取 0.05）
    'f1': 1.0,             # 瑞利锚定下限频率 (Hz)——首跑后改为模态 1 阶频率
    'f2': 5.0,             # 瑞利锚定上限频率 (Hz)——首跑后改为模态高阶(如 3 阶)频率
}

# 分析步 / 作业
job_cfg = {
    'n_eigen': 6,          # 模态步提取阶数（验 T1 用）
    'num_cpus': 4,
    'memory_percent': 90,
    'history_freq': 1,     # 历史输出频率（每增量步）
    'submit': True,        # True=建模后直接提交；False=只建模存 cae
}

# 网格
mesh_cfg = {
    'elems_per_member': 1,  # 每根梁/柱单元数。集中质量模型须=1（避免无质量中间节点导致特征值病态）；
                            #   要细分须改用结构密度(material_cfg.density>0)提供分布质量
    'elem': 'B21',          # 平面线性梁(Timoshenko)：1单元/构件不生成内部节点→所有节点皆有集中质量。
                            #   B22/B23 会生成无质量内部节点，与纯集中质量模型冲突；step3 上非线性沿用 B21+纤维截面
}


# ==========================================================
#  日志（精简版，沿用主脚本 log_step 习惯）
# ==========================================================

_DEFAULT_SCRIPT_NAME = 'frame_fixedbase_v1.py'


def _script_dir():
    f = globals().get('__file__')
    if f:
        return os.path.dirname(os.path.abspath(f))
    return os.getcwd()


def log_step(logger=None, message=None, *args):
    """首次调用初始化日志器；后续带累计用时输出。"""
    if not hasattr(log_step, '_logger'):
        log_filename = logger if isinstance(logger, str) else 'frame_fixedbase.log'
        _logger = logging.getLogger('frame_fb')
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


# ==========================================================
#  输入与配置注入
# ==========================================================

def find_acc_txt(logger=None):
    """读取 cwd 下第一条两列加速度记录 .txt，返回 (文件名, t数组, acc数组, dt)。"""
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


def load_case_config(logger=None):
    """若 cwd 有 case_config.json 则覆盖默认配置（frame/material/damping/job/mesh 五类，均可选）。"""
    global frame_cfg, material_cfg, damping_cfg, job_cfg, mesh_cfg
    path = os.path.join(os.getcwd(), 'case_config.json')
    if not os.path.isfile(path):
        if logger:
            log_step(logger, u'未发现 case_config.json，使用脚本内默认配置')
        return
    with open(path, 'r') as fh:
        cfg = _ensure_str(json.load(fh))
    for key, target in (('frame_cfg', frame_cfg), ('material_cfg', material_cfg),
                        ('damping_cfg', damping_cfg), ('job_cfg', job_cfg), ('mesh_cfg', mesh_cfg)):
        if key in cfg and isinstance(cfg[key], dict):
            target.update(cfg[key])
    if logger:
        log_step(logger, u'已注入 case_config.json: %s', ', '.join(sorted(cfg.keys())))


# ==========================================================
#  物理：瑞利阻尼系数
# ==========================================================

def rayleigh_coeffs(xi, f1, f2):
    """两端 ξ 相等的瑞利系数 (alpha, beta)：α=2ξω1ω2/(ω1+ω2)，β=2ξ/(ω1+ω2)。"""
    w1 = 2.0 * math.pi * f1
    w2 = 2.0 * math.pi * f2
    alpha = 2.0 * xi * w1 * w2 / (w1 + w2)
    beta = 2.0 * xi / (w1 + w2)
    return alpha, beta


# ==========================================================
#  建模
# ==========================================================

def build_frame_model(logger):
    """建立二维固定基础框架：几何(线框)、梁截面、集中质量、网格、节点集。返回 (model, inst_name)。"""
    nb = int(frame_cfg['n_bay'])
    ns = int(frame_cfg['n_story'])
    bw = float(frame_cfg['bay_width'])
    sh = float(frame_cfg['story_height'])

    model = mdb.models['Model-1'] if 'Model-1' in mdb.models else mdb.Model(name='Model-1')
    # 列(x)与层(y)坐标
    xs = [j * bw for j in range(nb + 1)]            # x: 0..nb*bw
    ys = [k * sh for k in range(ns + 1)]            # y: 0(基底)..ns*sh

    # ---- 线框几何（柱 + 梁；端点重合自动生成节点顶点）----
    sheet = max(xs[-1], ys[-1]) * 2.0
    sk = model.ConstrainedSketch(name='__frame__', sheetSize=sheet)
    for x in xs:                                    # 柱：每条柱线逐层分段（保证每层有顶点）
        for k in range(ns):
            sk.Line(point1=(x, ys[k]), point2=(x, ys[k + 1]))
    for k in range(1, ns + 1):                      # 梁：每层每跨一段（基底无梁）
        for j in range(nb):
            sk.Line(point1=(xs[j], ys[k]), point2=(xs[j + 1], ys[k]))
    part = model.Part(name='Frame', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseWire(sketch=sk)
    del model.sketches['__frame__']
    log_step(logger, u'框架线框已建: 跨=%d 层=%d, 柱段=%d 梁段=%d', nb, ns, len(xs) * ns, ns * nb)

    # ---- 材料 ----
    mat = model.Material(name=material_cfg['name'])
    mat.Elastic(table=((material_cfg['E'], material_cfg['nu']),))
    if material_cfg['density'] > 0:
        mat.Density(table=((material_cfg['density'],),))
    alpha, beta = rayleigh_coeffs(damping_cfg['ratio'], damping_cfg['f1'], damping_cfg['f2'])
    mat.Damping(alpha=alpha, beta=beta)
    log_step(logger, u'材料 %s: E=%.2e Pa, ν=%.2f, 瑞利 α=%.4f β=%.6f (ξ=%.3f @ %.2f/%.2f Hz)',
             material_cfg['name'], material_cfg['E'], material_cfg['nu'], alpha, beta,
             damping_cfg['ratio'], damping_cfg['f1'], damping_cfg['f2'])

    # ---- 梁截面（矩形；integration=DURING_ANALYSIS 为 step3 纤维/塑性留口）----
    col = frame_cfg['column']; bm = frame_cfg['beam']
    model.RectangularProfile(name='ColProf', a=col['width'], b=col['depth'])  # a=出平面, b=平面内(主控弯曲)
    model.RectangularProfile(name='BeamProf', a=bm['width'], b=bm['depth'])
    # DURING_ANALYSIS：截面积分点处按材料本构积分（需材料 Elastic，已定义）；step3 上塑性时换纤维截面沿用此积分
    model.BeamSection(name='ColSec', profile='ColProf', material=material_cfg['name'],
                      integration=DURING_ANALYSIS, poissonRatio=material_cfg['nu'])
    model.BeamSection(name='BeamSec', profile='BeamProf', material=material_cfg['name'],
                      integration=DURING_ANALYSIS, poissonRatio=material_cfg['nu'])

    z = 0.0
    # 柱/梁边用解析中点 findAt 归类（坐标全已知，比按 index/label 稳）
    col_mids = [((x, (ys[k] + ys[k + 1]) / 2.0, z),) for x in xs for k in range(ns)]
    beam_mids = [(((xs[j] + xs[j + 1]) / 2.0, ys[k], z),) for k in range(1, ns + 1) for j in range(nb)]
    part.Set(edges=part.edges.findAt(*col_mids), name='COLS')
    part.Set(edges=part.edges.findAt(*beam_mids), name='BEAMS')
    part.SectionAssignment(region=part.sets['COLS'], sectionName='ColSec')
    part.SectionAssignment(region=part.sets['BEAMS'], sectionName='BeamSec')
    # 梁单元方向：2D 平面梁 n1 取出平面 (0,0,-1)
    part.assignBeamSectionOrientation(region=part.Set(edges=part.edges, name='ALL_E'),
                                      method=N1_COSINES, n1=(0.0, 0.0, -1.0))
    log_step(logger, u'截面分配: 柱段=%d, 梁段=%d', len(col_mids), len(beam_mids))

    # ---- 几何顶点集（基底 / 各层参考节点 / 各层全节点），均用解析坐标 findAt ----
    base_pts = [((x, ys[0], z),) for x in xs]
    part.Set(vertices=part.vertices.findAt(*base_pts), name='BASE')
    # 参考柱线 x=xs[0]：每层一个顶点，供层间位移角/楼层响应提取
    for k in range(1, ns + 1):
        part.Set(vertices=part.vertices.findAt(((xs[0], ys[k], z),)), name='FLOOR_%d' % k)
    # 各层全节点集（施加集中质量）
    floor_full = {}
    for k in range(1, ns + 1):
        fpts = [((x, ys[k], z),) for x in xs]
        nm = 'FLOORALL_%d' % k
        part.Set(vertices=part.vertices.findAt(*fpts), name=nm)
        floor_full[k] = (nm, len(xs))

    # ---- 网格 ----
    nseed = int(mesh_cfg.get('elems_per_member', 1))
    part.seedEdgeByNumber(edges=part.edges, number=nseed, constraint=FIXED)  # 每构件精确 nseed 段（保证无意外中间节点）
    _ELEM = {'B21': B21, 'B22': B22, 'B23': B23}
    elem_code = _ELEM.get(mesh_cfg.get('elem', 'B23'), B23)
    part.setElementType(regions=(part.edges,), elemTypes=(mesh.ElemType(elemCode=elem_code, elemLibrary=STANDARD),))
    part.generateMesh()
    log_step(logger, u'网格: %s, 每构件%d单元, 单元=%d, 节点=%d',
             mesh_cfg.get('elem', 'B23'), nseed, len(part.elements), len(part.nodes))

    # ---- 装配 ----
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)
    inst_name = 'Frame-1'
    asm.Instance(name=inst_name, part=part, dependent=ON)

    # ---- 楼层集中质量（每层节点各 floor_mass/节点数）----
    m_total = float(frame_cfg['floor_mass'])
    for k in range(1, ns + 1):
        nm, cnt = floor_full[k]
        m_node = m_total / float(cnt)
        region = asm.instances[inst_name].sets[nm]
        asm.engineeringFeatures.PointMassInertia(name='Mass_%d' % k, region=region, mass=m_node)
    log_step(logger, u'楼层集中质量: 每层 %.0f kg（%d 层，按节点均分）', m_total, ns)

    return model, inst_name


# ==========================================================
#  分析步 / 边界 / 输出
# ==========================================================

def setup_steps_and_loads(model, inst_name, acc_name, t, a, dt, logger):
    """模态步 + 隐式动力步；基底嵌固 + 水平加速度输入；历史输出(基底RF1, 各层U1/V1/A1)。"""
    ns = int(frame_cfg['n_story'])
    asm = model.rootAssembly
    base = asm.instances[inst_name].sets['BASE']

    # ---- 模态步（验 T1）----
    model.FrequencyStep(name='Step-Modal', previous='Initial',
                        numEigen=int(job_cfg['n_eigen']), normalization=MASS)

    # ---- 基底约束：竖向+转动恒定嵌固；水平在模态固定、地震步释放后由加速度 BC 驱动 ----
    model.DisplacementBC(name='BaseVR', createStepName='Initial', region=base,
                         u2=0.0, ur3=0.0)                       # 竖向/转动嵌固（全程）
    bc_h = model.DisplacementBC(name='BaseH', createStepName='Initial', region=base, u1=0.0)  # 水平：模态用

    # ---- 隐式动力步（地震时程）----
    tp = float(t[-1])
    model.ImplicitDynamicsStep(name='Step-EQ', previous='Step-Modal',
                               timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=dt,
                               maxNumInc=1000000, nlgeom=OFF, application=TRANSIENT_FIDELITY)

    # 地震步内释放水平嵌固，改施加基底水平加速度
    bc_h.deactivate('Step-EQ')
    amp_data = tuple((float(t[i]), float(a[i])) for i in range(len(t)))
    model.TabularAmplitude(name='AccAmp', data=amp_data, timeSpan=STEP)
    model.AccelerationBC(name='BaseAcc', createStepName='Step-EQ', region=base,
                         a1=1.0, amplitude='AccAmp')
    log_step(logger, u'地震输入: 基底水平加速度 BC（Amplitude=AccAmp, 时长=%.2fs, dt=%.4fs）', tp, dt)

    # ---- 历史输出 ----
    freq = int(job_cfg['history_freq'])
    # 基底反力(求和=基底剪力) + 基底运动 U1/A1(第1层层间位移角需基底位移；基底被加速度BC驱动 U1≠0)
    model.HistoryOutputRequest(name='H-BaseRF', createStepName='Step-EQ',
                               variables=('RF1', 'U1', 'A1'), region=base, frequency=freq)
    # 各层参考节点：绝对位移/速度/加速度（层间位移角 & 楼层加速度后处理用）
    for k in range(1, ns + 1):
        region = asm.instances[inst_name].sets['FLOOR_%d' % k]
        model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName='Step-EQ',
                                   variables=('U1', 'V1', 'A1'), region=region, frequency=freq)
    # 整体能量（数值健康检查：ALLKE/ALLIE/ALLVD）
    model.HistoryOutputRequest(name='H-Energy', createStepName='Step-EQ',
                               variables=('ALLKE', 'ALLIE', 'ALLVD', 'ALLAE'), frequency=freq * 10)
    log_step(logger, u'历史输出已配: 基底RF1 + %d 层 U1/V1/A1 + 能量', ns)


def write_meta(acc_name, t, a, dt, inst_name, logger):
    """写 frame_meta.json：几何/质量/手算锚点，供后处理校核与复现。"""
    ns = int(frame_cfg['n_story'])
    sh = float(frame_cfg['story_height'])
    H = ns * sh
    meta = {
        'script': _DEFAULT_SCRIPT_NAME,
        'acc_record': acc_name, 'dt': dt, 'duration': float(t[-1]),
        'pga': float(np.max(np.abs(a))),
        'frame': {'n_story': ns, 'n_bay': int(frame_cfg['n_bay']),
                  'story_height': sh, 'bay_width': float(frame_cfg['bay_width']),
                  'total_height': H, 'floor_mass': float(frame_cfg['floor_mass']),
                  'total_mass': float(frame_cfg['floor_mass']) * ns},
        'material': dict(material_cfg), 'damping': dict(damping_cfg),
        'inst_name': inst_name,
        # 手算锚点（验 T1）：经验式 T1≈0.1·N（N=层数, 混凝土框架粗估）与基础周期 0.075·H^0.75（ATC）
        'anchor_T1_empirical_0p1N': 0.1 * ns,
        'anchor_T1_atc_0p075H075': 0.075 * (H ** 0.75),
        'floor_sets': ['FLOOR_%d' % k for k in range(1, ns + 1)],
        'base_set': 'BASE',
    }
    with open('frame_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log_step(logger, u'frame_meta.json 已写: T1 经验锚点 0.1N=%.2fs, ATC=%.2fs',
             meta['anchor_T1_empirical_0p1N'], meta['anchor_T1_atc_0p075H075'])


def submit(model_name, logger):
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model_name, description='Fixed-base 2D frame (step1 validation)',
            type=ANALYSIS, memory=job_cfg['memory_percent'], memoryUnits=PERCENTAGE,
            getMemoryFromAnalysis=True, numCpus=job_cfg['num_cpus'], numDomains=job_cfg['num_cpus'],
            multiprocessingMode=DEFAULT, numGPUs=0, echoPrint=OFF, modelPrint=OFF,
            contactPrint=OFF, historyPrint=OFF)
    mdb.save()
    log_step(logger, u'%s 提交中...', job_name)
    t0 = time.time()
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, u'%s 完成, 耗时=%.2fs', job_name, time.time() - t0)


# ==========================================================
#  主入口
# ==========================================================

def main():
    logger = log_step('frame_fixedbase.log')
    t0 = time.time()
    try:
        log_step(logger, u'====== Step-1 固定基础框架建模开始 ======')
        load_case_config(logger)
        acc_name, t, a, dt = find_acc_txt(logger)

        cae = 'frame_n%d_b%d.cae' % (int(frame_cfg['n_story']), int(frame_cfg['n_bay']))
        mdb.saveAs(pathName=cae)
        if 'Model-1' not in mdb.models:
            mdb.Model(name='Model-1')

        model, inst_name = build_frame_model(logger)
        setup_steps_and_loads(model, inst_name, acc_name, t, a, dt, logger)
        write_meta(acc_name, t, a, dt, inst_name, logger)
        mdb.save()
        log_step(logger, u'基础模型已存: %s', cae)

        if job_cfg.get('submit', True):
            log_step(logger, u'====== 提交作业 ======')
            submit('Model-1', logger)
            log_step(logger, u'完成。后处理请运行: abaqus cae noGUI=postproc_frame_v1.py')
        else:
            log_step(logger, u'submit=False，仅建模存 cae，未提交')

        log_step(logger, u'====== 全部完成, 总耗时=%.2fs ======', time.time() - t0)
    except Exception as exc:
        log_step(logger, u'脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
