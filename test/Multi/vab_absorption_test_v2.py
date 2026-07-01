# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""方法验证 Tier 2 + 弹簧系数敏感性：黏弹性人工边界(VAB)吸收性 + spring_scale 扫描（Abaqus，独立自包含）。

v2 相对 v1：内部增加 spring_scale 外循环（一次运行扫完多个弹簧系数），自动汇总"残余反射 vs spring_scale"表。
背景：VAB 弹簧 kn=G/2R、kt=G/4R（α_n=0.5/α_t=0.25）是刘晶波 2D 常用值(α_n=1.0/α_t=0.5)的一半。本测试直接量化：
把弹簧拨到标准 Liu(spring_scale=2.0) 能否把残余反射(v1 测得近边界 ~12%)压下去。

刘晶波式算例：均匀弹性二维域内部施加竖向集中力脉冲(Ricker)，对比：
  ① small_fixed —— 小域 + 固定边界(虚假反射，坏对照；与 spring 无关，只建一次)；
  ② small_vab   —— 小域 + VAB，对 SPRING_SCALES 每档各建一次；
  ③ large_ref   —— 大域(时窗内反射未返回，准无限参考解；与 spring 无关，只建一次)。
残余反射 = max|U−U_ref|/max|U_ref|（逐观测点：源处/中途/近边界）。

判读：
  • 近边界反射随 spring_scale↑ 明显降 → 残差含低频弹簧分量，标准 Liu 修一部分。
  • 几乎不动 → 反射由斜入射阻尼器失配主导，弹簧救不了(需海绵/PML)。

运行：abaqus cae noGUI=vab_absorption_test_v2.py
输出：屏幕 + vab_spring_sweep_summary.txt(逐测点反射 vs spring_scale)；vab_obs_*.csv(各模型时程，供绘图)。
"""

from abaqus import *  # 内核
from abaqusConstants import *  # 常量
from caeModules import *  # 模块对象(part/material/section/assembly/step/load/mesh 等)
from regionToolset import Region  # 区域对象(SectionAssignment/弹簧-阻尼器 用)
import os, math  # 标准库

# ===== 可调参数 =====
RHO = 2000.0; NU = 0.3; VS = 500.0  # 密度/泊松比/剪切波速(均匀弹性)
FC = 8.0                # Ricker 中心频率(Hz)，波长=VS/FC=62.5 m
MESH = 6.25             # 网格(m)，约 10 单元/波长
DT = 0.001; T_TOTAL = 1.2  # 时间步与总时长(s)
W_SMALL = 500.0         # 小域边长(m)，波源到边界 250 m，反射在 1.0 s 返回波源
W_LARGE = 900.0         # 大域边长(m)，波源到边界 450 m，时窗内反射不返回观测点
F_AMP = 1.0e6           # 集中力幅值(N)，仅相对比较，量级任意
OBS_OFFSETS = [(0.0, 0.0), (120.0, 0.0), (180.0, 0.0)]  # 观测点相对波源偏移(m)：源处/中途/近边界
SPRING_SCALES = [1.0, 2.0]  # 弹簧系数缩放扫描：1.0=现行(α_n0.5/α_t0.25)、2.0=标准Liu(1.0/0.5)
# ====================

CP = VS * math.sqrt((2.0 * (1.0 - NU)) / (1.0 - 2.0 * NU))  # 纵波速
GG = RHO * VS * VS  # 剪切模量
EE = 2.0 * GG * (1.0 + NU)  # 杨氏模量


def _fmt(k):  # 1.0→'1'、0.5→'0p5'（模型名安全）
    return ('%g' % float(k)).replace('.', 'p')


def _ricker(dt, T, fc, t0):  # Ricker 幅值序列 [(t,a),...]，峰=1
    out = []
    t = 0.0
    while t <= T + 0.5 * dt:
        x = math.pi * fc * (t - t0); r = (1.0 - 2.0 * x * x) * math.exp(-x * x)
        out.append((t, r)); t += dt
    return out


def build_box(model_name, W, load_xy, boundary, spring_scale=1.0):
    """建均匀方域 [0,W]^2，波源在 load_xy，boundary∈{'fixed','vab','free'}，spring_scale 缩放 VAB 弹簧。

    返回 (model, inst_name, load_label, obs_labels)。spring_scale 仅在 boundary=='vab' 时生效。
    """
    if model_name in mdb.models:
        del mdb.models[model_name]
    m = mdb.Model(name=model_name)
    s = m.ConstrainedSketch(name='sk', sheetSize=4 * W)
    s.rectangle(point1=(0.0, 0.0), point2=(W, W))
    part = m.Part(name='P', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    mat = m.Material(name='soil'); mat.Elastic(table=((EE, NU),)); mat.Density(table=((RHO,),))
    m.HomogeneousSolidSection(name='sec', material='soil', thickness=None)
    part.SectionAssignment(region=Region(faces=part.faces[:]), sectionName='sec')
    part.seedPart(size=MESH, deviationFactor=0.1, minSizeFactor=0.1)
    part.setElementType(regions=(part.faces[:],), elemTypes=(mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD),))
    part.generateMesh()
    a = m.rootAssembly; a.DatumCsysByDefault(CARTESIAN); inst = a.Instance(name='P-1', part=part, dependent=ON)

    # 波源节点(最近 load_xy) 与 观测节点
    def nearest(x0, y0):
        best = None
        for nd in inst.nodes:
            d = (nd.coordinates[0] - x0) ** 2 + (nd.coordinates[1] - y0) ** 2
            if best is None or d < best[0]:
                best = (d, nd.label)
        return best[1]
    load_label = nearest(load_xy[0], load_xy[1])
    obs_labels = [nearest(load_xy[0] + ox, load_xy[1] + oy) for (ox, oy) in OBS_OFFSETS]

    # 分析步
    step = m.ImplicitDynamicsStep(name='dyn', previous='Initial', timePeriod=T_TOTAL,
                                  timeIncrementationMethod=FIXED, initialInc=DT, maxNumInc=1000000,
                                  nlgeom=OFF, application=TRANSIENT_FIDELITY)
    # 观测点场输出(逐增量 U)
    obs_seq = inst.nodes.sequenceFromLabels(tuple(obs_labels))
    a.Set(nodes=obs_seq, name='OBS')
    m.FieldOutputRequest(name='F-OBS', createStepName='dyn', variables=('U',), frequency=1,
                         region=a.sets['OBS'])
    m.fieldOutputRequests['F-Output-1'].setValues(variables=('U',), frequency=10000000)  # 整体降频

    # 边界处理
    tol = 1e-3
    edges = {
        'L': [nd for nd in inst.nodes if abs(nd.coordinates[0] - 0.0) < tol],
        'R': [nd for nd in inst.nodes if abs(nd.coordinates[0] - W) < tol],
        'B': [nd for nd in inst.nodes if abs(nd.coordinates[1] - 0.0) < tol],
        'T': [nd for nd in inst.nodes if abs(nd.coordinates[1] - W) < tol],
    }
    if boundary == 'fixed':
        allb = []
        for v in edges.values():
            allb += v
        seq = inst.nodes.sequenceFromLabels(tuple(sorted(set(nd.label for nd in allb))))
        m.EncastreBC(name='fix', createStepName='Initial', region=a.Set(nodes=seq, name='FIX'))
    elif boundary == 'vab':
        _apply_vab(m, a, inst, edges, load_xy, spring_scale)
    # 'free' 不处理

    # 集中力(竖向 Y) + Ricker 幅值
    amp = _ricker(DT, T_TOTAL, FC, 1.0 / FC)  # t0 取一个周期，保证起始平滑
    m.TabularAmplitude(name='ric', data=tuple(amp), smooth=SOLVER_DEFAULT, timeSpan=STEP)
    ln = inst.nodes.sequenceFromLabels((load_label,))
    m.ConcentratedForce(name='src', createStepName='dyn', region=a.Set(nodes=ln, name='SRC'),
                        cf2=F_AMP, amplitude='ric', distributionType=UNIFORM)
    return m, 'P-1', load_label, obs_labels


def _apply_vab(m, a, inst, edges, load_xy, spring_scale):
    """对四条边界节点施加接地弹簧-阻尼器(VAB)，系数同主脚本口径；spring_scale 缩放弹簧 kn,kt(不动阻尼器)。"""
    lx, ly = load_xy
    axis = {'L': 1, 'R': 1, 'B': 0, 'T': 0}  # 沿边界变化的坐标：左右边界沿 y(轴1)、底顶边界沿 x(轴0)
    dofs = {'L': (1, 2), 'R': (1, 2), 'B': (2, 1), 'T': (2, 1)}  # (法向dof, 切向dof)：左右法向x切向y、底顶法向y切向x
    spr_on = (spring_scale > 0.0)  # spring_scale=0 纯黏性时关闭弹簧(Abaqus 要求 ON 的刚度>0)
    for tag, nodes in edges.items():
        ax = axis[tag]
        srt = sorted(nodes, key=lambda nd: nd.coordinates[ax])
        n = len(srt)
        for i, nd in enumerate(srt):
            c = nd.coordinates[ax]
            if n == 1:
                l = 0.0
            elif i == 0:
                l = abs(srt[1].coordinates[ax] - c) / 2.0
            elif i == n - 1:
                l = abs(c - srt[i - 1].coordinates[ax]) / 2.0
            else:
                l = abs(srt[i + 1].coordinates[ax] - srt[i - 1].coordinates[ax]) / 2.0
            R = max(1.0, math.hypot(nd.coordinates[0] - lx, nd.coordinates[1] - ly))  # 波源到节点距离
            kn = GG / (2.0 * R) * l * spring_scale; cn = RHO * CP * l  # 法向弹簧(×spring_scale)/阻尼
            kt = GG / (4.0 * R) * l * spring_scale; ct = RHO * VS * l  # 切向弹簧(×spring_scale)/阻尼
            dof_n, dof_t = dofs[tag]
            reg = Region(nodes=inst.nodes.sequenceFromLabels((nd.label,)))
            a.engineeringFeatures.SpringDashpotToGround(name='sd_%s_%d_n' % (tag, nd.label), region=reg,
                orientation=None, dof=dof_n, springBehavior=(ON if spr_on else OFF),
                springStiffness=(kn if spr_on else 1.0), dashpotBehavior=ON, dashpotCoefficient=cn)
            a.engineeringFeatures.SpringDashpotToGround(name='sd_%s_%d_t' % (tag, nd.label), region=reg,
                orientation=None, dof=dof_t, springBehavior=(ON if spr_on else OFF),
                springStiffness=(kt if spr_on else 1.0), dashpotBehavior=ON, dashpotCoefficient=ct)


def run_and_extract(model_name, inst_name, obs_labels):
    """提交并从 ODB 提取各观测点竖向位移 U2 时程。返回 {obs_index:[(t,u2),...]}。"""
    from odbAccess import openOdb
    job = model_name
    if job in mdb.jobs:
        del mdb.jobs[job]
    mdb.Job(name=job, model=model_name, numCpus=1).submit(consistencyChecking=OFF)
    mdb.jobs[job].waitForCompletion()
    odb = openOdb(job + '.odb', readOnly=True)
    step = odb.steps['dyn']
    inst = list(odb.rootAssembly.instances.values())[0]
    series = {i: [] for i in range(len(obs_labels))}
    for fr in step.frames:
        tt = fr.frameValue
        u = fr.fieldOutputs['U']
        bynode = {v.nodeLabel: v.data for v in u.values}
        for i, lab in enumerate(obs_labels):
            if lab in bynode:
                series[i].append((tt, float(bynode[lab][1])))  # U2(竖向)
    odb.close()
    return series


def _maxabs_list(vals):  # 列表绝对值最大；空返回 0（Py2.7 无 max(default=)）
    av = [abs(x) for x in vals]
    return max(av) if av else 0.0


def _write_obs_csv(cwd, name, series):  # 写该模型三观测点 U2 时程 csv(供绘图)
    with open(os.path.join(cwd, 'vab_obs_%s.csv' % name), 'w') as f:
        f.write('t,U2_src,U2_mid,U2_edge\n')
        n = max(len(series[i]) for i in series)
        for k in range(n):
            row = ['%.5f' % series[0][k][0]] if k < len(series[0]) else ['']
            for i in range(3):
                row.append('%.6e' % (series[i][k][1] if k < len(series[i]) else 0.0))
            f.write(','.join(row) + '\n')


def main():
    cwd = os.getcwd()
    print('扫 spring_scale = %s，验证 VAB 残余反射随弹簧系数的变化...' % SPRING_SCALES)
    # 固定边界(坏对照) + 大域参考：与 spring_scale 无关，各建一次
    res = {}
    for name, W, ld, bnd in [('small_fixed', W_SMALL, (W_SMALL / 2, W_SMALL / 2), 'fixed'),
                             ('large_ref', W_LARGE, (W_LARGE / 2, W_LARGE / 2), 'vab')]:
        print('  -> %s (W=%g, boundary=%s)' % (name, W, bnd))
        m, inst_name, ll, obs = build_box(name, W, ld, bnd, spring_scale=1.0)
        res[name] = run_and_extract(name, inst_name, obs)
        _write_obs_csv(cwd, name, res[name])
    # small_vab 逐档扫 spring_scale
    vab_res = {}  # spring_scale -> series
    for k in SPRING_SCALES:
        name = 'small_vab_s%s' % _fmt(k)
        print('  -> %s (spring_scale=%g, α_n=%.2f/α_t=%.2f)' % (name, k, 0.5 * k, 0.25 * k))
        m, inst_name, ll, obs = build_box(name, W_SMALL, (W_SMALL / 2, W_SMALL / 2), 'vab', spring_scale=k)
        vab_res[k] = run_and_extract(name, inst_name, obs)
        _write_obs_csv(cwd, name, vab_res[k])

    # 逐观测点残余反射：以 large_ref 为参考
    ref = res['large_ref']
    def reflection(series, oi):  # 第 oi 测点残余反射%
        ref_u = [u for _, u in ref[oi]]
        u = [v for _, v in series[oi]]
        n = min(len(u), len(ref_u))
        mr = _maxabs_list(ref_u)
        if n == 0 or mr == 0:
            return float('nan')
        return max(abs(u[k] - ref_u[k]) for k in range(n)) / mr * 100.0

    # 汇总表
    L = ['========== VAB 残余反射 随弹簧系数 spring_scale 扫描 ==========',
         '材料: rho=%g, Vs=%g, Vp=%.0f, nu=%g | Ricker fc=%g Hz | 网格 %g m | dt=%g, T=%g' % (RHO, VS, CP, NU, FC, MESH, DT, T_TOTAL),
         '小域 W=%g(源到边界%g), 大域参考 W=%g | 残余反射=max|U-U_ref|/max|U_ref|' % (W_SMALL, W_SMALL / 2, W_LARGE),
         'spring_scale=1.0 -> alpha_n0.5/alpha_t0.25(现行); =2.0 -> 标准Liu(1.0/0.5)',
         '-' * 60,
         '%-18s %-6s %-9s %-9s %-9s' % ('边界', 'a_n', '+0m', '+120m', '+180m'),
         '-' * 60]
    # 固定边界对照行
    L.append('%-18s %-6s %-9.1f %-9.1f %-9.1f' % ('固定(坏对照)', '-',
             reflection(res['small_fixed'], 0), reflection(res['small_fixed'], 1), reflection(res['small_fixed'], 2)))
    # 各 spring_scale 行
    for k in SPRING_SCALES:
        note = '(现行)' if abs(k - 1.0) < 1e-9 else ('(标准Liu)' if abs(k - 2.0) < 1e-9 else '')
        tag = 'VAB s=%g%s' % (k, note)
        L.append('%-18s %-6.2f %-9.1f %-9.1f %-9.1f' % (tag, 0.5 * k,
                 reflection(vab_res[k], 0), reflection(vab_res[k], 1), reflection(vab_res[k], 2)))
    L.append('-' * 60)

    # 判读：近边界(+180m)反射随 spring_scale 怎么动
    edge = [(k, reflection(vab_res[k], 2)) for k in SPRING_SCALES]
    edge = [(k, r) for (k, r) in edge if r == r]  # 去 nan
    if len(edge) >= 2:
        lo_k, lo_r = edge[0]; hi_k, hi_r = edge[-1]  # 按 SPRING_SCALES 顺序首尾
        drop = (lo_r - hi_r) / lo_r * 100.0 if lo_r else float('nan')
        L.append('近边界(+180m)反射: spring_scale=%g 时 %.1f%% -> spring_scale=%g 时 %.1f%% (降 %.0f%%)'
                 % (lo_k, lo_r, hi_k, hi_r, drop))
        if drop >= 25.0:
            L.append('判读: 反射随弹簧加强明显下降 -> 残差含低频弹簧分量，标准Liu(2.0)修一部分，建议正文采用。')
        elif drop <= 5.0:
            L.append('判读: 反射几乎不随弹簧变化 -> 由斜入射阻尼器失配主导，弹簧救不了(需海绵层/PML)。')
        else:
            L.append('判读: 反射中等下降 -> 弹簧是部分因素，标准Liu略有改善但非根治。')
    text = '\n'.join(L); print(text)
    with open(os.path.join(cwd, 'vab_spring_sweep_summary.txt'), 'w') as f:
        f.write(text)
    print('\n已写出 vab_spring_sweep_summary.txt 与 vab_obs_*.csv。')


if __name__ == '__main__' or True:
    main()
