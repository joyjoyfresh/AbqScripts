# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""方法验证 Tier 2：黏弹性人工边界(VAB)吸收性验证脚本（Abaqus，独立自包含）。

刘晶波式算例：均匀弹性二维域内部施加竖向集中力脉冲(Ricker)，对比三种处理在观测点的位移时程：
  ① small_fixed —— 小域 + 固定边界(显示虚假反射，作"坏"对照)；
  ② small_vab   —— 小域 + 黏弹性边界(VAB，应消除反射)；
  ③ large_ref   —— 大域(边界足够远，分析时窗内反射未返回，作"准无限"参考解)。
判据：small_vab 与 large_ref 在观测点时程基本重合；残余反射 = max|U_vab−U_ref|/max|U_ref| 很小(如<5–10%)，
      且明显小于 small_fixed 的反射。证 VAB 有效吸收外行散射波。

VAB 系数采用与主建模脚本同一口径：Kn=G/(2R)·l, Cn=ρ·cp·l, Kt=G/(4R)·l, Ct=ρ·cs·l
  (R=波源到该边界节点距离, l=该节点沿边界的影响长度)。

运行：abaqus cae noGUI=vab_absorption_test_v1.py
输出：屏幕 + vab_absorption_summary.txt(残余反射量化)；vab_obs_*.csv(各模型观测点位移时程，供绘图)。
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
# ====================

CP = VS * math.sqrt((2.0 * (1.0 - NU)) / (1.0 - 2.0 * NU))  # 纵波速
GG = RHO * VS * VS  # 剪切模量
EE = 2.0 * GG * (1.0 + NU)  # 杨氏模量


def _ricker(dt, T, fc, t0):  # Ricker 幅值序列 [(t,a),...]，峰=1
    out = []
    t = 0.0
    while t <= T + 0.5 * dt:
        x = math.pi * fc * (t - t0); r = (1.0 - 2.0 * x * x) * math.exp(-x * x)
        out.append((t, r)); t += dt
    return out


def build_box(model_name, W, load_xy, boundary):
    """建均匀方域 [0,W]^2，波源在 load_xy，boundary∈{'fixed','vab','free'}；返回 (model, inst_name, load_label, obs_labels)。"""
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
        _apply_vab(m, a, inst, edges, load_xy)
    # 'free' 不处理

    # 集中力(竖向 Y) + Ricker 幅值
    amp = _ricker(DT, T_TOTAL, FC, 1.0 / FC)  # t0 取一个周期，保证起始平滑
    m.TabularAmplitude(name='ric', data=tuple(amp), smooth=SOLVER_DEFAULT, timeSpan=STEP)
    ln = inst.nodes.sequenceFromLabels((load_label,))
    m.ConcentratedForce(name='src', createStepName='dyn', region=a.Set(nodes=ln, name='SRC'),
                        cf2=F_AMP, amplitude='ric', distributionType=UNIFORM)
    return m, 'P-1', load_label, obs_labels


def _apply_vab(m, a, inst, edges, load_xy):
    """对四条边界节点施加接地弹簧-阻尼器(VAB)，系数同主脚本口径。"""
    lx, ly = load_xy
    axis = {'L': 1, 'R': 1, 'B': 0, 'T': 0}  # 沿边界变化的坐标：左右边界沿 y(轴1)、底顶边界沿 x(轴0)
    dofs = {'L': (1, 2), 'R': (1, 2), 'B': (2, 1), 'T': (2, 1)}  # (法向dof, 切向dof)：左右法向x切向y、底顶法向y切向x
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
            kn = GG / (2.0 * R) * l; cn = RHO * CP * l  # 法向弹簧/阻尼
            kt = GG / (4.0 * R) * l; ct = RHO * VS * l  # 切向弹簧/阻尼
            dof_n, dof_t = dofs[tag]
            reg = Region(nodes=inst.nodes.sequenceFromLabels((nd.label,)))
            a.engineeringFeatures.SpringDashpotToGround(name='sd_%s_%d_n' % (tag, nd.label), region=reg,
                orientation=None, dof=dof_n, springBehavior=ON, springStiffness=kn, dashpotBehavior=ON, dashpotCoefficient=cn)
            a.engineeringFeatures.SpringDashpotToGround(name='sd_%s_%d_t' % (tag, nd.label), region=reg,
                orientation=None, dof=dof_t, springBehavior=ON, springStiffness=kt, dashpotBehavior=ON, dashpotCoefficient=ct)


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


def _maxabs(series):  # 取序列竖向幅值绝对值的最大值；空序列返回 0（Py2.7 无 max(default=) 参数）
    vals = [abs(u) for _, u in series]
    return max(vals) if vals else 0.0


def main():
    cwd = os.getcwd()
    print('建模并求解三模型(small_fixed / small_vab / large_ref)...')
    cfgs = [('small_fixed', W_SMALL, (W_SMALL / 2, W_SMALL / 2), 'fixed'),
            ('small_vab', W_SMALL, (W_SMALL / 2, W_SMALL / 2), 'vab'),
            ('large_ref', W_LARGE, (W_LARGE / 2, W_LARGE / 2), 'vab')]
    res = {}
    for name, W, ld, bnd in cfgs:
        print('  -> %s (W=%g, boundary=%s)' % (name, W, bnd))
        m, inst_name, ll, obs = build_box(name, W, ld, bnd)
        res[name] = run_and_extract(name, inst_name, obs)
        # 写时程 csv(供绘图)
        with open(os.path.join(cwd, 'vab_obs_%s.csv' % name), 'w') as f:
            f.write('t,U2_src,U2_mid,U2_edge\n')
            n = max(len(res[name][i]) for i in res[name])
            for k in range(n):
                row = ['%.5f' % res[name][0][k][0]] if k < len(res[name][0]) else ['']
                for i in range(3):
                    row.append('%.6e' % (res[name][i][k][1] if k < len(res[name][i]) else 0.0))
                f.write(','.join(row) + '\n')

    # 残余反射：以 large_ref 为参考，对齐时刻比较"近边界"观测点(index 2)
    ref = res['large_ref'][2]; ref_t = [t for t, _ in ref]; ref_u = [u for _, u in ref]
    def reflection(series):
        u = [v for _, v in series]
        n = min(len(u), len(ref_u))
        if n == 0 or _maxabs(ref) == 0:
            return float('nan')
        dmax = max(abs(u[k] - ref_u[k]) for k in range(n))
        return dmax / _maxabs(ref) * 100.0
    r_fixed = reflection(res['small_fixed'][2]); r_vab = reflection(res['small_vab'][2])

    L = ['========== 黏弹性边界(VAB) 吸收性验证 ==========',
         '材料: ρ=%g, Vs=%g, Vp=%.0f, ν=%g | Ricker fc=%g Hz | 网格 %g m | dt=%g, T=%g' % (RHO, VS, CP, NU, FC, MESH, DT, T_TOTAL),
         '小域 W=%g(源到边界%g), 大域参考 W=%g(源到边界%g)' % (W_SMALL, W_SMALL / 2, W_LARGE, W_LARGE / 2),
         '观测点(近边界, 相对源 +%g m)竖向位移 vs 大域参考的残余反射:' % OBS_OFFSETS[2][0],
         '-' * 52,
         '  小域+固定边界  残余反射 = %.1f%%   (坏对照, 应很大)' % r_fixed,
         '  小域+VAB       残余反射 = %.1f%%   (应明显小于固定边界)' % r_vab,
         '-' * 52]
    if r_vab == r_vab and r_fixed == r_fixed:
        if r_vab < 10.0 and r_vab < 0.5 * r_fixed:
            L.append('结论：VAB 将虚假反射从 %.1f%% 降到 %.1f%% → 黏弹性边界有效吸收外行散射波，验证通过。' % (r_fixed, r_vab))
        else:
            L.append('结论：VAB 残余反射 %.1f%% 仍偏大，建议加密边界网格或增大波源到边界距离后复核。' % r_vab)
    text = '\n'.join(L); print(text)
    with open(os.path.join(cwd, 'vab_absorption_summary.txt'), 'w') as f:
        f.write(text)
    print('\n已写出 vab_absorption_summary.txt 与 vab_obs_*.csv(可绘三条时程对比图)。')


if __name__ == '__main__' or True:
    main()
