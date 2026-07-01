# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软楔 2D 共振模态【存在性】验证脚本（Abaqus 频率提取）。

目的：直接回答"坡顶软楔的 2D 面波共振模 Abaqus 到底能不能表示"。
做法：建一个与论文同口径的简化台阶坡(软表层50m + 覆盖层 + 基岩，底+两侧固定)，
      跑 *Frequency(Lanczos) 特征值分析，提取若干阶振型，
      再按"模态位移能量在坡顶软楔里的占比"给每阶打分，
      标记落在输入频带(默认 1.5~8Hz)内、且能量高度集中于坡顶软楔的【局部模态】。

判据：
  - 找到 1.5~8Hz 内、坡顶软楔能量占比高(>阈值)的模态 → 该 2D 共振模 Abaqus 表示得了，
    那么动力工况里放大偏低就只是激励/网格频散/时窗问题(可调)，不是方法天花板；
  - 找不到(或只有铺满整层的 1D 型模态、无坡顶集中) → 论文那种"尖锐 2D 俘获共振"本身偏弱，
    反过来支持对论文 7.6 峰值的质疑。

运行方式(Windows 命令行，在本文件所在目录)：
    abaqus cae noGUI=eigen_softwedge_v1.py
输出：
  - 屏幕打印各阶频率与软楔能量占比表，并高亮命中的局部模态；
  - eigen_softwedge_summary.txt 汇总；
  - softwedge_eigen.odb 可在 Abaqus/Viewer 里直接看振型云图。

注意：这是【独立简化模型】，几何/材料与正式建模脚本同口径，但不含黏弹性边界/等效力
      (模态分析只需 M、K，阻尼器对实模态无意义)；底+两侧固定是为压低整体块体模态、
      让局部软楔模态凸显出来(坡顶离两侧 ~1000m，远大于软层波长，固定侧边不影响坡顶模态)。
"""

from abaqus import *  # Abaqus 内核主对象(mdb 等)
from abaqusConstants import *  # Abaqus 常量(LANCZOS/CPE4R/ON 等)
from caeModules import *  # part/mesh/section 等模块对象
import os  # 路径操作
import math  # 数学函数

# ========== 可调参数 ==========
RHO = 2500.0          # 密度 kg/m^3(各层相同，与论文一致)
NU = 0.3              # 泊松比(各层相同)
VS_SURFACE = 400.0    # 表层(软层)剪切波速 m/s —— Vs1/Vs2=0.5
VS_OVERLYING = 800.0  # 覆盖层剪切波速 m/s —— Vs2
VS_BEDROCK = 2000.0   # 基岩剪切波速 m/s —— VR

BEDROCK_THICK = 200.0  # 基岩层厚 m(y:0~200)
H = 400.0              # 上覆总厚 m(上平台面 y=BEDROCK_THICK+H=600)
H_LOWER = 200.0        # 下平台上覆厚 m(下平台面 y=BEDROCK_THICK+H_LOWER=400)
SLOPE_HEIGHT = 200.0   # 坡高 m(= H - H_LOWER)
X_CREST = 1000.0       # 坡顶 x
TOTAL_L = 1800.0       # 模型总长 m
SURF_THICK = 50.0      # 表层(软层)厚度 h1 m

MESH_SIZE = 4.0       # 网格尺寸 m(模态分析用，4m 足够；要更准可改 2)
NUM_EIGEN = 200       # 提取模态阶数(若打印的最高频率没到 8Hz，请调大本值；底+两侧固定已抬高整体模态、利于覆盖软楔带)
ELEM = CPE4R          # 单元类型(平面应变减缩积分)

BAND_LO, BAND_HI = 1.5, 8.0   # 关注的输入频带(Hz)：命中此带内的局部模态才算"可被激励的共振"
WEDGE_X_LO, WEDGE_X_HI = 850.0, 1080.0  # 坡顶软楔 x 范围(用于能量占比统计)
WEDGE_Y = 545.0       # 软楔 y 下界(软层底界 y=550 略放宽)
LOC_FLAG = 0.30       # 局部化阈值：坡顶软楔能量占比 > 此值即判为"坡顶软楔局部模态"

MODEL_NAME = 'SoftWedgeEigen'  # 模型名
PART_NAME = 'Slope'            # 零件名
INST_NAME = 'Slope-1'         # 实例名
JOB_NAME = 'softwedge_eigen'  # 作业名/ODB 名
# ================================

# 由波速换算杨氏模量 E = 2*rho*Vs^2*(1+nu)(各向同性弹性)
def _E_from_vs(vs):
    """返回杨氏模量 Pa。G=rho*Vs^2；E=2G(1+nu)。"""
    G = RHO * vs * vs  # 剪切模量
    return 2.0 * G * (1.0 + NU)  # 杨氏模量


# 关键几何派生量
Y_SURF_UP = BEDROCK_THICK + H        # 上平台地表 y = 600
Y_SURF_LOW = BEDROCK_THICK + H_LOWER  # 下平台地表 y = 400
Y_BEDROCK_TOP = BEDROCK_THICK         # 基岩顶 y = 200
Y_SOFT_BOT = Y_SURF_UP - SURF_THICK   # 软层底界 y = 550
X_TOE = X_CREST + SLOPE_HEIGHT        # 坡脚 x = 1200(45°时水平投影=坡高)
# 软层底界 y=550 与坡面相交处的 x：坡面 y=600-(x-1000) → x = 1000 + (600-550) = 1050
X_SOFT_PINCH = X_CREST + (Y_SURF_UP - Y_SOFT_BOT)  # = 1050


def build_model():
    """建立简化台阶坡模型：外轮廓 + 两条水平分区线(分出 软层/覆盖层/基岩)，赋材料、网格。"""
    if MODEL_NAME in mdb.models:
        del mdb.models[MODEL_NAME]  # 清理同名旧模型
    model = mdb.Model(name=MODEL_NAME)

    # ---- 外轮廓多段线(逆时针)：左下→右下→右平台面→坡脚→坡顶→上平台左→闭合 ----
    s = model.ConstrainedSketch(name='outline', sheetSize=4000.0)
    s.Line(point1=(0.0, 0.0), point2=(TOTAL_L, 0.0))            # 底边
    s.Line(point1=(TOTAL_L, 0.0), point2=(TOTAL_L, Y_SURF_LOW))  # 右边(到下平台面)
    s.Line(point1=(TOTAL_L, Y_SURF_LOW), point2=(X_TOE, Y_SURF_LOW))  # 下平台面(右→坡脚)
    s.Line(point1=(X_TOE, Y_SURF_LOW), point2=(X_CREST, Y_SURF_UP))   # 坡面(坡脚→坡顶)
    s.Line(point1=(X_CREST, Y_SURF_UP), point2=(0.0, Y_SURF_UP))      # 上平台面(坡顶→左)
    s.Line(point1=(0.0, Y_SURF_UP), point2=(0.0, 0.0))               # 左边(闭合)
    part = model.Part(name=PART_NAME, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)

    # ---- 分区线：y=200(基岩顶) 全宽；y=550(软层底) 从左边到坡面交点(1050) ----
    sp = model.ConstrainedSketch(name='partlines', sheetSize=4000.0)
    sp.Line(point1=(0.0, Y_BEDROCK_TOP), point2=(TOTAL_L, Y_BEDROCK_TOP))  # 基岩顶界(全宽)
    sp.Line(point1=(0.0, Y_SOFT_BOT), point2=(X_SOFT_PINCH, Y_SOFT_BOT))   # 软层底界(到坡面尖灭点)
    part.PartitionFaceBySketch(faces=part.faces[:], sketch=sp)  # 一次分出三块面

    # ---- 三种材料 + 截面 ----
    specs = [('SOFT', VS_SURFACE), ('OVERLYING', VS_OVERLYING), ('BEDROCK', VS_BEDROCK)]
    for nm, vs in specs:
        mat = model.Material(name=nm)
        mat.Elastic(table=((_E_from_vs(vs), NU),))  # 弹性(E, nu)
        mat.Density(table=((RHO,),))                # 密度
        model.HomogeneousSolidSection(name=nm + '_SEC', material=nm, thickness=None)

    # ---- 按面所在位置(findAt)赋截面 ----
    pick = {
        'BEDROCK':   (900.0, 100.0),   # 基岩面内一点(y<200)
        'OVERLYING': (900.0, 400.0),   # 覆盖层面内一点(200<y<550)
        'SOFT':      (500.0, 575.0),   # 软层面内一点(550<y<600，上平台)
    }
    for nm, (px, py) in pick.items():
        f = part.faces.findAt(((px, py, 0.0),))  # 定位该位置的面
        region = part.Set(faces=f, name=nm + '_FACE')
        part.SectionAssignment(region=region, sectionName=nm + '_SEC')

    # ---- 网格 ----
    part.seedPart(size=MESH_SIZE, deviationFactor=0.1, minSizeFactor=0.1)
    et = mesh.ElemType(elemCode=ELEM, elemLibrary=STANDARD)  # 四节点平面应变
    et3 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)  # 三角(过渡兜底)
    part.setElementType(regions=(part.faces[:],), elemTypes=(et, et3))
    part.generateMesh()

    # ---- 装配 ----
    a = model.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    a.Instance(name=INST_NAME, part=part, dependent=ON)

    # ---- 边界条件：底边 + 左右两侧固定(Initial 步)----
    inst = a.instances[INST_NAME]
    tol = 1.0  # 选边容差
    e_bottom = inst.edges.getByBoundingBox(-tol, -tol, -tol, TOTAL_L + tol, tol, tol)  # y≈0
    e_left = inst.edges.getByBoundingBox(-tol, -tol, -tol, tol, Y_SURF_UP + tol, tol)  # x≈0
    e_right = inst.edges.getByBoundingBox(TOTAL_L - tol, -tol, -tol, TOTAL_L + tol, Y_SURF_UP + tol, tol)  # x≈L
    fixed = a.Set(edges=e_bottom + e_left + e_right, name='FIXED_OUTER')
    model.EncastreBC(name='fix_outer', createStepName='Initial', region=fixed)

    # ---- 频率提取步 ----
    # 不指定 normalization：默认按质量正则化。SIM/Lanczos 求解器只支持质量正则化；
    # 本脚本局部化指标用模态内能量比值(软楔/整体)，与正则化方式无关，故无需 DISPLACEMENT。
    model.FrequencyStep(name='Freq', previous='Initial', eigensolver=LANCZOS,
                        numEigen=NUM_EIGEN)
    return model


def run_job():
    """提交作业并等待完成，返回 ODB 路径。"""
    if JOB_NAME in mdb.jobs:
        del mdb.jobs[JOB_NAME]
    job = mdb.Job(name=JOB_NAME, model=MODEL_NAME, type=ANALYSIS, numCpus=1)
    job.submit(consistencyChecking=OFF)
    job.waitForCompletion()
    return os.path.join(os.getcwd(), JOB_NAME + '.odb')


def postprocess(odb_path):
    """读 ODB：逐阶算坡顶软楔/整软层的模态位移能量占比，打印表并标记局部模态。"""
    from odbAccess import openOdb  # 仅后处理需要
    odb = openOdb(odb_path, readOnly=True)
    step = odb.steps['Freq']
    inst = odb.rootAssembly.instances[INST_NAME.upper()] if INST_NAME.upper() in odb.rootAssembly.instances \
        else list(odb.rootAssembly.instances.values())[0]  # 取实例(名称大小写兜底)

    # 节点坐标表 label -> (x, y)
    coord = {}
    for nd in inst.nodes:
        coord[nd.label] = (nd.coordinates[0], nd.coordinates[1])

    # 各区域节点掩码
    def in_wedge(x, y):  # 坡顶软楔
        return (WEDGE_X_LO <= x <= WEDGE_X_HI) and (y >= WEDGE_Y)

    def in_softlayer(x, y):  # 整个软层(y>=软层底界)
        return y >= WEDGE_Y

    rows = []  # (mode, freq, frac_wedge, frac_soft)
    fmin, fmax = 1e9, -1e9
    for fr in step.frames:
        mode = getattr(fr, 'mode', None)
        freq = getattr(fr, 'frequency', None)  # Abaqus 频率步：每帧的频率(Hz)
        if freq is None or freq <= 0:
            continue  # 跳过基态/无效帧
        fmin = min(fmin, freq); fmax = max(fmax, freq)
        u = fr.fieldOutputs['U']
        tot = 0.0; w_sum = 0.0; s_sum = 0.0
        for v in u.values:
            xy = coord.get(v.nodeLabel)
            if xy is None:
                continue
            ux, uy = v.data[0], v.data[1]
            e = ux * ux + uy * uy  # 该节点模态位移能量(∝|U|^2)
            tot += e
            if e > 0.0:
                x, y = xy
                if in_softlayer(x, y):
                    s_sum += e
                    if in_wedge(x, y):
                        w_sum += e
        if tot <= 0:
            continue
        rows.append((mode, freq, w_sum / tot, s_sum / tot))
    odb.close()

    # 打印结果
    lines = []
    lines.append('========== 软楔模态提取结果 ==========')
    lines.append('提取阶数=%d, 频率范围=%.3f ~ %.3f Hz' % (len(rows), fmin, fmax))
    if fmax < BAND_HI:
        lines.append('!! 警告：最高频率 %.2fHz 未覆盖关注带上限 %.2fHz，请把 NUM_EIGEN 调大后重跑' % (fmax, BAND_HI))
    lines.append('坡顶软楔 x∈[%.0f,%.0f], y≥%.0f；局部化阈值=%.2f；关注带=%.1f~%.1fHz'
                 % (WEDGE_X_LO, WEDGE_X_HI, WEDGE_Y, LOC_FLAG, BAND_LO, BAND_HI))
    lines.append('-' * 64)
    lines.append('%-6s %-10s %-14s %-14s %-6s' % ('mode', 'freq(Hz)', '软楔能量占比', '整软层能量占比', '命中?'))

    hits = []
    # 按软楔占比从高到低，挑出最像"坡顶软楔局部模态"的前若干阶展示；同时逐阶全表写入文件
    rows_sorted = sorted(rows, key=lambda r: r[2], reverse=True)
    for (mode, freq, fw, fs) in rows_sorted:
        hit = (BAND_LO <= freq <= BAND_HI) and (fw >= LOC_FLAG)
        flag = 'YES' if hit else ''
        if hit:
            hits.append((mode, freq, fw, fs))
        lines.append('%-6s %-10.3f %-14.3f %-14.3f %-6s' % (mode, freq, fw, fs, flag))

    lines.append('-' * 64)
    if hits:
        lines.append('结论：在 %.1f~%.1fHz 内找到 %d 个【坡顶软楔局部模态】(软楔能量占比≥%.2f)。'
                     % (BAND_LO, BAND_HI, len(hits), LOC_FLAG))
        lines.append('  → 该 2D 共振模 Abaqus 能表示；动力工况放大偏低应归因于激励/网格频散/时窗，可继续调。')
        for (mode, freq, fw, fs) in hits:
            lines.append('    命中 mode=%s, f=%.3fHz, 软楔占比=%.3f' % (mode, freq, fw))
    else:
        lines.append('结论：在 %.1f~%.1fHz 内【未找到】坡顶软楔局部模态(占比均<%.2f)。'
                     % (BAND_LO, BAND_HI, LOC_FLAG))
        lines.append('  → 该尖锐 2D 俘获共振本身偏弱；反过来支持对论文 7.6 峰值的质疑。')
        lines.append('  (若最高频率未到 %.1fHz，请先调大 NUM_EIGEN 再下结论)' % BAND_HI)

    text = '\n'.join(lines)
    print(text)
    with open('eigen_softwedge_summary.txt', 'w') as f:
        f.write(text)
    print('\n已写出 eigen_softwedge_summary.txt；可在 Abaqus/Viewer 打开 %s.odb 查看振型。' % JOB_NAME)


if __name__ == '__main__' or True:  # Abaqus noGUI 执行时直接跑
    print('开始建模...')
    build_model()
    print('提交频率提取作业...')
    odb_path = run_job()
    print('后处理 ODB: %s' % odb_path)
    postprocess(odb_path)
