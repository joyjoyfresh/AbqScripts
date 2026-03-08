# -*- coding: utf-8 -*-

from abaqus import *
from abaqusConstants import *
from abaqus import mdb
from regionToolset import Region
from caeModules import *
import mesh
import numpy as np
import math
import os
import time
import logging
import traceback


# 获取脚本名（不含扩展名）
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

LOGGER = None
STEP_COUNTER = 0
RUN_START_TIME = None
LAST_STEP_TIME = None


def setup_logger(log_file=None):
    """初始化日志器，同时输出到屏幕和文件。"""
    global LOGGER, STEP_COUNTER, RUN_START_TIME, LAST_STEP_TIME
    if LOGGER is not None:
        return LOGGER

    # 默认日志文件名为脚本名.log
    if log_file is None:
        log_file = SCRIPT_NAME + '.log'

    logging.addLevelName(logging.DEBUG, '调试')
    logging.addLevelName(logging.INFO, '信息')
    logging.addLevelName(logging.WARNING, '警告')
    logging.addLevelName(logging.ERROR, '错误')
    logging.addLevelName(logging.CRITICAL, '严重')

    logger = logging.getLogger(SCRIPT_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    LOGGER = logger
    STEP_COUNTER = 0
    RUN_START_TIME = time.time()
    LAST_STEP_TIME = RUN_START_TIME
    return logger


def log_step(logger, message, *args):
    """输出带步骤编号的详细日志，便于定位异常位置。"""
    global STEP_COUNTER, RUN_START_TIME, LAST_STEP_TIME
    now = time.time()
    if RUN_START_TIME is None:
        RUN_START_TIME = now
    if LAST_STEP_TIME is None:
        LAST_STEP_TIME = RUN_START_TIME

    delta_last = now - LAST_STEP_TIME
    delta_total = now - RUN_START_TIME
    STEP_COUNTER += 1
    logger.info('[步骤 %04d][用时%.3fs][总用时%.3fs] ' + message,
                STEP_COUNTER, delta_last, delta_total, *args)
    LAST_STEP_TIME = now


def create_model(width, height, cs, vv, density, mesh_size,
                 time_period, initial_inc, logger=None):
    """
    创建二维平面应变模型：几何、材料、截面、装配、分析步、网格

    参数:
        width      (float): 模型宽度 (m)
        height     (float): 模型高度 (m)
        cs         (float): 剪切波速 (m/s)
        vv         (float): 泊松比
        density    (float): 密度 (kg/m³)
        mesh_size  (float): 网格尺寸 (m)
        time_period(float): 分析步总时长 (s)
        initial_inc(float): 固定增量步大小 (s)
    """
    logger = logger or setup_logger()
    t0 = time.time()
    model_name = 'Model-1'
    part_name = 'Part-1'
    log_step(logger, '创建模型开始: 宽度=%.3f, 高度=%.3f, 剪切波速=%.3f, 泊松比=%.3f, 密度=%.3f, 网格=%.3f',
             width, height, cs, vv, density, mesh_size)

    # --- 清除已有模型并重建 ---
    try:
        del mdb.models[model_name]
        log_step(logger, '已删除已有模型: %s', model_name)
    except KeyError:
        log_step(logger, '未找到需要删除的已有模型: %s', model_name)
    model = mdb.Model(name=model_name)
    log_step(logger, '已创建模型: %s', model_name)

    # ============ 创建二维矩形Part ============
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(width, height) * 2)
    s.rectangle(point1=(0.0, 0.0), point2=(width, height))
    log_step(logger, '已创建矩形草图')
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del model.sketches['__profile__']
    log_step(logger, '已创建零件并生成壳基体: %s', part_name)

    # ============ 材料与截面 ============
    GG = density * cs ** 2
    EE = 2 * GG * (1 + vv)
    log_step(logger, '材料参数已计算: G=%.6e, E=%.6e', GG, EE)

    mat = model.Material(name='Soil')
    mat.Elastic(table=((EE, vv),))
    mat.Density(table=((density,),))
    log_step(logger, '材料已定义: Soil')

    model.HomogeneousSolidSection(name='Section-1', material='Soil',
                                  thickness=1.0)
    log_step(logger, '截面已创建: Section-1')

    faces = part.faces
    region = part.Set(faces=faces, name='Set-All')
    part.SectionAssignment(region=region, sectionName='Section-1',
                           offset=0.0, offsetType=MIDDLE_SURFACE,
                           offsetField='', thicknessAssignment=FROM_SECTION)
    log_step(logger, '截面已分配到所有面')

    # ============ 装配 ============
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    assembly.Instance(name='Part-1-1', part=part, dependent=ON)
    log_step(logger, '装配实例已创建: Part-1-1')

    # ============ 分析步 ============
    model.ImplicitDynamicsStep(
        name='step-earthquake', previous='Initial',
        timePeriod=time_period, timeIncrementationMethod=FIXED, initialInc=initial_inc,
        maxNumInc=1000000,
        nlgeom=OFF, application=MODERATE_DISSIPATION)
    log_step(logger, '动力分析步已创建: 时长=%.6f, 初始增量=%.6f',
             time_period, initial_inc)

    # 修改：设置场输出请求，确保每一帧数据都能输出到 odb 文件中
    # 默认情况下，Abaqus 只有几个关键帧。我们这里强制每个增量步都输出。
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'E', 'U', 'V', 'A', 'RF'),
        frequency=10)
    log_step(logger, '场输出请求已更新: 频率=10')

    # ============ 网格划分 ============
    # 设置网格控制：四边形单元(QUAD)，结构化划分(STRUCTURED)
    pickedRegions = part.faces
    part.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)
    log_step(logger, '网格控制已设置: 四边形 + 结构化')

    # 近似全局尺寸设为 mesh_size
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    log_step(logger, '已播种网格: 尺寸=%.3f', mesh_size)

    # 指定单元类型：平面应变四节点双线性单元 (CPE4)，取消减缩积分 (取消了尾部的 'R')
    elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1,))
    part.generateMesh()
    log_step(logger, '已生成网格: CPE4 单元')

    # 重新生成装配体以同步网格
    assembly.regenerate()
    log_step(logger, '装配已重新生成')
    log_step(logger, '创建模型完成: 节点=%d, 单元=%d, 耗时=%.2fs',
             len(part.nodes), len(part.elements), time.time() - t0)


def VAB_oblique(angle, cs, vv, density, logger=None):
    """
    主函数：为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力

    参数:
        angle  (float): SV波入射角度（度），0为垂直入射
        cs     (float): 剪切波速 (m/s)
        vv     (float): 泊松比
        density(float): 密度 (kg/m³)
    """
    # ============ 基本参数 ============
    logger = logger or setup_logger()
    t0 = time.time()
    VELtxt = 'VEL.txt'             # 速度时程文件（两列：时间, 速度）
    step_name = 'step-earthquake'  # 地震分析步名称
    log_step(logger, '斜入射人工边界开始: 入射角=%.3f, 剪切波速=%.3f, 泊松比=%.3f, 密度=%.3f',
             angle, cs, vv, density)

    # ============ 获取装配体和Part ============
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()
    log_step(logger, '已加载装配并重新生成')

    part = mdb.models['Model-1'].parts['Part-1']
    nodes = part.nodes
    log_step(logger, '已加载零件并获取节点')

    # ============ 提取边界节点 ============
    x_list = [node.coordinates[0] for node in nodes]
    y_list = [node.coordinates[1] for node in nodes]
    xmin = min(x_list)
    xmax = max(x_list)
    ymin = min(y_list)
    tol = 1e-6
    log_step(logger, '边界范围已识别: xmin=%.3f, xmax=%.3f, ymin=%.3f', xmin, xmax, ymin)

    # ====创建左边界、右边界、底边界节点集====
    l_nodes_list = [node for node in nodes if abs(node.coordinates[0] - xmin) < tol]
    r_nodes_list = [node for node in nodes if abs(node.coordinates[0] - xmax) < tol]
    b_nodes_list = [node for node in nodes if abs(node.coordinates[1] - ymin) < tol]

    l_nodes = part.nodes.sequenceFromLabels([node.label for node in l_nodes_list])
    r_nodes = part.nodes.sequenceFromLabels([node.label for node in r_nodes_list])
    b_nodes = part.nodes.sequenceFromLabels([node.label for node in b_nodes_list])

    part.Set(name='l_nodes', nodes=l_nodes)
    part.Set(name='r_nodes', nodes=r_nodes)
    part.Set(name='b_nodes', nodes=b_nodes)
    log_step(logger, '边界节点集已创建: 左=%d, 右=%d, 底=%d',
             len(l_nodes), len(r_nodes), len(b_nodes))

    # ============ 材料参数计算 ============
    GG = density * cs ** 2                    # 剪切模量
    EE = 2 * GG * (1 + vv)                    # 弹性模量
    lam = 2 * GG * vv / (1 - 2 * vv)          # 拉梅常数 λ
    cp = math.sqrt((lam + 2 * GG) / density)  # 纵波波速
    log_step(logger, '波速/材料参数已计算: G=%.6e, E=%.6e, lam=%.6e, cp=%.3f', GG, EE, lam, cp)

    # ============ 获取模型尺寸 ============
    part = mdb.models['Model-1'].parts['Part-1']
    l_nodes = part.sets['l_nodes'].nodes
    l_ymax_node = max(l_nodes, key=lambda node: node.coordinates[1])
    xmin = l_ymax_node.coordinates[0]
    ymax_l = l_ymax_node.coordinates[1]

    b_nodes = part.sets['b_nodes'].nodes
    ymin = b_nodes[0].coordinates[1]

    r_nodes = part.sets['r_nodes'].nodes
    r_ymax_node = max(r_nodes, key=lambda node: node.coordinates[1])
    xmax = r_ymax_node.coordinates[0]
    ymax_r = r_ymax_node.coordinates[1]

    ymax = max(ymax_l, ymax_r)
    log_step(logger, '由边界节点得到模型尺寸: Lx=%.3f, Ly=%.3f',
             xmax - xmin, ymax - ymin)

    # ============ 计算节点影响长度 ============
    def get_node_influence(part, set_name, sort_axis='y', ascending=False):
        """
        获取边界节点的影响长度（半距离），返回 [n, 4] 数组：节点号、x、y、影响长度
        """
        nodes = part.sets[set_name].nodes
        node_data = []
        for node in nodes:
            node_data.append([node.label, node.coordinates[0], node.coordinates[1]])

        node_data = np.array(node_data)
        axis = 1 if sort_axis == 'x' else 2
        node_data = node_data[node_data[:, axis].argsort()]
        if not ascending:
            node_data = node_data[::-1]

        n = node_data.shape[0]
        influence = np.zeros(n)
        for i in range(n):
            if i == 0:
                influence[i] = abs(node_data[i, axis] - node_data[i + 1, axis]) / 2
            elif i == n - 1:
                influence[i] = abs(node_data[i, axis] - node_data[i - 1, axis]) / 2
            else:
                influence[i] = abs(node_data[i - 1, axis] - node_data[i + 1, axis]) / 2

        node_data = np.hstack((node_data, influence.reshape(-1, 1)))
        return node_data

    part = mdb.models['Model-1'].parts['Part-1']
    node_data_l = get_node_influence(part, 'l_nodes', sort_axis='y', ascending=False)
    node_data_r = get_node_influence(part, 'r_nodes', sort_axis='y', ascending=False)
    node_data_b = get_node_influence(part, 'b_nodes', sort_axis='x', ascending=True)
    log_step(logger, '节点影响长度已计算')

    # ============ 粘弹性人工边界参数（刘晶波公式） ============
    kn = GG / 2 / ymax       # 法向弹簧刚度系数
    cn = density * cp         # 法向阻尼系数
    kt = GG / 4 / ymax       # 切向弹簧刚度系数
    ct = density * cs         # 切向阻尼系数
    log_step(logger, '弹簧-阻尼系数已计算: kn=%.6e, cn=%.6e, kt=%.6e, ct=%.6e', kn, cn, kt, ct)

    def add_spring_damper(node_data):
        """将弹簧刚度和阻尼系数乘以影响长度，追加到 node_data"""
        influence = node_data[:, 3]
        kns = kn * influence
        cns = cn * influence
        kts = kt * influence
        cts = ct * influence
        return np.hstack((node_data,
                           kns.reshape(-1, 1),
                           cns.reshape(-1, 1),
                           kts.reshape(-1, 1),
                           cts.reshape(-1, 1)))

    node_data_l = add_spring_damper(node_data_l)
    node_data_r = add_spring_damper(node_data_r)
    node_data_b = add_spring_damper(node_data_b)
    log_step(logger, '弹簧-阻尼系数已分配到所有边界节点')

    # ============ 在Abaqus中添加弹簧-阻尼器到地面 ============
    model = mdb.models['Model-1']
    assembly = model.rootAssembly
    instance = assembly.instances['Part-1-1']

    def add_spring_dashpot(node_data, prefix, dof_n, dof_t):
        """为每个边界节点添加法向和切向弹簧-阻尼器"""
        for row in node_data:
            node_label = int(row[0])
            kn = row[4]
            cn = row[5]
            kt = row[6]
            ct = row[7]
            node_array = instance.nodes.sequenceFromLabels([node_label])
            if len(node_array) == 0:
                logger.warning('创建弹簧-阻尼器时，实例中不存在节点 %d', node_label)
                continue
            region = Region(nodes=node_array)
            assembly.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_normal'.format(prefix, node_label),
                region=region, orientation=None, dof=dof_n,
                springBehavior=ON, springStiffness=kn,
                dashpotBehavior=ON, dashpotCoefficient=cn)
            assembly.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_tangent'.format(prefix, node_label),
                region=region, orientation=None, dof=dof_t,
                springBehavior=ON, springStiffness=kt,
                dashpotBehavior=ON, dashpotCoefficient=ct)

    # 左/右边界: dof_n=1(x方向), dof_t=2(y方向)
    # 底边界:    dof_n=2(y方向), dof_t=1(x方向)
    add_spring_dashpot(node_data_l, prefix='l', dof_n=1, dof_t=2)
    add_spring_dashpot(node_data_r, prefix='r', dof_n=1, dof_t=2)
    add_spring_dashpot(node_data_b, prefix='b', dof_n=2, dof_t=1)
    log_step(logger, '弹簧-阻尼器创建完成')

    # ============ 入射角与反射系数计算 ============
    if angle == 0:
        angle = 1e-10  # 避免除零
    else:
        angle = round(angle, 4)
    log_step(logger, '入射角已规范化: 入射角=%.6f 度', angle)

    alpha = np.radians(angle)                    # SV波入射角(弧度)
    alpha_critical = np.arcsin(cs / cp)          # 临界角
    if alpha >= alpha_critical:
        raise ValueError('The incident angle is greater than or equal to the critical angle.')

    beta_p = np.arcsin(cp * np.sin(alpha) / cs)  # 反射P波角度

    # 反射系数 A1 (SV波反射系数), A2 (P波反射系数)
    numerator_A1 = cs ** 2 * np.sin(2 * alpha) * np.sin(2 * beta_p) - cp ** 2 * np.cos(2 * alpha) ** 2
    denominator_A1 = cs ** 2 * np.sin(2 * alpha) * np.sin(2 * beta_p) + cp ** 2 * np.cos(2 * alpha) ** 2
    A1 = numerator_A1 / denominator_A1

    numerator_A2 = 2 * cp * cs * np.sin(2 * alpha) * np.cos(2 * alpha)
    A2 = numerator_A2 / denominator_A1
    log_step(logger, '反射参数已计算: alpha=%.6f 弧度, beta_p=%.6f 弧度, A1=%.6f, A2=%.6f',
             alpha, beta_p, A1, A2)

    # ============ 读取速度时程并积分得到位移 ============
    if not os.path.exists(VELtxt):
        raise IOError('未找到速度文件: {}'.format(VELtxt))
    VEL = np.loadtxt(VELtxt)
    if VEL.ndim != 2 or VEL.shape[1] < 2 or VEL.shape[0] < 2:
        raise ValueError('VEL.txt 至少需要 2 行 2 列: [time, velocity]')
    time_arr = VEL[:, 0]
    vel = VEL[:, 1]
    dt = VEL[1, 0] - VEL[0, 0]
    if dt <= 0:
        raise ValueError('VEL.txt 中时间步长无效: dt 必须 > 0')
    log_step(logger, 'VEL 已读取: 样本数=%d, dt=%.6f, 总时长=%.6f',
             len(time_arr), dt, time_arr[-1])

    dis = np.zeros_like(vel)
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 梯形积分
    DIS = np.column_stack((time_arr, dis))
    log_step(logger, '速度已积分为位移')

    max_time = VEL[-1, 0]
    Ly = ymax - ymin  # 模型高度
    Lx = xmax - xmin  # 模型宽度
    log_step(logger, '用于延迟计算的模型尺寸: Lx=%.3f, Ly=%.3f', Lx, Ly)

    # ============ 计算各节点的波到达延迟时间 ============
    def calc_node_delay(node_data, boundary, alpha, beta_p, cs, cp, Ly, Lx):
        """
        计算节点延迟时间
        """
        n = node_data.shape[0]
        det = np.zeros((n, 4))
        det[:, 0] = node_data[:, 0]

        for i in range(n):
            x0 = node_data[i, 1]
            y0 = node_data[i, 2]

            if boundary == 'l':
                t1 = y0 * np.cos(alpha) / cs
                t2 = (2 * Ly - y0) * np.cos(alpha) / cs
                t3 = ((Ly - y0) / (cp * np.cos(beta_p))
                      + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs)
                det[i, 1] = t1
                det[i, 2] = t2
                det[i, 3] = t3

            elif boundary == 'r':
                t7 = y0 * np.cos(alpha) / cs + Lx * np.sin(alpha) / cs
                t8 = (2 * Ly - y0) * np.cos(alpha) / cs + Lx * np.sin(alpha) / cs
                t9 = ((Ly - y0) / (cp * np.cos(beta_p))
                      + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs
                      + Lx * np.sin(alpha) / cs)
                det[i, 1] = t7
                det[i, 2] = t8
                det[i, 3] = t9

            elif boundary == 'b':
                t4 = x0 * np.sin(alpha) / cs
                t5 = (2 * Ly + x0 * np.tan(alpha)) * np.cos(alpha) / cs
                t6 = (Ly / (cp * np.cos(beta_p))
                      + (Ly * np.cos(alpha) + x0 * np.sin(alpha)
                         - Ly * np.tan(beta_p) * np.sin(alpha)) / cs)
                det[i, 1] = t4
                det[i, 2] = t5
                det[i, 3] = t6

            else:
                raise ValueError("boundary must be 'l', 'r', or 'b'")

        return det

    det_l = calc_node_delay(node_data_l, 'l', alpha, beta_p, cs, cp, Ly, Lx)
    det_r = calc_node_delay(node_data_r, 'r', alpha, beta_p, cs, cp, Ly, Lx)
    det_b = calc_node_delay(node_data_b, 'b', alpha, beta_p, cs, cp, Ly, Lx)
    log_step(logger, '左/右/底 边界节点延迟时间已计算')

    # 如果最大延迟超过输入时程长度，则补零延长
    detmax = max(np.max(det_l[:, 1:]), np.max(det_r[:, 1:]), np.max(det_b[:, 1:]))
    if max_time < detmax:
        n_add = int(np.ceil((detmax - max_time) / dt))
        new_times = VEL[-1, 0] + dt * np.arange(1, n_add + 1)
        new_vel = np.zeros((n_add, 2))
        new_vel[:, 0] = new_times
        VEL = np.vstack([VEL, new_vel])
        DIS = np.vstack([DIS, new_vel])
        log_step(logger, 'VEL/DIS 已用零延长: 增加行数=%d, 新总时长=%.6f',
                 n_add, VEL[-1, 0])
    else:
        log_step(logger, 'VEL/DIS 无需延长')

    # ============ 延迟时间对齐到时间步 ============
    def round_delay(det, dt):
        det[:, 1:4] = np.round(det[:, 1:4] / dt) * dt
        return det

    det_l = round_delay(det_l, dt)
    det_r = round_delay(det_r, dt)
    det_b = round_delay(det_b, dt)
    log_step(logger, '延迟时间已对齐到 dt 网格')

    # ============ 信号延迟工具函数 ============
    def delay_signal(u0, delay_t, dt):
        """将信号延迟 delay_t 时间"""
        n_delay = int(np.round(delay_t / dt))
        N = u0.shape[0]
        new_len = N + n_delay
        delayed = np.zeros((new_len, 2))
        delayed[:, 0] = np.arange(new_len) * dt
        delayed[n_delay:, 1] = u0[:, 1]
        return delayed

    def pad_to(arr, length, dt):
        """将数组补零到指定长度"""
        if arr.shape[0] < length:
            pad = np.zeros((length - arr.shape[0], 2))
            pad[:, 0] = np.arange(arr.shape[0], length) * dt
            arr = np.vstack([arr, pad])
        return arr

    # ============ 计算自由场位移和速度 ============
    field_data = {}  # 用字典存储中间结果

    def calc_freefield_u_and_dotu_general(node_data, det, timeseries, dt,
                                           alpha, beta_p, A1, A2,
                                           suffix1, suffix2, prefix):
        """
        对各边界（左、右、底）计算自由场 ux/uy 或 dotux/dotuy 时程
        """
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            idx = np.where(det[:, 0] == node_id)[0][0]
            tA = det[idx, 1]
            tB = det[idx, 2]
            tC = det[idx, 3]

            u0_tA = delay_signal(timeseries, tA, dt)
            u0_tB = delay_signal(timeseries, tB, dt)
            u0_tC = delay_signal(timeseries, tC, dt)

            max_len = max(u0_tA.shape[0], u0_tB.shape[0], u0_tC.shape[0])
            u0_tA = pad_to(u0_tA, max_len, dt)
            u0_tB = pad_to(u0_tB, max_len, dt)
            u0_tC = pad_to(u0_tC, max_len, dt)

            # 自由场位移/速度叠加（入射SV + 反射SV + 反射P）
            ux = (u0_tA[:, 1] * np.cos(alpha)
                  - A1 * u0_tB[:, 1] * np.cos(alpha)
                  + A2 * u0_tC[:, 1] * np.sin(beta_p))
            uy = (-u0_tA[:, 1] * np.sin(alpha)
                  - A1 * u0_tB[:, 1] * np.sin(alpha)
                  - A2 * u0_tC[:, 1] * np.cos(beta_p))

            ux_arr = np.zeros((max_len, 2))
            uy_arr = np.zeros((max_len, 2))
            ux_arr[:, 0] = u0_tA[:, 0]
            uy_arr[:, 0] = u0_tA[:, 0]
            ux_arr[:, 1] = ux
            uy_arr[:, 1] = uy

            field_data['{}-{}-{}'.format(node_id, prefix, suffix1)] = ux_arr
            field_data['{}-{}-{}'.format(node_id, prefix, suffix2)] = uy_arr

    # 计算位移自由场
    calc_freefield_u_and_dotu_general(node_data_l, det_l, DIS, dt, alpha, beta_p, A1, A2, 'ux', 'uy', 'l')
    calc_freefield_u_and_dotu_general(node_data_r, det_r, DIS, dt, alpha, beta_p, A1, A2, 'ux', 'uy', 'r')
    calc_freefield_u_and_dotu_general(node_data_b, det_b, DIS, dt, alpha, beta_p, A1, A2, 'ux', 'uy', 'b')
    log_step(logger, '左/右/底 自由场位移已计算')
    # 计算速度自由场
    calc_freefield_u_and_dotu_general(node_data_l, det_l, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'l')
    calc_freefield_u_and_dotu_general(node_data_r, det_r, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'r')
    calc_freefield_u_and_dotu_general(node_data_b, det_b, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'b')
    log_step(logger, '左/右/底 自由场速度已计算')

    # ============ 计算自由场应力 ============
    def calc_freefield_sigma_general(node_data, det, VEL, dt,
                                      alpha, beta_p, A1, A2,
                                      GG, cs, lam, cp, prefix):
        """
        对各边界（左、右、底）计算自由场应力 sigmax/sigmay 时程
        """
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            idx = np.where(det[:, 0] == node_id)[0][0]
            tA = det[idx, 1]
            tB = det[idx, 2]
            tC = det[idx, 3]

            v0_tA = delay_signal(VEL, tA, dt)
            v0_tB = delay_signal(VEL, tB, dt)
            v0_tC = delay_signal(VEL, tC, dt)

            max_len = max(v0_tA.shape[0], v0_tB.shape[0], v0_tC.shape[0])
            v0_tA = pad_to(v0_tA, max_len, dt)
            v0_tB = pad_to(v0_tB, max_len, dt)
            v0_tC = pad_to(v0_tC, max_len, dt)

            sin2a = np.sin(2 * alpha)
            cos2a = np.cos(2 * alpha)
            sinbp = np.sin(beta_p)
            sin2bp = np.sin(beta_p) ** 2
            sin2bp_2 = np.sin(2 * beta_p)
            cosbp = np.cos(beta_p)
            cosbp2 = cosbp ** 2

            if prefix == 'l':
                sigmax = (GG / cs * sin2a * (v0_tA[:, 1] - A1 * v0_tB[:, 1])
                          + A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC[:, 1])
                sigmay = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          - A2 * GG * sin2bp_2 / cp * v0_tC[:, 1])
            elif prefix == 'r':
                sigmax = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          - A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC[:, 1])
                sigmay = (GG / cs * cos2a * (-v0_tA[:, 1] - A1 * v0_tB[:, 1])
                          + A2 * GG * sin2bp_2 / cp * v0_tC[:, 1])
            elif prefix == 'b':
                sigmax = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          - A2 * GG * sin2bp_2 / cp * v0_tC[:, 1])
                sigmay = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB[:, 1])
                          + A2 * (lam + 2 * GG * cosbp2) / cp * v0_tC[:, 1])
            else:
                raise ValueError("prefix must be 'l', 'r' or 'b'")

            sigmax_arr = np.zeros((max_len, 2))
            sigmay_arr = np.zeros((max_len, 2))
            sigmax_arr[:, 0] = v0_tA[:, 0]
            sigmay_arr[:, 0] = v0_tA[:, 0]
            sigmax_arr[:, 1] = sigmax
            sigmay_arr[:, 1] = sigmay

            field_data['{}-{}-sigmax'.format(node_id, prefix)] = sigmax_arr
            field_data['{}-{}-sigmay'.format(node_id, prefix)] = sigmay_arr

    calc_freefield_sigma_general(node_data_l, det_l, VEL, dt, alpha, beta_p, A1, A2, GG, cs, lam, cp, 'l')
    calc_freefield_sigma_general(node_data_r, det_r, VEL, dt, alpha, beta_p, A1, A2, GG, cs, lam, cp, 'r')
    calc_freefield_sigma_general(node_data_b, det_b, VEL, dt, alpha, beta_p, A1, A2, GG, cs, lam, cp, 'b')
    log_step(logger, '左/右/底 自由场应力已计算')

    # ============ 计算等效节点力 ============
    def calc_equiv_node_force_general(node_data, prefix):
        """
        计算等效节点力 fx, fy
        node_data: [节点号, x, y, A, kn, cn, kt, ct]
        prefix: 'l'（左侧）、'r'（右侧）、'b'（底边）
        """
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            A = node_data[i, 3]       # 影响长度
            kn = node_data[i, 4]
            cn = node_data[i, 5]
            kt = node_data[i, 6]
            ct = node_data[i, 7]

            ux_arr = field_data['{}-{}-ux'.format(node_id, prefix)]
            dotux_arr = field_data['{}-{}-dotux'.format(node_id, prefix)]
            sigmax_arr = field_data['{}-{}-sigmax'.format(node_id, prefix)]
            uy_arr = field_data['{}-{}-uy'.format(node_id, prefix)]
            dotuy_arr = field_data['{}-{}-dotuy'.format(node_id, prefix)]
            sigmay_arr = field_data['{}-{}-sigmay'.format(node_id, prefix)]

            min_len = min(ux_arr.shape[0], dotux_arr.shape[0], sigmax_arr.shape[0],
                          uy_arr.shape[0], dotuy_arr.shape[0], sigmay_arr.shape[0])

            ux = ux_arr[:min_len, 1]
            dotux = dotux_arr[:min_len, 1]
            sigmax = sigmax_arr[:min_len, 1]
            uy = uy_arr[:min_len, 1]
            dotuy = dotuy_arr[:min_len, 1]
            sigmay = sigmay_arr[:min_len, 1]
            time = ux_arr[:min_len, 0]

            # 等效节点力 = 弹簧力 + 阻尼力 + 应力贡献
            if prefix in ('l', 'r'):
                fx = kn * ux + cn * dotux + A * sigmax
                fy = kt * uy + ct * dotuy + A * sigmay
            elif prefix == 'b':
                fx = kt * ux + ct * dotux + A * sigmax
                fy = kn * uy + cn * dotuy + A * sigmay
            else:
                raise ValueError("prefix must be 'l', 'r' or 'b'")

            fx_arr = np.zeros((min_len, 2))
            fy_arr = np.zeros((min_len, 2))
            fx_arr[:, 0] = time
            fy_arr[:, 0] = time
            fx_arr[:, 1] = fx
            fy_arr[:, 1] = fy

            field_data['{}-{}-fx'.format(node_id, prefix)] = fx_arr
            field_data['{}-{}-fy'.format(node_id, prefix)] = fy_arr

    calc_equiv_node_force_general(node_data_l, 'l')
    calc_equiv_node_force_general(node_data_r, 'r')
    calc_equiv_node_force_general(node_data_b, 'b')
    log_step(logger, '等效节点力时程已计算')

    # ============ 创建幅值曲线 (Amplitude) ============
    def batch_add_node_force_amplitude(node_data, prefix):
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            fx_arr = field_data['{}-{}-fx'.format(node_id, prefix)]
            fy_arr = field_data['{}-{}-fy'.format(node_id, prefix)]

            ampli_fx = tuple(tuple(row) for row in fx_arr)
            ampli_fy = tuple(tuple(row) for row in fy_arr)

            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)

            mdb.models['Model-1'].TabularAmplitude(
                data=ampli_fx, name=name_amp_fx,
                smooth=SOLVER_DEFAULT, timeSpan=STEP)
            mdb.models['Model-1'].TabularAmplitude(
                data=ampli_fy, name=name_amp_fy,
                smooth=SOLVER_DEFAULT, timeSpan=STEP)

    batch_add_node_force_amplitude(node_data_l, 'l')
    batch_add_node_force_amplitude(node_data_r, 'r')
    batch_add_node_force_amplitude(node_data_b, 'b')
    log_step(logger, '所有边界节点的幅值曲线已创建')

    # ============ 施加集中力载荷 ============
    def batch_add_node_force(node_data, prefix, step_name):
        a = mdb.models['Model-1'].rootAssembly
        instance_name = 'Part-1-1'
        n = a.instances[instance_name].nodes

        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)
            name_load_fx = 'load-{}-{}-fx'.format(node_id, prefix)
            name_load_fy = 'load-{}-{}-fy'.format(node_id, prefix)

            node_array = n.sequenceFromLabels([node_id])
            if len(node_array) == 0:
                logger.warning('施加载荷时，实例中不存在节点 %d (实例: %s)', node_id, instance_name)
                continue

            region = [node_array]
            mdb.models['Model-1'].ConcentratedForce(
                name=name_load_fx, createStepName=step_name,
                region=region, cf1=1.0, amplitude=name_amp_fx,
                distributionType=UNIFORM, field='', localCsys=None)
            mdb.models['Model-1'].ConcentratedForce(
                name=name_load_fy, createStepName=step_name,
                region=region, cf2=1.0, amplitude=name_amp_fy,
                distributionType=UNIFORM, field='', localCsys=None)

    batch_add_node_force(node_data_l, 'l', step_name)
    batch_add_node_force(node_data_r, 'r', step_name)
    batch_add_node_force(node_data_b, 'b', step_name)
    log_step(logger, '所有边界节点已施加集中力')
    log_step(logger, '斜入射人工边界完成: 总节点数=%d, 耗时=%.2fs',
             node_data_l.shape[0] + node_data_r.shape[0] + node_data_b.shape[0],
             time.time() - t0)


def submit_job(job_name='Job-VAB', num_cpus=7, memory_percent=90, logger=None):
    """
    创建并提交Abaqus作业

    参数:
        job_name       (str):  作业名称
        num_cpus       (int):  CPU数量
        memory_percent (int):  内存百分比
    """
    logger = logger or setup_logger()
    t0 = time.time()
    model_name = 'Model-1'
    log_step(logger, '提交作业开始: 作业名=%s, CPU 数量=%d, 内存=%d%%',
             job_name, num_cpus, memory_percent)

    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
        log_step(logger, '已删除已有作业: %s', job_name)

    mdb.Job(name=job_name, model=model_name,
            description='VAB oblique SV-wave analysis',
            type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0,
            queue=None, memory=memory_percent, memoryUnits=PERCENTAGE,
            getMemoryFromAnalysis=True, explicitPrecision=SINGLE,
            nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,
            contactPrint=OFF, historyPrint=OFF,
            numCpus=num_cpus, numDomains=num_cpus,
            multiprocessingMode=DEFAULT, numGPUs=0)
    log_step(logger, '作业对象已创建: %s', job_name)

    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    log_step(logger, '作业已提交: %s', job_name)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, '作业已完成: %s', job_name)
    log_step(logger, '提交作业完成: 耗时=%.2fs', time.time() - t0)


if __name__ == '__main__':
    logger = setup_logger()
    total_start = time.time()
    try:
        log_step(logger, '脚本入口')
        # ===== 参数设置 =====
        angle = 15        # SV波入射角度（度）
        E = 20e9          # 杨氏模量 (Pa) (相当于之前的剪切波速 1754 m/s)
        vv = 0.3          # 泊松比
        density = 2500    # 密度 (kg/m³)

        # 由杨氏模量自动计算剪切波速 (m/s)
        cs = math.sqrt((E / (2 * (1 + vv))) / density)

        width = 2000.0     # 模型宽度 (m)
        height = 1000.0    # 模型高度 (m)
        mesh_size = 5.0   # 网格尺寸 (m)
        log_step(logger, '输入参数已准备: 入射角=%.3f, 杨氏模量E=%.6e, 泊松比=%.3f, 密度=%.3f, 宽度=%.3f, 高度=%.3f, 网格=%.3f',
                 angle, E, vv, density, width, height, mesh_size)

        # ============ 自动读取VEL.txt文件里的分析步总时长和固定增量步大小 ============
        time_period = 2.0     # 默认分析步总时长 (s) (找不到文件时的备用值)
        initial_inc = 0.001   # 默认初始增量步 (s) (找不到文件时的备用值)

        VELtxt = 'VEL.txt'
        if os.path.exists(VELtxt):
            vel_data = np.loadtxt(VELtxt)
            if vel_data.ndim == 2 and vel_data.shape[0] >= 2 and vel_data.shape[1] >= 2:
                time_arr = vel_data[:, 0]
                dt = time_arr[1] - time_arr[0]
                if dt > 0:
                    time_period = time_arr[-1]   # 分析步总时长 (s)
                    initial_inc = dt             # 初始增量步 (s)
                    log_step(logger, '已从 VEL.txt 自动设置分析步: 时长=%.6f, 初始增量=%.6f',
                             time_period, initial_inc)
                else:
                    log_step(logger, 'VEL.txt 中 dt <= 0，分析步设置回退默认值')
            else:
                log_step(logger, 'VEL.txt 格式无效，分析步设置回退默认值')
        else:
            log_step(logger, '未找到 VEL.txt，分析步设置回退默认值')

        job_name = 'Job-VAB'  # 作业名称
        num_cpus = 7         # CPU数量
        cae_name = 'VAB_oblique.cae'  # CAE文件名（可修改）
        log_step(logger, '作业参数已准备: 作业名=%s, CPU 数量=%d', job_name, num_cpus)

        # ===== 执行流程 =====
        # 1. 创建模型（几何、材料、网格、装配、分析步）
        create_model(width, height, cs, vv, density, mesh_size,
                     time_period, initial_inc, logger=logger)

        # 第1步完成后保存CAE文件以便检查模型基础信息
        mdb.saveAs(pathName=cae_name)
        log_step(logger, '已保存 CAE 快照: %s', cae_name)

        # 2. 施加粘弹性人工边界和等效节点力
        VAB_oblique(angle, cs, vv, density, logger=logger)
        # 步骤2：在同一个CAE文件上保存添加了边界和载荷后的状态
        mdb.save()
        log_step(logger, '已在人工边界设置后保存 CAE')

        # 3. 提交作业
        submit_job(job_name=job_name, num_cpus=num_cpus, logger=logger)
        # 步骤3：最后即使提交了作业也再保存一次最终状态
        mdb.save()
        log_step(logger, '最终 CAE 已保存；总耗时=%.2fs', time.time() - total_start)
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise
