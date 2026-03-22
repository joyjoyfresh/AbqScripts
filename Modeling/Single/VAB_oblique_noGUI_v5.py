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


def find_vel_txt(logger=None):
    """
    自动在当前工作目录查找 .txt 文件作为速度时程输入。
    - 仅1个 txt 文件：直接使用
    - 多个 txt 文件：优先使用名称含 'vel' 的（不区分大小写），否则取字母序第一个并警告
    - 0个 txt 文件：抛出异常
    """
    cwd = os.getcwd()
    txt_files = [f for f in os.listdir(cwd) if f.lower().endswith('.txt')]

    if len(txt_files) == 0:
        raise IOError('当前目录 {} 下未找到任何 .txt 文件'.format(cwd))

    if len(txt_files) == 1:
        path = txt_files[0]
        if logger:
            log_step(logger, '自动识别速度时程文件: %s', path)
        return path

    vel_files = [f for f in txt_files if 'vel' in f.lower()]
    if len(vel_files) == 1:
        path = vel_files[0]
        if logger:
            log_step(logger, '自动识别速度时程文件 (多文件中匹配 vel): %s', path)
        return path

    path = sorted(txt_files)[0]
    if logger:
        log_step(logger, '当前目录存在多个 .txt 文件，自动选用: %s（共 %d 个）', path, len(txt_files))
    return path


def log_step(logger=None, message=None, *args, **kwargs):
    """
    合并的日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。

    初始化:    logger = log_step(log_file='VAB-oblique.log')
    记录日志:  log_step(logger, '消息 %s', val)
    """
    if not hasattr(log_step, '_logger'):
        log_file = kwargs.get('log_file', 'VAB-oblique.log')

        logging.addLevelName(logging.DEBUG, '调试')
        logging.addLevelName(logging.INFO, '信息')
        logging.addLevelName(logging.WARNING, '警告')
        logging.addLevelName(logging.ERROR, '错误')
        logging.addLevelName(logging.CRITICAL, '严重')

        _logger = logging.getLogger()
        _logger.setLevel(logging.INFO)
        _logger.propagate = False

        if not _logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setFormatter(formatter)
            _logger.addHandler(file_handler)

        log_step._logger = _logger
        log_step._start_time = time.time()

    if message is not None:
        now = time.time()
        delta_total = now - log_step._start_time
        log_step._logger.info('[总用时%.3fs] ' + message, delta_total, *args)

    return log_step._logger


def get_time_params_from_vel(logger=None):
    """
    从 VEL.txt 文件读取分析步总时长和固定增量步大小。

    参数:
        logger: 日志对象
    返回:
        (time_period, initial_inc): 分析步总时长和初始增量步
    """
    time_period = 2.0   # 默认分析步总时长 (s)
    initial_inc = 0.001  # 默认初始增量步 (s)

    try:
        VELtxt = find_vel_txt(logger)
        vel_data = np.loadtxt(VELtxt)
        if vel_data.ndim == 2 and vel_data.shape[0] >= 2 and vel_data.shape[1] >= 2:
            time_arr = vel_data[:, 0]
            dt = time_arr[1] - time_arr[0]
            if dt > 0:
                time_period = time_arr[-1]
                initial_inc = dt
                log_step(logger, '已从 %s 自动设置分析步: 时长=%.6f, 初始增量=%.6f',
                         VELtxt, time_period, initial_inc)
            else:
                log_step(logger, '%s 中 dt <= 0，将使用默认值', VELtxt)
        else:
            log_step(logger, '%s 格式无效，将使用默认值', VELtxt)
    except Exception as e:
        log_step(logger, '读取速度时程文件失败: %s，将使用默认值', str(e))

    return time_period, initial_inc


def create_model(total_L, h, i, cs, vv, density, mesh_size,
                 time_period, initial_inc, H_lower=None, cae_name=None, logger=None):
    """
    创建二维平面应变模型：几何、材料、截面、装配、分析步、网格

    参数:
        total_L     (float): 模型总水平长度 (m)
        h           (float): 斜坡高度 (m)
        i           (float): 斜坡倾角 (°)
        cs          (float): 剪切波速 (m/s)
        vv          (float): 泊松比
        density     (float): 密度 (kg/m³)
        mesh_size   (float): 网格尺寸 (m)
        time_period (float): 分析步总时长 (s)
        initial_inc (float): 固定增量步大小 (s)
        H_lower     (float): 下垫面高度 (m)，默认为 2*h
    几何逻辑（6个关键点，逆时针闭合）:
        w_slope = h / tan(i)         斜坡水平投影宽度
        L_rem   = total_L - w_slope  剩余水平长度
        L_flat  = L_rem / 2          左/右平台水平长度
        P1=(0, 0),  P2=(total_L, 0),
        P3=(total_L, H_lower),  P4=(L_flat+w_slope, H_lower),
        P5=(L_flat, H_lower+h), P6=(0, H_lower+h)
    """
    logger = logger or log_step()
    t0 = time.time()
    n = 1
    while 'Model-%d' % n in mdb.models:
        n += 1
    model_name = 'Model-%d' % n

    if h <= 0:
        raise ValueError('h 必须 > 0')
    if i <= 0 or i >= 90:
        raise ValueError('倾角 i 必须在 (0, 90) 范围内')
    if H_lower <= 0:
        raise ValueError('H_lower 必须 > 0')

    w_slope = h / math.tan(math.radians(i))
    L_rem = total_L - w_slope
    if L_rem <= 0:
        raise ValueError('斜坡水平投影 w_slope=%.3f 超过总长 total_L=%.3f' % (w_slope, total_L))
    L_flat = L_rem / 2.0
    H_upper = H_lower + h   # 左侧（上覆）地表高度

    log_step(logger, '创建模型开始: total_L=%.3f, h=%.3f, i=%.4f°, H_lower=%.3f, H_upper=%.3f, w_slope=%.3f, L_flat=%.3f, cs=%.3f, vv=%.3f, density=%.3f, mesh=%.3f',
             total_L, h, i, H_lower, H_upper, w_slope, L_flat, cs, vv, density, mesh_size)

    model = mdb.Model(name=model_name)
    log_step(logger, '已创建模型: %s', model_name)

    # ============ 创建二维坡地 Part（6节点多边形） ============
    pn = 1
    while 'Part-%d' % pn in model.parts:
        pn += 1
    part_name = 'Part-%d' % pn
    # P1(0,0) → P2(total_L,0) → P3(total_L,H_lower) → P4(L_flat+w_slope,H_lower)
    #         → P5(L_flat,H_upper) → P6(0,H_upper) → 闭合
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(total_L, H_upper) * 2)
    s.Line(point1=(0.0, 0.0),                   point2=(total_L, 0.0))               # 底边
    s.Line(point1=(total_L, 0.0),               point2=(total_L, H_lower))           # 右边界
    s.Line(point1=(total_L, H_lower),           point2=(L_flat + w_slope, H_lower))  # 右平台地表
    s.Line(point1=(L_flat + w_slope, H_lower),  point2=(L_flat, H_upper))            # 斜坡
    s.Line(point1=(L_flat, H_upper),            point2=(0.0, H_upper))               # 左平台地表
    s.Line(point1=(0.0, H_upper),               point2=(0.0, 0.0))                   # 左边界
    log_step(logger, '已创建6节点坡地草图: L_flat=%.3f, w_slope=%.3f, H_lower=%.3f, H_upper=%.3f',
             L_flat, w_slope, H_lower, H_upper)
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del model.sketches['__profile__']
    log_step(logger, '已创建零件并生成壳基体: %s', part_name)

    # ============ 材料与截面 ============
    GG = density * cs ** 2
    EE = 2 * GG * (1 + vv)
    log_step(logger, '材料参数已计算: G=%.6e, E=%.6e', GG, EE)

    mat_n = 1
    while 'Soil-%d' % mat_n in model.materials:
        mat_n += 1
    mat_name = 'Soil-%d' % mat_n
    mat = model.Material(name=mat_name)
    mat.Elastic(table=((EE, vv),))
    mat.Density(table=((density,),))
    log_step(logger, '材料已定义: %s', mat_name)

    model.HomogeneousSolidSection(name='Section-1', material=mat_name,
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
    in_ = 1
    while '%s-%d' % (part_name, in_) in assembly.instances:
        in_ += 1
    inst_name = '%s-%d' % (part_name, in_)
    assembly.Instance(name=inst_name, part=part, dependent=ON)
    log_step(logger, '装配实例已创建: %s', inst_name)

    # ============ 分析步 ============
    model.ImplicitDynamicsStep(
        name='step-earthquake', previous='Initial',
        timePeriod=time_period, timeIncrementationMethod=FIXED, initialInc=initial_inc,
        maxNumInc=1000000,
        nlgeom=OFF, application=MODERATE_DISSIPATION)
    log_step(logger, '动力分析步已创建: 时长=%.6f, 初始增量=%.6f',
             time_period, initial_inc)

    # 修改：设置场输出请求，每10帧数据都能输出到 odb 文件中
    model.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'E', 'U', 'V', 'A', 'RF'),
        frequency=10)
    log_step(logger, '场输出请求已更新: 频率=10')

    # ============ 网格划分 ============
    # 坡地多边形优先使用自由网格，避免结构化网格在斜顶边失败。
    pickedRegions = part.faces
    part.setMeshControls(regions=pickedRegions, elemShape=QUAD_DOMINATED, technique=FREE)
    log_step(logger, '网格控制已设置: 四边形主导 + 自由网格')

    # 近似全局尺寸设为 mesh_size
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    log_step(logger, '已播种网格: 尺寸=%.3f', mesh_size)

    # 指定单元类型：平面应变四节点/三节点单元
    elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)
    part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1, elemType2))
    part.generateMesh()
    log_step(logger, '已生成网格: CPE4/CPE3 单元')

    # 重新生成装配体以同步网格
    assembly.regenerate()
    log_step(logger, '装配已重新生成')
    log_step(logger, '创建模型完成: 节点=%d, 单元=%d, 耗时=%.2fs',
             len(part.nodes), len(part.elements), time.time() - t0)
    if cae_name:
        mdb.saveAs(pathName=cae_name)
        log_step(logger, '已保存 CAE 快照: %s', cae_name)
    return model_name, part_name, inst_name


def VAB_oblique(angle, cs, vv, density, model_name='Model-1', part_name='Part-1', inst_name='Part-1-1', logger=None):
    """
    主函数：为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力

    参数:
        angle  (float): SV波入射角度（度），0为垂直入射
        cs     (float): 剪切波速 (m/s)
        vv     (float): 泊松比
        density(float): 密度 (kg/m³)
    """
    # ============ 基本参数 ============
    logger = logger or log_step()
    t0 = time.time()
    VELtxt = find_vel_txt(logger)  # 自动查找当前目录的速度时程文件（两列：时间, 速度）
    step_name = 'step-earthquake'  # 地震分析步名称
    log_step(logger, '斜入射人工边界开始: 入射角=%.3f, 剪切波速=%.3f, 泊松比=%.3f, 密度=%.3f',
             angle, cs, vv, density)

    # ============ 获取装配体和Part ============
    a = mdb.models[model_name].rootAssembly
    a.regenerate()
    log_step(logger, '已加载装配并重新生成')

    part = mdb.models[model_name].parts[part_name]
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
    part = mdb.models[model_name].parts[part_name]
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

    part = mdb.models[model_name].parts[part_name]
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
    model = mdb.models[model_name]
    assembly = model.rootAssembly
    instance = assembly.instances[inst_name]

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

            mdb.models[model_name].TabularAmplitude(
                data=ampli_fx, name=name_amp_fx,
                smooth=SOLVER_DEFAULT, timeSpan=STEP)
            mdb.models[model_name].TabularAmplitude(
                data=ampli_fy, name=name_amp_fy,
                smooth=SOLVER_DEFAULT, timeSpan=STEP)

    batch_add_node_force_amplitude(node_data_l, 'l')
    batch_add_node_force_amplitude(node_data_r, 'r')
    batch_add_node_force_amplitude(node_data_b, 'b')
    log_step(logger, '所有边界节点的幅值曲线已创建')

    # ============ 施加集中力载荷 ============
    def batch_add_node_force(node_data, prefix, step_name):
        a = mdb.models[model_name].rootAssembly
        instance_name = inst_name
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
            mdb.models[model_name].ConcentratedForce(
                name=name_load_fx, createStepName=step_name,
                region=region, cf1=1.0, amplitude=name_amp_fx,
                distributionType=UNIFORM, field='', localCsys=None)
            mdb.models[model_name].ConcentratedForce(
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
    mdb.save()
    log_step(logger, '已在人工边界设置后保存 CAE')


def submit_job(num_cpus=7, memory_percent=90, model_name='Model-1', logger=None):
    """
    创建并提交Abaqus作业

    参数:
        job_name       (str):  作业名称
        num_cpus       (int):  CPU数量
        memory_percent (int):  内存百分比
    """
    logger = logger or log_step()
    t0 = time.time()
    n = 1
    while 'Job-%d' % n in mdb.jobs:
        n += 1
    job_name = 'Job-%d' % n
    log_step(logger, '提交作业开始: 作业名=%s, CPU 数量=%d, 内存=%d%%',
             job_name, num_cpus, memory_percent)

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

    mdb.save()
    log_step(logger, '最终 CAE 已保存')
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    log_step(logger, '作业已提交: %s', job_name)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, '作业已完成: %s', job_name)
    log_step(logger, '提交作业完成: 耗时=%.2fs', time.time() - t0)


if __name__ == '__main__':
    log_file = 'VAB-oblique.log'  # 日志文件名（可修改）
    logger = log_step(log_file=log_file)
    total_start = time.time()
    try:
        log_step(logger, '脚本入口')
        # ===== 土体参数设置 =====
        angle = 15        # *SV波入射角度（度）
        E = 20e9          # *杨氏模量 (Pa)
        vv = 0.3          # *泊松比
        density = 2500    # *密度 (kg/m³)
        cs = math.sqrt((E / (2 * (1 + vv))) / density)  # 由杨氏模量自动计算剪切波速 (m/s)

        # ====== 模型参数设置 =======
        h = 200.0                 # *斜坡高度 (m)
        i = 30.0                  # *斜坡倾角 (°)
        mesh_size_manual = 5      # *手动设置网格尺寸 (m)
        f_max = 3.33              # *目标最高频率 (Hz)，用于自动计算网格尺寸
        n_per_wave = 10           # *每波长最少单元数（建议 8~10）
        H_lower = 2.0 * h   # 下垫面高度 = 2h
        total_L = 6.0 * h + h / math.tan(math.radians(i))  # 总长 = 左平台3h + 坡宽 + 右平台3h
        mesh_size_auto = cs / (f_max * n_per_wave)   # 自动计算网格尺寸 = Vs / (f_max * n)
        mesh_size = min(mesh_size_auto, mesh_size_manual)  # 取自动与手动中的较小值

        # ====== 作业参数设置 ======
        num_cpus = 7                          # *CPU数量
        cae_name = 'VAB_oblique_slope.cae'    # *CAE文件名（可修改）
        log_step(logger, '输入参数已准备')
        time_period, initial_inc = get_time_params_from_vel(logger=logger)   # 自动读取VEL.txt文件里的分析步总时长和固定增量步大小

        # ======= 执行流程 ========
        # 1. 创建模型（几何、材料、网格、装配、分析步）
        model_name, part_name, inst_name = create_model(total_L, h, i, cs, vv, density, mesh_size,
                     time_period, initial_inc, H_lower=H_lower, cae_name=cae_name, logger=logger)
        # 2. 施加粘弹性人工边界和等效节点力
        VAB_oblique(angle, cs, vv, density, model_name=model_name, part_name=part_name, inst_name=inst_name, logger=logger)
        # 3. 提交作业
        submit_job(num_cpus=num_cpus, model_name=model_name, logger=logger)
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise