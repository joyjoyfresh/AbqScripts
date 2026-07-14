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




def wave_vectors(wave_type, direction, angle, G, lam, c):
    """
    返回统一波场系数: (ux, uy, sig_xx, sig_yy, tau_xy)
    所有表达式由解析统一推导, 用于自由场应力和等效力计算
    """
    p = math.sin(angle) / c
    if direction == 'up':
        q = math.cos(angle) / c
        if wave_type == 'SV':
            ux, uy = math.cos(angle), -math.sin(angle)
        else: # P
            ux, uy = math.sin(angle), math.cos(angle)
    else: # down
        q = - math.cos(angle) / c
        if wave_type == 'SV':
            ux, uy = -math.cos(angle), -math.sin(angle)
        else: # P
            ux, uy = math.sin(angle), -math.cos(angle)

    sig_xx = - (lam + 2*G) * p * ux - lam * q * uy
    sig_yy = - (lam + 2*G) * q * uy - lam * p * ux
    tau_xy = - G * (q * ux + p * uy)

    return ux, uy, sig_xx, sig_yy, tau_xy
    
def _compute_interface_sv_coeff(alpha1, cs1, cp1, rho1, cs2, cp2, rho2, lam1, lam2):
    p = math.sin(alpha1) / cs1
    if p * cp1 >= 1.0 or p * cs2 >= 1.0 or p * cp2 >= 1.0:
        raise ValueError("Critical angle exceeded at interface.")
    
    beta1 = math.asin(p * cp1)
    alpha2 = math.asin(p * cs2)
    beta2 = math.asin(p * cp2)
    
    G1 = rho1 * cs1**2
    G2 = rho2 * cs2**2
    
    A = np.zeros((4, 4))
    B = np.zeros(4)
    # x = [Rss, Rsp, Tss, Tsp]^T
    A[0, 0] = -math.cos(alpha1); A[0, 1] = math.sin(beta1)
    A[0, 2] = -math.cos(alpha2); A[0, 3] = -math.sin(beta2)
    B[0] = -math.cos(alpha1)
    
    A[1, 0] = -math.sin(alpha1); A[1, 1] = -math.cos(beta1)
    A[1, 2] = math.sin(alpha2); A[1, 3] = -math.cos(beta2)
    B[1] = math.sin(alpha1)
    
    ux, uy, sxx, syy, sxy = wave_vectors('SV', 'up', alpha1, G1, lam1, cs1); tau_inc, sig_inc = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('SV', 'down', alpha1, G1, lam1, cs1); tau_rss, sig_rss = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('P', 'down', beta1, G1, lam1, cp1); tau_rsp, sig_rsp = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('SV', 'up', alpha2, G2, lam2, cs2); tau_tss, sig_tss = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('P', 'up', beta2, G2, lam2, cp2); tau_tsp, sig_tsp = sxy, syy
    
    A[2, 0] = tau_rss; A[2, 1] = tau_rsp; A[2, 2] = -tau_tss; A[2, 3] = -tau_tsp
    B[2] = -tau_inc
    
    A[3, 0] = sig_rss; A[3, 1] = sig_rsp; A[3, 2] = -sig_tss; A[3, 3] = -sig_tsp
    B[3] = -sig_inc
    
    X = np.linalg.solve(A, B)
    return X[0], X[1], X[2], X[3], beta1, alpha2, beta2

def _compute_free_surface_sv_coeff(alpha, cs, cp, rho, lam):
    p = math.sin(alpha) / cs
    beta = math.asin(p * cp)
    G = rho * cs**2
    ux, uy, sxx, syy, sxy = wave_vectors('SV', 'up', alpha, G, lam, cs); tau_inc, sig_inc = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('SV', 'down', alpha, G, lam, cs); tau_rss, sig_rss = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('P', 'down', beta, G, lam, cp); tau_rsp, sig_rsp = sxy, syy
    A = np.zeros((2, 2)); B = np.zeros(2)
    A[0, 0] = tau_rss; A[0, 1] = tau_rsp; B[0] = -tau_inc
    A[1, 0] = sig_rss; A[1, 1] = sig_rsp; B[1] = -sig_inc
    X = np.linalg.solve(A, B)
    return X[0], X[1], beta

def _compute_free_surface_p_coeff(alpha_p, cs, cp, rho, lam):
    p = math.sin(alpha_p) / cp
    beta_sv = math.asin(p * cs)
    G = rho * cs**2
    ux, uy, sxx, syy, sxy = wave_vectors('P', 'up', alpha_p, G, lam, cp); tau_inc, sig_inc = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('SV', 'down', beta_sv, G, lam, cs); tau_rss, sig_rss = sxy, syy
    ux, uy, sxx, syy, sxy = wave_vectors('P', 'down', alpha_p, G, lam, cp); tau_rsp, sig_rsp = sxy, syy
    A = np.zeros((2, 2)); B = np.zeros(2)
    A[0, 0] = tau_rss; A[0, 1] = tau_rsp; B[0] = -tau_inc
    A[1, 0] = sig_rss; A[1, 1] = sig_rsp; B[1] = -sig_inc
    X = np.linalg.solve(A, B)
    return X[0], X[1], beta_sv

DEFAULT_STEP_NAME = 'Step-earthquake'  # 定义默认分析步名称
BOUNDARY_SET_NAMES = ('Left_boundary', 'Right_boundary', 'Bottom_boundary')  # 定义基础边界节点集名称
BOUNDARY_SEQUENCE = ('l', 'r', 'b')  # 定义边界处理顺序


def _next_available_name(prefix, existing_container):
    """按前缀生成可用名称（如 Part-1, Part-2）。"""
    index = 1  # 初始化名称序号
    while '%s-%d' % (prefix, index) in existing_container:  # 循环查找未被占用的名称
        index += 1  # 若已存在则递增序号
    return '%s-%d' % (prefix, index)  # 返回首个可用名称


def _normalize_output_variables(variables):
    """规范化输出变量为元组，满足 Abaqus 接口要求。"""
    if isinstance(variables, str):  # 若为单个字符串
        return (variables,)  # 转为单元素元组
    if isinstance(variables, list):  # 若为列表
        return tuple(variables)  # 转为元组
    return variables  # 其他情况原样返回


def _compute_wave_speed_from_elastic_modulus(elastic_modulus, poisson_ratio, density):
    """根据杨氏模量、泊松比和密度计算剪切波速。"""
    return math.sqrt((elastic_modulus / (2 * (1 + poisson_ratio))) / density)  # 按弹性理论公式计算 Vs


def _compute_material_params(cs, vv, density):
    """根据剪切波速、泊松比和密度计算材料参数。"""  # 定义函数用途说明
    G = density * cs ** 2  # 计算剪切模量 G
    E = 2 * G * (1 + vv)  # 计算杨氏模量 E
    lam = E * vv / ((1 + vv) * (1 - 2 * vv))  # 计算拉梅常数 lambda
    cp = math.sqrt((lam + 2 * G) / density)  # 计算纵波波速 cp
    return G, E, lam, cp  # 返回后续计算所需的材料参数


def log_step(logger=None, message=None, *args):
    """
    日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。
    初始化:    logger = log_step('mylog.log')  # 传入日志文件名
               logger = log_step()            # 使用默认文件名 'logfile.log'
    记录日志:  log_step(logger, '消息 %s', val)
    """
    if not hasattr(log_step, '_logger'):
        if logger is not None and isinstance(logger, str):
            log_filename = logger
            logger = None
        else:
            log_filename = 'logfile.log'

        _logger = logging.getLogger('abqpy') # 日志器名称
        _logger.setLevel(logging.INFO)
        _logger.propagate = False

        _logger.handlers = []
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        log_step._logger = _logger
        log_step._start_time = time.time()
        log_step._log_filename = log_filename

        return _logger

    if message is not None:
        now = time.time()
        delta_total = now - log_step._start_time
        log_step._logger.info('[%.3fs] ' + message, delta_total, *args)

    return log_step._logger


def find_acc_txt(logger=None):
    """
    查找当前工作目录下所有 .txt 文件，并读取每个加速度文件的分析步时长和增量步。
    返回按文件名排序的列表: [(filename, time_period, initial_inc), ...]
    若无文件则抛出异常。
    """
    cwd = os.getcwd()
    txt_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.txt')])
    if len(txt_files) == 0:
        raise IOError('当前目录 {} 下未找到任何 .txt 文件'.format(cwd))

    result = []
    for f in txt_files:
        time_period = 2.0
        initial_inc = 0.001
        try:
            acc_data = np.loadtxt(f)
            if acc_data.ndim == 2 and acc_data.shape[0] >= 2 and acc_data.shape[1] >= 2:
                time_arr = acc_data[:, 0]
                dt = time_arr[1] - time_arr[0]
                if dt > 0:
                    time_period = time_arr[-1]
                    initial_inc = dt
                    if logger:
                        log_step(logger, '已从加速度文件 %s 读取分析步参数: 时长=%.2f, 初始增量=%.3f',
                                 f, time_period, initial_inc)
                else:
                    if logger:
                        log_step(logger, '%s 中 dt <= 0，将使用默认值', f)
            else:
                if logger:
                    log_step(logger, '%s 格式无效，将使用默认值', f)
        except Exception as e:
            if logger:
                log_step(logger, '读取加速度时程文件失败: %s，将使用默认值', str(e))
        result.append((f, time_period, initial_inc))

    return result


def create_model(total_L, h, i, cs1, vv1, density1, cs2, vv2, density2, mesh_size,
                 H_lower=None, cae_name=None,
                 logger=None):
    """
    创建二维平面应变模型：几何、材料、截面、装配、网格（不含分析步）
    参数:
        total_L     (float): 模型总水平长度 (m)
        h           (float): 斜坡高度 (m)
        i           (float): 斜坡倾角 (°)
        cs          (float): 剪切波速 (m/s)
        vv          (float): 泊松比
        density     (float): 密度 (kg/m³)
        mesh_size   (float): 网格尺寸 (m)
        H_lower     (float): 下垫面高度 (m)，默认为 2*h
    几何逻辑（6个关键点，逆时针闭合）:
        w_slope    = h / tan(i)                斜坡水平投影宽度
        left_flat  = 3h                         左平台固定长度
        right_flat = total_L - left_flat - w_slope  右平台长度（自动剩余）
        P1=(0, 0),  P2=(total_L, 0),
        P3=(total_L, H_lower),  P4=(left_flat+w_slope, H_lower),
        P5=(left_flat, H_lower+h), P6=(0, H_lower+h)
    """
    logger = logger or log_step()
    model_name = 'Model-1'

    if h <= 0:
        raise ValueError('h 必须 > 0')
    if i <= 0 or i >= 90:
        raise ValueError('倾角 i 必须在 (0, 90) 范围内')
    if H_lower is None:
        H_lower = 2.0 * h
    if H_lower <= 0:
        raise ValueError('H_lower 必须 > 0')

    w_slope = h / math.tan(math.radians(i))
    left_flat = 3.0 * h
    right_flat = total_L - left_flat - w_slope
    if right_flat <= 0:
        raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                         (total_L, left_flat, w_slope))
    H_upper = H_lower + h   # 左侧（上覆）地表高度
    
    if cae_name:
        mdb.saveAs(pathName=cae_name)
        log_step(logger, '工程文件保存为 %s', cae_name)
    model = mdb.Model(name=model_name)
    log_step(logger, '%s 基础模型开始创建', model_name)

    # ============ 创建二维坡地 Part（6节点多边形） ============
    part_name = _next_available_name('Part', model.parts)
    # P1(0,0) → P2(total_L,0) → P3(total_L,H_lower) → P4(left_flat+w_slope,H_lower)
    #         → P5(left_flat,H_upper) → P6(0,H_upper) → 闭合
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(total_L, H_upper) * 2)
    s.Line(point1=(0.0, 0.0),                   point2=(total_L, 0.0))                 # 底边
    s.Line(point1=(total_L, 0.0),               point2=(total_L, H_lower))             # 右边界
    s.Line(point1=(total_L, H_lower),           point2=(left_flat + w_slope, H_lower)) # 右平台地表
    s.Line(point1=(left_flat + w_slope, H_lower),  point2=(left_flat, H_upper))        # 斜坡
    s.Line(point1=(left_flat, H_upper),            point2=(0.0, H_upper))              # 左平台地表
    s.Line(point1=(0.0, H_upper),               point2=(0.0, 0.0))                     # 左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del model.sketches['__profile__']
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)

    # ============ 网格前按坡底点水平切分面，分为上下两部分 ============
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(
        name='__partition__', sheetSize=max(total_L, H_upper) * 2
    )
    partition_sketch.Line(point1=(left_flat + w_slope, H_lower), point2=(0.0, H_lower))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__partition__']
    log_step(logger, '%s 网格前切分完成: 上下两部分', model_name)

    # ============ 材料与双层截面分配 ============
    GG1 = density1 * cs1 ** 2
    EE1 = 2 * GG1 * (1 + vv1)
    mat_name_1 = _next_available_name('Material-L1', model.materials)
    mat1 = model.Material(name=mat_name_1)
    mat1.Elastic(table=((EE1, vv1),))
    mat1.Density(table=((density1,),))
    sec_name_1 = _next_available_name('Section-L1', model.sections)
    model.HomogeneousSolidSection(name=sec_name_1, material=mat_name_1, thickness=1.0)

    GG2 = density2 * cs2 ** 2
    EE2 = 2 * GG2 * (1 + vv2)
    mat_name_2 = _next_available_name('Material-L2', model.materials)
    mat2 = model.Material(name=mat_name_2)
    mat2.Elastic(table=((EE2, vv2),))
    mat2.Density(table=((density2,),))
    sec_name_2 = _next_available_name('Section-L2', model.sections)
    model.HomogeneousSolidSection(name=sec_name_2, material=mat_name_2, thickness=1.0)

    faces_L1_seq = part.faces.getByBoundingBox(yMax=H_lower + 1e-4)
    faces_L2_seq = part.faces.getByBoundingBox(yMin=H_lower - 1e-4)
    
    if len(faces_L1_seq) > 0:
        part.SectionAssignment(region=Region(faces=faces_L1_seq), sectionName=sec_name_1,
                               offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    if len(faces_L2_seq) > 0:
        part.SectionAssignment(region=Region(faces=faces_L2_seq), sectionName=sec_name_2,
                               offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    log_step(logger, '%s 双层截面已分配', model_name)

    # ============ 装配 ============
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    inst_name = _next_available_name(part_name, assembly.instances)
    assembly.Instance(name=inst_name, part=part, dependent=ON)
    log_step(logger, '%s 装配实例已创建: %s', model_name, inst_name)

    # ============ 网格划分 ============

    pickedRegions = part.faces
    part.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)
    log_step(logger, '%s 网格控制已设置: 四边形 + 结构化', model_name)

    # 近似全局尺寸设为 mesh_size
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    log_step(logger, '%s 已播种网格: 尺寸=%.0f', model_name, mesh_size)

    # 指定单元类型：平面应变四节点，取消缩减积分
    elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1,))
    part.generateMesh()
    log_step(logger, '%s 已生成网格: CPE4 单元', model_name)

    # 重新生成装配体以同步网格
    assembly.regenerate()
    log_step(logger, '%s 装配已重新生成', model_name)

    # ============ 创建边界节点集（左/右/底） ============
    x_list = [node.coordinates[0] for node in part.nodes]
    y_list = [node.coordinates[1] for node in part.nodes]
    xmin = min(x_list)
    xmax = max(x_list)
    ymin = min(y_list)
    tol = 1e-6

    l_nodes_list = [node for node in part.nodes if abs(node.coordinates[0] - xmin) < tol]
    r_nodes_list = [node for node in part.nodes if abs(node.coordinates[0] - xmax) < tol]
    b_nodes_list = [node for node in part.nodes if abs(node.coordinates[1] - ymin) < tol]

    l_labels = tuple(node.label for node in l_nodes_list)
    r_labels = tuple(node.label for node in r_nodes_list)
    b_labels = tuple(node.label for node in b_nodes_list)

    part.SetFromNodeLabels(nodeLabels=l_labels, name='Left_boundary')
    part.SetFromNodeLabels(nodeLabels=r_labels, name='Right_boundary')
    part.SetFromNodeLabels(nodeLabels=b_labels, name='Bottom_boundary')
    log_step(logger, '%s 边界节点集已在Part中创建: 左=%d, 右=%d, 底=%d', model_name, len(l_labels), len(r_labels), len(b_labels))

    # ============ 创建顶面节点集（用于后处理 PGA） ============
    # 按几何边界精确筛选：左平台 + 斜坡 + 右平台，避免把内部节点误判为顶面节点。
    top_tol = max(1e-6, mesh_size * 1e-3)  # 定义顶面识别容差，兼顾数值误差与网格尺度
    top_surface_labels = []  # 初始化顶面节点编号列表

    for node in part.nodes:  # 遍历全部节点并按几何方程判断是否位于顶面
        x = node.coordinates[0]  # 读取节点x坐标
        y = node.coordinates[1]  # 读取节点y坐标
        is_on_top = False  # 初始化“位于顶面”标记

        if (0.0 - top_tol) <= x <= (left_flat + top_tol):  # 判断是否位于左平台x范围
            if abs(y - H_upper) <= top_tol:  # 左平台顶面满足 y = H_upper
                is_on_top = True  # 标记为顶面节点
        elif (left_flat - top_tol) <= x <= (left_flat + w_slope + top_tol):  # 判断是否位于斜坡x范围
            y_slope = H_upper - (x - left_flat) * h / w_slope  # 计算当前x对应的斜坡理论y坐标
            if abs(y - y_slope) <= top_tol:  # 斜坡顶面满足线性方程
                is_on_top = True  # 标记为顶面节点
        elif (left_flat + w_slope - top_tol) <= x <= (total_L + top_tol):  # 判断是否位于右平台x范围
            if abs(y - H_lower) <= top_tol:  # 右平台顶面满足 y = H_lower
                is_on_top = True  # 标记为顶面节点

        if is_on_top:  # 若当前节点位于顶面
            top_surface_labels.append(node.label)  # 记录顶面节点编号

    top_surface_labels = tuple(sorted(set(top_surface_labels)))  # 去重并排序，生成Abaqus需要的节点标签元组
    if len(top_surface_labels) == 0:
        raise ValueError('%s 未识别到顶部边界节点，请检查几何参数与容差设置' % model_name)

    part.SetFromNodeLabels(nodeLabels=top_surface_labels, name='TOP_SURFACE')
    log_step(logger, '%s 顶面节点集已创建: TOP_SURFACE 节点数=%d', model_name, len(top_surface_labels))

    mdb.save() 
    return model_name, part_name, inst_name




def VAB_oblique(angle, cs1, vv1, density1, cs2, vv2, density2, H_lower,
                model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None):
    """
    双层模型主函数：施加粘弹性人工边界和等效节点力。
    """
    logger = logger or log_step()
    t0 = time.time()
    step_name = step_name or DEFAULT_STEP_NAME
    log_step(logger, '%s 双层模型开始创建人工边界', model_name)

    a = mdb.models[model_name].rootAssembly
    a.regenerate()
    model = mdb.models[model_name]
    part = model.parts[part_name]
    instance = a.instances[inst_name]

    missing_boundary_sets = [name for name in BOUNDARY_SET_NAMES if name not in part.sets]
    if missing_boundary_sets:
        raise KeyError('%s 缺少边界节点集: %s' % (model_name, '/'.join(missing_boundary_sets)))

    def get_instance_nodes_from_part_set(set_name):
        return instance.nodes.sequenceFromLabels(tuple(node.label for node in part.sets[set_name].nodes))

    G1, E1, lam1, cp1 = _compute_material_params(cs1, vv1, density1)
    G2, E2, lam2, cp2 = _compute_material_params(cs2, vv2, density2)

    l_nodes = get_instance_nodes_from_part_set('Left_boundary')
    b_nodes = get_instance_nodes_from_part_set('Bottom_boundary')
    r_nodes = get_instance_nodes_from_part_set('Right_boundary')

    ymax = max(max(node.coordinates[1] for node in l_nodes), max(node.coordinates[1] for node in r_nodes))
    xmax = max(node.coordinates[0] for node in r_nodes)

    def get_node_influence(nodes, sort_axis='y', ascending=False):
        node_data = np.array([[node.label, node.coordinates[0], node.coordinates[1]] for node in nodes], dtype=float)
        axis = 1 if sort_axis == 'x' else 2
        node_data = node_data[node_data[:, axis].argsort()]
        if not ascending: node_data = node_data[::-1]
        n, coord = node_data.shape[0], node_data[:, axis]
        influence = np.zeros(n)
        if n > 1:
            influence[0] = abs(coord[0] - coord[1]) / 2.0
            influence[-1] = abs(coord[-1] - coord[-2]) / 2.0
            if n > 2: influence[1:-1] = np.abs(coord[:-2] - coord[2:]) / 2.0
        return np.hstack((node_data, influence.reshape(-1, 1)))

    node_data_l = get_node_influence(l_nodes, sort_axis='y', ascending=False)
    node_data_r = get_node_influence(r_nodes, sort_axis='y', ascending=False)
    node_data_b = get_node_influence(b_nodes, sort_axis='x', ascending=True)

    def add_spring_damper_and_record(node_data):
        props = []
        for row in node_data:
            node_id, x, y, A = int(row[0]), row[1], row[2], row[3]
            is_L1 = (y <= H_lower + 1e-4)
            G_cur, rho_cur, cs_cur, cp_cur = (G1, density1, cs1, cp1) if is_L1 else (G2, density2, cs2, cp2)
            kn = G_cur / 2 / ymax * A
            cn = rho_cur * cp_cur * A
            kt = G_cur / 4 / ymax * A
            ct = rho_cur * cs_cur * A
            props.append([kn, cn, kt, ct])
        return np.hstack((node_data, np.array(props)))

    node_data_l = add_spring_damper_and_record(node_data_l)
    node_data_r = add_spring_damper_and_record(node_data_r)
    node_data_b = add_spring_damper_and_record(node_data_b)
    
    def apply_spring_dashpot(node_data, prefix, dof_n, dof_t):
        for row in node_data:
            node_id, kn, cn, kt, ct = int(row[0]), row[4], row[5], row[6], row[7]
            region = Region(nodes=instance.nodes.sequenceFromLabels([node_id]))
            a.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_normal'.format(prefix, node_id), region=region, dof=dof_n,
                springBehavior=ON, springStiffness=kn, dashpotBehavior=ON, dashpotCoefficient=cn)
            a.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_tangent'.format(prefix, node_id), region=region, dof=dof_t,
                springBehavior=ON, springStiffness=kt, dashpotBehavior=ON, dashpotCoefficient=ct)

    boundary_dof = {'l': (1, 2), 'r': (1, 2), 'b': (2, 1)}
    apply_spring_dashpot(node_data_l, 'l', *boundary_dof['l'])
    apply_spring_dashpot(node_data_r, 'r', *boundary_dof['r'])
    apply_spring_dashpot(node_data_b, 'b', *boundary_dof['b'])
    log_step(logger, '%s 弹簧-阻尼器已全部施加', model_name)

    angle = 1e-10 if angle == 0 else round(angle, 4)
    alpha1 = np.radians(angle)
    p = np.sin(alpha1) / cs1
    
    Rss, Rsp, Tss, Tsp, beta1, alpha2, beta2 = _compute_interface_sv_coeff(alpha1, cs1, cp1, density1, cs2, cp2, density2, lam1, lam2)
    A1_2, A2_2, _ = _compute_free_surface_sv_coeff(alpha2, cs2, cp2, density2, lam2)
    B1_2, B2_2, _ = _compute_free_surface_p_coeff(beta2, cs2, cp2, density2, lam2)

    ACC = np.loadtxt(acc_file)
    time_arr = ACC[:, 0]
    acc = ACC[:, 1]
    dt = time_arr[1] - time_arr[0]
    
    vel = np.zeros_like(acc); vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)
    VEL = np.column_stack((time_arr, vel))
    dis = np.zeros_like(vel); dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)
    DIS = np.column_stack((time_arr, dis))
    
    max_time = time_arr[-1]
    
    # 获取需要延迟的最大时间用于扩充数据
    # 计算波的最长传播时间
    c_min = min(cs1, cs2)
    max_delay = xmax / c_min + ymax / c_min * 4.0
    if max_time < max_delay + 5.0:
        n_add = int(np.ceil((max_delay + 5.0 - max_time) / dt))
        new_times = VEL[-1, 0] + dt * np.arange(1, n_add + 1)
        new_vel = np.zeros((n_add, 2)); new_vel[:, 0] = new_times
        VEL = np.vstack([VEL, new_vel])
        DIS = np.vstack([DIS, new_vel])

    def delay_signal(u0, delay_t, dt):
        n_delay = int(np.round(delay_t / dt))
        delayed = np.zeros((u0.shape[0] + n_delay, 2))
        delayed[:, 0] = np.arange(delayed.shape[0]) * dt
        delayed[n_delay:, 1] = u0[:, 1]
        return delayed

    cache_disp = {}; cache_vel = {}
    def get_delayed_disp(delay_t):
        n_delay = int(np.round(delay_t / dt))
        if n_delay not in cache_disp: cache_disp[n_delay] = delay_signal(DIS, n_delay * dt, dt)
        return cache_disp[n_delay]
        
    def get_delayed_vel(delay_t):
        n_delay = int(np.round(delay_t / dt))
        if n_delay not in cache_vel: cache_vel[n_delay] = delay_signal(VEL, n_delay * dt, dt)
        return cache_vel[n_delay]

    def pad_min_len(arrays):
        max_len = max(arr.shape[0] for arr in arrays)
        padded = []
        for arr in arrays:
            if arr.shape[0] < max_len:
                p = np.zeros((max_len, 2))
                p[:arr.shape[0], :] = arr
                p[arr.shape[0]:, 0] = arr[-1, 0] + dt * np.arange(1, max_len - arr.shape[0] + 1)
                padded.append(p)
            else:
                padded.append(arr)
        return max_len, padded

    H1 = H_lower
    H2 = ymax - H_lower
    p_wave = np.sin(alpha1) / cs1

    field_data = {}
    
    def process_boundary(node_data, prefix):
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            x0, y0 = node_data[i, 1], node_data[i, 2]
            A, kn, cn, kt, ct = node_data[i, 3:]

            is_L1 = (y0 <= H1 + 1e-4)
            base_t = x0 * p_wave
            
            if is_L1:
                waves = [
                    ('SV', 'up', alpha1, 1.0, base_t + y0 * math.cos(alpha1)/cs1),
                    ('SV', 'down', alpha1, Rss, base_t + (2*H1 - y0) * math.cos(alpha1)/cs1),
                    ('P',  'down', beta1,   Rsp, base_t + H1 * math.cos(alpha1)/cs1 + (H1 - y0) * math.cos(beta1)/cp1),
                ]
                G_cur, lam_cur, c_cur = G1, lam1, cs1
            else:
                y2 = y0 - H1
                t_SV_inc = base_t + H1 * math.cos(alpha1)/cs1
                waves = [
                    ('SV', 'up', alpha2, Tss, t_SV_inc + y2 * math.cos(alpha2)/cs2),
                    ('SV', 'down', alpha2, Tss * A1_2, t_SV_inc + (2*H2 - y2) * math.cos(alpha2)/cs2),
                    ('P', 'down', beta2, Tss * A2_2, t_SV_inc + H2 * math.cos(alpha2)/cs2 + (H2 - y2) * math.cos(beta2)/cp2),
                    ('P', 'up', beta2, Tsp, t_SV_inc + y2 * math.cos(beta2)/cp2),
                    ('SV', 'down', alpha2, Tsp * B1_2, t_SV_inc + H2 * math.cos(beta2)/cp2 + (H2 - y2) * math.cos(alpha2)/cs2),
                    ('P', 'down', beta2, Tsp * B2_2, t_SV_inc + (2*H2 - y2) * math.cos(beta2)/cp2),
                ]
                G_cur, lam_cur, c_cur = G2, lam2, cs2
                
            val_arrays = []
            param_arrays = []
            
            for w_type, w_dir, w_ang, w_amp, w_delay in waves:
                ux, uy, sxx, syy, sxy = wave_vectors(w_type, w_dir, w_ang, G_cur, lam_cur, cs1 if is_L1 else cs2)
                # Correct param c is local for stress. Actually wave_vectors internally uses cp for P waves so we pass 1.0 safely? 
                # NO! wave_vectors requires correct local wave speed.
                c_val = cp1 if w_type=='P' and is_L1 else cs1 if is_L1 else cp2 if w_type=='P' else cs2
                ux, uy, sxx, syy, sxy = wave_vectors(w_type, w_dir, w_ang, G_cur, lam_cur, c_val)
                
                param_arrays.append((w_amp, ux, uy, sxx, syy, sxy))
                val_arrays.append(get_delayed_disp(w_delay))
                val_arrays.append(get_delayed_vel(w_delay))
                
            mlen, padded = pad_min_len(val_arrays)
            
            total_ux = np.zeros(mlen); total_uy = np.zeros(mlen)
            total_dotux = np.zeros(mlen); total_dotuy = np.zeros(mlen)
            total_sxx = np.zeros(mlen); total_syy = np.zeros(mlen); total_sxy = np.zeros(mlen)
            
            for k in range(len(waves)):
                w_amp, ux, uy, sxx, syy, sxy = param_arrays[k]
                disp_arr = padded[2*k][:, 1]
                vel_arr = padded[2*k+1][:, 1]
                total_ux += w_amp * ux * disp_arr
                total_uy += w_amp * uy * disp_arr
                total_dotux += w_amp * ux * vel_arr
                total_dotuy += w_amp * uy * vel_arr
                total_sxx += w_amp * sxx * vel_arr
                total_syy += w_amp * syy * vel_arr
                total_sxy += w_amp * sxy * vel_arr

            if prefix == 'l':
                fs_x = total_sxx; fs_y = total_sxy
            elif prefix == 'r':
                fs_x = -total_sxx; fs_y = -total_sxy
            elif prefix == 'b':
                fs_x = total_sxy; fs_y = total_syy
                
            fx = kn * total_ux + cn * total_dotux + A * fs_x
            fy = kt * total_uy + ct * total_dotuy + A * fs_y
            
            fx_arr = np.zeros((mlen, 2))
            fy_arr = np.zeros((mlen, 2))
            fx_arr[:, 0] = padded[0][:, 0]
            fy_arr[:, 0] = padded[0][:, 0]
            fx_arr[:, 1] = fx
            fy_arr[:, 1] = fy
            
            field_data['{}-{}-fx'.format(node_id, prefix)] = fx_arr
            field_data['{}-{}-fy'.format(node_id, prefix)] = fy_arr

        log_step(logger, '%s 边界 %s 自由场和等效力计算完成', model_name, prefix)

    process_boundary(node_data_l, 'l')
    process_boundary(node_data_r, 'r')
    process_boundary(node_data_b, 'b')

    def apply_forces(node_data, prefix):
        instance_name = inst_name
        n = a.instances[instance_name].nodes
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            fx_arr = field_data['{}-{}-fx'.format(node_id, prefix)]
            fy_arr = field_data['{}-{}-fy'.format(node_id, prefix)]

            ampli_fx = tuple(tuple(row) for row in fx_arr)
            ampli_fy = tuple(tuple(row) for row in fy_arr)

            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)

            mdb.models[model_name].TabularAmplitude(data=ampli_fx, name=name_amp_fx, smooth=SOLVER_DEFAULT, timeSpan=STEP)
            mdb.models[model_name].TabularAmplitude(data=ampli_fy, name=name_amp_fy, smooth=SOLVER_DEFAULT, timeSpan=STEP)

            region = Region(nodes=n.sequenceFromLabels([node_id]))
            mdb.models[model_name].ConcentratedForce(
                name='load-{}-{}-fx'.format(node_id, prefix), createStepName=step_name,
                region=region, cf1=1.0, amplitude=name_amp_fx, distributionType=UNIFORM)
            mdb.models[model_name].ConcentratedForce(
                name='load-{}-{}-fy'.format(node_id, prefix), createStepName=step_name,
                region=region, cf2=1.0, amplitude=name_amp_fy, distributionType=UNIFORM)

    apply_forces(node_data_l, 'l')
    apply_forces(node_data_r, 'r')
    apply_forces(node_data_b, 'b')
    
    mdb.save()
    log_step(logger, '%s 粘弹性人工边界施加完毕: 耗时=%.2fs', model_name, time.time() - t0)

def build_models(acc_info, base_model, part_name, inst_name, angle,
                 cs1, vv1, density1, cs2, vv2, density2, H_lower,
                 step_name=DEFAULT_STEP_NAME, variables=('S', 'U', 'V'), frequency=10, logger=None):
    logger = logger or log_step()
    variables = _normalize_output_variables(variables)
    model_names = []
    for acc_file, tp, inc in acc_info:
        new_model_name = os.path.splitext(acc_file)[0]
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])
        model = mdb.models[new_model_name]
        model.ImplicitDynamicsStep(
            name=step_name, previous='Initial',
            timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=inc,
            maxNumInc=1000000, nlgeom=OFF, application=MODERATE_DISSIPATION)
        model.fieldOutputRequests['F-Output-1'].setValues(variables=variables, frequency=frequency)
        mdb.save()
        VAB_oblique(angle, cs1, vv1, density1, cs2, vv2, density2, H_lower,
                    model_name=new_model_name, part_name=part_name,
                    inst_name=inst_name, acc_file=acc_file, step_name=step_name, logger=logger)
        model_names.append(new_model_name)
    return model_names

def submit_job(num_cpus=7, memory_percent=90, model_name='Model-1', logger=None):
    logger = logger or log_step()
    t0 = time.time()
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
    mdb.Job(name=job_name, model=model_name, description='VAB oblique SV-wave double layer analysis',
            type=ANALYSIS, numCpus=num_cpus, numDomains=num_cpus,
            memory=memory_percent, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True)
    mdb.save()
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, '%s已完成: 耗时=%.2fs', job_name, time.time() - t0)

def main():
    logger = log_step('VAB_oblique_double_v1.log')
    total_start = time.time()

    material_cfg_1 = {
        'angle': 30,
        'elastic_modulus': 32e9,
        'poisson_ratio': 0.25,
        'density': 2650,
    }
    material_cfg_2 = {
        'elastic_modulus': 10e9,
        'poisson_ratio': 0.3,
        'density': 2000,
    }
    geometry_cfg = {
        'h': 100,
        'i': 45,
        'mesh_size_manual': 4,
        'f_max': 15,
        'n_per_wave': 10,
    }
    job_cfg = {
        'variables': ('U', 'V', 'A'),
        'frequency': 1,
        'num_cpus': 7,
        'memory_percent': 90,
    }

    try:
        cs1 = _compute_wave_speed_from_elastic_modulus(material_cfg_1['elastic_modulus'], material_cfg_1['poisson_ratio'], material_cfg_1['density'])
        cs2 = _compute_wave_speed_from_elastic_modulus(material_cfg_2['elastic_modulus'], material_cfg_2['poisson_ratio'], material_cfg_2['density'])

        h = geometry_cfg['h']
        H_lower = 2.0 * h
        total_L = 8.0 * h
        mesh_size_auto = min(cs1, cs2) / (geometry_cfg['f_max'] * geometry_cfg['n_per_wave'])
        mesh_size = min(mesh_size_auto, geometry_cfg['mesh_size_manual'])

        cae_name = 'double_h{}_i{}_a{}.cae'.format(h, geometry_cfg['i'], material_cfg_1['angle'])
        acc_info = find_acc_txt(logger)

        base_model, part_name, inst_name = create_model(
            total_L=total_L, h=h, i=geometry_cfg['i'],
            cs1=cs1, vv1=material_cfg_1['poisson_ratio'], density1=material_cfg_1['density'],
            cs2=cs2, vv2=material_cfg_2['poisson_ratio'], density2=material_cfg_2['density'],
            mesh_size=mesh_size, H_lower=H_lower, cae_name=cae_name, logger=logger)

        model_names = build_models(
            acc_info=acc_info, base_model=base_model, part_name=part_name, inst_name=inst_name,
            angle=material_cfg_1['angle'],
            cs1=cs1, vv1=material_cfg_1['poisson_ratio'], density1=material_cfg_1['density'],
            cs2=cs2, vv2=material_cfg_2['poisson_ratio'], density2=material_cfg_2['density'],
            H_lower=H_lower,
            step_name=DEFAULT_STEP_NAME, variables=job_cfg['variables'], frequency=job_cfg['frequency'], logger=logger)

        for model_name in model_names:
            submit_job(num_cpus=job_cfg['num_cpus'], memory_percent=job_cfg['memory_percent'], model_name=model_name, logger=logger)

        log_step(logger, '所有作业已完成，总耗时=%.2fs', time.time() - total_start)
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))
        import traceback
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise

if __name__ == '__main__':
    main()
