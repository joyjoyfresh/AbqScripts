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


def create_model(total_L, h, i, cs, vv, density, mesh_size,
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
    pn = 1
    while 'Part-%d' % pn in model.parts:
        pn += 1
    part_name = 'Part-%d' % pn
    # P1(0,0) → P2(total_L,0) → P3(total_L,H_lower) → P4(left_flat+w_slope,H_lower)
    #         → P5(left_flat,H_upper) → P6(0,H_upper) → 闭合
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(total_L, H_upper) * 2)
    s.Line(point1=(0.0, 0.0),                   point2=(total_L, 0.0))               # 底边
    s.Line(point1=(total_L, 0.0),               point2=(total_L, H_lower))           # 右边界
    s.Line(point1=(total_L, H_lower),           point2=(left_flat + w_slope, H_lower))  # 右平台地表
    s.Line(point1=(left_flat + w_slope, H_lower),  point2=(left_flat, H_upper))         # 斜坡
    s.Line(point1=(left_flat, H_upper),            point2=(0.0, H_upper))                # 左平台地表
    s.Line(point1=(0.0, H_upper),               point2=(0.0, 0.0))                   # 左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR,
                      type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del model.sketches['__profile__']
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)

    # ============ 材料与截面 ============
    GG = density * cs ** 2
    EE = 2 * GG * (1 + vv)

    mat_n = 1
    while 'Material-%d' % mat_n in model.materials:
        mat_n += 1
    mat_name = 'Material-%d' % mat_n
    mat = model.Material(name=mat_name)
    mat.Elastic(table=((EE, vv),))
    mat.Density(table=((density,),))
    log_step(logger, '%s 材料已定义: %s', model_name, mat_name)

    sec_n = 1
    while 'Section-%d' % sec_n in model.sections:
        sec_n += 1
    sec_name = 'Section-%d' % sec_n
    model.HomogeneousSolidSection(name=sec_name, material=mat_name, thickness=1.0)
    log_step(logger, '%s 截面已创建: %s', model_name, sec_name)

    faces = part.faces
    region = Region(faces=faces)
    part.SectionAssignment(region=region, sectionName=sec_name,
                           offset=0.0, offsetType=MIDDLE_SURFACE,
                           offsetField='', thicknessAssignment=FROM_SECTION)
    log_step(logger, '%s 截面已分配到所有面', model_name)

    # ============ 装配 ============
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    in_ = 1
    while '%s-%d' % (part_name, in_) in assembly.instances:
        in_ += 1
    inst_name = '%s-%d' % (part_name, in_)
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

    part.SetFromNodeLabels(nodeLabels=l_labels, name='l')
    part.SetFromNodeLabels(nodeLabels=r_labels, name='r')
    part.SetFromNodeLabels(nodeLabels=b_labels, name='b')
    log_step(logger, '%s 边界节点集已在Part中创建: 左=%d, 右=%d, 底=%d', model_name, len(l_labels), len(r_labels), len(b_labels))
    
    mdb.save() 
    return model_name, part_name, inst_name


def create_node_sets(model_name, part_name=None, target_coords_map=None, tol=1e-3, logger=None,
                     total_L=None, h=None, i=None, H_lower=None,
                     slope_obs_count=None,
                     upper_obs_count=0, upper_obs_spacing=0.0,
                     lower_obs_count=0, lower_obs_spacing=0.0):
    """
    参考 NodeSet_create_v3 的思路：按坐标在 Part 中创建节点集。
    支持两种模式：
    1) 直接传入 target_coords_map: {set_name: (x, y, z)}
    2) 不传 target_coords_map，由几何参数自动生成 M/U/D 观察点
       - M: 斜坡上从顶到底均匀分布
       - U: 从坡顶向左按间距分布
         - D: 从坡脚向右按间距分布（右平台长度由几何剩余确定）
    若容差内未找到节点，则自动回退到最近节点。
    """
    logger = logger or log_step()

    # 自动生成观察点坐标（M/U/D）
    if target_coords_map is None:
        if h is None or i is None or slope_obs_count is None:
            raise ValueError('自动创建观察点时，h/i/slope_obs_count 不能为空')
        if H_lower is None:
            H_lower = 2.0 * h
        if total_L is None:
            total_L = 8.0 * h

        if slope_obs_count < 1:
            raise ValueError('slope_obs_count 必须 >= 1')
        if upper_obs_count < 0 or lower_obs_count < 0:
            raise ValueError('upper_obs_count/lower_obs_count 不能为负数')
        if upper_obs_spacing < 0 or lower_obs_spacing < 0:
            raise ValueError('upper_obs_spacing/lower_obs_spacing 不能为负数')

        w_slope = h / math.tan(math.radians(i))
        left_flat = 3.0 * h
        right_flat = total_L - left_flat - w_slope
        if right_flat <= 0:
            raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                             (total_L, left_flat, w_slope))
        H_upper = H_lower + h

        slope_upper = (left_flat, H_upper, 0.0)
        slope_lower = (left_flat + w_slope, H_lower, 0.0)
        target_coords_map = {}
        coord_used = []

        def is_dup(pt, eps=1e-10):
            for q in coord_used:
                if abs(pt[0] - q[0]) < eps and abs(pt[1] - q[1]) < eps and abs(pt[2] - q[2]) < eps:
                    return True
            return False

        def add_obs(name, pt):
            if not is_dup(pt):
                target_coords_map[name] = pt
                coord_used.append(pt)
            else:
                log_step(logger, '%s 观察点 %s 与已有点重合，已跳过: (%.6f, %.6f, %.6f)',
                         model_name, name, pt[0], pt[1], pt[2])

        # M 点：斜坡上从上到下均匀分布（含顶点与坡脚）
        if slope_obs_count == 1:
            add_obs('M1', slope_upper)
        else:
            for k in range(slope_obs_count):
                t = float(k) / float(slope_obs_count - 1)
                x = slope_upper[0] + t * (slope_lower[0] - slope_upper[0])
                y = slope_upper[1] + t * (slope_lower[1] - slope_upper[1])
                add_obs('M{}'.format(k + 1), (x, y, 0.0))

        # U 点：从坡顶向左（不含坡顶）
        if upper_obs_count > 0:
            if upper_obs_spacing <= 0.0:
                log_step(logger, '%s upper_obs_spacing=%.6f 非法（需>0），U点已全部跳过', model_name, upper_obs_spacing)
            else:
                max_u = int(math.floor((slope_upper[0] - 0.0) / upper_obs_spacing + 1e-12))
                eff_u = min(upper_obs_count, max_u)
                if eff_u < upper_obs_count:
                    log_step(logger,
                             '%s U点请求数量=%d 超出上平台范围，已截断为 %d（spacing=%.3f, 平台长度=%.3f）',
                             model_name, upper_obs_count, eff_u, upper_obs_spacing, slope_upper[0])
                for k in range(eff_u):
                    x = slope_upper[0] - (k + 1) * upper_obs_spacing
                    y = H_upper
                    add_obs('U{}'.format(k + 1), (x, y, 0.0))

        # D 点：从坡脚向右（不含坡脚）
        if lower_obs_count > 0:
            if lower_obs_spacing <= 0.0:
                log_step(logger, '%s lower_obs_spacing=%.6f 非法（需>0），D点已全部跳过', model_name, lower_obs_spacing)
            else:
                right_len = total_L - slope_lower[0]
                max_d = int(math.floor(right_len / lower_obs_spacing + 1e-12))
                eff_d = min(lower_obs_count, max_d)
                if eff_d < lower_obs_count:
                    log_step(logger,
                             '%s D点请求数量=%d 超出下平台范围，已截断为 %d（spacing=%.3f, 平台长度=%.3f）',
                             model_name, lower_obs_count, eff_d, lower_obs_spacing, right_len)
                for k in range(eff_d):
                    x = slope_lower[0] + (k + 1) * lower_obs_spacing
                    y = H_lower
                    add_obs('D{}'.format(k + 1), (x, y, 0.0))

    model = mdb.models[model_name]
    if part_name is None:
        if len(model.parts) != 1:
            raise ValueError('未指定 part_name 且模型中 Part 数量不为 1，请显式传入 part_name')
        part_name = list(model.parts.keys())[0]
    if part_name not in model.parts:
        raise KeyError('模型 %s 中不存在 Part: %s' % (model_name, part_name))
    part = model.parts[part_name]

    def normalize_set_name(raw_name):
        """将任意输入名转换为 Abaqus 可接受的集合名。"""
        s = str(raw_name).strip()
        if not s:
            s = 'OBS_SET'
        cleaned = []
        for ch in s:
            if ch.isalnum() or ch == '_':
                cleaned.append(ch)
            else:
                cleaned.append('_')
        name = ''.join(cleaned)
        if not name:
            name = 'OBS_SET'
        if name[0].isdigit():
            name = '#' + name
        if len(name) > 80:
            name = name[:80]
        return name

    created_sets = []
    used_names = set(part.sets.keys())
    for raw_set_name, target in target_coords_map.items():
        set_name = normalize_set_name(raw_set_name)
        if set_name in used_names:
            idx = 1
            base_name = set_name
            while True:
                candidate = '{}_{}'.format(base_name, idx)
                if candidate not in used_names:
                    set_name = candidate
                    break
                idx += 1
        used_names.add(set_name)
        tx, ty, tz = target
        matched_labels = []
        best_match = None  # (node_label, dist)

        for node in part.nodes:
            x = node.coordinates[0]
            y = node.coordinates[1]
            z = node.coordinates[2] if len(node.coordinates) > 2 else 0.0
            dist = ((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2) ** 0.5

            if best_match is None or dist < best_match[1]:
                best_match = (node.label, dist)

            if dist < tol:
                matched_labels.append(node.label)

        if not matched_labels:
            if best_match is None:
                log_step(logger, '%s 观察点集合 %s 创建失败：Part中无节点可用', model_name, set_name)
                continue
            matched_labels = [best_match[0]]
            log_step(logger, '%s 观察点集合 %s 未找到容差内节点，已定位到最近节点', model_name, set_name)

        part.SetFromNodeLabels(nodeLabels=tuple(matched_labels), name=set_name)
        created_sets.append(set_name)
    
    mdb.save() 
    if created_sets:
        log_step(logger, '%s Part观察点集合已创建: %s', model_name, ', '.join(created_sets))
    return created_sets


def VAB_oblique(angle, cs, vv, density, model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None):
    """
    主函数：为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力
    参数:
        angle  (float): SV波入射角度（度），0为垂直入射
        cs     (float): 剪切波速 (m/s)
        vv     (float): 泊松比
        density(float): 密度 (kg/m³)
        acc_file  (str): 指定加速度时程文件名
        step_name (str): 分析步名称，None 时使用 'Step-earthquake'
    """
    # ============ 基本参数 ============
    logger = logger or log_step()
    t0 = time.time()
    step_name = step_name or 'Step-earthquake'
    log_step(logger, '%s 模型开始创建人工边界', model_name)

    # ============ 获取装配体 ============
    a = mdb.models[model_name].rootAssembly
    a.regenerate()

    model = mdb.models[model_name]
    if part_name not in model.parts:
        raise KeyError('%s 中不存在Part: %s' % (model_name, part_name))
    part = model.parts[part_name]
    if inst_name not in a.instances:
        raise KeyError('%s 中不存在实例: %s' % (model_name, inst_name))
    instance = a.instances[inst_name]

    # ============ 复用基础模型中的边界节点集（Part层） ============
    if 'l' not in part.sets or 'r' not in part.sets or 'b' not in part.sets:
        raise KeyError('%s 缺少Part边界节点集 l/r/b，请先在 create_model 中创建' % model_name)
    log_step(logger, '%s 复用已有Part边界节点集: l/r/b', model_name)

    def get_instance_nodes_from_part_set(set_name):
        labels = tuple(node.label for node in part.sets[set_name].nodes)
        if not labels:
            raise ValueError('%s Part节点集 %s 为空' % (model_name, set_name))
        return instance.nodes.sequenceFromLabels(labels)

    # ============ 材料参数计算 ============
    GG = density * cs ** 2                    # 剪切模量
    EE = 2 * GG * (1 + vv)                    # 弹性模量
    lam = 2 * GG * vv / (1 - 2 * vv)          # 拉梅常数 λ
    cp = math.sqrt((lam + 2 * GG) / density)  # 纵波波速

    # ============ 获取模型尺寸 ============
    l_nodes = get_instance_nodes_from_part_set('l')
    l_ymax_node = max(l_nodes, key=lambda node: node.coordinates[1])
    xmin = l_ymax_node.coordinates[0]
    ymax_l = l_ymax_node.coordinates[1]

    b_nodes = get_instance_nodes_from_part_set('b')
    ymin = b_nodes[0].coordinates[1]

    r_nodes = get_instance_nodes_from_part_set('r')
    r_ymax_node = max(r_nodes, key=lambda node: node.coordinates[1])
    xmax = r_ymax_node.coordinates[0]
    ymax_r = r_ymax_node.coordinates[1]

    ymax = max(ymax_l, ymax_r)

    # ============ 计算节点影响长度 ============
    def get_node_influence(nodes, sort_axis='y', ascending=False):
        """
        获取边界节点的影响长度（半距离），返回 [n, 4] 数组：节点号、x、y、影响长度
        """
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

    node_data_l = get_node_influence(l_nodes, sort_axis='y', ascending=False)
    node_data_r = get_node_influence(r_nodes, sort_axis='y', ascending=False)
    node_data_b = get_node_influence(b_nodes, sort_axis='x', ascending=True)
    log_step(logger, '%s 节点影响长度已计算', model_name)

    # ============ 粘弹性人工边界参数（刘晶波公式） ============
    kn = GG / 2 / ymax       # 法向弹簧刚度系数
    cn = density * cp         # 法向阻尼系数
    kt = GG / 4 / ymax       # 切向弹簧刚度系数
    ct = density * cs         # 切向阻尼系数
    log_step(logger, '%s 弹簧-阻尼系数已计算', model_name)

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
    log_step(logger, '%s 弹簧-阻尼系数已分配到所有边界节点', model_name)

    # ============ 在Abaqus中添加弹簧-阻尼器到地面 ============
    assembly = model.rootAssembly

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
    log_step(logger, '%s 弹簧-阻尼器创建完成', model_name)

    # ============ 入射角与反射系数计算 ============
    if angle == 0:
        angle = 1e-10  # 避免除零
    else:
        angle = round(angle, 4)

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
    log_step(logger, '%s 反射参数已计算', model_name)

    # ============ 读取加速度时程并积分得到速度/位移 ============
    if not acc_file:
        raise ValueError('acc_file 不能为空，请传入加速度时程文件')
    ACC = np.loadtxt(acc_file)
    if ACC.ndim != 2 or ACC.shape[1] < 2 or ACC.shape[0] < 2:
        raise ValueError('加速度文件至少需要 2 行 2 列: [time, acceleration]')
    time_arr = ACC[:, 0]
    acc = ACC[:, 1]
    dt = ACC[1, 0] - ACC[0, 0]
    if dt <= 0:
        raise ValueError('加速度文件中时间步长无效: dt 必须 > 0')
    log_step(logger, '%s 已读取加速度时程 %s', model_name, acc_file)

    vel = np.zeros_like(acc)
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)  # a -> v 梯形积分
    VEL = np.column_stack((time_arr, vel))
    log_step(logger, '%s 加速度已积分为速度', model_name)

    dis = np.zeros_like(vel)
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 梯形积分
    DIS = np.column_stack((time_arr, dis))
    log_step(logger, '%s 速度已积分为位移', model_name)

    max_time = ACC[-1, 0]
    Ly = ymax - ymin  # 模型高度
    Lx = xmax - xmin  # 模型宽度

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
    log_step(logger, '%s 左/右/底 边界节点延迟时间已计算', model_name)

    # 如果最大延迟超过输入时程长度，则补零延长
    detmax = max(np.max(det_l[:, 1:]), np.max(det_r[:, 1:]), np.max(det_b[:, 1:]))
    if max_time < detmax:
        n_add = int(np.ceil((detmax - max_time) / dt))
        new_times = VEL[-1, 0] + dt * np.arange(1, n_add + 1)
        new_vel = np.zeros((n_add, 2))
        new_vel[:, 0] = new_times
        VEL = np.vstack([VEL, new_vel])
        DIS = np.vstack([DIS, new_vel])
        log_step(logger, '%s VEL/DIS 已用零延长: 增加行数=%d, 新总时长=%.3f',
                 model_name, n_add, VEL[-1, 0])
    else:
        log_step(logger, '%s VEL/DIS 无需延长', model_name)

    # ============ 延迟时间对齐到时间步 ============
    def round_delay(det, dt):
        det[:, 1:4] = np.round(det[:, 1:4] / dt) * dt
        return det

    det_l = round_delay(det_l, dt)
    det_r = round_delay(det_r, dt)
    det_b = round_delay(det_b, dt)
    log_step(logger, '%s 延迟时间已对齐到 dt 网格', model_name)

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
    log_step(logger, '%s 左/右/底 自由场位移已计算', model_name)
    # 计算速度自由场
    calc_freefield_u_and_dotu_general(node_data_l, det_l, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'l')
    calc_freefield_u_and_dotu_general(node_data_r, det_r, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'r')
    calc_freefield_u_and_dotu_general(node_data_b, det_b, VEL, dt, alpha, beta_p, A1, A2, 'dotux', 'dotuy', 'b')
    log_step(logger, '%s 左/右/底 自由场速度已计算', model_name)

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
    log_step(logger, '%s 左/右/底 自由场应力已计算', model_name)

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
    log_step(logger, '%s 等效节点力时程已计算', model_name)

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
    log_step(logger, '%s 所有边界节点的幅值曲线已创建', model_name)

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
            region = Region(nodes=node_array)
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
    log_step(logger, '%s 所有边界节点已施加集中力', model_name)
    mdb.save()
    log_step(logger, '%s 粘弹性人工边界完成: 耗时=%.2fs', model_name, time.time() - t0)


def build_models(acc_info, base_model, part_name, inst_name, angle, cs, vv, density, 
                 step_name='Step-earthquake',variables=('S', 'U', 'V'), frequency=10, logger=None):
    """
    根据加速度时程信息批量复制模型、创建分析步、施加人工边界。

    参数:
        acc_info    (list): find_acc_txt 返回的列表 [(acc_file, tp, inc), ...]
        base_model  (str): 基础模型名称
        part_name   (str): 零件名称
        inst_name   (str): 实例名称
        angle/cs/vv/density: 人工边界参数
        step_name   (str): 分析步名称
    返回:
        model_names (list): 新创建的模型名称列表
    """
    logger = logger or log_step()

    # Abaqus 要求 variables 为字符串序列；若传入单个字符串则自动转换为单元素元组
    if isinstance(variables, str):
        variables = (variables,)
    elif isinstance(variables, list):
        variables = tuple(variables)

    model_names = []
    for acc_file, tp, inc in acc_info:
        new_model_name = os.path.splitext(acc_file)[0]
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])
        log_step(logger, '%s 模型已从 %s 复制', new_model_name, base_model)

        # 创建分析步
        model = mdb.models[new_model_name]
        model.ImplicitDynamicsStep(
            name=step_name, previous='Initial',
            timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=inc,
            maxNumInc=1000000,
            nlgeom=OFF, application=MODERATE_DISSIPATION)
        model.fieldOutputRequests['F-Output-1'].setValues(
            variables=variables, frequency=frequency)
        mdb.save()
        log_step(logger, '%s 分析步已创建, 时长=%.2f, 增量=%.3f',
                 new_model_name, tp, inc)

        # 施加粘弹性人工边界和等效节点力
        VAB_oblique(angle, cs, vv, density,
                    model_name=new_model_name, part_name=part_name,
                    inst_name=inst_name,
                    acc_file=acc_file, step_name=step_name,
                    logger=logger)
        model_names.append(new_model_name)

    return model_names


def submit_job(num_cpus=7, memory_percent=90, model_name='Model-1', logger=None):
    """
    创建并提交Abaqus作业

    参数:
        model_name     (str):  模型名称（默认同时作为作业名称）
        num_cpus       (int):  CPU数量
        memory_percent (int):  内存百分比
    """
    logger = logger or log_step()
    t0 = time.time()
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
        log_step(logger, '检测到同名旧作业，已删除: %s', job_name)
    log_step(logger, '%s作业开始提交, CPU 数量=%d, 内存=%d%%',
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

    mdb.save()
    log_step(logger, '%s作业已提交，正在等待完成...', job_name)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, '%s已完成: 耗时=%.2fs', job_name, time.time() - t0)


if __name__ == '__main__':
    logger = log_step('VAB_oblique_noGUI_v8.log') # *日志文件名
    total_start = time.time()
    try:
        log_step(logger, '脚本开始执行')
        # ===== 土体参数设置 =====
        angle = 10                  # *SV波入射角度（度）
        E = 32e9                    # *杨氏模量 (Pa)
        vv = 0.25                   # *泊松比
        density = 2650              # *密度 (kg/m³)
        cs = math.sqrt((E / (2 * (1 + vv))) / density)  # 由杨氏模量自动计算剪切波速 (m/s)

        # ====== 模型参数设置 =======
        h = 100                     # *斜坡高度 (m)
        i = 45                      # *斜坡倾角 (°)
        mesh_size_manual = 2        # *手动设置网格尺寸 (m)
        f_max = 15                  # *目标最高频率 (Hz)，用于自动计算网格尺寸
        n_per_wave = 10             # *每波长最少单元数（建议 8~10）
        H_lower = 2.0 * h   # 下垫面高度 = 2h
        total_L = 8.0 * h   # 总长固定为 8h（左平台固定 3h，右平台由剩余长度自动确定）
        mesh_size_auto = cs / (f_max * n_per_wave)   # 自动计算网格尺寸 = Vs / (f_max * n)
        mesh_size = min(mesh_size_auto, mesh_size_manual)  # 取自动与手动中的较小值

        # ====== 观察点参数设置 ======
        slope_obs_count = 3         # *斜坡观察点数量（M1..，从上到下均匀分布，含顶点与坡脚）
        upper_obs_count = 0         # *上层平台观察点数量（U1..，从坡顶向左）
        upper_obs_spacing = 50.0    # *上层平台观察点间距 (m)
        lower_obs_count = 0         # *下层平台观察点数量（D1..，从坡脚向右）
        lower_obs_spacing = 50.0    # *下层平台观察点间距 (m)

        # ====== 作业参数设置 ======
        cae_name = 'h'+str(h)+'_i'+str(i)+'_a'+str(angle)+'.cae' # *CAE文件名（可修改）
        variables = ('U', 'V', 'A',)                             # *输出变量
        frequency = 10              # *输出频率 (Hz)
        num_cpus = 7                # *CPU数量
        memory_percent = 90         # *内存百分比

        # 1. 查找加速度时程文件
        acc_info = find_acc_txt(logger)
        
        # 2. 创建基础模型（几何、材料、网格、装配，不含分析步）
        base_model, part_name, inst_name = create_model(
            total_L, h, i, cs, vv, density, mesh_size,
            H_lower=H_lower, cae_name=cae_name,
            logger=logger)

        # 3. 在基础模型上创建观察点集合（M/U/D）
        create_node_sets(
            model_name=base_model, part_name=part_name, target_coords_map=None, tol=1e-3, logger=logger,
            total_L=total_L, h=h, i=i, H_lower=H_lower,
            slope_obs_count=slope_obs_count,
            upper_obs_count=upper_obs_count, upper_obs_spacing=upper_obs_spacing,
            lower_obs_count=lower_obs_count, lower_obs_spacing=lower_obs_spacing)

        # 4. 为每个txt文件复制模型、创建分析步、施加人工边界
        model_names = build_models(
            acc_info=acc_info, base_model=base_model, part_name=part_name, inst_name=inst_name, angle=angle, cs=cs, 
            vv=vv, density=density, step_name='Step-earthquake',variables=variables, frequency=frequency, logger=logger)

        # 5. 依次提交作业
        for mn in model_names:
            submit_job(num_cpus=num_cpus, memory_percent=memory_percent, model_name=mn, logger=logger)
        log_step(logger, '所有作业已完成，总耗时=%.2fs', time.time() - total_start)

        
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())