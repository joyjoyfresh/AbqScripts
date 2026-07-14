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


DEFAULT_STEP_NAME = 'Step-earthquake'  # 定义默认分析步名称
BOUNDARY_SET_NAMES = ('Left_boundary', 'Right_boundary', 'Bottom_boundary')  # 定义基础边界节点集名称
BOUNDARY_SEQUENCE = ('l', 'r', 'b')  # 定义边界处理顺序


def main():
    """脚本主入口：组织参数、建模、施加边界并提交作业。"""
    logger = log_step('VAB_oblique_TAF_v1.log')  # 初始化日志并写入当前版本日志文件
    total_start = time.time()  # 记录主流程起始时间

    # 统一集中配置参数，便于维护和批量改动。
    material_cfg = {
        'angle': 30,  # 设置 SV 波入射角度（度）
        'elastic_modulus': 32e9,  # 设置杨氏模量（Pa）
        'poisson_ratio': 0.25,  # 设置泊松比
        'density': 2650,  # 设置密度（kg/m^3）
    }
    geometry_cfg = {
        'h': 100,  # 设置斜坡高度（m）
        'i': 45,  # 设置斜坡倾角（度）
        'mesh_size_manual': 4,  # 设置手动网格尺寸上限（m）
        'f_max': 15,  # 设置目标最高频率（Hz）
        'n_per_wave': 10,  # 设置每波长单元数
    }
    job_cfg = {
        'variables': ('U', 'V', 'A'),  # 设置场输出变量
        'frequency': 1,  # 设置输出频率
        'num_cpus': 7,  # 设置并行 CPU 数量
        'memory_percent': 90,  # 设置作业内存百分比
    }

    try:
        log_step(logger, '脚本开始执行')  # 写入脚本启动日志

        cs = _compute_wave_speed_from_elastic_modulus(  # 根据材料参数计算剪切波速
            material_cfg['elastic_modulus'],
            material_cfg['poisson_ratio'],
            material_cfg['density'])

        h = geometry_cfg['h']  # 读取斜坡高度
        H_lower = 2.0 * h  # 设置斜坡模型下垫面高度
        H_flat = 2.5 * h  # 设置平坦自由场模型总高度
        total_L = 8.0 * h  # 设置总模型长度
        mesh_size_auto = cs / (geometry_cfg['f_max'] * geometry_cfg['n_per_wave'])  # 按波长准则计算自动网格尺寸
        mesh_size = min(mesh_size_auto, geometry_cfg['mesh_size_manual'])  # 取自动尺寸与手动上限中的较小值

        cae_name = 'h{}_i{}_a{}.cae'.format(h, geometry_cfg['i'], material_cfg['angle'])  # 生成 CAE 文件名

        acc_info = find_acc_txt(logger)  # 读取当前目录内全部加速度时程信息

        base_model, part_name, inst_name = create_model(  # 创建基础几何与网格模型
            total_L=total_L,
            h=h,
            i=geometry_cfg['i'],
            cs=cs,
            vv=material_cfg['poisson_ratio'],
            density=material_cfg['density'],
            mesh_size=mesh_size,
            H_lower=H_lower,
            cae_name=cae_name,
            logger=logger)

        flat_base_model, flat_part_name, flat_inst_name = create_flat_model(  # 创建平坦自由场基础模型
            total_L=total_L,
            H_flat=H_flat,
            cs=cs,
            vv=material_cfg['poisson_ratio'],
            density=material_cfg['density'],
            mesh_size=mesh_size,
            logger=logger)

        slope_model_names = build_models(  # 依据不同地震动复制斜坡模型并施加等效边界
            acc_info=acc_info,
            base_model=base_model,
            part_name=part_name,
            inst_name=inst_name,
            angle=material_cfg['angle'],
            cs=cs,
            vv=material_cfg['poisson_ratio'],
            density=material_cfg['density'],
            step_name=DEFAULT_STEP_NAME,
            variables=job_cfg['variables'],
            frequency=job_cfg['frequency'],
            model_scene='slope',
            logger=logger)

        flat_model_names = build_models(  # 依据不同地震动复制平坦自由场模型并施加等效边界
            acc_info=acc_info,
            base_model=flat_base_model,
            part_name=flat_part_name,
            inst_name=flat_inst_name,
            angle=material_cfg['angle'],
            cs=cs,
            vv=material_cfg['poisson_ratio'],
            density=material_cfg['density'],
            step_name=DEFAULT_STEP_NAME,
            variables=job_cfg['variables'],
            frequency=job_cfg['frequency'],
            model_scene='flat',
            logger=logger)

        model_names = slope_model_names + flat_model_names  # 合并两类模型名称用于统一提交作业

        for model_name in model_names:  # 顺序提交每个模型作业
            submit_job(
                num_cpus=job_cfg['num_cpus'],
                memory_percent=job_cfg['memory_percent'],
                model_name=model_name,
                logger=logger)

        log_step(logger, '所有作业已完成，总耗时=%.2fs', time.time() - total_start)  # 输出总耗时日志
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))  # 记录异常摘要
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 记录完整堆栈便于定位问题
        raise  # 继续抛出异常，避免失败被静默吞掉


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


def _build_model_name_from_record(acc_file, scene_tag):
    """按“记录名-场景名”规则生成模型名，如 El_Centro-slope。"""
    record_name = os.path.splitext(os.path.basename(acc_file))[0]  # 从加速度文件名中提取不含扩展名的记录名
    if not record_name:  # 校验记录名不能为空
        raise ValueError('无法从加速度文件生成记录名: %s' % acc_file)  # 记录名为空时抛出异常
    if scene_tag not in ('slope', 'flat'):  # 校验场景标签是否受支持
        raise ValueError('scene_tag 仅支持 slope 或 flat，当前为: %s' % scene_tag)  # 场景标签非法时抛出异常
    return '{}-{}'.format(record_name, scene_tag)  # 返回“记录名-场景名”格式模型名


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

    # ============ 材料与截面 ============
    GG = density * cs ** 2
    EE = 2 * GG * (1 + vv)

    mat_name = _next_available_name('Material', model.materials)
    mat = model.Material(name=mat_name)
    mat.Elastic(table=((EE, vv),))
    mat.Density(table=((density,),))
    log_step(logger, '%s 材料已定义: %s', model_name, mat_name)

    sec_name = _next_available_name('Section', model.sections)
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
    inst_name = _next_available_name(part_name, assembly.instances)
    assembly.Instance(name=inst_name, part=part, dependent=ON)
    log_step(logger, '%s 装配实例已创建: %s', model_name, inst_name)

    # ============ 网格划分 ============
    # 网格前按坡底点水平切分面，分为上下两部分
    # 分割线：从坡底点 (left_flat + w_slope, H_lower) 水平连到左边界 (0, H_lower)
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(
        name='__partition__', sheetSize=max(total_L, H_upper) * 2
    )
    partition_sketch.Line(point1=(left_flat + w_slope, H_lower), point2=(0.0, H_lower))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__partition__']
    log_step(logger, '%s 网格前切分完成: 上下两部分', model_name)

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


def create_flat_model(total_L, H_flat, cs, vv, density, mesh_size, logger=None):
    """创建二维平面应变平坦自由场模型：矩形几何、材料、截面、装配与网格。"""
    logger = logger or log_step()  # 复用已有日志器或初始化默认日志器
    model_name = 'Model-2'  # 指定平坦自由场基础模型名称

    if total_L <= 0:  # 校验模型长度参数
        raise ValueError('total_L 必须 > 0')  # 长度非法时抛出异常
    if H_flat <= 0:  # 校验模型高度参数
        raise ValueError('H_flat 必须 > 0')  # 高度非法时抛出异常

    model = mdb.Model(name=model_name)  # 创建平坦自由场基础模型
    log_step(logger, '%s 基础模型开始创建（平坦自由场）', model_name)  # 记录平坦模型创建开始日志

    part_name = _next_available_name('Part', model.parts)  # 生成平坦模型零件名称
    sketch = model.ConstrainedSketch(name='__flat_profile__', sheetSize=max(total_L, H_flat) * 2)  # 创建平坦模型草图
    sketch.Line(point1=(0.0, 0.0), point2=(total_L, 0.0))  # 绘制矩形底边
    sketch.Line(point1=(total_L, 0.0), point2=(total_L, H_flat))  # 绘制矩形右边界
    sketch.Line(point1=(total_L, H_flat), point2=(0.0, H_flat))  # 绘制矩形顶边
    sketch.Line(point1=(0.0, H_flat), point2=(0.0, 0.0))  # 绘制矩形左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)  # 创建二维可变形体零件
    part.BaseShell(sketch=sketch)  # 由草图生成壳体面
    del model.sketches['__flat_profile__']  # 删除临时草图对象
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)  # 记录零件创建完成日志

    GG = density * cs ** 2  # 按剪切波速计算剪切模量
    EE = 2 * GG * (1 + vv)  # 按线弹性关系计算杨氏模量

    mat_name = _next_available_name('Material', model.materials)  # 生成材料名称
    mat = model.Material(name=mat_name)  # 创建材料对象
    mat.Elastic(table=((EE, vv),))  # 定义弹性参数
    mat.Density(table=((density,),))  # 定义密度参数
    log_step(logger, '%s 材料已定义: %s', model_name, mat_name)  # 记录材料定义日志

    sec_name = _next_available_name('Section', model.sections)  # 生成截面名称
    model.HomogeneousSolidSection(name=sec_name, material=mat_name, thickness=1.0)  # 创建均质实体截面
    log_step(logger, '%s 截面已创建: %s', model_name, sec_name)  # 记录截面创建日志

    faces = part.faces  # 获取零件全部面
    region = Region(faces=faces)  # 将全部面打包为区域对象
    part.SectionAssignment(  # 向全部面分配截面
        region=region,
        sectionName=sec_name,
        offset=0.0,
        offsetType=MIDDLE_SURFACE,
        offsetField='',
        thicknessAssignment=FROM_SECTION)
    log_step(logger, '%s 截面已分配到所有面', model_name)  # 记录截面分配日志

    assembly = model.rootAssembly  # 获取根装配体对象
    assembly.DatumCsysByDefault(CARTESIAN)  # 设置默认笛卡尔坐标系
    inst_name = _next_available_name(part_name, assembly.instances)  # 生成实例名称
    assembly.Instance(name=inst_name, part=part, dependent=ON)  # 创建装配实例
    log_step(logger, '%s 装配实例已创建: %s', model_name, inst_name)  # 记录实例创建日志

    picked_regions = part.faces  # 选取全部面用于网格控制
    part.setMeshControls(regions=picked_regions, elemShape=QUAD, technique=STRUCTURED)  # 设置结构化四边形网格
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)  # 设定全局播种尺寸
    elem_type = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)  # 指定平面应变四节点单元
    part.setElementType(regions=(picked_regions,), elemTypes=(elem_type,))  # 将单元类型分配给全部面
    part.generateMesh()  # 执行网格生成
    assembly.regenerate()  # 重新生成装配同步网格信息
    log_step(logger, '%s 网格已生成: 尺寸=%.3f, 单元=CPE4', model_name, mesh_size)  # 记录网格生成日志

    x_list = [node.coordinates[0] for node in part.nodes]  # 提取所有节点x坐标
    y_list = [node.coordinates[1] for node in part.nodes]  # 提取所有节点y坐标
    xmin = min(x_list)  # 计算最小x坐标
    xmax = max(x_list)  # 计算最大x坐标
    ymin = min(y_list)  # 计算最小y坐标
    ymax = max(y_list)  # 计算最大y坐标
    tol = 1e-6  # 设置边界识别容差

    l_nodes_list = [node for node in part.nodes if abs(node.coordinates[0] - xmin) < tol]  # 筛选左边界节点
    r_nodes_list = [node for node in part.nodes if abs(node.coordinates[0] - xmax) < tol]  # 筛选右边界节点
    b_nodes_list = [node for node in part.nodes if abs(node.coordinates[1] - ymin) < tol]  # 筛选底边界节点
    t_nodes_list = [node for node in part.nodes if abs(node.coordinates[1] - ymax) < tol]  # 筛选顶边界节点

    l_labels = tuple(node.label for node in l_nodes_list)  # 生成左边界标签元组
    r_labels = tuple(node.label for node in r_nodes_list)  # 生成右边界标签元组
    b_labels = tuple(node.label for node in b_nodes_list)  # 生成底边界标签元组
    t_labels = tuple(node.label for node in t_nodes_list)  # 生成顶边界标签元组

    part.SetFromNodeLabels(nodeLabels=l_labels, name='Left_boundary')  # 创建左边界节点集
    part.SetFromNodeLabels(nodeLabels=r_labels, name='Right_boundary')  # 创建右边界节点集
    part.SetFromNodeLabels(nodeLabels=b_labels, name='Bottom_boundary')  # 创建底边界节点集
    part.SetFromNodeLabels(nodeLabels=t_labels, name='TOP_SURFACE')  # 创建顶面节点集
    log_step(  # 写入平坦模型边界节点集数量日志
        logger,
        '%s 边界节点集已在Part中创建: 左=%d, 右=%d, 底=%d, 顶=%d',
        model_name,
        len(l_labels),
        len(r_labels),
        len(b_labels),
        len(t_labels))

    mdb.save()  # 保存当前CAE数据库
    return model_name, part_name, inst_name  # 返回平坦基础模型及其零件/实例名称


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
    step_name = step_name or DEFAULT_STEP_NAME
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
    missing_boundary_sets = [name for name in BOUNDARY_SET_NAMES if name not in part.sets]
    if missing_boundary_sets:
        raise KeyError('%s 缺少Part边界节点集: %s，请先在 create_model 中创建' %
                       (model_name, '/'.join(missing_boundary_sets)))
    log_step(logger, '%s 复用已有Part边界节点集: %s', model_name, '/'.join(BOUNDARY_SET_NAMES))

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
    l_nodes = get_instance_nodes_from_part_set('Left_boundary')
    l_ymax_node = max(l_nodes, key=lambda node: node.coordinates[1])
    xmin = l_ymax_node.coordinates[0]
    ymax_l = l_ymax_node.coordinates[1]

    b_nodes = get_instance_nodes_from_part_set('Bottom_boundary')
    ymin = b_nodes[0].coordinates[1]

    r_nodes = get_instance_nodes_from_part_set('Right_boundary')
    r_ymax_node = max(r_nodes, key=lambda node: node.coordinates[1])
    xmax = r_ymax_node.coordinates[0]
    ymax_r = r_ymax_node.coordinates[1]

    ymax = max(ymax_l, ymax_r)

    # ============ 计算节点影响长度 ============
    def get_node_influence(nodes, sort_axis='y', ascending=False):
        """
        获取边界节点的影响长度（半距离），返回 [n, 4] 数组：节点号、x、y、影响长度
        """
        node_data = np.array([[node.label, node.coordinates[0], node.coordinates[1]] for node in nodes], dtype=float)  # 一次性构造节点数组
        axis = 1 if sort_axis == 'x' else 2  # 指定排序轴索引
        node_data = node_data[node_data[:, axis].argsort()]  # 按目标轴升序排序
        if not ascending:  # 若需要降序则翻转
            node_data = node_data[::-1]  # 执行降序翻转

        n = node_data.shape[0]  # 获取节点数量
        if n == 1:  # 若仅有一个节点
            influence = np.array([0.0])  # 影响长度记为0
        else:  # 若节点数量大于1
            coord = node_data[:, axis]  # 提取排序轴坐标
            influence = np.empty(n)  # 预分配影响长度数组
            influence[0] = abs(coord[0] - coord[1]) / 2.0  # 首节点取与次节点半距
            influence[-1] = abs(coord[-1] - coord[-2]) / 2.0  # 末节点取与前节点半距
            if n > 2:  # 若存在中间节点
                influence[1:-1] = np.abs(coord[:-2] - coord[2:]) / 2.0  # 中间节点用前后跨一节点半距

        node_data = np.hstack((node_data, influence.reshape(-1, 1)))  # 拼接影响长度列
        return node_data  # 返回节点数据与影响长度

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

    # 统一定义边界自由度规则：左/右边界法向x切向y，底边界法向y切向x
    boundary_dof = {
        'l': (1, 2),
        'r': (1, 2),
        'b': (2, 1),
    }
    boundary_node_data = {
        'l': node_data_l,
        'r': node_data_r,
        'b': node_data_b,
    }
    for boundary in BOUNDARY_SEQUENCE:
        dof_n, dof_t = boundary_dof[boundary]
        add_spring_dashpot(boundary_node_data[boundary], prefix=boundary, dof_n=dof_n, dof_t=dof_t)
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

    def build_det_map(det):
        """将 det 数组转换为 {node_id: (tA, tB, tC)} 字典，避免循环内重复检索"""
        return {int(row[0]): (row[1], row[2], row[3]) for row in det}  # 建立节点到三种到时的映射

    det_map_l = build_det_map(det_l)  # 左边界延迟映射
    det_map_r = build_det_map(det_r)  # 右边界延迟映射
    det_map_b = build_det_map(det_b)  # 底边界延迟映射

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

    def make_delay_cache(timeseries, dt):
        """按离散步数缓存延迟信号，避免相同延迟重复构造数组"""
        cache = {}  # 定义延迟缓存字典

        def get_delayed(delay_t):
            n_delay = int(np.round(delay_t / dt))  # 计算离散延迟步数
            if n_delay not in cache:  # 若缓存中不存在该延迟
                cache[n_delay] = delay_signal(timeseries, n_delay * dt, dt)  # 构造并缓存延迟信号
            return cache[n_delay]  # 返回缓存的延迟信号

        return get_delayed  # 返回闭包函数供后续调用

    def pad_to(arr, length, dt):
        """将数组补零到指定长度"""
        if arr.shape[0] < length:
            pad = np.zeros((length - arr.shape[0], 2))
            pad[:, 0] = np.arange(arr.shape[0], length) * dt
            arr = np.vstack([arr, pad])
        return arr

    # ============ 计算自由场位移和速度 ============
    field_data = {}  # 用字典存储中间结果

    def calc_freefield_u_and_dotu_general(node_data, det_map, timeseries, dt,
                                           alpha, beta_p, A1, A2,
                                           suffix1, suffix2, prefix):
        """
        对各边界（左、右、底）计算自由场 ux/uy 或 dotux/dotuy 时程
        """
        get_delayed = make_delay_cache(timeseries, dt)  # 为当前时程创建延迟缓存访问器
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            tA, tB, tC = det_map[node_id]  # 通过字典O(1)获取三种到时

            u0_tA = get_delayed(tA)  # 获取A波延迟信号
            u0_tB = get_delayed(tB)  # 获取B波延迟信号
            u0_tC = get_delayed(tC)  # 获取C波延迟信号

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

    boundary_det_map = {
        'l': det_map_l,
        'r': det_map_r,
        'b': det_map_b,
    }

    # 计算位移自由场
    for boundary in BOUNDARY_SEQUENCE:
        calc_freefield_u_and_dotu_general(
            boundary_node_data[boundary], boundary_det_map[boundary], DIS, dt,
            alpha, beta_p, A1, A2, 'ux', 'uy', boundary)
    log_step(logger, '%s 左/右/底 自由场位移已计算', model_name)
    # 计算速度自由场
    for boundary in BOUNDARY_SEQUENCE:
        calc_freefield_u_and_dotu_general(
            boundary_node_data[boundary], boundary_det_map[boundary], VEL, dt,
            alpha, beta_p, A1, A2, 'dotux', 'dotuy', boundary)
    log_step(logger, '%s 左/右/底 自由场速度已计算', model_name)

    # ============ 计算自由场应力 ============
    def calc_freefield_sigma_general(node_data, det_map, VEL, dt,
                                      alpha, beta_p, A1, A2,
                                      GG, cs, lam, cp, prefix):
        """
        对各边界（左、右、底）计算自由场应力 sigmax/sigmay 时程
        """
        get_delayed = make_delay_cache(VEL, dt)  # 为速度时程创建延迟缓存访问器
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            tA, tB, tC = det_map[node_id]  # 通过字典O(1)获取三种到时

            v0_tA = get_delayed(tA)  # 获取A波延迟速度
            v0_tB = get_delayed(tB)  # 获取B波延迟速度
            v0_tC = get_delayed(tC)  # 获取C波延迟速度

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

    for boundary in BOUNDARY_SEQUENCE:
        calc_freefield_sigma_general(
            boundary_node_data[boundary], boundary_det_map[boundary], VEL, dt,
            alpha, beta_p, A1, A2, GG, cs, lam, cp, boundary)
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

    for boundary in BOUNDARY_SEQUENCE:
        calc_equiv_node_force_general(boundary_node_data[boundary], boundary)
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

    for boundary in BOUNDARY_SEQUENCE:
        batch_add_node_force_amplitude(boundary_node_data[boundary], boundary)
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

    for boundary in BOUNDARY_SEQUENCE:
        batch_add_node_force(boundary_node_data[boundary], boundary, step_name)
    log_step(logger, '%s 所有边界节点已施加集中力', model_name)
    mdb.save()
    log_step(logger, '%s 粘弹性人工边界完成: 耗时=%.2fs', model_name, time.time() - t0)


def build_models(acc_info, base_model, part_name, inst_name, angle, cs, vv, density,
                 step_name=DEFAULT_STEP_NAME, variables=('S', 'U', 'V'), frequency=10,
                 model_scene='slope', logger=None):
    """
    根据加速度时程信息批量复制模型、创建分析步、施加人工边界。

    参数:
        acc_info    (list): find_acc_txt 返回的列表 [(acc_file, tp, inc), ...]
        base_model  (str): 基础模型名称
        part_name   (str): 零件名称
        inst_name   (str): 实例名称
        angle/cs/vv/density: 人工边界参数
        step_name   (str): 分析步名称
        model_scene (str): 模型场景标识，支持 slope 或 flat
    返回:
        model_names (list): 新创建的模型名称列表
    """
    logger = logger or log_step()

    # Abaqus 要求 variables 为字符串序列；若传入单个字符串则自动转换为单元素元组
    variables = _normalize_output_variables(variables)

    model_names = []
    for acc_file, tp, inc in acc_info:
        new_model_name = _build_model_name_from_record(acc_file, model_scene)  # 按“记录名-场景名”自动生成模型名
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])
        log_step(logger, '%s 模型已从 %s 复制', new_model_name, base_model)

        # 创建分析步
        model = mdb.models[new_model_name]
        model.ImplicitDynamicsStep(
            name=step_name, previous='Initial',
            timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=inc,
            maxNumInc=1000000,
            nlgeom=OFF, application=MODERATE_DISSIPATION)

        # 全局场输出
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
    main()