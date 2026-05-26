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
    logger = log_step('VAB_oblique_TAF_double.log')  # 初始化日志并写入当前版本日志文件
    total_start = time.time()  # 记录主流程起始时间

    # 统一配置参数（根据 Shen 等 - 2025 论文设置）
    material_cfg = {
        'angle': 15,  # SV 波入射角度（度）
        'bedrock': {
            'elastic_modulus': 26e9,  # 基岩杨氏模量（Pa），对应 Vs = 2000 m/s
            'poisson_ratio': 0.3,  # 基岩泊松比
            'density': 2500,  # 基岩密度（kg/m^3）
        },
        'overlying': {
            'poisson_ratio': 0.3,
            'density': 2500,
            'velocity_ratio': 2.5,  # VR / Vs2 阻抗比，对应 Vs2 = 2000 / 2.5 = 800 m/s
        },
        'surface': {
            'poisson_ratio': 0.3,
            'density': 2500,
            'velocity_ratio': 0.5,  # Vs1 / Vs2 阻抗比，对应 Vs1 = 800 * 0.5 = 400 m/s
            'relative_thickness': 0.25,  # h1 / (H - h) 相对厚度
        },
        'max_reflect_order': 3,  # 设置多次反射/透射最大阶数
    }

    geometry_cfg = {
        'H_minus_h': 200.0,  # 斜坡高度 H - h (m)
        'i': 45.0,  # 斜坡倾角 (度)
        'h_over_H': 0.5,  # 深度比 h / H
        'total_L': 1800.0,  # 总模型长度 (m)
        'left_flat': 1000.0,  # 上平台长度 (m)
        'bedrock_thickness': 200.0,  # 基岩层厚度 (m)
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

        # 波动参数计算
        cs_bedrock = _compute_wave_speed_from_elastic_modulus(
            material_cfg['bedrock']['elastic_modulus'],
            material_cfg['bedrock']['poisson_ratio'],
            material_cfg['bedrock']['density']
        )
        cs_overlying = cs_bedrock / material_cfg['overlying']['velocity_ratio']
        cs_surface = cs_overlying * material_cfg['surface']['velocity_ratio']

        # 几何高度计算
        H_minus_h = geometry_cfg['H_minus_h']
        h_over_H = geometry_cfg['h_over_H']
        H = H_minus_h / (1.0 - h_over_H)
        h = H - H_minus_h
        bedrock_thickness = geometry_cfg['bedrock_thickness']
        H_lower = bedrock_thickness + h
        H_flat = bedrock_thickness + H
        H_upper = bedrock_thickness + H
        w_slope = H_minus_h / math.tan(math.radians(geometry_cfg['i']))
        total_L = geometry_cfg['total_L']
        left_flat = geometry_cfg['left_flat']
        h1 = material_cfg['surface']['relative_thickness'] * H_minus_h

        # 网格尺寸设为高度 (H - h) 的 4%
        mesh_size = 0.04 * H_minus_h

        cae_name = 'h{}_i{}_a{}.cae'.format(int(H_minus_h), int(geometry_cfg['i']), int(material_cfg['angle']))

        acc_info = find_acc_txt(logger)  # 读取当前目录内全部加速度时程信息

        base_model, part_name, inst_name = create_model(  # 创建基础几何与网格模型
            total_L=total_L,
            H_minus_h=H_minus_h,
            i=geometry_cfg['i'],
            h_over_H=h_over_H,
            bedrock_thickness=bedrock_thickness,
            cs_bedrock=cs_bedrock,
            vv_bedrock=material_cfg['bedrock']['poisson_ratio'],
            density_bedrock=material_cfg['bedrock']['density'],
            cs_overlying=cs_overlying,
            vv_overlying=material_cfg['overlying']['poisson_ratio'],
            density_overlying=material_cfg['overlying']['density'],
            cs_surface=cs_surface,
            vv_surface=material_cfg['surface']['poisson_ratio'],
            density_surface=material_cfg['surface']['density'],
            h1=h1,
            mesh_size=mesh_size,
            cae_name=cae_name,
            logger=logger
        )

        flat_base_model, flat_part_name, flat_inst_name = create_flat_model(  # 创建平坦自由场基础模型
            total_L=total_L,
            H_flat=H_flat,
            bedrock_thickness=bedrock_thickness,
            cs_bedrock=cs_bedrock,
            vv_bedrock=material_cfg['bedrock']['poisson_ratio'],
            density_bedrock=material_cfg['bedrock']['density'],
            cs_overlying=cs_overlying,
            vv_overlying=material_cfg['overlying']['poisson_ratio'],
            density_overlying=material_cfg['overlying']['density'],
            cs_surface=cs_surface,
            vv_surface=material_cfg['surface']['poisson_ratio'],
            density_surface=material_cfg['surface']['density'],
            h1=h1,
            mesh_size=mesh_size,
            logger=logger
        )

        slope_model_names = build_models(  # 依据不同地震动复制斜坡模型并施加等效边界
            acc_info=acc_info,
            base_model=base_model,
            part_name=part_name,
            inst_name=inst_name,
            angle=material_cfg['angle'],
            cs_bedrock=cs_bedrock,
            vv_bedrock=material_cfg['bedrock']['poisson_ratio'],
            density_bedrock=material_cfg['bedrock']['density'],
            cs_overlying=cs_overlying,
            vv_overlying=material_cfg['overlying']['poisson_ratio'],
            density_overlying=material_cfg['overlying']['density'],
            cs_surface=cs_surface,
            vv_surface=material_cfg['surface']['poisson_ratio'],
            density_surface=material_cfg['surface']['density'],
            bedrock_thickness=bedrock_thickness,
            h1=h1,
            H_upper=H_upper,
            H_lower=H_lower,
            left_flat=left_flat,
            w_slope=w_slope,
            max_reflect_order=material_cfg['max_reflect_order'],
            step_name=DEFAULT_STEP_NAME,
            variables=job_cfg['variables'],
            frequency=job_cfg['frequency'],
            model_scene='slope',
            logger=logger
        )

        flat_model_names = build_models(  # 依据不同地震动复制平坦自由场模型并施加等效边界
            acc_info=acc_info,
            base_model=flat_base_model,
            part_name=flat_part_name,
            inst_name=flat_inst_name,
            angle=material_cfg['angle'],
            cs_bedrock=cs_bedrock,
            vv_bedrock=material_cfg['bedrock']['poisson_ratio'],
            density_bedrock=material_cfg['bedrock']['density'],
            cs_overlying=cs_overlying,
            vv_overlying=material_cfg['overlying']['poisson_ratio'],
            density_overlying=material_cfg['overlying']['density'],
            cs_surface=cs_surface,
            vv_surface=material_cfg['surface']['poisson_ratio'],
            density_surface=material_cfg['surface']['density'],
            bedrock_thickness=bedrock_thickness,
            h1=h1,
            H_upper=H_upper,
            H_lower=H_upper,  # 平坦自由场上部和下部都是统一总高度 H_upper
            left_flat=left_flat,
            w_slope=0.001,  # 平坦自由场相当于倾斜宽度极小
            max_reflect_order=material_cfg['max_reflect_order'],
            step_name=DEFAULT_STEP_NAME,
            variables=job_cfg['variables'],
            frequency=job_cfg['frequency'],
            model_scene='flat',
            logger=logger
        )

        model_names = slope_model_names + flat_model_names  # 合并两类模型名称用于统一提交作业

        for model_name in model_names:  # 顺序提交每个模型作业
            submit_job(
                num_cpus=job_cfg['num_cpus'],
                memory_percent=job_cfg['memory_percent'],
                model_name=model_name,
                logger=logger
            )

        log_step(logger, '所有作业已完成，总耗时=%.2fs', time.time() - total_start)  # 输出总耗时日志
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))  # 记录异常摘要
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 记录完整堆栈
        raise


def _next_available_name(prefix, existing_container):
    """按前缀生成可用名称（如 Part-1, Part-2）。"""
    index = 1
    while '%s-%d' % (prefix, index) in existing_container:
        index += 1
    return '%s-%d' % (prefix, index)


def _normalize_output_variables(variables):
    """规范化输出变量为元组，满足 Abaqus 接口要求。"""
    if isinstance(variables, str):
        return (variables,)
    if isinstance(variables, list):
        return tuple(variables)
    return variables


def _compute_wave_speed_from_elastic_modulus(elastic_modulus, poisson_ratio, density):
    """根据杨氏模量、泊松比和密度计算剪切波速。"""
    return math.sqrt((elastic_modulus / (2 * (1 + poisson_ratio))) / density)


def _compute_elastic_modulus_from_wave_speed(cs, vv, density):
    """根据剪切波速、泊松比和密度计算杨氏模量 E。"""
    GG = density * (cs ** 2)
    EE = 2 * GG * (1 + vv)
    return EE


def _safe_arcsin(value):
    """对 arcsin 输入做截断，避免浮点超界。"""
    return math.asin(max(-1.0, min(1.0, value)))


def _compute_material_params(cs, vv, density):
    """根据 Vs、泊松比、密度计算材料参数。"""
    GG = density * cs ** 2
    EE = 2 * GG * (1 + vv)
    lam = 2 * GG * vv / (1 - 2 * vv)
    cp = math.sqrt((lam + 2 * GG) / density)
    return {'GG': GG, 'EE': EE, 'lam': lam, 'cp': cp, 'cs': cs, 'vv': vv, 'density': density}


def _compute_interface_sv_coeff(alpha1, mat1, mat2):
    """计算 SV 波在两层界面的等效反射/透射系数（阻抗近似）。"""
    z1s = mat1['density'] * mat1['cs'] * max(1e-8, math.cos(alpha1))
    sin_a2 = mat2['cs'] * math.sin(alpha1) / mat1['cs']
    alpha2 = _safe_arcsin(sin_a2)
    z2s = mat2['density'] * mat2['cs'] * max(1e-8, math.cos(alpha2))
    denom = z1s + z2s if abs(z1s + z2s) > 1e-12 else 1e-12
    rss = (z2s - z1s) / denom
    tss = 2.0 * z2s / denom
    rsp = 0.0
    tsp = 0.0
    return {'Rss': rss, 'Rsp': rsp, 'Tss': tss, 'Tsp': tsp, 'alpha2': alpha2}


def _compute_free_surface_sv_coeff(alpha, cp, cs):
    """计算 SV 波在自由面的反射系数。"""
    beta_p = _safe_arcsin(cp * math.sin(alpha) / cs)
    numerator_a1 = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) - cp ** 2 * math.cos(2 * alpha) ** 2
    denominator = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) + cp ** 2 * math.cos(2 * alpha) ** 2
    if abs(denominator) < 1e-12:
        denominator = 1e-12
    a1 = numerator_a1 / denominator
    a2 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator
    return {'A1': a1, 'A2': a2, 'beta': beta_p}


def _compute_free_surface_p_coeff(beta, cp, cs):
    """计算 P 波在自由面的反射系数。"""
    alpha = _safe_arcsin(cs * math.sin(beta) / cp)
    numerator_b2 = cp ** 2 * math.cos(2 * alpha) ** 2 - cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)
    denominator = cp ** 2 * math.cos(2 * alpha) ** 2 + cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)
    if abs(denominator) < 1e-12:
        denominator = 1e-12
    b2 = numerator_b2 / denominator
    b1 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator
    return {'B1': b1, 'B2': b2, 'alpha': alpha}


def _build_model_name_from_record(acc_file, scene_tag):
    """按“记录名-场景名”规则生成模型名。"""
    record_name = os.path.splitext(os.path.basename(acc_file))[0]
    if not record_name:
        raise ValueError('无法从加速度文件生成记录名: %s' % acc_file)
    if scene_tag not in ('slope', 'flat'):
        raise ValueError('scene_tag 仅支持 slope 或 flat，当前为: %s' % scene_tag)
    return '{}-{}'.format(record_name, scene_tag)


def log_step(logger=None, message=None, *args):
    """日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。"""
    if not hasattr(log_step, '_logger'):
        if logger is not None and isinstance(logger, str):
            log_filename = logger
            logger = None
        else:
            log_filename = 'logfile.log'

        _logger = logging.getLogger('abqpy')
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
    """查找当前工作目录下所有 .txt 文件，并读取每个加速度文件的分析步时长和增量步。"""
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


def create_model(total_L, H_minus_h, i, h_over_H, bedrock_thickness,
                 cs_bedrock, vv_bedrock, density_bedrock,
                 cs_overlying, vv_overlying, density_overlying,
                 cs_surface, vv_surface, density_surface,
                 h1, mesh_size, cae_name=None, logger=None):
    """创建二维平面应变模型：几何、材料、截面、装配、网格（不含分析步）"""
    logger = logger or log_step()
    model_name = 'Model-1'

    # 几何计算
    H = H_minus_h / (1.0 - h_over_H)
    h = H - H_minus_h
    H_lower = bedrock_thickness + h
    H_upper = bedrock_thickness + H
    w_slope = H_minus_h / math.tan(math.radians(i))
    left_flat = 1000.0  # 上平台长度固定为 1000m

    right_flat = total_L - left_flat - w_slope
    if right_flat <= 0:
        raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                         (total_L, left_flat, w_slope))

    if cae_name:
        mdb.saveAs(pathName=cae_name)
        log_step(logger, '工程文件保存为 %s', cae_name)
    model = mdb.Model(name=model_name)
    log_step(logger, '%s 基础模型开始创建', model_name)

    # 创建二维坡地 Part
    part_name = _next_available_name('Part', model.parts)
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(total_L, H_upper) * 2)
    s.Line(point1=(0.0, 0.0),                   point2=(total_L, 0.0))                 # 底边
    s.Line(point1=(total_L, 0.0),               point2=(total_L, H_lower))             # 右边界
    s.Line(point1=(total_L, H_lower),           point2=(left_flat + w_slope, H_lower)) # 右平台地表
    s.Line(point1=(left_flat + w_slope, H_lower),  point2=(left_flat, H_upper))        # 斜坡
    s.Line(point1=(left_flat, H_upper),            point2=(0.0, H_upper))              # 左平台地表
    s.Line(point1=(0.0, H_upper),               point2=(0.0, 0.0))                     # 左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del model.sketches['__profile__']
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)

    # 材料与截面
    # 计算弹性模量
    EE_bedrock = _compute_elastic_modulus_from_wave_speed(cs_bedrock, vv_bedrock, density_bedrock)
    EE_overlying = _compute_elastic_modulus_from_wave_speed(cs_overlying, vv_overlying, density_overlying)
    EE_surface = _compute_elastic_modulus_from_wave_speed(cs_surface, vv_surface, density_surface)

    mat_bedrock_name = _next_available_name('Material-Bedrock', model.materials)
    mat_bedrock = model.Material(name=mat_bedrock_name)
    mat_bedrock.Elastic(table=((EE_bedrock, vv_bedrock),))
    mat_bedrock.Density(table=((density_bedrock,),))

    mat_overlying_name = _next_available_name('Material-Overlying', model.materials)
    mat_overlying = model.Material(name=mat_overlying_name)
    mat_overlying.Elastic(table=((EE_overlying, vv_overlying),))
    mat_overlying.Density(table=((density_overlying,),))

    mat_surface_name = _next_available_name('Material-Surface', model.materials)
    mat_surface = model.Material(name=mat_surface_name)
    mat_surface.Elastic(table=((EE_surface, vv_surface),))
    mat_surface.Density(table=((density_surface,),))

    sec_bedrock_name = _next_available_name('Section-Bedrock', model.sections)
    model.HomogeneousSolidSection(name=sec_bedrock_name, material=mat_bedrock_name, thickness=1.0)

    sec_overlying_name = _next_available_name('Section-Overlying', model.sections)
    model.HomogeneousSolidSection(name=sec_overlying_name, material=mat_overlying_name, thickness=1.0)

    sec_surface_name = _next_available_name('Section-Surface', model.sections)
    model.HomogeneousSolidSection(name=sec_surface_name, material=mat_surface_name, thickness=1.0)

    # 装配
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    inst_name = _next_available_name(part_name, assembly.instances)
    assembly.Instance(name=inst_name, part=part, dependent=ON)

    # ============ 切分面以划分网格与材料区域 ============
    # 1. 垂直切分（ crest & toe ）
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(name='__vert_partition__', sheetSize=max(total_L, H_upper) * 2)
    partition_sketch.Line(point1=(left_flat, 0.0), point2=(left_flat, H_upper))
    partition_sketch.Line(point1=(left_flat + w_slope, 0.0), point2=(left_flat + w_slope, H_lower))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__vert_partition__']
    log_step(logger, '%s 几何垂直切分完成', model_name)

    # 2. 水平切分基岩界面 (y = bedrock_thickness)
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(name='__bedrock_partition__', sheetSize=max(total_L, H_upper) * 2)
    partition_sketch.Line(point1=(0.0, bedrock_thickness), point2=(total_L, bedrock_thickness))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__bedrock_partition__']
    log_step(logger, '%s 基岩水平面切分完成', model_name)

    # 3. 切分表层界面 (y = y_surface - h1)
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(name='__surface_partition__', sheetSize=max(total_L, H_upper) * 2)
    partition_sketch.Line(point1=(0.0, H_upper - h1), point2=(left_flat, H_upper - h1))
    partition_sketch.Line(point1=(left_flat, H_upper - h1), point2=(left_flat + w_slope, H_lower - h1))
    partition_sketch.Line(point1=(left_flat + w_slope, H_lower - h1), point2=(total_L, H_lower - h1))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__surface_partition__']
    log_step(logger, '%s 坡地表层线切分完成', model_name)

    # 设置网格控制：四边形 + 结构化
    pickedRegions = part.faces
    part.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1,))
    part.generateMesh()
    log_step(logger, '%s 已生成网格: CPE4 单元，尺寸=%.2f', model_name, mesh_size)

    # ============ 按质心坐标分配截面 ============
    def _get_surf_y(x):
        if x <= left_flat:
            return H_upper
        elif x >= left_flat + w_slope:
            return H_lower
        else:
            return H_upper - (x - left_flat) * H_minus_h / w_slope

    sec_assignments = {
        'bedrock': [],
        'overlying': [],
        'surface': []
    }

    for face in part.faces:
        centroid = face.getCentroid()
        xc = centroid[0] if len(centroid) >= 2 else centroid[0][0]
        yc = centroid[1] if len(centroid) >= 2 else centroid[0][1]

        if yc < bedrock_thickness:
            sec_assignments['bedrock'].append(face)
        else:
            y_surf = _get_surf_y(xc)
            if y_surf - yc < h1:
                sec_assignments['surface'].append(face)
            else:
                sec_assignments['overlying'].append(face)

    def _to_face_sequence(face_list):
        face_seq = part.faces[0:0]
        for face in face_list:
            face_seq = face_seq + part.faces[face.index:face.index + 1]
        return face_seq

    if sec_assignments['bedrock']:
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['bedrock'])),
                               sectionName=sec_bedrock_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    if sec_assignments['overlying']:
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['overlying'])),
                               sectionName=sec_overlying_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    if sec_assignments['surface']:
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['surface'])),
                               sectionName=sec_surface_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    log_step(logger, '%s 截面属性分配完成: Bedrock=%d, Overlying=%d, Surface=%d',
             model_name, len(sec_assignments['bedrock']), len(sec_assignments['overlying']), len(sec_assignments['surface']))

    # 重新生成装配体以同步网格
    assembly.regenerate()

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
    log_step(logger, '%s 边界节点集已创建: 左=%d, 右=%d, 底=%d', model_name, len(l_labels), len(r_labels), len(b_labels))

    # ============ 创建顶面节点集 ============
    top_tol = max(1e-6, mesh_size * 1e-3)
    top_surface_labels = []

    for node in part.nodes:
        x = node.coordinates[0]
        y = node.coordinates[1]
        is_on_top = False

        if (0.0 - top_tol) <= x <= (left_flat + top_tol):
            if abs(y - H_upper) <= top_tol:
                is_on_top = True
        elif (left_flat - top_tol) <= x <= (left_flat + w_slope + top_tol):
            y_slope = H_upper - (x - left_flat) * H_minus_h / w_slope
            if abs(y - y_slope) <= top_tol:
                is_on_top = True
        elif (left_flat + w_slope - top_tol) <= x <= (total_L + top_tol):
            if abs(y - H_lower) <= top_tol:
                is_on_top = True

        if is_on_top:
            top_surface_labels.append(node.label)

    top_surface_labels = tuple(sorted(set(top_surface_labels)))
    if len(top_surface_labels) == 0:
        raise ValueError('%s 未识别到顶部边界节点，请检查几何参数与容差设置' % model_name)

    part.SetFromNodeLabels(nodeLabels=top_surface_labels, name='TOP_SURFACE')
    log_step(logger, '%s 顶面节点集已创建: TOP_SURFACE=%d', model_name, len(top_surface_labels))

    mdb.save()
    return model_name, part_name, inst_name


def create_flat_model(total_L, H_flat, bedrock_thickness,
                      cs_bedrock, vv_bedrock, density_bedrock,
                      cs_overlying, vv_overlying, density_overlying,
                      cs_surface, vv_surface, density_surface,
                      h1, mesh_size, logger=None):
    """创建二维平面应变平坦自由场模型：矩形几何、材料、截面、装配与网格。"""
    logger = logger or log_step()
    model_name = 'Model-2'

    model = mdb.Model(name=model_name)
    log_step(logger, '%s 基础模型开始创建（平坦自由场）', model_name)

    part_name = _next_available_name('Part', model.parts)
    sketch = model.ConstrainedSketch(name='__flat_profile__', sheetSize=max(total_L, H_flat) * 2)
    sketch.Line(point1=(0.0, 0.0), point2=(total_L, 0.0))
    sketch.Line(point1=(total_L, 0.0), point2=(total_L, H_flat))
    sketch.Line(point1=(total_L, H_flat), point2=(0.0, H_flat))
    sketch.Line(point1=(0.0, H_flat), point2=(0.0, 0.0))
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=sketch)
    del model.sketches['__flat_profile__']
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)

    # 材料与截面
    EE_bedrock = _compute_elastic_modulus_from_wave_speed(cs_bedrock, vv_bedrock, density_bedrock)
    EE_overlying = _compute_elastic_modulus_from_wave_speed(cs_overlying, vv_overlying, density_overlying)
    EE_surface = _compute_elastic_modulus_from_wave_speed(cs_surface, vv_surface, density_surface)

    mat_bedrock_name = _next_available_name('Material-Bedrock', model.materials)
    mat_bedrock = model.Material(name=mat_bedrock_name)
    mat_bedrock.Elastic(table=((EE_bedrock, vv_bedrock),))
    mat_bedrock.Density(table=((density_bedrock,),))

    mat_overlying_name = _next_available_name('Material-Overlying', model.materials)
    mat_overlying = model.Material(name=mat_overlying_name)
    mat_overlying.Elastic(table=((EE_overlying, vv_overlying),))
    mat_overlying.Density(table=((density_overlying,),))

    mat_surface_name = _next_available_name('Material-Surface', model.materials)
    mat_surface = model.Material(name=mat_surface_name)
    mat_surface.Elastic(table=((EE_surface, vv_surface),))
    mat_surface.Density(table=((density_surface,),))

    sec_bedrock_name = _next_available_name('Section-Bedrock', model.sections)
    model.HomogeneousSolidSection(name=sec_bedrock_name, material=mat_bedrock_name, thickness=1.0)

    sec_overlying_name = _next_available_name('Section-Overlying', model.sections)
    model.HomogeneousSolidSection(name=sec_overlying_name, material=mat_overlying_name, thickness=1.0)

    sec_surface_name = _next_available_name('Section-Surface', model.sections)
    model.HomogeneousSolidSection(name=sec_surface_name, material=mat_surface_name, thickness=1.0)

    # 装配
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    inst_name = _next_available_name(part_name, assembly.instances)
    assembly.Instance(name=inst_name, part=part, dependent=ON)

    # ============ 水平切分面 ============
    # 1. 基岩水平切分 (y = bedrock_thickness)
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(name='__flat_bedrock_partition__', sheetSize=max(total_L, H_flat) * 2)
    partition_sketch.Line(point1=(total_L, bedrock_thickness), point2=(0.0, bedrock_thickness))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__flat_bedrock_partition__']

    # 2. 表层水平切分 (y = H_flat - h1)
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(name='__flat_surface_partition__', sheetSize=max(total_L, H_flat) * 2)
    partition_sketch.Line(point1=(total_L, H_flat - h1), point2=(0.0, H_flat - h1))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__flat_surface_partition__']
    log_step(logger, '%s 平坦自由场网格前切割完成', model_name)

    picked_regions = part.faces
    part.setMeshControls(regions=picked_regions, elemShape=QUAD, technique=STRUCTURED)
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
    elem_type = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    part.setElementType(regions=(picked_regions,), elemTypes=(elem_type,))
    part.generateMesh()
    log_step(logger, '%s 平坦模型网格已生成: 尺寸=%.2f', model_name, mesh_size)

    # ============ 截面分配 ============
    sec_assignments = {
        'bedrock': [],
        'overlying': [],
        'surface': []
    }

    for face in part.faces:
        centroid = face.getCentroid()
        yc = centroid[1] if len(centroid) >= 2 else centroid[0][1]

        if yc < bedrock_thickness:
            sec_assignments['bedrock'].append(face)
        elif H_flat - yc < h1:
            sec_assignments['surface'].append(face)
        else:
            sec_assignments['overlying'].append(face)

    def _to_face_sequence(face_list):
        face_seq = part.faces[0:0]
        for face in face_list:
            face_seq = face_seq + part.faces[face.index:face.index + 1]
        return face_seq

    if sec_assignments['bedrock']:
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['bedrock'])),
                               sectionName=sec_bedrock_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    if sec_assignments['overlying']:
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['overlying'])),
                               sectionName=sec_overlying_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    if sec_assignments['surface']:
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['surface'])),
                               sectionName=sec_surface_name, offset=0.0, offsetType=MIDDLE_SURFACE,
                               offsetField='', thicknessAssignment=FROM_SECTION)
    log_step(logger, '%s 截面属性分配完成（平坦自由场）', model_name)

    assembly.regenerate()

    x_list = [node.coordinates[0] for node in part.nodes]
    y_list = [node.coordinates[1] for node in part.nodes]
    xmin = min(x_list)
    xmax = max(x_list)
    ymin = min(y_list)
    ymax = max(y_list)
    tol = 1e-6

    l_nodes_list = [node for node in part.nodes if abs(node.coordinates[0] - xmin) < tol]
    r_nodes_list = [node for node in part.nodes if abs(node.coordinates[0] - xmax) < tol]
    b_nodes_list = [node for node in part.nodes if abs(node.coordinates[1] - ymin) < tol]
    t_nodes_list = [node for node in part.nodes if abs(node.coordinates[1] - ymax) < tol]

    l_labels = tuple(node.label for node in l_nodes_list)
    r_labels = tuple(node.label for node in r_nodes_list)
    b_labels = tuple(node.label for node in b_nodes_list)
    t_labels = tuple(node.label for node in t_nodes_list)

    part.SetFromNodeLabels(nodeLabels=l_labels, name='Left_boundary')
    part.SetFromNodeLabels(nodeLabels=r_labels, name='Right_boundary')
    part.SetFromNodeLabels(nodeLabels=b_labels, name='Bottom_boundary')
    part.SetFromNodeLabels(nodeLabels=t_labels, name='TOP_SURFACE')

    mdb.save()
    return model_name, part_name, inst_name


def VAB_oblique(angle,
                cs_bedrock, vv_bedrock, density_bedrock,
                cs_overlying, vv_overlying, density_overlying,
                cs_surface, vv_surface, density_surface,
                bedrock_thickness, h1,
                H_upper, H_lower, left_flat, w_slope,
                max_reflect_order=1,
                model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None):
    """为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力。"""
    logger = logger or log_step()
    t0 = time.time()
    step_name = step_name or DEFAULT_STEP_NAME
    log_step(logger, '%s 模型开始创建人工边界', model_name)

    a = mdb.models[model_name].rootAssembly
    a.regenerate()

    model = mdb.models[model_name]
    if part_name not in model.parts:
        raise KeyError('%s 中不存在Part: %s' % (model_name, part_name))
    part = model.parts[part_name]
    if inst_name not in a.instances:
        raise KeyError('%s 中不存在实例: %s' % (model_name, inst_name))
    instance = a.instances[inst_name]

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

    # 材料参数计算
    mat_bedrock = _compute_material_params(cs_bedrock, vv_bedrock, density_bedrock)
    mat_overlying = _compute_material_params(cs_overlying, vv_overlying, density_overlying)
    mat_surface = _compute_material_params(cs_surface, vv_surface, density_surface)

    def _pick_material_by_node(x_coord, y_coord):
        if y_coord < bedrock_thickness + 1e-4:
            return mat_bedrock
        # 计算当前 x 坐标对应的地表高度
        if x_coord <= left_flat:
            y_surf = H_upper
        elif x_coord >= left_flat + w_slope:
            y_surf = H_lower
        else:
            y_surf = H_upper - (x_coord - left_flat) * (H_upper - H_lower) / w_slope
        
        if y_surf - y_coord < h1 + 1e-4:
            return mat_surface
        else:
            return mat_overlying

    # 获取模型尺寸
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

    # 计算节点影响长度
    def get_node_influence(nodes, sort_axis='y', ascending=False):
        node_data = np.array([[node.label, node.coordinates[0], node.coordinates[1]] for node in nodes], dtype=float)
        axis = 1 if sort_axis == 'x' else 2
        node_data = node_data[node_data[:, axis].argsort()]
        if not ascending:
            node_data = node_data[::-1]

        n = node_data.shape[0]
        if n == 1:
            influence = np.array([0.0])
        else:
            coord = node_data[:, axis]
            influence = np.empty(n)
            influence[0] = abs(coord[0] - coord[1]) / 2.0
            influence[-1] = abs(coord[-1] - coord[-2]) / 2.0
            if n > 2:
                influence[1:-1] = np.abs(coord[:-2] - coord[2:]) / 2.0

        node_data = np.hstack((node_data, influence.reshape(-1, 1)))
        return node_data

    node_data_l = get_node_influence(l_nodes, sort_axis='y', ascending=False)
    node_data_r = get_node_influence(r_nodes, sort_axis='y', ascending=False)
    node_data_b = get_node_influence(b_nodes, sort_axis='x', ascending=True)
    log_step(logger, '%s 节点影响长度已计算', model_name)

    # 粘弹性人工边界参数 (根据节点所在材质层动态赋值)
    def add_spring_damper(node_data):
        influence = node_data[:, 3]
        kns = np.zeros_like(influence)
        cns = np.zeros_like(influence)
        kts = np.zeros_like(influence)
        cts = np.zeros_like(influence)
        for idx in range(node_data.shape[0]):
            x0 = node_data[idx, 1]
            y0 = node_data[idx, 2]
            mat = _pick_material_by_node(x0, y0)
            kn_coeff = mat['GG'] / 2.0 / ymax
            cn_coeff = mat['density'] * mat['cp']
            kt_coeff = mat['GG'] / 4.0 / ymax
            ct_coeff = mat['density'] * mat['cs']
            kns[idx] = kn_coeff * influence[idx]
            cns[idx] = cn_coeff * influence[idx]
            kts[idx] = kt_coeff * influence[idx]
            cts[idx] = ct_coeff * influence[idx]
        return np.hstack((node_data,
                           kns.reshape(-1, 1),
                           cns.reshape(-1, 1),
                           kts.reshape(-1, 1),
                           cts.reshape(-1, 1)))

    node_data_l = add_spring_damper(node_data_l)
    node_data_r = add_spring_damper(node_data_r)
    node_data_b = add_spring_damper(node_data_b)
    log_step(logger, '%s 弹簧-阻尼系数已分配到所有边界节点', model_name)

    # 添加弹簧阻尼器到地面
    def add_spring_dashpot(node_data, prefix, dof_n, dof_t):
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
        angle = 1e-10
    else:
        angle = round(angle, 4)

    alpha1 = math.radians(angle)
    interface_coeff_12 = _compute_interface_sv_coeff(alpha1, mat_bedrock, mat_overlying)
    alpha2 = interface_coeff_12['alpha2']
    beta1 = _safe_arcsin(mat_bedrock['cp'] * math.sin(alpha1) / mat_bedrock['cs'])
    beta2 = _safe_arcsin(mat_overlying['cp'] * math.sin(alpha2) / mat_overlying['cs']) if abs(math.sin(alpha2)) > 0 else 1e-10

    free_sv_2 = _compute_free_surface_sv_coeff(alpha2, mat_overlying['cp'], mat_overlying['cs'])
    free_p_2 = _compute_free_surface_p_coeff(beta2, mat_overlying['cp'], mat_overlying['cs'])
    interface_coeff_21 = _compute_interface_sv_coeff(alpha2, mat_overlying, mat_bedrock)

    rss_primary = interface_coeff_12['Rss']
    rsp_primary = interface_coeff_12['Rsp']
    tss_primary = interface_coeff_12['Tss']
    tsp_primary = interface_coeff_12['Tsp']

    h2 = max(0.0, ymax - bedrock_thickness)
    cycle_sv = free_sv_2['A1'] * interface_coeff_21['Rss']
    cycle_p = free_p_2['B2'] * interface_coeff_21['Rss']
    order_count = max(0, int(max_reflect_order))

    sum_cycle_sv = 0.0
    sum_cycle_p = 0.0
    for order_idx in range(order_count + 1):
        sum_cycle_sv += cycle_sv ** order_idx
        sum_cycle_p += cycle_p ** order_idx

    Rss_eff = rss_primary + tss_primary * free_sv_2['A1'] * interface_coeff_21['Tss'] * sum_cycle_sv
    Rsp_eff = rsp_primary + tss_primary * free_sv_2['A2'] * interface_coeff_21['Tss'] * sum_cycle_p
    cycle_delay_sv = (2.0 * h2 * math.cos(alpha2) / mat_overlying['cs']) if h2 > 0 else 0.0
    cycle_delay_p = (2.0 * h2 * math.cos(beta2) / mat_overlying['cp']) if h2 > 0 else 0.0
    log_step(logger, '%s 反射参数计算完成: Rss_eff=%.4f, Rsp_eff=%.4f', model_name, Rss_eff, Rsp_eff)

    # 映射回基准变量名 (底层使用 bedrock)
    alpha = alpha1
    beta_p = beta1
    A1 = Rss_eff
    A2 = Rsp_eff
    GG = mat_bedrock['GG']
    lam = mat_bedrock['lam']
    cs = mat_bedrock['cs']
    cp = mat_bedrock['cp']

    # 读取加速度时程并积分
    if not acc_file:
        raise ValueError('acc_file 不能为空')
    ACC = np.loadtxt(acc_file)
    if ACC.ndim != 2 or ACC.shape[1] < 2 or ACC.shape[0] < 2:
        raise ValueError('加速度文件格式不满足 [time, acceleration]')
    time_arr = ACC[:, 0]
    acc = ACC[:, 1]
    dt = ACC[1, 0] - ACC[0, 0]
    if dt <= 0:
        raise ValueError('加速度 dt 必须 > 0')

    vel = np.zeros_like(acc)
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)
    VEL = np.column_stack((time_arr, vel))

    dis = np.zeros_like(vel)
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)
    DIS = np.column_stack((time_arr, dis))

    max_time = ACC[-1, 0]
    Ly = bedrock_thickness - ymin
    Lx = xmax - xmin

    # 计算各节点的波到达延迟时间
    def calc_node_delay(node_data, boundary, alpha, beta_p, cs, cp, Ly, Lx):
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

    # 补零延长
    detmax = max(np.max(det_l[:, 1:]), np.max(det_r[:, 1:]), np.max(det_b[:, 1:]))
    if max_time < detmax:
        n_add = int(np.ceil((detmax - max_time) / dt))
        new_times = VEL[-1, 0] + dt * np.arange(1, n_add + 1)
        new_vel = np.zeros((n_add, 2))
        new_vel[:, 0] = new_times
        VEL = np.vstack([VEL, new_vel])
        DIS = np.vstack([DIS, new_vel])
        log_step(logger, '%s VEL/DIS 已用零延长: 增加行数=%d', model_name, n_add)
    else:
        log_step(logger, '%s VEL/DIS 无需延长', model_name)

    # 对齐延迟时间
    def round_delay(det, dt):
        det[:, 1:4] = np.round(det[:, 1:4] / dt) * dt
        return det

    det_l = round_delay(det_l, dt)
    det_r = round_delay(det_r, dt)
    det_b = round_delay(det_b, dt)

    det_map_l = {int(row[0]): (row[1], row[2], row[3]) for row in det_l}
    det_map_r = {int(row[0]): (row[1], row[2], row[3]) for row in det_r}
    det_map_b = {int(row[0]): (row[1], row[2], row[3]) for row in det_b}

    def delay_signal(u0, delay_t, dt):
        n_delay = int(np.round(delay_t / dt))
        N = u0.shape[0]
        new_len = N + n_delay
        delayed = np.zeros((new_len, 2))
        delayed[:, 0] = np.arange(new_len) * dt
        delayed[n_delay:, 1] = u0[:, 1]
        return delayed

    def make_delay_cache(timeseries, dt):
        cache = {}
        def get_delayed(delay_t):
            n_delay = int(np.round(delay_t / dt))
            if n_delay not in cache:
                cache[n_delay] = delay_signal(timeseries, n_delay * dt, dt)
            return cache[n_delay]
        return get_delayed

    def pad_to(arr, length, dt):
        if arr.shape[0] < length:
            pad = np.zeros((length - arr.shape[0], 2))
            pad[:, 0] = np.arange(arr.shape[0], length) * dt
            arr = np.vstack([arr, pad])
        return arr

    # ============ 计算自由场位移和速度 ============
    field_data = {}

    def calc_freefield_u_and_dotu_general(node_data, det_map, timeseries, dt,
                                           alpha, beta_p, A1, A2,
                                           suffix1, suffix2, prefix):
        get_delayed = make_delay_cache(timeseries, dt)
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            tA, tB, tC = det_map[node_id]

            u0_tA = get_delayed(tA)
            u0_tB_list = []
            u0_tC_list = []
            for order_idx in range(order_count + 1):
                delay_b = tB + order_idx * cycle_delay_sv
                delay_c = tC + order_idx * cycle_delay_p
                u0_tB_list.append(get_delayed(delay_b))
                u0_tC_list.append(get_delayed(delay_c))

            max_len = u0_tA.shape[0]
            for arr in u0_tB_list + u0_tC_list:
                max_len = max(max_len, arr.shape[0])

            u0_tA = pad_to(u0_tA, max_len, dt)
            u0_tB_sum = np.zeros(max_len)
            u0_tC_sum = np.zeros(max_len)
            for order_idx in range(order_count + 1):
                arr_b = pad_to(u0_tB_list[order_idx], max_len, dt)
                arr_c = pad_to(u0_tC_list[order_idx], max_len, dt)
                u0_tB_sum += (cycle_sv ** order_idx) * arr_b[:, 1]
                u0_tC_sum += (cycle_p ** order_idx) * arr_c[:, 1]

            ux = (u0_tA[:, 1] * np.cos(alpha)
                  - A1 * u0_tB_sum * np.cos(alpha)
                  + A2 * u0_tC_sum * np.sin(beta_p))
            uy = (-u0_tA[:, 1] * np.sin(alpha)
                  - A1 * u0_tB_sum * np.sin(alpha)
                  - A2 * u0_tC_sum * np.cos(beta_p))

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
    log_step(logger, '%s 自由场位移已计算', model_name)

    # 计算速度自由场
    for boundary in BOUNDARY_SEQUENCE:
        calc_freefield_u_and_dotu_general(
            boundary_node_data[boundary], boundary_det_map[boundary], VEL, dt,
            alpha, beta_p, A1, A2, 'dotux', 'dotuy', boundary)
    log_step(logger, '%s 自由场速度已计算', model_name)

    # ============ 计算自由场应力 ============
    def calc_freefield_sigma_general(node_data, det_map, VEL, dt,
                                      alpha, beta_p, A1, A2,
                                      GG, cs, lam, cp, prefix):
        get_delayed = make_delay_cache(VEL, dt)
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            tA, tB, tC = det_map[node_id]

            v0_tA = get_delayed(tA)
            v0_tB_list = []
            v0_tC_list = []
            for order_idx in range(order_count + 1):
                delay_b = tB + order_idx * cycle_delay_sv
                delay_c = tC + order_idx * cycle_delay_p
                v0_tB_list.append(get_delayed(delay_b))
                v0_tC_list.append(get_delayed(delay_c))

            max_len = v0_tA.shape[0]
            for arr in v0_tB_list + v0_tC_list:
                max_len = max(max_len, arr.shape[0])

            v0_tA = pad_to(v0_tA, max_len, dt)
            v0_tB_sum = np.zeros(max_len)
            v0_tC_sum = np.zeros(max_len)
            for order_idx in range(order_count + 1):
                arr_b = pad_to(v0_tB_list[order_idx], max_len, dt)
                arr_c = pad_to(v0_tC_list[order_idx], max_len, dt)
                v0_tB_sum += (cycle_sv ** order_idx) * arr_b[:, 1]
                v0_tC_sum += (cycle_p ** order_idx) * arr_c[:, 1]

            sin2a = np.sin(2 * alpha)
            cos2a = np.cos(2 * alpha)
            sin2bp = np.sin(beta_p) ** 2
            sin2bp_2 = np.sin(2 * beta_p)
            cosbp = np.cos(beta_p)
            cosbp2 = cosbp ** 2

            if prefix == 'l':
                sigmax = (GG / cs * sin2a * (v0_tA[:, 1] - A1 * v0_tB_sum)
                          + A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC_sum)
                sigmay = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB_sum)
                          - A2 * GG * sin2bp_2 / cp * v0_tC_sum)
            elif prefix == 'r':
                sigmax = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB_sum)
                          - A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC_sum)
                sigmay = (GG / cs * cos2a * (-v0_tA[:, 1] - A1 * v0_tB_sum)
                          + A2 * GG * sin2bp_2 / cp * v0_tC_sum)
            elif prefix == 'b':
                sigmax = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB_sum)
                          - A2 * GG * sin2bp_2 / cp * v0_tC_sum)
                sigmay = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB_sum)
                          + A2 * (lam + 2 * GG * cosbp2) / cp * v0_tC_sum)
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
    log_step(logger, '%s 自由场应力已计算', model_name)

    # ============ 计算等效节点力 ============
    def calc_equiv_node_force_general(node_data, prefix):
        for i in range(node_data.shape[0]):
            node_id = int(node_data[i, 0])
            A = node_data[i, 3]
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

    # 创建幅值曲线 (Amplitude)
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

    # 施加集中力载荷
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


def build_models(acc_info, base_model, part_name, inst_name,
                 angle,
                 cs_bedrock, vv_bedrock, density_bedrock,
                 cs_overlying, vv_overlying, density_overlying,
                 cs_surface, vv_surface, density_surface,
                 bedrock_thickness, h1,
                 H_upper, H_lower, left_flat, w_slope,
                 max_reflect_order=1,
                 step_name=DEFAULT_STEP_NAME, variables=('S', 'U', 'V'), frequency=10,
                 model_scene='slope', logger=None):
    """根据加速度时程信息批量复制模型、创建分析步、施加人工边界。"""
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
        VAB_oblique(angle=angle,
                    cs_bedrock=cs_bedrock, vv_bedrock=vv_bedrock, density_bedrock=density_bedrock,
                    cs_overlying=cs_overlying, vv_overlying=vv_overlying, density_overlying=density_overlying,
                    cs_surface=cs_surface, vv_surface=vv_surface, density_surface=density_surface,
                    bedrock_thickness=bedrock_thickness, h1=h1,
                    H_upper=H_upper, H_lower=H_lower, left_flat=left_flat, w_slope=w_slope,
                    max_reflect_order=max_reflect_order,
                    model_name=new_model_name, part_name=part_name,
                    inst_name=inst_name,
                    acc_file=acc_file, step_name=step_name,
                    logger=logger)
        model_names.append(new_model_name)

    return model_names


def submit_job(num_cpus=7, memory_percent=90, model_name='Model-1', logger=None):
    """创建并提交Abaqus作业"""
    logger = logger or log_step()
    t0 = time.time()
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
        log_step(logger, '检测到同名旧作业，已删除: %s', job_name)
    log_step(logger, '%s作业开始提交, CPU 数量=%d, 内存=%d%%',
             job_name, num_cpus, memory_percent)

    mdb.Job(name=job_name, model=model_name,
            description='VAB oblique SV-wave analysis (Three-layered slope)',
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
