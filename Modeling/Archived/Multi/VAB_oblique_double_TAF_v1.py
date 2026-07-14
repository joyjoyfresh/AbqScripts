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
    logger = log_step('VAB_oblique_double_v2.log')  # 初始化日志并写入当前版本日志文件
    total_start = time.time()  # 记录主流程起始时间

    # 统一集中配置参数，便于维护和批量改动。
    material_cfg = {
        'angle': 30,  # 设置 SV 波入射角度（度）
        'layer1': {
            'elastic_modulus': 2.6e10,  # 设置 Layer1 杨氏模量（Pa），对应 Vs = 2000 m/s
            'poisson_ratio': 0.3,  # 设置 Layer1 泊松比
            'density': 2500,  # 设置 Layer1 密度（kg/m^3）
        },
        'layer2': {
            'elastic_modulus': 1.664e10,  # 设置 Layer2 杨氏模量（Pa），对应 Vs = 1600 m/s (VR/Vs = 1.25)
            'poisson_ratio': 0.3,  # 设置 Layer2 泊松比
            'density': 2500,  # 设置 Layer2 密度（kg/m^3）
        },
        'max_reflect_order': 3,  # 设置多次反射/透射最大阶数
    }
    geometry_cfg = {
        'slope_h': 200.0,  # 设置斜坡高度（m）
        'left_flat': 1000.0,  # 设置左平台长度（m）
        'right_total': 800.0,  # 设置从坡顶到右边界的总水平长度（m）
        'i': 45,  # 设置斜坡倾角（度）
        'mesh_size_manual': 4,  # 设置手动网格尺寸上限（m）
        'f_max': 15,  # 设置目标最高频率（Hz）
        'n_per_wave': 10,  # 设置每波长单元数
        'interface_h': 200.0,  # 设置用户自定义界面高度（m，即基岩层厚度）
    }
    job_cfg = {
        'variables': ('U', 'V', 'A'),  # 设置场输出变量
        'frequency': 1,  # 设置输出频率
        'num_cpus': 7,  # 设置并行 CPU 数量
        'memory_percent': 90,  # 设置作业内存百分比
    }

    try:
        log_step(logger, '脚本开始执行')  # 写入脚本启动日志

        cs1 = _compute_wave_speed_from_elastic_modulus(  # 根据 Layer1 材料参数计算剪切波速
            material_cfg['layer1']['elastic_modulus'],
            material_cfg['layer1']['poisson_ratio'],
            material_cfg['layer1']['density'])
        cs2 = _compute_wave_speed_from_elastic_modulus(  # 根据 Layer2 材料参数计算剪切波速
            material_cfg['layer2']['elastic_modulus'],
            material_cfg['layer2']['poisson_ratio'],
            material_cfg['layer2']['density'])

        slope_h = geometry_cfg['slope_h']  # 读取斜坡高度
        left_flat = geometry_cfg['left_flat']  # 读取左平台长度
        interface_h = geometry_cfg['interface_h']  # 基岩层厚度/界面高度
        H_lower = interface_h + 200.0  # 下垫面高度（保证 h/H = 0.5，h = 200m，H = 400m）
        total_L = left_flat + geometry_cfg['right_total']  # 总模型长度
        mesh_size_auto = min(cs1, cs2) / (geometry_cfg['f_max'] * geometry_cfg['n_per_wave'])  # 按较慢层波速计算自动网格尺寸
        mesh_size = min(mesh_size_auto, geometry_cfg['mesh_size_manual'])  # 取自动尺寸与手动上限中的较小值

        cae_name = 'h{}_i{}_a{}.cae'.format(int(slope_h), geometry_cfg['i'], material_cfg['angle'])  # 生成 CAE 文件名
        H_flat = H_lower + slope_h  # 平坦自由场模型总高度（与坡地模型左侧等高）

        acc_info = find_acc_txt(logger)  # 读取当前目录内全部加速度时程信息

        base_model, part_name, inst_name = create_model(  # 创建基础几何与网格模型
            total_L=total_L,
            slope_h=slope_h,
            left_flat=left_flat,
            i=geometry_cfg['i'],
            cs1=cs1,
            vv1=material_cfg['layer1']['poisson_ratio'],
            density1=material_cfg['layer1']['density'],
            cs2=cs2,
            vv2=material_cfg['layer2']['poisson_ratio'],
            density2=material_cfg['layer2']['density'],
            interface_h=interface_h,
            mesh_size=mesh_size,
            H_lower=H_lower,
            cae_name=cae_name,
            logger=logger)

        flat_base_model, flat_part_name, flat_inst_name = create_flat_model(  # 创建平坦自由场基础模型
            total_L=total_L,
            H_flat=H_flat,
            cs1=cs1,
            vv1=material_cfg['layer1']['poisson_ratio'],
            density1=material_cfg['layer1']['density'],
            cs2=cs2,
            vv2=material_cfg['layer2']['poisson_ratio'],
            density2=material_cfg['layer2']['density'],
            interface_h=interface_h,
            mesh_size=mesh_size,
            logger=logger)

        slope_model_names = build_models(  # 依据不同地震动复制坡地模型并施加等效边界
            acc_info=acc_info,
            base_model=base_model,
            part_name=part_name,
            inst_name=inst_name,
            angle=material_cfg['angle'],
            cs1=cs1,
            vv1=material_cfg['layer1']['poisson_ratio'],
            density1=material_cfg['layer1']['density'],
            cs2=cs2,
            vv2=material_cfg['layer2']['poisson_ratio'],
            density2=material_cfg['layer2']['density'],
            H1=geometry_cfg['interface_h'],
            max_reflect_order=material_cfg['max_reflect_order'],
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
            cs1=cs1,
            vv1=material_cfg['layer1']['poisson_ratio'],
            density1=material_cfg['layer1']['density'],
            cs2=cs2,
            vv2=material_cfg['layer2']['poisson_ratio'],
            density2=material_cfg['layer2']['density'],
            H1=geometry_cfg['interface_h'],
            max_reflect_order=material_cfg['max_reflect_order'],
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


def _build_model_name_from_record(acc_file, scene_tag):
    """按"记录名-场景名"规则生成模型名，如 El_Centro-slope。"""
    record_name = os.path.splitext(os.path.basename(acc_file))[0]  # 从加速度文件名中提取不含扩展名的记录名
    if not record_name:  # 校验记录名不能为空
        raise ValueError('无法从加速度文件生成记录名: %s' % acc_file)  # 记录名为空时抛出异常
    if scene_tag not in ('slope', 'flat'):  # 校验场景标签是否受支持
        raise ValueError('scene_tag 仅支持 slope 或 flat，当前为: %s' % scene_tag)  # 场景标签非法时抛出异常
    return '{}-{}'.format(record_name, scene_tag)  # 返回"记录名-场景名"格式模型名


def _compute_wave_speed_from_elastic_modulus(elastic_modulus, poisson_ratio, density):
    """根据杨氏模量、泊松比和密度计算剪切波速。"""
    return math.sqrt((elastic_modulus / (2 * (1 + poisson_ratio))) / density)  # 按弹性理论公式计算 Vs


def _safe_arcsin(value):
    """对 arcsin 输入做截断，避免浮点超界。"""
    return math.asin(max(-1.0, min(1.0, value)))  # 将输入限制在[-1,1]后再计算反三角函数


def _compute_material_params(cs, vv, density):
    """根据 Vs、泊松比、密度计算材料参数。"""
    GG = density * cs ** 2  # 计算剪切模量 G
    EE = 2 * GG * (1 + vv)  # 计算杨氏模量 E
    lam = 2 * GG * vv / (1 - 2 * vv)  # 计算拉梅常数 lambda
    cp = math.sqrt((lam + 2 * GG) / density)  # 计算纵波波速 Vp
    return {'GG': GG, 'EE': EE, 'lam': lam, 'cp': cp, 'cs': cs, 'vv': vv, 'density': density}  # 返回参数字典


def _compute_interface_sv_coeff(alpha1, mat1, mat2):
    """计算 SV 波在两层界面的等效反射/透射系数（阻抗近似）。"""
    z1s = mat1['density'] * mat1['cs'] * max(1e-8, math.cos(alpha1))  # 计算 Layer1 对应的 SV 斜入射阻抗
    sin_a2 = mat2['cs'] * math.sin(alpha1) / mat1['cs']  # 按 Snell 定律计算 Layer2 的 SV 正弦项
    alpha2 = _safe_arcsin(sin_a2)  # 计算 Layer2 的 SV 折射角
    z2s = mat2['density'] * mat2['cs'] * max(1e-8, math.cos(alpha2))  # 计算 Layer2 对应的 SV 斜入射阻抗
    denom = z1s + z2s if abs(z1s + z2s) > 1e-12 else 1e-12  # 防止分母过小导致数值爆炸
    rss = (z2s - z1s) / denom  # 计算 SV 反射系数
    tss = 2.0 * z2s / denom  # 计算 SV 透射系数
    rsp = 0.0  # 当前版本中 P-SV 转换采用近似忽略
    tsp = 0.0  # 当前版本中 SV-P 转换采用近似忽略
    return {'Rss': rss, 'Rsp': rsp, 'Tss': tss, 'Tsp': tsp, 'alpha2': alpha2}  # 返回界面系数与折射角


def _compute_free_surface_sv_coeff(alpha, cp, cs):
    """计算 SV 波在自由面的反射系数。"""
    beta_p = _safe_arcsin(cp * math.sin(alpha) / cs)  # 计算反射 P 波角度
    numerator_a1 = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) - cp ** 2 * math.cos(2 * alpha) ** 2  # 计算 A1 分子
    denominator = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) + cp ** 2 * math.cos(2 * alpha) ** 2  # 计算公共分母
    if abs(denominator) < 1e-12:  # 判断分母是否接近零
        denominator = 1e-12  # 对极小分母进行保护
    a1 = numerator_a1 / denominator  # 计算 SV->SV 反射系数
    a2 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator  # 计算 SV->P 反射系数
    return {'A1': a1, 'A2': a2, 'beta': beta_p}  # 返回系数与反射角


def _compute_free_surface_p_coeff(beta, cp, cs):
    """计算 P 波在自由面的反射系数。"""
    alpha = _safe_arcsin(cs * math.sin(beta) / cp)  # 计算反射 SV 波角度
    numerator_b2 = cp ** 2 * math.cos(2 * alpha) ** 2 - cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 计算 B2 分子
    denominator = cp ** 2 * math.cos(2 * alpha) ** 2 + cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 计算公共分母
    if abs(denominator) < 1e-12:  # 判断分母是否接近零
        denominator = 1e-12  # 对极小分母进行保护
    b2 = numerator_b2 / denominator  # 计算 P->P 反射系数
    b1 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator  # 计算 P->SV 反射系数
    return {'B1': b1, 'B2': b2, 'alpha': alpha}  # 返回系数与反射角


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


def create_model(total_L, slope_h, left_flat, i,
                 cs1, vv1, density1,
                 cs2, vv2, density2,
                 interface_h,
                 mesh_size,
                 H_lower=None, cae_name=None,
                 logger=None):
    """
    创建二维平面应变模型：几何、材料、截面、装配、网格（不含分析步）
    参数:
        total_L     (float): 模型总水平长度 (m)
        slope_h     (float): 斜坡高度 (m)
        left_flat   (float): 左侧（上部）平地长度 (m)
        i           (float): 斜坡倾角 (°)
        cs1/cs2     (float): Layer 1/2 剪切波速 (m/s)
        vv1/vv2     (float): Layer 1/2 泊松比
        density1/2  (float): Layer 1/2 密度 (kg/m³)
        interface_h (float): 用户设定界面高度 (m)
        mesh_size   (float): 网格尺寸 (m)
        H_lower     (float): 下垫面高度 (m)
    几何逻辑（6个关键点，逆时针闭合）:
        w_slope    = slope_h / tan(i)          斜坡水平投影宽度
        right_flat = total_L - left_flat - w_slope  右平台长度（自动剩余）
        P1=(0, 0),  P2=(total_L, 0),
        P3=(total_L, H_lower),  P4=(left_flat+w_slope, H_lower),
        P5=(left_flat, H_upper), P6=(0, H_upper)
    """
    logger = logger or log_step()
    model_name = 'Model-1'

    if slope_h <= 0:
        raise ValueError('slope_h 必须 > 0')
    if i <= 0 or i >= 90:
        raise ValueError('倾角 i 必须在 (0, 90) 范围内')
    if H_lower is None:
        H_lower = interface_h + 200.0
    if H_lower <= 0:
        raise ValueError('H_lower 必须 > 0')
    if interface_h <= 0:
        raise ValueError('interface_h 必须 > 0')

    w_slope = slope_h / math.tan(math.radians(i))
    right_flat = total_L - left_flat - w_slope
    if right_flat <= 0:
        raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                         (total_L, left_flat, w_slope))
    H_upper = H_lower + slope_h   # 左侧（上覆）地表高度
    
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
    mat1_params = _compute_material_params(cs1, vv1, density1)  # 计算 Layer1 材料参数
    mat2_params = _compute_material_params(cs2, vv2, density2)  # 计算 Layer2 材料参数

    mat1_name = _next_available_name('Material-L1', model.materials)  # 生成 Layer1 材料名
    mat1 = model.Material(name=mat1_name)  # 创建 Layer1 材料对象
    mat1.Elastic(table=((mat1_params['EE'], vv1),))  # 写入 Layer1 弹性参数
    mat1.Density(table=((density1,),))  # 写入 Layer1 密度参数

    mat2_name = _next_available_name('Material-L2', model.materials)  # 生成 Layer2 材料名
    mat2 = model.Material(name=mat2_name)  # 创建 Layer2 材料对象
    mat2.Elastic(table=((mat2_params['EE'], vv2),))  # 写入 Layer2 弹性参数
    mat2.Density(table=((density2,),))  # 写入 Layer2 密度参数
    log_step(logger, '%s 双层材料已定义: %s, %s', model_name, mat1_name, mat2_name)  # 记录双层材料创建完成

    sec1_name = _next_available_name('Section-L1', model.sections)  # 生成 Layer1 截面名
    model.HomogeneousSolidSection(name=sec1_name, material=mat1_name, thickness=1.0)  # 创建 Layer1 截面
    sec2_name = _next_available_name('Section-L2', model.sections)  # 生成 Layer2 截面名
    model.HomogeneousSolidSection(name=sec2_name, material=mat2_name, thickness=1.0)  # 创建 Layer2 截面
    log_step(logger, '%s 双层截面已创建: %s, %s', model_name, sec1_name, sec2_name)  # 记录双层截面创建完成

    # ============ 装配 ============
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)
    inst_name = _next_available_name(part_name, assembly.instances)
    assembly.Instance(name=inst_name, part=part, dependent=ON)
    log_step(logger, '%s 装配实例已创建: %s', model_name, inst_name)

    # ============ 网格划分与几何切分 ============
    # 1. 网格前按坡底点水平切分面，分为上下两部分
    # 分割线：从坡底点 (left_flat + w_slope, H_lower) 水平连到左边界 (0, H_lower)
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(
        name='__partition__', sheetSize=max(total_L, H_upper) * 2
    )
    partition_sketch.Line(point1=(left_flat + w_slope, H_lower), point2=(0.0, H_lower))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__partition__']
    log_step(logger, '%s 网格前切割完成 (坡底点水平切分)', model_name)

    # 2. 在 interface_h 处水平切分面，确保基岩与土层边界清晰
    part_faces = part.faces
    partition_sketch2 = model.ConstrainedSketch(
        name='__partition_interface__', sheetSize=max(total_L, H_upper) * 2
    )
    partition_sketch2.Line(point1=(0.0, interface_h), point2=(total_L, interface_h))
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch2)
    del model.sketches['__partition_interface__']
    log_step(logger, '%s 基岩水平面切分完成 (interface_h = %.3f)', model_name, interface_h)

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

    # 按界面高度将面分配到不同截面，允许界面高度与几何分割线不一致。
    layer1_faces = []  # 初始化 Layer1 面列表
    layer2_faces = []  # 初始化 Layer2 面列表
    for face in part.faces:  # 遍历当前零件所有面
        centroid = face.getCentroid()  # 获取面质心坐标
        yc = centroid[1] if len(centroid) >= 2 else centroid[0][1]  # 兼容不同版本返回格式读取 y 坐标
        if yc < interface_h:  # 判断质心是否位于界面以下
            layer1_faces.append(face)  # 将面归类到 Layer1
        else:  # 处理界面以上面
            layer2_faces.append(face)  # 将面归类到 Layer2

    def _to_face_sequence(face_list):  # 定义列表到 GeomSequence 的转换函数
        face_seq = part.faces[0:0]  # 构造空的 FaceArray 作为初始序列
        for face in face_list:  # 遍历目标面列表
            face_seq = face_seq + part.faces[face.index:face.index + 1]  # 逐个拼接为 Abaqus 认可的 FaceArray
        return face_seq  # 返回可用于 Region 的 FaceArray

    if len(layer1_faces) > 0:  # 判断 Layer1 面集合是否非空
        layer1_face_seq = _to_face_sequence(layer1_faces)  # 将 Layer1 列表转换为 GeomSequence
        part.SectionAssignment(region=Region(faces=layer1_face_seq), sectionName=sec1_name, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)  # 对 Layer1 面批量分配 Layer1 截面
    if len(layer2_faces) > 0:  # 判断 Layer2 面集合是否非空
        layer2_face_seq = _to_face_sequence(layer2_faces)  # 将 Layer2 列表转换为 GeomSequence
        part.SectionAssignment(region=Region(faces=layer2_face_seq), sectionName=sec2_name, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)  # 对 Layer2 面批量分配 Layer2 截面
    log_step(logger, '%s 截面分配完成: interface_h=%.3f, L1面=%d, L2面=%d', model_name, interface_h, len(layer1_faces), len(layer2_faces))  # 记录界面高度分层结果

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
            y_slope = H_upper - (x - left_flat) * slope_h / w_slope  # 计算当前x对应的斜坡理论y坐标
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


def create_flat_model(total_L, H_flat,
                     cs1, vv1, density1,
                     cs2, vv2, density2,
                     interface_h,
                     mesh_size, logger=None):
    """创建二维平面应变平坦自由场模型：矩形几何、双层材料、截面、装配与网格。
    参数:
        total_L     (float): 模型总水平长度 (m)
        H_flat      (float): 模型总高度 (m)
        cs1/cs2     (float): Layer 1/2 剪切波速 (m/s)
        vv1/vv2     (float): Layer 1/2 泊松比
        density1/2  (float): Layer 1/2 密度 (kg/m³)
        interface_h (float): 界面高度 (m)，Layer1 与 Layer2 的分界
        mesh_size   (float): 网格尺寸 (m)
    """
    logger = logger or log_step()  # 复用已有日志器或初始化默认日志器
    model_name = 'Model-2'  # 指定平坦自由场基础模型名称

    if total_L <= 0:  # 校验模型长度参数
        raise ValueError('total_L 必须 > 0')  # 长度非法时抛出异常
    if H_flat <= 0:  # 校验模型高度参数
        raise ValueError('H_flat 必须 > 0')  # 高度非法时抛出异常

    model = mdb.Model(name=model_name)  # 创建平坦自由场基础模型
    log_step(logger, '%s 基础模型开始创建（平坦自由场）', model_name)  # 记录平坦模型创建开始日志

    # ============ 创建矩形 Part ============
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

    # ============ 双层材料与截面 ============
    mat1_params = _compute_material_params(cs1, vv1, density1)  # 计算 Layer1 材料参数
    mat2_params = _compute_material_params(cs2, vv2, density2)  # 计算 Layer2 材料参数

    mat1_name = _next_available_name('Material-L1', model.materials)  # 生成 Layer1 材料名
    mat1 = model.Material(name=mat1_name)  # 创建 Layer1 材料对象
    mat1.Elastic(table=((mat1_params['EE'], vv1),))  # 写入 Layer1 弹性参数
    mat1.Density(table=((density1,),))  # 写入 Layer1 密度参数

    mat2_name = _next_available_name('Material-L2', model.materials)  # 生成 Layer2 材料名
    mat2 = model.Material(name=mat2_name)  # 创建 Layer2 材料对象
    mat2.Elastic(table=((mat2_params['EE'], vv2),))  # 写入 Layer2 弹性参数
    mat2.Density(table=((density2,),))  # 写入 Layer2 密度参数
    log_step(logger, '%s 双层材料已定义: %s, %s', model_name, mat1_name, mat2_name)  # 记录双层材料创建完成

    sec1_name = _next_available_name('Section-L1', model.sections)  # 生成 Layer1 截面名
    model.HomogeneousSolidSection(name=sec1_name, material=mat1_name, thickness=1.0)  # 创建 Layer1 截面
    sec2_name = _next_available_name('Section-L2', model.sections)  # 生成 Layer2 截面名
    model.HomogeneousSolidSection(name=sec2_name, material=mat2_name, thickness=1.0)  # 创建 Layer2 截面
    log_step(logger, '%s 双层截面已创建: %s, %s', model_name, sec1_name, sec2_name)  # 记录双层截面创建完成

    # ============ 装配 ============
    assembly = model.rootAssembly  # 获取根装配体对象
    assembly.DatumCsysByDefault(CARTESIAN)  # 设置默认笛卡尔坐标系
    inst_name = _next_available_name(part_name, assembly.instances)  # 生成实例名称
    assembly.Instance(name=inst_name, part=part, dependent=ON)  # 创建装配实例
    log_step(logger, '%s 装配实例已创建: %s', model_name, inst_name)  # 记录实例创建日志

    # ============ 水平切分面（interface_h 处分界） ============
    part_faces = part.faces  # 获取当前全部面
    partition_sketch = model.ConstrainedSketch(
        name='__flat_interface_partition__', sheetSize=max(total_L, H_flat) * 2
    )  # 创建切分草图
    partition_sketch.Line(point1=(0.0, interface_h), point2=(total_L, interface_h))  # 在 interface_h 处绘制水平分割线
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)  # 执行面切分
    del model.sketches['__flat_interface_partition__']  # 删除临时草图
    log_step(logger, '%s 平坦自由场界面切分完成 (interface_h=%.3f)', model_name, interface_h)  # 记录切分完成

    # ============ 网格划分 ============
    picked_regions = part.faces  # 选取全部面用于网格控制
    part.setMeshControls(regions=picked_regions, elemShape=QUAD, technique=STRUCTURED)  # 设置结构化四边形网格
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)  # 设定全局播种尺寸
    elem_type = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)  # 指定平面应变四节点单元
    part.setElementType(regions=(picked_regions,), elemTypes=(elem_type,))  # 将单元类型分配给全部面
    part.generateMesh()  # 执行网格生成
    log_step(logger, '%s 网格已生成: 尺寸=%.3f, 单元=CPE4', model_name, mesh_size)  # 记录网格生成日志

    # ============ 按界面高度分配截面 ============
    layer1_faces = []  # 初始化 Layer1 面列表
    layer2_faces = []  # 初始化 Layer2 面列表
    for face in part.faces:  # 遍历当前零件所有面
        centroid = face.getCentroid()  # 获取面质心坐标
        yc = centroid[1] if len(centroid) >= 2 else centroid[0][1]  # 兼容不同版本返回格式读取 y 坐标
        if yc < interface_h:  # 判断质心是否位于界面以下
            layer1_faces.append(face)  # 将面归类到 Layer1
        else:  # 处理界面以上面
            layer2_faces.append(face)  # 将面归类到 Layer2

    def _to_face_sequence(face_list):  # 定义列表到 GeomSequence 的转换函数
        face_seq = part.faces[0:0]  # 构造空的 FaceArray 作为初始序列
        for face in face_list:  # 遍历目标面列表
            face_seq = face_seq + part.faces[face.index:face.index + 1]  # 逐个拼接为 Abaqus 认可的 FaceArray
        return face_seq  # 返回可用于 Region 的 FaceArray

    if len(layer1_faces) > 0:  # 判断 Layer1 面集合是否非空
        layer1_face_seq = _to_face_sequence(layer1_faces)  # 将 Layer1 列表转换为 GeomSequence
        part.SectionAssignment(region=Region(faces=layer1_face_seq), sectionName=sec1_name, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)  # 对 Layer1 面批量分配 Layer1 截面
    if len(layer2_faces) > 0:  # 判断 Layer2 面集合是否非空
        layer2_face_seq = _to_face_sequence(layer2_faces)  # 将 Layer2 列表转换为 GeomSequence
        part.SectionAssignment(region=Region(faces=layer2_face_seq), sectionName=sec2_name, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)  # 对 Layer2 面批量分配 Layer2 截面
    log_step(logger, '%s 截面分配完成: interface_h=%.3f, L1面=%d, L2面=%d', model_name, interface_h, len(layer1_faces), len(layer2_faces))  # 记录界面高度分层结果

    # 重新生成装配体以同步网格
    assembly.regenerate()  # 同步网格信息
    log_step(logger, '%s 装配已重新生成', model_name)  # 记录装配同步日志

    # ============ 创建边界节点集（左/右/底/顶） ============
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


def VAB_oblique(angle,
                cs1, vv1, density1,
                cs2, vv2, density2,
                H1,
                max_reflect_order=1,
                model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None):
    """
    主函数：为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力
    参数:
        angle  (float): SV波入射角度（度），0为垂直入射
        cs1/2     (float): 双层剪切波速 (m/s)
        vv1/2     (float): 双层泊松比
        density1/2(float): 双层密度 (kg/m³)
        H1        (float): 界面高度 (m)
        max_reflect_order (int): 多次界面反射/透射迭代阶数
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
    mat1 = _compute_material_params(cs1, vv1, density1)  # 计算 Layer1 材料参数
    mat2 = _compute_material_params(cs2, vv2, density2)  # 计算 Layer2 材料参数

    def _pick_material_by_y(y_coord):
        """按节点 y 坐标选择层参数。"""
        return mat1 if y_coord < H1 else mat2  # 小于界面高度取 Layer1，否则取 Layer2

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
    if H1 <= ymin or H1 >= ymax:
        raise ValueError('H1 必须位于模型高度范围内: (%.3f, %.3f)' % (ymin, ymax))

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
    log_step(logger, '%s 开始按界面高度 H1=%.3f 计算分层弹簧-阻尼系数', model_name, H1)

    def add_spring_damper(node_data):
        """将弹簧刚度和阻尼系数乘以影响长度，追加到 node_data"""
        influence = node_data[:, 3]  # 读取影响长度数组
        kns = np.zeros_like(influence)  # 初始化法向刚度数组
        cns = np.zeros_like(influence)  # 初始化法向阻尼数组
        kts = np.zeros_like(influence)  # 初始化切向刚度数组
        cts = np.zeros_like(influence)  # 初始化切向阻尼数组
        for idx in range(node_data.shape[0]):  # 遍历节点逐个按层赋值
            y0 = node_data[idx, 2]  # 读取节点 y 坐标
            mat = _pick_material_by_y(y0)  # 依据界面高度选择层参数
            kn_coeff = mat['GG'] / 2.0 / ymax  # 计算当前层法向弹簧刚度系数
            cn_coeff = mat['density'] * mat['cp']  # 计算当前层法向阻尼系数
            kt_coeff = mat['GG'] / 4.0 / ymax  # 计算当前层切向弹簧刚度系数
            ct_coeff = mat['density'] * mat['cs']  # 计算当前层切向阻尼系数
            kns[idx] = kn_coeff * influence[idx]  # 写入法向刚度
            cns[idx] = cn_coeff * influence[idx]  # 写入法向阻尼
            kts[idx] = kt_coeff * influence[idx]  # 写入切向刚度
            cts[idx] = ct_coeff * influence[idx]  # 写入切向阻尼
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
    if angle == 0:  # 判断入射角是否为零
        angle = 1e-10  # 使用极小角度避免后续除零
    else:  # 处理非零入射角
        angle = round(angle, 4)  # 将角度统一保留四位小数

    alpha1 = math.radians(angle)  # 计算 Layer1 中 SV 入射角
    interface_coeff_12 = _compute_interface_sv_coeff(alpha1, mat1, mat2)  # 计算 1->2 界面系数
    alpha2 = interface_coeff_12['alpha2']  # 读取 Layer2 中 SV 折射角
    beta1 = _safe_arcsin(mat1['cp'] * math.sin(alpha1) / mat1['cs'])  # 计算 Layer1 中 P 角
    beta2 = _safe_arcsin(mat2['cp'] * math.sin(alpha2) / mat2['cs']) if abs(math.sin(alpha2)) > 0 else 1e-10  # 计算 Layer2 中 P 角

    free_sv_2 = _compute_free_surface_sv_coeff(alpha2, mat2['cp'], mat2['cs'])  # 计算 Layer2 自由面 SV 反射系数
    free_p_2 = _compute_free_surface_p_coeff(beta2, mat2['cp'], mat2['cs'])  # 计算 Layer2 自由面 P 反射系数
    interface_coeff_21 = _compute_interface_sv_coeff(alpha2, mat2, mat1)  # 计算 2->1 界面系数

    rss_primary = interface_coeff_12['Rss']  # 读取一次界面 SV 反射系数
    rsp_primary = interface_coeff_12['Rsp']  # 读取一次界面 P 反射系数
    tss_primary = interface_coeff_12['Tss']  # 读取一次界面 SV 透射系数
    tsp_primary = interface_coeff_12['Tsp']  # 读取一次界面 P 透射系数

    # 使用有限阶截断叠加 Layer2 往返反射影响，形成等效界面反射系数。
    h2 = max(0.0, ymax - H1)  # 计算 Layer2 有效厚度
    cycle_sv = free_sv_2['A1'] * interface_coeff_21['Rss']  # 计算 SV 分量的单次往返衰减因子
    cycle_p = free_p_2['B2'] * interface_coeff_21['Rss']  # 计算 P 分量的单次往返衰减因子
    order_count = max(0, int(max_reflect_order))  # 将迭代阶数转为非负整数

    sum_cycle_sv = 0.0  # 初始化 SV 往返叠加和
    sum_cycle_p = 0.0  # 初始化 P 往返叠加和
    for order_idx in range(order_count + 1):  # 按阶数累加多次反射贡献
        sum_cycle_sv += cycle_sv ** order_idx  # 累加 SV 往返几何级数项
        sum_cycle_p += cycle_p ** order_idx  # 累加 P 往返几何级数项

    Rss_eff = rss_primary + tss_primary * free_sv_2['A1'] * interface_coeff_21['Tss'] * sum_cycle_sv  # 计算等效 SV 反射系数
    Rsp_eff = rsp_primary + tss_primary * free_sv_2['A2'] * interface_coeff_21['Tss'] * sum_cycle_p  # 计算等效 P 反射系数
    cycle_delay_sv = (2.0 * h2 * math.cos(alpha2) / mat2['cs']) if h2 > 0 else 0.0  # 计算 SV 每次往返附加时间
    cycle_delay_p = (2.0 * h2 * math.cos(beta2) / mat2['cp']) if h2 > 0 else 0.0  # 计算 P 每次往返附加时间
    log_step(logger, '%s 双层反射参数已计算: Rss_eff=%.4f, Rsp_eff=%.4f, order=%d', model_name, Rss_eff, Rsp_eff, order_count)  # 记录双层反射参数

    # 兼容后续原有公式变量命名，基准层采用 Layer1 参数。
    alpha = alpha1  # 将基准 SV 角映射到旧变量名
    beta_p = beta1  # 将基准 P 角映射到旧变量名
    A1 = Rss_eff  # 将等效 SV 反射系数映射到旧变量名
    A2 = Rsp_eff  # 将等效 P 反射系数映射到旧变量名
    GG = mat1['GG']  # 将 Layer1 剪切模量映射到旧变量名
    lam = mat1['lam']  # 将 Layer1 拉梅常数映射到旧变量名
    cs = mat1['cs']  # 将 Layer1 剪切波速映射到旧变量名
    cp = mat1['cp']  # 将 Layer1 纵波波速映射到旧变量名

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
    Ly = H1 - ymin  # 将主反射路径高度基准设为用户界面高度
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
        if n_delay < 0:
            n_delay = 0
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
            u0_tB_list = []  # 初始化多阶 SV 反射时程列表
            u0_tC_list = []  # 初始化多阶 P 反射时程列表
            for order_idx in range(order_count + 1):  # 循环生成多次反射延迟信号
                delay_b = tB + order_idx * cycle_delay_sv  # 计算当前阶 SV 延迟时间
                delay_c = tC + order_idx * cycle_delay_p  # 计算当前阶 P 延迟时间
                u0_tB_list.append(get_delayed(delay_b))  # 存入当前阶 SV 延迟时程
                u0_tC_list.append(get_delayed(delay_c))  # 存入当前阶 P 延迟时程

            max_len = u0_tA.shape[0]  # 以 A 波长度初始化最大长度
            for arr in u0_tB_list + u0_tC_list:  # 遍历全部反射分量长度
                max_len = max(max_len, arr.shape[0])  # 更新统一对齐长度

            u0_tA = pad_to(u0_tA, max_len, dt)  # 将 A 波补零到统一长度
            u0_tB_sum = np.zeros(max_len)  # 初始化多阶 SV 反射叠加值
            u0_tC_sum = np.zeros(max_len)  # 初始化多阶 P 反射叠加值
            for order_idx in range(order_count + 1):  # 逐阶叠加 SV/P 反射分量
                arr_b = pad_to(u0_tB_list[order_idx], max_len, dt)  # 当前阶 SV 时程补零对齐
                arr_c = pad_to(u0_tC_list[order_idx], max_len, dt)  # 当前阶 P 时程补零对齐
                u0_tB_sum += (cycle_sv ** order_idx) * arr_b[:, 1]  # 按阶数权重叠加 SV 分量
                u0_tC_sum += (cycle_p ** order_idx) * arr_c[:, 1]  # 按阶数权重叠加 P 分量

            # 自由场位移/速度叠加（入射SV + 反射SV + 反射P）
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
            v0_tB_list = []  # 初始化多阶 SV 反射速度时程列表
            v0_tC_list = []  # 初始化多阶 P 反射速度时程列表
            for order_idx in range(order_count + 1):  # 循环生成多次反射延迟速度
                delay_b = tB + order_idx * cycle_delay_sv  # 计算当前阶 SV 延迟时间
                delay_c = tC + order_idx * cycle_delay_p  # 计算当前阶 P 延迟时间
                v0_tB_list.append(get_delayed(delay_b))  # 存入当前阶 SV 速度时程
                v0_tC_list.append(get_delayed(delay_c))  # 存入当前阶 P 速度时程

            max_len = v0_tA.shape[0]  # 以 A 波长度初始化最大长度
            for arr in v0_tB_list + v0_tC_list:  # 遍历全部反射分量长度
                max_len = max(max_len, arr.shape[0])  # 更新统一对齐长度

            v0_tA = pad_to(v0_tA, max_len, dt)  # 将 A 波补零到统一长度
            v0_tB_sum = np.zeros(max_len)  # 初始化多阶 SV 反射速度叠加值
            v0_tC_sum = np.zeros(max_len)  # 初始化多阶 P 反射速度叠加值
            for order_idx in range(order_count + 1):  # 逐阶叠加 SV/P 反射速度分量
                arr_b = pad_to(v0_tB_list[order_idx], max_len, dt)  # 当前阶 SV 速度补零对齐
                arr_c = pad_to(v0_tC_list[order_idx], max_len, dt)  # 当前阶 P 速度补零对齐
                v0_tB_sum += (cycle_sv ** order_idx) * arr_b[:, 1]  # 按阶数权重叠加 SV 速度
                v0_tC_sum += (cycle_p ** order_idx) * arr_c[:, 1]  # 按阶数权重叠加 P 速度

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


def build_models(acc_info, base_model, part_name, inst_name,
                 angle,
                 cs1, vv1, density1,
                 cs2, vv2, density2,
                 H1,
                 max_reflect_order=1,
                 step_name=DEFAULT_STEP_NAME, variables=('S', 'U', 'V'), frequency=10,
                 model_scene='slope', logger=None):
    """
    根据加速度时程信息批量复制模型、创建分析步、施加人工边界。

    参数:
        acc_info    (list): find_acc_txt 返回的列表 [(acc_file, tp, inc), ...]
        base_model  (str): 基础模型名称
        part_name   (str): 零件名称
        inst_name   (str): 实例名称
        双层材料参数 + H1 + max_reflect_order: 人工边界参数
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
        new_model_name = _build_model_name_from_record(acc_file, model_scene)  # 按"记录名-场景名"自动生成模型名
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
        VAB_oblique(angle,
                cs1, vv1, density1,
                cs2, vv2, density2,
                H1,
                max_reflect_order=max_reflect_order,
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