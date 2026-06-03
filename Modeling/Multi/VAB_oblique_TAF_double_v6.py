# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8

try:  # 尝试导入 Abaqus 接口（仅在 Abaqus 环境内可用）
    from abaqus import *  # 导入 Abaqus 主接口
    from abaqusConstants import *  # 导入 Abaqus 常量
    from abaqus import mdb  # 导入建模数据库对象
    from regionToolset import Region  # 导入区域工具
    from caeModules import *  # 导入 CAE 模块工具
    import mesh  # 导入网格模块
    _ABAQUS_AVAILABLE = True  # 标记 Abaqus 环境可用
except ImportError:  # 在纯 Python 环境（如自检脚本）中无 Abaqus
    _ABAQUS_AVAILABLE = False  # 标记 Abaqus 不可用，仅允许调用自由场计算相关纯数值函数
import numpy as np  # 导入数值计算库
import math  # 导入数学模块
import os  # 导入操作系统接口
import time  # 导入时间模块
import logging  # 导入日志模块
import traceback  # 导入异常堆栈模块


DEFAULT_STEP_NAME = 'Step-earthquake'  # 定义默认分析步名称
BOUNDARY_SET_NAMES = ('Left_boundary', 'Right_boundary', 'Bottom_boundary')  # 定义基础边界节点集名称
BOUNDARY_SEQUENCE = ('l', 'r', 'b')  # 定义边界处理顺序


def main():
    """脚本主入口：组织参数、建模、施加边界并提交作业。"""  # 说明主入口用途
    logger = log_step('VAB_oblique_TAF_double.log')  # 初始化日志并写入当前版本日志文件
    total_start = time.time()  # 记录主流程起始时间

    # 统一配置参数
    material_cfg = {  # 定义材料参数配置
        'angle': 15,  # 设置 SV 波入射角度（度）
        'bedrock': {  # 定义基岩材料参数
            'elastic_modulus': 26e9,  # 设置基岩杨氏模量（Pa），对应 Vs = 2000 m/s
            'poisson_ratio': 0.3,  # 设置基岩泊松比
            'density': 2500,  # 设置基岩密度（kg/m^3）
        },  # 结束基岩材料参数
        'overlying': {  # 定义覆盖层材料参数
            'poisson_ratio': 0.3,  # 设置覆盖层泊松比
            'density': 2500,  # 设置覆盖层密度
            'velocity_ratio': 1.25,  # 设置 VR / Vs 阻抗比，对应 Vs = 1600 m/s
        },  # 结束覆盖层材料参数
    }  # 结束材料参数配置

    geometry_cfg = {  # 定义几何参数配置
        'H_minus_h': 200.0,  # 设置斜坡高度差 H - h (m)
        'i': 45.0,  # 设置斜坡倾角 (度)
        'h_over_H': 0.5,  # 设置深度比 h / H
        'total_L': 1800.0,  # 设置总模型长度 (m)
        'left_flat': 1000.0,  # 设置上平台长度 (m)
        'bedrock_thickness': 200.0,  # 设置基岩层厚度 (m)
    }  # 结束几何参数配置

    job_cfg = {  # 定义作业参数配置
        'variables': ('U', 'V', 'A'),  # 设置场输出变量
        'frequency': 1,  # 设置输出频率
        'num_cpus': 8,  # 设置并行 CPU 数量
        'memory_percent': 90,  # 设置作业内存百分比
    }  # 结束作业参数配置

    try:
        log_step(logger, '脚本开始执行')  # 写入脚本启动日志

        # 波动参数计算
        cs_bedrock = _compute_wave_speed_from_elastic_modulus(
            material_cfg['bedrock']['elastic_modulus'],
            material_cfg['bedrock']['poisson_ratio'],
            material_cfg['bedrock']['density']
        )
        cs_overlying = cs_bedrock / material_cfg['overlying']['velocity_ratio']  # 计算覆盖层剪切波速

        # 几何高度计算
        H_minus_h = geometry_cfg['H_minus_h']  # 读取斜坡高度差
        h_over_H = geometry_cfg['h_over_H']  # 读取深度比
        H = H_minus_h / (1.0 - h_over_H)  # 计算总覆盖层厚度
        h = H - H_minus_h  # 计算下部覆盖层高度
        bedrock_thickness = geometry_cfg['bedrock_thickness']  # 读取基岩层厚度
        H_lower = bedrock_thickness + h  # 计算坡脚地表高度
        H_flat = bedrock_thickness + H  # 计算平坦场地总高度
        H_upper = bedrock_thickness + H  # 计算坡顶地表高度
        w_slope = H_minus_h / math.tan(math.radians(geometry_cfg['i']))  # 计算坡面水平长度
        total_L = geometry_cfg['total_L']  # 读取模型总长度
        left_flat = geometry_cfg['left_flat']  # 读取左平台长度

        mesh_size = 4  # 网格尺寸设为 4 m

        cae_name = 'h{}_i{}_a{}.cae'.format(int(H_minus_h), int(geometry_cfg['i']), int(material_cfg['angle']))  # 生成工程文件名

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
            bedrock_thickness=bedrock_thickness,
            H_upper=H_upper,
            H_lower=H_lower,
            left_flat=left_flat,
            w_slope=w_slope,
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
            bedrock_thickness=bedrock_thickness,
            H_upper=H_upper,
            H_lower=H_upper,  # 平坦自由场上部和下部都是统一总高度 H_upper
            left_flat=left_flat,
            w_slope=0.001,  # 平坦自由场相当于倾斜宽度极小
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
    except Exception as exc:  # 捕获脚本运行异常
        log_step(logger, '脚本失败: %s', str(exc))  # 记录异常摘要
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 记录完整堆栈
        raise  # 继续抛出异常以便上层处理


def _next_available_name(prefix, existing_container):  # 定义生成唯一名称的辅助函数
    """按前缀生成可用名称（如 Part-1, Part-2）。"""  # 说明函数用途
    index = 1  # 初始化序号
    while '%s-%d' % (prefix, index) in existing_container:  # 循环查找未占用名称
        index += 1  # 序号递增
    return '%s-%d' % (prefix, index)  # 返回可用名称


def _normalize_output_variables(variables):  # 定义输出变量规范化函数
    """规范化输出变量为元组，满足 Abaqus 接口要求。"""  # 说明函数用途
    if isinstance(variables, str):  # 判断是否为单个字符串
        return (variables,)  # 转换为单元素元组
    if isinstance(variables, list):  # 判断是否为列表
        return tuple(variables)  # 转换为元组
    return variables  # 其他类型保持原样返回


def _compute_wave_speed_from_elastic_modulus(elastic_modulus, poisson_ratio, density):  # 定义波速反算函数
    """根据杨氏模量、泊松比和密度计算剪切波速。"""  # 说明函数用途
    return math.sqrt((elastic_modulus / (2 * (1 + poisson_ratio))) / density)  # 返回计算得到的剪切波速


def _compute_elastic_modulus_from_wave_speed(cs, vv, density):  # 定义弹性模量反算函数
    """根据剪切波速、泊松比和密度计算杨氏模量 E。"""  # 说明函数用途
    GG = density * (cs ** 2)  # 计算剪切模量
    EE = 2 * GG * (1 + vv)  # 计算杨氏模量
    return EE  # 返回杨氏模量


def _safe_arcsin(value):  # 定义安全反正弦函数
    """对 arcsin 输入做截断，避免浮点超界。"""  # 说明函数用途
    return math.asin(max(-1.0, min(1.0, value)))  # 将输入截断到合法范围后再求反正弦


def _compute_material_params(cs, vv, density):  # 定义材料参数计算函数
    """根据 Vs、泊松比、密度计算材料参数。"""  # 说明函数用途
    GG = density * cs ** 2  # 计算剪切模量
    EE = 2 * GG * (1 + vv)  # 计算杨氏模量
    lam = 2 * GG * vv / (1 - 2 * vv)  # 计算拉梅常数
    cp = math.sqrt((lam + 2 * GG) / density)  # 计算纵波波速
    return {'GG': GG, 'EE': EE, 'lam': lam, 'cp': cp, 'cs': cs, 'vv': vv, 'density': density}  # 返回材料参数字典


def _build_model_name_from_record(acc_file, scene_tag):  # 定义模型命名函数
    """按"记录名-场景名"规则生成模型名。"""  # 说明函数用途
    record_name = os.path.splitext(os.path.basename(acc_file))[0]  # 提取不带扩展名的记录名
    if not record_name:  # 检查记录名是否为空
        raise ValueError('无法从加速度文件生成记录名: %s' % acc_file)  # 抛出命名错误
    if scene_tag not in ('slope', 'flat'):  # 检查场景标签是否合法
        raise ValueError('scene_tag 仅支持 slope 或 flat，当前为: %s' % scene_tag)  # 抛出场景错误
    return '{}-{}'.format(record_name, scene_tag)  # 返回组合后的模型名


def log_step(logger=None, message=None, *args):  # 定义日志记录函数
    """日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。"""  # 说明函数用途
    if not hasattr(log_step, '_logger'):  # 判断日志器是否已经初始化
        if logger is not None and isinstance(logger, str):  # 判断是否直接传入日志文件名
            log_filename = logger  # 保存日志文件名
            logger = None  # 清空外部 logger 引用
        else:  # 处理未传入文件名的情况
            log_filename = 'logfile.log'  # 使用默认日志文件名

        _logger = logging.getLogger('abqpy')  # 创建或获取日志器
        _logger.setLevel(logging.INFO)  # 设置日志等级
        _logger.propagate = False  # 禁止向父日志器传播

        _logger.handlers = []  # 清空旧处理器
        formatter = logging.Formatter(  # 构建日志格式器
            '%(asctime)s [%(levelname)s] %(message)s',  # 设置日志输出格式
            datefmt='%Y-%m-%d %H:%M:%S'  # 设置时间格式
        )  # 结束格式器构建

        file_handler = logging.FileHandler(log_filename, mode='w')  # 创建文件日志处理器
        file_handler.setFormatter(formatter)  # 绑定日志格式器
        _logger.addHandler(file_handler)  # 添加文件处理器到日志器

        log_step._logger = _logger  # 保存日志器到函数属性
        log_step._start_time = time.time()  # 记录日志起始时间
        log_step._log_filename = log_filename  # 保存日志文件名

        return _logger  # 返回初始化后的日志器

    if message is not None:  # 判断是否需要输出日志
        now = time.time()  # 获取当前时间
        delta_total = now - log_step._start_time  # 计算总耗时
        log_step._logger.info('[%.3fs] ' + message, delta_total, *args)  # 输出带耗时的日志

    return log_step._logger  # 返回已初始化的日志器


def find_acc_txt(logger=None):  # 定义加速度文件检索函数
    """查找当前工作目录下所有 .txt 文件，并读取每个加速度文件的分析步时长和增量步。"""  # 说明函数用途
    cwd = os.getcwd()  # 获取当前工作目录
    txt_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.txt')])  # 收集全部 txt 文件
    if len(txt_files) == 0:  # 判断是否找到文件
        raise IOError('当前目录 {} 下未找到任何 .txt 文件'.format(cwd))  # 抛出文件缺失异常

    result = []  # 初始化结果列表
    for f in txt_files:  # 遍历每个加速度文件
        time_period = 2.0  # 设置默认分析时长
        initial_inc = 0.001  # 设置默认初始增量
        try:  # 尝试读取文件内容
            acc_data = np.loadtxt(f)  # 读取加速度时程数据
            if acc_data.ndim == 2 and acc_data.shape[0] >= 2 and acc_data.shape[1] >= 2:  # 判断数据格式是否有效
                time_arr = acc_data[:, 0]  # 提取时间列
                dt = time_arr[1] - time_arr[0]  # 计算时间步长
                if dt > 0:  # 判断步长是否有效
                    time_period = time_arr[-1]  # 获取分析时长
                    initial_inc = dt  # 设置初始增量
                    if logger:  # 判断是否需要记录日志
                        log_step(logger, '已从加速度文件 %s 读取分析步参数: 时长=%.2f, 初始增量=%.3f',
                                 f, time_period, initial_inc)  # 输出读取成功日志
                else:  # 处理步长无效的情况
                    if logger:  # 判断是否需要记录日志
                        log_step(logger, '%s 中 dt <= 0，将使用默认值', f)  # 输出默认值日志
            else:  # 处理格式不合法的情况
                if logger:  # 判断是否需要记录日志
                    log_step(logger, '%s 格式无效，将使用默认值', f)  # 输出格式错误日志
        except Exception as e:  # 捕获读取异常
            if logger:  # 判断是否需要记录日志
                log_step(logger, '读取加速度时程文件失败: %s，将使用默认值', str(e))  # 输出读取失败日志
        result.append((f, time_period, initial_inc))  # 保存文件参数结果

    return result  # 返回全部文件信息


def create_model(total_L, H_minus_h, i, h_over_H, bedrock_thickness,
                 cs_bedrock, vv_bedrock, density_bedrock,
                 cs_overlying, vv_overlying, density_overlying,
                 mesh_size, cae_name=None, logger=None):
    """创建二维平面应变模型：几何、材料、截面、装配、网格（不含分析步）"""  # 说明函数用途
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    model_name = 'Model-1'  # 设置基础模型名称

    H = H_minus_h / (1.0 - h_over_H)  # 根据高度比反算总高度
    h = H - H_minus_h  # 计算下部高度
    H_lower = bedrock_thickness + h  # 计算坡脚地表高度
    H_upper = bedrock_thickness + H  # 计算坡顶地表高度
    w_slope = H_minus_h / math.tan(math.radians(i))  # 计算坡面水平长度
    left_flat = 1000.0  # 设置左侧平台长度为固定值

    right_flat = total_L - left_flat - w_slope  # 计算右侧平台长度
    if right_flat <= 0:  # 检查右平台是否有效
        raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                         (total_L, left_flat, w_slope))  # 抛出几何错误

    if cae_name:  # 判断是否需要保存 cae 文件
        mdb.saveAs(pathName=cae_name)  # 另存为新的工程文件
        log_step(logger, '工程文件保存为 %s', cae_name)  # 记录保存日志
    model = mdb.Model(name=model_name)  # 创建基础模型
    log_step(logger, '%s 基础模型开始创建', model_name)  # 记录模型创建日志

    # 创建二维坡地 Part
    part_name = _next_available_name('Part', model.parts)  # 生成零件名称
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(total_L, H_upper) * 2)  # 创建轮廓草图
    s.Line(point1=(0.0, 0.0),                   point2=(total_L, 0.0))                 # 绘制底边
    s.Line(point1=(total_L, 0.0),               point2=(total_L, H_lower))             # 绘制右边界
    s.Line(point1=(total_L, H_lower),           point2=(left_flat + w_slope, H_lower)) # 绘制右平台地表
    s.Line(point1=(left_flat + w_slope, H_lower),  point2=(left_flat, H_upper))        # 绘制斜坡段
    s.Line(point1=(left_flat, H_upper),            point2=(0.0, H_upper))              # 绘制左平台地表
    s.Line(point1=(0.0, H_upper),               point2=(0.0, 0.0))                     # 绘制左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)  # 创建二维可变形零件
    part.BaseShell(sketch=s)  # 由草图生成壳体基体
    del model.sketches['__profile__']  # 删除临时草图
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)  # 记录零件创建日志

    EE_bedrock = _compute_elastic_modulus_from_wave_speed(cs_bedrock, vv_bedrock, density_bedrock)  # 计算基岩弹性模量
    EE_overlying = _compute_elastic_modulus_from_wave_speed(cs_overlying, vv_overlying, density_overlying)  # 计算覆盖层弹性模量

    mat_bedrock_name = _next_available_name('Material-Bedrock', model.materials)  # 生成基岩材料名
    mat_bedrock = model.Material(name=mat_bedrock_name)  # 创建基岩材料
    mat_bedrock.Elastic(table=((EE_bedrock, vv_bedrock),))  # 定义基岩弹性参数
    mat_bedrock.Density(table=((density_bedrock,),))  # 定义基岩密度

    mat_overlying_name = _next_available_name('Material-Overlying', model.materials)  # 生成覆盖层材料名
    mat_overlying = model.Material(name=mat_overlying_name)  # 创建覆盖层材料
    mat_overlying.Elastic(table=((EE_overlying, vv_overlying),))  # 定义覆盖层弹性参数
    mat_overlying.Density(table=((density_overlying,),))  # 定义覆盖层密度

    sec_bedrock_name = _next_available_name('Section-Bedrock', model.sections)  # 生成基岩截面名
    model.HomogeneousSolidSection(name=sec_bedrock_name, material=mat_bedrock_name, thickness=1.0)  # 创建基岩截面

    sec_overlying_name = _next_available_name('Section-Overlying', model.sections)  # 生成覆盖层截面名
    model.HomogeneousSolidSection(name=sec_overlying_name, material=mat_overlying_name, thickness=1.0)  # 创建覆盖层截面

    # 装配
    assembly = model.rootAssembly  # 获取装配体对象
    assembly.DatumCsysByDefault(CARTESIAN)  # 设置默认笛卡尔坐标系
    inst_name = _next_available_name(part_name, assembly.instances)  # 生成实例名称
    assembly.Instance(name=inst_name, part=part, dependent=ON)  # 创建零件实例

    # ============ 切分面以划分网格与材料区域 ============
    # 1. 垂直切分（ crest & toe ）
    part_faces = part.faces  # 获取当前面集合
    partition_sketch = model.ConstrainedSketch(name='__vert_partition__', sheetSize=max(total_L, H_upper) * 2)  # 创建垂直切分草图
    partition_sketch.Line(point1=(left_flat, 0.0), point2=(left_flat, H_upper))  # 绘制左平台竖向切线
    partition_sketch.Line(point1=(left_flat + w_slope, 0.0), point2=(left_flat + w_slope, H_lower))  # 绘制坡脚竖向切线
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)  # 按草图切分面
    del model.sketches['__vert_partition__']  # 删除临时切分草图
    log_step(logger, '%s 几何垂直切分完成', model_name)  # 记录切分完成日志

    # 2. 水平切分基岩界面 (y = bedrock_thickness)
    part_faces = part.faces  # 获取当前面集合
    partition_sketch = model.ConstrainedSketch(name='__bedrock_partition__', sheetSize=max(total_L, H_upper) * 2)  # 创建水平切分草图
    partition_sketch.Line(point1=(0.0, bedrock_thickness), point2=(total_L, bedrock_thickness))  # 绘制基岩界面
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)  # 按基岩界面切分面
    del model.sketches['__bedrock_partition__']  # 删除临时切分草图
    log_step(logger, '%s 基岩水平面切分完成', model_name)  # 记录切分完成日志

    # 设置网格控制：四边形 + 结构化
    pickedRegions = part.faces  # 选取全部面作为网格区域
    part.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)  # 设置四边形结构化网格
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)  # 设置全局网格尺寸
    elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)  # 定义平面应变四节点单元
    part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1,))  # 分配单元类型
    part.generateMesh()  # 生成网格
    log_step(logger, '%s 已生成网格: CPE4 单元，尺寸=%.2f', model_name, mesh_size)  # 记录网格生成日志

    # ============ 按质心坐标分配截面 ============
    sec_assignments = {  # 初始化截面分配容器
        'bedrock': [],  # 保存基岩面
        'overlying': []  # 保存覆盖层面
    }  # 结束截面分配容器

    for face in part.faces:  # 遍历所有面
        centroid = face.getCentroid()  # 获取面质心
        yc = centroid[1] if len(centroid) >= 2 else centroid[0][1]  # 读取质心纵坐标

        if yc < bedrock_thickness:  # 判断是否位于基岩层
            sec_assignments['bedrock'].append(face)  # 归入基岩截面
        else:  # 其余部分归入覆盖层
            sec_assignments['overlying'].append(face)  # 归入覆盖层截面

    def _to_face_sequence(face_list):  # 定义面序列转换函数
        face_seq = part.faces[0:0]  # 创建空面序列
        for face in face_list:  # 遍历面列表
            face_seq = face_seq + part.faces[face.index:face.index + 1]  # 逐个拼接面对象
        return face_seq  # 返回面序列

    if sec_assignments['bedrock']:  # 判断是否存在基岩面
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['bedrock'])),  # 为基岩分配截面
                               sectionName=sec_bedrock_name, offset=0.0, offsetType=MIDDLE_SURFACE,  # 指定基岩截面参数
                               offsetField='', thicknessAssignment=FROM_SECTION)  # 结束基岩截面分配
    if sec_assignments['overlying']:  # 判断是否存在覆盖层面
        part.SectionAssignment(region=Region(faces=_to_face_sequence(sec_assignments['overlying'])),  # 为覆盖层分配截面
                               sectionName=sec_overlying_name, offset=0.0, offsetType=MIDDLE_SURFACE,  # 指定覆盖层截面参数
                               offsetField='', thicknessAssignment=FROM_SECTION)  # 结束覆盖层截面分配
    log_step(logger, '%s 截面属性分配完成: Bedrock=%d, Overlying=%d',  # 记录截面分配日志
             model_name, len(sec_assignments['bedrock']), len(sec_assignments['overlying']))  # 输出各区域面数

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
                      mesh_size, logger=None):
    """创建二维平面应变平坦自由场模型：矩形几何、材料、截面、装配与网格。"""  # 说明函数用途
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    model_name = 'Model-2'  # 设置平坦自由场模型名称

    model = mdb.Model(name=model_name)  # 创建平坦自由场模型
    log_step(logger, '%s 基础模型开始创建（平坦自由场）', model_name)  # 记录模型创建日志

    part_name = _next_available_name('Part', model.parts)  # 生成零件名称
    sketch = model.ConstrainedSketch(name='__flat_profile__', sheetSize=max(total_L, H_flat) * 2)  # 创建矩形草图
    sketch.Line(point1=(0.0, 0.0), point2=(total_L, 0.0))  # 绘制底边
    sketch.Line(point1=(total_L, 0.0), point2=(total_L, H_flat))  # 绘制右边界
    sketch.Line(point1=(total_L, H_flat), point2=(0.0, H_flat))  # 绘制顶边
    sketch.Line(point1=(0.0, H_flat), point2=(0.0, 0.0))  # 绘制左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)  # 创建二维可变形零件
    part.BaseShell(sketch=sketch)  # 由草图生成壳体基体
    del model.sketches['__flat_profile__']  # 删除临时草图
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)  # 记录零件创建日志

    EE_bedrock = _compute_elastic_modulus_from_wave_speed(cs_bedrock, vv_bedrock, density_bedrock)  # 计算基岩弹性模量
    EE_overlying = _compute_elastic_modulus_from_wave_speed(cs_overlying, vv_overlying, density_overlying)  # 计算覆盖层弹性模量

    mat_bedrock_name = _next_available_name('Material-Bedrock', model.materials)  # 生成基岩材料名
    mat_bedrock = model.Material(name=mat_bedrock_name)  # 创建基岩材料
    mat_bedrock.Elastic(table=((EE_bedrock, vv_bedrock),))  # 定义基岩弹性参数
    mat_bedrock.Density(table=((density_bedrock,),))  # 定义基岩密度

    mat_overlying_name = _next_available_name('Material-Overlying', model.materials)  # 生成覆盖层材料名
    mat_overlying = model.Material(name=mat_overlying_name)  # 创建覆盖层材料
    mat_overlying.Elastic(table=((EE_overlying, vv_overlying),))  # 定义覆盖层弹性参数
    mat_overlying.Density(table=((density_overlying,),))  # 定义覆盖层密度

    sec_bedrock_name = _next_available_name('Section-Bedrock', model.sections)  # 生成基岩截面名
    model.HomogeneousSolidSection(name=sec_bedrock_name, material=mat_bedrock_name, thickness=1.0)  # 创建基岩截面

    sec_overlying_name = _next_available_name('Section-Overlying', model.sections)  # 生成覆盖层截面名
    model.HomogeneousSolidSection(name=sec_overlying_name, material=mat_overlying_name, thickness=1.0)  # 创建覆盖层截面

    # 装配
    assembly = model.rootAssembly  # 获取装配体对象
    assembly.DatumCsysByDefault(CARTESIAN)  # 设置默认笛卡尔坐标系
    inst_name = _next_available_name(part_name, assembly.instances)  # 生成实例名称
    assembly.Instance(name=inst_name, part=part, dependent=ON)  # 创建零件实例

    # ============ 水平切分面 ============
    part_faces = part.faces  # 获取当前面集合
    partition_sketch = model.ConstrainedSketch(name='__flat_bedrock_partition__', sheetSize=max(total_L, H_flat) * 2)  # 创建基岩界面草图
    partition_sketch.Line(point1=(total_L, bedrock_thickness), point2=(0.0, bedrock_thickness))  # 绘制基岩界面
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)  # 按界面切分面
    del model.sketches['__flat_bedrock_partition__']  # 删除临时草图
    log_step(logger, '%s 平坦自由场网格前切割完成', model_name)  # 记录切分日志

    picked_regions = part.faces  # 选取全部面作为网格区域
    part.setMeshControls(regions=picked_regions, elemShape=QUAD, technique=STRUCTURED)  # 设置结构化四边形网格
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)  # 设置网格种子尺寸
    elem_type = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)  # 定义单元类型
    part.setElementType(regions=(picked_regions,), elemTypes=(elem_type,))  # 分配单元类型
    part.generateMesh()  # 生成网格
    log_step(logger, '%s 平坦模型网格已生成: 尺寸=%.2f', model_name, mesh_size)  # 记录网格日志

    # ============ 截面分配 ============
    sec_assignments = {
        'bedrock': [],
        'overlying': []
    }

    for face in part.faces:
        centroid = face.getCentroid()
        yc = centroid[1] if len(centroid) >= 2 else centroid[0][1]

        if yc < bedrock_thickness:
            sec_assignments['bedrock'].append(face)
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


# ============================================================
#  双层自由场核心实现：斜入射 SV 波的直接多波 Zoeppritz 界面解
#  （基岩入射 SV + 界面反/透射 + 覆盖层自由面多次混响，含 SV<->P 转换）
#  说明：v5 曾保留一个未被调用的 Thomson-Haskell 传播矩阵 _layer_propagator，
#        其内部组装重复且与本实现无关，v6 已删除以避免误导。
# ============================================================

def _propagator_matrix_freefield(omega, p, mat_bedrock, mat_overlying,
                                  h_bedrock, h_overlying, y_target, y_bottom):
    """
    用传播矩阵法计算任意深度 y_target 处的频域自由场响应。

    模型分层（从下到上，y 坐标递增）：
      [y_bottom, y_bottom+h_bedrock)    → 基岩（入射 SV 波来自下方无限半空间）
      [y_bottom+h_bedrock, y_top]       → 覆盖层（顶部为自由面）

    y_bottom : 模型底边 y 坐标 (m)
    h_bedrock: 基岩层厚度 (m)
    h_overlying: 覆盖层厚度（当前边界位置对应）(m)
    y_target : 目标节点 y 坐标 (m)
    p        : 水平慢度 (s/m)（由 Snell 定律 = sin(alpha1)/cs_bedrock）
    omega    : 角频率 (rad/s)

    返回: dict，包含 'ux'(复数), 'uy', 'sxx', 'syy', 'sxy'，
          每个值代表该深度处单位入射 SV 幅值对应的频域响应
    """
    cs1 = mat_bedrock['cs']  # 基岩剪切波速
    cp1 = mat_bedrock['cp']  # 基岩纵波波速
    cs2 = mat_overlying['cs']  # 覆盖层剪切波速
    cp2 = mat_overlying['cp']  # 覆盖层纵波波速
    GG1 = mat_bedrock['GG']  # 基岩剪切模量
    lam1 = mat_bedrock['lam']  # 基岩拉梅常数
    GG2 = mat_overlying['GG']  # 覆盖层剪切模量
    lam2 = mat_overlying['lam']  # 覆盖层拉梅常数

    y_intf = y_bottom + h_bedrock  # 基岩/覆盖层界面 y 坐标
    y_top = y_intf + h_overlying  # 自由面 y 坐标

    # ---- 计算各层的垂直慢度 ----
    def _qval(c, p_val):  # 定义垂直慢度计算函数
        """计算垂直慢度 q = sqrt(1/c^2 - p^2)"""  # 说明函数用途
        val = (1.0 / c) ** 2 - p_val ** 2  # 计算慢度平方
        if val >= 0:  # 判断是否为实数根
            return complex(math.sqrt(val), 0)  # 返回实数垂直慢度
        else:  # 处理倏逝波情况
            return complex(0, math.sqrt(-val))  # 返回纯虚数垂直慢度

    qs1 = _qval(cs1, p)  # 基岩 SV 垂直慢度
    qp1 = _qval(cp1, p)  # 基岩 P 垂直慢度
    qs2 = _qval(cs2, p)  # 覆盖层 SV 垂直慢度
    qp2 = _qval(cp2, p)  # 覆盖层 P 垂直慢度

    # ---- 统一使用 e^{i*omega*(p*x - q*y)} 约定 ----
    # 上行波: e^{+i*omega*q*y}（朝 +y 方向传播）
    # 下行波: e^{-i*omega*q*y}（朝 -y 方向传播）

    def _phase(q_val, dy):  # 定义相位因子计算函数
        """计算从参考面传播 dy 距离后的相位因子 e^{i*omega*q*dy}"""  # 说明函数用途
        return np.exp(1j * omega * q_val * dy)  # 返回相位因子

    # ---- 构建界面约束方程，求各波幅值 ----
    # 未知量（共 6 个，基岩层内反射 + 界面透射 + 覆盖层内部振幅）：
    #   基岩侧（入射和反射）：
    #     a_sv1_dn  : 入射 SV 下行（归一化为 1.0）
    #     a_sv1_up  : 反射 SV 上行（从界面反射）—— Rss
    #     a_p1_up   : 反射 P 上行 —— Rsp
    #   覆盖层侧（向上透射，然后在自由面反射）：
    #     a_sv2_up  : 透射 SV 上行 —— Tss
    #     a_sv2_dn  : 覆盖层内 SV 下行（自由面 SV 反射）
    #     a_p2_up   : 透射 P 上行 —— Tsp
    #     a_p2_dn   : 覆盖层内 P 下行（自由面 P 反射）
    #
    # 传播矩阵自动把覆盖层的多次混响全部编码到界面处的幅值里。
    # 做法：
    #   1) 自由面条件（y=y_top）: sigma_yy = tau_xy = 0
    #      给出 a_sv2_dn、a_p2_dn 用 a_sv2_up、a_p2_up 表示
    #   2) 界面连续条件（y=y_intf）: ux,uy,sigma_yy,tau_xy 连续
    #      给出 4 个方程，未知量 [Rss, Rsp, Tss, Tsp]

    # 位移和应力系数（单位幅值平面波）
    # 约定：将 SV 波参数化为位移的 SV 分量（沿传播方向垂直的偏振方向）
    # SV 波位移方向（上行）: ux = cos(alpha), uy = -sin(alpha)（ -> 相对 x 轴 )
    # 但用慢度形式: ux_sv_up = -qs1/abs(qs1+1e-100), uy_sv_up = -p  (取绝对值归一化)
    # 为保持与 v3/v4 一致，以 "单位速度幅值" 约定

    # ── 基岩层 ──
    # SV 上行（幅值 1.0）在界面 y_intf 处的位移和应力
    #   ux = cos(alpha1)  uy = sin(alpha1) （SV 上行）
    #   注: q = qs1（实部）对应 alpha1 = arcsin(p*cs1)
    def _sv_disp_stress(q_sv, q_p_dummy, GG, lam, cs, cp, p_val, direction, wave_type):  # 定义位移和应力系数函数
        """
        计算单位速度幅值平面波的位移方向和应力系数（频域，每单位速度幅值）。
        direction: 'up' 或 'down'
        wave_type: 'SV' 或 'P'
        返回 (ux, uy, sigma_xx, sigma_yy, tau_xy)，均为每单位速度幅值的系数
        """
        sgn = 1.0 if direction == 'up' else -1.0  # 上行取正，下行取负
        if wave_type == 'SV':  # 处理 SV 波
            q = q_sv  # 选用 SV 垂直慢度
            # 位移方向（SV 偏振方向，垂直于传播方向）
            ux_d = q / (1.0 / cs)  # SV 水平位移分量（归一化到速度幅值）
            uy_d = -sgn * p_val / (1.0 / cs)  # SV 垂直位移分量
        else:  # 处理 P 波
            q = q_p_dummy  # 选用 P 垂直慢度
            ux_d = p_val / (1.0 / cp)  # P 水平位移分量
            uy_d = sgn * q / (1.0 / cp)  # P 垂直位移分量
        # 应力系数（以速度幅值为参考，sigma = C * p * vel）
        sig_xx = -(lam * (p_val * ux_d + sgn * q * uy_d) + 2.0 * GG * p_val * ux_d)  # σ_xx 系数
        sig_yy = -(lam * (p_val * ux_d + sgn * q * uy_d) + 2.0 * GG * sgn * q * uy_d)  # σ_yy 系数
        tau_xy = -GG * (sgn * q * ux_d + p_val * uy_d)  # τ_xy 系数
        return ux_d, uy_d, sig_xx, sig_yy, tau_xy  # 返回系数

    # 基岩侧各波在 y=y_intf 处的系数（相位均取 0）
    ux_sv1u, uy_sv1u, sxx_sv1u, syy_sv1u, sxy_sv1u = _sv_disp_stress(qs1, qp1, GG1, lam1, cs1, cp1, p, 'up', 'SV')  # 入射 SV 上行
    ux_sv1d, uy_sv1d, sxx_sv1d, syy_sv1d, sxy_sv1d = _sv_disp_stress(qs1, qp1, GG1, lam1, cs1, cp1, p, 'down', 'SV')  # 反射 SV 下行
    ux_p1d, uy_p1d, sxx_p1d, syy_p1d, sxy_p1d = _sv_disp_stress(qs1, qp1, GG1, lam1, cs1, cp1, p, 'down', 'P')  # 反射 P 下行

    # 覆盖层侧各波在 y=y_intf 处的系数（相位取 0，均向上传播至自由面）
    ux_sv2u, uy_sv2u, sxx_sv2u, syy_sv2u, sxy_sv2u = _sv_disp_stress(qs2, qp2, GG2, lam2, cs2, cp2, p, 'up', 'SV')  # 透射 SV 上行
    ux_p2u, uy_p2u, sxx_p2u, syy_p2u, sxy_p2u = _sv_disp_stress(qs2, qp2, GG2, lam2, cs2, cp2, p, 'up', 'P')  # 透射 P 上行

    # 自由面条件：sigma_yy(y_top) = 0, tau_xy(y_top) = 0
    # 在 y_top 处，SV2上行波和 P2上行波都已到达自由面
    # 反射后产生 SV2下行和 P2下行
    # 自由面边界给出 [a_sv2_dn, a_p2_dn] 的表达式：
    # sigma_yy = 0: a_sv2_up*syy_sv2u + a_sv2_dn*syy_sv2d + a_p2_up*syy_p2u + a_p2_dn*syy_p2d = 0
    # tau_xy   = 0: a_sv2_up*sxy_sv2u + a_sv2_dn*sxy_sv2d + a_p2_up*sxy_p2u + a_p2_dn*sxy_p2d = 0
    # 其中 syy_sv2d = syy_sv2u（syy 不随方向改变符号对于 SV），sxy_sv2d = -sxy_sv2u（tau_xy 改变符号）
    # 实际上更精确地做法：
    # 对覆盖层施加 2×2 自由面 Zoeppritz，一次性给出从 (Tss,Tsp) 到 (a_sv2_dn,a_p2_dn)

    ux_sv2d, uy_sv2d, sxx_sv2d, syy_sv2d, sxy_sv2d = _sv_disp_stress(qs2, qp2, GG2, lam2, cs2, cp2, p, 'down', 'SV')  # 覆盖层 SV 下行
    ux_p2d, uy_p2d, sxx_p2d, syy_p2d, sxy_p2d = _sv_disp_stress(qs2, qp2, GG2, lam2, cs2, cp2, p, 'down', 'P')  # 覆盖层 P 下行

    # 相位因子（从 y_intf 传播到 y_top 的上行波，再反射回来）
    phi_s2_up = _phase(qs2, h_overlying)  # SV2 从界面到自由面的相位
    phi_p2_up = _phase(qp2, h_overlying)  # P2 从界面到自由面的相位

    # 自由面方程（2×2）：
    # [syy_sv2d*phi_s2  syy_p2d*phi_p2] [a_sv2_dn_ref] = -[syy_sv2u*phi_s2_up  syy_p2u*phi_p2_up] [Tss]
    # [sxy_sv2d*phi_s2  sxy_p2d*phi_p2]                   [sxy_sv2u*phi_s2_up  sxy_p2u*phi_p2_up] [Tsp]
    # （phi_s2 = conj(phi_s2_up) 因为下行波的相位 = -上行波的相位）

    phi_s2_dn = np.conj(phi_s2_up)  # SV2 下行相位（从自由面回到界面）
    phi_p2_dn = np.conj(phi_p2_up)  # P2 下行相位

    # 自由面 2×2 反射矩阵 F_surf：
    # [a_sv2_dn_at_intf] = F_surf * [a_sv2_up_at_intf]
    # F_surf = -inv([syy_sv2d*phi_dn  syy_p2d*phi_dn; ...]) * [syy_sv2u*phi_up  ...]
    A_surf = np.array([
        [syy_sv2d * phi_s2_dn, syy_p2d * phi_p2_dn],  # sigma_yy = 0 方程的已知项
        [sxy_sv2d * phi_s2_dn, sxy_p2d * phi_p2_dn],  # tau_xy = 0 方程的已知项
    ], dtype=complex)  # 组装自由面方程左端矩阵

    B_surf_sv = np.array([
        -syy_sv2u * phi_s2_up,  # SV 上行波对 sigma_yy 的贡献
        -sxy_sv2u * phi_s2_up,  # SV 上行波对 tau_xy 的贡献
    ], dtype=complex)  # 组装 SV 贡献向量

    B_surf_p = np.array([
        -syy_p2u * phi_p2_up,  # P 上行波对 sigma_yy 的贡献
        -sxy_p2u * phi_p2_up,  # P 上行波对 tau_xy 的贡献
    ], dtype=complex)  # 组装 P 贡献向量

    det_Asurf = A_surf[0, 0] * A_surf[1, 1] - A_surf[0, 1] * A_surf[1, 0]  # 计算 2×2 行列式
    if abs(det_Asurf) < 1e-30:  # 检查行列式是否退化
        # 退化情况（几乎不会发生）：直接返回零响应
        return {'ux': 0.0, 'uy': 0.0, 'sxx': 0.0, 'syy': 0.0, 'sxy': 0.0}  # 返回零响应

    inv_Asurf = np.array([  # 计算 2×2 逆矩阵
        [A_surf[1, 1], -A_surf[0, 1]],  # 逆矩阵第一行
        [-A_surf[1, 0], A_surf[0, 0]],  # 逆矩阵第二行
    ], dtype=complex) / det_Asurf  # 除以行列式

    # F_surf 将 [Tss, Tsp] 映射到 [a_sv2_dn_at_intf, a_p2_dn_at_intf]
    # F_sv 列：对应 Tss 的情况（Tsp=0）
    F_sv = inv_Asurf.dot(B_surf_sv)  # 自由面 SV 反射映射
    F_p = inv_Asurf.dot(B_surf_p)  # 自由面 P 反射映射
    # 即: a_sv2_dn_at_intf = F_sv[0]*Tss + F_p[0]*Tsp
    #     a_p2_dn_at_intf  = F_sv[1]*Tss + F_p[1]*Tsp

    # ---- 界面连续条件（4×4）----
    # 基岩侧（以 y_intf 为参考面）：
    #   入射 SV 上行（幅值 1.0）+ 反射 SV 下行（幅值 Rss）+ 反射 P 下行（幅值 Rsp）
    # 覆盖层侧（以 y_intf 为参考面）：
    #   透射 SV 上行（幅值 Tss）+ 透射 P 上行（幅值 Tsp）
    #   + 覆盖层 SV 下行（幅值 F_sv[0]*Tss + F_p[0]*Tsp）
    #   + 覆盖层 P 下行（幅值 F_sv[1]*Tss + F_p[1]*Tsp）
    #
    # 注意：基岩侧入射 SV 上行波在 y_intf 处的相位取决于它从底部出发到界面
    # 若设底部 y=y_bottom 为参考点（入射幅值 1.0 在此处），则界面处的相位为
    # phi_sv1_up_to_intf = e^{i*omega*qs1*h_bedrock}

    phi_sv1_up = _phase(qs1, h_bedrock)  # 基岩 SV 上行从底部到界面的相位
    # 反射波从界面出发到底部再到本节点 y_target（对于 y_target < y_intf 即基岩节点）

    # 以界面 y_intf 为参考面构建方程
    # 覆盖层侧在界面处的有效位移/应力（叠加透射 + 覆盖层内反射下行）
    def _cov_at_intf(Tss_dummy, Tsp_dummy):  # 定义覆盖层界面处叠加计算函数
        """计算覆盖层侧在界面处的位移/应力（透射 + 自由面反射）"""  # 说明函数用途
        a_sv2_dn = F_sv[0] * Tss_dummy + F_p[0] * Tsp_dummy  # 覆盖层内 SV 下行幅值
        a_p2_dn = F_sv[1] * Tss_dummy + F_p[1] * Tsp_dummy  # 覆盖层内 P 下行幅值
        ux = (Tss_dummy * ux_sv2u + Tsp_dummy * ux_p2u +  # 透射波贡献
              a_sv2_dn * ux_sv2d + a_p2_dn * ux_p2d)  # 覆盖层内反射波贡献
        uy = (Tss_dummy * uy_sv2u + Tsp_dummy * uy_p2u +  # 透射波贡献
              a_sv2_dn * uy_sv2d + a_p2_dn * uy_p2d)  # 覆盖层内反射波贡献
        syy = (Tss_dummy * syy_sv2u + Tsp_dummy * syy_p2u +  # 透射波应力贡献
               a_sv2_dn * syy_sv2d + a_p2_dn * syy_p2d)  # 覆盖层内反射波应力贡献
        sxy = (Tss_dummy * sxy_sv2u + Tsp_dummy * sxy_p2u +  # 透射波应力贡献
               a_sv2_dn * sxy_sv2d + a_p2_dn * sxy_p2d)  # 覆盖层内反射波应力贡献
        return ux, uy, syy, sxy  # 返回界面处位移和应力

    # 从 Tss=1,Tsp=0 时覆盖层侧的贡献（用于构建矩阵列）
    cov_ux_sv, cov_uy_sv, cov_syy_sv, cov_sxy_sv = _cov_at_intf(1.0, 0.0)  # Tss 列贡献
    cov_ux_p, cov_uy_p, cov_syy_p, cov_sxy_p = _cov_at_intf(0.0, 1.0)  # Tsp 列贡献

    # 4×4 方程组: A_mat * [Rss, Rsp, Tss, Tsp]^T = B_vec
    # 行顺序: ux, uy, tau_xy, sigma_yy
    A_mat = np.zeros((4, 4), dtype=complex)  # 初始化界面方程矩阵

    # 基岩侧（列 0: Rss，列 1: Rsp）
    A_mat[0, 0] = ux_sv1d  # Rss 对 ux 的贡献（基岩 SV 下行）
    A_mat[1, 0] = uy_sv1d  # Rss 对 uy 的贡献
    A_mat[2, 0] = sxy_sv1d  # Rss 对 tau_xy 的贡献
    A_mat[3, 0] = syy_sv1d  # Rss 对 sigma_yy 的贡献

    A_mat[0, 1] = ux_p1d  # Rsp 对 ux 的贡献（基岩 P 下行）
    A_mat[1, 1] = uy_p1d  # Rsp 对 uy 的贡献
    A_mat[2, 1] = sxy_p1d  # Rsp 对 tau_xy 的贡献
    A_mat[3, 1] = syy_p1d  # Rsp 对 sigma_yy 的贡献

    # 覆盖层侧（列 2: Tss，列 3: Tsp，取负号因为移到右端）
    A_mat[0, 2] = -cov_ux_sv  # Tss 对 ux 的贡献（覆盖层侧，取负移至左端）
    A_mat[1, 2] = -cov_uy_sv  # Tss 对 uy 的贡献
    A_mat[2, 2] = -cov_sxy_sv  # Tss 对 tau_xy 的贡献
    A_mat[3, 2] = -cov_syy_sv  # Tss 对 sigma_yy 的贡献

    A_mat[0, 3] = -cov_ux_p  # Tsp 对 ux 的贡献（覆盖层侧）
    A_mat[1, 3] = -cov_uy_p  # Tsp 对 uy 的贡献
    A_mat[2, 3] = -cov_sxy_p  # Tsp 对 tau_xy 的贡献
    A_mat[3, 3] = -cov_syy_p  # Tsp 对 sigma_yy 的贡献

    # 右端项：入射 SV 上行波（幅值 1.0，以 y_intf 为参考面）
    B_vec = np.array([
        -ux_sv1u,  # ux 连续：入射贡献移至右端
        -uy_sv1u,  # uy 连续
        -sxy_sv1u,  # tau_xy 连续
        -syy_sv1u,  # sigma_yy 连续
    ], dtype=complex)  # 组装右端向量

    # 求解线性方程组
    try:  # 尝试求解
        X = np.linalg.solve(A_mat, B_vec)  # 求解 4×4 方程组
    except np.linalg.LinAlgError:  # 处理奇异矩阵
        return {'ux': 0.0, 'uy': 0.0, 'sxx': 0.0, 'syy': 0.0, 'sxy': 0.0}  # 返回零响应

    Rss, Rsp, Tss, Tsp = X[0], X[1], X[2], X[3]  # 提取四个界面系数

    # 覆盖层内的下行幅值（在界面处）
    a_sv2_dn_intf = F_sv[0] * Tss + F_p[0] * Tsp  # 覆盖层内 SV 下行幅值（界面处）
    a_p2_dn_intf = F_sv[1] * Tss + F_p[1] * Tsp  # 覆盖层内 P 下行幅值（界面处）

    # ---- 计算目标节点 y_target 处的响应 ----
    if y_target <= y_intf + 1e-4:  # 判断节点是否在基岩层（含界面）
        # 基岩层节点：入射 SV 上行 + 反射 SV 下行 + 反射 P 下行
        dy_from_bottom = y_target - y_bottom  # 节点相对底部的高度
        phi_sv_inc = _phase(qs1, dy_from_bottom)  # 入射 SV 在节点处的相位
        phi_sv_ref = _phase(qs1, 2.0 * h_bedrock - dy_from_bottom)  # 反射 SV 在节点处的相位（从界面返回）
        phi_p_ref = (_phase(qp1, h_bedrock) *  # P 波到界面的相位
                     _phase(qp1, h_bedrock - dy_from_bottom))  # P 从界面返回到节点的相位

        ux_tot = (1.0 * ux_sv1u * phi_sv_inc +  # 入射 SV 贡献
                  Rss * ux_sv1d * phi_sv_ref +  # 反射 SV 贡献
                  Rsp * ux_p1d * phi_p_ref)  # 反射 P 贡献
        uy_tot = (1.0 * uy_sv1u * phi_sv_inc +  # 入射 SV 贡献
                  Rss * uy_sv1d * phi_sv_ref +  # 反射 SV 贡献
                  Rsp * uy_p1d * phi_p_ref)  # 反射 P 贡献
        sxx_tot = (1.0 * sxx_sv1u * phi_sv_inc +  # 入射 SV 应力贡献
                   Rss * sxx_sv1d * phi_sv_ref +  # 反射 SV 应力贡献
                   Rsp * sxx_p1d * phi_p_ref)  # 反射 P 应力贡献
        syy_tot = (1.0 * syy_sv1u * phi_sv_inc +  # 入射 SV 应力贡献
                   Rss * syy_sv1d * phi_sv_ref +  # 反射 SV 应力贡献
                   Rsp * syy_p1d * phi_p_ref)  # 反射 P 应力贡献
        sxy_tot = (1.0 * sxy_sv1u * phi_sv_inc +  # 入射 SV 应力贡献
                   Rss * sxy_sv1d * phi_sv_ref +  # 反射 SV 应力贡献
                   Rsp * sxy_p1d * phi_p_ref)  # 反射 P 应力贡献

    else:  # 处理覆盖层节点
        # 覆盖层节点：4 个波（透射 SV↑ + 覆盖层 SV↓ + 透射 P↑ + 覆盖层 P↓）
        # 自动包含多次混响（通过 a_sv2_dn, a_p2_dn 的系数）
        dy_from_intf = y_target - y_intf  # 节点在覆盖层中的高度（从界面起算）

        phi_sv2u = _phase(qs2, dy_from_intf)  # SV2 上行在节点处的相位
        phi_sv2d = _phase(qs2, -dy_from_intf)  # SV2 下行在节点处的相位（界面处幅值对应从自由面反射后的到达值）
        phi_p2u = _phase(qp2, dy_from_intf)  # P2 上行在节点处的相位
        phi_p2d = _phase(qp2, -dy_from_intf)  # P2 下行在节点处的相位

        ux_tot = (Tss * ux_sv2u * phi_sv2u +  # 透射 SV 上行贡献
                  a_sv2_dn_intf * ux_sv2d * phi_sv2d +  # 覆盖层 SV 下行贡献
                  Tsp * ux_p2u * phi_p2u +  # 透射 P 上行贡献
                  a_p2_dn_intf * ux_p2d * phi_p2d)  # 覆盖层 P 下行贡献
        uy_tot = (Tss * uy_sv2u * phi_sv2u +  # 透射 SV 上行贡献
                  a_sv2_dn_intf * uy_sv2d * phi_sv2d +  # 覆盖层 SV 下行贡献
                  Tsp * uy_p2u * phi_p2u +  # 透射 P 上行贡献
                  a_p2_dn_intf * uy_p2d * phi_p2d)  # 覆盖层 P 下行贡献
        sxx_tot = (Tss * sxx_sv2u * phi_sv2u +  # 透射 SV 应力贡献
                   a_sv2_dn_intf * sxx_sv2d * phi_sv2d +  # 覆盖层 SV 下行应力贡献
                   Tsp * sxx_p2u * phi_p2u +  # 透射 P 应力贡献
                   a_p2_dn_intf * sxx_p2d * phi_p2d)  # 覆盖层 P 下行应力贡献
        syy_tot = (Tss * syy_sv2u * phi_sv2u +  # 透射 SV 应力贡献
                   a_sv2_dn_intf * syy_sv2d * phi_sv2d +  # 覆盖层 SV 下行应力贡献
                   Tsp * syy_p2u * phi_p2u +  # 透射 P 应力贡献
                   a_p2_dn_intf * syy_p2d * phi_p2d)  # 覆盖层 P 下行应力贡献
        sxy_tot = (Tss * sxy_sv2u * phi_sv2u +  # 透射 SV 应力贡献
                   a_sv2_dn_intf * sxy_sv2d * phi_sv2d +  # 覆盖层 SV 下行应力贡献
                   Tsp * sxy_p2u * phi_p2u +  # 透射 P 应力贡献
                   a_p2_dn_intf * sxy_p2d * phi_p2d)  # 覆盖层 P 下行应力贡献

    return {  # 返回频域响应字典
        'ux': ux_tot,  # x 向位移频域响应
        'uy': uy_tot,  # y 向位移频域响应
        'sxx': sxx_tot,  # σ_xx 频域响应
        'syy': syy_tot,  # σ_yy 频域响应
        'sxy': sxy_tot,  # τ_xy 频域响应
    }  # 结束返回字典


def _compute_freefield_at_node(y_target, x_target, mat_bedrock, mat_overlying,
                                h_bedrock, h_overlying, y_bottom,
                                p_horiz, vel_freq, freq_arr, dt, N_fft):
    """
    用传播矩阵法计算单个节点的自由场位移、速度和应力时程。

    vel_freq  : 输入速度时程（底部入射 SV 幅值）的频谱（复数数组，长度 N_fft//2+1）
    freq_arr  : 对应的频率数组 (Hz)
    p_horiz   : 水平慢度 (s/m)
    h_overlying: 当前边界位置（左/右边界）对应的覆盖层厚度

    返回: dict，包含 'ux', 'uy', 'dotux', 'dotuy', 'sxx', 'syy', 'sxy'
          每个值均为时域数组，长度 N_fft
    """
    N_pos = N_fft // 2 + 1  # 正频率个数（包含直流分量）

    ux_freq = np.zeros(N_pos, dtype=complex)  # 初始化 ux 频谱数组
    uy_freq = np.zeros(N_pos, dtype=complex)  # 初始化 uy 频谱数组
    sxx_freq = np.zeros(N_pos, dtype=complex)  # 初始化 sxx 频谱数组
    syy_freq = np.zeros(N_pos, dtype=complex)  # 初始化 syy 频谱数组
    sxy_freq = np.zeros(N_pos, dtype=complex)  # 初始化 sxy 频谱数组

    for k in range(N_pos):  # 遍历每个频率分量
        f_k = freq_arr[k]  # 当前频率（Hz）
        omega_k = 2.0 * math.pi * f_k  # 计算角频率

        # 水平传播相位延迟（Snell 定律给出 e^{i*omega*p*x}）
        phase_x = np.exp(1j * omega_k * p_horiz * x_target)  # 计算水平传播相位

        if abs(omega_k) < 1e-20:  # 处理直流分量
            # 直流分量（静态）：响应取零（加速度基线处理后无静态位移）
            continue  # 跳过直流分量

        # 调用传播矩阵法求频域传递函数
        resp = _propagator_matrix_freefield(  # 计算频域传播矩阵响应
            omega_k, p_horiz,  # 传入角频率和水平慢度
            mat_bedrock, mat_overlying,  # 传入两层材料参数
            h_bedrock, h_overlying,  # 传入两层厚度
            y_target, y_bottom  # 传入目标节点和底部 y 坐标
        )

        # 乘以输入速度谱（速度幅值）和水平相位（位移 = 速度/i*omega）
        v_k = vel_freq[k] * phase_x  # 当前频率的速度幅值（含水平传播）

        ux_freq[k] = resp['ux'] * v_k / (1j * omega_k)  # 位移 = 速度 / (i*omega)
        uy_freq[k] = resp['uy'] * v_k / (1j * omega_k)  # 位移 y 分量
        sxx_freq[k] = resp['sxx'] * v_k  # 应力 sxx（= 系数 × 速度）
        syy_freq[k] = resp['syy'] * v_k  # 应力 syy
        sxy_freq[k] = resp['sxy'] * v_k  # 应力 sxy

    # 速度频谱（= i*omega * 位移频谱）
    omega_arr = 2.0 * math.pi * freq_arr  # 计算角频率数组
    dotux_freq = 1j * omega_arr * ux_freq  # 速度 x 分量
    dotuy_freq = 1j * omega_arr * uy_freq  # 速度 y 分量

    # IFFT 回时域（取实部）
    ux_t = np.fft.irfft(ux_freq, n=N_fft)  # x 向位移时域
    uy_t = np.fft.irfft(uy_freq, n=N_fft)  # y 向位移时域
    dotux_t = np.fft.irfft(dotux_freq, n=N_fft)  # x 向速度时域
    dotuy_t = np.fft.irfft(dotuy_freq, n=N_fft)  # y 向速度时域
    sxx_t = np.fft.irfft(sxx_freq, n=N_fft)  # σ_xx 时域
    syy_t = np.fft.irfft(syy_freq, n=N_fft)  # σ_yy 时域
    sxy_t = np.fft.irfft(sxy_freq, n=N_fft)  # τ_xy 时域

    return {  # 返回时域结果字典
        'ux': ux_t,  # x 向位移时程
        'uy': uy_t,  # y 向位移时程
        'dotux': dotux_t,  # x 向速度时程
        'dotuy': dotuy_t,  # y 向速度时程
        'sxx': sxx_t,  # σ_xx 应力时程
        'syy': syy_t,  # σ_yy 应力时程
        'sxy': sxy_t,  # τ_xy 应力时程
    }  # 结束返回字典


def VAB_oblique(angle,
                cs_bedrock, vv_bedrock, density_bedrock,
                cs_overlying, vv_overlying, density_overlying,
                bedrock_thickness,
                H_upper, H_lower, left_flat, w_slope,
                model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None):
    """为二维模型施加粘弹性人工边界（弹簧-阻尼器）和地震动斜向输入等效节点力。"""  # 说明函数用途
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    t0 = time.time()  # 记录函数开始时间
    step_name = step_name or DEFAULT_STEP_NAME  # 使用默认分析步名称
    log_step(logger, '%s 模型开始创建人工边界', model_name)  # 记录人工边界开始日志

    assembly = mdb.models[model_name].rootAssembly  # 获取装配体
    assembly.regenerate()  # 重新生成装配体

    model = mdb.models[model_name]  # 获取目标模型
    if part_name not in model.parts:  # 检查零件是否存在
        raise KeyError('%s 中不存在Part: %s' % (model_name, part_name))  # 抛出零件缺失异常
    part = model.parts[part_name]  # 获取零件对象
    if inst_name not in assembly.instances:  # 检查实例是否存在
        raise KeyError('%s 中不存在实例: %s' % (model_name, inst_name))  # 抛出实例缺失异常
    instance = assembly.instances[inst_name]  # 获取实例对象

    missing_boundary_sets = [name for name in BOUNDARY_SET_NAMES if name not in part.sets]  # 检查边界节点集是否齐全
    if missing_boundary_sets:  # 判断是否存在缺失节点集
        raise KeyError('%s 缺少Part边界节点集: %s，请先在 create_model 中创建' %
                       (model_name, '/'.join(missing_boundary_sets)))  # 抛出节点集缺失异常
    log_step(logger, '%s 复用已有Part边界节点集: %s', model_name, '/'.join(BOUNDARY_SET_NAMES))  # 记录节点集复用日志

    def get_instance_nodes_from_part_set(set_name):  # 定义从零件集获取实例节点的辅助函数
        labels = tuple(node.label for node in part.sets[set_name].nodes)  # 提取节点标签
        if not labels:  # 判断节点集是否为空
            raise ValueError('%s Part节点集 %s 为空' % (model_name, set_name))  # 抛出空节点集异常
        return instance.nodes.sequenceFromLabels(labels)  # 按标签获取实例节点序列

    # 材料参数计算
    mat_bedrock = _compute_material_params(cs_bedrock, vv_bedrock, density_bedrock)  # 计算基岩材料参数
    mat_overlying = _compute_material_params(cs_overlying, vv_overlying, density_overlying)  # 计算覆盖层材料参数

    # 获取模型尺寸
    l_nodes = get_instance_nodes_from_part_set('Left_boundary')  # 获取左边界节点
    l_ymax_node = max(l_nodes, key=lambda node: node.coordinates[1])  # 取左边界最高节点
    xmin = l_ymax_node.coordinates[0]  # 记录左边界 x 坐标
    ymax_l = l_ymax_node.coordinates[1]  # 记录左边界最高 y 坐标

    b_nodes = get_instance_nodes_from_part_set('Bottom_boundary')  # 获取底边节点
    ymin = b_nodes[0].coordinates[1]  # 记录底边 y 坐标

    r_nodes = get_instance_nodes_from_part_set('Right_boundary')  # 获取右边界节点
    r_ymax_node = max(r_nodes, key=lambda node: node.coordinates[1])  # 取右边界最高节点
    xmax = r_ymax_node.coordinates[0]  # 记录右边界 x 坐标
    ymax_r = r_ymax_node.coordinates[1]  # 记录右边界最高 y 坐标

    ymax = max(ymax_l, ymax_r)  # 取左右边界最高点中的较大值

    # 计算节点影响长度
    def get_node_influence(nodes, sort_axis='y', ascending=False):  # 定义节点影响长度计算函数
        node_data = np.array([[node.label, node.coordinates[0], node.coordinates[1]] for node in nodes], dtype=float)  # 生成节点数据表
        axis = 1 if sort_axis == 'x' else 2  # 根据排序轴选择坐标列
        node_data = node_data[node_data[:, axis].argsort()]  # 按指定坐标排序
        if not ascending:  # 判断是否需要倒序
            node_data = node_data[::-1]  # 反转排序结果

        n = node_data.shape[0]  # 统计节点数量
        if n == 1:  # 处理单节点情况
            influence = np.array([0.0])  # 单节点影响长度设为零
        else:  # 处理多节点情况
            coord = node_data[:, axis]  # 提取排序坐标
            influence = np.empty(n)  # 创建影响长度数组
            influence[0] = abs(coord[0] - coord[1]) / 2.0  # 计算首节点影响长度
            influence[-1] = abs(coord[-1] - coord[-2]) / 2.0  # 计算末节点影响长度
            if n > 2:  # 处理中间节点
                influence[1:-1] = np.abs(coord[:-2] - coord[2:]) / 2.0  # 计算中间节点影响长度

        node_data = np.hstack((node_data, influence.reshape(-1, 1)))  # 将影响长度拼接到数据表
        return node_data  # 返回节点影响数据

    node_data_l = get_node_influence(l_nodes, sort_axis='y', ascending=False)  # 计算左边界节点影响长度
    node_data_r = get_node_influence(r_nodes, sort_axis='y', ascending=False)  # 计算右边界节点影响长度
    node_data_b = get_node_influence(b_nodes, sort_axis='x', ascending=True)  # 计算底边节点影响长度
    log_step(logger, '%s 节点影响长度已计算', model_name)  # 记录节点影响长度日志

    # 粘弹性人工边界参数（根据节点所在材质层动态赋值）
    def _pick_material_by_node(x_coord, y_coord):  # 定义按节点坐标选择材料的函数
        if y_coord < bedrock_thickness + 1e-4:  # 判断节点是否位于基岩层
            return mat_bedrock  # 返回基岩材料参数
        else:  # 否则认为属于覆盖层
            return mat_overlying  # 返回覆盖层材料参数

    def add_spring_damper(node_data):  # 定义弹簧阻尼参数计算函数
        influence = node_data[:, 3]  # 提取节点影响长度
        kns = np.zeros_like(influence)  # 初始化法向刚度数组
        cns = np.zeros_like(influence)  # 初始化法向阻尼数组
        kts = np.zeros_like(influence)  # 初始化切向刚度数组
        cts = np.zeros_like(influence)  # 初始化切向阻尼数组
        for idx in range(node_data.shape[0]):  # 遍历所有边界节点
            x0 = node_data[idx, 1]  # 读取节点 x 坐标
            y0 = node_data[idx, 2]  # 读取节点 y 坐标
            mat = _pick_material_by_node(x0, y0)  # 根据节点位置选择材料参数
            kn_coeff = mat['GG'] / 2.0 / ymax  # 计算法向刚度系数
            cn_coeff = mat['density'] * mat['cp']  # 计算法向阻尼系数
            kt_coeff = mat['GG'] / 4.0 / ymax  # 计算切向刚度系数
            ct_coeff = mat['density'] * mat['cs']  # 计算切向阻尼系数
            kns[idx] = kn_coeff * influence[idx]  # 计算法向刚度
            cns[idx] = cn_coeff * influence[idx]  # 计算法向阻尼
            kts[idx] = kt_coeff * influence[idx]  # 计算切向刚度
            cts[idx] = ct_coeff * influence[idx]  # 计算切向阻尼
        return np.hstack((node_data,  # 拼接原始节点数据
                           kns.reshape(-1, 1),  # 拼接法向刚度
                           cns.reshape(-1, 1),  # 拼接法向阻尼
                           kts.reshape(-1, 1),  # 拼接切向刚度
                           cts.reshape(-1, 1)))  # 拼接切向阻尼

    node_data_l = add_spring_damper(node_data_l)  # 为左边界分配弹簧阻尼参数
    node_data_r = add_spring_damper(node_data_r)  # 为右边界分配弹簧阻尼参数
    node_data_b = add_spring_damper(node_data_b)  # 为底边分配弹簧阻尼参数
    log_step(logger, '%s 弹簧-阻尼系数已分配到所有边界节点', model_name)  # 记录参数分配日志

    # 添加弹簧阻尼器到模型
    def add_spring_dashpot(node_data, prefix, dof_n, dof_t):  # 定义创建弹簧阻尼器的函数
        for row in node_data:  # 遍历每个边界节点
            node_label = int(row[0])  # 读取节点标签
            kn = row[4]  # 读取法向刚度
            cn = row[5]  # 读取法向阻尼
            kt = row[6]  # 读取切向刚度
            ct = row[7]  # 读取切向阻尼
            node_array = instance.nodes.sequenceFromLabels([node_label])  # 通过标签获取实例节点
            if len(node_array) == 0:  # 判断节点是否存在
                logger.warning('创建弹簧-阻尼器时，实例中不存在节点 %d', node_label)  # 输出警告日志
                continue  # 跳过当前节点
            region = Region(nodes=node_array)  # 构建节点区域
            assembly.engineeringFeatures.SpringDashpotToGround(  # 创建法向弹簧阻尼器
                name='SpringDashpot_{}_{}_normal'.format(prefix, node_label),  # 设置法向元件名称
                region=region, orientation=None, dof=dof_n,  # 设置区域和自由度
                springBehavior=ON, springStiffness=kn,  # 设置弹簧行为和刚度
                dashpotBehavior=ON, dashpotCoefficient=cn)  # 设置阻尼行为和阻尼系数
            assembly.engineeringFeatures.SpringDashpotToGround(  # 创建切向弹簧阻尼器
                name='SpringDashpot_{}_{}_tangent'.format(prefix, node_label),  # 设置切向元件名称
                region=region, orientation=None, dof=dof_t,  # 设置区域和自由度
                springBehavior=ON, springStiffness=kt,  # 设置弹簧行为和刚度
                dashpotBehavior=ON, dashpotCoefficient=ct)  # 设置阻尼行为和阻尼系数

    boundary_dof = {  # 定义各边界对应的法向与切向自由度
        'l': (1, 2),  # 左边界法向为 1（x）、切向为 2（y）
        'r': (1, 2),  # 右边界法向为 1（x）、切向为 2（y）
        'b': (2, 1),  # 底边法向为 2（y）、切向为 1（x）
    }  # 结束自由度映射
    boundary_node_data = {  # 定义各边界对应的节点数据
        'l': node_data_l,  # 左边界节点数据
        'r': node_data_r,  # 右边界节点数据
        'b': node_data_b,  # 底边节点数据
    }  # 结束节点数据映射
    for boundary in BOUNDARY_SEQUENCE:  # 按边界顺序创建弹簧阻尼器
        dof_n, dof_t = boundary_dof[boundary]  # 读取当前边界自由度配置
        add_spring_dashpot(boundary_node_data[boundary], prefix=boundary, dof_n=dof_n, dof_t=dof_t)  # 施加弹簧阻尼器
    log_step(logger, '%s 弹簧-阻尼器创建完成', model_name)  # 记录创建完成日志

    # ============ 入射角处理 ============
    if angle == 0:  # 判断入射角是否为零
        angle = 1e-10  # 用极小角度替代零角度
    else:  # 处理非零角度
        angle = round(angle, 4)  # 保留四位小数
    alpha1 = math.radians(angle)  # 将角度转换为弧度
    p_horiz = math.sin(alpha1) / cs_bedrock  # 计算水平慢度（Snell 定律，所有层共享）
    log_step(logger, '%s 水平慢度 p = %.8f s/m', model_name, p_horiz)  # 记录水平慢度

    # ============ 读取加速度时程并积分 ============
    if not acc_file:  # 判断加速度文件是否为空
        raise ValueError('acc_file 不能为空')  # 抛出参数缺失异常
    ACC = np.loadtxt(acc_file)  # 读取加速度时程
    if ACC.ndim != 2 or ACC.shape[1] < 2 or ACC.shape[0] < 2:  # 检查加速度文件格式
        raise ValueError('加速度文件格式不满足 [time, acceleration]')  # 抛出格式异常
    time_arr = ACC[:, 0]  # 提取时间列
    acc = ACC[:, 1]  # 提取加速度列
    dt = ACC[1, 0] - ACC[0, 0]  # 计算时间步长
    if dt <= 0:  # 检查步长是否有效
        raise ValueError('加速度 dt 必须 > 0')  # 抛出步长异常

    # 积分得到速度时程（梯形积分 + 基线校正，抑制低频漂移）
    acc = acc - np.mean(acc)  # 去除加速度零频偏移，避免积分后速度产生线性漂移
    vel = np.zeros_like(acc)  # 初始化速度数组
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)  # 通过梯形积分计算速度
    _vel_trend = np.polyfit(time_arr, vel, 1)  # 最小二乘拟合速度的线性趋势项
    vel = vel - (_vel_trend[0] * time_arr + _vel_trend[1])  # 扣除线性趋势完成基线校正（位移=速度/(iω) 会放大低频误差）
    log_step(logger, '%s 速度基线校正完成: 去趋势斜率=%.3e', model_name, _vel_trend[0])  # 记录基线校正日志

    N_orig = len(vel)  # 原始速度时程长度

    # 补零到 2 的幂次方，保证 FFT 效率
    N_fft = 1  # 初始化 FFT 长度
    while N_fft < N_orig:  # 寻找最小的 2 的幂次方
        N_fft *= 2  # 倍增 FFT 长度
    N_fft *= 2  # 额外翻倍以避免时域混叠

    vel_padded = np.zeros(N_fft)  # 创建补零后的速度数组
    vel_padded[:N_orig] = vel  # 将原始速度数据填入

    freq_arr = np.fft.rfftfreq(N_fft, d=dt)  # 计算正频率数组（Hz）
    vel_freq = np.fft.rfft(vel_padded)  # 计算速度时程的正频率 FFT

    log_step(logger, '%s FFT 完成: N_orig=%d, N_fft=%d', model_name, N_orig, N_fft)  # 记录 FFT 参数

    # ============ 逐节点计算自由场并组装等效力 ============
    Lx = xmax - xmin  # 计算模型横向跨度

    # 定义各边界面力的外法向（v5 修正）
    # 左边界 x=xmin：外域在左侧，外法向 n = (-1, 0)
    #   面力 tx = σ·n: tx = -σ_xx,  ty = -τ_xy
    # 右边界 x=xmax：外域在右侧，外法向 n = (+1, 0)
    #   面力 tx = σ·n: tx = +σ_xx,  ty = +τ_xy
    # 底边  y=ymin：外域在下方，外法向 n = (0, -1)
    #   面力 tx = σ·n: tx = -τ_xy,  ty = -σ_yy

    field_data = {}  # 初始化等效力缓存字典

    def process_boundary_node(node_row, prefix):  # 定义单节点等效力计算函数
        """用传播矩阵法计算单个边界节点的等效节点力时程，存入 field_data。"""  # 说明函数用途
        node_id = int(node_row[0])  # 读取节点标签
        x0 = node_row[1]  # 读取节点 x 坐标
        y0 = node_row[2]  # 读取节点 y 坐标
        A_inf = node_row[3]  # 读取节点影响长度（面积 = 影响长度 × 单位厚度）
        kn = node_row[4]  # 读取法向刚度
        cn = node_row[5]  # 读取法向阻尼
        kt = node_row[6]  # 读取切向刚度
        ct = node_row[7]  # 读取切向阻尼

        # 确定当前节点对应的覆盖层有效厚度
        if prefix == 'l':  # 左边界
            h_ov_local = max(0.0, ymax_l - bedrock_thickness)  # 左边界覆盖层厚度
        elif prefix == 'r':  # 右边界
            h_ov_local = max(0.0, ymax_r - bedrock_thickness)  # 右边界覆盖层厚度
        else:  # 底边界
            h_ov_local = max(0.0, ymax - bedrock_thickness)  # 底边界覆盖层厚度

        h_br_local = bedrock_thickness  # 基岩层厚度（固定）

        # 调用传播矩阵法计算自由场时程
        ff = _compute_freefield_at_node(  # 计算节点自由场
            y_target=y0,  # 目标节点 y 坐标
            x_target=x0,  # 目标节点 x 坐标（用于水平传播相位）
            mat_bedrock=mat_bedrock,  # 基岩材料参数
            mat_overlying=mat_overlying,  # 覆盖层材料参数
            h_bedrock=h_br_local,  # 基岩层厚度
            h_overlying=h_ov_local,  # 覆盖层有效厚度
            y_bottom=ymin,  # 模型底边 y 坐标
            p_horiz=p_horiz,  # 水平慢度
            vel_freq=vel_freq,  # 输入速度频谱
            freq_arr=freq_arr,  # 频率数组
            dt=dt,  # 时间步长
            N_fft=N_fft  # FFT 长度
        )

        ux = ff['ux'][:N_orig]  # 截取 x 向位移时程（去掉补零部分）
        uy = ff['uy'][:N_orig]  # 截取 y 向位移时程
        dotux = ff['dotux'][:N_orig]  # 截取 x 向速度时程
        dotuy = ff['dotuy'][:N_orig]  # 截取 y 向速度时程
        sxx = ff['sxx'][:N_orig]  # 截取 σ_xx 时程
        syy = ff['syy'][:N_orig]  # 截取 σ_yy 时程
        sxy = ff['sxy'][:N_orig]  # 截取 τ_xy 时程
        t_arr = time_arr[:N_orig]  # 截取时间轴

        # ---- v5 核心修正：面力符号约定（退化点 3）----
        # 等效节点力 = K·u_ff + C·v_ff + A·(σ·n_exterior)
        # 严格按外法向方向计算面力 t = σ·n
        if prefix == 'l':  # 左边界，外法向 n = (-1, 0)
            tx = -sxx  # t_x = σ_xx·(-1) = -σ_xx
            ty = -sxy  # t_y = τ_xy·(-1) = -τ_xy
            # 等效力：x 为法向（弹簧 kn），y 为切向（弹簧 kt）
            fx = kn * ux + cn * dotux + A_inf * tx  # x 向等效力
            fy = kt * uy + ct * dotuy + A_inf * ty  # y 向等效力
        elif prefix == 'r':  # 右边界，外法向 n = (+1, 0)
            tx = +sxx  # t_x = σ_xx·(+1) = +σ_xx
            ty = +sxy  # t_y = τ_xy·(+1) = +τ_xy
            fx = kn * ux + cn * dotux + A_inf * tx  # x 向等效力
            fy = kt * uy + ct * dotuy + A_inf * ty  # y 向等效力
        else:  # 底边界，外法向 n = (0, -1)
            tx = -sxy  # t_x = τ_xy·(0) + σ_xx·(0) = τ_yx·(-1) = -τ_xy（对称）
            ty = -syy  # t_y = σ_yy·(-1) = -σ_yy
            # 底边：x 为切向（弹簧 kt），y 为法向（弹簧 kn）
            fx = kt * ux + ct * dotux + A_inf * tx  # x 向等效力（切向弹簧）
            fy = kn * uy + cn * dotuy + A_inf * ty  # y 向等效力（法向弹簧）

        # 组装时程数组
        fx_arr = np.column_stack((t_arr, fx))  # 组合 x 向力时程
        fy_arr = np.column_stack((t_arr, fy))  # 组合 y 向力时程

        field_data['{}-{}-fx'.format(node_id, prefix)] = fx_arr  # 缓存 x 向力时程
        field_data['{}-{}-fy'.format(node_id, prefix)] = fy_arr  # 缓存 y 向力时程

    # 逐边界逐节点计算等效节点力
    for boundary in BOUNDARY_SEQUENCE:  # 遍历每个边界
        nd = boundary_node_data[boundary]  # 获取当前边界的节点数据
        for i in range(nd.shape[0]):  # 遍历节点
            process_boundary_node(nd[i, :], boundary)  # 计算并缓存当前节点等效力
        log_step(logger, '%s %s 边界等效节点力计算完成', model_name, boundary)  # 记录边界完成日志

    log_step(logger, '%s 所有边界波场叠加与等效节点力计算完成（传播矩阵法）', model_name)  # 记录整体完成日志

    # ============ 创建幅值曲线（Amplitude）============
    def batch_add_node_force_amplitude(node_data, prefix):  # 定义批量创建幅值曲线函数
        for i in range(node_data.shape[0]):  # 遍历节点数据
            node_id = int(node_data[i, 0])  # 读取节点标签
            fx_arr = field_data['{}-{}-fx'.format(node_id, prefix)]  # 读取 x 向力时程
            fy_arr = field_data['{}-{}-fy'.format(node_id, prefix)]  # 读取 y 向力时程

            ampli_fx = tuple(tuple(row) for row in fx_arr)  # 将 x 向时程转换为幅值数据
            ampli_fy = tuple(tuple(row) for row in fy_arr)  # 将 y 向时程转换为幅值数据

            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)  # 生成 x 向幅值名称
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)  # 生成 y 向幅值名称

            mdb.models[model_name].TabularAmplitude(  # 创建 x 向幅值曲线
                data=ampli_fx, name=name_amp_fx,  # 传入数据和名称
                smooth=SOLVER_DEFAULT, timeSpan=STEP)  # 设置平滑和时间跨度
            mdb.models[model_name].TabularAmplitude(  # 创建 y 向幅值曲线
                data=ampli_fy, name=name_amp_fy,  # 传入数据和名称
                smooth=SOLVER_DEFAULT, timeSpan=STEP)  # 设置平滑和时间跨度

    for boundary in BOUNDARY_SEQUENCE:  # 逐边界创建幅值曲线
        batch_add_node_force_amplitude(boundary_node_data[boundary], boundary)  # 调用幅值创建函数
    log_step(logger, '%s 所有边界节点的幅值曲线已创建', model_name)  # 记录幅值曲线日志

    # ============ 施加集中力载荷 ============
    def batch_add_node_force(node_data, prefix, step_name):  # 定义批量施加载荷函数
        assembly = mdb.models[model_name].rootAssembly  # 获取当前模型装配体
        instance_name = inst_name  # 记录实例名称
        n = assembly.instances[instance_name].nodes  # 获取实例节点集合

        for i in range(node_data.shape[0]):  # 遍历节点数据
            node_id = int(node_data[i, 0])  # 读取节点标签
            name_amp_fx = 'AMP-{}-{}-fx'.format(node_id, prefix)  # 生成 x 向幅值名
            name_amp_fy = 'AMP-{}-{}-fy'.format(node_id, prefix)  # 生成 y 向幅值名
            name_load_fx = 'load-{}-{}-fx'.format(node_id, prefix)  # 生成 x 向载荷名
            name_load_fy = 'load-{}-{}-fy'.format(node_id, prefix)  # 生成 y 向载荷名

            node_array = n.sequenceFromLabels([node_id])  # 按标签查找实例节点
            if len(node_array) == 0:  # 判断节点是否存在
                logger.warning('施加载荷时，实例中不存在节点 %d (实例: %s)', node_id, instance_name)  # 输出警告日志
                continue  # 跳过当前节点
            region = Region(nodes=node_array)  # 构建节点区域
            mdb.models[model_name].ConcentratedForce(  # 创建 x 向集中力
                name=name_load_fx, createStepName=step_name,  # 设置载荷名称和分析步
                region=region, cf1=1.0, amplitude=name_amp_fx,  # 设置作用区域和幅值
                distributionType=UNIFORM, field='', localCsys=None)  # 设置载荷分布
            mdb.models[model_name].ConcentratedForce(  # 创建 y 向集中力
                name=name_load_fy, createStepName=step_name,  # 设置载荷名称和分析步
                region=region, cf2=1.0, amplitude=name_amp_fy,  # 设置作用区域和幅值
                distributionType=UNIFORM, field='', localCsys=None)  # 设置载荷分布

    for boundary in BOUNDARY_SEQUENCE:  # 逐边界施加载荷
        batch_add_node_force(boundary_node_data[boundary], boundary, step_name)  # 调用载荷创建函数
    log_step(logger, '%s 所有边界节点已施加集中力', model_name)  # 记录载荷施加日志
    mdb.save()  # 保存模型数据库
    log_step(logger, '%s 粘弹性人工边界完成: 耗时=%.2fs', model_name, time.time() - t0)  # 记录结束耗时


def build_models(acc_info, base_model, part_name, inst_name,
                 angle,
                 cs_bedrock, vv_bedrock, density_bedrock,
                 cs_overlying, vv_overlying, density_overlying,
                 bedrock_thickness,
                 H_upper, H_lower, left_flat, w_slope,
                 step_name=DEFAULT_STEP_NAME, variables=('S', 'U', 'V'), frequency=10,
                 model_scene='slope', logger=None):
    """根据加速度时程信息批量复制模型、创建分析步、施加人工边界。"""  # 说明函数用途
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器

    variables = _normalize_output_variables(variables)  # 规范化场输出变量列表

    model_names = []  # 初始化模型名称列表
    for acc_file, tp, inc in acc_info:  # 遍历每个加速度记录
        new_model_name = _build_model_name_from_record(acc_file, model_scene)  # 按记录名和场景生成模型名
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])  # 复制基础模型
        log_step(logger, '%s 模型已从 %s 复制', new_model_name, base_model)  # 记录复制日志

        model = mdb.models[new_model_name]  # 获取新模型对象
        model.ImplicitDynamicsStep(  # 创建隐式动力分析步
            name=step_name, previous='Initial',  # 设置分析步名称和前置步
            timePeriod=tp, timeIncrementationMethod=FIXED, initialInc=inc,  # 设置时长和初始增量
            maxNumInc=1000000,  # 设置最大增量步数
            nlgeom=OFF, application=MODERATE_DISSIPATION)  # 关闭几何非线性并设置阻尼

        model.fieldOutputRequests['F-Output-1'].setValues(  # 设置场输出请求
            variables=variables, frequency=frequency)  # 指定输出变量和频率

        mdb.save()  # 保存模型数据库
        log_step(logger, '%s 分析步已创建, 时长=%.2f, 增量=%.3f',  # 记录分析步创建日志
                 new_model_name, tp, inc)  # 输出时长和初始增量

        VAB_oblique(angle=angle,  # 调用人工边界构建函数
                    cs_bedrock=cs_bedrock, vv_bedrock=vv_bedrock, density_bedrock=density_bedrock,  # 传入基岩参数
                    cs_overlying=cs_overlying, vv_overlying=vv_overlying, density_overlying=density_overlying,  # 传入覆盖层参数
                    bedrock_thickness=bedrock_thickness,  # 传入基岩厚度
                    H_upper=H_upper, H_lower=H_lower, left_flat=left_flat, w_slope=w_slope,  # 传入几何参数
                    model_name=new_model_name, part_name=part_name,  # 传入模型和零件名称
                    inst_name=inst_name,  # 传入实例名称
                    acc_file=acc_file, step_name=step_name,  # 传入加速度文件和分析步名称
                    logger=logger)  # 传入日志器
        model_names.append(new_model_name)  # 记录新模型名称

    return model_names  # 返回模型名称列表


def submit_job(num_cpus=7, memory_percent=90, model_name='Model-1', logger=None):
    """创建并提交Abaqus作业"""  # 说明函数用途
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    t0 = time.time()  # 记录作业开始时间
    job_name = 'job-' + model_name  # 生成作业名称
    if job_name in mdb.jobs:  # 判断是否存在同名旧作业
        del mdb.jobs[job_name]  # 删除旧作业
        log_step(logger, '检测到同名旧作业，已删除: %s', job_name)  # 记录删除日志
    log_step(logger, '%s作业开始提交, CPU 数量=%d, 内存=%d%%',  # 记录提交日志
             job_name, num_cpus, memory_percent)  # 输出 CPU 和内存配置

    mdb.Job(name=job_name, model=model_name,  # 创建 Abaqus 作业
            description='VAB oblique SV-wave analysis (Two-layered slope)',  # 设置作业描述
            type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0,  # 设置作业调度参数
            queue=None, memory=memory_percent, memoryUnits=PERCENTAGE,  # 设置内存参数
            getMemoryFromAnalysis=True, explicitPrecision=SINGLE,  # 设置精度参数
            nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,  # 关闭冗余输出
            contactPrint=OFF, historyPrint=OFF,  # 关闭接触与历史输出
            numCpus=num_cpus, numDomains=num_cpus,  # 设置 CPU 数量与并行域数量
            multiprocessingMode=DEFAULT, numGPUs=0)  # 设置多处理器并行模式与 GPU 核心数

    mdb.save()  # 保存模型数据库
    log_step(logger, '%s作业已提交，正在等待完成...', job_name)  # 记录作业提交日志
    mdb.jobs[job_name].submit(consistencyChecking=OFF)  # 提交作业并关闭一致性检查
    mdb.jobs[job_name].waitForCompletion()  # 等待作业完成
    log_step(logger, '%s已完成: 耗时=%.2fs', job_name, time.time() - t0)  # 记录作业完成耗时


if __name__ == '__main__':  # 判断是否直接运行脚本
    main()  # 调用主入口函数
