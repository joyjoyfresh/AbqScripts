# -*- coding: utf-8 -*-  # 指定源文件编码为 UTF-8

from abaqus import *  # 导入 Abaqus 主接口
from abaqusConstants import *  # 导入 Abaqus 常量
from abaqus import mdb  # 导入建模数据库对象
from regionToolset import Region  # 导入区域工具
from caeModules import *  # 导入 CAE 模块工具
import mesh  # 导入网格模块
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
        'max_reflect_order': 3,  # 设置多次反射/透射最大阶数
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
        cs_overlying = cs_bedrock / material_cfg['overlying']['velocity_ratio']

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

        # 网格尺寸设为高度 (H - h) 的 4%
        mesh_size = 4

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
            bedrock_thickness=bedrock_thickness,
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


def _compute_interface_sv_coeff(alpha1, mat1, mat2):  # 定义界面 SV 波系数计算函数
    """计算 SV 波在两层界面的等效反射/透射系数（阻抗近似）。"""  # 说明函数用途
    z1s = mat1['density'] * mat1['cs'] * max(1e-8, math.cos(alpha1))  # 计算入射侧等效阻抗
    sin_a2 = mat2['cs'] * math.sin(alpha1) / mat1['cs']  # 计算透射角正弦值
    alpha2 = _safe_arcsin(sin_a2)  # 计算透射角
    z2s = mat2['density'] * mat2['cs'] * max(1e-8, math.cos(alpha2))  # 计算透射侧等效阻抗
    denom = z1s + z2s if abs(z1s + z2s) > 1e-12 else 1e-12  # 计算分母并避免除零
    rss = (z2s - z1s) / denom  # 计算反射系数
    tss = 2.0 * z2s / denom  # 计算透射系数
    rsp = 0.0  # 设置转换反射系数为零
    tsp = 0.0  # 设置转换透射系数为零
    return {'Rss': rss, 'Rsp': rsp, 'Tss': tss, 'Tsp': tsp, 'alpha2': alpha2}  # 返回界面系数字典


def _compute_free_surface_sv_coeff(alpha, cp, cs):  # 定义 SV 波自由面系数计算函数
    """计算 SV 波在自由面的反射系数。"""  # 说明函数用途
    beta_p = _safe_arcsin(cp * math.sin(alpha) / cs)  # 计算自由面转换角
    numerator_a1 = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) - cp ** 2 * math.cos(2 * alpha) ** 2  # 计算 A1 分子
    denominator = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) + cp ** 2 * math.cos(2 * alpha) ** 2  # 计算公共分母
    if abs(denominator) < 1e-12:  # 检查分母是否过小
        denominator = 1e-12  # 避免除零
    a1 = numerator_a1 / denominator  # 计算反射系数 A1
    a2 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator  # 计算转换系数 A2
    return {'A1': a1, 'A2': a2, 'beta': beta_p}  # 返回自由面系数字典


def _compute_free_surface_p_coeff(beta, cp, cs):  # 定义 P 波自由面系数计算函数
    """计算 P 波在自由面的反射系数。"""  # 说明函数用途
    alpha = _safe_arcsin(cs * math.sin(beta) / cp)  # 计算对应入射角
    numerator_b2 = cp ** 2 * math.cos(2 * alpha) ** 2 - cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 计算 B2 分子
    denominator = cp ** 2 * math.cos(2 * alpha) ** 2 + cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 计算公共分母
    if abs(denominator) < 1e-12:  # 检查分母是否过小
        denominator = 1e-12  # 避免除零
    b2 = numerator_b2 / denominator  # 计算反射系数 B2
    b1 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator  # 计算转换系数 B1
    return {'B1': b1, 'B2': b2, 'alpha': alpha}  # 返回自由面系数字典


def _build_model_name_from_record(acc_file, scene_tag):  # 定义模型命名函数
    """按“记录名-场景名”规则生成模型名。"""  # 说明函数用途
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
    # 1. 基岩水平切分 (y = bedrock_thickness)
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


def VAB_oblique(angle,
                cs_bedrock, vv_bedrock, density_bedrock,
                cs_overlying, vv_overlying, density_overlying,
                bedrock_thickness,
                H_upper, H_lower, left_flat, w_slope,
                max_reflect_order=1,
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

    def _pick_material_by_node(x_coord, y_coord):  # 定义按节点坐标选择材料的函数
        if y_coord < bedrock_thickness + 1e-4:  # 判断节点是否位于基岩层
            return mat_bedrock  # 返回基岩材料参数
        else:  # 否则认为属于覆盖层
            return mat_overlying  # 返回覆盖层材料参数

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

    # 粘弹性人工边界参数 (根据节点所在材质层动态赋值)
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

    # 添加弹簧阻尼器到地面
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
        'l': (1, 2),  # 左边界法向为 1、切向为 2
        'r': (1, 2),  # 右边界法向为 1、切向为 2
        'b': (2, 1),  # 底边法向为 2、切向为 1
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

    # ============ 入射角与反射系数计算 ============
    if angle == 0:  # 判断入射角是否为零
        angle = 1e-10  # 用极小角度替代零角度
    else:  # 处理非零角度
        angle = round(angle, 4)  # 保留四位小数

    alpha1 = math.radians(angle)  # 将角度转换为弧度
    interface_coeff_12 = _compute_interface_sv_coeff(alpha1, mat_bedrock, mat_overlying)  # 计算界面 1->2 系数
    alpha2 = interface_coeff_12['alpha2']  # 读取透射角
    beta1 = _safe_arcsin(mat_bedrock['cp'] * math.sin(alpha1) / mat_bedrock['cs'])  # 计算基岩 P 波角
    beta2 = _safe_arcsin(mat_overlying['cp'] * math.sin(alpha2) / mat_overlying['cs']) if abs(math.sin(alpha2)) > 0 else 1e-10  # 计算覆盖层 P 波角

    free_sv_2 = _compute_free_surface_sv_coeff(alpha2, mat_overlying['cp'], mat_overlying['cs'])  # 计算覆盖层自由面 SV 系数
    free_p_2 = _compute_free_surface_p_coeff(beta2, mat_overlying['cp'], mat_overlying['cs'])  # 计算覆盖层自由面 P 系数
    interface_coeff_21 = _compute_interface_sv_coeff(alpha2, mat_overlying, mat_bedrock)  # 计算界面 2->1 系数

    rss_primary = interface_coeff_12['Rss']  # 提取主反射系数
    rsp_primary = interface_coeff_12['Rsp']  # 提取转换反射系数
    tss_primary = interface_coeff_12['Tss']  # 提取主透射系数
    tsp_primary = interface_coeff_12['Tsp']  # 提取转换透射系数

    h2 = max(0.0, ymax - bedrock_thickness)  # 计算覆盖层厚度
    cycle_sv = free_sv_2['A1'] * interface_coeff_21['Rss']  # 计算 SV 循环项
    cycle_p = free_p_2['B2'] * interface_coeff_21['Rss']  # 计算 P 循环项
    order_count = max(0, int(max_reflect_order))  # 计算反射阶数上限

    sum_cycle_sv = 0.0  # 初始化 SV 累积项
    sum_cycle_p = 0.0  # 初始化 P 累积项
    for order_idx in range(order_count + 1):  # 按阶数累加循环项
        sum_cycle_sv += cycle_sv ** order_idx  # 累加 SV 几何级数
        sum_cycle_p += cycle_p ** order_idx  # 累加 P 几何级数

    Rss_eff = rss_primary + tss_primary * free_sv_2['A1'] * interface_coeff_21['Tss'] * sum_cycle_sv  # 计算等效反射系数
    Rsp_eff = rsp_primary + tss_primary * free_sv_2['A2'] * interface_coeff_21['Tss'] * sum_cycle_p  # 计算等效转换系数
    cycle_delay_sv = (2.0 * h2 * math.cos(alpha2) / mat_overlying['cs']) if h2 > 0 else 0.0  # 计算 SV 循环延迟
    cycle_delay_p = (2.0 * h2 * math.cos(beta2) / mat_overlying['cp']) if h2 > 0 else 0.0  # 计算 P 循环延迟
    log_step(logger, '%s 反射参数计算完成: Rss_eff=%.4f, Rsp_eff=%.4f', model_name, Rss_eff, Rsp_eff)  # 记录反射参数日志

    # 映射回基准变量名 (底层使用 bedrock)
    alpha = alpha1  # 映射入射角变量
    beta_p = beta1  # 映射 P 波角变量
    A1 = Rss_eff  # 映射等效反射系数
    A2 = Rsp_eff  # 映射等效转换系数
    GG = mat_bedrock['GG']  # 映射基岩剪切模量
    lam = mat_bedrock['lam']  # 映射基岩拉梅常数
    cs = mat_bedrock['cs']  # 映射基岩剪切波速
    cp = mat_bedrock['cp']  # 映射基岩纵波波速

    # 读取加速度时程并积分
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

    vel = np.zeros_like(acc)  # 初始化速度数组
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)  # 通过梯形积分计算速度
    VEL = np.column_stack((time_arr, vel))  # 组合速度时程

    dis = np.zeros_like(vel)  # 初始化位移数组
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 通过梯形积分计算位移
    DIS = np.column_stack((time_arr, dis))  # 组合位移时程

    max_time = ACC[-1, 0]  # 读取原始时程末时刻
    Ly = bedrock_thickness - ymin  # 计算基岩界面高度差
    Lx = xmax - xmin  # 计算模型横向跨度

    # 计算各节点的波到达延迟时间
    def calc_node_delay(node_data, boundary, alpha, beta_p, cs, cp, Ly, Lx,
                        cs2, cp2, alpha2, beta2, ymax_col):  # 定义节点延迟时间计算函数
        n = node_data.shape[0]  # 读取节点数量
        det = np.zeros((n, 4))  # 初始化延迟时间表
        det[:, 0] = node_data[:, 0]  # 将节点标签写入结果表第一列

        for i in range(n):  # 遍历每个边界节点
            x0 = node_data[i, 1]  # 读取节点 x 坐标
            y0 = node_data[i, 2]  # 读取节点 y 坐标

            if boundary == 'l':  # 处理左边界节点
                if y0 <= Ly:  # 判断是否处于基岩段
                    t1 = y0 * np.cos(alpha) / cs  # 计算第一段延迟
                    t2 = (2 * Ly - y0) * np.cos(alpha) / cs  # 计算第二段延迟
                    t3 = ((Ly - y0) / (cp * np.cos(beta_p))  # 计算第三段延迟的第一部分
                          + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs)  # 计算第三段延迟的第二部分
                else:  # 处理覆盖层段
                    t1 = Ly * np.cos(alpha) / cs + (y0 - Ly) * np.cos(alpha2) / cs2  # 计算第一段延迟
                    t2 = Ly * np.cos(alpha) / cs + (2 * ymax_col - Ly - y0) * np.cos(alpha2) / cs2  # 计算第二段延迟
                    t3 = Ly * np.cos(alpha) / cs + (y0 - Ly) * np.cos(beta2) / cp2  # 计算第三段延迟
                det[i, 1] = t1  # 写入第一列延迟
                det[i, 2] = t2  # 写入第二列延迟
                det[i, 3] = t3  # 写入第三列延迟

            elif boundary == 'r':  # 处理右边界节点
                if y0 <= Ly:  # 判断是否处于基岩段
                    t1 = y0 * np.cos(alpha) / cs  # 计算第一段延迟
                    t2 = (2 * Ly - y0) * np.cos(alpha) / cs  # 计算第二段延迟
                    t3 = ((Ly - y0) / (cp * np.cos(beta_p))  # 计算第三段延迟的第一部分
                          + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs)  # 计算第三段延迟的第二部分
                else:  # 处理覆盖层段
                    t1 = Ly * np.cos(alpha) / cs + (y0 - Ly) * np.cos(alpha2) / cs2  # 计算第一段延迟
                    t2 = Ly * np.cos(alpha) / cs + (2 * ymax_col - Ly - y0) * np.cos(alpha2) / cs2  # 计算第二段延迟
                    t3 = Ly * np.cos(alpha) / cs + (y0 - Ly) * np.cos(beta2) / cp2  # 计算第三段延迟
                det[i, 1] = t1 + Lx * np.sin(alpha) / cs  # 写入并叠加横向传播延迟
                det[i, 2] = t2 + Lx * np.sin(alpha) / cs  # 写入并叠加横向传播延迟
                det[i, 3] = t3 + Lx * np.sin(alpha) / cs  # 写入并叠加横向传播延迟

            elif boundary == 'b':  # 处理底边节点
                t4 = x0 * np.sin(alpha) / cs  # 计算底边第一段延迟
                t5 = (2 * Ly + x0 * np.tan(alpha)) * np.cos(alpha) / cs  # 计算底边第二段延迟
                t6 = (Ly / (cp * np.cos(beta_p))  # 计算底边第三段延迟的第一部分
                      + (Ly * np.cos(alpha) + x0 * np.sin(alpha)  # 计算底边第三段延迟的第二部分
                         - Ly * np.tan(beta_p) * np.sin(alpha)) / cs)  # 计算底边第三段延迟的第三部分
                det[i, 1] = t4  # 写入第一列延迟
                det[i, 2] = t5  # 写入第二列延迟
                det[i, 3] = t6  # 写入第三列延迟
            else:  # 处理非法边界类型
                raise ValueError("boundary must be 'l', 'r', or 'b'")  # 抛出边界类型异常
        return det  # 返回延迟时间表

    det_l = calc_node_delay(node_data_l, 'l', alpha, beta_p, cs, cp, Ly, Lx,  # 计算左边界延迟表
                            cs_overlying, mat_overlying['cp'], alpha2, beta2, ymax_l)  # 传入覆盖层参数
    det_r = calc_node_delay(node_data_r, 'r', alpha, beta_p, cs, cp, Ly, Lx,  # 计算右边界延迟表
                            cs_overlying, mat_overlying['cp'], alpha2, beta2, ymax_r)  # 传入覆盖层参数
    det_b = calc_node_delay(node_data_b, 'b', alpha, beta_p, cs, cp, Ly, Lx,  # 计算底边延迟表
                            cs_overlying, mat_overlying['cp'], alpha2, beta2, ymax)  # 传入覆盖层参数
    log_step(logger, '%s 左/右/底 边界节点延迟时间已计算', model_name)  # 记录延迟时间日志

    # 补零延长
    detmax = max(np.max(det_l[:, 1:]), np.max(det_r[:, 1:]), np.max(det_b[:, 1:]))  # 计算最大延迟时间
    if max_time < detmax:  # 判断原始时程是否不足以覆盖所有延迟
        n_add = int(np.ceil((detmax - max_time) / dt))  # 计算需要补零的步数
        new_times = VEL[-1, 0] + dt * np.arange(1, n_add + 1)  # 生成新增时间序列
        new_vel = np.zeros((n_add, 2))  # 创建补零数组
        new_vel[:, 0] = new_times  # 写入补零段时间
        VEL = np.vstack([VEL, new_vel])  # 将补零段追加到速度时程
        DIS = np.vstack([DIS, new_vel])  # 将补零段追加到位移时程
        log_step(logger, '%s VEL/DIS 已用零延长: 增加行数=%d', model_name, n_add)  # 记录补零日志
    else:  # 处理无需延长的情况
        log_step(logger, '%s VEL/DIS 无需延长', model_name)  # 记录无需延长日志

    # 对齐延迟时间
    def round_delay(det, dt):  # 定义延迟时间取整函数
        det[:, 1:4] = np.round(det[:, 1:4] / dt) * dt  # 将延迟时间对齐到时间步长
        return det  # 返回取整后的延迟表

    det_l = round_delay(det_l, dt)  # 对左边界延迟时间取整
    det_r = round_delay(det_r, dt)  # 对右边界延迟时间取整
    det_b = round_delay(det_b, dt)  # 对底边延迟时间取整

    det_map_l = {int(row[0]): (row[1], row[2], row[3]) for row in det_l}  # 构建左边界延迟映射
    det_map_r = {int(row[0]): (row[1], row[2], row[3]) for row in det_r}  # 构建右边界延迟映射
    det_map_b = {int(row[0]): (row[1], row[2], row[3]) for row in det_b}  # 构建底边延迟映射

    def delay_signal(u0, delay_t, dt):  # 定义信号延迟函数
        n_delay = int(np.round(delay_t / dt))  # 将延迟时间转换为步数
        N = u0.shape[0]  # 获取原始序列长度
        new_len = N + n_delay  # 计算延迟后长度
        delayed = np.zeros((new_len, 2))  # 创建延迟后的时程数组
        delayed[:, 0] = np.arange(new_len) * dt  # 生成新的时间轴
        delayed[n_delay:, 1] = u0[:, 1]  # 将原始信号平移到延迟位置
        return delayed  # 返回延迟信号

    def make_delay_cache(timeseries, dt):  # 定义延迟缓存工厂函数
        cache = {}  # 初始化缓存字典
        def get_delayed(delay_t):  # 定义延迟获取函数
            n_delay = int(np.round(delay_t / dt))  # 将延迟时间转换为步数
            if n_delay not in cache:  # 判断缓存中是否已有对应延迟
                cache[n_delay] = delay_signal(timeseries, n_delay * dt, dt)  # 生成并缓存延迟信号
            return cache[n_delay]  # 返回缓存中的延迟信号
        return get_delayed  # 返回延迟获取函数

    def pad_to(arr, length, dt):  # 定义数组补零函数
        if arr.shape[0] < length:  # 判断是否需要补零
            pad = np.zeros((length - arr.shape[0], 2))  # 创建补零数组
            pad[:, 0] = np.arange(arr.shape[0], length) * dt  # 为补零段补充时间轴
            arr = np.vstack([arr, pad])  # 将补零段拼接到末尾
        return arr  # 返回补零后的数组

    # ============ 计算自由场位移和速度 ============
    field_data = {}  # 初始化自由场结果缓存

    def calc_freefield_u_and_dotu_general(node_data, det_map, timeseries, dt,
                                           alpha, beta_p, A1, A2,
                                           suffix1, suffix2, prefix):  # 定义自由场位移速度计算函数
        get_delayed = make_delay_cache(timeseries, dt)  # 创建延迟信号缓存器
        for i in range(node_data.shape[0]):  # 遍历边界节点
            node_id = int(node_data[i, 0])  # 读取节点标签
            tA, tB, tC = det_map[node_id]  # 读取三段延迟时间

            u0_tA = get_delayed(tA)  # 获取主传播延迟信号
            u0_tB_list = []  # 初始化 B 路径信号列表
            u0_tC_list = []  # 初始化 C 路径信号列表
            for order_idx in range(order_count + 1):  # 遍历反射阶数
                delay_b = tB + order_idx * cycle_delay_sv  # 计算 SV 循环延迟
                delay_c = tC + order_idx * cycle_delay_p  # 计算 P 循环延迟
                u0_tB_list.append(get_delayed(delay_b))  # 追加 B 路径延迟信号
                u0_tC_list.append(get_delayed(delay_c))  # 追加 C 路径延迟信号

            max_len = u0_tA.shape[0]  # 初始化最大长度
            for arr in u0_tB_list + u0_tC_list:  # 遍历所有延迟信号
                max_len = max(max_len, arr.shape[0])  # 统计统一长度

            u0_tA = pad_to(u0_tA, max_len, dt)  # 补齐主传播信号长度
            u0_tB_sum = np.zeros(max_len)  # 初始化 B 路径累加数组
            u0_tC_sum = np.zeros(max_len)  # 初始化 C 路径累加数组
            for order_idx in range(order_count + 1):  # 按阶数叠加路径信号
                arr_b = pad_to(u0_tB_list[order_idx], max_len, dt)  # 补齐 B 路径信号
                arr_c = pad_to(u0_tC_list[order_idx], max_len, dt)  # 补齐 C 路径信号
                u0_tB_sum += (cycle_sv ** order_idx) * arr_b[:, 1]  # 累加 B 路径贡献
                u0_tC_sum += (cycle_p ** order_idx) * arr_c[:, 1]  # 累加 C 路径贡献

            ux = (u0_tA[:, 1] * np.cos(alpha)  # 计算 x 向位移
                  - A1 * u0_tB_sum * np.cos(alpha)  # 叠加反射修正项
                  + A2 * u0_tC_sum * np.sin(beta_p))  # 叠加转换修正项
            uy = (-u0_tA[:, 1] * np.sin(alpha)  # 计算 y 向位移
                  - A1 * u0_tB_sum * np.sin(alpha)  # 叠加反射修正项
                  - A2 * u0_tC_sum * np.cos(beta_p))  # 叠加转换修正项

            ux_arr = np.zeros((max_len, 2))  # 创建 x 向时程数组
            uy_arr = np.zeros((max_len, 2))  # 创建 y 向时程数组
            ux_arr[:, 0] = u0_tA[:, 0]  # 写入时间轴
            uy_arr[:, 0] = u0_tA[:, 0]  # 写入时间轴
            ux_arr[:, 1] = ux  # 写入 x 向结果
            uy_arr[:, 1] = uy  # 写入 y 向结果

            field_data['{}-{}-{}'.format(node_id, prefix, suffix1)] = ux_arr  # 缓存 x 向结果
            field_data['{}-{}-{}'.format(node_id, prefix, suffix2)] = uy_arr  # 缓存 y 向结果

    boundary_det_map = {  # 定义各边界延迟映射
        'l': det_map_l,  # 左边界映射
        'r': det_map_r,  # 右边界映射
        'b': det_map_b,  # 底边映射
    }  # 结束边界延迟映射

    # 计算位移自由场
    for boundary in BOUNDARY_SEQUENCE:  # 逐边界计算位移自由场
        calc_freefield_u_and_dotu_general(  # 调用自由场计算函数
            boundary_node_data[boundary], boundary_det_map[boundary], DIS, dt,  # 传入位移时程与延迟表
            alpha, beta_p, A1, A2, 'ux', 'uy', boundary)  # 传入输出后缀与边界标识
    log_step(logger, '%s 自由场位移已计算', model_name)  # 记录位移自由场日志

    # 计算速度自由场
    for boundary in BOUNDARY_SEQUENCE:  # 逐边界计算速度自由场
        calc_freefield_u_and_dotu_general(  # 调用自由场计算函数
            boundary_node_data[boundary], boundary_det_map[boundary], VEL, dt,  # 传入速度时程与延迟表
            alpha, beta_p, A1, A2, 'dotux', 'dotuy', boundary)  # 传入输出后缀与边界标识
    log_step(logger, '%s 自由场速度已计算', model_name)  # 记录速度自由场日志

    # ============ 计算自由场应力 ============
    def calc_freefield_sigma_general(node_data, det_map, VEL, dt,
                                      alpha, beta_p, A1, A2,
                                      GG, cs, lam, cp, prefix):  # 定义自由场应力计算函数
        get_delayed = make_delay_cache(VEL, dt)  # 创建速度延迟缓存器
        for i in range(node_data.shape[0]):  # 遍历边界节点
            node_id = int(node_data[i, 0])  # 读取节点标签
            tA, tB, tC = det_map[node_id]  # 读取三段延迟时间

            v0_tA = get_delayed(tA)  # 获取主传播速度信号
            v0_tB_list = []  # 初始化 B 路径速度列表
            v0_tC_list = []  # 初始化 C 路径速度列表
            for order_idx in range(order_count + 1):  # 遍历反射阶数
                delay_b = tB + order_idx * cycle_delay_sv  # 计算 SV 循环延迟
                delay_c = tC + order_idx * cycle_delay_p  # 计算 P 循环延迟
                v0_tB_list.append(get_delayed(delay_b))  # 追加 B 路径速度信号
                v0_tC_list.append(get_delayed(delay_c))  # 追加 C 路径速度信号

            max_len = v0_tA.shape[0]  # 初始化最大长度
            for arr in v0_tB_list + v0_tC_list:  # 遍历所有延迟信号
                max_len = max(max_len, arr.shape[0])  # 统计统一长度

            v0_tA = pad_to(v0_tA, max_len, dt)  # 补齐主传播速度信号
            v0_tB_sum = np.zeros(max_len)  # 初始化 B 路径累加数组
            v0_tC_sum = np.zeros(max_len)  # 初始化 C 路径累加数组
            for order_idx in range(order_count + 1):  # 按阶数叠加路径信号
                arr_b = pad_to(v0_tB_list[order_idx], max_len, dt)  # 补齐 B 路径信号
                arr_c = pad_to(v0_tC_list[order_idx], max_len, dt)  # 补齐 C 路径信号
                v0_tB_sum += (cycle_sv ** order_idx) * arr_b[:, 1]  # 累加 B 路径贡献
                v0_tC_sum += (cycle_p ** order_idx) * arr_c[:, 1]  # 累加 C 路径贡献

            sin2a = np.sin(2 * alpha)  # 计算双角正弦项
            cos2a = np.cos(2 * alpha)  # 计算双角余弦项
            sin2bp = np.sin(beta_p) ** 2  # 计算 P 波角相关项
            sin2bp_2 = np.sin(2 * beta_p)  # 计算双倍 P 波角正弦项
            cosbp = np.cos(beta_p)  # 计算 P 波角余弦
            cosbp2 = cosbp ** 2  # 计算余弦平方项

            if prefix == 'l':  # 处理左边界应力
                sigmax = (GG / cs * sin2a * (v0_tA[:, 1] - A1 * v0_tB_sum)  # 计算 x 向应力
                          + A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC_sum)  # 叠加转换项
                sigmay = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB_sum)  # 计算 y 向应力
                          - A2 * GG * sin2bp_2 / cp * v0_tC_sum)  # 叠加转换项
            elif prefix == 'r':  # 处理右边界应力
                sigmax = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB_sum)  # 计算 x 向应力
                          - A2 * (lam + 2 * GG * sin2bp) / cp * v0_tC_sum)  # 叠加转换项
                sigmay = (GG / cs * cos2a * (-v0_tA[:, 1] - A1 * v0_tB_sum)  # 计算 y 向应力
                          + A2 * GG * sin2bp_2 / cp * v0_tC_sum)  # 叠加转换项
            elif prefix == 'b':  # 处理底边应力
                sigmax = (GG / cs * cos2a * (v0_tA[:, 1] + A1 * v0_tB_sum)  # 计算 x 向应力
                          - A2 * GG * sin2bp_2 / cp * v0_tC_sum)  # 叠加转换项
                sigmay = (GG / cs * sin2a * (-v0_tA[:, 1] + A1 * v0_tB_sum)  # 计算 y 向应力
                          + A2 * (lam + 2 * GG * cosbp2) / cp * v0_tC_sum)  # 叠加转换项
            else:  # 处理非法边界前缀
                raise ValueError("prefix must be 'l', 'r' or 'b'")  # 抛出前缀异常

            sigmax_arr = np.zeros((max_len, 2))  # 创建 x 向应力数组
            sigmay_arr = np.zeros((max_len, 2))  # 创建 y 向应力数组
            sigmax_arr[:, 0] = v0_tA[:, 0]  # 写入时间轴
            sigmay_arr[:, 0] = v0_tA[:, 0]  # 写入时间轴
            sigmax_arr[:, 1] = sigmax  # 写入 x 向应力结果
            sigmay_arr[:, 1] = sigmay  # 写入 y 向应力结果

            field_data['{}-{}-sigmax'.format(node_id, prefix)] = sigmax_arr  # 缓存 x 向应力时程
            field_data['{}-{}-sigmay'.format(node_id, prefix)] = sigmay_arr  # 缓存 y 向应力时程

    for boundary in BOUNDARY_SEQUENCE:  # 逐边界计算自由场应力
        calc_freefield_sigma_general(  # 调用自由场应力计算函数
            boundary_node_data[boundary], boundary_det_map[boundary], VEL, dt,  # 传入速度时程与延迟表
            alpha, beta_p, A1, A2, GG, cs, lam, cp, boundary)  # 传入材料参数与边界标识
    log_step(logger, '%s 自由场应力已计算', model_name)  # 记录应力自由场日志

    # ============ 计算等效节点力 ============
    def calc_equiv_node_force_general(node_data, prefix):  # 定义等效节点力计算函数
        for i in range(node_data.shape[0]):  # 遍历边界节点
            node_id = int(node_data[i, 0])  # 读取节点标签
            A = node_data[i, 3]  # 读取影响长度
            kn = node_data[i, 4]  # 读取法向刚度
            cn = node_data[i, 5]  # 读取法向阻尼
            kt = node_data[i, 6]  # 读取切向刚度
            ct = node_data[i, 7]  # 读取切向阻尼

            ux_arr = field_data['{}-{}-ux'.format(node_id, prefix)]  # 读取 x 向位移时程
            dotux_arr = field_data['{}-{}-dotux'.format(node_id, prefix)]  # 读取 x 向速度时程
            sigmax_arr = field_data['{}-{}-sigmax'.format(node_id, prefix)]  # 读取 x 向应力时程
            uy_arr = field_data['{}-{}-uy'.format(node_id, prefix)]  # 读取 y 向位移时程
            dotuy_arr = field_data['{}-{}-dotuy'.format(node_id, prefix)]  # 读取 y 向速度时程
            sigmay_arr = field_data['{}-{}-sigmay'.format(node_id, prefix)]  # 读取 y 向应力时程

            min_len = min(ux_arr.shape[0], dotux_arr.shape[0], sigmax_arr.shape[0],  # 计算统一长度
                          uy_arr.shape[0], dotuy_arr.shape[0], sigmay_arr.shape[0])  # 计算统一长度

            ux = ux_arr[:min_len, 1]  # 截取 x 向位移值
            dotux = dotux_arr[:min_len, 1]  # 截取 x 向速度值
            sigmax = sigmax_arr[:min_len, 1]  # 截取 x 向应力值
            uy = uy_arr[:min_len, 1]  # 截取 y 向位移值
            dotuy = dotuy_arr[:min_len, 1]  # 截取 y 向速度值
            sigmay = sigmay_arr[:min_len, 1]  # 截取 y 向应力值
            time = ux_arr[:min_len, 0]  # 截取时间轴

            if prefix in ('l', 'r'):  # 处理侧边界
                fx = kn * ux + cn * dotux + A * sigmax  # 计算 x 向等效力
                fy = kt * uy + ct * dotuy + A * sigmay  # 计算 y 向等效力
            elif prefix == 'b':  # 处理底边界
                fx = kt * ux + ct * dotux + A * sigmax  # 计算 x 向等效力
                fy = kn * uy + cn * dotuy + A * sigmay  # 计算 y 向等效力
            else:  # 处理非法前缀
                raise ValueError("prefix must be 'l', 'r' or 'b'")  # 抛出前缀异常

            fx_arr = np.zeros((min_len, 2))  # 创建 x 向力时程数组
            fy_arr = np.zeros((min_len, 2))  # 创建 y 向力时程数组
            fx_arr[:, 0] = time  # 写入时间轴
            fy_arr[:, 0] = time  # 写入时间轴
            fx_arr[:, 1] = fx  # 写入 x 向力值
            fy_arr[:, 1] = fy  # 写入 y 向力值

            field_data['{}-{}-fx'.format(node_id, prefix)] = fx_arr  # 缓存 x 向力时程
            field_data['{}-{}-fy'.format(node_id, prefix)] = fy_arr  # 缓存 y 向力时程

    for boundary in BOUNDARY_SEQUENCE:  # 逐边界计算等效节点力
        calc_equiv_node_force_general(boundary_node_data[boundary], boundary)  # 调用等效节点力计算函数
    log_step(logger, '%s 等效节点力时程已计算', model_name)  # 记录等效力日志

    # 创建幅值曲线 (Amplitude)
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

    # 施加集中力载荷
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
                 max_reflect_order=1,
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
                    max_reflect_order=max_reflect_order,  # 传入反射阶数
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
            numCpus=num_cpus, numGPUs=0)  # 设置并行计算资源

    mdb.save()  # 保存模型数据库
    log_step(logger, '%s作业已提交，正在等待完成...', job_name)  # 记录作业提交日志
    mdb.jobs[job_name].submit(consistencyChecking=OFF)  # 提交作业并关闭一致性检查
    mdb.jobs[job_name].waitForCompletion()  # 等待作业完成
    log_step(logger, '%s已完成: 耗时=%.2fs', job_name, time.time() - t0)  # 记录作业完成耗时


if __name__ == '__main__':  # 判断是否直接运行脚本
    main()  # 调用主入口函数
