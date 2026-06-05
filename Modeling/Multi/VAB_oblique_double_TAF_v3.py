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
from collections import namedtuple  # 导入命名元组用于参数打包


# ==========================================================
#  配置参数（实验参数集中在此处修改）
# ==========================================================


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


mesh_size = 4  # 网格尺寸设为 4 m


# ==========================================================
#  模块常量与全局状态
# ==========================================================


DEFAULT_STEP_NAME = 'Step-earthquake'  # 定义默认分析步名称
BOUNDARY_SET_NAMES = ('Left_boundary', 'Right_boundary', 'Bottom_boundary')  # 定义基础边界节点集名称
BOUNDARY_SEQUENCE = ('l', 'r', 'b')  # 定义边界处理顺序


_FF_TRANSFER_CACHE = {}  # 模块级传递函数缓存：键为节点几何+材料+频率轴指纹，值为频域传递数组字典


# ============================================================
#  参数打包对象（v7：用结构化对象取代散标量，缩短函数签名、便于阅读）
# ============================================================
# Material：单层材料的基本输入（剪切波速、泊松比、密度）；
#   派生量 GG/lam/cp/EE 仍由物理核心函数按需计算，故此处只存输入。
Material = namedtuple('Material', ['cs', 'vv', 'density'])  # 单层材料输入
# Site：场地两层材料 + 基岩层厚度
Site = namedtuple('Site', ['bedrock', 'overlying', 'bedrock_thickness'])  # 双层场地
# Geometry：斜坡几何（输入项 + 一次算好的派生项）
Geometry = namedtuple('Geometry', [
    'total_L', 'i', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness',  # 输入项
    'H', 'h', 'H_upper', 'H_lower', 'H_flat', 'w_slope'])  # 派生项
# BoundaryNode：单个边界节点的几何与粘弹性边界参数（取代裸 numpy 列索引）
BoundaryNode = namedtuple('BoundaryNode',
                          ['label', 'x', 'y', 'influence', 'kn', 'cn', 'kt', 'ct'])  # 边界节点
# FreeFieldCtx：等效力计算所需的上下文（一次打包，避免长参数列表）
FreeFieldCtx = namedtuple('FreeFieldCtx', [  # 自由场上下文命名元组
    'mat_bedrock', 'mat_overlying', 'geom', 'ymax_l', 'ymax_r', 'ymin',  # 两层材料、几何、各边界高度信息
    'p_horiz', 'vel_freq', 'freq_arr', 'dt', 'N_fft', 'N_orig', 'time_arr'])  # 水平慢度、输入频谱与时频长度


# ==========================================================
#  通用工具函数
# ==========================================================


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


def _safe_arcsin(value):  # 定义安全反正弦函数
    """对 arcsin 输入做截断，避免浮点超界。"""  # 说明函数用途
    return math.asin(max(-1.0, min(1.0, value)))  # 将输入截断到合法范围后再求反正弦


# ==========================================================
#  物理与数值计算
# ==========================================================


def _compute_wave_speed_from_elastic_modulus(elastic_modulus, poisson_ratio, density):  # 定义波速反算函数
    """根据杨氏模量、泊松比和密度计算剪切波速。"""  # 说明函数用途
    return math.sqrt((elastic_modulus / (2 * (1 + poisson_ratio))) / density)  # 返回计算得到的剪切波速


def _compute_elastic_modulus_from_wave_speed(cs, vv, density):  # 定义弹性模量反算函数
    """根据剪切波速、泊松比和密度计算杨氏模量 E。"""  # 说明函数用途
    GG = density * (cs ** 2)  # 计算剪切模量
    EE = 2 * GG * (1 + vv)  # 计算杨氏模量
    return EE  # 返回杨氏模量


def _compute_material_params(cs, vv, density):  # 定义材料参数计算函数
    """根据 Vs、泊松比、密度计算材料参数。"""  # 说明函数用途
    GG = density * cs ** 2  # 计算剪切模量
    EE = 2 * GG * (1 + vv)  # 计算杨氏模量
    lam = 2 * GG * vv / (1 - 2 * vv)  # 计算拉梅常数
    cp = math.sqrt((lam + 2 * GG) / density)  # 计算纵波波速
    return {'GG': GG, 'EE': EE, 'lam': lam, 'cp': cp, 'cs': cs, 'vv': vv, 'density': density}  # 返回材料参数字典


def _integrate_acc_to_velocity(acc, dt, time_arr):  # 定义加速度积分并基线校正的纯数值函数
    """加速度梯形积分为速度并做基线校正（去零偏 + 线性去趋势），抑制低频漂移。

    acc      : 加速度时程数组
    dt       : 时间步长 (s)
    time_arr : 与 acc 对应的时间轴数组
    返回 (vel, slope)：校正后的速度数组与被扣除的速度线性趋势斜率
    """
    acc = acc - np.mean(acc)  # 去除加速度零频偏移，避免积分后速度产生线性漂移
    vel = np.zeros_like(acc)  # 初始化速度数组
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)  # 通过梯形积分计算速度
    trend = np.polyfit(time_arr, vel, 1)  # 最小二乘拟合速度的线性趋势项
    vel = vel - (trend[0] * time_arr + trend[1])  # 扣除线性趋势完成基线校正（位移=速度/(iω) 会放大低频误差）
    return vel, trend[0]  # 返回校正后速度与趋势斜率


def _surface_y_at(x, H_upper, H_lower, left_flat, w_slope):  # 定义按横坐标计算地表高度的纯函数
    """返回横坐标 x 处的地表 y 坐标（用于底边节点取其正上方柱子的覆盖层厚度）。

    几何：坡顶平台高 H_upper，坡脚平台高 H_lower，二者之间为线性坡面。
    x <= left_flat            : 坡顶平台，地表 = H_upper
    left_flat < x <= +w_slope : 坡面段，地表沿 x 从 H_upper 线性降到 H_lower
    x  > left_flat + w_slope  : 坡脚平台，地表 = H_lower
    """
    w = max(w_slope, 1e-9)  # 防止除零（平坦模型 w_slope 取极小值）
    if x <= left_flat:  # 判断是否位于坡顶平台
        return H_upper  # 返回坡顶平台高度
    if x <= left_flat + w:  # 判断是否位于坡面段
        return H_upper - (x - left_flat) * (H_upper - H_lower) / w  # 返回坡面线性过渡高度
    return H_lower  # 其余位于坡脚平台，返回坡脚高度


# ==========================================================
#  几何构造与命名
# ==========================================================


def make_geometry(total_L, H_minus_h, i, h_over_H, left_flat, bedrock_thickness):  # 定义斜坡几何构造函数
    """根据斜坡几何输入计算全部派生量并打包为 Geometry。"""  # 说明函数用途
    H = H_minus_h / (1.0 - h_over_H)  # 计算总覆盖层厚度
    h = H - H_minus_h  # 计算下部覆盖层高度
    H_upper = bedrock_thickness + H  # 计算坡顶地表高度
    H_lower = bedrock_thickness + h  # 计算坡脚地表高度
    H_flat = bedrock_thickness + H  # 计算平坦场地总高度（= 坡顶高度）
    w_slope = H_minus_h / math.tan(math.radians(i))  # 计算坡面水平长度
    return Geometry(total_L=total_L, i=i, left_flat=left_flat, H_minus_h=H_minus_h,  # 组装并返回几何对象（填入输入项）
                    h_over_H=h_over_H, bedrock_thickness=bedrock_thickness,  # 继续填入输入项
                    H=H, h=h, H_upper=H_upper, H_lower=H_lower, H_flat=H_flat, w_slope=w_slope)  # 填入派生项


def make_flat_geometry(geom):  # 定义平坦自由场几何派生函数
    """由斜坡几何派生平坦自由场几何：上下表面统一为 H_upper、坡面宽度取极小值。"""  # 说明函数用途
    return geom._replace(H_lower=geom.H_upper, w_slope=0.001)  # 替换两项后返回新几何对象


def _build_model_name_from_record(acc_file, scene_tag):  # 定义模型命名函数
    """按"记录名-场景名"规则生成模型名。"""  # 说明函数用途
    record_name = os.path.splitext(os.path.basename(acc_file))[0]  # 提取不带扩展名的记录名
    if not record_name:  # 检查记录名是否为空
        raise ValueError('无法从加速度文件生成记录名: %s' % acc_file)  # 抛出命名错误
    if scene_tag not in ('slope', 'flat'):  # 检查场景标签是否合法
        raise ValueError('scene_tag 仅支持 slope 或 flat，当前为: %s' % scene_tag)  # 抛出场景错误
    return '{}-{}'.format(record_name, scene_tag)  # 返回组合后的模型名


# ==========================================================
#  输入读取
# ==========================================================


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


# ============================================================
#  双层自由场核心实现（v7 向量化）：斜入射 SV 波的直接多波 Zoeppritz 界面解
#  （基岩入射 SV + 界面反/透射 + 覆盖层自由面多次混响，含 SV<->P 转换）
#  v7 优化：垂直慢度 q 与位移/应力系数均与频率无关，仅相位因子随 ω 变化；
#          故将原逐频率 Python 循环（每频率组装并求解 4×4）改为对全部频率一次性
#          向量化（批量 np.linalg.solve），并对"传递函数"按节点几何缓存、跨地震记录
#          复用（不同记录仅 vel_freq 不同）。数值与逐频率实现等价（最大相对偏差
#          ~1e-16，见 test/test_vectorize_equiv.py）。
# ============================================================


def _freefield_transfer(y_target, x_target, mat_bedrock, mat_overlying,
                        h_bedrock, h_overlying, y_bottom,
                        p_horiz, freq_arr, dt, N_fft):
    """向量化计算单节点"每单位输入速度谱"的频域传递函数，并按节点几何缓存。

    返回 dict：'ux','uy','dotux','dotuy','sxx','syy','sxy'，每个为长度 N_pos 的复数数组，
    满足 <量>_freq = 传递函数 * vel_freq（DC 分量已置零）。
    传递函数只依赖节点几何/材料/频率轴，与具体地震记录（vel_freq）无关，故可跨记录复用。
    """
    cs1 = mat_bedrock['cs']  # 基岩剪切波速
    cp1 = mat_bedrock['cp']  # 基岩纵波波速
    cs2 = mat_overlying['cs']  # 覆盖层剪切波速
    cp2 = mat_overlying['cp']  # 覆盖层纵波波速
    GG1 = mat_bedrock['GG']  # 基岩剪切模量
    lam1 = mat_bedrock['lam']  # 基岩拉梅常数
    GG2 = mat_overlying['GG']  # 覆盖层剪切模量
    lam2 = mat_overlying['lam']  # 覆盖层拉梅常数

    # 缓存键：节点几何 + 慢度 + 频率轴指纹 + 材料标量（四舍五入以稳定浮点键）
    key = (round(y_target, 6), round(x_target, 6), round(h_bedrock, 6),  # 节点与层厚几何
           round(h_overlying, 6), round(y_bottom, 6), round(p_horiz, 12),  # 覆盖层厚、底高、水平慢度
           int(N_fft), round(dt, 12),  # 频率轴指纹（FFT 长度与步长）
           cs1, cp1, cs2, cp2, GG1, lam1, GG2, lam2)  # 两层材料标量
    cached = _FF_TRANSFER_CACHE.get(key)  # 查缓存
    if cached is not None:  # 命中则直接返回
        return cached  # 返回缓存的传递函数

    omega = 2.0 * math.pi * freq_arr  # 角频率数组
    y_intf = y_bottom + h_bedrock  # 基岩/覆盖层界面 y 坐标

    def _qval(c):  # 计算垂直慢度 q = sqrt(1/c^2 - p^2)（与频率无关）
        val = (1.0 / c) ** 2 - p_horiz ** 2  # 慢度平方
        if val >= 0:  # 实数根
            return complex(math.sqrt(val), 0)  # 返回实数垂直慢度
        return complex(0, math.sqrt(-val))  # 倏逝波返回纯虚数

    qs1 = _qval(cs1)  # 基岩 SV 垂直慢度
    qp1 = _qval(cp1)  # 基岩 P 垂直慢度
    qs2 = _qval(cs2)  # 覆盖层 SV 垂直慢度
    qp2 = _qval(cp2)  # 覆盖层 P 垂直慢度

    def _coeff(q_sv, q_p, GG, lam, cs, cp, direction, wave_type):  # 单位速度幅值平面波的位移/应力系数（频率无关）
        sgn = 1.0 if direction == 'up' else -1.0  # 上行取正、下行取负
        if wave_type == 'SV':  # 处理 SV 波
            q = q_sv  # 选用 SV 垂直慢度
            ux_d = q / (1.0 / cs)  # 水平位移分量
            uy_d = -sgn * p_horiz / (1.0 / cs)  # 垂直位移分量
        else:  # 处理 P 波
            q = q_p  # 选用 P 垂直慢度
            ux_d = p_horiz / (1.0 / cp)  # 水平位移分量
            uy_d = sgn * q / (1.0 / cp)  # 垂直位移分量
        sig_xx = -(lam * (p_horiz * ux_d + sgn * q * uy_d) + 2.0 * GG * p_horiz * ux_d)  # σ_xx 系数
        sig_yy = -(lam * (p_horiz * ux_d + sgn * q * uy_d) + 2.0 * GG * sgn * q * uy_d)  # σ_yy 系数
        tau_xy = -GG * (sgn * q * ux_d + p_horiz * uy_d)  # τ_xy 系数
        return ux_d, uy_d, sig_xx, sig_yy, tau_xy  # 返回五个系数

    # 各波在界面参考面处的系数（相位取 0，频率无关，整段只算一次）
    ux_sv1u, uy_sv1u, sxx_sv1u, syy_sv1u, sxy_sv1u = _coeff(qs1, qp1, GG1, lam1, cs1, cp1, 'up', 'SV')  # 入射 SV 上行
    ux_sv1d, uy_sv1d, sxx_sv1d, syy_sv1d, sxy_sv1d = _coeff(qs1, qp1, GG1, lam1, cs1, cp1, 'down', 'SV')  # 反射 SV 下行
    ux_p1d, uy_p1d, sxx_p1d, syy_p1d, sxy_p1d = _coeff(qs1, qp1, GG1, lam1, cs1, cp1, 'down', 'P')  # 反射 P 下行
    ux_sv2u, uy_sv2u, sxx_sv2u, syy_sv2u, sxy_sv2u = _coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'up', 'SV')  # 透射 SV 上行
    ux_p2u, uy_p2u, sxx_p2u, syy_p2u, sxy_p2u = _coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'up', 'P')  # 透射 P 上行
    ux_sv2d, uy_sv2d, sxx_sv2d, syy_sv2d, sxy_sv2d = _coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'down', 'SV')  # 覆盖层 SV 下行
    ux_p2d, uy_p2d, sxx_p2d, syy_p2d, sxy_p2d = _coeff(qs2, qp2, GG2, lam2, cs2, cp2, 'down', 'P')  # 覆盖层 P 下行

    # 自由面/界面相位因子（随频率变化，向量化）
    phi_s2_up = np.exp(1j * omega * qs2 * h_overlying)  # SV2 从界面到自由面相位
    phi_p2_up = np.exp(1j * omega * qp2 * h_overlying)  # P2 从界面到自由面相位
    phi_s2_dn = np.conj(phi_s2_up)  # SV2 下行相位（自由面回界面）
    phi_p2_dn = np.conj(phi_p2_up)  # P2 下行相位

    # 自由面 2×2 反射：逐频率解析求逆（向量化）
    A00 = syy_sv2d * phi_s2_dn  # σ_yy=0 方程 SV 项
    A01 = syy_p2d * phi_p2_dn  # σ_yy=0 方程 P 项
    A10 = sxy_sv2d * phi_s2_dn  # τ_xy=0 方程 SV 项
    A11 = sxy_p2d * phi_p2_dn  # τ_xy=0 方程 P 项
    det = A00 * A11 - A01 * A10  # 2×2 行列式
    bad_surf = np.abs(det) < 1e-30  # 标记自由面退化 bin（det≈0），其响应最终置零（对齐 v6 零响应分支）
    det = np.where(bad_surf, 1.0, det)  # 退化 bin 用 1.0 占位避免 F 放大，最终统一清零
    iA00 = A11 / det  # 逆矩阵 (0,0)
    iA01 = -A01 / det  # 逆矩阵 (0,1)
    iA10 = -A10 / det  # 逆矩阵 (1,0)
    iA11 = A00 / det  # 逆矩阵 (1,1)
    Bsv0 = -syy_sv2u * phi_s2_up  # SV 上行对 σ_yy 贡献
    Bsv1 = -sxy_sv2u * phi_s2_up  # SV 上行对 τ_xy 贡献
    Bp0 = -syy_p2u * phi_p2_up  # P 上行对 σ_yy 贡献
    Bp1 = -sxy_p2u * phi_p2_up  # P 上行对 τ_xy 贡献
    Fsv0 = iA00 * Bsv0 + iA01 * Bsv1  # 自由面 SV 反射映射第一分量
    Fsv1 = iA10 * Bsv0 + iA11 * Bsv1  # 自由面 SV 反射映射第二分量
    Fp0 = iA00 * Bp0 + iA01 * Bp1  # 自由面 P 反射映射第一分量
    Fp1 = iA10 * Bp0 + iA11 * Bp1  # 自由面 P 反射映射第二分量

    # 覆盖层侧在界面处的叠加贡献（透射 + 自由面反射下行）
    cov_ux_sv = ux_sv2u + Fsv0 * ux_sv2d + Fsv1 * ux_p2d  # Tss 列对 ux 的贡献
    cov_uy_sv = uy_sv2u + Fsv0 * uy_sv2d + Fsv1 * uy_p2d  # Tss 列对 uy 的贡献
    cov_syy_sv = syy_sv2u + Fsv0 * syy_sv2d + Fsv1 * syy_p2d  # Tss 列对 σ_yy 的贡献
    cov_sxy_sv = sxy_sv2u + Fsv0 * sxy_sv2d + Fsv1 * sxy_p2d  # Tss 列对 τ_xy 的贡献
    cov_ux_p = ux_p2u + Fp0 * ux_sv2d + Fp1 * ux_p2d  # Tsp 列对 ux 的贡献
    cov_uy_p = uy_p2u + Fp0 * uy_sv2d + Fp1 * uy_p2d  # Tsp 列对 uy 的贡献
    cov_syy_p = syy_p2u + Fp0 * syy_sv2d + Fp1 * syy_p2d  # Tsp 列对 σ_yy 的贡献
    cov_sxy_p = sxy_p2u + Fp0 * sxy_sv2d + Fp1 * sxy_p2d  # Tsp 列对 τ_xy 的贡献

    # 界面连续条件 4×4 方程组（批量装配 + 批量求解全部频率）
    Nk = omega.shape[0]  # 频率个数
    A = np.zeros((Nk, 4, 4), dtype=complex)  # 批量系数矩阵
    A[:, 0, 0] = ux_sv1d; A[:, 1, 0] = uy_sv1d; A[:, 2, 0] = sxy_sv1d; A[:, 3, 0] = syy_sv1d  # Rss 列
    A[:, 0, 1] = ux_p1d; A[:, 1, 1] = uy_p1d; A[:, 2, 1] = sxy_p1d; A[:, 3, 1] = syy_p1d  # Rsp 列
    A[:, 0, 2] = -cov_ux_sv; A[:, 1, 2] = -cov_uy_sv; A[:, 2, 2] = -cov_sxy_sv; A[:, 3, 2] = -cov_syy_sv  # Tss 列
    A[:, 0, 3] = -cov_ux_p; A[:, 1, 3] = -cov_uy_p; A[:, 2, 3] = -cov_sxy_p; A[:, 3, 3] = -cov_syy_p  # Tsp 列
    B = np.empty((Nk, 4, 1), dtype=complex)  # 批量右端向量
    B[:, 0, 0] = -ux_sv1u; B[:, 1, 0] = -uy_sv1u; B[:, 2, 0] = -sxy_sv1u; B[:, 3, 0] = -syy_sv1u  # 入射 SV 上行移至右端
    dc_mask = np.abs(omega) < 1e-20  # 直流分量掩码（omega≈0，v6 原本跳过不求解）
    if np.any(dc_mask):  # 存在直流分量时
        A[dc_mask] = np.eye(4, dtype=complex)  # 将 DC bin 方程置为单位阵，避免其奇异使整批求解崩溃
        B[dc_mask, :, 0] = 0.0  # DC bin 右端置零（其响应后续统一清零）
    try:  # 优先批量求解全部频率
        X = np.linalg.solve(A, B)[:, :, 0]  # 批量求解 4×4 方程组
        sing_mask = np.zeros(Nk, dtype=bool)  # 批量成功则无奇异 bin
    except np.linalg.LinAlgError:  # 某些 bin 奇异时回退逐 bin 求解（对齐 v6 的 except 零响应）
        X = np.zeros((Nk, 4), dtype=complex)  # 初始化解数组
        sing_mask = np.zeros(Nk, dtype=bool)  # 初始化奇异 bin 掩码
        for k in range(Nk):  # 逐频率求解
            try:  # 尝试求解单个 bin
                X[k] = np.linalg.solve(A[k], B[k, :, 0])  # 求解该 bin 的 4×4
            except np.linalg.LinAlgError:  # 该 bin 奇异
                sing_mask[k] = True  # 标记为奇异，其响应置零
    Rss = X[:, 0]; Rsp = X[:, 1]; Tss = X[:, 2]; Tsp = X[:, 3]  # 提取四个界面系数（频率数组）
    a_sv2 = Fsv0 * Tss + Fp0 * Tsp  # 覆盖层内 SV 下行幅值（界面处）
    a_p2 = Fsv1 * Tss + Fp1 * Tsp  # 覆盖层内 P 下行幅值（界面处）

    # 目标节点处响应（分支由 y_target 决定，对全频率一致）
    if y_target <= y_intf + 1e-4:  # 基岩层节点
        dy = y_target - y_bottom  # 节点相对底部高度
        phi_inc = np.exp(1j * omega * qs1 * dy)  # 入射 SV 相位
        phi_ref = np.exp(1j * omega * qs1 * (2.0 * h_bedrock - dy))  # 反射 SV 相位
        phi_pref = np.exp(1j * omega * qp1 * (2.0 * h_bedrock - dy))  # 反射 P 相位
        ux = ux_sv1u * phi_inc + Rss * ux_sv1d * phi_ref + Rsp * ux_p1d * phi_pref  # ux 频域响应
        uy = uy_sv1u * phi_inc + Rss * uy_sv1d * phi_ref + Rsp * uy_p1d * phi_pref  # uy 频域响应
        sxx = sxx_sv1u * phi_inc + Rss * sxx_sv1d * phi_ref + Rsp * sxx_p1d * phi_pref  # σ_xx 频域响应
        syy = syy_sv1u * phi_inc + Rss * syy_sv1d * phi_ref + Rsp * syy_p1d * phi_pref  # σ_yy 频域响应
        sxy = sxy_sv1u * phi_inc + Rss * sxy_sv1d * phi_ref + Rsp * sxy_p1d * phi_pref  # τ_xy 频域响应
    else:  # 覆盖层节点
        dyi = y_target - y_intf  # 节点在覆盖层中的高度
        p_su = np.exp(1j * omega * qs2 * dyi)  # SV2 上行相位
        p_sd = np.exp(-1j * omega * qs2 * dyi)  # SV2 下行相位
        p_pu = np.exp(1j * omega * qp2 * dyi)  # P2 上行相位
        p_pd = np.exp(-1j * omega * qp2 * dyi)  # P2 下行相位
        ux = Tss * ux_sv2u * p_su + a_sv2 * ux_sv2d * p_sd + Tsp * ux_p2u * p_pu + a_p2 * ux_p2d * p_pd  # ux 频域响应
        uy = Tss * uy_sv2u * p_su + a_sv2 * uy_sv2d * p_sd + Tsp * uy_p2u * p_pu + a_p2 * uy_p2d * p_pd  # uy 频域响应
        sxx = Tss * sxx_sv2u * p_su + a_sv2 * sxx_sv2d * p_sd + Tsp * sxx_p2u * p_pu + a_p2 * sxx_p2d * p_pd  # σ_xx 频域响应
        syy = Tss * syy_sv2u * p_su + a_sv2 * syy_sv2d * p_sd + Tsp * syy_p2u * p_pu + a_p2 * syy_p2d * p_pd  # σ_yy 频域响应
        sxy = Tss * sxy_sv2u * p_su + a_sv2 * sxy_sv2d * p_sd + Tsp * sxy_p2u * p_pu + a_p2 * sxy_p2d * p_pd  # τ_xy 频域响应

    # 组装"每单位输入速度谱"传递函数：位移=响应×水平相位/(iω)，应力=响应×水平相位
    phase_x = np.exp(1j * omega * p_horiz * x_target)  # 水平传播相位（Snell）
    inv_iw = np.zeros_like(omega, dtype=complex)  # 1/(iω)，DC 处保持 0
    nz = np.abs(omega) >= 1e-20  # 非直流频率掩码
    inv_iw[nz] = 1.0 / (1j * omega[nz])  # 仅非零频率求 1/(iω)
    T_ux = ux * phase_x * inv_iw  # ux 传递函数
    T_uy = uy * phase_x * inv_iw  # uy 传递函数
    T_dotux = 1j * omega * T_ux  # x 向速度传递函数（= iω·位移）
    T_dotuy = 1j * omega * T_uy  # y 向速度传递函数
    T_sxx = sxx * phase_x  # σ_xx 传递函数
    T_syy = syy * phase_x  # σ_yy 传递函数
    T_sxy = sxy * phase_x  # τ_xy 传递函数
    zero_mask = (~nz) | bad_surf | sing_mask  # 需置零的 bin：直流 + 自由面退化 + 4×4 奇异（对齐 v6 零响应）
    for arr in (T_ux, T_uy, T_dotux, T_dotuy, T_sxx, T_syy, T_sxy):  # 遍历各传递函数
        arr[zero_mask] = 0.0  # 清零这些 bin（DC 静态 + 退化/奇异回退零响应）

    transfer = {'ux': T_ux, 'uy': T_uy, 'dotux': T_dotux, 'dotuy': T_dotuy,  # 打包传递函数字典
                'sxx': T_sxx, 'syy': T_syy, 'sxy': T_sxy}  # 应力分量传递函数
    _FF_TRANSFER_CACHE[key] = transfer  # 写入缓存供跨记录复用
    return transfer  # 返回传递函数


def _compute_freefield_at_node(y_target, x_target, mat_bedrock, mat_overlying,
                                h_bedrock, h_overlying, y_bottom,
                                p_horiz, vel_freq, freq_arr, dt, N_fft):
    """计算单节点自由场位移/速度/应力时程（向量化传播矩阵 + 传递函数缓存）。

    [输入幅值约定（#5）] vel_freq 视为"基底入射上行 SV 波"幅值 E 的频谱，
      自由岩面运动 = 2E（自由面效应），TAF 取比值时该归一化精确抵消。
    返回 dict：'ux','uy','dotux','dotuy','sxx','syy','sxy'，均为长度 N_fft 的时域数组。
    """
    T = _freefield_transfer(y_target, x_target, mat_bedrock, mat_overlying,  # 取（缓存的）频域传递函数
                            h_bedrock, h_overlying, y_bottom,  # 传入几何
                            p_horiz, freq_arr, dt, N_fft)  # 传入慢度与频率轴
    return {  # 频域 = 传递函数 × 输入速度谱，再 IFFT 回时域
        'ux': np.fft.irfft(T['ux'] * vel_freq, n=N_fft),  # x 向位移时程
        'uy': np.fft.irfft(T['uy'] * vel_freq, n=N_fft),  # y 向位移时程
        'dotux': np.fft.irfft(T['dotux'] * vel_freq, n=N_fft),  # x 向速度时程
        'dotuy': np.fft.irfft(T['dotuy'] * vel_freq, n=N_fft),  # y 向速度时程
        'sxx': np.fft.irfft(T['sxx'] * vel_freq, n=N_fft),  # σ_xx 时程
        'syy': np.fft.irfft(T['syy'] * vel_freq, n=N_fft),  # σ_yy 时程
        'sxy': np.fft.irfft(T['sxy'] * vel_freq, n=N_fft),  # τ_xy 时程
    }  # 结束返回字典


# ==========================================================
#  建模（几何/材料/网格）
# ==========================================================


def create_model(site, geom, mesh_size, cae_name=None, logger=None):
    """创建二维平面应变斜坡模型：几何、材料、截面、装配、网格（不含分析步）。

    site: Site 对象（两层材料 + 基岩厚度）
    geom: Geometry 对象（斜坡几何，含派生量）
    """  # 说明函数用途与参数
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    model_name = 'Model-1'  # 设置基础模型名称

    total_L = geom.total_L  # 读取模型总长度
    H_lower = geom.H_lower  # 读取坡脚地表高度
    H_upper = geom.H_upper  # 读取坡顶地表高度
    H_minus_h = geom.H_minus_h  # 读取斜坡高度差（用于坡面顶点识别）
    w_slope = geom.w_slope  # 读取坡面水平长度
    left_flat = geom.left_flat  # 读取左平台长度
    bedrock_thickness = geom.bedrock_thickness  # 读取基岩层厚度

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

    EE_bedrock = _compute_elastic_modulus_from_wave_speed(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 计算基岩弹性模量
    EE_overlying = _compute_elastic_modulus_from_wave_speed(site.overlying.cs, site.overlying.vv, site.overlying.density)  # 计算覆盖层弹性模量

    mat_bedrock_name = _next_available_name('Material-Bedrock', model.materials)  # 生成基岩材料名
    mat_bedrock = model.Material(name=mat_bedrock_name)  # 创建基岩材料
    mat_bedrock.Elastic(table=((EE_bedrock, site.bedrock.vv),))  # 定义基岩弹性参数
    mat_bedrock.Density(table=((site.bedrock.density,),))  # 定义基岩密度

    mat_overlying_name = _next_available_name('Material-Overlying', model.materials)  # 生成覆盖层材料名
    mat_overlying = model.Material(name=mat_overlying_name)  # 创建覆盖层材料
    mat_overlying.Elastic(table=((EE_overlying, site.overlying.vv),))  # 定义覆盖层弹性参数
    mat_overlying.Density(table=((site.overlying.density,),))  # 定义覆盖层密度

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


def create_flat_model(site, geom, mesh_size, logger=None):
    """创建二维平面应变平坦自由场模型：矩形几何、材料、截面、装配与网格。

    site: Site 对象（两层材料 + 基岩厚度）
    geom: Geometry 对象（取其总长、平坦总高 H_flat、基岩厚度）
    """  # 说明函数用途与参数
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    model_name = 'Model-2'  # 设置平坦自由场模型名称

    total_L = geom.total_L  # 读取模型总长度
    H_flat = geom.H_flat  # 读取平坦场地总高度
    bedrock_thickness = geom.bedrock_thickness  # 读取基岩层厚度

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

    EE_bedrock = _compute_elastic_modulus_from_wave_speed(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 计算基岩弹性模量
    EE_overlying = _compute_elastic_modulus_from_wave_speed(site.overlying.cs, site.overlying.vv, site.overlying.density)  # 计算覆盖层弹性模量

    mat_bedrock_name = _next_available_name('Material-Bedrock', model.materials)  # 生成基岩材料名
    mat_bedrock = model.Material(name=mat_bedrock_name)  # 创建基岩材料
    mat_bedrock.Elastic(table=((EE_bedrock, site.bedrock.vv),))  # 定义基岩弹性参数
    mat_bedrock.Density(table=((site.bedrock.density,),))  # 定义基岩密度

    mat_overlying_name = _next_available_name('Material-Overlying', model.materials)  # 生成覆盖层材料名
    mat_overlying = model.Material(name=mat_overlying_name)  # 创建覆盖层材料
    mat_overlying.Elastic(table=((EE_overlying, site.overlying.vv),))  # 定义覆盖层弹性参数
    mat_overlying.Density(table=((site.overlying.density,),))  # 定义覆盖层密度

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


# ==========================================================
#  人工边界 VAB（弹簧-阻尼器 + 等效节点力）
# ==========================================================


def _make_boundary_nodes(nodes, sort_axis, ascending, pick_material, ymax):  # 定义构建边界节点列表的函数
    """对一条边界的实例节点排序、计算影响长度与弹簧/阻尼系数，返回 BoundaryNode 列表。

    nodes       : Abaqus 实例节点序列
    sort_axis   : 'x' 或 'y'，沿该轴排序并据相邻间距求影响长度
    ascending   : 是否升序（底边升序、侧边降序，沿用 v6 行为）
    pick_material: 函数 (x, y) -> 材料参数 dict，用于按节点所在层取系数
    ymax        : 弹簧刚度公式中的参考长度 R
    """  # 说明函数用途与参数
    arr = np.array([[node.label, node.coordinates[0], node.coordinates[1]] for node in nodes], dtype=float)  # 生成节点数据表
    axis = 1 if sort_axis == 'x' else 2  # 根据排序轴选择坐标列
    arr = arr[arr[:, axis].argsort()]  # 按指定坐标排序
    if not ascending:  # 判断是否需要倒序
        arr = arr[::-1]  # 反转排序结果

    n = arr.shape[0]  # 统计节点数量
    if n == 1:  # 处理单节点情况
        influence = np.array([0.0])  # 单节点影响长度设为零
    else:  # 处理多节点情况
        coord = arr[:, axis]  # 提取排序坐标
        influence = np.empty(n)  # 创建影响长度数组
        influence[0] = abs(coord[0] - coord[1]) / 2.0  # 计算首节点影响长度
        influence[-1] = abs(coord[-1] - coord[-2]) / 2.0  # 计算末节点影响长度
        if n > 2:  # 处理中间节点
            influence[1:-1] = np.abs(coord[:-2] - coord[2:]) / 2.0  # 计算中间节点影响长度

    result = []  # 初始化边界节点列表
    for idx in range(n):  # 遍历每个节点
        x0 = arr[idx, 1]  # 读取节点 x 坐标
        y0 = arr[idx, 2]  # 读取节点 y 坐标
        inf = influence[idx]  # 读取节点影响长度
        mat = pick_material(x0, y0)  # 按节点位置选择材料参数
        kn = mat['GG'] / 2.0 / ymax * inf  # 计算法向刚度
        cn = mat['density'] * mat['cp'] * inf  # 计算法向阻尼
        kt = mat['GG'] / 4.0 / ymax * inf  # 计算切向刚度
        ct = mat['density'] * mat['cs'] * inf  # 计算切向阻尼
        result.append(BoundaryNode(label=int(arr[idx, 0]), x=x0, y=y0, influence=inf,  # 组装边界节点
                                   kn=kn, cn=cn, kt=kt, ct=ct))  # 填入弹簧/阻尼系数
    return result  # 返回边界节点列表


def _add_spring_dashpots(assembly, instance, nodes_by_boundary, model_name, logger):  # 定义施加弹簧阻尼器的函数
    """为三条边界的所有节点创建接地弹簧-阻尼器（法向 + 切向）。"""  # 说明函数用途
    boundary_dof = {'l': (1, 2), 'r': (1, 2), 'b': (2, 1)}  # 各边界 (法向自由度, 切向自由度)
    for boundary in BOUNDARY_SEQUENCE:  # 按边界顺序处理
        dof_n, dof_t = boundary_dof[boundary]  # 读取当前边界自由度配置
        for bn in nodes_by_boundary[boundary]:  # 遍历该边界每个节点
            node_array = instance.nodes.sequenceFromLabels([bn.label])  # 通过标签获取实例节点
            if len(node_array) == 0:  # 判断节点是否存在
                logger.warning('创建弹簧-阻尼器时，实例中不存在节点 %d', bn.label)  # 输出警告日志
                continue  # 跳过当前节点
            region = Region(nodes=node_array)  # 构建节点区域
            assembly.engineeringFeatures.SpringDashpotToGround(  # 创建法向弹簧阻尼器
                name='SpringDashpot_{}_{}_normal'.format(boundary, bn.label),  # 设置法向元件名称
                region=region, orientation=None, dof=dof_n,  # 设置区域和自由度
                springBehavior=ON, springStiffness=bn.kn,  # 设置弹簧行为和刚度
                dashpotBehavior=ON, dashpotCoefficient=bn.cn)  # 设置阻尼行为和阻尼系数
            assembly.engineeringFeatures.SpringDashpotToGround(  # 创建切向弹簧阻尼器
                name='SpringDashpot_{}_{}_tangent'.format(boundary, bn.label),  # 设置切向元件名称
                region=region, orientation=None, dof=dof_t,  # 设置区域和自由度
                springBehavior=ON, springStiffness=bn.kt,  # 设置弹簧行为和刚度
                dashpotBehavior=ON, dashpotCoefficient=bn.ct)  # 设置阻尼行为和阻尼系数
    log_step(logger, '%s 弹簧-阻尼器创建完成', model_name)  # 记录创建完成日志


def _build_equivalent_forces(nodes_by_boundary, ctx):  # 定义计算等效节点力时程的函数
    """逐边界逐节点计算等效节点力时程，返回 {'<label>-<边界>-fx/fy': Nx2 数组}。

    等效力 = K·u_ff + C·v̇_ff + A·(σ_ff·n_外)，n 为各边界外法向：
      左 n=(-1,0): tx=-σxx, ty=-τxy；右 n=(+1,0): tx=+σxx, ty=+τxy；
      底 n=(0,-1): tx=-τxy,  ty=-σyy。
    角点处理（v6）：左下/右下角点同属侧边与底边两个集合，会各算一次并叠加——
      这是 VAB 角点的标准正确处理（形函数横跨两个边界单元，面力/弹簧/阻尼本应叠加），
      不可折半。Abaqus 中同节点多个 ConcentratedForce 相加、多个接地弹簧并联相加。
    """  # 说明函数用途与外法向/角点约定
    field_data = {}  # 初始化等效力缓存字典
    geom = ctx.geom  # 取出几何对象
    No = ctx.N_orig  # 原始时程长度
    t_arr = ctx.time_arr[:No]  # 截取时间轴
    for boundary in BOUNDARY_SEQUENCE:  # 遍历每个边界
        for bn in nodes_by_boundary[boundary]:  # 遍历该边界每个节点
            # 确定当前节点正上方柱子的覆盖层有效厚度（#2：底边按 x 取值）
            if boundary == 'l':  # 左边界
                h_ov = max(0.0, ctx.ymax_l - geom.bedrock_thickness)  # 左边界覆盖层厚度
            elif boundary == 'r':  # 右边界
                h_ov = max(0.0, ctx.ymax_r - geom.bedrock_thickness)  # 右边界覆盖层厚度
            else:  # 底边界
                surf_y = _surface_y_at(bn.x, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 该底节点正上方地表高度
                h_ov = max(0.0, surf_y - geom.bedrock_thickness)  # 对应覆盖层厚度（坡顶厚、坡脚薄、坡面过渡）

            ff = _compute_freefield_at_node(  # 调用自由场引擎计算该节点时程
                y_target=bn.y, x_target=bn.x,  # 目标节点坐标
                mat_bedrock=ctx.mat_bedrock, mat_overlying=ctx.mat_overlying,  # 两层材料参数
                h_bedrock=geom.bedrock_thickness, h_overlying=h_ov,  # 基岩厚 + 覆盖层有效厚
                y_bottom=ctx.ymin, p_horiz=ctx.p_horiz,  # 底边 y 与水平慢度
                vel_freq=ctx.vel_freq, freq_arr=ctx.freq_arr, dt=ctx.dt, N_fft=ctx.N_fft)  # 频域输入

            ux = ff['ux'][:No]  # 截取 x 向位移
            uy = ff['uy'][:No]  # 截取 y 向位移
            dotux = ff['dotux'][:No]  # 截取 x 向速度
            dotuy = ff['dotuy'][:No]  # 截取 y 向速度
            sxx = ff['sxx'][:No]  # 截取 σ_xx
            syy = ff['syy'][:No]  # 截取 σ_yy
            sxy = ff['sxy'][:No]  # 截取 τ_xy

            if boundary == 'l':  # 左边界，外法向 n=(-1,0)，x 为法向、y 为切向
                fx = bn.kn * ux + bn.cn * dotux + bn.influence * (-sxx)  # x 向等效力
                fy = bn.kt * uy + bn.ct * dotuy + bn.influence * (-sxy)  # y 向等效力
            elif boundary == 'r':  # 右边界，外法向 n=(+1,0)，x 为法向、y 为切向
                fx = bn.kn * ux + bn.cn * dotux + bn.influence * (sxx)  # x 向等效力
                fy = bn.kt * uy + bn.ct * dotuy + bn.influence * (sxy)  # y 向等效力
            else:  # 底边界，外法向 n=(0,-1)，x 为切向、y 为法向
                fx = bn.kt * ux + bn.ct * dotux + bn.influence * (-sxy)  # x 向等效力（切向弹簧）
                fy = bn.kn * uy + bn.cn * dotuy + bn.influence * (-syy)  # y 向等效力（法向弹簧）

            field_data['{}-{}-fx'.format(bn.label, boundary)] = np.column_stack((t_arr, fx))  # 缓存 x 向力时程
            field_data['{}-{}-fy'.format(bn.label, boundary)] = np.column_stack((t_arr, fy))  # 缓存 y 向力时程
    return field_data  # 返回等效力缓存


def _apply_amplitudes_and_loads(model_name, inst_name, nodes_by_boundary, field_data, step_name, logger):  # 定义施加幅值与集中力的函数
    """为每个边界节点创建幅值曲线（TabularAmplitude）并施加 x/y 向集中力。"""  # 说明函数用途
    model = mdb.models[model_name]  # 获取目标模型
    nodes = model.rootAssembly.instances[inst_name].nodes  # 获取实例节点集合
    for boundary in BOUNDARY_SEQUENCE:  # 遍历每个边界
        for bn in nodes_by_boundary[boundary]:  # 遍历该边界每个节点
            fx_arr = field_data['{}-{}-fx'.format(bn.label, boundary)]  # 读取 x 向力时程
            fy_arr = field_data['{}-{}-fy'.format(bn.label, boundary)]  # 读取 y 向力时程
            name_amp_fx = 'AMP-{}-{}-fx'.format(bn.label, boundary)  # 生成 x 向幅值名
            name_amp_fy = 'AMP-{}-{}-fy'.format(bn.label, boundary)  # 生成 y 向幅值名
            model.TabularAmplitude(data=tuple(tuple(row) for row in fx_arr),  # 创建 x 向幅值曲线
                                   name=name_amp_fx, smooth=SOLVER_DEFAULT, timeSpan=STEP)  # 设置平滑与时间跨度
            model.TabularAmplitude(data=tuple(tuple(row) for row in fy_arr),  # 创建 y 向幅值曲线
                                   name=name_amp_fy, smooth=SOLVER_DEFAULT, timeSpan=STEP)  # 设置平滑与时间跨度
            node_array = nodes.sequenceFromLabels([bn.label])  # 按标签查找实例节点
            if len(node_array) == 0:  # 判断节点是否存在
                logger.warning('施加载荷时，实例中不存在节点 %d (实例: %s)', bn.label, inst_name)  # 输出警告日志
                continue  # 跳过当前节点
            region = Region(nodes=node_array)  # 构建节点区域
            model.ConcentratedForce(name='load-{}-{}-fx'.format(bn.label, boundary),  # 创建 x 向集中力
                                    createStepName=step_name, region=region, cf1=1.0, amplitude=name_amp_fx,  # 设置分析步/区域/幅值
                                    distributionType=UNIFORM, field='', localCsys=None)  # 设置载荷分布
            model.ConcentratedForce(name='load-{}-{}-fy'.format(bn.label, boundary),  # 创建 y 向集中力
                                    createStepName=step_name, region=region, cf2=1.0, amplitude=name_amp_fy,  # 设置分析步/区域/幅值
                                    distributionType=UNIFORM, field='', localCsys=None)  # 设置载荷分布
    log_step(logger, '%s 幅值曲线与集中力已创建', model_name)  # 记录完成日志


def VAB_oblique(site, geom, angle,
                model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None):
    """为二维模型施加粘弹性人工边界（弹簧-阻尼器）与斜入射 SV 波等效节点力。

    site: Site 对象（两层材料 + 基岩厚度）
    geom: Geometry 对象（几何，含 H_upper/H_lower/left_flat/w_slope/bedrock_thickness）
    angle: SV 波入射角（度）
    """  # 说明函数用途与参数
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
    mat_bedrock = _compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 计算基岩材料参数
    mat_overlying = _compute_material_params(site.overlying.cs, site.overlying.vv, site.overlying.density)  # 计算覆盖层材料参数

    # 获取模型尺寸（左/右边界最高点与底边 y）
    l_nodes = get_instance_nodes_from_part_set('Left_boundary')  # 获取左边界节点
    ymax_l = max(l_nodes, key=lambda node: node.coordinates[1]).coordinates[1]  # 记录左边界最高 y 坐标

    b_nodes = get_instance_nodes_from_part_set('Bottom_boundary')  # 获取底边节点
    ymin = b_nodes[0].coordinates[1]  # 记录底边 y 坐标

    r_nodes = get_instance_nodes_from_part_set('Right_boundary')  # 获取右边界节点
    ymax_r = max(r_nodes, key=lambda node: node.coordinates[1]).coordinates[1]  # 记录右边界最高 y 坐标

    ymax = max(ymax_l, ymax_r)  # 取左右边界最高点中的较大值（弹簧刚度参考长度 R）

    # 按节点所在材质层选择材料参数（基岩/覆盖层）
    def pick_material(x_coord, y_coord):  # 定义按节点坐标选择材料的函数
        if y_coord < geom.bedrock_thickness + 1e-4:  # 判断节点是否位于基岩层
            return mat_bedrock  # 返回基岩材料参数
        return mat_overlying  # 否则返回覆盖层材料参数

    # 构建三条边界的节点列表（含影响长度与弹簧-阻尼系数）
    nodes_by_boundary = {  # 各边界 -> BoundaryNode 列表
        'l': _make_boundary_nodes(l_nodes, 'y', False, pick_material, ymax),  # 左边界（沿 y 降序）
        'r': _make_boundary_nodes(r_nodes, 'y', False, pick_material, ymax),  # 右边界（沿 y 降序）
        'b': _make_boundary_nodes(b_nodes, 'x', True, pick_material, ymax),  # 底边界（沿 x 升序）
    }  # 结束节点列表构建
    log_step(logger, '%s 边界节点影响长度与弹簧-阻尼系数已计算', model_name)  # 记录计算日志

    _add_spring_dashpots(assembly, instance, nodes_by_boundary, model_name, logger)  # 施加接地弹簧-阻尼器

    # ============ 入射角处理 ============
    if angle == 0:  # 判断入射角是否为零
        angle = 1e-10  # 用极小角度替代零角度
    else:  # 处理非零角度
        angle = round(angle, 4)  # 保留四位小数
    alpha1 = math.radians(angle)  # 将角度转换为弧度
    p_horiz = math.sin(alpha1) / site.bedrock.cs  # 计算水平慢度（Snell 定律，所有层共享）
    log_step(logger, '%s 水平慢度 p = %.8f s/m', model_name, p_horiz)  # 记录水平慢度

    # ============ 读取加速度时程并积分 ============
    # [输入幅值约定（#5）] 输入加速度记录积分得到的速度，被当作"基底入射上行 SV 波"
    #   幅值 E（见 _compute_freefield_at_node）。自由岩面对应 2E（自由面效应），
    #   故模型地表运动 ≈ 2×输入记录，属正确物理而非误差。
    #   TAF = PGA_slope / PGA_flat 相对平坦自由场取比值，线性分析下此归一化在比值中
    #   抵消、对 TAF 无影响；仅"绝对 PGA"引用时需按 E / 2E 约定换算（如视记录为
    #   基岩露头 2E，则应取入射 = 记录/2）。当前实现按"记录 = 入射波 E"。
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
    vel, _vel_slope = _integrate_acc_to_velocity(acc, dt, time_arr)  # 调用纯函数完成积分与基线校正
    log_step(logger, '%s 速度基线校正完成: 去趋势斜率=%.3e', model_name, _vel_slope)  # 记录基线校正日志

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
    ctx = FreeFieldCtx(  # 打包等效力计算所需上下文
        mat_bedrock=mat_bedrock, mat_overlying=mat_overlying, geom=geom,  # 材料与几何
        ymax_l=ymax_l, ymax_r=ymax_r, ymin=ymin,  # 边界高度信息
        p_horiz=p_horiz, vel_freq=vel_freq, freq_arr=freq_arr, dt=dt,  # 输入波场
        N_fft=N_fft, N_orig=N_orig, time_arr=time_arr)  # 时频长度与时间轴
    field_data = _build_equivalent_forces(nodes_by_boundary, ctx)  # 计算所有边界节点等效力时程
    log_step(logger, '%s 所有边界等效节点力计算完成', model_name)  # 记录计算完成日志

    # ============ 创建幅值曲线并施加集中力 ============
    _apply_amplitudes_and_loads(model_name, inst_name, nodes_by_boundary, field_data, step_name, logger)  # 施加幅值与载荷
    mdb.save()  # 保存模型数据库
    log_step(logger, '%s 粘弹性人工边界完成: 耗时=%.2fs', model_name, time.time() - t0)  # 记录结束耗时


# ==========================================================
#  批量建模与作业提交
# ==========================================================


def build_models(acc_info, base_model, part_name, inst_name,
                 site, geom, angle, job,
                 step_name=DEFAULT_STEP_NAME, model_scene='slope', logger=None):
    """根据加速度时程信息批量复制模型、创建分析步、施加人工边界。

    site/geom: 场地材料与几何对象（直接转发给 VAB_oblique）
    angle    : SV 波入射角（度）
    job      : 作业配置 dict，读取 'variables'（场输出变量）与 'frequency'（输出频率）
    """  # 说明函数用途与参数
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器

    variables = _normalize_output_variables(job['variables'])  # 规范化场输出变量列表
    frequency = job['frequency']  # 读取场输出频率

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

        VAB_oblique(site=site, geom=geom, angle=angle,  # 调用人工边界构建函数（传入场地与几何对象）
                    model_name=new_model_name, part_name=part_name, inst_name=inst_name,  # 传入模型/零件/实例名称
                    acc_file=acc_file, step_name=step_name, logger=logger)  # 传入加速度文件、分析步与日志器
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


# ==========================================================
#  主入口
# ==========================================================


def main():
    """脚本主入口：组织参数、建模、施加边界并提交作业。"""  # 说明主入口用途
    logger = log_step('VAB_oblique_TAF_double.log')  # 初始化日志并写入当前版本日志文件
    total_start = time.time()  # 记录主流程起始时间
    _FF_TRANSFER_CACHE.clear()  # 清空上一次运行残留的自由场传递函数缓存，避免跨运行内存累积

    try:
        log_step(logger, '脚本开始执行')  # 写入脚本启动日志

        # 波动参数计算
        cs_bedrock = _compute_wave_speed_from_elastic_modulus(
            material_cfg['bedrock']['elastic_modulus'],
            material_cfg['bedrock']['poisson_ratio'],
            material_cfg['bedrock']['density']
        )
        cs_overlying = cs_bedrock / material_cfg['overlying']['velocity_ratio']  # 计算覆盖层剪切波速

        # 打包场地材料与斜坡几何为结构化对象（取代散标量透传）
        site = Site(  # 构建双层场地对象
            bedrock=Material(cs=cs_bedrock,  # 基岩剪切波速
                             vv=material_cfg['bedrock']['poisson_ratio'],  # 基岩泊松比
                             density=material_cfg['bedrock']['density']),  # 基岩密度（构成基岩材料）
            overlying=Material(cs=cs_overlying,  # 覆盖层剪切波速
                               vv=material_cfg['overlying']['poisson_ratio'],  # 覆盖层泊松比
                               density=material_cfg['overlying']['density']),  # 覆盖层密度（构成覆盖层材料）
            bedrock_thickness=geometry_cfg['bedrock_thickness'])  # 基岩层厚度

        geom = make_geometry(  # 构建斜坡几何对象（含全部派生量）
            total_L=geometry_cfg['total_L'],  # 模型总长度
            H_minus_h=geometry_cfg['H_minus_h'],  # 斜坡高度差
            i=geometry_cfg['i'],  # 斜坡倾角
            h_over_H=geometry_cfg['h_over_H'],  # 深度比 h/H
            left_flat=geometry_cfg['left_flat'],  # 上平台长度
            bedrock_thickness=geometry_cfg['bedrock_thickness'])  # 基岩层厚度
        geom_flat = make_flat_geometry(geom)  # 派生平坦自由场几何

        cae_name = 'h{}_i{}_a{}.cae'.format(int(geom.H_minus_h), int(geom.i), int(material_cfg['angle']))  # 生成工程文件名
        acc_info = find_acc_txt(logger)  # 读取当前目录内全部加速度时程信息

        base_model, part_name, inst_name = create_model(  # 创建斜坡基础几何与网格模型
            site=site, geom=geom, mesh_size=mesh_size, cae_name=cae_name, logger=logger)  # 传入场地/几何/网格/文件名

        flat_base_model, flat_part_name, flat_inst_name = create_flat_model(  # 创建平坦自由场基础模型
            site=site, geom=geom, mesh_size=mesh_size, logger=logger)  # 传入场地/几何/网格

        slope_model_names = build_models(  # 批量复制斜坡模型并施加等效边界
            acc_info=acc_info, base_model=base_model, part_name=part_name, inst_name=inst_name,  # 地震动信息与基础模型/零件/实例
            site=site, geom=geom, angle=material_cfg['angle'],  # 场地、斜坡几何与入射角
            job=job_cfg, model_scene='slope', logger=logger)  # 作业配置与斜坡场景标签

        flat_model_names = build_models(  # 批量复制平坦自由场模型并施加等效边界
            acc_info=acc_info, base_model=flat_base_model, part_name=flat_part_name, inst_name=flat_inst_name,  # 地震动信息与平坦基础模型/零件/实例
            site=site, geom=geom_flat, angle=material_cfg['angle'],  # 场地、平坦几何与入射角
            job=job_cfg, model_scene='flat', logger=logger)  # 作业配置与平坦场景标签

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


if __name__ == '__main__':  # 判断是否直接运行脚本
    main()  # 调用主入口函数
