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
import io  # 导入 io 模块（提供 Py2/Py3 通用、可指定编码的文件接口，用于写 case_meta.json）
import json  # 导入 JSON 模块（写出 case_meta.json）
import sys  # 导入系统模块（用于 Py2/Py3 版本判断）
import time  # 导入时间模块
import logging  # 导入日志模块
import traceback  # 导入异常堆栈模块
from collections import namedtuple  # 导入命名元组用于参数打包


# ==========================================================
#  配置参数（实验参数集中在此处修改）
# ==========================================================


# ==========================================================
#  【本版 = v8：research_plan_v1.md §3.1 (A1~A5) 落地核对与频域框架(C1)前置支撑】
#
#  §3.1 状态核对（详见 ML/research_plan_v1.md 更新版）：
#    A1 边界系数逐节点取局部材料   → v4 起已实现（v7 升级为按局部埋深落带），本版无改动；
#    A2 层状斜入射自由场（频域）   → v6 fd 引擎（全局矩阵法≡Thomson-Haskell）已内联单文件实现，
#                                    本版补【临界角校验】硬化（≥临界角拒绝建模，>30° 告警）；
#    A3 网格按最软层（K-L 判据）   → v5 起已实现（mesh_cfg.auto），本版无改动；
#    A4 阻尼分层赋值              → v3 起已实现；本版新增 damping_cfg['anchor']='dual'
#                                    双控锚定：瑞利拟合下限取 min(f1_factor·fc, 场地基频 f_site)；
#    A5 三段式验证协议            → 本版新增 _fd_engine_selfcheck 建模前自动解析对拍
#                                    （半空间退化 + 单层 SH 解析，误差>1e-3 即中止），结果写 meta。
#
#  频域框架（C1）前置支撑（全部单文件内完成，Abaqus Python 2.7 兼容）：
#    ① time_cfg['tail_seconds']：分析步时长与 fd 自由场时窗同步延长"静默尾段"，
#       供宽频脉冲提取传递函数 H(f) 时捕捉坡体混响衰减（默认 0 = 与 v7 行为完全一致）；
#    ② run_cfg['surface_only']：新增 TOP_SURFACE 节点集 A/U 全时程场输出 + 整体场输出降频，
#       ODB 体积骤降，服务"地表全时程提取"数据管线（默认 False = 与 v7 行为一致）；
#    ③ 旧版 numpy 硬化：lstsq rcond=-1（兼容 Abaqus Py2.7 自带 numpy）。
#
#  ▼ 以下为 v7 原说明（机制全部保留）：
# ==========================================================
# ==========================================================
#  【本版 = v7：按 Modeling/Multi/plan.md 诊断结论的三项升级】
#
#  诊断结论（详见 plan.md 与 test/verify_farfield_1d_v6.py）：
#    v6 的 FE 远场台阶与一维粘弹精确解逐项吻合（<2%），"软土模拟失真"不成立；
#    与论文图15 的偏差是软/硬一致的整体倍率，来自与论文的几何/参数/口径不一致。
#
#  ① 表层几何模式开关 surface_geometry（material_cfg 新增键，可由 case_config.json 注入）：
#       'horizontal'（默认）= v6 行为：固定厚度表层为固定高程水平带，仅存在于上平台；
#       'terrain' = 表层沿地形等厚铺设（上平台/坡面/下平台均有表层）——
#       论文图15 中坡脚以右软/硬两曲线明显分离，证明论文采用该几何。
#       实现：分层带新增 fix 字段（'elevation'/'fill'/'depth'），_band_bounds_at 统一
#       换算任意柱内各带上下界；几何切分新增沿地形折线 _partition_terrain；
#       截面分配 / 边界弹簧选材 / 自由场柱构造（fd 与 ray 引擎共用）全部按"局部埋深"取层。
#       horizontal 模式下全部带为 'elevation'，各处行为与 v6 完全等价。
#
#  ② case_meta.json 新增 ff_theory 块（远场一维理论台阶，自动 QA 锚点）：
#       建模时用 fd 引擎对左(上平台)/右(下平台)边界柱直接计算地表 PGA 理论值，
#       按 TAF = PGA / (factor_h × PGA_in) 口径写出 taf_h/taf_v；
#       FE 远场平台值应与之一致（±5%），供 Compute_TAF_v3 自动核对。
#       另：geometry 块写出 x_crest/x_toe，derived 块写出 a0 = fc × a0_base，
#       绘图脚本对位不再从 PGA csv 反推几何。
#
#  ③ mesh_cfg 新增 'elem' 键（'CPE4' 默认 / 'CPE4R' 减缩积分提速选项）。
#       本次诊断证明 CPE4 + 现网格已足够，此项仅为后续大模型提速预留。
#
#  ▼ 以下为 v6 原说明（机制全部保留）：
# ==========================================================
# ==========================================================
#  【本版 = v6：在 v5 基础上的两项核心升级（对齐 Shen 等 2025 论文口径）】
#
#  ① 自由场引擎升级为"频域精确分层解"（fd 引擎，默认启用）：
#       对每个边界节点所在的水平成层柱（基岩半空间 + 有限层 + 自由面），
#       逐频用全局矩阵法精确求解平面波边值问题（见 _fd_solve_column）。
#       与 v5 射线法相比：界面 SV<->P 转换、任意阶多次波、层间耦合、
#       随频率变化的瑞利阻尼衰减全部被精确包含——
#       不再有 MAX_REFLECT_ORDER 截断、阻抗近似与应力/位移系数口径问题。
#       v5 射线法整体保留（freefield_cfg['engine']='ray'），供回归对比。
#
#  ② TAF 归一化口径修正（配合 Compute_TAF_v2.py）：
#       论文式(5) 的分母是【基岩半空间自由场】的水平 PGA（解析值，垂直入射=2×输入），
#       不是同地层平坦模型的地表 PGA——后者会把软表层的一维场地放大抵消掉，
#       导致 v5 算出的 TAF 显著小于论文（图15 三层软表层最严重，差 2.5~5 倍）。
#       证据：论文图13/图15 远离坡体的平台处 TAF 不回到 1，而等于该柱一维放大倍数。
#       本版在 case_meta.json 写出 ff_normalization 块（A1/A2/factor_h 等），
#       由 Compute_TAF_v2.py 用解析分母重算 TAF；平坦模型降级为 QA 对照
#       （run_cfg['run_flat'] 可关闭以节省一半算力）。
#
#  ▼ 已知局限（方法差异，撰写方法节须注明）：
#    (a) Rayleigh 阻尼 α/β 对 P/S 同一阻尼比，无法独立实现论文 Qp=2Qs；
#        本版仅按 Qs=qs_factor·cs 施加，SV 主导可接受。
#    (b) 单元为 CPE4 线性（+楔形 CPE3），对比论文 SPECFEM2D 谱元高阶；
#        靠网格自适应（mesh_cfg）缓解数值频散。
#    (c) 弹簧刚度参考长度取固定 R=ymax，系数 0.5/0.25（Liu 2006 标准取值）。
#
#  配置注入：运行时若工况文件夹内存在 case_config.json 则覆盖默认配置
#    （v6 新增支持 freefield_cfg / run_cfg 两类注入）。
#  layers 约定：基岩之上的有限层列表，"从上到下"(顶层在前)；
#    []→单层均质；[overlying]→双层；[surface, overlying]→三层(论文图15)。
#    velocity_ratio = V_R / V_S（相对基岩波速比）；除最底覆盖层外其余层须给 thickness(m)。
# ==========================================================
material_cfg = {  # 默认材料参数配置（双层，对齐 double_v4；可被 case_config.json 覆盖）
    'angle': 15,  # 设置 SV 波入射角度（度）：与 double_v4 一致（批处理可替换 0/15）
    'surface_geometry': 'horizontal',  # v7：表层几何 'horizontal'=固定高程水平带(=v6) / 'terrain'=沿地形等厚铺设（论文图15 口径）
    'bedrock': {  # 定义基岩材料参数（底部半空间 V_R = 2000 m/s）
        'elastic_modulus': 26e9,  # 基岩杨氏模量（Pa），对应 Vs = 2000 m/s
        'poisson_ratio': 0.3,  # 基岩泊松比
        'density': 2500,  # 基岩密度（kg/m^3）
    },  # 结束基岩材料参数
    # layers：仅一项覆盖层 → 双层模型（M=1），与 double_v4 的 overlying 完全对应。
    'layers': [  # 定义基岩之上的有限层（仅覆盖层）
        {'name': 'overlying', 'velocity_ratio': 1.25, 'poisson_ratio': 0.3,  # 覆盖层 Vr/Vs=1.25(Vs=1600)，对齐 double_v4
         'density': 2500},  # 覆盖层密度（厚度由几何决定，故不写 thickness）
    ],  # 结束有限层列表
}  # 结束材料参数配置

geometry_cfg = {  # 定义几何参数配置（对齐 double_v4）
    'H_minus_h': 200.0,  # 斜坡高度差 H - h (m)
    'i': 45.0,  # 斜坡倾角 (度)：批处理可替换 30/60
    'h_over_H': 0.5,  # 深度比 h / H
    'total_L': 1800.0,  # 总模型长度 (m)
    'left_flat': 1000.0,  # 上平台长度 (m)
    'bedrock_thickness': 200.0,  # 基岩层厚度 (m)
}  # 结束几何参数配置

job_cfg = {  # 定义作业参数配置
    'variables': ('U', 'V', 'A'),  # 设置场输出变量
    'frequency': 1,  # 设置输出频率
    'num_cpus': 8,  # 设置并行 CPU 数量
    'memory_percent': 90,  # 设置作业内存百分比
}  # 结束作业参数配置


mesh_size = 4  # 网格尺寸设为 4 m


damping_cfg = {                 # 材料阻尼配置（对齐论文 Q 衰减；可被 case_config.json 覆盖）
    'enable': True,             # 是否施加材料阻尼（False 则退化为 v2 无阻尼行为）
    'method': 'rayleigh',       # 'rayleigh'=双频拟合(α+β，两端 ξ 相等≈恒定 Q) / 'stiffness'=仅刚度比例(β)
    'qs_factor': 0.05,          # Qs = qs_factor*cs（论文 coarse-grain 法，cs 单位 m/s）
    'q_bedrock': 999.0,         # 基岩品质因子(≈无衰减)
    'fc': None,                 # 输入波主频(Hz)：None=从加速度记录自动估计；可显式/注入覆盖
    'f1_factor': 0.5,           # 双频拟合下限 = f1_factor*fc
    'f2_factor': 2.5,           # 双频拟合上限 = f2_factor*fc（≈Ricker 高频边界）
    'anchor': 'input',          # v8：拟合锚定 'input'=仅输入主频(=v7) / 'dual'=场地基频+输入主频双控
}                               # 结束阻尼配置


mesh_cfg = {                    # 网格自适应配置（项②：对齐论文式1 Kuhlemeyer-Lysmer 判据）
    'auto': True,               # True=自动按最软层/最高频率计算 Δl_max；False=强制使用 mesh_size
    'elems_per_wavelength': 10, # 每波长最少单元数（论文取 10，即 Δl≤cs_min/(10·fmax)）
    'fmax_factor': 2.5,         # fmax = fmax_factor*fc（Ricker 子波有效频带上限估计，覆盖 2~3σ 宽度）
    'min_size': 0.5,            # 网格下限（m）：防止过软层或超高频时计算量爆炸
    'elem': 'CPE4',             # v7：单元类型 'CPE4'(默认，全积分) / 'CPE4R'(减缩积分提速选项)
}                               # 结束网格自适应配置


time_cfg = {                    # 时间步充分性校验配置（项③：安全护栏，当前 dt=1e-3 均达标不触发重采样）
    'check': True,              # True=检查输入记录 dt 是否满足最低步/周期要求
    'min_steps_per_fmax_period': 20,  # 每 fmax 周期最少采样步数（dt <= 1/(fmax*20) 才达标）
    'resample_if_violate': True,      # 不满足时是否自动重采样（np.interp 升采样）
    'tail_seconds': 0.0,              # v8：静默尾段时长(s)——分析步与 fd 自由场时窗同步延长（H(f) 提取用）
}                               # 结束时间步校验配置


freefield_cfg = {               # v6 新增：自由场引擎配置（可被 case_config.json 覆盖）
    'engine': 'fd',             #! 'fd'=频域精确分层自由场（默认，含界面 SV<->P 与全部多次波）；'ray'=v5 射线法（回归对比用）
    'include_damping': True,    # fd 引擎是否在自由场中计入与模型介质一致的瑞利阻尼
    'spectrum_tol': 1e-7,       # 仅求解幅值谱 > tol*max 的频率分量（其余置零，省时且高频数值稳定）
    'fcut': None,               # 频率上限(Hz)：None=仅按谱幅值掩码自适应截断
    'pad_factor': 4,            # FFT 补零倍数（>=2，防止时域卷绕污染响应窗口）
}                               # 结束自由场引擎配置


run_cfg = {                     # v6 新增：运行控制配置（可被 case_config.json 覆盖）
    'run_flat': False,           # 是否建模并求解平坦对照模型（v6 起 TAF 分母为解析值，flat 仅作 QA；量产可设 False 省一半算力）
    'surface_only': False,       # v8：True=仅 TOP_SURFACE 输出 A/U 全时程+整体场输出降频（ODB 瘦身，频域框架用）
}                               # 结束运行控制配置


# ==========================================================
#  模块常量与全局状态
# ==========================================================


DEFAULT_STEP_NAME = 'Step-earthquake'  # 定义默认分析步名称
BOUNDARY_SET_NAMES = ('Left_boundary', 'Right_boundary', 'Bottom_boundary')  # 定义基础边界节点集名称
BOUNDARY_SEQUENCE = ('l', 'r', 'b')  # 定义边界处理顺序


MAX_REFLECT_ORDER = 3  # 射线法覆盖层内多次反射/透射的几何级数截断阶数（默认3，可由 case_config.json 注入；多腔组合数≈(order+1)^M，M=有限层数，增大 order 会显著增加计算量）


_REFL_COEFF_CACHE = {}  # 等效反射/转换系数缓存：键为柱地表高度+入射角+层结构指纹，值为 (Rss_eff, Rsp_eff)


# ============================================================
#  参数打包对象（v7：用结构化对象取代散标量，缩短函数签名、便于阅读）
# ============================================================
# Material：单层材料的基本输入（剪切波速、泊松比、密度、固定厚度、名称）；
#   派生量 GG/lam/cp/EE 仍由物理核心函数按需计算，故此处只存输入。
#   thickness=None 表示：基岩半空间，或最底有限层（覆盖层，厚度由几何决定）。
Material = namedtuple('Material', ['cs', 'vv', 'density', 'thickness', 'name'])  # 单层材料输入
# Site：基岩半空间 + 有限层列表（layers 从上到下）+ 基岩层厚度
Site = namedtuple('Site', ['bedrock', 'layers', 'bedrock_thickness'])  # 多层场地（支持 0/1/2... 个有限层）
# Geometry：斜坡几何（输入项 + 一次算好的派生项）
Geometry = namedtuple('Geometry', [
    'total_L', 'i', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness',  # 输入项
    'H', 'h', 'H_upper', 'H_lower', 'H_flat', 'w_slope',  # 派生项
    'layer_interfaces'])  # 派生项：固定层间界面 y（从下到上，不含基岩界面），用于切分与材料分配
# BoundaryNode：单个边界节点的几何与粘弹性边界参数（取代裸 numpy 列索引）
BoundaryNode = namedtuple('BoundaryNode',
                          ['label', 'x', 'y', 'influence', 'kn', 'cn', 'kt', 'ct'])  # 边界节点
# FreeFieldCtx：射线法等效力计算所需的上下文（一次打包，避免长参数列表）
#   推广要点：自由场按每个节点"所在水平成层柱"逐层计算，故携带场地分层(strat)而非单一覆盖层。
FreeFieldCtx = namedtuple('FreeFieldCtx', [  # 自由场上下文命名元组
    'site', 'geom', 'strat', 'ymax_l', 'ymax_r', 'ymin',  # 场地、几何、分层带、各边界高度信息
    'alpha', 'beta_p', 'p_horiz',  # 基岩 SV 入射角、基岩 P 反射角、水平慢度（Snell 守恒）
    'GG', 'lam', 'cs', 'cp',  # 基岩剪切模量/拉梅常数/波速（投影与应力公式沿用 v4 用基岩标量）
    'VEL', 'DIS', 'dt', 'time_arr', 'max_reflect_order',  # 速度/位移时程、步长、时间轴、反射阶数
    'acc', 'damp_terms', 'ffcfg'])  # v6 新增：原始加速度记录、各带瑞利系数表、自由场引擎配置


# ==========================================================
#  通用工具函数
# ==========================================================


_DEFAULT_SCRIPT_NAME = 'VAB_oblique_TAF_multilayer_v8.py'  # 本脚本已知文件名（__file__ 缺失时的兜底）


def _script_path():  # 安全获取当前脚本绝对路径（Abaqus 内核可能不定义 __file__）
    """返回脚本绝对路径；Abaqus 用 execfile/kernel 执行时全局可能无 __file__，此时退化为当前目录下的已知脚本名。"""
    f = globals().get('__file__')  # 安全读取 __file__（缺失返回 None，不抛 NameError）
    if f:  # __file__ 存在时
        return os.path.abspath(f)  # 返回其绝对路径
    return os.path.join(os.getcwd(), _DEFAULT_SCRIPT_NAME)  # 兜底：当前工作目录(工况文件夹) + 已知脚本名


def _script_name():  # 安全获取当前脚本文件名（仅名字部分）
    """返回脚本文件名（如 'VAB_oblique_TAF_multilayer_v4.py'），不依赖 __file__。"""
    return os.path.basename(_script_path())  # 取路径基名


def _script_dir():  # 安全获取当前脚本所在目录（工况文件夹）
    """返回脚本所在目录；__file__ 缺失时退化为当前工作目录。"""
    return os.path.dirname(_script_path())  # 取路径目录部分


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


def _ensure_str(obj):  # 递归将 unicode 转为原生 str（Py2 兼容 Abaqus API）
    """递归将 json.load 产生的 unicode 转为 Python 2 原生 str（bytes），Py3 保持不变。

    Abaqus Python 2 的 C++ API（如 model.Material(name=...)）只接受 str，拒绝 unicode。
    而 json.load 在 Py2 下默认返回 unicode，因此加载 case_config.json 后必须转换。
    """  # 说明函数用途与必要性
    if sys.version_info[0] >= 3:  # Py3：str 即 unicode，Abaqus 2020+ 原生接受
        return obj  # 无需转换
    # Py2 分支：json.load 返回 unicode，需递归转为 str
    if isinstance(obj, unicode):  # 字符串节点
        return obj.encode('utf-8')  # unicode → str（UTF-8 字节串）
    if isinstance(obj, dict):  # 字典节点
        return {_ensure_str(k): _ensure_str(v) for k, v in obj.items()}  # 递归转换键与值
    if isinstance(obj, list):  # 列表节点
        return [_ensure_str(item) for item in obj]  # 递归转换每个元素
    if isinstance(obj, tuple):  # 元组节点
        return tuple(_ensure_str(item) for item in obj)  # 递归转换每个元素
    return obj  # 数值等其他类型原样返回


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


def _estimate_dominant_freq(acc, dt):  # 从加速度时程估计主频 fc
    """对加速度时程做 FFT，返回幅值谱最大处频率（DC 置 0）。

    acc : 一维加速度数组（可含均值偏移）；dt：时间步长 (s)。
    返回主导频率 fc (Hz)；Ricker 子波即得其中心频率。
    """
    acc_centered = acc - np.mean(acc)  # 去均值，避免 DC 分量干扰
    n = len(acc_centered)  # 信号长度
    spectrum = np.abs(np.fft.rfft(acc_centered))  # 单边幅值谱
    freqs = np.fft.rfftfreq(n, dt)  # 对应频率轴
    spectrum[0] = 0.0  # 置零 DC 分量
    idx_max = np.argmax(spectrum)  # 幅值最大处索引
    return float(freqs[idx_max])  # 返回主导频率


def _damping_ratio_from_q(cs, is_bedrock, dcfg):  # 由 Q 值换算阻尼比 ξ
    """根据剪切波速与是否基岩，计算品质因子 Q 与阻尼比 ξ=1/(2Q)。

    cs : 该层剪切波速 (m/s)；is_bedrock：是否基岩层；dcfg：阻尼配置（含 qs_factor/q_bedrock）。
    返回 (Q, xi)。
    """
    if is_bedrock:  # 基岩层
        Q = dcfg['q_bedrock']  # 基岩 Q≈999（近乎无衰减）
    else:  # 有限层
        Q = dcfg['qs_factor'] * cs  # Qs = qs_factor * cs（论文 coarse-grain 法）
    xi = 1.0 / (2.0 * Q)  # 阻尼比 ξ = 1/(2Q)
    return Q, xi  # 返回品质因子与阻尼比


def _rayleigh_coeffs(xi, dcfg, fc):  # 由阻尼比计算瑞利阻尼系数 α, β
    """按指定方法将阻尼比 ξ 换算为 Abaqus 瑞利阻尼系数 (alpha, beta)。

    xi : 阻尼比；dcfg：含 method/f1_factor/f2_factor；fc：主频 (Hz)。
    method=='stiffness'：α=0、β=ξ/(π·fc)（fc 处 ξ 精确）；
    其余（rayleigh 双频拟合）：在 f1=f1_factor·fc 与 f2=f2_factor·fc 两端 ξ 相等≈恒定 Q。
    返回 (alpha, beta)。
    """
    if dcfg['method'] == 'stiffness':  # 仅刚度比例阻尼
        alpha = 0.0  # 质量比例系数为零
        beta = xi / (math.pi * fc)  # β = ξ/(π·fc)，使 fc 处 ξ 精确
    else:  # 默认 rayleigh 双频拟合
        f1 = dcfg['f1_factor'] * fc  # 拟合下限频率（默认锚定输入主频）
        if dcfg.get('anchor') == 'dual' and dcfg.get('f_site'):  # v8：双控锚定（场地基频+输入主频）
            f1 = min(f1, float(dcfg['f_site']))  # 下限取较小者，使拟合带覆盖场地基频
        w1 = 2.0 * math.pi * f1  # 拟合下限圆频率
        w2 = 2.0 * math.pi * dcfg['f2_factor'] * fc  # 拟合上限圆频率
        alpha = 2.0 * xi * w1 * w2 / (w1 + w2)  # 瑞利 α（两端 ξ 相等≈恒定 Q）
        beta = 2.0 * xi / (w1 + w2)  # 瑞利 β
    return alpha, beta  # 返回瑞利阻尼系数


def _resolve_damping(dcfg, fc_est):  # 解析阻尼配置（补全 fc 字段）
    """拷贝阻尼配置，若 fc 为空则用自动估计值 fc_est 填充，返回解析后的配置 dict。

    供建材（_create_band_materials_sections）与元数据（_write_case_meta）共用同一份解析结果，
    避免 fc 口径漂移。显式给定的 fc 不被覆盖。
    """
    resolved = dict(dcfg)  # 浅拷贝配置（不就地修改原全局）
    if resolved.get('fc') is None:  # 未显式指定主频
        resolved['fc'] = fc_est  # 使用自动估计值
    return resolved  # 返回解析后配置


def _site_fundamental_freq(site, geom):  # v8：估算上平台柱场地基频 f_site = 1/Ts
    """Ts = 4·Σ(d_i/Vs_i)（上平台柱各有限层垂直走时），无有限层返回 None。

    固定厚度层取其 thickness；最底覆盖层厚度 = H_upper − 基岩厚 − Σ固定厚度。
    供 damping_cfg['anchor']=='dual' 的瑞利双控拟合使用（f1 锚定 min(f1_factor·fc, f_site)），
    对应 research_plan_v1.md §3.1-A4 "场地基频与输入卓越频率双控"。
    """
    if not site.layers:  # 均质场地（无有限层）
        return None  # 无场地基频概念
    fixed_sum = sum([L.thickness for L in site.layers if L.thickness is not None])  # 固定厚度之和
    travel = 0.0  # 累计垂直走时 Σd/Vs
    for L in site.layers:  # 自上而下遍历有限层
        d = L.thickness if L.thickness is not None else (geom.H_upper - geom.bedrock_thickness - fixed_sum)  # 该层厚度
        if d > 0:  # 厚度有效时累加
            travel += d / float(L.cs)  # 累加该层垂直走时
    if travel <= 0:  # 异常兜底
        return None  # 无法估算
    return 1.0 / (4.0 * travel)  # 场地基频 f_site = 1/(4Σd/Vs)


def _compute_interface_sv_coeff(alpha1, mat1, mat2):  # 定义界面 SV 波系数计算函数
    """计算 SV 波在两层界面的等效反射/透射系数（阻抗近似，忽略 SV<->P 转换）。

    alpha1 : 在 mat1 中的 SV 入射角（弧度）
    mat1   : 入射侧材料参数字典；mat2：透射侧材料参数字典
    返回 dict：Rss/Rsp/Tss/Tsp 与透射角 alpha2（Rsp=Tsp=0，射线法 v3 的关键近似）
    """
    z1s = mat1['density'] * mat1['cs'] * max(1e-8, math.cos(alpha1))  # 计算入射侧等效阻抗
    sin_a2 = mat2['cs'] * math.sin(alpha1) / mat1['cs']  # 由 Snell 定律计算透射角正弦值
    alpha2 = _safe_arcsin(sin_a2)  # 计算透射角
    z2s = mat2['density'] * mat2['cs'] * max(1e-8, math.cos(alpha2))  # 计算透射侧等效阻抗
    denom = z1s + z2s if abs(z1s + z2s) > 1e-12 else 1e-12  # 计算分母并避免除零
    rss = (z2s - z1s) / denom  # 计算反射系数
    tss = 2.0 * z2s / denom  # 计算透射系数
    rsp = 0.0  # 设置转换反射系数为零（阻抗近似忽略 SV->P）
    tsp = 0.0  # 设置转换透射系数为零（阻抗近似忽略 SV->P）
    return {'Rss': rss, 'Rsp': rsp, 'Tss': tss, 'Tsp': tsp, 'alpha2': alpha2}  # 返回界面系数字典


def _compute_free_surface_sv_coeff(alpha, cp, cs):  # 定义 SV 波自由面系数计算函数
    """计算 SV 波在自由面的反射系数 A1（SV->SV）与转换系数 A2（SV->P）。"""  # 说明函数用途
    beta_p = _safe_arcsin(cp * math.sin(alpha) / cs)  # 计算自由面 P 波转换角
    numerator_a1 = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) - cp ** 2 * math.cos(2 * alpha) ** 2  # 计算 A1 分子
    denominator = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) + cp ** 2 * math.cos(2 * alpha) ** 2  # 计算公共分母
    if abs(denominator) < 1e-12:  # 检查分母是否过小
        denominator = 1e-12  # 避免除零
    a1 = numerator_a1 / denominator  # 计算反射系数 A1
    a2 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator  # 计算转换系数 A2
    return {'A1': a1, 'A2': a2, 'beta': beta_p}  # 返回自由面系数字典


def _compute_free_surface_p_coeff(beta, cp, cs):  # 定义 P 波自由面系数计算函数
    """计算 P 波在自由面的反射系数 B2（P->P）与转换系数 B1（P->SV）。"""  # 说明函数用途
    alpha = _safe_arcsin(cs * math.sin(beta) / cp)  # 计算对应的 SV 角
    numerator_b2 = cp ** 2 * math.cos(2 * alpha) ** 2 - cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 计算 B2 分子
    denominator = cp ** 2 * math.cos(2 * alpha) ** 2 + cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)  # 计算公共分母
    if abs(denominator) < 1e-12:  # 检查分母是否过小
        denominator = 1e-12  # 避免除零
    b2 = numerator_b2 / denominator  # 计算反射系数 B2
    b1 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator  # 计算转换系数 B1
    return {'B1': b1, 'B2': b2, 'alpha': alpha}  # 返回自由面系数字典


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


def _build_stratigraphy(site, geom, ymin=0.0, surface_geometry='horizontal'):  # 定义场地分层带构造函数（v7：支持表层几何模式）
    """把场地分层展开为"从下到上"的标称材料带列表，供建模与自由场逐层取用。

    返回 list，每项 dict：{'name','mat'(Material),'y0','y1','fix',...}；
    y0/y1 为该带的【标称】下/上界 y（按上平台地表 H_upper 计），
    fix 标记该带边界如何随柱地表高度换算（v7 新增，见 _band_bounds_at）：
      'elevation' : 上下界为固定高程（基岩带；horizontal 模式下的全部带，=v6 行为）；
      'depth'     : 上下界为距局部地表的固定埋深 d0/d1（terrain 模式的固定厚度表层）；
      'fill'      : 下界固定高程 y0、上界=局部地表−dtop（terrain 模式的最底有限层/覆盖层）。
    顺序：基岩带在前（最底），向上依次为覆盖层(最底有限层)…表层(最顶有限层)。
    单层(site.layers 为空)时只返回基岩带（其上界取坡顶 H_upper，即全场均质基岩）。
    """  # 说明函数用途与返回结构
    H_upper = geom.H_upper  # 坡顶地表高度（最顶有限层的标称上界基准）
    bt = geom.bedrock_thickness  # 基岩界面 y
    layers_td = list(site.layers)  # 有限层（从上到下）
    if not layers_td:  # 处理单层情况（无有限层）
        return [{'name': site.bedrock.name, 'mat': site.bedrock, 'y0': ymin, 'y1': H_upper, 'fix': 'elevation'}]  # 全场均质基岩带
    terrain = (surface_geometry == 'terrain')  # 是否沿地形等厚铺设（v7）
    bands_td = []  # 初始化"从上到下"的有限层带列表
    y_top = H_upper  # 从坡顶开始向下推各层标称上界
    depth_top = 0.0  # 从局部地表向下累计的固定埋深（terrain 模式用）
    for L in layers_td:  # 自上而下遍历有限层
        if L.thickness is not None:  # 固定厚度层（表层等）
            band = {'name': L.name, 'mat': L, 'y0': y_top - L.thickness, 'y1': y_top}  # 标称上下界（上平台口径）
            if terrain:  # terrain 模式：该层按"距局部地表的埋深"定位
                band['fix'] = 'depth'  # 标记为埋深带
                band['d0'] = depth_top  # 该层顶埋深
                band['d1'] = depth_top + L.thickness  # 该层底埋深
            else:  # horizontal 模式：固定高程水平带（=v6）
                band['fix'] = 'elevation'  # 标记为高程带
            bands_td.append(band)  # 记录该层带
            y_top -= L.thickness  # 标称上界下移一个固定厚度
            depth_top += L.thickness  # 累计固定埋深
        else:  # 最底有限层（覆盖层，厚度由几何决定，填充至基岩界面）
            band = {'name': L.name, 'mat': L, 'y0': bt, 'y1': y_top}  # 标称：基岩界面到剩余高度
            if terrain:  # terrain 模式：上界跟随"局部地表 − 累计固定埋深"
                band['fix'] = 'fill'  # 标记为填充带
                band['dtop'] = depth_top  # 其上界距局部地表的埋深
            else:  # horizontal 模式（=v6）
                band['fix'] = 'elevation'  # 标记为高程带
            bands_td.append(band)  # 记录覆盖层带
    bedrock_band = {'name': site.bedrock.name, 'mat': site.bedrock, 'y0': ymin, 'y1': bt, 'fix': 'elevation'}  # 基岩带（恒为固定高程）
    bands_bt = list(reversed(bands_td))  # 反转为"从下到上"
    return [bedrock_band] + bands_bt  # 基岩带在前 + 有限层带（从下到上）


def _band_bounds_at(band, ys):  # 定义按柱地表高度换算带上下界的纯函数（v7 新增）
    """返回带 band 在"地表高程为 ys 的柱"内的 (y0, y1)。

    'elevation' 带原样返回标称值；'depth' 带返回 (ys−d1, ys−d0)；
    'fill' 带返回 (y0, ys−dtop)。horizontal 模式下全部带为 elevation，结果与 v6 一致。
    建模截面分配、边界弹簧选材、自由场柱构造统一经由本函数取界，保证四处口径一致。
    """
    fix = band.get('fix', 'elevation')  # 带定位方式（缺省按固定高程，兼容旧带结构）
    if fix == 'depth':  # 固定埋深带（terrain 模式表层）
        return ys - band['d1'], ys - band['d0']  # 上下界 = 局部地表减去底/顶埋深
    if fix == 'fill':  # 填充带（terrain 模式覆盖层）
        return band['y0'], ys - band['dtop']  # 下界固定高程、上界跟随地表
    return band['y0'], band['y1']  # 固定高程带原样返回标称值


# ==========================================================
#  几何构造与命名
# ==========================================================


def make_geometry(total_L, H_minus_h, i, h_over_H, left_flat, bedrock_thickness, fixed_thicknesses=None):  # 定义斜坡几何构造函数
    """根据斜坡几何输入计算全部派生量并打包为 Geometry。

    fixed_thicknesses: 顶部各有限层的固定厚度列表（从上到下，不含最底覆盖层），
        用于推算固定层间界面 y。空/None 表示双层或单层（无固定层间界面）。
    """  # 说明函数用途与参数
    H = H_minus_h / (1.0 - h_over_H)  # 计算总覆盖层厚度
    h = H - H_minus_h  # 计算下部覆盖层高度
    H_upper = bedrock_thickness + H  # 计算坡顶地表高度
    H_lower = bedrock_thickness + h  # 计算坡脚地表高度
    H_flat = bedrock_thickness + H  # 计算平坦场地总高度（= 坡顶高度）
    w_slope = H_minus_h / math.tan(math.radians(i))  # 计算坡面水平长度
    fixed = list(fixed_thicknesses or [])  # 规范化固定厚度列表（默认空）
    layer_interfaces = []  # 初始化固定层间界面 y 列表（从下到上）
    cum = 0.0  # 初始化从顶部累计的固定厚度
    for t in fixed:  # 自上而下遍历各固定层厚度
        cum += t  # 累加固定层厚度
        layer_interfaces.append(H_upper - cum)  # 该层底界面 y = 坡顶 - 累计固定厚度
    layer_interfaces = sorted(layer_interfaces)  # 由下到上排序（便于切分与材料分配）
    return Geometry(total_L=total_L, i=i, left_flat=left_flat, H_minus_h=H_minus_h,  # 组装并返回几何对象（填入输入项）
                    h_over_H=h_over_H, bedrock_thickness=bedrock_thickness,  # 继续填入输入项
                    H=H, h=h, H_upper=H_upper, H_lower=H_lower, H_flat=H_flat, w_slope=w_slope,  # 填入派生项
                    layer_interfaces=layer_interfaces)  # 填入固定层间界面 y 列表


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
#  多层自由场核心实现（射线法 / 到时延迟叠加，由 v4 双层推广为 1/2/3... 层层栈）：
#  基岩入射 SV → 各界面反/透射（阻抗近似，忽略界面 SV<->P 转换 Rsp=Tsp=0）
#  → 各有限层内自由面/界面多次混响（按 max_reflect_order 截断的几何级数）。
#  时域实现：对每个边界节点按其"所在水平成层柱"的几何到时延迟速度/位移时程后线性叠加。
#
#  推广要点（M=有限层数）：
#   ①等效反射系数 Rss_eff/Rsp_eff 由"自顶向下递归的层栈反射"求得（_effective_refl_coeffs），
#     M=1 时严格退化为 v4 的单腔几何级数；
#   ②混响在时域按"各有限层独立腔"叠加（_column_cavities + _superpose_paths 的腔积枚举），
#     M=1 时严格退化为 v4 单腔；M>=2 为截断近似（忽略跨腔耦合与界面 SV<->P）；
#   ③Rsp_eff 仅取顶层腔 P 混响 + 各层直透（顶面转换），M=1 时退化为 v4；
#   ④基岩/均质节点沿用 v4 到时公式；有限层节点用穿层走时累加（M=1 与 v4 一致）；
#   ⑤投影与应力沿用 v4：用基岩角度与基岩材料标量（继承 v4 近似）。
#  保留 v7 改良：①输入速度用基线校正积分 _integrate_acc_to_velocity；
#               ②各柱厚度/层组成按节点取（侧边用该侧最高点、底边按 x）。
# ============================================================


def _delay_signal(u0, n_delay, dt):  # 定义将时程延迟整数步的函数
    """将时程 u0(Nx2) 整体延迟 n_delay 个时间步，返回延长后的 (N+n_delay)x2 数组。"""  # 说明函数用途
    N = u0.shape[0]  # 原始序列长度
    new_len = N + n_delay  # 延迟后总长度
    delayed = np.zeros((new_len, 2))  # 创建延迟后数组
    delayed[:, 0] = np.arange(new_len) * dt  # 生成新时间轴
    delayed[n_delay:, 1] = u0[:, 1]  # 将原信号平移到延迟位置
    return delayed  # 返回延迟信号


def _make_delay_cache(timeseries, dt):  # 定义延迟信号缓存工厂
    """返回一个按延迟步数缓存延迟信号的访问器，跨节点复用以减少重复构造。"""  # 说明函数用途
    cache = {}  # 初始化缓存字典
    def get_delayed(delay_t):  # 定义按延迟时间取信号的闭包
        n_delay = int(np.round(delay_t / dt))  # 将延迟时间换算为离散步数
        if n_delay not in cache:  # 缓存未命中
            cache[n_delay] = _delay_signal(timeseries, n_delay, dt)  # 构造并缓存
        return cache[n_delay]  # 返回缓存信号
    return get_delayed  # 返回访问器


def _pad_to(arr, length, dt):  # 定义把 (M,2) 数组补零到指定长度的函数
    """将 arr(Mx2) 末尾补零延长到 length 行，补零段补充时间轴。"""  # 说明函数用途
    if arr.shape[0] < length:  # 需要补零时
        pad = np.zeros((length - arr.shape[0], 2))  # 创建补零段
        pad[:, 0] = np.arange(arr.shape[0], length) * dt  # 补齐时间轴
        arr = np.vstack([arr, pad])  # 拼接
    return arr  # 返回补齐后数组


def _calc_node_delay(boundary, x0, y0, Ly, Lx,
                     alpha, beta_p, cs, cp, alpha2, beta2, cs2, cp2, ymax_col):
    """计算单个边界节点的三段到时 (tA, tB, tC)：入射 SV、反射 SV、反射/转换 P。

    boundary : 'l'/'r'/'b'；x0,y0：节点坐标；Ly：界面相对底边高度；Lx：模型横向跨度
    返回 (t1, t2, t3) 三段延迟时间（秒）。基岩段用 cs/cp/alpha/beta_p，
    覆盖层段用 cs2/cp2/alpha2/beta2，与 v3 射线法一致。
    """
    if boundary in ('l', 'r'):  # 处理左/右边界节点
        if y0 <= Ly:  # 节点位于基岩段
            t1 = y0 * np.cos(alpha) / cs  # 入射 SV 到时
            t2 = (2 * Ly - y0) * np.cos(alpha) / cs  # 反射 SV 到时
            t3 = ((Ly - y0) / (cp * np.cos(beta_p))  # 反射 P 到时（第一部分）
                  + (Ly - (Ly - y0) * np.tan(alpha) * np.tan(beta_p)) * np.cos(alpha) / cs)  # 第二部分
        else:  # 节点位于覆盖层段
            t1 = Ly * np.cos(alpha) / cs + (y0 - Ly) * np.cos(alpha2) / cs2  # 入射 SV 到时
            t2 = Ly * np.cos(alpha) / cs + (2 * ymax_col - Ly - y0) * np.cos(alpha2) / cs2  # 反射 SV 到时
            t3 = Ly * np.cos(alpha) / cs + (y0 - Ly) * np.cos(beta2) / cp2  # 反射 P 到时
        if boundary == 'r':  # 右边界叠加横向传播延迟
            shift = Lx * np.sin(alpha) / cs  # 横向传播延迟量
            t1 += shift; t2 += shift; t3 += shift  # 三段同时叠加
        return t1, t2, t3  # 返回三段到时
    elif boundary == 'b':  # 处理底边节点（位于基岩，纯基岩波场）
        t4 = x0 * np.sin(alpha) / cs  # 入射 SV 到时
        t5 = (2 * Ly + x0 * np.tan(alpha)) * np.cos(alpha) / cs  # 反射 SV 到时
        t6 = (Ly / (cp * np.cos(beta_p))  # 反射 P 到时（第一部分）
              + (Ly * np.cos(alpha) + x0 * np.sin(alpha)  # 第二部分
                 - Ly * np.tan(beta_p) * np.sin(alpha)) / cs)  # 第三部分
        return t4, t5, t6  # 返回三段到时
    else:  # 非法边界
        raise ValueError("boundary must be 'l', 'r', or 'b'")  # 抛出异常


def _column_seg(cs, vv, density, alpha_p, y0, y1, name):  # 定义构造单个柱内层段的函数
    """根据材料与水平慢度构造柱内一层段（含派生波速、角度、垂直慢度因子、上下界）。"""  # 说明函数用途
    params = _compute_material_params(cs, vv, density)  # 计算该层材料派生参数（GG/lam/cp 等）
    alpha = _safe_arcsin(alpha_p * cs)  # 由 Snell 守恒求该层 SV 角
    beta = _safe_arcsin(alpha_p * params['cp'])  # 由 Snell 守恒求该层 P 角
    return {'name': name, 'mat': params, 'cs': cs, 'cp': params['cp'],  # 打包层段：名称、材料参数、波速
            'GG': params['GG'], 'lam': params['lam'], 'density': density,  # 剪切模量/拉梅常数/密度
            'alpha': alpha, 'beta': beta,  # SV/P 角
            'cos_alpha': math.cos(alpha), 'cos_beta': math.cos(beta),  # 垂直慢度用余弦
            'y0': y0, 'y1': y1}  # 该层段下界与上界 y


def _build_column(strat, ymax_col, alpha_p, ymin):  # 定义构造节点所在成层柱的函数（v7：经 _band_bounds_at 通用化）
    """由场地分层带 strat 与该柱地表高度 ymax_col 构造"从下到上"的柱层段列表。

    各带上下界经 _band_bounds_at 按该柱地表高度换算（terrain 模式的埋深带随地表移动），
    再裁剪到 [ymin, ymax_col] 并用 y_floor 逐带钳位保证不重叠、不留缝。
    horizontal 模式（全部 elevation 带）结果与 v6 完全一致。
    单层场地（strat 仅一条基岩带）返回单层段柱（全 bedrock 至地表）。
    """  # 说明函数用途
    tol = 1e-6  # 设置带相交容差
    column = []  # 初始化柱层段列表
    y_floor = ymin  # 当前柱内已占用高度的上沿（自底向上推进）
    for band in strat:  # 从下到上遍历各标称材料带
        y0, y1 = _band_bounds_at(band, ymax_col)  # 按该柱地表高度换算带上下界（v7）
        y0 = max(y0, y_floor)  # 下界钳位到已占用上沿（防带间重叠）
        y1 = min(y1, ymax_col)  # 上界截断到地表
        if y1 <= y0 + tol:  # 换算/截断后无有效厚度
            continue  # 跳过（该柱无此带）
        mat = band['mat']  # 取该带材料输入
        column.append(_column_seg(mat.cs, mat.vv, mat.density, alpha_p, y0, y1, band['name']))  # 追加层段
        y_floor = y1  # 推进已占用上沿
    return column  # 返回从下到上的柱层段列表


def _seg_at(column, y0):  # 定义在柱中按 y 坐标查找对应层段的辅助函数
    """返回柱 column 中包含 y0 的层段 dict（键同 _column_seg：alpha/beta/GG/cs/lam/cp 等）。

    column 从下到上排列（_build_column 产物）；y0 在某段 [seg['y0'], seg['y1']] 内则返回该段。
    找不到时退回最顶层段（兜底，不应发生）。
    用于项①自由场层内材料一致化：对有限层侧边节点，改用本层材料而非基岩标量。
    """
    for seg in reversed(column):  # 从顶层向下查找（软表层通常在顶，先中）
        if seg['y0'] - 1e-6 <= y0 <= seg['y1'] + 1e-6:  # 节点 y 落入该层段范围
            return seg  # 返回本层段
    return column[-1]  # 兜底返回最顶层段（正常不走此分支）


def _effective_refl_coeffs(column, oc):  # 定义层栈等效反射/转换系数计算函数
    """自顶向下递归求基岩中上行 SV 的等效自由面反射 Rss_eff 与 SV->P 转换 Rsp_eff。

    沿用 v4 的界面 SV 阻抗近似与自由面完整 SV 反射/转换；M=1 时严格退化为 v4 的单腔几何级数。
    column：从下到上的柱层段（column[0]=基岩，column[-1]=最顶有限层或均质介质）。
    """  # 说明函数用途
    topL = column[-1]  # 取最顶层段
    free_sv = _compute_free_surface_sv_coeff(topL['alpha'], topL['cp'], topL['cs'])  # 顶面 SV 反射/转换系数
    if len(column) == 1:  # 均质柱（无有限层覆盖）
        return free_sv['A1'], free_sv['A2']  # 直接返回自由面 SV 反射与 SV->P 转换
    free_p = _compute_free_surface_p_coeff(topL['beta'], topL['cp'], topL['cs'])  # 顶面 P 反射/转换系数
    nseg = len(column)  # 柱层段总数
    Rtop = free_sv['A1']  # 当前层顶反射（从最顶层的自由面 A1 起）
    A2_top = free_sv['A2']  # 顶面 SV->P 转换系数
    B2_top = free_p['B2']  # 顶面 P->P 反射系数
    T_up = 1.0  # SV 自下而上穿过各界面的累计透射
    T_down = 1.0  # P（用 SV 近似）自上而下穿过各界面的累计透射
    Rbot_top_layer = None  # 顶层腔底界面反射（P 混响用）
    for k in range(nseg - 1, 0, -1):  # 自顶层向下遍历各界面（column[k] 上、column[k-1] 下）
        upper = column[k]  # 界面上方层段
        lower = column[k - 1]  # 界面下方层段
        intf_lo = _compute_interface_sv_coeff(lower['alpha'], lower['mat'], upper['mat'])  # 下方入射：反射回下方 Rss + 上透 Tss
        intf_hi = _compute_interface_sv_coeff(upper['alpha'], upper['mat'], lower['mat'])  # 上方下行：反射回上方 Rss + 下透 Tss
        Rbot = intf_hi['Rss']  # 本层底界面反射（下行反射回上行）
        if k == nseg - 1:  # 记录顶层腔底反射
            Rbot_top_layer = Rbot  # 供 P 混响使用
        cyc = Rtop * Rbot  # 本层腔一次 SV 混响幅值因子
        sum_cyc = sum([cyc ** j for j in range(oc + 1)])  # 截断几何级数和（用列表避免通配 sum 拒收生成器）
        Rbottom = intf_lo['Rss'] + intf_lo['Tss'] * Rtop * intf_hi['Tss'] * sum_cyc  # 本层底界面等效反射
        T_up *= intf_lo['Tss']  # 累计上行透射
        T_down *= intf_hi['Tss']  # 累计下行透射
        Rtop = Rbottom  # 该等效反射成为下一层（更低层）看到的顶反射
    Rss_eff = Rtop  # 递归至基岩界面：基岩中上行 SV 的等效反射
    cyc_p = B2_top * Rbot_top_layer  # 顶层腔一次 P 混响幅值因子
    sum_cyc_p = sum([cyc_p ** j for j in range(oc + 1)])  # P 混响截断几何级数和
    Rsp_eff = T_up * A2_top * T_down * sum_cyc_p  # 等效 SV->P 转换（上透→顶面转换→下透→顶腔混响）
    return Rss_eff, Rsp_eff  # 返回等效反射与转换系数


def _column_cavities(column, oc):  # 定义柱内各混响腔（用于时域延迟叠加）的计算函数
    """返回 (cavities_sv, cavities_p)：各有限层 SV 混响腔 (cycle, cdelay) 列表 + 顶层 P 混响腔。

    cycle = 该层腔顶反射×底反射（幅值）；cdelay = 该层往返垂直走时。M=1 时退化为 v4 单腔。
    """  # 说明函数用途
    nseg = len(column)  # 柱层段总数
    cavities_sv = []  # 初始化 SV 混响腔列表
    cavities_p = []  # 初始化 P 混响腔列表（仅顶层腔）
    for k in range(1, nseg):  # 遍历各有限层（column[1..nseg-1]）
        layer = column[k]  # 当前有限层段
        lower = column[k - 1]  # 其下方层段
        thick = layer['y1'] - layer['y0']  # 该层厚度
        if thick <= 0:  # 厚度无效则跳过
            continue  # 跳过该层
        Rbot = _compute_interface_sv_coeff(layer['alpha'], layer['mat'], lower['mat'])['Rss']  # 底界面反射（下行反射回上行）
        if k == nseg - 1:  # 顶层：顶反射取自由面 SV 反射
            Rtop = _compute_free_surface_sv_coeff(layer['alpha'], layer['cp'], layer['cs'])['A1']  # 自由面 SV 反射
        else:  # 内层：顶反射取上界面反射
            upper = column[k + 1]  # 其上方层段
            Rtop = _compute_interface_sv_coeff(layer['alpha'], layer['mat'], upper['mat'])['Rss']  # 上界面反射
        cdelay_sv = 2.0 * thick * layer['cos_alpha'] / layer['cs']  # 该层 SV 往返垂直走时
        cavities_sv.append((Rtop * Rbot, cdelay_sv))  # 记录该层 SV 混响腔
        if k == nseg - 1:  # 顶层腔额外贡献 P 混响（转换在顶面发生）
            B2 = _compute_free_surface_p_coeff(layer['beta'], layer['cp'], layer['cs'])['B2']  # 自由面 P 反射
            cdelay_p = 2.0 * thick * layer['cos_beta'] / layer['cp']  # 该层 P 往返垂直走时
            cavities_p.append((B2 * Rbot, cdelay_p))  # 记录顶层 P 混响腔
    return cavities_sv, cavities_p  # 返回 SV/P 混响腔列表


def _tt(column, y_lo, y_hi, wave):  # 定义柱内 y_lo→y_hi 垂直走时累加函数
    """逐层累加从 y_lo 到 y_hi（y_lo<y_hi）的垂直走时；wave='SV' 用 cos_alpha/cs，'P' 用 cos_beta/cp。"""  # 说明函数用途
    t = 0.0  # 初始化走时
    for seg in column:  # 遍历柱内各层段
        lo = max(y_lo, seg['y0'])  # 本层段内的下限
        hi = min(y_hi, seg['y1'])  # 本层段内的上限
        if hi > lo:  # 区间有效时累加
            if wave == 'SV':  # SV 波
                t += (hi - lo) * seg['cos_alpha'] / seg['cs']  # 累加 SV 垂直走时
            else:  # P 波
                t += (hi - lo) * seg['cos_beta'] / seg['cp']  # 累加 P 垂直走时
    return t  # 返回总走时


def _superpose_paths(get_delayed, tA, tB, tC, cavities_sv, cavities_p, order_count, dt):  # 定义多腔混响叠加函数
    """对一个节点叠加主路径与各有限层混响，返回 (时间轴, A路径值, B路径累加, C路径累加)。

    A：主到时 tA 的延迟信号；B：反射 SV 路径在各腔往返组合下的混响累加；
    C：反射/转换 P 路径在顶层腔混响下的累加。各腔几何级数按 order_count 截断，腔间取乘积枚举。
    单腔（M=1）时严格退化为 v4 的 Σ_k cycle^k·delayed(t + k·cdelay)。
    """  # 说明函数用途
    def _combos(cavities):  # 定义将一组腔展开为 (幅值, 附加延迟) 组合的内函数
        combo = [(1.0, 0.0)]  # 初始组合：无混响（幅值1、零延迟）
        for (cyc, cd) in cavities:  # 逐腔做几何级数与已有组合的乘积
            new = []  # 新组合容器
            for (amp, dl) in combo:  # 遍历已有组合
                for j in range(order_count + 1):  # 该腔的截断阶数
                    new.append((amp * (cyc ** j), dl + j * cd))  # 叠加该腔第 j 阶（幅值相乘、延迟相加）
            combo = new  # 更新组合
        return combo  # 返回全部组合
    combo_b = _combos(cavities_sv)  # B 路径各腔组合
    combo_c = _combos(cavities_p)  # C 路径各腔组合（仅顶层 P 腔）
    sig_b = [(amp, get_delayed(tB + dl)) for amp, dl in combo_b]  # B 路径各组合的延迟信号
    sig_c = [(amp, get_delayed(tC + dl)) for amp, dl in combo_c]  # C 路径各组合的延迟信号
    u0_tA = get_delayed(tA)  # 主到时延迟信号
    max_len = u0_tA.shape[0]  # 统计统一长度
    for _amp, arr in sig_b + sig_c:  # 遍历所有路径信号
        max_len = max(max_len, arr.shape[0])  # 更新最大长度
    u0_tA = _pad_to(u0_tA, max_len, dt)  # 补齐主路径
    sumB = np.zeros(max_len)  # B 路径累加器
    sumC = np.zeros(max_len)  # C 路径累加器
    for amp, arr in sig_b:  # 叠加 B 路径各组合
        sumB += amp * _pad_to(arr, max_len, dt)[:, 1]  # 累加（带组合幅值）
    for amp, arr in sig_c:  # 叠加 C 路径各组合
        sumC += amp * _pad_to(arr, max_len, dt)[:, 1]  # 累加（带组合幅值）
    return u0_tA[:, 0], u0_tA[:, 1], sumB, sumC  # 返回时间轴与三路结果


def _compute_freefield_at_node(boundary, x0, y0, ymax_col, ctx, get_vel, get_dis):
    """射线法计算单节点自由场时程，返回 dict：time/ux/uy/dotux/dotuy/sigmax/sigmay。

    boundary : 'l'/'r'/'b'；x0,y0：节点坐标；ymax_col：该柱地表高度（决定层组成、层厚与到时）；
    ctx      : FreeFieldCtx（含基岩角度、水平慢度、场地分层、基岩材料标量、VEL/DIS、dt 等）；
    get_vel/get_dis：速度/位移时程的延迟缓存访问器（跨节点复用）。
    多层推广：按该柱层栈求等效系数与各腔混响；投影/应力沿用 v4（基岩角度 + 基岩材料标量）。
    """
    geom = ctx.geom  # 取几何对象
    Lx = geom.total_L  # 模型横向跨度（xmin=0）
    bt = geom.bedrock_thickness  # 基岩界面 y
    dt = ctx.dt  # 时间步长
    oc = max(0, int(ctx.max_reflect_order))  # 反射阶数上限
    p = ctx.p_horiz  # 水平慢度

    column = _build_column(ctx.strat, ymax_col, p, ctx.ymin)  # 构造该节点所在成层柱
    nseg = len(column)  # 柱层段数
    key = (round(ymax_col, 4), round(p, 12))  # 等效系数缓存键（同一柱地表高度+入射角复用）
    cached = _REFL_COEFF_CACHE.get(key)  # 查缓存
    if cached is None:  # 未命中
        cached = _effective_refl_coeffs(column, oc)  # 计算等效反射/转换系数
        _REFL_COEFF_CACHE[key] = cached  # 写入缓存
    Rss_eff, Rsp_eff = cached  # 取等效系数
    cavities_sv, cavities_p = _column_cavities(column, oc)  # 计算该柱各混响腔

    if boundary == 'b' or y0 <= bt + 1e-6:  # 基岩节点或均质节点：沿用 v4 到时公式
        Ly = bt if nseg >= 2 else ymax_col  # 反射点：有基岩界面取界面，否则（均质）取自由面
        col0 = column[0]  # 最底层段（基岩或均质介质）
        tA, tB, tC = _calc_node_delay(boundary, x0, y0, Ly, Lx,  # 计算三段到时（v4 公式）
                                      ctx.alpha, ctx.beta_p, ctx.cs, ctx.cp,  # 基岩角度/波速
                                      col0['alpha'], col0['beta'], col0['cs'], col0['cp'], ymax_col)  # 占位（基岩分支不用）
    else:  # 有限层节点：穿层走时累加（反射点为自由面）
        tA = _tt(column, ctx.ymin, y0, 'SV')  # 入射 SV：自底到节点
        tB = _tt(column, ctx.ymin, ymax_col, 'SV') + _tt(column, y0, ymax_col, 'SV')  # 反射 SV：自底到自由面 + 自由面回节点
        tC = _tt(column, ctx.ymin, bt, 'SV') + _tt(column, bt, y0, 'P')  # 转换 P：基岩段 SV + 覆盖段 P 上行到节点
        if boundary == 'r':  # 右边界叠加横向传播延迟
            shift = Lx * math.sin(ctx.alpha) / ctx.cs  # 横向传播延迟量（基岩角度）
            tA += shift; tB += shift; tC += shift  # 三段同时叠加

    # 位移自由场：对位移时程 DIS 做多腔混响叠加
    td, dA, dB, dC = _superpose_paths(get_dis, tA, tB, tC, cavities_sv, cavities_p, oc, dt)  # 位移路径叠加
    # 速度自由场：对速度时程 VEL 做多腔混响叠加（速度与应力共用此叠加结果）
    _tv, vA, vB, vC = _superpose_paths(get_vel, tA, tB, tC, cavities_sv, cavities_p, oc, dt)  # 速度路径叠加

    A1 = Rss_eff  # 等效自由面 SV 反射系数（该柱）
    A2 = Rsp_eff  # 等效自由面 SV->P 转换系数（该柱）

    # ── 项①：层内材料/角度一致化 ────────────────────────────────────────────────
    # 底边节点或基岩段侧边节点：沿用基岩标量（与 v4 完全等价，单层退化精确）
    # 有限层侧边节点（y0 > bedrock_interface）：改用本层 alpha/beta/GG/cs/lam/cp，
    #   使面力计算与已按本层取值的弹簧/阻尼系数口径一致（v4 此处用基岩标量，存在矛盾）。
    # 注意：等效反射幅值 Rss_eff/Rsp_eff 仍用阻抗近似层栈递归（射线近似，不因本修改改变）。
    use_local = (boundary in ('l', 'r')) and (y0 > bt + 1e-6)  # 是否为有限层侧边节点
    if use_local:  # 有限层侧边节点：取本层 alpha/beta 与本层材料标量
        local_seg = _seg_at(column, y0)  # 按节点 y 坐标查找所在层段
        a = local_seg['alpha']   # 本层 SV 入射角（Snell 守恒折射角）
        bp = local_seg['beta']   # 本层 P 角（Snell 守恒折射 P 角）
        GG = local_seg['GG']     # 本层剪切模量
        cs = local_seg['cs']     # 本层剪切波速
        lam = local_seg['lam']   # 本层拉梅常数
        cp = local_seg['cp']     # 本层纵波波速
    else:  # 基岩节点或底边节点：沿用基岩标量（与 v4 等价）
        a = ctx.alpha    # 基岩 SV 入射角
        bp = ctx.beta_p  # 基岩 P 反射角
        GG = ctx.GG      # 基岩剪切模量
        cs = ctx.cs      # 基岩剪切波速
        lam = ctx.lam    # 基岩拉梅常数
        cp = ctx.cp      # 基岩纵波波速
    # ── 层内一致化结束 ──────────────────────────────────────────────────────────

    ux = dA * np.cos(a) - A1 * dB * np.cos(a) + A2 * dC * np.sin(bp)  # x 向位移
    uy = -dA * np.sin(a) - A1 * dB * np.sin(a) - A2 * dC * np.cos(bp)  # y 向位移
    dotux = vA * np.cos(a) - A1 * vB * np.cos(a) + A2 * vC * np.sin(bp)  # x 向速度
    dotuy = -vA * np.sin(a) - A1 * vB * np.sin(a) - A2 * vC * np.cos(bp)  # y 向速度

    sin2a = np.sin(2 * a)    # 双角正弦（基于上方选定的 a）
    cos2a = np.cos(2 * a)    # 双角余弦
    sin2bp = np.sin(bp) ** 2  # P 角正弦平方
    sin2bp_2 = np.sin(2 * bp)  # 双倍 P 角正弦
    cosbp2 = np.cos(bp) ** 2  # P 角余弦平方

    if boundary == 'l':  # 左边界应力（外法向已内嵌）
        sigmax = (GG / cs * sin2a * (vA - A1 * vB)  # σ_xx
                  + A2 * (lam + 2 * GG * sin2bp) / cp * vC)  # 叠加转换项
        sigmay = (GG / cs * cos2a * (vA + A1 * vB)  # σ_yy
                  - A2 * GG * sin2bp_2 / cp * vC)  # 叠加转换项
    elif boundary == 'r':  # 右边界应力
        sigmax = (GG / cs * sin2a * (-vA + A1 * vB)  # σ_xx
                  - A2 * (lam + 2 * GG * sin2bp) / cp * vC)  # 叠加转换项
        sigmay = (GG / cs * cos2a * (-vA - A1 * vB)  # σ_yy
                  + A2 * GG * sin2bp_2 / cp * vC)  # 叠加转换项
    else:  # 底边界应力（基岩节点，始终用基岩标量，已由上方 use_local 保证）
        sigmax = (GG / cs * cos2a * (vA + A1 * vB)  # σ_xx
                  - A2 * GG * sin2bp_2 / cp * vC)  # 叠加转换项
        sigmay = (GG / cs * sin2a * (-vA + A1 * vB)  # σ_yy
                  + A2 * (lam + 2 * GG * cosbp2) / cp * vC)  # 叠加转换项

    return {'time': td, 'ux': ux, 'uy': uy, 'dotux': dotux, 'dotuy': dotuy,  # 打包返回
            'sigmax': sigmax, 'sigmay': sigmay}  # 应力分量


# ============================================================
#  v6 频域精确分层自由场引擎（fd 引擎，默认）
#
#  原理：对每个边界节点所在的"水平成层柱"（基岩半空间 + 若干有限层 + 顶部自由面），
#  在频率域逐频精确求解平面波边值问题（全局矩阵法）：
#    - 未知量：基岩反射 P/SV + 每个有限层的上/下行 P 与 SV（共 4M+2 个，M=有限层数）；
#    - 方程：顶部自由面 σyy=σxy=0（2 条）+ 每个界面 ux/uy/σyy/σxy 连续（4M 条）；
#    - 入射：基岩中上行 SV，单位位移幅值（实际幅值谱在节点评估时乘 U0(ω)=−A(ω)/ω²）。
#  与 v5 射线法相比：界面 SV<->P 转换、任意阶多次波、层间耦合、随频率变化的
#  瑞利阻尼衰减全部精确包含（无截断、无阻抗近似、无应力/位移系数口径问题）。
#
#  波相位约定：e^{i(ωt − ωp·x − ωk_y·y)}；上行波 k_y=+q（参考层底）、下行波 k_y=−q（参考层顶），
#  使所有相位因子在层内均为衰减方向，数值稳定。极化约定与论文式(2)（半空间解）严格一致：
#    Pu=(cp·p, cp·qp)  Pd=(cp·p, −cp·qp)  Su=(cs·qs, −cs·p)  Sd=(−cs·qs, −cs·p)
#  半空间退化（M=0）时本引擎逐项还原 v5 的解析公式（见 test_fd_freefield_v6.py T1）。
# ============================================================


_FD_SOLVER_CACHE = {}  # fd 引擎缓存：键=round(ymax_col,4) 的柱解；另含 '_input' 键缓存输入谱


def _next_pow2(n):  # 求不小于 n 的最小 2 的幂
    """返回不小于 n 的最小 2 的整数次幂（FFT 长度用）。"""
    m = 1  # 初始化幂值
    while m < n:  # 不足 n 时继续翻倍
        m *= 2  # 翻倍
    return m  # 返回 2 的幂


def _band_damping_terms(strat, damping):  # 计算各材料带的瑞利阻尼系数表
    """返回 {带名: (alpha, beta)}；damping 未启用时全部为 (0,0)。

    与 _create_band_materials_sections 完全同口径（同一 Q→ξ→(α,β) 公式），
    保证 fd 自由场的衰减与 Abaqus 模型内介质一致。
    """
    terms = {}  # 初始化系数表
    for idx, band in enumerate(strat):  # 从下到上遍历各材料带（idx==0 即基岩）
        if damping and damping.get('enable'):  # 启用材料阻尼时
            _Q, xi = _damping_ratio_from_q(band['mat'].cs, idx == 0, damping)  # 该带品质因子与阻尼比
            a_ray, b_ray = _rayleigh_coeffs(xi, damping, damping['fc'])  # 该带瑞利系数 (α,β)
        else:  # 未启用阻尼
            a_ray, b_ray = 0.0, 0.0  # 弹性（无衰减）
        terms[band['name']] = (a_ray, b_ray)  # 按带名记录
    return terms  # 返回系数表


def _fd_input_spectrum(ctx):  # 计算（并缓存）输入波的位移谱与求解频点
    """对输入加速度补零 FFT，返回输入谱缓存 dict（同一模型的所有节点共享）。

    内容：Nfft/Nout/nfreq/idx（被求解频点索引）/omega（对应圆频率）/U0（对应位移谱）。
    位移谱 U0 = −A(ω)/ω²（e^{iωt} 约定下加速度两次积分）；仅保留幅值谱 > tol·max 的频点。
    """
    cached = _FD_SOLVER_CACHE.get('_input')  # 查输入谱缓存
    if cached is not None:  # 命中缓存
        return cached  # 直接返回
    ffcfg = ctx.ffcfg or freefield_cfg  # 自由场引擎配置
    acc = np.asarray(ctx.acc, dtype=float)  # 输入加速度记录
    dt = float(ctx.dt)  # 时间步长
    N = acc.shape[0]  # 原始记录长度
    pad = max(2, int(ffcfg.get('pad_factor', 4)))  # 补零倍数（防时域卷绕）
    Nfft = _next_pow2(N * pad)  # FFT 长度（2 的幂）
    A = np.fft.rfft(acc, n=Nfft)  # 加速度单边谱
    freqs = np.fft.rfftfreq(Nfft, dt)  # 频率轴 (Hz)
    tol = float(ffcfg.get('spectrum_tol', 1e-7))  # 谱幅值掩码阈值
    amax = float(np.max(np.abs(A))) if A.size else 0.0  # 谱峰值
    mask = np.abs(A) > tol * amax  # 幅值显著的频点掩码
    mask[0] = False  # 排除直流分量（ω=0 不可除）
    fcut = ffcfg.get('fcut')  # 显式频率上限
    if fcut:  # 给定上限时附加截断
        mask = mask & (freqs <= float(fcut))  # 截断高频
    idx = np.nonzero(mask)[0]  # 被求解频点索引
    omega = 2.0 * math.pi * freqs[idx]  # 对应圆频率数组
    U0 = -A[idx] / (omega ** 2)  # 位移谱（加速度谱两次积分）
    tail = float(ffcfg.get('tail_seconds', 0.0) or 0.0)  # v8：静默尾段时长（捕捉混响衰减，H(f) 提取用）
    Nout = min(Nfft, N + int(round(tail / dt))) if tail > 0 else N  # 输出窗口长度（不超过 FFT 长度防卷绕）
    cached = {'Nfft': Nfft, 'Nout': Nout, 'dt': dt, 'nfreq': len(freqs),  # 打包缓存内容
              'idx': idx, 'omega': omega, 'U0': U0}  # 频点索引/圆频率/位移谱
    _FD_SOLVER_CACHE['_input'] = cached  # 写入缓存
    return cached  # 返回输入谱缓存


def _fd_layer_params(seg, omega, p, damp_terms, include_damping):  # 计算单层段逐频复参数
    """返回该层段逐频复数材料与垂直慢度 dict：{'qs','qp','mu','lam','csC','cpC','p'}。

    瑞利阻尼以复模量/复密度计入：ρ̃=ρ(1−iα/ω)、μ̃=μ(1+iωβ)、λ̃=λ(1+iωβ)，
    与 Abaqus 连续介质方程 ρü+αρu̇=∇·σ(1+β∂t) 的频域形式严格一致。
    垂直慢度取衰减分支（Im(q)<=0），保证局部参考相位因子恒为衰减方向。
    """
    rho = float(seg['density'])  # 该层密度
    mu0 = rho * seg['cs'] ** 2  # 实剪切模量
    lam0 = rho * (seg['cp'] ** 2 - 2.0 * seg['cs'] ** 2)  # 实拉梅常数
    if include_damping:  # 自由场计入与 FE 介质一致的阻尼
        a_ray, b_ray = damp_terms.get(seg['name'], (0.0, 0.0))  # 该带 (α,β)
    else:  # 自由场按弹性计算
        a_ray, b_ray = 0.0, 0.0  # 无阻尼
    rhoC = rho * (1.0 - 1j * a_ray / omega)  # 复密度（质量比例阻尼）
    sfac = 1.0 + 1j * omega * b_ray  # 刚度比例阻尼因子 (1+iωβ)
    muC = mu0 * sfac  # 复剪切模量
    lamC = lam0 * sfac  # 复拉梅常数
    cs2 = muC / rhoC  # 复剪切波速平方
    cp2 = (lamC + 2.0 * muC) / rhoC  # 复纵波波速平方
    qs = np.sqrt(1.0 / cs2 - p * p)  # SV 垂直慢度（复）
    qp = np.sqrt(1.0 / cp2 - p * p)  # P 垂直慢度（复）
    qs = np.where(qs.imag > 0.0, -qs, qs)  # 强制衰减分支 Im(q)<=0
    qp = np.where(qp.imag > 0.0, -qp, qp)  # 强制衰减分支 Im(q)<=0
    return {'qs': qs, 'qp': qp, 'mu': muC, 'lam': lamC,  # 打包逐频复参数
            'csC': np.sqrt(cs2), 'cpC': np.sqrt(cp2), 'p': p}  # 复波速与水平慢度


def _fd_wave_params(seg, la, kind):  # 构造某层段某类波的极化与相位参数
    """kind ∈ {'Pu','Pd','Su','Sd'}；返回 {'dx','dy','ky','yref'}（dx/dy/ky 为逐频复数组）。

    极化约定与论文式(2)一致（半空间退化逐项还原 v5 公式）：
      Pu=(cp·p, cp·qp)  Pd=(cp·p, −cp·qp)  Su=(cs·qs, −cs·p)  Sd=(−cs·qs, −cs·p)
    相位 e^{−iω·ky·(y−yref)}：上行波 ky=+q 参考层底 y0、下行波 ky=−q 参考层顶 y1。
    """
    qs, qp, csC, cpC, p = la['qs'], la['qp'], la['csC'], la['cpC'], la['p']  # 取该层逐频参数
    if kind == 'Pu':  # 上行 P 波
        return {'dx': cpC * p, 'dy': cpC * qp, 'ky': qp, 'yref': seg['y0']}  # 极化/相位/参考层底
    if kind == 'Pd':  # 下行 P 波
        return {'dx': cpC * p, 'dy': -cpC * qp, 'ky': -qp, 'yref': seg['y1']}  # 极化/相位/参考层顶
    if kind == 'Su':  # 上行 SV 波
        return {'dx': csC * qs, 'dy': -csC * p, 'ky': qs, 'yref': seg['y0']}  # 极化/相位/参考层底
    return {'dx': -csC * qs, 'dy': -csC * p, 'ky': -qs, 'yref': seg['y1']}  # 下行 SV 波


def _fd_field_coeffs(wave, la, omega, p, y):  # 某波在高程 y 处的 5 个场量系数
    """返回 (ux, uy, σyy, σxy, σxx) 的逐频系数（乘以该波幅值即得场量谱）。

    位移 u = d·a·e^{iω(t−px)}·e^{−iω·ky·(y−yref)} ⇒ ∂x→−iωp、∂y→−iω·ky，
    σyy = λ∂ux/∂x+(λ+2μ)∂uy/∂y；σxy = μ(∂ux/∂y+∂uy/∂x)；σxx = (λ+2μ)∂ux/∂x+λ∂uy/∂y。
    """
    ph = np.exp(-1j * omega * wave['ky'] * (y - wave['yref']))  # 垂直相位因子（含衰减）
    dx = wave['dx'] * ph  # 含相位的 x 向位移系数
    dy = wave['dy'] * ph  # 含相位的 y 向位移系数
    lam, mu = la['lam'], la['mu']  # 复拉梅常数与剪切模量
    miw = -1j * omega  # 公共因子 −iω
    syy = miw * (lam * p * dx + (lam + 2.0 * mu) * wave['ky'] * dy)  # σyy 系数
    sxy = miw * mu * (wave['ky'] * dx + p * dy)  # σxy 系数
    sxx = miw * ((lam + 2.0 * mu) * p * dx + lam * wave['ky'] * dy)  # σxx 系数
    return dx, dy, syy, sxy, sxx  # 返回 5 个场量系数


def _fd_solve_column(column, p, omega, damp_terms, include_damping):  # 柱频域全局矩阵求解
    """对一根成层柱逐频求解【单位入射上行 SV】的全部波幅，返回柱解 dict。

    column：从下到上层段（column[0]=基岩半空间段，_build_column 产物）。
    未知量排序：[Pd0, Sd0, Pu1, Pd1, Su1, Sd1, ..., PuM, PdM, SuM, SdM]（0=基岩）。
    方程：每个界面 ux/uy/σyy/σxy 连续（4M 条）+ 顶部自由面 σyy=σxy=0（2 条）。
    返回 {'amps','las','waves','inc','column'}；amps 形状 (n频点, 4M+2)。
    """
    nseg = len(column)  # 柱层段数
    M = nseg - 1  # 有限层数
    las = [_fd_layer_params(seg, omega, p, damp_terms, include_damping) for seg in column]  # 各层逐频复参数
    waves = []  # 各层未知波参数表：waves[m] = [(未知量列号, 波参数), ...]
    waves.append([(0, _fd_wave_params(column[0], las[0], 'Pd')),  # 基岩反射下行 P
                  (1, _fd_wave_params(column[0], las[0], 'Sd'))])  # 基岩反射下行 SV
    col = 2  # 下一个未知量列号
    for m in range(1, nseg):  # 各有限层的 4 个波
        wm = []  # 该层波列表
        for kind in ('Pu', 'Pd', 'Su', 'Sd'):  # 上/下行 P 与 SV
            wm.append((col, _fd_wave_params(column[m], las[m], kind)))  # 记录列号与波参数
            col += 1  # 列号递增
        waves.append(wm)  # 追加该层
    nunk = col  # 未知量总数（=4M+2）
    inc = _fd_wave_params(column[0], las[0], 'Su')  # 入射上行 SV（单位幅值，参考柱底 ymin）
    nb = omega.shape[0]  # 频点数
    A = np.zeros((nb, nunk, nunk), dtype=complex)  # 批量系数矩阵（每频点一个）
    b = np.zeros((nb, nunk), dtype=complex)  # 批量右端项
    row = 0  # 当前方程行号
    for j in range(M):  # 逐个界面写连续条件（界面 j 在 column[j] 顶）
        Y = column[j]['y1']  # 界面高程
        for sgn, m in ((1.0, j), (-1.0, j + 1)):  # 界面下方层(+) 与上方层(−)
            la = las[m]  # 该层复参数
            for cidx, w in waves[m]:  # 该层各未知波
                ux, uy, syy, sxy, _sxx = _fd_field_coeffs(w, la, omega, p, Y)  # 界面处场量系数
                A[:, row + 0, cidx] += sgn * ux  # ux 连续
                A[:, row + 1, cidx] += sgn * uy  # uy 连续
                A[:, row + 2, cidx] += sgn * syy  # σyy 连续
                A[:, row + 3, cidx] += sgn * sxy  # σxy 连续
            if m == 0:  # 入射波（已知）贡献移到右端
                ux, uy, syy, sxy, _sxx = _fd_field_coeffs(inc, la, omega, p, Y)  # 入射波界面场量
                b[:, row + 0] -= sgn * ux  # 移项：ux
                b[:, row + 1] -= sgn * uy  # 移项：uy
                b[:, row + 2] -= sgn * syy  # 移项：σyy
                b[:, row + 3] -= sgn * sxy  # 移项：σxy
        row += 4  # 行号推进 4 条
    Ys = column[-1]['y1']  # 地表高程（最顶层段上界）
    laT = las[-1]  # 最顶层复参数
    for cidx, w in waves[-1]:  # 顶层各未知波参与自由面条件
        _ux, _uy, syy, sxy, _sxx = _fd_field_coeffs(w, laT, omega, p, Ys)  # 地表处应力系数
        A[:, row + 0, cidx] += syy  # 自由面 σyy=0
        A[:, row + 1, cidx] += sxy  # 自由面 σxy=0
    if M == 0:  # 半空间退化：入射波也参与自由面条件
        _ux, _uy, syy, sxy, _sxx = _fd_field_coeffs(inc, laT, omega, p, Ys)  # 入射波地表应力
        b[:, row + 0] -= syy  # 移项：σyy
        b[:, row + 1] -= sxy  # 移项：σxy
    try:  # 优先批量直解（numpy 支持堆叠方阵；右端项需补列维以兼容各 numpy 版本）
        amps = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (nb,nunk,1) 求解后去掉列维 → (nb, nunk)
    except np.linalg.LinAlgError:  # 个别频点奇异时逐频最小二乘兜底
        amps = np.zeros((nb, nunk), dtype=complex)  # 初始化结果
        for k in range(nb):  # 逐频点求解
            amps[k] = np.linalg.lstsq(A[k], b[k], rcond=-1)[0]  # 最小二乘解（rcond=-1 兼容 Abaqus Py2.7 旧版 numpy）
    return {'amps': amps, 'las': las, 'waves': waves, 'inc': inc, 'column': column}  # 返回柱解


def _fd_eval_column(sol, omega, p, y):  # 在柱内高程 y 处评估单位入射的 7 个场量谱
    """返回 dict{'ux','uy','vx','vy','sxx','syy','sxy'}（逐频复数组，单位入射幅值）。"""
    column = sol['column']  # 柱层段
    seg_idx = 0  # 默认基岩段（兜底）
    for k in range(len(column) - 1, -1, -1):  # 自顶向下查找节点所在层段
        if column[k]['y0'] - 1e-6 <= y <= column[k]['y1'] + 1e-6:  # 落入该层段
            seg_idx = k  # 记录层段索引
            break  # 停止查找
    la = sol['las'][seg_idx]  # 该层复参数
    amps = sol['amps']  # 全部未知波幅
    ux = np.zeros_like(omega, dtype=complex)  # x 向位移谱
    uy = np.zeros_like(omega, dtype=complex)  # y 向位移谱
    syy = np.zeros_like(omega, dtype=complex)  # σyy 谱
    sxy = np.zeros_like(omega, dtype=complex)  # σxy 谱
    sxx = np.zeros_like(omega, dtype=complex)  # σxx 谱
    for cidx, w in sol['waves'][seg_idx]:  # 叠加该层各未知波
        cux, cuy, csyy, csxy, csxx = _fd_field_coeffs(w, la, omega, p, y)  # 该波场量系数
        a = amps[:, cidx]  # 该波逐频幅值
        ux += a * cux; uy += a * cuy  # 位移叠加
        syy += a * csyy; sxy += a * csxy; sxx += a * csxx  # 应力叠加
    if seg_idx == 0:  # 基岩段需叠加入射波本身
        cux, cuy, csyy, csxy, csxx = _fd_field_coeffs(sol['inc'], la, omega, p, y)  # 入射波场量
        ux += cux; uy += cuy  # 位移叠加
        syy += csyy; sxy += csxy; sxx += csxx  # 应力叠加
    vx = 1j * omega * ux  # x 向速度谱（e^{iωt} 约定）
    vy = 1j * omega * uy  # y 向速度谱
    return {'ux': ux, 'uy': uy, 'vx': vx, 'vy': vy, 'sxx': sxx, 'syy': syy, 'sxy': sxy}  # 返回场量谱


def _fd_freefield_at_node(boundary, x0, y0, ymax_col, ctx):  # fd 引擎单节点自由场
    """fd 引擎：计算单个边界节点的自由场时程，返回 dict（接口与射线法引擎一致）。

    boundary : 'l'/'r'/'b'；x0,y0：节点坐标；ymax_col：该柱地表高度；ctx：FreeFieldCtx。
    步骤：①取输入谱缓存 ②取/解该柱频域解 ③节点谱 = 单位解 × U0(ω) × e^{−iωp·x0}
          ④逆 FFT 截断回原时长 ⑤按边界外法向嵌入应力符号。
    """
    inp = _fd_input_spectrum(ctx)  # 输入谱缓存（全模型共享）
    key = round(ymax_col, 4)  # 柱解缓存键（同地表高度柱复用）
    sol = _FD_SOLVER_CACHE.get(key)  # 查柱解缓存
    if sol is None:  # 未命中
        column = _build_column(ctx.strat, ymax_col, ctx.p_horiz, ctx.ymin)  # 构造该柱层段
        sol = _fd_solve_column(column, ctx.p_horiz, inp['omega'], ctx.damp_terms,  # 频域求解
                               bool((ctx.ffcfg or {}).get('include_damping', True)))  # 是否计入阻尼
        _FD_SOLVER_CACHE[key] = sol  # 写入缓存
    fields = _fd_eval_column(sol, inp['omega'], ctx.p_horiz, y0)  # 单位入射场量谱
    shift = np.exp(-1j * inp['omega'] * ctx.p_horiz * x0)  # 水平传播相位（左边界 x0=0 不移）
    scale = inp['U0'] * shift  # 输入位移谱 × 水平相位
    out = {}  # 时域结果容器
    for name in ('ux', 'uy', 'vx', 'vy', 'sxx', 'syy', 'sxy'):  # 逐场量逆变换
        spec = np.zeros(inp['nfreq'], dtype=complex)  # 全频带谱（未求解频点为零）
        spec[inp['idx']] = fields[name] * scale  # 填入被求解频点
        out[name] = np.fft.irfft(spec, n=inp['Nfft'])[:inp['Nout']]  # 逆 FFT 并截断回原时长
    if boundary == 'l':  # 左边界外法向 n=(−1,0)：面力 = −σxx, −σxy
        sigmax = -out['sxx']; sigmay = -out['sxy']  # 嵌入外法向符号
    elif boundary == 'r':  # 右边界外法向 n=(+1,0)：面力 = +σxx, +σxy
        sigmax = out['sxx']; sigmay = out['sxy']  # 嵌入外法向符号
    else:  # 底边界外法向 n=(0,−1)：面力 = −σxy, −σyy
        sigmax = -out['sxy']; sigmay = -out['syy']  # 嵌入外法向符号
    t_out = np.arange(inp['Nout']) * inp['dt']  # v8：输出时间轴（含静默尾段；tail=0 时与原记录时轴等价）
    return {'time': t_out, 'ux': out['ux'], 'uy': out['uy'],  # 打包返回
            'dotux': out['vx'], 'dotuy': out['vy'], 'sigmax': sigmax, 'sigmay': sigmay}  # 速度与应力


def _fd_engine_selfcheck(logger=None):  # v8：fd 引擎建模前内置自检（§3.1-A5 验证协议自动化）
    """两项解析对拍：①弹性均质半空间垂直入射地表水平位移=2E；②单层场地 1Hz 传递函数 vs SH 解析解。

    返回 {'halfspace_err','single_layer_err'}（相对误差）；任一误差 > 1e-3 抛 RuntimeError 中止建模，
    防止 fd 引擎被无意改动后静默产出错误的等效节点力。计算量毫秒级，每次建模自动执行。
    """
    p0 = 1e-15  # 近垂直入射水平慢度
    # ① 半空间退化：均质基岩柱（Vs=2000, ν=0.3），地表 |ux| 应=2.0（自由面放大）
    col1 = [_column_seg(2000.0, 0.3, 2500.0, p0, 0.0, 400.0, 'bedrock')]  # 均质柱
    om1 = 2.0 * math.pi * np.array([1.0, 3.0, 7.0])  # 三个校核频率
    sol1 = _fd_solve_column(col1, p0, om1, {'bedrock': (0.0, 0.0)}, True)  # 弹性求解
    f1 = _fd_eval_column(sol1, om1, p0, 400.0)  # 地表场量谱
    err1 = float(np.max(np.abs(np.abs(f1['ux']) - 2.0))) / 2.0  # 相对误差
    # ② 单层对拍：200m/Vs=800 覆盖层 + Vs=2000 基岩，1Hz，|ux| vs SH 解析 2/|cos(kh)+i·α·sin(kh)|
    col2 = [_column_seg(2000.0, 0.3, 2500.0, p0, 0.0, 200.0, 'bedrock'),  # 基岩段
            _column_seg(800.0, 0.3, 2500.0, p0, 200.0, 400.0, 'cover')]  # 覆盖层段
    om2 = np.array([2.0 * math.pi * 1.0])  # 单频 1Hz
    sol2 = _fd_solve_column(col2, p0, om2, {'bedrock': (0.0, 0.0), 'cover': (0.0, 0.0)}, True)  # 弹性求解
    f2 = _fd_eval_column(sol2, om2, p0, 400.0)  # 地表场量谱
    kh = 2.0 * math.pi * 1.0 * 200.0 / 800.0  # 层内相位角 ωh/Vs
    ana = 2.0 / abs(complex(math.cos(kh), (800.0 / 2000.0) * math.sin(kh)))  # SH 解析地表幅值（阻抗比=0.4）
    err2 = abs(abs(f2['ux'][0]) - ana) / ana  # 相对误差
    result = {'halfspace_err': err1, 'single_layer_err': err2}  # 打包自检结果
    if logger:  # 有日志器时记录
        log_step(logger, 'fd 引擎自检: 半空间误差=%.2e, 单层解析误差=%.2e（阈值 1e-3）', err1, err2)  # 输出日志
    if err1 > 1e-3 or err2 > 1e-3:  # 任一项超阈值
        raise RuntimeError('fd 引擎自检失败: halfspace_err=%.3e, single_layer_err=%.3e' % (err1, err2))  # 中止建模
    return result  # 返回自检误差


# ==========================================================
#  建模（几何/材料/网格）
# ==========================================================


def _max_element_size(site, fc, mcfg):  # 定义 Kuhlemeyer-Lysmer 自适应最大单元尺寸计算函数
    """按 Kuhlemeyer-Lysmer 准则计算允许的最大单元尺寸 Δl_max。

    Δl_max = cs_min / (elems_per_wavelength * fmax)，其中 fmax = fmax_factor * fc。
    site : Site 对象（基岩 + 有限层列表）；fc：输入波主频(Hz)；mcfg：mesh_cfg 配置 dict。
    返回 Δl_max(m)，已受 mcfg['min_size'] 兜底（防止过小导致计算量爆炸）。
    """
    epw = float(mcfg.get('elems_per_wavelength', 10))  # 每波长单元数（默认10）
    ff = float(mcfg.get('fmax_factor', 2.5))  # fmax 倍数因子（默认2.5）
    min_sz = float(mcfg.get('min_size', 0.5))  # 网格下限（默认0.5m）
    cs_candidates = [site.bedrock.cs]  # 初始化候选波速列表（基岩 Vs）
    for L in site.layers:  # 遍历各有限层
        cs_candidates.append(L.cs)  # 加入该层波速
    cs_min = min(cs_candidates)  # 取最小剪切波速（最软层）
    fmax = ff * fc  # 有效最高频率（Ricker 主频的 fmax_factor 倍）
    delta_l = cs_min / (epw * fmax)  # Kuhlemeyer-Lysmer 最大单元尺寸
    return max(delta_l, min_sz)  # 受下限约束后返回


def _interface_partitions(strat):  # 定义从分层带提取切分界面的函数（v7：区分水平界面与沿地形界面）
    """返回 (horiz_y, depth_d)：水平切分界面 y 列表 + 沿地形切分埋深 d 列表（均从下到上）。

    每条带（除最底基岩带）的【下界】即一条材料界面：
      elevation/fill 带的下界为固定高程水平线（y0）→ 归入 horiz_y；
      depth 带的下界为"地表整体下移 d1"的沿地形折线 → 归入 depth_d。
    horizontal 模式 depth_d 恒为空，horiz_y 与 v6 的 _interface_y_list 等价。
    单层（仅一条带）返回 ([], [])（无需切分）。
    """  # 说明函数用途
    horiz, depth = [], []  # 初始化两类界面列表
    for band in strat[1:]:  # 从下到上遍历各带（跳过最底基岩带，其下界为模型底）
        if band.get('fix', 'elevation') == 'depth':  # 埋深带：下界沿地形
            depth.append(band['d1'])  # 记录该界面埋深
        else:  # 高程/填充带：下界为水平线
            horiz.append(band['y0'])  # 记录该界面高程
    return horiz, depth  # 返回 (水平界面, 沿地形界面)


def _create_band_materials_sections(model, strat, damping=None):  # 定义逐带创建材料与截面的函数（v3：支持材料阻尼）
    """为分层带（从下到上）逐带创建材料与均质截面，返回 [(band, sec_name), ...]。

    damping: 解析后的阻尼配置 dict（含 enable/method/fc/qs_factor/q_bedrock 等）；
             None 或 enable=False 时退化为 v2 无阻尼行为。strat[0] 恒为基岩（_build_stratigraphy 保证）。
    """  # 说明函数用途与参数
    band_sections = []  # 初始化带-截面映射列表
    for idx, band in enumerate(strat):  # 从下到上遍历每条材料带（idx==0 即基岩）
        mat = band['mat']  # 取该带材料输入
        EE = _compute_elastic_modulus_from_wave_speed(mat.cs, mat.vv, mat.density)  # 由波速反算弹性模量
        mat_name = _next_available_name('Material-%s' % band['name'], model.materials)  # 生成材料名（含层名）
        m = model.Material(name=mat_name)  # 创建材料
        m.Elastic(table=((EE, mat.vv),))  # 定义弹性参数
        m.Density(table=((mat.density,),))  # 定义密度
        if damping and damping.get('enable'):  # 启用材料阻尼时按 Q 衰减施加瑞利阻尼
            is_bedrock = (idx == 0)  # 基岩带（最底带）→ 用 q_bedrock≈999
            Q, xi = _damping_ratio_from_q(mat.cs, is_bedrock, damping)  # 由波速换算品质因子 Q 与阻尼比 ξ
            a_ray, b_ray = _rayleigh_coeffs(xi, damping, damping['fc'])  # 由 ξ 换算瑞利系数 (α, β)
            m.Damping(alpha=a_ray, beta=b_ray)  # 施加 Abaqus 瑞利阻尼（α 质量比例 + β 刚度比例）
        sec_name = _next_available_name('Section-%s' % band['name'], model.sections)  # 生成截面名（含层名）
        model.HomogeneousSolidSection(name=sec_name, material=mat_name, thickness=1.0)  # 创建均质实体截面
        band_sections.append((band, sec_name))  # 记录该带与其截面名
    return band_sections  # 返回带-截面映射


def _partition_horizontal(model, part, geom, y_list, name_prefix):  # 定义按一组水平界面切分面的函数
    """对 y_list 中每条水平界面逐条 PartitionFaceBySketch 切分（切线自动裁剪到实体内）。"""  # 说明函数用途
    for idx, y in enumerate(y_list):  # 遍历每条水平界面 y
        part_faces = part.faces  # 获取当前面集合
        sk_name = '__%s_%d__' % (name_prefix, idx)  # 生成临时草图名
        sk = model.ConstrainedSketch(name=sk_name, sheetSize=max(geom.total_L, geom.H_upper) * 2)  # 创建水平切分草图
        sk.Line(point1=(0.0, y), point2=(geom.total_L, y))  # 绘制该界面水平切线
        part.PartitionFaceBySketch(faces=part_faces, sketch=sk)  # 按草图切分面
        del model.sketches[sk_name]  # 删除临时草图


def _partition_terrain(model, part, geom, depth_list, name_prefix):  # 定义按沿地形折线切分面的函数（v7 新增）
    """对 depth_list 中每个埋深 d，用"地表整体下移 d"的三段折线切分（上平台-坡面-下平台）。

    折线节点：(0, H_upper−d) → (left_flat, H_upper−d) → (left_flat+w_slope, H_lower−d)
              → (total_L, H_lower−d)，即表层底界面沿地形等深。
    """  # 说明函数用途
    for idx, d in enumerate(depth_list):  # 遍历每条沿地形界面（按埋深）
        sk_name = '__%s_%d__' % (name_prefix, idx)  # 生成临时草图名
        sk = model.ConstrainedSketch(name=sk_name, sheetSize=max(geom.total_L, geom.H_upper) * 2)  # 创建切分草图
        x1 = geom.left_flat  # 坡顶 x 坐标
        x2 = geom.left_flat + geom.w_slope  # 坡脚 x 坐标
        sk.Line(point1=(0.0, geom.H_upper - d), point2=(x1, geom.H_upper - d))  # 上平台段水平线（地表下移 d）
        sk.Line(point1=(x1, geom.H_upper - d), point2=(x2, geom.H_lower - d))  # 坡面段斜线（与坡面平行）
        sk.Line(point1=(x2, geom.H_lower - d), point2=(geom.total_L, geom.H_lower - d))  # 下平台段水平线
        part.PartitionFaceBySketch(faces=part.faces, sketch=sk)  # 按草图切分面
        del model.sketches[sk_name]  # 删除临时草图


def _assign_sections_by_band(part, band_sections, surface_y_fn=None):  # 定义按质心落带分配截面的函数（v7：支持按局部埋深落带）
    """按面质心落入哪条材料带分配对应截面，返回 [(层名, 面数), ...]。

    surface_y_fn: 函数 x → 该 x 处地表高程；terrain 模式据此把质心换算到局部柱内
    （经 _band_bounds_at 取带界）。None 或 horizontal 模式时等价于 v6 的按标称 y 落带。
    """  # 说明函数用途与参数
    def _to_face_sequence(face_list):  # 定义面序列转换内函数
        face_seq = part.faces[0:0]  # 创建空面序列
        for face in face_list:  # 遍历面列表
            face_seq = face_seq + part.faces[face.index:face.index + 1]  # 逐个拼接面对象
        return face_seq  # 返回面序列
    tol = 1e-6  # 设置带边界容差
    buckets = [[] for _ in band_sections]  # 为每条带准备面桶
    for face in part.faces:  # 遍历所有面
        centroid = face.getCentroid()  # 获取面质心
        if len(centroid) >= 2 and not hasattr(centroid[0], '__len__'):  # 质心为 (x, y, ...) 平铺形式
            xc, yc = centroid[0], centroid[1]  # 读取质心横纵坐标
        else:  # 质心为 ((x, y, z),) 嵌套形式
            xc, yc = centroid[0][0], centroid[0][1]  # 读取质心横纵坐标
        ys = surface_y_fn(xc) if surface_y_fn else None  # 该质心所在柱的地表高程（terrain 落带用）
        placed = False  # 标记是否已归带
        for bi, (band, _sec) in enumerate(band_sections):  # 从下到上遍历各带
            y0, y1 = _band_bounds_at(band, ys) if ys is not None else (band['y0'], band['y1'])  # 带上下界（v7 统一换算）
            if y0 - tol <= yc < y1 + tol:  # 判断质心是否落入该带 [y0, y1)
                buckets[bi].append(face)  # 归入该带
                placed = True  # 置归带标记
                break  # 跳出带循环
        if not placed:  # 处理未落入任何带的兜底情况
            buckets[-1].append(face)  # 归入最顶带
    counts = []  # 初始化面数统计
    for (band, sec_name), face_list in zip(band_sections, buckets):  # 遍历每带及其面桶
        if face_list:  # 该带存在面时分配截面
            part.SectionAssignment(region=Region(faces=_to_face_sequence(face_list)),  # 为该带分配截面
                                   sectionName=sec_name, offset=0.0, offsetType=MIDDLE_SURFACE,  # 指定截面参数
                                   offsetField='', thicknessAssignment=FROM_SECTION)  # 结束截面分配
        counts.append((band['name'], len(face_list)))  # 记录该带面数
    return counts  # 返回各带面数统计


def create_model(site, geom, mesh_size, cae_name=None, logger=None, damping=None,
                 surface_geometry='horizontal', elem_name='CPE4'):
    """创建二维平面应变斜坡模型：几何、材料、截面、装配、网格（不含分析步）。

    site: Site 对象（基岩 + 有限层列表 + 基岩厚度，支持 1/2/3... 层）
    geom: Geometry 对象（斜坡几何，含派生量与固定层间界面）
    damping: 解析后的阻尼配置 dict（转发给材料创建，与平坦模型同参以保证 TAF 分母一致）
    surface_geometry: v7 表层几何模式 'horizontal'(=v6) / 'terrain'(表层沿地形等厚铺设)
    elem_name: v7 单元类型 'CPE4'(默认) / 'CPE4R'
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

    # ============ 逐层创建材料与截面（支持 1/2/3... 层）============
    strat = _build_stratigraphy(site, geom, surface_geometry=surface_geometry)  # 构造从下到上的分层带（v7：含表层几何模式）
    band_sections = _create_band_materials_sections(model, strat, damping)  # 逐带创建材料与截面（含瑞利阻尼）
    log_step(logger, '%s 已创建 %d 个材料带的材料与截面 (表层几何=%s)', model_name, len(strat), surface_geometry)  # 记录材料/截面创建日志

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

    # 2. 切分各材料界面（v7：水平界面 + terrain 模式的沿地形折线界面）
    interfaces, depth_ifaces = _interface_partitions(strat)  # 提取水平界面 y 与沿地形界面埋深 d
    _partition_horizontal(model, part, geom, interfaces, 'hpartition')  # 逐条水平切分
    if depth_ifaces:  # terrain 模式：存在沿地形界面
        _partition_terrain(model, part, geom, depth_ifaces, 'tpartition')  # 逐条沿地形折线切分（v7）
    log_step(logger, '%s 材料界面切分完成: 水平=%d, 沿地形=%d', model_name, len(interfaces), len(depth_ifaces))  # 记录切分完成日志

    # 设置网格控制：默认结构化四边形；若有界面切过坡面（楔形）或存在沿地形界面则退为自由四边形为主
    cuts_slope = bool(depth_ifaces) or any(H_lower + 1e-6 < y < H_upper - 1e-6 for y in interfaces)  # 是否存在界面切过坡面
    pickedRegions = part.faces  # 选取全部面作为网格区域
    if cuts_slope:  # 坡面被切出表层楔形（无法结构化）
        part.setMeshControls(regions=pickedRegions, elemShape=QUAD_DOMINATED, technique=FREE)  # 自由四边形为主网格（容许少量三角）
        log_step(logger, '%s 检测到界面切过坡面（表层楔形），网格采用 FREE QUAD_DOMINATED', model_name)  # 记录网格策略
    else:  # 无楔形（M<=1 或界面在坡脚以下）
        part.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)  # 结构化四边形（同 v4）
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)  # 设置全局网格尺寸
    elem_code = CPE4R if str(elem_name).upper() == 'CPE4R' else CPE4  # v7：按 mesh_cfg['elem'] 选择单元类型
    elemType1 = mesh.ElemType(elemCode=elem_code, elemLibrary=STANDARD)  # 定义平面应变四节点单元
    elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)  # 定义平面应变三节点单元（自由网格过渡用）
    part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1, elemType2))  # 分配单元类型（四+三节点）
    part.generateMesh()  # 生成网格
    log_step(logger, '%s 已生成网格: %s 单元，尺寸=%.2f', model_name, str(elem_name).upper(), mesh_size)  # 记录网格生成日志

    # ============ 按质心落带分配截面（v7：terrain 模式按局部埋深落带）============
    surf_fn = lambda xc: _surface_y_at(xc, H_upper, H_lower, left_flat, w_slope)  # 局部地表高度函数
    counts = _assign_sections_by_band(part, band_sections, surf_fn)  # 逐带按质心分配截面
    log_step(logger, '%s 截面属性分配完成: %s', model_name,  # 记录截面分配日志
             ', '.join('%s=%d' % (n, c) for n, c in counts))  # 输出各带面数

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


def create_flat_model(site, geom, mesh_size, logger=None, damping=None,
                      surface_geometry='horizontal', elem_name='CPE4'):
    """创建二维平面应变平坦自由场模型：矩形几何、材料、截面、装配与网格。

    site: Site 对象（基岩 + 有限层列表 + 基岩厚度，支持 1/2/3... 层）
    geom: Geometry 对象（取其总长、平坦总高 H_flat、基岩厚度；全场各层带齐全）
    damping: 解析后的阻尼配置 dict（与斜坡模型同参，保证 TAF=坡地/自由场 的分母口径一致）
    surface_geometry: v7 表层几何模式（平坦地表两种模式等价，但带结构须与斜坡模型同口径）
    elem_name: v7 单元类型 'CPE4'(默认) / 'CPE4R'
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

    # ============ 逐层创建材料与截面（平坦自由场，全场各带齐全）============
    strat = _build_stratigraphy(site, geom, surface_geometry=surface_geometry)  # 构造从下到上的分层带（v7：与斜坡同口径）
    band_sections = _create_band_materials_sections(model, strat, damping)  # 逐带创建材料与截面（含瑞利阻尼，与斜坡同参）
    log_step(logger, '%s 已创建 %d 个材料带的材料与截面（平坦自由场）', model_name, len(strat))  # 记录材料/截面创建日志

    # 装配
    assembly = model.rootAssembly  # 获取装配体对象
    assembly.DatumCsysByDefault(CARTESIAN)  # 设置默认笛卡尔坐标系
    inst_name = _next_available_name(part_name, assembly.instances)  # 生成实例名称
    assembly.Instance(name=inst_name, part=part, dependent=ON)  # 创建零件实例

    # ============ 水平切分各材料界面（v7：沿地形界面在平坦模型中退化为水平线）============
    interfaces, depth_ifaces = _interface_partitions(strat)  # 提取水平界面与沿地形界面埋深
    interfaces = sorted(interfaces + [H_flat - d for d in depth_ifaces])  # 平坦地表：埋深界面 = H_flat − d 的水平线
    _partition_horizontal(model, part, geom, interfaces, 'flat_hpartition')  # 逐条水平切分
    log_step(logger, '%s 平坦自由场水平界面切分完成: 界面数=%d', model_name, len(interfaces))  # 记录切分日志

    picked_regions = part.faces  # 选取全部面作为网格区域
    part.setMeshControls(regions=picked_regions, elemShape=QUAD, technique=STRUCTURED)  # 设置结构化四边形网格
    part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)  # 设置网格种子尺寸
    elem_code = CPE4R if str(elem_name).upper() == 'CPE4R' else CPE4  # v7：按 mesh_cfg['elem'] 选择单元类型
    elem_type = mesh.ElemType(elemCode=elem_code, elemLibrary=STANDARD)  # 定义单元类型
    part.setElementType(regions=(picked_regions,), elemTypes=(elem_type,))  # 分配单元类型
    part.generateMesh()  # 生成网格
    log_step(logger, '%s 平坦模型网格已生成: %s, 尺寸=%.2f', model_name, str(elem_name).upper(), mesh_size)  # 记录网格日志

    # ============ 按质心落带分配截面（平坦地表：地表高程恒为 H_flat）============
    counts = _assign_sections_by_band(part, band_sections, lambda _xc: H_flat)  # 逐带按质心分配截面（v7）
    log_step(logger, '%s 截面属性分配完成（平坦自由场）: %s', model_name,  # 记录截面分配日志
             ', '.join('%s=%d' % (n, c) for n, c in counts))  # 输出各带面数

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
    """逐边界逐节点用射线法计算自由场并组装等效节点力时程，返回 {'<label>-<边界>-fx/fy': Nx2 数组}。

    等效力 = K·u_ff + C·v̇_ff + A·σ_ff，其中应力 σ_ff 的各边界公式已内嵌外法向符号
      （见 _compute_freefield_at_node），故此处面力项统一取 +A·σ：
      侧边(l/r)：fx=kn·ux+cn·u̇x+A·σx, fy=kt·uy+ct·u̇y+A·σy；
      底边(b)  ：fx=kt·ux+ct·u̇x+A·σx, fy=kn·uy+cn·u̇y+A·σy。
    角点处理：左下/右下角点同属侧边与底边两个集合，会各算一次并叠加（VAB 角点标准处理，不折半）。
    时间轴：射线法按到时延迟会延长时程，故各节点力时程取其自身（延长后）时间轴，不截断到原长。
    """  # 说明函数用途与外法向/角点约定
    field_data = {}  # 初始化等效力缓存字典
    geom = ctx.geom  # 取出几何对象
    engine = (ctx.ffcfg or {}).get('engine', 'ray')  # v6：自由场引擎选择（'fd' 或 'ray'）
    get_vel = None  # 射线法速度延迟缓存（fd 引擎不需要）
    get_dis = None  # 射线法位移延迟缓存（fd 引擎不需要）
    if engine != 'fd':  # 仅射线法路径需要延迟缓存
        get_vel = _make_delay_cache(ctx.VEL, ctx.dt)  # 速度时程延迟缓存（跨节点复用）
        get_dis = _make_delay_cache(ctx.DIS, ctx.dt)  # 位移时程延迟缓存（跨节点复用）
    for boundary in BOUNDARY_SEQUENCE:  # 遍历每个边界
        for bn in nodes_by_boundary[boundary]:  # 遍历该边界每个节点
            # 确定当前节点所在柱子的地表高度 ymax_col（#2：底边按 x 取值）
            if boundary == 'l':  # 左边界
                ymax_col = ctx.ymax_l  # 左边界柱地表高度
            elif boundary == 'r':  # 右边界
                ymax_col = ctx.ymax_r  # 右边界柱地表高度
            else:  # 底边界
                ymax_col = _surface_y_at(bn.x, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 该底节点正上方地表高度

            if engine == 'fd':  # v6 默认：频域精确分层自由场
                ff = _fd_freefield_at_node(boundary, bn.x, bn.y, ymax_col, ctx)  # fd 引擎自由场时程
            else:  # 回归对比：v5 射线法
                ff = _compute_freefield_at_node(boundary, bn.x, bn.y, ymax_col, ctx, get_vel, get_dis)  # 射线法自由场时程

            time = ff['time']  # 延长后的时间轴
            ux = ff['ux']; uy = ff['uy']  # 位移分量
            dotux = ff['dotux']; dotuy = ff['dotuy']  # 速度分量
            sigmax = ff['sigmax']; sigmay = ff['sigmay']  # 应力分量（已含外法向符号）

            if boundary in ('l', 'r'):  # 侧边界：x 为法向、y 为切向
                fx = bn.kn * ux + bn.cn * dotux + bn.influence * sigmax  # x 向等效力（法向弹簧+阻尼+面力）
                fy = bn.kt * uy + bn.ct * dotuy + bn.influence * sigmay  # y 向等效力（切向弹簧+阻尼+面力）
            else:  # 底边界：x 为切向、y 为法向
                fx = bn.kt * ux + bn.ct * dotux + bn.influence * sigmax  # x 向等效力（切向弹簧）
                fy = bn.kn * uy + bn.cn * dotuy + bn.influence * sigmay  # y 向等效力（法向弹簧）

            field_data['{}-{}-fx'.format(bn.label, boundary)] = np.column_stack((time, fx))  # 缓存 x 向力时程
            field_data['{}-{}-fy'.format(bn.label, boundary)] = np.column_stack((time, fy))  # 缓存 y 向力时程
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
                acc_file=None, step_name=None, logger=None, tcfg=None, fc_used=None,
                ffcfg=None, damping=None, surface_geometry='horizontal'):
    """为二维模型施加粘弹性人工边界（弹簧-阻尼器）与斜入射 SV 波等效节点力。

    site     : Site 对象（基岩 + 有限层列表 + 基岩厚度，支持 1/2/3... 层）
    geom     : Geometry 对象（几何，含 H_upper/H_lower/left_flat/w_slope/bedrock_thickness/layer_interfaces）
    angle    : SV 波入射角（度）
    tcfg     : time_cfg 配置 dict（项③时间步校验）；None=跳过检查
    fc_used  : 网格/阻尼已用主频(Hz)；提供时用于 fmax 校验（不提供则从记录自动估计）
    ffcfg    : freefield_cfg 配置 dict（v6：'fd'=频域精确分层自由场 / 'ray'=v5 射线法）
    damping  : 解析后的阻尼配置 dict（v6：fd 引擎据此使自由场衰减与 FE 介质一致）
    surface_geometry: v7 表层几何模式（与建模同口径，决定边界弹簧选材与自由场柱分层）
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

    # 材料参数计算与场地分层
    mat_bedrock = _compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 计算基岩材料参数
    strat = _build_stratigraphy(site, geom, ymin=0.0, surface_geometry=surface_geometry)  # 构造场地分层带（v7：与建模同口径）
    _strat_params = [_compute_material_params(b['mat'].cs, b['mat'].vv, b['mat'].density) for b in strat]  # 各带材料派生参数（弹簧系数用）

    # 获取模型尺寸（左/右边界最高点与底边 y）
    l_nodes = get_instance_nodes_from_part_set('Left_boundary')  # 获取左边界节点
    ymax_l = max(l_nodes, key=lambda node: node.coordinates[1]).coordinates[1]  # 记录左边界最高 y 坐标

    b_nodes = get_instance_nodes_from_part_set('Bottom_boundary')  # 获取底边节点
    ymin = b_nodes[0].coordinates[1]  # 记录底边 y 坐标

    r_nodes = get_instance_nodes_from_part_set('Right_boundary')  # 获取右边界节点
    ymax_r = max(r_nodes, key=lambda node: node.coordinates[1]).coordinates[1]  # 记录右边界最高 y 坐标

    ymax = max(ymax_l, ymax_r)  # 取左右边界最高点中的较大值（弹簧刚度参考长度 R）

    # 按节点所在材质层选择材料参数（v7：经 _band_bounds_at 按局部柱落带，支持任意层数与 terrain 模式）
    def pick_material(x_coord, y_coord):  # 定义按节点坐标选择材料的函数
        ys = _surface_y_at(x_coord, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 该节点所在柱的地表高程
        for band, params in zip(strat, _strat_params):  # 从下到上遍历各材料带
            y0, y1 = _band_bounds_at(band, ys)  # 按柱地表换算该带上下界（horizontal 模式=标称值）
            if y0 - 1e-4 <= y_coord < y1 + 1e-4:  # 判断节点 y 是否落入该带
                return params  # 返回该带材料参数
        return _strat_params[-1]  # 兜底返回最顶带材料参数

    # 构建三条边界的节点列表（含影响长度与弹簧-阻尼系数）
    nodes_by_boundary = {  # 各边界 -> BoundaryNode 列表
        'l': _make_boundary_nodes(l_nodes, 'y', False, pick_material, ymax),  # 左边界（沿 y 降序）
        'r': _make_boundary_nodes(r_nodes, 'y', False, pick_material, ymax),  # 右边界（沿 y 降序）
        'b': _make_boundary_nodes(b_nodes, 'x', True, pick_material, ymax),  # 底边界（沿 x 升序）
    }  # 结束节点列表构建
    log_step(logger, '%s 边界节点影响长度与弹簧-阻尼系数已计算', model_name)  # 记录计算日志

    _add_spring_dashpots(assembly, instance, nodes_by_boundary, model_name, logger)  # 施加接地弹簧-阻尼器

    # ============ 入射角处理与水平慢度 ============
    if angle == 0:  # 判断入射角是否为零
        angle = 1e-10  # 用极小角度替代零角度
    else:  # 处理非零角度
        angle = round(angle, 4)  # 保留四位小数
    alpha1 = math.radians(angle)  # 将角度转换为弧度（基岩 SV 入射角）

    cs1 = mat_bedrock['cs']  # 基岩剪切波速
    cp1 = mat_bedrock['cp']  # 基岩纵波波速
    # ── v8：SV 入射临界角校验（§3.1-A2 硬化） ──────────────────────────────────
    # 基岩中 SV→P 临界角 = asin(cs/cp)（ν=0.3 时≈32.31°）；超临界后自由面反射 P 为非均匀波，
    # ray 引擎实角公式与 TAF 解析分母 factor_h 均失效，故达到临界角直接拒绝建模；
    # 论文工况以 30° 为上限，超过时仅告警。
    crit_deg = math.degrees(math.asin(cs1 / cp1))  # 基岩 SV 临界角（度）
    if angle >= crit_deg - 1e-6:  # 入射角达到/超过临界角
        raise ValueError('入射角 %.2f° >= 基岩临界角 %.2f°（超临界非均匀波不在本方法适用域内）' % (angle, crit_deg))  # 拒绝建模
    if angle > 30.0:  # 超过论文采用的上限
        log_step(logger, '%s 警告: 入射角 %.2f° > 30°（论文上限），已接近临界角 %.2f°，结果须谨慎使用',
                 model_name, angle, crit_deg)  # 输出告警
    p_horiz = math.sin(alpha1) / cs1  # 水平慢度（Snell 守恒，全场不变）
    beta1 = _safe_arcsin(cp1 * math.sin(alpha1) / cs1)  # 基岩 P 波反射角
    order_count = max(0, int(MAX_REFLECT_ORDER))  # 几何级数截断阶数（仅 ray 引擎使用）
    _REFL_COEFF_CACHE.clear()  # 清空等效系数缓存（不同模型/入射角不可复用）
    _FD_SOLVER_CACHE.clear()  # v6：清空 fd 柱解与输入谱缓存（不同模型/输入不可复用）
    ffcfg_used = dict(freefield_cfg)  # 以全局默认为底
    if ffcfg:  # 传入了自由场配置
        ffcfg_used.update(ffcfg)  # 覆盖默认
    ffcfg_used['tail_seconds'] = float((tcfg or {}).get('tail_seconds', 0.0) or 0.0)  # v8：静默尾段（fd 自由场时窗延长）
    damp_terms = _band_damping_terms(strat, damping)  # v6：各材料带瑞利系数表（fd 自由场衰减一致化）
    log_step(logger, '%s 自由场引擎=%s(阻尼一致化=%s): 入射角=%.4f°, 水平慢度 p=%.6e, 层数(含基岩)=%d',  # 记录引擎与参数
             model_name, ffcfg_used.get('engine'), ffcfg_used.get('include_damping'),  # 引擎与阻尼开关
             angle, p_horiz, len(strat))  # 输出入射角/慢度/层数

    # ============ 读取加速度时程并积分（保留 v7 基线校正）============
    # [输入幅值约定（#5）] 加速度记录积分得到的速度被当作"基底入射上行 SV 波"幅值 E；
    #   自由岩面对应 2E（自由面效应），TAF=PGA_slope/PGA_flat 取比值时该归一化抵消。
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

    # ── 项③：时间步充分性校验（安全护栏） ────────────────────────────────────────
    # 当前 dt=1e-3 对 fmax≈20Hz 约有 50 步/周期，远超 20 步门槛，不触发重采样。
    # 此护栏为换用粗采样输入波时提供自动保障，不影响常规 Ricker 4/6/8Hz 运行。
    if tcfg and tcfg.get('check'):  # 启用时间步校验
        fc_for_check = fc_used if fc_used else _estimate_dominant_freq(acc, dt)  # 取已知主频或自动估计
        fmax_check = float(time_cfg.get('fmax_factor', 2.5) if hasattr(time_cfg, 'get') else 2.5) * fc_for_check  # fmax 估计
        min_steps = float(tcfg.get('min_steps_per_fmax_period', 20))  # 最低步/周期要求
        steps_per_period = (1.0 / fmax_check) / dt if (fmax_check > 0 and dt > 0) else 9999.0  # 实际步/周期
        log_step(logger, '%s 时间步校验: dt=%.4fs, fmax=%.2fHz, 步/周期=%.1f(要求>=%.0f)',
                 model_name, dt, fmax_check, steps_per_period, min_steps)  # 记录校验信息
        if steps_per_period < min_steps and tcfg.get('resample_if_violate', True):  # 不满足且需重采样
            dt_new = (1.0 / fmax_check) / min_steps  # 目标步长（保证最低步/周期）
            log_step(logger, '%s 时间步不足，自动重采样: dt %.4fs -> %.4fs', model_name, dt, dt_new)  # 记录重采样
            t_new = np.arange(time_arr[0], time_arr[-1] + dt_new * 0.5, dt_new)  # 新时间轴
            acc = np.interp(t_new, time_arr, acc)  # 重采样加速度
            time_arr = t_new  # 更新时间轴
            dt = dt_new  # 更新步长
    # ── 时间步校验结束 ───────────────────────────────────────────────────────────

    # 积分得到速度时程（梯形积分 + 基线校正，抑制低频漂移），再积分得到位移时程
    vel, _vel_slope = _integrate_acc_to_velocity(acc, dt, time_arr)  # 加速度→速度（含基线校正）
    log_step(logger, '%s 速度基线校正完成: 去趋势斜率=%.3e', model_name, _vel_slope)  # 记录基线校正日志
    dis = np.zeros_like(vel)  # 初始化位移数组
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 速度梯形积分得到位移
    VEL = np.column_stack((time_arr, vel))  # 组合速度时程 [t, v]
    DIS = np.column_stack((time_arr, dis))  # 组合位移时程 [t, u]

    # ============ 逐节点用射线法计算自由场并组装等效力 ============
    ctx = FreeFieldCtx(  # 打包等效力计算所需上下文（fd 与 ray 引擎共用）
        site=site, geom=geom, strat=strat,  # 场地、几何、分层带
        ymax_l=ymax_l, ymax_r=ymax_r, ymin=ymin,  # 各边界高度信息
        alpha=alpha1, beta_p=beta1, p_horiz=p_horiz,  # 基岩入射角/P 反射角、水平慢度
        GG=mat_bedrock['GG'], lam=mat_bedrock['lam'], cs=cs1, cp=cp1,  # 基岩材料标量（投影/应力用）
        VEL=VEL, DIS=DIS, dt=dt, time_arr=time_arr, max_reflect_order=order_count,  # 时程、步长、阶数
        acc=acc, damp_terms=damp_terms, ffcfg=ffcfg_used)  # v6：原始加速度、各带瑞利系数、引擎配置
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
                 step_name=DEFAULT_STEP_NAME, model_scene='slope', logger=None,
                 tcfg=None, fc_used=None, ffcfg=None, damping=None, surface_geometry='horizontal',
                 surface_only=False):
    """根据加速度时程信息批量复制模型、创建分析步、施加人工边界。

    site/geom : 场地材料与几何对象（直接转发给 VAB_oblique）
    angle     : SV 波入射角（度）
    job       : 作业配置 dict，读取 'variables'（场输出变量）与 'frequency'（输出频率）
    tcfg      : time_cfg 配置（项③）；None=跳过时间步校验
    fc_used   : 网格/阻尼已用主频(Hz)，转发给 VAB_oblique 用于 fmax 估计
    ffcfg     : 自由场引擎配置（v6），转发给 VAB_oblique
    damping   : 解析后阻尼配置（v6），转发给 VAB_oblique 用于 fd 自由场衰减一致化
    surface_geometry: v7 表层几何模式，转发给 VAB_oblique（须与 create_model 同口径）
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
        tail = float((tcfg or {}).get('tail_seconds', 0.0) or 0.0)  # v8：静默尾段时长（与 fd 自由场时窗一致）
        model.ImplicitDynamicsStep(  # 创建隐式动力分析步
            name=step_name, previous='Initial',  # 设置分析步名称和前置步
            timePeriod=tp + tail, timeIncrementationMethod=FIXED, initialInc=inc,  # 设置时长（含尾段）和初始增量
            maxNumInc=1000000,  # 设置最大增量步数
            nlgeom=OFF, application=TRANSIENT_FIDELITY)  # 关闭几何非线性；TRANSIENT_FIDELITY(α≈-0.05)降数值阻尼，让物理材料阻尼主导

        model.fieldOutputRequests['F-Output-1'].setValues(  # 设置场输出请求
            variables=variables, frequency=frequency)  # 指定输出变量和频率
        if surface_only:  # v8：仅地表节点集输出全时程（ODB 瘦身，服务频域框架"全时程提取"）
            surf_region = model.rootAssembly.allInstances[inst_name].sets['TOP_SURFACE']  # 地表节点集（实例级）
            model.FieldOutputRequest(name='F-Output-Surface', createStepName=step_name,  # 新建地表场输出请求
                                     variables=('A', 'U'), frequency=frequency, region=surf_region)  # 地表 A/U 每增量步
            model.fieldOutputRequests['F-Output-1'].setValues(  # 整体场输出降频（仅留抽检帧）
                variables=variables, frequency=10000000)  # 设为极大间隔（几乎只输出首末帧）
            log_step(logger, '%s 输出瘦身: TOP_SURFACE 全时程 A/U + 整体场输出降频', new_model_name)  # 记录瘦身日志

        mdb.save()  # 保存模型数据库
        log_step(logger, '%s 分析步已创建, 时长=%.2f(含尾段 %.2f), 增量=%.3f',  # 记录分析步创建日志
                 new_model_name, tp + tail, tail, inc)  # 输出时长和初始增量

        VAB_oblique(site=site, geom=geom, angle=angle,  # 调用人工边界构建函数（传入场地与几何对象）
                    model_name=new_model_name, part_name=part_name, inst_name=inst_name,  # 传入模型/零件/实例名称
                    acc_file=acc_file, step_name=step_name, logger=logger,  # 传入加速度文件、分析步与日志器
                    tcfg=tcfg, fc_used=fc_used, ffcfg=ffcfg, damping=damping,  # 时间步校验 + v6 引擎/阻尼配置
                    surface_geometry=surface_geometry)  # v7：表层几何模式（与建模同口径）
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
            description='VAB oblique SV-wave analysis (Multi-layered slope)',  # 设置作业描述
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


def build_site(material_cfg, geometry_cfg):
    """由配置构建 Site 对象（基岩 + 从上到下的有限层列表），并校验层厚约束。

    单层(layers 为空)→ 仅基岩；双层 → 基岩 + 覆盖层；三层 → 基岩 + 表层 + 覆盖层。
    返回 (site, fixed_thicknesses)，fixed_thicknesses 为顶部各固定层厚度（从上到下，供 make_geometry 用）。
    """  # 说明函数用途与返回
    cs_bedrock = _compute_wave_speed_from_elastic_modulus(  # 由基岩弹性模量反算剪切波速
        material_cfg['bedrock']['elastic_modulus'],  # 基岩杨氏模量
        material_cfg['bedrock']['poisson_ratio'],  # 基岩泊松比
        material_cfg['bedrock']['density'])  # 基岩密度
    bedrock = Material(cs=cs_bedrock, vv=material_cfg['bedrock']['poisson_ratio'],  # 构建基岩材料（半空间）
                       density=material_cfg['bedrock']['density'], thickness=None, name='Bedrock')  # 基岩无固定厚度
    layers_cfg = material_cfg.get('layers', [])  # 读取有限层列表（从上到下）
    layers = []  # 初始化有限层 Material 列表
    fixed_thicknesses = []  # 初始化顶部固定层厚度列表（从上到下）
    nL = len(layers_cfg)  # 有限层数量
    for idx, lc in enumerate(layers_cfg):  # 自上而下遍历各有限层配置
        cs = cs_bedrock / lc['velocity_ratio']  # 由相对基岩波速比计算该层剪切波速
        is_bottom = (idx == nL - 1)  # 是否为最底有限层（覆盖层，厚度由几何决定）
        thickness = None if is_bottom else lc['thickness']  # 最底层厚度为 None，其余取固定厚度
        if not is_bottom:  # 非最底层须有固定厚度
            fixed_thicknesses.append(lc['thickness'])  # 记录固定厚度
        layers.append(Material(cs=cs, vv=lc['poisson_ratio'], density=lc['density'],  # 构建该有限层材料
                               thickness=thickness, name=lc['name']))  # 填入厚度与名称
    site = Site(bedrock=bedrock, layers=layers, bedrock_thickness=geometry_cfg['bedrock_thickness'])  # 组装场地对象
    # 层厚约束校验：覆盖层须有正厚度（坡顶 H - 顶部固定厚度之和 > 0）
    H = geometry_cfg['H_minus_h'] / (1.0 - geometry_cfg['h_over_H'])  # 总覆盖厚度 H
    if sum(fixed_thicknesses) >= H - 1e-6 and nL >= 1:  # 顶部固定层之和不得吃光覆盖层厚度
        raise ValueError('顶部固定层厚度之和(%.2f) >= 总覆盖厚 H(%.2f)，覆盖层无正厚度' %  # 抛出层厚错误
                         (sum(fixed_thicknesses), H))  # 输出冲突数值
    return site, fixed_thicknesses  # 返回场地对象与固定厚度列表


def _deep_merge(base, override):  # 递归合并两份配置字典
    """dict 逐键递归合并；其余类型（含 list，如 layers）整体替换。返回合并后的新 dict。"""
    out = dict(base)  # 复制基底，避免就地修改
    for k, v in (override or {}).items():  # 遍历覆盖项
        if isinstance(v, dict) and isinstance(out.get(k), dict):  # 双方均为 dict 才递归合并
            out[k] = _deep_merge(out[k], v)  # 递归合并子字典（如 bedrock）
        else:  # 其余情况（标量/列表/新键）
            out[k] = v  # 整体替换/新增（layers 列表整体替换即可改层数）
    return out  # 返回合并结果


def _load_case_config(material_cfg, geometry_cfg, mesh_size, damping_cfg, logger,
                      mesh_cfg_in=None, time_cfg_in=None, max_reflect_order_in=None):  # 加载工况配置注入
    """若当前工况文件夹存在 case_config.json，则用其覆盖默认配置（支持部分覆盖或整体改 layers/几何/网格/阻尼）。

    v6 新增支持 freefield_cfg / run_cfg 两类注入（自由场引擎与运行控制）。
    v7 注入要点：material_cfg 可含 "surface_geometry": "terrain"（表层沿地形等厚铺设），
    mesh_cfg 可含 "elem": "CPE4R"（减缩积分单元）——均经既有深合并机制生效，无需新键名。
    v8 注入要点：time_cfg 可含 "tail_seconds"（静默尾段秒数），run_cfg 可含 "surface_only"
    （地表全时程输出瘦身），damping_cfg 可含 "anchor": "dual"（场地基频双控锚定）。
    case_config.json 结构（各键可选，缺省即用默认）：
      {
        "material_cfg": {"surface_geometry": "terrain", ...}, "geometry_cfg": {...}, "mesh_size": 4,
        "damping_cfg": {...}, "mesh_cfg": {"elem": "CPE4"}, "time_cfg": {...},
        "max_reflect_order": 3,
        "freefield_cfg": {"engine": "fd"}, "run_cfg": {"run_flat": true}
      }
    返回覆盖后的 (material_cfg, geometry_cfg, mesh_size, damping_cfg, mesh_cfg, time_cfg,
                  max_reflect_order, freefield_cfg, run_cfg)。
    无文件或解析失败则原样返回默认。
    """
    mesh_cfg_out = dict(mesh_cfg_in) if mesh_cfg_in else dict(mesh_cfg)  # 初始化网格自适应配置（用全局默认）
    time_cfg_out = dict(time_cfg_in) if time_cfg_in else dict(time_cfg)  # 初始化时间步校验配置（用全局默认）
    max_reflect_order_out = max_reflect_order_in if max_reflect_order_in is not None else MAX_REFLECT_ORDER  # 反射阶数
    ff_cfg_out = dict(freefield_cfg)  # 初始化自由场引擎配置（用全局默认）
    run_cfg_out = dict(run_cfg)  # 初始化运行控制配置（用全局默认）
    path = os.path.join(os.getcwd(), 'case_config.json')  # 约定的配置注入文件
    if not os.path.isfile(path):  # 无注入文件 → 用默认配置单独运行
        if logger:  # 记录提示
            log_step(logger, '未发现 case_config.json，使用脚本内默认配置')  # 输出默认配置提示
        return (material_cfg, geometry_cfg, mesh_size, damping_cfg, mesh_cfg_out,  # 原样返回默认
                time_cfg_out, max_reflect_order_out, ff_cfg_out, run_cfg_out)  # 含 v6 新增两项
    try:  # 尝试读取并覆盖
        import json  # 导入 JSON 模块
        with io.open(path, 'r', encoding='utf-8') as f:  # 打开配置文件（io.open：Py2 内置 open 不支持 encoding 关键字）
            cfg = _ensure_str(json.load(f))  # 解析并递归转换 unicode→str（Py2 下 Abaqus API 只接受 str）
        if isinstance(cfg.get('material_cfg'), dict):  # 提供了材料覆盖
            material_cfg = _deep_merge(material_cfg, cfg['material_cfg'])  # 合并材料配置
        if isinstance(cfg.get('geometry_cfg'), dict):  # 提供了几何覆盖
            geometry_cfg = _deep_merge(geometry_cfg, cfg['geometry_cfg'])  # 合并几何配置
        if cfg.get('mesh_size') is not None:  # 提供了网格覆盖
            mesh_size = cfg['mesh_size']  # 覆盖网格尺寸
        if isinstance(cfg.get('damping_cfg'), dict):  # 提供了阻尼覆盖（批量调参/关阻尼）
            damping_cfg = _deep_merge(damping_cfg, cfg['damping_cfg'])  # 合并阻尼配置
        if isinstance(cfg.get('mesh_cfg'), dict):  # 项②：提供了网格自适应覆盖
            mesh_cfg_out = _deep_merge(mesh_cfg_out, cfg['mesh_cfg'])  # 合并网格自适应配置
        if isinstance(cfg.get('time_cfg'), dict):  # 项③：提供了时间步校验覆盖
            time_cfg_out = _deep_merge(time_cfg_out, cfg['time_cfg'])  # 合并时间步校验配置
        if cfg.get('max_reflect_order') is not None:  # 项④：提供了反射阶数覆盖
            max_reflect_order_out = int(cfg['max_reflect_order'])  # 覆盖反射截断阶数
        if isinstance(cfg.get('freefield_cfg'), dict):  # v6：提供了自由场引擎覆盖
            ff_cfg_out = _deep_merge(ff_cfg_out, cfg['freefield_cfg'])  # 合并自由场引擎配置
        if isinstance(cfg.get('run_cfg'), dict):  # v6：提供了运行控制覆盖
            run_cfg_out = _deep_merge(run_cfg_out, cfg['run_cfg'])  # 合并运行控制配置
        if logger:  # 记录成功
            log_step(logger, '已加载 case_config.json 覆盖默认配置: 入射角=%s, 层数(有限层)=%d, i=%s, 阻尼=%s/%s, mesh_auto=%s, order=%s, 引擎=%s, run_flat=%s',  # 输出关键覆盖项
                     material_cfg.get('angle'), len(material_cfg.get('layers', [])), geometry_cfg.get('i'),
                     damping_cfg.get('enable'), damping_cfg.get('method'),
                     mesh_cfg_out.get('auto'), max_reflect_order_out,
                     ff_cfg_out.get('engine'), run_cfg_out.get('run_flat'))  # 输出引擎与平坦模型开关
    except Exception as _e:  # 解析失败
        if logger:  # 记录告警
            log_step(logger, '加载 case_config.json 失败(改用默认配置): %s', str(_e))  # 输出失败告警
    return (material_cfg, geometry_cfg, mesh_size, damping_cfg, mesh_cfg_out,  # 返回（可能被覆盖的）配置
            time_cfg_out, max_reflect_order_out, ff_cfg_out, run_cfg_out)  # 含 v6 新增两项


def _meta_f(value):  # 把数值安全转为内置 float（兼容 numpy 标量）
    """将 numpy/字符串等数值规范化为内置 float；None 原样返回，不可转换返回 None。"""
    if value is None:  # 空值
        return None  # 原样返回
    try:  # 尝试转换
        return float(value)  # 返回内置浮点
    except (TypeError, ValueError):  # 不可转换
        return None  # 返回 None


def _meta_material(name, cs, vv, density, thickness=None):  # 打包单层材料为规范字典
    """返回 {name, cs, vv, density, thickness}；thickness=None 表示半空间或由几何决定。"""
    return {'name': str(name), 'cs': _meta_f(cs), 'vv': _meta_f(vv),  # 层名与波速、泊松比
            'density': _meta_f(density), 'thickness': _meta_f(thickness)}  # 密度与厚度


def _damping_meta(site, damping):  # 把阻尼配置与逐层换算结果打包为元数据块
    """返回阻尼元数据 dict（含逐层 cs/Q/xi/alpha/beta），供 case_meta.json 记录与下游核对。

    site    : Site 对象（基岩 + 从上到下有限层）；damping：解析后的阻尼配置（含 fc）。
    damping=None 或 enable=False 时返回 {'enable': False}。逐层顺序：基岩在前，再各有限层（从上到下）。
    """
    if not (damping and damping.get('enable')):  # 未启用阻尼
        return {'enable': False}  # 仅记录关闭状态
    fc = damping.get('fc')  # 解析后的主频
    per_layer = []  # 初始化逐层阻尼明细
    mats = [(site.bedrock, True)] + [(L, False) for L in site.layers]  # 基岩在前 + 各有限层（从上到下）
    for mat, is_bedrock in mats:  # 遍历各层（含基岩）
        Q, xi = _damping_ratio_from_q(mat.cs, is_bedrock, damping)  # 该层品质因子与阻尼比
        a_ray, b_ray = _rayleigh_coeffs(xi, damping, fc)  # 该层瑞利系数
        per_layer.append({'name': str(mat.name), 'cs': _meta_f(mat.cs),  # 层名与波速
                          'Q': _meta_f(Q), 'xi': _meta_f(xi),  # 品质因子与阻尼比
                          'alpha': _meta_f(a_ray), 'beta': _meta_f(b_ray)})  # 瑞利系数
    return {'enable': True, 'method': damping.get('method'), 'fc': _meta_f(fc),  # 总体配置
            'anchor': damping.get('anchor', 'input'), 'f_site': _meta_f(damping.get('f_site')),  # v8：锚定方式与场地基频
            'qs_factor': _meta_f(damping.get('qs_factor')), 'q_bedrock': _meta_f(damping.get('q_bedrock')),  # Q 换算因子
            'f1_factor': _meta_f(damping.get('f1_factor')), 'f2_factor': _meta_f(damping.get('f2_factor')),  # 双频拟合边界
            'layers': per_layer}  # 逐层阻尼明细（基岩在前）


def _write_case_meta(material_cfg, geom, site, mesh_size, script_name, logger, damping=None, ffcfg=None,
                     sgeom='horizontal', acc_path=None, selfcheck=None):  # 写出统一工况元数据（v7 理论台阶 + v8 自检）
    """把本工况参数固化为当前工况文件夹的 case_meta.json（自包含、不依赖任何外部模块）。失败仅告警、不中断建模。

    本函数是工况元数据的【唯一写入者 / 单一真相源】：所有派生量公式（a0_base、波速比、模型类型等）只在此处定义；
    下游 Collect/Plot 仅读取生成的 JSON 数据、不共享本处代码，因此不存在跨 Abaqus/普通 Python 的字段口径漂移。
    damping: 解析后的阻尼配置（含 fc），写入 'damping' 块（逐层 Q/xi/alpha/beta），供复现与追溯。
    ffcfg  : 自由场引擎配置（v6），写入 'freefield' 块。
    v6 新增 'ff_normalization' 块：TAF 解析分母（基岩半空间自由地表运动，论文式(5) 口径），
    供 Compute_TAF_v2.py 直接读取：PGA_ff_h = factor_h × max|输入加速度|。
    v7 新增：'surface_geometry' 键、geometry 块 x_crest/x_toe、derived 块 a0、
    'ff_theory' 块（用 fd 引擎算左/右远场柱的一维理论台阶 taf_h/taf_v，QA 锚点）。
    sgeom    : v7 表层几何模式；acc_path：首条输入记录路径（理论台阶用，None 则跳过 ff_theory）。
    返回 ff_theory dict（计算成功时），否则 None（供主流程日志打印）。
    """
    try:  # 元数据写出不应影响建模主流程
        bedrock = _meta_material(site.bedrock.name, site.bedrock.cs, site.bedrock.vv,  # 基岩材料字典
                                 site.bedrock.density, site.bedrock.thickness)  # 密度与厚度
        layers = [_meta_material(L.name, L.cs, L.vv, L.density, L.thickness) for L in site.layers]  # 各有限层（从上到下）
        geometry = {'i': geom.i, 'total_L': geom.total_L, 'left_flat': geom.left_flat,  # 几何输入项
                    'H_minus_h': geom.H_minus_h, 'h_over_H': geom.h_over_H,  # 斜坡高度差与深度比
                    'bedrock_thickness': geom.bedrock_thickness, 'H': geom.H, 'h': geom.h,  # 基岩厚度与覆盖厚度
                    'w_slope': geom.w_slope,  # 坡面水平长度
                    'x_crest': geom.left_flat, 'x_toe': geom.left_flat + geom.w_slope}  # v7：坡顶/坡脚 x（绘图 #1/#2 直接读取）
        geometry = {k: _meta_f(v) for k, v in geometry.items()}  # 几何统一转 float
        n_finite = len(layers)  # 有限层数（不含基岩）
        has_bedrock = site.bedrock is not None  # 是否存在基岩半空间
        n_total = n_finite + (1 if has_bedrock else 0)  # 总介质层数（含基岩）
        model_type = 'single' if n_total <= 1 else ('double' if n_total == 2 else 'multilayer')  # 模型类型判定
        Hmh = geometry.get('H_minus_h')  # 斜坡高度差 H-h
        slope_height = Hmh if Hmh is not None else geometry.get('h')  # 斜坡特征高度（a0 归一化用，单层退化用 h）
        vs_bedrock = bedrock['cs'] if has_bedrock else None  # 基岩剪切波速 Vr
        vs_surface = layers[0]['cs'] if layers else vs_bedrock  # 最顶有限层 Vs1（无有限层退化为基岩）
        vs_cover = layers[-1]['cs'] if layers else vs_surface  # 最底覆盖层 Vs2（a0 用）
        vr_over_vs2 = (vs_bedrock / vs_cover) if (vs_bedrock and vs_cover) else None  # Vr/Vs2 抗阻比
        vs1_over_vs2 = (vs_surface / vs_cover) if (vs_surface and vs_cover) else None  # Vs1/Vs2 软硬比
        a0_base = (2.0 * slope_height / vs_cover) if (slope_height and vs_cover) else None  # a0 = f_c(Hz) * a0_base
        fc_meta = (damping or {}).get('fc')  # 解析后主频（阻尼关闭/未估计时可能为 None）
        a0_val = (fc_meta * a0_base) if (fc_meta and a0_base) else None  # v7：无量纲频率 a0 = fc × a0_base
        derived = {  # 派生量集中区（公式单一真相源）
            'n_finite_layers': n_finite,  # 有限层数（不含基岩）
            'n_layers_total': n_total,  # 总层数（含基岩）
            'vs_bedrock': _meta_f(vs_bedrock),  # 基岩 Vr
            'vs_surface': _meta_f(vs_surface),  # 表层 Vs1
            'vs_cover': _meta_f(vs_cover),  # 覆盖层 Vs2
            'vr_over_vs2': _meta_f(vr_over_vs2),  # Vr/Vs2
            'vs1_over_vs2': _meta_f(vs1_over_vs2),  # Vs1/Vs2
            'slope_height': _meta_f(slope_height),  # a0 归一化用斜坡特征高度
            'a0_base': _meta_f(a0_base),  # a0 换算基数
            'a0': _meta_f(a0_val),  # v7：无量纲频率 a0（与论文工况对位用）
        }
        out_dir = os.path.abspath(os.getcwd())  # 当前工况文件夹（建模运行目录）
        meta = {  # 组装规范元数据（嵌套结构，供下游通用展平）
            'schema_version': 1,  # schema 版本号
            'model_type': model_type,  # 模型类型 single/double/multilayer
            'model_script': str(script_name),  # 建模脚本文件名
            'incident_angle': _meta_f(material_cfg['angle']),  # SV 入射角 θs（度）
            'surface_geometry': str(sgeom),  # v7：表层几何模式 horizontal/terrain
            'mesh_size': _meta_f(mesh_size),  # 网格尺寸（m）
            'geometry': geometry,  # 几何参数（含派生 H/h/w_slope）
            'bedrock': bedrock,  # 基岩材料字典
            'layers': layers,  # 从上到下的有限层材料字典列表
            'derived': derived,  # 派生量
            'damping': _damping_meta(site, damping),  # 材料阻尼块（逐层 Q/xi/alpha/beta，可复现）
            'record': None,  # 输入波记录名（以各 CSV 文件为准，留空）
            'extra': {},  # 附加自定义键值
            'folder': os.path.basename(out_dir.rstrip('/\\')),  # 工况文件夹名（来源标识）
        }
        # ── v6：TAF 解析分母（基岩半空间自由地表运动，论文式(5) 口径） ────────────
        ang_deg = float(material_cfg['angle'])  # SV 入射角（度）
        alpha_r = math.radians(ang_deg if abs(ang_deg) > 1e-12 else 1e-10)  # 入射角弧度（零角用极小值）
        mat_b = _compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)  # 基岩派生参数
        fs = _compute_free_surface_sv_coeff(alpha_r, mat_b['cp'], mat_b['cs'])  # 基岩自由面 SV 反射/转换系数
        factor_h = (1.0 - fs['A1']) * math.cos(alpha_r) + fs['A2'] * math.sin(fs['beta'])  # 水平分量放大系数（0°时=2）
        factor_v = -((1.0 + fs['A1']) * math.sin(alpha_r) + fs['A2'] * math.cos(fs['beta']))  # 竖向分量系数（0°时=0）
        meta['ff_normalization'] = {  # TAF 解析分母块（Compute_TAF_v2.py 读取）
            'method': 'bedrock_halfspace_free_surface',  # 分母口径：基岩半空间自由地表运动
            'A1': _meta_f(fs['A1']), 'A2': _meta_f(fs['A2']),  # 自由面 SV->SV 反射 / SV->P 转换系数
            'beta_deg': _meta_f(math.degrees(fs['beta'])),  # 自由面 P 波角（度）
            'factor_h': _meta_f(factor_h),  # PGA_ff_h = factor_h × max|输入加速度|
            'factor_v': _meta_f(factor_v),  # 竖向自由场系数（仅记录；论文式(5) 统一除以水平分母）
            'note': 'TAF_h=PGA_h/(factor_h*PGA_in); TAF_v=PGA_v/(factor_h*PGA_in)',  # 使用说明
        }
        meta['freefield'] = {  # v6 自由场引擎信息（追溯用）
            'engine': (ffcfg or {}).get('engine'),  # 引擎类型 fd/ray
            'include_damping': (ffcfg or {}).get('include_damping'),  # 自由场是否计入阻尼
        }
        # ── v7：远场一维理论台阶 ff_theory（自动 QA 锚点） ──────────────────────────
        # 用 fd 引擎对左(上平台 H_upper)/右(下平台 H_lower)边界柱计算地表加速度时程，
        # 按 TAF = PGA / (factor_h × PGA_in) 口径得理论台阶；FE 远场平台值应与之一致(±5%)，
        # 由 Compute_TAF_v3 自动核对——任何网格/阻尼/引擎回归当场暴露。
        ff_theory = None  # 理论台阶块初始化
        if acc_path and os.path.isfile(acc_path):  # 提供了输入记录时才计算
            try:  # 理论台阶计算失败不影响元数据主体
                rec = np.loadtxt(acc_path)  # 读取输入记录 [time, acc]
                acc0 = rec[:, 1]  # 加速度列
                dt0 = float(rec[1, 0] - rec[0, 0])  # 时间步长
                strat_t = _build_stratigraphy(site, geom, ymin=0.0, surface_geometry=sgeom)  # 与建模同口径分层
                damp_terms = _band_damping_terms(strat_t, damping)  # 各带瑞利系数（与 FE 介质一致）
                p0 = math.sin(alpha_r) / mat_b['cs']  # 水平慢度（基岩入射角，Snell 守恒）
                Nfft = _next_pow2(len(acc0) * 4)  # FFT 长度（补零 4 倍防卷绕）
                A0 = np.fft.rfft(acc0, n=Nfft)  # 加速度单边谱
                freqs0 = np.fft.rfftfreq(Nfft, dt0)  # 频率轴
                mask0 = np.abs(A0) > 1e-7 * float(np.max(np.abs(A0)))  # 谱幅值掩码（同 fd 引擎 tol）
                mask0[0] = False  # 排除直流分量
                idx0 = np.nonzero(mask0)[0]  # 被求解频点索引
                om0 = 2.0 * math.pi * freqs0[idx0]  # 对应圆频率
                incl = bool((ffcfg or {}).get('include_damping', True))  # 是否计入阻尼（与 fd 引擎一致）
                denom0 = factor_h * float(np.max(np.abs(acc0)))  # 解析分母 = factor_h × PGA_in
                ff_theory = {'fc_used': _meta_f((damping or {}).get('fc')),  # 瑞利拟合主频
                             'damped': bool(damping and damping.get('enable') and incl),  # 理论值是否含阻尼
                             'note': 'fd 引擎一维柱地表 PGA/(factor_h*PGA_in)；FE 远场台阶应与之一致(±5%)'}  # 使用说明
                for tag, ys in (('left', geom.H_upper), ('right', geom.H_lower)):  # 左(上平台)/右(下平台)两柱
                    col_t = _build_column(strat_t, ys, p0, 0.0)  # 构造该柱层段（terrain 模式自动含表层）
                    sol_t = _fd_solve_column(col_t, p0, om0, damp_terms, incl)  # 频域求解（单位入射）
                    fld = _fd_eval_column(sol_t, om0, p0, ys)  # 地表场量谱（单位入射）
                    spec_t = np.zeros(len(freqs0), dtype=complex)  # x 向加速度全频谱容器
                    spec_t[idx0] = fld['ux'] * A0[idx0]  # a(ω) = u_unit × A（位移两次积分与 −ω² 相消）
                    ax0 = np.fft.irfft(spec_t, n=Nfft)  # x 向加速度时程
                    spec_t = np.zeros(len(freqs0), dtype=complex)  # y 向加速度全频谱容器
                    spec_t[idx0] = fld['uy'] * A0[idx0]  # y 向加速度谱
                    ay0 = np.fft.irfft(spec_t, n=Nfft)  # y 向加速度时程
                    ff_theory[tag] = {'taf_h': _meta_f(float(np.max(np.abs(ax0))) / denom0),  # 该柱水平理论台阶
                                      'taf_v': _meta_f(float(np.max(np.abs(ay0))) / denom0),  # 该柱竖向理论台阶
                                      'surface_y': _meta_f(ys),  # 该柱地表高程
                                      'layers': [seg['name'] for seg in col_t]}  # 该柱层组成（自检用）
            except Exception as _fe:  # 理论台阶计算异常
                if logger:  # 有日志器时记录告警
                    log_step(logger, 'ff_theory 计算失败(不影响建模): %s', str(_fe))  # 输出告警
                ff_theory = None  # 置空
        meta['ff_theory'] = ff_theory  # v7：理论台阶块（可能为 None）
        meta['selfcheck'] = selfcheck  # v8：fd 引擎自检误差（halfspace_err/single_layer_err）
        text = json.dumps(meta, ensure_ascii=False, indent=2, default=_meta_f)  # 序列化为字符串（保留中文，default 兜底 numpy 标量）
        if isinstance(text, bytes):  # Py2 下 ensure_ascii=False 可能返回 bytes
            text = text.decode('utf-8')  # 解码为 unicode 以匹配 io.open 文本写入
        path = os.path.join(out_dir, 'case_meta.json')  # 目标路径
        with io.open(path, 'w', encoding='utf-8') as f:  # 以 UTF-8 文本模式打开（Py2 内置 open 不支持 encoding 关键字）
            f.write(text)  # 写出序列化文本
        if logger:  # 有日志器时记录
            log_step(logger, 'case_meta.json 已写出: %s', path)  # 输出成功日志
        return ff_theory  # v7：返回理论台阶块供主流程日志打印
    except Exception as _e:  # 捕获任何写出异常
        if logger:  # 有日志器时记录告警
            log_step(logger, 'case_meta.json 写出失败(不影响建模): %s', str(_e))  # 输出失败告警
        return None  # v7：失败时无理论台阶可返回


def main():
    """脚本主入口：组织参数、建模、施加边界并提交作业。"""  # 说明主入口用途
    global material_cfg, geometry_cfg, mesh_size, damping_cfg, MAX_REFLECT_ORDER  # 声明为全局以便用注入配置整体覆盖（所有 callee 均按值取用，安全）
    logger = log_step('VAB_oblique_TAF_multilayer.log')  # 初始化日志并写入当前版本日志文件
    total_start = time.time()  # 记录主流程起始时间

    try:
        log_step(logger, '脚本开始执行 (VAB_oblique_TAF_multilayer_v8)')  # 写入脚本启动日志

        # 配置注入：若工况文件夹有 case_config.json 则覆盖默认配置（v6 新增自由场引擎/运行控制注入）
        (material_cfg, geometry_cfg, mesh_size, damping_cfg,
         _mesh_cfg, _time_cfg, _max_reflect_order, _ff_cfg, _run_cfg) = _load_case_config(  # 加载并覆盖
            material_cfg, geometry_cfg, mesh_size, damping_cfg, logger)  # 传入默认配置与日志器
        MAX_REFLECT_ORDER = _max_reflect_order  # 全局更新反射截断阶数（仅 ray 引擎使用）

        # 构建场地材料（基岩 + 有限层列表）并校验层厚
        site, fixed_thicknesses = build_site(material_cfg, geometry_cfg)  # 由配置构建场地对象与固定层厚度
        n_total_layers = 1 + len(site.layers)  # 总层数（含基岩）
        sgeom = str(material_cfg.get('surface_geometry', 'horizontal'))  # v7：表层几何模式（可由 case_config.json 注入）
        if sgeom not in ('horizontal', 'terrain'):  # 校验模式合法性
            raise ValueError("surface_geometry 仅支持 'horizontal' 或 'terrain'，当前: %s" % sgeom)  # 抛出配置错误
        log_step(logger, '场地分层构建完成: 总层数(含基岩)=%d, 有限层=%s, 表层几何=%s',  # 记录分层信息
                 n_total_layers, [L.name for L in site.layers], sgeom)  # 输出层名与表层几何模式

        geom = make_geometry(  # 构建斜坡几何对象（含全部派生量）
            total_L=geometry_cfg['total_L'],  # 模型总长度
            H_minus_h=geometry_cfg['H_minus_h'],  # 斜坡高度差
            i=geometry_cfg['i'],  # 斜坡倾角
            h_over_H=geometry_cfg['h_over_H'],  # 深度比 h/H
            left_flat=geometry_cfg['left_flat'],  # 上平台长度
            bedrock_thickness=geometry_cfg['bedrock_thickness'],  # 基岩层厚度
            fixed_thicknesses=fixed_thicknesses)  # 顶部固定层厚度（推算固定层间界面）
        geom_flat = make_flat_geometry(geom)  # 派生平坦自由场几何

        acc_info = find_acc_txt(logger)  # 读取当前目录内全部加速度时程信息（提前，供主频估计与建模复用）

        # 解析材料阻尼：若未显式指定主频 fc，则用首条加速度记录估计（多记录同 fc 是标准用法；不同 fc 时以首条为准）
        fc_est = None  # 初始化主频估计值
        if damping_cfg.get('enable') and damping_cfg.get('fc') is None and acc_info:  # 启用阻尼且需自动估计 fc
            try:  # 估计失败不应中断建模
                _acc0 = np.loadtxt(acc_info[0][0])  # 读取首条加速度记录 [time, acc]
                fc_est = _estimate_dominant_freq(_acc0[:, 1], _acc0[1, 0] - _acc0[0, 0])  # 由首条记录估计主频
                log_step(logger, '阻尼主频自动估计: fc=%.3f Hz（源记录: %s）', fc_est, acc_info[0][0])  # 记录估计结果
            except Exception as _e:  # 估计异常
                log_step(logger, '阻尼主频估计失败(将依赖显式 fc 或回退): %s', str(_e))  # 输出告警
        damping = _resolve_damping(damping_cfg, fc_est)  # 解析阻尼配置（补全 fc，供建材与 meta 共用）
        fc_resolved = damping.get('fc')  # 解析后的主频（网格/时间步校验复用）
        if damping.get('enable') and damping.get('anchor') == 'dual':  # v8：双控锚定（场地基频+输入主频）
            _f_site = _site_fundamental_freq(site, geom)  # 估算上平台柱场地基频
            if _f_site:  # 估算成功
                damping['f_site'] = _f_site  # 写入解析后配置（建材/fd 自由场/meta 共用同一份，口径一致）
                log_step(logger, '阻尼双控锚定: f_site=%.3f Hz（瑞利拟合下限取 min(f1_factor*fc, f_site)）', _f_site)  # 记录锚定
        log_step(logger, '材料阻尼: enable=%s, method=%s, anchor=%s, fc=%s',  # 记录最终阻尼设定
                 damping.get('enable'), damping.get('method'), damping.get('anchor', 'input'), fc_resolved)  # 输出配置

        # ── 项②：网格自适应 ─────────────────────────────────────────────────────
        mesh_used = mesh_size  # 默认使用配置指定尺寸
        if _mesh_cfg.get('auto') and fc_resolved:  # 启用自适应且已知主频
            delta_l = _max_element_size(site, fc_resolved, _mesh_cfg)  # 按最软层/最高频计算 Δl_max
            mesh_used = min(mesh_size, delta_l)  # 取较小值（不超过 Kuhlemeyer-Lysmer 限值）
            if mesh_used < mesh_size:  # 自适应网格比配置值更细
                log_step(logger, '网格自适应: Δl_max=%.2fm -> mesh_used=%.2fm (原 mesh_size=%.2fm)',
                         delta_l, mesh_used, mesh_size)  # 记录网格细化日志
            else:  # 配置值已满足判据
                log_step(logger, '网格判据: Δl_max=%.2fm >= mesh_size=%.2fm，无需细化', delta_l, mesh_size)  # 记录日志
        else:  # 未启用自适应或主频未知
            log_step(logger, '网格尺寸: %.2fm（自适应未启用或 fc 未知）', mesh_used)  # 记录日志
        # ── 网格自适应结束 ──────────────────────────────────────────────────────

        selfcheck = _fd_engine_selfcheck(logger)  # v8：fd 引擎建模前自检（解析对拍，失败即中止，毫秒级）

        first_rec = acc_info[0][0] if acc_info else None  # v7：首条输入记录（理论台阶计算用）
        ff_theory = _write_case_meta(material_cfg, geom, site, mesh_used, _script_name(), logger,  # 写出统一工况元数据 case_meta.json
                                     damping=damping, ffcfg=_ff_cfg, sgeom=sgeom, acc_path=first_rec,  # 含 v7 ff_theory/x_crest/x_toe/a0
                                     selfcheck=selfcheck)  # v8：自检结果写入 meta
        if ff_theory and ff_theory.get('left') and ff_theory.get('right'):  # v7：打印理论台阶（QA 锚点）
            log_step(logger, '远场一维理论台阶: 左(上平台) TAF_h=%.3f TAF_v=%.3f | 右(下平台) TAF_h=%.3f TAF_v=%.3f (FE 远场应与之一致±5%%)',
                     ff_theory['left']['taf_h'], ff_theory['left']['taf_v'],  # 左柱理论值
                     ff_theory['right']['taf_h'], ff_theory['right']['taf_v'])  # 右柱理论值

        run_flat = bool(_run_cfg.get('run_flat', True))  # 是否运行平坦对照模型（v6 起仅作 QA，可关闭省算力）
        log_step(logger, '运行控制: run_flat=%s, 自由场引擎=%s', run_flat, _ff_cfg.get('engine'))  # 记录运行控制

        cae_name = 'h{}_i{}_a{}_L{}.cae'.format(int(geom.H_minus_h), int(geom.i),  # 生成工程文件名（含层数）
                                                int(material_cfg['angle']), n_total_layers)  # 文件名追加总层数

        base_model, part_name, inst_name = create_model(  # 创建斜坡基础几何与网格模型
            site=site, geom=geom, mesh_size=mesh_used, cae_name=cae_name, logger=logger, damping=damping,  # 场地/几何/自适应网格/文件名/阻尼
            surface_geometry=sgeom, elem_name=_mesh_cfg.get('elem', 'CPE4'))  # v7：表层几何模式 + 单元类型

        if run_flat:  # 需要平坦对照模型时才创建
            flat_base_model, flat_part_name, flat_inst_name = create_flat_model(  # 创建平坦自由场基础模型
                site=site, geom=geom, mesh_size=mesh_used, logger=logger, damping=damping,  # 场地/几何/自适应网格/阻尼（与斜坡同参）
                surface_geometry=sgeom, elem_name=_mesh_cfg.get('elem', 'CPE4'))  # v7：表层几何模式 + 单元类型

        slope_model_names = build_models(  # 批量复制斜坡模型并施加等效边界
            acc_info=acc_info, base_model=base_model, part_name=part_name, inst_name=inst_name,  # 地震动信息与基础模型/零件/实例
            site=site, geom=geom, angle=material_cfg['angle'],  # 场地、斜坡几何与入射角
            job=job_cfg, model_scene='slope', logger=logger,  # 作业配置与斜坡场景标签
            tcfg=_time_cfg, fc_used=fc_resolved, ffcfg=_ff_cfg, damping=damping,  # 时间步校验 + v6 引擎/阻尼
            surface_geometry=sgeom, surface_only=bool(_run_cfg.get('surface_only', False)))  # v7 表层几何 + v8 输出瘦身

        flat_model_names = []  # 平坦模型名称列表（默认空）
        if run_flat:  # 需要平坦对照模型时才建模
            flat_model_names = build_models(  # 批量复制平坦自由场模型并施加等效边界
                acc_info=acc_info, base_model=flat_base_model, part_name=flat_part_name, inst_name=flat_inst_name,  # 地震动信息与平坦基础模型/零件/实例
                site=site, geom=geom_flat, angle=material_cfg['angle'],  # 场地、平坦几何与入射角
                job=job_cfg, model_scene='flat', logger=logger,  # 作业配置与平坦场景标签
                tcfg=_time_cfg, fc_used=fc_resolved, ffcfg=_ff_cfg, damping=damping,  # 时间步校验 + v6 引擎/阻尼
                surface_geometry=sgeom, surface_only=bool(_run_cfg.get('surface_only', False)))  # v7 表层几何 + v8 输出瘦身

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
# v8 end  # 文件结束标记
