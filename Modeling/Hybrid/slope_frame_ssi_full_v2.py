# -*- coding: utf-8 -*-
"""
坡顶框架结构地震响应（TSSI）自包含建模脚本
================================================================================
1. 功能：在 v3 坡面波动引擎（上土下岩分层、粘弹性边界、斜入射等效力、EQL 非线性等）基础上，
   通过 tssi_cfg['enable'] 开关控制是否叠加坡顶多层框架（采用 Tie 耦合）。
2. 后处理：
   - 启用 TSSI：每波建立 SSI 模型，用 Postprocess/Hybrid/Postprocess_SSI_response_v1.py 处理。
   - 禁用 TSSI：退化为纯坡地模型，用 Postprocess/General/Postprocess_PGA_v3.py 处理。
3. 运行方式：abaqus cae noGUI=slope_frame_ssi_full_v1.py （工作目录存放加速度记录文件）
4. 兼容性：兼容 Python 2.7。
"""

from abaqus import *
from abaqusConstants import *
from abaqus import mdb
from regionToolset import Region
from caeModules import *
import mesh
import numpy as np
import math
import os
import io
import json
import sys
import time
import logging
import traceback
from collections import namedtuple


# ==========================================================
#  配置参数（默认值定义在这里，工况目录下的 case_config.json 可覆盖这些默认值）
# ==========================================================

# 默认材料参数配置
material_cfg = {
    'angle': 15,                        # SV 波入射角度（度）
    'surface_geometry': 'horizontal',   #! 表层几何 'horizontal'=固定高程水平带 / 'terrain'=沿地形等厚铺设
    # 基岩材料参数（剪切波速直接给定，杨氏模量由 E=2ρVs²(1+ν) 内部换算）
    'bedrock': {
        'vs': 2000.0,                   # 基岩剪切波速（m/s）
        'poisson_ratio': 0.3,           # 基岩泊松比
        'density': 2500,                # 基岩密度（kg/m^3）
    },
    # 基岩之上的土层（自顶向下配置，每层显式给定厚度；剩余深度全部归基岩；layers=[] 即全基岩坡）
    'layers': [
        {'name': 'surface',             # 表层名称
        'vs': 400.0,                    # 表层剪切波速（m/s）
        'poisson_ratio': 0.3,           # 该层泊松比
        'density': 2500,                # 该层密度（kg/m^3）
        'thickness': 50},               # 该层厚度（m）
        {'name': 'overlying',           # 覆盖层
        'vs': 800.0,                    # 覆盖层剪切波速（m/s）
        'poisson_ratio': 0.3,           # 覆盖层泊松比
        'density': 2500,                # 覆盖层密度（kg/m^3）
        'thickness': 350},              # 该层厚度（m，坡顶面以下 50+350=400，与旧默认模型一致）
    ],
}

# 几何参数配置
geometry_cfg = {
    'slope_height': 200.0,              # 坡高 hs = 坡顶与坡脚地表高程差 (m)——唯一绝对尺度
    'slope_angle': 45.0,                # 坡角 (度)
    'crest_window': 3.0,                # 坡顶观测窗（hs 倍数，计划书 A_max；地形放大 x/h≈3~4 已衰减回 1）
    'toe_window': 2.0,                  # 坡脚观测窗（hs 倍数，计划书 C_max）
    'side_clearance': 1.0,              # 侧向边界净空（hs 倍数，观测窗外留给 VAB 的距离）
    'base_depth': 2.0,                  # 坡脚面以下模型深度（hs 倍数，恒定不随地层变；土层扣完剩余全归基岩）
}

# 作业参数配置
job_cfg = {
    'variables': ('U', 'V', 'A'),       # 场输出变量（位移/速度/加速度）
    'frequency': 1,                     # 场输出频率（每隔该增量步输出一帧）
    'num_cpus': 8,                      # 并行 CPU 数量
    'memory_percent': 90,               # 作业内存百分比
}

# 材料阻尼配置
damping_cfg = {
    'enable': True,                     #! 是否施加材料阻尼（False 则退化为无阻尼行为）
    'method': 'rayleigh',               # 'rayleigh'=双频拟合(α+β，两端 ξ 相等≈恒定 Q) / 'stiffness'=仅刚度比例(β)
    'constant_xi': 0.01,                #! 统一恒定阻尼比(如 0.05)：None=关闭(按波速计算)；指定值时将忽略 qs_factor，对所有有限土层施加统一阻尼
    'qs_factor': 0.05,                  # Qs = qs_factor*cs（coarse-grain 法，cs 单位 m/s）
    'q_bedrock': 999.0,                 # 基岩品质因子(≈无衰减)
    'fc': None,                         # 输入波主频(Hz)：None=从加速度记录自动估计；可显式/注入覆盖
    'f1_factor': 0.5,                   # 双频拟合下限 = f1_factor*fc
    'f2_factor': 2.5,                   # 双频拟合上限 = f2_factor*fc（≈Ricker 高频边界）
    'anchor': 'perband',                # 拟合锚定 'perband'=逐层按各层共振频带重锚定 / 'input'=仅输入主频 / 'dual'=场地基频+输入主频双控
    'harmonics_cover': 3.0,             # perband 模式下拟合上限覆盖到各层共振基频的几次谐波（f2≥harmonics_cover·f_layer）
}

# 网格自适应配置
mesh_cfg = {
    'size': 4.0,                        # 基准/全局网格尺寸（m）。auto=True 时作为上限(mesh_used=min(size,Δl_max))；auto=False 时强制采用
    'auto': True,                       #! True=自动按最软层/最高频率计算 Δl_max（不超过 size）；False=强制使用 size
    'elems_per_wavelength': 10,         # 每波长最少单元数（论文取 10，即 Δl≤cs_min/(10·fmax)）
    'fmax_factor': 2.5,                 # fmax = fmax_factor*fc（Ricker 子波有效频带上限估计，覆盖 2~3σ 宽度）
    'min_size': 0.2,                    # 网格下限（m）：防止过软层或超高频时计算量爆炸
    'elem': 'CPE4R',                    #! 单元类型: 'CPE4'/'CPE4R'(线性) 或 'CPE8'/'CPE8R'(二次,低频散,边界自动用一致权重1/6:2/3)
    'graded': True,                     # 分层非均匀网格——按层波速比缩放单元尺寸(软层细/深部粗,自由四边形平滑过渡)；False=全局均匀
    'max_band_ratio': 4.0,              # 最粗层单元 ≤ 该倍数×最细层(=mesh_used)，限制过渡比以保证网格质量
    'max_size': None,                   # 单元绝对上限(m)，None=不额外限制(仅受 max_band_ratio 约束)
    'resolve_harmonics': 3.0,           # 薄软层加密——网格至少解析到各层共振基频 f_layer=cs/(4h) 的该倍数谐波(与 perband 阻尼 harmonics_cover 对齐)；0/None=关闭
    'min_elems_through_thickness': 6,   # 每个有限层厚度方向至少单元数(保证薄层驻波/混响形态)；该判据优先于 min_size
}

# 时间步配置
time_cfg = {
    'check': True,                      # True=仅诊断：输入 dt 偏粗(步/周期不足)时输出警告，但【不】改变 dt；False=连诊断也跳过
    'min_steps_per_fmax_period': 20,    # 诊断阈值：每 fmax 周期建议的最少步数（dt <= 1/(fmax*20) 视为充足）
    'tail_seconds': 0.0,                #! 静默尾段时长(s)——分析步与 fd 自由场时窗同步延长（H(f) 提取用）
}

# 自由场引擎配置
freefield_cfg = {
    'engine': 'fd',                     #! 'fd'=频域精确分层自由场（默认，含界面 SV<->P 与全部多次波）；'ray'=射线法（回归对比用）
    'include_damping': True,            # fd 引擎是否在自由场中计入与模型介质一致的瑞利阻尼
    'spectrum_tol': 1e-7,               # 仅求解幅值谱 > tol*max 的频率分量（其余置零，省时且高频数值稳定）
    'fcut': None,                       # 频率上限(Hz)：None=仅按谱幅值掩码自适应截断
    'pad_factor': 4,                    # FFT 补零倍数（>=2，防止时域卷绕污染响应窗口）
}

# 运行控制配置
run_cfg = {
    'surface_only': True,               # True=仅 TOP_SURFACE 输出 A/U 全时程+整体场输出降频（ODB 瘦身，频域框架用）
    'critical_angle_check': False,      # True=入射角达到/超过 SV→P 临界角时拒绝建模(硬性拦截)；False=仅输出警告不中断(探索超临界工况)
    'wave_files': None,                 #! 地震波文件路径：None=扫工况目录全部.txt(旧行为)/字符串或列表=指定文件(绝对路径或相对工况目录)
}

# 人工边界配置
boundary_cfg = {
    'dashpot_scale': 1.0,               # 阻尼器(吸收)cn,ct 缩放：1.0=全吸收(Liu)/0<k<1=弱吸收/0=纯弹簧全反射
    'spring_scale': 1.0,                # 弹簧(恢复)kn,kt 缩放：1.0=现行(α0.5/0.25)/2.0=标准Liu/0=纯黏性(弹簧关)
    'sponge_enable': False,             # 边界内侧阻尼海绵层开关(opt-in)：L/R/B 内侧加渐变阻尼带吸收残余反射
    'sponge_width': 0.0,                # 海绵带宽 m：0=自动 max(10×基准网格,8%域宽)；graded 网格远场粗须用域宽项
    'sponge_grades': 5,                 # 海绵分级数：阻尼从内缘 0 渐增到贴边界，级越多越平滑
    'sponge_xi_max': 0.3,               # 贴边界处附加阻尼比(占主频 fc)：海绵最外层 ξ 附加量，0.3≈强吸收
}

# 土体非线性：等效线性(EQL) 配置
eql_cfg = {
    'enable': False,                    #! True=建模前对软层做 EQL 应变相容(降Vs/增ξ)；False=保持线性(=原行为)
    'curve': 'darendeli',               # 经验曲线(可切换对比): 'darendeli'(通用) / 'seed_idriss_sand'(砂) / 'vucetic_dobry'(黏土)
    'nonlinear_layers': ['surface'],    # 参与非线性的层名(其余层保持线性；岩层一般不进入非线性)
    'PI': 15.0,                         # 塑性指数(Darendeli/Vucetic-Dobry 用)
    'sigma0_kpa': 100.0,                # 平均有效围压 kPa(Darendeli 用)
    'strain_ratio': 0.65,               # 有效剪应变 = 该比例 × 峰值剪应变
    'tol': 0.02,                        # 收敛容差(Vs 相对变化)
    'max_iter': 15,                     # 最大迭代次数(1D 内迭代)
    'mode': '1d',                       #! '1d'=1D应变相容→喂2D(默认,快/稳/可验证) / '2d_element'=逐单元2D EQL(重型,需Abaqus实测)
    'max_outer_iter': 4,                # 2d_element: 2D FE 重跑次数(外迭代)
    'n_strain_bins': 12,                # 2d_element: 按应变给软层单元分箱的箱数(控制材料数量)
    'converge_g': 0.05,                 # 2d_element: 外迭代收敛容差(各箱 G 相对变化)
}


# ==========================================================
#  TSSI 坡顶建筑/SSI 配置（与引擎配置并列在文件头）
# ==========================================================

# TSSI 总开关配置
tssi_cfg = {
    'enable': False,                     # True=在坡顶追加框架(Tie耦合); False=退化为 v3 纯坡地模型
    'scene': 'ssi',                      #! 三胞胎场景 'ssi'=全耦合(默认) / 'freefield'=纯坡地提取坡顶运动 / 'fixed'=固定基础框架单体
    'fixed_input': None,                 #! fixed 场景基底输入加速度 .txt 路径(绝对/相对工况目录); None=自动查找 crest_motion_*.txt
    'history_freq': 1,                   # 框架历史输出(U1/A1等)采样频率：每隔该增量步记录一次
    'nonlinear': True,                   # True=梁柱用CDP混凝土+钢筋(非线性纤维截面); False=退回纯弹性(step2 行为)
    'gravity': 'off',                    # 重力级别 'off'=现状(v1基线) / 'structure'=Level A 仅结构自重(动力步前静力步) / 'full'=Level B 全模型(P2,未实现)
    'crest_offset_B': 0.0,               # 距坡肩距离 M/B(0=右缘贴坡肩=v1基线; step4 扫描参数)
    'T_fixed': None,                     # 固定基础基本周期(s,周期延长基准); None=按层数 0.1N 估算(默认5层=0.5s=v1值), 有实测值时注入覆盖
    'nlgeom': False,                     # 几何非线性(P-Δ大位移) False=OFF(v1基线) / True=ON(强震层间角>1%时批量开)
    'cdp_min_inc_factor': 1.0e-4,        # P1#9 CDP动力步最小增量=initialInc×该系数(收敛降级链:不收敛时调小,见下方降级链注释)
}

# 坡顶框架（B21 梁 + 楼层集中质量；坐坡顶右缘贴坡肩 x=left_flat）
frame_cfg = {
    'n_story': 5,                        # 框架层数
    'n_bay': 3,                          # 框架跨数
    'story_height': 3.0,                 # 层高（m）
    'bay_width': 6.0,                    # 跨度（m）
    'column': {
        'width': 0.5,                    # 柱截面宽度（m）
        'depth': 0.5,                    # 柱截面高度（m）
    },
    'beam': {
        'width': 0.3,                    # 梁截面宽度（m）
        'depth': 0.6,                    # 梁截面高度（m）
    },
    'floor_mass': 5.0e4,                 # 每层楼板集中质量（kg）
}

# 框架材料配置（混凝土 C30，含瑞利阻尼拟合频段 + step3 CDP 本构参数）
frame_material_cfg = {
    'name': 'Concrete_C30',              # 材料名称
    'E': 30.0e9,                         # 弹性模量（Pa，GB50010 表4.1.5 C30 弹性模量 Ec）
    'nu': 0.2,                           # 泊松比
    'density': 10.0,                     # 密度（kg/m^3，已按结构等效折算；真实楼层质量走 floor_mass 集中质量）
    'damping_ratio': 0.05,               # 瑞利阻尼目标阻尼比
    'rayleigh_mode': 'fixed',            #! 拟合频段锚定 'fixed'=固定 f1/f2(v1基线) / 'modal'=按 T1 自动锚定(f1=0.8/T1,f2=5/T1 覆盖前三阶)
    'f1': 1.0,                           # 瑞利阻尼拟合下限频率（Hz，rayleigh_mode='fixed' 时用）
    'f2': 5.0,                           # 瑞利阻尼拟合上限频率（Hz，rayleigh_mode='fixed' 时用）
    'fc_mpa': 20.1,                      # 轴心抗压强度标准值 fck（MPa，GB50010 表4.1.3-1 C30）
    'ft_mpa': 2.01,                      # 轴心抗拉强度标准值 ftk（MPa，GB50010 表4.1.3-1 C30）
    'dilation_angle': 30.0,              # CDP 膨胀角（度，混凝土常用取值）
    'eccentricity': 0.1,                 # CDP 流动势偏心率(默认)
    'fb0_fc0': 1.16,                     # CDP 双轴/单轴抗压强度比(默认)
    'K': 0.6667,                         # CDP 拉压子午线形状系数(默认)
    'viscosity': 0.0005,                 # CDP 粘性正则化参数(小值,助收敛,不显著改变本构)
}

# 框架钢筋配置（HRB400，角部配筋；纤维截面用 *Rebar, element=BEAM 关键字注入，Abaqus/CAE 不支持梁钢筋图形化建模）
rebar_cfg = {
    'material': 'Rebar_HRB400',          # 钢筋材料名称
    'Es': 200.0e9,                       # 钢筋弹性模量（Pa）
    'nu': 0.3,                           # 钢筋泊松比
    'fy': 400.0e6,                       # 屈服强度（Pa，HRB400）
    'hardening_ratio': 0.01,             # 屈服后切线模量/Es(小值强化,防止理想弹塑性数值病态)
    'density': 7850.0,                   # 钢筋密度（kg/m^3，仅用于材料定义；梁钢筋质量不计入求解——已按 step1 说明用 floor_mass 集中质量代表结构质量）
    'cover': 0.03,                       # 保护层厚度（m，到钢筋中心近似简化，不再扣半径）
    'column': {
        'ratio': 0.015,                  # 柱配筋率(总钢筋面积/截面毛面积)，取常规RC柱经验值
        'bar_diameter': 0.022,           # 单根钢筋直径(m)，仅用于摆放校验，面积按配筋率反算
    },
    'beam': {
        'ratio': 0.008,                  # 梁配筋率
        'bar_diameter': 0.018,           # 单根钢筋直径(m)
    },
}

# 基础形式配置（P1#7/#8：柱脚下条形基础板 + 可选接触升级；默认 'tie' 保持 v1 柱脚点绑定行为）
foundation_cfg = {
    'type': 'tie',                       #! 'tie'=柱脚点绑定土面(v1基线) / 'footing'=柱脚下加实体条形基础板(CPE4)+分布式绑定土面
    'width': None,                       # 基础板宽(m)；None=按框架宽×1.2 自动(保证≥框架宽，两侧各挑出0.1倍)
    'thickness': 0.8,                    # 基础板厚(m)——坐在坡顶面上，顶面接柱脚
    'E': 30.0e9,                         # 基础混凝土弹性模量(Pa，C30)
    'nu': 0.2,                           # 基础泊松比
    'density': 2500.0,                   # 基础密度(kg/m³，混凝土；重力步与惯性均计入)
    'mesh_size': None,                   # 基础板网格尺寸(m)；None=取 thickness/2
    'contact': False,                    #! True=基础底与土面【硬接触+库仑摩擦】(可提离滑移,强震批C专用) / False=Tie绑定(默认)
    'mu': 0.5,                           # 土-基础库仑摩擦系数(contact=True 时用，μ≈0.4~0.6)
}


# ==========================================================
#  模块常量与全局状态
# ==========================================================

_DEFAULT_SCRIPT_NAME = 'slope_frame_ssi_full_v2.py'  # __file__ 缺失时的兜底文件名
DEFAULT_STEP_NAME = 'Step-earthquake'  # 默认分析步名称
GRAVITY_STEP_NAME = 'Step-gravity'  # 重力静力步名称（P0#1：动力步前施加自重）
GRAVITY_G = 9.81  # 重力加速度（m/s²，重力两步法与楼层节点力共用）
BOUNDARY_SET_NAMES = ('Left_boundary', 'Right_boundary', 'Bottom_boundary')  # 边界节点集名称
BOUNDARY_SEQUENCE = ('l', 'r', 'b')  # 边界处理顺序(左/右/底)

MAX_REFLECT_ORDER = 3   # ray 覆盖层多次反射截断阶数(默认3，可由 case_config.json 顶层 max_reflect_order 覆盖)
_REFL_COEFF_CACHE = {}  # ray 等效反射/转换系数缓存(运行时填充)
_FD_SOLVER_CACHE = {}  # fd 引擎缓存：键=round(ymax_col,4) 的柱解；另含 '_input' 键缓存输入谱


# ============================================================
#  参数打包对象（用结构化对象取代散标量，缩短函数签名、便于阅读）
# ============================================================
# Material：单层材料的基本输入（剪切波速、泊松比、密度、固定厚度、名称）；
#   派生量 GG/lam/cp/EE 仍由物理核心函数按需计算，故此处只存输入。
#   thickness=None 表示：基岩半空间，或最底有限层（覆盖层，厚度由几何决定）。
Material = namedtuple('Material', ['cs', 'vv', 'density', 'thickness', 'name'])  # 单层材料输入
# Site：基岩半空间 + 土层列表（layers 从上到下，厚度均显式给定）+ 基岩顶面高程（=坡顶地表−Σ土层厚）
Site = namedtuple('Site', ['bedrock', 'layers', 'bedrock_thickness'])  # 多层场地（支持 0/1/2... 个土层）
# Geometry：斜坡几何（外形输入项 + 一次算好的派生项；H/h/h_over_H 为派生记录量，h 可负=基岩坡面出露）
Geometry = namedtuple('Geometry', [
    'total_L', 'i', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness',  # 外形与地层派生
    'H', 'h', 'H_upper', 'H_lower', 'H_flat', 'w_slope',  # 派生项
    'layer_interfaces'])  # 派生项：土层间界面 y（从下到上，不含基岩顶面），用于切分与材料分配
# BoundaryNode：单个边界节点的几何与粘弹性边界参数（取代裸 numpy 列索引）
BoundaryNode = namedtuple('BoundaryNode',
                          ['label', 'x', 'y', 'influence', 'kn', 'cn', 'kt', 'ct'])  # 边界节点
# FreeFieldCtx：射线法等效力计算所需的上下文（一次打包，避免长参数列表）
#   推广要点：自由场按每个节点"所在水平成层柱"逐层计算，故携带场地分层(strat)而非单一覆盖层。
FreeFieldCtx = namedtuple('FreeFieldCtx', [  # 自由场上下文命名元组
    'site', 'geom', 'strat', 'ymax_l', 'ymax_r', 'ymin',  # 场地、几何、分层带、各边界高度信息
    'alpha', 'beta_p', 'p_horiz',  # 基岩 SV 入射角、基岩 P 反射角、水平慢度（Snell 守恒）
    'GG', 'lam', 'cs', 'cp',  # 基岩剪切模量/拉梅常数/波速（投影与应力公式使用基岩标量）
    'VEL', 'DIS', 'dt', 'time_arr', 'max_reflect_order',  # 速度/位移时程、步长、时间轴、反射阶数
    'acc', 'damp_terms', 'ffcfg'])  # 原始加速度记录、各带瑞利系数表、自由场引擎配置


# ==========================================================
#  通用工具函数
# ==========================================================

def _script_path():  # 安全获取当前脚本绝对路径（Abaqus 内核可能不定义 __file__）
    """返回脚本绝对路径；Abaqus 用 execfile/kernel 执行时全局可能无 __file__，此时退化为当前目录下的已知脚本名。"""
    f = globals().get('__file__')
    if f:  # __file__ 存在时
        return os.path.abspath(f)
    return os.path.join(os.getcwd(), _DEFAULT_SCRIPT_NAME)  # 兜底：当前工作目录(工况文件夹) + 已知脚本名


def _script_name():
    """返回脚本文件名（如 'VAB_oblique_TAF_multilayer_v8.py'），不依赖 __file__。"""
    return os.path.basename(_script_path())


def _script_dir():
    """返回脚本所在目录；__file__ 缺失时退化为当前工作目录。"""
    return os.path.dirname(_script_path())


def log_step(logger=None, message=None, *args):
    """日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。"""
    if not hasattr(log_step, '_logger'):
        if logger is not None and isinstance(logger, str):
            log_filename = logger
            logger = None
        else:
            script_name = _script_name()  # 获取当前脚本名
            log_filename = os.path.splitext(script_name)[0] + '.log'  # 使用与脚本同名的日志文件名

        _logger = logging.getLogger('abqpy')
        _logger.setLevel(logging.INFO)
        _logger.propagate = False  # 禁止向父日志器传播

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


def _next_available_name(prefix, existing_container):
    """按前缀生成可用名称（如 Part-1, Part-2）。"""
    index = 1
    while '%s-%d' % (prefix, index) in existing_container:  # 循环查找未占用名称
        index += 1  # 序号递增
    return '%s-%d' % (prefix, index)


def _normalize_output_variables(variables):
    """规范化输出变量为元组，满足 Abaqus 接口要求。"""
    if isinstance(variables, str):
        return (variables,)
    if isinstance(variables, list):
        return tuple(variables)
    return variables


def _safe_arcsin(value):
    """对 arcsin 输入做截断，避免浮点超界。"""
    return math.asin(max(-1.0, min(1.0, value)))  # 将输入截断到合法范围后再求反正弦


def _ensure_str(obj):  # 递归将 unicode 转为原生 str（Py2 兼容 Abaqus API）
    """递归将 json.load 产生的 unicode 转为 Python 2 原生 str（bytes），Py3 保持不变。

    Abaqus Python 2 的 C++ API（如 model.Material(name=...)）只接受 str，拒绝 unicode。
    而 json.load 在 Py2 下默认返回 unicode，因此加载 case_config.json 后必须转换。
    """
    if sys.version_info[0] >= 3:  # Py3：str 即 unicode，Abaqus 2020+ 原生接受
        return obj  # 无需转换
    # Py2 分支：json.load 返回 unicode，需递归转为 str
    if isinstance(obj, unicode):  # 字符串节点
        return obj.encode('utf-8')  # unicode → str（UTF-8 字节串）
    if isinstance(obj, dict):  # 字典节点
        return {_ensure_str(k): _ensure_str(v) for k, v in obj.items()}
    if isinstance(obj, list):  # 列表节点
        return [_ensure_str(item) for item in obj]
    if isinstance(obj, tuple):  # 元组节点
        return tuple(_ensure_str(item) for item in obj)
    return obj  # 数值等其他类型原样返回


# ==========================================================
#  物理与数值计算
# ==========================================================


def _compute_elastic_modulus_from_wave_speed(cs, vv, density):
    """根据剪切波速、泊松比和密度计算杨氏模量 E。"""
    GG = density * (cs ** 2)
    EE = 2 * GG * (1 + vv)
    return EE


def _compute_material_params(cs, vv, density):
    """根据 Vs、泊松比、密度计算材料参数。"""
    GG = density * cs ** 2
    EE = 2 * GG * (1 + vv)
    lam = 2 * GG * vv / (1 - 2 * vv)
    cp = math.sqrt((lam + 2 * GG) / density)
    return {'GG': GG, 'EE': EE, 'lam': lam, 'cp': cp, 'cs': cs, 'vv': vv, 'density': density}


def _estimate_dominant_freq(acc, dt):
    """对加速度时程做 FFT，返回幅值谱最大处频率（DC 置 0）。

    acc : 一维加速度数组（可含均值偏移）；dt：时间步长 (s)。
    返回主导频率 fc (Hz)；Ricker 子波即得其中心频率。
    """
    acc_centered = acc - np.mean(acc)  # 去均值，避免 DC 分量干扰
    n = len(acc_centered)
    spectrum = np.abs(np.fft.rfft(acc_centered))
    freqs = np.fft.rfftfreq(n, dt)
    spectrum[0] = 0.0  # 置零 DC 分量
    idx_max = np.argmax(spectrum)
    return float(freqs[idx_max])


def _damping_ratio_from_q(cs, is_bedrock, dcfg, layer_name=None):  # 由 Q 值换算阻尼比 ξ(支持逐层 ξ 覆盖)
    """根据剪切波速与是否基岩，计算品质因子 Q 与阻尼比 ξ=1/(2Q)。
    
    支持通过 dcfg['constant_xi'] 指定全场有限土层的统一恒定阻尼比。
    cs : 该层剪切波速 (m/s)；is_bedrock：是否基岩层；dcfg：阻尼配置（含 qs_factor/q_bedrock/constant_xi）。
    返回 (Q, xi)。
    """
    if is_bedrock:  # 基岩层
        Q = dcfg['q_bedrock']  # 基岩 Q≈999（近乎无衰减）
        xi = 1.0 / (2.0 * Q)
    else:  # 有限层
        xi_by = dcfg.get('xi_by_layer')  # EQL 注入的逐层 ξ(最优先)
        constant_xi = dcfg.get('constant_xi')
        if xi_by and layer_name in xi_by:
            xi = float(xi_by[layer_name])  # 该层应变相容阻尼比(EQL)
            Q = 1.0 / (2.0 * xi)
        elif constant_xi is not None:
            xi = float(constant_xi)  # 使用统一恒定阻尼比
            Q = 1.0 / (2.0 * xi)
        else:
            Q = dcfg['qs_factor'] * cs  # Qs = qs_factor*cs（论文 coarse-grain 法）
            xi = 1.0 / (2.0 * Q)
    return Q, xi


def _rayleigh_coeffs(xi, dcfg, fc, f_layer=None):  # 由阻尼比计算瑞利阻尼系数 α, β（支持逐层重锚定）
    """按指定方法将阻尼比 ξ 换算为 Abaqus 瑞利阻尼系数 (alpha, beta)。

    xi      : 阻尼比；dcfg：含 method/f1_factor/f2_factor/anchor/harmonics_cover；fc：输入波主频 (Hz)。
    f_layer : 该层自身一维共振基频 (Hz)，仅 anchor=='perband' 时使用；None=退化为输入主频锚定（如基岩）。
    method=='stiffness'：α=0、β=ξ/(π·fc)（fc 处 ξ 精确）；
    其余（rayleigh 双频拟合）按 anchor 选取拟合频带 [f1,f2]，两端 ξ 相等≈恒定 Q：
      'input'  ：f1=f1_factor·fc，            f2=f2_factor·fc；
      'dual'   ：f1=min(f1_factor·fc, f_site)，f2=f2_factor·fc（覆盖场地基频）；
      'perband'：f1=min(f1_factor·fc, f_layer)，f2=max(f2_factor·fc, harmonics_cover·f_layer)
                 —— 把该层自身共振及其前几阶谐波纳入拟合带，避免软薄层混响落在 β 主导的
                 高频段被过阻尼（默认）。f2 抬高后 fc 处略偏欠阻尼，对保留混响是有利且偏保守的。
    返回 (alpha, beta)。
    """
    if dcfg['method'] == 'stiffness':  # 仅刚度比例阻尼
        alpha = 0.0
        beta = xi / (math.pi * fc)  # β = ξ/(π·fc)，使 fc 处 ξ 精确
    else:  # 默认 rayleigh 双频拟合
        anchor = dcfg.get('anchor', 'input')
        f1 = dcfg['f1_factor'] * fc
        f2 = dcfg['f2_factor'] * fc
        if anchor == 'perband' and f_layer and f_layer > 0:  # 逐层按该层共振频带重锚定
            hc = float(dcfg.get('harmonics_cover', 3.0))
            f1 = min(f1, float(f_layer))  # 下限纳入该层共振基频
            f2 = max(f2, hc * float(f_layer))  # 上限纳入共振谐波，防混响被高频 β 过阻尼
        elif anchor == 'dual' and dcfg.get('f_site'):  # 双控锚定（场地基频+输入主频）
            f1 = min(f1, float(dcfg['f_site']))  # 下限取较小者，使拟合带覆盖场地基频
        w1 = 2.0 * math.pi * f1
        w2 = 2.0 * math.pi * f2
        alpha = 2.0 * xi * w1 * w2 / (w1 + w2)  # 两端 ξ 相等≈恒定 Q
        beta = 2.0 * xi / (w1 + w2)
    return alpha, beta


def _resolve_damping(dcfg, fc_est):  # 解析阻尼配置（补全 fc 字段）
    """拷贝阻尼配置，若 fc 为空则用自动估计值 fc_est 填充，返回解析后的配置 dict。

    供建材（_create_band_materials_sections）与元数据（_write_case_meta）共用同一份解析结果，
    避免 fc 口径漂移。显式给定的 fc 不被覆盖。
    """
    resolved = dict(dcfg)  # 浅拷贝，不就地修改原全局
    if resolved.get('fc') is None:  # 未显式指定主频
        resolved['fc'] = fc_est
    return resolved


def _site_fundamental_freq(site, geom):  # 估算上平台柱场地基频 f_site = 1/Ts
    """Ts = 4·Σ(d_i/Vs_i)（上平台柱各有限层垂直走时），无有限层返回 None。

    固定厚度层取其 thickness；最底覆盖层厚度 = H_upper − 基岩厚 − Σ固定厚度。
    供 damping_cfg['anchor']=='dual' 的瑞利双控拟合使用（f1 锚定 min(f1_factor·fc, f_site)），
    对应 research_plan.md 中"场地基频与输入卓越频率双控"的设计要点。
    """
    if not site.layers:  # 均质场地（无土层）
        return None
    travel = 0.0
    for L in site.layers:  # 自上而下遍历土层（厚度均显式给定）
        travel += float(L.thickness) / float(L.cs)
    if travel <= 0:
        return None
    return 1.0 / (4.0 * travel)  # 场地基频 f_site = 1/(4Σd/Vs)


def _band_resonance_freq(band):
    """返回材料带 band 的四分之一波长共振基频 (Hz)，供 perband 逐层重锚定使用。

    厚度 d 取标称上下界之差 (y1-y0)；d<=0 时返回 None（无有限厚度，如退化带/基岩半空间）。
    薄软层 d 小 → f 高，正是需要把瑞利拟合上限抬高、以免混响被高频过阻尼的层。
    与 _material_resonance_freq 数值同口径（厚度定义一致），保证建材/自由场/meta 三处 (α,β) 同源。
    """
    d = float(band['y1']) - float(band['y0'])  # 该带标称厚度（上平台口径）
    if d <= 0:
        return None
    return float(band['mat'].cs) / (4.0 * d)  # 四分之一波长共振基频


def _material_resonance_freq(mat, site, geom):  # 由材料层厚估算共振基频（meta 用，与分层带同口径）
    """返回土层 mat 的共振基频 cs/(4·d)；厚度均显式给定。

    仅对土层调用（基岩半空间无共振概念，应由调用方传 None）。厚度无效返回 None。
    """
    d = float(mat.thickness)  # 该层厚度（土层厚度均显式给定）
    if d <= 0:
        return None
    return float(mat.cs) / (4.0 * d)  # 四分之一波长共振基频


def _compute_interface_sv_coeff(alpha1, mat1, mat2):
    """计算 SV 波在两层界面的等效反射/透射系数（阻抗近似，忽略 SV<->P 转换）。

    alpha1 : 在 mat1 中的 SV 入射角（弧度）
    mat1   : 入射侧材料参数字典；mat2：透射侧材料参数字典
    返回 dict：Rss/Rsp/Tss/Tsp 与透射角 alpha2（Rsp=Tsp=0，阻抗近似的关键简化）
    """
    z1s = mat1['density'] * mat1['cs'] * max(1e-8, math.cos(alpha1))
    sin_a2 = mat2['cs'] * math.sin(alpha1) / mat1['cs']  # Snell 定律算透射角正弦
    alpha2 = _safe_arcsin(sin_a2)
    z2s = mat2['density'] * mat2['cs'] * max(1e-8, math.cos(alpha2))
    denom = z1s + z2s if abs(z1s + z2s) > 1e-12 else 1e-12  # 避免除零
    rss = (z2s - z1s) / denom
    tss = 2.0 * z2s / denom
    rsp = 0.0
    tsp = 0.0
    return {'Rss': rss, 'Rsp': rsp, 'Tss': tss, 'Tsp': tsp, 'alpha2': alpha2}


def _compute_free_surface_sv_coeff(alpha, cp, cs):
    """计算 SV 波在自由面的反射系数 A1（SV->SV）与转换系数 A2（SV->P）。"""
    beta_p = _safe_arcsin(cp * math.sin(alpha) / cs)  # 自由面 P 波转换角
    numerator_a1 = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) - cp ** 2 * math.cos(2 * alpha) ** 2
    denominator = cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta_p) + cp ** 2 * math.cos(2 * alpha) ** 2
    if abs(denominator) < 1e-12:
        denominator = 1e-12  # 避免除零
    a1 = numerator_a1 / denominator
    a2 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator
    return {'A1': a1, 'A2': a2, 'beta': beta_p}


def _compute_free_surface_p_coeff(beta, cp, cs):
    """计算 P 波在自由面的反射系数 B2（P->P）与转换系数 B1（P->SV）。"""
    alpha = _safe_arcsin(cs * math.sin(beta) / cp)
    numerator_b2 = cp ** 2 * math.cos(2 * alpha) ** 2 - cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)
    denominator = cp ** 2 * math.cos(2 * alpha) ** 2 + cs ** 2 * math.sin(2 * alpha) * math.sin(2 * beta)
    if abs(denominator) < 1e-12:
        denominator = 1e-12  # 避免除零
    b2 = numerator_b2 / denominator
    b1 = (2 * cp * cs * math.sin(2 * alpha) * math.cos(2 * alpha)) / denominator
    return {'B1': b1, 'B2': b2, 'alpha': alpha}


def _integrate_acc_to_velocity(acc, dt, time_arr):
    """加速度梯形积分为速度并做基线校正（去零偏 + 线性去趋势），抑制低频漂移。

    acc      : 加速度时程数组
    dt       : 时间步长 (s)
    time_arr : 与 acc 对应的时间轴数组
    返回 (vel, slope)：校正后的速度数组与被扣除的速度线性趋势斜率
    """
    acc = acc - np.mean(acc)  # 去均值，避免积分后速度线性漂移
    vel = np.zeros_like(acc)
    vel[1:] = np.cumsum((acc[:-1] + acc[1:]) / 2 * dt)  # 梯形积分
    trend = np.polyfit(time_arr, vel, 1)
    vel = vel - (trend[0] * time_arr + trend[1])  # 扣除线性趋势（位移=速度/(iω) 会放大低频误差）
    return vel, trend[0]


def _surface_y_at(x, H_upper, H_lower, left_flat, w_slope):
    """返回横坐标 x 处的地表 y 坐标（用于底边节点取其正上方柱子的覆盖层厚度）。

    几何：坡顶平台高 H_upper，坡脚平台高 H_lower，二者之间为线性坡面。
    x <= left_flat            : 坡顶平台，地表 = H_upper
    left_flat < x <= +w_slope : 坡面段，地表沿 x 从 H_upper 线性降到 H_lower
    x  > left_flat + w_slope  : 坡脚平台，地表 = H_lower
    """
    w = max(w_slope, 1e-9)  # 防止除零（平坦模型 w_slope 取极小值）
    if x <= left_flat:
        return H_upper
    if x <= left_flat + w:
        return H_upper - (x - left_flat) * (H_upper - H_lower) / w
    return H_lower


def _build_stratigraphy(site, geom, ymin=0.0, surface_geometry='horizontal'):
    """把场地分层展开为"从下到上"的标称材料带列表，供建模与自由场逐层取用。

    返回 list，每项 dict：{'name','mat'(Material),'y0','y1','fix',...}；
    y0/y1 为该带的【标称】下/上界 y（按上平台地表 H_upper 计），
    fix 标记该带边界如何随柱地表高度换算（见 _band_bounds_at）：
      'elevation' : 上下界为固定高程（horizontal 模式下的全部带）；
      'depth'     : 上下界为距局部地表的固定埋深 d0/d1（terrain 模式的土层带）；
      'fill'      : 下界固定高程 y0、上界=局部地表−dtop（terrain 模式的基岩带，顶面随地形）。
    顺序：基岩带在前（最底），向上依次为各土层（每层厚度均显式给定，剩余深度归基岩）。
    全基岩坡(site.layers 为空)时只返回基岩带（其上界取坡顶 H_upper）。
    """
    H_upper = geom.H_upper  # 坡顶地表高度（最顶土层标称上界基准）
    bt = geom.bedrock_thickness  # 基岩顶面高程（=坡顶地表 − Σ土层厚）
    layers_td = list(site.layers)  # 土层（从上到下）
    if not layers_td:
        return [{'name': site.bedrock.name, 'mat': site.bedrock, 'y0': ymin, 'y1': H_upper, 'fix': 'elevation'}]  # 全场均质基岩带
    terrain = (surface_geometry == 'terrain')  # 是否沿地形等厚铺设
    bands_td = []
    y_top = H_upper
    depth_top = 0.0
    for L in layers_td:  # 自上而下遍历土层（厚度均显式给定）
        band = {'name': L.name, 'mat': L, 'y0': y_top - L.thickness, 'y1': y_top}  # 标称上下界（上平台口径）
        if terrain:  # terrain 模式：该层按"距局部地表的埋深"定位
            band['fix'] = 'depth'
            band['d0'] = depth_top  # 该层顶埋深
            band['d1'] = depth_top + L.thickness
        else:  # horizontal 模式：固定高程水平带
            band['fix'] = 'elevation'
        bands_td.append(band)
        y_top -= L.thickness
        depth_top += L.thickness
    bedrock_band = {'name': site.bedrock.name, 'mat': site.bedrock, 'y0': ymin, 'y1': bt}  # 基岩带（标称上界=基岩顶面）
    if terrain:  # terrain 模式：基岩顶面随地形（=局部地表 − 土层总厚）
        bedrock_band['fix'] = 'fill'
        bedrock_band['dtop'] = depth_top  # 基岩顶面距局部地表的埋深（=Σ土层厚）
    else:  # horizontal 模式：基岩顶面为固定高程
        bedrock_band['fix'] = 'elevation'
    bands_bt = list(reversed(bands_td))  # 反转为"从下到上"
    return [bedrock_band] + bands_bt  # 基岩带在前 + 土层带（从下到上）


def _band_bounds_at(band, ys):
    """返回带 band 在"地表高程为 ys 的柱"内的 (y0, y1)。

    'elevation' 带原样返回标称值；'depth' 带返回 (ys−d1, ys−d0)；
    'fill' 带返回 (y0, ys−dtop)。horizontal 模式下全部带为 elevation。
    建模截面分配、边界弹簧选材、自由场柱构造统一经由本函数取界，保证四处口径一致。
    """
    fix = band.get('fix', 'elevation')  # 缺省按固定高程（兼容旧带结构）
    if fix == 'depth':  # 固定埋深带（terrain 模式表层）
        return ys - band['d1'], ys - band['d0']  # 上下界 = 局部地表减去底/顶埋深
    if fix == 'fill':  # 填充带（terrain 模式覆盖层）
        return band['y0'], ys - band['dtop']  # 下界固定高程、上界跟随地表
    return band['y0'], band['y1']


# ==========================================================
#  几何构造与命名
# ==========================================================


def _resolve_geometry_cfg(gcfg, logger=None):  # 无量纲几何设计 → 引擎绝对尺寸
    """把以坡高 hs 为基准的无量纲几何设计换算成引擎所需的绝对尺寸。

    参数
    ----
    gcfg : dict
        无量纲几何设计（6 个键全部必填，观测窗/净空/深度均为 hs 倍数）：
        slope_height 坡高 hs (m，唯一绝对尺度)；slope_angle 坡角(度)；
        crest_window/toe_window 坡顶/坡脚观测窗；side_clearance 侧向边界净空；
        base_depth 坡脚面以下模型深度（恒定，不随地层配置变）。
    logger : 日志器（None 则不打印换算结果）

    返回
    ----
    dict：引擎几何键 {total_L, left_flat, H_minus_h, i, H_lower}，
    其中 H_lower=base_depth·hs 为坡脚地表高程（模型底边 y=0）。

    换算公式：left_flat=(crest_window+side_clearance)·hs；w_slope=hs/tan(i)；
    right_flat=(toe_window+side_clearance)·hs；total_L=三段之和。
    几何只定外形，地层划分（各土层厚度、基岩顶面）全部由 material_cfg['layers'] 决定。
    """
    hs = float(gcfg['slope_height'])  # 坡高（唯一绝对尺度）
    i_deg = float(gcfg['slope_angle'])  # 坡角（度）
    crest_win = float(gcfg['crest_window'])  # 坡顶观测窗倍数
    toe_win = float(gcfg['toe_window'])  # 坡脚观测窗倍数
    clear = float(gcfg['side_clearance'])  # 侧向净空倍数
    base = float(gcfg['base_depth'])  # 坡脚面以下深度倍数
    if not (hs > 0.0):
        raise ValueError('slope_height(坡高)必须>0，当前: %r' % gcfg['slope_height'])  # 尺度非法
    if not (0.0 < i_deg < 90.0):
        raise ValueError('slope_angle 需在(0,90)度内，当前: %r' % gcfg['slope_angle'])  # 坡角非法
    if min(crest_win, toe_win, clear) < 0.0:
        raise ValueError('crest_window/toe_window/side_clearance 需>=0: %r/%r/%r'
                         % (crest_win, toe_win, clear))  # 倍数非法
    if base < 1.0:
        raise ValueError('base_depth 需>=1（保证底部至少留出基岩净空），当前: %r' % gcfg['base_depth'])  # 深度不足
    left_flat = (crest_win + clear) * hs  # 坡顶平台 = 观测窗 + 净空
    right_flat = (toe_win + clear) * hs  # 坡脚平台 = 观测窗 + 净空
    w_slope = hs / math.tan(math.radians(i_deg))  # 坡面水平长（仅换算 total_L 用，派生量仍由 make_geometry 统一算）
    total_L = left_flat + w_slope + right_flat  # 总长随 hs、坡角浮动
    resolved = {'H_minus_h': hs,  # 引擎键：斜坡高度差
                'i': i_deg,  # 引擎键：坡角
                'left_flat': left_flat,  # 引擎键：上平台长度
                'total_L': total_L,  # 引擎键：模型总长
                'H_lower': base * hs}  # 引擎键：坡脚地表高程（=坡脚面以下模型深度）
    if logger is not None:  # 打印换算结果，供日志核对
        log_step(logger, '几何换算(hs=%.1f m): left_flat=%.1f(%.2fh) + w_slope=%.1f(%.2fh) + right_flat=%.1f(%.2fh) = total_L=%.1f(%.2fh); 坡脚面以下深度=%.1f(%.2fh)',
                 hs, left_flat, left_flat / hs, w_slope, w_slope / hs,
                 right_flat, right_flat / hs, total_L, total_L / hs,
                 resolved['H_lower'], base)
    return resolved


def make_geometry(total_L, H_minus_h, i, left_flat, toe_surface_y, soil_thicknesses=None):
    """根据斜坡外形与土层厚度表计算全部派生量并打包为 Geometry。

    toe_surface_y    : 坡脚地表高程（=base_depth·hs，模型底边 y=0，所有工况一致）
    soil_thicknesses : 各土层厚度列表（从上到下，全部显式给定；空/None=全基岩坡）

    基岩顶面高程 bedrock_thickness = 坡顶地表 − Σ土层厚（土层扣完剩余全归基岩）。
    H/h/h_over_H 仅作派生记录量（对位论文口径）：H=坡顶下土层总厚，h=坡脚下土层厚，
    h 可为负（土层总厚<坡高 → 基岩在坡面出露），全基岩坡时 H=0、h_over_H=None。
    """
    soil = [float(t) for t in (soil_thicknesses or [])]  # 土层厚度表（从上到下）
    H_lower = float(toe_surface_y)  # 坡脚地表高程
    H_upper = H_lower + H_minus_h  # 坡顶地表高程
    H_flat = H_upper  # 平坦对照模型地表高程（与坡顶齐平）
    w_slope = H_minus_h / math.tan(math.radians(i))
    total_soil = sum(soil)  # 土层总厚
    bedrock_thickness = H_upper - total_soil  # 基岩顶面高程（=旧口径"基岩厚度"，底边 y=0）
    H = total_soil  # 派生记录：坡顶下土层总厚
    h = H_lower - bedrock_thickness  # 派生记录：坡脚下土层厚（可为负=基岩坡面出露）
    h_over_H = (h / H) if H > 0.0 else None  # 派生记录：深度比（全基岩坡无定义）
    layer_interfaces = []
    cum = 0.0
    for t in soil[:-1]:  # 自上而下遍历土层间界面（最底土层底界=基岩顶面，单列字段）
        cum += t
        layer_interfaces.append(H_upper - cum)  # 该层底界面 y = 坡顶 - 累计厚度
    layer_interfaces = sorted(layer_interfaces)
    return Geometry(total_L=total_L, i=i, left_flat=left_flat, H_minus_h=H_minus_h,
                    h_over_H=h_over_H, bedrock_thickness=bedrock_thickness,
                    H=H, h=h, H_upper=H_upper, H_lower=H_lower, H_flat=H_flat, w_slope=w_slope,
                    layer_interfaces=layer_interfaces)


def _build_model_name_from_record(acc_file, scene_tag):
    """按"记录名-场景名"规则生成模型名。"""
    record_name = os.path.splitext(os.path.basename(acc_file))[0]
    if not record_name:
        raise ValueError('无法从加速度文件生成记录名: %s' % acc_file)
    if scene_tag not in ('slope', 'flat'):
        raise ValueError('scene_tag 仅支持 slope 或 flat，当前为: %s' % scene_tag)
    return '{}-{}'.format(record_name, scene_tag)


# ==========================================================
#  输入读取
# ==========================================================


def find_acc_txt(logger=None, wave_files=None):
    """检索加速度时程文件，读取每个文件的分析步时长和增量步。

    wave_files : None=扫当前工作目录全部 .txt（旧行为）；
                 字符串或列表=使用指定文件（绝对路径或相对工况目录），
                 来源为 case_config.json 的 run_cfg['wave_files']，任一文件缺失即报错中止。
    返回 [(文件路径, 分析时长, 初始增量), ...]。
    """
    cwd = os.getcwd()
    if wave_files:  # 注入了指定波形文件 → 按给定路径读取，不扫目录
        if not isinstance(wave_files, (list, tuple)):  # 允许单条字符串写法
            wave_files = [wave_files]
        txt_files = []
        for p in wave_files:
            p = str(p)  # JSON 注入经 _ensure_str 已为 str，此处兜底
            full = p if os.path.isabs(p) else os.path.abspath(os.path.join(cwd, p))  # 相对路径按工况目录解析
            if not os.path.isfile(full):  # 指定文件必须存在，缺失立即中止（防止静默漏波）
                raise IOError('run_cfg[wave_files] 指定的地震波文件不存在: {}'.format(full))
            txt_files.append(full)
        if logger:
            log_step(logger, '使用注入的地震波文件(run_cfg.wave_files): 共 %d 条: %s',
                     len(txt_files), ', '.join(txt_files))
    else:  # 旧行为：扫工况目录下全部 .txt
        txt_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.txt')])
        if logger:
            log_step(logger, '开始检索加速度时程文件: 目录=%s, 命中 %d 个 .txt 文件', cwd, len(txt_files))
        if len(txt_files) == 0:
            raise IOError('当前目录 {} 下未找到任何 .txt 文件'.format(cwd))  # 抛出文件缺失异常

    result = []
    for f in txt_files:
        time_period = 2.0  # 设置默认分析时长
        initial_inc = 0.001  # 设置默认初始增量
        try:  # 尝试读取文件内容
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
                        log_step(logger, '%s 中 dt <= 0，将使用默认值', f)  # 输出默认值日志
            else:
                if logger:
                    log_step(logger, '%s 格式无效，将使用默认值', f)
        except Exception as e:  # 捕获读取异常
            if logger:
                log_step(logger, '读取加速度时程文件失败: %s，将使用默认值', str(e))
        result.append((f, time_period, initial_inc))

    if logger:
        log_step(logger, '加速度时程检索完成: 共 %d 条记录, 时长范围=%.2f~%.2fs, 增量范围=%.4f~%.4fs',
                 len(result),
                 min(r[1] for r in result), max(r[1] for r in result),  # 时长范围
                 min(r[2] for r in result), max(r[2] for r in result))  # 增量范围
    return result


# ============================================================
#  多层自由场计算核心（基于“射线反射与时延叠加”的任意多层土波动算法）
# ============================================================


def _delay_signal(u0, n_delay, dt):
    """将时程 u0(Nx2) 整体延迟 n_delay 个时间步，返回延长后的 (N+n_delay)x2 数组。"""
    N = u0.shape[0]  # 原始序列长度
    new_len = N + n_delay  # 延迟后总长度
    delayed = np.zeros((new_len, 2))
    delayed[:, 0] = np.arange(new_len) * dt
    delayed[n_delay:, 1] = u0[:, 1]
    return delayed


def _make_delay_cache(timeseries, dt):
    """返回一个按延迟步数缓存延迟信号的访问器，跨节点复用以减少重复构造。"""
    cache = {}
    def get_delayed(delay_t):
        n_delay = int(np.round(delay_t / dt))
        if n_delay not in cache:  # 缓存未命中
            cache[n_delay] = _delay_signal(timeseries, n_delay, dt)
        return cache[n_delay]
    return get_delayed


def _pad_to(arr, length, dt):
    """将 arr(Mx2) 末尾补零延长到 length 行，补零段补充时间轴。"""
    if arr.shape[0] < length:
        pad = np.zeros((length - arr.shape[0], 2))
        pad[:, 0] = np.arange(arr.shape[0], length) * dt  # 补齐时间轴
        arr = np.vstack([arr, pad])
    return arr


def _calc_node_delay(boundary, x0, y0, Ly, Lx,
                     alpha, beta_p, cs, cp, alpha2, beta2, cs2, cp2, ymax_col):
    """计算单个边界节点的三段到时 (tA, tB, tC)：入射 SV、反射 SV、反射/转换 P。

    boundary : 'l'/'r'/'b'；x0,y0：节点坐标；Ly：界面相对底边高度；Lx：模型横向跨度
    返回 (t1, t2, t3) 三段延迟时间（秒）。基岩段用 cs/cp/alpha/beta_p，
    覆盖层段用 cs2/cp2/alpha2/beta2（射线法口径）。
    """
    if boundary in ('l', 'r'):
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
        return t1, t2, t3
    elif boundary == 'b':
        t4 = x0 * np.sin(alpha) / cs  # 入射 SV 到时
        t5 = (2 * Ly + x0 * np.tan(alpha)) * np.cos(alpha) / cs  # 反射 SV 到时
        t6 = (Ly / (cp * np.cos(beta_p))  # 反射 P 到时（第一部分）
              + (Ly * np.cos(alpha) + x0 * np.sin(alpha)  # 第二部分
                 - Ly * np.tan(beta_p) * np.sin(alpha)) / cs)  # 第三部分
        return t4, t5, t6
    else:
        raise ValueError("boundary must be 'l', 'r', or 'b'")  # 抛出异常


def _column_seg(cs, vv, density, alpha_p, y0, y1, name):
    """根据材料与水平慢度构造柱内一层段（含派生波速、角度、垂直慢度因子、上下界）。"""
    params = _compute_material_params(cs, vv, density)
    alpha = _safe_arcsin(alpha_p * cs)  # 由 Snell 守恒求该层 SV 角
    beta = _safe_arcsin(alpha_p * params['cp'])  # 由 Snell 守恒求该层 P 角
    return {'name': name, 'mat': params, 'cs': cs, 'cp': params['cp'],
            'GG': params['GG'], 'lam': params['lam'], 'density': density,  # 剪切模量/拉梅常数/密度
            'alpha': alpha, 'beta': beta,  # SV/P 角
            'cos_alpha': math.cos(alpha), 'cos_beta': math.cos(beta),  # 垂直慢度用余弦
            'y0': y0, 'y1': y1}  # 该层段下界与上界 y


def _build_column(strat, ymax_col, alpha_p, ymin):
    """由场地分层带 strat 与该柱地表高度 ymax_col 构造"从下到上"的柱层段列表。

    各带上下界经 _band_bounds_at 按该柱地表高度换算（terrain 模式的埋深带随地表移动），
    再裁剪到 [ymin, ymax_col] 并用 y_floor 逐带钳位保证不重叠、不留缝。
    horizontal 模式（全部 elevation 带）结果与双层模型完全一致。
    单层场地（strat 仅一条基岩带）返回单层段柱（全 bedrock 至地表）。
    """
    tol = 1e-6
    column = []
    y_floor = ymin  # 当前柱内已占用高度的上沿（自底向上推进）
    for band in strat:
        y0, y1 = _band_bounds_at(band, ymax_col)
        y0 = max(y0, y_floor)  # 下界钳位到已占用上沿（防带间重叠）
        y1 = min(y1, ymax_col)  # 上界截断到地表
        if y1 <= y0 + tol:  # 换算/截断后无有效厚度
            continue
        mat = band['mat']
        column.append(_column_seg(mat.cs, mat.vv, mat.density, alpha_p, y0, y1, band['name']))
        y_floor = y1  # 推进已占用上沿
    return column


def _seg_at(column, y0):  # 定义在柱中按 y 坐标查找对应层段的辅助函数
    """返回柱 column 中包含 y0 的层段 dict（键同 _column_seg：alpha/beta/GG/cs/lam/cp 等）。

    column 从下到上排列（_build_column 产物）；y0 在某段 [seg['y0'], seg['y1']] 内则返回该段。
    找不到时退回最顶层段（兜底，不应发生）。
    用于项①自由场层内材料一致化：对有限层侧边节点，改用本层材料而非基岩标量。
    """
    for seg in reversed(column):
        if seg['y0'] - 1e-6 <= y0 <= seg['y1'] + 1e-6:  # 节点 y 落入该层段范围
            return seg
    return column[-1]  # 兜底返回最顶层段（正常不走此分支）


def _effective_refl_coeffs(column, oc):
    """自顶向下递归求基岩中上行 SV 的等效自由面反射 Rss_eff 与 SV->P 转换 Rsp_eff。

    沿用界面 SV 阻抗近似与自由面完整 SV 反射/转换；M=1 时严格退化为单腔几何级数。
    column：从下到上的柱层段（column[0]=基岩，column[-1]=最顶有限层或均质介质）。
    """
    topL = column[-1]
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
        if k == nseg - 1:
            Rbot_top_layer = Rbot  # 供 P 混响使用
        cyc = Rtop * Rbot  # 本层腔一次 SV 混响幅值因子
        sum_cyc = sum([cyc ** j for j in range(oc + 1)])  # 截断几何级数和（用列表避免通配 sum 拒收生成器）
        Rbottom = intf_lo['Rss'] + intf_lo['Tss'] * Rtop * intf_hi['Tss'] * sum_cyc  # 本层底界面等效反射
        T_up *= intf_lo['Tss']  # 累计上行透射
        T_down *= intf_hi['Tss']  # 累计下行透射
        Rtop = Rbottom  # 该等效反射成为下一层（更低层）看到的顶反射
    Rss_eff = Rtop
    cyc_p = B2_top * Rbot_top_layer  # 顶层腔一次 P 混响幅值因子
    sum_cyc_p = sum([cyc_p ** j for j in range(oc + 1)])  # P 混响截断几何级数和
    Rsp_eff = T_up * A2_top * T_down * sum_cyc_p  # 等效 SV->P 转换（上透→顶面转换→下透→顶腔混响）
    return Rss_eff, Rsp_eff


def _column_cavities(column, oc):  # 定义柱内各混响腔（用于时域延迟叠加）的计算函数
    """返回 (cavities_sv, cavities_p)：各有限层 SV 混响腔 (cycle, cdelay) 列表 + 顶层 P 混响腔。

    cycle = 该层腔顶反射×底反射（幅值）；cdelay = 该层往返垂直走时。M=1 时退化为单腔。
    """
    nseg = len(column)  # 柱层段总数
    cavities_sv = []
    cavities_p = []
    for k in range(1, nseg):
        layer = column[k]  # 当前有限层段
        lower = column[k - 1]  # 其下方层段
        thick = layer['y1'] - layer['y0']  # 该层厚度
        if thick <= 0:  # 厚度无效则跳过
            continue
        Rbot = _compute_interface_sv_coeff(layer['alpha'], layer['mat'], lower['mat'])['Rss']  # 底界面反射（下行反射回上行）
        if k == nseg - 1:  # 顶层：顶反射取自由面 SV 反射
            Rtop = _compute_free_surface_sv_coeff(layer['alpha'], layer['cp'], layer['cs'])['A1']  # 自由面 SV 反射
        else:  # 内层：顶反射取上界面反射
            upper = column[k + 1]  # 其上方层段
            Rtop = _compute_interface_sv_coeff(layer['alpha'], layer['mat'], upper['mat'])['Rss']  # 上界面反射
        cdelay_sv = 2.0 * thick * layer['cos_alpha'] / layer['cs']  # 该层 SV 往返垂直走时
        cavities_sv.append((Rtop * Rbot, cdelay_sv))
        if k == nseg - 1:  # 顶层腔额外贡献 P 混响（转换在顶面发生）
            B2 = _compute_free_surface_p_coeff(layer['beta'], layer['cp'], layer['cs'])['B2']  # 自由面 P 反射
            cdelay_p = 2.0 * thick * layer['cos_beta'] / layer['cp']  # 该层 P 往返垂直走时
            cavities_p.append((B2 * Rbot, cdelay_p))
    return cavities_sv, cavities_p


def _tt(column, y_lo, y_hi, wave):
    """逐层累加从 y_lo 到 y_hi（y_lo<y_hi）的垂直走时；wave='SV' 用 cos_alpha/cs，'P' 用 cos_beta/cp。"""
    t = 0.0
    for seg in column:
        lo = max(y_lo, seg['y0'])  # 本层段内的下限
        hi = min(y_hi, seg['y1'])  # 本层段内的上限
        if hi > lo:  # 区间有效时累加
            if wave == 'SV':  # SV 波
                t += (hi - lo) * seg['cos_alpha'] / seg['cs']
            else:  # P 波
                t += (hi - lo) * seg['cos_beta'] / seg['cp']
    return t


def _superpose_paths(get_delayed, tA, tB, tC, cavities_sv, cavities_p, order_count, dt):
    """对一个节点叠加主路径与各有限层混响，返回 (时间轴, A路径值, B路径累加, C路径累加)。

    A：主到时 tA 的延迟信号；B：反射 SV 路径在各腔往返组合下的混响累加；
    C：反射/转换 P 路径在顶层腔混响下的累加。各腔几何级数按 order_count 截断，腔间取乘积枚举。
    单腔（M=1）时严格退化为单腔 Σ_k cycle^k·delayed(t + k·cdelay) 形式。
    """
    def _combos(cavities):
        combo = [(1.0, 0.0)]  # 初始组合：无混响（幅值1、零延迟）
        for (cyc, cd) in cavities:  # 逐腔做几何级数与已有组合的乘积
            new = []  # 新组合容器
            for (amp, dl) in combo:
                for j in range(order_count + 1):  # 该腔的截断阶数
                    new.append((amp * (cyc ** j), dl + j * cd))  # 叠加该腔第 j 阶（幅值相乘、延迟相加）
            combo = new
        return combo
    combo_b = _combos(cavities_sv)  # B 路径各腔组合
    combo_c = _combos(cavities_p)  # C 路径各腔组合（仅顶层 P 腔）
    sig_b = [(amp, get_delayed(tB + dl)) for amp, dl in combo_b]  # B 路径各组合的延迟信号
    sig_c = [(amp, get_delayed(tC + dl)) for amp, dl in combo_c]  # C 路径各组合的延迟信号
    u0_tA = get_delayed(tA)  # 主到时延迟信号
    max_len = u0_tA.shape[0]
    for _amp, arr in sig_b + sig_c:
        max_len = max(max_len, arr.shape[0])
    u0_tA = _pad_to(u0_tA, max_len, dt)  # 补齐主路径
    sumB = np.zeros(max_len)  # B 路径累加器
    sumC = np.zeros(max_len)  # C 路径累加器
    for amp, arr in sig_b:  # 叠加 B 路径各组合
        sumB += amp * _pad_to(arr, max_len, dt)[:, 1]
    for amp, arr in sig_c:  # 叠加 C 路径各组合
        sumC += amp * _pad_to(arr, max_len, dt)[:, 1]
    return u0_tA[:, 0], u0_tA[:, 1], sumB, sumC


def _compute_freefield_at_node(boundary, x0, y0, ymax_col, ctx, get_vel, get_dis):
    """射线法计算单节点自由场时程，返回 dict：time/ux/uy/dotux/dotuy/sigmax/sigmay。

    boundary : 'l'/'r'/'b'；x0,y0：节点坐标；ymax_col：该柱地表高度（决定层组成、层厚与到时）；
    ctx      : FreeFieldCtx（含基岩角度、水平慢度、场地分层、基岩材料标量、VEL/DIS、dt 等）；
    get_vel/get_dis：速度/位移时程的延迟缓存访问器（跨节点复用）。
    多层推广：按该柱层栈求等效系数与各腔混响；投影/应力沿用基岩角度 + 基岩材料标量。
    """
    geom = ctx.geom
    Lx = geom.total_L  # 模型横向跨度（xmin=0）
    bt = geom.bedrock_thickness  # 基岩界面 y
    dt = ctx.dt  # 时间步长
    oc = max(0, int(ctx.max_reflect_order))  # 反射阶数上限
    p = ctx.p_horiz  # 水平慢度

    column = _build_column(ctx.strat, ymax_col, p, ctx.ymin)
    nseg = len(column)  # 柱层段数
    key = (round(ymax_col, 4), round(p, 12))  # 等效系数缓存键（同一柱地表高度+入射角复用）
    cached = _REFL_COEFF_CACHE.get(key)  # 查缓存
    if cached is None:
        cached = _effective_refl_coeffs(column, oc)
        _REFL_COEFF_CACHE[key] = cached  # 写入缓存
    Rss_eff, Rsp_eff = cached
    cavities_sv, cavities_p = _column_cavities(column, oc)

    if boundary == 'b' or y0 <= bt + 1e-6:  # 基岩节点或均质节点：沿用单层到时公式
        Ly = bt if nseg >= 2 else ymax_col  # 反射点：有基岩界面取界面，否则（均质）取自由面
        col0 = column[0]  # 最底层段（基岩或均质介质）
        tA, tB, tC = _calc_node_delay(boundary, x0, y0, Ly, Lx,  # 计算三段到时（单层公式）
                                      ctx.alpha, ctx.beta_p, ctx.cs, ctx.cp,  # 基岩角度/波速
                                      col0['alpha'], col0['beta'], col0['cs'], col0['cp'], ymax_col)  # 占位（基岩分支不用）
    else:  # 有限层节点：穿层走时累加（反射点为自由面）
        tA = _tt(column, ctx.ymin, y0, 'SV')  # 入射 SV：自底到节点
        tB = _tt(column, ctx.ymin, ymax_col, 'SV') + _tt(column, y0, ymax_col, 'SV')  # 反射 SV：自底到自由面 + 自由面回节点
        tC = _tt(column, ctx.ymin, bt, 'SV') + _tt(column, bt, y0, 'P')
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
    # 底边节点或基岩段侧边节点：沿用基岩标量（与单层模型完全等价，单层退化精确）
    # 有限层侧边节点（y0 > bedrock_interface）：改用本层 alpha/beta/GG/cs/lam/cp，
    #   使面力计算与已按本层取值的弹簧/阻尼系数口径一致（单层模型此处用基岩标量，存在矛盾）。
    # 注意：等效反射幅值 Rss_eff/Rsp_eff 仍用阻抗近似层栈递归（射线近似，不因本修改改变）。
    use_local = (boundary in ('l', 'r')) and (y0 > bt + 1e-6)  # 是否为有限层侧边节点
    if use_local:  # 有限层侧边节点：取本层 alpha/beta 与本层材料标量
        local_seg = _seg_at(column, y0)
        a = local_seg['alpha']   # 本层 SV 入射角（Snell 守恒折射角）
        bp = local_seg['beta']   # 本层 P 角（Snell 守恒折射 P 角）
        GG = local_seg['GG']     # 本层剪切模量
        cs = local_seg['cs']     # 本层剪切波速
        lam = local_seg['lam']   # 本层拉梅常数
        cp = local_seg['cp']     # 本层纵波波速
    else:  # 基岩节点或底边节点：沿用基岩标量（与单层模型等价）
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

    return {'time': td, 'ux': ux, 'uy': uy, 'dotux': dotux, 'dotuy': dotuy,
            'sigmax': sigmax, 'sigmay': sigmay}  # 应力分量


def _next_pow2(n):  # 求不小于 n 的最小 2 的幂
    """返回不小于 n 的最小 2 的整数次幂（FFT 长度用）。"""
    m = 1
    while m < n:  # 不足 n 时继续翻倍
        m *= 2  # 翻倍
    return m


def _band_damping_terms(strat, damping):  # 计算各材料带的瑞利阻尼系数表
    """返回 {带名: (alpha, beta)}；damping 未启用时全部为 (0,0)。

    与 _create_band_materials_sections 完全同口径（同一 Q→ξ→(α,β) 公式），
    保证 fd 自由场的衰减与 Abaqus 模型内介质一致。
    """
    terms = {}
    for idx, band in enumerate(strat):  # 从下到上遍历各材料带（idx==0 即基岩）
        if damping and damping.get('enable'):  # 启用材料阻尼时
            _Q, xi = _damping_ratio_from_q(band['mat'].cs, idx == 0, damping, band['name'])  # 该带品质因子与阻尼比
            f_layer = None if idx == 0 else _band_resonance_freq(band)  # 基岩无共振→None；有限层取该带共振基频
            a_ray, b_ray = _rayleigh_coeffs(xi, damping, damping['fc'], f_layer)  # 该带瑞利系数 (α,β)（perband 逐层重锚定）
        else:  # 未启用阻尼
            a_ray, b_ray = 0.0, 0.0  # 弹性（无衰减）
        terms[band['name']] = (a_ray, b_ray)
    return terms


def _fd_input_spectrum(ctx):
    """对输入加速度补零 FFT，返回输入谱缓存 dict（同一模型的所有节点共享）。

    内容：Nfft/Nout/nfreq/idx（被求解频点索引）/omega（对应圆频率）/U0（对应位移谱）。
    位移谱 U0 = −A(ω)/ω²（e^{iωt} 约定下加速度两次积分）；仅保留幅值谱 > tol·max 的频点。
    """
    cached = _FD_SOLVER_CACHE.get('_input')  # 查输入谱缓存
    if cached is not None:
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
    tail = float(ffcfg.get('tail_seconds', 0.0) or 0.0)  # 静默尾段时长（捕捉混响衰减，H(f) 提取用）
    Nout = min(Nfft, N + int(round(tail / dt))) if tail > 0 else N
    cached = {'Nfft': Nfft, 'Nout': Nout, 'dt': dt, 'nfreq': len(freqs),
              'idx': idx, 'omega': omega, 'U0': U0}  # 频点索引/圆频率/位移谱
    _FD_SOLVER_CACHE['_input'] = cached  # 写入缓存
    return cached


def _fd_layer_params(seg, omega, p, damp_terms, include_damping):
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
    return {'qs': qs, 'qp': qp, 'mu': muC, 'lam': lamC,
            'csC': np.sqrt(cs2), 'cpC': np.sqrt(cp2), 'p': p}  # 复波速与水平慢度


def _fd_wave_params(seg, la, kind):
    """kind ∈ {'Pu','Pd','Su','Sd'}；返回 {'dx','dy','ky','yref'}（dx/dy/ky 为逐频复数组）。

    极化约定与论文式(2)一致（半空间退化逐项还原射线法公式）：
      Pu=(cp·p, cp·qp)  Pd=(cp·p, −cp·qp)  Su=(cs·qs, −cs·p)  Sd=(−cs·qs, −cs·p)
    相位 e^{−iω·ky·(y−yref)}：上行波 ky=+q 参考层底 y0、下行波 ky=−q 参考层顶 y1。
    """
    qs, qp, csC, cpC, p = la['qs'], la['qp'], la['csC'], la['cpC'], la['p']
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
    return dx, dy, syy, sxy, sxx


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
            wm.append((col, _fd_wave_params(column[m], las[m], kind)))
            col += 1  # 列号递增
        waves.append(wm)
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
        amps = np.zeros((nb, nunk), dtype=complex)
        for k in range(nb):  # 逐频点求解
            amps[k] = np.linalg.lstsq(A[k], b[k], rcond=-1)[0]  # 最小二乘解（rcond=-1 兼容 Abaqus Py2.7 旧版 numpy）
    return {'amps': amps, 'las': las, 'waves': waves, 'inc': inc, 'column': column}


def _fd_eval_column(sol, omega, p, y):  # 在柱内高程 y 处评估单位入射的 7 个场量谱
    """返回 dict{'ux','uy','vx','vy','sxx','syy','sxy'}（逐频复数组，单位入射幅值）。"""
    column = sol['column']  # 柱层段
    seg_idx = 0  # 默认基岩段（兜底）
    for k in range(len(column) - 1, -1, -1):  # 自顶向下查找节点所在层段
        if column[k]['y0'] - 1e-6 <= y <= column[k]['y1'] + 1e-6:  # 落入该层段
            seg_idx = k
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
    return {'ux': ux, 'uy': uy, 'vx': vx, 'vy': vy, 'sxx': sxx, 'syy': syy, 'sxy': sxy}


def _fd_freefield_at_node(boundary, x0, y0, ymax_col, ctx):  # fd 引擎单节点自由场
    """fd 引擎：计算单个边界节点的自由场时程，返回 dict（接口与射线法引擎一致）。

    boundary : 'l'/'r'/'b'；x0,y0：节点坐标；ymax_col：该柱地表高度；ctx：FreeFieldCtx。
    步骤：①取输入谱缓存 ②取/解该柱频域解 ③节点谱 = 单位解 × U0(ω) × e^{−iωp·x0}
          ④逆 FFT 截断回原时长 ⑤按边界外法向嵌入应力符号。
    """
    inp = _fd_input_spectrum(ctx)  # 输入谱缓存（全模型共享）
    key = round(ymax_col, 4)  # 柱解缓存键（同地表高度柱复用）
    sol = _FD_SOLVER_CACHE.get(key)  # 查柱解缓存
    if sol is None:
        column = _build_column(ctx.strat, ymax_col, ctx.p_horiz, ctx.ymin)
        sol = _fd_solve_column(column, ctx.p_horiz, inp['omega'], ctx.damp_terms,  # 频域求解
                               bool((ctx.ffcfg or {}).get('include_damping', True)))  # 是否计入阻尼
        _FD_SOLVER_CACHE[key] = sol  # 写入缓存
    fields = _fd_eval_column(sol, inp['omega'], ctx.p_horiz, y0)  # 单位入射场量谱
    shift = np.exp(-1j * inp['omega'] * ctx.p_horiz * x0)  # 水平传播相位（左边界 x0=0 不移）
    scale = inp['U0'] * shift  # 输入位移谱 × 水平相位
    out = {}  # 时域结果容器
    for name in ('ux', 'uy', 'vx', 'vy', 'sxx', 'syy', 'sxy'):  # 逐场量逆变换
        spec = np.zeros(inp['nfreq'], dtype=complex)  # 全频带谱（未求解频点为零）
        spec[inp['idx']] = fields[name] * scale
        out[name] = np.fft.irfft(spec, n=inp['Nfft'])[:inp['Nout']]  # 逆 FFT 并截断回原时长
    if boundary == 'l':  # 左边界外法向 n=(−1,0)：面力 = −σxx, −σxy
        sigmax = -out['sxx']; sigmay = -out['sxy']  # 嵌入外法向符号
    elif boundary == 'r':  # 右边界外法向 n=(+1,0)：面力 = +σxx, +σxy
        sigmax = out['sxx']; sigmay = out['sxy']  # 嵌入外法向符号
    else:  # 底边界外法向 n=(0,−1)：面力 = −σxy, −σyy
        sigmax = -out['sxy']; sigmay = -out['syy']  # 嵌入外法向符号
    t_out = np.arange(inp['Nout']) * inp['dt']
    return {'time': t_out, 'ux': out['ux'], 'uy': out['uy'],
            'dotux': out['vx'], 'dotuy': out['vy'], 'sigmax': sigmax, 'sigmay': sigmay}  # 速度与应力


def _fd_engine_selfcheck(logger=None):  # fd 引擎建模前内置自检（验证协议自动化）
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
            _column_seg(800.0, 0.3, 2500.0, p0, 200.0, 400.0, 'cover')]
    om2 = np.array([2.0 * math.pi * 1.0])  # 单频 1Hz
    sol2 = _fd_solve_column(col2, p0, om2, {'bedrock': (0.0, 0.0), 'cover': (0.0, 0.0)}, True)  # 弹性求解
    f2 = _fd_eval_column(sol2, om2, p0, 400.0)  # 地表场量谱
    kh = 2.0 * math.pi * 1.0 * 200.0 / 800.0  # 层内相位角 ωh/Vs
    ana = 2.0 / abs(complex(math.cos(kh), (800.0 / 2000.0) * math.sin(kh)))  # SH 解析地表幅值（阻抗比=0.4）
    err2 = abs(abs(f2['ux'][0]) - ana) / ana  # 相对误差
    result = {'halfspace_err': err1, 'single_layer_err': err2}
    if logger:
        log_step(logger, 'fd 引擎自检: 半空间误差=%.2e, 单层解析误差=%.2e（阈值 1e-3）', err1, err2)
    if err1 > 1e-3 or err2 > 1e-3:  # 任一项超阈值
        raise RuntimeError('fd 引擎自检失败: halfspace_err=%.3e, single_layer_err=%.3e' % (err1, err2))  # 中止建模
    return result


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
    cs_candidates = [site.bedrock.cs]
    for L in site.layers:
        cs_candidates.append(L.cs)
    cs_min = min(cs_candidates)
    fmax = ff * fc  # 有效最高频率（Ricker 主频的 fmax_factor 倍）
    delta_l = cs_min / (epw * fmax)  # Kuhlemeyer-Lysmer 最大单元尺寸
    return max(delta_l, min_sz)  # 受下限约束后返回


def _interface_partitions(strat):
    """返回 (horiz_y, depth_d)：水平切分界面 y 列表 + 沿地形切分埋深 d 列表（均从下到上）。

    每条带（除最底基岩带）的【下界】即一条材料界面：
      elevation/fill 带的下界为固定高程水平线（y0）→ 归入 horiz_y；
      depth 带的下界为"地表整体下移 d1"的沿地形折线 → 归入 depth_d。
    horizontal 模式 depth_d 恒为空。
    单层（仅一条带）返回 ([], [])（无需切分）。
    """
    horiz, depth = [], []
    for band in strat[1:]:
        if band.get('fix', 'elevation') == 'depth':  # 埋深带：下界沿地形
            depth.append(band['d1'])
        else:  # 高程/填充带：下界为水平线
            horiz.append(band['y0'])
    return horiz, depth


def _create_band_materials_sections(model, strat, damping=None):
    """为分层带（从下到上）逐带创建材料与均质截面，返回 [(band, sec_name), ...]。

    damping: 解析后的阻尼配置 dict（含 enable/method/fc/qs_factor/q_bedrock 等）；
             None 或 enable=False 时退化为无阻尼行为。strat[0] 恒为基岩（_build_stratigraphy 保证）。
    """
    band_sections = []
    for idx, band in enumerate(strat):  # 从下到上遍历每条材料带（idx==0 即基岩）
        mat = band['mat']
        EE = _compute_elastic_modulus_from_wave_speed(mat.cs, mat.vv, mat.density)
        mat_name = _next_available_name('Material-%s' % band['name'], model.materials)
        m = model.Material(name=mat_name)
        m.Elastic(table=((EE, mat.vv),))
        m.Density(table=((mat.density,),))
        if damping and damping.get('enable'):  # 启用材料阻尼时按 Q 衰减施加瑞利阻尼
            is_bedrock = (idx == 0)  # 基岩带（最底带）→ 用 q_bedrock≈999
            Q, xi = _damping_ratio_from_q(mat.cs, is_bedrock, damping, band['name'])  # 由波速换算品质因子 Q 与阻尼比 ξ
            f_layer = None if is_bedrock else _band_resonance_freq(band)  # 基岩无共振→None；有限层取该带共振基频
            a_ray, b_ray = _rayleigh_coeffs(xi, damping, damping['fc'], f_layer)  # 由 ξ 换算瑞利系数 (α, β)（perband 逐层重锚定）
            m.Damping(alpha=a_ray, beta=b_ray)  # 施加 Abaqus 瑞利阻尼（α 质量比例 + β 刚度比例）
        sec_name = _next_available_name('Section-%s' % band['name'], model.sections)
        model.HomogeneousSolidSection(name=sec_name, material=mat_name, thickness=1.0)
        band_sections.append((band, sec_name))
    return band_sections


def _partition_horizontal(model, part, geom, y_list, name_prefix):
    """对 y_list 中每条水平界面逐条 PartitionFaceBySketch 切分（切线自动裁剪到实体内）。"""
    for idx, y in enumerate(y_list):
        part_faces = part.faces
        sk_name = '__%s_%d__' % (name_prefix, idx)
        sk = model.ConstrainedSketch(name=sk_name, sheetSize=max(geom.total_L, geom.H_upper) * 2)
        sk.Line(point1=(0.0, y), point2=(geom.total_L, y))  # 绘制该界面水平切线
        part.PartitionFaceBySketch(faces=part_faces, sketch=sk)
        del model.sketches[sk_name]


def _partition_terrain(model, part, geom, depth_list, name_prefix):
    """对 depth_list 中每个埋深 d，用"地表整体下移 d"的三段折线切分（上平台-坡面-下平台）。

    折线节点：(0, H_upper−d) → (left_flat, H_upper−d) → (left_flat+w_slope, H_lower−d)
              → (total_L, H_lower−d)，即表层底界面沿地形等深。
    """
    for idx, d in enumerate(depth_list):
        sk_name = '__%s_%d__' % (name_prefix, idx)
        sk = model.ConstrainedSketch(name=sk_name, sheetSize=max(geom.total_L, geom.H_upper) * 2)
        x1 = geom.left_flat  # 坡顶 x 坐标
        x2 = geom.left_flat + geom.w_slope  # 坡脚 x 坐标
        sk.Line(point1=(0.0, geom.H_upper - d), point2=(x1, geom.H_upper - d))  # 上平台段水平线（地表下移 d）
        sk.Line(point1=(x1, geom.H_upper - d), point2=(x2, geom.H_lower - d))  # 坡面段斜线（与坡面平行）
        sk.Line(point1=(x2, geom.H_lower - d), point2=(geom.total_L, geom.H_lower - d))  # 下平台段水平线
        part.PartitionFaceBySketch(faces=part.faces, sketch=sk)
        del model.sketches[sk_name]


def _assign_sections_by_band(part, band_sections, surface_y_fn=None):
    """按面质心落入哪条材料带分配对应截面，返回 [(层名, 面数), ...]。

    surface_y_fn: 函数 x → 该 x 处地表高程；terrain 模式据此把质心换算到局部柱内
    （经 _band_bounds_at 取带界）。None 或 horizontal 模式时等价于按标称 y 落带。
    """
    def _to_face_sequence(face_list):
        face_seq = part.faces[0:0]
        for face in face_list:
            face_seq = face_seq + part.faces[face.index:face.index + 1]  # 逐个拼接面对象
        return face_seq
    tol = 1e-6
    buckets = [[] for _ in band_sections]
    for face in part.faces:
        centroid = face.getCentroid()
        if len(centroid) >= 2 and not hasattr(centroid[0], '__len__'):  # 质心为 (x, y, ...) 平铺形式
            xc, yc = centroid[0], centroid[1]
        else:  # 质心为 ((x, y, z),) 嵌套形式
            xc, yc = centroid[0][0], centroid[0][1]
        ys = surface_y_fn(xc) if surface_y_fn else None  # 该质心所在柱的地表高程（terrain 落带用）
        placed = False  # 标记是否已归带
        for bi, (band, _sec) in enumerate(band_sections):
            y0, y1 = _band_bounds_at(band, ys) if ys is not None else (band['y0'], band['y1'])  # 带上下界
            if y0 - tol <= yc < y1 + tol:
                buckets[bi].append(face)
                placed = True  # 置归带标记
                break  # 跳出带循环
        if not placed:  # 处理未落入任何带的兜底情况
            buckets[-1].append(face)
    counts = []
    for (band, sec_name), face_list in zip(band_sections, buckets):
        if face_list:  # 该带存在面时分配截面
            part.SectionAssignment(region=Region(faces=_to_face_sequence(face_list)),
                                   sectionName=sec_name, offset=0.0, offsetType=MIDDLE_SURFACE,  # 指定截面参数
                                   offsetField='', thicknessAssignment=FROM_SECTION)
        counts.append((band['name'], len(face_list)))
    return counts


def _apply_damping_sponge(model, part, strat, damping, surf_fn, mesh_size, fc, logger, model_name):
    """边界内侧阻尼海绵层(opt-in)：对距 L/R/B 人工边界 sponge_width 内的单元，按到边界归一距离分级，
    在该带原材料上叠加渐增的刚度比例瑞利阻尼 β，吸收漏过 VAB 的残余外行波。

    默认关闭(boundary_cfg['sponge_enable']=False)→直接返回，不改变既有行为。分级 0(海绵内缘,无附加)
    → ngrade-1(贴边界, ξ 附加=sponge_xi_max)，渐变避免阻尼突变自身反射。每个(材料带,级)组合复制带材料
    并叠加 β，按单元集 SectionAssignment 覆盖原面截面(后赋材覆盖先赋材)。顶为自由面，不设海绵。
    注：与 eql_cfg['mode']='2d_element' 同时启用时，软层重叠单元以后运行者为准(EQL 在分析后外迭代赋材)。
    """
    if not boundary_cfg.get('sponge_enable', False):  # 未启用→零风险返回
        return
    if not fc or float(fc) <= 0.0:  # 海绵 β 需主频换算，缺 fc 则跳过并告警
        if logger:
            log_step(logger, '%s 阻尼海绵层：缺主频 fc，未生效(需输入波主频)', model_name)
        return
    ngrade = max(1, int(boundary_cfg.get('sponge_grades', 5)))  # 分级数
    xi_max = float(boundary_cfg.get('sponge_xi_max', 0.3))      # 贴边界处附加阻尼比(占主频)
    xs = [n.coordinates[0] for n in part.nodes]; ys_all = [n.coordinates[1] for n in part.nodes]  # 域包围盒
    xmin, xmax, ymin = min(xs), max(xs), min(ys_all)
    sw = float(boundary_cfg.get('sponge_width', 0.0))           # 海绵带宽 m
    if sw <= 0.0:  # 0=自动：max(10×基准网格, 8%域宽)——graded 网格远场单元粗，纯 10×网格常过小、捕不到单元
        sw = max(10.0 * float(mesh_size), 0.08 * (xmax - xmin))
    groups = {}  # (带idx, 级) -> [单元标签]
    for el in part.elements:
        node_idx = el.connectivity  # 单元节点内部索引元组
        coords = [part.nodes[i].coordinates for i in node_idx]  # 各节点坐标
        xc = sum([p[0] for p in coords]) / len(coords)  # 质心 x（用列表避免通配 sum 拒收生成器）
        yc = sum([p[1] for p in coords]) / len(coords)  # 质心 y
        d = min(xc - xmin, xmax - xc, yc - ymin)  # 到 L/R/B 最近距离(顶部自由面不计)
        if d >= sw:  # 海绵带外→保持原材料
            continue
        g = min(ngrade - 1, int((1.0 - d / sw) * ngrade))  # 归一深度 0(内缘)→1(边界) 映射到级
        ys = surf_fn(xc)  # 该柱地表高程(落带用)
        bi = 0  # 兜底归基岩
        for i, b in enumerate(strat):  # 质心落入哪条材料带
            y0, y1 = _band_bounds_at(b, ys)
            if y0 - 1e-4 <= yc < y1 + 1e-4:
                bi = i; break
        groups.setdefault((bi, g), []).append(el.label)
    if not groups:  # 海绵带内无单元(sw 过小或网格过粗)
        if logger:
            log_step(logger, '%s 阻尼海绵层：带宽 %.2fm 内无单元，未生效(增大 sponge_width)', model_name, sw)
        return
    n_assigned = 0
    for (bi, g), labs in groups.items():  # 逐(带,级)复制材料+叠加 β+按单元集赋材
        band = strat[bi]; mat = band['mat']
        EE = _compute_elastic_modulus_from_wave_speed(mat.cs, mat.vv, mat.density)  # 该带弹模
        if damping and damping.get('enable'):  # 复刻该带原瑞利阻尼(_create_band_materials_sections 同口径)
            is_bedrock = (bi == 0)
            _Q, xi0 = _damping_ratio_from_q(mat.cs, is_bedrock, damping, band['name'])
            f_layer = None if is_bedrock else _band_resonance_freq(band)
            a0, b0 = _rayleigh_coeffs(xi0, damping, damping['fc'], f_layer)
        else:  # 原本无阻尼→海绵附加为唯一阻尼
            a0, b0 = 0.0, 0.0
        b_extra = (xi_max * (g + 0.5) / ngrade) / (math.pi * float(fc))  # 刚度比例 β：ξ_add=βπf → β=ξ_add/(πf)
        mname = _next_available_name('Mat-Sponge_b%d_g%d' % (bi, g), model.materials)
        m = model.Material(name=mname); m.Elastic(table=((EE, mat.vv),)); m.Density(table=((mat.density,),))
        m.Damping(alpha=a0, beta=b0 + b_extra)  # 原阻尼 + 海绵附加 β
        sname = _next_available_name('Sec-Sponge_b%d_g%d' % (bi, g), model.sections)
        model.HomogeneousSolidSection(name=sname, material=mname, thickness=1.0)
        setname = _next_available_name('Sponge_b%d_g%d' % (bi, g), part.sets)
        part.SetFromElementLabels(name=setname, elementLabels=tuple(sorted(labs)))  # 该(带,级)单元集
        part.SectionAssignment(region=part.sets[setname], sectionName=sname,  # 覆盖原面截面
                               offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
        n_assigned += len(labs)
    if logger:
        log_step(logger, '%s 阻尼海绵层已施加：带宽=%.1fm, %d级, 贴边界附加ξ=%.0f%%, 覆盖 %d 单元/%d 组',
                 model_name, sw, ngrade, 100 * xi_max, n_assigned, len(groups))


def _band_graded_sizes(strat, mesh_used, mcfg, fc=None):  # 各带目标单元尺寸（波速比缩放 + 软层谐波/穿层加密）
    """返回与 strat 等长的各带目标尺寸列表（从下到上）。纯函数，便于单测。

    每带尺寸 = min(
        mesh_used × (cs/cs_min),                        # ① 波速比缩放：深部粗、最软层=mesh_used（基准）
        cs / (epw · f_resolve),                         # ② 共振谐波波长判据：仅薄软层会更细
        thickness / min_elems_through_thickness )       # ③ 穿层单元数判据：保证薄层足够层数
    其中 f_resolve = max(fmax_factor·fc, resolve_harmonics·f_layer)，f_layer = cs/(4·thickness)；
    使网格解析频带与 perband 阻尼保留频带(harmonics_cover·f_layer)对齐——薄软层混响高次谐波不再欠采样。
    上限受 max_band_ratio(最粗≤该倍×mesh_used)、max_size 约束；
    下限取 min(min_size, ③)，即【穿层判据优先于 min_size】，薄软层不会被 min_size 钳到层数不足。
    resolve_harmonics=0/None 时退化为纯波速比缩放版（仅①）。
    """
    cs_min = min([b['mat'].cs for b in strat])  # 最软层波速
    epw = float(mcfg.get('elems_per_wavelength', 10))  # 每波长单元数
    fmax_factor = float(mcfg.get('fmax_factor', 2.5))  # 输入频带 fmax 倍数
    max_size = mcfg.get('max_size')  # 绝对上限(m)，None=不限
    max_ratio = float(mcfg.get('max_band_ratio', 4.0))  # 过渡比上限
    min_size = float(mcfg.get('min_size', 0.5))  # 单元下限(m)
    rh = mcfg.get('resolve_harmonics', 3.0)  # 解析谐波次数
    rh = float(rh) if rh else 0.0
    n_thk = mcfg.get('min_elems_through_thickness', 6)  # 穿层最少单元数
    n_thk = float(n_thk) if n_thk else 0.0
    sizes = []  # 各带目标尺寸
    for b in strat:
        cs = float(b['mat'].cs)  # 该带波速
        thk = float(b['y1']) - float(b['y0'])  # 该带标称厚度
        s = mesh_used * (cs / float(cs_min))  # ① 波速比缩放（尺寸∝波速，最软层=mesh_used）
        dl_thick = (thk / n_thk) if (n_thk > 0 and thk > 0) else None  # ③ 穿层判据尺寸
        if rh > 0 and thk > 0:  # ② 共振谐波加密（仅薄软层会触发更细）
            f_layer = cs / (4.0 * thk)  # 该层一维共振基频
            f_resolve = rh * f_layer  # 至少解析到该层若干阶谐波
            if fc:  # 已知输入主频时并入输入频带上限
                f_resolve = max(f_resolve, fmax_factor * float(fc))
            dl_wave = cs / (epw * f_resolve)  # 波长判据尺寸
            s = min(s, dl_wave)
        if dl_thick:  # ③ 穿层判据
            s = min(s, dl_thick)
        s = min(s, mesh_used * max_ratio)  # 过渡比上限（限制最粗端）
        if max_size:  # 绝对上限
            s = min(s, float(max_size))  # 施加绝对上限
        floor = min(min_size, dl_thick) if dl_thick else min_size  # 下限：穿层判据优先于 min_size
        s = max(s, floor)  # 施加下限
        sizes.append(s)
    return sizes


def _elem_codes(elem_name):  # 由单元名映射 (主单元码, 三角过渡单元码, 是否二次单元)
    """支持 CPE4/CPE4R(线性) 与 CPE8/CPE8R(二次)；二次时三角过渡用 CPE6M(修正二次三角)。

    返回 (main_code, tri_code, is_quadratic)。is_quadratic=True 时边界节点含中节点，
    须改用二次单元边的一致权重(角:中=1/6:2/3)，否则等效力/黏弹性边界注入失真、远场不再=一维理论。
    """
    nm = str(elem_name).upper()  # 统一大写
    if nm == 'CPE8R':  # 二次减缩积分(低频散，本次验证用)
        return CPE8R, CPE6M, True
    if nm == 'CPE8':  # 二次全积分
        return CPE8, CPE6M, True
    if nm == 'CPE4R':  # 线性减缩积分(默认)
        return CPE4R, CPE3, False
    return CPE4, CPE3, False  # 兜底：线性全积分


def _seed_graded_mesh(part, strat, surface_y_fn, mesh_used, mcfg, elem_name, logger, model_name='Model-1', fc=None):  # 分层非均匀网格
    """对 part 施加分层非均匀网格：软层细、深部粗，自由四边形为主(QUAD_DOMINATED)平滑过渡。

    做法：先按最粗尺寸全局打底，再【由粗到细】逐带对其边按各带尺寸加密(constraint=FINER)，
    使共享界面边被更细的种子覆盖、过渡发生在较粗一侧。各带尺寸见 _band_graded_sizes。
    surface_y_fn: x→地表高程(用于按局部柱定位质心落带，与 _assign_sections_by_band 同口径)。
    fc: 输入波主频(Hz)，转发给 _band_graded_sizes 做软层谐波加密（None 时仅按谐波/穿层判据）。
    返回 [(层名, 尺寸, 面数), ...] 供日志记录。
    """
    sizes = _band_graded_sizes(strat, mesh_used, mcfg, fc)  # 各带目标尺寸（从下到上，含软层谐波加密）
    part.setMeshControls(regions=part.faces, elemShape=QUAD_DOMINATED, technique=FREE)  # 全自由四边形网格(容许过渡三角)
    part.seedPart(size=max(sizes), deviationFactor=0.1, minSizeFactor=0.1)  # 先用最粗尺寸全局打底
    tol = 1e-6  # 落带容差
    info = []
    for bi, b in enumerate(strat):
        face_list = []  # 该带的面
        for face in part.faces:
            c = face.getCentroid()  # 面质心
            if len(c) >= 2 and not hasattr(c[0], '__len__'):  # 平铺 (x,y,..)
                xc, yc = c[0], c[1]
            else:  # 嵌套 ((x,y,z),)
                xc, yc = c[0][0], c[0][1]
            ys = surface_y_fn(xc) if surface_y_fn else None  # 该柱地表高程
            y0, y1 = _band_bounds_at(b, ys) if ys is not None else (b['y0'], b['y1'])  # 该带上下界(同口径)
            if y0 - tol <= yc < y1 + tol:  # 质心落入该带
                face_list.append(face)
        if not face_list:  # 该带无面
            info.append((b['name'], sizes[bi], 0))
            continue
        edge_idx = set()  # 该带所有面的边索引(去重)
        for face in face_list:
            for ei in face.getEdges():  # 该面各边索引
                edge_idx.add(ei)
        edge_seq = part.edges[0:0]  # 空边序列
        for ei in sorted(edge_idx):
            edge_seq = edge_seq + part.edges[ei:ei + 1]  # 逐条拼接
        part.seedEdgeBySize(edges=edge_seq, size=sizes[bi], deviationFactor=0.1,
                            minSizeFactor=0.1, constraint=FINER)  # FINER：细种子覆盖共享边上的粗种子
        info.append((b['name'], sizes[bi], len(face_list)))
    elem_code, tri_code, _is_quad = _elem_codes(elem_name)  # 选择单元类型(CPE4/4R 线性 或 CPE8/8R 二次)
    et1 = mesh.ElemType(elemCode=elem_code, elemLibrary=STANDARD)  # 主单元(四/八节点平面应变)
    et2 = mesh.ElemType(elemCode=tri_code, elemLibrary=STANDARD)  # 过渡三角(线性CPE3/二次CPE6M)
    part.setElementType(regions=(part.faces,), elemTypes=(et1, et2))  # 分配单元类型
    part.generateMesh()
    return info


def create_model(site, geom, mesh_size, cae_name=None, logger=None, damping=None,
                 surface_geometry='horizontal', elem_name='CPE4', mesh_cfg=None, fc=None):
    """创建二维平面应变斜坡模型：几何、材料、截面、装配、网格（不含分析步）。

    site: Site 对象（基岩 + 有限层列表 + 基岩厚度，支持 1/2/3... 层）
    geom: Geometry 对象（斜坡几何，含派生量与固定层间界面）
    damping: 解析后的阻尼配置 dict（转发给材料创建，与平坦模型同参以保证 TAF 分母一致）
    surface_geometry: 表层几何模式 'horizontal' / 'terrain'(表层沿地形等厚铺设)
    elem_name: 单元类型 'CPE4'(默认) / 'CPE4R'
    mesh_cfg: 网格配置 dict；含 'graded' 时启用分层非均匀网格(软层细/深部粗)，缺省/None 为均匀网格
    fc: 输入波主频(Hz)，graded 网格下用于软层共振谐波加密（None 时仅按谐波/穿层判据）
    """
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    model_name = 'Model-1'

    total_L = geom.total_L
    H_lower = geom.H_lower
    H_upper = geom.H_upper
    H_minus_h = geom.H_minus_h  # 读取斜坡高度差（用于坡面顶点识别）
    w_slope = geom.w_slope
    left_flat = geom.left_flat
    bedrock_thickness = geom.bedrock_thickness

    right_flat = total_L - left_flat - w_slope
    if right_flat <= 0:
        raise ValueError('右平台长度<=0: total_L=%.3f, left_flat=%.3f, w_slope=%.3f' %
                         (total_L, left_flat, w_slope))  # 抛出几何错误

    if cae_name:
        mdb.saveAs(pathName=cae_name)  # 另存为新的工程文件
        log_step(logger, '工程文件保存为 %s', cae_name)
    model = mdb.Model(name=model_name)
    log_step(logger, '%s 基础模型开始创建', model_name)

    # 创建二维坡地 Part
    part_name = _next_available_name('Part', model.parts)
    s = model.ConstrainedSketch(name='__profile__', sheetSize=max(total_L, H_upper) * 2)
    s.Line(point1=(0.0, 0.0),                   point2=(total_L, 0.0))                 # 绘制底边
    s.Line(point1=(total_L, 0.0),               point2=(total_L, H_lower))             # 绘制右边界
    s.Line(point1=(total_L, H_lower),           point2=(left_flat + w_slope, H_lower)) # 绘制右平台地表
    s.Line(point1=(left_flat + w_slope, H_lower),  point2=(left_flat, H_upper))        # 绘制斜坡段
    s.Line(point1=(left_flat, H_upper),            point2=(0.0, H_upper))              # 绘制左平台地表
    s.Line(point1=(0.0, H_upper),               point2=(0.0, 0.0))                     # 绘制左边界
    part = model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseShell(sketch=s)
    del model.sketches['__profile__']
    log_step(logger, '%s 已创建零件并生成壳基体: %s', model_name, part_name)

    # ============ 逐层创建材料与截面（支持 1/2/3... 层）============
    strat = _build_stratigraphy(site, geom, surface_geometry=surface_geometry)
    band_sections = _create_band_materials_sections(model, strat, damping)  # 逐带创建材料与截面（含瑞利阻尼）
    log_step(logger, '%s 已创建 %d 个材料带的材料与截面 (表层几何=%s)', model_name, len(strat), surface_geometry)

    # 装配
    assembly = model.rootAssembly
    assembly.DatumCsysByDefault(CARTESIAN)  # 设置默认笛卡尔坐标系
    inst_name = _next_available_name(part_name, assembly.instances)
    assembly.Instance(name=inst_name, part=part, dependent=ON)
    log_step(logger, '%s 装配实例已创建: %s (dependent=ON)', model_name, inst_name)

    # ============ 切分面以划分网格与材料区域 ============
    # 1. 垂直切分（ crest & toe ）
    part_faces = part.faces
    partition_sketch = model.ConstrainedSketch(name='__vert_partition__', sheetSize=max(total_L, H_upper) * 2)
    partition_sketch.Line(point1=(left_flat, 0.0), point2=(left_flat, H_upper))  # 绘制左平台竖向切线
    partition_sketch.Line(point1=(left_flat + w_slope, 0.0), point2=(left_flat + w_slope, H_lower))  # 绘制坡脚竖向切线
    part.PartitionFaceBySketch(faces=part_faces, sketch=partition_sketch)
    del model.sketches['__vert_partition__']
    log_step(logger, '%s 几何垂直切分完成', model_name)

    # 2. 切分各材料界面（水平界面 + terrain 模式的沿地形折线界面）
    interfaces, depth_ifaces = _interface_partitions(strat)
    _partition_horizontal(model, part, geom, interfaces, 'hpartition')  # 逐条水平切分
    if depth_ifaces:  # terrain 模式：存在沿地形界面
        _partition_terrain(model, part, geom, depth_ifaces, 'tpartition')  # 逐条沿地形折线切分
    log_step(logger, '%s 材料界面切分完成: 水平=%d, 沿地形=%d', model_name, len(interfaces), len(depth_ifaces))

    mesh_cfg = mesh_cfg or {}
    graded = bool(mesh_cfg.get('graded', False)) and len(strat) > 1  # 启用分层非均匀且存在有限层（单层无需分层）
    pickedRegions = part.faces  # 选取全部面作为网格区域
    if graded:  # 分层非均匀网格——软层细、深部粗，自由四边形平滑过渡
        surf_fn_mesh = lambda xc: _surface_y_at(xc, H_upper, H_lower, left_flat, w_slope)  # 落带用地表高程函数
        ginfo = _seed_graded_mesh(part, strat, surf_fn_mesh, mesh_size, mesh_cfg, elem_name, logger, model_name, fc)  # 施加分层网格(含软层谐波加密)
        log_step(logger, '%s 分层非均匀网格(FREE QUAD_DOMINATED): %s', model_name,
                 ', '.join('%s=%.2fm(%d面)' % (n, s, c) for n, s, c in ginfo))
    else:  # 均匀网格
        # 设置网格控制：默认结构化四边形；若有界面切过坡面（楔形）或存在沿地形界面则退为自由四边形为主
        cuts_slope = bool(depth_ifaces) or any(H_lower + 1e-6 < y < H_upper - 1e-6 for y in interfaces)  # 是否存在界面切过坡面
        if cuts_slope:  # 坡面被切出表层楔形（无法结构化）
            part.setMeshControls(regions=pickedRegions, elemShape=QUAD_DOMINATED, technique=FREE)  # 自由四边形为主网格（容许少量三角）
            log_step(logger, '%s 检测到界面切过坡面（表层楔形），网格采用 FREE QUAD_DOMINATED', model_name)
        else:  # 无楔形（M<=1 或界面在坡脚以下）
            part.setMeshControls(regions=pickedRegions, elemShape=QUAD, technique=STRUCTURED)  # 结构化四边形
        part.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
        elem_code, tri_code, _is_quad = _elem_codes(elem_name)  # 主单元+三角过渡(支持二次)
        elemType1 = mesh.ElemType(elemCode=elem_code, elemLibrary=STANDARD)
        elemType2 = mesh.ElemType(elemCode=tri_code, elemLibrary=STANDARD)
        part.setElementType(regions=(pickedRegions,), elemTypes=(elemType1, elemType2))  # 分配单元类型（四/八 + 三角过渡）
        part.generateMesh()
        log_step(logger, '%s 已生成均匀网格: %s 单元，尺寸=%.2f', model_name, str(elem_name).upper(), mesh_size)

    # ============ 按质心落带分配截面（terrain 模式按局部埋深落带）============
    surf_fn = lambda xc: _surface_y_at(xc, H_upper, H_lower, left_flat, w_slope)  # 局部地表高度函数
    counts = _assign_sections_by_band(part, band_sections, surf_fn)  # 逐带按质心分配截面
    log_step(logger, '%s 截面属性分配完成: %s', model_name,
             ', '.join('%s=%d' % (n, c) for n, c in counts))

    # 边界内侧阻尼海绵层（opt-in，默认关闭→不改变既有行为；在带截面之后覆盖边界区单元截面）
    _apply_damping_sponge(model, part, strat, damping, surf_fn, mesh_size, fc, logger, model_name)

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

    n_elems = len(part.elements)
    n_nodes = len(part.nodes)
    log_step(logger, '%s 网格统计: 单元=%d, 节点=%d', model_name, n_elems, n_nodes)

    mdb.save()
    log_step(logger, '%s 基础模型创建完成并已保存 (part=%s, inst=%s)', model_name, part_name, inst_name)
    return model_name, part_name, inst_name


# ==========================================================
#  人工边界 VAB（弹簧-阻尼器 + 等效节点力）
# ==========================================================


def _make_boundary_nodes(nodes, sort_axis, ascending, pick_material, ymax, logger=None, model_name='Model-1', boundary_tag='?', quadratic=False):
    """对一条边界的实例节点排序、计算影响长度(权重)与弹簧/阻尼系数，返回 BoundaryNode 列表。

    nodes       : Abaqus 实例节点序列
    sort_axis   : 'x' 或 'y'，沿该轴排序并据相邻间距求影响长度
    ascending   : 是否升序（底边升序、侧边降序，沿用现有行为）
    pick_material: 函数 (x, y) -> 材料参数 dict，用于按节点所在层取系数
    ymax        : 弹簧刚度公式中的参考长度 R
    quadratic   : 是否二次单元(CPE8/8R)。True 时边界节点为 角-中-角 交替，
                  须用二次单元边一致权重(角:中=1/6:2/3)，否则黏弹性边界/等效力注入失真。

    权重口径：线性单元用相邻节点半距(tributary)；二次单元按每条单元边长 Le
      分配 角=Le/6、中=2Le/3、角=Le/6 累加(角节点由相邻两边各得 Le/6)。两者总权重均=边长之和。
    """
    arr = np.array([[node.label, node.coordinates[0], node.coordinates[1]] for node in nodes], dtype=float)
    axis = 1 if sort_axis == 'x' else 2  # 根据排序轴选择坐标列
    arr = arr[arr[:, axis].argsort()]
    if not ascending:
        arr = arr[::-1]  # 反转排序结果

    n = arr.shape[0]
    if n == 1:
        influence = np.array([0.0])  # 单节点影响长度设为零
    elif quadratic and n >= 3 and (n % 2 == 1):
        # 二次单元：排序后 角-中-角-中-...-角(节点数为奇)，逐单元边按 1/6:2/3:1/6 一致权重累加
        coord = arr[:, axis]
        influence = np.zeros(n)
        for k in range(0, n - 1, 2):  # 每条二次边：角 k、中 k+1、角 k+2
            Le = abs(coord[k + 2] - coord[k])  # 该单元边长度(两角节点间距)
            influence[k] += Le / 6.0           # 左角节点 +Le/6
            influence[k + 1] += 2.0 * Le / 3.0  # 中节点 +2Le/3
            influence[k + 2] += Le / 6.0       # 右角节点 +Le/6
    else:
        if quadratic and logger:  # 想用二次却不满足 角-中 交替(节点数偶/异常) → 退回 tributary 并告警
            log_step(logger, '%s 边界[%s]警告: 二次单元但节点数=%d 非奇数，权重退回线性 tributary(请检查网格)',
                     model_name, boundary_tag, n)
        coord = arr[:, axis]
        influence = np.empty(n)
        influence[0] = abs(coord[0] - coord[1]) / 2.0
        influence[-1] = abs(coord[-1] - coord[-2]) / 2.0
        if n > 2:
            influence[1:-1] = np.abs(coord[:-2] - coord[2:]) / 2.0

    dscale = float(boundary_cfg.get('dashpot_scale', 1.0))  # 边界阻尼器(吸收)缩放系数（1=全吸收/0=全反射）
    sscale = float(boundary_cfg.get('spring_scale', 1.0))   # 边界弹簧(恢复)缩放系数（1=现行/2=标准Liu/0=纯黏性）
    result = []
    for idx in range(n):
        x0 = arr[idx, 1]
        y0 = arr[idx, 2]
        inf = influence[idx]
        mat = pick_material(x0, y0)
        kn = mat['GG'] / 2.0 / ymax * inf * sscale  # 法向弹簧(恢复) × 缩放
        cn = mat['density'] * mat['cp'] * inf * dscale  # 法向阻尼器(吸收) × 缩放
        kt = mat['GG'] / 4.0 / ymax * inf * sscale  # 切向弹簧(恢复) × 缩放
        ct = mat['density'] * mat['cs'] * inf * dscale  # 切向阻尼器(吸收) × 缩放
        result.append(BoundaryNode(label=int(arr[idx, 0]), x=x0, y=y0, influence=inf,
                                   kn=kn, cn=cn, kt=kt, ct=ct))
    if logger and abs(dscale - 1.0) > 1e-9:  # 非标准吸收时显式告警（对照实验留痕）
        log_step(logger, '%s 边界[%s] 阻尼器吸收缩放 dashpot_scale=%.3f（1=全吸收/0=纯弹簧全反射）',
                 model_name, boundary_tag, dscale)
    if logger and abs(sscale - 1.0) > 1e-9:  # 非现行弹簧系数时显式告警（对照实验留痕）
        log_step(logger, '%s 边界[%s] 弹簧恢复缩放 spring_scale=%.3f（1=现行α_n0.5/α_t0.25, 2=标准Liu, 0=纯黏性）',
                 model_name, boundary_tag, sscale)
    if logger and n > 0:
        kns = [b.kn for b in result]; cns = [b.cn for b in result]
        log_step(logger, '%s 边界[%s]节点=%d, 影响长度=%.3f~%.3f, 法向刚度kn=%.3e~%.3e, 法向阻尼cn=%.3e~%.3e',
                 model_name, boundary_tag, n,  # 模型名/边界标签/节点数
                 float(influence.min()), float(influence.max()),  # 影响长度范围
                 min(kns), max(kns), min(cns), max(cns))  # 刚度/阻尼范围
    return result


def _add_spring_dashpots(assembly, instance, nodes_by_boundary, model_name, logger):
    """为三条边界的所有节点创建接地弹簧-阻尼器（法向 + 切向）。"""
    boundary_dof = {'l': (1, 2), 'r': (1, 2), 'b': (2, 1)}  # 各边界 (法向自由度, 切向自由度)
    total_created = 0
    for boundary in BOUNDARY_SEQUENCE:
        dof_n, dof_t = boundary_dof[boundary]
        n_b = len(nodes_by_boundary[boundary])  # 该边界节点数
        for bn in nodes_by_boundary[boundary]:
            node_array = instance.nodes.sequenceFromLabels([bn.label])  # 通过标签获取实例节点
            if len(node_array) == 0:
                logger.warning('创建弹簧-阻尼器时，实例中不存在节点 %d', bn.label)
                continue
            region = Region(nodes=node_array)
            # dashpot_scale=0(全反射)时 cn/ct=0；spring_scale=0(纯黏性)时 kn/kt=0。
            # Abaqus 要求 *Behavior=ON 的系数>0，故系数<=0 时改对应 Behavior=OFF，并传正占位系数(OFF 下被忽略)。
            dash_on_n = bn.cn > 0.0  # 法向是否启用阻尼器
            dash_on_t = bn.ct > 0.0  # 切向是否启用阻尼器
            spr_on_n = bn.kn > 0.0   # 法向是否启用弹簧(spring_scale=0 纯黏性时关闭)
            spr_on_t = bn.kt > 0.0   # 切向是否启用弹簧(spring_scale=0 纯黏性时关闭)
            assembly.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_normal'.format(boundary, bn.label),
                region=region, orientation=None, dof=dof_n,
                springBehavior=(ON if spr_on_n else OFF),
                springStiffness=(bn.kn if spr_on_n else 1.0),
                dashpotBehavior=(ON if dash_on_n else OFF),
                dashpotCoefficient=(bn.cn if dash_on_n else 1.0))
            assembly.engineeringFeatures.SpringDashpotToGround(
                name='SpringDashpot_{}_{}_tangent'.format(boundary, bn.label),
                region=region, orientation=None, dof=dof_t,
                springBehavior=(ON if spr_on_t else OFF),
                springStiffness=(bn.kt if spr_on_t else 1.0),
                dashpotBehavior=(ON if dash_on_t else OFF),
                dashpotCoefficient=(bn.ct if dash_on_t else 1.0))
            total_created += 2  # 每节点创建法向+切向两个元件
        log_step(logger, '%s 边界[%s]弹簧-阻尼器已创建: %d 节点 -> %d 元件', model_name, boundary, n_b, n_b * 2)
    log_step(logger, '%s 弹簧-阻尼器创建完成: 合计 %d 个元件', model_name, total_created)


def _build_equivalent_forces(nodes_by_boundary, ctx, logger=None, model_name='Model-1'):
    """逐边界逐节点用射线法计算自由场并组装等效节点力时程，返回 {'<label>-<边界>-fx/fy': Nx2 数组}。
    等效力 = K·u_ff + C·v̇_ff + A·σ_ff，其中应力 σ_ff 的各边界公式已内嵌外法向符号
      （见 _compute_freefield_at_node），故此处面力项统一取 +A·σ：
      侧边(l/r)：fx=kn·ux+cn·u̇x+A·σx, fy=kt·uy+ct·u̇y+A·σy；
      底边(b)  ：fx=kt·ux+ct·u̇x+A·σx, fy=kn·uy+cn·u̇y+A·σy。
    角点处理：左下/右下角点同属侧边与底边两个集合，会各算一次并叠加（VAB 角点标准处理，不折半）。
    时间轴：射线法按到时延迟会延长时程，故各节点力时程取其自身（延长后）时间轴，不截断到原长。
    """  # 说明函数用途与外法向/角点约定
    field_data = {}
    geom = ctx.geom
    engine = (ctx.ffcfg or {}).get('engine', 'ray')  # 自由场引擎选择（'fd' 或 'ray'）
    get_vel = None  # 射线法速度延迟缓存（fd 引擎不需要）
    get_dis = None  # 射线法位移延迟缓存（fd 引擎不需要）
    if engine != 'fd':  # 仅射线法路径需要延迟缓存
        get_vel = _make_delay_cache(ctx.VEL, ctx.dt)  # 速度时程延迟缓存（跨节点复用）
        get_dis = _make_delay_cache(ctx.DIS, ctx.dt)  # 位移时程延迟缓存（跨节点复用）
    if logger:
        _total_nodes = sum([len(v) for v in nodes_by_boundary.values()])  # 三边界节点总数
        log_step(logger, '%s 开始计算等效节点力: 引擎=%s, 边界节点合计=%d (左=%d/右=%d/底=%d)',
                 model_name, engine, _total_nodes,  # 引擎与总数
                 len(nodes_by_boundary['l']), len(nodes_by_boundary['r']), len(nodes_by_boundary['b']))  # 各边界节点数
    for boundary in BOUNDARY_SEQUENCE:
        _t_b = time.time()  # 该边界计算起始时间
        for bn in nodes_by_boundary[boundary]:
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

            t_arr = ff['time']  # 延长后的时间轴
            ux = ff['ux']; uy = ff['uy']  # 位移分量
            dotux = ff['dotux']; dotuy = ff['dotuy']  # 速度分量
            sigmax = ff['sigmax']; sigmay = ff['sigmay']  # 应力分量（已含外法向符号）

            if boundary in ('l', 'r'):  # 侧边界：x 为法向、y 为切向
                fx = bn.kn * ux + bn.cn * dotux + bn.influence * sigmax  # x 向等效力（法向弹簧+阻尼+面力）
                fy = bn.kt * uy + bn.ct * dotuy + bn.influence * sigmay  # y 向等效力（切向弹簧+阻尼+面力）
            else:  # 底边界：x 为切向、y 为法向
                fx = bn.kt * ux + bn.ct * dotux + bn.influence * sigmax  # x 向等效力（切向弹簧）
                fy = bn.kn * uy + bn.cn * dotuy + bn.influence * sigmay  # y 向等效力（法向弹簧）

            field_data['{}-{}-fx'.format(bn.label, boundary)] = np.column_stack((t_arr, fx))  # 缓存 x 向力时程
            field_data['{}-{}-fy'.format(bn.label, boundary)] = np.column_stack((t_arr, fy))  # 缓存 y 向力时程
        if logger:
            log_step(logger, '%s 边界[%s]等效力计算完成: %d 节点, 耗时=%.2fs',
                     model_name, boundary, len(nodes_by_boundary[boundary]), time.time() - _t_b)
    return field_data


def _apply_amplitudes_and_loads(model_name, inst_name, nodes_by_boundary, field_data, step_name, logger):
    """为每个边界节点创建幅值曲线（TabularAmplitude）并施加 x/y 向集中力。"""
    model = mdb.models[model_name]
    nodes = model.rootAssembly.instances[inst_name].nodes
    for boundary in BOUNDARY_SEQUENCE:
        for bn in nodes_by_boundary[boundary]:
            fx_arr = field_data['{}-{}-fx'.format(bn.label, boundary)]
            fy_arr = field_data['{}-{}-fy'.format(bn.label, boundary)]
            name_amp_fx = 'AMP-{}-{}-fx'.format(bn.label, boundary)
            name_amp_fy = 'AMP-{}-{}-fy'.format(bn.label, boundary)
            model.TabularAmplitude(data=tuple(tuple(row) for row in fx_arr),
                                   name=name_amp_fx, smooth=SOLVER_DEFAULT, timeSpan=STEP)
            model.TabularAmplitude(data=tuple(tuple(row) for row in fy_arr),
                                   name=name_amp_fy, smooth=SOLVER_DEFAULT, timeSpan=STEP)
            node_array = nodes.sequenceFromLabels([bn.label])
            if len(node_array) == 0:
                logger.warning('施加载荷时，实例中不存在节点 %d (实例: %s)', bn.label, inst_name)
                continue
            region = Region(nodes=node_array)
            model.ConcentratedForce(name='load-{}-{}-fx'.format(bn.label, boundary),
                                    createStepName=step_name, region=region, cf1=1.0, amplitude=name_amp_fx,
                                    distributionType=UNIFORM, field='', localCsys=None)
            model.ConcentratedForce(name='load-{}-{}-fy'.format(bn.label, boundary),
                                    createStepName=step_name, region=region, cf2=1.0, amplitude=name_amp_fy,
                                    distributionType=UNIFORM, field='', localCsys=None)
    log_step(logger, '%s 幅值曲线与集中力已创建', model_name)


def VAB_oblique(site, geom, angle,
                model_name='Model-1', part_name='Part-1', inst_name='Part-1-1',
                acc_file=None, step_name=None, logger=None, tcfg=None, fc_used=None,
                ffcfg=None, damping=None, surface_geometry='horizontal',
                critical_angle_check=True, elem_name='CPE4'):
    """为二维模型施加粘弹性人工边界（弹簧-阻尼器）与斜入射 SV 波等效节点力。

    site     : Site 对象（基岩 + 有限层列表 + 基岩厚度，支持 1/2/3... 层）
    geom     : Geometry 对象（几何，含 H_upper/H_lower/left_flat/w_slope/bedrock_thickness/layer_interfaces）
    angle    : SV 波入射角（度）
    tcfg     : time_cfg 配置 dict（项③时间步校验）；None=跳过检查
    fc_used  : 网格/阻尼已用主频(Hz)；提供时用于 fmax 校验（不提供则从记录自动估计）
    ffcfg    : freefield_cfg 配置 dict（v6：'fd'=频域精确分层自由场 / 'ray'=v5 射线法）
    damping  : 解析后的阻尼配置 dict（v6：fd 引擎据此使自由场衰减与 FE 介质一致）
    surface_geometry: v7 表层几何模式（与建模同口径，决定边界弹簧选材与自由场柱分层）
    critical_angle_check: v8 临界角校验开关（True=超临界拒绝建模 / False=仅告警不中断）
    elem_name: v3 单元类型（'CPE4/4R' 线性 / 'CPE8/8R' 二次）。二次时边界节点含中节点，
               自动改用二次单元边一致权重(角:中=1/6:2/3)，以保持远场=一维理论的验证。
    """
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    t0 = time.time()
    step_name = step_name or DEFAULT_STEP_NAME  # 使用默认分析步名称
    log_step(logger, '%s 模型开始创建人工边界', model_name)

    assembly = mdb.models[model_name].rootAssembly
    assembly.regenerate()

    model = mdb.models[model_name]
    if part_name not in model.parts:
        raise KeyError('%s 中不存在Part: %s' % (model_name, part_name))  # 抛出零件缺失异常
    part = model.parts[part_name]
    if inst_name not in assembly.instances:
        raise KeyError('%s 中不存在实例: %s' % (model_name, inst_name))  # 抛出实例缺失异常
    instance = assembly.instances[inst_name]

    missing_boundary_sets = [name for name in BOUNDARY_SET_NAMES if name not in part.sets]
    if missing_boundary_sets:
        raise KeyError('%s 缺少Part边界节点集: %s，请先在 create_model 中创建' %
                       (model_name, '/'.join(missing_boundary_sets)))  # 抛出节点集缺失异常
    log_step(logger, '%s 复用已有Part边界节点集: %s', model_name, '/'.join(BOUNDARY_SET_NAMES))

    def get_instance_nodes_from_part_set(set_name):
        labels = tuple(node.label for node in part.sets[set_name].nodes)
        if not labels:
            raise ValueError('%s Part节点集 %s 为空' % (model_name, set_name))  # 抛出空节点集异常
        return instance.nodes.sequenceFromLabels(labels)

    # 材料参数计算与场地分层
    mat_bedrock = _compute_material_params(site.bedrock.cs, site.bedrock.vv, site.bedrock.density)
    strat = _build_stratigraphy(site, geom, ymin=0.0, surface_geometry=surface_geometry)  # 构造场地分层带（v7：与建模同口径）
    _strat_params = [_compute_material_params(b['mat'].cs, b['mat'].vv, b['mat'].density) for b in strat]  # 各带材料派生参数（弹簧系数用）

    # 获取模型尺寸（左/右边界最高点与底边 y）
    l_nodes = get_instance_nodes_from_part_set('Left_boundary')
    ymax_l = max(l_nodes, key=lambda node: node.coordinates[1]).coordinates[1]

    b_nodes = get_instance_nodes_from_part_set('Bottom_boundary')
    ymin = b_nodes[0].coordinates[1]

    r_nodes = get_instance_nodes_from_part_set('Right_boundary')
    ymax_r = max(r_nodes, key=lambda node: node.coordinates[1]).coordinates[1]

    ymax = max(ymax_l, ymax_r)

    # 按节点所在材质层选择材料参数（v7：经 _band_bounds_at 按局部柱落带，支持任意层数与 terrain 模式）
    def pick_material(x_coord, y_coord):
        ys = _surface_y_at(x_coord, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope)  # 该节点所在柱的地表高程
        for band, params in zip(strat, _strat_params):
            y0, y1 = _band_bounds_at(band, ys)
            if y0 - 1e-4 <= y_coord < y1 + 1e-4:
                return params
        return _strat_params[-1]  # 兜底返回最顶带材料参数

    # 构建三条边界的节点列表（含影响长度/一致权重与弹簧-阻尼系数）
    _, _, _is_quad = _elem_codes(elem_name)  # 是否二次单元(决定边界节点权重口径)
    nodes_by_boundary = {  # 各边界 -> BoundaryNode 列表
        'l': _make_boundary_nodes(l_nodes, 'y', False, pick_material, ymax, logger, model_name, 'l', quadratic=_is_quad),  # 左边界（沿 y 降序）
        'r': _make_boundary_nodes(r_nodes, 'y', False, pick_material, ymax, logger, model_name, 'r', quadratic=_is_quad),  # 右边界（沿 y 降序）
        'b': _make_boundary_nodes(b_nodes, 'x', True, pick_material, ymax, logger, model_name, 'b', quadratic=_is_quad),  # 底边界（沿 x 升序）
    }
    _n_bn = sum([len(v) for v in nodes_by_boundary.values()])  # 三边界节点总数
    log_step(logger, '%s 边界节点影响长度与弹簧-阻尼系数已计算: 左=%d, 右=%d, 底=%d (合计=%d, 参考长度R=%.2f)',
             model_name, len(nodes_by_boundary['l']), len(nodes_by_boundary['r']),  # 左右底节点数
             len(nodes_by_boundary['b']), _n_bn, ymax)  # 合计数与参考长度

    _add_spring_dashpots(assembly, instance, nodes_by_boundary, model_name, logger)  # 施加接地弹簧-阻尼器

    # ============ 入射角处理与水平慢度 ============
    if angle == 0:
        angle = 1e-10
    else:
        angle = round(angle, 4)  # 保留四位小数
    alpha1 = math.radians(angle)

    cs1 = mat_bedrock['cs']  # 基岩剪切波速
    cp1 = mat_bedrock['cp']  # 基岩纵波波速
    # ── v8：SV 入射临界角校验（§3.1-A2 硬化，可由 run_cfg['critical_angle_check'] 关闭） ──
    # 基岩中 SV→P 临界角 = asin(cs/cp)（ν=0.3 时≈32.31°）；超临界后自由面反射 P 为非均匀波，
    # ray 引擎实角公式与 TAF 解析分母 factor_h 均失效，故默认达到临界角直接拒绝建模；
    # 关闭校验时仅输出警告、不中断，便于探索超临界工况。论文工况以 30° 为上限。
    crit_deg = math.degrees(math.asin(cs1 / cp1))  # 基岩 SV 临界角（度）
    if angle >= crit_deg - 1e-6:  # 入射角达到/超过临界角
        if critical_angle_check:  # 启用校验：硬性拦截
            raise ValueError('入射角 %.2f° >= 基岩临界角 %.2f°（超临界非均匀波不在本方法适用域内）' % (angle, crit_deg))  # 拒绝建模
        else:  # 关闭校验：仅告警不中断
            log_step(logger, '%s 警告: 入射角 %.2f° >= 基岩临界角 %.2f°（已关闭临界角校验，超临界结果可能不可靠）',
                     model_name, angle, crit_deg)
    if angle > 30.0:  # 超过论文采用的上限
        log_step(logger, '%s 警告: 入射角 %.2f° > 30°（论文上限），已接近临界角 %.2f°，结果须谨慎使用',
                 model_name, angle, crit_deg)
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
    log_step(logger, '%s 自由场引擎=%s(阻尼一致化=%s): 入射角=%.4f°, 水平慢度 p=%.6e, 层数(含基岩)=%d',
             model_name, ffcfg_used.get('engine'), ffcfg_used.get('include_damping'),  # 引擎与阻尼开关
             angle, p_horiz, len(strat))

    # ============ 读取加速度时程并积分（保留 v7 基线校正）============
    # [输入幅值约定（#5）] 加速度记录积分得到的速度被当作"基底入射上行 SV 波"幅值 E；
    #   自由岩面对应 2E（自由面效应），TAF=PGA_slope/PGA_flat 取比值时该归一化抵消。
    if not acc_file:
        raise ValueError('acc_file 不能为空')  # 抛出参数缺失异常
    ACC = np.loadtxt(acc_file)
    if ACC.ndim != 2 or ACC.shape[1] < 2 or ACC.shape[0] < 2:
        raise ValueError('加速度文件格式不满足 [time, acceleration]')  # 抛出格式异常
    time_arr = ACC[:, 0]
    acc = ACC[:, 1]
    dt = ACC[1, 0] - ACC[0, 0]
    if dt <= 0:
        raise ValueError('加速度 dt 必须 > 0')  # 抛出步长异常

    # ── 项③：时间步充分性【诊断】（v9：仅警告，不再重采样） ──────────────────────────
    # 模拟步长始终 = 输入地震动 txt 的 dt（与分析步 initialInc 同源），保证自由场与分析步 dt 严格一致。
    # 若输入 dt 偏粗（每 fmax 周期步数不足），仅输出警告，由用户自行决定是否更换更细 dt 的输入波。
    if tcfg and tcfg.get('check'):  # 启用时间步诊断（默认关闭；不论开关 dt 都不会被改变）
        fc_for_check = fc_used if fc_used else _estimate_dominant_freq(acc, dt)
        fmax_check = 2.5 * fc_for_check  # fmax 估计（≈Ricker 高频边界，与网格 K-L 判据同口径）
        min_steps = float(tcfg.get('min_steps_per_fmax_period', 20))  # 建议的最低步/周期
        steps_per_period = (1.0 / fmax_check) / dt if (fmax_check > 0 and dt > 0) else 9999.0  # 实际步/周期
        if steps_per_period < min_steps:  # 步数不足：仅警告，不改 dt
            log_step(logger, '%s 警告: 输入 dt=%.4fs 偏粗 (fmax=%.2fHz 仅 %.1f 步/周期 < 建议 %.0f)，'
                             '如需更高精度请改用更细 dt 的输入波（脚本不自动重采样）',
                     model_name, dt, fmax_check, steps_per_period, min_steps)
        else:  # 步数充足
            log_step(logger, '%s 时间步诊断: dt=%.4fs, fmax=%.2fHz, %.1f 步/周期(>=%.0f) 达标',
                     model_name, dt, fmax_check, steps_per_period, min_steps)
    # ── 时间步诊断结束 ───────────────────────────────────────────────────────────────

    # 积分得到速度时程（梯形积分 + 基线校正，抑制低频漂移），再积分得到位移时程
    vel, _vel_slope = _integrate_acc_to_velocity(acc, dt, time_arr)  # 加速度→速度（含基线校正）
    log_step(logger, '%s 速度基线校正完成: 去趋势斜率=%.3e', model_name, _vel_slope)
    dis = np.zeros_like(vel)
    dis[1:] = np.cumsum((vel[:-1] + vel[1:]) / 2 * dt)  # 速度梯形积分得到位移
    VEL = np.column_stack((time_arr, vel))  # 组合速度时程 [t, v]
    DIS = np.column_stack((time_arr, dis))  # 组合位移时程 [t, u]

    # ============ 逐节点用射线法计算自由场并组装等效力 ============
    ctx = FreeFieldCtx(
        site=site, geom=geom, strat=strat,  # 场地、几何、分层带
        ymax_l=ymax_l, ymax_r=ymax_r, ymin=ymin,  # 各边界高度信息
        alpha=alpha1, beta_p=beta1, p_horiz=p_horiz,  # 基岩入射角/P 反射角、水平慢度
        GG=mat_bedrock['GG'], lam=mat_bedrock['lam'], cs=cs1, cp=cp1,  # 基岩材料标量（投影/应力用）
        VEL=VEL, DIS=DIS, dt=dt, time_arr=time_arr, max_reflect_order=order_count,  # 时程、步长、阶数
        acc=acc, damp_terms=damp_terms, ffcfg=ffcfg_used)  # v6：原始加速度、各带瑞利系数、引擎配置
    field_data = _build_equivalent_forces(nodes_by_boundary, ctx, logger, model_name)
    log_step(logger, '%s 所有边界等效节点力计算完成: 共 %d 条力时程', model_name, len(field_data))

    # ============ 创建幅值曲线并施加集中力 ============
    _apply_amplitudes_and_loads(model_name, inst_name, nodes_by_boundary, field_data, step_name, logger)  # 施加幅值与载荷
    mdb.save()
    log_step(logger, '%s 粘弹性人工边界完成: 耗时=%.2fs', model_name, time.time() - t0)


# ==========================================================
#  批量建模与作业提交
# ==========================================================


def build_models(acc_info, base_model, part_name, inst_name,
                 site, geom, angle, job,
                 step_name=DEFAULT_STEP_NAME, model_scene='slope', logger=None,
                 tcfg=None, fc_used=None, ffcfg=None, damping=None, surface_geometry='horizontal',
                 surface_only=False, critical_angle_check=True, elem_name='CPE4'):
    """根据加速度时程信息批量复制模型、创建分析步、施加人工边界。

    site/geom : 场地材料与几何对象（直接转发给 VAB_oblique）
    angle     : SV 波入射角（度）
    job       : 作业配置 dict，读取 'variables'（场输出变量）与 'frequency'（输出频率）
    tcfg      : time_cfg 配置（项③）；None=跳过时间步校验
    fc_used   : 网格/阻尼已用主频(Hz)，转发给 VAB_oblique 用于 fmax 估计
    ffcfg     : 自由场引擎配置（v6），转发给 VAB_oblique
    damping   : 解析后阻尼配置（v6），转发给 VAB_oblique 用于 fd 自由场衰减一致化
    surface_geometry: v7 表层几何模式，转发给 VAB_oblique（须与 create_model 同口径）
    """
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器

    variables = _normalize_output_variables(job['variables'])
    frequency = job['frequency']

    model_names = []
    for acc_file, tp, inc in acc_info:
        new_model_name = _build_model_name_from_record(acc_file, model_scene)
        mdb.Model(name=new_model_name, objectToCopy=mdb.models[base_model])
        log_step(logger, '%s 模型已从 %s 复制', new_model_name, base_model)

        model = mdb.models[new_model_name]
        tail = float((tcfg or {}).get('tail_seconds', 0.0) or 0.0)  # v8：静默尾段时长（与 fd 自由场时窗一致）

        # 先建地震动力步（previous='Initial'）：默认 F-Output-1 锚在此步，U/V/A 场输出合法。
        # P0#1 重力静力步随后用 previous='Initial' 插到它【前面】（Abaqus 自动把地震步 previous 重排为重力步），
        # 从而静力步不含 V/A 场输出，避免"静力步非法变量"报错。
        # P1#10 几何非线性(P-Δ)：tssi_cfg['nlgeom'] 控制，默认 OFF=v1 基线；强震大位移时开 ON。
        nlgeom_flag = ON if tssi_cfg.get('nlgeom', False) else OFF
        # ── P1#9 CDP 收敛降级链（不收敛时按序尝试，前三档不改架构，均可由 case_config 注入实现；D1 转 Explicit 需人工决策不默认做）──
        #   ① 增大 CDP 粘性正则化：注入 frame_material_cfg['viscosity'] 5e-4 → 2e-3（助收敛，量级不显著改本构）；
        #   ② 减小最小增量：注入 tssi_cfg['cdp_min_inc_factor'] 1e-4 → 1e-5（放宽自动增量下限，允许更细回缩）；
        #   ③ 自动增量放松：已用 AUTOMATIC（遇不收敛自动回缩重试），无需额外动作；
        #   ④【决策点 D1，不默认】转 Explicit：架构级改动，斜入射等效节点力施加方式需重核。
        min_inc_factor = float(tssi_cfg.get('cdp_min_inc_factor', 1.0e-4))
        if tssi_cfg.get('enable') and tssi_cfg.get('nonlinear', True):  # step3：CDP材料非线性用自动增量(可回缩重试)
            model.ImplicitDynamicsStep(
                name=step_name, previous='Initial',
                timePeriod=tp + tail, timeIncrementationMethod=AUTOMATIC,
                initialInc=inc, minInc=inc * min_inc_factor, maxInc=inc, maxNumInc=1000000,
                nlgeom=nlgeom_flag, application=TRANSIENT_FIDELITY)  # 收敛降级见上；仍不收敛按 D1 转 Explicit
        else:  # 纯坡地/弹性框架(已验证行为)：固定增量不变
            model.ImplicitDynamicsStep(
                name=step_name, previous='Initial',
                timePeriod=tp + tail, timeIncrementationMethod=FIXED, initialInc=inc,
                maxNumInc=1000000,
                nlgeom=nlgeom_flag, application=TRANSIENT_FIDELITY)  # TRANSIENT_FIDELITY(α≈-0.05)降数值阻尼，让物理材料阻尼主导

        model.fieldOutputRequests['F-Output-1'].setValues(
            variables=variables, frequency=frequency)  # 指定输出变量和频率
        if surface_only:  # v8：仅地表节点集输出全时程（ODB 瘦身，服务频域框架"全时程提取"）
            surf_region = model.rootAssembly.allInstances[inst_name].sets['TOP_SURFACE']  # 地表节点集（实例级）
            model.FieldOutputRequest(name='F-Output-Surface', createStepName=step_name,  # 新建地表场输出请求
                                     variables=('A', 'U'), frequency=frequency, region=surf_region)  # 地表 A/U 每增量步
            model.fieldOutputRequests['F-Output-1'].setValues(  # 整体场输出降频（仅留抽检帧）
                variables=variables, frequency=10000000)  # 设为极大间隔（几乎只输出首末帧）
            log_step(logger, '%s 输出瘦身: TOP_SURFACE 全时程 A/U + 整体场输出降频', new_model_name)  # 记录瘦身日志

        # P0#1 重力两步法：在地震步【前】插 Static 通用步施加结构自重。previous='Initial' → Abaqus
        # 自动把该静力步插到 Initial 与地震步之间，地震步续接其静平衡状态。'off'(v1基线)则不插。
        grav_mode = str(tssi_cfg.get('gravity', 'off')) if tssi_cfg.get('enable') else 'off'
        if grav_mode != 'off':
            model.StaticStep(name=GRAVITY_STEP_NAME, previous='Initial', timePeriod=1.0,
                             initialInc=0.1, minInc=1.0e-6, maxInc=1.0, maxNumInc=100,
                             nlgeom=nlgeom_flag)  # 静力步，载荷 0→满 线性斜坡；nlgeom 与动力步一致(P-Δ 预载)
            _apply_frame_gravity(model, grav_mode, logger, new_model_name)  # 施加自重（框架GRAV + 楼层节点力）

        mdb.save()
        log_step(logger, '%s 分析步已创建, 时长=%.2f(含尾段 %.2f), 增量=%.3f',
                 new_model_name, tp + tail, tail, inc)

        VAB_oblique(site=site, geom=geom, angle=angle,  # 调用人工边界构建函数（传入场地与几何对象）
                    model_name=new_model_name, part_name=part_name, inst_name=inst_name,  # 传入模型/零件/实例名称
                    acc_file=acc_file, step_name=step_name, logger=logger,  # 传入加速度文件、分析步与日志器
                    tcfg=tcfg, fc_used=fc_used, ffcfg=ffcfg, damping=damping,  # 时间步校验 + v6 引擎/阻尼配置
                    surface_geometry=surface_geometry,  # v7：表层几何模式（与建模同口径）
                    critical_angle_check=critical_angle_check,  # v8：临界角校验开关透传
                    elem_name=elem_name)  # v3：单元类型(决定边界节点是否用二次一致权重)
        model_names.append(new_model_name)

    return model_names


def submit_job(num_cpus=7, memory_percent=90, model_name='Model-1', logger=None):
    """创建并提交Abaqus作业"""
    logger = logger or log_step()  # 在未传入日志器时使用默认日志器
    t0 = time.time()
    job_name = 'job-' + model_name
    if job_name in mdb.jobs:
        del mdb.jobs[job_name]
        log_step(logger, '检测到同名旧作业，已删除: %s', job_name)
    log_step(logger, '%s作业开始提交, CPU 数量=%d, 内存=%d%%',
             job_name, num_cpus, memory_percent)

    mdb.Job(name=job_name, model=model_name,  # 创建 Abaqus 作业
            description='VAB oblique SV-wave analysis (Multi-layered slope)',
            type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0,
            queue=None, memory=memory_percent, memoryUnits=PERCENTAGE,
            getMemoryFromAnalysis=True, explicitPrecision=SINGLE,
            nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,  # 关闭冗余输出
            contactPrint=OFF, historyPrint=OFF,  # 关闭接触与历史输出
            numCpus=num_cpus, numDomains=num_cpus,
            multiprocessingMode=DEFAULT, numGPUs=0)

    mdb.save()
    log_step(logger, '%s作业已提交，正在等待完成...', job_name)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    mdb.jobs[job_name].waitForCompletion()
    log_step(logger, '%s已完成: 耗时=%.2fs', job_name, time.time() - t0)


# ==========================================================
#  工况配置加载与元数据写出（主入口辅助函数）
# ==========================================================


def build_site(material_cfg, geometry_cfg):
    """由配置构建 Site 对象（基岩 + 从上到下的土层列表），并校验厚度与基岩净空。

    每个土层必须显式给定正厚度 thickness；剩余深度全部归基岩（半空间+底部净空）。
    layers 为空 → 全基岩坡。返回 (site, soil_thicknesses)，
    soil_thicknesses 为各土层厚度（从上到下，供 make_geometry 推界面用）。
    """
    cs_bedrock = float(material_cfg['bedrock']['vs'])  # 基岩剪切波速（直接给定）
    if cs_bedrock <= 0.0:
        raise ValueError('bedrock.vs(基岩剪切波速)必须>0，当前: %r' % material_cfg['bedrock']['vs'])  # 波速非法
    bedrock = Material(cs=cs_bedrock, vv=material_cfg['bedrock']['poisson_ratio'],  # 构建基岩材料（半空间）
                       density=material_cfg['bedrock']['density'], thickness=None, name='Bedrock')  # 基岩无固定厚度
    layers_cfg = material_cfg.get('layers', [])
    layers = []
    soil_thicknesses = []
    for lc in layers_cfg:  # 自上而下遍历各土层配置
        cs = float(lc['vs'])  # 该层剪切波速（直接给定）
        if cs <= 0.0:
            raise ValueError('层[%s].vs(剪切波速)必须>0，当前: %r' % (lc.get('name'), lc['vs']))  # 波速非法
        if lc.get('thickness') is None:
            raise ValueError('层[%s]必须显式给定 thickness（土层不再自动填充，剩余厚度归基岩）' % lc.get('name'))  # 厚度缺失
        t = float(lc['thickness'])  # 该层厚度
        if t <= 0.0:
            raise ValueError('层[%s].thickness 必须>0，当前: %r' % (lc.get('name'), lc['thickness']))  # 厚度非法
        soil_thicknesses.append(t)
        layers.append(Material(cs=cs, vv=lc['poisson_ratio'], density=lc['density'],
                               thickness=t, name=lc['name']))
    # 基岩净空校验：基岩顶面高程 = 坡顶地表 − Σ土层厚，须留 ≥2·坡高 的底部 VAB 净空
    hs = float(geometry_cfg['H_minus_h'])  # 坡高
    H_upper = float(geometry_cfg['H_lower']) + hs  # 坡顶地表高程
    bedrock_top = H_upper - sum(soil_thicknesses)  # 基岩顶面高程
    if bedrock_top < 2.0 * hs - 1e-6:
        raise ValueError('土层总厚(%.2f)过大: 基岩顶面高程 %.2f < 底部净空要求 2h=%.2f（需 Σt ≤ base_depth·h − h）' %
                         (sum(soil_thicknesses), bedrock_top, 2.0 * hs))  # 抛出净空错误
    site = Site(bedrock=bedrock, layers=layers, bedrock_thickness=bedrock_top)
    return site, soil_thicknesses


def _deep_merge(base, override):
    """dict 逐键递归合并；其余类型（含 list，如 layers）整体替换。返回合并后的新 dict。"""
    out = dict(base)  # 复制基底，避免就地修改
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):  # 双方均为 dict 才递归合并
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v  # 整体替换/新增（layers 列表整体替换即可改层数）
    return out


def _load_case_config(material_cfg, geometry_cfg, damping_cfg, logger,
                      mesh_cfg_in=None, time_cfg_in=None, max_reflect_order_in=None):  # 加载工况配置注入
    """加载并注入用户自定义工况配置（来自 case_config.json），覆盖默认配置。
    支持对材料、几何、阻尼、网格、时间步、自由场引擎、等效线性化(EQL)、人工边界等配置项进行部分或整体覆盖。
    返回覆盖后的完整配置元组：(material_cfg, geometry_cfg, damping_cfg, mesh_cfg, time_cfg, 
                          max_reflect_order, freefield_cfg, run_cfg)。
    """
    mesh_cfg_out = dict(mesh_cfg_in) if mesh_cfg_in else dict(mesh_cfg)  # 初始化网格自适应配置（用全局默认）
    time_cfg_out = dict(time_cfg_in) if time_cfg_in else dict(time_cfg)  # 初始化时间步校验配置（用全局默认）
    max_reflect_order_out = max_reflect_order_in if max_reflect_order_in is not None else MAX_REFLECT_ORDER  # 反射阶数
    ff_cfg_out = dict(freefield_cfg)  # 初始化自由场引擎配置（用全局默认）
    run_cfg_out = dict(run_cfg)  # 初始化运行控制配置（用全局默认）
    path = os.path.join(os.getcwd(), 'case_config.json')  # 约定的配置注入文件
    if not os.path.isfile(path):  # 无注入文件 → 用默认配置单独运行
        if logger:
            log_step(logger, '未发现 case_config.json，使用脚本内默认配置')  # 输出默认配置提示
        return (material_cfg, geometry_cfg, damping_cfg, mesh_cfg_out,  # 原样返回默认
                time_cfg_out, max_reflect_order_out, ff_cfg_out, run_cfg_out)  # 含 v6 新增两项
    try:  # 尝试读取并覆盖
        import json  # 导入 JSON 模块
        if logger:  # 记录发现注入文件
            log_step(logger, '发现 case_config.json，开始加载并覆盖默认配置: %s', path)
        with io.open(path, 'r', encoding='utf-8') as f:  # 打开配置文件（io.open：Py2 内置 open 不支持 encoding 关键字）
            cfg = _ensure_str(json.load(f))  # 解析并递归转换 unicode→str（Py2 下 Abaqus API 只接受 str）
        _merged_keys = []  # 记录已合并的配置段名（用于日志）
        if isinstance(cfg.get('material_cfg'), dict):  # 提供了材料覆盖
            material_cfg = _deep_merge(material_cfg, cfg['material_cfg'])  # 合并材料配置
            _merged_keys.append('material_cfg')
        if isinstance(cfg.get('geometry_cfg'), dict):  # 提供了几何覆盖
            geometry_cfg = _deep_merge(geometry_cfg, cfg['geometry_cfg'])  # 合并几何配置
            _merged_keys.append('geometry_cfg')
        if isinstance(cfg.get('damping_cfg'), dict):  # 提供了阻尼覆盖（批量调参/关阻尼）
            damping_cfg = _deep_merge(damping_cfg, cfg['damping_cfg'])  # 合并阻尼配置
            _merged_keys.append('damping_cfg')
        if isinstance(cfg.get('mesh_cfg'), dict):  # 项②：提供了网格配置覆盖（含 size）
            mesh_cfg_out = _deep_merge(mesh_cfg_out, cfg['mesh_cfg'])  # 合并网格配置
            _merged_keys.append('mesh_cfg')
        if cfg.get('mesh_size') is not None:  # v9.1：兼容旧写法——顶层 mesh_size 映射到 mesh_cfg['size']
            mesh_cfg_out['size'] = cfg['mesh_size']  # 覆盖基准网格尺寸（向后兼容）
            _merged_keys.append('mesh_size(旧写法)')  # 记录兼容映射
        if isinstance(cfg.get('time_cfg'), dict):  # 项③：提供了时间步校验覆盖
            time_cfg_out = _deep_merge(time_cfg_out, cfg['time_cfg'])  # 合并时间步校验配置
            _merged_keys.append('time_cfg')
        if cfg.get('max_reflect_order') is not None:  # 项④：提供了反射阶数覆盖
            max_reflect_order_out = int(cfg['max_reflect_order'])  # 覆盖反射截断阶数
            _merged_keys.append('max_reflect_order')
        if isinstance(cfg.get('freefield_cfg'), dict):  # v6：提供了自由场引擎覆盖
            ff_cfg_out = _deep_merge(ff_cfg_out, cfg['freefield_cfg'])  # 合并自由场引擎配置
            _merged_keys.append('freefield_cfg')
        if isinstance(cfg.get('run_cfg'), dict):  # v6：提供了运行控制覆盖
            run_cfg_out = _deep_merge(run_cfg_out, cfg['run_cfg'])  # 合并运行控制配置
            _merged_keys.append('run_cfg')
        if isinstance(cfg.get('eql_cfg'), dict):  # 土体非线性 EQL 覆盖
            eql_cfg.update(cfg['eql_cfg'])  # 就地更新全局 eql_cfg(扁平字典)
            _merged_keys.append('eql_cfg')
        if isinstance(cfg.get('boundary_cfg'), dict):  # 人工边界吸收缩放覆盖（边界吸收对照实验）
            boundary_cfg.update(cfg['boundary_cfg'])  # 就地更新全局 boundary_cfg(扁平字典，与 eql_cfg 同模式)
            _merged_keys.append('boundary_cfg')
        for _tssi_key, _tssi_dict in (('tssi_cfg', tssi_cfg), ('frame_cfg', frame_cfg),
                                      ('frame_material_cfg', frame_material_cfg), ('rebar_cfg', rebar_cfg),
                                      ('foundation_cfg', foundation_cfg)):  # P0#5+P1#7：TSSI各段含嵌套dict(column/beam)需深合并，就地更新(无需global)
            if isinstance(cfg.get(_tssi_key), dict):
                _merged = _deep_merge(_tssi_dict, cfg[_tssi_key])
                _tssi_dict.clear(); _tssi_dict.update(_merged)
                _merged_keys.append(_tssi_key)
        if logger:
            log_step(logger, '已合并配置段: %s', ', '.join(_merged_keys) if _merged_keys else '(无)')
            log_step(logger, '已加载 case_config.json 覆盖默认配置: 入射角=%s, 层数(有限层)=%d, 坡角=%s, 阻尼=%s/%s, mesh_auto=%s, order=%s, 引擎=%s',  # 输出关键覆盖项
                     material_cfg.get('angle'), len(material_cfg.get('layers', [])), geometry_cfg.get('slope_angle'),
                     damping_cfg.get('enable'), damping_cfg.get('method'),
                     mesh_cfg_out.get('auto'), max_reflect_order_out,
                     ff_cfg_out.get('engine'))
    except Exception as _e:  # 解析失败
        if logger:
            log_step(logger, '加载 case_config.json 失败(改用默认配置): %s', str(_e))
    return (material_cfg, geometry_cfg, damping_cfg, mesh_cfg_out,
            time_cfg_out, max_reflect_order_out, ff_cfg_out, run_cfg_out)  # 含 v6 新增两项


def _meta_f(value):  # 把数值安全转为内置 float（兼容 numpy 标量）
    """将 numpy/字符串等数值规范化为内置 float；None 原样返回，不可转换返回 None。"""
    if value is None:  # 空值
        return None  # 原样返回
    try:  # 尝试转换
        return float(value)
    except (TypeError, ValueError):  # 不可转换
        return None


def _meta_material(name, cs, vv, density, thickness=None):
    """返回 {name, cs, vv, density, thickness}；thickness=None 表示半空间或由几何决定。"""
    return {'name': str(name), 'cs': _meta_f(cs), 'vv': _meta_f(vv),  # 层名与波速、泊松比
            'density': _meta_f(density), 'thickness': _meta_f(thickness)}  # 密度与厚度


def _damping_meta(site, damping, geom=None):  # 把阻尼配置与逐层换算结果打包为元数据块（v9：含逐层共振频率）
    """返回阻尼元数据 dict（含逐层 cs/Q/xi/f_layer/alpha/beta），供 case_meta.json 记录与下游核对。

    site    : Site 对象（基岩 + 从上到下有限层）；damping：解析后的阻尼配置（含 fc）；
    geom    : Geometry 对象（v9：perband 模式据此推算各有限层共振基频 f_layer，None 则不记录 f_layer）。
    damping=None 或 enable=False 时返回 {'enable': False}。逐层顺序：基岩在前，再各有限层（从上到下）。
    """
    if not (damping and damping.get('enable')):  # 未启用阻尼
        return {'enable': False}  # 仅记录关闭状态
    fc = damping.get('fc')  # 解析后的主频
    per_layer = []
    mats = [(site.bedrock, True)] + [(L, False) for L in site.layers]  # 基岩在前 + 各有限层（从上到下）
    for mat, is_bedrock in mats:
        Q, xi = _damping_ratio_from_q(mat.cs, is_bedrock, damping, str(mat.name))  # 该层品质因子与阻尼比
        f_layer = None if (is_bedrock or geom is None) else _material_resonance_freq(mat, site, geom)  # v9：有限层共振基频（与建材同口径）
        a_ray, b_ray = _rayleigh_coeffs(xi, damping, fc, f_layer)  # 该层瑞利系数（perband 时随 f_layer 重锚定）
        per_layer.append({'name': str(mat.name), 'cs': _meta_f(mat.cs),  # 层名与波速
                          'Q': _meta_f(Q), 'xi': _meta_f(xi),  # 品质因子与阻尼比
                          'f_layer': _meta_f(f_layer),  # v9：该层一维共振基频（perband QA 锚点）
                          'alpha': _meta_f(a_ray), 'beta': _meta_f(b_ray)})  # 瑞利系数
    return {'enable': True, 'method': damping.get('method'), 'fc': _meta_f(fc),  # 总体配置
            'anchor': damping.get('anchor', 'input'), 'f_site': _meta_f(damping.get('f_site')),  # v8：锚定方式与场地基频
            'harmonics_cover': _meta_f(damping.get('harmonics_cover')),  # v9：perband 拟合上限覆盖的共振谐波次数
            'qs_factor': _meta_f(damping.get('qs_factor')), 'q_bedrock': _meta_f(damping.get('q_bedrock')),  # Q 换算因子
            'f1_factor': _meta_f(damping.get('f1_factor')), 'f2_factor': _meta_f(damping.get('f2_factor')),  # 双频拟合边界
            'layers': per_layer}  # 逐层阻尼明细（基岩在前）


def _write_case_meta(material_cfg, geom, site, mesh_size, script_name, logger, damping=None, ffcfg=None,
                     sgeom='horizontal', acc_path=None, selfcheck=None, eql_info=None):  # 写出统一工况元数据（v7 理论台阶 + v8 自检 + v2 EQL）
    """写出工况元数据 case_meta.json，固化当前建模与配置的全部参数。
    作为工况元数据的单一真相源，记录所有材料、几何、解析阻尼、自由场配置及一维理论台阶等参数，
    供下游分析与后处理脚本直接读取。失败仅告警，不影响建模主流程。
    返回：
        一维解析自由场理论台阶(ff_theory)字典（计算成功时），否则返回 None。
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
        vs_cover = layers[-1]['cs'] if layers else vs_surface  # 最底覆盖层 Vs2（a0 归一化 + Vr/Vs2 抗阻比，对齐论文 VR/Vs 口径）
        vs_min = min([L['cs'] for L in layers]) if layers else vs_bedrock  # 最软有限层波速（仅作诊断记录：含软表层时即 Vs1）
        vr_over_vs2 = (vs_bedrock / vs_cover) if (vs_bedrock and vs_cover) else None  # Vr/Vs2 抗阻比
        vs1_over_vs2 = (vs_surface / vs_cover) if (vs_surface and vs_cover) else None  # Vs1/Vs2 软硬比
        a0_base = (2.0 * slope_height / vs_cover) if (slope_height and vs_cover) else None  # a0 = fc(Hz) × a0_base，按上覆层 Vs2 归一化（与论文 a0=2fc(H−h)/Vs2 一致；上一版误改为 Vs1 已撤回）
        fc_meta = (damping or {}).get('fc')  # 解析后主频（阻尼关闭/未估计时可能为 None）
        a0_val = (fc_meta * a0_base) if (fc_meta and a0_base) else None  # v7：无量纲频率 a0 = fc × a0_base
        derived = {  # 派生量集中区（公式单一真相源）
            'n_finite_layers': n_finite,  # 有限层数（不含基岩）
            'n_layers_total': n_total,  # 总层数（含基岩）
            'vs_bedrock': _meta_f(vs_bedrock),  # 基岩 Vr
            'vs_surface': _meta_f(vs_surface),  # 表层 Vs1
            'vs_cover': _meta_f(vs_cover),
            'vs_min': _meta_f(vs_min),  # v9.1：最软有限层波速（a0 归一化用）
            'vr_over_vs2': _meta_f(vr_over_vs2),  # Vr/Vs2
            'vs1_over_vs2': _meta_f(vs1_over_vs2),  # Vs1/Vs2
            'slope_height': _meta_f(slope_height),  # a0 归一化用斜坡特征高度
            'a0_base': _meta_f(a0_base),  # a0 换算基数（v9.1：按最软层 vs_min）
            'a0': _meta_f(a0_val),  # v7：无量纲频率 a0（与论文工况对位用）
        }
        out_dir = os.path.abspath(os.getcwd())  # 当前工况文件夹（建模运行目录）
        meta = {
            'schema_version': 1,  # schema 版本号
            'model_type': model_type,  # 模型类型 single/double/multilayer
            'model_script': str(script_name),  # 建模脚本文件名
            'incident_angle': _meta_f(material_cfg['angle']),  # SV 入射角 θs（度）
            'surface_geometry': str(sgeom),  # v7：表层几何模式 horizontal/terrain
            'mesh_size': _meta_f(mesh_size),  # 网格尺寸（m）
            'geometry': geometry,  # 几何参数（含派生 H/h/w_slope）
            'bedrock': bedrock,  # 基岩材料字典
            'layers': layers,
            'derived': derived,  # 派生量
            'damping': _damping_meta(site, damping, geom),  # 材料阻尼块（逐层 Q/xi/f_layer/alpha/beta，可复现；v9 传 geom 算共振频率）
            'eql': (eql_info if eql_info else {'enable': False}),  # v2 土体非线性 EQL 结果(各非线性层 γ_eff/G_Gmax/Vs0→Vs/ξ)
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
            'note': 'TAF_h=PGA_h/(factor_h*PGA_in); TAF_v=PGA_v/(factor_h*PGA_in)',
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
                rec = np.loadtxt(acc_path)
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
                             'note': 'fd 引擎一维柱地表 PGA/(factor_h*PGA_in)；FE 远场台阶应与之一致(±5%)'}
                for tag, ys in (('left', geom.H_upper), ('right', geom.H_lower)):  # 左(上平台)/右(下平台)两柱
                    col_t = _build_column(strat_t, ys, p0, 0.0)
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
                if logger:
                    log_step(logger, 'ff_theory 计算失败(不影响建模): %s', str(_fe))
                ff_theory = None  # 置空
        meta['ff_theory'] = ff_theory  # v7：理论台阶块（可能为 None）
        meta['selfcheck'] = selfcheck  # v8：fd 引擎自检误差（halfspace_err/single_layer_err）
        text = json.dumps(meta, ensure_ascii=False, indent=2, default=_meta_f)  # 序列化为字符串（保留中文，default 兜底 numpy 标量）
        if isinstance(text, bytes):  # Py2 下 ensure_ascii=False 可能返回 bytes
            text = text.decode('utf-8')  # 解码为 unicode 以匹配 io.open 文本写入
        path = os.path.join(out_dir, 'case_meta.json')  # 目标路径
        with io.open(path, 'w', encoding='utf-8') as f:  # 以 UTF-8 文本模式打开（Py2 内置 open 不支持 encoding 关键字）
            f.write(text)  # 写出序列化文本
        if logger:
            log_step(logger, 'case_meta.json 已写出: %s', path)
        return ff_theory  # v7：返回理论台阶块供主流程日志打印
    except Exception as _e:  # 捕获任何写出异常
        if logger:
            log_step(logger, 'case_meta.json 写出失败(不影响建模): %s', str(_e))
        return None  # v7：失败时无理论台阶可返回


# ==========================================================
#  土体非线性：等效线性(EQL / SHAKE 式) —— v1：1D 应变相容 → 喂 2D
# ==========================================================

def _eql_mod_damp(gamma, curve, PI, sigma0_kpa):  # 可切换经验曲线：γ→(G/Gmax, ξ)
    """给定工程剪应变 gamma(小数)，返回 (G/Gmax, xi)。三种曲线可切换对比。"""
    g = max(abs(float(gamma)), 1e-9)
    if curve == 'seed_idriss_sand':  # 砂/砾(Seed-Idriss)
        gref, a, xi_min, xi_max = 6.0e-4, 0.92, 0.010, 0.28
    elif curve == 'vucetic_dobry':  # 黏土(随 PI)
        gref, a = 3.0e-4 * (1.0 + PI / 30.0), 0.85
        xi_min, xi_max = max(0.008, 0.025 - 0.0001 * PI), 0.25
    else:  # darendeli(通用,含围压/PI)
        gref = (0.0352 + 0.0010 * PI) * (sigma0_kpa / 100.0) ** 0.35 / 100.0
        a, xi_min, xi_max = 0.919, 0.008 + 0.00005 * PI, 0.24
    GG = 1.0 / (1.0 + (g / gref) ** a)  # 双曲骨架模量折减(Darendeli 口径)
    # 阻尼 = 最小阻尼 + Darendeli 修正 × Masing 滞回阻尼。
    # 替代旧线性近似 ξ=ξmin+k(1-GG)：后者在中低应变把阻尼算高约2倍(0.02%给10%、γr给16%)。
    # Masing 滞回阻尼(a=1 双曲闭式，分数)：γ→0 时自然趋于0，避免低应变虚高阻尼。
    DM = max(0.0, (4.0 * (g - gref * math.log((g + gref) / gref)) / (g * g / (g + gref)) - 2.0) / math.pi)
    b = 0.60  # Darendeli 修正系数(真实土阻尼/纯Masing≈0.6, N=1)；真实土比纯 Masing 阻尼低
    xi = min(max(xi_min + b * (GG ** 0.1) * DM, xi_min), xi_max)  # 修正后阻尼，封顶 xi_max
    return GG, xi


def _eql_layer_strain(freqs, Vs, rho, h, xi, Vb, rhob, xib, fcut):  # 竖直入射 SH：各层中点应变传递函数
    """传播矩阵法返回各有限层中点应变谱 γ_mid/outcrop位移。零频与 >fcut 跳过(防溢出)。"""
    nL = len(Vs)
    strain = np.zeros((nL, len(freqs)), dtype=complex)
    for fi, f in enumerate(freqs):
        w = 2.0 * math.pi * f
        if w == 0.0 or f > fcut:
            continue
        try:
            s0, s1 = 1.0 + 0j, 0.0 + 0j  # 地表 [u, τ]
            for j in range(nL):
                b = Vs[j] * np.sqrt(1.0 + 2j * xi[j]); mu = rho[j] * b * b; k = w / b
                c, sn = np.cos(k * h[j] / 2.0), np.sin(k * h[j] / 2.0)
                t_mid = mu * k * sn * s0 + c * s1
                strain[j, fi] = t_mid / (rho[j] * Vs[j] * Vs[j])  # γ=τ/G
                c, sn = np.cos(k * h[j]), np.sin(k * h[j])
                s0, s1 = c * s0 - sn / (mu * k) * s1, mu * k * sn * s0 + c * s1
            bb = Vb * np.sqrt(1.0 + 2j * xib); mub = rhob * bb * bb; kb = w / bb
            A = (s0 + 1j * s1 / (kb * mub)) / 2.0  # outcrop=2A
            if not np.isfinite(A) or abs(A) < 1e-30:
                strain[:, fi] = 0.0; continue
            strain[:, fi] /= (2.0 * A)
        except Exception:
            strain[:, fi] = 0.0
    return strain


def _run_freefield_eql(site, geom, eql_cfg, acc_in, dt, logger=None):  # 一维自由场 EQL(SHAKE)
    """对上平台自由场柱做应变相容迭代，返回 (更新后的 site, xi_by_layer, info)。

    仅 eql_cfg['nonlinear_layers'] 中的层非线性(降 Vs、增 ξ)；输入 acc_in 视为基岩入射 E，露头=2E。
    """
    curve = eql_cfg.get('curve', 'darendeli'); PI = float(eql_cfg.get('PI', 0.0))
    sigma0 = float(eql_cfg.get('sigma0_kpa', 100.0)); ratio = float(eql_cfg.get('strain_ratio', 0.65))
    tol = float(eql_cfg.get('tol', 0.02)); maxit = int(eql_cfg.get('max_iter', 15))
    nonlin_names = set(eql_cfg.get('nonlinear_layers', ['surface']))
    Vs0, rho, h, names, nonlin = [], [], [], [], []
    for L in site.layers:
        Vs0.append(float(L.cs)); rho.append(float(L.density)); h.append(float(L.thickness))  # 土层厚度均显式给定
        names.append(L.name); nonlin.append(L.name in nonlin_names)
    Vs0 = np.array(Vs0); rho = np.array(rho); h = np.array(h); nonlin = np.array(nonlin)
    Vb, rhob, xib = float(site.bedrock.cs), float(site.bedrock.density), 0.005
    acc = 2.0 * np.asarray(acc_in, dtype=float)  # 露头=2E
    n = len(acc); A = np.fft.rfft(acc); fr = np.fft.rfftfreq(n, dt); w = 2.0 * math.pi * fr
    U0 = np.zeros_like(A); U0[1:] = -A[1:] / w[1:] ** 2
    fc_hint = _estimate_dominant_freq(acc_in, dt) or 4.0
    fcut = min(0.45 / dt, max(12.0, 6.0 * fc_hint))
    Vs = Vs0.copy(); xi = np.where(nonlin, 0.02, 0.02); GG = np.ones(len(Vs)); geff = np.zeros(len(Vs))
    iters = 0
    if logger:  # 记录 EQL 迭代开始
        log_step(logger, 'EQL 迭代开始: 非线性层=%s, 曲线=%s, 截断频率fcut=%.2fHz, 最大迭代=%d, 收敛容差=%.3f',
                 [names[j] for j in range(len(Vs)) if nonlin[j]], curve, fcut, maxit, tol)
    for it in range(maxit):
        iters = it + 1
        st = _eql_layer_strain(fr, Vs, rho, h, xi, Vb, rhob, xib, fcut)
        Vs_new = Vs.copy(); xi_new = xi.copy()
        for j in range(len(Vs)):
            if not nonlin[j]:
                continue
            gt = np.fft.irfft(st[j] * U0, n=n)
            geff[j] = ratio * float(np.max(np.abs(gt)))
            gg, x = _eql_mod_damp(geff[j], curve, PI, sigma0)
            GG[j] = gg; Vs_new[j] = Vs0[j] * math.sqrt(gg); xi_new[j] = x
        rel = float(np.max(np.abs(Vs_new - Vs) / np.maximum(Vs, 1e-9)))
        Vs, xi = Vs_new, xi_new
        if logger:
            _nonlin_idx = [j for j in range(len(Vs)) if nonlin[j]]  # 非线性层索引
            _geff_str = ', '.join('%s=%.3f%%' % (names[j], geff[j] * 100) for j in _nonlin_idx)  # 各层有效应变
            log_step(logger, 'EQL 迭代%d/%d: 最大Vs相对变化=%.4f (容差%.3f), 有效应变 %s',
                     iters, maxit, rel, tol, _geff_str)
        if rel < tol:
            if logger:
                log_step(logger, 'EQL 收敛于第 %d 次迭代 (相对变化=%.4f < 容差%.3f)', iters, rel, tol)
            break
    new_layers = []; xi_by_layer = {}
    for j, L in enumerate(site.layers):
        if nonlin[j]:
            new_layers.append(L._replace(cs=float(Vs[j]))); xi_by_layer[L.name] = float(xi[j])
        else:
            new_layers.append(L)
    new_site = site._replace(layers=new_layers)
    info = {names[j]: {'Vs0': float(Vs0[j]), 'Vs': float(Vs[j]), 'GG': float(GG[j]),
                       'xi': float(xi[j]), 'geff': float(geff[j])} for j in range(len(Vs)) if nonlin[j]}
    if logger:
        for nm, d in info.items():
            log_step(logger, 'EQL[%s] 层%s: gamma_eff=%.4f%% G/Gmax=%.3f Vs %.0f->%.0f xi=%.1f%% (迭代%d次)',
                     curve, nm, d['geff'] * 100, d['GG'], d['Vs0'], d['Vs'], d['xi'] * 100, iters)
    return new_site, xi_by_layer, info


# ==========================================================
#  土体非线性 ② 逐单元 2D EQL
# ==========================================================

def _eql_bins_from_strains(geff_by_elem, n_bins):  # 纯函数：按有效剪应变给单元分箱(对数)
    """把 {单元号: γ_eff} 分成 n_bins 个对数箱。返回 (bin_of_elem{单元:箱号}, bin_geff[各箱代表应变])。"""
    if not geff_by_elem:
        return {}, []
    vals = np.array([max(float(v), 1e-9) for v in geff_by_elem.values()])
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo * (1.0 + 1e-6):  # 应变近似一致 → 单箱
        edges = np.array([lo * 0.999, hi * 1.001])
    else:
        edges = np.logspace(math.log10(lo), math.log10(hi), n_bins + 1)
    edges[-1] *= 1.0001  # 含右端
    nb = len(edges) - 1
    bin_geff = [math.sqrt(edges[i] * edges[i + 1]) for i in range(nb)]  # 各箱代表应变(几何中点)
    bin_of_elem = {}
    for e, g in geff_by_elem.items():
        idx = int(np.clip(np.searchsorted(edges, max(float(g), 1e-9), side='right') - 1, 0, nb - 1))
        bin_of_elem[e] = idx
    return bin_of_elem, bin_geff


def _eql_bin_props(bin_geff, Vs0_soil, eql_cfg):  # 纯函数：各应变箱的 (Vs, ξ)
    """按各箱代表应变查曲线，返回 [(Vs, xi), ...]。Vs=Vs0_soil·√(G/Gmax)。"""
    curve = eql_cfg.get('curve', 'darendeli'); PI = float(eql_cfg.get('PI', 0.0))
    s0 = float(eql_cfg.get('sigma0_kpa', 100.0))
    out = []
    for g in bin_geff:
        gg, xi = _eql_mod_damp(g, curve, PI, s0)
        out.append((Vs0_soil * math.sqrt(gg), xi, gg))
    return out


def _read_element_max_shear_strain(odb_path, inst_name, elem_labels=None):  # [需 Abaqus 实测] 读各单元最大剪应变
    """打开 ODB，对(软层)单元逐帧取最大工程剪应变 γ_xy=2·E12，返回 {单元号: γ_max}。
    要求建模时对软层单元请求了应变场输出 'E'。仅在 Abaqus 内可运行(odbAccess)。"""
    from odbAccess import openOdb  # Abaqus 专用
    odb = openOdb(odb_path, readOnly=True)
    try:
        step = odb.steps[list(odb.steps.keys())[-1]]  # 最后一个分析步
        want = set(elem_labels) if elem_labels is not None else None
        gmax = {}
        for frame in step.frames:  # 逐帧
            if 'E' not in frame.fieldOutputs:
                continue
            for v in frame.fieldOutputs['E'].values:  # 各积分点应变
                lab = v.elementLabel
                if want is not None and lab not in want:
                    continue
                comp = v.data  # 平面应变 [E11,E22,E33,E12]
                gxy = abs(2.0 * comp[3]) if len(comp) >= 4 else abs(2.0 * comp[-1])  # 工程剪应变
                if gxy > gmax.get(lab, 0.0):
                    gmax[lab] = gxy
        return gmax
    finally:
        odb.close()


def _soil_element_labels(part, strat, geom, nonlin_names, surface_geometry):  # 软层单元标签(按质心落带)
    """返回属于 nonlin_names 各带的单元标签集合(供逐单元 EQL 赋材/读应变)。"""
    surf_fn = (lambda xc: _surface_y_at(xc, geom.H_upper, geom.H_lower, geom.left_flat, geom.w_slope))
    name2band = {b['name']: b for b in strat}
    labels = set()
    for el in part.elements:
        node_idx = el.connectivity  # 单元节点内部索引元组
        coords = [part.nodes[i].coordinates for i in node_idx]  # 各节点坐标
        xc = sum([p[0] for p in coords]) / len(coords)  # 质心 x（用列表避免通配 sum 拒收生成器）
        yc = sum([p[1] for p in coords]) / len(coords)  # 质心 y
        ys = surf_fn(xc)
        for nm in nonlin_names:  # 落入哪条非线性带
            b = name2band.get(nm)
            if b is None:
                continue
            y0, y1 = _band_bounds_at(b, ys)
            if y0 - 1e-4 <= yc < y1 + 1e-4:
                labels.add(el.label); break
    return labels


def _apply_element_eql_materials(model_name, part_name, soil_elem_labels, geff_by_elem,
                                 Vs0_soil, vv_soil, rho_soil, eql_cfg, damping, fc, logger=None):
    """[需 Abaqus 实测] 按应变分箱，为软层单元逐箱建材并按单元集赋材(降Vs/增ξ)。返回 (bin_of_elem, bin_props)。"""
    n_bins = int(eql_cfg.get('n_strain_bins', 12))
    bin_of_elem, bin_geff = _eql_bins_from_strains(geff_by_elem, n_bins)
    props = _eql_bin_props(bin_geff, Vs0_soil, eql_cfg)  # [(Vs,xi,GG),...]
    model = mdb.models[model_name]; part = model.parts[part_name]
    for bi, (Vs_b, xi_b, gg_b) in enumerate(props):  # 逐箱建材+赋材
        labs = tuple(sorted([e for e, idx in bin_of_elem.items() if idx == bi]))
        if not labs:
            continue
        EE = _compute_elastic_modulus_from_wave_speed(Vs_b, vv_soil, rho_soil)
        mname = _next_available_name('Mat-EQLbin%d' % bi, model.materials)
        m = model.Material(name=mname); m.Elastic(table=((EE, vv_soil),)); m.Density(table=((rho_soil,),))
        a_ray, b_ray = _rayleigh_coeffs(xi_b, damping, fc)  # 该箱 ξ→瑞利系数
        m.Damping(alpha=a_ray, beta=b_ray)
        sname = _next_available_name('Sec-EQLbin%d' % bi, model.sections)
        model.HomogeneousSolidSection(name=sname, material=mname, thickness=1.0)
        setname = 'EQLbin%d' % bi
        part.SetFromElementLabels(name=setname, elementLabels=labs)  # 该箱单元集
        part.SectionAssignment(region=part.sets[setname], sectionName=sname,
                               offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='', thicknessAssignment=FROM_SECTION)
    if logger:
        log_step(logger, '逐单元EQL赋材: %d箱, Vs范围 %.0f~%.0f, ξ范围 %.1f~%.1f%%', len(props),
                 min(p[0] for p in props), max(p[0] for p in props),
                 100*min(p[1] for p in props), 100*max(p[1] for p in props))
    return bin_of_elem, props


def _add_soil_strain_output(model_name, part_name, inst_name, soil_labels, step_name, freq=1, logger=None):  # [需 Abaqus 实测] 软层单元应变场 E 输出
    """为软层单元创建单元集 EQL_SOIL 并请求应变场 E(逐单元 2D EQL 读取最大剪应变的前提)。
    仅对软层、逐增量输出，避免整模型 E 输出导致 ODB 过大。"""
    if not soil_labels:
        return
    model = mdb.models[model_name]; part = model.parts[part_name]
    part.SetFromElementLabels(name='EQL_SOIL', elementLabels=tuple(sorted(soil_labels)))  # 软层单元集
    region = model.rootAssembly.allInstances[inst_name].sets['EQL_SOIL']  # 实例级区域
    model.FieldOutputRequest(name='F-Output-E', createStepName=step_name, variables=('E',),  # 应变场 E
                             frequency=freq, region=region)  # 仅软层、逐增量
    if logger:
        log_step(logger, '已为软层 %d 个单元添加应变场 E 输出(逐单元 EQL 读取用)', len(soil_labels))


def _run_2d_element_eql(base_model, part_name, inst_name, site, geom, eql_cfg, damping, fc,
                        ffcfg, tcfg, sgeom, run_cfg, acc_rec, angle, job_cfg, logger, elem_name='CPE4'):
    """[需 Abaqus 实测] 逐单元 2D EQL 驱动：对代表性记录迭代 [建/赋材→提交→读ODB应变→更新] 至收敛。

    注意：本驱动需建模时对软层单元请求 'E' 应变场输出；mesh 跨外迭代保持不变(只改材料)。
    返回最终模型名。扫描多强度=按不同输入多跑(沿用 Autorun)。
    """
    ratio = float(eql_cfg.get('strain_ratio', 0.65)); n_out = int(eql_cfg.get('max_outer_iter', 4))
    conv = float(eql_cfg.get('converge_g', 0.05)); nonlin = set(eql_cfg.get('nonlinear_layers', ['surface']))
    strat = _build_stratigraphy(site, geom, surface_geometry=sgeom)
    soil = [b for b in strat if b['name'] in nonlin]
    Vs0_soil = float(soil[0]['mat']['cs']) if soil else float(site.layers[0].cs)
    vv_soil = float(site.layers[0].vv); rho_soil = float(site.layers[0].density)
    # 初始建模(1 条记录, 含边界+分析步)，复用 build_models
    names = build_models([acc_rec], base_model, part_name, inst_name, site, geom, angle, job_cfg,
                         model_scene='slope', logger=logger, tcfg=tcfg, fc_used=fc, ffcfg=ffcfg,
                         damping=damping, surface_geometry=sgeom,
                         surface_only=bool(run_cfg.get('surface_only', False)),
                         critical_angle_check=bool(run_cfg.get('critical_angle_check', True)),
                         elem_name=elem_name)  # v3：单元类型透传(二次单元边界一致权重)
    model_name = names[0]
    part = mdb.models[model_name].parts[part_name]
    soil_labels = _soil_element_labels(part, strat, geom, nonlin, sgeom)  # 软层单元
    _add_soil_strain_output(model_name, part_name, inst_name, soil_labels, DEFAULT_STEP_NAME, 1, logger)  # 软层应变场 E 输出(读应变前提)
    prev = None
    for outer in range(n_out):  # 外迭代
        submit_job(num_cpus=job_cfg['num_cpus'], memory_percent=job_cfg['memory_percent'],
                   model_name=model_name, logger=logger)
        odb_path = 'job-%s.odb' % model_name
        gmax = _read_element_max_shear_strain(odb_path, inst_name, soil_labels)  # 读应变
        geff = {e: ratio * g for e, g in gmax.items()}  # 有效应变
        bin_of_elem, props = _apply_element_eql_materials(model_name, part_name, soil_labels, geff,
                                                          Vs0_soil, vv_soil, rho_soil, eql_cfg, damping, fc, logger)
        Vs_now = [p[0] for p in props]
        if prev is not None and len(prev) == len(Vs_now):  # 收敛检查(各箱 Vs/G 相对变化)
            rel = max(abs(a - b) / max(b, 1e-9) for a, b in zip(Vs_now, prev))
            log_step(logger, '逐单元EQL 外迭代%d: 最大Vs相对变化=%.3f', outer + 1, rel)
            if rel < conv:
                break
        prev = Vs_now
    log_step(logger, '逐单元 2D EQL 完成(外迭代%d次): %s', outer + 1, model_name)
    return model_name


# ==========================================================
#  TSSI 坡顶建筑结构（tssi_cfg['enable']=True 时在 v3 坡地模型上追加框架 + Tie 耦合）
# ==========================================================


def rayleigh_coeffs(xi, f1, f2):
    w1 = 2.0 * math.pi * f1; w2 = 2.0 * math.pi * f2
    return 2.0 * xi * w1 * w2 / (w1 + w2), 2.0 * xi / (w1 + w2)


def _frame_T1_estimate():
    """框架固定基础基本周期 T1（s）：有注入实测值(tssi_cfg['T_fixed'])则用之，否则按 0.1N 经验估算。

    供 P0#4 瑞利阻尼 modal 锚定与 P0#6 tssi_meta 的 T_fixed 共用，保证两处口径一致。
    默认 5 层 → 0.1×5=0.5s（与 v1 硬编码 0.5 一致）。
    """
    T_inj = tssi_cfg.get('T_fixed')          # 注入的实测/指定周期
    if T_inj is not None:
        return float(T_inj)
    return 0.1 * int(frame_cfg['n_story'])   # 0.1N 经验估算


# ==========================================================
#  step3: 混凝土 CDP 单轴本构曲线（GB50010-2010 附录 C.2 经验公式）
# ==========================================================

def _monotone_inelastic_pairs(x_vals, stress_fn, damage_fn, eps_peak, E0):
    """把(x=ε/ε_peak)采样点转成 Abaqus 要求的单调递增(应力,非弹性应变)与(损伤因子,非弹性应变)表。

    x_vals按升序输入；返回 (hardening_pairs, damage_pairs)，均为 [(值, 非弹性应变Pa/无量纲), ...]，
    首行非弹性应变强制为 0（代表本构开始偏离线弹性的起点应力/损伤）。
    """
    hardening, damage = [], []
    zero_stress, zero_damage = None, 0.0
    for x in x_vals:
        eps_total = x * eps_peak
        sigma = stress_fn(x)
        d = min(max(damage_fn(x), 0.0), 0.999)
        eps_in = eps_total - sigma / E0
        if eps_in <= 0.0:
            zero_stress, zero_damage = sigma, d  # 仍处线性段，记录该段末尾值供首行使用
            continue
        if not hardening:  # 第一次进入非线性，写入首行(非弹性应变=0)
            hardening.append((zero_stress if zero_stress is not None else sigma, 0.0))
            damage.append((zero_damage, 0.0))
        if eps_in > hardening[-1][1] * (1.0 + 1e-9):  # 严格递增才追加(丢弃数值噪声导致的非递增点)
            hardening.append((sigma, eps_in))
            damage.append((d, eps_in))
    if not hardening:  # 全程未进入非线性(极端参数兜底)，退化为单点
        hardening.append((zero_stress if zero_stress is not None else stress_fn(x_vals[-1]), 0.0))
        damage.append((zero_damage, 0.0))
    return hardening, damage


def _gb50010_concrete_cdp_curves(fc_mpa, ft_mpa, Ec_pa, n_pts=40):
    """按 GB50010-2010 附录 C.2 单轴应力-应变曲线 + Sidoroff 能量等效法损伤因子生成 CDP 材料曲线(SI 单位)。

    fc_mpa/ft_mpa: 轴心抗压/抗拉强度代表值(MPa)；Ec_pa: 弹性模量(Pa)。
    返回 dict: comp_hardening/comp_damage/tension_stiffening/tension_damage，均为 Abaqus 表格式元组序列。
    注：损伤因子采用 Sidoroff 能量等效原理 d=1-sqrt(σ/(E₀·ε))，而非规范附录原始 dc/dt 公式，
    后者在上升段损伤增长过快会导致 Abaqus 换算塑性应变递减(FATAL ERROR)。能量等效法数学上
    保证 εpl 非负单调递增，且与 GB50010 应力-应变曲线完全兼容，为学界主流做法。
    """
    fc, ft, E0 = fc_mpa * 1.0e6, ft_mpa * 1.0e6, Ec_pa
    eps_c = (700.0 + 172.0 * math.sqrt(fc_mpa)) * 1.0e-6         # 峰值压应变
    alpha_a = 2.4 - 0.0125 * fc_mpa                              # 受压上升段参数
    alpha_d = 0.157 * fc_mpa ** 0.785 - 0.905                    # 受压下降段参数
    rho_c = fc / (E0 * eps_c)                                    # 应力比(受压)
    eps_t = 65.0e-6 * ft_mpa ** 0.54                             # 峰值拉应变
    alpha_t = 0.312 * ft_mpa ** 2                                # 受拉下降段参数
    rho_t = ft / (E0 * eps_t)

    def comp_y(x):
        return (alpha_a * x + (3.0 - 2.0 * alpha_a) * x ** 2 + (alpha_a - 2.0) * x ** 3) if x <= 1.0 \
            else x / (alpha_d * (x - 1.0) ** 2 + x)

    def comp_d(x):  # Sidoroff 能量等效: d = 1 - sqrt(σ/(E0·ε)) = 1 - sqrt(y·ρ/x)
        y = comp_y(x)
        ratio = y * rho_c / x if x > 0.0 else 1.0               # σ/(E0·ε)
        return 0.0 if ratio >= 1.0 else 1.0 - math.sqrt(ratio)

    def tens_y(x):
        return (1.2 * x - 0.2 * x ** 6) if x <= 1.0 else x / (alpha_t * (x - 1.0) ** 1.7 + x)

    def tens_d(x):  # Sidoroff 能量等效: d = 1 - sqrt(σ/(E0·ε)) = 1 - sqrt(y·ρ/x)
        y = tens_y(x)
        ratio = y * rho_t / x if x > 0.0 else 1.0               # σ/(E0·ε)
        return 0.0 if ratio >= 1.0 else 1.0 - math.sqrt(ratio)

    x_c = [0.02 + i * (5.0 - 0.02) / (n_pts - 1) for i in range(n_pts)]      # 受压：x 至 5(深入软化段)
    x_t = [0.02 + i * (12.0 - 0.02) / (n_pts - 1) for i in range(n_pts)]     # 受拉：x 至 12(拉伸软化更慢)
    comp_hardening, comp_damage = _monotone_inelastic_pairs(
        x_c, lambda x: comp_y(x) * fc, comp_d, eps_c, E0)
    tension_stiffening, tension_damage = _monotone_inelastic_pairs(
        x_t, lambda x: tens_y(x) * ft, tens_d, eps_t, E0)
    return {'comp_hardening': tuple(comp_hardening), 'comp_damage': tuple(comp_damage),
            'tension_stiffening': tuple(tension_stiffening), 'tension_damage': tuple(tension_damage)}


def _corner_rebar_positions(width, depth, cover, ratio):
    """矩形截面 4 角配筋：按配筋率反算单根钢筋面积，四角对称布置。返回 [(area, x1, x2), ...]，x1沿宽度、x2沿深度(截面局部轴,原点=形心)。"""
    a_total = ratio * width * depth
    a_bar = a_total / 4.0
    x1 = width / 2.0 - cover
    x2 = depth / 2.0 - cover
    return [(a_bar, x1, x2), (a_bar, -x1, x2), (a_bar, x1, -x2), (a_bar, -x1, -x2)]


# ==========================================================
#  框架 Part（局部基底 y=0；同 step2a）
# ==========================================================

def build_frame_part(model, logger):
    nb = int(frame_cfg['n_bay']); ns = int(frame_cfg['n_story'])
    bw = float(frame_cfg['bay_width']); sh = float(frame_cfg['story_height'])
    xs = [j * bw for j in range(nb + 1)]; ys = [k * sh for k in range(ns + 1)]; z = 0.0
    sk = model.ConstrainedSketch(name='__frame__', sheetSize=max(xs[-1], ys[-1]) * 2.0)
    for x in xs:
        for k in range(ns):
            sk.Line(point1=(x, ys[k]), point2=(x, ys[k + 1]))
    for k in range(1, ns + 1):
        for j in range(nb):
            sk.Line(point1=(xs[j], ys[k]), point2=(xs[j + 1], ys[k]))
    part = model.Part(name='Frame', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    part.BaseWire(sketch=sk)
    del model.sketches['__frame__']

    mc = frame_material_cfg
    mat = model.Material(name=mc['name'])
    mat.Elastic(table=((mc['E'], mc['nu']),))
    mat.Density(table=((mc['density'],),))
    # P0#4：瑞利阻尼拟合频段。'fixed'=固定 f1/f2(v1基线)；'modal'=按结构基本周期 T1 自动锚定前三阶
    if str(mc.get('rayleigh_mode', 'fixed')) == 'modal':
        T1_est = _frame_T1_estimate()          # 0.1N 估算或注入实测值
        f1_ray = 0.8 / T1_est                  # 下限≈0.8/T1(略低于一阶)
        f2_ray = 5.0 / T1_est                  # 上限≈5/T1(覆盖前三阶)
        log_step(logger, u'框架瑞利阻尼(modal): T1≈%.3fs -> f1=%.3fHz, f2=%.3fHz', T1_est, f1_ray, f2_ray)
    else:
        f1_ray = mc['f1']; f2_ray = mc['f2']   # 固定频段(v1基线)
    a_r, b_r = rayleigh_coeffs(mc['damping_ratio'], f1_ray, f2_ray)
    mat.Damping(alpha=a_r, beta=b_r)
    if tssi_cfg.get('nonlinear', True):  # step3：混凝土 CDP 非线性(GB50010 附录C.2 本构) + 钢筋弹塑性材料
        cdp = _gb50010_concrete_cdp_curves(mc['fc_mpa'], mc['ft_mpa'], mc['E'])
        mat.ConcreteDamagedPlasticity(table=((mc['dilation_angle'], mc['eccentricity'], mc['fb0_fc0'],
                                              mc['K'], mc['viscosity']),))
        mat.concreteDamagedPlasticity.ConcreteCompressionHardening(table=cdp['comp_hardening'])
        mat.concreteDamagedPlasticity.ConcreteCompressionDamage(table=cdp['comp_damage'])
        mat.concreteDamagedPlasticity.ConcreteTensionStiffening(table=cdp['tension_stiffening'], type=STRAIN)
        mat.concreteDamagedPlasticity.ConcreteTensionDamage(table=cdp['tension_damage'], type=STRAIN)
        rc = rebar_cfg
        mat_s = model.Material(name=rc['material'])
        mat_s.Elastic(table=((rc['Es'], rc['nu']),))
        mat_s.Density(table=((rc['density'],),))
        mat_s.Plastic(table=((rc['fy'], 0.0), (rc['fy'] + rc['hardening_ratio'] * rc['Es'] * 0.05, 0.05)))
        log_step(logger, u'框架材料: 混凝土CDP(fc=%.1fMPa,ft=%.2fMPa) + 钢筋%s(fy=%.0fMPa) 已定义',
                 mc['fc_mpa'], mc['ft_mpa'], rc['material'], rc['fy'] / 1.0e6)
    col = frame_cfg['column']; bm = frame_cfg['beam']
    model.RectangularProfile(name='ColProf', a=col['width'], b=col['depth'])
    model.RectangularProfile(name='BeamProf', a=bm['width'], b=bm['depth'])
    model.BeamSection(name='ColSec', profile='ColProf', material=mc['name'], integration=DURING_ANALYSIS, poissonRatio=mc['nu'])
    model.BeamSection(name='BeamSec', profile='BeamProf', material=mc['name'], integration=DURING_ANALYSIS, poissonRatio=mc['nu'])
    col_mids = [((x, (ys[k] + ys[k + 1]) / 2.0, z),) for x in xs for k in range(ns)]
    beam_mids = [(((xs[j] + xs[j + 1]) / 2.0, ys[k], z),) for k in range(1, ns + 1) for j in range(nb)]
    part.Set(edges=part.edges.findAt(*col_mids), name='COLS')
    part.Set(edges=part.edges.findAt(*beam_mids), name='BEAMS')
    col_base_mids = [((x, (ys[0] + ys[1]) / 2.0, z),) for x in xs]  # P0#3：底层柱段中点(基底剪力 SF 直取用)
    part.Set(edges=part.edges.findAt(*col_base_mids), name='COLS_BASE')
    part.SectionAssignment(region=part.sets['COLS'], sectionName='ColSec')
    part.SectionAssignment(region=part.sets['BEAMS'], sectionName='BeamSec')
    part.assignBeamSectionOrientation(region=part.Set(edges=part.edges, name='ALL_E'), method=N1_COSINES, n1=(0.0, 0.0, -1.0))
    part.Set(vertices=part.vertices.findAt(*[((x, ys[0], z),) for x in xs]), name='BASE')
    floor_full = {}
    for k in range(1, ns + 1):
        part.Set(vertices=part.vertices.findAt(((xs[0], ys[k], z),)), name='FLOOR_%d' % k)
        part.Set(vertices=part.vertices.findAt(*[((x, ys[k], z),) for x in xs]), name='FLOORALL_%d' % k)
        floor_full[k] = ('FLOORALL_%d' % k, len(xs))
    part.seedEdgeByNumber(edges=part.edges, number=1, constraint=FIXED)
    part.setElementType(regions=(part.edges,), elemTypes=(mesh.ElemType(elemCode=B21, elemLibrary=STANDARD),))
    part.generateMesh()
    log_step(logger, u'框架 Part: B21, 层=%d 跨=%d, 单元=%d', ns, nb, len(part.elements))
    return part, floor_full, ns


# ==========================================================
#  坡顶框架 Tie 耦合 + 历史输出 + TSSI 元数据
# ==========================================================

def add_frame_on_crest(model_name, geom, soil_part_name, soil_inst_name, logger):
    """框架坐坡顶 + 基础耦合到坡顶土面 + 楼层集中质量。返回 (frame_inst, ns)。

    基础形式(foundation_cfg['type'])：
      · 'tie'(默认,v1基线)：柱脚顶点直接 Tie 到坡顶土面(tieRotations=OFF 铰接、点绑定)；
      · 'footing'(P1#7)：柱脚下加实体条形基础板(CPE4R)，柱脚 Tie 基础顶(tieRotations=ON 刚接)，
        基础底 Tie 土面(分布式,缓解点绑定应力集中)；foundation contact=True 时基础底改硬接触+摩擦(P1#8,per-model)。
    """
    model = mdb.models[model_name]
    asm = model.rootAssembly
    frame, floor_full, ns = build_frame_part(model, logger)
    frame_inst = 'Frame-1'
    asm.Instance(name=frame_inst, part=frame, dependent=ON)
    fw = int(frame_cfg['n_bay']) * float(frame_cfg['bay_width'])
    # P0#2：距坡肩距离 M=crest_offset_B×fw。0=右缘贴坡肩(x=left_flat, v1基线)；>0 时框架整体左移，step4 扫描退让距离
    crest_off_B = float(tssi_cfg.get('crest_offset_B', 0.0))
    x_off = float(geom.left_flat) - fw - crest_off_B * fw
    if x_off < -1.0e-6:  # 框架左缘越出上平台左端，几何非法
        raise ValueError('crest_offset_B=%.3f 过大：框架左缘 x=%.2f 越出上平台[0,%.2f]，请减小或加大 left_flat'
                         % (crest_off_B, x_off, geom.left_flat))
    # P1#7：基础形式。footing 时框架整体上抬一个基础板厚 T（坐在基础顶），基础板占 [H_upper, H_upper+T]
    use_footing = (str(foundation_cfg.get('type', 'tie')) == 'footing')
    foot_T = float(foundation_cfg.get('thickness', 0.8)) if use_footing else 0.0
    asm.translate(instanceList=(frame_inst,), vector=(x_off, geom.H_upper + foot_T, 0.0))
    m_total = float(frame_cfg['floor_mass'])
    for k in range(1, ns + 1):
        nm, cnt = floor_full[k]
        asm.engineeringFeatures.PointMassInertia(
            name='Mass_%d' % k, region=asm.instances[frame_inst].sets[nm], mass=m_total / float(cnt))
    # 坡顶平台土面 [0,left_flat] 建 Tie/接触 主面
    soil_part = model.parts[soil_part_name]
    crest_edge = soil_part.edges.findAt(((geom.left_flat * 0.5, geom.H_upper, 0.0),))
    soil_part.Surface(side1Edges=crest_edge, name='CREST_SURF')
    if use_footing:  # P1#7：实体条形基础板 + 柱脚刚接基础顶 + 基础底耦合土面
        _build_footing(model, geom, x_off, fw, foot_T, soil_inst_name, logger, model_name)
    else:  # v1 基线：柱脚顶点直接 Tie 土面（铰接、点绑定）
        model.Tie(name='FrameSoil', master=asm.instances[soil_inst_name].surfaces['CREST_SURF'],
                  slave=asm.instances[frame_inst].sets['BASE'],
                  positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=OFF, thickness=ON)
    # 坡顶参考节点(TOP_SURFACE 中 x 最接近 left_flat)——供 SSI 后处理(框架基础运动/TAF)
    top = asm.instances[soil_inst_name].sets['TOP_SURFACE']
    crest_node = min(top.nodes, key=lambda n: abs(n.coordinates[0] - geom.left_flat))
    asm.Set(name='CREST_REF', nodes=asm.instances[soil_inst_name].nodes.sequenceFromLabels([crest_node.label]))
    log_step(logger, u'[%s] 坡顶框架已挂(x_off=%.1f,y=%.1f,M/B=%.2f,M=%.1fm,基础=%s)+耦合, 坡顶参考x=%.0f',
             model_name, x_off, geom.H_upper + foot_T, crest_off_B, crest_off_B * fw,
             ('footing T=%.2fm' % foot_T) if use_footing else 'tie', crest_node.coordinates[0])
    return frame_inst, ns


def _build_footing(model, geom, x_off, fw, foot_T, soil_inst_name, logger, model_name):
    """P1#7：建实体条形基础板(CPE4R)并耦合。基础占 x∈[cx-W/2,cx+W/2], y∈[H_upper,H_upper+T]。
    · 柱脚 BASE(y=H_upper+T) Tie 基础顶 FOOT_TOP，tieRotations=ON（刚接，传弯矩）；
    · 基础底 FOOT_BOT(y=H_upper) 与土面 CREST_SURF：contact=False 时 Tie，True 时留给 add_footing_contact 建接触。
    """
    asm = model.rootAssembly
    W = float(foundation_cfg['width']) if foundation_cfg.get('width') else fw * 1.2  # None=框架宽×1.2
    if W < fw - 1.0e-6:  # 基础须不窄于框架
        raise ValueError('foundation width=%.2f 小于框架宽 %.2f' % (W, fw))
    cx = x_off + fw / 2.0                       # 框架(与基础)水平中心
    x_left = cx - W / 2.0                        # 基础左缘
    if x_left < -1.0e-6 or (cx + W / 2.0) > geom.left_flat + 1.0e-6:  # 越出上平台
        raise ValueError('基础板 x∈[%.2f,%.2f] 越出上平台[0,%.2f]，请减小 width/crest_offset 或加大 left_flat'
                         % (x_left, cx + W / 2.0, geom.left_flat))
    msize = float(foundation_cfg['mesh_size']) if foundation_cfg.get('mesh_size') else max(foot_T / 2.0, 0.1)
    # 基础板 Part（局部坐标 x∈[0,W], y∈[0,T]）
    fs = model.ConstrainedSketch(name='__footing__', sheetSize=max(W, foot_T) * 2.0)
    fs.rectangle(point1=(0.0, 0.0), point2=(W, foot_T))
    fpart = model.Part(name='Footing', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    fpart.BaseShell(sketch=fs)
    del model.sketches['__footing__']
    fmat = model.Material(name='Concrete_Footing')
    fmat.Elastic(table=((float(foundation_cfg['E']), float(foundation_cfg['nu'])),))
    fmat.Density(table=((float(foundation_cfg['density']),),))
    model.HomogeneousSolidSection(name='FootSec', material='Concrete_Footing', thickness=1.0)  # 2D 平面应变单位厚
    fpart.SectionAssignment(region=Region(faces=fpart.faces), sectionName='FootSec')
    fpart.Set(faces=fpart.faces, name='ALL_F')  # 基础全单元集(重力 GRAV 用)
    fpart.Surface(side1Edges=fpart.edges.findAt(((W / 2.0, foot_T, 0.0),)), name='FOOT_TOP')  # 基础顶面
    fpart.Surface(side1Edges=fpart.edges.findAt(((W / 2.0, 0.0, 0.0),)), name='FOOT_BOT')      # 基础底面
    fpart.seedPart(size=msize, deviationFactor=0.1, minSizeFactor=0.1)
    fpart.setElementType(regions=(fpart.faces,), elemTypes=(mesh.ElemType(elemCode=CPE4R, elemLibrary=STANDARD),))
    fpart.generateMesh()
    asm.Instance(name='Footing-1', part=fpart, dependent=ON)
    asm.translate(instanceList=('Footing-1',), vector=(x_left, geom.H_upper, 0.0))  # 就位：底=H_upper
    # 柱脚 Tie 基础顶（刚接）
    model.Tie(name='FrameFooting', master=asm.instances['Footing-1'].surfaces['FOOT_TOP'],
              slave=asm.instances['Frame-1'].sets['BASE'],
              positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)
    if not foundation_cfg.get('contact'):  # 基础底 Tie 土面（默认）；contact=True 时改由 add_footing_contact 建接触
        model.Tie(name='FootingSoil', master=asm.instances[soil_inst_name].surfaces['CREST_SURF'],
                  slave=asm.instances['Footing-1'].surfaces['FOOT_BOT'],
                  positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=OFF, thickness=ON)
    log_step(logger, u'[%s] 条形基础板已建: 宽%.2fm×厚%.2fm(CPE4R,%d单元), 柱脚刚接基础顶, 基础底%s土面',
             model_name, W, foot_T, len(fpart.elements), u'接触(待per-model)' if foundation_cfg.get('contact') else u'Tie')


def _apply_frame_gravity(model, grav_mode, logger, model_name):
    """P0#1 Level A：重力静力步内给坡顶框架施加自重（build_models 建静力步后调用）。

    · 梁柱自重：GRAV 分布力作用于 Frame 单元（默认密度小、量级次要，但物理完整）；
    · 楼层重力：显式向下节点力 = 每节点分摊质量×g。不依赖"点质量吃 GRAV"
      （Abaqus 中 GRAV 对 engineeringFeatures 点质量的作用不确定），直接加力可确保
      柱底轴力锚点 N≈Σ楼层质量·g/柱数（默认 5 层 4 柱 ≈613kN，见 §5 QA）。
    Level A 不给土体加重力（避免静-动边界切换），土由 VAB 接地弹簧承静载。
    grav_mode: 'structure'=Level A（本实现）；'full'=Level B 全模型 geostatic（P2 未实现，回退 Level A 并告警）。
    """
    asm = model.rootAssembly
    frame = asm.instances['Frame-1']
    if grav_mode == 'full':  # Level B 尚未实现
        log_step(logger, u'[%s] gravity=full(Level B 全模型geostatic)尚未实现，回退 Level A 仅结构自重', model_name)
    try:  # 梁柱自重(仅框架单元,土体不加)；密度小量级次要，失败不影响楼层力主锚点
        model.Gravity(name='Grav-frame', createStepName=GRAVITY_STEP_NAME,
                      comp2=-GRAVITY_G, distributionType=UNIFORM, region=frame.sets['ALL_E'])
    except Exception as _e:
        log_step(logger, u'[%s] 框架 GRAV 施加失败(忽略,梁柱自重次要): %s', model_name, str(_e))
    if 'Footing-1' in asm.instances.keys():  # P1#7：条形基础板自重(实体单元,GRAV 直接生效)
        try:
            model.Gravity(name='Grav-footing', createStepName=GRAVITY_STEP_NAME,
                          comp2=-GRAVITY_G, distributionType=UNIFORM, region=asm.instances['Footing-1'].sets['ALL_F'])
        except Exception as _e:
            log_step(logger, u'[%s] 基础板 GRAV 施加失败(忽略): %s', model_name, str(_e))
    ns = int(frame_cfg['n_story'])
    n_col = int(frame_cfg['n_bay']) + 1              # 每层节点数=柱数=跨数+1
    m_node = float(frame_cfg['floor_mass']) / float(n_col)  # 每节点分摊质量(与点质量分摊一致)
    f_node = m_node * GRAVITY_G                      # 每节点重力(N,向下)
    for k in range(1, ns + 1):  # 逐层楼板重力作向下节点力
        model.ConcentratedForce(name='Wt-Floor%d' % k, createStepName=GRAVITY_STEP_NAME,
                                region=frame.sets['FLOORALL_%d' % k], cf2=-f_node,
                                distributionType=UNIFORM, field='', localCsys=None)
    log_step(logger, u'[%s] 重力(Level A)已施加: 框架GRAV + %d层×%d节点(每节点%.1fkN); 预期柱底轴力≈%.0fkN(手算锚点)',
             model_name, ns, n_col, f_node / 1.0e3,
             ns * float(frame_cfg['floor_mass']) * GRAVITY_G / float(n_col) / 1.0e3)


def add_frame_outputs(model_name, logger):
    """给已建步的 SSI 模型配 TSSI 历史/场输出。build_models 建步后逐波调用。

    P0#3 输出补全：
      · 楼层：U1/U2/V1/A1/A2（增 U2/A2 摇摆/竖向）；
      · 柱脚 BASE：U1/U2（摇摆角 θ=(u2_前柱−u2_后柱)/B，后处理算）；
      · 底层柱单元 COLS_BASE：SF1/SF2/SM1（基底剪力直取，与 Σm·a 互校）；
      · nonlinear 时：Frame 单元集损伤场 DAMAGET/DAMAGEC/PEEQ（降频）+ 整体能量 ALLPD/ALLIE/ETOTAL。
    """
    model = mdb.models[model_name]
    asm = model.rootAssembly
    freq = int(tssi_cfg.get('history_freq', 1))
    ns = int(frame_cfg['n_story'])
    frame = asm.instances['Frame-1']
    # 静力合法量(位移U/截面力SF)锚到【首步】：有重力步时=Step-gravity，使其覆盖重力步(verify 读重力末帧柱底轴力)；
    # 含 V/A(速度/加速度)的量只在地震动力步有意义，锚到 Step-earthquake。
    first_step = GRAVITY_STEP_NAME if GRAVITY_STEP_NAME in model.steps.keys() else DEFAULT_STEP_NAME
    model.HistoryOutputRequest(name='H-Crest', createStepName=DEFAULT_STEP_NAME,
                               variables=('U1', 'A1'), region=asm.sets['CREST_REF'], frequency=freq)
    for k in range(1, ns + 1):  # 楼层：增 U2/A2（摇摆/竖向分量，Q5）；含 A/V 故锚地震步
        model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName=DEFAULT_STEP_NAME,
                                   variables=('U1', 'U2', 'V1', 'A1', 'A2'),
                                   region=frame.sets['FLOOR_%d' % k], frequency=freq)
    # 柱脚位移：U1/U2 供摇摆角 θ 计算（后处理取前/后柱 u2 差 / 框架宽）；静力合法，锚首步覆盖重力步
    model.HistoryOutputRequest(name='H-Base', createStepName=first_step,
                               variables=('U1', 'U2'), region=frame.sets['BASE'], frequency=freq)
    # 底层柱截面力：SF1(轴力)/SF2(剪力)/SM1(弯矩)，基底剪力 V=ΣSF2 直取；静力合法，锚首步以覆盖重力步(轴力锚点)
    model.HistoryOutputRequest(name='H-ColBase', createStepName=first_step,
                               variables=('SF1', 'SF2', 'SM1'), region=frame.sets['COLS_BASE'], frequency=freq)
    if tssi_cfg.get('nonlinear', True):  # 非线性：损伤场 + 能量（弹性时无损伤/塑性耗能，跳过）
        dmg_freq = max(1, freq) * 10  # 损伤场输出降频（ODB 瘦身，损伤演化慢于响应）
        model.FieldOutputRequest(name='F-Frame-Damage', createStepName=DEFAULT_STEP_NAME,
                                 variables=('DAMAGET', 'DAMAGEC', 'PEEQ', 'S', 'E'),
                                 region=frame.sets['ALL_E'], frequency=dmg_freq)
        model.HistoryOutputRequest(name='H-Energy', createStepName=DEFAULT_STEP_NAME,
                                   variables=('ALLPD', 'ALLIE', 'ALLKE', 'ALLSE', 'ETOTAL'), frequency=freq)
    log_step(logger, u'[%s] TSSI 输出已配: 坡顶参考 + %d 层(U/V/A含2向) + 柱脚U2 + 底柱SF + %s',
             model_name, ns, u'损伤场/能量' if tssi_cfg.get('nonlinear', True) else u'(弹性,无损伤)')


def add_footing_contact(model_name, soil_inst_name, logger):
    """P1#8：基础板底与坡顶土面建【硬接触 + 库仑摩擦】，允许提离/滑移（替代 FootingSoil Tie）。

    仅 foundation type='footing' 且 contact=True 时生效。接触需在已建分析步上创建，故本函数
    在 build_models 建步之后逐波调用（与 add_frame_outputs 并列）。接触自首步(有重力步=重力步)起激活，
    使基础在自重下压紧土面、强震下可提离/滑移，考察摇摆/滑移占比（仅批 C 强震代表工况建议启用）。
    收敛提示：硬接触+隐式在强震下较难收敛，配合 P1#9 降级链（增大 CDP 粘性、减小最小增量）。
    """
    if not (str(foundation_cfg.get('type', 'tie')) == 'footing' and foundation_cfg.get('contact')):
        return  # 非接触工况：基础底已在 _build_footing 里 Tie，无需处理
    if str(tssi_cfg.get('gravity', 'off')) == 'off':  # 接触需自重压紧,否则基础悬空可张开
        log_step(logger, u'[%s] 警告：contact=True 但 gravity=off，基础无自重压紧、可能初始张开，建议同开 gravity=structure', model_name)
    model = mdb.models[model_name]
    asm = model.rootAssembly
    mu = float(foundation_cfg.get('mu', 0.5))
    prop = model.ContactProperty('FootSoilProp')
    prop.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON, constraintEnforcementMethod=DEFAULT)  # 硬接触可提离
    prop.TangentialBehavior(formulation=PENALTY, directionality=ISOTROPIC, table=((mu,),),  # 库仑摩擦(罚)
                            maximumElasticSlip=FRACTION, fraction=0.005)
    first_step = GRAVITY_STEP_NAME if GRAVITY_STEP_NAME in model.steps.keys() else DEFAULT_STEP_NAME  # 自重步起激活
    model.SurfaceToSurfaceContactStd(
        name='FootSoilContact', createStepName=first_step,
        master=asm.instances[soil_inst_name].surfaces['CREST_SURF'],
        slave=asm.instances['Footing-1'].surfaces['FOOT_BOT'],
        sliding=FINITE, interactionProperty='FootSoilProp',
        adjustMethod=OVERCLOSED, initialClearance=OMIT, thickness=ON)  # 初始过闭合自动调整贴合
    log_step(logger, u'[%s] P1#8 基础-土 硬接触+库仑摩擦(μ=%.2f)已建, 自 %s 起激活(可提离滑移)',
             model_name, mu, first_step)


def _find_beamsection_keyword_index(sie_blocks, elset_tag):
    """在 keywordBlock.sieBlocks 中定位 '*Beam Section, elset=<elset_tag>...' 所在块索引(sieBlocks每项=整块非单行)，找不到返回 None。"""
    prefix = '*Beam Section'
    tag = 'elset=%s' % elset_tag
    for i, line in enumerate(sie_blocks):
        if line.startswith(prefix) and tag in line:
            return i
    return None


def add_frame_rebar(model_name, logger):
    """给已完成建步/边界/输出配置的 SSI 模型注入梁柱角部钢筋(*Rebar, element=BEAM 关键字)。

    Abaqus/CAE 图形化建模不支持梁截面钢筋（*REBAR 只能通过关键字编辑器/脚本 keywordBlock 注入），
    故用 model.keywordBlock 在 '*Beam Section, elset=COLS/BEAMS' 的截面尺寸数据行后插入钢筋定义。
    按 Abaqus 惯例，keyword 编辑应是对模型的【最后一步操作】(此后不应再对该模型做图形化修改)，
    故须在 build_models(建步+边界)与 add_frame_outputs(历史输出)都完成后、提交作业前调用。
    """
    if not tssi_cfg.get('nonlinear', True):  # 非线性关闭时(=step2弹性行为)不注入钢筋
        return
    model = mdb.models[model_name]
    rc = rebar_cfg
    col = frame_cfg['column']; bm = frame_cfg['beam']
    col_bars = _corner_rebar_positions(col['width'], col['depth'], rc['cover'], rc['column']['ratio'])
    beam_bars = _corner_rebar_positions(bm['width'], bm['depth'], rc['cover'], rc['beam']['ratio'])
    model.keywordBlock.synchVersions(storeNodesAndElements=False)

    def _rebar_text(bars, tag, elset_tag):
        # NAME/MATERIAL 是 *Rebar 关键字参数(非数据行字段)；数据行首列须是elset/单元标签(非面积)，此处整个elset共用同一根钢筋定义
        blocks = []
        for k, (area, x1, x2) in enumerate(bars):
            name = '%s-%d' % (tag, k + 1)
            blocks.append('*Rebar, element=BEAM, material=%s, name=%s\n%s, %.6e, %.6f, %.6f'
                           % (rc['material'], name, elset_tag, area, x1, x2))
        return '\n'.join(blocks)

    # 注：sieBlocks 每项是【整块】(*Beam Section 关键字行+其截面尺寸数据行合并为一个 String)，
    # insert(position, text) 语义是"插到 position 这一块之后"，故直接用 idx（非 idx+1/+2）。
    idx_c = _find_beamsection_keyword_index(model.keywordBlock.sieBlocks, 'COLS')
    if idx_c is not None:
        model.keywordBlock.insert(idx_c, _rebar_text(col_bars, 'RebarC', 'COLS'))
    else:
        log_step(logger, u'[%s] 未找到 *Beam Section elset=COLS 关键字块，柱钢筋注入跳过', model_name)

    idx_b = _find_beamsection_keyword_index(model.keywordBlock.sieBlocks, 'BEAMS')  # 上次插入后块索引已整体后移，需重新查找
    if idx_b is not None:
        model.keywordBlock.insert(idx_b, _rebar_text(beam_bars, 'RebarB', 'BEAMS'))
    else:
        log_step(logger, u'[%s] 未找到 *Beam Section elset=BEAMS 关键字行，梁钢筋注入跳过', model_name)

    n_eff_col = col_bars[0][0] / (math.pi * rc['column']['bar_diameter'] ** 2 / 4.0)  # 单角等效Φd实际根数(配筋率反算面积/单根面积)
    n_eff_beam = beam_bars[0][0] / (math.pi * rc['beam']['bar_diameter'] ** 2 / 4.0)
    log_step(logger, u'[%s] 梁柱钢筋已注入: 柱4角×%.0fmm²(≈%.1f根Φ%.0fmm), 梁4角×%.0fmm²(≈%.1f根Φ%.0fmm)',
             model_name, col_bars[0][0] * 1.0e6, n_eff_col, rc['column']['bar_diameter'] * 1000,
             beam_bars[0][0] * 1.0e6, n_eff_beam, rc['beam']['bar_diameter'] * 1000)


def write_tssi_meta(logger):
    """写 tssi_meta.json：框架参数(供 SSI 后处理 Postprocess_SSI_response 读取)。"""
    meta = {'n_story': int(frame_cfg['n_story']), 'n_bay': int(frame_cfg['n_bay']),
            'story_height': float(frame_cfg['story_height']), 'floor_mass': float(frame_cfg['floor_mass']),
            'inst_frame': 'Frame-1',
            'scene': str(tssi_cfg.get('scene', 'ssi')),  # 三胞胎场景标记(ssi/freefield/fixed)
            'T_fixed_step1': _frame_T1_estimate(),  # P0#6：固定基础 T1(周期延长基准)，注入实测优先否则0.1N，去硬编码0.5
            'crest_offset_B': float(tssi_cfg.get('crest_offset_B', 0.0)),  # P0#2：距坡肩距离 M/B(step4 扫描)
            'gravity': str(tssi_cfg.get('gravity', 'off')),  # P0#1：重力级别(off/structure/full)
            'nlgeom': bool(tssi_cfg.get('nlgeom', False)),  # P1#10：几何非线性(P-Δ)
            'foundation_type': str(foundation_cfg.get('type', 'tie')),  # P1#7：tie/footing
            'foundation_contact': bool(foundation_cfg.get('contact', False)),  # P1#8：基础底是否硬接触
            'nonlinear': bool(tssi_cfg.get('nonlinear', True)),  # step3: True=CDP混凝土+钢筋纤维截面
            'concrete_fc_mpa': frame_material_cfg.get('fc_mpa'), 'concrete_ft_mpa': frame_material_cfg.get('ft_mpa'),
            'rebar_ratio_column': rebar_cfg['column']['ratio'], 'rebar_ratio_beam': rebar_cfg['beam']['ratio']}
    with open('tssi_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    log_step(logger, u'tssi_meta.json 已写(框架参数, 供 SSI 后处理)')


# ==========================================================
#  三胞胎去耦场景（scene='freefield'/'fixed'/'ssi'）
# ==========================================================


def _add_crest_ref_for_freefield(base_model, geom, inst_name, logger):
    """freefield 场景：在纯坡地模型上补建 CREST_REF 参考点集 + 历史输出。

    freefield 场景 tssi_cfg['enable']=False 不建框架，但仍需在坡顶创建
    CREST_REF（坡肩附近节点），以便解算后提取坡顶运动供 fixed 场景输入。
    逻辑复用 add_frame_on_crest 中 CREST_REF 的创建方式。
    """
    model = mdb.models[base_model]
    asm = model.rootAssembly
    # 在 TOP_SURFACE 中找 x 最接近坡肩(left_flat)的节点
    top = asm.instances[inst_name].sets['TOP_SURFACE']
    # P0#2：距坡肩距离 M=crest_offset_B×框架宽；freefield 场景也需按同 offset 取参考点，与 ssi 一致
    fw = int(frame_cfg['n_bay']) * float(frame_cfg['bay_width'])  # 框架宽度
    crest_off_B = float(tssi_cfg.get('crest_offset_B', 0.0))  # 距坡肩距离倍数
    x_target = float(geom.left_flat) - fw / 2.0 - crest_off_B * fw  # 框架中心 x（与 ssi 场景同口径）
    x_target = max(0.0, min(x_target, float(geom.left_flat)))  # 截断到上平台范围
    crest_node = min(top.nodes, key=lambda n: abs(n.coordinates[0] - x_target))  # 最近节点
    asm.Set(name='CREST_REF', nodes=asm.instances[inst_name].nodes.sequenceFromLabels([crest_node.label]))
    log_step(logger, u'[freefield] 坡顶参考点 CREST_REF 已建: x=%.1f(目标x=%.1f), y=%.1f',
             crest_node.coordinates[0], x_target, crest_node.coordinates[1])


def _add_freefield_crest_outputs(model_name, inst_name, logger):
    """给 freefield 场景的各波模型配 CREST_REF 历史输出（A1/A2/U1/U2）。

    在 build_models 建步之后逐波调用（与 ssi 场景的 add_frame_outputs 对应）。
    输出变量覆盖水平+竖向，供 extract_crest_motion 提取供 fixed 场景消费。
    """
    model = mdb.models[model_name]
    asm = model.rootAssembly
    freq = int(tssi_cfg.get('history_freq', 1))
    model.HistoryOutputRequest(name='H-Crest', createStepName=DEFAULT_STEP_NAME,
                               variables=('U1', 'U2', 'A1', 'A2'),
                               region=asm.sets['CREST_REF'], frequency=freq)
    log_step(logger, u'[%s] freefield 坡顶输出已配: CREST_REF U1/U2/A1/A2', model_name)


def extract_crest_motion(odb_path, step_name=None, logger=None):
    """从已解算的 freefield ODB 提取 CREST_REF 节点加速度时程，导出为 .txt 文件。

    参数
    ----
    odb_path   : freefield ODB 文件路径（如 'job-Ricker3-slope.odb'）
    step_name  : 分析步名称（None=使用 DEFAULT_STEP_NAME）
    logger     : 日志器（None=不打印）

    返回
    ----
    dict : {'h': 导出的水平加速度文件路径, 'v': 导出的竖向加速度文件路径}

    导出文件格式：两列（时间, 加速度 m/s²），与 v2 输入记录格式一致。
    文件名：crest_motion_h.txt / crest_motion_v.txt，写入 ODB 同目录。
    此函数不在 main() 中自动调用（freefield ODB 须先解算完成），
    由外部 Autorun 批处理脚本或后处理脚本调用。
    """
    from odbAccess import openOdb  # Abaqus ODB 访问模块（仅在调用时导入）
    step_name = step_name or DEFAULT_STEP_NAME  # 默认分析步名
    odb = openOdb(str(odb_path), readOnly=True)
    try:
        # 定位 CREST_REF 节点（Assembly 级节点集）
        asm_keys = list(odb.rootAssembly.nodeSets.keys())
        # Abaqus ODB 中节点集名称可能全大写
        crest_key = None
        for k in asm_keys:
            if k.upper() == 'CREST_REF':
                crest_key = k
                break
        if crest_key is None:
            raise KeyError(u'ODB 中未找到 CREST_REF 节点集（freefield 场景须先建 CREST_REF）: %s' % odb_path)
        node_set = odb.rootAssembly.nodeSets[crest_key]
        # 获取实例名和节点标签（CREST_REF 只含一个节点）
        inst_name = list(odb.rootAssembly.instances.keys())[0]  # 单 Part 模型只有一个实例
        node_label = node_set.nodes[0][0].label  # nodeSets.nodes 是 [[node,...],...]（按实例分组）
        # 从 historyRegions 提取加速度时程
        step = odb.steps[step_name]
        hr_key = 'Node %s.%d' % (inst_name, node_label)  # Abaqus 历史区域键格式
        # 尝试精确匹配，不区分大小写
        hr = None
        for k in step.historyRegions.keys():
            if k.upper() == hr_key.upper():
                hr = step.historyRegions[k]
                break
        if hr is None:
            raise KeyError(u'ODB 历史区域未找到 CREST_REF 节点: 期望 %s, 可用: %s' % (hr_key, list(step.historyRegions.keys())[:5]))
        # 提取 A1（水平）和 A2（竖向）
        out_dir = os.path.dirname(os.path.abspath(str(odb_path)))
        result = {}
        for var, suffix in [('A1', 'h'), ('A2', 'v')]:
            if var in hr.historyOutputs:
                data = np.array(hr.historyOutputs[var].data, dtype=float)
                out_file = os.path.join(out_dir, 'crest_motion_%s.txt' % suffix)
                np.savetxt(out_file, data, fmt='%.10e', delimiter='\t',
                           header='Time(s)\t%s(m/s2) extracted from %s CREST_REF' % (var, os.path.basename(str(odb_path))))
                result[suffix] = out_file
                if logger:
                    log_step(logger, u'坡顶运动已导出: %s, 点数=%d, |%s|max=%.4f m/s²',
                             out_file, len(data), var, float(np.max(np.abs(data[:, 1]))))
            else:
                if logger:
                    log_step(logger, u'警告: ODB 中 CREST_REF 无 %s 输出，跳过', var)
    finally:
        odb.close()
    return result


def _find_fixed_input(logger):
    """查找 fixed 场景的基底输入加速度文件。

    优先使用 tssi_cfg['fixed_input'] 指定的路径（绝对或相对工况目录）；
    未指定时自动在工况目录查找 crest_motion_h.txt（extract_crest_motion 的默认输出）。
    返回 (文件路径, 时间段, 增量步长) 元组，与 find_acc_txt 返回格式对齐。
    """
    fi = tssi_cfg.get('fixed_input')  # 用户指定路径
    if fi:
        path = str(fi)
        if not os.path.isabs(path):  # 相对路径按工况目录解析
            path = os.path.join(os.getcwd(), path)
    else:
        path = os.path.join(os.getcwd(), 'crest_motion_h.txt')  # 默认查找
    if not os.path.isfile(path):
        raise IOError(u'fixed 场景输入文件不存在: %s（需先跑 freefield 并调用 extract_crest_motion 导出）' % path)
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(u'fixed 输入文件格式错误(需两列 时间/加速度): %s' % path)
    t = data[:, 0].astype(float)
    a = data[:, 1].astype(float)
    dt = float(t[1] - t[0])
    tp = float(t[-1])
    if logger:
        log_step(logger, u'[fixed] 基底输入: %s, 点数=%d, dt=%.5fs, 时长=%.2fs, PGA=%.4f m/s²',
                 path, len(t), dt, tp, float(np.max(np.abs(a))))
    return path, t, a, dt, tp


def build_fixed_model(logger):
    """建固定基础框架单体模型（三胞胎 fixed 场景）。

    逻辑参考 frame_ssi_v1.py 的 build_fixed_scene()，但完整适配 v2 配置体系：
    - 框架 Part：复用 build_frame_part（含 CDP+钢筋材料定义）
    - 柱脚边界：嵌固（u2=0, ur3=0）+ AccelerationBC（输入 freefield 坡顶运动）
    - 重力步：继承 tssi_cfg['gravity']（与 ssi 场景一致）
    - 分析步：AUTOMATIC（CDP 非线性时）或 FIXED（弹性时）
    - 输出：与 ssi 场景同口径（楼层 U/V/A、柱脚 U、底柱 SF、损伤/能量）
    - 钢筋注入：add_frame_rebar（keyword 注入，最后调用）

    返回 model_name 列表（与 build_models 返回格式一致，仅含一个模型名）。
    """
    # 读取输入时程
    input_path, t_in, a_in, dt_in, tp_in = _find_fixed_input(logger)
    record_name = os.path.splitext(os.path.basename(input_path))[0]  # 记录名（用于模型命名）
    model_name = '%s-fixed' % record_name  # 模型命名

    # 清理同名旧模型
    if model_name in mdb.models:
        del mdb.models[model_name]
    model = mdb.Model(name=model_name)
    asm = model.rootAssembly
    asm.DatumCsysByDefault(CARTESIAN)

    # 建框架 Part（复用 build_frame_part：含 CDP 本构 + 瑞利阻尼 + 钢筋材料定义）
    frame, floor_full, ns = build_frame_part(model, logger)
    frame_inst = 'Frame-1'
    asm.Instance(name=frame_inst, part=frame, dependent=ON)

    # 楼层集中质量（与 add_frame_on_crest 同逻辑）
    m_total = float(frame_cfg['floor_mass'])
    for k in range(1, ns + 1):
        nm, cnt = floor_full[k]
        asm.engineeringFeatures.PointMassInertia(
            name='Mass_%d' % k, region=asm.instances[frame_inst].sets[nm], mass=m_total / float(cnt))

    # 尾段时长（与 ssi 场景一致）
    tail = float(tssi_cfg.get('tail_seconds', 0.0) if 'tail_seconds' in tssi_cfg else 0.0)

    # 分析步（隐式动力）
    nlgeom_flag = ON if tssi_cfg.get('nlgeom', False) else OFF
    min_inc_factor = float(tssi_cfg.get('cdp_min_inc_factor', 1.0e-4))
    step_name = DEFAULT_STEP_NAME
    if tssi_cfg.get('nonlinear', True):  # CDP 非线性用自动增量
        model.ImplicitDynamicsStep(
            name=step_name, previous='Initial',
            timePeriod=tp_in + tail, timeIncrementationMethod=AUTOMATIC,
            initialInc=dt_in, minInc=dt_in * min_inc_factor, maxInc=dt_in, maxNumInc=1000000,
            nlgeom=nlgeom_flag, application=TRANSIENT_FIDELITY)
    else:  # 弹性用固定增量
        model.ImplicitDynamicsStep(
            name=step_name, previous='Initial',
            timePeriod=tp_in + tail, timeIncrementationMethod=FIXED, initialInc=dt_in,
            maxNumInc=1000000, nlgeom=nlgeom_flag, application=TRANSIENT_FIDELITY)

    # 场输出（与 ssi 场景同口径）
    variables = _normalize_output_variables(job_cfg.get('variables', ('U', 'V', 'A')))
    frequency = int(job_cfg.get('frequency', 1))
    model.fieldOutputRequests['F-Output-1'].setValues(variables=variables, frequency=frequency)

    # P0#1 重力步（与 ssi 场景一致，fixed 场景也需预加轴压以保证 CDP 柱损伤形态正确）
    grav_mode = str(tssi_cfg.get('gravity', 'off'))
    if grav_mode != 'off':
        model.StaticStep(name=GRAVITY_STEP_NAME, previous='Initial', timePeriod=1.0,
                         initialInc=0.1, minInc=1.0e-6, maxInc=1.0, maxNumInc=100,
                         nlgeom=nlgeom_flag)
        # 施加重力（仅框架自重+楼层力，无土体、无基础板）
        frame_region = asm.instances[frame_inst].sets['ALL_E']
        try:
            model.Gravity(name='Grav-frame', createStepName=GRAVITY_STEP_NAME,
                          comp2=-GRAVITY_G, distributionType=UNIFORM, region=frame_region)
        except Exception as _e:
            log_step(logger, u'[%s] 框架 GRAV 施加失败(忽略): %s', model_name, str(_e))
        n_col = int(frame_cfg['n_bay']) + 1
        m_node = float(frame_cfg['floor_mass']) / float(n_col)
        f_node = m_node * GRAVITY_G
        for k in range(1, ns + 1):
            model.ConcentratedForce(name='Wt-Floor%d' % k, createStepName=GRAVITY_STEP_NAME,
                                    region=asm.instances[frame_inst].sets['FLOORALL_%d' % k], cf2=-f_node,
                                    distributionType=UNIFORM, field='', localCsys=None)
        log_step(logger, u'[%s] 重力(Level A)已施加: 框架GRAV + %d层楼层力', model_name, ns)

    # 柱脚边界条件：嵌固（竖向+转动约束）
    base = asm.instances[frame_inst].sets['BASE']
    model.DisplacementBC(name='BaseVR', createStepName='Initial', region=base, u2=0.0, ur3=0.0)

    # 基底加速度输入（freefield 坡顶运动）
    amp_data = tuple((float(t_in[i]), float(a_in[i])) for i in range(len(t_in)))
    model.TabularAmplitude(name='AccAmp', data=amp_data, timeSpan=STEP)
    model.AccelerationBC(name='BaseAcc', createStepName=step_name, region=base, a1=1.0, amplitude='AccAmp')
    log_step(logger, u'[%s] 柱脚嵌固 + 基底 AccelerationBC 已施加(输入=%s)', model_name, os.path.basename(input_path))

    # 历史输出（与 ssi 场景同口径）
    freq = int(tssi_cfg.get('history_freq', 1))
    first_step = GRAVITY_STEP_NAME if GRAVITY_STEP_NAME in model.steps.keys() else step_name
    # 楼层：U1/U2/V1/A1/A2
    for k in range(1, ns + 1):
        model.HistoryOutputRequest(name='H-Floor%d' % k, createStepName=step_name,
                                   variables=('U1', 'U2', 'V1', 'A1', 'A2'),
                                   region=asm.instances[frame_inst].sets['FLOOR_%d' % k], frequency=freq)
    # 柱脚位移：U1/U2（摇摆角计算用）
    model.HistoryOutputRequest(name='H-Base', createStepName=first_step,
                               variables=('U1', 'U2'), region=base, frequency=freq)
    # 底层柱截面力：SF1/SF2/SM1（基底剪力直取）
    model.HistoryOutputRequest(name='H-ColBase', createStepName=first_step,
                               variables=('SF1', 'SF2', 'SM1'),
                               region=asm.instances[frame_inst].sets['COLS_BASE'], frequency=freq)
    # 柱脚反力（fixed 场景特有，校核用）
    model.HistoryOutputRequest(name='H-BaseRF', createStepName=step_name,
                               variables=('RF1', 'RF2'), region=base, frequency=freq)
    if tssi_cfg.get('nonlinear', True):  # 损伤场+能量
        dmg_freq = max(1, freq) * 10
        model.FieldOutputRequest(name='F-Frame-Damage', createStepName=step_name,
                                 variables=('DAMAGET', 'DAMAGEC', 'PEEQ', 'S', 'E'),
                                 region=asm.instances[frame_inst].sets['ALL_E'], frequency=dmg_freq)
        model.HistoryOutputRequest(name='H-Energy', createStepName=step_name,
                                   variables=('ALLPD', 'ALLIE', 'ALLKE', 'ALLSE', 'ETOTAL'), frequency=freq)

    mdb.save()
    log_step(logger, u'[%s] fixed 场景模型已建: %d层%d跨框架, 嵌固基础, %s, 时长=%.2fs',
             model_name, ns, int(frame_cfg['n_bay']),
             u'CDP非线性' if tssi_cfg.get('nonlinear', True) else u'弹性',
             tp_in + tail)

    # 钢筋注入（keyword 编辑，须是对模型的最后操作）
    add_frame_rebar(model_name, logger)

    return [model_name]


# ==========================================================
#  主入口
# ==========================================================



def main():
    """脚本主入口：组织参数、建模、施加边界并提交作业。"""
    global material_cfg, geometry_cfg, damping_cfg, MAX_REFLECT_ORDER  # 声明为全局以便用注入配置整体覆盖（所有 callee 均按值取用，安全）
    logger = log_step()  # 自动使用与脚本同名的日志文件
    total_start = time.time()

    try:
        log_step(logger, '脚本开始执行 (slope_frame_ssi_full_v2)')  # 写入脚本启动日志

        # 配置注入：若工况文件夹有 case_config.json 则覆盖默认配置（v6 新增自由场引擎/运行控制注入）
        (material_cfg, geometry_cfg, damping_cfg,
         _mesh_cfg, _time_cfg, _max_reflect_order, _ff_cfg, _run_cfg) = _load_case_config(  # 加载并覆盖
            material_cfg, geometry_cfg, damping_cfg, logger)  # 传入默认配置与日志器
        mesh_size = float(_mesh_cfg.get('size', 4.0))  # v9.1：基准网格尺寸改由 mesh_cfg['size'] 提供（兼容旧顶层 mesh_size）
        MAX_REFLECT_ORDER = _max_reflect_order  # 全局更新反射截断阶数（仅 ray 引擎使用）

        geometry_cfg = _resolve_geometry_cfg(geometry_cfg, logger)  # 无量纲几何设计 → 绝对尺寸（研究计划§2.1，total_L 浮动）

        # 构建场地材料（基岩 + 土层列表）并校验厚度与基岩净空
        site, soil_thicknesses = build_site(material_cfg, geometry_cfg)
        n_total_layers = 1 + len(site.layers)  # 总层数（含基岩）
        sgeom = str(material_cfg.get('surface_geometry', 'horizontal'))  # v7：表层几何模式（可由 case_config.json 注入）
        if sgeom not in ('horizontal', 'terrain'):  # 校验模式合法性
            raise ValueError("surface_geometry 仅支持 'horizontal' 或 'terrain'，当前: %s" % sgeom)  # 抛出配置错误
        log_step(logger, '场地分层构建完成: 总层数(含基岩)=%d, 有限层=%s, 表层几何=%s',
                 n_total_layers, [L.name for L in site.layers], sgeom)
        log_step(logger, '基岩: Vs=%.0f m/s, ν=%.2f, ρ=%.0f kg/m³',
                 site.bedrock.cs, site.bedrock.vv, site.bedrock.density)
        for _L in site.layers:  # 逐层记录土层波速与厚度
            _vr = site.bedrock.cs / _L.cs if _L.cs > 0 else None  # 该层相对基岩波速比
            log_step(logger, '土层[%s]: Vs=%.0f m/s, Vr/Vs=%.2f, ν=%.2f, ρ=%.0f, 厚度=%.1f m',
                     _L.name, _L.cs, _vr, _L.vv, _L.density, _L.thickness)
        log_step(logger, '基岩顶面高程=%.1f m（坡脚面以下深度 %.1f m 恒定，土层扣完剩余归基岩）',
                 site.bedrock_thickness, geometry_cfg['H_lower'])

        geom = make_geometry(
            total_L=geometry_cfg['total_L'],  # 模型总长度
            H_minus_h=geometry_cfg['H_minus_h'],  # 斜坡高度差
            i=geometry_cfg['i'],  # 斜坡倾角
            left_flat=geometry_cfg['left_flat'],  # 上平台长度
            toe_surface_y=geometry_cfg['H_lower'],  # 坡脚地表高程（坡脚面以下深度恒定）
            soil_thicknesses=soil_thicknesses)  # 各土层厚度（从上到下，推层间界面与基岩顶面）

        acc_info = find_acc_txt(logger, wave_files=_run_cfg.get('wave_files'))  # 波形来源：注入路径优先，否则扫工况目录

        # 解析材料阻尼：若未显式指定主频 fc，则用首条加速度记录估计（多记录同 fc 是标准用法；不同 fc 时以首条为准）
        fc_est = None
        if damping_cfg.get('enable') and damping_cfg.get('fc') is None and acc_info:  # 启用阻尼且需自动估计 fc
            try:  # 估计失败不应中断建模
                _acc0 = np.loadtxt(acc_info[0][0])
                fc_est = _estimate_dominant_freq(_acc0[:, 1], _acc0[1, 0] - _acc0[0, 0])
                log_step(logger, '阻尼主频自动估计: fc=%.3f Hz（源记录: %s）', fc_est, acc_info[0][0])
            except Exception as _e:  # 估计异常
                log_step(logger, '阻尼主频估计失败(将依赖显式 fc 或回退): %s', str(_e))
        damping = _resolve_damping(damping_cfg, fc_est)  # 解析阻尼配置（补全 fc，供建材与 meta 共用）
        fc_resolved = damping.get('fc')  # 解析后的主频（网格/时间步校验复用）
        if damping.get('enable') and damping.get('anchor') == 'dual':  # v8：双控锚定（场地基频+输入主频）
            _f_site = _site_fundamental_freq(site, geom)  # 估算上平台柱场地基频
            if _f_site:  # 估算成功
                damping['f_site'] = _f_site  # 写入解析后配置（建材/fd 自由场/meta 共用同一份，口径一致）
                log_step(logger, '阻尼双控锚定: f_site=%.3f Hz（瑞利拟合下限取 min(f1_factor*fc, f_site)）', _f_site)
        if damping.get('enable') and damping.get('anchor') == 'perband' and fc_resolved:  # v9：逐层重锚定（QA 打印各层共振频带）
            _strat_log = _build_stratigraphy(site, geom, ymin=0.0, surface_geometry=sgeom)  # 与建材同口径分层
            _hc = float(damping.get('harmonics_cover', 3.0))  # 谐波覆盖次数
            for _idx, _band in enumerate(_strat_log):  # 逐带打印共振频率与拟合上限
                if _idx == 0:  # 基岩带跳过（无共振、Q≈999）
                    continue
                _fl = _band_resonance_freq(_band)  # 该层共振基频
                if _fl:  # 有效时记录
                    _f2_eff = max(damping['f2_factor'] * fc_resolved, _hc * _fl)  # 该层实际拟合上限
                    log_step(logger, '逐层重锚定: 层 %s f_layer=%.3f Hz -> 拟合上限 f2=%.3f Hz（旧 input 锚定仅 %.3f Hz）',
                             _band['name'], _fl, _f2_eff, damping['f2_factor'] * fc_resolved)
        log_step(logger, '材料阻尼: enable=%s, method=%s, anchor=%s, fc=%s',
                 damping.get('enable'), damping.get('method'), damping.get('anchor', 'input'), fc_resolved)

        # ── 土体非线性：等效线性(EQL) 在建 FE 前更新 site/damping ──
        _eql_meta = {'enable': False}  # EQL 结果(写入 case_meta)
        if eql_cfg.get('enable') and site.layers and acc_info:  # 启用 EQL 且有有限层与输入
            try:
                _acc_eql = np.loadtxt(acc_info[0][0])  # 首条输入(代表强度)
                _dt_eql = float(_acc_eql[1, 0] - _acc_eql[0, 0])
                site, _xi_by, _eql_info = _run_freefield_eql(site, geom, eql_cfg, _acc_eql[:, 1], _dt_eql, logger)
                damping['xi_by_layer'] = _xi_by  # 注入逐层 ξ(建材/自由场/meta 同口径)
                _eql_meta = {'enable': True, 'mode': eql_cfg.get('mode', '1d'), 'curve': eql_cfg.get('curve'),
                             'PI': eql_cfg.get('PI'), 'sigma0_kpa': eql_cfg.get('sigma0_kpa'), 'layers': _eql_info}
                log_step(logger, '土体非线性 EQL: 曲线=%s, 非线性层=%s', eql_cfg.get('curve'), list(_xi_by.keys()))
            except Exception as _e:  # EQL 失败回退线性, 不中断建模
                log_step(logger, 'EQL 失败(回退线性): %s', str(_e))

        # ── 项②：网格自适应 ─────────────────────────────────────────────────────
        mesh_used = mesh_size  # 默认使用配置指定尺寸
        if _mesh_cfg.get('auto') and fc_resolved:  # 启用自适应且已知主频
            delta_l = _max_element_size(site, fc_resolved, _mesh_cfg)
            mesh_used = min(mesh_size, delta_l)  # 取较小值（不超过 Kuhlemeyer-Lysmer 限值）
            if mesh_used < mesh_size:  # 自适应网格比配置值更细
                log_step(logger, '网格自适应: Δl_max=%.2fm -> mesh_used=%.2fm (原 mesh_size=%.2fm)',
                         delta_l, mesh_used, mesh_size)
            else:  # 配置值已满足判据
                log_step(logger, '网格判据: Δl_max=%.2fm >= mesh_size=%.2fm，无需细化', delta_l, mesh_size)
        else:  # 未启用自适应或主频未知
            log_step(logger, '网格尺寸: %.2fm（自适应未启用或 fc 未知）', mesh_used)
        # ── 网格自适应结束 ──────────────────────────────────────────────────────

        log_step(logger, '====== 阶段: fd 引擎建模前自检(解析对拍) ======')
        selfcheck = _fd_engine_selfcheck(logger)  # v8：fd 引擎建模前自检（解析对拍，失败即中止，毫秒级）

        log_step(logger, '====== 阶段: 写出工况元数据 case_meta.json ======')

        first_rec = acc_info[0][0] if acc_info else None  # v7：首条输入记录（理论台阶计算用）
        ff_theory = _write_case_meta(material_cfg, geom, site, mesh_used, _script_name(), logger,  # 写出统一工况元数据 case_meta.json
                                     damping=damping, ffcfg=_ff_cfg, sgeom=sgeom, acc_path=first_rec,  # 含 v7 ff_theory/x_crest/x_toe/a0
                                     selfcheck=selfcheck, eql_info=_eql_meta)  # v8 自检 + v2 EQL 结果写入 meta
        if ff_theory and ff_theory.get('left') and ff_theory.get('right'):  # v7：打印理论台阶（QA 锚点）
            log_step(logger, '远场一维理论台阶: 左(上平台) TAF_h=%.3f TAF_v=%.3f | 右(下平台) TAF_h=%.3f TAF_v=%.3f (FE 远场应与之一致±5%%)',
                     ff_theory['left']['taf_h'], ff_theory['left']['taf_v'],  # 左柱理论值
                     ff_theory['right']['taf_h'], ff_theory['right']['taf_v'])  # 右柱理论值

        log_step(logger, '运行控制: 自由场引擎=%s（已移除平坦对照模型，TAF 分母用解析自由场）', _ff_cfg.get('engine'))

        # ── 三胞胎场景调度 ────────────────────────────────────────────────────
        scene = str(tssi_cfg.get('scene', 'ssi'))  # 三胞胎场景：'ssi'(默认)/'freefield'/'fixed'
        if scene not in ('ssi', 'freefield', 'fixed'):
            raise ValueError("tssi_cfg['scene'] 仅支持 'ssi'/'freefield'/'fixed'，当前: %s" % scene)
        log_step(logger, '三胞胎场景: scene=%s', scene)

        # ── scene='fixed'：固定基础框架单体，不建土体 ─────────────────────────
        if scene == 'fixed':
            log_step(logger, '====== 阶段: fixed 场景——固定基础框架单体(不建土体/VAB) ======')
            write_tssi_meta(logger)  # 写 tssi_meta（含 scene='fixed'）
            model_names = build_fixed_model(logger)  # 建模 + 钢筋注入

            log_step(logger, '====== 阶段: 提交作业(共 %d 个模型) ======', len(model_names))
            for model_name in model_names:
                submit_job(
                    num_cpus=job_cfg['num_cpus'],
                    memory_percent=job_cfg['memory_percent'],
                    model_name=model_name,
                    logger=logger
                )

            log_step(logger, '所有作业已完成，总耗时=%.2fs', time.time() - total_start)
            return  # fixed 场景提前返回，不走坡地建模流程

        # ── scene='freefield' 预处理：强制 enable=False ───────────────────────
        if scene == 'freefield':
            tssi_cfg['enable'] = False  # freefield 场景不建框架
            log_step(logger, '[freefield] 已强制 tssi_cfg.enable=False（纯坡地模型）')

        # ── scene='ssi' 预处理：强制 enable=True ─────────────────────────────
        if scene == 'ssi':
            tssi_cfg['enable'] = True  # ssi 场景必须有框架
            log_step(logger, '[ssi] 已强制 tssi_cfg.enable=True（全耦合模型）')

        # ── 坡地建模流程（freefield 和 ssi 共用） ──────────────────────────────
        cae_name = 'h{}_i{}_a{}_L{}.cae'.format(int(geom.H_minus_h), int(geom.i),
                                                int(material_cfg['angle']), n_total_layers)  # 文件名追加总层数
        log_step(logger, '====== 阶段: 创建基础几何与网格模型 (cae=%s) ======', cae_name)

        base_model, part_name, inst_name = create_model(
            site=site, geom=geom, mesh_size=mesh_used, cae_name=cae_name, logger=logger, damping=damping,  # 场地/几何/自适应网格/文件名/阻尼
            surface_geometry=sgeom, elem_name=_mesh_cfg.get('elem', 'CPE4'), mesh_cfg=_mesh_cfg, fc=fc_resolved)  # v7 单元类型 + v9 分层网格 + v9.1 软层谐波加密(fc)

        if tssi_cfg.get('enable'):  # ssi 场景: 在坡地基础模型上追加坡顶框架(Tie 耦合); build_models 会复制到各波
            add_frame_on_crest(base_model, geom, part_name, inst_name, logger)
            write_tssi_meta(logger)

        if scene == 'freefield':  # freefield 场景: 补建 CREST_REF 参考点（不建框架但需提取坡顶运动）
            _add_crest_ref_for_freefield(base_model, geom, inst_name, logger)

        if eql_cfg.get('enable') and eql_cfg.get('mode') == '2d_element' and site.layers and acc_info:  # ② 逐单元 2D EQL(自管迭代提交)
            log_step(logger, '====== 阶段: 逐单元 2D EQL 迭代(自管建/提/读/更新) ======')
            _run_2d_element_eql(base_model, part_name, inst_name, site, geom, eql_cfg, damping, fc_resolved,
                                _ff_cfg, _time_cfg, sgeom, _run_cfg, acc_info[0], material_cfg['angle'], job_cfg, logger,
                                elem_name=_mesh_cfg.get('elem', 'CPE4'))  # v3：单元类型透传
            model_names = []
        else:  # 1D 应变相容 或 线性: 原批量建模+提交
            log_step(logger, '====== 阶段: 批量复制模型并施加人工边界(共 %d 条记录) ======', len(acc_info))
            slope_model_names = build_models(  # 批量复制斜坡模型并施加等效边界
                acc_info=acc_info, base_model=base_model, part_name=part_name, inst_name=inst_name,  # 地震动信息与基础模型/零件/实例
                site=site, geom=geom, angle=material_cfg['angle'],  # 场地、斜坡几何与入射角
                job=job_cfg, model_scene='slope', logger=logger,  # 作业配置与斜坡场景标签
                tcfg=_time_cfg, fc_used=fc_resolved, ffcfg=_ff_cfg, damping=damping,  # 时间步校验 + v6 引擎/阻尼
                surface_geometry=sgeom, surface_only=bool(_run_cfg.get('surface_only', False)),  # v7 表层几何 + v8 输出瘦身
                critical_angle_check=bool(_run_cfg.get('critical_angle_check', True)),  # v8：临界角校验开关
                elem_name=_mesh_cfg.get('elem', 'CPE4'))  # v3：单元类型透传(CPE8R 时边界自动用二次一致权重)
            model_names = slope_model_names  # 待提交的模型名称（已移除平坦对照模型）

        if tssi_cfg.get('enable') and model_names:  # ssi 场景: 各波 SSI 模型追加框架层历史输出(步已建)
            for _mn in model_names:
                add_frame_outputs(_mn, logger)
                add_footing_contact(_mn, inst_name, logger)  # P1#8：footing+contact 时建基础-土硬接触(步已建后)
                add_frame_rebar(_mn, logger)  # step3：keyword注入梁柱钢筋，须最后调用(其后不再对模型做图形化修改)

        if scene == 'freefield' and model_names:  # freefield 场景: 各波模型追加 CREST_REF 历史输出
            for _mn in model_names:
                _add_freefield_crest_outputs(_mn, inst_name, logger)

        log_step(logger, '====== 阶段: 提交作业(共 %d 个模型) ======', len(model_names))
        for model_name in model_names:
            submit_job(
                num_cpus=job_cfg['num_cpus'],
                memory_percent=job_cfg['memory_percent'],
                model_name=model_name,
                logger=logger
            )

        log_step(logger, '所有作业已完成，总耗时=%.2fs', time.time() - total_start)
    except Exception as exc:  # 捕获脚本运行异常
        log_step(logger, '脚本失败: %s', str(exc))
        logger.error('异常堆栈:\n%s', traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
