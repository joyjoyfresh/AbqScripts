# -*- coding: utf-8 -*-
"""工况元数据统一模块（单/双/多层模型通用的"唯一真相源"）。

背景与动机：各建模脚本把工况参数（坡角、坡高、覆盖层/基岩厚度、入射角、抗阻比、网格…）
  只编码在【文件夹名】里，且每个 Autorun 命名规则各不相同；后续 Collect/Plot 只能靠正则
  反猜，无法通用于不同层数与任意参数扫描。本模块把"工况参数"固化为一个机器可读的
  case_meta.json（统一 schema），由建模脚本在求解出全部派生量后写入各自工况文件夹：
    - 建模脚本：main() 末尾调用 build_meta(...) + write_case_meta(...)（单/双/多层都走同一 schema）；
    - Collect：读各文件夹的 case_meta.json，展平为 index.csv 的统一列（不再靠文件夹名反猜）；
    - Plot：直接按 index.csv 的规范列分组出图（坡角 i、入射角 θs、a0 等）。

本模块为【纯 Python】（仅依赖 json/os/math/re），可同时被 Abaqus 端建模脚本与普通 Python
  端的 Collect/Plot 导入，确保三方共用同一份字段定义、避免口径漂移。

层序约定：layers 为【从上到下】的有限层（覆盖层/表层…），不含基岩；
  layers[0]=最顶层（表层），layers[-1]=最底有限层（紧贴基岩的覆盖层，其 Vs 即 Vs2）。
  单层均质模型：bedrock=None，layers=[均质介质]；双层：bedrock + 1 个覆盖层；三层：bedrock + 表层 + 覆盖层。
"""

import os  # 导入系统路径模块
import re  # 导入正则模块（频率解析用）
import json  # 导入 JSON 读写模块
import math  # 导入数学模块（派生量计算用）

SCHEMA_VERSION = 1  # 元数据 schema 版本号（字段变更时递增）
META_FILENAME = 'case_meta.json'  # 统一的元数据文件名


# ==============================================================================
#  数值与材料辅助
# ==============================================================================
def _f(value):  # 把数值安全转为内置 float（兼容 numpy 标量）
    """将 numpy/字符串等数值规范化为内置 float；None 原样返回，无法转换返回 None。"""
    if value is None:  # 空值
        return None  # 原样返回
    try:  # 尝试转换
        return float(value)  # 返回内置浮点
    except (TypeError, ValueError):  # 不可转换
        return None  # 返回 None


def material(name, cs, vv, density, thickness=None):  # 构造单层材料的规范字典
    """打包一层材料为统一字典：name 层名、cs 剪切波速(m/s)、vv 泊松比、density 密度、thickness 厚度(None=由几何决定/半空间)。"""
    return {'name': str(name), 'cs': _f(cs), 'vv': _f(vv),  # 层名与波速、泊松比
            'density': _f(density), 'thickness': _f(thickness)}  # 密度与厚度（None 表示半空间或由几何决定）


# ==============================================================================
#  规范元数据构建
# ==============================================================================
def build_meta(model_script, incident_angle, geometry, bedrock, layers, mesh_size,
               model_type=None, record=None, extra=None):  # 构建规范工况元数据
    """汇总建模脚本求解后的全部参数为统一 schema 字典。

    model_script   : 建模脚本文件名（如 'VAB_oblique_TAF_multilayer_v3.py'）。
    incident_angle : SV 波入射角 θs（度）。
    geometry       : 几何参数字典（尽量含 i/total_L/left_flat/H_minus_h/h_over_H/bedrock_thickness/H/h/w_slope；
                     单层模型可只给 i/h/total_L 等可得项，缺失项自动留空）。
    bedrock        : 基岩材料字典（material(...) 产物）或 None（单层均质模型无基岩半空间区分时传 None）。
    layers         : 从上到下的有限层材料字典列表（material(...) 产物）；单层均质模型传 [均质层]。
    mesh_size      : 网格尺寸（m）。
    model_type     : 'single'/'double'/'multilayer'；为 None 时按有限层数自动判定。
    record         : 可选，本工况输入波记录名（一般留空，记录名以各 CSV 文件为准）。
    extra          : 可选，附加自定义键值（原样并入 meta['extra']）。
    返回：规范元数据 dict（可直接交 write_case_meta 落盘）。
    """
    layers = [l for l in (layers or []) if l]  # 规范化有限层列表（剔除空项）
    n_finite = len(layers)  # 有限层数量（不含基岩）
    has_bedrock = bedrock is not None  # 是否存在基岩半空间
    n_total = n_finite + (1 if has_bedrock else 0)  # 总介质层数（含基岩）
    if model_type is None:  # 未显式指定模型类型时按总介质层数自动判定
        model_type = 'single' if n_total <= 1 else ('double' if n_total == 2 else 'multilayer')  # 1层均质/2层双层/≥3多层
    geom = {k: _f(v) for k, v in (geometry or {}).items()}  # 规范化几何为 float 字典
    # 派生几何：缺 H/h/w_slope 时按输入项补算（与建模脚本 make_geometry 同口径）
    Hmh = geom.get('H_minus_h')  # 斜坡高度差 H-h
    hoH = geom.get('h_over_H')  # 深度比 h/H
    if geom.get('H') is None and Hmh is not None and hoH is not None and abs(1.0 - hoH) > 1e-9:  # 可补算 H
        geom['H'] = Hmh / (1.0 - hoH)  # 总覆盖厚度 H
    if geom.get('h') is None and geom.get('H') is not None and Hmh is not None:  # 可补算 h
        geom['h'] = geom['H'] - Hmh  # 下部覆盖高度 h
    if geom.get('w_slope') is None and Hmh is not None and geom.get('i'):  # 可补算坡面水平长度
        geom['w_slope'] = Hmh / math.tan(math.radians(geom['i'])) if geom['i'] not in (0, 90) else None  # 坡面宽度
    # 斜坡特征高度（a0 归一化用）：优先 H_minus_h，单层用 h
    slope_height = Hmh if Hmh is not None else geom.get('h')  # 斜坡特征高度
    # 各层波速口径
    vs_bedrock = bedrock['cs'] if has_bedrock else None  # 基岩 Vr
    vs_surface = layers[0]['cs'] if layers else vs_bedrock  # 最顶有限层 Vs1（无有限层退化为基岩）
    vs_cover = layers[-1]['cs'] if layers else vs_surface  # 最底覆盖层 Vs2（a0 用）
    # 抗阻/波速比（缺项留 None）
    vr_over_vs2 = (vs_bedrock / vs_cover) if (vs_bedrock and vs_cover) else None  # 基岩/覆盖层 Vr/Vs2
    vs1_over_vs2 = (vs_surface / vs_cover) if (vs_surface and vs_cover) else None  # 表层/覆盖层 Vs1/Vs2
    # a0 归一化基数：a0 = f_c(Hz) * a0_base，其中 a0_base = 2*(斜坡特征高度)/Vs2
    a0_base = (2.0 * slope_height / vs_cover) if (slope_height and vs_cover) else None  # 无量纲频率换算基数
    derived = {  # 派生量集中区
        'n_finite_layers': n_finite,  # 有限层数（不含基岩）
        'n_layers_total': n_finite + (1 if has_bedrock else 0),  # 总层数（含基岩）
        'vs_bedrock': _f(vs_bedrock),  # 基岩剪切波速 Vr
        'vs_surface': _f(vs_surface),  # 表层剪切波速 Vs1
        'vs_cover': _f(vs_cover),  # 覆盖层剪切波速 Vs2
        'vr_over_vs2': _f(vr_over_vs2),  # Vr/Vs2 抗阻比
        'vs1_over_vs2': _f(vs1_over_vs2),  # Vs1/Vs2 软硬比
        'slope_height': _f(slope_height),  # a0 归一化用斜坡特征高度
        'a0_base': _f(a0_base),  # a0 = f_c * a0_base
    }
    meta = {  # 组装规范元数据
        'schema_version': SCHEMA_VERSION,  # schema 版本
        'model_type': model_type,  # 模型类型 single/double/multilayer
        'model_script': str(model_script),  # 建模脚本文件名
        'incident_angle': _f(incident_angle),  # SV 入射角 θs（度）
        'mesh_size': _f(mesh_size),  # 网格尺寸（m）
        'geometry': geom,  # 几何参数（含派生）
        'bedrock': bedrock,  # 基岩材料字典或 None
        'layers': layers,  # 从上到下的有限层材料字典列表
        'derived': derived,  # 派生量
        'record': (str(record) if record is not None else None),  # 可选输入波记录名
        'extra': dict(extra) if extra else {},  # 附加自定义键值
    }
    return meta  # 返回规范元数据


# ==============================================================================
#  落盘与读取
# ==============================================================================
def write_case_meta(out_dir, meta, filename=META_FILENAME):  # 写出元数据 JSON
    """把规范元数据写入 out_dir/filename（UTF-8、缩进、保留中文）；若 meta 缺 folder 则补当前文件夹名。返回写出路径。"""
    out_dir = os.path.abspath(out_dir)  # 规范化输出目录
    meta = dict(meta)  # 拷贝避免污染入参
    meta.setdefault('folder', os.path.basename(out_dir.rstrip('/\\')))  # 补充工况文件夹名（来源标识）
    path = os.path.join(out_dir, filename)  # 目标路径
    with open(path, 'w', encoding='utf-8') as f:  # 打开文件写入
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_f)  # 序列化（default 兜底 numpy 标量）
    return path  # 返回写出路径


def read_case_meta(folder, filename=META_FILENAME):  # 读取某文件夹的元数据
    """读取 folder/filename 并返回 dict；不存在或解析失败返回 None。"""
    path = os.path.join(folder, filename)  # 元数据路径
    if not os.path.isfile(path):  # 文件不存在
        return None  # 返回空
    try:  # 尝试读取解析
        with open(path, 'r', encoding='utf-8') as f:  # 打开文件
            return json.load(f)  # 返回解析结果
    except Exception:  # 解析失败
        return None  # 返回空


# ==============================================================================
#  展平为 index.csv 的统一列
# ==============================================================================
def flatten_for_index(meta):  # 把嵌套 meta 展平为 index.csv 的一行（统一列）
    """返回扁平 dict，列名稳定，供 Collect 写入 index.csv 与 Plot 直接读取。

    关键列：model_type、model_script、incident_angle(θs)、mesh_size、slope_i(坡角)、
      total_L/left_flat/H_minus_h/h_over_H/bedrock_thickness/H/h/w_slope、
      n_finite_layers/n_layers_total、vs_bedrock/vs_surface/vs_cover、
      vr_over_vs2/vs1_over_vs2/slope_height/a0_base、layers_json(完整层信息)。
    """
    g = meta.get('geometry', {}) or {}  # 几何字典
    d = meta.get('derived', {}) or {}  # 派生字典
    flat = {  # 组装扁平列
        'model_type': meta.get('model_type'),  # 模型类型
        'model_script': meta.get('model_script'),  # 建模脚本
        'incident_angle': meta.get('incident_angle'),  # 入射角 θs
        'mesh_size': meta.get('mesh_size'),  # 网格尺寸
        'slope_i': g.get('i'),  # 坡角 i
        'total_L': g.get('total_L'),  # 模型总长
        'left_flat': g.get('left_flat'),  # 上平台长度
        'H_minus_h': g.get('H_minus_h'),  # 斜坡高度差
        'h_over_H': g.get('h_over_H'),  # 深度比
        'bedrock_thickness': g.get('bedrock_thickness'),  # 基岩厚度
        'H': g.get('H'),  # 总覆盖厚度
        'h': g.get('h'),  # 下部覆盖高度
        'w_slope': g.get('w_slope'),  # 坡面水平长度
        'n_finite_layers': d.get('n_finite_layers'),  # 有限层数
        'n_layers_total': d.get('n_layers_total'),  # 总层数
        'vs_bedrock': d.get('vs_bedrock'),  # 基岩 Vr
        'vs_surface': d.get('vs_surface'),  # 表层 Vs1
        'vs_cover': d.get('vs_cover'),  # 覆盖层 Vs2
        'vr_over_vs2': d.get('vr_over_vs2'),  # Vr/Vs2
        'vs1_over_vs2': d.get('vs1_over_vs2'),  # Vs1/Vs2
        'slope_height': d.get('slope_height'),  # a0 归一化高度
        'a0_base': d.get('a0_base'),  # a0 换算基数
        'layers_json': json.dumps(meta.get('layers', []), ensure_ascii=False),  # 完整层信息（紧凑 JSON 保真）
    }
    return flat  # 返回扁平列


INDEX_META_FIELDS = [  # index.csv 中由 meta 展平而来的列顺序（供 Collect 写表头）
    'model_type', 'model_script', 'incident_angle', 'mesh_size', 'slope_i',
    'total_L', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness', 'H', 'h', 'w_slope',
    'n_finite_layers', 'n_layers_total', 'vs_bedrock', 'vs_surface', 'vs_cover',
    'vr_over_vs2', 'vs1_over_vs2', 'slope_height', 'a0_base', 'layers_json',
]


# ==============================================================================
#  频率与 a0 解析（Plot 端用）
# ==============================================================================
def parse_freq_hz(record):  # 从记录名解析输入波主频 f_c(Hz)
    """匹配 record 中形如 '4Hz'/'ricker_4Hz' 的频率字段，返回浮点 Hz；无则返回 None。"""
    if record is None:  # 空记录
        return None  # 返回空
    m = re.search(r'(\d+(?:\.\d+)?)\s*Hz', str(record), re.IGNORECASE)  # 匹配数字+Hz
    return float(m.group(1)) if m else None  # 命中返回频率，否则 None


def freq_to_a0(freq_hz, a0_base):  # 由主频与 a0 基数算无量纲频率
    """a0 = f_c(Hz) * a0_base；任一为空返回 None。"""
    if freq_hz is None or a0_base is None:  # 缺输入
        return None  # 返回空
    return float(freq_hz) * float(a0_base)  # 返回 a0


def record_to_a0(record, a0_base):  # 直接由记录名与 a0 基数算 a0
    """组合 parse_freq_hz 与 freq_to_a0：从记录名解析主频再换算 a0；失败返回 None。"""
    return freq_to_a0(parse_freq_hz(record), a0_base)  # 返回 a0
