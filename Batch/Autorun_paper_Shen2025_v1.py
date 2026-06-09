# -*- coding: utf-8 -*-
"""批量跑【Shen 等 2025 论文全部 68 个唯一工况】的驱动脚本 v1。

论文：The combined amplification effects of topography and stratigraphy of layered
rock slopes under vertically and obliquely incident seismic waves（Shen 等，2025）。

工况体系（4 集，去重后恰好 68 个独立工况，与论文"68 个独立数据点"一致）：
  集 A：坡角 i∈{30,45,60}°，h/H=0.5，Vr/Vs=1.25，fc∈{2,4,6,8}Hz，θs∈{0,15}° → 24 工况
  集 B：i=30°，h/H∈{0.25,0.75}，Vr/Vs=1.25，fc∈{2,4,6,8}Hz，θs∈{0,15}° → 16 工况
        （h/H=0.5 已含于集 A，不重复）
  集 C：i=45°，h/H=0.5，Vr/Vs∈{2.5,5.0}，a0=2.0，θs∈{0,15}° → 4 工况
        （Vr/Vs=1.25 的 i=45 已含于集 A，不重复；C 的 fc 由 a0=2 各自推算）
  集 D：i=45°，h/H=0.5，Vr/Vs2=2.5(双层)，Vs1/Vs2∈{0.5,0.75,2.0}，
        h1/(H-h)∈{0.25,0.5,0.75,1.0}，a0=2.0→fc=4Hz，θs∈{0,15}° → 24 工况

【设计要点 — 每工况一个文件夹、单一 Ricker 频率】
  v3 建模脚本按工况注入 damping_cfg.fc 拟合 Rayleigh 阻尼，必须对应单一 fc；
  若一个文件夹塞多个频率，后面的工况会用错阻尼。因此本脚本"一工况=一fc=一文件夹"。
  已有的 Ricker 文件（2/4/6/8 Hz）覆盖全部工况，无需新建波形。

运行方式（求解输出建在仓库外 scratch 盘，避免污染 git）：
  abaqus cae noGUI=Batch/Autorun_paper_Shen2025_v1.py -- C:\\Abaqus\\Shen2025_paper
  或：python Batch/Autorun_paper_Shen2025_v1.py  C:\\Abaqus\\Shen2025_paper
  （后续）：python Postprocess/General/Collect_results_v2.py  C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_Multi_TAF_v2.py      C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_Fig15_compare_v2.py  C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_dist_param_v1.py     C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_peakTAF_v1.py        C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_PGA_box_v1.py        C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_FAS_spectrogram_v1.py C:\\Abaqus\\Shen2025_paper
           python Postprocess/Multi/Plot_seismograms_v1.py    C:\\Abaqus\\Shen2025_paper

【解释器注意】SCRIPT_SEQUENCE 的前两个脚本（建模/PGA 提取）需要 Abaqus Python；
  第三个（Compute_TAF）需要 pandas。若 Abaqus Python 无 pandas，可从 SCRIPT_SEQUENCE
  移除 Compute_TAF，汇总后单独用普通 Python 跑：
    python Postprocess/General/Compute_TAF_v1.py  <工况文件夹路径>
"""

import os  # 导入操作系统路径与目录模块
import json  # 导入 JSON 模块用于写出 case_config.json
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器
import re  # 导入正则模块（用于 _sanitize）
import concurrent.futures  # 导入并发模块（本脚本 MAX_WORKERS=1，但框架保留）


# ==============================================================================
#  全局配置（仅修改此处即可调整根目录/文件夹前缀/并发数/删除类型）
# ==============================================================================

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')  # 默认根目录（可被命令行参数覆盖）
FOLDER_PREFIX = "shen2025-"  # 工况文件夹统一前缀（与旧 multi- 前缀区分）
MAX_WORKERS = 1  # 并行处理文件夹数：每个 Abaqus 作业已用 8 CPU，设为 1 避免超订
CONFIG_FILENAME = "case_config.json"  # 注入给建模脚本的配置文件名

# 建模、提取、计算 TAF 三步脚本（按绝对路径，对齐目录整理后的当前位置）
_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # 仓库根
SCRIPT_SEQUENCE = [  # 每工况文件夹内按序执行的脚本
    os.path.join(_REPO, r"Modeling\Multi\VAB_oblique_TAF_multilayer_v3.py"),  # 建模（v3：配置注入 + 材料阻尼 + 写 case_meta.json）
    os.path.join(_REPO, r"Postprocess\General\Postprocess_PGA_v2.py"),  # PGA 提取（需 Abaqus Python + odbAccess）
    os.path.join(_REPO, r"Postprocess\General\Compute_TAF_v1.py"),  # TAF 计算（需 pandas/numpy，详见模块顶部注释）
]

# Ricker 输入波文件夹（已有 2/4/6/8 Hz，覆盖全部 68 工况）
_WAVE_DIR = os.path.join(_REPO, r"Wave\Impulse\Acceleration")  # Ricker 波文件夹

# 每次 Abaqus 作业跑完后永久删除的中间文件类型（不放回收站；快照图暂不做，故 .odb 也删以节省磁盘）
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 中间文件扩展名列表


# ==============================================================================
#  工况参数辅助函数
# ==============================================================================

def _overlying_layer(vr):  # 构建覆盖层材料配置（双层模型的唯一有限层）
    """vr = Vr/Vs（基岩波速 / 该层波速），论文基岩 Vr=2000 m/s；不给 thickness 表示厚度由几何决定。"""
    return {'name': 'overlying', 'velocity_ratio': vr, 'poisson_ratio': 0.3, 'density': 2500}  # 覆盖层材料字典


def _surface_layer(vr, thickness):  # 构建表层材料配置（三层模型的顶层）
    """vr = Vr/Vs1，thickness 为该层固定厚度 h1（m）。"""
    return {'name': 'surface', 'velocity_ratio': vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': thickness}  # 表层材料字典（含厚度）


def _ricker_path(fc):  # 根据主频 fc 返回对应 Ricker 文件路径
    """fc 必须为已有文件的频率（2/4/6/8 Hz）；否则返回路径不存在，build_source_files 会报错。"""
    fc_str = _fmt_num(fc)  # 格式化频率（去掉无意义的 .0）
    return os.path.join(_WAVE_DIR, "ricker_wavelet_{}Hz.txt".format(fc_str))  # 拼接完整路径


# ==============================================================================
#  68 工况参数表（程序化生成，按 config 签名自动去重）
# ==============================================================================
#
#  每个工况 dict 结构：
#    {"fc": float,       ← 输入波主频（用于选 Ricker 文件 + 注入 damping_cfg.fc）
#     "config": {        ← 注入给建模脚本的完整 case_config（深合并到 v3 默认配置）
#       "material_cfg": {"angle": θs, "layers": [...]},
#       "geometry_cfg": {"i": i, "h_over_H": r},
#       "damping_cfg":  {"fc": fc}   ← 显式给定主频，Rayleigh 阻尼在此 fc 精确拟合
#     }}
#
# ==============================================================================

def _build_cases():  # 程序化构建全部 68 工况（按 A/B/C/D 四集组织）
    """返回去重后的工况列表；每项含 fc 与完整 config。"""
    cases = []  # 全部工况收集器

    # ---- 集 A：坡角 i 的影响 (Section 3.1) ----
    # 固定：h/H=0.5，Vr/Vs=1.25(Vs=1600)；变量：i∈{30,45,60}，fc∈{2,4,6,8}Hz，θs∈{0,15}°
    for i_deg in [30.0, 45.0, 60.0]:  # 遍历坡角
        for angle in [0, 15]:  # 遍历入射角
            for fc in [2.0, 4.0, 6.0, 8.0]:  # 遍历无量纲频率对应的中心频率
                cases.append({  # 添加一个工况
                    "fc": fc,  # 输入波主频
                    "config": {  # 注入配置
                        "material_cfg": {"angle": angle, "layers": [_overlying_layer(1.25)]},  # 双层(Vr/Vs=1.25)，入射角
                        "geometry_cfg": {"i": i_deg, "h_over_H": 0.5},  # 坡角与厚度比
                        "damping_cfg": {"fc": fc},  # 显式主频供 Rayleigh 阻尼拟合
                    },
                    "_set": "A",  # 工况集标签（仅供调试，不进入文件夹名）
                })  # 集 A 工况

    # ---- 集 B：覆盖层厚度比 h/H 的影响 (Section 3.2) ----
    # 固定：i=30°，Vr/Vs=1.25；变量：h/H∈{0.25,0.75}（0.5 已含于集 A）
    for h_over_H in [0.25, 0.75]:  # 遍历厚度比（不含 0.5）
        for angle in [0, 15]:  # 遍历入射角
            for fc in [2.0, 4.0, 6.0, 8.0]:  # 遍历频率
                cases.append({  # 添加一个工况
                    "fc": fc,  # 输入波主频
                    "config": {  # 注入配置
                        "material_cfg": {"angle": angle, "layers": [_overlying_layer(1.25)]},  # 双层(Vr/Vs=1.25)
                        "geometry_cfg": {"i": 30.0, "h_over_H": h_over_H},  # 固定坡角 30°，变厚度比
                        "damping_cfg": {"fc": fc},  # 显式主频
                    },
                    "_set": "B",  # 工况集标签
                })  # 集 B 工况

    # ---- 集 C：基岩-覆盖层阻抗比 Vr/Vs 的影响 (Section 3.3) ----
    # 固定：i=45°，h/H=0.5，a0=2.0；变量：Vr/Vs∈{2.5,5.0}，θs∈{0,15}°
    # Vr/Vs=1.25 对应 i=45 工况已在集 A（A12/A16），不重复
    # fc 由 a0=2.0 推算：fc = a0*Vs/(2*(H-h)) = 2.0*Vs/400
    #   Vr/Vs=2.5 → Vs=800 → fc=4 Hz；Vr/Vs=5.0 → Vs=400 → fc=2 Hz
    for vr, fc in [(2.5, 4.0), (5.0, 2.0)]:  # 遍历阻抗比及其对应频率
        for angle in [0, 15]:  # 遍历入射角
            cases.append({  # 添加一个工况
                "fc": fc,  # 输入波主频（由 a0=2.0 推算）
                "config": {  # 注入配置
                    "material_cfg": {"angle": angle, "layers": [_overlying_layer(vr)]},  # 双层(指定 Vr/Vs)
                    "geometry_cfg": {"i": 45.0, "h_over_H": 0.5},  # 固定坡角与厚度比
                    "damping_cfg": {"fc": fc},  # 显式主频
                },
                "_set": "C",  # 工况集标签
            })  # 集 C 工况

    # ---- 集 D：双层覆盖层地表层参数的联合影响 (Section 3.4) ----
    # 固定：i=45°，h/H=0.5，Vr/Vs2=2.5(覆盖层)，a0=2.0→fc=4Hz；
    # 变量：Vs1/Vs2∈{0.5,0.75,2.0} × h1/(H-h)∈{0.25,0.5,0.75,1.0} × θs∈{0,15}°
    # velocity_ratio = Vr/Vs（Vr=2000 m/s 基岩）：
    #   Vs1=400(soft)   → vr=5.0；Vs1=600(medium) → vr=10/3≈3.333；Vs1=1600(hard) → vr=1.25
    # H-h=200 m；h1/(H-h)∈{0.25,0.5,0.75,1.0} → h1∈{50,100,150,200} m
    _D_SURFACE_VRS = [5.0, 10.0 / 3.0, 1.25]  # 表层 velocity_ratio（从软到硬：Vs1/Vs2=0.5/0.75/2.0）
    _D_H1_VALUES = [50.0, 100.0, 150.0, 200.0]  # 表层实际厚度 h1（m），对应 h1/(H-h)=0.25/0.5/0.75/1.0
    for surf_vr in _D_SURFACE_VRS:  # 遍历表层波速比
        for h1 in _D_H1_VALUES:  # 遍历表层厚度
            for angle in [0, 15]:  # 遍历入射角
                cases.append({  # 添加一个工况
                    "fc": 4.0,  # 固定 fc=4 Hz（a0=2.0，Vs2=800 m/s，H-h=200 m）
                    "config": {  # 注入配置
                        "material_cfg": {  # 材料配置
                            "angle": angle,  # 入射角
                            "layers": [  # 三层：表层 + 覆盖层
                                _surface_layer(surf_vr, h1),  # 顶部表层（厚度固定）
                                _overlying_layer(2.5),  # 下部覆盖层(Vr/Vs2=2.5)
                            ],
                        },
                        "geometry_cfg": {"i": 45.0, "h_over_H": 0.5},  # 固定坡角与厚度比
                        "damping_cfg": {"fc": 4.0},  # 显式主频
                    },
                    "_set": "D",  # 工况集标签
                })  # 集 D 工况

    return cases  # 返回全部工况


PARAMETER_CASES = _build_cases()  # 调用构建函数，生成全部工况列表


# ==============================================================================
#  公用辅助函数（源自 Autorun_TAF_multilayer_v2，保持原逻辑不变）
# ==============================================================================

def build_source_files(static_source_paths, script_sequence):  # 构建源文件映射并检测缺失/重名
    """返回 (source_files 映射, missing 列表, duplicate_names 列表)。"""
    source_files = {}  # 初始化源文件映射
    missing = []  # 初始化缺失列表
    duplicate_names = []  # 初始化重名冲突列表
    for source_path in list(static_source_paths) + list(script_sequence):  # 合并静态与脚本
        target_name = os.path.basename(source_path)  # 目标文件名
        if not os.path.isfile(source_path):  # 源文件不存在
            missing.append((target_name, source_path)); continue  # 记录缺失并跳过
        if target_name in source_files and source_files[target_name] != source_path:  # 同名不同源冲突
            duplicate_names.append((target_name, source_path, source_files[target_name])); continue  # 记录冲突并跳过
        source_files[target_name] = source_path  # 写入映射
    return source_files, missing, duplicate_names  # 返回结果


def _fmt_num(v):  # 数值转简洁字符串（去掉无意义的 .0）
    """45.0→'45'，1.25→'1.25'，3.33333→'3.33333'；非数值原样返回。"""
    try:  # 尝试按数值处理
        f = float(v)  # 转浮点
        return str(int(f)) if f == int(f) else ('%g' % f)  # 整数去小数点，否则用 %g 紧凑表示
    except (TypeError, ValueError):  # 非数值
        return str(v)  # 原样转字符串


def _sanitize(text):  # 规范化为合法文件夹名片段
    """非 [0-9A-Za-z._-] 的字符替换为连字符，去除两端连字符。"""
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-") or "x"  # 清洗并兜底


def name_from_config(config):  # 由注入配置自动生成可读唯一的工况名后缀
    """编码 config 中的层结构/入射角/几何/网格；不包含 damping_cfg，fc 由调用方追加。"""
    mat = config.get("material_cfg") or {}  # 材料覆盖
    geo = config.get("geometry_cfg") or {}  # 几何覆盖
    tokens = []  # 名称片段
    layers = mat.get("layers")  # 有限层列表
    if isinstance(layers, list):  # 编码层结构
        segs = []  # 各层片段
        for L in layers:  # 自上而下遍历
            seg = "vr" + _fmt_num(L.get("velocity_ratio"))  # 波速比片段
            if L.get("thickness") is not None:  # 有固定厚度则附加
                seg += "t" + _fmt_num(L["thickness"])  # 厚度片段
            segs.append(seg)  # 收集
        tokens.append("L%d_%s" % (len(layers), "-".join(segs)))  # L{层数}_{各层}
    if "angle" in mat:  # 入射角
        tokens.append("a" + _fmt_num(mat["angle"]))  # a{angle}
    for key, pre in (("i", "i"), ("H_minus_h", "H"), ("h_over_H", "hoH"),  # 几何关键键 → 短前缀
                     ("bedrock_thickness", "br"), ("total_L", "L"), ("left_flat", "lf")):
        if key in geo:  # 覆盖了该几何键
            tokens.append(pre + _fmt_num(geo[key]))  # 追加片段
    if config.get("mesh_size") is not None:  # 网格尺寸
        tokens.append("m" + _fmt_num(config["mesh_size"]))  # m{mesh}
    for scope, skip in ((mat, {"angle", "layers"}), (geo, {"i", "H_minus_h", "h_over_H",
                                                             "bedrock_thickness", "total_L", "left_flat"})):
        for k, v in scope.items():  # 遍历未专门编码的标量键
            if k in skip or isinstance(v, (dict, list)):  # 跳过已编码与嵌套结构
                continue  # 继续
            tokens.append("%s%s" % (_sanitize(k), _fmt_num(v)))  # 通用键值片段
    return "-".join(tokens) if tokens else "default"  # 拼接（全空则用 default）


def build_folder_name(case):  # 生成工况文件夹名（在 name_from_config 基础上附加 fc 标识）
    """格式：{FOLDER_PREFIX}{config编码}-fc{fc}Hz；fc 在此显式追加，确保同几何不同频率名称唯一。"""
    tag = case.get("name") or case.get("folder_tag")  # 可选手填标签（向后兼容）
    if not tag:  # 未手填 → 自动取名
        tag = name_from_config(case.get("config") or {})  # 由配置生成基础名
        fc = case.get("fc")  # 该工况主频
        if fc is not None:  # 追加频率标识（区分同几何不同频率的工况）
            tag = "{}-fc{}Hz".format(tag, _fmt_num(fc))  # 格式：原名-fc{N}Hz
    return "{}{}".format(FOLDER_PREFIX, _sanitize(tag))  # 前缀 + 规范化标签


def create_and_fill_folder(folder_path, source_files, config):  # 创建并填充单个工况目录
    """新建目录、拷入源文件（脚本 + 单一 Ricker）、写出 case_config.json。"""
    os.makedirs(folder_path, exist_ok=True)  # 创建目录
    for target_name, src_path in source_files.items():  # 遍历所有源文件
        shutil.copy2(src_path, os.path.join(folder_path, target_name))  # 拷入并保留元数据
    _config_to_write = {k: v for k, v in (config or {}).items() if not k.startswith("_")}  # 去掉 _set 等调试键
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:  # 写出注入配置
        json.dump(_config_to_write, f, ensure_ascii=False, indent=2)  # 序列化 config


def run_scripts_in_folder(folder_path, run_order):  # 在目录内按顺序执行脚本
    """逐个用当前 Python 解释器在目录内执行；任一失败返回 False。"""
    for script_name in run_order:  # 按顺序遍历
        script_path = os.path.join(folder_path, script_name)  # 脚本完整路径
        if not os.path.isfile(script_path):  # 脚本缺失
            print("错误：脚本不存在 -> {}".format(script_path)); return False  # 报错并失败
        print("开始执行：{}".format(script_path))  # 开始日志
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)  # 在目录内执行
        if result.returncode != 0:  # 执行失败
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode)); return False  # 报错并失败
        print("完成执行：{}".format(script_name))  # 完成日志
    return True  # 全部成功


def delete_files_by_type(folder_path, file_types):  # 永久删除目录下指定类型的文件
    """按扩展名永久删除中间文件（不放回收站）；file_types 为空则跳过。"""
    if not file_types:  # 未指定类型
        return  # 直接返回
    normalized = {t.lower() if t.startswith(".") else "." + t.lower() for t in file_types}  # 规范扩展名集合
    deleted = {t: 0 for t in normalized}  # 各类型删除计数
    failed = []  # 失败列表
    for name in sorted(os.listdir(folder_path)):  # 遍历目录
        fp = os.path.join(folder_path, name)  # 完整路径
        if not os.path.isfile(fp):  # 跳过非文件
            continue  # 继续
        ext = os.path.splitext(name)[1].lower()  # 扩展名
        if ext not in normalized:  # 非目标类型
            continue  # 跳过
        try:  # 尝试删除
            os.remove(fp)  # 永久删除
            deleted[ext] += 1  # 计数加一
        except OSError as exc:  # 系统异常
            failed.append((fp, str(exc)))  # 记录失败
    for ext, n in sorted(deleted.items()):  # 汇总
        if n > 0:  # 只打印有删除的类型
            print("已删除 {} 文件数量：{}".format(ext, n))  # 打印数量
    if failed:  # 有失败
        print("警告：以下文件删除失败：")  # 警告标题
        for fp, err in failed:  # 遍历失败
            print("  - {} -> {}".format(fp, err))  # 打印明细


# ==============================================================================
#  主控制流程
# ==============================================================================

def main():  # 主入口：组织工况 → 检查 → 并行建模/后处理 → 汇总
    """读命令行参数获取 scratch 根目录，验证所有源文件存在，按 PARAMETER_CASES 批量处理。"""
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR  # scratch 根目录
    root_dir = os.path.abspath(root_dir)  # 转绝对路径
    types_to_delete = list(DELETE_FILE_TYPES)  # 待删除的文件类型副本
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]  # 执行顺序（文件名）

    print("目标根目录：{}".format(root_dir))  # 打印根目录
    print("要直接删除的文件类型：{}".format(types_to_delete if types_to_delete else "无"))  # 打印待删除类型
    print("工况总数：{}".format(len(PARAMETER_CASES)))  # 打印工况数

    # 预检：各集脚本文件存在（Ricker 存在性在每工况 process_folder 中检查）
    for p in SCRIPT_SEQUENCE:  # 逐脚本检查
        if not os.path.isfile(p):  # 不存在
            print("错误：脚本缺失 -> {}".format(p)); sys.exit(1)  # 报错退出

    # 规划工况：文件夹名去重
    folder_plan = []  # 文件夹计划
    seen = {}  # 文件夹名 → 工况序号（用于报告重复）
    set_counts = {}  # 各集工况计数
    for idx, case in enumerate(PARAMETER_CASES, start=1):  # 遍历工况
        if not isinstance(case, dict) or "config" not in case:  # 校验结构
            print("错误：第 {} 组工况缺少 config。".format(idx)); sys.exit(1)  # 报错退出
        folder_name = build_folder_name(case)  # 生成文件夹名
        if folder_name in seen:  # 重名
            print("错误：工况 #{} 与工况 #{} 生成了相同的文件夹名 -> {}".format(
                idx, seen[folder_name], folder_name)); sys.exit(1)  # 报错退出
        seen[folder_name] = idx  # 记录已用名
        folder_plan.append((folder_name, case["config"], case.get("fc")))  # 记录(文件夹,配置,fc)
        s = case.get("_set", "?")  # 工况集标签
        set_counts[s] = set_counts.get(s, 0) + 1  # 统计各集数量

    # 打印各集工况数汇总（用于人工核对）
    for s in sorted(set_counts):  # 按集名排序
        print("  集 {}：{} 个工况".format(s, set_counts[s]))  # 打印各集数量
    print("  合计：{} 个唯一工况".format(len(folder_plan)))  # 打印总数

    os.makedirs(root_dir, exist_ok=True)  # 确保根目录存在
    failed_folders = []  # 失败目录列表

    def process_folder(item):  # 处理单个工况文件夹的工作函数
        """item=(folder_name, config, fc)；按 fc 选 Ricker、写 config、执行脚本序列；返回(path,ok)。"""
        folder_name, config, fc = item  # 解包
        folder_path = os.path.join(root_dir, folder_name)  # 文件夹完整路径
        print("\n==============================")  # 分隔符
        print("开始处理文件夹：{} (fc={} Hz, 集={})".format(
            folder_path, fc, config.get("_set", "?")))  # 当前文件夹与工况信息

        # 为该工况构建源文件映射：脚本 + 单一 Ricker
        ricker_path = _ricker_path(fc)  # 该工况对应的 Ricker 波文件路径
        source_files, missing, dups = build_source_files([ricker_path], SCRIPT_SEQUENCE)  # 构建映射
        if missing:  # 缺失文件
            print("错误：工况 {} 缺失源文件：{}".format(folder_name, [m[1] for m in missing]))  # 报错
            return folder_path, False  # 返回失败
        if dups:  # 重名冲突
            print("错误：工况 {} 文件重名：{}".format(folder_name, dups))  # 报错
            return folder_path, False  # 返回失败

        create_and_fill_folder(folder_path, source_files, config)  # 创建目录并注入配置
        ok = run_scripts_in_folder(folder_path, run_order)  # 顺序执行脚本
        if types_to_delete:  # 需清理中间文件
            delete_files_by_type(folder_path, types_to_delete)  # 永久删除中间文件
        else:  # 不清理
            print("已跳过文件删除（未指定要删除的文件类型）。")  # 提示
        return folder_path, ok  # 返回结果

    print("\n开始并行批处理，最大并发任务数：{}".format(MAX_WORKERS))  # 启动提示
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 线程池
        for folder_path, ok in executor.map(process_folder, folder_plan):  # 并行执行并收集结果
            if not ok:  # 失败
                failed_folders.append(folder_path)  # 记录失败

    print("\n==============================")  # 总结分隔符
    if failed_folders:  # 有失败
        print("批处理结束：存在失败文件夹（{}个）。".format(len(failed_folders)))  # 失败标题
        for path in failed_folders:  # 遍历失败
            print("  - {}".format(path))  # 打印路径
        sys.exit(2)  # 异常退出
    print("批处理结束：全部 {} 个工况文件夹处理完成。".format(len(folder_plan)))  # 全部成功
    print("\n提示：随后按顺序运行：")  # 后续步骤标题
    print("  1. python Postprocess/General/Collect_results_v2.py  {}".format(root_dir))  # 步骤1
    print("  2. python Postprocess/Multi/Plot_Multi_TAF_v2.py     {}".format(root_dir))  # 步骤2（Fig8）
    print("  3. python Postprocess/Multi/Plot_Fig15_compare_v2.py {}".format(root_dir))  # 步骤3（Fig15）
    print("  4. python Postprocess/Multi/Plot_dist_param_v1.py    {}".format(root_dir))  # 步骤4（Fig11/13）
    print("  5. python Postprocess/Multi/Plot_peakTAF_v1.py       {}".format(root_dir))  # 步骤5（Fig9/12/16）
    print("  6. python Postprocess/Multi/Plot_PGA_box_v1.py       {}".format(root_dir))  # 步骤6（Fig20）
    print("  7. python Postprocess/Multi/Plot_FAS_spectrogram_v1.py {}".format(root_dir))  # 步骤7（Fig17）
    print("  8. python Postprocess/Multi/Plot_seismograms_v1.py   {}".format(root_dir))  # 步骤8（Fig10c/14c）


if __name__ == "__main__":  # 主入口判断
    main()  # 运行主流程
