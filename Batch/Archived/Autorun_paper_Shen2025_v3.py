# -*- coding: utf-8 -*-
"""
批量跑【Shen 等 2025 论文全部 68 个唯一工况】的驱动脚本 v3。

工况体系（4 集，去重后恰好 68 个独立工况，与论文"68 个独立数据点"一致）：
  集 A：坡角 i∈{30,45,60}°，h/H=0.5，Vr/Vs=1.25，fc∈{2,4,6,8}Hz，θs∈{0,15}° → 24 工况
  集 B：i=30°，h/H∈{0.25,0.75}，Vr/Vs=1.25，fc∈{2,4,6,8}Hz，θs∈{0,15}° → 16 工况
        （h/H=0.5 已含于集 A，不重复）
  集 C：i=45°，h/H=0.5，Vr/Vs∈{2.5,5.0}，a0=2.0，θs∈{0,15}° → 4 工况
        （Vr/Vs=1.25 的 i=45 已含于集 A，不重复；C 的 fc 由 a0=2 各自推算）
  集 D：i=45°，h/H=0.5，Vr/Vs2=2.5(双层)，Vs1/Vs2∈{0.5,0.75,2.0}，
        h1/(H-h)∈{0.25,0.5,0.75,1.0}，a0=2.0→fc=4Hz，θs∈{0,15}° → 24 工况

【v3 新增特性 — 进度跟踪与断点续跑】
  ● 通过 progress.json 文件持久化记录每个工况的完成状态（pending/running/done/failed）。
  ● 启动时自动检测进度文件：
    - 状态为 done 的工况直接跳过；
    - 状态为 running 的工况视为上次被中断，删除其文件夹后重新执行；
    - 状态为 failed 的工况同样删除文件夹后重新执行。
  ● 实时显示进度条、已用时间、预计剩余时间（ETA）。

【设计要点 — 每工况一个文件夹、单一 Ricker 频率】
  v3 建模脚本按工况注入 damping_cfg.fc 拟合 Rayleigh 阻尼，必须对应单一 fc；
  若一个文件夹塞多个频率，后面的工况会用错阻尼。因此本脚本"一工况=一fc=一文件夹"。
  已有的 Ricker 文件（2/4/6/8 Hz）覆盖全部工况，无需新建波形。

【解释器注意】SCRIPT_SEQUENCE 的前两个脚本（建模/PGA 提取）需要 Abaqus Python；
  第三个（Compute_TAF）需要 pandas。若 Abaqus Python 无 pandas，可从 SCRIPT_SEQUENCE
  移除 Compute_TAF，汇总后单独用普通 Python 跑
"""

import os  # 导入操作系统路径与目录模块
import json  # 导入 JSON 模块用于写出 case_config.json 及 progress.json
import shutil  # 导入文件复制/删除模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器
import re  # 导入正则模块（用于 _sanitize）
import time  # 导入时间模块（用于进度计时与 ETA 计算）
import datetime  # 导入日期时间模块（用于格式化时间戳）
import concurrent.futures  # 导入并发模块（本脚本 MAX_WORKERS=1，但框架保留）


# ==============================================================================
#  全局配置（仅修改此处即可调整根目录/文件夹前缀/并发数/删除类型）
# ==============================================================================

FOLDER_PREFIX = "shen2025-"  # 工况文件夹统一前缀（与旧 multi- 前缀区分）
MAX_WORKERS = 2  # 并行处理文件夹数：每个 Abaqus 作业已用 8 CPU，设为 1 避免超订
CONFIG_FILENAME = "case_config.json"  # 注入给建模脚本的配置文件名
PROGRESS_FILENAME = "progress.json"  # 进度持久化文件名（v3 新增）

# 建模、提取、计算 TAF 三步脚本（直接指定绝对路径）
SCRIPT_SEQUENCE = [  # 每工况文件夹内按序执行的脚本
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_TAF_multilayer_v8.py",  # 建模（v6：fd 频域精确自由场引擎）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算（v2：解析自由场分母，论文式(5) 口径）
]

# Ricker 输入波文件夹（已有 2/4/6/8 Hz，覆盖全部 68 工况）
_WAVE_DIR = r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration"  # Ricker 波文件夹

# 每次 Abaqus 作业跑完后永久删除的中间文件类型（不放回收站；快照图暂不做，故 .odb 也删以节省磁盘）
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 中间文件扩展名列表

# 全部工况求解完成后自动执行的后处理脚本（直接指定绝对路径）
POST_SCRIPT_SEQUENCE = [  # 汇总与跨工况出图脚本
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py",  # 汇总各工况 case_meta.json 到 results/index.csv
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_Multi_TAF_v3.py",  # 图8 排版
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_Fig15_compare_v3.py",  # 图15 对比
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_dist_param_v1.py",  # 图11/13 分布参数
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_peakTAF_v1.py",  # 图9/12/16 峰值TAF
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_PGA_box_v1.py",  # 图20 PGA盒须
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_FAS_spectrogram_v1.py",  # 图17 FAS谱图
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_seismograms_v1.py",  # 图10c/14c 地震波
]


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
#  进度持久化管理（v3 新增）
# ==============================================================================

# 工况状态枚举常量
STATUS_PENDING = "pending"    # 待处理（尚未开始）
STATUS_RUNNING = "running"    # 运行中（若启动时发现此状态则视为上次被中断）
STATUS_DONE = "done"          # 已完成
STATUS_FAILED = "failed"      # 已失败（脚本返回非零退出码）


def _load_progress(progress_path):  # 从磁盘加载进度文件
    """加载 progress.json；文件不存在或损坏则返回空字典。"""
    if not os.path.isfile(progress_path):  # 文件不存在
        return {}  # 返回空（首次运行）
    try:  # 尝试读取
        with open(progress_path, 'r', encoding='utf-8') as f:  # 打开文件
            data = json.load(f)  # 解析 JSON
        if isinstance(data, dict):  # 校验类型
            return data  # 返回数据
        print("警告：progress.json 格式异常（非字典），将重新初始化。")  # 格式异常
        return {}  # 返回空
    except (json.JSONDecodeError, IOError) as exc:  # 解析或 IO 异常
        print("警告：读取 progress.json 失败 -> {}，将重新初始化。".format(exc))  # 报警告
        return {}  # 返回空


def _save_progress(progress_path, progress_data):  # 将进度数据写入磁盘
    """原子写：先写临时文件再重命名，避免写入过程中断导致文件损坏。"""
    tmp_path = progress_path + ".tmp"  # 临时文件路径
    with open(tmp_path, 'w', encoding='utf-8') as f:  # 写临时文件
        json.dump(progress_data, f, ensure_ascii=False, indent=2)  # 序列化 JSON
        f.flush()  # 刷新缓冲区
        os.fsync(f.fileno())  # 强制写入磁盘
    # Windows 上 os.rename 在目标存在时会报错，改用 os.replace（Python 3.3+）
    os.replace(tmp_path, progress_path)  # 原子替换（覆盖旧文件）


def _update_status(progress_path, progress_data, folder_name, status, extra=None):  # 更新单个工况状态并持久化
    """更新指定工况状态并立即写盘；extra 为可选附加信息字典（如时间戳）。"""
    entry = progress_data.get(folder_name, {})  # 获取已有条目（或创建新的）
    entry["status"] = status  # 设置状态
    entry["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 更新时间戳
    if extra:  # 有附加信息
        entry.update(extra)  # 合并
    progress_data[folder_name] = entry  # 写回字典
    _save_progress(progress_path, progress_data)  # 持久化到磁盘


# ==============================================================================
#  进度显示辅助函数（v3 新增）
# ==============================================================================

def _format_duration(seconds):  # 将秒数格式化为可读时间字符串
    """将秒数格式化为 'Xh Ym Zs' 或 'Ym Zs' 或 'Zs' 的可读格式。"""
    seconds = int(seconds)  # 取整
    if seconds < 0:  # 防御负数
        return "0s"  # 返回零
    h = seconds // 3600  # 小时
    m = (seconds % 3600) // 60  # 分钟
    s = seconds % 60  # 秒
    if h > 0:  # 有小时
        return "{}h {:02d}m {:02d}s".format(h, m, s)  # 时分秒
    if m > 0:  # 有分钟
        return "{}m {:02d}s".format(m, s)  # 分秒
    return "{}s".format(s)  # 仅秒


def _print_progress_bar(current, total, elapsed, bar_width=40):  # 打印进度条到控制台
    """打印一行进度条，含百分比、已用时间和 ETA。"""
    if total <= 0:  # 无工况
        return  # 跳过
    ratio = current / total  # 完成比例
    filled = int(bar_width * ratio)  # 已填充的字符数
    bar = "█" * filled + "░" * (bar_width - filled)  # 进度条字符串
    percent = ratio * 100  # 百分比

    elapsed_str = _format_duration(elapsed)  # 格式化已用时间
    if current > 0:  # 已完成至少一个工况（可计算 ETA）
        eta_seconds = (elapsed / current) * (total - current)  # 预估剩余时间
        eta_str = _format_duration(eta_seconds)  # 格式化 ETA
    else:  # 尚无完成工况
        eta_str = "计算中..."  # 无法估算

    print("\r  进度 |{}| {}/{} ({:.1f}%)  已用: {}  ETA: {}    ".format(
        bar, current, total, percent, elapsed_str, eta_str), end="")  # 覆盖同一行
    if current >= total:  # 全部完成
        print()  # 换行


def _print_case_header(idx, total, folder_name, fc, set_label, elapsed):  # 打印工况开始的标题块
    """打印当前工况的详细标题信息，含序号、进度、工况参数。"""
    print("\n" + "=" * 70)  # 分隔线
    print("  工况 [{}/{}]  集={}  fc={}Hz".format(idx, total, set_label, fc))  # 序号与参数
    print("  文件夹: {}".format(folder_name))  # 文件夹名
    print("  开始时间: {}  (累计已用: {})".format(
        datetime.datetime.now().strftime("%H:%M:%S"),  # 当前时间
        _format_duration(elapsed)))  # 累计已用时间
    print("=" * 70)  # 分隔线


# ==============================================================================
#  公用辅助函数（源自 Autorun_paper_Shen2025_v2，保持原逻辑不变）
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
        print("  ▶ 执行：{}".format(script_name))  # 开始日志（缩进以区分工况标题）
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)  # 在目录内执行
        if result.returncode != 0:  # 执行失败
            print("  ✗ 失败：{} (返回码={})".format(script_name, result.returncode)); return False  # 报错并失败
        print("  ✓ 完成：{}".format(script_name))  # 完成日志
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
            print("  已删除 {} 文件数量：{}".format(ext, n))  # 打印数量
    if failed:  # 有失败
        print("  警告：以下文件删除失败：")  # 警告标题
        for fp, err in failed:  # 遍历失败
            print("    - {} -> {}".format(fp, err))  # 打印明细


def _remove_folder(folder_path):  # 完整删除工况文件夹（用于中断恢复）
    """安全删除整个工况文件夹；若删除失败打印警告但不中止流程。"""
    if not os.path.isdir(folder_path):  # 文件夹不存在
        return  # 无需操作
    try:  # 尝试递归删除
        shutil.rmtree(folder_path)  # 递归删除整个目录树
        print("  ♻ 已删除被中断的文件夹：{}".format(os.path.basename(folder_path)))  # 确认删除
    except OSError as exc:  # 系统异常
        print("  警告：删除文件夹失败 -> {} ({})".format(folder_path, exc))  # 报警告


# ==============================================================================
#  主控制流程
# ==============================================================================

def main():  # 主入口：组织工况 → 检查进度 → 断点续跑 → 并行建模/后处理 → 汇总
    """读命令行参数获取 scratch 根目录，验证所有源文件存在，按 PARAMETER_CASES 批量处理。
    支持从 progress.json 恢复进度，跳过已完成工况，删除并重跑被中断工况。"""
    root_dir = os.path.dirname(os.path.abspath(__file__))  # 固定为当前脚本所在目录
    types_to_delete = list(DELETE_FILE_TYPES)  # 待删除的文件类型副本
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]  # 执行顺序（文件名）
    progress_path = os.path.join(root_dir, PROGRESS_FILENAME)  # 进度文件完整路径

    print("=" * 70)  # 总标题分隔线
    print("  Shen2025 批量驱动脚本 v3 — 断点续跑模式")  # 脚本标题
    print("  启动时间: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))  # 启动时间
    print("=" * 70)  # 分隔线
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
        _set_label = case.get("_set", "?")  # 工况集标签
        folder_plan.append((folder_name, case["config"], case.get("fc"), _set_label))  # 记录(文件夹,配置,fc,集标签)
        s = _set_label  # 工况集标签
        set_counts[s] = set_counts.get(s, 0) + 1  # 统计各集数量

    # 打印各集工况数汇总（用于人工核对）
    for s in sorted(set_counts):  # 按集名排序
        print("  集 {}：{} 个工况".format(s, set_counts[s]))  # 打印各集数量
    print("  合计：{} 个唯一工况".format(len(folder_plan)))  # 打印总数

    os.makedirs(root_dir, exist_ok=True)  # 确保根目录存在

    # ------------------------------------------------------------------
    #  进度恢复逻辑（v3 新增）
    # ------------------------------------------------------------------
    progress_data = _load_progress(progress_path)  # 加载已有进度
    n_skipped = 0  # 跳过的已完成工况数
    n_interrupted = 0  # 被中断需重来的工况数
    n_failed_redo = 0  # 上次失败需重来的工况数

    # 预扫描：统计各状态并处理被中断/失败的工况文件夹
    for folder_name, config, fc, _set in folder_plan:  # 遍历所有计划工况
        entry = progress_data.get(folder_name, {})  # 获取进度条目
        status = entry.get("status", STATUS_PENDING)  # 获取状态（默认 pending）
        folder_path = os.path.join(root_dir, folder_name)  # 文件夹完整路径

        if status == STATUS_DONE:  # 已完成
            n_skipped += 1  # 计数
        elif status == STATUS_RUNNING:  # 上次被中断（运行中途程序退出）
            n_interrupted += 1  # 计数
            print("  ⚠ 检测到被中断的工况：{} — 将删除文件夹并重新执行".format(folder_name))  # 提示
            _remove_folder(folder_path)  # 删除不完整的文件夹
            _update_status(progress_path, progress_data, folder_name, STATUS_PENDING)  # 重置为 pending
        elif status == STATUS_FAILED:  # 上次执行失败
            n_failed_redo += 1  # 计数
            print("  ⚠ 检测到失败的工况：{} — 将删除文件夹并重新执行".format(folder_name))  # 提示
            _remove_folder(folder_path)  # 删除失败的文件夹
            _update_status(progress_path, progress_data, folder_name, STATUS_PENDING)  # 重置为 pending

    # 打印恢复摘要
    n_todo = len(folder_plan) - n_skipped  # 本次需执行的工况数
    if n_skipped > 0 or n_interrupted > 0 or n_failed_redo > 0:  # 有恢复动作
        print("\n--- 断点续跑摘要 ---")  # 摘要标题
        print("  已完成（跳过）：{} 个".format(n_skipped))  # 跳过数
        if n_interrupted > 0:  # 有被中断的
            print("  被中断（删除重来）：{} 个".format(n_interrupted))  # 中断数
        if n_failed_redo > 0:  # 有失败的
            print("  上次失败（删除重来）：{} 个".format(n_failed_redo))  # 失败数
        print("  本次待执行：{} 个".format(n_todo))  # 待执行数
        print("--------------------")  # 摘要结束
    else:  # 首次运行
        print("\n首次运行，未检测到进度文件（或进度为空）。")  # 首次提示
        print("本次待执行：{} 个工况".format(n_todo))  # 待执行数

    if n_todo == 0:  # 全部已完成
        print("\n所有 {} 个工况均已完成，无需重新执行。直接进入后处理阶段。".format(len(folder_plan)))  # 全完成提示
    else:  # 有待执行工况
        # ---- 并行批处理工况（含进度跟踪） ----
        failed_folders = []  # 失败目录列表
        completed_count = 0  # 本轮已完成计数（仅计本次运行中完成的）
        batch_start_time = time.time()  # 批处理开始时间

        def process_folder(item, seq_idx, total_todo):  # 处理单个工况文件夹的工作函数（含进度更新）
            """item=(folder_name, config, fc, _set)；按 fc 选 Ricker、写 config、执行脚本序列；返回(path,ok)。
            seq_idx: 本次运行中的序号（1-based），total_todo: 本次需执行的总数。"""
            folder_name, config, fc, _set = item  # 解包
            folder_path = os.path.join(root_dir, folder_name)  # 文件夹完整路径
            elapsed = time.time() - batch_start_time  # 已用时间

            _print_case_header(seq_idx, total_todo, folder_name, fc, _set, elapsed)  # 打印工况标题

            # 标记为 running（若此后程序崩溃，下次启动时会检测到并删除重来）
            _update_status(progress_path, progress_data, folder_name, STATUS_RUNNING,
                           extra={"started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                  "set": _set, "fc": fc})  # 持久化 running 状态

            # 为该工况构建源文件映射：脚本 + 单一 Ricker
            ricker_path = _ricker_path(fc)  # 该工况对应的 Ricker 波文件路径
            source_files, missing, dups = build_source_files([ricker_path], SCRIPT_SEQUENCE)  # 构建映射
            if missing:  # 缺失文件
                print("  错误：工况 {} 缺失源文件：{}".format(folder_name, [m[1] for m in missing]))  # 报错
                _update_status(progress_path, progress_data, folder_name, STATUS_FAILED,
                               extra={"error": "缺失源文件"})  # 标记失败
                return folder_path, False  # 返回失败
            if dups:  # 重名冲突
                print("  错误：工况 {} 文件重名：{}".format(folder_name, dups))  # 报错
                _update_status(progress_path, progress_data, folder_name, STATUS_FAILED,
                               extra={"error": "文件重名"})  # 标记失败
                return folder_path, False  # 返回失败

            create_and_fill_folder(folder_path, source_files, config)  # 创建目录并注入配置
            ok = run_scripts_in_folder(folder_path, run_order)  # 顺序执行脚本

            if ok:  # 执行成功
                _update_status(progress_path, progress_data, folder_name, STATUS_DONE,
                               extra={"completed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})  # 标记完成
            else:  # 执行失败
                _update_status(progress_path, progress_data, folder_name, STATUS_FAILED,
                               extra={"error": "脚本执行失败"})  # 标记失败

            if types_to_delete:  # 需清理中间文件
                delete_files_by_type(folder_path, types_to_delete)  # 永久删除中间文件
            else:  # 不清理
                print("  已跳过文件删除（未指定要删除的文件类型）。")  # 提示
            return folder_path, ok  # 返回结果

        # 筛选待执行的工况（跳过已完成）
        todo_items = []  # 待执行列表
        for item in folder_plan:  # 遍历全部工况
            folder_name = item[0]  # 文件夹名
            entry = progress_data.get(folder_name, {})  # 进度条目
            if entry.get("status") == STATUS_DONE:  # 已完成
                continue  # 跳过
            todo_items.append(item)  # 加入待执行

        total_todo = len(todo_items)  # 本次需执行总数

        print("\n开始批处理，最大并发任务数：{}".format(MAX_WORKERS))  # 启动提示
        _print_progress_bar(0, total_todo, 0)  # 打印初始进度条

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 线程池
            # 逐个提交以保证 seq_idx 对应实际执行顺序（并发数通常为 1-2，不影响效率）
            futures = {}  # future → (seq_idx, item) 映射
            for seq_idx, item in enumerate(todo_items, start=1):  # 遍历待执行工况
                future = executor.submit(process_folder, item, seq_idx, total_todo)  # 提交任务
                futures[future] = (seq_idx, item)  # 记录映射

            for future in concurrent.futures.as_completed(futures):  # 按完成顺序收集结果
                seq_idx, item = futures[future]  # 获取序号与工况
                try:  # 获取结果
                    folder_path, ok = future.result()  # 解包
                except Exception as exc:  # 未预期异常
                    folder_name = item[0]  # 文件夹名
                    folder_path = os.path.join(root_dir, folder_name)  # 完整路径
                    print("  ✗ 工况 {} 发生未预期异常：{}".format(folder_name, exc))  # 报错
                    _update_status(progress_path, progress_data, folder_name, STATUS_FAILED,
                                   extra={"error": str(exc)})  # 标记失败
                    ok = False  # 视为失败
                if not ok:  # 失败
                    failed_folders.append(folder_path)  # 记录失败
                completed_count += 1  # 本轮已完成计数
                _print_progress_bar(completed_count, total_todo, time.time() - batch_start_time)  # 更新进度条

        # 批处理总结
        batch_elapsed = time.time() - batch_start_time  # 批处理总耗时
        print("\n" + "=" * 70)  # 总结分隔线
        print("  批处理完成  耗时: {}".format(_format_duration(batch_elapsed)))  # 耗时
        if failed_folders:  # 有失败
            print("  ✗ 失败文件夹（{}个）：".format(len(failed_folders)))  # 失败标题
            for path in failed_folders:  # 遍历失败
                print("    - {}".format(os.path.basename(path)))  # 打印路径
            print("  提示：失败工况已标记为 failed，下次运行将自动删除文件夹并重试。")  # 提示
        else:  # 全部成功
            print("  ✓ 本次执行的 {} 个工况全部成功。".format(total_todo))  # 全部成功
        print("=" * 70)  # 分隔线

        if failed_folders:  # 有失败 → 不进入后处理
            print("\n存在失败工况，跳过后处理阶段。请修复问题后重新运行脚本。")  # 提示
            sys.exit(2)  # 异常退出

    # ------------------------------------------------------------------
    #  后处理阶段：拷贝后处理脚本到根目录并顺序执行
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)  # 后处理分隔符
    print("  开始自动后处理脚本阶段...")  # 阶段标题
    print("=" * 70)  # 分隔符
    post_run_order = []  # 收集后处理脚本文件名
    for src_path in POST_SCRIPT_SEQUENCE:  # 遍历后处理脚本
        if not os.path.isfile(src_path):  # 源脚本不存在
            print("错误：后处理脚本缺失 -> {}".format(src_path))  # 报错
            sys.exit(3)  # 退出
        target_name = os.path.basename(src_path)  # 目标文件名
        dst_path = os.path.join(root_dir, target_name)  # 目标路径
        shutil.copy2(src_path, dst_path)  # 拷贝到根目录
        post_run_order.append(target_name)  # 记录执行顺序
        print("  已拷贝后处理脚本：{}".format(target_name))  # 确认拷贝

    for post_idx, script_name in enumerate(post_run_order, start=1):  # 按顺序执行后处理脚本
        script_path = os.path.join(root_dir, script_name)  # 脚本完整路径
        print("\n  ▶ [{}/{}] 执行后处理：{}".format(post_idx, len(post_run_order), script_name))  # 开始日志
        result = subprocess.run([sys.executable, script_name, root_dir], cwd=root_dir, check=False)  # 在根目录执行并传入目录参数
        if result.returncode != 0:  # 执行失败
            print("  ✗ 后处理失败：{} (返回码={})".format(script_name, result.returncode))  # 报错
            sys.exit(4)  # 后处理失败退出
        print("  ✓ 完成后处理：{}".format(script_name))  # 完成日志

    # 全部完成总结
    print("\n" + "=" * 70)  # 最终总结分隔线
    print("  ✓ 全部流程完成！")  # 完成标题
    print("  工况总数: {} | 后处理脚本: {} 个".format(len(folder_plan), len(post_run_order)))  # 统计
    print("  完成时间: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))  # 完成时间
    print("=" * 70)  # 分隔线


if __name__ == "__main__":  # 主入口判断
    main()  # 运行主流程
