# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""小论文批次 05/05：C001—C010 真实波直接闭环（全局执行序号 084—093）。

依据《毕业论文第四至第六章与英文期刊小论文融合研究实施总计划最终版》§4.8：
4 个代表系统（P007/P039/P061/B007）× 记录 EQ01/EQ02，加 P039/B007 × EQ03。
只能在 B001—B012 完成评价、代理永久锁定后运行（计划 §4.8/§4.10）。
输入波为 Wave/Seismic/Sp_EQ 下预处理记录（0.1g，dt=1 ms，
来源与预处理参数见 sp_eq_input_manifest.json）。

运行形式：
  python Autorun_ch4_sp_05_C_v1.py [求解输出根目录]
缺省输出根目录为 C:\\Users\\12462\\Documents\\Code\\AbqScripts\\Run\\ch4_sp_05_C，
工况文件夹按全局执行顺序命名（case-084-C001 … case-093-C010）。
"""

import os  # 导入操作系统相关路径与目录操作模块
import json  # 导入 JSON 模块用于写出注入配置文件
import shutil  # 导入文件复制与高层级文件操作模块
import subprocess  # 导入子进程执行模块以运行其他脚本
import sys  # 导入系统模块用于获取 Python 解释器路径与退出程序
import concurrent.futures  # 导入并发模块以实现多工况文件夹并行执行
import datetime  # 导入时间模块用于清单追踪
import math  # 导入数学模块用于临界角计算
import time  # 导入时间模块用于轮询子进程与进度日志
import threading  # 导入线程锁，避免并发工况终端输出互相穿插

ROOT_DIR = r"C:\Users\12462\Documents\Code\AbqScripts\Run\ch4_sp_05_C"  # 默认求解输出根目录（小论文批次 05/05：C001—C010），可用命令行参数覆盖
FOLDER_PREFIX = "case-"  # 各工况文件夹的命名统一前缀
DELETE_FILE_TYPES = [".odb", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".jnl", ".com", ".rpy", ".rec"]  # 数据提取成功后删除的过程文件；CAE按当前存储策略保留
REQUIRED_RESULT_FILES = ["surface_results.npz", "surface_results.xlsx"]  # 清理前必须同时存在且非空的规范数据产物
POSTPROCESS_STATUS_FILENAME = "postprocess_status.json"  # 后处理数据提取状态；不依赖可能吞退出码的Abaqus批处理包装器
MAX_WORKERS = 4  # 单机最多同时运行4个建模/求解工况，已由G1r正式批次验证
POSTPROCESS_WORKERS = 1  # 单工况后处理并发数，默认1以减少与求解争用内存
CONFIG_FILENAME = "case_config.json"  # 注入给建模或计算脚本的配置文件名
TERMINAL_PROGRESS_POLL_SECONDS = 300.0  # autorun每5min直读sta，建模进度日志仅作回退
PROCESS_STATUS_POLL_SECONDS = 5.0  # 轻量检查子进程是否结束，避免低频进度轮询阻塞后续步骤
_PROGRESS_PRINT_LOCK = threading.Lock()  # 并发工况共用终端输出锁

# Abaqus 启动路径与需要由 Abaqus Python 运行的脚本名单
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'
ABAQUS_CAE_SCRIPTS = {'slope_frame_ssi_full_v2.py'}
ABAQUS_PYTHON_SCRIPTS = {'Postprocess_All_surface_v2.py'}
ABAQUS_SCRIPTS = ABAQUS_CAE_SCRIPTS | ABAQUS_PYTHON_SCRIPTS  # 兼容既有检查

# 注入配置的顶层键白名单
ALLOWED_CONFIG_KEYS = {
    'material_cfg', 'geometry_cfg', 'damping_cfg', 'mesh_cfg',
    'time_cfg', 'run_cfg', 'eql_cfg', 'tssi_cfg',
}

MODEL_SCRIPT_SEQUENCE = [  # 建模线程池连续执行的脚本绝对路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\slope_frame_ssi_full_v2.py",  # 建模脚本（读取 case_config.json，含层内材料一致化/网格自适应/时间步校验）
]

CASE_POSTPROCESS_SCRIPT_SEQUENCE = [  # 建模完成后进入独立线程池的单工况后处理脚本
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Postprocess_All_surface_v2.py",  # 后处理提取脚本路径 / 提取单工况数据
]

SCRIPT_SEQUENCE = MODEL_SCRIPT_SEQUENCE + CASE_POSTPROCESS_SCRIPT_SEQUENCE  # 兼容源文件复制清单

POST_SCRIPT_SEQUENCE = []  # 全局汇总由小论文专用后处理统一执行，autorun 不再跨批次自动聚合

# ==========================================================
#  小论文统一数值设置（计划 §4.3）与工况组装辅助
# ==========================================================

REPO_ROOT = r"C:\Users\12462\Documents\Code\AbqScripts"  # 仓库根目录
G1B_WAVE = os.path.join(REPO_ROOT, "Wave", "Impulse", "Acceleration", "G1b_frequency_gate", "g1b_multisine_phase_a.txt")  # 统一系统识别输入（0.1g，dt=1 ms，0.5—12 Hz）
EQ_WAVE_DIR = os.path.join(REPO_ROOT, "Wave", "Seismic", "Sp_EQ")  # 真实波输入目录
SLOPE_HEIGHT = 100.0  # 坡高（m），小论文固定
INCIDENCE_ANGLE = 15.0  # SV 波入射角（度），小论文固定
COVER_DENSITY = 2125  # 覆盖层密度（kg/m³）=0.85×2500
COVER_POISSON = 0.35  # 覆盖层泊松比
COVER_XI = 0.03  # 覆盖层阻尼比 3%
BEDROCK_CFG = {'vs': 2000.0, 'poisson_ratio': 0.3, 'density': 2500}  # 基岩参数，小论文固定
BEDROCK_XI = 5.0e-4  # 基岩阻尼比（≈0.05%）
TAIL_SECONDS = 6.0  # 静默尾段时长（s）


def cover_layer(vs, thickness):
    """生成小论文单层覆盖层配置（水平带，置于基岩之上）。"""
    return {'name': 'surface', 'vs': float(vs), 'poisson_ratio': COVER_POISSON,
            'density': COVER_DENSITY, 'thickness': float(thickness)}


def base_config(slope_angle, layers, wave_files, extra=None):
    """组装小论文统一工况配置：坡高100 m/入射15°/尾段6 s/freefield 场景/incremental 初态。

    参数:
        slope_angle: 坡角（度）。
        layers: 覆盖层列表；均质基岩坡传 []。
        wave_files: 输入波文件绝对路径列表。
        extra: 需并入配置的附加覆盖字典（如 V 批次的网格/计算域/尾段变体）。
    """
    config = {
        'material_cfg': {'angle': INCIDENCE_ANGLE, 'bedrock': dict(BEDROCK_CFG), 'layers': layers},  # 入射角与地层
        'geometry_cfg': {'slope_height': SLOPE_HEIGHT, 'slope_angle': float(slope_angle),
                         'crest_window': 4.0, 'toe_window': 3.0,
                         'side_clearance': 1.0, 'base_depth': 3.0},  # 观测窗4h/3h + 侧向净距1H + 基底深度3H
        'damping_cfg': {'constant_xi': COVER_XI, 'bedrock_xi': BEDROCK_XI},  # 覆盖层 3% / 基岩 0.05%
        'time_cfg': {'tail_seconds': TAIL_SECONDS},  # 静默尾段
        'run_cfg': {
            'wave_files': list(wave_files),
            'frf_cfg': {'fmax_hz': 12.0},  # 复频响输出上限
            'response_spectrum_cfg': {'enable': True},  # 真实波重构阶段计算工程反应谱指标
        },
        'tssi_cfg': {'enable': False, 'scene': 'freefield'},  # 纯坡地自由场场景（无结构）
    }
    if extra:  # 并入批次特定覆盖（浅层按键合并，仅用于同段内少量键）
        for key, val in extra.items():
            if isinstance(val, dict) and isinstance(config.get(key), dict):
                config[key].update(val)
            else:
                config[key] = val
    return config


EQ_WAVES = {  # 预处理真实波输入（0.1g，dt=1 ms）
    'EQ01': os.path.join(EQ_WAVE_DIR, 'sp_eq01_el_centro_0p1g_dt1ms.txt'),
    'EQ02': os.path.join(EQ_WAVE_DIR, 'sp_eq02_kobe_0p1g_dt1ms.txt'),
    'EQ03': os.path.join(EQ_WAVE_DIR, 'sp_eq03_chichi_0p1g_dt1ms.txt'),
}
_C_SYSTEMS = {  # 闭环系统（坡角, d/h, rv），定义与 P/B 批次一致
    'P007': (15.0, 0.60, 0.45),
    'P039': (45.0, 0.60, 0.60),
    'P061': (60.0, 1.40, 0.30),
    'B007': (37.5, 0.80, 0.675),
}
_C_TABLE = [  # (工况ID, 系统, 记录)：计划 §4.8，全局执行序号 084—093
    ('C001', 'P007', 'EQ01'), ('C002', 'P007', 'EQ02'),
    ('C003', 'P039', 'EQ01'), ('C004', 'P039', 'EQ02'),
    ('C005', 'P061', 'EQ01'), ('C006', 'P061', 'EQ02'),
    ('C007', 'B007', 'EQ01'), ('C008', 'B007', 'EQ02'),
    ('C009', 'P039', 'EQ03'), ('C010', 'B007', 'EQ03'),
]
PARAMETER_CASES = [
    {"name": "%03d-%s" % (84 + _k, _cid),
     "config": base_config(_C_SYSTEMS[_sys][0],
                           [cover_layer(_C_SYSTEMS[_sys][2] * 2000.0, _C_SYSTEMS[_sys][1] * SLOPE_HEIGHT)],
                           [EQ_WAVES[_eq]])}
    for _k, (_cid, _sys, _eq) in enumerate(_C_TABLE, start=1)
]


def validate_postprocess_status(folder_path):  # 核验后处理数据文件是否成功写出
    """读取后处理状态文件，返回 ``(通过, 原因)``。"""
    status_path = os.path.join(folder_path, POSTPROCESS_STATUS_FILENAME)
    if not os.path.isfile(status_path):
        return False, "缺少{}".format(POSTPROCESS_STATUS_FILENAME)
    try:
        with open(status_path, 'r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except Exception as exc:
        return False, "{}无法读取: {}".format(
            POSTPROCESS_STATUS_FILENAME, str(exc),
        )
    if not bool(payload.get('success', False)):
        return False, "{}报告数据提取失败: {}".format(
            POSTPROCESS_STATUS_FILENAME, payload.get('reason') or 'unknown',
        )
    return True, "completed"


def _case_already_done(root_dir, folder_name):
    """检查工况目录是否已有成功的后处理结果，用于跳过已完成的工况。"""
    status_path = os.path.join(root_dir, folder_name, POSTPROCESS_STATUS_FILENAME)
    if not os.path.isfile(status_path):
        return False
    try:
        with open(status_path, 'r', encoding='utf-8') as f:
            return bool(json.load(f).get('success', False))
    except Exception:
        return False


def validate_config(config):
    """验证工况配置是否合规。

    根据 G0 生产准备规则：
    1. 配置段只能使用允许 of 白名单键。对未知键立即失败。
    2. 临界角校验硬性拦截。
    """
    if not config:
        return
    for key in config.keys():
        if key not in ALLOWED_CONFIG_KEYS:
            raise ValueError("错误：发现了不属于白名单的非法顶层配置键: '{}'。允许的键包括: {}".format(
                key, sorted(list(ALLOWED_CONFIG_KEYS))))

    # 临界角预检
    material_cfg = config.get("material_cfg") or {}
    run_cfg = config.get("run_cfg") or {}
    critical_angle_check = run_cfg.get("critical_angle_check", True)
    angle = material_cfg.get("angle")
    if angle is not None:
        bedrock = material_cfg.get("bedrock") or {}
        vs = bedrock.get("vs", 2000.0)
        pr = bedrock.get("poisson_ratio", 0.3)
        if 0.0 < pr < 0.5:
            sin_crit = math.sqrt((1.0 - 2.0 * pr) / (2.0 * (1.0 - pr)))
            crit_deg = math.degrees(math.asin(sin_crit))
        else:
            crit_deg = 32.3115  # 默认 ν=0.3 下的临界角
        if angle >= crit_deg - 1e-6 and critical_angle_check:
            raise ValueError("错误：入射角 {}° 达到或超过基岩临界角 {:.2f}°（超临界非均匀波不在方法适用域内，硬性拦截）".format(angle, crit_deg))


def _write_run_manifest(root_dir, folder_plan, source_files, status_dict=None):
    """写出或更新本批次运行的主清单 run_manifest.json。"""
    manifest_path = os.path.join(root_dir, 'run_manifest.json')
    previous_data = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                previous_data = json.load(f)
        except Exception:
            pass

    previous_cases = previous_data.get('cases', {}) if isinstance(previous_data, dict) else {}
    manifest_data = {
        'created_at': previous_data.get('created_at') or datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'source_files': sorted(source_files.keys()),
        'execution_policy': {
            'pipeline_mode': 'model_then_async_case_postprocess',
            'model_workers': MAX_WORKERS,
            'case_postprocess_workers': POSTPROCESS_WORKERS,
            'global_postprocess_after_all_cases': True,
            'cleanup_after_required_results': bool(DELETE_FILE_TYPES),
            'cleanup_file_types': list(DELETE_FILE_TYPES),
            'cleanup_required_results': list(REQUIRED_RESULT_FILES),
            'retain_cae': True,
        },
    }

    cases = {}
    for folder_name, config in folder_plan:
        case_status = status_dict.get(folder_name, 'planned') if status_dict else 'planned'
        previous_case = previous_cases.get(folder_name, {})
        cases[folder_name] = {
            'case_id': folder_name,
            'status': case_status,
            'added_at': previous_case.get('added_at') or datetime.datetime.now().isoformat(),
            'updated_at': datetime.datetime.now().isoformat(),
        }

    manifest_data['cases'] = cases
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, ensure_ascii=False, indent=2, sort_keys=True)


def build_source_files(script_sequence):  # 建立脚本文件映射并检测缺失与冲突
    """建立脚本文件映射并检测缺失与冲突。

    参数说明:
        script_sequence (list): 执行脚本完整路径列表。

    返回值:
        tuple: (source_files 映射字典, missing 缺失文件列表, duplicate_names 冲突文件列表)
    """
    source_files = {}  # 初始化源文件名字与物理路径的映射字典
    missing = []  # 初始化不存在的文件记录列表
    duplicate_names = []  # 初始化发生重名冲突的文件记录列表
    for source_path in list(script_sequence):  # 遍历所有配置的脚本路径
        target_name = os.path.basename(source_path)  # 获取文件名作为目标文件夹下的名称
        if not os.path.isfile(source_path):  # 若源物理文件不存在
            missing.append((target_name, source_path))  # 记录缺失的文件名与路径
            continue  # 继续处理下一个文件
        if target_name in source_files and source_files[target_name] != source_path:  # 若同名文件指向不同源路径
            duplicate_names.append((target_name, source_path, source_files[target_name]))  # 记录重名冲突
            continue  # 继续处理下一个文件
        source_files[target_name] = source_path  # 写入映射字典
    return source_files, missing, duplicate_names  # 返回映射与错误检测结果


def ensure_sources_exist(missing_items):  # 校验并打印缺失的源文件
    """校验并打印缺失的源文件。

    参数说明:
        missing_items (list): 缺失文件列表。

    返回值:
        bool: 若无缺失返回 True，否则返回 False。
    """
    if not missing_items:  # 若无缺失文件
        return True  # 校验通过
    print("错误：以下源文件不存在，请检查配置路径：")  # 打印错误提示头
    for name, path in missing_items:  # 遍历缺失记录
        print("  - {} -> {}".format(name, path))  # 打印缺失的文件名与原路径
    return False  # 校验失败


def ensure_no_duplicate_targets(duplicate_items):  # 校验并打印重名冲突的文件
    """校验并打印重名冲突的文件。

    参数说明:
        duplicate_items (list): 冲突文件列表。

    返回值:
        bool: 若无冲突返回 True，否则返回 False。
    """
    if not duplicate_items:  # 若无冲突文件
        return True  # 校验通过
    print("错误：以下文件复制后会重名，请调整路径或文件名：")  # 打印错误提示头
    for target_name, current_path, existing_path in duplicate_items:  # 遍历冲突记录
        print("  - {} -> {} (已存在来源: {})".format(target_name, current_path, existing_path))  # 打印冲突细节
    return False  # 校验失败


def _fmt_num(v):  # 格式化数值为紧凑字符串
    """格式化数值为紧凑字符串。

    参数说明:
        v (float/int/str): 需要被格式化的变量。

    返回值:
        str: 去掉无意义小数点或特殊字符的紧凑字符串。
    """
    try:  # 尝试将变量转换为浮点数处理
        f = float(v)  # 转换为浮点型
        return str(int(f)) if f == int(f) else ('%g' % f)  # 整数则去掉小数点，小数用 %g 紧凑表示
    except (TypeError, ValueError):  # 若转换失败
        return str(v)  # 原样返回字符串形式


def _sanitize(text):  # 清浅地清洗字符串为合法的文件目录名
    """清洗字符串为合法的文件目录名。

    参数说明:
        text (str): 待清洗的原始字符串。

    返回值:
        str: 仅包含字母、数字、点、下划线与连字符的合法目录名。
    """
    import re  # 局部导入正则模块
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-") or "x"  # 替换非法字符为连字符并清除两端空格


def name_from_config(config):  # 根据配置字典自动生成唯一的工况文件夹名后缀
    """根据配置字典自动生成唯一的工况文件夹名后缀。

    参数说明:
        config (dict): 参数覆盖字典。

    返回值:
        str: 拼接后的工况目录名后缀。
    """
    tokens = []  # 初始化名称片段列表
    for key, val in sorted(config.items()):  # 遍历一级配置键值对
        if isinstance(val, dict):  # 若值为子字典结构
            for sub_key, sub_val in sorted(val.items()):  # 遍历二级键值对
                if not isinstance(sub_val, (dict, list)):  # 仅提取标量基本类型
                    tokens.append("{}{}".format(sub_key, _fmt_num(sub_val)))  # 拼接二级键名与数值
        elif not isinstance(val, list):  # 若值为一级标量基本类型
            tokens.append("{}{}".format(key, _fmt_num(val)))  # 拼接一级键名与数值
    return "-".join(tokens) if tokens else "default"  # 用连字符拼接片段，全空则使用 default


def build_folder_name(case):  # 确定工况文件夹名称
    """确定工况文件夹名称。

    参数说明:
        case (dict): 单个工况字典。

    返回值:
        str: 完整的工况文件夹名称（带前缀）。
    """
    tag = case.get("name") or case.get("folder_tag")  # 获取显式指定的名称或标签
    if not tag:  # 若未指定
        tag = name_from_config(case.get("config") or {})  # 根据配置字典自动生成
    return "{}{}".format(FOLDER_PREFIX, _sanitize(tag))  # 返回拼接前缀后的规范化名称


def create_and_fill_folder(folder_path, source_files, config):  # 创建目录并注入配置文件与源文件
    """创建目录并注入配置文件与源文件。

    参数说明:
        folder_path (str): 目标工况文件夹绝对路径。
        source_files (dict): 源文件物理路径映射字典。
        config (dict): 参数覆盖字典。
    """
    validate_config(config)  # 强校验配置
    os.makedirs(folder_path, exist_ok=True)  # 创建工况目录
    for target_name, src_path in source_files.items():  # 遍历待拷贝的所有文件映射
        shutil.copy2(src_path, os.path.join(folder_path, target_name))  # 拷入工况目录并保留元数据
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:  # 打开注入的配置文件
        json.dump(config or {}, f, ensure_ascii=False, indent=2)  # 序列化配置为 JSON 文件


def _read_new_job_progress(log_path, offset, not_before):
    """增量读取建模日志中的作业进度行，返回新偏移和消息列表。"""
    if not os.path.isfile(log_path):
        return offset, []
    try:
        if os.path.getmtime(log_path) + 1.0 < not_before:
            return offset, []  # 忽略同目录上一次运行遗留的旧日志
        size = os.path.getsize(log_path)
        if size < offset:
            offset = 0  # 建模脚本以写模式重建日志后，从文件头重新读取
        with open(log_path, 'rb') as handle:
            handle.seek(offset)
            data = handle.read()
            new_offset = handle.tell()
        text = data.decode('utf-8', 'replace')
        messages = []
        for line in text.splitlines():
            marker = '作业进度:'
            if marker in line:
                messages.append(line[line.index(marker):].strip())
        return new_offset, messages
    except (IOError, OSError, ValueError):
        return offset, []


def _read_model_step_totals(log_path):
    """从建模日志读取模型名与地震动力步总时长的对应关系。"""
    totals = {}
    if not log_path or not os.path.isfile(log_path):
        return totals
    marker = ' 分析步已创建, 时长='
    try:
        with open(log_path, 'rb') as handle:
            lines = handle.read().decode('utf-8', 'replace').splitlines()
        for line in lines:
            if marker not in line:
                continue
            prefix, value_text = line.split(marker, 1)
            prefix_fields = prefix.split()
            if not prefix_fields:
                continue
            try:
                totals[prefix_fields[-1]] = float(value_text.split('(', 1)[0])
            except (TypeError, ValueError):
                continue
    except (IOError, OSError, ValueError):
        return {}
    return totals


def _target_dynamic_step_number(folder_path):
    """依据工况配置返回Step-earthquake在sta中的步序号。"""
    config_path = os.path.join(folder_path, CONFIG_FILENAME)
    try:
        with open(config_path, 'r', encoding='utf-8') as handle:
            config = json.load(handle)
    except (IOError, OSError, ValueError, TypeError):
        return 1
    tssi = config.get('tssi_cfg') or {}
    gravity_enabled = (
        bool(tssi.get('enable'))
        and str(tssi.get('scene', 'ssi')).lower() != 'freefield'
        and str(tssi.get('gravity', 'off')).lower() != 'off'
    )
    return 2 if gravity_enabled else 1


def _read_sta_step_progress(sta_path, target_step_number):
    """从Abaqus/Standard状态文件尾部读取目标分析步最新步时间。"""
    try:
        with open(sta_path, 'rb') as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144), os.SEEK_SET)
            tail = handle.read().decode('ascii', 'ignore')
        for line in reversed(tail.splitlines()):
            fields = line.split()
            if len(fields) < 9:
                continue
            try:
                step_number = int(fields[0])
                int(fields[1])
                step_time = float(fields[7].replace('D', 'E').replace('d', 'e'))
            except (TypeError, ValueError):
                continue
            if step_number == int(target_step_number):
                return step_time
    except (IOError, OSError, ValueError):
        return None
    return None


def _read_sta_job_progress(folder_path, not_before, progress_log):
    """直接读取本次运行最新sta，返回一条可供终端打印的进度消息。"""
    candidates = []
    try:
        for name in os.listdir(folder_path):
            if not name.lower().endswith('.sta'):
                continue
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            modified = os.path.getmtime(path)
            if modified + 1.0 >= not_before:
                candidates.append((modified, os.path.getsize(path), path))
    except (IOError, OSError, ValueError):
        return []
    if not candidates:
        return []
    sta_path = max(candidates)[2]
    job_name = os.path.splitext(os.path.basename(sta_path))[0]
    model_name = job_name[4:] if job_name.startswith('job-') else job_name
    totals = _read_model_step_totals(progress_log)
    total_seconds = totals.get(model_name)
    if total_seconds is None and len(totals) == 1:
        total_seconds = next(iter(totals.values()))
    if total_seconds is None or total_seconds <= 0.0:
        return []
    target_step = _target_dynamic_step_number(folder_path)
    current_seconds = _read_sta_step_progress(sta_path, target_step)
    note = ''
    if current_seconds is None:
        current_seconds = 0.0
        note = '，目标动力步尚未开始'
    current_seconds = max(0.0, float(current_seconds))
    percent = min(100.0, 100.0 * current_seconds / total_seconds)
    return [
        '作业进度: {}，已算到 {:.3f} 秒/共 {:.3f} 秒（{:.1f}%，Step-earthquake{}）'.format(
            job_name, current_seconds, total_seconds, percent, note,
        )
    ]


def _case_terminal_prefix(folder_path, case_index=None, case_total=None):
    """生成包含计划序号、总数和工况名的终端前缀。"""
    case_name = os.path.basename(os.path.normpath(folder_path))
    if case_index is not None and case_total is not None:
        return "[工况 {}/{}][{}]".format(
            int(case_index), int(case_total), case_name,
        )
    return "[{}]".format(case_name)


def _print_case_message(folder_path, message, case_index=None,
                        case_total=None):
    """按整行输出工况消息，防止多工况并发时字符交叉。"""
    prefix = _case_terminal_prefix(
        folder_path, case_index=case_index, case_total=case_total,
    )
    with _PROGRESS_PRINT_LOCK:
        print("{} {}".format(prefix, message), flush=True)


def _print_job_progress(folder_path, messages, case_index=None,
                        case_total=None):
    """按整行输出进度，防止多工况并发时字符交叉。"""
    if not messages:
        return
    prefix = _case_terminal_prefix(
        folder_path, case_index=case_index, case_total=case_total,
    )
    with _PROGRESS_PRINT_LOCK:
        for message in messages:
            print("[运行状态]{} {}".format(prefix, message), flush=True)


def run_scripts_in_folder(folder_path, run_order, step_offset=0,
                          stage_label='stage', case_index=None,
                          case_total=None):  # 在工况文件夹内按顺序执行指定阶段脚本
    """在工况文件夹内按顺序执行指定的脚本，自动分发 Abaqus cae 和普通 Python 解释器。

    参数说明:
        folder_path (str): 工况文件夹绝对路径。
        run_order (list): 当前阶段脚本文件名（不含路径）顺序列表。
        step_offset (int): 当前阶段之前已有的脚本数量。
        stage_label (str): 日志中的阶段标签。
        case_index (int): 工况在本批次中的序号。
        case_total (int): 本批次工况总数。

    返回值:
        bool: 若全部顺利执行返回 True，任意脚本执行失败则返回 False。
    """
    for local_idx, script_name in enumerate(run_order, start=1):  # 遍历待执行的脚本
        idx = int(step_offset) + local_idx  # 保持拆分流水线后的全流程步骤编号
        script_path = os.path.join(folder_path, script_name)  # 拼接绝对路径
        if not os.path.isfile(script_path):  # 若物理文件不存在
            print("错误：脚本不存在 -> {}".format(script_path))  # 打印不存在 of 错误提示
            return False  # 返回失败

        # 判断执行命令
        if script_name in ABAQUS_CAE_SCRIPTS:
            cmd = [ABAQUS_CMD, 'cae', 'noGUI=' + script_name]
            log_filename = "autorun_step{:02d}_{}_{}.log".format(
                idx, stage_label, os.path.splitext(script_name)[0],
            )
        elif script_name in ABAQUS_PYTHON_SCRIPTS:
            cmd = [ABAQUS_CMD, 'python', script_name]
            log_filename = "autorun_step{:02d}_{}_{}.log".format(
                idx, stage_label, os.path.splitext(script_name)[0],
            )
        else:
            cmd = [sys.executable, script_name]
            log_filename = "autorun_step{:02d}_post_{}.log".format(idx, os.path.splitext(script_name)[0])

        log_path = os.path.join(folder_path, log_filename)
        postprocess_status_path = os.path.join(
            folder_path, POSTPROCESS_STATUS_FILENAME,
        )
        if (
            script_name == 'Postprocess_All_surface_v2.py'
            and os.path.isfile(postprocess_status_path)
        ):
            os.remove(postprocess_status_path)  # 当前进程启动前删除旧状态，避免误用历史执行结果
        _print_case_message(
            folder_path,
            "开始执行: {} (命令: {})".format(script_name, ' '.join(cmd)),
            case_index=case_index, case_total=case_total,
        )

        env = os.environ.copy()
        if script_name not in ABAQUS_SCRIPTS:
            env['PYTHONIOENCODING'] = 'utf-8'

        with open(log_path, 'wb') as handle:
            handle.write("命令: {}\n工作目录: {}\n\n".format(' '.join(cmd), folder_path).encode('utf-8'))
            handle.flush()
            try:
                started_at = time.time()
                process = subprocess.Popen(cmd, cwd=folder_path, stdout=handle,
                                           stderr=subprocess.STDOUT, env=env)
                progress_offset = 0
                progress_log = os.path.join(
                    folder_path, os.path.splitext(script_name)[0] + '.log',
                ) if script_name == 'slope_frame_ssi_full_v2.py' else None
                next_progress_poll = started_at + float(TERMINAL_PROGRESS_POLL_SECONDS)
                while process.poll() is None:
                    now = time.time()
                    if progress_log and now >= next_progress_poll:
                        progress_offset, messages = _read_new_job_progress(
                            progress_log, progress_offset, started_at,
                        )
                        sta_messages = _read_sta_job_progress(
                            folder_path, started_at, progress_log,
                        )
                        if sta_messages:
                            messages = sta_messages  # sta是求解器当前状态，优先于可能滞后的建模日志
                        _print_job_progress(
                            folder_path, messages,
                            case_index=case_index, case_total=case_total,
                        )
                        next_progress_poll = now + float(TERMINAL_PROGRESS_POLL_SECONDS)
                    time.sleep(max(0.5, float(PROCESS_STATUS_POLL_SECONDS)))
                if progress_log:
                    progress_offset, messages = _read_new_job_progress(
                        progress_log, progress_offset, started_at,
                    )
                    sta_messages = _read_sta_job_progress(
                        folder_path, started_at, progress_log,
                    )
                    if sta_messages:
                        messages = sta_messages
                    _print_job_progress(
                        folder_path, messages,
                        case_index=case_index, case_total=case_total,
                    )
                returncode = process.returncode
            except Exception as e:
                handle.write("\n子进程启动异常: {}\n".format(str(e)).encode('utf-8'))
                returncode = -999

        if returncode != 0:  # 若执行退出码不为 0
            print("错误：{} 执行失败，返回码={}，详情见日志：{}".format(script_name, returncode, log_path))  # 打印执行失败提示
            return False  # 返回失败
        if script_name == 'Postprocess_All_surface_v2.py':
            status_ok, status_reason = validate_postprocess_status(folder_path)
            if not status_ok:
                print(
                    "错误：{} 的数据提取状态失败，详情={}；保留ODB供诊断。".format(
                        script_name, status_reason,
                    )
                )
                return False
        _print_case_message(
            folder_path, "完成执行：{}".format(script_name),
            case_index=case_index, case_total=case_total,
        )
    return True  # 返回成功


def delete_files_by_type(folder_path, file_types, run_ok):  # 永久删除指定后缀的中间文件以节省空间
    """在流水线通过且规范数据产物完整时清理过程文件，并写出审计。

    参数说明:
        folder_path (str): 工况文件夹绝对路径。
        file_types (list): 待删除的文件后缀名列表。
        run_ok (bool): 工况步骤是否顺利运行完成。

    返回值:
        bool: 全部目标文件删除成功时返回 True；跳过或存在失败时返回 False。
    """
    if not file_types:  # 若未配置删除类型
        return True  # 直接返回

    required_paths = [
        os.path.join(folder_path, name) for name in REQUIRED_RESULT_FILES
    ]
    results_ready = all(
        os.path.isfile(path) and os.path.getsize(path) > 0
        for path in required_paths
    )
    if not (run_ok and results_ready):
        print(
            "警告：由于工况执行失败或规范数据产物不完整，"
            "跳过过程文件清理并完整保留以便诊断。"
        )
        return False

    normalized = {t.lower() if t.startswith(".") else "." + t.lower() for t in file_types}  # 规范化文件后缀名为小写格式
    deleted = {t: 0 for t in normalized}  # 初始化各类型删除文件数计数器
    deleted_files = []
    failed = []  # 初始化删除失败文件记录列表
    retained_results = []
    for path in required_paths:
        retained_results.append({
            'name': os.path.basename(path),
            'size_bytes': os.path.getsize(path),
        })
    for name in sorted(os.listdir(folder_path)):  # 遍历工况目录下的所有文件名
        fp = os.path.join(folder_path, name)  # 拼接绝对路径
        if not os.path.isfile(fp):  # 若非文件结构
            continue  # 跳过处理
        ext = os.path.splitext(name)[1].lower()  # 提取文件后缀名并转换为小写
        if ext not in normalized:  # 若不在待删除列表中
            continue  # 跳过处理
        size_bytes = os.path.getsize(fp)
        try:  # 尝试删除物理文件
            os.remove(fp)  # 永久删除文件
            deleted[ext] += 1  # 计数器加一
            deleted_files.append({
                'name': name, 'extension': ext, 'size_bytes': size_bytes,
            })
        except OSError as exc:  # 若触发系统错误
            failed.append((fp, str(exc)))  # 记录错误信息
    for ext, n in sorted(deleted.items()):  # 遍历打印删除结果统计
        print("已删除 {} 文件数量：{}".format(ext, n))  # 打印各类型文件删除数量
    if failed:  # 若存在删除失败的文件
        print("警告：以下文件删除失败：")  # 打印警告标题
        for fp, err in failed:  # 遍历打印失败记录
            print("  - {} -> {}".format(fp, err))  # 打印失败的物理路径与错误信息
    released_bytes = sum(item['size_bytes'] for item in deleted_files)
    audit = {
        'schema_version': 1,
        'cleaned_at': datetime.datetime.now().isoformat(),
        'case_dir': os.path.abspath(folder_path),
        'status': 'partial' if failed else 'completed',
        'cleanup_file_types': sorted(normalized),
        'retained_cae': True,
        'retained_results': retained_results,
        'deleted_file_count': len(deleted_files),
        'released_bytes': released_bytes,
        'deleted_files': deleted_files,
        'failed_files': [
            {'path': path, 'error': error} for path, error in failed
        ],
    }
    audit_path = os.path.join(folder_path, 'cleanup_audit.json')
    with open(audit_path, 'w', encoding='utf-8') as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(
        "过程文件清理完成：删除{}个文件，释放{:.2f} GiB；审计={}".format(
            len(deleted_files), released_bytes / float(1024 ** 3), audit_path,
        )
    )
    return not failed


def run_folder_pipeline(root_dir, folder_plan, source_files, status_dict,
                        model_run_order, case_post_run_order,
                        types_to_delete):
    """用独立线程池衔接建模与单工况后处理，返回失败目录。"""
    failed_folders = []
    manifest_lock = threading.Lock()
    total_cases = len(folder_plan)
    case_positions = dict(
        (folder_name, index)
        for index, (folder_name, _config) in enumerate(folder_plan, start=1)
    )

    def update_status(folder_name, status):
        with manifest_lock:
            status_dict[folder_name] = status
            _write_run_manifest(
                root_dir, folder_plan, source_files, status_dict,
            )

    def run_model(item):
        folder_name, config = item
        folder_path = os.path.join(root_dir, folder_name)
        case_index = case_positions[folder_name]
        _print_case_message(
            folder_path, "开始建模：{}".format(folder_path),
            case_index=case_index, case_total=total_cases,
        )
        update_status(folder_name, 'model_running')
        try:
            create_and_fill_folder(folder_path, source_files, config)
            ok = run_scripts_in_folder(
                folder_path, model_run_order, step_offset=0,
                stage_label='model', case_index=case_index,
                case_total=total_cases,
            )
        except Exception as err:
            print("异常：建模阶段失败 -> {}".format(str(err)))
            ok = False
        update_status(folder_name, 'model_completed' if ok else 'model_failed')
        return item, folder_path, ok

    def run_case_postprocess(item, folder_path):
        folder_name, _config = item
        case_index = case_positions[folder_name]
        update_status(folder_name, 'postprocess_running')
        _print_case_message(
            folder_path, "开始单工况后处理：{}".format(folder_path),
            case_index=case_index, case_total=total_cases,
        )
        try:
            ok = run_scripts_in_folder(
                folder_path, case_post_run_order,
                step_offset=len(model_run_order),
                stage_label='postprocess', case_index=case_index,
                case_total=total_cases,
            )
        except Exception as err:
            print("异常：单工况后处理失败 -> {}".format(str(err)))
            ok = False
        update_status(
            folder_name, 'data_ready' if ok else 'postprocess_failed',
        )
        if types_to_delete:
            delete_files_by_type(folder_path, types_to_delete, ok)
        else:
            print("已跳过文件删除（没有指定要删除的文件类型）。")
        if ok:
            _print_case_message(
                folder_path, "流水线完成",
                case_index=case_index, case_total=total_cases,
            )
        return folder_path, ok

    print(
        "开始流水线批处理：总工况={}，建模并发={}，单工况后处理并发={}".format(
            total_cases, MAX_WORKERS, POSTPROCESS_WORKERS,
        )
    )
    post_futures = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
    ) as model_executor, concurrent.futures.ThreadPoolExecutor(
        max_workers=POSTPROCESS_WORKERS,
    ) as post_executor:
        model_futures = [
            model_executor.submit(run_model, item) for item in folder_plan
        ]
        for future in concurrent.futures.as_completed(model_futures):
            item, folder_path, ok = future.result()
            if ok:
                post_futures.append(post_executor.submit(
                    run_case_postprocess, item, folder_path,
                ))
            else:
                failed_folders.append(folder_path)
        for future in concurrent.futures.as_completed(post_futures):
            folder_path, ok = future.result()
            if not ok:
                failed_folders.append(folder_path)
    return failed_folders


def main():  # 批处理主控制流程
    """批处理主控制流程。"""
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR  # 支持从命令行参数接收保存目录，否则使用默认值
    types_to_delete = list(DELETE_FILE_TYPES)  # 获取待删除中间文件格式列表的副本
    print("目标根目录：{}".format(root_dir))  # 打印目标根目录
    print("自动删除文件类型：{}".format(types_to_delete if types_to_delete else "无"))  # 打印待删除类型
    source_files, missing_items, duplicate_items = build_source_files(SCRIPT_SEQUENCE)  # 构建复制文件字典并查错
    model_run_order = [os.path.basename(p) for p in MODEL_SCRIPT_SEQUENCE]  # 建模阶段脚本顺序
    case_post_run_order = [os.path.basename(p) for p in CASE_POSTPROCESS_SCRIPT_SEQUENCE]  # 单工况后处理顺序
    if not ensure_sources_exist(missing_items):  # 校验并处理源文件缺失
        sys.exit(1)  # 异常退出
    if not ensure_no_duplicate_targets(duplicate_items):  # 校验并处理复制文件名冲突
        sys.exit(1)  # 异常退出
    if not PARAMETER_CASES:  # 若工况列表为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组工况。")  # 打印工况缺失提示
        sys.exit(1)  # 异常退出
    if not model_run_order:
        print("错误：MODEL_SCRIPT_SEQUENCE 为空，流水线没有建模阶段。")
        sys.exit(1)
    if MAX_WORKERS < 1 or POSTPROCESS_WORKERS < 1:
        print("错误：建模和单工况后处理并发数必须均不小于1。")
        sys.exit(1)

    folder_plan = []  # 初始化工况目录计划列表
    seen = set()  # 初始化工况文件夹去重名字集合
    status_dict = {}  # 用于在清单中保存各个工况的运行状态

    for idx, case in enumerate(PARAMETER_CASES, start=1):  # 遍历所有定义的工况
        if not isinstance(case, dict) or "config" not in case:  # 校验工况节点合法性
            print("错误：第 {} 组工况缺少 config。".format(idx))  # 打印格式错误提示
            sys.exit(1)  # 异常退出
        folder_name = build_folder_name(case)  # 确定工况目录名
        if folder_name in seen:  # 若检测到重复的工况目录名
            print("错误：工况生成了重复文件夹名 -> {}".format(folder_name))  # 打印命名冲突提示
            sys.exit(1)  # 异常退出
        seen.add(folder_name)  # 加入已生成名称集合中
        if _case_already_done(root_dir, folder_name):  # 检查工况是否已有成功的后处理结果
            print("跳过已完成工况：{}".format(folder_name))
            continue
        folder_plan.append((folder_name, case["config"]))  # 记录到待处理工况目录中
        status_dict[folder_name] = 'planned'

    if not folder_plan:
        print("所有工况均已完成，无需重新运行。")
        return

    os.makedirs(root_dir, exist_ok=True)  # 创建结果总输出根目录

    # 写入初始 planned 清单
    _write_run_manifest(root_dir, folder_plan, source_files, status_dict)

    failed_folders = run_folder_pipeline(
        root_dir, folder_plan, source_files, status_dict,
        model_run_order, case_post_run_order, types_to_delete,
    )
    print("\n==============================")  # 打印批处理结束分隔符
    if failed_folders:  # 若存在执行失败的工况
        print("批处理结束：存在失败文件夹（{}个）。".format(len(failed_folders)))  # 打印失败总数统计
        for path in failed_folders:  # 遍历失败的路径列表
            print("  - {}".format(path))  # 打印失败详情
        sys.exit(2)  # 返回状态码2退出系统
    print("批处理结束：全部 {} 个工况文件夹处理完成。".format(len(folder_plan)))  # 打印全部成功提示

    # 全局后处理汇总阶段
    if POST_SCRIPT_SEQUENCE:  # 若配置了后处理汇总脚本
        print("\n==============================")  # 打印后处理分隔符
        print("开始自动后处理脚本阶段...")  # 打印后处理启动提示
        post_run_order = []  # 初始化后处理脚本文件名列表
        for src_path in POST_SCRIPT_SEQUENCE:  # 遍历所有配置的后处理物理路径
            if not os.path.isfile(src_path):  # 若后处理物理文件不存在
                print("错误：后处理脚本缺失 -> {}".format(src_path))  # 打印错误提示
                sys.exit(3)  # 异常退出
            target_name = os.path.basename(src_path)  # 获取后处理脚本文件名
            dst_path = os.path.join(root_dir, target_name)  # 拼接总根目录下目标拷贝路径
            shutil.copy2(src_path, dst_path)  # 复制后处理脚本至根目录下执行以保持环境一致
            post_run_order.append(target_name)  # 记录到后处理文件名列表中
            print("已拷贝后处理脚本：{}".format(target_name))  # 打印拷贝成功提示
        for script_name in post_run_order:  # 遍历执行拷贝完成的脚本文件
            script_path = os.path.join(root_dir, script_name)  # 拼接路径
            print("开始执行后处理：{}".format(script_path))  # 打印执行提示
            result = subprocess.run([sys.executable, script_name, root_dir], cwd=root_dir, check=False)  # 在根目录下用当前解释器运行并传入根目录参数
            if result.returncode != 0:  # 若执行不成功
                print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))  # 打印错误提示
                sys.exit(4)  # 异常退出
            print("完成后处理：{}".format(script_name))  # 打印完成提示


if __name__ == "__main__":  # 判断为主入口运行
    main()  # 执行批处理流程
