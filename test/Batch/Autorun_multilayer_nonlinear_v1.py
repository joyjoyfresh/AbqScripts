# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""论文图15（双层软/硬表层）复刻【验证】Autorun —— 配置注入方式，含"时窗修复"。

本脚本仿照 Batch/Autorun_TAF_multilayer_v3.py 编写，专门用于复刻 Shen 等(2025) 图15：
  双层模型 i=45°、Vr/Vs2=2.5(Vs2=800)、h/H=0.5、a0=2.0(fc=4Hz Ricker)，
  软表层 Vs1/Vs2=0.5(velocity_ratio=5.0) / 硬表层 Vs1/Vs2=2.0(velocity_ratio=1.25)，
  表层相对厚度 h1/(H-h)=0.25(h1=50m) / 0.75(h1=150m)，入射角 0°/15° —— 共 8 工况。

调用建模脚本 Modeling/Multi/VAB_oblique_multilayer_nonlinear_v3.py（读取 case_config.json 覆盖默认配置）。

────────────────────────────────────────────────────────────────────────────
【与旧验证脚本的关键差异 —— 三处"按诊断结论修复"的全局注入（见 COMMON_OVERRIDES）】
  之前 test-4hz 的软层工况复刻失败（软斜入射峰值 TAF_h≈2.0，论文≈7.6）。诊断结论：
  FE 远场平台已精确等于解析一维台阶(1.70)，但近坡顶的【二维陷波/瑞利波放大缺失】——
  根因是【模拟时窗太短】(只有输入记录 2.0s，tail=0)，软层慢波(Vs1=400, 瑞利≈370m/s)
  的陷波/沿面瑞利波来不及发育就被截断。硬层是快波、2.0s 内已发育完，故硬层/均质图能复刻。

  ① time_cfg.tail_seconds = 6.0  —— 在输入记录后追加静默尾段，让软层陷波/瑞利波充分发育、
     捕到迟到的峰值。建模脚本会同步延长分析步时长与 fd 自由场时窗（见建模脚本 §build_models）。
  ② freefield_cfg.pad_factor = 8 —— FFT 补零倍数加大，使 fd 自由场时窗(Nout)覆盖 2+6=8s。
  ③ damping_cfg.constant_xi = None —— 关掉默认的统一 ξ=0.01(Q=50)，改用论文口径 Qs=0.05·Vs
     (软表层 Vs1=400→Qs=20→ξ=0.025；覆盖层 Vs2=800→Qs=40→ξ=0.0125；基岩 Q=999)。

  注意（绝对值口径）：本链沿用 Compute_TAF_v2（分母=基岩露头解析自由场 factor_h·PGA_in）。
  时窗修复后软层峰值会大幅回升、与论文趋势一致；但若要与论文图15【绝对值】严格对齐，
  还需把 TAF 分母改为"局部一维分层自由场"（论文式(5) 的 Assimaki 地形口径，远场归一到 1）——
  那是 Compute_TAF 的另一处后处理改动，不在本 Autorun 范围内。
────────────────────────────────────────────────────────────────────────────

每个工况文件夹内执行顺序：建模 → 提取 PGA → 计算 TAF。
跑完后再 Collect 汇总各工况 case_meta.json，最后 Plot_Fig15_compare 出图与论文对比。
"""

import os  # 导入操作系统路径与目录模块
import json  # 导入 JSON 模块用于写出 case_config.json
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器
import concurrent.futures  # 导入并发模块以实现多文件夹并行执行

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置目标模型根目录（各工况文件夹建在此；可由命令行参数覆盖）
FOLDER_PREFIX = "multi-"  # 设置目标文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 每个文件夹脚本执行后要直接删除的文件类型；为空则不删除
MAX_WORKERS = 2  # 并行处理文件夹的最大线程数（注意：时窗拉长到 8s 后单作业耗时约 4×，按机器算力酌情调）
CONFIG_FILENAME = "case_config.json"  # 注入给建模脚本的配置文件名（建模脚本会读取它）

# ── 时窗修复全局参数（集中在此便于调/复原；置 None 可关闭对应修复回到旧行为） ──
TAIL_SECONDS = 6.0     # ①静默尾段秒数：让软层陷波/瑞利波发育完。None=不延长(回到旧 2.0s 行为)
PAD_FACTOR = 8         # ②fd 自由场 FFT 补零倍数：须保证 Nout≥(记录+尾段)。None=用建模脚本默认(4)
USE_PAPER_Q = True     # ③True=用论文 Qs=0.05·Vs(constant_xi=None)；False=保持脚本默认统一 ξ=0.01

# 固定源文件（随每个工况文件夹拷入）：仅输入波 .txt（建模脚本自包含写出 case_meta.json，无需拷模块）
STATIC_SOURCE_PATHS = [  # 定义固定源文件完整路径列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4 Hz Ricker（a0=2.0 @ Vs2=800，图15 工况）
]  # 结束固定源文件完整路径定义

# 每个工况文件夹按顺序执行的脚本（绝对路径）
SCRIPT_SEQUENCE = [  # 定义脚本顺序配置列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模脚本 v3（读取 case_config.json）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算（基岩露头分母，见顶部口径说明）
]  # 结束脚本顺序配置定义

# 全部工况求解完成后自动执行的后处理脚本（绝对路径）
POST_SCRIPT_SEQUENCE = [  # 汇总与跨工况出图脚本
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py",  # 汇总各工况 case_meta.json 到 results/index.csv
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_Fig15_compare_v3.py",  # 三层/双层图15 出图与论文对比
]

# ============================================================
#  全局注入（每个工况都叠加）：时窗修复三件套。case 专属 config 会再深合并到其上。
# ============================================================
def _build_common_overrides():  # 由顶部开关构造全局注入字典（仅放开启的项）
    """按 TAIL_SECONDS/PAD_FACTOR/USE_PAPER_Q 三个开关，生成要叠加到每个工况的 config 覆盖。"""
    common = {}  # 初始化全局覆盖
    if TAIL_SECONDS is not None:  # ①延长时窗
        common.setdefault("time_cfg", {})["tail_seconds"] = float(TAIL_SECONDS)  # 注入静默尾段秒数
    if PAD_FACTOR is not None:  # ②加大 FFT 补零
        common.setdefault("freefield_cfg", {})["pad_factor"] = int(PAD_FACTOR)  # 注入 fd 补零倍数
    if USE_PAPER_Q:  # ③用论文 Qs=0.05·Vs
        common.setdefault("damping_cfg", {})["constant_xi"] = None  # 关掉统一 ξ，回到按波速算 Q 的支路
    return common  # 返回全局覆盖


COMMON_OVERRIDES = _build_common_overrides()  # 模块级全局注入（所有工况共享）


# 双层(图15)层模板：surface(表层 Vs1) + overlying(覆盖层 Vs2)。velocity_ratio = Vr/Vs。
def _layers2(surf_vr, surf_thick):  # 生成双层 layers（表层软硬/厚度可变 + 覆盖层）
    """surf_vr: 表层 velocity_ratio(Vr/Vs1，软=5.0→Vs1=400 / 硬=1.25→Vs1=1600)；surf_thick: 表层厚度 h1(m)。"""
    return [  # 从上到下：表层 + 覆盖层
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},  # 表层(固定厚度)
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},  # 覆盖层 Vr/Vs2=2.5(Vs2=800)，厚度由几何定
    ]  # 结束双层 layers


PARAMETER_CASES = [  # 论文图15：i=45 固定，软/硬 × 厚度(0.25/0.75) × 入射角(0/15) = 8 工况
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers2(5.0,  50.0)},  "geometry_cfg": {"i": 45.0}}},   # D01 软 Vs1/Vs2=0.5, h1/(H-h)=0.25, 0°
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers2(1.25, 50.0)},  "geometry_cfg": {"i": 45.0}}},   # D17 硬 Vs1/Vs2=2.0, 0.25, 0°
    {"config": {"material_cfg": {"angle": 15, "layers": _layers2(5.0,  50.0)},  "geometry_cfg": {"i": 45.0}}},   # D05 软, 0.25, 15°
    {"config": {"material_cfg": {"angle": 15, "layers": _layers2(1.25, 50.0)},  "geometry_cfg": {"i": 45.0}}},   # D21 硬, 0.25, 15°
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers2(5.0,  150.0)}, "geometry_cfg": {"i": 45.0}}},   # D03 软, 0.75, 0°
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers2(1.25, 150.0)}, "geometry_cfg": {"i": 45.0}}},   # D19 硬, 0.75, 0°
    {"config": {"material_cfg": {"angle": 15, "layers": _layers2(5.0,  150.0)}, "geometry_cfg": {"i": 45.0}}},   # D07 软, 0.75, 15°（论文最强放大，TAF_h≈7.6）
    {"config": {"material_cfg": {"angle": 15, "layers": _layers2(1.25, 150.0)}, "geometry_cfg": {"i": 45.0}}},   # D23 硬, 0.75, 15°
]  # 结束变参数工况列表定义


def _deep_merge(base, override):  # dict 逐键递归合并；其余类型(含 list，如 layers)整体替换
    """与建模脚本同口径：双方均为 dict 才递归合并，否则 override 整体替换/新增。返回新 dict。"""
    out = dict(base)  # 复制基底，避免就地修改
    for k, v in (override or {}).items():  # 遍历覆盖项
        if isinstance(v, dict) and isinstance(out.get(k), dict):  # 双方均 dict → 递归
            out[k] = _deep_merge(out[k], v)  # 递归合并
        else:  # 其余 → 整体替换
            out[k] = v  # 替换/新增
    return out  # 返回合并结果


def merged_case_config(case_config):  # 把全局时窗修复叠加到工况专属 config 上
    """先放全局 COMMON_OVERRIDES，再深合并工况专属 config（专属项优先；二者键互不冲突即各自生效）。"""
    return _deep_merge(COMMON_OVERRIDES, case_config or {})  # 全局打底 + 工况覆盖


def build_source_files(static_source_paths, script_sequence):  # 构建源文件名→源路径映射并检测缺失/重名
    """返回 (source_files 映射, missing 缺失列表, duplicate_names 重名冲突列表)。"""
    source_files = {}  # 初始化源文件映射
    missing = []  # 初始化缺失列表
    duplicate_names = []  # 初始化重名冲突列表
    for source_path in list(static_source_paths) + list(script_sequence):  # 合并固定源与脚本统一处理
        target_name = os.path.basename(source_path)  # 目标文件名
        if not os.path.isfile(source_path):  # 源文件不存在
            missing.append((target_name, source_path)); continue  # 记录缺失并跳过
        if target_name in source_files and source_files[target_name] != source_path:  # 同名不同源冲突
            duplicate_names.append((target_name, source_path, source_files[target_name])); continue  # 记录冲突并跳过
        source_files[target_name] = source_path  # 写入映射
    return source_files, missing, duplicate_names  # 返回结果


def ensure_sources_exist(missing_items):  # 检查缺失源文件
    """有缺失则打印并返回 False。"""
    if not missing_items:  # 无缺失
        return True  # 检查通过
    print("错误：以下源文件不存在，请先检查配置路径：")  # 错误标题
    for name, path in missing_items:  # 遍历缺失
        print("  - {} -> {}".format(name, path))  # 打印明细
    return False  # 检查失败


def ensure_no_duplicate_targets(duplicate_items):  # 检查重名目标冲突
    """有冲突则打印并返回 False。"""
    if not duplicate_items:  # 无冲突
        return True  # 检查通过
    print("错误：以下文件复制后会重名，请调整路径或文件名：")  # 错误标题
    for target_name, current_path, existing_path in duplicate_items:  # 遍历冲突
        print("  - {} -> {} (已存在来源: {})".format(target_name, current_path, existing_path))  # 打印明细
    return False  # 检查失败


def _fmt_num(v):  # 把数值格式化为简洁字符串（去掉无意义的 .0）
    """45.0→'45'、1.25→'1.25'、150.0→'150'；非数值原样转字符串。"""
    try:  # 尝试按数值处理
        f = float(v)  # 转浮点
        return str(int(f)) if f == int(f) else ('%g' % f)  # 整数去小数点，否则用 %g 紧凑表示
    except (TypeError, ValueError):  # 非数值
        return str(v)  # 原样转字符串


def _sanitize(text):  # 规范化为合法文件夹名片段
    """非 [0-9A-Za-z._-] 的字符替换为连字符，并去除两端连字符。"""
    import re  # 局部导入正则
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-") or "x"  # 清洗并兜底


def name_from_config(config):  # 由注入配置自动生成可读、唯一的工况名（避免手写出错）
    """从 config 关键参数拼出文件夹名：层结构(L{n}_vr..t..)、入射角 a、几何 i 等、网格 m。

    命名只为人类可读 + 区分工况；工况真实身份以各文件夹 case_meta.json 为准（下游分组靠它）。
    只编码 material_cfg/geometry_cfg/mesh_size，故全局时窗修复(time/ff/damping)不参与命名、不影响文件夹名。
    """
    mat = config.get("material_cfg") or {}  # 材料覆盖
    geo = config.get("geometry_cfg") or {}  # 几何覆盖
    tokens = []  # 名称片段
    layers = mat.get("layers")  # 有限层列表（若覆盖了）
    if isinstance(layers, list):  # 编码层结构（能区分单/双/三层与各层波速比、厚度）
        segs = []  # 各层片段
        for L in layers:  # 自上而下遍历有限层
            seg = "vr" + _fmt_num(L.get("velocity_ratio"))  # 该层相对波速比
            if L.get("thickness") is not None:  # 有固定厚度则附加
                seg += "t" + _fmt_num(L["thickness"])  # 厚度片段
            segs.append(seg)  # 收集
        tokens.append("L%d_%s" % (len(layers), "-".join(segs)))  # L{层数}_{各层}
    if "angle" in mat:  # 入射角
        tokens.append("a" + _fmt_num(mat["angle"]))  # a{angle}
    for key, pre in (("i", "i"), ("H_minus_h", "H"), ("h_over_H", "hoH"),  # 几何关键键 → 短前缀
                     ("bedrock_thickness", "br"), ("total_L", "L"), ("left_flat", "lf")):  # 续
        if key in geo:  # 覆盖了该几何键
            tokens.append(pre + _fmt_num(geo[key]))  # 追加片段
    if config.get("mesh_size") is not None:  # 网格尺寸（旧写法）
        tokens.append("m" + _fmt_num(config["mesh_size"]))  # m{mesh}
    return "-".join(tokens) if tokens else "default"  # 拼接（全空则用 default）


def build_folder_name(case):  # 生成工况文件夹名（优先手填 name/folder_tag，否则按 config 自动取名）
    """优先用 case 显式给的 name 或 folder_tag；都没有则由【工况专属 config】自动生成（不含全局修复）。"""
    tag = case.get("name") or case.get("folder_tag")  # 可选的手填标签（向后兼容）
    if not tag:  # 未手填 → 自动取名（用专属 config，保证与旧验证目录名一致、便于对比）
        tag = name_from_config(case.get("config") or {})  # 由专属配置生成
    return "{}{}".format(FOLDER_PREFIX, _sanitize(tag))  # 前缀 + 规范化标签


def create_and_fill_folder(folder_path, source_files, config):  # 创建并填充单个工况目录
    """新建目录、拷入固定源与脚本、写出 case_config.json（已叠加全局时窗修复的最终 config）。"""
    os.makedirs(folder_path, exist_ok=True)  # 创建目录
    for target_name, src_path in source_files.items():  # 遍历所有源文件
        shutil.copy2(src_path, os.path.join(folder_path, target_name))  # 拷入并保留元数据
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:  # 写出注入配置
        json.dump(config or {}, f, ensure_ascii=False, indent=2)  # 序列化最终 config


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


def delete_files_by_type(folder_path, file_types):  # 直接删除目录下指定类型的文件
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
            os.remove(fp)  # 直接永久删除文件
            deleted[ext] += 1  # 删除计数加一
        except OSError as exc:  # 系统异常
            failed.append((fp, str(exc)))  # 记录失败
    for ext, n in sorted(deleted.items()):  # 汇总各类型
        print("已删除 {} 文件数量：{}".format(ext, n))  # 打印数量
    if failed:  # 有失败
        print("警告：以下文件删除失败：")  # 警告标题
        for fp, err in failed:  # 遍历失败
            print("  - {} -> {}".format(fp, err))  # 打印明细


def main():  # 主控制流程
    """组织源文件 → 规划各工况文件夹 → 并行建模/后处理 → 清理 → 汇总出图。"""
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR  # 根目录：命令行参数或默认
    types_to_delete = list(DELETE_FILE_TYPES)  # 待删除的文件类型副本
    print("目标根目录：{}".format(root_dir))  # 打印根目录
    print("全局时窗修复注入：{}".format(json.dumps(COMMON_OVERRIDES, ensure_ascii=False)))  # 打印全局修复（便于核对生效）
    print("要直接删除的文件类型：{}".format(types_to_delete if types_to_delete else "无"))  # 打印待删除类型
    source_files, missing_items, duplicate_items = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)  # 构建源映射
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]  # 执行顺序（文件名）
    if not ensure_sources_exist(missing_items):  # 校验源文件存在
        sys.exit(1)  # 缺失则退出
    if not ensure_no_duplicate_targets(duplicate_items):  # 校验无重名冲突
        sys.exit(1)  # 冲突则退出
    if not PARAMETER_CASES:  # 工况表为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组工况。"); sys.exit(1)  # 报错退出
    # 规划各工况：文件夹名去重 + 关联其【已叠加全局修复】的最终配置
    folder_plan = []  # 文件夹计划
    seen = set()  # 文件夹名去重集合
    for idx, case in enumerate(PARAMETER_CASES, start=1):  # 遍历工况
        if not isinstance(case, dict) or "config" not in case:  # 校验工况结构
            print("错误：第 {} 组工况缺少 config。".format(idx)); sys.exit(1)  # 报错退出
        folder_name = build_folder_name(case)  # 生成文件夹名（仅用专属 config）
        if folder_name in seen:  # 重名
            print("错误：工况生成了重复文件夹名 -> {}".format(folder_name)); sys.exit(1)  # 报错退出
        seen.add(folder_name)  # 记录文件夹名
        folder_plan.append((folder_name, merged_case_config(case["config"])))  # 记录(文件夹, 全局修复+专属配置)
    os.makedirs(root_dir, exist_ok=True)  # 确保根目录存在
    failed_folders = []  # 失败目录列表

    def process_folder(item):  # 处理单个工况文件夹的工作函数
        """item=(folder_name, config)；返回 (folder_path, ok)。"""
        folder_name, config = item  # 解包
        folder_path = os.path.join(root_dir, folder_name)  # 文件夹完整路径
        print("\n==============================")  # 分隔符
        print("开始处理文件夹：{}".format(folder_path))  # 当前文件夹
        create_and_fill_folder(folder_path, source_files, config)  # 创建并注入配置
        ok = run_scripts_in_folder(folder_path, run_order)  # 顺序执行脚本
        if types_to_delete:  # 需清理
            delete_files_by_type(folder_path, types_to_delete)  # 直接永久删除中间文件
        else:  # 不清理
            print("已跳过文件删除（没有指定要删除的文件类型）。")  # 提示
        return folder_path, ok  # 返回结果

    print("开始并行批处理，最大并发任务数：{}".format(MAX_WORKERS))  # 启动提示
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

    # 后处理阶段：拷贝后处理脚本到根目录并顺序执行
    print("\n==============================")  # 后处理分隔符
    print("开始自动后处理脚本阶段...")  # 阶段标题
    post_run_order = []  # 收集后处理脚本文件名
    for src_path in POST_SCRIPT_SEQUENCE:  # 遍历后处理脚本
        if not os.path.isfile(src_path):  # 源脚本不存在
            print("错误：后处理脚本缺失 -> {}".format(src_path))  # 报错
            sys.exit(3)  # 退出
        target_name = os.path.basename(src_path)  # 目标文件名
        dst_path = os.path.join(root_dir, target_name)  # 目标路径
        shutil.copy2(src_path, dst_path)  # 拷贝到根目录
        post_run_order.append(target_name)  # 记录执行顺序
        print("已拷贝后处理脚本：{}".format(target_name))  # 确认拷贝

    for script_name in post_run_order:  # 按顺序执行后处理脚本
        script_path = os.path.join(root_dir, script_name)  # 脚本完整路径
        print("开始执行后处理：{}".format(script_path))  # 开始日志
        result = subprocess.run([sys.executable, script_name, root_dir], cwd=root_dir, check=False)  # 在根目录执行并传入目录参数
        if result.returncode != 0:  # 执行失败
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))  # 报错
            sys.exit(4)  # 后处理失败退出
        print("完成后处理：{}".format(script_name))  # 完成日志


if __name__ == "__main__":  # 主入口
    main()  # 运行主流程
