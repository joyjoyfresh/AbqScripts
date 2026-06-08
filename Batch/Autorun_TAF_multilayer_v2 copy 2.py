# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""批量跑【多种变参数工况】的合并版 Autorun（配置注入方式）。

合并了原 Autorun_TAF_multilayer_v1（三层图15 扫描）与 Autorun_TAF_multilayer_verify_v1（双层 i/angle 扫描），
统一调用合并版建模脚本 Modeling/Multi/VAB_oblique_TAF_multilayer_v4.py。

与旧版"正则替换标量"不同，本版用【配置注入】：每个工况在 PARAMETER_CASES 里给一份 config 覆盖
（material_cfg/geometry_cfg/mesh_size 的部分或全部），脚本把它写进工况文件夹的 case_config.json，
建模脚本 v4 运行时读取并覆盖默认配置。于是一个批处理即可任意改：
  层数(单/双/三层，靠 material_cfg.layers 列表长度)、各层波速比/泊松比/密度/厚度、
  几何(坡角 i、坡高 H_minus_h、深度比 h_over_H、覆盖层/基岩厚、总长、平台长)、入射角 angle、网格 mesh_size。

每个工况文件夹内执行顺序：建模(v4) → 提取 PGA → 计算 TAF → 每文件夹出图。
跑完后再用 Postprocess/General/Collect_results_v2.py 汇总各工况 case_meta.json 到 results/index.csv，
最后用 Postprocess/Multi/Plot_Multi_TAF_v4.py 跨工况出图（论文图8 排版）。
"""

import os  # 导入操作系统路径与目录模块
import json  # 导入 JSON 模块用于写出 case_config.json
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器
import concurrent.futures  # 导入并发模块以实现多文件夹并行执行

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置目标模型根目录（各工况文件夹建在此）
FOLDER_PREFIX = "multi-"  # 设置目标文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 每个文件夹脚本执行后要直接删除的文件类型；为空则不删除
MAX_WORKERS = 4  # 并行处理文件夹的最大线程数
CONFIG_FILENAME = "case_config.json"  # 注入给建模脚本的配置文件名（建模脚本 v4 会读取它）

# 固定源文件（随每个工况文件夹拷入）：仅输入波 .txt（建模脚本已自包含写出 case_meta.json，无需再拷模块）
STATIC_SOURCE_PATHS = [  # 定义固定源文件完整路径列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4 Hz Ricker 输入波（a0=2.0 @ Vs2=800）
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_6Hz.txt",  # 6 Hz Ricker 输入波
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_8Hz.txt",  # 8 Hz Ricker 输入波
]  # 结束固定源文件完整路径定义

# 每个工况文件夹按顺序执行的脚本（路径已对齐目录整理后的新位置）
SCRIPT_SEQUENCE = [  # 定义脚本顺序配置列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_TAF_multilayer_v4.py",  # 合并版建模脚本（读取 case_config.json，自包含写出 case_meta.json）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v1.py",  # TAF 计算
]  # 结束脚本顺序配置定义

# ============================================================
#  变参数工况表：每项只需写 {config: 注入给建模脚本的配置覆盖}。
#  文件夹名【自动由 config 生成】(见 name_from_config)，无需手写、不会写错；
#    如确需自定义名字，可加可选键 "name": "你的名字" 覆盖自动命名。
#  config 内 material_cfg/geometry_cfg/mesh_size 均可只写要改的键（部分覆盖）；
#  改 layers 列表即可切换单/双/三层（列表整体替换默认 layers）。
#  下表给出两类示例：A) 三层图15 软/硬×厚度×角度；B) 双层 i/angle 扫描。按需增删。
# ============================================================
# 三层(图15)层模板：surface(表层 Vs1) + overlying(覆盖层 Vs2)。velocity_ratio = Vr/Vs。
def _layers3(surf_vr, surf_thick):  # 生成三层 layers（表层软硬/厚度可变）
    """surf_vr: 表层 velocity_ratio(Vr/Vs1，软=5.0/硬=1.25)；surf_thick: 表层厚度 h1(m)。"""
    return [  # 从上到下：表层 + 覆盖层
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},  # 表层(固定厚度)
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},  # 覆盖层 Vr/Vs2=2.5(Vs2=800)，厚度由几何定
    ]  # 结束三层 layers


PARAMETER_CASES = [  # 定义变参数工况列表（文件夹名自动由 config 生成）
    # ---- B) 双层（沿用默认 overlying，仅扫坡角与入射角）= 4 工况 ----
    {"config": {"material_cfg": {"angle": 0},  "geometry_cfg": {"i": 30.0}}},  # 双层, i=30, 0°
    {"config": {"material_cfg": {"angle": 15}, "geometry_cfg": {"i": 30.0}}},  # 双层, i=30, 15°
]  # 结束变参数工况列表定义


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


def name_from_config(config):  # 由注入配置自动生成可读、唯一的工况名（避免手写 folder_tag 出错）
    """从 config 的关键参数拼出文件夹名后缀：层结构(L{n}_vr..t..)、入射角 a、几何 i/H/hoH 等、网格 m。

    命名只为"人类可读 + 区分工况"；工况真实身份以各文件夹的 case_meta.json 为准（下游分组按它，不靠文件夹名）。
    覆盖了哪些键就编码哪些键，故两个不同 config 一般得到不同名字；万一仍重名，main() 会报错提示。
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
    if config.get("mesh_size") is not None:  # 网格尺寸
        tokens.append("m" + _fmt_num(config["mesh_size"]))  # m{mesh}
    # 兜底：把未被上面专门编码、却被覆盖的其余标量项也并入，确保不同 config 名字不同
    for scope, skip in ((mat, {"angle", "layers"}), (geo, {"i", "H_minus_h", "h_over_H", "bedrock_thickness", "total_L", "left_flat"})):
        for k, v in scope.items():  # 遍历该作用域
            if k in skip or isinstance(v, (dict, list)):  # 跳过已编码键与嵌套结构
                continue  # 继续
            tokens.append("%s%s" % (_sanitize(k), _fmt_num(v)))  # 通用片段 键+值
    return "-".join(tokens) if tokens else "default"  # 拼接（全空则用 default）


def build_folder_name(case):  # 生成工况文件夹名（优先手填 name/folder_tag，否则按 config 自动取名）
    """优先用 case 显式给的 name 或 folder_tag；都没有则由 config 自动生成。统一加 FOLDER_PREFIX 前缀。"""
    tag = case.get("name") or case.get("folder_tag")  # 可选的手填标签（向后兼容）
    if not tag:  # 未手填 → 自动取名
        tag = name_from_config(case.get("config") or {})  # 由配置生成
    return "{}{}".format(FOLDER_PREFIX, _sanitize(tag))  # 前缀 + 规范化标签


def create_and_fill_folder(folder_path, source_files, config):  # 创建并填充单个工况目录
    """新建目录、拷入固定源与脚本、写出 case_config.json（注入配置）。"""
    os.makedirs(folder_path, exist_ok=True)  # 创建目录
    for target_name, src_path in source_files.items():  # 遍历所有源文件
        shutil.copy2(src_path, os.path.join(folder_path, target_name))  # 拷入并保留元数据
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:  # 写出注入配置
        json.dump(config or {}, f, ensure_ascii=False, indent=2)  # 序列化 config（缺省为空，建模脚本用默认配置）


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
    """按扩展名永久删除中间文件（不放回收站、不移入垃圾桶）；file_types 为空则跳过。"""
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
    """组织源文件 → 规划各工况文件夹 → 并行建模/后处理 → 清理。"""
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR  # 根目录：命令行参数或默认
    types_to_delete = list(DELETE_FILE_TYPES)  # 待删除的文件类型副本
    print("目标根目录：{}".format(root_dir))  # 打印根目录
    print("要直接删除的文件类型：{}".format(types_to_delete if types_to_delete else "无"))  # 打印待删除类型
    source_files, missing_items, duplicate_items = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)  # 构建源映射
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]  # 执行顺序（文件名）
    if not ensure_sources_exist(missing_items):  # 校验源文件存在
        sys.exit(1)  # 缺失则退出
    if not ensure_no_duplicate_targets(duplicate_items):  # 校验无重名冲突
        sys.exit(1)  # 冲突则退出
    if not PARAMETER_CASES:  # 工况表为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组工况。"); sys.exit(1)  # 报错退出
    # 规划各工况：文件夹名去重 + 关联其注入配置
    folder_plan = []  # 文件夹计划
    seen = set()  # 文件夹名去重集合
    for idx, case in enumerate(PARAMETER_CASES, start=1):  # 遍历工况
        if not isinstance(case, dict) or "config" not in case:  # 校验工况结构
            print("错误：第 {} 组工况缺少 config。".format(idx)); sys.exit(1)  # 报错退出
        folder_name = build_folder_name(case)  # 生成文件夹名
        if folder_name in seen:  # 重名
            print("错误：工况生成了重复文件夹名 -> {}".format(folder_name)); sys.exit(1)  # 报错退出
        seen.add(folder_name)  # 记录文件夹名
        folder_plan.append((folder_name, case["config"]))  # 记录(文件夹, 配置)
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
        print("批处理结束：存在失败文件夹。")  # 失败标题
        for path in failed_folders:  # 遍历失败
            print("  - {}".format(path))  # 打印路径
        sys.exit(2)  # 异常退出
    print("批处理结束：全部文件夹处理完成。")  # 全部成功
    print("提示：随后运行 Postprocess/General/Collect_results_v2.py 汇总，再用 Postprocess/Multi/Plot_Multi_TAF_v4.py 跨工况出图。")  # 后续步骤提示


if __name__ == "__main__":  # 主入口
    main()  # 运行主流程
