# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软楔 2D 面波共振【收敛性验证】批处理（配置注入方式）。

目的：检验"软表层放大偏低"是不是 CPE4R 线性单元的数值频散 / 时窗截断造成的，
即论文图18 那种"软层多次反射 + 坡顶瑞利波多径上行叠加"在 Abaqus 里到底能不能堆起来。

做法：固定同一软薄层工况(表层 Vs1/Vs2=0.5, h1=50m, 斜入射15°, 坡角 i=45°, 8Hz Ricker)，
只单独变三件事，看坡顶峰值放大系数是否随之上爬：
  ① 网格尺寸 size：4→2→1m（CPE4R 每波长单元数 ~11→23→46，直接测数值频散）；
  ② 单元类型 elem：CPE4R(减缩积分) vs CPE4(全积分)（单元敏感性）；
  ③ 静默尾段 tail_seconds：0→4s（让后到的慢瑞利波/混响发育完整，不被时窗截断）。

判据：
  - 峰值随网格加密(②m4→m2→m1)明显上爬且未收敛 → 数值频散在压共振，加密/升阶可救；
  - 峰值随尾段(tail0→tail4)上爬 → 之前是混响被截断；
  - 都纹丝不动 → 才指向方法天花板（局部黏弹性边界+等效力 撑不起该 2D 共振）。

单元：CPE4R(线性减缩) / CPE4(线性全积分) / CPE8R(二次减缩,低频散) —— 由建模脚本 v3 支持，
      二次单元边界自动改用一致权重(角:中=1/6:2/3)以保持远场=一维理论。
      CPE8R@4m 的有效分辨率≈CPE4R@2m，是测"数值频散是否压低软楔放大"的最直接对照。

每个工况文件夹内执行顺序：建模 → 提取 PGA → 计算 TAF。跑完汇总 case_meta 到 results/index.csv。
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
MAX_WORKERS = 1  # 并行处理文件夹的最大线程数（收敛工况网格细、内存大，建议串行=1，避免 m1 与他案同时占内存）
CONFIG_FILENAME = "case_config.json"  # 注入给建模脚本的配置文件名

# 固定源文件（随每个工况文件夹拷入）：仅输入波 .txt
STATIC_SOURCE_PATHS = [  # 定义固定源文件完整路径列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_8Hz.txt",  # 8Hz 文件(fc≈4Hz)，与已有软层 8Hz 基线同源，便于直接对比
]  # 结束固定源文件完整路径定义

# 每个工况文件夹按顺序执行的脚本
SCRIPT_SEQUENCE = [  # 定义脚本顺序配置列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模脚本 v3（支持 CPE8R 二次单元 + 边界一致权重）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算
]  # 结束脚本顺序配置定义

# 全部工况求解完成后自动执行的后处理脚本
POST_SCRIPT_SEQUENCE = [  # 汇总脚本（收敛工况无图15结构，故只汇总不出图）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py",  # 汇总各工况 case_meta.json 到 results/index.csv
]


# 三层(图15)层模板：surface(表层 Vs1) + overlying(覆盖层 Vs2)。velocity_ratio = Vs_bedrock/Vs_layer。
def _layers3(surf_vr, surf_thick):  # 生成三层 layers（表层软硬/厚度可变）
    """surf_vr: 表层 velocity_ratio(软=5.0→Vs1=400/硬=1.25→Vs1=1600)；surf_thick: 表层厚度 h1(m)。"""
    return [  # 从上到下：表层 + 覆盖层
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},  # 表层(固定厚度)
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},  # 覆盖层 Vr/Vs2=2.5(Vs2=800)，厚度由几何定
    ]  # 结束三层 layers


def _soft_base():  # 软薄层主工况的材料/几何(各收敛工况共用，只在外面追加 mesh_cfg/time_cfg)
    """表层 Vs1/Vs2=0.5(软), h1=50m, 斜入射15°, 坡角 i=45°。每次返回全新 dict，避免别名共享。"""
    return {"material_cfg": {"angle": 15, "layers": _layers3(5.0, 50.0)}, "geometry_cfg": {"i": 45.0}}


PARAMETER_CASES = [  # 收敛工况：同一软薄层，只单独变 网格/单元/尾段
    # 名称必须显式给(否则只编码 material/geometry，几个工况会重名冲突)。
    # ① 基线(=当前默认行为)：4m / CPE4R / 无尾段
    {"config": dict(_soft_base(), mesh_cfg={"size": 4.0, "elem": "CPE4R"}, time_cfg={"tail_seconds": 0.0}),
     "name": "conv-soft-m4-CPE4R-tail0"},
    # ② 只加尾段：4m / CPE4R / tail=4s —— 对比①隔离"时窗截断"效应
    {"config": dict(_soft_base(), mesh_cfg={"size": 4.0, "elem": "CPE4R"}, time_cfg={"tail_seconds": 4.0}),
     "name": "conv-soft-m4-CPE4R-tail4"},
    # ③ 网格加密×2：2m / CPE4R / tail=4s —— 对比②测数值频散
    {"config": dict(_soft_base(), mesh_cfg={"size": 2.0, "elem": "CPE4R"}, time_cfg={"tail_seconds": 4.0}),
     "name": "conv-soft-m2-CPE4R-tail4"},
    # ④ 网格加密×4：1m / CPE4R / tail=4s —— 收敛点【较重，单元数大，跑得慢；嫌慢可注释本行】
    {"config": dict(_soft_base(), mesh_cfg={"size": 1.0, "elem": "CPE4R"}, time_cfg={"tail_seconds": 4.0}),
     "name": "conv-soft-m1-CPE4R-tail4"},
    # ⑤ 单元敏感性：2m / CPE4(全积分) / tail=4s —— 对比③看减缩 vs 全积分
    {"config": dict(_soft_base(), mesh_cfg={"size": 2.0, "elem": "CPE4"}, time_cfg={"tail_seconds": 4.0}),
     "name": "conv-soft-m2-CPE4-tail4"},
    # ⑥ 二次单元(低频散)：4m / CPE8R / tail=4s —— 与①②同网格对照，CPE8R@4m≈CPE4R@2m，最直接测频散
    {"config": dict(_soft_base(), mesh_cfg={"size": 4.0, "elem": "CPE8R"}, time_cfg={"tail_seconds": 4.0}),
     "name": "conv-soft-m4-CPE8R-tail4"},
    # ⑦ 二次单元加密：2m / CPE8R / tail=4s —— 与③(CPE4R@2m)对照，看升阶是否还能再抬
    {"config": dict(_soft_base(), mesh_cfg={"size": 2.0, "elem": "CPE8R"}, time_cfg={"tail_seconds": 4.0}),
     "name": "conv-soft-m2-CPE8R-tail4"},
]  # 结束收敛工况列表定义


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


def build_folder_name(case):  # 生成工况文件夹名（本批处理一律要求显式 name）
    """各收敛工况只差 mesh_cfg/time_cfg(不参与自动命名)，故必须显式给 name，否则会重名冲突。"""
    tag = case.get("name")  # 取显式名称
    if not tag:  # 未给名称
        raise ValueError("收敛工况必须显式提供 'name'（否则文件夹会重名）")  # 直接报错提醒
    return "{}{}".format(FOLDER_PREFIX, tag)  # 前缀 + 名称


def create_and_fill_folder(folder_path, source_files, config):  # 创建并填充单个工况目录
    """新建目录、拷入固定源与脚本、写出 case_config.json（注入配置）。"""
    os.makedirs(folder_path, exist_ok=True)  # 创建目录
    for target_name, src_path in source_files.items():  # 遍历所有源文件
        shutil.copy2(src_path, os.path.join(folder_path, target_name))  # 拷入并保留元数据
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:  # 写出注入配置
        json.dump(config or {}, f, ensure_ascii=False, indent=2)  # 序列化 config


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
    """按扩展名永久删除中间文件；file_types 为空则跳过。"""
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
    """组织源文件 → 规划各工况文件夹 → 建模/后处理 → 清理 → 汇总。"""
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
        print("错误：PARAMETER_CASES 为空。"); sys.exit(1)  # 报错退出
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

    print("开始批处理，最大并发任务数：{}".format(MAX_WORKERS))  # 启动提示
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 线程池
        for folder_path, ok in executor.map(process_folder, folder_plan):  # 执行并收集结果
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
