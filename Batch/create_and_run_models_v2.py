# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""批量创建 fuke-10-* 文件夹并按顺序执行脚本。"""  # 说明脚本用途

import os  # 导入操作系统路径与目录模块
import re  # 导入正则模块用于替换参数
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置目标模型根目录
FOLDER_PREFIX = "fuke-10-"  # 设置目标文件夹前缀
FOLDER_START = 1  # 设置起始文件夹编号
FOLDER_END = 12  # 设置结束文件夹编号
STOP_ON_ERROR = True  # 设置出现脚本错误时是否立即停止

WORKSPACE_DIR = r'C:\Users\12462\Documents\Code\AbqScripts'   # 设置工作区根目录

STATIC_SOURCE_FILE_NAMES = [  # 定义固定不带版本号变化的源文件名列表
    "El_Centro_scaled.txt",  # 定义 El_Centro 加速度时程文件名
    "Loma_Prieta_scaled.txt",  # 定义 Loma_Prieta 加速度时程文件名
    "Northridge_scaled.txt",  # 定义 Northridge 加速度时程文件名
]  # 结束固定源文件名列表定义

VERSIONED_SCRIPT_PREFIXES = [  # 定义需要自动选择最新版本的脚本前缀顺序
    "VAB_oblique_noGUI",  # 定义建模脚本前缀
    "Postprocess_PGA",  # 定义 PGA 后处理脚本前缀
    "Distribution_PGA",  # 定义分布后处理脚本前缀
]  # 结束版本化脚本前缀定义

PREFERRED_RELATIVE_PATHS = {  # 定义固定源文件的优先相对路径列表
    "El_Centro_scaled.txt": [os.path.join("Wave", "Seismic", "El_Centro_scaled.txt"), os.path.join("Wave", "Seismic", "Scaled", "El_Centro_scaled.txt")],  # 定义 El_Centro 优先路径
    "Loma_Prieta_scaled.txt": [os.path.join("Wave", "Seismic", "Loma_Prieta_scaled.txt"), os.path.join("Wave", "Seismic", "Scaled", "Loma_Prieta_scaled.txt")],  # 定义 Loma_Prieta 优先路径
    "Northridge_scaled.txt": [os.path.join("Wave", "Seismic", "Northridge_scaled.txt"), os.path.join("Wave", "Seismic", "Scaled", "Northridge_scaled.txt")],  # 定义 Northridge 优先路径
}  # 结束优先相对路径定义

VERSIONED_SCRIPT_PREFERRED_DIRS = {  # 定义版本化脚本的优先目录列表
    "VAB_oblique_noGUI": [os.path.join("Modeling", "Single")],  # 定义建模脚本优先目录
    "Postprocess_PGA": [os.path.join("Postprocess")],  # 定义 PGA 后处理脚本优先目录
    "Distribution_PGA": [os.path.join("Postprocess")],  # 定义分布后处理脚本优先目录
}  # 结束版本化脚本优先目录定义


def find_file_in_workspace(file_name):  # 定义在工作区内递归查找文件的函数
    candidates = []  # 初始化候选路径列表
    for current_root, _, files in os.walk(WORKSPACE_DIR):  # 遍历工作区下的所有目录与文件
        if file_name in files:  # 判断当前目录是否包含目标文件名
            candidates.append(os.path.join(current_root, file_name))  # 记录候选文件完整路径
    if not candidates:  # 判断是否未找到任何候选路径
        return None  # 返回空表示未找到文件
    candidates.sort(key=lambda p: (len(os.path.normpath(p)), os.path.normpath(p)))  # 按路径长度与字典序排序以稳定选择结果
    return candidates[0]  # 返回排序后首个候选路径


def find_latest_versioned_script(script_prefix):  # 定义按前缀查找最新版本脚本的函数
    pattern = re.compile(r"^{}_v(\d+)\.py$".format(re.escape(script_prefix)))  # 定义脚本名匹配模式并捕获版本号
    preferred_dirs = [os.path.normpath(os.path.join(WORKSPACE_DIR, rel_dir)) for rel_dir in VERSIONED_SCRIPT_PREFERRED_DIRS.get(script_prefix, [])]  # 构造当前前缀的优先目录绝对路径
    candidates = []  # 初始化候选脚本列表
    for current_root, _, files in os.walk(WORKSPACE_DIR):  # 遍历工作区下的所有目录与文件
        norm_root = os.path.normpath(current_root)  # 规范化当前遍历目录路径
        for file_name in files:  # 遍历当前目录中的文件名
            match = pattern.match(file_name)  # 判断文件名是否匹配目标前缀与版本格式
            if not match:  # 判断当前文件是否不匹配
                continue  # 跳过不匹配文件
            version = int(match.group(1))  # 解析匹配到的版本号整数值
            abs_path = os.path.join(current_root, file_name)  # 拼接候选脚本绝对路径
            preferred_rank = 0 if norm_root in preferred_dirs else 1  # 计算候选脚本是否位于优先目录
            candidates.append((version, preferred_rank, abs_path, file_name))  # 记录候选脚本信息用于排序
    if not candidates:  # 判断是否未找到任何候选脚本
        return None, None  # 返回空表示未找到脚本
    candidates.sort(key=lambda item: (-item[0], item[1], len(os.path.normpath(item[2])), os.path.normpath(item[2])))  # 按版本降序、目录优先级、路径长度、路径字典序排序
    latest = candidates[0]  # 读取排序后的最佳候选结果
    return latest[3], latest[2]  # 返回最新脚本文件名与绝对路径


def resolve_latest_versioned_scripts():  # 定义解析三类脚本最新版本的函数
    latest_name_map = {}  # 初始化脚本前缀到最新文件名的映射
    latest_path_map = {}  # 初始化最新文件名到绝对路径的映射
    missing_prefixes = []  # 初始化缺失脚本前缀列表
    for script_prefix in VERSIONED_SCRIPT_PREFIXES:  # 按既定顺序遍历版本化脚本前缀
        latest_name, latest_path = find_latest_versioned_script(script_prefix)  # 查找当前前缀的最新脚本
        if latest_name is None or latest_path is None:  # 判断当前前缀是否未找到脚本
            missing_prefixes.append(script_prefix)  # 记录缺失前缀
            continue  # 跳过当前前缀继续处理下一个
        latest_name_map[script_prefix] = latest_name  # 记录当前前缀解析到的最新文件名
        latest_path_map[latest_name] = latest_path  # 记录最新文件名对应的绝对路径
    return latest_name_map, latest_path_map, missing_prefixes  # 返回最新脚本名映射、路径映射与缺失前缀


def resolve_source_files(source_file_names, pre_resolved_paths=None):  # 定义解析源文件实际路径的函数
    resolved = {}  # 初始化已解析路径映射
    missing = []  # 初始化缺失文件列表
    pre_resolved_paths = pre_resolved_paths or {}  # 初始化预解析路径映射默认值
    for file_name in source_file_names:  # 遍历所有所需源文件名
        if file_name in pre_resolved_paths:  # 判断当前文件是否已有预解析路径
            resolved[file_name] = pre_resolved_paths[file_name]  # 写入预解析路径结果
            continue  # 跳过当前文件继续处理下一个
        preferred_rel_list = PREFERRED_RELATIVE_PATHS.get(file_name, [])  # 读取当前文件的优先相对路径列表
        resolved_path = None  # 初始化当前文件解析结果
        for preferred_rel in preferred_rel_list:  # 遍历当前文件的优先路径
            preferred_abs = os.path.join(WORKSPACE_DIR, preferred_rel)  # 拼接优先路径对应的绝对路径
            if os.path.isfile(preferred_abs):  # 判断优先路径是否存在文件
                resolved_path = preferred_abs  # 记录已命中的优先路径
                break  # 命中优先路径后结束当前文件的优先查找
        if resolved_path is None:  # 判断优先路径是否均未命中
            resolved_path = find_file_in_workspace(file_name)  # 在整个工作区执行递归兜底查找
        if resolved_path is None:  # 判断是否最终仍未找到文件
            missing.append(file_name)  # 记录缺失文件名
            continue  # 跳过当前文件继续处理下一个
        resolved[file_name] = resolved_path  # 写入当前文件解析到的实际路径
    return resolved, missing  # 返回解析结果与缺失列表


def ensure_sources_exist(source_files, missing_files):  # 定义检查源文件是否存在的函数
    missing = []  # 初始化缺失文件列表
    for name in missing_files:  # 遍历解析阶段已识别的缺失文件名
        missing.append((name, "<未在工作区中找到>"))  # 记录未找到的文件信息
    for name, path in source_files.items():  # 遍历所有已解析源文件映射
        if not os.path.isfile(path):  # 判断当前源文件是否存在
            missing.append((name, path))  # 记录缺失文件信息
    if missing:  # 判断是否存在缺失文件
        print("错误：以下源文件不存在，请先检查路径：")  # 输出错误提示标题
        for name, path in missing:  # 遍历缺失文件信息
            print("  - {} -> {}".format(name, path))  # 输出缺失文件明细
        return False  # 返回检查失败
    return True  # 返回检查成功


def update_mesh_size_manual(vab_script_path, mesh_value):  # 定义更新网格参数的函数
    with open(vab_script_path, "r", encoding="utf-8", errors="ignore") as f:  # 读取目标脚本文本
        content = f.read()  # 读取完整脚本内容
    pattern = r"('mesh_size_manual'\s*:\s*)(\d+(?:\.\d+)?)"  # 定义参数匹配正则表达式
    replaced_content, replace_count = re.subn(pattern, r"\g<1>{}".format(mesh_value), content, count=1)  # 仅替换首个匹配项
    if replace_count == 0:  # 判断是否未匹配到目标参数
        raise ValueError("未在 {} 中找到参数 'mesh_size_manual'".format(vab_script_path))  # 抛出参数缺失异常
    with open(vab_script_path, "w", encoding="utf-8", newline="") as f:  # 以写入模式覆盖脚本内容
        f.write(replaced_content)  # 写回替换后的脚本文本
    print("已更新 {} 的 mesh_size_manual={}".format(vab_script_path, mesh_value))  # 输出参数更新结果


def create_and_fill_folder(folder_path, folder_index, source_files, vab_script_name):  # 定义创建并填充单个模型目录的函数
    os.makedirs(folder_path, exist_ok=True)  # 创建目标目录并允许已存在
    for target_name, src_path in source_files.items():  # 遍历所有需要复制的文件
        dst_path = os.path.join(folder_path, target_name)  # 拼接目标文件完整路径
        shutil.copy2(src_path, dst_path)  # 复制文件并保留时间戳等元数据
    vab_script_path = os.path.join(folder_path, vab_script_name)  # 定位当前文件夹中的建模脚本
    update_mesh_size_manual(vab_script_path, folder_index)  # 将网格参数更新为当前文件夹编号


def run_scripts_in_folder(folder_path, run_order):  # 定义在单个目录内按顺序执行脚本的函数
    for script_name in run_order:  # 按预设顺序遍历脚本
        script_path = os.path.join(folder_path, script_name)  # 生成当前脚本完整路径
        if not os.path.isfile(script_path):  # 检查当前脚本是否存在
            print("错误：脚本不存在 -> {}".format(script_path))  # 输出脚本缺失错误
            return False  # 返回执行失败
        print("开始执行：{}".format(script_path))  # 输出脚本开始执行信息
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)  # 使用当前 Python 解释器在目标目录执行脚本
        if result.returncode != 0:  # 判断脚本是否执行失败
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))  # 输出脚本失败信息
            return False  # 返回执行失败
        print("完成执行：{}".format(script_name))  # 输出脚本执行完成信息
    return True  # 返回全部脚本执行成功


def main():  # 定义主函数
    root_dir = ROOT_DIR  # 使用默认根目录
    if len(sys.argv) >= 2:  # 判断是否通过命令行传入根目录参数
        root_dir = sys.argv[1]  # 使用命令行传入的根目录覆盖默认值

    print("目标根目录：{}".format(root_dir))  # 输出本次运行根目录
    latest_name_map, latest_path_map, missing_prefixes = resolve_latest_versioned_scripts()  # 解析三类脚本的最新版本
    if missing_prefixes:  # 判断是否存在未找到版本化脚本的前缀
        print("错误：以下脚本前缀未找到任何版本文件，请先检查工作区：")  # 输出缺失前缀标题
        for script_prefix in missing_prefixes:  # 遍历缺失脚本前缀
            print("  - {}".format(script_prefix))  # 输出缺失脚本前缀明细
        sys.exit(1)  # 缺失版本化脚本时退出程序
    print("已解析到以下最新版本脚本：")  # 输出最新版本脚本标题
    for script_prefix in VERSIONED_SCRIPT_PREFIXES:  # 按固定顺序遍历脚本前缀
        latest_name = latest_name_map[script_prefix]  # 读取当前前缀解析到的最新文件名
        print("  - {} -> {}".format(script_prefix, latest_name))  # 输出前缀与最新脚本名对应关系

    source_file_names = list(STATIC_SOURCE_FILE_NAMES) + [latest_name_map[prefix] for prefix in VERSIONED_SCRIPT_PREFIXES]  # 组装本次运行需要复制的完整源文件名列表
    run_order = [latest_name_map[prefix] for prefix in VERSIONED_SCRIPT_PREFIXES]  # 组装本次运行的动态脚本执行顺序
    vab_script_name = latest_name_map["VAB_oblique_noGUI"]  # 读取本次运行用于改网格参数的建模脚本文件名

    source_files, missing_files = resolve_source_files(source_file_names, pre_resolved_paths=latest_path_map)  # 解析所有源文件的实际路径
    if source_files:  # 判断是否存在可用源文件
        print("已解析到以下源文件路径：")  # 输出解析结果标题
        for name in source_file_names:  # 按固定顺序遍历源文件名
            if name in source_files:  # 判断当前文件是否解析成功
                print("  - {} -> {}".format(name, source_files[name]))  # 输出文件名与对应解析路径
    if not ensure_sources_exist(source_files, missing_files):  # 检查所有源文件可用性
        sys.exit(1)  # 源文件缺失时退出程序

    os.makedirs(root_dir, exist_ok=True)  # 确保根目录存在
    failed_folders = []  # 初始化失败文件夹记录列表

    for index in range(FOLDER_START, FOLDER_END + 1):  # 按编号顺序遍历所有目标文件夹
        folder_name = "{}{}".format(FOLDER_PREFIX, index)  # 生成文件夹名称
        folder_path = os.path.join(root_dir, folder_name)  # 生成文件夹完整路径
        print("\n==============================")  # 输出分隔线便于阅读日志
        print("处理文件夹：{}".format(folder_path))  # 输出当前处理文件夹信息
        create_and_fill_folder(folder_path, index, source_files, vab_script_name)  # 创建并填充当前文件夹并同步网格参数
        ok = run_scripts_in_folder(folder_path, run_order)  # 在当前文件夹内按顺序执行脚本
        if not ok:  # 判断当前文件夹是否执行失败
            failed_folders.append(folder_path)  # 记录失败文件夹路径
            if STOP_ON_ERROR:  # 判断是否配置为失败即停止
                print("检测到失败，按配置停止后续文件夹处理。")  # 输出停止处理提示
                break  # 跳出循环结束批处理

    print("\n==============================")  # 输出结束分隔线
    if failed_folders:  # 判断是否存在失败文件夹
        print("批处理结束：存在失败文件夹。")  # 输出失败总结标题
        for path in failed_folders:  # 遍历失败文件夹列表
            print("  - {}".format(path))  # 输出失败文件夹路径
        sys.exit(2)  # 以非零状态码退出
    print("批处理结束：全部文件夹处理完成。")  # 输出全部成功信息


if __name__ == "__main__":  # 判断是否为直接运行入口
    main()  # 执行主函数