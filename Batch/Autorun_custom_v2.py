# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""遍历当前目录下全部子文件夹并按顺序执行脚本。"""  # 说明脚本用途

import os  # 导入操作系统路径与目录模块
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器与命令行参数

ROOT_DIR = os.getcwd()  # 设置默认根目录为脚本启动时的当前工作目录
STOP_ON_ERROR = True  # 设置出现脚本错误时是否立即停止
SKIP_FOLDER_IF_SCRIPTS_EXIST = False  # 设置当子文件夹已存在全部执行脚本时是否跳过该文件夹

DELETE_BY_NAMES = [  # 定义按具体文件名删除的配置列表
    #"abaqus.rpy",
]  # 结束按具体文件名删除配置定义
DELETE_BY_EXTENSIONS = [  # 定义按文件扩展名删除的配置列表
    ".png", 
]  # 结束按文件扩展名删除配置定义

CUSTOM_COPY_PATHS = [  # 定义需要复制进每个子目录的自定义文件完整路径列表
    #r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled\El_Centro_scaled.txt",  # 定义 El_Centro 文件完整路径
    #r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled\Loma_Prieta_scaled.txt",  # 定义 Loma_Prieta 文件完整路径
    #r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled\Northridge_scaled.txt",  # 定义 Northridge 文件完整路径
]  # 结束自定义复制文件路径定义

SCRIPT_SEQUENCE = [  # 定义脚本顺序配置列表
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Distribution_PGA_v6.py"},  # 定义分布脚本完整路径
]  # 结束脚本顺序配置定义


def collect_subfolders(root_dir):  # 定义收集根目录下一级子文件夹的函数
    subfolders = []  # 初始化子文件夹路径列表
    for entry_name in sorted(os.listdir(root_dir)):  # 按名称排序遍历根目录下全部条目
        entry_path = os.path.join(root_dir, entry_name)  # 拼接当前条目的完整路径
        if not os.path.isdir(entry_path):  # 判断当前条目是否为文件夹
            continue  # 跳过非文件夹条目
        subfolders.append(entry_path)  # 记录当前子文件夹路径
    return subfolders  # 返回子文件夹路径列表


def build_copy_sources(custom_copy_paths, script_sequence):  # 定义构建复制源文件映射并校验冲突的函数
    copy_sources = {}  # 初始化复制源文件映射
    missing_items = []  # 初始化缺失源文件列表
    duplicate_items = []  # 初始化同名冲突列表
    combined_paths = list(custom_copy_paths)  # 复制自定义文件路径列表用于组合处理
    combined_paths.extend([script_config["path"] for script_config in script_sequence])  # 追加执行脚本路径确保脚本会被复制到子目录
    for source_path in combined_paths:  # 遍历所有需要复制的源文件路径
        target_name = os.path.basename(source_path)  # 提取复制后的目标文件名
        if not os.path.isfile(source_path):  # 判断源文件是否存在
            missing_items.append((target_name, source_path))  # 记录缺失源文件信息
            continue  # 跳过当前缺失项继续处理
        if target_name in copy_sources and copy_sources[target_name] != source_path:  # 判断是否出现同名不同路径冲突
            duplicate_items.append((target_name, source_path, copy_sources[target_name]))  # 记录同名冲突详情
            continue  # 跳过冲突项继续处理
        copy_sources[target_name] = source_path  # 记录目标文件名到源路径的映射
    return copy_sources, missing_items, duplicate_items  # 返回复制映射与校验结果


def ensure_sources_exist(missing_items):  # 定义检查缺失源文件列表的函数
    if not missing_items:  # 判断是否不存在缺失项
        return True  # 返回检查成功
    print("错误：以下源文件不存在，请先检查配置路径：")  # 输出缺失文件提示标题
    for target_name, source_path in missing_items:  # 遍历缺失源文件信息
        print("  - {} -> {}".format(target_name, source_path))  # 输出缺失文件明细
    return False  # 返回检查失败


def ensure_no_duplicate_targets(duplicate_items):  # 定义检查同名目标冲突的函数
    if not duplicate_items:  # 判断是否不存在同名冲突
        return True  # 返回检查成功
    print("错误：以下源文件复制后文件名冲突，请调整路径或文件名：")  # 输出同名冲突提示标题
    for target_name, current_path, existing_path in duplicate_items:  # 遍历冲突信息
        print("  - {} -> {} (已存在来源: {})".format(target_name, current_path, existing_path))  # 输出冲突明细
    return False  # 返回检查失败


def copy_files_to_folder(folder_path, copy_sources):  # 定义将源文件复制到指定子目录的函数
    for target_name, source_path in copy_sources.items():  # 遍历全部待复制文件映射
        destination_path = os.path.join(folder_path, target_name)  # 拼接当前文件目标路径
        shutil.copy2(source_path, destination_path)  # 复制源文件到当前子目录并保留元数据


def find_existing_scripts(folder_path, run_order):  # 定义检查文件夹内已存在脚本的函数
    existing_scripts = []  # 初始化已存在脚本列表
    for script_name in run_order:  # 按执行顺序遍历脚本文件名
        script_path = os.path.join(folder_path, script_name)  # 生成当前脚本完整路径
        if not os.path.isfile(script_path):  # 判断当前脚本是否不存在
            continue  # 跳过不存在脚本
        existing_scripts.append(script_name)  # 记录已存在脚本文件名
    return existing_scripts  # 返回已存在脚本列表


def has_all_scripts(folder_path, run_order):  # 定义判断文件夹是否已存在全部执行脚本的函数
    existing_script_set = set(find_existing_scripts(folder_path, run_order))  # 获取当前文件夹中已存在脚本集合
    required_script_set = set(run_order)  # 构造当前任务要求的脚本集合
    return required_script_set.issubset(existing_script_set)  # 返回是否已包含全部要求脚本


def run_scripts_in_folder(folder_path, run_order):  # 定义在单个目录内按顺序执行脚本的函数
    if not run_order:  # 判断当前是否未配置待执行脚本
        print("未配置待执行脚本，当前文件夹仅执行删除步骤。")  # 输出仅删除模式提示
        return True  # 返回执行成功以便继续后续流程
    for script_name in run_order:  # 按预设顺序遍历脚本
        script_path = os.path.join(folder_path, script_name)  # 生成当前脚本完整路径
        print("开始执行：{}".format(script_path))  # 输出脚本开始执行信息
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)  # 使用当前 Python 解释器在目标目录执行脚本
        if result.returncode != 0:  # 判断脚本是否执行失败
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))  # 输出脚本失败信息
            return False  # 返回执行失败
        print("完成执行：{}".format(script_name))  # 输出脚本执行完成信息
    return True  # 返回全部脚本执行成功


def normalize_delete_names(delete_names):  # 定义规范化文件名删除配置的函数
    normalized_names = set()  # 初始化规范化文件名集合
    for file_name in delete_names:  # 遍历原始文件名配置列表
        stripped_name = str(file_name).strip()  # 去除文件名两端空白字符
        if not stripped_name:  # 判断文件名是否为空
            continue  # 跳过空文件名配置
        normalized_names.add(stripped_name.lower())  # 记录小写后的文件名用于大小写不敏感匹配
    return normalized_names  # 返回规范化文件名集合


def normalize_delete_extensions(delete_extensions):  # 定义规范化扩展名删除配置的函数
    normalized_extensions = set()  # 初始化规范化扩展名集合
    for extension in delete_extensions:  # 遍历原始扩展名配置列表
        stripped_extension = str(extension).strip().lower()  # 去除扩展名两端空白并转换为小写
        if not stripped_extension:  # 判断扩展名是否为空
            continue  # 跳过空扩展名配置
        if not stripped_extension.startswith("."):  # 判断扩展名是否缺少前导点
            stripped_extension = ".{}".format(stripped_extension)  # 自动补齐扩展名前导点
        normalized_extensions.add(stripped_extension)  # 记录规范化后的扩展名
    return normalized_extensions  # 返回规范化扩展名集合


def delete_custom_files_in_folder(folder_path, delete_names, delete_extensions):  # 定义按文件名与扩展名删除文件的函数
    normalized_names = normalize_delete_names(delete_names)  # 规范化按文件名删除配置
    normalized_extensions = normalize_delete_extensions(delete_extensions)  # 规范化按扩展名删除配置
    if not normalized_names and not normalized_extensions:  # 判断是否没有任何删除规则
        print("未配置任何删除规则，跳过删除步骤。")  # 输出未配置删除规则提示
        return  # 直接结束删除步骤
    deleted_files = []  # 初始化删除成功文件路径列表
    failed_files = []  # 初始化删除失败文件路径与错误信息列表
    for entry_name in sorted(os.listdir(folder_path)):  # 遍历当前子文件夹下全部条目名称
        entry_path = os.path.join(folder_path, entry_name)  # 组装当前条目的完整路径
        if not os.path.isfile(entry_path):  # 判断当前条目是否为普通文件
            continue  # 跳过非普通文件条目
        lower_name = entry_name.lower()  # 获取当前文件名的小写形式用于匹配
        _, extension = os.path.splitext(lower_name)  # 提取当前文件的小写扩展名
        matched_by_name = lower_name in normalized_names  # 判断当前文件是否命中文件名删除规则
        matched_by_extension = extension in normalized_extensions  # 判断当前文件是否命中扩展名删除规则
        if not matched_by_name and not matched_by_extension:  # 判断当前文件是否未命中任何删除规则
            continue  # 跳过不需要删除的文件
        try:  # 尝试删除命中规则的文件
            os.remove(entry_path)  # 删除当前文件
            deleted_files.append(entry_path)  # 记录删除成功文件路径
        except OSError as exc:  # 捕获删除文件时的系统异常
            failed_files.append((entry_path, str(exc)))  # 记录删除失败文件路径与错误信息
    print("已删除文件数量：{}".format(len(deleted_files)))  # 输出删除成功文件数量
    if failed_files:  # 判断是否存在删除失败文件
        print("警告：以下文件删除失败：")  # 输出删除失败提示标题
        for failed_path, error_text in failed_files:  # 遍历删除失败信息列表
            print("  - {} -> {}".format(failed_path, error_text))  # 输出删除失败明细


def main():  # 定义主函数
    root_dir = ROOT_DIR  # 使用默认根目录
    if len(sys.argv) >= 2:  # 判断是否通过命令行传入根目录参数
        root_dir = sys.argv[1]  # 使用命令行传入的根目录覆盖默认值

    print("目标根目录：{}".format(root_dir))  # 输出本次运行根目录
    copy_sources, missing_items, duplicate_items = build_copy_sources(CUSTOM_COPY_PATHS, SCRIPT_SEQUENCE)  # 构建自定义文件与执行脚本的复制映射
    run_order = [os.path.basename(script_config["path"]) for script_config in SCRIPT_SEQUENCE]  # 按脚本完整路径解析执行顺序文件名列表
    if not run_order:  # 判断脚本执行顺序是否为空
        print("提示：SCRIPT_SEQUENCE 为空，将以仅删除模式遍历子文件夹。")  # 输出仅删除模式提示
    if not ensure_sources_exist(missing_items):  # 检查所有复制源文件是否存在
        sys.exit(1)  # 源文件缺失时退出程序
    if not ensure_no_duplicate_targets(duplicate_items):  # 检查复制目标文件名是否冲突
        sys.exit(1)  # 文件名冲突时退出程序

    if not os.path.isdir(root_dir):  # 判断根目录是否存在且为文件夹
        print("错误：根目录不存在或不是文件夹 -> {}".format(root_dir))  # 输出根目录无效提示
        sys.exit(1)  # 根目录无效时退出程序

    folder_paths = collect_subfolders(root_dir)  # 收集根目录下全部一级子文件夹
    if not folder_paths:  # 判断是否没有可处理的子文件夹
        print("错误：当前目录下未找到任何子文件夹。")  # 输出未找到子文件夹提示
        sys.exit(1)  # 无可处理文件夹时退出程序

    failed_folders = []  # 初始化失败文件夹记录列表
    skipped_folders = []  # 初始化跳过文件夹记录列表
    succeeded_folders = []  # 初始化成功文件夹记录列表

    for folder_path in folder_paths:  # 按名称顺序遍历所有目标子文件夹
        print("\n==============================")  # 输出分隔线便于阅读日志
        print("处理文件夹：{}".format(folder_path))  # 输出当前处理文件夹信息
        if SKIP_FOLDER_IF_SCRIPTS_EXIST and run_order:  # 判断是否配置为按已存在脚本规则跳过且已配置待执行脚本
            existing_scripts = find_existing_scripts(folder_path, run_order)  # 获取当前文件夹中已存在的执行脚本列表
            if existing_scripts:  # 判断当前文件夹是否已存在至少一个执行脚本
                print("检测到已存在执行脚本：{}".format(folder_path))  # 输出检测到已有脚本的提示
                for script_name in existing_scripts:  # 遍历已存在脚本列表
                    print("  - 已存在：{}".format(script_name))  # 输出已存在脚本明细
            if has_all_scripts(folder_path, run_order):  # 判断是否已存在全部脚本且配置为跳过
                print("按配置跳过该文件夹（已存在全部执行脚本）。")  # 输出按配置跳过提示
                skipped_folders.append(folder_path)  # 记录跳过文件夹路径
                continue  # 跳过当前文件夹继续处理下一个
        try:  # 尝试复制文件到当前子文件夹
            copy_files_to_folder(folder_path, copy_sources)  # 将自定义文件与执行脚本先复制到当前子文件夹
        except OSError as exc:  # 捕获复制文件过程中的系统异常
            print("错误：复制文件失败 -> {}".format(exc))  # 输出复制失败错误信息
            failed_folders.append(folder_path)  # 记录复制失败文件夹路径
            if STOP_ON_ERROR:  # 判断是否配置为失败即停止
                print("检测到失败，按配置停止后续文件夹处理。")  # 输出停止处理提示
                break  # 跳出循环结束批处理
            continue  # 当前文件夹失败后继续处理下一个

        delete_custom_files_in_folder(folder_path, DELETE_BY_NAMES, DELETE_BY_EXTENSIONS)  # 按配置自动执行删除或在无规则时跳过删除步骤
        ok = run_scripts_in_folder(folder_path, run_order)  # 在当前文件夹内按顺序执行脚本
        if ok:  # 判断当前文件夹是否执行成功
            succeeded_folders.append(folder_path)  # 记录执行成功的文件夹路径
            continue  # 当前文件夹成功后继续处理下一个

        failed_folders.append(folder_path)  # 记录失败文件夹路径
        if STOP_ON_ERROR:  # 判断是否配置为失败即停止
            print("检测到失败，按配置停止后续文件夹处理。")  # 输出停止处理提示
            break  # 跳出循环结束批处理

    print("\n==============================")  # 输出结束分隔线
    print("成功文件夹数量：{}".format(len(succeeded_folders)))  # 输出成功文件夹数量
    print("跳过文件夹数量：{}".format(len(skipped_folders)))  # 输出跳过文件夹数量
    print("失败文件夹数量：{}".format(len(failed_folders)))  # 输出失败文件夹数量

    if skipped_folders:  # 判断是否存在跳过文件夹
        print("\n已跳过文件夹：")  # 输出跳过文件夹标题
        for path in skipped_folders:  # 遍历跳过文件夹列表
            print("  - {}".format(path))  # 输出跳过文件夹路径

    if failed_folders:  # 判断是否存在失败文件夹
        print("\n批处理结束：存在失败文件夹。")  # 输出失败总结标题
        for path in failed_folders:  # 遍历失败文件夹列表
            print("  - {}".format(path))  # 输出失败文件夹路径
        sys.exit(2)  # 以非零状态码退出

    print("\n批处理结束：全部可执行文件夹处理完成。")  # 输出全部成功信息


if __name__ == "__main__":  # 判断是否为直接运行入口
    main()  # 执行主函数
