# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""批量创建文件夹并按顺序执行脚本。"""  # 说明脚本用途

import os  # 导入操作系统路径与目录模块
import re  # 导入正则模块用于替换参数
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置目标模型根目录
FOLDER_PREFIX = "fuke-ALL-"  # 设置目标文件夹前缀
STOP_ON_ERROR = True  # 设置出现脚本错误时是否立即停止

STATIC_SOURCE_PATHS = [  # 定义固定源文件完整路径列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled\El_Centro_scaled.txt",  # 定义 El_Centro 文件完整路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled\Loma_Prieta_scaled.txt",  # 定义 Loma_Prieta 文件完整路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Seismic\Scaled\Northridge_scaled.txt",  # 定义 Northridge 文件完整路径
]  # 结束固定源文件完整路径定义

SCRIPT_SEQUENCE = [  # 定义脚本顺序配置列表
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Single\VAB_oblique_TAF_v2.py", "parameter_target": True},  # 定义建模脚本完整路径与是否作为参数替换目标
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Postprocess_PGA_v6.py", "parameter_target": False},  # 定义后处理脚本完整路径
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Distribution_TAF_v3.py", "parameter_target": False},  # 定义分布脚本完整路径
]  # 结束脚本顺序配置定义

PARAMETER_CASES = [  # 定义参数方案列表
    {"params": {"h": 400, "i": 15, "angle": 0}},
    {"params": {"h": 400, "i": 15, "angle": 5}},
    {"params": {"h": 400, "i": 15, "angle": 10}},
    {"params": {"h": 400, "i": 15, "angle": 15}},
    {"params": {"h": 400, "i": 15, "angle": 20}},
    {"params": {"h": 400, "i": 15, "angle": 25}},
    {"params": {"h": 400, "i": 15, "angle": 30}},
    {"params": {"h": 400, "i": 30, "angle": 0}},
    {"params": {"h": 400, "i": 30, "angle": 5}},
    {"params": {"h": 400, "i": 30, "angle": 10}},
    {"params": {"h": 400, "i": 30, "angle": 15}},
    {"params": {"h": 400, "i": 30, "angle": 20}},
    {"params": {"h": 400, "i": 30, "angle": 25}},
    {"params": {"h": 400, "i": 45, "angle": 0}},
    {"params": {"h": 400, "i": 45, "angle": 5}},
    {"params": {"h": 400, "i": 45, "angle": 10}},
    {"params": {"h": 400, "i": 45, "angle": 15}},
    {"params": {"h": 400, "i": 45, "angle": 20}},
    {"params": {"h": 400, "i": 45, "angle": 25}},
    {"params": {"h": 400, "i": 45, "angle": 30}},
    {"params": {"h": 400, "i": 60, "angle": 0}},
    {"params": {"h": 400, "i": 60, "angle": 5}},
    {"params": {"h": 400, "i": 60, "angle": 10}},
    {"params": {"h": 400, "i": 60, "angle": 15}},
    {"params": {"h": 400, "i": 60, "angle": 20}},
    {"params": {"h": 400, "i": 60, "angle": 25}},
    {"params": {"h": 400, "i": 60, "angle": 30}},
    {"params": {"h": 400, "i": 75, "angle": 0}},
    {"params": {"h": 400, "i": 75, "angle": 5}},
    {"params": {"h": 400, "i": 75, "angle": 10}},
    {"params": {"h": 400, "i": 75, "angle": 15}},
    {"params": {"h": 400, "i": 75, "angle": 20}},
    {"params": {"h": 400, "i": 75, "angle": 25}},
    {"params": {"h": 400, "i": 75, "angle": 30}},
]  # 结束参数方案列表定义


def build_source_files(static_source_paths, script_sequence):  # 定义根据完整路径配置构建源文件映射的函数
    source_files = {}  # 初始化源文件映射
    missing = []  # 初始化缺失文件列表
    duplicate_names = []  # 初始化重名目标文件列表
    for source_path in static_source_paths:  # 遍历固定源文件完整路径
        target_name = os.path.basename(source_path)  # 从完整路径解析目标文件名
        if not os.path.isfile(source_path):  # 判断源文件是否存在
            missing.append((target_name, source_path))  # 记录缺失文件与对应路径
            continue  # 跳过当前文件继续处理下一个
        if target_name in source_files:  # 判断目标文件名是否已存在映射
            duplicate_names.append((target_name, source_path, source_files[target_name]))  # 记录重名冲突信息
            continue  # 跳过冲突项继续处理下一个
        source_files[target_name] = source_path  # 写入目标文件名到源路径的映射
    for script_config in script_sequence:  # 遍历脚本顺序配置项
        source_path = script_config["path"]  # 读取脚本完整路径
        target_name = os.path.basename(source_path)  # 从完整路径解析脚本文件名
        if not os.path.isfile(source_path):  # 判断脚本文件是否存在
            missing.append((target_name, source_path))  # 记录缺失脚本与对应路径
            continue  # 跳过当前脚本继续处理下一个
        if target_name in source_files and source_files[target_name] != source_path:  # 判断是否出现同名不同源路径冲突
            duplicate_names.append((target_name, source_path, source_files[target_name]))  # 记录重名冲突信息
            continue  # 跳过冲突项继续处理下一个
        source_files[target_name] = source_path  # 写入脚本文件名到源路径的映射
    return source_files, missing, duplicate_names  # 返回源文件映射、缺失列表与重名冲突列表


def ensure_sources_exist(missing_items):  # 定义检查缺失源文件列表的函数
    if not missing_items:  # 判断是否不存在缺失项
        return True  # 返回检查成功
    print("错误：以下源文件不存在，请先检查配置路径：")  # 输出错误提示标题
    for name, path in missing_items:  # 遍历缺失文件信息
        print("  - {} -> {}".format(name, path))  # 输出缺失文件明细
    return False  # 返回检查失败


def ensure_no_duplicate_targets(duplicate_items):  # 定义检查重名目标文件冲突的函数
    if not duplicate_items:  # 判断是否不存在重名冲突
        return True  # 返回检查成功
    print("错误：以下文件复制后会重名，请调整路径或文件名：")  # 输出重名冲突提示标题
    for target_name, current_path, existing_path in duplicate_items:  # 遍历重名冲突信息
        print("  - {} -> {} (已存在来源: {})".format(target_name, current_path, existing_path))  # 输出重名冲突明细
    return False  # 返回检查失败


def to_python_literal(value):  # 定义将参数值转换为 Python 字面量字符串的函数
    if isinstance(value, bool):  # 判断参数值是否为布尔类型
        return "True" if value else "False"  # 返回布尔值对应的 Python 文本
    if isinstance(value, (int, float)) and not isinstance(value, bool):  # 判断参数值是否为数值类型
        return repr(value)  # 返回数值对应的文本表示
    if value is None:  # 判断参数值是否为 None
        return "None"  # 返回 None 的 Python 文本
    if isinstance(value, str):  # 判断参数值是否为字符串类型
        return repr(value)  # 返回字符串对应的带引号文本表示
    raise TypeError("参数值类型不支持：{}".format(type(value).__name__))  # 抛出不支持类型异常


def sanitize_name_fragment(text):  # 定义将文本规范化为文件夹名片段的函数
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-")  # 将非法字符替换为连字符并去除两端连字符
    return cleaned or "value"  # 返回规范化结果并在空值时回退到默认片段


def build_folder_name(case_config):  # 定义根据参数方案构建文件夹名的函数
    custom_tag = case_config.get("folder_tag")  # 读取用户可选的自定义文件夹标签
    if custom_tag is not None:  # 判断是否配置了自定义标签
        return "{}{}".format(FOLDER_PREFIX, sanitize_name_fragment(custom_tag))  # 返回前缀加自定义标签构成的文件夹名
    params = case_config.get("params", {})  # 读取当前方案的参数字典
    fragments = []  # 初始化文件夹名片段列表
    for key, value in params.items():  # 按参数书写顺序遍历参数键值对
        key_part = sanitize_name_fragment(key)  # 生成参数名片段
        value_part = sanitize_name_fragment(value)  # 生成参数值片段
        fragments.append("{}{}".format(key_part, value_part))  # 记录参数名值直接拼接组合片段
    suffix = "_".join(fragments) if fragments else "case"  # 拼接文件夹后缀并在空参数时回退默认值
    return "{}{}".format(FOLDER_PREFIX, suffix)  # 返回前缀加参数后缀构成的文件夹名


def update_parameters_in_script(script_path, params):  # 定义按参数字典更新脚本参数的函数
    with open(script_path, "r", encoding="utf-8", errors="ignore") as f:  # 读取目标脚本文本
        content = f.read()  # 读取完整脚本内容
    for param_name, param_value in params.items():  # 遍历当前方案需要修改的参数键值对
        pattern = r"('{}'\s*:\s*)([^,\n\r}}]+)".format(re.escape(param_name))  # 构造当前参数的正则匹配表达式
        replacement = r"\g<1>{}".format(to_python_literal(param_value))  # 构造当前参数的替换文本
        content, replace_count = re.subn(pattern, replacement, content, count=1)  # 仅替换当前参数首次匹配项
        if replace_count == 0:  # 判断当前参数是否在脚本中未匹配到
            raise ValueError("未在 {} 中找到参数 '{}'".format(script_path, param_name))  # 抛出参数缺失异常
    with open(script_path, "w", encoding="utf-8", newline="") as f:  # 以写入模式覆盖脚本内容
        f.write(content)  # 写回替换后的脚本文本
    print("已更新 {} 的参数：{}".format(script_path, ", ".join(params.keys())))  # 输出参数更新结果


def create_and_fill_folder(folder_path, source_files, parameter_script_name, params):  # 定义创建并填充单个模型目录的函数
    os.makedirs(folder_path, exist_ok=True)  # 创建目标目录并允许已存在
    for target_name, src_path in source_files.items():  # 遍历所有需要复制的文件
        dst_path = os.path.join(folder_path, target_name)  # 拼接目标文件完整路径
        shutil.copy2(src_path, dst_path)  # 复制文件并保留时间戳等元数据
    parameter_script_path = os.path.join(folder_path, parameter_script_name)  # 定位当前文件夹中的参数替换目标脚本
    update_parameters_in_script(parameter_script_path, params)  # 按当前方案更新目标脚本参数


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
    source_files, missing_items, duplicate_items = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)  # 根据完整路径配置构建源文件映射
    run_order = [os.path.basename(script_config["path"]) for script_config in SCRIPT_SEQUENCE]  # 按脚本完整路径解析执行顺序文件名列表
    parameter_script_candidates = [os.path.basename(script_config["path"]) for script_config in SCRIPT_SEQUENCE if script_config.get("parameter_target")]  # 收集参数替换目标脚本候选列表
    parameter_script_name = parameter_script_candidates[0] if parameter_script_candidates else None  # 读取首个参数替换目标脚本名

    if source_files:  # 判断是否存在可用源文件
        print("已解析到以下源文件路径：")  # 输出解析结果标题
        for source_path in STATIC_SOURCE_PATHS:  # 按固定源文件顺序遍历完整路径
            target_name = os.path.basename(source_path)  # 从完整路径解析目标文件名
            if target_name in source_files:  # 判断当前文件是否解析成功
                print("  - {} -> {}".format(target_name, source_files[target_name]))  # 输出文件名与对应解析路径
        for script_config in SCRIPT_SEQUENCE:  # 按脚本顺序遍历脚本配置
            target_name = os.path.basename(script_config["path"])  # 从脚本完整路径解析目标文件名
            if target_name in source_files:  # 判断当前脚本是否解析成功
                print("  - {} -> {}".format(target_name, source_files[target_name]))  # 输出脚本名与对应解析路径
    if not ensure_sources_exist(missing_items):  # 检查所有源文件可用性
        sys.exit(1)  # 源文件缺失时退出程序
    if not ensure_no_duplicate_targets(duplicate_items):  # 检查复制目标名是否冲突
        sys.exit(1)  # 重名冲突时退出程序
    if not run_order:  # 判断脚本执行顺序是否为空
        print("错误：SCRIPT_SEQUENCE 为空，至少需要配置一个待执行脚本。")  # 输出脚本顺序为空提示
        sys.exit(1)  # 配置为空时退出程序
    if parameter_script_name is None:  # 判断是否未配置参数替换目标脚本
        print("错误：SCRIPT_SEQUENCE 中未设置 parameter_target=True，无法执行参数替换。")  # 输出参数目标脚本缺失提示
        sys.exit(1)  # 缺失网格脚本时退出程序
    if len(parameter_script_candidates) > 1:  # 判断参数替换目标脚本是否配置了多个
        print("错误：SCRIPT_SEQUENCE 中 parameter_target=True 只能配置一个。")  # 输出参数目标脚本数量错误提示
        sys.exit(1)  # 参数目标脚本数量错误时退出程序
    if not PARAMETER_CASES:  # 判断参数方案列表是否为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组参数方案。")  # 输出参数方案为空提示
        sys.exit(1)  # 参数方案为空时退出程序
    folder_plan = []  # 初始化文件夹计划列表
    folder_name_set = set()  # 初始化文件夹名称去重集合
    for case_index, case_config in enumerate(PARAMETER_CASES, start=1):  # 按顺序遍历参数方案配置
        if not isinstance(case_config, dict):  # 判断当前参数方案是否为字典类型
            print("错误：第 {} 组参数方案不是字典。".format(case_index))  # 输出参数方案类型错误提示
            sys.exit(1)  # 参数方案类型错误时退出程序
        params = case_config.get("params")  # 读取当前参数方案中的参数字典
        if not isinstance(params, dict) or not params:  # 判断参数字典是否有效且非空
            print("错误：第 {} 组参数方案缺少有效的 params 字典。".format(case_index))  # 输出参数字典错误提示
            sys.exit(1)  # 参数字典错误时退出程序
        folder_name = build_folder_name(case_config)  # 根据当前参数方案生成文件夹名称
        if folder_name in folder_name_set:  # 判断文件夹名称是否重复
            print("错误：参数方案生成了重复文件夹名 -> {}".format(folder_name))  # 输出重复文件夹名提示
            sys.exit(1)  # 文件夹名重复时退出程序
        folder_name_set.add(folder_name)  # 记录当前文件夹名用于后续去重
        folder_plan.append((folder_name, params))  # 记录当前文件夹名与参数字典

    os.makedirs(root_dir, exist_ok=True)  # 确保根目录存在
    failed_folders = []  # 初始化失败文件夹记录列表

    for folder_name, params in folder_plan:  # 按参数方案顺序遍历所有目标文件夹计划
        folder_path = os.path.join(root_dir, folder_name)  # 生成文件夹完整路径
        print("\n==============================")  # 输出分隔线便于阅读日志
        print("处理文件夹：{}".format(folder_path))  # 输出当前处理文件夹信息
        create_and_fill_folder(folder_path, source_files, parameter_script_name, params)  # 创建并填充当前文件夹并执行参数替换
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