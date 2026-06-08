# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""批量创建文件夹并按顺序执行脚本。"""

import os  # 导入操作系统路径与目录模块
import re  # 导入正则模块用于替换参数
import shutil  # 导入文件复制模块
import subprocess  # 导入子进程执行模块
import sys  # 导入系统模块用于获取 Python 解释器
import concurrent.futures  # 导入并发模块以实现多文件夹并行执行
import ctypes  # 导入 ctypes 模块用于调用 Windows 系统 API
from ctypes import wintypes  # 从 ctypes 中导入 wintypes 以使用 Windows 的标准数据类型

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置目标模型根目录
FOLDER_PREFIX = "double-"  # 设置目标文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 设置每个文件夹脚本执行后要删除的文件类型列表，例如 [".odb", ".lck"]。若为空则不删除。
USE_RECYCLE_BIN = False  # 设置删除文件时是放入回收站（True）还是直接永久删除（False）

STATIC_SOURCE_PATHS = [  # 定义固定源文件完整路径列表
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 定义 El_Centro 文件完整路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_6Hz.txt",  # 定义 Loma_Prieta 文件完整路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_8Hz.txt",  # 定义 Northridge 文件完整路径
]  # 结束固定源文件完整路径定义

SCRIPT_SEQUENCE = [  # 定义脚本顺序配置列表
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_double_TAF_v4.py", "parameter_target": True},  # 定义建模脚本完整路径与是否作为参数替换目标（原 v7 已重命名为 v4）
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py", "parameter_target": False},  # 定义后处理脚本完整路径
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v1.py", "parameter_target": False},  # 定义分布脚本完整路径
    {"path": r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_Multi_TAF_v2.py", "parameter_target": False},  # 定义分布脚本完整路径
]  # 结束脚本顺序配置定义

PARAMETER_CASES = [  # 定义参数方案列表
    {"params": { "i": 30,  "angle": 0}},  # 定义第一组参数方案：角度为30度和0度
    {"params": { "i": 30, "angle": 15}},  # 定义第二组参数方案：角度为30度和15度
]  # 结束参数方案列表定义

class SHFILEOPSTRUCTW(ctypes.Structure):  # 定义 SHFILEOPSTRUCTW 结构体以匹配 Windows API
    """Windows SHFILEOPSTRUCTW 结构体的 ctypes 实现，用于调用系统回收站功能"""  # 结构体文档说明
    _fields_ = [  # 定义结构体的成员字段列表
        ("hwnd", wintypes.HWND),  # 窗口句柄成员，类型为 HWND
        ("wFunc", wintypes.UINT),  # 操作类型成员，类型为 UINT
        ("pFrom", wintypes.LPCWSTR),  # 源文件路径成员，类型为 LPCWSTR，要求双空字符结尾
        ("pTo", wintypes.LPCWSTR),  # 目的文件路径成员，类型为 LPCWSTR
        ("fFlags", ctypes.c_ushort),  # 操作控制标志成员，类型为 WORD（c_ushort）
        ("fAnyOperationsAborted", wintypes.BOOL),  # 是否中止操作成员，类型为 BOOL
        ("hNameMappings", wintypes.LPVOID),  # 文件名映射句柄成员，类型为 LPVOID
        ("lpszProgressTitle", wintypes.LPCWSTR),  # 进度条标题成员，类型为 LPCWSTR
    ]  # 结束结构体成员字段列表定义

def send_to_recycle_bin(path):  # 定义将指定文件移入 Windows 回收站的函数
    """
    path: 待删除文件的绝对或相对路径
    返回值: 布尔值，True 表示移入回收站成功，False 表示失败
    """
    abs_path = os.path.abspath(path) + "\0\0"  # 将路径转换为绝对路径，并以双空字符结尾（Windows API 要求）
    fileop = SHFILEOPSTRUCTW()  # 创建结构体实例
    fileop.hwnd = None  # 将窗口句柄设为 None（无父窗口）
    fileop.wFunc = 3  # 设置操作类型为 FO_DELETE（值为 3）
    fileop.pFrom = abs_path  # 设置源路径为以双空字符结尾的绝对路径
    fileop.pTo = None  # 将目的路径设为 None（删除操作无需目的路径）
    fileop.fFlags = 0x0040 | 0x0010 | 0x0400 | 0x0004  # 设置标志：允许撤销、不弹出确认框、无错误界面、静默模式
    fileop.fAnyOperationsAborted = False  # 初始化操作中止状态为 False
    fileop.hNameMappings = None  # 将名称映射句柄设为 None
    fileop.lpszProgressTitle = None  # 将进度条标题设为 None
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(fileop))  # 调用 Windows 垃圾箱 API
    return result == 0  # 返回操作结果，若返回值为 0 则表示成功

def build_source_files(static_source_paths, script_sequence):  # 定义根据完整路径配置构建源文件映射的函数
    """
    static_source_paths: 固定源文件完整路径列表
    script_sequence: 脚本顺序配置列表
    返回值: 源文件映射、缺失列表与重名冲突列表
    """
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
    """
    missing_items: 缺失的源文件列表
    返回值: 布尔值，True 表示所有文件均存在，False 表示有缺失
    """
    if not missing_items:  # 判断是否不存在缺失项
        return True  # 返回检查成功
    print("错误：以下源文件不存在，请先检查配置路径：")  # 输出错误提示标题
    for name, path in missing_items:  # 遍历缺失文件信息
        print("  - {} -> {}".format(name, path))  # 输出缺失文件明细
    return False  # 返回检查失败

def ensure_no_duplicate_targets(duplicate_items):  # 定义检查重名目标文件冲突的函数
    """
    duplicate_items: 重名目标文件列表
    返回值: 布尔值，True 表示无重名冲突，False 表示有冲突
    """
    if not duplicate_items:  # 判断是否不存在重名冲突
        return True  # 返回检查成功
    print("错误：以下文件复制后会重名，请调整路径或文件名：")  # 输出重名冲突提示标题
    for target_name, current_path, existing_path in duplicate_items:  # 遍历重名冲突信息
        print("  - {} -> {} (已存在来源: {})".format(target_name, current_path, existing_path))  # 输出重名冲突明细
    return False  # 返回检查失败

def to_python_literal(value):  # 定义将参数值转换为 Python 字面量字符串的函数
    """
    value: 待转换的值
    返回值: 对应的 Python 表达文本
    """
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
    """
    text: 待规范化的文本
    返回值: 规范化后的文件夹名片段
    """
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-")  # 将非法字符替换为连字符并去除两端连字符
    return cleaned or "value"  # 返回规范化结果并在空值时回退到默认片段

def build_folder_name(case_config):  # 定义根据参数方案构建文件夹名的函数
    """
    case_config: 参数方案配置字典
    返回值: 文件夹名称字符串
    """
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
    """
    script_path: 待替换参数的脚本文件路径
    params: 需替换的参数字典
    """
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
    """
    folder_path: 模型目录路径
    source_files: 需拷贝的源文件字典
    parameter_script_name: 需做参数替换的目标脚本名
    params: 参数字典
    """
    os.makedirs(folder_path, exist_ok=True)  # 创建目标目录并允许已存在
    for target_name, src_path in source_files.items():  # 遍历所有需要复制的文件
        dst_path = os.path.join(folder_path, target_name)  # 拼接目标文件完整路径
        shutil.copy2(src_path, dst_path)  # 复制文件并保留时间戳等元数据
    parameter_script_path = os.path.join(folder_path, parameter_script_name)  # 定位当前文件夹中的参数替换目标脚本
    update_parameters_in_script(parameter_script_path, params)  # 按当前方案更新目标脚本参数

def run_scripts_in_folder(folder_path, run_order):  # 定义在单个目录内按顺序执行脚本的函数
    """
    folder_path: 执行目标目录
    run_order: 需顺序执行的脚本名列表
    返回值: 布尔值，True 表示所有脚本成功，False 表示发生错误
    """
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

def delete_files_in_folder(folder_path, file_types, use_recycle_bin=True):  # 定义删除单个目录下指定类型文件的函数
    """
    folder_path: 目标文件夹路径
    file_types: 待删除的文件类型列表
    use_recycle_bin: 是否将文件移入回收站（True）还是直接永久删除（False）
    """
    if not file_types:  # 如果没有指定类型，则直接返回
        return  # 结束并返回空值
    normalized_types = {t.lower() if t.startswith(".") else "." + t.lower() for t in file_types}  # 将类型格式化为包含点的小写扩展名集合
    deleted_files = {t: [] for t in normalized_types}  # 初始化已删除文件分类字典
    failed_files = []  # 初始化删除失败文件信息列表
    for file_name in sorted(os.listdir(folder_path)):  # 遍历目录下全部文件与子目录名称并排序
        file_path = os.path.join(folder_path, file_name)  # 组装当前条目的完整路径
        if not os.path.isfile(file_path):  # 判断当前条目是否为普通文件
            continue  # 跳过非文件类型条目
        ext = os.path.splitext(file_name)[1].lower()  # 获取文件扩展名并转换为小写
        if ext not in normalized_types:  # 判断当前文件扩展名是否在待删除列表中
            continue  # 跳过非指定类型文件
        try:  # 尝试执行删除操作
            if use_recycle_bin:  # 如果需要放入回收站
                success = send_to_recycle_bin(file_path)  # 调用回收站删除函数
                if success:  # 判断是否移入回收站成功
                    deleted_files[ext].append(file_path)  # 记录成功移入回收站的文件
                else:  # 回收站删除失败时
                    failed_files.append((file_path, "SHFileOperationW failed"))  # 记录回收站删除失败信息
            else:  # 如果是直接物理删除
                os.remove(file_path)  # 物理删除文件
                deleted_files[ext].append(file_path)  # 记录直接物理删除成功的文件
        except OSError as exc:  # 捕获文件删除时的系统级异常
            failed_files.append((file_path, str(exc)))  # 记录删除失败文件路径与异常说明
    for ext, files in sorted(deleted_files.items()):  # 遍历按扩展名分类的成功删除结果
        if files:  # 判断该类型下是否存在已删除文件
            print("已删除 {} 文件数量：{}".format(ext, len(files)))  # 输出已删除的文件类型与数量
        else:  # 处理该类型下无待删除文件的情况
            print("未找到需要删除的 {} 文件。".format(ext))  # 输出未找到该类型文件提示
    if failed_files:  # 判断是否存在删除失败的记录
        print("警告：以下文件删除失败：")  # 输出失败提示标题
        for failed_path, error_text in failed_files:  # 遍历每个删除失败的文件信息
            print("  - {} -> {}".format(failed_path, error_text))  # 输出失败文件路径与错误内容

def main():  # 定义主函数
    """
    主控制流程函数
    """
    root_dir = ROOT_DIR  # 使用默认根目录
    if len(sys.argv) >= 2:  # 判断是否通过命令行传入根目录参数
        root_dir = sys.argv[1]  # 使用命令行传入的第一个参数作为根目录
    types_to_delete = list(DELETE_FILE_TYPES)  # 复制要删除的文件类型列表
    print("目标根目录：{}".format(root_dir))  # 输出本次运行根目录
    print("要删除的文件类型：{}".format(types_to_delete if types_to_delete else "无"))  # 输出本次运行待删除的文件类型
    print("是否使用回收站：{}".format(USE_RECYCLE_BIN))  # 输出是否使用回收站的配置信息
    source_files, missing_items, duplicate_items = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)  # 构建源文件映射与冲突分析
    run_order = [os.path.basename(script_config["path"]) for script_config in SCRIPT_SEQUENCE]  # 解析待执行脚本的文件名顺序列表
    parameter_script_candidates = [os.path.basename(script_config["path"]) for script_config in SCRIPT_SEQUENCE if script_config.get("parameter_target")]  # 获取参数替换脚本候选列表
    parameter_script_name = parameter_script_candidates[0] if parameter_script_candidates else None  # 设置第一个候选脚本为参数替换目标
    if source_files:  # 判断是否存在解析成功的源文件
        print("已解析到以下源文件路径：")  # 输出源文件路径解析结果标题
        for source_path in STATIC_SOURCE_PATHS:  # 遍历静态源文件路径
            target_name = os.path.basename(source_path)  # 获取目标文件名
            if target_name in source_files:  # 判断是否在源文件映射中
                print("  - {} -> {}".format(target_name, source_files[target_name]))  # 输出文件名与真实源路径映射关系
        for script_config in SCRIPT_SEQUENCE:  # 遍历执行序列中的脚本
            target_name = os.path.basename(script_config["path"])  # 获取脚本文件名
            if target_name in source_files:  # 判断是否在源文件映射中
                print("  - {} -> {}".format(target_name, source_files[target_name]))  # 输出脚本名与对应真实源路径
    if not ensure_sources_exist(missing_items):  # 校验所有源文件是否均存在
        sys.exit(1)  # 若有源文件缺失则退出程序
    if not ensure_no_duplicate_targets(duplicate_items):  # 校验目标路径是否发生命名冲突
        sys.exit(1)  # 若存在冲突则退出程序
    if not run_order:  # 判断脚本执行顺序列表是否为空
        print("错误：SCRIPT_SEQUENCE 为空，至少需要配置一个待执行脚本。")  # 输出顺序为空的错误提示
        sys.exit(1)  # 执行顺序为空时退出程序
    if parameter_script_name is None:  # 判断是否未配置参数替换目标脚本
        print("错误：SCRIPT_SEQUENCE 中未设置 parameter_target=True，无法执行参数替换。")  # 输出目标缺失错误提示
        sys.exit(1)  # 无法替换参数时退出程序
    if len(parameter_script_candidates) > 1:  # 判断参数替换目标脚本是否配置了多个
        print("错误：SCRIPT_SEQUENCE 中 parameter_target=True 只能配置一个。")  # 输出目标过多的错误提示
        sys.exit(1)  # 目标过多时退出程序
    if not PARAMETER_CASES:  # 判断参数方案列表是否为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组参数方案。")  # 输出方案为空的错误提示
        sys.exit(1)  # 方案为空时退出程序
    folder_plan = []  # 初始化文件夹处理计划列表
    folder_name_set = set()  # 初始化已存在文件夹名称集合以用于去重
    for case_index, case_config in enumerate(PARAMETER_CASES, start=1):  # 遍历所有参数方案
        if not isinstance(case_config, dict):  # 判断参数方案是否为字典
            print("错误：第 {} 组参数方案不是字典。".format(case_index))  # 输出类型错误的提示
            sys.exit(1)  # 类型错误时退出程序
        params = case_config.get("params")  # 获取当前方案下的参数字典
        if not isinstance(params, dict) or not params:  # 校验参数字典的合法性与非空性
            print("错误：第 {} 组参数方案缺少有效的 params 字典。".format(case_index))  # 输出参数非法的错误提示
            sys.exit(1)  # 参数非法时退出程序
        folder_name = build_folder_name(case_config)  # 构建当前方案对应的文件夹名称
        if folder_name in folder_name_set:  # 判断生成的文件夹名是否发生重复
            print("错误：参数方案生成了重复文件夹名 -> {}".format(folder_name))  # 输出文件夹重名错误提示
            sys.exit(1)  # 重名时退出程序
        folder_name_set.add(folder_name)  # 记录文件夹名到去重集合中
        folder_plan.append((folder_name, params))  # 记录文件夹名与对应的参数键值对
    os.makedirs(root_dir, exist_ok=True)  # 创建或确保根目录存在
    failed_folders = []  # 初始化失败目录记录列表
    def process_folder(item):  # 定义处理单个文件夹的内部多线程工作函数
        """
        item: 元组 (folder_name, params)
        返回值: 元组 (folder_path, ok)
        """
        folder_name, params = item  # 解包文件夹名与参数字典
        folder_path = os.path.join(root_dir, folder_name)  # 拼接目标文件夹的完整路径
        print("\n==============================")  # 输出日志段落分隔符
        print("开始处理文件夹：{}".format(folder_path))  # 输出当前正在处理的文件夹路径
        create_and_fill_folder(folder_path, source_files, parameter_script_name, params)  # 创建文件夹、拷贝源文件并更新参数
        ok = run_scripts_in_folder(folder_path, run_order)  # 顺序执行文件夹内的所有脚本
        if types_to_delete:  # 判断是否配置了需要删除的文件类型
            delete_files_in_folder(folder_path, types_to_delete, use_recycle_bin=USE_RECYCLE_BIN)  # 执行文件删除并按配置决定是否移入回收站
        else:  # 若没有指定需要删除的文件类型
            print("已跳过文件删除（没有指定要删除的文件类型）。")  # 输出跳过删除的日志信息
        return folder_path, ok  # 返回该目录的路径与运行成功标志
    max_workers = 2  # 设置并发线程最大数量为 2
    print("开始并行批处理，最大并发任务数：{}".format(max_workers))  # 输出并发启动提示信息
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  # 启动线程池上下文管理器
        results = executor.map(process_folder, folder_plan)  # 将处理计划提交到线程池执行
        for folder_path, ok in results:  # 遍历所有文件夹的并发处理结果
            if not ok:  # 判断执行是否失败
                failed_folders.append(folder_path)  # 记录失败文件夹路径
    print("\n==============================")  # 输出执行总结前段的分隔符
    if failed_folders:  # 判断是否存在失败的文件夹
        print("批处理结束：存在失败文件夹。")  # 输出失败总结信息标题
        for path in failed_folders:  # 遍历失败的文件夹列表
            print("  - {}".format(path))  # 输出具体失败的路径
        sys.exit(2)  # 以状态码 2 异常退出程序
    print("批处理结束：全部文件夹处理完成。")  # 输出全部处理成功的信息

if __name__ == "__main__":  # 判断当前是否为主入口直接运行
    main()  # 调用主入口控制逻辑函数
