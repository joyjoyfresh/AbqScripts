# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""通用批量运行与配置注入的 Autorun 模板。

用户可直接复制此模板并修改工况配置列表与路径，用于各类有限元参数扫描。
支持：
  1. 静态文件自动复制到各工况目录。
  2. 脚本的顺序执行（在各工况目录下独立运行）。
  3. 通过 json 配置注入参数，避免正则修改脚本。
  4. 多线程并行执行。
  5. 自动清理指定的中间大文件（如 odb/inp 等）。
  6. 执行全局汇总与出图的后处理脚本。
  7. 地震波经 run_cfg.wave_files 注入（波形文件留在源目录，不拷贝进工况文件夹，
     要求建模脚本支持 case_config.json 的 run_cfg['wave_files'] 键）。
"""

import os  # 导入操作系统相关路径与目录操作模块
import json  # 导入 JSON 模块用于写出注入配置文件
import shutil  # 导入文件复制与高层级文件操作模块
import subprocess  # 导入子进程执行模块以运行其他脚本
import sys  # 导入系统模块用于获取 Python 解释器路径与退出程序
import concurrent.futures  # 导入并发模块以实现多工况文件夹并行执行

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置默认的模型根目录，各工况文件夹建在此
FOLDER_PREFIX = "case-"  # 各工况文件夹的命名统一前缀
DELETE_FILE_TYPES = [".odb", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim"]  # 执行后自动删除的中间文件类型，留空则不删除
MAX_WORKERS = 2  # 并行处理工况文件夹的最大线程数
CONFIG_FILENAME = "case_config.json"  # 注入给建模或计算脚本的配置文件名

STATIC_SOURCE_PATHS = [  # 固定源文件路径列表，将拷入每个工况文件夹（地震波不走这里，用 WAVE_FILES 注入）
    # r"C:\path\to\your\control_file.dat",  # 示例源文件路径 / 需替换为实际控制文件路径
]

WAVE_FILES = [  # 全局兜底地震波列表：工况 config 未写 run_cfg.wave_files 时注入；留空且工况也未写=不注入（建模脚本回退扫工况目录 .txt）
    # r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 示例主脉冲 / 排首条兼作阻尼 fc 自估源
]

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行的脚本绝对路径
    # r"C:\path\to\your\modeling.py",  # 建模脚本路径 / 读取 case_config.json 执行建模
    # r"C:\path\to\your\postprocess.py",  # 后处理提取脚本路径 / 提取单工况数据
]

POST_SCRIPT_SEQUENCE = [  # 全部工况求解完成后自动在根目录执行的全局后处理脚本绝对路径
    # r"C:\path\to\your\collect.py",  # 汇总脚本路径 / 收集各工况 meta.json
    # r"C:\path\to\your\plot.py",  # 绘图脚本路径 / 绘制汇总对比图表
]

PARAMETER_CASES = [  # 变参数工况列表，每项需包含 config 配置覆盖字典；工况级 run_cfg.wave_files 优先于全局 WAVE_FILES
    {
        "name": "case_example_1",  # 工况名称 / 可选，若省略则按 config 自动命名（含波形文件名）
        "config": {  # 参数覆盖字典 / 建模脚本中需通过读取 json 文件覆盖默认配置
            "material": {"velocity": 800.0, "density": 2500.0},  # 材料配置 / 速度与密度值
            "geometry": {"slope_angle": 30.0},  # 几何配置 / 边坡角度
            "mesh_size": 2.0,  # 网格尺寸 / 网格最大尺寸
            "run_cfg": {"wave_files": [  # 工况级地震波 / 每个工况可指定不同波
                r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 本工况用 4Hz 主脉冲
            ]},
        }
    },
    {
        "name": "case_example_2",  # 工况名称 / 可选，若省略则按 config 自动命名（含波形文件名）
        "config": {  # 参数覆盖字典
            "material": {"velocity": 1200.0, "density": 2500.0},  # 材料配置 / 速度与密度值
            "geometry": {"slope_angle": 45.0},  # 几何配置 / 边坡角度
            "mesh_size": 2.0,  # 网格尺寸 / 网格最大尺寸
            "run_cfg": {"wave_files": [  # 工况级地震波 / 与工况1不同波
                r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_2Hz.txt",  # 本工况用 2Hz 补低频脉冲
            ]},
        }
    }
]


def build_source_files(static_source_paths, script_sequence):  # 建立源文件映射并检测缺失与冲突
    """建立源文件映射并检测缺失与冲突。

    参数说明:
        static_source_paths (list): 静态固定源文件完整路径列表。
        script_sequence (list): 执行脚本完整路径列表。

    返回值:
        tuple: (source_files 映射字典, missing 缺失文件列表, duplicate_names 冲突文件列表)
    """
    source_files = {}  # 初始化源文件名字与物理路径的映射字典
    missing = []  # 初始化不存在的文件记录列表
    duplicate_names = []  # 初始化发生重名冲突的文件记录列表
    for source_path in list(static_source_paths) + list(script_sequence):  # 遍历所有配置 of 源文件和脚本路径
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


def _sanitize(text):  # 清洗字符串为合法的文件目录名
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
                if sub_key == "wave_files" and isinstance(sub_val, (list, tuple)):  # 波形列表：取文件名主干参与命名（区分同参数不同波的工况）
                    tokens.extend(os.path.splitext(os.path.basename(str(p)))[0] for p in sub_val)  # 逐条追加波名片段
                elif not isinstance(sub_val, (dict, list)):  # 仅提取标量基本类型
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


def resolve_wave_files(config):  # 解析单工况最终使用的地震波列表（工况级优先，全局 WAVE_FILES 兜底）
    """解析单工况最终使用的地震波列表。

    参数说明:
        config (dict): 单工况参数覆盖字典。

    返回值:
        list: 绝对路径波形列表；工况级 run_cfg.wave_files 优先，否则用全局 WAVE_FILES，均未配置返回空列表。
    """
    wf = ((config or {}).get("run_cfg") or {}).get("wave_files")  # 工况级波形配置
    if wf:  # 工况显式指定
        wf_list = wf if isinstance(wf, (list, tuple)) else [wf]  # 允许单条字符串写法
    else:  # 未指定则用全局兜底
        wf_list = WAVE_FILES
    return [os.path.abspath(str(p)) for p in wf_list]  # 统一转绝对路径


def inject_wave_files(config):  # 把解析后的地震波列表写入工况配置的 run_cfg.wave_files
    """把解析后的地震波列表写入工况配置的 run_cfg.wave_files。

    参数说明:
        config (dict): 单工况参数覆盖字典。

    返回值:
        dict: 注入后的配置副本（波形路径已绝对化）；全局与工况级均未配置波形时原样返回。
    """
    wave_list = resolve_wave_files(config)  # 解析最终波形列表
    if not wave_list:  # 无任何波形配置
        return config  # 原样返回（建模脚本回退扫目录旧行为）
    new_config = dict(config)  # 浅拷贝避免污染 PARAMETER_CASES 原字典
    new_run_cfg = dict(config.get("run_cfg") or {})  # 拷贝运行控制配置
    new_run_cfg["wave_files"] = wave_list  # 写入绝对化后的波形列表
    new_config["run_cfg"] = new_run_cfg  # 挂回配置副本
    return new_config  # 返回注入后的配置


def create_and_fill_folder(folder_path, source_files, config):  # 创建目录并注入配置文件与源文件
    """创建目录并注入配置文件与源文件。

    参数说明:
        folder_path (str): 目标工况文件夹绝对路径。
        source_files (dict): 源文件物理路径映射字典。
        config (dict): 参数覆盖字典。
    """
    os.makedirs(folder_path, exist_ok=True)  # 创建工况目录
    for target_name, src_path in source_files.items():  # 遍历待拷贝的所有文件映射
        shutil.copy2(src_path, os.path.join(folder_path, target_name))  # 拷入工况目录并保留元数据
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:  # 打开注入的配置文件
        json.dump(config or {}, f, ensure_ascii=False, indent=2)  # 序列化配置为 JSON 文件


def run_scripts_in_folder(folder_path, run_order):  # 在工况文件夹内按顺序执行指定的脚本
    """在工况文件夹内按顺序执行指定的脚本。

    参数说明:
        folder_path (str): 工况文件夹绝对路径。
        run_order (list): 执行脚本文件名（不含路径）顺序列表。

    返回值:
        bool: 若全部顺利执行返回 True，任意脚本执行失败则返回 False。
    """
    for script_name in run_order:  # 遍历待执行的脚本文件名
        script_path = os.path.join(folder_path, script_name)  # 拼接工况目录下的脚本绝对路径
        if not os.path.isfile(script_path):  # 若物理文件不存在
            print("错误：脚本不存在 -> {}".format(script_path))  # 打印不存在的错误提示
            return False  # 返回失败
        print("开始执行：{}".format(script_path))  # 打印启动执行提示
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)  # 用当前解释器在工况目录下执行该脚本
        if result.returncode != 0:  # 若执行退出码不为 0
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))  # 打印执行失败提示
            return False  # 返回失败
        print("完成执行：{}".format(script_name))  # 打印执行成功提示
    return True  # 返回成功


def delete_files_by_type(folder_path, file_types):  # 永久删除指定后缀的中间文件以节省空间
    """永久删除指定后缀的中间文件以节省空间。

    参数说明:
        folder_path (str): 工况文件夹绝对路径。
        file_types (list): 待删除的文件后缀名列表。
    """
    if not file_types:  # 若未配置删除类型
        return  # 直接返回
    normalized = {t.lower() if t.startswith(".") else "." + t.lower() for t in file_types}  # 规范化文件后缀名为小写格式
    deleted = {t: 0 for t in normalized}  # 初始化各类型删除文件数计数器
    failed = []  # 初始化删除失败文件记录列表
    for name in sorted(os.listdir(folder_path)):  # 遍历工况目录下的所有文件名
        fp = os.path.join(folder_path, name)  # 拼接绝对路径
        if not os.path.isfile(fp):  # 若非文件结构
            continue  # 跳过处理
        ext = os.path.splitext(name)[1].lower()  # 提取文件后缀名并转换为小写
        if ext not in normalized:  # 若不在待删除列表中
            continue  # 跳过处理
        try:  # 尝试删除物理文件
            os.remove(fp)  # 永久删除文件
            deleted[ext] += 1  # 计数器加一
        except OSError as exc:  # 若触发系统错误
            failed.append((fp, str(exc)))  # 记录错误信息
    for ext, n in sorted(deleted.items()):  # 遍历打印删除结果统计
        print("已删除 {} 文件数量：{}".format(ext, n))  # 打印各类型文件删除数量
    if failed:  # 若存在删除失败的文件
        print("警告：以下文件删除失败：")  # 打印警告标题
        for fp, err in failed:  # 遍历打印失败记录
            print("  - {} -> {}".format(fp, err))  # 打印失败的物理路径与错误信息


def main():  # 批处理主控制流程
    """批处理主控制流程。"""
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR  # 支持从命令行参数接收保存目录，否则使用默认值
    types_to_delete = list(DELETE_FILE_TYPES)  # 获取待删除中间文件格式列表的副本
    print("目标根目录：{}".format(root_dir))  # 打印目标根目录
    print("自动删除文件类型：{}".format(types_to_delete if types_to_delete else "无"))  # 打印待删除类型
    source_files, missing_items, duplicate_items = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)  # 构建复制文件字典并查错
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]  # 整理得到工况目录下需顺序执行的文件名列表
    if not ensure_sources_exist(missing_items):  # 校验并处理源文件缺失
        sys.exit(1)  # 异常退出
    if not ensure_no_duplicate_targets(duplicate_items):  # 校验并处理复制文件名冲突
        sys.exit(1)  # 异常退出
    missing_waves = [("WAVE_FILES", p) for p in WAVE_FILES if not os.path.isfile(os.path.abspath(str(p)))]  # 校验全局波形存在性
    for idx, case in enumerate(PARAMETER_CASES, start=1):  # 校验各工况级波形存在性
        if isinstance(case, dict):  # 非法工况节点留给后续 config 校验报错
            missing_waves.extend(("第{}组工况".format(idx), p) for p in resolve_wave_files(case.get("config"))
                                 if not os.path.isfile(p))  # 记录该工况缺失的波形
    if missing_waves:  # 存在缺失波形文件
        print("错误：以下地震波文件不存在，请检查 WAVE_FILES 或工况级 run_cfg.wave_files：")  # 打印错误提示头
        for src, p in missing_waves:  # 遍历缺失记录
            print("  - [{}] {}".format(src, p))  # 打印来源与缺失路径
        sys.exit(1)  # 异常退出
    if not PARAMETER_CASES:  # 若工况列表为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组工况。")  # 打印工况缺失提示
        sys.exit(1)  # 异常退出
    folder_plan = []  # 初始化工况目录计划列表
    seen = set()  # 初始化工况文件夹去重名字集合
    for idx, case in enumerate(PARAMETER_CASES, start=1):  # 遍历所有定义的工况
        if not isinstance(case, dict) or "config" not in case:  # 校验工况节点合法性
            print("错误：第 {} 组工况缺少 config。".format(idx))  # 打印格式错误提示
            sys.exit(1)  # 异常退出
        folder_name = build_folder_name(case)  # 确定工况目录名
        if folder_name in seen:  # 若检测到重复的工况目录名
            print("错误：工况生成了重复文件夹名 -> {}".format(folder_name))  # 打印命名冲突提示
            sys.exit(1)  # 异常退出
        seen.add(folder_name)  # 加入已生成名称集合中
        folder_plan.append((folder_name, case["config"]))  # 记录到待处理工况目录中
    os.makedirs(root_dir, exist_ok=True)  # 创建结果总输出根目录
    failed_folders = []  # 初始化保存失败的工况目录列表

    def process_folder(item):  # 处理单个工况任务的核心闭包函数
        """处理单个工况任务的核心闭包函数。

        参数说明:
            item (tuple): (工况文件夹名, 参数覆盖字典)。

        返回值:
            tuple: (工况绝对路径, 执行是否成功标志)。
        """
        folder_name, config = item  # 解包工况名与配置字典
        folder_path = os.path.join(root_dir, folder_name)  # 拼接工况绝对路径
        print("\n==============================")  # 打印工况分隔符
        print("开始处理文件夹：{}".format(folder_path))  # 打印开始处理的工况路径
        create_and_fill_folder(folder_path, source_files, inject_wave_files(config))  # 建立工况文件夹并注入配置（含地震波路径）与物理源文件
        ok = run_scripts_in_folder(folder_path, run_order)  # 顺序在目录下启动指定运行脚本
        if types_to_delete:  # 若设置了需要清理的中间格式文件
            delete_files_by_type(folder_path, types_to_delete)  # 清理中间大文件
        else:  # 若不清理
            print("已跳过文件删除（没有指定要删除的文件类型）。")  # 打印跳过提示
        return folder_path, ok  # 返回工况绝对路径与执行结果

    print("开始并行批处理，最大并发任务数：{}".format(MAX_WORKERS))  # 打印批处理启动信息
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:  # 开启并发线程池
        for folder_path, ok in executor.map(process_folder, folder_plan):  # 并行分发并阻塞收集返回状态
            if not ok:  # 若该工况执行失败
                failed_folders.append(folder_path)  # 记录该失败工况的绝对路径
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
