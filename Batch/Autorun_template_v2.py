# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""通用批量运行与配置注入的 Autorun 模板（v2）。

用户可直接复制此模板并修改工况配置列表与路径，用于各类有限元参数扫描。
支持：
  1. 脚本的顺序执行（在各工况目录下独立运行）。
  2. 通过 json 配置注入参数，避免正则修改脚本。
  3. 多线程并行执行。
  4. 自动清理指定的中间大文件（如 odb/inp 等）。
  5. 执行全局汇总与出图的后处理脚本。
"""

import os  # 导入操作系统相关路径与目录操作模块
import json  # 导入 JSON 模块用于写出注入配置文件
import shutil  # 导入文件复制与高层级文件操作模块
import subprocess  # 导入子进程执行模块以运行其他脚本
import sys  # 导入系统模块用于获取 Python 解释器路径与退出程序
import concurrent.futures  # 导入并发模块以实现多工况文件夹并行执行
import hashlib  # 导入哈希模块用于计算配置与源文件散列
import datetime  # 导入时间模块用于清单追踪
import math  # 导入数学模块用于临界角计算

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 设置默认的模型根目录，各工况文件夹建在此
FOLDER_PREFIX = "case-"  # 各工况文件夹的命名统一前缀
DELETE_FILE_TYPES = [".odb", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", "jnl", "com"]  # 执行后自动删除的中间文件类型，留空则不删除
MAX_WORKERS = 2  # 并行处理工况文件夹的最大线程数
CONFIG_FILENAME = "case_config.json"  # 注入给建模或计算脚本的配置文件名

# Abaqus 启动路径与需要由 Abaqus Python 运行的脚本名单
ABAQUS_CMD = os.environ.get('ABAQUS_CMD') or r'C:\SIMULIA\Commands\abaqus.bat'
ABAQUS_SCRIPTS = {
    'slope_frame_ssi_full_v2.py',
    'Postprocess_All_surface_v2.py',
}

# 注入配置的顶层键白名单
ALLOWED_CONFIG_KEYS = {
    'material_cfg', 'geometry_cfg', 'damping_cfg', 'mesh_cfg',
    'time_cfg', 'freefield_cfg', 'run_cfg', 'eql_cfg', 'tssi_cfg',
}

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行的脚本绝对路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\slope_frame_ssi_full_v2.py",  # 建模脚本（读取 case_config.json，含层内材料一致化/网格自适应/时间步校验）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Postprocess_All_surface_v2.py",  # 后处理提取脚本路径 / 提取单工况数据
]

POST_SCRIPT_SEQUENCE = [  # 全部工况求解完成后自动在根目录执行的全局后处理脚本绝对路径
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Collect_All_results_v2.py",  # 汇总各工况 case_meta.json 到 results/index.csv
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Plot_Hybrid_surface_v2.py",  # 绘图脚本路径 / 绘制汇总对比图表
]

PARAMETER_CASES = [  # 变参数工况列表，每项需包含 config 配置覆盖字典
    {
        "name": "case_example_1",  # 工况名称 / 可选，若省略则按 config 自动命名
        "config": {  # 参数覆盖字典 / 建模脚本中需通过读取 json 文件覆盖默认配置
            "material_cfg": {"angle": 0.0},  # SV 波入射角度（度）
            "geometry_cfg": {"slope_angle": 30.0},  # 几何配置 / 边坡角度
            "run_cfg": {"wave_files": [  # 工况级地震波 / 每个工况可指定不同波
                r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 本工况用 4Hz 主脉冲
            ]},
        }
    },
    {
        "name": "case_example_2",  # 工况名称 / 可选，若省略则按 config 自动命名
        "config": {  # 参数覆盖字典
            "material_cfg": {"angle": 15.0},  # SV 波入射角度（度）
            "geometry_cfg": {"slope_angle": 45.0},  # 几何配置
            "run_cfg": {"wave_files": [  # 工况级地震波 / 与工况1不同波
                r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_2Hz.txt",  # 本工况用 2Hz 补低频脉冲
            ]},
        }
    }
]


def get_git_commit():
    """获取当前 Git 提交哈希值。"""
    try:
        res = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT_DIR, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if res.returncode == 0:
            return res.stdout.decode('utf-8').strip()
    except Exception:
        pass
    return "unknown"


def compute_file_sha256(path):
    """计算文件的 SHA-256 哈希值。"""
    if not os.path.isfile(path):
        return "missing"
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def compute_dict_sha256(d):
    """计算字典的 SHA-256 哈希值。"""
    text = json.dumps(d or {}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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
    manifest_data = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
        except Exception:
            pass

    script_hashes = {}
    for name, path in source_files.items():
        script_hashes[name] = compute_file_sha256(path)

    manifest_data.update({
        'created_at': manifest_data.get('created_at') or datetime.datetime.now().isoformat(),
        'updated_at': datetime.datetime.now().isoformat(),
        'git_commit': get_git_commit(),
        'source_files': script_hashes,
    })

    cases = manifest_data.get('cases', {})
    for folder_name, config in folder_plan:
        config_hash = compute_dict_sha256(config)
        case_status = status_dict.get(folder_name, 'planned') if status_dict else 'planned'
        if folder_name not in cases:
            cases[folder_name] = {
                'case_id': folder_name,
                'config_sha256': config_hash,
                'status': case_status,
                'added_at': datetime.datetime.now().isoformat(),
            }
        else:
            cases[folder_name]['status'] = case_status
            cases[folder_name]['updated_at'] = datetime.datetime.now().isoformat()

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


def run_scripts_in_folder(folder_path, run_order):  # 在工况文件夹内按顺序执行指定的脚本
    """在工况文件夹内按顺序执行指定的脚本，自动分发 Abaqus cae 和普通 Python 解释器。

    参数说明:
        folder_path (str): 工况文件夹绝对路径。
        run_order (list): 执行脚本文件名（不含路径）顺序列表。

    返回值:
        bool: 若全部顺利执行返回 True，任意脚本执行失败则返回 False。
    """
    for idx, script_name in enumerate(run_order, start=1):  # 遍历待执行的脚本
        script_path = os.path.join(folder_path, script_name)  # 拼接绝对路径
        if not os.path.isfile(script_path):  # 若物理文件不存在
            print("错误：脚本不存在 -> {}".format(script_path))  # 打印不存在 of 错误提示
            return False  # 返回失败

        # 判断执行命令
        if script_name in ABAQUS_SCRIPTS:
            cmd = [ABAQUS_CMD, 'cae', 'noGUI=' + script_name]
            log_filename = "autorun_step{:02d}_model_{}.log".format(idx, os.path.splitext(script_name)[0])
        else:
            cmd = [sys.executable, script_name]
            log_filename = "autorun_step{:02d}_post_{}.log".format(idx, os.path.splitext(script_name)[0])

        log_path = os.path.join(folder_path, log_filename)
        print("开始执行: {} (命令: {})".format(script_name, ' '.join(cmd)))  # 打印启动执行提示

        env = os.environ.copy()
        if script_name not in ABAQUS_SCRIPTS:
            env['PYTHONIOENCODING'] = 'utf-8'

        with open(log_path, 'wb') as handle:
            handle.write("命令: {}\n工作目录: {}\n\n".format(' '.join(cmd), folder_path).encode('utf-8'))
            handle.flush()
            try:
                result = subprocess.run(cmd, cwd=folder_path, stdout=handle, stderr=subprocess.STDOUT, check=False, env=env)
                returncode = result.returncode
            except Exception as e:
                handle.write("\n子进程启动异常: {}\n".format(str(e)).encode('utf-8'))
                returncode = -999

        if returncode != 0:  # 若执行退出码不为 0
            print("错误：{} 执行失败，返回码={}，详情见日志：{}".format(script_name, returncode, log_path))  # 打印执行失败提示
            return False  # 返回失败
        print("完成执行：{}".format(script_name))  # 打印执行成功提示
    return True  # 返回成功


def delete_files_by_type(folder_path, file_types, run_ok):  # 永久删除指定后缀的中间文件以节省空间
    """在 QA 通过（run_ok 为 True 且存在数据产物）时清理大文件，否则完整保留以便诊断。

    参数说明:
        folder_path (str): 工况文件夹绝对路径。
        file_types (list): 待删除的文件后缀名列表。
        run_ok (bool): 工况步骤是否顺利运行完成。
    """
    if not file_types:  # 若未配置删除类型
        return  # 直接返回

    # QA与产物检查：如果没有成功运行，或者没有最终 NPZ 产物，跳过删除大文件
    npz_path = os.path.join(folder_path, 'surface_results.npz')
    has_npz = os.path.isfile(npz_path)

    if not (run_ok and has_npz):
        print("警告：由于工况执行失败或缺失最终 NPZ 产物，跳过中间大文件清理，完整保留以便诊断。")
        return

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
    source_files, missing_items, duplicate_items = build_source_files(SCRIPT_SEQUENCE)  # 构建复制文件字典并查错
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]  # 整理得到工况目录下需顺序执行的文件名列表
    if not ensure_sources_exist(missing_items):  # 校验并处理源文件缺失
        sys.exit(1)  # 异常退出
    if not ensure_no_duplicate_targets(duplicate_items):  # 校验并处理复制文件名冲突
        sys.exit(1)  # 异常退出
    if not PARAMETER_CASES:  # 若工况列表为空
        print("错误：PARAMETER_CASES 为空，请至少配置一组工况。")  # 打印工况缺失提示
        sys.exit(1)  # 异常退出

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
        folder_plan.append((folder_name, case["config"]))  # 记录到待处理工况目录中
        status_dict[folder_name] = 'planned'

    os.makedirs(root_dir, exist_ok=True)  # 创建结果总输出根目录

    # 写入初始 planned 清单
    _write_run_manifest(root_dir, folder_plan, source_files, status_dict)

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

        # 更新清单为 running 状态
        status_dict[folder_name] = 'running'
        _write_run_manifest(root_dir, folder_plan, source_files, status_dict)

        try:
            create_and_fill_folder(folder_path, source_files, config)  # 建立工况文件夹并注入配置与物理源文件
            ok = run_scripts_in_folder(folder_path, run_order)  # 顺序在目录下启动指定运行脚本
        except Exception as err:
            print("异常：创建或执行工况失败 -> {}".format(str(err)))
            ok = False

        if ok:
            status_dict[folder_name] = 'passed'
        else:
            status_dict[folder_name] = 'failed'
        _write_run_manifest(root_dir, folder_plan, source_files, status_dict)

        if types_to_delete:  # 若设置了需要清理的中间格式文件
            delete_files_by_type(folder_path, types_to_delete, ok)  # 清理中间大文件
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
