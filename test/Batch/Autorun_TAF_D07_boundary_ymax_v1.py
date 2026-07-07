# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
r"""D07 软厚层边界自由场诊断批处理。

运行示例：
  python C:\Users\12462\Documents\Code\AbqScripts\test\Batch\Autorun_TAF_D07_boundary_ymax_v1.py C:\Users\12462\Documents\Code\AbqScripts\test\Abaqus\taf-d07-boundary-ymax-test

只跑论文 Fig.15(b) 关键工况 D07：Vs1/Vs2=0.50、h1/(H-h)=0.75、theta=15deg。
"""

import concurrent.futures  # 并行执行工况
import json  # 写入 case_config.json
import os  # 路径处理
import shutil  # 复制脚本与波形
import subprocess  # 调用子脚本
import sys  # 读取命令行参数


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 未传参时退回脚本所在目录
FOLDER_PREFIX = "d07-"  # 工况文件夹前缀
CONFIG_FILENAME = "case_config.json"  # 建模脚本读取的配置文件名
MAX_WORKERS = 1  # Abaqus 单工况已用多核，默认串行更稳
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 自动清理的求解中间文件

REPO = r"C:\Users\12462\Documents\Code\AbqScripts"  # 仓库根目录

STATIC_SOURCE_PATHS = [  # 每个工况都拷入同一个 4Hz Ricker
    os.path.join(REPO, r"Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt"),  # D07 对应 a0=2.0 -> fc=4Hz
]

SCRIPT_SEQUENCE = [  # 每个工况内依次执行
    os.path.join(REPO, r"Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py"),  # 建模与求解
    os.path.join(REPO, r"Postprocess\General\Postprocess_PGA_v3.py"),  # 提取 PGA/时程
    os.path.join(REPO, r"Postprocess\General\Compute_TAF_v2.py"),  # 计算 TAF
]

POST_SCRIPT_SEQUENCE = [  # 全部工况完成后只汇总，不强制出 Fig15 图
    os.path.join(REPO, r"Postprocess\General\Collect_results_v2.py"),  # 复制 CSV 并生成 results/index.csv
]


def _layers_d07():  # 返回 D07 的三层材料配置
    """D07：表层 Vs1=400、覆盖层 Vs2=800、h1=150m。"""
    return [
        {"name": "surface", "velocity_ratio": 5.0, "poisson_ratio": 0.3, "density": 2500, "thickness": 150.0},
        {"name": "overlying", "velocity_ratio": 2.5, "poisson_ratio": 0.3, "density": 2500},
    ]


def _case(bottom_ymax_mode, damping_enable=True):  # 生成单个诊断工况配置
    """bottom_ymax_mode: local/upper/lower；damping_enable 控制材料阻尼。"""
    cfg = {
        "material_cfg": {
            "angle": 15,
            "layers": _layers_d07(),
        },
        "geometry_cfg": {
            "i": 45.0,
        },
        "time_cfg": {
            "tail_seconds": 6.0,
        },
        "freefield_cfg": {
            "pad_factor": 8,
            "bottom_ymax_mode": bottom_ymax_mode,
        },
        "damping_cfg": {
            "enable": bool(damping_enable),
            "constant_xi": None,
        },
    }
    if not damping_enable:
        cfg["freefield_cfg"]["include_damping"] = False  # 关阻尼时自由场也按弹性算
    return cfg


PARAMETER_CASES = [  # 最小诊断矩阵
    {"name": "local-damped", "config": _case("local", True)},  # 当前基准：底边界按局部地表柱高
    {"name": "upper-damped", "config": _case("upper", True)},  # 底边界统一用上平台柱
    {"name": "lower-damped", "config": _case("lower", True)},  # 底边界统一用下平台柱
    {"name": "local-nodamp", "config": _case("local", False)},  # 当前基准关材料阻尼
]


def build_source_files():  # 检查并汇总需要拷入工况目录的文件
    """返回 {目标文件名: 源路径}，若缺文件或重名冲突则直接退出。"""
    source_files = {}
    for source_path in STATIC_SOURCE_PATHS + SCRIPT_SEQUENCE:
        target_name = os.path.basename(source_path)
        if not os.path.isfile(source_path):
            raise RuntimeError("源文件不存在: %s" % source_path)
        if target_name in source_files and source_files[target_name] != source_path:
            raise RuntimeError("复制后文件重名: %s" % target_name)
        source_files[target_name] = source_path
    return source_files


def create_and_fill_folder(folder_path, source_files, config):  # 创建工况目录并写配置
    """拷贝脚本/波形并写入 case_config.json。"""
    os.makedirs(folder_path, exist_ok=True)
    for target_name, source_path in source_files.items():
        shutil.copy2(source_path, os.path.join(folder_path, target_name))
    with open(os.path.join(folder_path, CONFIG_FILENAME), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def run_scripts_in_folder(folder_path):  # 按顺序执行工况目录内脚本
    """任一脚本失败则返回 False。"""
    for source_path in SCRIPT_SEQUENCE:
        script_name = os.path.basename(source_path)
        print("开始执行: %s" % os.path.join(folder_path, script_name))
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)
        if result.returncode != 0:
            print("错误: %s 返回码=%s" % (script_name, result.returncode))
            return False
    return True


def delete_files_by_type(folder_path):  # 清理 Abaqus 中间文件
    """按扩展名删除求解中间文件。"""
    suffixes = {x.lower() for x in DELETE_FILE_TYPES}
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in suffixes:
            try:
                os.remove(path)
            except OSError as exc:
                print("警告: 删除失败 %s -> %s" % (path, exc))


def process_folder(item):  # 执行一个工况
    """item=(folder_name, config)，返回 (folder_path, ok)。"""
    folder_name, config = item
    folder_path = os.path.join(process_folder.root_dir, folder_name)
    print("\n==============================")
    print("开始处理: %s" % folder_path)
    create_and_fill_folder(folder_path, process_folder.source_files, config)
    ok = run_scripts_in_folder(folder_path)
    delete_files_by_type(folder_path)
    return folder_path, ok


def run_post_scripts(root_dir):  # 执行根目录级汇总脚本
    """拷贝并执行 POST_SCRIPT_SEQUENCE。"""
    for source_path in POST_SCRIPT_SEQUENCE:
        target_name = os.path.basename(source_path)
        target_path = os.path.join(root_dir, target_name)
        shutil.copy2(source_path, target_path)
        print("开始执行后处理: %s" % target_path)
        result = subprocess.run([sys.executable, target_name, root_dir], cwd=root_dir, check=False)
        if result.returncode != 0:
            raise RuntimeError("后处理失败: %s 返回码=%s" % (target_name, result.returncode))


def main():  # 批处理入口
    """创建 4 个 D07 诊断工况并顺序运行。"""
    root_dir = os.path.abspath(sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR)
    os.makedirs(root_dir, exist_ok=True)
    source_files = build_source_files()
    folder_plan = [(FOLDER_PREFIX + case["name"], case["config"]) for case in PARAMETER_CASES]
    process_folder.root_dir = root_dir
    process_folder.source_files = source_files

    failed = []
    print("目标根目录: %s" % root_dir)
    print("工况数: %d, 最大并发: %d" % (len(folder_plan), MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for folder_path, ok in executor.map(process_folder, folder_plan):
            if not ok:
                failed.append(folder_path)
    if failed:
        print("存在失败工况:")
        for path in failed:
            print("  - %s" % path)
        sys.exit(2)
    run_post_scripts(root_dir)
    print("全部完成: %s" % root_dir)


if __name__ == "__main__":
    main()
