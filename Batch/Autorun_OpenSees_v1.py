# -*- coding: utf-8 -*-
"""批量跑【多种变参数工况】的 OpenSees 动力模拟合并版 Autorun。

统一调用合并版 OpenSees 建模与后处理一体化脚本 Modeling/Multi/VAB_oblique_TAF_multilayer_opensees_v1.py，
摆脱对 Abaqus 软件的依赖，使用普通的 Python 解释器即可一键跑完。
"""

import os
import json
import shutil
import subprocess
import sys
import concurrent.futures

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_PREFIX = "multi-"
DELETE_FILE_TYPES = [] # OpenSees 版本无大文件需要自动删除
MAX_WORKERS = 4 # 并行处理的最大线程数（OpenSees 计算快，可调高并发）
CONFIG_FILENAME = "case_config.json"

STATIC_SOURCE_PATHS = [
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_6Hz.txt",
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_8Hz.txt",
]

SCRIPT_SEQUENCE = [
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\OpenSees\VAB_oblique_OpenSees_v1.py", # 一体化建模模拟与PGA提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py", # 计算 TAF
]

POST_SCRIPT_SEQUENCE = [
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py", # 汇总各工况 case_meta.json
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_Multi_TAF_v3.py", # 双层图8
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Multi\Plot_Fig15_compare_v3.py", # 三层图15
]

def _layers3(surf_vr, surf_thick):
    return [
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
    ]

# 与原 Autorun_TAF_multilayer_v3.py 完全一致的工况表
PARAMETER_CASES = [
    # ---- A) 三层（论文图15）：i=45 固定，软/硬×厚度×角度 = 8 工况 ----
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers3(5.0, 50.0)},  "geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers3(1.25, 50.0)}, "geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 15, "layers": _layers3(5.0, 50.0)},  "geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 15, "layers": _layers3(1.25, 50.0)}, "geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers3(5.0, 150.0)}, "geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 0,  "layers": _layers3(1.25, 150.0)},"geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 15, "layers": _layers3(5.0, 150.0)}, "geometry_cfg": {"i": 45.0}}},
    {"config": {"material_cfg": {"angle": 15, "layers": _layers3(1.25, 150.0)},"geometry_cfg": {"i": 45.0}}},
    # ---- B) 双层（沿用默认 overlying，仅扫坡角与入射角）= 4 工况 ----
    {"config": {"material_cfg": {"angle": 0},  "geometry_cfg": {"i": 30.0}}},
    {"config": {"material_cfg": {"angle": 15}, "geometry_cfg": {"i": 30.0}}},
    {"config": {"material_cfg": {"angle": 0},  "geometry_cfg": {"i": 60.0}}},
    {"config": {"material_cfg": {"angle": 15}, "geometry_cfg": {"i": 60.0}}},
]

def build_source_files(static_source_paths, script_sequence):
    source_files = {}
    missing = []
    duplicate_names = []
    for source_path in list(static_source_paths) + list(script_sequence):
        target_name = os.path.basename(source_path)
        if not os.path.isfile(source_path):
            missing.append((target_name, source_path)); continue
        if target_name in source_files and source_files[target_name] != source_path:
            duplicate_names.append((target_name, source_path, source_files[target_name])); continue
        source_files[target_name] = source_path
    return source_files, missing, duplicate_names

def ensure_sources_exist(missing_items):
    if not missing_items:
        return True
    print("错误：以下源文件不存在，请先检查配置路径：")
    for name, path in missing_items:
        print("  - {} -> {}".format(name, path))
    return False

def ensure_no_duplicate_targets(duplicate_items):
    if not duplicate_items:
        return True
    print("错误：以下文件复制后会重名，请调整路径或文件名：")
    for target_name, current_path, existing_path in duplicate_items:
        print("  - {} -> {} (已存在来源: {})".format(target_name, current_path, existing_path))
    return False

def _fmt_num(v):
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else ('%g' % f)
    except (TypeError, ValueError):
        return str(v)

def _sanitize(text):
    import re
    return re.sub(r"[^0-9A-Za-z._-]+", "-", str(text)).strip("-") or "x"

def name_from_config(config):
    mat = config.get("material_cfg") or {}
    geo = config.get("geometry_cfg") or {}
    tokens = []
    layers = mat.get("layers")
    if isinstance(layers, list):
        segs = []
        for L in layers:
            seg = "vr" + _fmt_num(L.get("velocity_ratio"))
            if L.get("thickness") is not None:
                seg += "t" + _fmt_num(L["thickness"])
            segs.append(seg)
        tokens.append("L%d_%s" % (len(layers), "-".join(segs)))
    if "angle" in mat:
        tokens.append("a" + _fmt_num(mat["angle"]))
    for key, pre in (("i", "i"), ("H_minus_h", "H"), ("h_over_H", "hoH"),
                     ("bedrock_thickness", "br"), ("total_L", "L"), ("left_flat", "lf")):
        if key in geo:
            tokens.append(pre + _fmt_num(geo[key]))
    if config.get("mesh_size") is not None:
        tokens.append("m" + _fmt_num(config["mesh_size"]))
    for scope, skip in ((mat, {"angle", "layers"}), (geo, {"i", "H_minus_h", "h_over_H", "bedrock_thickness", "total_L", "left_flat"})):
        for k, v in scope.items():
            if k in skip or isinstance(v, (dict, list)):
                continue
            tokens.append("%s%s" % (_sanitize(k), _fmt_num(v)))
    return "-".join(tokens) if tokens else "default"

def build_folder_name(case):
    tag = case.get("name") or case.get("folder_tag")
    if not tag:
        tag = name_from_config(case.get("config") or {})
    return "{}{}".format(FOLDER_PREFIX, _sanitize(tag))

def create_and_fill_folder(folder_path, source_files, config):
    os.makedirs(folder_path, exist_ok=True)
    for target_name, src_path in source_files.items():
        shutil.copy2(src_path, os.path.join(folder_path, target_name))
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config or {}, f, ensure_ascii=False, indent=2)

def run_scripts_in_folder(folder_path, run_order):
    for script_name in run_order:
        script_path = os.path.join(folder_path, script_name)
        if not os.path.isfile(script_path):
            print("错误：脚本不存在 -> {}".format(script_path)); return False
        print("开始执行：{}".format(script_path))
        result = subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)
        if result.returncode != 0:
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode)); return False
        print("完成执行：{}".format(script_name))
    return True

def delete_files_by_type(folder_path, file_types):
    if not file_types:
        return
    normalized = {t.lower() if t.startswith(".") else "." + t.lower() for t in file_types}
    failed = []
    for name in sorted(os.listdir(folder_path)):
        fp = os.path.join(folder_path, name)
        if not os.path.isfile(fp):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in normalized:
            continue
        try:
            os.remove(fp)
        except OSError as exc:
            failed.append((fp, str(exc)))
    if failed:
        print("警告：以下文件删除失败：")
        for fp, err in failed:
            print("  - {} -> {}".format(fp, err))

def main():
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR
    types_to_delete = list(DELETE_FILE_TYPES)
    print("目标根目录：{}".format(root_dir))
    source_files, missing_items, duplicate_items = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    if not ensure_sources_exist(missing_items):
        sys.exit(1)
    if not ensure_no_duplicate_targets(duplicate_items):
        sys.exit(1)
    if not PARAMETER_CASES:
        print("错误：PARAMETER_CASES 为空，请至少配置一组工况。"); sys.exit(1)
        
    folder_plan = []
    seen = set()
    for idx, case in enumerate(PARAMETER_CASES, start=1):
        if not isinstance(case, dict) or "config" not in case:
            print("错误：第 {} 组工况缺少 config。".format(idx)); sys.exit(1)
        folder_name = build_folder_name(case)
        if folder_name in seen:
            print("错误：工况生成了重复文件夹名 -> {}".format(folder_name)); sys.exit(1)
        seen.add(folder_name)
        folder_plan.append((folder_name, case["config"]))
    os.makedirs(root_dir, exist_ok=True)
    failed_folders = []

    def process_folder(item):
        folder_name, config = item
        folder_path = os.path.join(root_dir, folder_name)
        print("\n==============================")
        print("开始处理文件夹：{}".format(folder_path))
        create_and_fill_folder(folder_path, source_files, config)
        ok = run_scripts_in_folder(folder_path, run_order)
        if types_to_delete:
            delete_files_by_type(folder_path, types_to_delete)
        return folder_path, ok

    print("开始并行批处理，最大并发任务数：{}".format(MAX_WORKERS))
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for folder_path, ok in executor.map(process_folder, folder_plan):
            if not ok:
                failed_folders.append(folder_path)
    print("\n==============================")
    if failed_folders:
        print("批处理结束：存在失败文件夹（{}个）。".format(len(failed_folders)))
        for path in failed_folders:
            print("  - {}".format(path))
        sys.exit(2)
    print("批处理结束：全部 {} 个工况文件夹处理完成。".format(len(folder_plan)))

    # 后处理阶段
    print("\n==============================")
    print("开始自动后处理脚本阶段...")
    post_run_order = []
    for src_path in POST_SCRIPT_SEQUENCE:
        if not os.path.isfile(src_path):
            print("错误：后处理脚本缺失 -> {}".format(src_path))
            sys.exit(3)
        target_name = os.path.basename(src_path)
        dst_path = os.path.join(root_dir, target_name)
        shutil.copy2(src_path, dst_path)
        post_run_order.append(target_name)
        print("已拷贝后处理脚本：{}".format(target_name))

    for script_name in post_run_order:
        script_path = os.path.join(root_dir, script_name)
        print("开始执行后处理：{}".format(script_path))
        result = subprocess.run([sys.executable, script_name, root_dir], cwd=root_dir, check=False)
        if result.returncode != 0:
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode))
            sys.exit(4)
        print("完成后处理：{}".format(script_name))

if __name__ == "__main__":
    main()
