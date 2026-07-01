# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""方法验证 Tier 4【时间步收敛 + 边界距离无关性】批处理。

对同一软薄层工况(线性,Vs1/Vs2=0.5,h1=50,斜入射15°,i45,4m/CPE4R)，做两组数值验证：
  ① 时间步收敛：dt=0.001(base) vs dt=0.0005(dthalf，同波形细采样)，坡面响应应不变(<2%)。
  ② 边界距离无关性：加宽(left_flat/total_L 加大)、加深(bedrock_thickness 加大)，坡面响应应不变(<几%)。
均为线性(eql 关)，证 FE 本身的数值收敛与边界充分远，与论文图15绝对值无关。

运行(与 Autorun 同):  python Autorun_verify_numerical_v1.py [可选:工作目录]
跑完自动出对照表(各工况 远场/坡顶 TAF 与 base 的偏差)。
"""

import os, json, shutil, subprocess, sys  # 标准库

ROOT_DIR = sys.argv[1] if len(sys.argv) >= 2 else os.path.dirname(os.path.abspath(__file__))  # 工况根目录(可命令行覆盖)
FOLDER_PREFIX = "multi-"  # 文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 删中间文件
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名
WAVE_DIR = r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration"  # 输入波目录

SCRIPT_SEQUENCE = [
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模 v3
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF
]
POST_SCRIPT_SEQUENCE = [
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py",
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\test\verify_numerical_report_v1.py",  # 对照表
]


def _soft(geom_extra=None):  # 软薄层 4m/CPE4R 线性 + 可选几何覆盖
    cfg = {
        "material_cfg": {"angle": 15, "layers": [
            {'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 50.0},
            {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
        ]},
        "geometry_cfg": dict({"i": 45.0}, **(geom_extra or {})),
        "mesh_cfg": {"size": 4.0, "elem": "CPE4R"},
        "time_cfg": {"tail_seconds": 0.0},
        "eql_cfg": {"enable": False},  # 线性
    }
    return cfg


def _wave(name):
    return os.path.join(WAVE_DIR, name)


PARAMETER_CASES = [
    # 基线：默认几何(left_flat=1000,total_L=1800,bedrock=200)，dt=0.001
    {"config": _soft(), "name": "verify-base", "input_file": _wave("ricker_wavelet_8Hz.txt")},
    # ① 时间步减半：dt=0.0005(同波形)，几何同 base
    {"config": _soft(), "name": "verify-dthalf", "input_file": _wave("ricker_wavelet_8Hz_dt0005.txt")},
    # ② 边界加宽：left_flat 1000→1600, total_L 1800→2800(侧边界远离坡体)
    {"config": _soft({"left_flat": 1600.0, "total_L": 2800.0}), "name": "verify-wide", "input_file": _wave("ricker_wavelet_8Hz.txt")},
    # ② 边界加深：bedrock_thickness 200→500(底边界下移)
    {"config": _soft({"bedrock_thickness": 500.0}), "name": "verify-deep", "input_file": _wave("ricker_wavelet_8Hz.txt")},
]


def build_source_files(seq):
    src = {}; miss = []
    for sp in seq:
        if os.path.isfile(sp):
            src[os.path.basename(sp)] = sp
        else:
            miss.append(sp)
    return src, miss


def create_and_fill_folder(folder, src, config, input_file):
    os.makedirs(folder, exist_ok=True)
    for name, sp in src.items():
        shutil.copy2(sp, os.path.join(folder, name))
    if not os.path.isfile(input_file):
        raise IOError("输入不存在: %s" % input_file)
    shutil.copy2(input_file, os.path.join(folder, os.path.basename(input_file)))
    with open(os.path.join(folder, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config or {}, f, ensure_ascii=False, indent=2)


def run_scripts_in_folder(folder, run_order):
    for name in run_order:
        sp = os.path.join(folder, name)
        if not os.path.isfile(sp):
            print("错误：脚本不存在 -> %s" % sp); return False
        print("开始执行：%s" % name)
        if subprocess.run([sys.executable, name], cwd=folder, check=False).returncode != 0:
            print("错误：%s 执行失败" % name); return False
        print("完成执行：%s" % name)
    return True


def delete_files_by_type(folder, types):
    norm = {t.lower() for t in types}
    for name in sorted(os.listdir(folder)):
        fp = os.path.join(folder, name)
        if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in norm:
            try:
                os.remove(fp)
            except OSError as e:
                print("删除失败 %s: %s" % (fp, e))


def main():
    root = ROOT_DIR
    print("工况根目录：%s" % root)
    src, miss = build_source_files(SCRIPT_SEQUENCE)
    if miss:
        print("错误：脚本缺失："); [print("  -", m) for m in miss]; sys.exit(1)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    os.makedirs(root, exist_ok=True)
    seen = set(); failed = []
    for case in PARAMETER_CASES:
        name = FOLDER_PREFIX + case["name"]
        if name in seen:
            print("错误：重复文件夹名 %s" % name); sys.exit(1)
        seen.add(name)
        folder = os.path.join(root, name)
        print("\n==============================\n处理：%s" % folder)
        create_and_fill_folder(folder, src, case["config"], case["input_file"])
        if not run_scripts_in_folder(folder, run_order):
            failed.append(folder)
        delete_files_by_type(folder, DELETE_FILE_TYPES)
    print("\n==============================")
    print(("存在失败工况(%d)" % len(failed)) if failed else ("全部 %d 个验证工况完成。" % len(PARAMETER_CASES)))
    for sp in POST_SCRIPT_SEQUENCE:
        if not os.path.isfile(sp):
            print("后处理脚本缺失: %s" % sp); continue
        shutil.copy2(sp, os.path.join(root, os.path.basename(sp)))
        print("执行后处理：%s" % os.path.basename(sp))
        subprocess.run([sys.executable, os.path.basename(sp), root], cwd=root, check=False)


if __name__ == "__main__":
    main()
