# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软楔【持续正弦(谐波)激励】批处理 —— 收尾实验：稳态放大 vs 瞬态。

目的：模态测试已证软楔 2D 共振模存在(2.91/3.84Hz 等)，但瞬态 Ricker 只激出坡顶放大 ~2.0。
      本实验改用【多周期持续正弦】在各软楔模态频率上稳态驱动，看坡顶【稳态】放大能否冲高：
        稳态明显>瞬态 2.0 → 纯属"短脉冲激不满共振"，则论文的 Ricker 也该 ~2.0，7.6 更可疑；
        稳态也上不去(~2.0) → 是更深的方法层面差异(等效力+VAB vs SPECFEM)。

每工况用各自的正弦输入(频率不同)，软薄层(Vs1/Vs2=0.5,h1=50,斜入射15°,i45)，网格 4m/CPE4R
(收敛研究已证 4m 足够、升阶/加密只差 3.6%，故用快配置)，无尾段(稳态在正弦持续段内)。

运行(与 Autorun 同):  python Autorun_softwedge_harmonic_v1.py [可选:工作目录]
"""

import os  # 路径与目录
import json  # 写 case_config.json
import shutil  # 复制文件
import subprocess  # 子进程执行
import sys  # 解释器与参数

ROOT_DIR = sys.argv[1] if len(sys.argv) >= 2 else os.path.dirname(os.path.abspath(__file__))  # 工况根目录(可命令行覆盖)
FOLDER_PREFIX = "multi-"  # 文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 跑完删除的中间文件
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名

WAVE_DIR = r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration"  # 正弦输入所在目录

# 每工况按顺序执行的脚本(建模用 v3)
SCRIPT_SEQUENCE = [
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模脚本 v3
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算
]

POST_SCRIPT_SEQUENCE = [  # 全部跑完后汇总 + 谐波专用判读
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py",  # 汇总到 results/index.csv
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\test\softwedge_harmonic_report_v1.py",  # 谐波对照表+判读(稳态 vs 瞬态)
]


def _soft_4m():  # 软薄层 + 4m/CPE4R 配置(每次返回新 dict)
    """表层 Vs1/Vs2=0.5,h1=50,斜入射15°,i45；4m 减缩积分；无尾段。"""
    return {
        "material_cfg": {"angle": 15, "layers": [
            {'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 50.0},
            {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
        ]},
        "geometry_cfg": {"i": 45.0},
        "mesh_cfg": {"size": 4.0, "elem": "CPE4R"},
        "time_cfg": {"tail_seconds": 0.0},
    }


# 各工况：频率 -> 正弦输入文件 + 文件夹名。2.0Hz=软层一维基频(对照)，2.91/3.84Hz=软楔 2D 局部模态
PARAMETER_CASES = [
    {"config": _soft_4m(), "name": "harm-sine2Hz",    "input_file": os.path.join(WAVE_DIR, "sine_2Hz.txt")},     # 1D 基频(对照)
    {"config": _soft_4m(), "name": "harm-sine2p91Hz", "input_file": os.path.join(WAVE_DIR, "sine_2p91Hz.txt")},  # 软楔模态 mode47
    {"config": _soft_4m(), "name": "harm-sine3p84Hz", "input_file": os.path.join(WAVE_DIR, "sine_3p84Hz.txt")},  # 软楔模态 mode95(最强)
]


def build_source_files(script_sequence):  # 脚本源映射 + 缺失检测
    """返回 (源文件名->路径 映射, 缺失列表)。仅脚本(输入波按工况单独拷)。"""
    source_files = {}; missing = []
    for sp in script_sequence:
        name = os.path.basename(sp)
        if not os.path.isfile(sp):
            missing.append(sp); continue
        source_files[name] = sp
    return source_files, missing


def create_and_fill_folder(folder_path, source_files, config, input_file):  # 建目录+拷脚本+拷该工况正弦+写配置
    """每工况只拷【自己的】正弦输入(find_acc_txt 会跑目录下所有 .txt，故一folder一输入)。"""
    os.makedirs(folder_path, exist_ok=True)
    for name, src in source_files.items():
        shutil.copy2(src, os.path.join(folder_path, name))  # 拷脚本
    if not os.path.isfile(input_file):
        raise IOError("正弦输入不存在: %s" % input_file)  # 输入缺失直接报错
    shutil.copy2(input_file, os.path.join(folder_path, os.path.basename(input_file)))  # 拷该工况正弦
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config or {}, f, ensure_ascii=False, indent=2)  # 写注入配置


def run_scripts_in_folder(folder_path, run_order):  # 目录内顺序执行脚本
    """逐个用当前解释器执行；任一失败返回 False。"""
    for name in run_order:
        sp = os.path.join(folder_path, name)
        if not os.path.isfile(sp):
            print("错误：脚本不存在 -> %s" % sp); return False
        print("开始执行：%s" % name)
        if subprocess.run([sys.executable, name], cwd=folder_path, check=False).returncode != 0:
            print("错误：%s 执行失败" % name); return False
        print("完成执行：%s" % name)
    return True


def delete_files_by_type(folder_path, file_types):  # 删中间文件
    """按扩展名永久删除中间文件。"""
    norm = {t.lower() for t in file_types}
    for name in sorted(os.listdir(folder_path)):
        fp = os.path.join(folder_path, name)
        if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in norm:
            try:
                os.remove(fp)
            except OSError as e:
                print("删除失败 %s: %s" % (fp, e))


def main():  # 主流程
    """规划工况 → 串行建模/后处理 → 清理 → 汇总。"""
    root = ROOT_DIR
    print("工况根目录：%s" % root)
    source_files, missing = build_source_files(SCRIPT_SEQUENCE)
    if missing:
        print("错误：以下脚本缺失："); [print("  -", m) for m in missing]; sys.exit(1)
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
        create_and_fill_folder(folder, source_files, case["config"], case["input_file"])
        ok = run_scripts_in_folder(folder, run_order)
        delete_files_by_type(folder, DELETE_FILE_TYPES)
        if not ok:
            failed.append(folder)
    print("\n==============================")
    if failed:
        print("存在失败工况(%d)：" % len(failed)); [print("  -", f) for f in failed]
    else:
        print("全部 %d 个谐波工况完成。" % len(PARAMETER_CASES))
    # 后处理汇总
    for sp in POST_SCRIPT_SEQUENCE:
        if not os.path.isfile(sp):
            print("后处理脚本缺失: %s" % sp); continue
        dst = os.path.join(root, os.path.basename(sp)); shutil.copy2(sp, dst)
        print("执行后处理：%s" % os.path.basename(sp))
        subprocess.run([sys.executable, os.path.basename(sp), root], cwd=root, check=False)


if __name__ == "__main__":
    main()
