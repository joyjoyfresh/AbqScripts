# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""软土场地【非线性放大·强度扫描】批处理 —— 课题主线。

主题：软土场地考虑土体非线性的地震响应放大随【输入强度】如何变化（这是论文未做的新内容）。
做法：同一软薄层工况(Vs1/Vs2=0.5,h1=50,斜入射15°,i45)，输入 2Hz Ricker 按 PGA 分 5 档(0.05~0.8g)，
      每档跑【非线性 EQL(等效线性,应变相容降Vs/增ξ)】；另跑 1 条【线性参照】(TAF 与幅值无关，取一档即可)。
预期：PGA↑ → 软层剪应变↑ → G/Gmax↓、ξ↑、共振频率下移 → 坡顶放大相对线性【下降并重分布】。
      这条"非线性如何抑制软土放大"的曲线就是核心成果，且不依赖 2D 俘获绝对值是否对上论文。

非线性方法：脚本内置 EQL(可切换 Darendeli/Seed-Idriss/Vucetic-Dobry 曲线)，mode='1d'(标准 SHAKE 式,稳)。
网格 4m/CPE4R(收敛已证够用)。每档用各自的分档 Ricker 输入。

运行(与 Autorun 同):  python Autorun_softsoil_nonlinear_v1.py [可选:工作目录]
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
WAVE_DIR = r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration"  # 分档 Ricker 所在目录

SCRIPT_SEQUENCE = [  # 每工况按顺序执行(建模用 v3)
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模脚本 v3(含 EQL)
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算
]
POST_SCRIPT_SEQUENCE = [  # 全部跑完后汇总 + 强度扫描判读
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Collect_results_v2.py",  # 汇总到 results/index.csv
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\test\softsoil_nonlinear_report_v1.py",  # 强度扫描对照表(线性 vs 非线性)
]


def _soft(eql_enable):  # 软薄层 + 4m/CPE4R + EQL 开关(每次返回新 dict)
    """eql_enable: True=非线性(EQL 1D 应变相容,Darendeli 曲线) / False=线性参照。"""
    return {
        "material_cfg": {"angle": 15, "layers": [
            {'name': 'surface', 'velocity_ratio': 5.0, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': 50.0},
            {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
        ]},
        "geometry_cfg": {"i": 45.0},
        "mesh_cfg": {"size": 4.0, "elem": "CPE4R"},
        "time_cfg": {"tail_seconds": 0.0},
        "eql_cfg": {"enable": bool(eql_enable), "mode": "1d", "curve": "darendeli", "nonlinear_layers": ["surface"]},
    }


def _wave(tag):  # 分档 Ricker 路径
    return os.path.join(WAVE_DIR, "ricker2Hz_%s.txt" % tag)


# 工况：1 条线性参照(TAF 与幅值无关) + 5 档非线性 EQL
PARAMETER_CASES = [
    {"config": _soft(False), "name": "lin-ref",    "input_file": _wave("0p2g")},   # 线性参照(任一档即可，取0.2g)
    {"config": _soft(True),  "name": "nlin-0p05g", "input_file": _wave("0p05g")},  # 非线性 0.05g(近线性)
    {"config": _soft(True),  "name": "nlin-0p1g",  "input_file": _wave("0p1g")},   # 非线性 0.1g
    {"config": _soft(True),  "name": "nlin-0p2g",  "input_file": _wave("0p2g")},   # 非线性 0.2g
    {"config": _soft(True),  "name": "nlin-0p4g",  "input_file": _wave("0p4g")},   # 非线性 0.4g
    {"config": _soft(True),  "name": "nlin-0p8g",  "input_file": _wave("0p8g")},   # 非线性 0.8g(强非线性)
]


def build_source_files(seq):  # 脚本源映射 + 缺失检测
    src = {}; miss = []
    for sp in seq:
        if os.path.isfile(sp):
            src[os.path.basename(sp)] = sp
        else:
            miss.append(sp)
    return src, miss


def create_and_fill_folder(folder, src, config, input_file):  # 建目录+拷脚本+拷该档 Ricker+写配置
    """每工况只拷自己那一档 Ricker(find_acc_txt 跑目录下所有 .txt，故一folder一输入)。"""
    os.makedirs(folder, exist_ok=True)
    for name, sp in src.items():
        shutil.copy2(sp, os.path.join(folder, name))
    if not os.path.isfile(input_file):
        raise IOError("分档输入不存在: %s" % input_file)
    shutil.copy2(input_file, os.path.join(folder, os.path.basename(input_file)))
    with open(os.path.join(folder, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config or {}, f, ensure_ascii=False, indent=2)


def run_scripts_in_folder(folder, run_order):  # 目录内顺序执行
    for name in run_order:
        sp = os.path.join(folder, name)
        if not os.path.isfile(sp):
            print("错误：脚本不存在 -> %s" % sp); return False
        print("开始执行：%s" % name)
        if subprocess.run([sys.executable, name], cwd=folder, check=False).returncode != 0:
            print("错误：%s 执行失败" % name); return False
        print("完成执行：%s" % name)
    return True


def delete_files_by_type(folder, types):  # 删中间文件
    norm = {t.lower() for t in types}
    for name in sorted(os.listdir(folder)):
        fp = os.path.join(folder, name)
        if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in norm:
            try:
                os.remove(fp)
            except OSError as e:
                print("删除失败 %s: %s" % (fp, e))


def main():  # 主流程
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
    if failed:
        print("存在失败工况(%d)：" % len(failed)); [print("  -", f) for f in failed]
    else:
        print("全部 %d 个工况完成。" % len(PARAMETER_CASES))
    for sp in POST_SCRIPT_SEQUENCE:
        if not os.path.isfile(sp):
            print("后处理脚本缺失: %s" % sp); continue
        shutil.copy2(sp, os.path.join(root, os.path.basename(sp)))
        print("执行后处理：%s" % os.path.basename(sp))
        subprocess.run([sys.executable, os.path.basename(sp), root], cwd=root, check=False)


if __name__ == "__main__":
    main()
