# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""【P0#1 重力两步法 冒烟测试】Autorun —— gravity=off(v1基线) vs structure(Level A) 对照。

验证目标：`Modeling/Hybrid/slope_frame_ssi_full_v2.py` 新增的重力两步法(动力步前插 Static
`Step-gravity` 施加结构自重)能建步、能收敛，且柱底轴力命中手算锚点(§5 QA，偏差<2%)。
两工况同几何同波同框架，仅 `tssi_cfg.gravity` 不同：
  - grav-off ：gravity='off'   —— 无静力步，等于 v1 基线行为(对照，verify 脚本应报"跳过")。
  - grav-on  ：gravity='structure' —— Level A 仅结构自重；verify 脚本核对柱底轴力。
均用 elastic(nonlinear=False) 小框架，先把重力"管道"验通，再谈 CDP+重力。

缩小规模沿用 step2b 小几何 + 2 层 1 跨小框架，仅一条 4Hz Ricker 波，求解快。
柱底轴力手算(默认 2 层 1 跨 2 柱, floor_mass=5e4)：2×5e4×9.81/2 ≈ 490.5 kN/柱。

运行方式（本 Autorun 用【系统 Python3】起——它内部再用 `abaqus cae noGUI=` 派发建模/读odb子脚本到
abaqus 内核；不能用 abaqus python 起本文件，abaqus2021 内核 Py2.7 无 subprocess.run）：
  python test/Batch/Autorun_gravity_smoketest_v1.py  C:\\Users\\12462\\Documents\\Code\\AbqScripts\\test\\Abaqus\\gravity_smoketest
求解产物大，默认已指到 test/Abaqus/gravity_smoketest（gitignore 排除，不入库）。
已实跑验证：grav-on 柱底轴力 490.7 vs 手算 490.5 kN/柱(+0.04%,PASS)；grav-off 单步无静力步(=v1)。
"""

import os  # 路径与目录
import json  # 写 case_config.json
import glob  # 扫描结果文件
import shutil  # 复制源文件
import subprocess  # 执行子进程
import sys  # 解释器与参数
import concurrent.futures  # 多文件夹并行

# 默认输出根目录：test/Abaqus/gravity_smoketest（gitignore 排除，不入库）
DEFAULT_OUT = r"C:\Users\12462\Documents\Code\AbqScripts\test\Abaqus\gravity_smoketest"
FOLDER_PREFIX = "grav-"  # 文件夹前缀
DELETE_FILE_TYPES = []  # 冒烟：保留 .msg/.sta/.dat 收敛诊断
MAX_WORKERS = 2  # 并行文件夹数(2 工况)
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名

STATIC_SOURCE_PATHS = [  # 随每个工况拷入的固定源(仅输入波)
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4Hz Ricker
]

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Hybrid\slope_frame_ssi_full_v2.py",  # TSSI 建模(读 case_config.json)
    r"C:\Users\12462\Documents\Code\AbqScripts\test\Batch\verify_gravity_axial_v1.py",  # 柱底轴力校验(读 odb)
]

# 缩小几何+网格，加速冒烟。注意：v2 用【新无量纲几何 schema】(slope_height 唯一绝对尺度)，
# 非 v1 的 H_minus_h/i/h_over_H；土层默认 400m 太深，须一并覆盖为薄软层适配小坡。
SMALL_GEOMETRY = {"slope_height": 30.0, "slope_angle": 45.0}  # 小坡 hs=30m
SMALL_MATERIAL = {"layers": [  # 覆盖默认 50+350m 深土，改单薄软层适配浅模型
    {"name": "surface", "vs": 400.0, "poisson_ratio": 0.3, "density": 2500, "thickness": 10},
]}
SMALL_MESH = {"size": 5.0, "auto": False, "graded": False, "elem": "CPE4R"}  # 非自适应粗网格
SMALL_FRAME = {"n_story": 2, "n_bay": 1}  # 2 层 1 跨最小规模


def build_cases():  # 生成 grav-off/grav-on 两对照工况(同几何同波，仅 gravity 不同)
    """每项={name, gravity, config}。"""
    base_common = {"geometry_cfg": SMALL_GEOMETRY, "material_cfg": SMALL_MATERIAL,
                   "mesh_cfg": SMALL_MESH, "frame_cfg": SMALL_FRAME}
    cases = []
    for name, gravity in (("off", "off"), ("on", "structure")):
        cfg = dict(base_common)  # 浅拷贝共有项
        cfg["tssi_cfg"] = {"enable": True, "nonlinear": False, "gravity": gravity}  # 核心旋钮：弹性框架 + 重力级别
        cases.append({"name": name, "gravity": gravity, "config": cfg})
    return cases


CASES = build_cases()  # 全部对照工况


# ============================================================
#  通用助手(与 Autorun_TSSI_step3_test_v1 同口径，复制保持自包含)
# ============================================================
def build_source_files(static_source_paths, script_sequence):  # 源文件名→源路径映射 + 缺失/重名检测
    source_files, missing, dup = {}, [], []
    for source_path in list(static_source_paths) + list(script_sequence):
        target_name = os.path.basename(source_path)
        if not os.path.isfile(source_path):
            missing.append((target_name, source_path)); continue
        if target_name in source_files and source_files[target_name] != source_path:
            dup.append((target_name, source_path, source_files[target_name])); continue
        source_files[target_name] = source_path
    return source_files, missing, dup


def create_and_fill_folder(folder_path, source_files, config):  # 建目录、拷源、写 case_config.json
    os.makedirs(folder_path, exist_ok=True)
    for target_name, src_path in source_files.items():
        shutil.copy2(src_path, os.path.join(folder_path, target_name))
    with open(os.path.join(folder_path, CONFIG_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(config or {}, f, ensure_ascii=False, indent=2)


def _find_abaqus_cmd():  # 定位 abaqus 启动器(建模脚本 import abaqus/odbAccess，须经 abaqus 内核跑)
    for c in (os.environ.get("ABAQUS_CMD"), r"C:\SIMULIA\Commands\abaqus.bat", "abaqus", "abq2021"):
        if c and (os.path.isfile(c) or shutil.which(c)):
            return c
    return r"C:\SIMULIA\Commands\abaqus.bat"  # 兜底


ABAQUS_CMD = _find_abaqus_cmd()  # abaqus 命令


def run_scripts_in_folder(folder_path, run_order):  # 目录内按顺序执行脚本
    # 注意：本 Autorun 在【系统 Python3】下跑(需 subprocess.run)，但建模/读odb脚本 import 的是
    # abaqus/odbAccess，必须经【abaqus 内核】执行，故统一用 `abaqus cae noGUI=脚本`(cae 内核也能 openOdb)，
    # 【不能】用 sys.executable(那是 py3.13,import abaqus 直接失败)。
    for script_name in run_order:
        script_path = os.path.join(folder_path, script_name)
        if not os.path.isfile(script_path):
            print("错误：脚本不存在 -> {}".format(script_path)); return False
        print("开始执行(abaqus cae noGUI)：{}".format(script_path))
        result = subprocess.run([ABAQUS_CMD, "cae", "noGUI=" + script_name], cwd=folder_path, check=False)
        if result.returncode != 0:
            print("错误：{} 执行失败，返回码={}".format(script_name, result.returncode)); return False
        print("完成执行：{}".format(script_name))
    return True


def read_axial_check(folder_path):  # 读该工况 gravity_axial_check.csv 首行；缺则 None
    import csv
    fs = glob.glob(os.path.join(folder_path, 'gravity_axial_check.csv'))
    if not fs:
        return None
    with open(fs[0], 'r') as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def main():  # 主流程：建/跑 grav-off + grav-on → 汇总柱底轴力校验
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_OUT
    print("目标根目录：{}".format(root_dir))
    print("实验：P0#1 重力两步法冒烟 | grav-off(v1基线,无静力步) vs grav-on(Level A 仅结构自重)")
    source_files, missing, dup = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    if missing:
        print("错误：源文件缺失："); [print("  -", n, "->", p) for n, p in missing]; sys.exit(1)
    if dup:
        print("错误：源文件重名冲突："); [print("  -", t) for t, *_ in dup]; sys.exit(1)

    plan = []  # 规划文件夹
    seen = set()
    for c in CASES:
        folder = FOLDER_PREFIX + c["name"]
        if folder in seen:
            print("错误：重复文件夹名 ->", folder); sys.exit(1)
        seen.add(folder)
        plan.append((folder, c))
    os.makedirs(root_dir, exist_ok=True)

    def process(item):
        folder, case = item
        folder_path = os.path.join(root_dir, folder)
        print("\n==============================")
        if glob.glob(os.path.join(folder_path, 'gravity_axial_check.csv')):  # 断点续跑
            print("跳过(已有轴力校验结果)：{}".format(folder))
            return folder_path, True
        print("开始处理：{} (gravity={})".format(folder, case["gravity"]))
        create_and_fill_folder(folder_path, source_files, case["config"])
        ok = run_scripts_in_folder(folder_path, run_order)
        if ok:
            delete_files_by_type(folder_path, DELETE_FILE_TYPES)
        return folder_path, ok

    print("开始并行批处理，最大并发：{}".format(MAX_WORKERS))
    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for folder_path, ok in ex.map(process, plan):
            if not ok:
                failed.append(folder_path)
    if failed:
        print("\n存在失败文件夹（{}个）：".format(len(failed)))
        [print("  -", p) for p in failed]

    # ===== 汇总：柱底轴力校验 =====
    print("\n==============================")
    print("实验汇总：Step-gravity 柱底轴力校验(§5 QA 锚点，偏差<2% 为 PASS)")
    print("-" * 78)
    print("%-10s %-9s %-16s %-16s %-9s %-8s" % ("case", "gravity", "手算kN/柱", "实测kN/柱", "相对误差", "判读"))
    print("-" * 78)
    for folder, case in plan:
        r = read_axial_check(os.path.join(root_dir, folder))
        if r is None:
            note = "无静力步(对照,正常)" if case["gravity"] == "off" else "<无校验结果,看.msg/.sta>"
            print("%-10s %-9s %s" % (case["name"], case["gravity"], note))
            continue
        exp_kn = float(r["expected_N_per_col"]) / 1.0e3
        act_kn = float(r["actual_mean_N_per_col"]) / 1.0e3
        print("%-10s %-9s %-16.1f %-16.1f %-9s %-8s"
              % (case["name"], case["gravity"], exp_kn, act_kn,
                 "%+.2f%%" % (100.0 * float(r["rel_err"])), r["verdict"]))
    print("-" * 78)
    print("判读：grav-on 若 PASS(<2%) → 重力两步法管道通、传力对；grav-off 应报'无静力步'(等于 v1)。")

    if failed:
        sys.exit(2)


def delete_files_by_type(folder_path, file_types):  # 永久删除指定类型中间文件
    if not file_types:
        return
    normalized = {t.lower() if t.startswith(".") else "." + t.lower() for t in file_types}
    for name in sorted(os.listdir(folder_path)):
        fp = os.path.join(folder_path, name)
        if os.path.isfile(fp) and os.path.splitext(name)[1].lower() in normalized:
            try:
                os.remove(fp)
            except OSError as exc:
                print("警告：删除失败 {} -> {}".format(fp, exc))


if __name__ == "__main__":
    main()
