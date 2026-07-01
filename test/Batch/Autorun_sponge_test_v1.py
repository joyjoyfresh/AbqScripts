# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""【阻尼海绵层冒烟+非扰动测试】Autorun —— 对软斜坡案扫 boundary_cfg.sponge_enable(off / on×2 档 xi_max)。

目的(诚实说明)：
  ① 冒烟测试 —— 确认 v3 新增的阻尼海绵层代码能在 Abaqus 跑通不崩(这是唯一没实测过的部分)。
     工况若出 TAF 结果 = 海绵建模+赋材+求解全通；若该工况失败(无 TAF)= 海绵代码有问题。
     另：跑完看建模 .log 里有无 "阻尼海绵层已施加：...覆盖 N 单元/M 组"，确认海绵确实生效。
  ② 非扰动确认 —— 坡顶峰(x≈坡肩)与远场平台(x∈[200,900])都【远离】L/R/B 边界内侧的海绵带
     (带宽自动≈10×网格<200m)，故海绵 on/off 对二者本就该几乎不变(边界对坡顶无影响已证)。
     off 与 on 坡顶峰一致(<5%) = 海绵不破坏坡面解。

注意：海绵的【降残余反射】效果在箱体测试(test/Multi/vab_absorption_test_v2.py)里才量得到，
      坡面 PGA TAF 看不到(坡顶/平台远离边界)。本表只验"能跑+不扰动"，不验"降反射"。

跑法：python <本文件> C:\\Abaqus\\sponge-test   （你的 python 会自动把建模脚本转交 abaqus）
输出：根目录 boundary_sponge_test_summary.csv + 屏幕表(坡顶峰/平台 vs 海绵档)。
"""

import os  # 路径与目录
import json  # 写 case_config.json / 汇总
import glob  # 扫描结果文件
import shutil  # 复制源文件
import subprocess  # 执行子进程
import sys  # 解释器与参数
import concurrent.futures  # 多文件夹并行

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 各工况文件夹建在此（可由命令行参数覆盖）
FOLDER_PREFIX = "sponge-"  # 文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 跑完直接删除的中间文件
MAX_WORKERS = 2  # 并行文件夹数
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名

# ── 实验参数（集中在此便于调） ──
TAIL_SECONDS = 4.0           # 静默尾段(s)：标准软案尾段
PAD_FACTOR = 8               # fd 自由场 FFT 补零倍数（须保证 Nout≥记录+尾段）
SPONGE_CASES = [             # (标签, sponge boundary_cfg)：off 基线 + 两档 xi_max
    ("off", {"sponge_enable": False}),
    ("on-x0p3", {"sponge_enable": True, "sponge_xi_max": 0.3}),
    ("on-x0p5", {"sponge_enable": True, "sponge_xi_max": 0.5}),
]

STATIC_SOURCE_PATHS = [  # 随每个工况拷入的固定源（仅输入波）
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4Hz Ricker（图15 a0=2.0）
]

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模(含 boundary_cfg.sponge_* 旋钮)
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算
]

PLATEAU_X = (200.0, 900.0)  # 上平台远场区间（x>200 避开左边界海绵带，取 TAF_h 中位数作平台值）


def _layers2(surf_vr, surf_thick):  # 双层 layers：表层 + 覆盖层
    return [
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
    ]


def build_cases():  # 生成海绵 off/on 工况（软斜入射 D07）
    """软斜入射 D07(Vs1/Vs2=0.5, h0.75, i45, 15°) 扫海绵开关。每项={name, label, config}。"""
    base_common = {  # 全扫共有注入，保证各档仅 sponge 配置不同
        "time_cfg": {"tail_seconds": float(TAIL_SECONDS)},
        "freefield_cfg": {"pad_factor": int(PAD_FACTOR)},
        "damping_cfg": {"constant_xi": None},   # 用论文 Qs=0.05·Vs
    }
    cases = []
    for label, bcfg in SPONGE_CASES:
        cfg = dict(base_common)  # 浅拷贝共有项
        cfg["material_cfg"] = {"angle": 15, "layers": _layers2(5.0, 150.0)}  # 软 Vs1/Vs2=0.5、h1=150(0.75)
        cfg["geometry_cfg"] = {"i": 45.0}
        cfg["boundary_cfg"] = dict(bcfg)  # 本实验核心：海绵开关/强度
        cases.append({"name": "soft-a15-%s" % label, "label": label, "config": cfg})
    return cases


CASES = build_cases()  # 全部工况


# ============================================================
#  通用助手（与 Autorun_boundary_spring_sweep_v1 同口径，复制保持自包含）
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


def run_scripts_in_folder(folder_path, run_order):  # 目录内按顺序执行脚本
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


def read_taf_peaks(folder_path):  # 读该工况 TAF-*.csv，返回 (peakH, peakV, plateauH)；缺则 None
    import pandas as pd
    fs = glob.glob(os.path.join(folder_path, 'TAF-*.csv'))
    if not fs:
        return None
    df = pd.read_csv(fs[0])
    up = df[(df['x'] >= PLATEAU_X[0]) & (df['x'] <= PLATEAU_X[1])]
    plateau = float(up['TAF_h'].median()) if len(up) else float('nan')
    return float(df['TAF_h'].max()), float(df['TAF_v'].max()), plateau


def main():  # 主流程：建/跑各工况 → 汇总"坡顶峰/平台 vs 海绵档" + 冒烟/非扰动判读
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR
    print("目标根目录：{}".format(root_dir))
    print("实验：阻尼海绵层 off/on 冒烟+非扰动测试 | tail=%.1fs pad=%d" % (TAIL_SECONDS, PAD_FACTOR))
    print("档位：%s" % [c[0] for c in SPONGE_CASES])
    source_files, missing, dup = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    if missing:
        print("错误：源文件缺失："); [print("  -", n, "->", p) for n, p in missing]; sys.exit(1)
    if dup:
        print("错误：源文件重名冲突："); [print("  -", t) for t, *_ in dup]; sys.exit(1)

    plan = []
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
        if glob.glob(os.path.join(folder_path, 'TAF-*.csv')):  # 断点续跑
            print("跳过(已有 TAF 结果)：{}".format(folder))
            return folder_path, True
        print("开始处理：{} (sponge={})".format(folder, case["label"]))
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
        print("（海绵 on 档失败 = 海绵代码有问题，需查建模 .log）")

    # ===== 汇总：坡顶峰/平台 vs 海绵档 =====
    print("\n==============================")
    print("实验汇总：坡顶峰/远场平台 TAF 随海绵 off/on 的变化（应几乎不变=非扰动）")
    print("-" * 56)
    print("%-12s %-9s %-9s %-9s" % ("海绵", "坡顶峰H", "坡顶峰V", "平台H"))
    print("-" * 56)
    rows = []
    for folder, case in plan:
        peaks = read_taf_peaks(os.path.join(root_dir, folder))
        if peaks is None:
            print("%-12s %s" % (case["label"], "<无TAF结果=该档失败>"))
            continue
        pH, pV, plat = peaks
        rows.append((case["label"], pH, pV, plat))
    for label, pH, pV, plat in rows:
        print("%-12s %-9.3f %-9.3f %-9.3f" % (label, pH, pV, plat))
    print("-" * 56)

    # 判读：① 冒烟(on 档有没有出结果) ② 非扰动(坡顶峰 off vs on)
    labels_ok = [r[0] for r in rows]
    on_ok = [l for l in labels_ok if l.startswith("on")]
    if "off" in labels_ok and on_ok:
        print("① 冒烟：海绵 on 档 %s 均跑出结果 → 海绵代码可运行。（再核建模 .log 有 '阻尼海绵层已施加'）" % on_ok)
        base = next(r[1] for r in rows if r[0] == "off")
        for label, pH, _pV, _plat in rows:
            if label.startswith("on") and base:
                d = abs(pH - base) / base * 100
                tag = "✓非扰动" if d <= 5 else "⚠坡顶峰变动>5%(查海绵带是否过宽侵入坡顶)"
                print("② %s 坡顶峰 %.3f vs off %.3f（差 %.1f%%）%s" % (label, pH, base, d, tag))
    elif on_ok == []:
        print("① 冒烟失败：海绵 on 档无结果 → 海绵代码在 Abaqus 端报错，查对应文件夹建模 .log。")

    # 写出汇总 CSV
    try:
        import csv
        with open(os.path.join(root_dir, "boundary_sponge_test_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["sponge", "peakH", "peakV", "plateauH"])
            for r in rows:
                w.writerow(r)
        print("汇总已写出：boundary_sponge_test_summary.csv")
    except Exception as e:
        print("汇总 CSV 写出失败：", e)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
