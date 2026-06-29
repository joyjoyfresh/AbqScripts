# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""【边界吸收对照实验】Autorun —— 验证"论文图15 软层 7.6 是否被 SPECFEM 边界 S 波反射人为抬高"。

背景（见对话诊断）：收敛后的等效力 ABAQUS 模型软斜入射坡顶峰只到 ~1.8(论文≈7.6)，且网格/单元/时窗/阻尼/
归一化都排除。fd 自由场输入经单独验证正确(自由面 BC 机器精度、SV→P 转换正常)。唯一剩下的物理杠杆 =
人工边界对坡顶散射面波/陷波的吸收强度。论文 SPECFEM 用 Stacey ABC，作者 §2.3.1 自承"对 S 波吸收差、
部分反射"。假设：被反射回域内的陷波 S 波在软层来回混响堆积 → 抬高放大，可能正是 7.6 的来源。

本实验：用建模脚本 v3 新增的 `boundary_cfg.dashpot_scale` 旋钮，对【软斜入射 D07】工况扫吸收强度
  k=1.0(全吸收,现状) / 0.5 / 0.2 / 0.0(纯弹簧,全反射封闭域)，外加【硬斜入射】k=1.0/0.0 作对照。
判读：
  • 峰值随 k↓ 急升、逼近 ~7.6  → 坐实论文值是边界反射混响人为抬高，你的全吸收 ~1.8 才是物理正确解。
  • 峰值几乎不动                 → 边界吸收也不是主因，论文 7.6 来源待查，需直接跑 SPECFEM2D 对拍。
  • 软层升、硬层几乎不升         → 该机制是软层陷波特有（与论文"软层斜入射暴增、硬层不明显"一致）。

注意：弱吸收/反射工况靠域内混响【堆积】，需较长时窗发育，故默认 tail=8s(总 10s，单作业较慢)。
若只想先看趋势可把 TAIL_SECONDS 调小。仅依赖纯 Python+pandas 出汇总，不依赖 Collect/Plot。
"""

import os  # 路径与目录
import re  # 文件名解析
import json  # 写 case_config.json / 汇总
import glob  # 扫描结果文件
import shutil  # 复制源文件
import subprocess  # 执行子进程
import sys  # 解释器与参数
import concurrent.futures  # 多文件夹并行

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 各工况文件夹建在此（可由命令行参数覆盖）
FOLDER_PREFIX = "bdry-"  # 文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 跑完直接删除的中间文件
MAX_WORKERS = 2  # 并行文件夹数
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名

# ── 实验参数（集中在此便于调） ──
TAIL_SECONDS = 8.0           # 静默尾段(s)：弱吸收/反射工况靠混响堆积，需较长时窗发育。想先看趋势可调小
PAD_FACTOR = 16              # fd 自由场 FFT 补零倍数（须保证 Nout≥记录+尾段）
DASHPOT_SCALES = [1.0, 0.5, 0.2, 0.0]  # 软层扫的吸收强度（1=全吸收 … 0=全反射）
HARD_CONTROL_SCALES = [1.0, 0.0]       # 硬层只取两端作对照

STATIC_SOURCE_PATHS = [  # 随每个工况拷入的固定源（仅输入波）
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4Hz Ricker（图15 a0=2.0）
]

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模(含 boundary_cfg.dashpot_scale 旋钮)
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算
]

PLATEAU_X = (200.0, 900.0)  # 上平台远场区间（取 TAF_h 中位数作平台值）


def _layers2(surf_vr, surf_thick):  # 双层 layers：表层 + 覆盖层
    return [
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
    ]


def _fmt_scale(k):  # 0.2→'0p2'、1.0→'1p0'（文件夹名安全）
    return ('%g' % float(k)).replace('.', 'p')


def build_cases():  # 生成对照实验工况（每个显式命名，含 dashpot_scale）
    """软斜入射 D07 扫吸收强度 + 硬斜入射两端对照。每项 = {name, kind, dscale, config}。"""
    base_common = {  # 每个工况共有的修复注入（论文 Q + 长时窗 + 大补零）
        "time_cfg": {"tail_seconds": float(TAIL_SECONDS)},
        "freefield_cfg": {"pad_factor": int(PAD_FACTOR)},
        "damping_cfg": {"constant_xi": None},   # 用论文 Qs=0.05·Vs
    }
    cases = []
    specs = [("soft", 5.0, DASHPOT_SCALES), ("hard", 1.25, HARD_CONTROL_SCALES)]  # 软=Vs1/Vs2=0.5；硬=2.0
    for kind, surf_vr, scales in specs:
        for k in scales:
            cfg = dict(base_common)  # 浅拷贝共有项
            cfg["material_cfg"] = {"angle": 15, "layers": _layers2(surf_vr, 150.0)}  # 斜入射 15°、h1=150(0.75)
            cfg["geometry_cfg"] = {"i": 45.0}
            cfg["boundary_cfg"] = {"dashpot_scale": float(k)}  # 本实验核心旋钮
            cases.append({
                "name": "L2_vr%s_t150-a15-d%s" % (_fmt_scale(surf_vr), _fmt_scale(k)),  # 含吸收强度，文件夹不冲突
                "kind": kind, "dscale": float(k), "config": cfg,
            })
    return cases


CASES = build_cases()  # 全部对照工况


# ============================================================
#  通用助手（与 Autorun_multilayer_nonlinear_v1 同口径，复制保持自包含）
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


def main():  # 主流程：建/跑各工况 → 汇总"峰值 vs 吸收强度"
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR
    print("目标根目录：{}".format(root_dir))
    print("实验：边界吸收 dashpot_scale 扫描 | tail=%.1fs pad=%d" % (TAIL_SECONDS, PAD_FACTOR))
    print("软层扫 %s ；硬层对照 %s" % (DASHPOT_SCALES, HARD_CONTROL_SCALES))
    source_files, missing, dup = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    if missing:
        print("错误：源文件缺失："); [print("  -", n, "->", p) for n, p in missing]; sys.exit(1)
    if dup:
        print("错误：源文件重名冲突："); [print("  -", t) for t, *_ in dup]; sys.exit(1)

    # 规划文件夹（名字已含 dscale，互不冲突）
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
        # 断点续跑：已有 TAF 结果则跳过（重跑只补未完成/失败的工况，省去重复长作业）
        if glob.glob(os.path.join(folder_path, 'TAF-*.csv')):
            print("跳过(已有 TAF 结果)：{}".format(folder))
            return folder_path, True
        print("开始处理：{} (kind={}, dashpot_scale={})".format(folder, case["kind"], case["dscale"]))
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
        # 不直接退出——已成功的工况仍出汇总

    # ===== 汇总：峰值 TAF vs 吸收强度 =====
    print("\n==============================")
    print("实验汇总：坡顶峰值 TAF 随边界吸收 dashpot_scale 的变化")
    print("（论文软斜入射坡顶 ≈ 7.6；全吸收 k=1.0 现状 ≈ 1.8）")
    print("-" * 64)
    print("%-8s %-8s %-8s %-8s %-8s" % ("kind", "dscale", "peakH", "peakV", "平台H"))
    print("-" * 64)
    rows = []
    for folder, case in plan:
        peaks = read_taf_peaks(os.path.join(root_dir, folder))
        if peaks is None:
            print("%-8s %-8.2f %s" % (case["kind"], case["dscale"], "<无TAF结果>"))
            continue
        pH, pV, plat = peaks
        rows.append((case["kind"], case["dscale"], pH, pV, plat))
    for kind, dscale, pH, pV, plat in sorted(rows, key=lambda r: (r[0], -r[1])):  # 按 kind、dscale 降序
        print("%-8s %-8.2f %-8.2f %-8.2f %-8.2f" % (kind, dscale, pH, pV, plat))
    print("-" * 64)

    # 判读：软层峰值随 k↓ 的增幅
    soft = sorted([r for r in rows if r[0] == "soft"], key=lambda r: -r[1])
    if len(soft) >= 2:
        hi_k = soft[0]   # k 最大（全吸收）
        lo_k = soft[-1]  # k 最小（最反射）
        ratio = lo_k[2] / hi_k[2] if hi_k[2] else float('nan')
        print("软层：k=%.2f 峰=%.2f → k=%.2f 峰=%.2f（增幅 ×%.2f）" % (hi_k[1], hi_k[2], lo_k[1], lo_k[2], ratio))
        if ratio >= 2.5:
            print("判读：峰值随吸收减弱大幅上升 → 论文 7.6 很可能是边界反射混响抬高，你的全吸收解才物理正确。")
        elif ratio <= 1.3:
            print("判读：峰值几乎不随吸收变化 → 边界吸收不是主因，需直接跑 SPECFEM2D 对拍定论。")
        else:
            print("判读：峰值中等上升 → 边界吸收是部分因素，但单凭它补不满到 7.6。")

    # 写出汇总 CSV（留痕）
    try:
        import csv
        with open(os.path.join(root_dir, "boundary_absorption_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["kind", "dashpot_scale", "peakH", "peakV", "plateauH"])
            for r in sorted(rows, key=lambda r: (r[0], -r[1])):
                w.writerow(r)
        print("汇总已写出：boundary_absorption_summary.csv")
    except Exception as e:
        print("汇总 CSV 写出失败：", e)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
