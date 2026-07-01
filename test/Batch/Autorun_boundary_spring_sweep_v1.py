# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""【边界弹簧系数敏感性】Autorun —— 扫 boundary_cfg.spring_scale，把刘晶波黏弹性边界的弹簧刚度
从现行(α_n=0.5/α_t=0.25)拨到标准 Liu(2.0×→α_n=1.0/α_t=0.5)及更高，做便宜的部分修正 + 稳健性。

背景：本方法 VAB 弹簧系数硬编码 kn=G/2R、kt=G/4R（α_n=0.5/α_t=0.25），恰是刘晶波 2D 常用值
(α_n=1.0/α_t=0.5)的一半，落在文献区间低端。弹簧是弹性恢复项，主管低频/静位移精度，不动高频波吸收(那是 ρc)。
本实验对【软斜入射 D07】工况扫 spring_scale = 0.5/1.0(现行)/2.0(标准Liu)/4.0，看：
  • 坡顶峰 TAF：应基本不变 → 收下第 9 个稳健性维度(峰对弹簧系数不敏感)，并堵"α 非标准"的质疑。
  • 远场平台 TAF：spring 影响低频边界精度，平台若随 spring_scale 漂移即是信号。
判读：
  • 坡顶峰极差 <5%  → 弹簧系数非关键，标准 Liu 与现行一致，稳健性成立。
  • 坡顶峰明显变   → α 有影响，正文宜用标准 Liu(2.0) 并报此敏感性。
注：残余反射(近边界 12%)随 spring_scale 怎么动，需另跑 vab_absorption_test（本表只看坡面 TAF）。
spring_scale=1.0 这一档应复现既有软斜案(~2.0)，作参照。仅纯 Python+pandas 出汇总。
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
FOLDER_PREFIX = "spring-"  # 文件夹前缀
DELETE_FILE_TYPES = [".odb", ".jnl", ".inp", ".msg", ".prt", ".dat", ".sta", ".sim", ".com"]  # 跑完直接删除的中间文件
MAX_WORKERS = 2  # 并行文件夹数
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名

# ── 实验参数（集中在此便于调） ──
TAIL_SECONDS = 4.0           # 静默尾段(s)：弹簧敏感性不靠混响堆积，标准软案尾段即可
PAD_FACTOR = 8               # fd 自由场 FFT 补零倍数（须保证 Nout≥记录+尾段）
SPRING_SCALES = [0.5, 1.0, 2.0, 4.0]  # 弹簧系数缩放：1.0=现行(α_n0.5/α_t0.25)、2.0=标准Liu(1.0/0.5)
CURRENT_ALPHA_N = 0.5        # 现行法向弹簧系数 α_n（spring_scale=1.0 时）；α_t=α_n/2，仅用于汇总标注

STATIC_SOURCE_PATHS = [  # 随每个工况拷入的固定源（仅输入波）
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4Hz Ricker（图15 a0=2.0）
]

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Multi\VAB_oblique_multilayer_nonlinear_v3.py",  # 建模(含 boundary_cfg.spring_scale 旋钮)
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Postprocess_PGA_v2.py",  # PGA 提取
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\General\Compute_TAF_v2.py",  # TAF 计算
]

PLATEAU_X = (200.0, 900.0)  # 上平台远场区间（取 TAF_h 中位数作平台值）


def _layers2(surf_vr, surf_thick):  # 双层 layers：表层 + 覆盖层
    return [
        {'name': 'surface', 'velocity_ratio': surf_vr, 'poisson_ratio': 0.3, 'density': 2500, 'thickness': surf_thick},
        {'name': 'overlying', 'velocity_ratio': 2.5, 'poisson_ratio': 0.3, 'density': 2500},
    ]


def _fmt_scale(k):  # 0.5→'0p5'、1.0→'1p0'（文件夹名安全）
    return ('%g' % float(k)).replace('.', 'p')


def build_cases():  # 生成弹簧系数敏感性工况（软斜入射 D07 扫 spring_scale）
    """软斜入射 D07(Vs1/Vs2=0.5, h1/(H-h)=0.75, i45, 15°) 扫弹簧系数。每项={name, sscale, config}。"""
    base_common = {  # 全扫共有注入（论文 Q + 尾段 + 补零），保证四档仅 spring_scale 不同
        "time_cfg": {"tail_seconds": float(TAIL_SECONDS)},
        "freefield_cfg": {"pad_factor": int(PAD_FACTOR)},
        "damping_cfg": {"constant_xi": None},   # 用论文 Qs=0.05·Vs
    }
    cases = []
    for k in SPRING_SCALES:
        cfg = dict(base_common)  # 浅拷贝共有项
        cfg["material_cfg"] = {"angle": 15, "layers": _layers2(5.0, 150.0)}  # 软 Vs1/Vs2=0.5、h1=150(0.75)
        cfg["geometry_cfg"] = {"i": 45.0}
        cfg["boundary_cfg"] = {"spring_scale": float(k)}  # 本实验核心旋钮
        cases.append({
            "name": "soft-a15-s%s" % _fmt_scale(k),  # 含 spring_scale，文件夹不冲突
            "sscale": float(k), "config": cfg,
        })
    return cases


CASES = build_cases()  # 全部敏感性工况


# ============================================================
#  通用助手（与 Autorun_boundary_absorption_test_v1 同口径，复制保持自包含）
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


def main():  # 主流程：建/跑各工况 → 汇总"坡顶峰/远场平台 vs spring_scale"
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR
    print("目标根目录：{}".format(root_dir))
    print("实验：边界弹簧系数 spring_scale 扫描 | tail=%.1fs pad=%d" % (TAIL_SECONDS, PAD_FACTOR))
    print("扫 %s （1.0=现行 α_n%.2f/α_t%.2f；2.0=标准Liu α_n%.2f/α_t%.2f）"
          % (SPRING_SCALES, CURRENT_ALPHA_N, CURRENT_ALPHA_N / 2, 2 * CURRENT_ALPHA_N, CURRENT_ALPHA_N))
    source_files, missing, dup = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    if missing:
        print("错误：源文件缺失："); [print("  -", n, "->", p) for n, p in missing]; sys.exit(1)
    if dup:
        print("错误：源文件重名冲突："); [print("  -", t) for t, *_ in dup]; sys.exit(1)

    # 规划文件夹（名字已含 spring_scale，互不冲突）
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
        # 断点续跑：已有 TAF 结果则跳过
        if glob.glob(os.path.join(folder_path, 'TAF-*.csv')):
            print("跳过(已有 TAF 结果)：{}".format(folder))
            return folder_path, True
        print("开始处理：{} (spring_scale={})".format(folder, case["sscale"]))
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

    # ===== 汇总：坡顶峰/远场平台 vs spring_scale =====
    print("\n==============================")
    print("实验汇总：坡顶峰/远场平台 TAF 随边界弹簧系数 spring_scale 的变化")
    print("（spring_scale=1.0 应复现既有软斜案 ~2.0 作参照）")
    print("-" * 64)
    print("%-10s %-8s %-8s %-8s %-8s" % ("spring", "α_n", "peakH", "peakV", "平台H"))
    print("-" * 64)
    rows = []
    for folder, case in plan:
        peaks = read_taf_peaks(os.path.join(root_dir, folder))
        if peaks is None:
            print("%-10.2f %-8s" % (case["sscale"], "<无TAF结果>"))
            continue
        pH, pV, plat = peaks
        rows.append((case["sscale"], CURRENT_ALPHA_N * case["sscale"], pH, pV, plat))
    for sscale, alpha_n, pH, pV, plat in sorted(rows, key=lambda r: r[0]):  # 按 spring_scale 升序
        print("%-10.2f %-8.2f %-8.2f %-8.2f %-8.2f" % (sscale, alpha_n, pH, pV, plat))
    print("-" * 64)

    # 判读：坡顶峰随 spring_scale 的变幅（稳健性）
    if len(rows) >= 2:
        peaks = [r[2] for r in rows]
        pmin, pmax, pmean = min(peaks), max(peaks), sum(peaks) / len(peaks)
        spread = (pmax - pmin) / pmean * 100 if pmean else float('nan')
        print("坡顶峰 spring_scale %.1f→%.1f 变幅：%.2f~%.2f（极差 %.1f%% of 均值）"
              % (SPRING_SCALES[0], SPRING_SCALES[-1], pmin, pmax, spread))
        if spread <= 5.0:
            print("判读：坡顶峰对弹簧系数不敏感(<5%) → 第9维稳健性成立；标准Liu(2.0)与现行一致，α取值非关键。")
        elif spread <= 15.0:
            print("判读：坡顶峰随弹簧系数中等变化(5~15%) → α有影响，正文宜用标准Liu(2.0)并报此敏感性。")
        else:
            print("判读：坡顶峰对弹簧系数敏感(>15%) → α选择关键，需审慎定标并讨论。")
        plats = [r[4] for r in rows]
        print("远场平台 TAF_h：%.3f~%.3f（应≈一维理论；spring 主要影响低频边界精度，漂移即信号）"
              % (min(plats), max(plats)))

    # 写出汇总 CSV（留痕）
    try:
        import csv
        with open(os.path.join(root_dir, "boundary_spring_sweep_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(["spring_scale", "alpha_n", "peakH", "peakV", "plateauH"])
            for r in sorted(rows, key=lambda r: r[0]):
                w.writerow(r)
        print("汇总已写出：boundary_spring_sweep_summary.csv")
    except Exception as e:
        print("汇总 CSV 写出失败：", e)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
