# -*- coding: utf-8 -*-  # 声明源码编码为 UTF-8
"""【TSSI step3 冒烟测试】Autorun —— 梁柱 CDP混凝土+钢筋非线性首次实跑，elastic/cdp 两工况对照。

背景：`Modeling/Hybrid/slope_frame_ssi_full_v1.py` 的 TSSI 框架（Tie耦合+CDP混凝土+钢筋纤维截面）
从未在 Abaqus 里真正跑过（此前只过了 py_compile + 结构检查）。本脚本跑【缩小规模】的 2 个对照工况：
  - elastic：tssi_cfg.nonlinear=False，退回 step2 纯弹性框架（已知机制应该能跑通，作对照基线）
  - cdp    ：tssi_cfg.nonlinear=True， step3 CDP混凝土+钢筋+AUTOMATIC增量（本次要验证的新代码路径）
判读：
  • elastic 都跑不过 → 是 Tie/框架本身的问题（与 CDP 无关）。
  • 只有 cdp 跑不过 → 是 CDP 材料/钢筋注入/收敛的问题，看该工况 .msg/.sta 里的不收敛记录。
  • 两者都过 → 比较 T_ssi/基底剪力/层间位移角/建筑放大：cdp 应因材料软化而周期更长、放大略降(定性方向)。

缩小规模沿用 step2b 已验证过的小几何（`frame_ssi_slope_v1.py` 里的参数）：
  H_minus_h=30/total_L=300/left_flat=120/bedrock_thickness=40，mesh=5m 非自适应；
  框架缩到 2 层 1 跨（默认 5 层 3 跨太大，先用最小规模冒烟）。仅一条 4Hz Ricker 波。

运行方式（工作目录用命令行传入，不传则用脚本所在目录；求解产物较大建议指到仓库外 C:/Abaqus）：
  abaqus cae noGUI=Batch/Autorun_TSSI_step3_test_v1.py -- C:\\Abaqus\\TSSI_step3_test
  或：python Batch/Autorun_TSSI_step3_test_v1.py  C:\\Abaqus\\TSSI_step3_test
"""

import os  # 路径与目录
import json  # 写 case_config.json / 汇总
import glob  # 扫描结果文件
import shutil  # 复制源文件
import subprocess  # 执行子进程
import sys  # 解释器与参数
import concurrent.futures  # 多文件夹并行

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 各工况文件夹建在此（可由命令行参数覆盖）
FOLDER_PREFIX = "tssi-"  # 文件夹前缀
DELETE_FILE_TYPES = []  # 冒烟测试：不清理中间产物(.msg/.sta/.dat 正是要看的收敛诊断，跟其余 Autorun 不同)
MAX_WORKERS = 2  # 并行文件夹数（只有2个工况，各跑各的互不影响）
CONFIG_FILENAME = "case_config.json"  # 注入配置文件名

STATIC_SOURCE_PATHS = [  # 随每个工况拷入的固定源（仅输入波）
    r"C:\Users\12462\Documents\Code\AbqScripts\Wave\Impulse\Acceleration\ricker_wavelet_4Hz.txt",  # 4Hz Ricker，够触发框架响应又不至于太长
]

SCRIPT_SEQUENCE = [  # 每个工况文件夹内按顺序执行
    r"C:\Users\12462\Documents\Code\AbqScripts\Modeling\Hybrid\slope_frame_ssi_full_v1.py",  # TSSI 建模脚本（读取 case_config.json）
    r"C:\Users\12462\Documents\Code\AbqScripts\Postprocess\Hybrid\Postprocess_SSI_response_v1.py",  # SSI 建筑响应提取(T_ssi/基底剪力/层间位移角/放大)
]

# 缩小几何+网格（沿用 step2b 已验证过的小规模），加速冒烟测试
SMALL_GEOMETRY = {
    "H_minus_h": 30.0, "i": 45.0, "h_over_H": 0.5,
    "total_L": 300.0, "left_flat": 120.0, "bedrock_thickness": 40.0,
}
SMALL_MESH = {"size": 5.0, "auto": False, "graded": False, "elem": "CPE4R"}
SMALL_FRAME = {"n_story": 2, "n_bay": 1}  # 默认 5 层 3 跨太大，先用最小规模冒烟


def build_cases():  # 生成 elastic/cdp 两个对照工况（同几何同波，仅 tssi_cfg.nonlinear 不同）
    """每项={name, nonlinear, config}。"""
    base_common = {"geometry_cfg": SMALL_GEOMETRY, "mesh_cfg": SMALL_MESH, "frame_cfg": SMALL_FRAME}
    cases = []
    for name, nonlinear in (("elastic", False), ("cdp", True)):
        cfg = dict(base_common)  # 浅拷贝共有项
        cfg["tssi_cfg"] = {"enable": True, "nonlinear": nonlinear}  # 本实验核心旋钮
        cases.append({"name": name, "nonlinear": nonlinear, "config": cfg})
    return cases


CASES = build_cases()  # 全部对照工况


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


def read_ssi_summary(folder_path):  # 读该工况 SSI-response-summary.csv 首行指标；缺则 None
    import csv
    fs = glob.glob(os.path.join(folder_path, 'SSI-response-summary.csv'))
    if not fs:
        return None
    with open(fs[0], 'r') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    r = rows[0]
    return {k: (float(v) if v not in ('', 'None') else None) for k, v in r.items() if k != 'record'}


def main():  # 主流程：建/跑 elastic+cdp 两工况 → 对照汇总(T_ssi/基底剪力/层间位移角/建筑放大)
    root_dir = sys.argv[1] if len(sys.argv) >= 2 else ROOT_DIR
    print("目标根目录：{}".format(root_dir))
    print("实验：TSSI step3 冒烟测试 | elastic(对照基线) vs cdp(CDP混凝土+钢筋新路径)")
    source_files, missing, dup = build_source_files(STATIC_SOURCE_PATHS, SCRIPT_SEQUENCE)
    run_order = [os.path.basename(p) for p in SCRIPT_SEQUENCE]
    if missing:
        print("错误：源文件缺失："); [print("  -", n, "->", p) for n, p in missing]; sys.exit(1)
    if dup:
        print("错误：源文件重名冲突："); [print("  -", t) for t, *_ in dup]; sys.exit(1)

    # 规划文件夹
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
        # 断点续跑：已有 SSI 结果则跳过
        if glob.glob(os.path.join(folder_path, 'SSI-response-summary.csv')):
            print("跳过(已有 SSI 结果)：{}".format(folder))
            return folder_path, True
        print("开始处理：{} (nonlinear={})".format(folder, case["nonlinear"]))
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

    # ===== 汇总：elastic vs cdp 对照 =====
    print("\n==============================")
    print("实验汇总：T_ssi/基底剪力/最大层间位移角/建筑放大 —— elastic(对照) vs cdp(step3)")
    print("-" * 78)
    print("%-10s %-8s %-8s %-14s %-10s %-10s" % ("case", "T_ssi", "周期比", "基底剪力(N)", "最大层间角", "建筑放大"))
    print("-" * 78)
    rows = []
    for folder, case in plan:
        s = read_ssi_summary(os.path.join(root_dir, folder))
        if s is None:
            print("%-10s %-8s" % (case["name"], "<无SSI结果，看该文件夹.msg/.sta定位不收敛>"))
            continue
        rows.append((case["name"], s))
        print("%-10s %-8.3f %-8.2f %-14.3e %-10.4f %-10.2f"
              % (case["name"], s.get('T_ssi') or 0, s.get('period_ratio') or 0,
                 s.get('base_shear') or 0, s.get('max_drift') or 0, s.get('building_amp') or 0))
    print("-" * 78)

    if len(rows) == 2:
        by_name = dict(rows)
        if 'elastic' in by_name and 'cdp' in by_name:
            e, c = by_name['elastic'], by_name['cdp']
            print("判读：cdp 相对 elastic —— T_ssi 变化 %+.1f%%，基底剪力变化 %+.1f%%，建筑放大变化 %+.1f%%"
                  % (100 * (c.get('T_ssi', 0) - e.get('T_ssi', 0)) / (e.get('T_ssi') or 1),
                     100 * (c.get('base_shear', 0) - e.get('base_shear', 0)) / (e.get('base_shear') or 1),
                     100 * (c.get('building_amp', 0) - e.get('building_amp', 0)) / (e.get('building_amp') or 1)))
            print("（材料软化预期：CDP 因混凝土/钢筋非线性刚度下降，T_ssi 应略增；具体量级只做定性核对，不当正式结果）")
    elif len(rows) < 2:
        print("判读：不足两个工况的结果，先看失败工况的日志/.msg/.sta，无法做 elastic/cdp 对照。")

    # 写出汇总 CSV（留痕）
    try:
        import csv
        with open(os.path.join(root_dir, "tssi_step3_smoketest_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["case", "T_ssi", "T_fixed", "period_ratio", "base_shear", "max_drift", "crest_pga", "roof_pga", "building_amp"])
            for name, s in rows:
                w.writerow([name, s.get('T_ssi'), s.get('T_fixed'), s.get('period_ratio'), s.get('base_shear'),
                            s.get('max_drift'), s.get('crest_pga'), s.get('roof_pga'), s.get('building_amp')])
        print("汇总已写出：tssi_step3_smoketest_summary.csv")
    except Exception as e:
        print("汇总 CSV 写出失败：", e)

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
