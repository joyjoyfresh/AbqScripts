# -*- coding: utf-8 -*-
"""跨工况结果收集器（仅服务 multilayer 系列；不改动建模/后处理管线）。

背景：各工况在自己的文件夹里产出 TAF-<记录>.csv / PGA-<记录>-slope|flat.csv，
  文件名只含波形记录名，工况信息（软/硬、厚度、角度…）只在文件夹名里。
  跨工况汇总时这些同名 CSV 会互相覆盖、丢失身份。

本脚本做法（零侵入）：遍历根目录下各工况文件夹，把其中的 TAF/PGA/TIMESERIES CSV
  【复制】到统一的 results/ 目录，并按"文件夹名"重命名为自描述、全局唯一的形式：
    TAF-<文件夹名>__<记录>.csv
    PGA-<文件夹名>__<记录>-slope.csv / -flat.csv
    TIMESERIES-<文件夹名>__<记录>-slope.csv / -flat.csv（整段时程，文件较大）
  同时写出 results/index.csv 清单（文件、来源文件夹、类型、记录、scene，
  以及尽量解析到的工况属性 Vs1/Vs2、h1/(H−h)、入射角），方便直接载入分组分析。
  原始文件保持原样不动，results/ 只是可自描述的集中副本。

运行（在存放各工况文件夹的目录下，例如 Batch/）：
  cd Batch && python ../Postprocess/Collect_TAF_results_v1.py
  或：python Postprocess/Collect_TAF_results_v1.py  <存放工况文件夹的根目录>  [输出目录]
"""

import os  # 导入系统接口模块
import re  # 导入正则模块
import sys  # 导入系统参数模块
import glob  # 导入文件匹配模块
import shutil  # 导入文件复制模块
import csv  # 导入 CSV 写入模块

# ==============================================================================
#  配置
# ==============================================================================
COLLECT_PREFIXES = ('TAF', 'PGA', 'TIMESERIES')  # 需收集的 CSV 文件类型前缀（含整段时程 TIMESERIES）
# 说明：TIMESERIES-*.csv 是全表面节点的完整加速度时程，文件较大；若不需要可从上面元组删除 'TIMESERIES'。
OUT_DIRNAME = 'results'  # 默认集中输出子目录名
SKIP_DIR_NAMES = {'results', '__pycache__', 'test', '.git'}  # 扫描时跳过的目录名
MODEL_SCRIPT_GLOB = 'VAB_oblique_TAF_multilayer_v*.py'  # 工况文件夹内建模脚本（用于精确解析参数）
SEP = '__'  # 工况标签与记录名之间的分隔符（便于后续按 SEP 拆分）


# ==============================================================================
#  工况属性解析（脚本优先，文件夹名回退）——与 Plot_Fig15_compare 保持一致口径
# ==============================================================================
def parse_case_from_script(folder):  # 从文件夹内建模脚本精确解析工况属性
    """返回 dict(vs1_vs2, h1_over, angle) 或 None。"""
    scripts = glob.glob(os.path.join(folder, MODEL_SCRIPT_GLOB))  # 查找建模脚本副本
    if not scripts:  # 无脚本
        return None
    try:
        text = open(scripts[0], 'r', encoding='utf-8', errors='ignore').read()  # 读取脚本
    except Exception:
        return None
    vrs = re.findall(r"'velocity_ratio'\s*:\s*([0-9.]+)", text)  # 所有 velocity_ratio（首=表层、次=覆盖层）
    ths = re.findall(r"'thickness'\s*:\s*([0-9.]+)", text)  # thickness（首=表层）
    ang = re.search(r"'angle'\s*:\s*([0-9.]+)", text)  # 入射角
    hmh = re.search(r"'H_minus_h'\s*:\s*([0-9.]+)", text)  # 斜坡高差
    if len(vrs) < 2 or not ths or not ang or not hmh:  # 非含表层多层配置
        return None
    surf_vr, over_vr = float(vrs[0]), float(vrs[1])  # 表层、覆盖层相对波速比
    vs1_vs2 = over_vr / surf_vr if surf_vr > 0 else float('nan')  # Vs1/Vs2 = over/surf
    h1_over = float(ths[0]) / float(hmh.group(1)) if float(hmh.group(1)) > 0 else float('nan')  # h1/(H−h)
    return {'vs1_vs2': vs1_vs2, 'h1_over': h1_over, 'angle': float(ang.group(1))}  # 返回属性


def parse_case_from_name(folder):  # 从文件夹名回退解析工况属性
    """按 soft/hard、t25/t75、a0/a15 命名约定识别，返回 dict（缺项为 None）。"""
    name = os.path.basename(folder).lower()  # 文件夹名（小写）
    vs1_vs2 = 0.5 if 'soft' in name else (2.0 if 'hard' in name else None)  # 软=0.5、硬=2.0
    m_t = re.search(r't(\d+)', name)  # 厚度标签
    m_a = re.search(r'a(-?\d+)', name)  # 角度标签
    h1_over = (int(m_t.group(1)) / 100.0) if m_t else None  # t25→0.25
    angle = float(m_a.group(1)) if m_a else None  # a15→15
    if vs1_vs2 is None and h1_over is None and angle is None:  # 全无则视为不可识别
        return None
    return {'vs1_vs2': vs1_vs2, 'h1_over': h1_over, 'angle': angle}  # 返回属性（可能含 None）


def parse_case(folder):  # 综合解析工况属性
    """脚本优先，回退文件夹名；都失败返回全 None 的占位 dict。"""
    return parse_case_from_script(folder) or parse_case_from_name(folder) \
        or {'vs1_vs2': None, 'h1_over': None, 'angle': None}  # 占位


# ==============================================================================
#  文件名拆解
# ==============================================================================
def split_csv_name(stem):  # 拆解 CSV 文件主干
    """把如 'TAF-ricker_4Hz' 或 'PGA-ricker_4Hz-slope' 拆为 (type, record, scene)。

    返回 (type, record, scene)；scene ∈ {'slope','flat',''}；非目标前缀返回 (None,..)。
    """
    m = re.match(r'(?i)^(TAF|PGA|TIMESERIES)[-_](.+)$', stem)  # 匹配类型前缀
    if not m:  # 非目标文件
        return None, None, None
    ftype = m.group(1).upper()  # 类型（大写）
    rest = m.group(2)  # 去前缀后的剩余
    scene = ''  # 初始化 scene
    low = rest.lower()  # 小写副本
    if low.endswith('-slope'):  # 坡地
        scene = 'slope'; record = rest[:-6]
    elif low.endswith('-flat'):  # 平地
        scene = 'flat'; record = rest[:-5]
    else:  # 无 scene 后缀
        record = rest
    return ftype, record, scene  # 返回拆解结果


# ==============================================================================
#  主流程
# ==============================================================================
def main():  # 主入口
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()  # 根目录：参数或当前目录
    root = os.path.abspath(root)  # 转绝对路径
    out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3 else os.path.join(root, OUT_DIRNAME)  # 输出目录
    os.makedirs(out_dir, exist_ok=True)  # 创建输出目录
    print('>>> 收集根目录: %s' % root)  # 提示根目录
    print('>>> 输出目录:   %s' % out_dir)  # 提示输出目录

    manifest = []  # 清单记录
    n_folders = 0  # 收录文件夹计数
    n_files = 0  # 复制文件计数
    for entry in sorted(os.listdir(root)):  # 遍历根目录条目
        folder = os.path.join(root, entry)  # 子文件夹完整路径
        if not os.path.isdir(folder) or entry in SKIP_DIR_NAMES or entry.startswith('.'):  # 跳过非目录/特殊目录
            continue
        # 收集该文件夹内目标 CSV（排除归一化中间文件）
        csvs = []  # 该文件夹待收集 CSV
        for prefix in COLLECT_PREFIXES:  # 遍历类型前缀
            csvs += glob.glob(os.path.join(folder, '%s-*.csv' % prefix))  # 匹配该类型
        csvs = [f for f in sorted(set(csvs)) if '-normalized' not in os.path.basename(f).lower()]  # 排除归一化临时表
        if not csvs:  # 该文件夹无目标 CSV
            continue
        attr = parse_case(folder)  # 解析工况属性
        n_folders += 1  # 文件夹计数
        print('--- %s  (Vs1/Vs2=%s, h1/(H-h)=%s, angle=%s) ---' %  # 打印工况
              (entry, attr['vs1_vs2'], attr['h1_over'], attr['angle']))
        for src in csvs:  # 遍历该文件夹 CSV
            stem = os.path.splitext(os.path.basename(src))[0]  # 文件主干
            ftype, record, scene = split_csv_name(stem)  # 拆解
            if ftype is None:  # 非目标文件
                continue
            scene_suffix = ('-' + scene) if scene else ''  # scene 后缀
            new_name = '%s-%s%s%s%s.csv' % (ftype, entry, SEP, record, scene_suffix)  # 新文件名：TYPE-<文件夹>__<记录>[-scene]
            dst = os.path.join(out_dir, new_name)  # 目标路径
            shutil.copy2(src, dst)  # 复制（保留时间戳）
            n_files += 1  # 文件计数
            manifest.append({  # 记录清单
                'collected_file': new_name, 'source_folder': entry, 'type': ftype,
                'record': record, 'scene': scene,
                'vs1_vs2': attr['vs1_vs2'], 'h1_over': attr['h1_over'], 'angle': attr['angle'],
            })
            print('    %s  ->  %s' % (os.path.basename(src), new_name))  # 打印复制

    if not manifest:  # 无收集
        print('未发现任何含 %s-*.csv 的工况文件夹。请在存放工况文件夹的目录下运行，或传入该目录。'
              % '/'.join(COLLECT_PREFIXES))
        return

    # 写出清单 index.csv
    index_path = os.path.join(out_dir, 'index.csv')  # 清单路径
    fields = ['collected_file', 'source_folder', 'type', 'record', 'scene', 'vs1_vs2', 'h1_over', 'angle']  # 列
    with open(index_path, 'w', newline='', encoding='utf-8-sig') as f:  # 写 CSV
        w = csv.DictWriter(f, fieldnames=fields)  # 字典写入器
        w.writeheader()  # 表头
        for row in manifest:  # 写每行
            w.writerow(row)
    print('\n>>> 完成：从 %d 个工况文件夹收集 %d 个 CSV 到 %s' % (n_folders, n_files, out_dir))  # 汇总
    print('>>> 清单：%s' % index_path)  # 清单路径


if __name__ == '__main__':  # 主入口
    main()  # 运行
