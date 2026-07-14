# -*- coding: utf-8 -*-
"""跨工况结果收集器 v2（基于 case_meta.json 的统一元数据，单/双/多层通用）。

相对 v1 的关键改变：不再靠文件夹名/建模脚本正则【反猜】工况参数，而是直接读取每个工况
  文件夹内由建模脚本写出的 case_meta.json（建模脚本自包含写出，本脚本不依赖任何共享模块），
  按固定映射把全部参数展平为统一规范列（slope_i、incident_angle、vs_cover、a0_base、layers_json…）
  写入 results/index.csv。这样无论单层/双层/三层、无论扫了哪些工况，都得到口径一致、可直接分组
  分析的数据库；下游 Plot 按规范列出图，不再做任何文件夹名解析。

做法（零侵入，原始文件不动）：遍历根目录下各工况文件夹，把其中 TAF/PGA/TIMESERIES 的 CSV
  【复制】到统一 results/ 目录并重命名为全局唯一：
    TAF-<文件夹名>__<记录>.csv / PGA-<文件夹名>__<记录>-slope|flat.csv / TIMESERIES-...。
  同时读各文件夹 case_meta.json，连同 文件/来源/类型/记录/scene 一并写 results/index.csv。
  缺 case_meta.json 的文件夹仍会收集 CSV，但元数据列留空并告警（建议重跑以补元数据）。

运行（在存放各工况文件夹的目录下，例如 Batch/）：
  cd Batch && python ../Postprocess/Collect_TAF_results_v2.py
  或：python Postprocess/Collect_TAF_results_v2.py  <存放工况文件夹的根目录>  [输出目录]
"""

import os  # 导入系统接口模块
import re  # 导入正则模块
import sys  # 导入系统参数模块
import glob  # 导入文件匹配模块
import shutil  # 导入文件复制模块
import csv  # 导入 CSV 写入模块
import io  # 导入 io 模块（UTF-8 读取 case_meta.json，Py2/Py3 通用）
import json  # 导入 JSON 模块（解析 case_meta.json、序列化 layers 列）


def _read_meta(folder):  # 读取某工况文件夹的 case_meta.json
    """读取 folder/case_meta.json 并返回 dict；不存在或解析失败返回 None。"""
    path = os.path.join(folder, 'case_meta.json')  # 元数据路径
    if not os.path.isfile(path):  # 文件不存在
        return None  # 返回空
    try:  # 尝试读取解析
        with io.open(path, 'r', encoding='utf-8') as f:  # UTF-8 文本模式（Py2/Py3 通用）
            return json.load(f)  # 返回解析结果
    except Exception:  # 解析失败
        return None  # 返回空


INDEX_META_FIELDS = [  # index.csv 中由 case_meta.json 展平而来的列（固定顺序、固定旧列名，供下游脚本稳定读取）
    'model_type', 'model_script', 'incident_angle', 'mesh_size', 'slope_i',
    'total_L', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness', 'H', 'h', 'w_slope',
    'n_finite_layers', 'n_layers_total', 'vs_bedrock', 'vs_surface', 'vs_cover',
    'vr_over_vs2', 'vs1_over_vs2', 'slope_height', 'a0_base', 'layers_json',
]


def _flatten(meta):  # 把嵌套 case_meta 展平为 index.csv 的一行（固定旧列名，兼容现有所有下游读取方）
    """按固定映射把嵌套 meta（geometry/derived/layers）展平为统一规范列（slope_i/a0_base/vs_cover…）。

    列名与重构前一致，故现有 index.csv 与所有用 pd.read_csv 读它的脚本均不受影响；字段缺失优雅返回 None。
    """
    g = meta.get('geometry', {}) or {}  # 几何子字典
    d = meta.get('derived', {}) or {}  # 派生子字典
    return {  # 组装扁平规范列
        'model_type': meta.get('model_type'),  # 模型类型
        'model_script': meta.get('model_script'),  # 建模脚本
        'incident_angle': meta.get('incident_angle'),  # 入射角 θs
        'mesh_size': meta.get('mesh_size'),  # 网格尺寸
        'slope_i': g.get('i'),  # 坡角 i
        'total_L': g.get('total_L'),  # 模型总长
        'left_flat': g.get('left_flat'),  # 上平台长度
        'H_minus_h': g.get('H_minus_h'),  # 斜坡高度差
        'h_over_H': g.get('h_over_H'),  # 深度比
        'bedrock_thickness': g.get('bedrock_thickness'),  # 基岩厚度
        'H': g.get('H'),  # 总覆盖厚度
        'h': g.get('h'),  # 下部覆盖高度
        'w_slope': g.get('w_slope'),  # 坡面水平长度
        'n_finite_layers': d.get('n_finite_layers'),  # 有限层数
        'n_layers_total': d.get('n_layers_total'),  # 总层数
        'vs_bedrock': d.get('vs_bedrock'),  # 基岩 Vr
        'vs_surface': d.get('vs_surface'),  # 表层 Vs1
        'vs_cover': d.get('vs_cover'),  # 覆盖层 Vs2
        'vr_over_vs2': d.get('vr_over_vs2'),  # Vr/Vs2
        'vs1_over_vs2': d.get('vs1_over_vs2'),  # Vs1/Vs2
        'slope_height': d.get('slope_height'),  # a0 归一化高度
        'a0_base': d.get('a0_base'),  # a0 换算基数
        'layers_json': json.dumps(meta.get('layers', []), ensure_ascii=False),  # 完整层信息（紧凑 JSON 保真）
    }


# ==============================================================================
#  配置
# ==============================================================================
COLLECT_PREFIXES = ('TAF', 'PGA', 'TIMESERIES')  # 需收集的 CSV 文件类型前缀（含整段时程 TIMESERIES）
OUT_DIRNAME = 'results'  # 默认集中输出子目录名
SKIP_DIR_NAMES = {'results', '__pycache__', 'test', '.git'}  # 扫描时跳过的目录名
SEP = '__'  # 工况标签与记录名之间的分隔符（便于后续按 SEP 拆分）
BASE_FIELDS = ['collected_file', 'source_folder', 'type', 'record', 'scene']  # index.csv 基础列（文件级）


# ==============================================================================
#  文件名拆解（与 v1 一致）
# ==============================================================================
def split_csv_name(stem):  # 拆解 CSV 文件主干
    """把如 'TAF-ricker_4Hz' 或 'PGA-ricker_4Hz-slope' 拆为 (type, record, scene)。

    返回 (type, record, scene)；scene ∈ {'slope','flat',''}；非目标前缀返回 (None,..)。
    """
    m = re.match(r'(?i)^(TAF|PGA|TIMESERIES)[-_](.+)$', stem)  # 匹配类型前缀
    if not m:  # 非目标文件
        return None, None, None  # 返回空三元组
    ftype = m.group(1).upper()  # 类型（大写）
    rest = m.group(2)  # 去前缀后的剩余
    scene = ''  # 初始化 scene
    low = rest.lower()  # 小写副本
    if low.endswith('-slope'):  # 坡地
        scene = 'slope'; record = rest[:-6]  # 截去 -slope
    elif low.endswith('-flat'):  # 平地
        scene = 'flat'; record = rest[:-5]  # 截去 -flat
    else:  # 无 scene 后缀
        record = rest  # 记录名即剩余
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
    n_missing_meta = 0  # 缺元数据的文件夹计数
    for entry in sorted(os.listdir(root)):  # 遍历根目录条目
        folder = os.path.join(root, entry)  # 子文件夹完整路径
        if not os.path.isdir(folder) or entry in SKIP_DIR_NAMES or entry.startswith('.'):  # 跳过非目录/特殊目录
            continue  # 跳过
        # 收集该文件夹内目标 CSV（排除归一化中间文件）
        csvs = []  # 该文件夹待收集 CSV
        for prefix in COLLECT_PREFIXES:  # 遍历类型前缀
            csvs += glob.glob(os.path.join(folder, '%s-*.csv' % prefix))  # 匹配该类型
        csvs = [f for f in sorted(set(csvs)) if '-normalized' not in os.path.basename(f).lower()]  # 排除归一化临时表
        if not csvs:  # 该文件夹无目标 CSV
            continue  # 跳过
        meta = _read_meta(folder)  # 读取该工况统一元数据（直接解析 JSON，不依赖共享模块）
        if meta is None:  # 缺元数据
            n_missing_meta += 1  # 计数
            meta_flat = {k: None for k in INDEX_META_FIELDS}  # 元数据列留空
            print('--- %s  (警告: 缺 case_meta.json，元数据列留空，建议重跑补元数据) ---' % entry)  # 告警
        else:  # 有元数据
            meta_flat = _flatten(meta)  # 展平为固定规范列
            print('--- %s  (type=%s, i=%s, θs=%s, n_layers=%s, Vr/Vs2=%s) ---' % (  # 打印工况摘要
                entry, meta_flat.get('model_type'), meta_flat.get('slope_i'),
                meta_flat.get('incident_angle'), meta_flat.get('n_layers_total'),
                meta_flat.get('vr_over_vs2')))
        n_folders += 1  # 文件夹计数
        for src in csvs:  # 遍历该文件夹 CSV
            stem = os.path.splitext(os.path.basename(src))[0]  # 文件主干
            ftype, record, scene = split_csv_name(stem)  # 拆解
            if ftype is None:  # 非目标文件
                continue  # 跳过
            scene_suffix = ('-' + scene) if scene else ''  # scene 后缀
            new_name = '%s-%s%s%s%s.csv' % (ftype, entry, SEP, record, scene_suffix)  # 新文件名：TYPE-<文件夹>__<记录>[-scene]
            dst = os.path.join(out_dir, new_name)  # 目标路径
            shutil.copy2(src, dst)  # 复制（保留时间戳）
            n_files += 1  # 文件计数
            row = {'collected_file': new_name, 'source_folder': entry, 'type': ftype,  # 文件级基础列
                   'record': record, 'scene': scene}  # 记录名与 scene
            row.update(meta_flat)  # 并入展平后的统一元数据列
            manifest.append(row)  # 记录清单
            print('    %s  ->  %s' % (os.path.basename(src), new_name))  # 打印复制

    if not manifest:  # 无收集
        print('未发现任何含 %s-*.csv 的工况文件夹。请在存放工况文件夹的目录下运行，或传入该目录。'
              % '/'.join(COLLECT_PREFIXES))  # 提示
        return  # 退出

    # 写出清单 index.csv（基础列 + 固定规范元数据列）
    index_path = os.path.join(out_dir, 'index.csv')  # 清单路径
    fields = BASE_FIELDS + list(INDEX_META_FIELDS)  # 完整列顺序（固定，兼容现有下游读取方）
    with open(index_path, 'w', newline='', encoding='utf-8-sig') as f:  # 写 CSV
        w = csv.DictWriter(f, fieldnames=fields)  # 字典写入器
        w.writeheader()  # 写表头
        for row in manifest:  # 写每行
            w.writerow({k: row.get(k) for k in fields})  # 仅写规范列
    print('\n>>> 完成：从 %d 个工况文件夹收集 %d 个 CSV 到 %s' % (n_folders, n_files, out_dir))  # 汇总
    if n_missing_meta:  # 有缺元数据的文件夹
        print('>>> 注意：%d 个文件夹缺 case_meta.json（元数据列留空）。请用已接入 case_meta 的建模脚本重跑以补全。' % n_missing_meta)  # 提示
    print('>>> 清单：%s' % index_path)  # 清单路径


if __name__ == '__main__':  # 主入口
    main()  # 运行
