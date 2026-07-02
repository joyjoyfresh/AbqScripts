# -*- coding: utf-8 -*-
"""跨工况结果收集器 v1（Hybrid 专用版）。

本脚本负责遍历指定根目录下的各个工况文件夹，收集其中的地震反应与谱比 CSV 文件，
并提取 case_meta.json 元数据及 surface_summary.json 中的关键标量指标（AR_max、suspect），
集中整理到 results 文件夹中，并生成统一规范的 index.csv 数据库清单，供后续作图或分析。
"""

import os  # 导入系统接口模块
import re  # 导入正则模块
import sys  # 导入系统参数模块
import glob  # 导入文件匹配模块
import shutil  # 导入文件复制模块
import csv  # 导入 CSV 写入模块
import io  # 导入 io 模块
import json  # 导入 JSON 模块


# ==============================================================================
#  配置
# ==============================================================================
COLLECT_PREFIXES = ('TAF', 'PGA', 'TIMESERIES', 'surface_response', 'H_surface_h', 'H_surface_v', 'H_topo_h')  # 收集前缀 / 包含旧前缀与新后处理前缀
KNOWN_PREFIXES = [  # 已知前缀转换映射表 / 长度从长到短排列防截断错误
    ('surface_response_', 'SURFACE_RESPONSE'),  # 地表响应
    ('H_surface_h_', 'H_SURFACE_H'),  # 水平地表传函
    ('H_surface_v_', 'H_SURFACE_V'),  # 竖向地表传函
    ('H_topo_h_', 'H_TOPO_H'),  # 地形水平谱比传函
    ('TIMESERIES-', 'TIMESERIES'),  # 时程中划线前缀
    ('TIMESERIES_', 'TIMESERIES'),  # 时程下划线前缀
    ('TAF-', 'TAF'),  # TAF 中划线前缀
    ('TAF_', 'TAF'),  # TAF 下划线前缀
    ('PGA-', 'PGA'),  # PGA 中划线前缀
    ('PGA_', 'PGA'),  # PGA 下划线前缀
]
OUT_DIRNAME = 'results'  # 默认集中输出子目录名 / 结果输出目录
SKIP_DIR_NAMES = {'results', '__pycache__', 'test', '.git'}  # 扫描时跳过的目录名 / 排除目录
SEP = '__'  # 工况标签与记录名之间的分隔符 / 统一命名分隔符
BASE_FIELDS = ['collected_file', 'source_folder', 'type', 'record', 'scene']  # index.csv 基础列 / 文件基础字段
INDEX_META_FIELDS = [  # 元数据扁平字段列表 / 用于 index.csv 对应列
    'model_type', 'model_script', 'incident_angle', 'mesh_size', 'slope_i',
    'total_L', 'left_flat', 'H_minus_h', 'h_over_H', 'bedrock_thickness', 'H', 'h', 'w_slope',
    'n_finite_layers', 'n_layers_total', 'vs_bedrock', 'vs_surface', 'vs_cover',
    'vr_over_vs2', 'vs1_over_vs2', 'slope_height', 'a0_base', 'layers_json',
]


def _read_meta(folder):  # 读取工况元数据 case_meta.json
    """读取工况文件夹下的元数据文件并返回字典。

    参数:
        folder (str): 工况文件夹的绝对或相对路径

    返回:
        dict: 解析后的元数据字典；若文件不存在或读取失败则返回 None
    """
    path = os.path.join(folder, 'case_meta.json')  # 元数据路径
    if not os.path.isfile(path):  # 文件不存在
        return None  # 返回空
    try:  # 尝试读取解析
        with io.open(path, 'r', encoding='utf-8') as f:  # 打开元数据文件
            return json.load(f)  # 返回解析出的字典
    except Exception:  # 解析异常
        return None  # 返回空


def _read_summary(folder):  # 读取地表响应摘要 surface_summary.json
    """读取地表响应摘要文件并提取各记录的指标映射。

    参数:
        folder (str): 工况文件夹的绝对或相对路径

    返回:
        dict: 键为 record 名字，值为包含 AR_max 和 suspect 字典的映射
    """
    path = os.path.join(folder, 'surface_summary.json')  # 摘要路径
    if not os.path.isfile(path):  # 文件不存在
        return {}  # 返回空字典
    try:  # 尝试读取解析
        with io.open(path, 'r', encoding='utf-8') as f:  # 打开摘要文件
            data = json.load(f)  # 加载 json 字典
            records = data.get('records', [])  # 获取记录列表
            mapping = {}  # 映射结果字典
            for r in records:  # 遍历记录项
                rec_name = r.get('record')  # 获取记录名称
                if rec_name:  # 记录名非空
                    mapping[rec_name] = {  # 填充指标
                        'AR_max': r.get('AR_max'),  # 提取最大放大倍数
                        'suspect': r.get('suspect'),  # 提取可疑标记
                    }
            return mapping  # 返回映射结果
    except Exception:  # 解析异常
        return {}  # 返回空字典


def _flatten(meta):  # 展平嵌套元数据为规范一维字典
    """把嵌套的元数据字典展平为 index.csv 规范的平铺列字典。

    参数:
        meta (dict): 原始元数据字典

    返回:
        dict: 展平后的字典，键为 INDEX_META_FIELDS 中的列名
    """
    g = meta.get('geometry', {}) or {}  # 几何子字典
    d = meta.get('derived', {}) or {}  # 派生子字典
    return {  # 组装规范扁平列
        'model_type': meta.get('model_type'),  # 模型类型
        'model_script': meta.get('model_script'),  # 建模脚本
        'incident_angle': meta.get('incident_angle'),  # 入射角度
        'mesh_size': meta.get('mesh_size'),  # 网格大小
        'slope_i': g.get('i'),  # 坡角
        'total_L': g.get('total_L'),  # 模型总长
        'left_flat': g.get('left_flat'),  # 左侧平地长度
        'H_minus_h': g.get('H_minus_h'),  # 斜坡高度
        'h_over_H': g.get('h_over_H'),  # 深度比
        'bedrock_thickness': g.get('bedrock_thickness'),  # 基岩厚度
        'H': g.get('H'),  # 总厚度
        'h': g.get('h'),  # 下部厚度
        'w_slope': g.get('w_slope'),  # 坡段水平跨度
        'n_finite_layers': d.get('n_finite_layers'),  # 有限层数
        'n_layers_total': d.get('n_layers_total'),  # 总层数
        'vs_bedrock': d.get('vs_bedrock'),  # 基岩波速
        'vs_surface': d.get('vs_surface'),  # 表层波速
        'vs_cover': d.get('vs_cover'),  # 覆盖层波速
        'vr_over_vs2': d.get('vr_over_vs2'),  # 剪切波速比
        'vs1_over_vs2': d.get('vs1_over_vs2'),  # 层间波速比
        'slope_height': d.get('slope_height'),  # 归一化高度
        'a0_base': d.get('a0_base'),  # a0 基础值
        'layers_json': json.dumps(meta.get('layers', []), ensure_ascii=False),  # 完整土层层信息 JSON
    }


def split_csv_name(stem):  # 剥离前缀并解析场景与记录名
    """通过最长匹配原则将文件名剥离为类型、记录名和场景。

    参数:
        stem (str): CSV 文件的主干名称

    返回:
        tuple: (ftype, record, scene) 三元组，匹配失败时返回 (None, None, None)
    """
    stem_lower = stem.lower()  # 小写比对副本
    matched_prefix = None  # 匹配前缀变量
    matched_type = None  # 匹配类型变量

    for pref, ttype in KNOWN_PREFIXES:  # 遍历已知前缀
        if stem_lower.startswith(pref.lower()):  # 检查前缀匹配
            matched_prefix = pref  # 记录匹配的前缀
            matched_type = ttype  # 记录对应的类型
            break  # 找到最长匹配后退出

    if not matched_prefix:  # 无有效前缀
        return None, None, None  # 返回空三元组

    rest = stem[len(matched_prefix):]  # 剥离前缀后的文本
    scene = ''  # 初始化场景变量
    low = rest.lower()  # 转为小写比对
    if low.endswith('-slope'):  # 坡地场景
        scene = 'slope'  # 标记坡地
        record = rest[:-6]  # 截去场景后缀
    elif low.endswith('-flat'):  # 平地场景
        scene = 'flat'  # 标记平地
        record = rest[:-5]  # 截去场景后缀
    else:  # 无场景标记
        record = rest  # 记录名即为剩余全部

    return matched_type, record, scene  # 返回拆解结果


def main():  # 主入口逻辑
    """收集脚本的主控制流程。

    遍历文件夹，对各工况目标 CSV 整理归档，并输出清单 index.csv。
    """
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()  # 收集根目录路径
    root = os.path.abspath(root)  # 转换为绝对路径
    out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3 else os.path.join(root, OUT_DIRNAME)  # 集中输出路径
    os.makedirs(out_dir, exist_ok=True)  # 创建集中输出目录
    print('>>> 收集根目录: %s' % root)  # 输出提示
    print('>>> 输出目录:   %s' % out_dir)  # 输出提示

    manifest = []  # 文件清单暂存器
    n_folders = 0  # 成功识别的工况文件夹数
    n_files = 0  # 已归档的文件总数
    n_missing_meta = 0  # 缺失元数据的文件夹数
    
    for entry in sorted(os.listdir(root)):  # 按文件夹名称排序遍历
        folder = os.path.join(root, entry)  # 工况文件夹绝对路径
        if not os.path.isdir(folder) or entry in SKIP_DIR_NAMES or entry.startswith('.'):  # 过滤非工况目标
            continue  # 跳过

        # 遍历目标前缀，进行多格式的 CSV 文件搜寻
        csvs = []  # 工况下匹配的 CSV 列表
        for prefix in COLLECT_PREFIXES:  # 遍历支持前缀
            csvs += glob.glob(os.path.join(folder, '%s-*.csv' % prefix))  # 匹配中划线模式
            csvs += glob.glob(os.path.join(folder, '%s_*.csv' % prefix))  # 匹配下划线模式

        # 去重并滤掉归一化时的临时输出
        csvs = [f for f in sorted(set(csvs)) if '-normalized' not in os.path.basename(f).lower()]  # 过滤后目标文件列表
        if not csvs:  # 当前文件夹无任何目标数据
            continue  # 跳过

        meta = _read_meta(folder)  # 读取 case_meta.json 元数据
        if meta is None:  # 元数据不存在
            n_missing_meta += 1  # 递增警告计数
            meta_flat = {k: None for k in INDEX_META_FIELDS}  # 填充为空值列
            print('--- %s  (警告: 缺 case_meta.json，元数据列留空，建议重跑补元数据) ---' % entry)  # 提示用户警告
        else:  # 元数据存在
            meta_flat = _flatten(meta)  # 展平元数据列
            print('--- %s  (type=%s, i=%s, θs=%s, n_layers=%s, Vr/Vs2=%s) ---' % (  # 打印工况摘要信息
                entry, meta_flat.get('model_type'), meta_flat.get('slope_i'),
                meta_flat.get('incident_angle'), meta_flat.get('n_layers_total'),
                meta_flat.get('vr_over_vs2')))

        # 尝试读取本工况生成的地表摘要以获取 AR_max 与 suspect 参数
        summary_map = _read_summary(folder)  # 读取摘要映射字典
        n_folders += 1  # 工况目录计数递增

        for src in csvs:  # 循环每个需要归档的文件
            stem = os.path.splitext(os.path.basename(src))[0]  # 提取主干名
            ftype, record, scene = split_csv_name(stem)  # 剥离并分析
            if ftype is None:  # 匹配前缀失败
                continue  # 跳过

            scene_suffix = ('-' + scene) if scene else ''  # 场景后缀处理
            # 采用标准化的大写前缀连接工况名及提取出的记录名以防冲突
            new_name = '%s-%s%s%s%s.csv' % (ftype, entry, SEP, record, scene_suffix)  # 新规范文件名
            dst = os.path.join(out_dir, new_name)  # 目标位置
            shutil.copy2(src, dst)  # 复制并保留原始时间戳
            n_files += 1  # 计数递增

            # 在 summary 映射中寻找对应记录的 AR_max 和 suspect 标量
            rec_summary = summary_map.get(record, {})  # 查找记录参数
            ar_max_val = rec_summary.get('AR_max')  # 提取 AR_max
            suspect_val = rec_summary.get('suspect')  # 提取 suspect

            row = {  # 初始化文件级行记录
                'collected_file': new_name,  # 归档后唯一名
                'source_folder': entry,  # 原始工况名
                'type': ftype,  # 数据类型大写表示
                'record': record,  # 记录名
                'scene': scene,  # 地形场景
                'AR_max': ar_max_val,  # 峰值放大
                'suspect': suspect_val,  # 是否异常警告
            }
            row.update(meta_flat)  # 合并工况结构元数据
            manifest.append(row)  # 追加至清单列表
            print('    %s  ->  %s' % (os.path.basename(src), new_name))  # 打印归档日志

    if not manifest:  # 清单为空则无处理
        print('未发现任何含有符合前缀 CSV 文件的工况文件夹。')  # 提示用户
        return  # 退出

    # 将清单写出至 index.csv 文件中
    index_path = os.path.join(out_dir, 'index.csv')  # 清单输出绝对路径
    fields = BASE_FIELDS + list(INDEX_META_FIELDS) + ['AR_max', 'suspect']  # 表头组合清单列
    with open(index_path, 'w', newline='', encoding='utf-8-sig') as f:  # UTF-8 带 BOM CSV 写入
        w = csv.DictWriter(f, fieldnames=fields)  # 创建写入器
        w.writeheader()  # 写表头
        for row in manifest:  # 循环行
            w.writerow({k: row.get(k) for k in fields})  # 单独提取写入规范的列数据

    print('\n>>> 完成：从 %d 个工况文件夹收集 %d 个 CSV 到 %s' % (n_folders, n_files, out_dir))  # 结束提示
    if n_missing_meta:  # 提示缺失元数据状况
        print('>>> 注意：%d 个文件夹缺 case_meta.json（元数据列留空）。请用已配置 case_meta 的建模脚本重跑。' % n_missing_meta)  # 元数据警告
    print('>>> 清单：%s' % index_path)  # 清单路径提示


if __name__ == '__main__':  # 程序入口
    main()  # 执行主流程
