# -*- coding: utf-8 -*-
"""跨工况结果收集器 v2（Hybrid 专用版）。

本脚本负责遍历指定根目录下的各个工况文件夹，收集 v2 后处理输出的 surface_results.npz，
并按其 manifest_json 中的逐记录 s 网格响应表展开 index.csv。保留对旧版临时 CSV 的兼容，
使集中目录可直接被 Plot_Hybrid_surface_v2.py 读取，不依赖已被打包清理的临时文件。
"""

import os  # 导入系统接口模块
import re  # 导入正则模块
import sys  # 导入系统参数模块
import glob  # 导入文件匹配模块
import shutil  # 导入文件复制模块
import csv  # 导入 CSV 写入模块
import io  # 导入 io 模块
import json  # 导入 JSON 模块
import hashlib  # 计算脚本和输入波哈希
import numpy as np  # 导入 NumPy 模块用于读取 NPZ 清单
try:
    if hasattr(sys, 'setdefaultencoding'):  # 仅在 Python 2 下执行
        eval("reload(sys)")  # 用 eval 动态执行，避开 Python 3 静态分析对未定义 reload 的报错
        sys.setdefaultencoding('utf-8')  # 设置默认编码
except Exception:
    pass



# ==============================================================================
#  配置
# ==============================================================================
COLLECT_PREFIXES = ('sgrid_response', 'sgrid_H_surface_h', 'sgrid_H_surface_v', 'sgrid_H_topo_h')  # 收集前缀 / 仅收集 v2 统一 s 子网格对齐产物
SURFACE_NPZ_NAME = 'surface_results.npz'  # v2 后处理的单工况最终数值包名称
SURFACE_NPZ_TYPE = 'SURFACE_RESULTS_NPZ'  # 供统一绘图脚本识别的最终数值包类型
KNOWN_PREFIXES = [  # 已知前缀转换映射表 / 长度从长到短排列防截断错误
    ('sgrid_H_surface_h_', 'SGRID_H_SURFACE_H'),  # 对齐水平传函（v2 重采样）
    ('sgrid_H_surface_v_', 'SGRID_H_SURFACE_V'),  # 对齐竖向传函（v2 重采样）
    ('sgrid_H_topo_h_', 'SGRID_H_TOPO_H'),  # 对齐地形谱比（v2 重采样）
    ('sgrid_response_', 'SGRID_RESPONSE'),  # 对齐响应表（v2 重采样）
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
AUDIT_FIELDS = [  # 结果侧审计字段 / 不以注入配置代替实际产物
    'actual_dt', 'duration_s', 'n_surface_nodes', 'n_model_nodes', 'n_elements', 'element_type',
    'damping_enable', 'damping_fc', 'damping_method', 'damping_anchor',
    'domain_xmin', 'domain_xmax', 'domain_ymin', 'domain_ymax', 'domain_source',
    'validation_geometry', 'script_hashes_json', 'input_wave_hashes_json',
    'qa_required', 'qa_theory', 'qa_reflection', 'qa_mesh', 'qa_time', 'qa_domain',
    'qa_energy', 'qa_external', 'overall_pass', 'qa_status', 'qa_gate_status_json',
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


def _read_json_file(path):  # 读取 JSON 配置，缺失或损坏时返回空字典
    if not os.path.isfile(path):
        return {}
    try:
        with io.open(path, 'r', encoding='utf-8') as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _sha256_file(path):  # 计算文件 SHA-256，缺失文件返回 None
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _repo_root_for_hashes():  # 定位仓库根目录以读取固定四脚本
    candidates = []
    here = os.path.abspath(os.path.dirname(__file__))
    candidates.append(os.path.abspath(os.path.join(here, '..', '..')))
    candidates.append(os.path.abspath(os.getcwd()))
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py')):
            return candidate
    return None


def _read_geometry_audit(folder):  # 读取建模阶段保存的几何和网格审计
    return _read_json_file(os.path.join(folder, 'geometry_validation.json'))


def _read_model_log_audit(folder):  # 从建模日志回退读取实际节点和单元数
    path = os.path.join(folder, 'slope_frame_ssi_full_v2.log')
    if not os.path.isfile(path):
        return {}
    try:
        text = io.open(path, 'r', encoding='utf-8', errors='ignore').read()
    except TypeError:  # Python 2 的 io.open 不接受 errors 参数
        text = io.open(path, 'r', encoding='utf-8').read()
    match = re.search(r'网格统计:\s*单元=(\d+),\s*节点=(\d+)', text)
    if not match:
        return {}
    return {'n_elements': int(match.group(1)), 'n_model_nodes': int(match.group(2))}


def _script_hashes(folder):  # 记录固定四脚本和工况内副本的哈希
    root = _repo_root_for_hashes()
    fixed = [
        ('Modeling/Hybrid/slope_frame_ssi_full_v2.py', os.path.join(root, 'Modeling', 'Hybrid', 'slope_frame_ssi_full_v2.py') if root else None),
        ('Postprocess/Hybrid/Postprocess_All_surface_v2.py', os.path.join(root, 'Postprocess', 'Hybrid', 'Postprocess_All_surface_v2.py') if root else None),
        ('Postprocess/Hybrid/Collect_All_results_v2.py', os.path.join(root, 'Postprocess', 'Hybrid', 'Collect_All_results_v2.py') if root else None),
        ('Postprocess/Hybrid/Plot_Hybrid_surface_v2.py', os.path.join(root, 'Postprocess', 'Hybrid', 'Plot_Hybrid_surface_v2.py') if root else None),
    ]
    result = {}
    for relative, canonical in fixed:
        local = os.path.join(folder, os.path.basename(relative))
        path = canonical if canonical and os.path.isfile(canonical) else local
        result[relative] = {'sha256': _sha256_file(path), 'source': path if path and os.path.isfile(path) else None}
    return result


def _input_wave_hashes(folder, config):  # 记录实际注入输入波文件哈希
    waves = ((config.get('run_cfg') or {}).get('wave_files') or []) if isinstance(config, dict) else []
    if not isinstance(waves, (list, tuple)):
        waves = [waves]
    result = {}
    for raw in waves:
        path = str(raw)
        if not os.path.isabs(path):
            path = os.path.join(folder, path)
        result[str(raw)] = {'sha256': _sha256_file(path), 'path': os.path.abspath(path)}
    return result


def _audit_fields(folder, meta, config, summary):  # 组合结果侧实际审计字段
    geometry_audit = _read_geometry_audit(folder)
    log_audit = _read_model_log_audit(folder)
    bbox = geometry_audit.get('bbox') or {}
    geometry = (meta or {}).get('geometry') or {}
    if not bbox:
        total_l = geometry.get('total_L')
        bedrock = geometry.get('bedrock_thickness')
        h = geometry.get('H') or geometry.get('H_minus_h') or 0.0
        bbox = {'xmin': 0.0, 'xmax': total_l, 'ymin': 0.0,
                'ymax': (float(bedrock) + float(h)) if bedrock is not None else None}
    mesh_cfg = (config or {}).get('mesh_cfg') or {}
    damping = (meta or {}).get('damping') or (config or {}).get('damping_cfg') or {}
    mesh_audit = dict(log_audit)
    mesh_audit.update({'n_model_nodes': geometry_audit.get('node_count', mesh_audit.get('n_model_nodes')),
                       'n_elements': geometry_audit.get('element_count', mesh_audit.get('n_elements'))})
    gates = summary.get('qa_gates') or {}
    gate_status = summary.get('qa_gate_status') or {}
    return {
        'actual_dt': summary.get('dt'), 'duration_s': summary.get('duration'),
        'n_surface_nodes': summary.get('n_nodes'), 'n_model_nodes': mesh_audit.get('n_model_nodes'),
        'n_elements': mesh_audit.get('n_elements'), 'element_type': mesh_cfg.get('elem'),
        'damping_enable': damping.get('enable'), 'damping_fc': damping.get('fc'),
        'damping_method': damping.get('method'), 'damping_anchor': damping.get('anchor'),
        'domain_xmin': bbox.get('xmin'), 'domain_xmax': bbox.get('xmax'),
        'domain_ymin': bbox.get('ymin'), 'domain_ymax': bbox.get('ymax'),
        'domain_source': 'geometry_validation.json' if geometry_audit else 'case_meta.geometry_fallback',
        'validation_geometry': (meta or {}).get('validation_geometry'),
        'script_hashes_json': json.dumps(_script_hashes(folder), ensure_ascii=True, sort_keys=True),
        'input_wave_hashes_json': json.dumps(_input_wave_hashes(folder, config), ensure_ascii=True, sort_keys=True),
        'qa_required': json.dumps(summary.get('qa_required'), ensure_ascii=True),
        'qa_theory': gates.get('theory'), 'qa_reflection': gates.get('reflection'),
        'qa_mesh': gates.get('mesh'), 'qa_time': gates.get('time'), 'qa_domain': gates.get('domain'),
        'qa_energy': gates.get('energy'), 'qa_external': gates.get('external'),
        'overall_pass': summary.get('overall_pass'), 'qa_status': summary.get('qa_status'),
        'qa_gate_status_json': json.dumps(gate_status, ensure_ascii=True, sort_keys=True),
    }


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
                        'dt': r.get('dt'), 'duration': r.get('duration'), 'n_nodes': r.get('n_nodes'),
                        'qa_required': r.get('qa_required'), 'qa_gates': r.get('qa_gates'),
                        'qa_gate_status': r.get('qa_gate_status'), 'overall_pass': r.get('overall_pass'),
                        'qa_status': r.get('qa_status'),
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


def _to_py2_str(val):  # 兼容 Py2/Py3 字符串转码
    """在 Python 2.7 下将 unicode 递归编码为 utf-8 字节串；Python 3 下保持原样。"""
    if sys.version_info[0] < 3:  # 处于 Python 2 环境
        if isinstance(val, unicode):  # unicode 字符
            return val.encode('utf-8')  # 编码
        elif isinstance(val, (list, tuple)):  # 列表元组
            return [_to_py2_str(item) for item in val]  # 递归转换
        elif isinstance(val, dict):  # 字典
            return {k: _to_py2_str(v) for k, v in val.items()}  # 递归转换键值
    return val  # 保持原样


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


def _npz_text(value):  # 兼容 Python 2/3 解析 NPZ 内 UTF-8 文本标量
    """将 NPZ 内的字节标量或 0 维数组转换为文本。"""
    if hasattr(value, 'item'):  # NumPy 标量或 0 维数组
        value = value.item()  # 取出实际标量
    if isinstance(value, bytes):  # Python 3 字节串
        return value.decode('utf-8')  # 按 UTF-8 解码
    try:
        if isinstance(value, unicode):  # Python 2 unicode
            return value  # 直接返回
    except NameError:  # Python 3 无 unicode 名称
        pass
    return str(value)  # 其他类型统一转文本


def _records_from_surface_npz(path):  # 从最终数值包中恢复记录名与场景
    """读取 manifest_json，返回包含 sgrid 响应表的 (record, scene) 列表。"""
    package = np.load(path)  # NPZ 由同链路后处理脚本生成，不使用 pickle
    try:
        manifest = json.loads(_npz_text(package['manifest_json']))  # 读取内部表清单
        records = []  # 初始化记录列表
        seen = set()  # 避免同一记录重复写入索引
        for item in manifest:  # 遍历打包的临时表名
            name = str(item.get('name', ''))  # 获取表文件名
            if not (name.startswith('sgrid_response_') and name.lower().endswith('.csv')):  # 仅保留统一 s 网格响应表
                continue  # 忽略传函等同记录附属表
            _, record, scene = split_csv_name(os.path.splitext(name)[0])  # 复用既有命名解析规则
            key = (record, scene)  # 构造去重键
            if record and key not in seen:  # 记录有效且未处理
                seen.add(key)  # 标记已处理
                records.append(key)  # 保存记录名与场景
        return records  # 返回展开后的记录列表
    finally:
        package.close()  # 及时关闭压缩包文件句柄


def _summary_from_surface_npz(path):  # 从最终数值包中恢复质量摘要
    """读取 surface_summary_json，返回按 record 索引的 AR_max 与 suspect 映射。"""
    package = np.load(path)  # 打开同工况最终数值包
    try:
        if 'surface_summary_json' not in package.files:  # 兼容尚未写入摘要的旧包
            return {}  # 无摘要时交由旧版 JSON 回退
        summary = json.loads(_npz_text(package['surface_summary_json']))  # 解析打包前的质量摘要
        mapping = {}  # 初始化按输入记录索引的摘要映射
        for item in summary.get('records', []):  # 遍历逐记录质量结果
            raw_record = item.get('record')  # 获取后处理摘要中的原始记录名
            _, record, _ = split_csv_name('sgrid_response_' + str(raw_record or ''))  # 与 NPZ 表名使用同一场景后缀规范化规则
            if record:  # 仅保存具名记录
                mapping[record] = {
                    'AR_max': item.get('AR_max'),  # 原始节点曲线上的最大水平放大
                    'suspect': item.get('suspect'),  # 最终远场或窗口收敛质量标记
                    'dt': item.get('dt'), 'duration': item.get('duration'), 'n_nodes': item.get('n_nodes'),
                    'qa_required': item.get('qa_required'), 'qa_gates': item.get('qa_gates'),
                    'qa_gate_status': item.get('qa_gate_status'), 'overall_pass': item.get('overall_pass'),
                    'qa_status': item.get('qa_status'),
                }
        return mapping  # 返回可直接合并到索引的摘要
    finally:
        package.close()  # 及时关闭压缩包文件句柄


def main():  # 主入口逻辑
    """收集脚本的主控制流程。

    遍历文件夹，对各工况最终 NPZ（及兼容 CSV）整理归档，并输出清单 index.csv。
    """
    root = sys.argv[1] if len(sys.argv) >= 2 else os.getcwd()  # 收集根目录路径
    root = os.path.abspath(root)  # 转换为绝对路径
    out_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) >= 3 else os.path.join(root, OUT_DIRNAME)  # 集中输出路径
    if not os.path.exists(out_dir):  # 检查输出目录是否存在
        try:  # 尝试创建目录
            os.makedirs(out_dir)  # 递归创建目录
        except OSError:  # 异常忽略
            pass
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

        # 遍历目标前缀，进行旧版临时 CSV 文件搜寻
        csvs = []  # 工况下匹配的 CSV 列表
        for prefix in COLLECT_PREFIXES:  # 遍历支持前缀
            csvs += glob.glob(os.path.join(folder, '%s-*.csv' % prefix))  # 匹配中划线模式
            csvs += glob.glob(os.path.join(folder, '%s_*.csv' % prefix))  # 匹配下划线模式

        # 去重并滤掉归一化时的临时输出
        csvs = [f for f in sorted(set(csvs)) if '-normalized' not in os.path.basename(f).lower()]  # 过滤后目标文件列表
        npz_path = os.path.join(folder, SURFACE_NPZ_NAME)  # 定位 v2 后处理最终数值包
        has_npz = os.path.isfile(npz_path)  # 判断最终数值包是否存在
        if not csvs and not has_npz:  # 当前文件夹无任何目标数据
            continue  # 跳过

        meta = _read_meta(folder)  # 读取 case_meta.json 元数据
        config = _read_json_file(os.path.join(folder, 'case_config.json'))  # 读取实际工况配置用于审计
        if meta is None:  # 元数据不存在
            n_missing_meta += 1  # 递增警告计数
            meta_flat = {k: None for k in INDEX_META_FIELDS}  # 填充为空值列
            print('--- %s  (警告: 缺 case_meta.json，元数据列留空，建议重跑补元数据) ---' % entry)  # 提示用户警告
        else:  # 元数据存在
            meta_flat = _flatten(meta)  # 展平元数据列
            print('--- %s  (type=%s, i=%s, theta_s=%s, n_layers=%s, Vr/Vs2=%s) ---' % (  # 打印工况摘要信息
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
            audit = _audit_fields(folder, meta or {}, config, rec_summary)  # 组合实际审计字段

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
            row.update(audit)  # 合并结果侧审计字段
            manifest.append(row)  # 追加至清单列表
            print('    %s  ->  %s' % (os.path.basename(src), new_name))  # 打印归档日志

        if has_npz:  # v2 正常链路：收集单工况最终 NPZ 并按内部记录展开索引
            try:
                npz_records = _records_from_surface_npz(npz_path)  # 从 manifest_json 恢复记录名
                summary_map.update(_summary_from_surface_npz(npz_path))  # 从打包摘要恢复 AR_max 与质量标记
            except Exception as exc:
                print('    警告：无法读取 %s 的 manifest_json：%s' % (SURFACE_NPZ_NAME, str(exc)))  # 保留工况并提示包损坏根因
                npz_records = []  # 不生成无法供绘图使用的错误索引行
            if not npz_records:  # 数值包无可绘制记录
                print('    警告：%s 中未发现 sgrid_response_<record>.csv，跳过绘图索引。' % SURFACE_NPZ_NAME)  # 提示后处理产物不完整
            else:
                npz_name = '%s-%s.npz' % (SURFACE_NPZ_TYPE, entry)  # 每个工况保留一个唯一最终数值包
                npz_dst = os.path.join(out_dir, npz_name)  # 构造集中归档路径
                shutil.copy2(npz_path, npz_dst)  # 复制压缩数值包并保留时间戳
                n_files += 1  # 最终数值包计入已归档文件数
                for record, scene in npz_records:  # 每条记录写一行，供绘图脚本逐记录检索
                    rec_summary = summary_map.get(record, {})  # 尝试保留兼容摘要指标
                    audit = _audit_fields(folder, meta or {}, config, rec_summary)  # 组合 NPZ 对应记录的实际审计字段
                    row = {
                        'collected_file': npz_name,  # 集中目录中的 NPZ 文件名
                        'source_folder': entry,  # 原始工况目录
                        'type': SURFACE_NPZ_TYPE,  # 最终数值包类型
                        'record': record,  # NPZ 内对应的输入记录名
                        'scene': scene,  # 坡地或平地场景标记
                        'AR_max': rec_summary.get('AR_max'),  # 兼容旧摘要中的峰值放大
                        'suspect': rec_summary.get('suspect'),  # 兼容旧摘要中的可疑标记
                    }
                    row.update(meta_flat)  # 合并工况结构元数据
                    row.update(audit)  # 合并结果侧审计字段
                    manifest.append(row)  # 写入按记录展开的索引行
                    print('    %s[%s]  ->  %s' % (SURFACE_NPZ_NAME, record, npz_name))  # 打印归档日志

    if not manifest:  # 清单为空则无处理
        print('未发现任何含有最终 NPZ 或兼容 CSV 文件的工况文件夹。')  # 提示用户
        return  # 退出

    # 将清单写出至 index.csv 文件中
    index_path = os.path.join(out_dir, 'index.csv')  # 清单输出绝对路径
    fields = BASE_FIELDS + list(INDEX_META_FIELDS) + ['AR_max', 'suspect'] + list(AUDIT_FIELDS)  # 表头组合清单列
    if sys.version_info[0] >= 3:  # 处于 Python 3 环境
        f = io.open(index_path, 'w', newline='', encoding='utf-8-sig')  # 自动写 BOM
    else:  # 处于 Python 2 环境
        f = open(index_path, 'wb')  # 二进制写入模式避免 csv write 编码冲突
        f.write('\xef\xbb\xbf')  # 手动写 UTF-8 BOM 字节以使 Excel 能够识别
        
    try:  # 安全块写入
        w = csv.DictWriter(f, fieldnames=fields)  # 创建写入器
        w.writeheader()  # 写表头
        for row in manifest:  # 循环行
            clean_row = {k: _to_py2_str(row.get(k)) for k in fields}  # 转换 unicode
            w.writerow(clean_row)  # 写入 csv 行
    finally:
        f.close()  # 确保关闭文件句柄

    print('\n>>> 完成：从 %d 个工况文件夹收集 %d 个数据文件到 %s' % (n_folders, n_files, out_dir))  # 结束提示
    if n_missing_meta:  # 提示缺失元数据状况
        print('>>> 注意：%d 个文件夹缺 case_meta.json（元数据列留空）。请用已配置 case_meta 的建模脚本重跑。' % n_missing_meta)  # 元数据警告
    print('>>> 清单：%s' % index_path)  # 清单路径提示


if __name__ == '__main__':  # 程序入口
    main()  # 执行主流程
