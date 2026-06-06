# -*- coding: utf-8 -*-
"""
PGA 后处理脚本 v7 —— 适用于 VAB_oblique_TAF_double_v2.py 创建的模型
改进点:
1) 顶面节点直接读取实例中的 TOP_SURFACE 节点集（全部节点），并按 x 坐标升序排列（空间排序）。
2) 动态计算尺度 h (斜坡高度)：
   - 对 slope 模型，计算顶面节点的最大与最小 y 坐标差值。
   - 对 flat 模型，查找同目录下对应的前缀相同且包含 -slope 的 ODB，并读取其差值；
     如无，查找目录下任意 slope 模型的差值；若再无，解析同目录下 *.cae 文件名中的 h 参数；若无，兜底使用 200.0。
3) 为避免丢失整个时程数据并防范 OOM：
   - 先保存顶面节点的所有时程数据：增量式（逐帧）写入 TIMESERIES-{name}.csv 中。
   - 帧循环完成后，再在内存中整合峰值（PGA）数据并按 x/h 排序，写入 PGA-{name}.csv 中。
"""

from odbAccess import openOdb  # 导入 Abaqus ODB 访问接口
import os  # 导入操作系统路径工具
import csv  # 导入 CSV 读写模块
import time  # 导入时间计时模块
import logging  # 导入日志记录模块
import traceback  # 导入异常堆栈打印模块
import glob  # 导入文件路径通配符匹配模块
import re  # 导入正则表达式模块

# ==============================================================================
#  配置项
# ==============================================================================
EXPORT_X_OVER_H = False  # 是否在 CSV 中导出 x/h 这一列（若需要兼容旧绘图脚本，可设为 True）


def log_step(logger=None, message=None, *args):
    """
    日志函数：首次调用时初始化日志器，后续调用输出带总用时的日志。
    初始化:    logger = log_step('mylog.log')  # 传入日志文件名
               logger = log_step()            # 使用默认文件名 'logfile.log'
    记录日志:  log_step(logger, '消息 %s', val)

    logger: str 或 logging.Logger 实例；首次调用时传入日志文件名字符串
    message: str，日志消息模板（支持 % 占位符）
    *args: 消息模板对应的填充参数
    返回: logging.Logger 实例
    """
    if not hasattr(log_step, '_logger'):  # 判断是否为首次调用（尚未初始化日志器）
        if logger is not None and isinstance(logger, str):  # 若传入参数为字符串，则作为日志文件名
            log_filename = logger  # 使用传入的字符串作为日志文件名
            logger = None  # 清空 logger，避免后续误用
        else:
            log_filename = 'logfile.log'  # 未指定文件名时使用默认文件名

        _logger = logging.getLogger('abqpy')  # 创建名为 'abqpy' 的日志器实例
        _logger.setLevel(logging.INFO)  # 设置日志级别为 INFO
        _logger.propagate = False  # 禁止日志向父日志器传播

        _logger.handlers = []  # 清空已有的处理器，防止重复添加
        formatter = logging.Formatter(  # 定义日志格式化器
            '%(asctime)s [%(levelname)s] %(message)s',  # 格式：时间 [级别] 消息
            datefmt='%Y-%m-%d %H:%M:%S'  # 时间格式：年-月-日 时:分:秒
        )

        file_handler = logging.FileHandler(log_filename, mode='w')  # 创建文件处理器，以写入模式打开日志文件
        file_handler.setFormatter(formatter)  # 为文件处理器设置格式化器
        _logger.addHandler(file_handler)  # 将文件处理器添加到日志器

        log_step._logger = _logger  # 将日志器保存为函数属性，供后续调用复用
        log_step._start_time = time.time()  # 记录脚本启动时间，用于计算总耗时
        log_step._log_filename = log_filename  # 保存日志文件名到函数属性

        return _logger  # 首次调用返回已初始化的日志器

    if message is not None:  # 若传入了消息内容，则记录日志
        now = time.time()  # 获取当前时间戳
        delta_total = now - log_step._start_time  # 计算从脚本启动到当前的总耗时
        log_step._logger.info('[%.3fs] ' + message, delta_total, *args)  # 输出带总耗时的 INFO 日志

    return log_step._logger  # 返回已初始化的日志器实例


# ==============================================================================
#  核心：从 TOP_SURFACE 节点集读取顶部表面节点（全部保留）
# ==============================================================================
def find_top_surface_nodes(instance):
    """
    直接读取实例中的 TOP_SURFACE 节点集，返回全部节点。

    instance: Abaqus ODB 装配实例对象
    返回:
        top_nodes: dict, {node_label: (x, y)}
        diagnostics: dict, 诊断信息
    """
    try:
        top_nset = instance.nodeSets['TOP_SURFACE']  # 尝试读取名为 TOP_SURFACE 的节点集
    except KeyError:
        # TOP_SURFACE 节点集不存在时返回空字典和诊断信息
        return {}, {'total_nodes': 0, 'selected_top_nodes': 0, 'source': 'TOP_SURFACE missing'}

    top_nodes = {}  # 初始化节点字典，键为节点编号，值为 (x, y) 坐标元组
    for node in top_nset.nodes:  # 遍历 TOP_SURFACE 节点集中的所有节点
        x = node.coordinates[0]  # 读取节点 x 坐标
        y = node.coordinates[1]  # 读取节点 y 坐标
        top_nodes[node.label] = (x, y)  # 以节点编号为键存储坐标

    diagnostics = {  # 构造诊断信息字典
        'total_nodes': len(top_nodes),  # 总节点数
        'selected_top_nodes': len(top_nodes),  # 选中的顶部节点数（此处全部保留）
        'source': 'TOP_SURFACE',  # 数据来源标识
    }
    return top_nodes, diagnostics  # 返回节点字典和诊断信息


def strip_job_prefix(name):
    """若名称以 job- 开头则去掉该前缀，避免输出文件名包含 job-。

    name: str，待处理的名称字符串
    返回: str，去除前缀后的名称
    """
    if name.lower().startswith('job-'):  # 判断名称是否以 job- 开头（忽略大小写）
        return name[4:]  # 截取 job- 之后的部分作为返回值
    return name  # 未匹配前缀则原样返回


# ==============================================================================
#  动态计算斜坡高度 h 的辅助方法
# ==============================================================================
def get_slope_height_from_odb(odb_path, logger=None):
    """从指定 ODB 文件中打开并读取 TOP_SURFACE 节点集的 y 坐标最大差值作为斜坡高度。

    odb_path: str，目标 ODB 文件的完整路径
    logger: logging.Logger 实例，用于记录日志（可选）
    返回: float 或 None，斜坡高度 h；若读取失败则返回 None
    """
    try:
        temp_odb = openOdb(path=odb_path, readOnly=True)  # 以只读模式打开辅助 ODB 文件
        try:
            assembly = temp_odb.rootAssembly  # 获取根装配体对象
            inst_keys = list(assembly.instances.keys())  # 获取所有装配实例的名称列表
            if not inst_keys:  # 若无任何实例则返回 None
                return None
            instance = assembly.instances[inst_keys[0]]  # 取第一个实例
            top_nset = instance.nodeSets['TOP_SURFACE']  # 读取 TOP_SURFACE 节点集
            y_coords = [node.coordinates[1] for node in top_nset.nodes]  # 提取所有顶面节点的 y 坐标
            if y_coords:  # 若 y 坐标列表非空
                dy = max(y_coords) - min(y_coords)  # 计算 y 坐标最大差值
                if dy > 1.0:  # 差值大于 1.0 时认为是有效的斜坡高度
                    return dy  # 返回斜坡高度
        finally:
            temp_odb.close()  # 确保无论是否出错都关闭 ODB 文件
    except Exception as e:
        if logger:  # 若提供了日志器则记录错误信息
            log_step(logger, '读取辅助 ODB %s 失败: %s', os.path.basename(odb_path), str(e))
    return None  # 读取失败时返回 None


def get_slope_height_from_cae(directory, logger=None):
    """
    尝试从目录下的 .cae 文件名中解析斜坡高度 h。
    文件名格式类似于 h200_i30_a15.cae，其中 h 后的数字为斜坡高度。

    directory: str，待搜索的目录路径
    logger: logging.Logger 实例，用于记录日志（可选）
    返回: float 或 None，解析出的斜坡高度；若解析失败则返回 None
    """
    try:
        cae_files = sorted(glob.glob(os.path.join(directory, '*.cae')))  # 获取目录下所有 .cae 文件并排序
        if cae_files:  # 若找到 .cae 文件
            cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]  # 取第一个文件的不含扩展名的文件名
            match = re.search(r'h(?P<h>-?\d+(?:\.\d+)?)_', cae_name)  # 用正则表达式从文件名中匹配 h 参数
            if match:  # 若正则匹配成功
                h_val = float(match.group('h'))  # 提取匹配到的 h 数值并转为浮点数
                if h_val > 0.0:  # 确保 h 值为正数
                    return h_val  # 返回解析出的斜坡高度
    except Exception as e:
        if logger:  # 若提供了日志器则记录错误信息
            log_step(logger, '从 CAE 文件名解析 h 失败: %s', str(e))
    return None  # 解析失败时返回 None


# ==============================================================================
#  处理单个 ODB 文件
# ==============================================================================
def process_one_odb(odb_path, logger=None):
    """
    处理单个 ODB 文件，提取顶面节点时程加速度并计算 PGA，输出 TIMESERIES 和 PGA 两个 CSV 文件。

    odb_path: str，目标 ODB 文件的完整路径
    logger: logging.Logger 实例，用于记录日志（可选）
    """
    logger = logger or log_step()  # 若未传入日志器则使用默认日志器
    odb_basename = os.path.basename(odb_path)  # 获取 ODB 文件的文件名（含扩展名）
    odb_stem = os.path.splitext(odb_basename)[0]  # 获取 ODB 文件的不含扩展名的文件名

    log_step(logger, '开始处理 ODB: %s', odb_basename)  # 记录开始处理日志

    odb = openOdb(path=odb_path, readOnly=True)  # 以只读模式打开 ODB 文件

    try:
        assembly = odb.rootAssembly  # 获取根装配体对象
        inst_keys = list(assembly.instances.keys())  # 获取所有装配实例的名称列表
        if not inst_keys:  # 若无任何实例则跳过该 ODB
            log_step(logger, '%s 无装配实例，跳过', odb_basename)
            return
        instance = assembly.instances[inst_keys[0]]  # 取第一个装配实例

        top_nodes, _ = find_top_surface_nodes(instance)  # 读取顶面节点字典（忽略诊断信息）
        if not top_nodes:  # 若未找到任何顶面节点则跳过
            log_step(logger, '%s 未找到顶部表面节点，跳过', odb_basename)
            return

        # 将节点标号按 x 坐标升序排列（保证横向空间顺序）
        top_labels = sorted(list(top_nodes.keys()), key=lambda lb: top_nodes[lb][0])  # 按 x 坐标升序排序节点标号列表
        label_to_idx = {}  # 初始化节点标号到索引的映射字典
        for i, lb in enumerate(top_labels):  # 遍历排序后的节点标号列表
            label_to_idx[lb] = i  # 建立节点标号到列表索引的映射
        n_top = len(top_labels)  # 记录顶面节点总数

        # ==========================================
        #  动态计算斜坡高度 h
        # ==========================================
        y_coords = [top_nodes[lb][1] for lb in top_labels]  # 提取所有顶面节点的 y 坐标
        y_max = max(y_coords)  # 计算顶面节点 y 坐标最大值
        y_min = min(y_coords)  # 计算顶面节点 y 坐标最小值
        dy = y_max - y_min  # 计算顶面 y 坐标差值，判断是否为斜坡模型
        if dy > 1.0:  # 差值大于 1.0 则认为是 slope 模型
            h = dy  # 直接使用顶面起伏差值作为斜坡高度
            log_step(logger, '%s 识别为 slope 模型，根据顶面起伏自动计算 h = y_max - y_min = %.6f', odb_basename, h)
        else:  # 差值不足 1.0 则认为是 flat 模型，需从其他来源获取 h
            h = None  # 初始化 h 为 None，后续逐级尝试获取
            dir_name = os.path.dirname(odb_path)  # 获取当前 ODB 文件所在目录
            # 尝试寻找对应的 slope ODB (例如 job-XXX-flat.odb -> job-XXX-slope.odb)
            if '-flat' in odb_stem.lower():  # 判断当前 ODB 是否为 flat 模型
                slope_stem = odb_stem.lower().replace('-flat', '-slope')  # 构造对应 slope 文件的文件名（不含扩展名）
                for f in os.listdir(dir_name):  # 遍历同目录下的所有文件
                    if f.lower() == slope_stem + '.odb':  # 找到匹配的 slope ODB 文件
                        slope_path = os.path.join(dir_name, f)  # 构造 slope ODB 文件的完整路径
                        h = get_slope_height_from_odb(slope_path, logger)  # 从对应 slope ODB 中读取斜坡高度
                        if h is not None:  # 若成功获取到斜坡高度则记录日志并退出循环
                            log_step(logger, '%s 识别为 flat 模型，从对应 slope 文件 %s 中获取 h = %.6f', odb_basename, f, h)
                            break

            # 如果没找到，尝试寻找目录下任意一个 slope ODB
            if h is None:  # 若仍未获取到斜坡高度，则继续查找其他 slope ODB
                for f in os.listdir(dir_name):  # 遍历同目录下的所有文件
                    if f.lower().endswith('.odb') and '-slope' in f.lower():  # 找到任意一个 slope ODB 文件
                        slope_path = os.path.join(dir_name, f)  # 构造该 slope ODB 文件的完整路径
                        h = get_slope_height_from_odb(slope_path, logger)  # 从该 slope ODB 中读取斜坡高度
                        if h is not None:  # 若成功获取到斜坡高度则记录日志并退出循环
                            log_step(logger, '%s 识别为 flat 模型，从目录下任意 slope 文件 %s 中获取 h = %.6f', odb_basename, f, h)
                            break

            # 如果依然没找到，尝试从当前目录下的 .cae 文件名解析
            if h is None:  # 若仍未获取到斜坡高度，则尝试从 .cae 文件名解析
                h = get_slope_height_from_cae(dir_name, logger)  # 从 .cae 文件名中解析斜坡高度
                if h is not None:  # 若从 .cae 文件名解析成功则记录日志
                    log_step(logger, '%s 识别为 flat 模型，从 .cae 文件名中成功解析 h = %.6f', odb_basename, h)

            # 最终兜底使用默认值 200.0
            if h is None:  # 若所有方法均无法获取斜坡高度，则使用默认值兜底
                h = 200.0  # 设置兜底默认斜坡高度为 200.0
                log_step(logger, '%s 识别为 flat 模型，未找到 slope 或 CAE 文件，使用默认值 h = %.6f', odb_basename, h)

        step_keys = list(odb.steps.keys())  # 获取所有分析步的名称列表
        if not step_keys:  # 若无任何分析步则跳过
            log_step(logger, '%s 无分析步，跳过', odb_basename)
            return

        step = odb.steps[step_keys[-1]]  # 取最后一个分析步（通常为动力分析步）
        frames = step.frames  # 获取该分析步的所有帧对象
        n_frames = len(frames)  # 记录总帧数
        if n_frames == 0:  # 若该分析步无任何帧数据则跳过
            log_step(logger, '%s 分析步 %s 无帧数据，跳过', odb_basename, step_keys[-1])
            return

        log_step(logger, '%s 使用分析步 %s，帧数=%d，顶部节点数=%d',
                 odb_basename, step_keys[-1], n_frames, n_top)  # 记录分析步基本信息日志

        pga_h = [0.0] * n_top  # 初始化各顶面节点水平 PGA 数组，初值为 0
        pga_v = [0.0] * n_top  # 初始化各顶面节点竖直 PGA 数组，初值为 0
        peak_h_time = [0.0] * n_top  # 初始化各节点水平 PGA 对应时刻数组
        peak_v_time = [0.0] * n_top  # 初始化各节点竖直 PGA 对应时刻数组

        top_nset = instance.nodeSets['TOP_SURFACE']  # 再次获取 TOP_SURFACE 节点集，用于帧数据子集提取
        log_step(logger, '%s 使用 TOP_SURFACE 节点集的全部节点进行后处理', odb_basename)

        # 准备输出文件名
        csv_stem = strip_job_prefix(odb_stem)  # 去除 ODB 文件名中的 job- 前缀，用于构造输出文件名
        timeseries_csv_name = 'TIMESERIES-{}.csv'.format(csv_stem)  # 构造时程数据 CSV 文件名
        pga_csv_name = 'PGA-{}.csv'.format(csv_stem)  # 构造 PGA 峰值表 CSV 文件名

        # ==========================================
        #  第一步：增量式逐帧保存时程数据到 TIMESERIES CSV
        # ==========================================
        log_step(logger, '%s 开始增量保存完整时程数据: %s', odb_basename, timeseries_csv_name)
        with open(timeseries_csv_name, 'w') as f_ts:  # 以写入模式打开时程 CSV 文件
            writer_ts = csv.writer(f_ts, lineterminator='\n')  # 创建 CSV 写入器，使用换行符分隔行

            # 写入表头
            header = ['Time']  # 初始化表头列表，首列为时间
            if EXPORT_X_OVER_H:  # 若配置要求导出 x/h 列
                header.append('x/h')  # 在表头中添加 x/h 列
            for label in top_labels:  # 为每个顶面节点添加水平加速度列名
                header.append('Node_{}_Accel_h'.format(label))  # 格式：Node_标号_Accel_h
            for label in top_labels:  # 为每个顶面节点添加竖直加速度列名
                header.append('Node_{}_Accel_v'.format(label))  # 格式：Node_标号_Accel_v
            writer_ts.writerow(header)  # 将表头行写入 CSV 文件

            for frame_idx, frame in enumerate(frames):  # 遍历所有帧（逐帧增量处理）
                if (frame_idx + 1) % max(1, n_frames // 10) == 0:  # 每完成约 10% 进度时记录一次日志
                    log_step(logger, '%s 处理帧进度: %d / %d', odb_basename, frame_idx + 1, n_frames)

                if 'A' not in frame.fieldOutputs:  # 若当前帧不含加速度场输出则跳过
                    continue

                acc_field = frame.fieldOutputs['A']  # 获取当前帧的加速度场输出对象
                t_cur = getattr(frame, 'frameValue', 0.0)  # 获取当前帧对应的时间值
                acc_values = acc_field.getSubset(region=top_nset).values  # 提取顶面节点集子集的加速度数据

                # 构建当前帧的数据映射 {node_label: (accel_h, accel_v)}
                frame_data = {}  # 初始化当前帧数据字典，键为节点标号，值为 (水平加速度, 竖直加速度)
                for val in acc_values:  # 遍历当前帧顶面节点的加速度数据
                    frame_data[val.nodeLabel] = (val.data[0], val.data[1])  # 存储水平与竖直加速度分量

                # 构造时程 data 行
                row = ['{:.6e}'.format(t_cur)]  # 初始化数据行，首列为当前帧时间（科学计数格式）
                if EXPORT_X_OVER_H:  # 若需要导出 x/h 列
                    first_node_x_over_h = top_nodes[top_labels[0]][0] / h  # 计算第一个顶面节点的 x/h 值
                    row.append('{:.6e}'.format(first_node_x_over_h))  # 将 x/h 值添加到数据行

                # 添加所有节点水平加速度
                for label in top_labels:  # 按空间顺序遍历所有顶面节点标号
                    accel_h = frame_data.get(label, (0.0, 0.0))[0]  # 获取当前节点水平加速度（缺失时默认 0）
                    row.append('{:.6e}'.format(accel_h))  # 将水平加速度添加到数据行

                # 添加所有节点竖直加速度
                for label in top_labels:  # 按空间顺序遍历所有顶面节点标号
                    accel_v = frame_data.get(label, (0.0, 0.0))[1]  # 获取当前节点竖直加速度（缺失时默认 0）
                    row.append('{:.6e}'.format(accel_v))  # 将竖直加速度添加到数据行

                writer_ts.writerow(row)  # 将当前帧的数据行写入 TIMESERIES CSV 文件

                # 更新峰值（PGA）数据
                for label in top_labels:  # 遍历所有顶面节点，更新各节点的峰值加速度
                    idx = label_to_idx[label]  # 获取当前节点在 PGA 数组中的索引
                    accel_h, accel_v = frame_data.get(label, (0.0, 0.0))  # 获取当前节点本帧的水平和竖直加速度
                    a1 = abs(accel_h)  # 取水平加速度绝对值
                    a2 = abs(accel_v)  # 取竖直加速度绝对值

                    if a1 > pga_h[idx]:  # 若当前帧水平加速度绝对值超过历史最大值
                        pga_h[idx] = a1  # 更新水平 PGA
                        peak_h_time[idx] = t_cur  # 更新水平 PGA 对应时刻

                    if a2 > pga_v[idx]:  # 若当前帧竖直加速度绝对值超过历史最大值
                        pga_v[idx] = a2  # 更新竖直 PGA
                        peak_v_time[idx] = t_cur  # 更新竖直 PGA 对应时刻

        log_step(logger, '%s 完整时程数据已先保存至 %s', odb_basename, timeseries_csv_name)  # 记录时程文件保存完成日志

        # ==========================================
        #  第二步：保存 PGA 峰值表
        # ==========================================
        log_step(logger, '%s 开始保存峰值（PGA）数据: %s', odb_basename, pga_csv_name)
        results = []  # 初始化 PGA 结果列表，每个元素为一个节点的结果字典
        for i, label in enumerate(top_labels):  # 遍历按空间顺序排列的顶面节点
            x_coord, y_coord = top_nodes[label]  # 获取当前节点的 x、y 坐标
            x_over_h = x_coord / h  # 计算当前节点的 x/h 归一化坐标
            results.append({  # 将当前节点的 PGA 结果添加到结果列表
                'x/h': x_over_h,  # x 归一化坐标
                'node_label': label,  # 节点标号
                'x': x_coord,  # 节点 x 坐标（单位：m）
                'y': y_coord,  # 节点 y 坐标（单位：m）
                'PGA_h': pga_h[i],  # 水平方向峰值地面加速度
                'PGA_v': pga_v[i],  # 竖直方向峰值地面加速度
                'peak_h_time': peak_h_time[i],  # 水平 PGA 出现的时刻（s）
                'peak_v_time': peak_v_time[i],  # 竖直 PGA 出现的时刻（s）
            })

        # 按 x 坐标进行升序排序（与按 x/h 排序等效）
        results.sort(key=lambda r: r['x'])  # 按节点 x 坐标升序排列结果列表

        with open(pga_csv_name, 'w') as f_pga:  # 以写入模式打开 PGA CSV 文件
            writer_pga = csv.writer(f_pga, lineterminator='\n')  # 创建 CSV 写入器

            pga_header = []  # 初始化 PGA 表头列表
            if EXPORT_X_OVER_H:  # 若需要导出 x/h 列
                pga_header.append('x/h')  # 在表头中添加 x/h 列
            pga_header.extend(['node_label', 'x', 'y', 'PGA_h', 'PGA_v', 'peak_h_time', 'peak_v_time'])  # 添加固定列名
            writer_pga.writerow(pga_header)  # 将表头行写入 PGA CSV 文件

            for r in results:  # 遍历按 x 坐标排序后的 PGA 结果列表
                row = []  # 初始化当前节点的数据行
                if EXPORT_X_OVER_H:  # 若需要导出 x/h 列
                    row.append('%.6f' % r['x/h'])  # 将 x/h 值格式化后添加到数据行
                row.extend([  # 将其余各列数据格式化后添加到数据行
                    '%d' % r['node_label'],  # 节点标号（整数格式）
                    '%.6f' % r['x'],  # x 坐标（6 位小数）
                    '%.6f' % r['y'],  # y 坐标（6 位小数）
                    '%.6f' % r['PGA_h'],  # 水平 PGA（6 位小数）
                    '%.6f' % r['PGA_v'],  # 竖直 PGA（6 位小数）
                    '%.6f' % r['peak_h_time'],  # 水平 PGA 出现时刻（6 位小数）
                    '%.6f' % r['peak_v_time'],  # 竖直 PGA 出现时刻（6 位小数）
                ])
                writer_pga.writerow(row)  # 将当前节点的数据行写入 PGA CSV 文件
        log_step(logger, '%s PGA 峰值表已后保存至 %s，节点数=%d', odb_basename, pga_csv_name, len(results))  # 记录 PGA 文件保存完成日志

    except Exception as exc:
        log_step(logger, '%s 处理失败: %s', odb_basename, str(exc))  # 记录异常信息
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 输出完整异常堆栈到日志
        raise  # 重新抛出异常，终止当前 ODB 处理
    finally:
        odb.close()  # 确保无论成功或失败都关闭 ODB 文件，释放资源


# ==============================================================================
#  主入口
# ==============================================================================
if __name__ == '__main__':
    logger = log_step('Postprocess_PGA.log')  # 初始化日志器，指定日志文件名
    total_start = time.time()  # 记录脚本整体开始时间
    try:
        log_step(logger, '脚本开始执行')  # 记录脚本启动日志

        cwd = os.getcwd()  # 获取当前工作目录路径
        odb_files = sorted([f for f in os.listdir(cwd) if f.lower().endswith('.odb')])  # 获取当前目录下所有 ODB 文件并排序
        if not odb_files:  # 若当前目录下无 ODB 文件则记录警告日志
            log_step(logger, '当前目录 %s 下未找到 .odb 文件', cwd)
        else:
            log_step(logger, '共找到 %d 个 ODB 文件', len(odb_files))  # 记录找到的 ODB 文件数量

        for odb_file in odb_files:  # 遍历当前目录下的所有 ODB 文件
            odb_path = os.path.join(cwd, odb_file)  # 构造当前 ODB 文件的完整路径
            process_one_odb(  # 调用单文件处理函数处理当前 ODB
                odb_path=odb_path,  # 传入 ODB 文件路径
                logger=logger  # 传入日志器实例
            )

        log_step(logger, '脚本执行完成，总耗时=%.2fs', time.time() - total_start)  # 记录脚本执行完成及总耗时
    except Exception as exc:
        log_step(logger, '脚本失败: %s', str(exc))  # 记录脚本失败信息
        logger.error('异常堆栈:\n%s', traceback.format_exc())  # 输出完整异常堆栈到日志
        raise  # 重新抛出异常，终止脚本
