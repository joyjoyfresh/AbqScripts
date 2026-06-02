# -*- coding: utf-8 -*-
"""
真实坐标 TAF 曲线绘制脚本
从已生成的 TAF-*.csv 文件中读取数据并绘制对比图，或由其他脚本导入调用。
"""

import os  # 导入系统接口模块
import re  # 导入正则表达式模块
import glob  # 导入全局文件匹配模块
import numpy as np  # 导入科学计算库 NumPy
import pandas as pd  # 导入数据分析库 Pandas
import matplotlib  # 导入绘图框架库 Matplotlib
matplotlib.use('Agg')  # 设置 Matplotlib 绘图后端为无界面的 Agg 模式
import matplotlib.pyplot as plt  # 导入画图子模块 Pyplot
import matplotlib.font_manager as fm  # 导入字体管理器模块
import matplotlib.ticker as mticker  # 导入刻度定位与格式化模块

# ==============================================================================
#  配置与常量
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
TAF_GLOB_PATTERN = 'TAF-*.csv'  # TAF 数据表文件模式
CUSTOM_YLIM = {'horizontal': (0, 2), 'vertical': (0, 1)}  # 自定义纵轴显示范围。可为 None (自适应)，元组 (min, max)，或字典分别设定 {'horizontal': (min, max), 'vertical': (min, max)}
#CUSTOM_YLIM = None  # 屏蔽的默认自适应纵轴配置，可用于切换为自适应模式
TARGET_INTERVALS = {'horizontal': 3, 'vertical': 5}  # 纵轴刻度区间划分数量。支持字典为水平/竖向图分别设置，也支持单个整数统一设置

# 默认坡体几何参数（当目录下不存在 PGA-*-slope.csv 文件时作为兜底使用）
DEFAULT_X_CREST = 800.0  # 默认坡顶 x 坐标
DEFAULT_X_TOE = 1000.0   # 默认坡脚 x 坐标
DEFAULT_H_VAL = 200.0     # 默认坡高 h

# 曲线样式映射（按频率升序循环使用）
CURVE_STYLES = [
    {'color': '#2ca02c', 'linestyle': '--', 'linewidth': 1.5},  # 第一条曲线: 绿色虚线
    {'color': '#1f77b4', 'linestyle': '-.', 'linewidth': 1.5},  # 第二条曲线: 蓝色点划线
    {'color': '#d62728', 'linestyle': '-', 'linewidth': 1.5},   # 第三条曲线: 红色实线
    {'color': '#9467bd', 'linestyle': ':', 'linewidth': 1.5},   # 第四条曲线: 紫色点线
    {'color': '#ff7f0e', 'linestyle': '-', 'linewidth': 1.5},   # 第五条曲线: 橙色实线
    {'color': '#8c564b', 'linestyle': '--', 'linewidth': 1.5},  # 第六条曲线: 棕色虚线
    {'color': '#e377c2', 'linestyle': '-.', 'linewidth': 1.5},  # 第七条曲线: 粉色点划线
    {'color': '#7f7f7f', 'linestyle': ':', 'linewidth': 1.5},   # 第八条曲线: 灰色点线
    {'color': '#17becf', 'linestyle': '-', 'linewidth': 1.5},   # 第九条曲线: 青色实线
    {'color': '#bcbd22', 'linestyle': '--', 'linewidth': 1.5},  # 第十条曲线: 黄绿色虚线
]

# ==============================================================================
#  中英文默认字体加载配置
# ==============================================================================
def build_font_properties():  # 定义加载系统字体的函数
    """
    加载并返回中英文默认字体属性。
    """
    cn_candidates = ['SimSun', 'NSimSun', 'Microsoft YaHei', 'SimHei']  # 候选的中文宋体/微软雅黑/黑体字体名列表
    en_candidates = ['Times New Roman', 'Times New Roman PS MT', 'Times']  # 候选的英文 Times New Roman 字体名列表
    cn_font = fm.FontProperties()  # 初始化默认的中文自定义字体属性实例
    en_font = fm.FontProperties()  # 初始化默认的英文自定义字体属性实例
    for name in cn_candidates:  # 遍历中文字体候选列表进行匹配加载
        try:  # 尝试载入当前名称的中文字体
            font_path = fm.findfont(name, fallback_to_default=False)  # 寻找系统内是否存在对应中文字体路径
            cn_font = fm.FontProperties(fname=font_path)  # 基于找到 of 路径实例化中文字体属性
            break  # 匹配成功则跳出中文加载循环
        except Exception:  # 处理加载失败的异常以继续尝试下一个候选字体
            continue  # 查找下一项候选字体
    for name in en_candidates:  # 遍历英文字体候选列表进行匹配加载
        try:  # 尝试载入当前名称的英文字体
            font_path = fm.findfont(name, fallback_to_default=False)  # 寻找系统内是否存在对应英文字体路径
            en_font = fm.FontProperties(fname=font_path)  # 基于找到的路径实例化英文字体属性
            break  # 匹配成功则跳出英文加载循环
        except Exception:  # 处理加载失败的异常以继续尝试下一个候选字体
            continue  # 查找下一项候选字体
    return cn_font, en_font  # 返回匹配完毕的中文字体属性与英文字体属性

CN_FONT, EN_FONT = build_font_properties()  # 加载并配置中英文默认字体属性

# ==============================================================================
#  辅助函数
# ==============================================================================
def get_incident_angle(cwd):  # 定义自适应获取入射角角度的函数
    """
    cwd: 当前工作目录路径
    """
    cae_files = sorted(glob.glob(os.path.join(cwd, '*.cae')))  # 检索并排序当前目录下的全部 CAE 数据库文件
    if cae_files:  # 判断是否找到了 CAE 文件
        cae_name = os.path.splitext(os.path.basename(cae_files[0]))[0]  # 提取第一个 CAE 文件的基础文件名
        match = re.search(r'a(-?\d+(?:\.\d+)?)$', cae_name)  # 正则匹配文件名尾部如 a30 等角度标志
        if match:  # 判断是否匹配到了角度数值
            return float(match.group(1))  # 返回提取并转换后的浮点型入射角度
    folder_name = os.path.basename(cwd)  # 提取当前工作文件夹的名称
    match_angle = re.search(r'angle(-?\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)  # 正则匹配文件夹名称中的 angle 字段
    if match_angle:  # 判断是否成功匹配 angle 关键字与数值
        return float(match_angle.group(1))  # 返回提取并转换后的浮点型角度值
    match_a = re.search(r'a(-?\d+(?:\.\d+)?)', folder_name, re.IGNORECASE)  # 正则匹配文件夹名称中的 a 角度字段
    if match_a:  # 判断是否匹配到 a 关键字
        return float(match_a.group(1))  # 返回解析后的浮点型角度值
    return 0.0  # 兜底返回默认值 0.0 度


def parse_wave_info(core_name):  # 解析文件名波形参数以供分组与排序的辅助函数
    """
    从文件名中提取基础波名（用于同一图表内分组绘制）和频率数值（用于排序）。
    """
    match_hz = re.search(r'[_-](?P<val>\d+(?:\.\d+)?)Hz', core_name, re.IGNORECASE)  # 正则匹配文件名中的频率字段
    if match_hz:  # 若提取成功
        freq = float(match_hz.group('val'))  # 提取频率值作为排序键
        base_name = core_name[:match_hz.start()].strip('_-')  # 提取不含频率的基础波名作为分组键
        return base_name, freq  # 返回分组波名与频率排序键
    return core_name, 0.0  # 兜底返回完整名称与默认排序键 0.0


def style_axes(ax):  # 定义设置图表坐标轴样式的函数
    """
    ax: 坐标轴对象实例
    """
    ax.set_facecolor('white')  # 设置背景颜色为白色
    ax.tick_params(direction='in', top=True, right=True, bottom=True, left=True, labelsize=11)  # 设置刻度线方向朝内且四面都显示，刻度标签字号为11
    for spine in ax.spines.values():  # 遍历获取坐标轴的四个边框线对象
        spine.set_color('black')  # 将各个边框线的颜色设置为黑色
        spine.set_linewidth(1.0)  # 设置边框线的粗细为1.0
    ax.grid(True, which='both', linestyle=':', color='#b0b0b0', linewidth=0.5)  # 开启点状网格线并设置网格线的颜色与宽度

# ==============================================================================
#  核心绘图函数
# ==============================================================================
def plot_taf_curves(taf_records, incident_angle, output_dir):  # 定义绘制不同频率 TAF 曲线对比图的主核心函数
    """
    taf_records: 包含所有波形数据的 TAF 记录列表
    incident_angle: 地震波入射角度
    output_dir: 绘图输出保存的目标目录
    """
    if not taf_records:  # 判断 TAF 记录列表是否为空
        print("无有效 TAF 数据记录，绘图终止。")  # 终端输出提示无数据
        return  # 退出函数
    df_records = pd.DataFrame(taf_records)  # 将 TAF 记录列表转换为 Pandas 的 DataFrame 结构进行数据管理
    grouped = df_records.groupby('base_motion')  # 按基础波形名称对数据表记录进行分组处理
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']  # 设置全局默认字体样式为 Times New Roman
    plt.rcParams['mathtext.fontset'] = 'stix'  # 设置 LaTeX 公式字体样式为 Stix
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号，避免负号渲染成乱码
    for motion_name, group_df in grouped:  # 遍历每个波形分组的数据内容
        print(f"\n>>> 正在为波形 [{motion_name}] 绘制对比图...")  # 输出当前绘图对应的波形名称
        sorted_group = group_df.sort_values(by='freq').reset_index(drop=True)  # 根据频率字段对分组记录进行升序排序并重置索引
        x_crest = sorted_group.loc[0, 'x_crest']  # 提取第一条曲线的坡顶绝对 x 坐标
        x_toe = sorted_group.loc[0, 'x_toe']  # 提取第一条曲线的坡脚绝对 x 坐标
        total_L = sorted_group.loc[0, 'total_L']  # 提取第一条曲线的坡面总长度
        for direction in ['horizontal', 'vertical']:  # 分别循环绘制水平向 (horizontal) 与 竖向 (vertical) 的 TAF 图表
            fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=300)  # 创建 5.2x4.0 英寸、分辨率 300dpi 的画布和坐标系
            style_axes(ax)  # 调用 style_axes 函数初始化坐标轴网格与边框外观
            for idx, row in sorted_group.iterrows():  # 遍历绘制当前方向下所有频率曲线
                style_cfg = CURVE_STYLES[idx % len(CURVE_STYLES)]  # 根据曲线索引循环获取预设样式配置
                y_data = row['taf_h'] if direction == 'horizontal' else row['taf_v']  # 根据水平向或竖向选择对应的 TAF 浮点数组
                ax.plot(row['x'], y_data, color=style_cfg['color'], linestyle=style_cfg['linestyle'], linewidth=style_cfg['linewidth'], label=row['legend'])  # 绘制对应曲线并指定配色、线型、宽度和图例名
            ax.set_xlim(0, total_L)  # 设置横坐标范围为 0 到坡面总长度
            tick_step = 600.0 if abs(total_L - 1800.0) < 50.0 else (total_L / 3.0)  # 若为 1800 米长坡则步长取 600 米，否则均分为三等份步长
            ax.set_xticks(np.arange(0, total_L + 1.0, tick_step))  # 生成并设置横坐标的主刻度点
            custom_lim = None  # 初始化自定义纵轴显示范围的限制变量
            if isinstance(CUSTOM_YLIM, tuple) and len(CUSTOM_YLIM) == 2:  # 判断全局配置是否为通用的二元元组
                custom_lim = CUSTOM_YLIM  # 直接采用该元组作为自定义纵轴范围配置
            elif isinstance(CUSTOM_YLIM, dict):  # 判断全局配置是否为字典类型
                custom_lim = CUSTOM_YLIM.get(direction)  # 根据当前方向从字典中获取对应的范围元组
            if isinstance(TARGET_INTERVALS, dict):  # 判断刻度区间数量是否为字典配置方式
                intervals = TARGET_INTERVALS.get(direction, 5)  # 获取当前绘制方向对应的刻度区间数量，默认值为5
            else:  # 处理非字典配置（即单个整数）的情况
                intervals = TARGET_INTERVALS  # 直接使用统一配置的刻度区间数量
            if custom_lim is not None:  # 若提取到了有效的自定义轴显示范围配置
                ylim_min, ylim_max = custom_lim  # 解包自定义的纵轴下限与上限
                y_span = ylim_max - ylim_min  # 计算纵轴显示数值的跨度大小
                y_step = y_span / intervals if y_span > 1e-6 else 1.0  # 使用当前方向区间数计算坐标刻度间距
            else:  # 处理未指定自定义纵轴范围的自适应计算逻辑
                all_y_values = np.concatenate([r['taf_h'] if direction == 'horizontal' else r['taf_v'] for _, r in sorted_group.iterrows()])  # 拼接所有绘制曲线的纵轴坐标点数据以求取极值
                min_y = all_y_values.min()  # 计算所有曲线数据点的最小值
                max_y = all_y_values.max()  # 计算所有曲线数据点的最大值
                y_span = max_y - min_y  # 计算数据的实际分布跨度范围
                raw_step = y_span / intervals if y_span > 1e-6 else 1.0  # 使用自适应刻度区间计算未圆整的基准步长
                magnitude = 10.0 ** np.floor(np.log10(raw_step))  # 计算步长的基数数量级
                nice_steps = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0]  # 声明常用美观的步长缩放因子列表
                y_step = magnitude * min(nice_steps, key=lambda s: abs(s * magnitude - raw_step))  # 精准匹配并计算最整洁的刻度步长值
                ylim_min = max(0.0, np.floor(min_y / y_step) * y_step)  # 计算自适应的纵轴刻度下限并确保非负
                ylim_max_needed = ylim_min + (max_y - ylim_min) * (5.0 / 4.0)  # 估算所需的纵轴上限空间以确保顶部留有1/5空白给图例
                ylim_max = np.ceil(ylim_max_needed / y_step) * y_step  # 对预估上限按步长进行向上圆整得到最终上限值
                if ylim_max <= max_y:  # 判断圆整后的上限是否小于等于最大数据值以应对浮点误差
                    ylim_max += y_step  # 若未完全包含最大数据值则额外递增一个步长高度
            ax.set_ylim(ylim_min, ylim_max)  # 设置纵轴显示界限
            ax.set_yticks(np.arange(ylim_min, ylim_max + 1e-9, y_step))  # 生成并设置纵坐标刻度标签的具体位置
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))  # 设置纵坐标标签的浮点数格式化为保留1位小数
            ax.axvline(x=x_crest, color='black', linestyle='--', linewidth=1.0)  # 绘制坡顶位置的黑色竖向虚线
            ax.axvline(x=x_toe, color='black', linestyle='--', linewidth=1.0)  # 绘制坡脚位置的黑色竖向虚线
            text_y = ylim_min + 0.92 * (ylim_max - ylim_min)  # 计算坡顶和坡脚编号文本的纵坐标位置
            offset_x = 0.02 * total_L  # 计算编号标注文本与虚线的水平偏移距离
            ax.text(x_crest - offset_x, text_y, '#1', fontsize=11, fontproperties=EN_FONT, va='top', ha='right')  # 在坡顶虚线左侧标注井号一
            ax.text(x_toe - offset_x, text_y, '#2', fontsize=11, fontproperties=EN_FONT, va='top', ha='right')  # 在坡脚虚线左侧标注井号二
            ax.set_xlabel('地表测点位置 (m)', fontsize=13, fontproperties=CN_FONT)  # 设定横轴说明文本为中文“地表测点位置 (m)”
            dir_label = '水平向' if direction == 'horizontal' else '竖向'  # 依据绘制分量设定标签方向的中文文本
            ax.set_ylabel(f'{dir_label} TAF', fontsize=13, fontproperties=CN_FONT)  # 设定纵轴说明文本格式为“分量向 TAF”
            ax.set_title(r'入射角 $\theta_s = %g^\circ$' % incident_angle, fontsize=14, fontproperties=CN_FONT, pad=10)  # 设定图表标题包含入射角角度的 LaTeX 数学表达式
            ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='black', framealpha=1.0, prop=EN_FONT, fontsize=10.5)  # 设置图例在左上角显示并设置背景、黑框和英文字体
            plt.tight_layout()  # 调整子图布局以防止坐标轴标签和图表被遮挡裁剪
            out_img_name = f"TAF-comparison-{motion_name}-{direction}.png"  # 构建输出的 PNG 图像文件名
            out_img_path = os.path.join(output_dir, out_img_name)  # 拼装包含保存目录的完整图像输出绝对路径
            plt.savefig(out_img_path, dpi=300, bbox_inches='tight')  # 保存当前绘制的图像并去除多余白边
            plt.close()  # 销毁当前图表对象释放内存
            print(f"  已成功输出对比图表: {out_img_name}")  # 终端输出保存成功的信息

# ==============================================================================
#  主处理逻辑
# ==============================================================================
def main():  # 定义主运行流程控制函数
    print(">>> 启动 TAF.csv 绘图程序 (Plot_Multi_TAF)...")  # 控制台输出程序启动信息
    taf_files = sorted(glob.glob(os.path.join(SCRIPT_DIR, TAF_GLOB_PATTERN)))  # 扫描并排序当前目录下的全部 TAF-*.csv 数据文件
    if not taf_files:  # 如果未找到任何符合文件匹配规则的数据文件
        print(f"错误: 目录 {SCRIPT_DIR} 下未找到匹配 {TAF_GLOB_PATTERN} 的 TAF 数据文件。")  # 输出错误信息指明未找到数据
        return  # 提前终止主函数的执行
    print(f"共发现 {len(taf_files)} 个 TAF 数据文件。")  # 控制台输出查找到的数据文件总数量
    incident_angle = get_incident_angle(SCRIPT_DIR)  # 自适应获取当前工作目录对应的地震波入射角度
    print(f"识别到入射角度 theta_s = {incident_angle}°")  # 输出识别到的入射角数值信息
    taf_records = []  # 初始化用于存放解析出所有 TAF 记录的数据列表
    for filepath in taf_files:  # 遍历读取到的每一个 TAF 数据文件路径
        filename = os.path.basename(filepath)  # 截取提取出文件名
        stem = os.path.splitext(filename)[0]  # 提取出文件名主体名称
        if stem.startswith('TAF-'):  # 判断文件主体是否以 TAF- 作为前缀
            base_key = stem[4:]  # 截取获取去掉前缀后的真实波形标识字符串
        else:  # 处理不含 TAF- 前缀的文件名
            base_key = stem  # 将整个文件名主体用作波形标识
        print(f"正在载入: {filename}...")  # 输出提示正在载入的文件名称
        try:  # 尝试读取 CSV 文件内容
            taf_df = pd.read_csv(filepath)  # 使用 Pandas 读取当前路径 of CSV 格式数据文件
            required_cols = {'x', 'TAF_h', 'TAF_v'}  # 定义文件必须包含的核心数据列名集合
            missing = required_cols - set(taf_df.columns)  # 计算当前数据表缺失的关键列集合
            if missing:  # 若发现确实存在缺失的列名
                print(f"  警告: 文件 {filename} 缺失关键列 {sorted(missing)}，跳过该文件。")  # 控制台打印警告信息
                continue  # 跳过当前文件的后续处理并进入下一个文件
            slope_x = taf_df['x'].to_numpy()  # 获取 x 列数据并转换为 NumPy 浮点型一维数组
            taf_h = taf_df['TAF_h'].to_numpy()  # 获取水平 TAF 列数据并转换为一维数组
            taf_v = taf_df['TAF_v'].to_numpy()  # 获取垂直 TAF 列数据并转换为一维数组
            total_L = slope_x.max()  # 提取出 x 绝对坐标的最大值作为坡面总长度
            x_crest, x_toe, h_val = None, None, None  # 初始化几何参数的局部变量
            pga_slope_pattern = os.path.join(SCRIPT_DIR, f"PGA-{base_key}-slope.csv")  # 构造对应的 PGA 斜坡数据表匹配模式
            pga_slope_files = glob.glob(pga_slope_pattern)  # 全局检索当前目录下对应的 PGA-*-slope.csv 几何表
            if not pga_slope_files:  # 若没找到特定波名对应的几何辅助数据文件
                pga_slope_files = glob.glob(os.path.join(SCRIPT_DIR, "PGA-*-slope.csv"))  # 尝试模糊检索目录下任意 PGA-*-slope.csv
            if pga_slope_files:  # 如果找到了坡度几何辅助文件
                try:  # 尝试载入并解析几何高度及关键控制点信息
                    slope_filepath = pga_slope_files[0]  # 取第一个搜索到的几何参数路径
                    slope_df = pd.read_csv(slope_filepath)  # 使用 Pandas 读取该坡度坐标数据
                    if 'x' in slope_df.columns and 'y' in slope_df.columns:  # 判断是否具有必须的 x 与 y 列
                        y_max = slope_df['y'].max()  # 获取 y 坐标的最大值作为坡顶位置高度
                        y_min = slope_df['y'].min()  # 获取 y 坐标的最小值作为坡底基准高度
                        h_val = y_max - y_min  # 计算两者差值得到坡高 h
                        x_crest = slope_df[slope_df['y'] >= y_max - 1e-3]['x'].max()  # 提取坡顶区域的最大 x 坐标点
                        x_toe = slope_df[slope_df['y'] <= y_min + 1e-3]['x'].min()  # 提取坡底起坡处的最小 x 坐标点
                        print(f"  已通过 {os.path.basename(slope_filepath)} 识别几何参数: h={h_val:.2f}m, 坡顶x={x_crest:.2f}m, 坡脚x={x_toe:.2f}m")  # 输出解析得到的几何值
                except Exception as ex:  # 捕获几何辅助文件读取过程中的异常
                    print(f"  读取 PGA 辅助文件获取几何信息失败: {ex}")  # 控制台打印读取异常提示
            if x_crest is None or x_toe is None or h_val is None:  # 判断几何控制点参数是否未能成功加载
                x_crest = DEFAULT_X_CREST  # 采用缺省兜底的坡顶 x 坐标
                x_toe = DEFAULT_X_TOE  # 采用缺省兜底的坡脚 x 坐标
                h_val = DEFAULT_H_VAL  # 采用缺省兜底的坡高值
                print(f"  未找到 PGA 原始文件，将使用预设几何参数: h={h_val:.2f}m, 坡顶x={x_crest:.2f}m, 坡脚x={x_toe:.2f}m")  # 输出警告并使用默认预设值
            base_motion, freq_val = parse_wave_info(base_key)  # 解析文件名提取得到波形分组基础名及数值频率
            legend_label = base_key  # 直接采用波形唯一关键字作为图例的展示名
            taf_records.append({  # 将提取重构的数据追加到 records 列表中
                'base_motion': base_motion.lower(),  # 将波形基础分组名称转换为小写字母
                'freq': freq_val,  # 当前文件的频率数值，用于后期曲线绘制排序
                'legend': legend_label,  # 绘图图例中显示的当前曲线标注文本
                'x': slope_x,  # 地表测点对应的绝对 x 坐标数据点
                'taf_h': taf_h,  # 水平方向的地表地形放大因子数组
                'taf_v': taf_v,  # 垂直方向的地表地形放大因子数组
                'x_crest': x_crest,  # 解析或默认得到的坡顶特征点 x 坐标
                'x_toe': x_toe,  # 解析或默认得到的坡脚特征点 x 坐标
                'total_L': total_L  # 数据集范围内的最大长度值
            })
        except Exception as e:  # 捕获对 TAF CSV 文件处理过程中发生的未知异常
            print(f"  处理 TAF 文件 {filename} 时出错: {e}")  # 终端输出具体的出错信息
    plot_taf_curves(taf_records, incident_angle, SCRIPT_DIR)  # 调用核心绘图对比函数执行输出
    print("\n>>> TAF.csv 绘图流程全部完成。")  # 终端提示整体流程绘图完毕

if __name__ == '__main__':  # 判断当前脚本是否作为主程序运行
    main()  # 调用 main 函数执行主处理流程
