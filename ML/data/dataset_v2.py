import os  # 导入操作系统工具
import re  # 导入正则表达式工具
import numpy as np  # 导入数值计算库
import pandas as pd  # 导入表格处理库
import torch  # 导入PyTorch主库
from torch.utils.data import Dataset  # 导入数据集基类

# 设置正则表达式，用于匹配类似 fuke-ALL-h100_i15_angle0 的文件夹名
FOLDER_PATTERN = re.compile(r"fuke-ALL-h([\d\.]+)_i([\d\.]+)_angle([\d\.]+)")  # 定义样本文件夹名匹配规则

def load_wave_file(filepath):  # 定义波形文件读取函数
    """
    从空格或制表符分隔的文本中读取波形文件。
    返回：
        t: 时间序列（np.ndarray）
        accel: 加速度序列（np.ndarray）
    """
    data = np.loadtxt(filepath)  # 读取文本波形数据
    t = data[:, 0]  # 提取时间列
    accel = data[:, 1]  # 提取加速度列
    return t, accel  # 返回时间和加速度

def resample_wave(t, accel, target_sf=100):  # 定义波形重采样函数
    """
    将波形重采样到目标采样频率（100Hz 对应 dt = 0.01s）。
    """
    dt = 1.0 / target_sf  # 计算目标采样间隔
    t_max = t[-1]  # 取出原始时间终点
    t_new = np.arange(0, t_max, dt)  # 生成新时间网格
    accel_new = np.interp(t_new, t, accel)  # 对加速度插值重采样
    return accel_new  # 返回重采样后的波形

def pad_wave(accel, max_len=4000):  # 定义波形补齐函数
    """
    将波形补齐到 max_len，并返回补齐后的波形和实际长度。
    """
    actual_len = len(accel)  # 计算原始长度
    if actual_len >= max_len:  # 判断是否需要截断
        return accel[:max_len], max_len  # 超长则截断并返回最大长度
    else:  # 处理不足长度的情况
        padded = np.zeros(max_len)  # 创建零填充数组
        padded[:actual_len] = accel  # 将原始波形拷贝到前部
        return padded, actual_len  # 返回补齐后的波形和实际长度

def load_raw_dataset(data_dir, target_sf=100, max_len=4000, num_points=161):  # 定义原始数据集加载函数
    """
    扫描 data_dir，解析样本，完成波形重采样和补齐，将曲线插值到统一网格，
    计算 TAF_v，并返回波形、条件、目标和样本元数据列表。
    """
    waves_list = []  # 初始化波形列表
    conds_list = []  # 初始化条件列表
    targets_list = []  # 初始化目标列表
    meta_list = []  # 初始化元数据列表
    grid_x = np.linspace(0.0, 8.0, num_points)  # 生成统一的x/h网格
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]  # 获取所有子文件夹
    for folder in folders:  # 遍历每个文件夹
        match = FOLDER_PATTERN.match(folder)  # 匹配文件夹名中的几何参数
        if not match:  # 判断是否匹配成功
            continue  # 不符合命名规则则跳过
        h = float(match.group(1))  # 解析坡高h
        i = float(match.group(2))  # 解析坡角i
        angle = float(match.group(3))  # 解析入射角angle
        folder_path = os.path.join(data_dir, folder)  # 拼接当前文件夹路径
        for wave_name in ["El_Centro", "Loma_Prieta", "Northridge"]:  # 遍历三种地震波
            wave_file = os.path.join(folder_path, f"{wave_name}_scaled.txt")  # 拼接波形文件路径
            if not os.path.exists(wave_file):  # 判断波形文件是否存在
                print(f"Warning: wave file {wave_file} not found. Skipping sample.")  # 打印缺失提示
                continue  # 缺失则跳过当前样本
            t, accel = load_wave_file(wave_file)  # 读取波形时间和加速度
            dt_approx = np.mean(np.diff(t))  # 计算原始平均采样间隔
            sf_approx = 1.0 / dt_approx  # 计算原始近似采样率
            if abs(sf_approx - target_sf) > 5.0:  # 判断是否需要重采样
                accel_resampled = resample_wave(t, accel, target_sf=target_sf)  # 执行重采样
            else:  # 采样率足够接近目标值
                accel_resampled = accel  # 直接使用原始波形
            accel_padded, actual_len = pad_wave(accel_resampled, max_len=max_len)  # 对波形补齐或截断
            pga_slope_file = os.path.join(folder_path, f"PGA-{wave_name}_scaled-slope.csv")  # 拼接坡面PGA文件路径
            if not os.path.exists(pga_slope_file):  # 判断坡面PGA文件是否存在
                print(f"Warning: PGA slope file {pga_slope_file} not found. Skipping sample.")  # 打印缺失提示
                continue  # 缺失则跳过当前样本
            df_slope = pd.read_csv(pga_slope_file)  # 读取坡面PGA表格
            df_slope = df_slope.sort_values(by="x/h")  # 按x/h排序
            slope_xh = df_slope["x/h"].values  # 提取坡面x/h列
            pga_h = df_slope["PGA_h"].values  # 提取坡面水平PGA列
            pga_v = df_slope["PGA_v"].values  # 提取坡面竖向PGA列
            pga_h_grid = np.interp(grid_x, slope_xh, pga_h)  # 将PGA_h插值到统一网格
            pga_v_grid = np.interp(grid_x, slope_xh, pga_v)  # 将PGA_v插值到统一网格
            taf_file = os.path.join(folder_path, f"TAF-{wave_name}.csv")  # 拼接TAF文件路径
            if not os.path.exists(taf_file):  # 判断TAF文件是否存在
                print(f"Warning: TAF file {taf_file} not found. Skipping sample.")  # 打印缺失提示
                continue  # 缺失则跳过当前样本
            df_taf = pd.read_csv(taf_file)  # 读取TAF表格
            df_taf = df_taf.sort_values(by="x/h")  # 按x/h排序
            taf_xh = df_taf["x/h"].values  # 提取TAF表格的x/h列
            taf_h = df_taf["TAF_h"].values  # 提取TAF_h列
            taf_h_grid = np.interp(grid_x, taf_xh, taf_h)  # 将TAF_h插值到统一网格
            pga_flat_file = os.path.join(folder_path, f"PGA-{wave_name}_scaled-flat.csv")  # 拼接平坦地表PGA文件路径
            if not os.path.exists(pga_flat_file):  # 判断平坦地表PGA文件是否存在
                print(f"Warning: PGA flat file {pga_flat_file} not found. Skipping sample.")  # 打印缺失提示
                continue  # 缺失则跳过当前样本
            df_flat = pd.read_csv(pga_flat_file)  # 读取平坦地表PGA表格
            df_flat = df_flat.sort_values(by="x/h")  # 按x/h排序
            flat_xh = df_flat["x/h"].values  # 提取平坦地表x/h列
            flat_pga_v = df_flat["PGA_v"].values  # 提取平坦地表竖向PGA列
            flat_pga_v_grid = np.interp(grid_x, flat_xh, flat_pga_v)  # 将平坦地表PGA_v插值到统一网格
            flat_pga_v_grid_safe = np.clip(flat_pga_v_grid, a_min=1e-6, a_max=None)  # 避免平坦地表值为零
            taf_v_grid = pga_v_grid / flat_pga_v_grid_safe  # 计算TAF_v
            taf_v_grid = np.clip(taf_v_grid, 0.0, 10.0)  # 裁剪TAF_v到合理范围
            target_curves = np.stack([pga_h_grid, pga_v_grid, taf_h_grid, taf_v_grid], axis=0)  # 堆叠四条目标曲线
            waves_list.append(accel_padded)  # 保存波形样本
            conds_list.append(np.array([h, i, angle]))  # 保存条件样本
            targets_list.append(target_curves)  # 保存目标样本
            meta_list.append({  # 保存元数据字典
                'case': folder,  # 记录案例文件夹名
                'wave': wave_name,  # 记录波形名称
                'actual_len': actual_len  # 记录实际长度
            })  # 结束元数据字典
    return (  # 返回整理好的数据集
        np.array(waves_list, dtype=np.float32),  # 返回波形数组
        np.array(conds_list, dtype=np.float32),  # 返回条件数组
        np.array(targets_list, dtype=np.float32),  # 返回目标数组
        meta_list  # 返回元数据列表
    )  # 结束返回元组

class SeisMLDataset(Dataset):  # 定义SeisML数据集类
    def __init__(self, waves, conditions, targets, meta, scaler, fit_scaler=False):  # 定义初始化方法
        """
        waves: 形状为 (N, MAX_TIME_STEPS) 的 np.ndarray 
        conditions: 形状为 (N, 3) 的 np.ndarray
        targets: 形状为 (N, 4, 161) 的 np.ndarray
        meta: 包含样本元数据的字典列表
        scaler: DataScaler 实例
        fit_scaler: 是否在当前数据上拟合归一化器
        """
        self.meta = meta  # 保存元数据
        self.scaler = scaler  # 保存归一化器引用
        if fit_scaler:  # 判断是否需要拟合归一化器
            self.scaler.fit(waves, conditions, targets)  # 在当前数据上拟合归一化器
        self.waves = self.scaler.transform_waves(waves)  # 对波形进行标准化
        self.conditions = self.scaler.transform_conditions(conditions)  # 对条件进行归一化
        self.targets = self.scaler.transform_targets(targets)  # 对目标进行标准化

    def __len__(self):  # 定义数据集长度方法
        return len(self.meta)  # 返回样本数量

    def __getitem__(self, idx):  # 定义索引取样方法
        wave = torch.tensor(self.waves[idx], dtype=torch.float32).unsqueeze(0)  # 将波形转为张量并增加通道维
        cond = torch.tensor(self.conditions[idx], dtype=torch.float32)  # 将条件转为张量
        length = torch.tensor(self.meta[idx]['actual_len'], dtype=torch.long)  # 将实际长度转为长整型张量
        target = torch.tensor(self.targets[idx], dtype=torch.float32)  # 将目标转为张量
        return wave, cond, length, target  # 返回单个样本的四元组
