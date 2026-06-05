import os  # 导入操作系统相关工具
import random  # 导入随机数模块
import json  # 导入JSON读写模块
import logging  # 导入日志模块
import numpy as np  # 导入数值计算库
import torch  # 导入PyTorch主库

def set_seed(seed=42):  # 定义固定随机种子的函数
    random.seed(seed)  # 固定Python随机种子
    np.random.seed(seed)  # 固定NumPy随机种子
    torch.manual_seed(seed)  # 固定PyTorch CPU随机种子
    torch.cuda.manual_seed(seed)  # 固定PyTorch当前GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 固定PyTorch所有GPU随机种子
    torch.backends.cudnn.deterministic = True  # 启用确定性卷积结果
    torch.backends.cudnn.benchmark = False  # 关闭卷积算法自动搜索

def setup_logger(log_file):  # 定义日志器初始化函数
    logger = logging.getLogger()  # 获取根日志器
    logger.setLevel(logging.INFO)  # 设置日志级别为INFO
    if logger.hasHandlers():  # 判断是否已有处理器
        logger.handlers.clear()  # 清空已有处理器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 定义日志格式
    ch = logging.StreamHandler()  # 创建控制台处理器
    ch.setFormatter(formatter)  # 为控制台处理器设置格式
    logger.addHandler(ch)  # 添加控制台处理器到日志器
    fh = logging.FileHandler(log_file, encoding='utf-8')  # 创建文件处理器
    fh.setFormatter(formatter)  # 为文件处理器设置格式
    logger.addHandler(fh)  # 添加文件处理器到日志器
    return logger  # 返回配置完成的日志器

class DataScaler:  # 定义数据归一化器类
    def __init__(self):  # 定义初始化方法
        self.stats = {}  # 初始化统计量字典

    def fit(self, waves, conditions, targets):  # 定义拟合统计量方法
        """
        waves: 形状为 (L,) 的波形列表或形状为 (N, L) 的补齐数组
        conditions: 形状为 (N, 3) 的条件数组（h、i、angle）
        targets: 形状为 (N, 4, 161) 的目标数组（PGA_h、PGA_v、TAF_h、TAF_v）
        """
        flat_waves = np.concatenate([w.flatten() for w in waves])  # 拼接所有波形为一维数组
        self.stats['wave_mean'] = float(np.mean(flat_waves))  # 计算波形均值并保存
        self.stats['wave_std'] = float(np.std(flat_waves))  # 计算波形标准差并保存
        if self.stats['wave_std'] == 0:  # 判断波形标准差是否为零
            self.stats['wave_std'] = 1.0  # 将零标准差修正为1
        self.stats['cond_min'] = conditions.min(axis=0).tolist()  # 计算条件最小值并保存
        self.stats['cond_max'] = conditions.max(axis=0).tolist()  # 计算条件最大值并保存
        self.stats['target_mean'] = targets.mean(axis=(0, 2)).tolist()  # 计算各目标通道均值并保存
        self.stats['target_std'] = targets.std(axis=(0, 2)).tolist()  # 计算各目标通道标准差并保存
        for idx in range(len(self.stats['target_std'])):  # 遍历每个目标通道的标准差
            if self.stats['target_std'][idx] == 0:  # 判断通道标准差是否为零
                self.stats['target_std'][idx] = 1.0  # 将零标准差修正为1

    def transform_waves(self, waves):  # 定义波形归一化函数
        """
        waves: 形状为 (L,) 的 np.ndarray 列表，或形状为 (N, L) 的补齐数组
        """
        mean = self.stats['wave_mean']  # 取出波形均值
        std = self.stats['wave_std']  # 取出波形标准差
        if isinstance(waves, list):  # 判断输入是否为列表
            return [(w - mean) / std for w in waves]  # 对列表中的每个波形逐个标准化
        return (waves - mean) / std  # 对数组形式波形直接标准化

    def transform_conditions(self, conditions):  # 定义条件归一化函数
        """
        conditions: 形状为 (N, 3) 的条件数组
        """
        c_min = np.array(self.stats['cond_min'])  # 取出条件最小值并转为数组
        c_max = np.array(self.stats['cond_max'])  # 取出条件最大值并转为数组
        diff = c_max - c_min  # 计算条件范围
        diff[diff == 0] = 1.0  # 将零范围修正为1
        return (conditions - c_min) / diff  # 执行Min-Max归一化

    def transform_targets(self, targets):  # 定义目标归一化函数
        """
        targets: 形状为 (N, 4, 161) 或 (4, 161) 的目标数组
        """
        mean = np.array(self.stats['target_mean'])  # 取出目标均值并转为数组
        std = np.array(self.stats['target_std'])  # 取出目标标准差并转为数组
        if len(targets.shape) == 3:  # 判断输入是否为批量三维数组
            mean = mean[np.newaxis, :, np.newaxis]  # 扩展均值维度以便广播
            std = std[np.newaxis, :, np.newaxis]  # 扩展标准差维度以便广播
        else:  # 处理单样本二维数组
            mean = mean[:, np.newaxis]  # 扩展均值维度以便广播
            std = std[:, np.newaxis]  # 扩展标准差维度以便广播
        return (targets - mean) / std  # 执行Z-score标准化

    def inverse_transform_targets(self, normalized_targets):  # 定义目标反归一化函数
        """
        normalized_targets： 为numpy数组或PyTorch张量，形状为(N, 4, 161) 或 (4, 161)
        """
        mean = np.array(self.stats['target_mean'])  # 取出目标均值并转为数组
        std = np.array(self.stats['target_std'])  # 取出目标标准差并转为数组
        is_tensor = False  # 标记输入是否为张量
        device = None  # 记录张量所在设备
        if isinstance(normalized_targets, torch.Tensor):  # 判断输入是否为张量
            is_tensor = True  # 标记输入为张量
            device = normalized_targets.device  # 记录张量设备
            normalized_targets = normalized_targets.detach().cpu().numpy()  # 转为CPU上的NumPy数组
        if len(normalized_targets.shape) == 3:  # 判断输入是否为批量三维数组
            mean = mean[np.newaxis, :, np.newaxis]  # 扩展均值维度以便广播
            std = std[np.newaxis, :, np.newaxis]  # 扩展标准差维度以便广播
        else:  # 处理单样本二维数组
            mean = mean[:, np.newaxis]  # 扩展均值维度以便广播
            std = std[:, np.newaxis]  # 扩展标准差维度以便广播
        unnormalized = normalized_targets * std + mean  # 执行反标准化
        if is_tensor:  # 判断是否需要返回张量
            return torch.from_numpy(unnormalized).to(device)  # 转回张量并放回原设备
        return unnormalized  # 直接返回NumPy数组

    def save(self, filepath):  # 定义统计量保存函数
        with open(filepath, 'w') as f:  # 以写入方式打开文件
            json.dump(self.stats, f, indent=4)  # 将统计量写入JSON文件

    def load(self, filepath):  # 定义统计量加载函数
        with open(filepath, 'r') as f:  # 以读取方式打开文件
            self.stats = json.load(f)  # 从JSON文件恢复统计量
