import os  # 导入操作系统相关功能模块
import torch  # 导入PyTorch深度学习框架

# 路径配置
DATA_DIR = r"E:\Abaqus\fuke-ALL"  # 数据集根目录
OUTPUT_DIR = r"c:\Users\12462\Documents\Code\AbqScripts\ML\outputs"  # 输出结果目录
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 若输出目录不存在则自动创建

# 数据预处理配置
WAVES = ["El_Centro", "Loma_Prieta", "Northridge"]  # 训练使用的地震波名称
TARGET_SF = 100  # 目标采样率为100Hz
MAX_TIME_STEPS = 4000  # 将波形补齐到4000个时间步
NUM_CURVE_POINTS = 161  # 统一的x/h采样点数量，范围为0.0到8.0，步长为0.05

# 特征与目标
COND_FEATURES = ["h", "i", "angle"]  # 条件输入特征
TARGET_CHANNELS = ["PGA_h", "PGA_v", "TAF_h", "TAF_v"]  # 输出目标通道
NUM_CHANNELS = len(TARGET_CHANNELS)  # 目标通道总数

# 训练超参数
TRAIN_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 优先使用GPU，否则使用CPU
    'optimizer': 'AdamW',  # 优化器类型
    'learning_rate': 1e-3,  # 初始学习率
    'lr_scheduler': 'ReduceLROnPlateau',  # 学习率调度器，通常用于验证集指标停滞时降学习率
    'batch_size': 16,  # 批大小
    'max_epochs': 500,  # 最大训练轮数
    'early_stopping_patience': 50,  # 早停等待轮数
    'weight_decay': 1e-4,  # 权重衰减系数
    'grad_clip': 1.0,  # 梯度裁剪阈值
    'random_seed': 42,  # 随机种子
    'num_folds': 5,  # 交叉验证折数
}

# 模型配置
MODEL_CONFIGS = {
    'cnn': {
        'wave_emb_dim': 128,  # 波形嵌入维度
        'dropout': 0.3,  # 丢弃率
    },
    'lstm': {
        'hidden_dim': 64,  # LSTM隐藏层维度
        'num_layers': 2,  # LSTM层数
        'wave_emb_dim': 128,  # 波形嵌入维度
        'dropout': 0.3,  # 丢弃率
    },
    'transformer': {
        'd_model': 64,  # Transformer模型维度
        'nhead': 4,  # 多头注意力头数
        'num_layers': 2,  # Transformer层数
        'dim_feedforward': 256,  # 前馈网络维度
        'wave_emb_dim': 128,  # 波形嵌入维度
        'dropout': 0.1,  # 丢弃率
    },
    'deeponet': {
        'wave_emb_dim': 128,  # 波形嵌入维度
        'trunk_hidden_dims': [64, 128, 128],  # Trunk网络隐藏层维度
    }
}
