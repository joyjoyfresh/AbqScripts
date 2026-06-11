import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
from models.base_model import BaseModel  # 导入基础模型封装类
from models.decoder import CurveDecoder  # 导入曲线解码器


class CNNEncoder(nn.Module):  # 定义CNN编码器模块
    def __init__(self, wave_emb_dim=128, dropout=0.3):  # 初始化编码器并设置嵌入维度与丢弃率
        super(CNNEncoder, self).__init__()  # 调用父类初始化方法
        self.conv = nn.Sequential(  # 定义卷积特征提取模块
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),  # 第一层卷积，通道从1变为16并下采样一半
            nn.BatchNorm1d(16),  # 对16个通道做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3),  # 第二层卷积，通道从16变为32并继续下采样
            nn.BatchNorm1d(32),  # 对32个通道做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),  # 第三层卷积，通道从32变为64并继续下采样
            nn.BatchNorm1d(64),  # 对64个通道做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),  # 第四层卷积，通道从64变为128并继续下采样
            nn.BatchNorm1d(128),  # 对128个通道做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.Conv1d(128, 128, kernel_size=7, stride=2, padding=3),  # 第五层卷积，保持128通道并继续下采样
            nn.BatchNorm1d(128),  # 对128个通道做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.AdaptiveAvgPool1d(1)  # 使用全局平均池化压缩时间维度到1
        )  # 结束卷积特征提取模块定义
        self.fc = nn.Sequential(  # 定义全连接映射模块
            nn.Linear(128, wave_emb_dim),  # 将128维特征映射到波形嵌入维度
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout)  # 按给定比例随机失活以防过拟合
        )  # 结束全连接映射模块定义

    def forward(self, wave):  # 定义前向传播过程
        out = self.conv(wave)  # 对输入波形提取卷积特征并得到池化结果
        out = out.squeeze(-1)  # 去掉最后一个长度为1的维度
        wave_emb = self.fc(out)  # 将特征映射为波形嵌入向量
        return wave_emb  # 返回波形嵌入结果


class CNNModel(BaseModel):  # 定义基于CNN编码器的完整模型
    def __init__(self, wave_emb_dim=128, num_points=161, num_channels=4, dropout=0.3):  # 初始化模型参数
        super(CNNModel, self).__init__()  # 调用父类初始化方法
        self.encoder = CNNEncoder(wave_emb_dim=wave_emb_dim, dropout=dropout)  # 创建CNN编码器实例
        self.decoder = CurveDecoder(wave_emb_dim=wave_emb_dim, num_points=num_points, num_channels=num_channels, dropout=dropout * 0.67)  # 创建曲线解码器并设置较低的dropout

    def forward(self, wave, cond, length=None):  # 定义前向传播过程
        wave_emb = self.encoder(wave)  # 将输入波形编码为嵌入向量
        out = self.decoder(wave_emb, cond)  # 将波形嵌入与条件信息输入解码器
        return out  # 返回模型预测结果
