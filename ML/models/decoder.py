import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块


class CurveDecoder(nn.Module):  # 定义曲线解码器模块
    def __init__(self, wave_emb_dim, num_points=161, num_channels=4, dropout=0.2):  # 初始化解码器并设置参数
        super(CurveDecoder, self).__init__()  # 调用父类初始化方法
        self.num_points = num_points  # 保存输出曲线的点数
        self.num_channels = num_channels  # 保存输出曲线的通道数
        in_dim = wave_emb_dim + 3  # 计算输入维度，包含波形嵌入和3个条件量
        self.mlp = nn.Sequential(  # 定义多层感知机解码器
            nn.Linear(in_dim, 128),  # 将输入特征映射到128维
            nn.BatchNorm1d(128),  # 对128维特征做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.Linear(128, 256),  # 将特征进一步映射到256维
            nn.BatchNorm1d(256),  # 对256维特征做批归一化
            nn.ReLU(),  # 使用ReLU激活函数增加非线性
            nn.Dropout(dropout),  # 按给定比例随机失活以防过拟合
            nn.Linear(256, num_channels * num_points)  # 输出展平后的所有通道和点数
        )  # 结束多层感知机定义

    def forward(self, wave_emb, cond):  # 定义前向传播过程
        x = torch.cat([wave_emb, cond], dim=1)  # 在特征维拼接波形嵌入和条件向量
        out = self.mlp(x)  # 通过多层感知机得到展平输出
        out = out.view(-1, self.num_channels, self.num_points)  # 重塑为批量、通道、点数的三维张量
        return out  # 返回曲线预测结果
