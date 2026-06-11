import math  # 导入数学计算模块
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
from models.base_model import BaseModel  # 导入基础模型类
from models.decoder import CurveDecoder  # 导入曲线解码器

class SinusoidalPositionalEncoding(nn.Module):  # 定义正弦位置编码模块
    def __init__(self, d_model, max_len=4000):  # 定义初始化方法
        super(SinusoidalPositionalEncoding, self).__init__()  # 初始化父类模块
        pe = torch.zeros(max_len, d_model)  # 创建位置编码缓存张量
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # 生成位置索引列向量
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # 生成频率缩放项
        pe[:, 0::2] = torch.sin(position * div_term)  # 将偶数维填充为正弦编码
        pe[:, 1::2] = torch.cos(position * div_term)  # 将奇数维填充为余弦编码
        pe = pe.unsqueeze(0)  # 扩展为批量维度，形状为 (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 将位置编码注册为缓冲区

    def forward(self, x):  # 定义前向传播方法
        return x + self.pe[:, :x.size(1), :]  # 将位置编码加到输入特征上

class TransformerEncoderModule(nn.Module):  # 定义 Transformer 编码模块
    def __init__(self, wave_emb_dim=128, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, max_len=4000):  # 定义初始化方法
        super(TransformerEncoderModule, self).__init__()  # 初始化父类模块

        self.embedding = nn.Linear(1, d_model)  # 将单通道输入映射到模型维度
        self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_len=max_len)  # 构建正弦位置编码

        encoder_layer = nn.TransformerEncoderLayer(  # 构建单层 Transformer 编码器
            d_model=d_model,  # 设置模型维度
            nhead=nhead,  # 设置多头注意力头数
            dim_feedforward=dim_feedforward,  # 设置前馈网络维度
            dropout=dropout,  # 设置 dropout 比例
            batch_first=True  # 设置批量维度在最前面
        )  # 结束编码器层构建
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 堆叠多个编码层

        self.fc = nn.Sequential(  # 构建最终投影层
            nn.Linear(d_model, wave_emb_dim),  # 将 Transformer 输出映射到嵌入维度
            nn.ReLU(),  # 添加激活函数
            nn.Dropout(dropout)  # 添加随机失活
        )  # 结束全连接层定义

    def forward(self, wave, length):  # 定义前向传播方法
        x = wave.transpose(1, 2)  # 将输入从 (Batch, 1, L) 转为 (Batch, L, 1)

        x = self.embedding(x)  # 对每个时间步做线性嵌入

        x = self.pos_embedding(x)  # 加入位置编码信息

        batch_size, seq_len, _ = x.size()  # 获取批量大小和序列长度
        device = x.device  # 获取当前张量所在设备
        range_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)  # 构造位置索引张量
        padding_mask = range_tensor >= length.unsqueeze(1)  # 生成填充位置掩码

        out = self.transformer(x, src_key_padding_mask=padding_mask)  # 通过 Transformer 编码序列

        mask = (~padding_mask).unsqueeze(-1)  # 构造有效位置掩码并扩展维度
        masked_out = out * mask  # 将填充位置置零以便池化

        sum_out = torch.sum(masked_out, dim=1)  # 沿时间维求和
        count = torch.clamp(torch.sum(mask, dim=1), min=1.0)  # 计算有效位置数量并避免除零
        pooled = sum_out / count  # 计算有效位置的平均表示

        wave_emb = self.fc(pooled)  # 将池化结果投影到目标嵌入空间

        return wave_emb  # 返回波形嵌入

class TransformerModel(BaseModel):  # 定义 Transformer 主模型
    def __init__(self, wave_emb_dim=128, d_model=64, nhead=4, num_layers=2,  # 定义初始化方法并设置编码器参数
                 dim_feedforward=256, num_points=161, num_channels=4, dropout=0.1):  # 设置解码器和正则化参数
        super(TransformerModel, self).__init__()  # 初始化父类模型

        self.encoder = TransformerEncoderModule(  # 构建 Transformer 编码器
            wave_emb_dim=wave_emb_dim,  # 设置输出嵌入维度
            d_model=d_model,  # 设置模型维度
            nhead=nhead,  # 设置注意力头数
            num_layers=num_layers,  # 设置编码层数
            dim_feedforward=dim_feedforward,  # 设置前馈网络维度
            dropout=dropout  # 设置 dropout 比例
        )  # 结束编码器构建
        self.decoder = CurveDecoder(  # 构建曲线解码器
            wave_emb_dim=wave_emb_dim,  # 设置输入嵌入维度
            num_points=num_points,  # 设置输出点数量
            num_channels=num_channels,  # 设置输出通道数量
            dropout=dropout * 2.0  # 设置解码器 dropout，大约为 0.2
        )  # 结束解码器构建

    def forward(self, wave, cond, length):  # 定义前向传播方法
        """输入波形、条件和长度，输出曲线结果。"""  # 说明前向函数用途
        wave_emb = self.encoder(wave, length)  # 使用编码器提取波形嵌入
        out = self.decoder(wave_emb, cond)  # 将波形嵌入与条件输入解码为曲线
        return out  # 返回模型输出
