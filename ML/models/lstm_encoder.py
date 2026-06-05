import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
from models.base_model import BaseModel  # 导入基础模型类
from models.decoder import CurveDecoder  # 导入曲线解码器

class SelfAttentionPooling(nn.Module):  # 定义自注意力池化模块
    """使用自注意力对可变长度的 LSTM 输出进行聚合。"""  # 说明模块用途

    def __init__(self, input_dim):  # 定义初始化方法
        super(SelfAttentionPooling, self).__init__()  # 初始化父类模块
        self.attn_linear = nn.Linear(input_dim, 1)  # 构建注意力打分层

    def forward(self, x, length):  # 定义前向传播方法
        """输入序列特征和有效长度，并返回池化后的向量。"""  # 说明输入输出
        batch_size, seq_len, input_dim = x.size()  # 读取批量大小、序列长度和特征维度

        scores = self.attn_linear(x)  # 计算原始注意力分数

        device = x.device  # 获取输入张量所在设备
        range_tensor = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)  # 构建位置索引张量
        mask = range_tensor < length.unsqueeze(1)  # 根据真实长度生成有效位置掩码

        scores = scores.squeeze(-1)  # 去掉最后一维得到二维分数矩阵
        scores = scores.masked_fill(~mask, -1e9)  # 将无效位置填充为极小值

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # 对分数做归一化并恢复三维形状

        pooled = torch.sum(x * weights, dim=1)  # 按注意力权重对序列特征求加权和

        return pooled  # 返回池化后的特征向量

class LSTMEncoder(nn.Module):  # 定义 LSTM 编码器
    def __init__(self, wave_emb_dim=128, hidden_dim=64, num_layers=2, dropout=0.3):  # 定义初始化方法
        super(LSTMEncoder, self).__init__()  # 初始化父类模块

        self.embedding = nn.Linear(1, 64)  # 将单通道输入映射到 64 维特征

        self.lstm = nn.LSTM(  # 构建双向 LSTM 编码器
            input_size=64,  # 设置输入特征维度
            hidden_size=hidden_dim,  # 设置隐藏状态维度
            num_layers=num_layers,  # 设置堆叠层数
            batch_first=True,  # 设置批量维度在最前面
            bidirectional=True,  # 启用双向编码
            dropout=dropout if num_layers > 1 else 0.0  # 根据层数决定是否启用 dropout
        )  # 结束 LSTM 构建

        self.pooling = SelfAttentionPooling(hidden_dim * 2)  # 构建自注意力池化层

        self.fc = nn.Sequential(  # 构建最终嵌入投影层
            nn.Linear(hidden_dim * 2, wave_emb_dim),  # 将双向输出映射到目标嵌入维度
            nn.ReLU(),  # 添加激活函数
            nn.Dropout(dropout)  # 添加随机失活
        )  # 结束全连接层定义

    def forward(self, wave, length):  # 定义前向传播方法
        x = wave.transpose(1, 2)  # 将输入从 (Batch, 1, L) 转为 (Batch, L, 1)

        x = self.embedding(x)  # 对序列每个时刻做线性嵌入

        length_cpu = length.cpu()  # 将长度张量移动到 CPU 以供打包序列使用
        packed_x = nn.utils.rnn.pack_padded_sequence(  # 将变长序列打包
            x, length_cpu, batch_first=True, enforce_sorted=False  # 保持原始顺序并关闭排序要求
        )  # 结束序列打包

        packed_out, _ = self.lstm(packed_x)  # 通过 LSTM 编码打包后的序列

        out, _ = nn.utils.rnn.pad_packed_sequence(  # 将打包输出还原为填充序列
            packed_out, batch_first=True, total_length=x.size(1)  # 保证输出长度与输入长度一致
        )  # 结束序列还原

        pooled = self.pooling(out, length)  # 使用自注意力对时间维度做聚合

        wave_emb = self.fc(pooled)  # 将池化结果投影到目标嵌入空间

        return wave_emb  # 返回波形嵌入

class LSTMModel(BaseModel):  # 定义 LSTM 主模型
    def __init__(self, wave_emb_dim=128, hidden_dim=64, num_layers=2, num_points=161, num_channels=4, dropout=0.3):  # 定义初始化方法
        super(LSTMModel, self).__init__()  # 初始化父类模型

        self.encoder = LSTMEncoder(  # 构建波形编码器
            wave_emb_dim=wave_emb_dim,  # 设置波形嵌入维度
            hidden_dim=hidden_dim,  # 设置 LSTM 隐藏维度
            num_layers=num_layers,  # 设置 LSTM 层数
            dropout=dropout  # 设置 dropout 比例
        )  # 结束编码器构建
        self.decoder = CurveDecoder(  # 构建曲线解码器
            wave_emb_dim=wave_emb_dim,  # 设置输入嵌入维度
            num_points=num_points,  # 设置输出点数量
            num_channels=num_channels,  # 设置输出通道数量
            dropout=dropout * 0.67  # 设置解码器 dropout 比例
        )  # 结束解码器构建

    def forward(self, wave, cond, length):  # 定义前向传播方法
        """输入波形、条件和长度，输出曲线结果。"""  # 说明前向函数用途
        wave_emb = self.encoder(wave, length)  # 使用编码器提取波形嵌入
        out = self.decoder(wave_emb, cond)  # 将波形嵌入与条件输入解码为曲线
        return out  # 返回模型输出
