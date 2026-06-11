import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
from models.base_model import BaseModel  # 导入基础模型类
from models.cnn_encoder import CNNEncoder  # 导入卷积编码器

class TrunkNet(nn.Module):  # 定义分支网络中的主干网络
    """将查询坐标 x/h 映射到表示空间。"""  # 说明主干网络的作用

    def __init__(self, out_dim=128, hidden_dims=[64, 128, 128]):  # 定义主干网络初始化方法
        super(TrunkNet, self).__init__()  # 初始化父类模块

        layers = []  # 创建全连接层容器
        in_dim = 1  # 设置输入维度为一维坐标
        for h_dim in hidden_dims:  # 逐层构建隐藏层
            layers.append(nn.Linear(in_dim, h_dim))  # 添加线性变换层
            layers.append(nn.ReLU())  # 添加 ReLU 激活层
            in_dim = h_dim  # 更新下一层输入维度
        layers.append(nn.Linear(in_dim, out_dim))  # 添加输出映射层

        self.mlp = nn.Sequential(*layers)  # 将层组合成顺序网络

    def forward(self, x):  # 定义前向传播方法
        """输入坐标张量并输出对应表示。"""  # 说明前向过程
        return self.mlp(x)  # 通过多层感知机计算输出

class DeepONetModel(BaseModel):  # 定义 DeepONet 主模型
    def __init__(self, wave_emb_dim=128, trunk_hidden_dims=[64, 128, 128],  # 定义初始化方法并设置分支维度
                 num_points=161, num_channels=4, dropout=0.3):  # 设置网格点数、通道数和 dropout
        super(DeepONetModel, self).__init__()  # 初始化父类模型

        self.num_points = num_points  # 保存网格点数量
        self.num_channels = num_channels  # 保存输出通道数量
        self.p_dim = wave_emb_dim  # 保存表示空间维度

        self.branch_encoder = CNNEncoder(wave_emb_dim=wave_emb_dim, dropout=dropout)  # 构建波形分支编码器
        self.branch_fc = nn.Sequential(  # 构建分支全连接融合层
            nn.Linear(wave_emb_dim + 3, 256),  # 将波形特征与条件特征映射到中间维度
            nn.ReLU(),  # 添加激活函数
            nn.Dropout(dropout),  # 添加随机失活
            nn.Linear(256, num_channels * wave_emb_dim)  # 输出每个通道对应的表示向量
        )  # 结束分支全连接层定义

        self.trunk_net = TrunkNet(out_dim=wave_emb_dim, hidden_dims=trunk_hidden_dims)  # 构建坐标主干网络

        grid_x = torch.linspace(0.0, 8.0, num_points).unsqueeze(1)  # 生成固定查询坐标网格
        self.register_buffer('grid_x', grid_x)  # 将网格注册为缓冲区以便随设备迁移

    def forward(self, wave, cond, length=None):  # 定义前向传播方法
        """输入波形、条件和长度，输出多通道空间分布。"""  # 说明前向函数用途
        batch_size = wave.size(0)  # 获取批量大小

        wave_emb = self.branch_encoder(wave)  # 提取波形编码特征
        branch_in = torch.cat([wave_emb, cond], dim=1)  # 拼接波形特征与条件特征
        branch_out = self.branch_fc(branch_in)  # 通过分支全连接层生成主干系数
        branch_out = branch_out.view(batch_size, self.num_channels, self.p_dim)  # 重塑为通道和表示维度

        trunk_out = self.trunk_net(self.grid_x)  # 对固定坐标网格进行编码

        out = torch.einsum('bcp,kp->bck', branch_out, trunk_out)  # 计算分支与主干的点积融合结果

        return out  # 返回模型输出
