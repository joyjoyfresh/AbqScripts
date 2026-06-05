import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块


class BaseModel(nn.Module):  # 定义基础模型类并继承nn.Module
    def __init__(self):  # 初始化基础模型对象
        super(BaseModel, self).__init__()  # 调用父类初始化方法

    def get_num_params(self):  # 统计模型中的可训练参数数量
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  # 返回所有可训练参数总数

    def save_checkpoint(self, filepath):  # 将模型权重保存到指定路径
        torch.save(self.state_dict(), filepath)  # 保存当前模型参数字典

    def load_checkpoint(self, filepath, device='cpu'):  # 从指定路径加载模型权重
        self.load_state_dict(torch.load(filepath, map_location=device))  # 按指定设备读取并恢复模型参数
