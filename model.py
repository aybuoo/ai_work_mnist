# model.py
import torch
from torch import nn

class MLP(nn.Module):
    """多层全连接神经网络（3层隐藏层）"""
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()  # 将28x28图像展平为784维向量
        self.layers = nn.Sequential(
            nn.Linear(784, 256),       # 隐藏层1：256神经元
            nn.ReLU(),
            nn.Dropout(0.2),           # 防止过拟合
            nn.Linear(256, 128),       # 隐藏层2：128神经元
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),        # 隐藏层3：64神经元
            nn.ReLU(),
            nn.Linear(64, 10)          # 输出层：10分类
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.flatten(x)
        return self.layers(x)

if __name__ == "__main__":
    # 验证模型输入输出维度
    model = MLP()
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    print(f"Input shape: (1, 1, 28, 28), Output shape: {output.shape}")
    # 预期输出：torch.Size([1, 10])