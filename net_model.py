"""
实战2
代码8-3
识别猫狗的卷积神经网络模型
"""
import torch.nn as nn                        # 引入torch.nn模块


class ICNET(nn.Module):                       # 定义类存储网络结构
    def __init__(self):
        super(ICNET, self).__init__()
        self.ConvNet = nn.Sequential(       # nn模块搭建卷积网络
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1,
                      bias=False),          # size：128,128,8
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1,
                      bias=False),          # size：128,128,8
            nn.ReLU(inplace=True),          # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # size：64,64,18

        )
        self.LinNet = nn.Sequential(       # nn模块搭建网络
            nn.Linear(64*64*8, 1000),      # 全连接层
            nn.ReLU(inplace=True),         # ReLU激活函数
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 2),
            nn.Softmax(dim=1)              # SoftMax分类激活函数
        )

    def forward(self, x):                   # 定义前向传播过程
        x = self.ConvNet(x)                 # 将x传入卷积网络
        x = x.view(x.size(0), 64*64*8)      # 展成一维数组
        out = self.LinNet(x)                # 接着通过全连接层
        return out                          # 返回预测值