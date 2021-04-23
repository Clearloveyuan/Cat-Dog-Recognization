"""
实战2
代码8-5
识别猫的卷积神经网络测试程序
"""
import torch                                                # 引入torch模块
import numpy as np                                          # 引入np模块

net = torch.load('net.pkl')                                 # 引入训练好的网络
x = np.load(file="cat_train_set.npy") / 255                  # 载入测试集并进行简单归一化
x = torch.tensor(x).type(torch.FloatTensor).cuda()          # 转换成tensor变量并传入GPU
y1 = torch.zeros(70)
y2 = torch.ones(70)
y0 = torch.cat((y1, y2))                                    # 设置标签用来计算准确率

y = net(x)                                                  # 输入网络得到结果

a1 = torch.max(y, 1)[1].cpu().data.numpy()                  # 数据传回CPU，返回数字较大的坐标
a2 = y0.data.numpy()                                        # 标签转换成numpy数组
print(a1)
print(f'准确率：{sum(a1 == a2)/140}')                        # 打印准确率