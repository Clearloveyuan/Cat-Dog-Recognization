"""
实战2
代码8-4
识别猫的卷积神经网络训练程序
（附加可视化）
"""
from net_model import ICNET                                 # 引入网络模型
import torch                                                # 引入torch模块
import torch.nn as nn                                       # 引入torch.nn模块
import numpy as np                                          # 引入np模块
import matplotlib.pyplot as plt                             # 引入matplotlib模块

net = ICNET().cuda()                                        # 将网络传入GPU
x = np.load(file="cat_train_set.npy") / 255                 # 载入训练集并进行简单归一化
x = torch.tensor(x).type(torch.FloatTensor).cuda()          # 转换成tensor变量并传入GPU
y1 = torch.zeros(70)
y2 = torch.ones(70)
y = torch.cat((y1, y2)).type(torch.LongTensor)
optimizer = torch.optim.SGD(net.parameters(), lr=0.03)      # 设置优化器
loss_func = nn.CrossEntropyLoss()                           # 设置损失函数

samplenum = 140        # 样本总数（训练集总数）
minibatch = 35         # 小批次样本大小
w_HR = 128             # 样本尺寸

x0 = np.zeros(minibatch * 3 * w_HR * w_HR)
x0 = np.reshape(x0, (minibatch, 3, w_HR, w_HR))       # 创建小批次空白样本
y0 = np.zeros(minibatch)                              # 创建小批次空白标签
x0 = torch.tensor(x0).type(torch.FloatTensor).cuda()
y0 = torch.tensor(y0).type(torch.LongTensor).cuda()   # 将小批次样本和标签传入GPU

plt.ion()       # 开启交互模式
x_plt = [0]     # x坐标
y_plt = [0]     # y坐标
for epoch in range(1000):                                             # epoch循环
    for iterations in range(int(samplenum / minibatch)):              # iterations循环
        k = 0
        for i in range(iterations * minibatch, iterations * minibatch + minibatch):     # 部分样本赋值给x0的循环
            x0[k, 0, :, :] = x[i, 0, :, :]
            x0[k, 1, :, :] = x[i, 1, :, :]
            x0[k, 2, :, :] = x[i, 2, :, :]
            y0[k] = y[i]         # 小批次标签
            k = k + 1

        out = net(x0)                                     # 实际输出
        loss = loss_func(out, y0)                         # 实际输出和期望输出传入损失函数
        optimizer.zero_grad()                             # 清除梯度
        loss.backward()                                   # 误差反向传播
        optimizer.step()                                  # 优化器开始优化
    if epoch % 50 == 0:                                   # 每50次显示
        plt.cla()            # 清除上一次绘图
        plt.xlim((0, 1000))  # 设置x坐标范围
        plt.xlabel('epoch')  # x轴的标题
        plt.ylim((0, 1))  # 设置y坐标范围
        plt.ylabel('loss')  # y轴的标题
        x_plt.append(epoch)  # 增加x坐标
        y_plt.append(loss.data) # 增加y坐标
        plt.plot(x_plt, y_plt, c='r', marker='x')  # 绘制折线图
        print(f'epoch:{epoch},loss:{loss}')     # 打印中间过程
        plt.pause(0.1)      # 停留显示
plt.ioff()  # 关闭交互模式
plt.show()  # 显示最后一幅图
torch.save(net, 'net.pkl')                                # 保存网络

