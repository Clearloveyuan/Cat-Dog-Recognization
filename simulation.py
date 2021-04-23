"""
实战2
代码8-6
模拟实际运用程序
"""
import torch                                                # 引入torch模块
import numpy as np                                          # 引入np模块
import cv2 as cv                                            # 引入cv模块

net = torch.load('net.pkl')                                 # 引入训练好的网络
img = cv.imread('./sample/dog.20.jpg')                       # 读取图片
img0 = cv.resize(img, (128, 128))                            # 更改图像尺寸
x = np.zeros(128*128*3)                                       # 新建空白输入
x = np.reshape(x, (1, 3, 128, 128))                         # 调整成输入规定的维度
x[0, 0, :, :] = img0[:, :, 0]/255
x[0, 1, :, :] = img0[:, :, 1]/255
x[0, 2, :, :] = img0[:, :, 2]/255                            # 赋值输入并进行简单归一化
x = torch.tensor(x).type(torch.FloatTensor).cuda()          # 转换成tensor变量并传入GPU

y = net(x)                                                  # 输入网络得到结果

max_num = torch.max(y, 1)[1]                                # 返回最大值的下标
if max_num == 0:                                            # if语句判断显示精确结果
    print('识别结果：图片中是猫')
    str = 'cat'
else:
    print('识别结果：图片中是狗')
    str = 'dog'
cv.imshow(f'{str}', img)                                       # 显示原始图片
cv.waitKey(0)                                                  # 定格显示