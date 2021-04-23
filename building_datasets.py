"""
实战2
代码8-1
构建样本集程序
数据集共有200张图片，包括100张猫和100张狗
选择前70张猫和前70张狗为训练集
后30张猫和后30张狗为测试集
"""
import cv2 as cv                                                   # 引入OpenCV模块
import numpy as np                                                 # 引入numpy模块
import os                                                          # 引入os模块


if __name__ == '__main__':
    # 输入x的长宽
    l_x = 128

    # 样本数量
    m = 200  # 样本总数
    m1 = 70  # 训练集数量
    m2 = 30  # 测试集数量
    train_set = np.zeros(m1*2 * l_x * l_x*3)                            # 创建零矩阵
    train_set = np.reshape(train_set, (m1*2, 3, l_x, l_x))              # 重塑成可以存储图片的形状
    test_set = np.zeros(m2*2 * l_x * l_x*3)                             # 创建零矩阵
    test_set = np.reshape(test_set, (m2*2, 3, l_x, l_x))                # 重塑成可以存储图片的形状
    sucess_mark = 0                                                   # 成功标记
                                                     # 测试集数量

    # 构建训练集
    for i in range(m1):
        path1 = f'./sample/cat.{i}.jpg'                           # 路径
        path2 = f'./sample/dog.{i}.jpg'
        if os.path.exists(path1) & os.path.exists(path2):         # 判断两个路径是否存在
            img1 = cv.imread(path1)
            img2 = cv.imread(path2)                               # 读取两个图片（一张猫一张狗）
            img1 = cv.resize(img1, (l_x, l_x))
            img2 = cv.resize(img2, (l_x, l_x))                           # 将图片调整到规定尺寸
            train_set[i, 0, :, :] = img1[:, :, 0]
            train_set[i, 1, :, :] = img1[:, :, 1]
            train_set[i, 2, :, :] = img1[:, :, 2]                          # 赋给训练集矩阵
            sucess_mark += 1                                           # 成功标记加一
            print("\r" + f'训练集总数：{m1*2}，当前第{(i+1)*2}个', end="", flush=True)
            train_set[m1+i, 0, :, :] = img2[:, :, 0]
            train_set[m1+i, 1, :, :] = img2[:, :, 1]
            train_set[m1+i, 2, :, :] = img2[:, :, 2]                          # 赋给训练集矩阵
            sucess_mark += 1                                           # 成功标记加一
        else:                                                          # 路径不存在的情况
            print(f'路径{path1}或{path2}不存在！')
            break                                                      # 报错之后直接跳出循环
    print('')                                                          # 换行格式

    # 构建测试集
    for i in range(70, 100):
        path1 = f'./sample/cat.{i}.jpg'                           # 路径
        path2 = f'./sample/dog.{i}.jpg'
        if os.path.exists(path1) & os.path.exists(path2):                                       # 判断路径是否存在
            # 读取图片并转换成灰度图
            img1 = cv.imread(path1)
            img2 = cv.imread(path2)
            img1 = cv.resize(img1, (l_x, l_x))
            img2 = cv.resize(img2, (l_x, l_x))
            test_set[i-70, 0, :, :] = img1[:, :, 0]                     # 赋给测试集矩阵
            test_set[i-70, 1, :, :] = img1[:, :, 1]                     # 赋给测试集矩阵
            test_set[i-70, 2, :, :] = img1[:, :, 2]                     # 赋给测试集矩阵
            sucess_mark += 1
            print("\r" + f'测试集总数：{m2*2}，当前第{(i-70+1)*2}个', end="", flush=True)
            test_set[m2+i-70, 0, :, :] = img2[:, :, 0]                     # 赋给测试集矩阵
            test_set[m2+i-70, 1, :, :] = img2[:, :, 1]                     # 赋给测试集矩阵
            test_set[m2+i-70, 2, :, :] = img2[:, :, 2]                     # 赋给测试集矩阵
            sucess_mark += 1
        else:
            print(f'路径{path1}或{path2}不存在！')
            break
    print('')
    if sucess_mark == 200:                    # 如果成功标记个数为样本总数则保存两个数据集
        np.save('cat_train_set.npy', train_set)
        np.save('cat_test_set.npy', test_set)
        print('生成成功！')