# -*- coding: utf-8 -*-

import minimizefx
import minimizem
import load_data
import random
import numpy as np
import logging
import initM
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
import mxnet as mx

M = np.random.randint(0, 1, size=(10, 10))
k = 10
a = 0.7
prefix = 'mymodel'
iteration = 0
batch_size = 6
num_epoch = 10
learning_rate = 0.02

Gama = 0.7
Lambda = 0.8

# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

# 进行数据的shuffle
train_label = train_label.reshape(train_label.shape[0], 1)
numpyx = x.asnumpy()
training_data = np.concatenate((numpyx, train_label), axis=1)
random.shuffle(training_data)

# 进行logger的初始化
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# 做十次循环
for i in range(0, 10):
    # 重新构建数据集


    x = training_data[:, :-1]
    train_label = training_data[:, -1]
    # 数据整合shuffle完毕

    train_data_count = 600
    # 数据截取
    x = x[:train_data_count]
    val_x = val_x[:train_data_count]
    train_label = train_label[:train_data_count]
    val_label = val_label[:train_data_count]
    print "训练集和验证集数据收集完毕，开始生成训练集以及验证集"
    # 更新模型迭代次数
    iteration = iteration + 100
    hdlr = logging.FileHandler('log-' + str(iteration) + '.txt')
    logger.addHandler(hdlr)
    self_made_m = minimizefx.fx_minimize(x, val_x, train_label, val_label, M, k, a, batch_size, prefix, iteration,
                           num_epoch, learning_rate)
    M, preds = minimizem.m_minimize(x, train_label, M, prefix, iteration, a, Gama, Lambda, k)

    # 保存M的结果
    np.savetxt('new-M' + str(iteration) + '-' + str(train_data_count), M, fmt=['%s']*M.shape[1], newline='\n')

    # 保存正确的结果常量M
    arrayM = initM.init_m(train_label)
    np.savetxt('constantM' + str(iteration) + '-' + str(train_data_count), arrayM, fmt=['%s']*arrayM.shape[1], newline='\n')

    # 计算出的f(xi)Mf(xj)T
    calculateM = np.zeros((train_label.shape[0], train_label.shape[0]))
    for xi in range(0, x.shape[0]):
        for xj in range(xi, x.shape[0]):
            calculateM[xi][xj] = np.dot(np.dot(preds[xi], M),  preds[xj].T)
            calculateM[xj][xi] = calculateM[xi][xj]
    np.savetxt('calculateM' + str(iteration) + '-' + str(train_data_count), calculateM, fmt=['%s']*calculateM.shape[1], newline='\n')

    # 计算总的loss
    U, Sigma, VT = randomized_svd(M, n_components=k,
                                  n_iter=5,
                                  random_state=None)
    mainloss = sum(Sigma)
    print 'keneral loss', mainloss
    for xi in range(0, x.shape[0]):
        for xj in range(0, x.shape[0]):
            if self_made_m[xi][xj] == 1:
                mainloss = mainloss + a * (calculateM[xi][xj] - 1) * (calculateM[xi][xj] - 1)
            else:
                mainloss = mainloss + (1 - a) * calculateM[xi][xj] * calculateM[xi][xj]

    f = open('mainloss.txt', 'a')
    f.write(str(mainloss))
    f.write('\n')
    f.close()
    print "第", i+1, "次mainloss: ", mainloss
    print "第", i+1, "次AMA迭代完成"