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
from numpy import linalg as la

M = np.random.randint(0, 2, size=(10, 10))
k = 10
a = 0.7
prefix = 'mymodel'
iteration = 0
batch_size = 64
num_epoch = 300
learning_rate = 0.02

Gama = 0.2
Lambda = 1

# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

# 进行数据的shuffle
train_label = train_label.reshape(train_label.shape[0], 1)
numpyx = x.asnumpy()
training_data = np.concatenate((numpyx, train_label), axis=1)
random.shuffle(training_data)
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

self_made_m = initM.init_m_random(train_label)
# 进行logger的初始化
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# 做十次循环
for i in range(0, 20):

    # 更新模型迭代次数
    iteration = iteration + 100
    hdlr = logging.FileHandler('log-' + str(iteration) + '.txt')
    logger.addHandler(hdlr)
    # minimizefx.fx_minimize(x, val_x, train_label, val_label, self_made_m, M, k, a, batch_size, prefix, iteration,
    #                        num_epoch, learning_rate)
    # 计算新f(x)旧M
    old_M = M
    # 加载f(x) model
    model_loaded = mx.model.FeedForward.load(prefix, iteration)
    print "模型加载完毕"
    # 获得新的train_label的preds
    count = 0
    beta = 0.5
    preds = minimizem.mx_predict(x, model_loaded)

    calculateold_M = np.zeros((train_label.shape[0], train_label.shape[0]))
    for xi in range(0, x.shape[0]):
        for xj in range(xi, x.shape[0]):
            calculateold_M[xi][xj] = np.dot(np.dot(preds[xi], old_M), preds[xj].T)
            calculateold_M[xj][xi] = calculateold_M[xi][xj]

    U, Sigma, VT = la.svd(old_M)
    old_mainloss = sum(Sigma)
    for xi in range(0, x.shape[0]):
        for xj in range(0, x.shape[0]):
            if self_made_m[xi][xj] == 1:
                old_mainloss = old_mainloss + a * (calculateold_M[xi][xj] - 1) * (calculateold_M[xi][xj] - 1)
            else:
                old_mainloss = old_mainloss + (1 - a) * calculateold_M[xi][xj] * calculateold_M[xi][xj]

    f = open('mloss' + str(iteration) + '.txt', 'a')
    f.write(str(old_mainloss))
    f.write('\n')
    f.close()
    # 计算新f(x)旧M 结束

    # #循环迭代优化M
    # for i in range(0, 3):
    #     New_M, preds = minimizem.m_minimize(x, self_made_m, M, prefix, iteration, a, Gama, Lambda, k)
    #     M = New_M
    #
    #     calculateM = np.zeros((train_label.shape[0], train_label.shape[0]))
    #     for xi in range(0, x.shape[0]):
    #         for xj in range(xi, x.shape[0]):
    #             calculateM[xi][xj] = np.dot(np.dot(preds[xi], M), preds[xj].T)
    #             calculateM[xj][xi] = calculateM[xi][xj]
    #
    #     # 计算总的loss
    #     U, Sigma, VT = la.svd(M)
    #     mainloss = sum(Sigma)
    #
    #     for xi in range(0, x.shape[0]):
    #         for xj in range(0, x.shape[0]):
    #             if self_made_m[xi][xj] == 1:
    #                 mainloss = mainloss + a * (calculateM[xi][xj] - 1) * (calculateM[xi][xj] - 1)
    #             else:
    #                 mainloss = mainloss + (1 - a) * calculateM[xi][xj] * calculateM[xi][xj]
    #
    #     f = open('mloss' + str(iteration) + '.txt', 'a')
    #     f.write(str(mainloss))
    #     f.write('\n')
    #     f.close()
    # # 循环迭代优化M 结束

    # M = minimizem.m_minimize_bynetwork(x, val_x, train_label, val_label, batch_size, self_made_m, prefix, iteration,
    #                                    num_epoch, 0.1, k, a)
    M = minimizem.m_minimize_admm(x, self_made_m, M, prefix, iteration, a, beta, Lambda, k)
    # 保存M的结果
    np.savetxt('new-M' + str(iteration) + '-' + str(train_data_count), M, fmt=['%s']*M.shape[1], newline='\n')

    # 保存正确的结果常量M
    arrayM = initM.init_m(train_label)
    np.savetxt('constantM' + str(iteration) + '-' + str(train_data_count), arrayM, fmt=['%s']*arrayM.shape[1], newline='\n')

    # 计算最后生成的M与老m做比较
    calculateM = np.zeros((train_label.shape[0], train_label.shape[0]))

    for xi in range(0, x.shape[0]):
        for xj in range(xi, x.shape[0]):
            calculateM[xi][xj] = np.dot(np.dot(preds[xi], M),  preds[xj].T)
            calculateM[xj][xi] = calculateM[xi][xj]

    np.savetxt('calculateM' + str(iteration) + '-' + str(train_data_count), calculateM, fmt=['%s']*calculateM.shape[1], newline='\n')

    # 计算总的loss
    U, Sigma, VT = la.svd(M)
    mainloss = sum(Sigma)

    print 'keneral loss', mainloss
    for xi in range(0, x.shape[0]):
        for xj in range(0, x.shape[0]):
            if self_made_m[xi][xj] == 1:
                mainloss = mainloss + a * (calculateM[xi][xj] - 1) * (calculateM[xi][xj] - 1)
            else:
                mainloss = mainloss + (1 - a) * calculateM[xi][xj] * calculateM[xi][xj]

    f = open('mainloss.txt', 'a')
    f.write(str(old_mainloss)+" "+str(mainloss))
    f.write('\n')
    f.close()
    print "第", i+1, "次mainloss: ", mainloss
    print "第", i+1, "次AMA迭代完成"
