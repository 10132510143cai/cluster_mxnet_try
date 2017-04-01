# -*- coding: utf-8 -*-

import minimizefx
import minimizem
import load_data
import random
import numpy as np
import logging

M = np.random.randint(0, 1, size=(10, 10))
k = 10
a = 0.7
prefix = 'mymodel'
iteration = 0
batch_size = 6
num_epoch = 10
learning_rate = 0.03

Gama = 0.7
Lambda = 0.8

# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

# 进行数据的shuffle
train_label = train_label.reshape(train_label.shape[0], 1)
numpyx = x.asnumpy()
training_data = np.concatenate((numpyx, train_label), axis=1)

# 做十次循环
for i in range(0, 10):
    # 重新构建数据集
    random.shuffle(training_data)

    x = training_data[:, :-1]
    train_label = training_data[:, -1]
    # 数据整合shuffle完毕

    # 数据截取
    x = x[:600]
    val_x = val_x[:600]
    train_label = train_label[:600]
    val_label = val_label[:600]
    print "训练集和验证集数据收集完毕，开始生成训练集以及验证集"
    # 更新模型迭代次数
    iteration = iteration + 100

    minimizefx.fx_minimize(x, val_x, train_label, val_label, M, k, a, batch_size, prefix, iteration,
                           num_epoch, learning_rate)
    M = minimizem.m_minimize(x, train_label, M, prefix, iteration, a, Gama, Lambda, k)
    print "第", i, "次AMA迭代完成"