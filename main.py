# -*- coding: utf-8 -*-

import Network_model as mlp_model
import mxnet as mx
import load_data
import initM
import random
import numpy as np
import logging

logging.getLogger().setLevel(logging.DEBUG)

# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

# 进行数据的shuffle
train_label = train_label.reshape(train_label.shape[0], 1)
numpyx = x.asnumpy()
training_data = np.concatenate((numpyx, train_label), axis=1)
random.shuffle(training_data)

x = training_data[:, :-1]
train_label = training_data[:, -1]
print train_label[0:10]
# 数据整合shuffle完毕

x = x[:600]
val_x = val_x[:600]
train_label = train_label[:600]
val_label = val_label[:600]

print "训练集和验证集数据收集完毕，开始生成训练集以及验证集"

# 初始化变量M
M = mx.ndarray.ones(shape=(10, 10))  # dimension 10*10
M = np.random.randint(0, 1, size=(10, 10))

M_shape = M.shape[0]
batch_size = 6

self_made_m = initM.init_m(train_label)
train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, self_made_m)

print "训练集+验证集生成完成"

# 加载训练网络
k = 10
a = 0.7
net = mlp_model.model_main(k, a)

# 训练模型
model = mx.model.FeedForward(
    symbol=net,  # network structure
    num_epoch=50,  # number of data passes for training
    learning_rate=0.01,  # learning rate of SGD
    initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
    arg_params={'M': M}
)

data_i_iter = []
data_j_iter = []
count = 0
metric = load_data.Auc()

print "网络加载完成，开始训练"
model.fit(
    X=train,  # training data
    eval_metric=metric,
    # eval_data=test,  # validation data
    batch_end_callback=mx.callback.Speedometer(batch_size, 200)  # output progress for each 200 data batches
)

# prefix = 'mymodel'
# iteration = 100
# model.save(prefix, iteration)
