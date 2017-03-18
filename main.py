# -*- coding: utf-8 -*-

import Network_model as mlp_model
import mxnet as mx
import load_data
import pandas as pd
import sys

sys.path.insert(0, "../../python")

# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

x = x[:600]
val_x = val_x[:600]
train_label = train_label[:600]
val_label = val_label[:600]
print "训练集和验证集数据收集完毕，开始生成训练集以及验证集"

M = mx.ndarray.ones(shape=(10, 10))  # dimension 10*10

print 'M的维数k为',
print M.shape[0]

M_shape = M.shape[0]
batch_size = 10

train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, M_shape)

print "训练集+验证集生成完成"

# 加载训练网络
k = 10
a = 0.5
net = mlp_model.model_main(k, a)

# 训练模型
model = mx.model.FeedForward(
    symbol=net,       # network structure
    num_epoch=10,     # number of data passes for training
    learning_rate=0.1,  # learning rate of SGD
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
    # batch_end_callback=mx.callback.Speedometer(batch_size, 200)  # output progress for each 200 data batches
)

model.save(sys.argv[2])



