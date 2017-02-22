# -*- coding: utf-8 -*-

import Network_model as mlp_model
import mxnet as mx
import load_data
import pandas as pd

# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

x = x[:600]
val_x = val_x[:600]
print "训练集和验证集数据收集完毕，开始生成训练集以及验证集"

M = mx.ndarray.ones(shape=(10, 10))  # dimension 10*10

print 'M的维数k为',
print M.shape[0]

M_shape = M.shape[0]

train, test = load_data.get_data(x, val_x, x, M_shape, M)

print "训练集+验证集生成完成"

# 加载训练网络
k = 10
a = 0.5
net = mlp_model.model_main(k, a)

# 训练模型
model = mx.model.FeedForward(
    symbol=net,       # network structure
    num_epoch=10,     # number of data passes for training
    learning_rate=0.1  # learning rate of SGD
)

data_i_iter = []
data_j_iter = []
count = 0

print "网络加载完成，开始训练"
model.fit(
    X=train,  # training data
    # eval_data=test,  # validation data
    # batch_end_callback=mx.callback.Speedometer(batch_size, 200)  # output progress for each 200 data batches
)


# for i in range(0, U.shape[0] - 1):
#     for j in range(0, U.shape[0] - 1):
#         data_i_iter.append(U[i])
#         data_j_iter.append(U[j])
#         count = count + 1
#         if count == 100:
#             count = 0
#             data_all = [mx.nd.array(data_i_iter),  mx.nd.array(data_j_iter), i, j]
#             data_names = ['dataxi', 'dataxj', 'i', 'j']
#             train_data = pd.DataFrame(data_all, columns=data_names, dtype=float)
#             train_data.to_csv(r"C:\Users\Mr_C\Desktop\cluster\train.csv", index=False, encoding="utf-8")
#
#             # 这里添加M中信息
#
#             train_iter = mx.io.csvIter(
#                 data_csv="C:\Users\Mr_C\Desktop\cluster\train.csv",
#                 data_shape=(100, 4)
#             )
#
#             val_iter = mx.io.csvIter(
#                 data_csv="C:\Users\Mr_C\Desktop\cluster\validate.csv",
#                 data_shape=(100, 4)
#             )
#
#             data_batch = mx.nd.array(data_names, data_all)
#             # train_data = mx.io.NDArrayIter(data_all, train_lbl, batch_size, shuffle=True)
#             # test_data = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)
#             model.fit(
#                 X=train_iter,  # training data
#                 eval_data=val_iter,  # validation data
#                 batch_end_callback=mx.callback.Speedometer(batch_size, 200)  # output progress for each 200 data batches
#             )




