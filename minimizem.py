# -*- coding: utf-8 -*-

import mxnet as mx
import load_data
import random
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

def sub_dict(dic, keys):
    return {k: dic[k] for k in keys}

def mx_predict(data, model):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    preds = []

    # 获得原来的网络设计
    k = 10
    # a = 0.7
    # network = mlp_model.model_main(k, a)
    # all_layers = network.get_internals()  # get internal output
    # print all_layers.list_outputs()
    # outputlist = all_layers.list_outputs()
    # fc2outputindex = outputlist.index('fc2_output')
    # my_sym = all_layers[fc2outputindex]

    # 重新设计相同的网络
    network = mlp_model.modelfx(k)

    my_mod = mx.mod.Module(symbol=network, context=mx.cpu())
    my_mod.bind(for_training=False, data_shapes=[('data', (1, 784))])

    params = model.get_params()  # get model paramters
    arg_params = params['arg_params']
    # print arg_params

    keys = ['fc1_weight', 'fc2_weight', 'fc1_bias', 'fc2_bias']
    # keys = ['fc_bias', 'fc_weight', 'fc1_x_bias', 'fc1_x_weight','fc2_x_bias', 'fc2_x_weight']
    arg_params = sub_dict(arg_params, keys)

    aux_params = params['aux_params']

    my_mod.set_params(arg_params, aux_params)

    for i in range(data.shape[0]):
        my_mod.forward(Batch([mx.nd.array(data[i].reshape(1, data.shape[1]))]))
        out = my_mod.get_outputs()[0].asnumpy()
        preds.append(out[0])

    return np.asarray(preds)


import Network_model as mlp_model
prefix = 'mymodel'
iteration = 100


# 加载f(x) model
model_loaded = mx.model.FeedForward.load(prefix, iteration)
print "模型加载完毕"
# 加载训练集以及数据集X
x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()

# 进行数据的shuffle
train_label = train_label.reshape(train_label.shape[0], 1)
numpyx = x.asnumpy()
training_data = np.concatenate((numpyx, train_label), axis=1)
random.shuffle(training_data)

x = training_data[:, :-1]
train_label = training_data[:, -1]
x = x[:600]
val_x = val_x[:600]
train_label = train_label[:600]
val_label = val_label[:600]
print train_label[0:10]
print "数据加载完毕"

batch_size = 6

# 获得新的train_label的preds
count = 0
preds = mx_predict(x, model_loaded)

M = np.random.randint(0, 1, size=(10, 10))
a = 0.7
y = 0.5

for i in range(0, x.shape[0]):
    for j in range(0, x.shape[0]):
        templeft = np.dot(preds[i].reshape((1, preds.shape[1])), M)
        templeft = np.dot(templeft, preds[j].T.reshape(preds.shape[1], 1))

        tempright = np.dot(preds[i].T.reshape(preds.shape[1], 1), preds[j].reshape((1, preds.shape[1])))

        if i == 0 and j == 0:
            if train_label[i] == train_label[j]:
                count = a * (templeft - 1) * tempright
            else:
                count = (1 - a) * templeft * tempright
        else:
            if train_label[i] == train_label[j]:
                count = count + a * (templeft - 1) * tempright
            else:
                count = count + (1 - a) * templeft * tempright

count = M - y * count
U, Sigma, VT = randomized_svd(count, n_components=10,
                              n_iter=5,
                              random_state=None)
print "U", U.shape
print "Sigma", Sigma.shape
print "V", VT.shape
# lossresult = model_loaded.predict(train)
# print lossresult.shape
# for i in range(0, self_made_m.shape[0]):
#     for j in range(0, self_made_m.shape[1]):
#         train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, self_made_m)
#         lossresult = model_loaded.predict(train)
#
#         if self_made_m[i][j] == 1:
#             train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, self_made_m)
#             lossresult = model_loaded.predict(train)
#             print lossresult
#         else:
#             data_all = [mx.nd.array(training_data[i]), mx.nd.array(training_data[j]), mx.nd.array([0])]
#             data_batch = load_data.Batch(data_names, data_all, label_names, label_all)
#             lossresult = model_loaded.predict(data_batch)
#             print lossresult


