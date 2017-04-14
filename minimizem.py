# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
import Network_model as mlp_model
from sklearn.utils.extmath import randomized_svd
import load_data
import random
from numpy import linalg as la

def sub_dict(dic, keys):
    return {k: dic[k] for k in keys}

def mx_predict(data, model):
    from collections import namedtuple
    Batch = namedtuple('Batch', ['data'])
    preds = []

    k = 10
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


def m_minimize(x, self_made_m, M, prefix, iteration, a, Gama, Lambda, k):
    # 加载f(x) model
    model_loaded = mx.model.FeedForward.load(prefix, iteration)
    print "模型加载完毕"

    # 获得新的train_label的preds
    count = 0
    preds = mx_predict(x, model_loaded)

    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[0]):
            templeft = np.dot(preds[i].reshape((1, preds.shape[1])), M)
            templeft = np.dot(templeft, preds[j].T.reshape(preds.shape[1], 1))

            tempright = np.dot(preds[i].T.reshape(preds.shape[1], 1), preds[j].reshape((1, preds.shape[1])))

            if i == 0 and j == 0:
                if self_made_m[i][j] == 1:
                    count = a * (templeft - 1) * tempright
                else:
                    count = (1 - a) * templeft * tempright
            else:
                if self_made_m[i][j] == 1:
                    count = count + a * (templeft - 1)  * tempright
                else:
                    count = count + (1 - a) * templeft * tempright

    np.savetxt('calculateM/count- ' + str(k), count, newline='\n')
    count2 = M - 2 * Gama * count
    U, Sigma, VT = la.svd(count2)

    print Sigma
    np.savetxt('calculateM/Sigma-' + str(k), Sigma, newline='\n')
    SigmaArray = np.zeros(shape=(10, 10))
    for i in range(0, SigmaArray.shape[0]):
        SigmaArray[i][i] = Sigma[i] - min(max(-Lambda * Gama, Sigma[i]), Lambda * Gama)

    print SigmaArray
    new_M = np.dot(U, SigmaArray)
    new_M = np.dot(new_M, VT)
    return new_M, preds


def m_minimize_bynetwork(x, val_x, train_label, val_label, batch_size, self_made_m, prefix, iteration, num_epoch, learning_rate, k, a):
    # 第二次初始化时使用加载的模型
    model_loaded = mx.model.FeedForward.load(prefix, iteration)
    # 加载初始化参数
    params = model_loaded.get_params()  # get model paramters
    arg_params = params['arg_params']

    # 初始化训练集
    train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, self_made_m)
    # 加载优化M的网络
    net = mlp_model.modelM(k, a)

    model = mx.model.FeedForward(
        symbol=net,  # network structure
        num_epoch=num_epoch+100,  # number of data passes for training
        learning_rate=learning_rate,  # learning rate of SGD
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        # optimizer=SelfOptimizer,
        # optimizer=optimizer,
        arg_params=arg_params
    )

    metric = load_data.Auc()
    print "网络加载完成，开始训练"
    model.fit(
        X=train,  # training data
        eval_metric=metric,
        # eval_data=test,  # validation data
        batch_end_callback=mx.callback.Speedometer(batch_size, 600 * 600 / batch_size, iteration=iteration,
                                                   minwhich='m-')
        # output progress for each 200 data batches
    )

    model.save(prefix+'-M', iteration)
    model_loaded = mx.model.FeedForward.load(prefix+'-M', iteration)
    params = model_loaded.get_params()  # get model paramters
    arg_params = params['arg_params']
    m = arg_params['M'].asnumpy()
    return m

# 测试用main
# x, val_x, train_iter, val_iter, train_label, val_label = load_data.load_data_main()
#
# # 进行数据的shuffle
# train_label = train_label.reshape(train_label.shape[0], 1)
# numpyx = x.asnumpy()
# training_data = np.concatenate((numpyx, train_label), axis=1)
# random.shuffle(training_data)
#
# x = training_data[:, :-1]
# train_label = training_data[:, -1]
# x = x[:600]
# val_x = val_x[:600]
# train_label = train_label[:600]
# val_label = val_label[:600]
# print train_label[0:10]
# print "数据加载完毕"
# M = np.random.randint(0, 1, size=(10, 10))
# m_minimize(x, train_label, M, prefix, iteration, a, Gama, Lambda, k)


