# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np
import Network_model as mlp_model
from sklearn.utils.extmath import randomized_svd
import load_data
import random
from numpy import linalg as la
from scipy.linalg import solve

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

def shrinkage(fore, back):
    U, Sigma, VT = la.svd(fore)
    SigmaArray = np.zeros(shape=(10, 10))
    for i in range(0, SigmaArray.shape[0]):
        SigmaArray[i][i] = Sigma[i] - min(max(-back, Sigma[i]), back)

    new_M = np.dot(U, SigmaArray)
    new_M = np.dot(new_M, VT)
    return new_M

def addLinearequal(a, preds, i, j, beta, self_made_m, rowdata):
    # 生成i,j位置的方程组集合
    data = np.zeros(shape=(1, preds.shape[1] * preds.shape[1]))
    # beta*N
    data[0][i * preds.shape[1] + j] = data[0][i * preds.shape[1] + j] + beta

    # 前半部分求和,x 代表列, y代表行

    for x in range(0, preds.shape[0]):
        for y in range(0, preds.shape[0]):
            fore = preds[x][i] * preds[y][j]
            if self_made_m[x][y] == 1:
                # 将-1操作转化为结果的改变
                rowdata[i][j] = rowdata[i][j] + fore * a
                reala = a
            else:
                reala = 1 - a


            # for zi in range(0, preds.shape[1]):
            #     for zj in range(0, preds.shape[1]):
            #         # 修改系数
            #         data[0][zi * preds.shape[1] + zj] = data[0][zi * preds.shape[1] + zj] + reala * fore * (
            #             preds[x][zi] * preds[y][zj])

            metricpreds = np.dot(preds[x].reshape(preds.shape[1], 1), preds[y].reshape(1, preds.shape[1]))
            data[0] = data[0] + reala * fore * metricpreds.reshape(1,metricpreds.shape[0] * metricpreds.shape[1])

    return data, rowdata

def m_minimize_admm(x, self_made_m, M, prefix, iteration, a, beta, Lambda, k):
    # 加载f(x) model
    model_loaded = mx.model.FeedForward.load(prefix, iteration)
    print "模型加载完毕"

    # 获得新的train_label的preds
    count = 0
    preds = mx_predict(x, model_loaded)

    N = M
    mainlosslist = []
    u = np.random.randint(0, 1, size=(10, 10))
    # M = 0
    flag = True
    lastM = N
    itercount = 0
    for mi in range(0, 10):
        # # 检查m步骤进行前关于M的loss
        # f = open('minimize_m_admm.txt', 'a')
        # f.write(str(mi))
        # f.write('\n')
        # U, Sigma, VT = la.svd(M)
        # admm_m_loss = Lambda * sum(abs(Sigma))
        # admm_m_loss = admm_m_loss - (u.T*(M-N)).trace()
        # U, Sigma, VT = la.svd(M-N)
        # maxeigenvalue = max(Sigma)
        # admm_m_loss = admm_m_loss + beta / 2 * maxeigenvalue
        # f.write(str(admm_m_loss))
        # f.write('\n')
        # f.close()
        # # 检查m步骤进行前关于M的loss 结束

        # 优化M
        M = shrinkage(N+u/beta, Lambda*beta)
        # 优化M结束

        # # 检查m步骤进行前关于M的loss
        # f = open('minimize_m_admm.txt', 'a')
        # U, Sigma, VT = la.svd(M)
        # admm_m_loss = Lambda * sum(abs(Sigma))
        # admm_m_loss = admm_m_loss - (u.T * (M - N)).trace()
        # U, Sigma, VT = la.svd(M - N)
        # maxeigenvalue = max(Sigma)
        # admm_m_loss = admm_m_loss + beta / 2 * maxeigenvalue
        # f.write(str(admm_m_loss))
        # f.write('\n')
        # f.close()
        # # 检查m步骤进行前关于M的loss 结束

        # 测试loss
        # 计算总的loss
        U, Sigma, VT = la.svd(M)
        mainloss = sum(Sigma)
        calculateM = np.zeros((preds.shape[0], preds.shape[0]))
        for ti in range(0, x.shape[0]):
            for tj in range(ti, x.shape[0]):
                calculateM[ti][tj] = np.dot(np.dot(preds[ti], M), preds[tj].T)
                calculateM[tj][ti] = calculateM[ti][tj]

        for xi in range(0, x.shape[0]):
            for xj in range(0, x.shape[0]):
                if self_made_m[xi][xj] == 1:
                    mainloss = mainloss + a * (calculateM[xi][xj] - 1) * (calculateM[xi][xj] - 1)
                else:
                    mainloss = mainloss + (1 - a) * calculateM[xi][xj] * calculateM[xi][xj]



        f = open('mloss' + str(iteration) + '.txt', 'a')
        f.write(str(mainloss))
        f.write('\n')
        f.close()

        # # 检查n步骤进行前关于N的loss
        # f = open('minimize_n_admm.txt', 'a')
        # f.write(str(mi))
        # f.write('\n')
        #
        # admm_n_loss = 0
        # calculateN = np.zeros((preds.shape[0], preds.shape[0]))
        # for ti in range(0, x.shape[0]):
        #     for tj in range(ti, x.shape[0]):
        #         calculateN[ti][tj] = np.dot(np.dot(preds[ti], N), preds[tj].T)
        #         calculateN[tj][ti] = calculateN[ti][tj]
        #
        # for xi in range(0, x.shape[0]):
        #     for xj in range(0, x.shape[0]):
        #         if self_made_m[xi][xj] == 1:
        #             admm_n_loss = admm_n_loss + a * (calculateN[xi][xj] - 1) * (calculateN[xi][xj] - 1)
        #         else:
        #             admm_n_loss = admm_n_loss + (1 - a) * calculateN[xi][xj] * calculateN[xi][xj]
        #
        # admm_n_loss = admm_n_loss - (u.T * (M - N)).trace()
        # U, Sigma, VT = la.svd(M - N)
        # maxeigenvalue = max(Sigma)
        # admm_n_loss = admm_n_loss + beta / 2 * maxeigenvalue
        # f.write(str(admm_n_loss))
        # f.write('\n')
        # f.close()
        # # 检查n步骤进行前关于N的loss 结束

        # # 优化N,生成k^2个结果集
        rowdata = beta * M - u

        # 生成各个位置关于x11-xkk的方程组
        print '开始生成方程组'
        linearequaldata = 0
        for i in range(0, preds.shape[1]):
            for j in range(0, preds.shape[1]):
                # 生成该位置的方程组集合
                temppredata, rowdata = addLinearequal(a, preds, i, j, beta, self_made_m, rowdata)
                if i == 0 and j == 0:
                    linearequaldata = temppredata
                else:
                    linearequaldata = np.row_stack((linearequaldata, temppredata))

        rowdata = rowdata.reshape((rowdata.shape[0] * rowdata.shape[0], 1))
        # 结果集生成完毕

        print '已生成k^2个方程组'
        # 已生成k^2*k^2维方程组
        resultN = solve(linearequaldata, rowdata)
        print(resultN)
        print resultN.shape
        N = resultN.reshape((preds.shape[1], preds.shape[1]))
        # 优化N完毕

        # # 检查n步骤进行前关于N的loss
        # f = open('minimize_n_admm.txt', 'a')
        # admm_n_loss = 0
        # calculateN = np.zeros((preds.shape[0], preds.shape[0]))
        # for ti in range(0, x.shape[0]):
        #     for tj in range(ti, x.shape[0]):
        #         calculateN[ti][tj] = np.dot(np.dot(preds[ti], N), preds[tj].T)
        #         calculateN[tj][ti] = calculateN[ti][tj]
        #
        # for xi in range(0, x.shape[0]):
        #     for xj in range(0, x.shape[0]):
        #         if self_made_m[xi][xj] == 1:
        #             admm_n_loss = admm_n_loss + a * (calculateN[xi][xj] - 1) * (calculateN[xi][xj] - 1)
        #         else:
        #             admm_n_loss = admm_n_loss + (1 - a) * calculateN[xi][xj] * calculateN[xi][xj]
        #
        # admm_n_loss = admm_n_loss - (u.T * (M - N)).trace()
        # U, Sigma, VT = la.svd(M - N)
        # maxeigenvalue = max(Sigma)
        # admm_n_loss = admm_n_loss + beta / 2 * maxeigenvalue
        # f.write(str(admm_n_loss))
        # f.write('\n')
        # f.close()
        # # 检查n步骤进行前关于N的loss 结束

        u = u - beta * ( M - N )
        # 优化u完毕

    return M
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


