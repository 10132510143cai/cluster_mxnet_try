# -*- coding: utf-8 -*-

import numpy as np
import mxnet as mx

def model_main(k, a):
    fc1_weight = mx.sym.Variable('fc1_weight')
    fc1_bias = mx.sym.Variable('fc1_bias')

    fc2_weight = mx.sym.Variable('fc2_weight')
    fc2_bias = mx.sym.Variable('fc2_bias')

    xipart_net = mx.sym.Variable(name='dataxi')
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xipart_net = mx.sym.Activation(xipart_net, name='relu1', act_type="relu")
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k, name="fc2")  # dimension 1*k

    xjpart_net = mx.sym.Variable(name='dataxj')

    xjpart_net = mx.sym.FullyConnected(data=xjpart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xjpart_net = mx.sym.Activation(data=xjpart_net, name='relu1', act_type="relu")
    xjpart_net = mx.sym.FullyConnected(data=xjpart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k, name="fc2")  # dimension 1*k

    #m = mx.sym.Variable(name='M')  # M k*k
    # m = mx.nd.array([1,2,3,4])
    m = mx.sym.Variable(name='M', shape=(10, 10))  # M k*k

    xipart_net = mx.symbol.dot(lhs=xipart_net, rhs=m)  # dimension 1*k
    # xjpart_net = mx.sym.transpose(data=xjpart_net, name='fc4')  # xj dimension k*1

    # net = xipart_net * xjpart_net
    # net = mx.sym.sum(data=net)  # dimension 1*1
    #
    # isinm = mx.sym.Variable('isinM')  #dimenstion batchsize*1
    #
    # # 设置loss_layer
    # loss_function = isinm * a * (net - 1) + (1-isinm) * (1 - a) * net
    # ls = mx.sym.MakeLoss(loss_function)

    ls = mx.sym.MakeLoss(xipart_net)
    print "模型构建完成"
    return ls

# # 测试main
# k = 1
# a = 0.5
#
# loss_function = model_main(k, a)
# input_shapes = {'dataxi': (1, 10), 'dataxj': (1, 10), 'isinM':(1, 1), 'M':(k, k)}  # 定义输入的shape
# arg_shapes, out_shapes, aux_shapes = loss_function.infer_shape(**input_shapes)
# print arg_shapes
# print out_shapes
# print aux_shapes
# print 'success'