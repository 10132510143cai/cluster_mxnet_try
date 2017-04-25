# -*- coding: utf-8 -*-
import logging
logging.getLogger().setLevel(logging.DEBUG)
import mxnet as mx

def model_main(k, a):
    # 设置共享的参数
    fc1_weight = mx.sym.Variable('fc1_weight')
    fc1_bias = mx.sym.Variable('fc1_bias')

    fc2_weight = mx.sym.Variable('fc2_weight')
    fc2_bias = mx.sym.Variable('fc2_bias')

    # 搭建对xi进行非线性映射的模型
    xipart_net = mx.sym.Variable(name='dataxi')
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xipart_net = mx.sym.Activation(xipart_net, name='relu1', act_type="relu")
    xipart_net = mx.sym.Dropout(data=xipart_net, p=0.2)
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k, name="fc2")  # dimension batch_size*k

    # 搭建对xj进行非线性映射的模型,两个模型相同
    xjpart_net = mx.sym.Variable(name='dataxj')
    xjpart_net = mx.sym.FullyConnected(data=xjpart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xjpart_net = mx.sym.Activation(data=xjpart_net, name='relu1', act_type="relu")
    xjpart_net = mx.sym.Dropout(data=xjpart_net, p=0.2)
    xjpart_net = mx.sym.FullyConnected(data=xjpart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k, name="fc2")  # dimension 1*k

    # 学习率为0的参数M
    m = mx.sym.Variable(name='M', shape=(k, k), attr={'lr_mult': '0'})  # M k*k

    # 开始进行矩阵乘运算，以下为loss 的计算形式
    xipart_net = mx.symbol.dot(lhs=xipart_net, rhs=m)  # dimension batchsize * k
    net = xipart_net * xjpart_net  # dimension batchsize * k
    net = mx.symbol.sum(data=net, axis=1)  # dimension batchsize * 1

    # 用于判断是否是同一个类的输入isinM
    isinm = mx.sym.Variable('isinM')  #dimenstion batchsize * 1

    isinmpart = net * isinm - isinm
    isinmpart = a * isinmpart * isinmpart  # 相当于单个值得二范数
    isinmpart = mx.symbol.sum_axis(isinmpart)  # dimension 1*1

    isnotinmpart = net * (1 - isinm)
    isnotinmpart = (1 - a) * isnotinmpart * isnotinmpart  # 相当于单个值得二范数
    isnotinmpart = mx.symbol.sum_axis(isnotinmpart)  # dimension 1*1

    loss_function = isinmpart + isnotinmpart

    # 将loss_function设为最终loss
    ls = mx.sym.MakeLoss(loss_function)
    print "模型构建完成"
    return ls

def modelfx(k):
    # 设置参数
    fc1_weight = mx.sym.Variable('fc1_weight')
    fc1_bias = mx.sym.Variable('fc1_bias')

    fc2_weight = mx.sym.Variable('fc2_weight')
    fc2_bias = mx.sym.Variable('fc2_bias')

    # 搭建网络
    xipart_net = mx.sym.Variable(name='data')
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xipart_net = mx.sym.Activation(xipart_net, name='relu1', act_type="relu")
    xipart_net = mx.sym.Dropout(data=xipart_net, p=0.2)
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k,
                                       name="fc2")  # dimension batch_size*k
    # 返回网络结构
    return xipart_net

def modelM(k, a):
    # 设置共享的参数,将其学习率设为0
    fc1_weight = mx.sym.Variable('fc1_weight', attr={'lr_mult': '0'})
    fc1_bias = mx.sym.Variable('fc1_bias', attr={'lr_mult': '0'})

    fc2_weight = mx.sym.Variable('fc2_weight', attr={'lr_mult': '0'})
    fc2_bias = mx.sym.Variable('fc2_bias', attr={'lr_mult': '0'})

    # 搭建对xi进行非线性映射的模型
    xipart_net = mx.sym.Variable(name='dataxi')
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xipart_net = mx.sym.Activation(xipart_net, name='relu1', act_type="relu")
    xipart_net = mx.sym.Dropout(data=xipart_net, p=0.2)
    xipart_net = mx.sym.FullyConnected(data=xipart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k,
                                       name="fc2")  # dimension batch_size*k

    # 搭建对xj进行非线性映射的模型,两个模型相同
    xjpart_net = mx.sym.Variable(name='dataxj')
    xjpart_net = mx.sym.FullyConnected(data=xjpart_net, weight=fc1_weight, bias=fc1_bias, num_hidden=64, name="fc1")
    xjpart_net = mx.sym.Activation(data=xjpart_net, name='relu1', act_type="relu")
    xjpart_net = mx.sym.Dropout(data=xjpart_net, p=0.2)
    xjpart_net = mx.sym.FullyConnected(data=xjpart_net, weight=fc2_weight, bias=fc2_bias, num_hidden=k,
                                       name="fc2")  # dimension 1*k

    # 学习率为0的参数M
    m = mx.sym.Variable(name='M', shape=(k, k))  # M k*k

    # 开始进行矩阵乘运算，以下为loss 的计算形式
    xipart_net = mx.symbol.dot(lhs=xipart_net, rhs=m)  # dimension batchsize * k
    net = xipart_net * xjpart_net  # dimension batchsize * k
    net = mx.symbol.sum(data=net, axis=1)  # dimension batchsize * 1

    # 用于判断是否是同一个类的输入isinM
    isinm = mx.sym.Variable('isinM')  # dimenstion batchsize * 1

    isinmpart = net * isinm - isinm
    isinmpart = a * isinmpart * isinmpart  # 相当于单个值得二范数
    isinmpart = mx.symbol.sum_axis(isinmpart)  # dimension 1*1

    isnotinmpart = net * (1 - isinm)
    isnotinmpart = (1 - a) * isnotinmpart * isnotinmpart  # 相当于单个值得二范数
    isnotinmpart = mx.symbol.sum_axis(isnotinmpart)  # dimension 1*1

    loss_function = isinmpart + isnotinmpart

    # 将loss_function设为最终loss
    ls = mx.sym.MakeLoss(loss_function)
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