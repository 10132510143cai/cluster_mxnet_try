# -*- coding: utf-8 -*-

import Network_model as mlp_model
import mxnet as mx
import load_data
import initM
import logging

def fx_minimize(x, val_x, train_label, val_label, self_made_m, M, k, a, batch_size, prefix, iteration, num_epoch, learning_rate, train_data_count):
    logging.getLogger().setLevel(logging.DEBUG)
    train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, self_made_m)
    print "训练集+验证集生成完成"
    # 加载训练网络
    net = mlp_model.model_main(k, a)
    internals = net.get_internals()
    arg_names = internals.list_arguments()
    lr_dict = dict()
    for arg_name in arg_names:
        if arg_name == 'M':
            lr_dict[arg_name] = 0

    if iteration == 100:
        # 训练模型
        t7 = load_data.SelfOptimizer(learning_rate=0.02, rescale_grad=(1.0 / batch_size))
        sgd = mx.optimizer.create('sgd', learning_rate=0.02)
        optimizer = mx.optimizer.create('sgd', learning_rate=0.02,
                               rescale_grad=(1.0 / batch_size))
        optimizer.set_lr_mult(lr_dict)
        model = mx.model.FeedForward(
            symbol=net,  # network structure
            num_epoch=num_epoch,  # number of data passes for training
            learning_rate=learning_rate,  # learning rate of SGD
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            # optimizer=SelfOptimizer,
            # optimizer=optimizer,
            arg_params={'M': M}
        )
        metric = load_data.Auc()

        print "网络加载完成，开始训练"
        model.fit(
            X=train,  # training data
            eval_metric=metric,
            # eval_data=test,  # validation data
            batch_end_callback=mx.callback.Speedometer(batch_size, train_data_count * train_data_count / batch_size,
                                                       iteration=iteration)
            # output progress for each 200 data batches
        )

        model.save(prefix, iteration)
    else:
        # 使用之前一次的模型
        model_loaded = mx.model.FeedForward.load(prefix, iteration -100)
        # 加载初始化参数
        params = model_loaded.get_params()  # get model paramters
        arg_params = params['arg_params']

        model = mx.model.FeedForward(
            symbol=net,  # network structure
            num_epoch=num_epoch,  # number of data passes for training
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
            batch_end_callback=mx.callback.Speedometer(batch_size, train_data_count * train_data_count / batch_size,
                                                       iteration=iteration,
                                                       minwhich='fx-')
            # output progress for each 200 data batches
        )
        model.save(prefix, iteration)

