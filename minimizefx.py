# -*- coding: utf-8 -*-

import Network_model as mlp_model
import mxnet as mx
import load_data
import initM
import logging

def fx_minimize(x, val_x, train_label, val_label, M, k, a, batch_size, prefix, iteration, num_epoch, learning_rate):
    logging.getLogger().setLevel(logging.DEBUG)
    self_made_m = initM.init_m_random(train_label)
    train, test = load_data.get_data(x, val_x, train_label, val_label, batch_size, self_made_m)
    print "训练集+验证集生成完成"

    # 加载训练网络
    net = mlp_model.model_main(k, a)

    # 训练模型
    model = mx.model.FeedForward(
        symbol=net,  # network structure
        num_epoch=num_epoch,  # number of data passes for training
        learning_rate=learning_rate,  # learning rate of SGD
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        arg_params={'M': M}
    )
    metric = load_data.Auc()

    print "网络加载完成，开始训练"
    model.fit(
        X=train,  # training data
        eval_metric=metric,
        # eval_data=test,  # validation data
        batch_end_callback=mx.callback.Speedometer(batch_size, 600*600/batch_size, iteration=iteration)  # output progress for each 200 data batches
    )

    model.save(prefix, iteration)
    return self_made_m
