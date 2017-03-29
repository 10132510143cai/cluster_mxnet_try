# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import urllib
import gzip
import struct
import matplotlib.pyplot as plt
import mxnet as mx
import random
import pandas as pd

class Auc(mx.metric.EvalMetric):
    def __init__(self):
        super(Auc, self).__init__('loss')

    def update(self, labels, preds):
        pred = preds[0].asnumpy().reshape(-1)
        # print pred
        self.sum_metric += np.sum(pred)
        # print "loss: ",
        # print self.sum_metric
        self.num_inst += len(pred)

class Batch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class MyDataIter(mx.io.DataIter):
    def __init__(self, z, label, batch_size, k, self_made_m):
        super(MyDataIter, self).__init__()
        self.data = z
        self.label = label
        self.batch_size = batch_size
        self.k = k
        self.self_made_m = self_made_m
        # self.provide_data = [('dataxi', (1,)), ('dataxj', (1,)), ('isinM', (1, 1)), ('M', (k, k))]
        self.provide_data = [('dataxi', (batch_size,  784)), ('dataxj', (batch_size,  784)), ('isinM', (batch_size, ))]
        # 输出数据的shape
        self.provide_label = []

    def __iter__(self):
        for k in range(self.data.shape[0] * self.data.shape[0] / self.batch_size):
            dataxi = []
            dataxj = []
            isinM = []

            # m = np.ones(self.k, self.k)
            for i in range(self.batch_size):
                j = k * self.batch_size + i
                tempxi = int(j / self.data.shape[0])
                tempxj = int(j - tempxi * self.data.shape[0])

                dataxi.append(self.data[tempxi])
                dataxj.append(self.data[tempxj])


                if self.self_made_m[tempxi][tempxj] == 1:
                    isinM.append(1)
                else:
                    isinM.append(0)

            # data_all = [mx.nd.array(dataxi), mx.nd.array(dataxj), mx.nd.array(isinM)]
            # data_names = ['dataxi', 'dataxj', 'isinM']
            data_all = [mx.nd.array(dataxi), mx.nd.array(dataxj), mx.nd.array(isinM)]
            data_names = ['dataxi', 'dataxj', 'isinM']

            # print "epoch",
            # print k

            label_all = []
            label_names = []

            data_batch = Batch(data_names, data_all, label_names, label_all)
            yield data_batch

    def reset(self):
        pass


def download_data(url, force_download=True):
    fname = url.split("/")[-1]
    if force_download or not os.path.exists(fname):
        urllib.urlretrieve(url, fname)
    return fname

def read_data(label_path, image_path):
    with gzip.open(label_path) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(image_path, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return label, image

def to4d(img):
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

# 加载数据集
def load_data_main():
    # path = 'C:\Users\JimGrey\PycharmProjects\cluster_mxnet_try/mnist/'
    path = 'C:\Users\Mr_C\Desktop\MNIST/'
    (train_lbl, train_img) = read_data(
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz')
    batch_size = 100
    x = mx.io.NDArrayIter(to4d(train_img)).data_list[0]
    x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))

    val_x = mx.io.NDArrayIter(to4d(val_img)).data_list[0]
    val_x = val_x.reshape((val_x.shape[0], val_x.shape[1]*val_x.shape[2]*val_x.shape[3]))

    train_label = train_lbl
    val_label = val_lbl
    train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

    return x, val_x, train_iter, val_iter, train_label, val_label

def get_data(train_z, val_z, train_label, val_label, batch_size, k, self_made_m):
    return MyDataIter(train_z, train_label, batch_size, k, self_made_m), MyDataIter(val_z, val_label, batch_size, k, self_made_m)

