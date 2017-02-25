# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import urllib
import gzip
import struct
import matplotlib.pyplot as plt
import mxnet as mx
import pandas as pd

class Batch(object):
    def __init__(self, data_names, data):
        self.data = data
        self.data_names = data_names

class MyDataIter(mx.io.DataIter):
    def __init__(self, z, x, batch_size, k):
        super(MyDataIter, self).__init__()
        print"初始化"
        self.data = z
        self.batch_size = batch_size
        self.x = x
        self.k = k
        # self.provide_data = [('dataxi', (1,)), ('dataxj', (1,)), ('isinM', (1, 1)), ('M', (k, k))]
        self.provide_data = [('dataxi', (batch_size,)), ('M', (batch_size, 10, 10))]
        # 输出数据的shape
        self.provide_label = []

    def __iter__(self):
        print"进入迭代步骤"
        for k in range(len(self.data) * len(self.data)):
            dataxi = []
            dataxj = []
            isinM = []
            mList = []
            print self.k
            m = np.ones(self.k, self.k)
            for i in range(self.batch_size):
                j = k * self.batch_size + i
                tempxi = int(j / len(self.data))
                tempxj = j - tempxi * len(self.data)
                dataxi.append(self.data[tempxi])
                dataxj.append(self.data[tempxj])
                mList.append(m)

                if self.x[tempxi].all() == self.x[tempxj].all():
                    isinM.append(1)
                else:
                    isinM.append(0)

            data_all = [mx.nd.array(dataxi), mx.nd.array(dataxj), mx.nd.array(isinM)]
            data_names = ['dataxi', 'dataxj', 'isinM']
            # label_all = [mx.nd.array(labels)]
            # label_names = ['label']

            data_batch = Batch(data_names, data_all)
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
    path = 'C:/Users/Mr_C/Desktop/MNIST/'
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

def get_data(train_z, val_z, x, batch_size, k):
    return MyDataIter(train_z, x, batch_size, k), MyDataIter(val_z, x, batch_size, k)

