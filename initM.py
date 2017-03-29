# -*- coding: utf-8 -*-
import random
import numpy as np
'''
init_m函数
param train_label 与训练集顺序一致的label集
return 代表关联的矩阵常量M
'''
def init_m(train_label):
    hashmap = {}
    for i in range(0, train_label.shape[0]):
        if train_label[i] not in hashmap:
            hashmap[train_label[i]] = [i]
        else:
            keylist = hashmap.get(train_label[i])
            keylist.append(i)
            hashmap[train_label[i]] = keylist

    arrayM = np.zeros((train_label.shape[0], train_label.shape[0]))

    for key in hashmap:
        keylist = hashmap.get(key)
        for i in range(0, len(keylist)):
            for j in range(i, len(keylist)):
                arrayM[keylist[i]][keylist[j]] = 1
                arrayM[keylist[j]][keylist[i]] = 1

    print arrayM
    return arrayM