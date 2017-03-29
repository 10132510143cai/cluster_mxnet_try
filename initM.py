# -*- coding: utf-8 -*-
import random
import numpy as np
'''
生成一个随机数，以0.3的概率生成连接组
'''
def random_number():
    a = random.random()
    if a <= 0.3:
        return 1
    else:
        return 0

'''
init_m函数是生成包含众多二元组的M，M中存在的二元组（i,j）代表i,j有联系
n  数据的维数
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