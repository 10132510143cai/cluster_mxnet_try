# -*- coding: utf-8 -*-
import random
import numpy as np
'''
init_m函数
param train_label 与训练集顺序一致的label集
return 代表关联的矩阵常量M
'''
def init_m(train_label):
    # 初始化一个hashmap
    hashmap = {}
    for i in range(0, train_label.shape[0]):
        # 如果该label不在hashmap中，则将label添加入hashmap中，创建一个list,将该训练数据的index放入list中
        if train_label[i] not in hashmap:
            hashmap[train_label[i]] = [i]

        # 如果label已在hashmap中，则将训练数据的index放入list中
        else:
            keylist = hashmap.get(train_label[i])
            keylist.append(i)
            hashmap[train_label[i]] = keylist

    # 根据hashmap的信息创建矩阵
    arrayM = np.zeros((train_label.shape[0], train_label.shape[0]))
    for key in hashmap:
        keylist = hashmap.get(key)
        for i in range(0, len(keylist)):
            for j in range(i, len(keylist)):
                arrayM[keylist[i]][keylist[j]] = 1
                arrayM[keylist[j]][keylist[i]] = 1

    print arrayM
    # 返回i,j条数据是否关联的矩阵
    return arrayM

def init_m_random(train_label):
    # 初始化一个hashmap
    hashmap = {}
    for i in range(0, train_label.shape[0]):
        # 如果该label不在hashmap中，则将label添加入hashmap中，创建一个list,将该训练数据的index放入list中
        if train_label[i] not in hashmap:
            hashmap[train_label[i]] = [i]

        # 如果label已在hashmap中，则将训练数据的index放入list中
        else:
            keylist = hashmap.get(train_label[i])
            keylist.append(i)
            hashmap[train_label[i]] = keylist

    arrayM = np.zeros((train_label.shape[0], train_label.shape[0]))

    # 根据hashmap的信息以一定概率创建矩阵
    for key in hashmap:
        keylist = hashmap.get(key)
        for i in range(0, len(keylist)):
            for j in range(i, len(keylist)):
                if random.random() <= 0.8:
                    arrayM[keylist[i]][keylist[j]] = 1
                    arrayM[keylist[j]][keylist[i]] = 1
                elif keylist[i] == keylist[j]:
                    arrayM[keylist[i]][keylist[j]] = 1

    print arrayM
    # 返回i,j条数据是否关联的矩阵
    return arrayM