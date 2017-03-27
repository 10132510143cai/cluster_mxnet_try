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
def init_m(n):
    arrayM = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, i+1):
            if i == j:
                arrayM[i][j] = 1
            elif random_number() == 1:
                arrayM[i][j] = 1
                arrayM[j][i] = 1

    print arrayM
    return arrayM