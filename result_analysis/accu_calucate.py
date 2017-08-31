# -*- coding: utf-8 -*-

import numpy as np

def accu_calcucate(iteration, threshold, traindatacount):
    calculateM = np.loadtxt('../calculateM'+str(iteration)+'-'+str(traindatacount))
    constantM = np.loadtxt('../constantM'+str(iteration)+'-'+str(traindatacount))

    diff = 0
    resultM = abs(calculateM) - constantM
    resultM = abs(resultM)
    np.savetxt('result-M' + str(iteration), resultM, fmt=['%s'] * resultM.shape[1], newline='\n')

    for x in range(0, resultM.shape[0]):
        for y in range(0, resultM.shape[1]):
            if resultM[x][y] > 1 - threshold:
                diff = diff +1

    accu = (resultM.shape[0] * resultM.shape[1] - diff) / float(resultM.shape[0] * resultM.shape[1])
    f = open('accu.txt', 'a')
    f.write(str(iteration))
    f.write(' ')
    f.write(str(threshold))
    f.write(' ')
    f.write(str(accu))
    f.write('\n')
    f.close()

for j in range(5, 10):
    for i in range(1, 13):
        accu_calcucate(i*100, j/10.0, 600)
