# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl
# Use numpy to load the data contained in the file
# ’fakedata.txt’ into a 2-D array called data
# data = np.loadtxt('data/epochloss-100.txt')
data = np.loadtxt('data/mloss100.txt')
data2 = np.loadtxt('data/mloss200.txt')
data3 = np.loadtxt('data/mloss300.txt')

# plot the first column as x, and second column as y

pl.plot(data, 'r--', label='first')
pl.plot(data2, 'b-*', label='second')
pl.plot(data3, 'y-o', label='third')
pl.ylabel('loss')
pl.legend()
pl.show()