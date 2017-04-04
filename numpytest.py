import numpy as np

a = np.array([[1, 2], [3, 4]])
print a.shape
b = np.array([[5, 6]])
print b.shape
c = np.concatenate((a, b.T), axis=1)
d = np.hstack((a, b))
print c.shape
print d.shape
