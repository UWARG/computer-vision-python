# Unit tests for geolocation module

import numpy as np

# Testing appending

p0 = np.array([0, 0])
g0 = np.array([1, 1])
p1 = np.array([2, 2])
g1 = np.array([3, 3])
p2 = np.array([4, 4])
g2 = np.array([5, 5])

x = 6
y = 7
p3 = np.array([8, 9])
pair3 = np.vstack((p3, [x, y]))
print(pair3)

pair0 = np.vstack((p0, g0))
print(pair0)
pair1 = np.vstack((p1, g1))
print(pair1)
pair2 = np.vstack((p2, g2))
print(pair2)

pairs = np.empty(shape=(0, 2, 2))
print(pairs)
pairs = np.concatenate((pairs, [pair1]))
print(pairs)
pairs = np.concatenate((pairs, [pair2]))
print(pairs)
pairs = np.concatenate((pairs, [pair3]))
print(pairs)
