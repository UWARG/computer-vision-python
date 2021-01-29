# Unit tests for geolocation module

import numpy as np

# Testing lists of vector addition TODO Remove
c = np.array([0.0, 0.0, -1.0])
u = np.array([1.0, 0.0, 0.0])
v = 1 * np.cross(c, u)

m = np.array([0.0, 1.0, -1.0, -1.0])
n = np.array([0.0, 1.0, -1.0, 1.0])

points = np.size(m)

m = np.atleast_2d(m).T
n = np.atleast_2d(n).T

c = np.tile(c, (points, 1))
u = np.tile(u, (points, 1))
v = np.tile(v, (points, 1))

a = c + m*u + n*v

print(a)

# Testing concatenation
