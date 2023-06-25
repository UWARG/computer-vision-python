import numpy as np

def get_distance(x_1, y_1, x_2, y_2):
    return np.sqrt(np.square(x_1-x_2)+np.square(y_1-y_2))