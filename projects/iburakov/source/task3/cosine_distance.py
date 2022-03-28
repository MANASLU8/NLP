import numpy as np


def cosine_distance(x, y):
    return 1 - np.dot(x, y) / (np.sqrt(np.sum(np.power(x, 2))) * np.sqrt(np.sum(np.power(y, 2))))
