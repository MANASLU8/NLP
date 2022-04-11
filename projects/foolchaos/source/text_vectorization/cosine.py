import numpy as np


def cosine_distance(vector1, vector2):
    return 1 - np.divide(np.dot(vector1, vector2),
                         np.sqrt(np.multiply(vector1, vector1).sum()) * np.sqrt(np.multiply(vector2, vector2).sum()))
