import numpy as np


def cosine_distance(vector1, vector2):
    vectors_dot_product = np.dot(vector1, vector2)
    vec1_sq = np.sqrt(np.multiply(vector1, vector1).sum())
    vec2_sq = np.sqrt(np.multiply(vector2, vector2).sum())
    cos_similarity = np.divide(vectors_dot_product, vec1_sq * vec2_sq)
    cos_distance = 1 - cos_similarity
    return cos_distance

