import numpy as np
from matplotlib import pyplot
# реализации метода подсчета косинусного расстояния между векторными представлениями текста


def cos_dist(vect1, vect2):
    vectors_dot_product = np.dot(vect1, vect2)
    vec1_sq = np.sqrt(np.multiply(vect1, vect1).sum())
    vec2_sq = np.sqrt(np.multiply(vect2, vect2).sum())
    sim = np.divide(vectors_dot_product, vec1_sq * vec2_sq)
    dist = 1 - sim
    return dist


def plot(name, resized_vectors, tokens):
    pyplot.scatter(resized_vectors[:, 0], resized_vectors[:, 1])
    words = tokens
    pyplot.title(name)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(resized_vectors[i, 0], resized_vectors[i, 1]))
    pyplot.show()
