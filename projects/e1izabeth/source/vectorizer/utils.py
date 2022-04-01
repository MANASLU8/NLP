import math
import os

import numpy as np
from matplotlib import pyplot
from nltk.corpus import stopwords
from sklearn.decomposition import PCA

eng_stopwords = stopwords.words("english")


total_docs_count = sum(len(files) for _, _, files in os.walk('../assets/train'))


def calc_tf_idf(n, tokens_len, word_len):
    tf_value = n / tokens_len
    idf_value = math.log10(total_docs_count / word_len)
    return tf_value * idf_value


def cosine_distance(vector1, vector2):
    vectors_dot_product = np.dot(vector1, vector2)
    vec1_sq = np.sqrt(np.dot(vector1, vector1).sum())
    vec2_sq = np.sqrt(np.dot(vector2, vector2).sum())
    cos_similarity = math.fabs(np.divide(vectors_dot_product, vec1_sq * vec2_sq))
    cos_distance = 1 - cos_similarity
    return cos_distance
    #return (1 - math.fabs(vector1.dot(vector2)/ (np.linalg.norm(vector1) * np.linalg.norm(vector2))))


def pca_reduce(vectors, dim):
    pca = PCA(n_components=dim)
    result = pca.fit_transform(vectors)
    return result


def create_plot(title, reduced_embeddings, all_tokens):
    pyplot.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    words = all_tokens
    pyplot.title(title)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    pyplot.show()
