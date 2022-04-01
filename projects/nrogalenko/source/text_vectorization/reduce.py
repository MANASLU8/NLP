from sklearn.decomposition import PCA


def pca_reduce(vectors, dim):
    # fit PCA model to the vectors
    pca = PCA(n_components=dim)
    result = pca.fit_transform(vectors)
    return result
