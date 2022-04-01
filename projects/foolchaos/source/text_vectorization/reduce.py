from sklearn.decomposition import PCA


def pca_reduce(vectors, dim):
    return PCA(n_components=dim).fit_transform(vectors)
