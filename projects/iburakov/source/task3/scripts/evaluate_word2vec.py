import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from paths import word2vec_model_filepath
from task3.cosine_distance import cosine_distance
from task3.token_dictionary import load_token_dictionary

TEST_WORDS = [
    ["Windows", ["OS", "MS-Windows", "Linux", "UNIX", "program", "curtains", "God", "inevitable", "Rockefeller"]],
    ["independent", [
        "sovereignty", "politics", "nations", "referendum", "territory", "Russia", "US", "HR", "PostScript", "baseball",
        "fprintf",
    ]],
    ["health", ["medical", "sports", "doctor", "blood", "heart", "city", "North", "communications", "Eric", "mailing"]],
]

_dct = load_token_dictionary()


def evaluate_vectorizer(token, other_tokens, vectorizer, plot_embedding_vectorizer_calls=10000):
    token_vec = vectorizer(token)
    other_token_vecs = [vectorizer(t) for t in other_tokens]
    other_token_dists = [cosine_distance(token_vec, v) for v in other_token_vecs]
    print(f"=== DISTANCES TO {token}:")
    print(pd.Series(other_token_dists, index=other_tokens).sort_values())

    pca = PCA(n_components=2)
    fitting_data = []
    for t in _dct.df.token.sample(n=plot_embedding_vectorizer_calls):
        try:
            fitting_data.append(vectorizer(t))
        except KeyError:
            continue
    pca.fit(fitting_data)
    token_vec_pca = pca.transform([token_vec])[0]
    other_token_vecs_pca = pca.transform(other_token_vecs)
    # token_vec_pca, *other_token_vecs_pca = pca.fit_transform([token_vec, *other_token_vecs])

    plt.scatter(*token_vec_pca)
    plt.annotate(token, token_vec_pca)
    x, y = zip(*other_token_vecs_pca)
    plt.scatter(x, y)
    for t, (xi, yi) in zip(other_tokens, zip(x, y)):
        plt.annotate(t, (xi, yi))
    plt.grid()
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.show()


if __name__ == '__main__':
    w2v = Word2Vec.load(str(word2vec_model_filepath))


    def w2v_vectorizer(t):
        return w2v.wv[t]


    for t, other_t in TEST_WORDS:
        evaluate_vectorizer(t, other_t, w2v_vectorizer)
