from sklearn.decomposition import PCA

from task3.scripts.evaluate_word2vec import evaluate_vectorizer, TEST_WORDS
from task3.scripts.train_word2vec import W2V_VECTOR_SIZE
from task3.token_dictionary import load_token_dictionary
from task3.vectorizing_simple import vectorize_text_simple

if __name__ == '__main__':
    _dct = load_token_dictionary()

    print(f"Training PCA...")
    pca = PCA(n_components=W2V_VECTOR_SIZE)
    fitting_data = []
    for t in _dct.df.token.sample(n=1000):
        try:
            fitting_data.append(vectorize_text_simple(t))
        except KeyError:
            continue
    for t, other_t in TEST_WORDS:
        fitting_data.append(vectorize_text_simple(t))
        fitting_data.extend(vectorize_text_simple(t) for t in other_t)
    pca.fit(fitting_data)


    def embedded_vectorizer(token: str):
        return pca.transform([vectorize_text_simple(token)])[0]


    print("Evaluating...")
    for t, other_t in TEST_WORDS:
        evaluate_vectorizer(t, other_t, embedded_vectorizer, 100)
