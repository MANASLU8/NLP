import joblib
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation

from tematic_modeling.lda import LDA
from tematic_modeling.lda_plot import create_plot
from vectorizer.freq_dictionary import read_freq_dict, reconstruct_entry_info, write_tokens_freq_dictionary
from vectorizer.vectorizer import TextVectorizer


def reconstruct_freq_dict(train_dict_path, test_dict_path):
    token_entries = reconstruct_entry_info(read_freq_dict(train_dict_path), ['../assets/test/Business', '../assets/test/World', '../assets/test/Sports', '../assets/test/Sci-Tech'])
    write_tokens_freq_dictionary(token_entries, test_dict_path)


def get_lda_model_path(iter, topic_num):
    return "../assets/lda_model_" + str(iter) + "_" + str(topic_num) + ".jl"


def train_lda_models_and_save(iterations, topics, matrix):
    for i in iterations:
        for topic_num in topics:
            lda_model = LDA(LatentDirichletAllocation(n_components=topic_num, max_iter=i, learning_method='online'), get_lda_model_path(iter, topic_num))
            lda_model.train(matrix)
            joblib.dump(lda_model, lda_model.model_path)


def main():
    train_dict_path = "../assets/vec_dict"
    train_matrix_path = "../assets/train-td-matrix.npz"
    test_dict_path = "../assets/test_vec_dict"
    test_matrix_path = "../assets/test-td-matrix.npz"
    perplexity_path = "../assets/perplexity"

    min_frequency = 4
    iterations = [2, 4, 8]
    topics = [2, 4, 5, 10, 20, 30, 40]
    top_num = 10
    # reconstruct_freq_dict(train_dict_path, test_dict_path)

    vectorizer = TextVectorizer(read_freq_dict(train_dict_path))
    vectorizer.save_matrix(min_frequency, train_matrix_path)
    train_matrix = sparse.load_npz(train_matrix_path)

    test_vectorizer = TextVectorizer(read_freq_dict(test_dict_path))
    test_vectorizer.save_matrix(min_frequency, test_matrix_path)
    test_matrix = sparse.load_npz(test_matrix_path)

    train_lda_models_and_save(iterations, topics, train_matrix)

    for i in iterations:
        for topic_num in topics:
            lda_model = LDA(joblib.load(get_lda_model_path(i, topic_num)), get_lda_model_path(i, topic_num))
            lda_model.write_perplexity_to_file("../assets/perplexity", test_matrix)
            dt_probability = lda_model.get_dt_matrix_and_save("../assets/dt_matrix", train_matrix)
            lda_model.top_words_by_topic_to_file("../assets/top_" + str(top_num), vectorizer, dt_probability, top_num)
    create_plot(perplexity_path)
    print("main")


if __name__ == "__main__":
    main()
