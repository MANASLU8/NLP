from matplotlib import pyplot

from .vectorize import use_tdidf_model
from .w2v import use_w2v_model


def create_plot(title, reduced_embeddings, all_tokens):
    pyplot.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    words = all_tokens
    pyplot.title(title)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
    pyplot.show()


def test_w2v_and_tfidf_models(model_path, tf_idf_matrix_file):
    chosen_tokens = ["football", "washington", "company"]
    similar_tokens = [["basketball", "hockey", "soccer"], ["california", "boston", "chicago"],
                      ["firm", "business", "agency"]]
    same_field_tokens = [["sport", "players", "game"], ["london", "paris", "moscow"],
                         ["organization", "bank", "economy"]]
    other_tokens = [["bombing", "russia", "trump"], ["consumer", "device", "computer"],
                    ["football", "coach", "players"]]
    for token in chosen_tokens:
        reduced_embeddings, all_tokens = use_w2v_model(model_path, token,
                                                       similar_tokens[chosen_tokens.index(token)],
                                                       same_field_tokens[chosen_tokens.index(token)],
                                                       other_tokens[chosen_tokens.index(token)])
        create_plot("w2v - " + token, reduced_embeddings, all_tokens)

        reduced_embeddings, all_tokens = use_tdidf_model(tf_idf_matrix_file, token,
                                                         similar_tokens[chosen_tokens.index(token)],
                                                         same_field_tokens[chosen_tokens.index(token)],
                                                         other_tokens[chosen_tokens.index(token)])
        # print(reduced_embeddings)
        create_plot("custom tf-idf - " + token, reduced_embeddings, all_tokens)


def test_different_w2v_models(model_paths):
    chosen_tokens = ["football", "washington", "company"]
    similar_tokens = [["basketball", "baseball", "soccer"], ["california", "boston", "chicago"],
                      ["firm", "business", "agency"]]
    same_field_tokens = [["sport", "players", "game"], ["london", "paris", "moscow"],
                         ["organization", "bank", "economy"]]
    other_tokens = [["bombing", "russia", "trump"], ["consumer", "device", "computer"],
                    ["football", "coach", "players"]]
    for token in chosen_tokens:
        for model_path in model_paths:
            reduced_embeddings, all_tokens = use_w2v_model(model_path, token,
                                                           similar_tokens[chosen_tokens.index(token)],
                                                           same_field_tokens[chosen_tokens.index(token)],
                                                           other_tokens[chosen_tokens.index(token)])
            create_plot("Model: " + model_path + " Token: " + token, reduced_embeddings, all_tokens)
