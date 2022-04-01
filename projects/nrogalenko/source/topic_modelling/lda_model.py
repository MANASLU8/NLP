import json
import os
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
import joblib


def get_dataset_row(dataset_file, row_num):
    f = open(dataset_file, "r")
    for i, line in enumerate(f):
        if i == row_num:
            return line


def get_perplexity_score(lda_model, test_td_matrix):
    return lda_model.perplexity(test_td_matrix)


def train_lda_model(topics_num, iter_num, matrix_file):
    td_matrix = sparse.load_npz(matrix_file)
    lda_model = LatentDirichletAllocation(n_components=topics_num, max_iter=iter_num, learning_method='online')\
        .fit(td_matrix)
    joblib.dump(lda_model, '../assets/lda-models/lda_model_' + str(iter_num) + '_' + str(topics_num) + '.jl')


def use_lda_model(model_path, dict_file, matrix_file, dataset_path, test_td_matrix_file, top_number_to_show):
    # load model
    lda_model = joblib.load(model_path)
    # get dictionary
    f = open(dict_file, "r")
    fr_dict = json.load(f)
    f.close()
    # get words
    feature_names = list(fr_dict.keys())
    # get document topic distribution
    td_matrix = sparse.load_npz(matrix_file)
    doc_topic = lda_model.transform(td_matrix)

    # preparing results dir
    filename_base = "lda_" + str(lda_model.n_components) + "_" + str(lda_model.max_iter)
    output_dir_path = "../assets/topic-modelling-results/" + filename_base + "/"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # writing doc-topic probability
    docs_probability_for_topic = [[] for _ in range(len(doc_topic[0]))]
    doc_topic_file = open(output_dir_path + "lda_doc_topic_matrix", "w+")
    for n, doc in enumerate(doc_topic):
        doc_topic_string = str(n + 1)
        for prob_num, topic_probability in enumerate(doc):
            docs_probability_for_topic[prob_num].append(topic_probability)
            doc_topic_string += ("\t" + str(topic_probability))
        doc_topic_file.write(doc_topic_string)
        doc_topic_file.write("\n")
    doc_topic_file.close()

    # writing top words and documents for topics
    top_words_file = open(output_dir_path + "top_" + str(top_number_to_show) + "_words_and_docs_by_topic", 'w+')
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_file.write("Topic %d:" % (topic_idx + 1))
        top_words_file.write("\n")
        top_words_file.write(" ".join([feature_names[i] for i in topic.argsort()[:-top_number_to_show - 1:-1]]))
        top_words_file.write("\n")
        top_documents_indices = sorted(range(len(docs_probability_for_topic[topic_idx])),
                                       key=lambda i: docs_probability_for_topic[topic_idx][i])[-top_number_to_show:]
        for index in top_documents_indices:
            top_words_file.write(get_dataset_row(dataset_path, index))
        top_words_file.write("\n")
    top_words_file.close()

    # writing perplexity score
    perplexity_file = open("../assets/topic-modelling-results/perplexity", "a+")
    test_td_matrix = sparse.load_npz(test_td_matrix_file)
    perplexity_file.write(str(lda_model.max_iter) + "," + str(lda_model.n_components) + ","
                          + str(get_perplexity_score(lda_model, test_td_matrix)) + "\n")
    perplexity_file.close()

