import os

import numpy as np


def get_features_and_labels_data(data_file, data_doc_embeddings_file):
    x = []
    y = []
    data = open(data_file, "r")
    embeddings = open(data_doc_embeddings_file, "r")

    for data_line, embeddings_line in zip(data, embeddings):
        label = int(data_line.split('","')[0][1:])
        y.append(label)
        doc_embedding = np.array(embeddings_line[:-1].split("\t"))
        x.append(doc_embedding.astype(float))
    data.close()
    embeddings.close()
    return x, y


def create_file_with_model_evaluation_result(evaluation_results_file, max_iter_num, training_duration):
    # save training time to evaluation table
    if not os.path.exists("../assets/svm-models-evaluation/"):
        os.makedirs("../assets/svm-models-evaluation/")
    need_headers = False
    if not os.path.exists("../assets/svm-models-evaluation/" + evaluation_results_file):
        need_headers = True
    eval_file = open("../assets/svm-models-evaluation/" + evaluation_results_file, "a+")
    if need_headers:
        # writing table header
        eval_file.write("Iterations\tTraining time (sec)\tAccuracy\tError rate\tRecall (micro)\t"
                        "Recall (macro)\tPrecision (micro)\tPrecision (macro)\tF1 (micro)\tF1 (macro)\n")
    eval_file.write(str(max_iter_num) + "\t" + str(training_duration) + "\n")
    eval_file.close()
