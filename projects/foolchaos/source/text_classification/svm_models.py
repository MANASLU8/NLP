import os

import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kernels import evaluation_result
from kernels import rbf_kernel, linear_kernel, sigmoid_kernel, poly_kernel
from .perfomance_evaluator import evaluate_model


def train_svm_rbf_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num, evaluation_results_file):
    rbf_kernel.train_svm_rbf_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num,
                                    evaluation_results_file)


def train_svm_linear_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num, evaluation_results_file):
    linear_kernel.train_svm_linear_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num,
                                          evaluation_results_file)


def train_svm_sigmoid_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num, evaluation_results_file):
    sigmoid_kernel.train_svm_sigmoid_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num,
                                            evaluation_results_file)


def train_svm_poly_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num, evaluation_results_file, deg):
    poly_kernel.train_svm_poly_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num,
                                      evaluation_results_file, deg)


def use_svm_model(svm_model_path, test_data_file, test_data_doc_embeddings_file, evaluation_results_file, iter_num):
    x_test, y_test = evaluation_result.get_features_and_labels_data(test_data_file, test_data_doc_embeddings_file)
    trans = StandardScaler()
    x_test = trans.fit_transform(x_test)
    svm_model = joblib.load(svm_model_path)
    accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro = evaluate_model(
        svm_model, x_test, y_test)

    # write experiment results to evaluation results file
    with open(evaluation_results_file, 'r') as file:
        data = file.readlines()
    for i in range(0, len(data)):
        if (data[i].startswith((str(iter_num) + "\t"))) or (data[i].startswith((str(iter_num) + " "))):
            data[i] = data[i][:-1] + "\t" + str(accuracy) + "\t" + str(error_rate) + "\t" + str(recall_micro) + "\t" + \
                      str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" + \
                      str(f1_micro) + "\t" + str(f1_macro) + "\n"
    with open(evaluation_results_file, 'w') as file:
        file.writelines(data)


def load(svm_model_path):
    return joblib.load(svm_model_path)


def use_model_with_dim_reduction_scale_after(svm_model_path, dim, results_file,
                                             test_data_file, test_data_doc_embeddings_file,
                                             train_data_file, train_data_doc_embeddings_file):
    svm_model = load(svm_model_path)
    pca = PCA(n_components=dim)
    trans = StandardScaler()

    x_train, y_train = evaluation_result.get_features_and_labels_data(train_data_file, train_data_doc_embeddings_file)
    x_train = pca.fit_transform(x_train)
    x_train = trans.fit_transform(x_train)
    svm_model.fit(x_train, y_train)

    x_test, y_test = evaluation_result.get_features_and_labels_data(test_data_file, test_data_doc_embeddings_file)
    x_test = pca.fit_transform(x_test)
    x_test = trans.fit_transform(x_test)
    accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro = \
        evaluate_model(svm_model, x_test, y_test)

    if not os.path.exists("../assets/svm-models-evaluation/"):
        os.makedirs("../assets/svm-models-evaluation/")
    need_headers = False
    if not os.path.exists("../assets/svm-models-evaluation/" + results_file):
        need_headers = True
    eval_file = open("../assets/svm-models-evaluation/" + results_file, "a+")
    if need_headers:
        # writing table header
        eval_file.write("Dim\tAccuracy\tError rate\tRecall (micro)\t"
                        "Recall (macro)\tPrecision (micro)\tPrecision (macro)\tF1 (micro)\tF1 (macro)\n")
    eval_file.write(str(dim) + "\t" + str(accuracy) + "\t" + str(error_rate) + "\t" + str(recall_micro) + "\t" +
                    str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" +
                    str(f1_micro) + "\t" + str(f1_macro) + "\n")
    eval_file.close()


def use_model_with_dim_reduction_scale_before(svm_model_path, dim, results_file,
                                              test_data_file, test_data_doc_embeddings_file,
                                              train_data_file, train_data_doc_embeddings_file):
    svm_model = load(svm_model_path)
    pca = PCA(n_components=dim)
    trans = StandardScaler()

    x_train, y_train = evaluation_result.get_features_and_labels_data(train_data_file, train_data_doc_embeddings_file)
    x_train = trans.fit_transform(x_train)
    x_train = pca.fit_transform(x_train)
    svm_model.fit(x_train, y_train)

    x_test, y_test = evaluation_result.get_features_and_labels_data(test_data_file, test_data_doc_embeddings_file)
    x_test = trans.fit_transform(x_test)
    x_test = pca.fit_transform(x_test)
    accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro = \
        evaluate_model(svm_model, x_test, y_test)

    if not os.path.exists("../assets/svm-models-evaluation/"):
        os.makedirs("../assets/svm-models-evaluation/")
    need_headers = False
    if not os.path.exists("../assets/svm-models-evaluation/" + results_file):
        need_headers = True
    eval_file = open("../assets/svm-models-evaluation/" + results_file, "a+")
    if need_headers:
        # writing table header
        eval_file.write("Dim\tAccuracy\tError rate\tRecall (micro)\t"
                        "Recall (macro)\tPrecision (micro)\tPrecision (macro)\tF1 (micro)\tF1 (macro)\n")
    eval_file.write(str(dim) + "\t" + str(accuracy) + "\t" + str(error_rate) + "\t" + str(recall_micro) + "\t" +
                    str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" +
                    str(f1_micro) + "\t" + str(f1_macro) + "\n")
    eval_file.close()
