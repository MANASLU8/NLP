from pathlib import Path
from lab5.model_eval import evaluate_model
import joblib
import numpy as np
import os
from datetime import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC


def svm_train(kernel, train_path, train_vectors_path, max_count, results_file):
    if kernel == "rbf":
        x_train, y_train = get_data(train_path, train_vectors_path)
        trans = StandardScaler()
        x_train = trans.fit_transform(x_train)
        svm_model = SVC(cache_size=10000, verbose=True, random_state=0, decision_function_shape="ovo", kernel='rbf',
                        max_iter=max_count)
        training_start = dt.now()
        svm_model.fit(x_train, y_train)
        joblib.dump(svm_model, str(Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                                        "svm", "svm_rbf"+str(max_count) + ".jl")))
        training_duration = (dt.now() - training_start).seconds
        create_result_file(results_file, max_count, training_duration)
    elif kernel == "linear":
        x_train, y_train = get_data(train_path, train_vectors_path)
        trans = StandardScaler()
        x_train = trans.fit_transform(x_train)
        svm_model = SVC(cache_size=10000, verbose=True, random_state=0, decision_function_shape="ovo", kernel='linear',
                        max_iter=max_count)
        training_start = dt.now()
        svm_model.fit(x_train, y_train)
        joblib.dump(svm_model, str(Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                                        "svm", "svm_linear"+str(max_count) + ".jl")))
        training_duration = (dt.now() - training_start).seconds
        create_result_file(results_file, max_count, training_duration)
    elif kernel == "sigmoid":
        x_train, y_train = get_data(train_path, train_vectors_path)
        trans = StandardScaler()
        x_train = trans.fit_transform(x_train)
        svm_model = SVC(cache_size=10000, verbose=True, random_state=0, decision_function_shape="ovo", kernel='sigmoid',
                        max_iter=max_count)
        training_start = dt.now()
        svm_model.fit(x_train, y_train)
        joblib.dump(svm_model, str(Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                                        "svm", "svm_sigmoid"+str(max_count) + ".jl")))
        training_duration = (dt.now() - training_start).seconds
        create_result_file(results_file, max_count, training_duration)


def get_data(data_path, data_vectors_path):
    x = []
    y = []
    data = open(data_path, "r")
    vecs = open(data_vectors_path, "r")

    for data_line, vecs_line in zip(data, vecs):
        label = int(data_line.split('","')[0][1:])
        y.append(label)
        doc_vec = np.array(vecs_line[:-1].split("\t"))
        x.append(doc_vec.astype(float))
    data.close()
    vecs.close()
    return x, y


def create_result_file(results_path, max_count, training_duration):
    # сохраняем время
    path_to_svm = Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                       "svm")
    print(path_to_svm)
    if not os.path.exists(path_to_svm):
        os.mkdir(path_to_svm)
    need_head = False
    if not os.path.exists(str(Path(path_to_svm, results_path+"_rez"))):
        need_head = True
    rez_file = open(str(Path(path_to_svm, results_path+"_rez")), "a+")
    if need_head:
        # writing table header
        rez_file.write("Iterations\tTraining time (sec)\tAccuracy\tError rate\tRecall (micro)\t"
                       "Recall (macro)\tPrecision (micro)\tPrecision (macro)\tF1 (micro)\tF1 (macro)\n")
    rez_file.write(str(max_count) + "\t" + str(training_duration) + "\n")
    rez_file.close()


def svm_fit(svm_path, test_path, test_vectors, results_path, count):
    x_test, y_test = get_data(test_path, test_vectors)
    trans = StandardScaler()
    x_test = trans.fit_transform(x_test)
    svm_model = joblib.load(svm_path)
    accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro = evaluate_model(
        svm_model, x_test, y_test)

    # запишем результаты эксперимента в файл
    with open(results_path, 'r') as file:
        data = file.readlines()
    for i in range(0, len(data)):
        if (data[i].startswith((str(count) + "\t"))) or (data[i].startswith((str(count) + " "))):
            data[i] = data[i][:-1] + "\t" + str(accuracy) + "\t" + str(error_rate) + "\t" + str(recall_micro) + "\t" + \
                      str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" + \
                      str(f1_micro) + "\t" + str(f1_macro) + "\n"
    with open(results_path, 'w') as file:
        file.writelines(data)


def svm_fir_dimension_redused_after(svm_path, dim, results_path,
                                    test_path, test_vector_path,
                                    train_path, train_vector_path):
    svm_model = joblib.load(svm_path)
    pca = PCA(n_components=dim)
    trans = StandardScaler()

    x_train, y_train = get_data(train_path, train_vector_path)
    x_train = pca.fit_transform(x_train)
    x_train = trans.fit_transform(x_train)
    svm_model.fit(x_train, y_train)

    x_test, y_test = get_data(test_path, test_vector_path)
    x_test = pca.fit_transform(x_test)
    x_test = trans.fit_transform(x_test)
    accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro = \
        evaluate_model(svm_model, x_test, y_test)

    path_to_svm = Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                       "svm")
    if not os.path.exists(path_to_svm):
        os.mkdir(path_to_svm)
    need_head = False
    if not os.path.exists(str(Path(path_to_svm, results_path))):
        need_head = True
    rez_file = open(str(Path(path_to_svm, results_path)), "a+")
    if need_head:
        # writing table header
        rez_file.write("Iterations\tTraining time (sec)\tAccuracy\tError rate\tRecall (micro)\t"
                       "Recall (macro)\tPrecision (micro)\tPrecision (macro)\tF1 (micro)\tF1 (macro)\n")
    rez_file.write(str(dim) + "\t" + str(accuracy) + "\t" + str(error_rate) + "\t" + str(recall_micro) + "\t" +
                   str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" +
                   str(f1_micro) + "\t" + str(f1_macro) + "\n")
    rez_file.close()


def svm_fit_dimensions_redusedb4(svm_path, dim, results_path,
                                 test_path, test_vector_path,
                                 train_path, train_vector_path):
    svm_model = joblib.load(svm_path)
    pca = PCA(n_components=dim)
    trans = StandardScaler()

    x_train, y_train = get_data(train_path, train_vector_path)
    x_train = trans.fit_transform(x_train)
    x_train = pca.fit_transform(x_train)
    svm_model.fit(x_train, y_train)

    x_test, y_test = get_data(test_path, test_vector_path)
    x_test = trans.fit_transform(x_test)
    x_test = pca.fit_transform(x_test)
    accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro = \
        evaluate_model(svm_model, x_test, y_test)

    path_to_svm = Path(str(Path(Path.cwd()))[:-len("source")], "assets",
                       "svm")
    if not os.path.exists(path_to_svm):
        os.mkdir(path_to_svm)
    need_head = False
    if not os.path.exists(str(Path(path_to_svm, results_path))):
        need_head = True
    rez_file = open(str(Path(path_to_svm, results_path)), "a+")
    if need_head:
        # пишем заголовок таблицы
        rez_file.write("Iterations\tTraining time (sec)\tAccuracy\tError rate\tRecall (micro)\t"
                       "Recall (macro)\tPrecision (micro)\tPrecision (macro)\tF1 (micro)\tF1 (macro)\n")
    rez_file.write(str(dim) + "\t" + str(accuracy) + "\t" + str(error_rate) + "\t" + str(recall_micro) + "\t" +
                   str(recall_macro) + "\t" + str(precision_micro) + "\t" + str(precision_macro) + "\t" +
                   str(f1_micro) + "\t" + str(f1_macro) + "\n")
    rez_file.close()
