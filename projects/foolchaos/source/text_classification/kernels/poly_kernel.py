from datetime import datetime as dt

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluation_result import create_file_with_model_evaluation_result
from .evaluation_result import get_features_and_labels_data


def train_svm_poly_kernel(train_data_file, train_data_doc_embeddings_file, max_iter_num, evaluation_results_file, deg):
    x_train, y_train = get_features_and_labels_data(train_data_file, train_data_doc_embeddings_file)
    trans = StandardScaler()
    x_train = trans.fit_transform(x_train)
    svm_model = SVC(cache_size=10000, verbose=True, random_state=0, decision_function_shape="ovo", kernel='poly',
                    max_iter=max_iter_num, degree=deg)
    training_start = dt.now()
    svm_model.fit(x_train, y_train)
    joblib.dump(svm_model, '../assets/svm-models/svm_model_poly__' + str(deg) + '_' + str(max_iter_num) + '__.jl')
    training_duration = (dt.now() - training_start).seconds
    create_file_with_model_evaluation_result(evaluation_results_file, max_iter_num, training_duration)
