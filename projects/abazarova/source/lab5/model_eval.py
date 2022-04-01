import math

metrics = {
    "1": {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "accuracy": 0.0,
        "error_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    },
    "2": {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "accuracy": 0.0,
        "error_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    },
    "3": {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "accuracy": 0.0,
        "error_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    },
    "4": {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "accuracy": 0.0,
        "error_rate": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0
    }
}


def evaluate_model(svm, x_test, y_test):

    true_labels = y_test
    model_rez = svm.predict(x_test)

    # создаем матрицу для каждого класса
    for label_num, label in enumerate(model_rez):
        lbl = true_labels[label_num]
        if label == lbl:
            metrics[str(lbl)]["TP"] += 1
            for clas in metrics:
                if clas != str(label):
                    metrics[clas]["TN"] += 1
        else:
            metrics[str(lbl)]["FN"] += 1
            metrics[str(label)]["FP"] += 1
            for clas in metrics:
                if (clas != str(label)) and (clas != str(lbl)):
                    metrics[clas]["TN"] += 1

    # считаем метрики для каждого класса
    for clas in metrics:
        cmetrics = metrics[clas]
        if (cmetrics["TP"] + cmetrics["FP"]) != 0:
            cmetrics["precision"] = (cmetrics["TP"] / (cmetrics["TP"] + cmetrics["FP"]))
        else:
            cmetrics["precision"] = math.nan
        if (cmetrics["TP"] + cmetrics["FN"]) != 0:
            cmetrics["recall"] = (cmetrics["TP"] / (cmetrics["TP"] + cmetrics["FN"]))
        else:
            cmetrics["recall"] = math.nan
        if (cmetrics["TN"] + cmetrics["TP"] + cmetrics["FN"] + cmetrics["FP"]) != 0:
            cmetrics["accuracy"] = (cmetrics["TN"] +
                                         cmetrics["TP"]) / (cmetrics["TN"] + cmetrics["TP"] +
                                                                 cmetrics["FN"] + cmetrics["FP"])
            cmetrics["error_rate"] = (cmetrics["FN"] +
                                           cmetrics["FP"]) / (cmetrics["TN"] + cmetrics["TP"] +
                                                                   cmetrics["FN"] + cmetrics["FP"])
        else:
            cmetrics["accuracy"] = math.nan
            cmetrics["error_rate"] = math.nan
        if (cmetrics["precision"] + cmetrics["recall"]) != 0:
            cmetrics["f1"] = 2 * ((cmetrics["precision"] * cmetrics["recall"]) /
                                       (cmetrics["precision"] + cmetrics["recall"]))
        else:
            cmetrics["f1"] = math.nan
        print(cmetrics)
    print("\n")
    # считаем финальные метрики по модели
    accuracy = 0.0
    error_rate = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    precision_micro_numerator = 0
    precision_micro_denominator = 0
    recall_micro_numerator = 0
    recall_micro_denominator = 0
    for clas in metrics:
        accuracy += metrics[clas]["accuracy"]
        error_rate += metrics[clas]["error_rate"]
        precision_macro += metrics[clas]["precision"]
        recall_macro += metrics[clas]["recall"]
        precision_micro_numerator += metrics[clas]["TP"]
        precision_micro_denominator += (metrics[clas]["TP"] + metrics[clas]["FP"])
        recall_micro_numerator += metrics[clas]["TP"]
        recall_micro_denominator += (metrics[clas]["TP"] + metrics[clas]["FN"])
    accuracy = accuracy / len(metrics)
    error_rate = error_rate / len(metrics)
    precision_macro = precision_macro / len(metrics)
    recall_macro = recall_macro / len(metrics)
    f1_macro = 2 * ((precision_macro * recall_macro) / (precision_macro + recall_macro))
    precision_micro = precision_micro_numerator / precision_micro_denominator
    recall_micro = recall_micro_numerator / recall_micro_denominator
    f1_micro = 2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro))
    return accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro
