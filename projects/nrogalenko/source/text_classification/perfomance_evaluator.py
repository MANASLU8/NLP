import math

metrics_by_class = {
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


def evaluate_model(svm_model, x_test, y_test):

    true_labels = y_test
    model_output = svm_model.predict(x_test)

    # create confusion matrix for each class
    for label_num, predicted_label in enumerate(model_output):
        true_label = true_labels[label_num]
        if predicted_label == true_label:
            metrics_by_class[str(true_label)]["TP"] += 1
            for doc_class in metrics_by_class:
                if doc_class != str(predicted_label):
                    metrics_by_class[doc_class]["TN"] += 1
        else:
            metrics_by_class[str(true_label)]["FN"] += 1
            metrics_by_class[str(predicted_label)]["FP"] += 1
            for doc_class in metrics_by_class:
                if (doc_class != str(predicted_label)) and (doc_class != str(true_label)):
                    metrics_by_class[doc_class]["TN"] += 1

    # calculate metrics for each class
    for doc_class in metrics_by_class:
        class_metrics = metrics_by_class[doc_class]
        if (class_metrics["TP"] + class_metrics["FP"]) != 0:
            class_metrics["precision"] = (class_metrics["TP"] / (class_metrics["TP"] + class_metrics["FP"]))
        else:
            class_metrics["precision"] = math.nan
        if (class_metrics["TP"] + class_metrics["FN"]) != 0:
            class_metrics["recall"] = (class_metrics["TP"] / (class_metrics["TP"] + class_metrics["FN"]))
        else:
            class_metrics["recall"] = math.nan
        if (class_metrics["TN"] + class_metrics["TP"] + class_metrics["FN"] + class_metrics["FP"]) != 0:
            class_metrics["accuracy"] = (class_metrics["TN"] + class_metrics["TP"]) / \
                                        (class_metrics["TN"] + class_metrics["TP"] + class_metrics["FN"] + class_metrics["FP"])
            class_metrics["error_rate"] = (class_metrics["FN"] + class_metrics["FP"]) / \
                                          (class_metrics["TN"] + class_metrics["TP"] + class_metrics["FN"] + class_metrics["FP"])
        else:
            class_metrics["accuracy"] = math.nan
            class_metrics["error_rate"] = math.nan
        if (class_metrics["precision"] + class_metrics["recall"]) != 0:
            class_metrics["f1"] = 2 * ((class_metrics["precision"] * class_metrics["recall"]) /
                                       (class_metrics["precision"] + class_metrics["recall"]))
        else:
            class_metrics["f1"] = math.nan
        print(class_metrics)
    print("\n")
    # calculate final metrics for model
    accuracy = 0.0
    error_rate = 0.0
    precision_macro = 0.0
    recall_macro = 0.0
    precision_micro_numerator = 0
    precision_micro_denominator = 0
    recall_micro_numerator = 0
    recall_micro_denominator = 0
    for doc_class in metrics_by_class:
        accuracy += metrics_by_class[doc_class]["accuracy"]
        error_rate += metrics_by_class[doc_class]["error_rate"]
        precision_macro += metrics_by_class[doc_class]["precision"]
        recall_macro += metrics_by_class[doc_class]["recall"]
        precision_micro_numerator += metrics_by_class[doc_class]["TP"]
        precision_micro_denominator += (metrics_by_class[doc_class]["TP"] + metrics_by_class[doc_class]["FP"])
        recall_micro_numerator += metrics_by_class[doc_class]["TP"]
        recall_micro_denominator += (metrics_by_class[doc_class]["TP"] + metrics_by_class[doc_class]["FN"])
    accuracy = accuracy / len(metrics_by_class)
    error_rate = error_rate / len(metrics_by_class)
    precision_macro = precision_macro / len(metrics_by_class)
    recall_macro = recall_macro / len(metrics_by_class)
    f1_macro = 2 * ((precision_macro * recall_macro) / (precision_macro + recall_macro))
    precision_micro = precision_micro_numerator / precision_micro_denominator
    recall_micro = recall_micro_numerator / recall_micro_denominator
    f1_micro = 2 * ((precision_micro * recall_micro) / (precision_micro + recall_micro))
    return accuracy, error_rate, recall_micro, recall_macro, precision_micro, precision_macro, f1_micro, f1_macro
