def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # из первого задания
    tp = tn = fp = fn = 0
    for pred, real in zip(prediction, ground_truth):
        if pred == real and real:
            tp += 1 
        if pred == real and not real:
            tn += 1
        if pred != real and real:
            fn += 1
        if pred != real and not real:
            fp += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    # из первого задания
    accuracy = 0
    for pred, real in zip(prediction, ground_truth):
        if pred == real:
            accuracy += 1
    accuracy /= len(prediction)
    return accuracy

