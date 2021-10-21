def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # из предыдущего задания
    accuracy = 0
    for pred, real in zip(prediction, ground_truth):
        if pred == real:
            accuracy += 1
    accuracy /= len(prediction)
    return accuracy
