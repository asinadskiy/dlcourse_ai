import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if len(predictions.shape) == 1: # если размерность - N
        probs = predictions - np.max(predictions) # вычитаем максимальное, чтобы не было больших чисел
        probs = np.exp(probs) # экспоненты оценок
        probs = probs / np.sum(probs) # для каждого элемента - по расчёт softmax по формуле
    elif len(predictions.shape) == 2: # если размерность batch_size, N
        probs = predictions - np.max(predictions, axis = 1).reshape(-1,1) # вычитаем максимальное по группам (убираем большие)
        probs = np.exp(probs) # экспоненты оценок
        probs = probs / np.sum(probs, axis = 1).reshape(-1,1) # softmax для каждого элемента
    else:
        raise Exception("Not implemented!")
    return probs

def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if len(probs.shape) == 1: # если размерность - N
        log_q_x = np.log(probs) # логарифмы решений модели
        p_x = np.zeros(probs.shape[0]) # все классы
        p_x[target_index] = 1 # класс, которому принадлежит сэмпл - для него P(x)=1
        CE_loss = -np.sum(p_x * log_q_x)
    elif len(probs.shape) == 2: # если размерность (1,N)
        log_q_x = np.log(probs) # логарифмы решений
        p_x = np.zeros((probs.shape[0], probs.shape[1])) # все классы - матрица
        for i in range(len(target_index)):
            p_x[i][target_index[i]] = 1 # класс, которому принадлежит сэмпл - для него P(x)=1
        CE_loss = -np.sum(p_x * log_q_x, axis=1) # умножаем на истинные классы и считаем
    else:
        raise Exception("Not implemented!")
    return CE_loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)   
    dprediction = probs.copy()
    if len(predictions.shape) == 1: # для одномерного случая
        dprediction[target_index] -= 1 # разворачиваем градиент
    elif len(predictions.shape) == 2: # для двумерного случая
        dprediction[np.arange(len(dprediction)), target_index] -= 1 # для каждого из предсказаний разворачиваем градиент
    else:
        raise Exception("Not implemented!")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W*W)
    grad = 2 * np.dot(W,reg_strength)

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    
    predictions = np.dot(X, W)

    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dpredictions)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            for batch in batches_indices: # для каждого набора данных
                x_ = X[batch]
                y_ = y[batch]
                loss, dW = linear_softmax(x_, self.W, y_) # softmax и изменение весов
                reg_loss, dregW = l2_regularization(self.W, reg)
                loss += reg_loss
                dW += dregW
                self.W += -learning_rate * dW # пересчитываем веса
            loss = np.mean(loss) # общая ошибка
            # end
            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        scores = np.dot(X, self.W) # исходные данные на веса
        y_pred = np.argmax(scores, axis=1) # поиск максимально вероятного
        
        return y_pred



                
                                                          

            

                
