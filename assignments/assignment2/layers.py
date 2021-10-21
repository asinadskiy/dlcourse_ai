import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # из предыдущего задания
    loss = reg_strength * np.sum(W * W) # увеличение ошибки в зависимости от сложности модели
    grad = 2 * np.dot(W,reg_strength) # градиент - куда двигать веса (зависит от исходной ошибки и весов)

    return loss, grad

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
    # из предыдущего задания
    probs = predictions.copy() # предсказанные вероятности
    if len(predictions.shape) == 1: # если размерность - N
        probs -= np.max(probs) # вычитаем максимальное, чтобы не было больших чисел
        probs = np.exp(probs) / np.sum(np.exp(probs)) # для каждого элемента - по расчёт softmax по формуле
    elif len(predictions.shape) == 2: # если размерность batch_size, N
        probs -= np.max(probs, axis = 1).reshape(-1,1) # вычитаем максимальное и строим в одномерный вектор
        probs = np.exp(probs)/np.sum(np.exp(probs),axis = 1).reshape(-1,1)
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
    # из предыдущего задания
    px = np.zeros_like(probs) # все классы
    for row in range(px.shape[0]): # для каждого сэмпла
        px[row,target_index[row]] = 1. # указываем истинный класс сэмпла
    qx = probs # решения модели
    ces = np.zeros_like(target_index, dtype=np.float) # массив кросс-энтропий
    for ix in range(0, target_index.shape[0]): # для каждого сэмпла
        ces[ix]= -1. * np.sum(px[ix] * np.log(qx[ix])) # кросс-энтропия по формуле -1*СУМ(P*logQ)
    ces = ces.reshape((target_index.shape[0],1)) # ответ для каждого сэмпла
    return  ces  


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    '''
    probs = softmax(preds) # вычисляем софтмакс
    real = np.zeros_like(preds) # реальные классы
    for row in range(real.shape[0]): # для каждого вектора
        real[row,target_index[row]] = 1. # указывваем истинный класс
    dprediction = probs - real # определяем матрицу ошибок
    loss = cross_entropy_loss(probs, target_index) # считаем кросс-энтропию
    loss = np.mean(loss) # среднее значение ошибки (по сэмплам)
    dprediction = dprediction / loss.shape[0] # средний градиент
    return loss, dprediction
    '''
    probs = softmax(preds) # вычисляем софтмакс
    real = np.zeros_like(preds) # реальные классы
    for row in range(real.shape[0]): # для каждого вектора
        real[row,target_index[row]] = 1. # указывваем истинный класс
    dprediction = probs - real # определяем матрицу ошибок
    loss = cross_entropy_loss(probs, target_index) # считаем кросс-энтропию
    dprediction /= loss.shape[0] # средний градиент
    loss = np.mean(loss) # среднее значение ошибки (по сэмплам)
    return loss, dprediction

class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        out = np.maximum(0, X) # самый вероятный
        self.cashe = X # запоминаем ответ
        return out

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = d_out.copy() # линейная область (y=x) для ReLU
        d_result[self.cashe < 0] = 0 # нелинейная область

        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X # входное значение
        out = np.dot(self.X, self.W.value) + self.B.value # умножение на вес и плюс смещение
        return out


    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot(self.X.T, d_out) # градиент для весов
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out) # градиент для смещений
        d_result = np.dot(d_out, self.W.value.T) # градиент от весов
        
        return d_result


    def params(self):
        return {'W': self.W, 'B': self.B}
