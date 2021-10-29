import numpy as np


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
    return ces

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
    # из предыдущего задания
    probs = softmax(predictions) # вычисляем софтмакс
    real = np.zeros_like(predictions) # реальные классы
    for row in range(real.shape[0]): # для каждого вектора
        real[row,target_index[row]] = 1. # указывваем истинный класс
    dprediction = probs - real # определяем матрицу ошибок
    loss = cross_entropy_loss(probs, target_index) # считаем кросс-энтропию
    dprediction /= loss.shape[0] # средний градиент
    loss = np.mean(loss) # среднее значение ошибки (по сэмплам)
    return loss, dprediction


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # из предыдущего задания
        out = np.maximum(0, X) # самый вероятный
        self.cashe = X # запоминаем ответ
        return out

    def backward(self, d_out):
        # из предыдущего задания
        d_result = d_out.copy() # линейная область (y=x) для ReLU
        d_result[self.cashe < 0] = 0 # нелинейная область
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # из предыдущего задания
        self.X = X # входное значение
        out = np.dot(self.X, self.W.value) + self.B.value # умножение на вес и плюс смещение
        return out

    def backward(self, d_out):
        # из предыдущего задания
        self.W.grad = np.dot(self.X.T, d_out) # градиент для весов
        self.B.grad = np.dot(np.ones([1, d_out.shape[0]]), d_out) # градиент для смещений
        d_result = np.dot(d_out, self.W.value.T) # градиент от весов      
        return d_result

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size # размер окна фильтра
        self.in_channels = in_channels # размер входных данных
        self.out_channels = out_channels # количество фильтров для обнаружения отдельных признаков
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        self.X = Param(X.copy()) # сохраняем параметр и место для градиента
        # параметры размерности выхода
        out_height = height + 1 - self.filter_size + 2 * self.padding # уменьшение на фильтр и плюс падднинг с каждой стороны
        out_width  = width  + 1 - self.filter_size + 2 * self.padding
        out = np.zeros([batch_size, out_height, out_width, self.out_channels]) # 4-мерный список выходных сигналов
        self.X.value = np.pad(self.X.value, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant') # расширяет список пустыми значениями, чтобы можно было пройти фильтром
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                slice_X_flat = self.X.value[:, y : (y + self.filter_size), x : (x + self.filter_size), :] # выбираем данные по X, Y и размеру фильтра
                slice_X_flat = slice_X_flat.reshape(batch_size, -1) # делаем плоский массив из слайса
                W_flat = self.W.value.reshape(-1, self.out_channels) # делаем плоский массив из весов
                out[:, y, x, :] = slice_X_flat.dot(W_flat) + self.B.value # вычисляем результат прохождения через слой (ax+b)
        return out


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.value.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_input = np.zeros_like(self.X.value) # входные данные
        W_flat = self.W.value.reshape(-1, self.out_channels) # плоский массив весов
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                slice_X_flat = self.X.value[:, y:y + self.filter_size, x:x + self.filter_size, :] # выбираем данные по X, Y и размеру фильтра
                slice_X_flat = slice_X_flat.reshape(batch_size, -1) # делаем плоский массив из слайса
                # передаваемое дальше изменение (результат умноженный на транспонированные веса)
                d_input[:, y:y + self.filter_size, x:x + self.filter_size, :] += np.dot(d_out[:, y, x, :], W_flat.T) \
                    .reshape(batch_size, self.filter_size, self.filter_size, self.in_channels) # и преобразовать к размеру фильтра
                # градиент весов как произведение транспонированных входов и потерь
                self.W.grad += np.dot(slice_X_flat.T, d_out[:, y, x, :]) \
                    .reshape(self.filter_size, self.filter_size, self.in_channels, out_channels) # и привести к размеру фильтра
                self.B.grad += np.sum(d_out[:, y, x, :], axis=0) # добавляем градиент смещений

        return d_input[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :] # веса без паддинга

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy() # входные данные
        out_height = (height - self.pool_size) // self.stride + 1 # исходный размер - размер окна // смещение окна
        out_width  = (width  - self.pool_size) // self.stride + 1
        
        out = np.zeros([batch_size, out_height, out_width, channels]) # массив выходов нужного размера
        
        for y in range(out_height): # для каждого итогового по оси
            for x in range(out_width):
                out[:, y, x, :] += np.amax(X[:, # максимальное по массиву из X
                                              (y * self.stride) : (y * self.stride + self.pool_size), # ограниченному размером
                                              (x * self.stride) : (x * self.stride + self.pool_size), # окна на этом шаге
                                              :], axis=(1, 2)) # максимальное по 1 и 2 координатам - X и Y
        
        return out # вернуть максимумы

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape # размеры входных данных
        _, out_height, out_width, _ = d_out.shape # размеры выходов
        
        d_input = np.zeros_like(self.X) # массив для градиентов
        
        batch_idxs = np.repeat(np.arange(batch_size), channels) # повторение элементов
        channel_idxs = np.tile(np.arange(channels), batch_size) # повторяет элементы batch_size раз
        
        for y in range(out_height): # для каждого элемента по оси
            for x in range(out_width):
                slice_X = self.X[:, # выбираем из массива входов
                                (y * self.stride) : (y * self.stride + self.pool_size), # ограниченному размером
                                (x * self.stride) : (x * self.stride + self.pool_size), # окна на этом шаге
                                :].reshape(batch_size, -1, channels) # распрямляя X и Y в одну координату
               
                max_idxs = np.argmax(slice_X, axis=1) # выбираем максимальное из X и Y (по 1 координате, в которую объединили)

                slice_d_input = d_input[:, # выбираем из массива изменений
                                        (y * self.stride) : (y * self.stride + self.pool_size), # ограниченному размером
                                        (x * self.stride) : (x * self.stride + self.pool_size), # окна на этом шаге
                                        :].reshape(batch_size, -1, channels) # распрямляя X и Y в одну координату
                
                slice_d_input[batch_idxs, max_idxs.flatten(), channel_idxs] = d_out[batch_idxs, y, x, channel_idxs] # пропускаем выходы налево

                d_input[:, # каждое из изменений делаем равным
                        (y * self.stride) : (y * self.stride + self.pool_size), # ограниченному размером
                        (x * self.stride) : (x * self.stride + self.pool_size), # окна на этом шаге
                        :] = slice_d_input.reshape(batch_size, self.pool_size, self.pool_size, channels) # изменениям слайса
        
        return d_input # и возвращаем их

    def params(self):
        return {}


class Flattener:
    """
    Преобразует четырёхмерные векторы в двумерные для перехода от свёрточных слоёв к полносвязным
    """
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = batch_size, height, width, channels # исходные размеры
        return X.reshape(batch_size, -1) # сопоставляем размерность с количеством сэмплов (в 2 измерения)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape) # сопоставление размерности с исходной (из 2 в 4 измерения)

    def params(self):
        # No params!
        return {}
