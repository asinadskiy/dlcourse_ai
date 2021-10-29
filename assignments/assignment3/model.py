import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        width, height, channels = input_shape # парсинг заданной размерности на измерения
        filter_size = 3 # размер фильтра (окна)
        padding = 1 # дополнение с каждой стороны
        pool_size = 4 # размер окна для определения максимума
        pool_stride = 4 # размер смещения фильтра на каждом шаге
        
        # первый уровень свёрточной сети
        self.Conv1 = ConvolutionalLayer(channels, conv1_channels, filter_size, padding) # свёртка: объединение, но та же размерность
        self.ReLU1 = ReLULayer() # добавляем нелинейность
        self.MaxPool1 = MaxPoolingLayer(pool_size, pool_stride) # объединение по размеру фильтра
        
        # второй уровень свёрточной сети
        self.Conv2 = ConvolutionalLayer(conv1_channels, conv2_channels, filter_size, padding) # ещё свёртка
        self.ReLU2 = ReLULayer() # нелинейность
        self.MaxPool2 = MaxPoolingLayer(pool_size, pool_stride) # и ещё свёртка по размеру фильтра
        
        # размеры нового рисунка - уменьшение в "размер шага" раз на каждый уровень
        left_width  = width  // pool_stride // pool_stride
        left_height = height // pool_stride // pool_stride
        
        self.Flat = Flattener() # создание из четырёхмерного объекта двумерного
        # полносвязный слой для вычисления ответа (наличия некоторых признаков)
        self.FullyConnected = FullyConnectedLayer(left_width * left_height * conv2_channels, n_output_classes)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for _, v in self.params().items(): # для каждого слоя
            v.grad = np.zeros(v.grad.shape) # обнуляем градиенты
        
        # прямой проход
        out = self.Conv1.forward(X) # свёртка
        out = self.ReLU1.forward(out) # нелинейность
        out = self.MaxPool1.forward(out) # выбор максимального из окна
        out = self.Conv2.forward(out) # свёртка 2
        out = self.ReLU2.forward(out) # нелинейность 2
        out = self.MaxPool2.forward(out) # выбор максимального из окна
        out = self.Flat.forward(out) # из многослойных данных в плоские
        out = self.FullyConnected.forward(out) # полносвязный слой для выделения признаков
          
        loss, d_out = softmax_with_cross_entropy(out, y) # вычисление ошибок

        # обратный проход - пропуск градентов через каждый слой для обновления весов
        # можно было циклом for layer: backward(), но так - более явно (хотя при изменении модели придётся переписывать)
        d_out = self.FullyConnected.backward(d_out)
        d_out = self.Flat.backward(d_out)
        d_out = self.MaxPool2.backward(d_out)
        d_out = self.ReLU2.backward(d_out)
        d_out = self.Conv2.backward(d_out)
        d_out = self.MaxPool1.backward(d_out)
        d_out = self.ReLU1.backward(d_out)
        d_out = self.Conv1.backward(d_out)
        
        return loss # возврат значения ошибки

    def predict(self, X):
        # You can probably copy the code from previous assignment
        # можно было циклом for layer: forward(), но так - более явно (хотя при изменении модели придётся переписывать)
        # прямой проход через все слои
        out = self.Conv1.forward(X)
        out = self.ReLU1.forward(out)
        out = self.MaxPool1.forward(out)
        out = self.Conv2.forward(out)
        out = self.ReLU2.forward(out)
        out = self.MaxPool2.forward(out)
        out = self.Flat.forward(out)
        out = self.FullyConnected.forward(out)
        
        pred = np.argmax(out, axis=1) # выбираем самые вероятные значения предсказаний
        return pred

    def params(self):
        result = {}

        # TODO: Aggregate all the params from all the layers
        # which have parameters
        # объединение всех слоёв в словарь
        name_to_layer = {"Conv1": self.Conv1, # свёртка
                      "Conv2": self.Conv2, # свёртка
                      "Fully": self.FullyConnected} # полносвязный
        
        for name, layer in name_to_layer.items(): # для каждой пары "имя" - "модель"
            for k, v in layer.params().items(): # для каждого из свойств конкретного слоя
                # k - тип (W или B), v - параметр (градиент)
                result['{}_{}'.format(name, k)] = v # в элемент словаря "тип слоя - переменная(W/B)" добавить ссылку на слой

        return result
