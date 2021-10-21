import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layers = []
        self.layers.append(FullyConnectedLayer(n_input = n_input, n_output = hidden_layer_size)) # полносвязный уровень
        self.layers.append(ReLULayer()) # уровень с нелинейностью ReLU
        self.layers.append(FullyConnectedLayer(n_input = hidden_layer_size, n_output = n_output)) # полносвязный уровень

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        
        params = self.params() # запоминаем собственные параметры
        for param_key in params: # для каждого из имён параметров
            param = params[param_key] # выбираем ссылку на него из словаря
            param.grad = np.zeros_like(param.grad) # обнуляем градиент
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        # прямое распространение
        X_next = X.copy() # значения (изменяются на каждом этапе)
        for layer in self.layers: # каждый слой сети
            X_next = layer.forward(X_next) # вычисляет знаение на основе входных данных и своих весов
        loss, grad = softmax_with_cross_entropy(X_next, y) # оценка ошибки

        
        # обратное распространение ошибки
        gradients = [] # список градиентов
        for layer in reversed(self.layers): # по всем слоям в обратном порядке
            grad = layer.backward(grad) # градиент для слоя
            gradients.append(grad) # сохраняем в список по уровням
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # регуляризация
        l2_reg = 0
        for layer, grad in zip(reversed(self.layers), gradients): # перебираем уровни и вычисленные градиенты
            grad_l2 = 0 # региляризованный градиент
            for params in layer.params(): # для весов и смещений уровня
                param = layer.params()[params] # выбираем ссылку на параметр из списка
                loss_d, grad_d = l2_regularization(param.value, self.reg) # регуляризуем
                param.grad += grad_d # добавляем регуляризацию (штраф за сложность модели)
                l2_reg += loss_d # добавляем градиент по слою
        loss += l2_reg # добавляем регуляризацию по всей сети
        
        return loss
    
    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int) # инициализируем список предсказания
        for layer in self.layers: # для каждого из слоёв сети
            X = layer.forward(X) # определяем его решения по входным данным
        pred = np.argmax(X, axis=1) # ищем наиболее вероятный по мнению модели класс
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        for layer_num in range(len(self.layers)): # для каждого уровня
            for i in self.layers[layer_num].params(): # для каждого параметра на уровне (вес и смещение)
                result[str(layer_num) + "_" + i] = self.layers[layer_num].params()[i] # запоминаем его
        return result
