import numpy as np


class SGD:
    def update(self, w, d_w, learning_rate):
        return w - d_w * learning_rate


class MomentumSGD:
    '''
    Implements Momentum SGD update
    '''
    def __init__(self, momentum=0.9):
        # из предыдущего задания
        self.momentum = momentum # коэффициент затухания
        self.velocity = 0 # скорость изменения
    
    def update(self, w, d_w, learning_rate):
        '''
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        '''
        # из предыдущего задания
        self.velocity *= self.momentum # делаем затухание
        self.velocity -= learning_rate * d_w # уменьшаем скорость в зависимости от качества обучения
        return w + self.velocity # текущие веса + скорость
