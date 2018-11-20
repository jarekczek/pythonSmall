# print('very beginning')
#
from typing import Dict, Generic, List, TypeVar
import copy
import collections
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import sys
import time

import keras.backend as K
import tensorflow as tf
from keras import activations
from keras.engine.base_layer import Layer
from keras.layers import Conv1D, Dense, LSTM
from keras.initializers import glorot_uniform
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from tensorflow.python.ops import array_ops, math_ops
import keras.losses

class CustomSublayer(Layer):
    def __init__(self, **kwargs):
        super(CustomSublayer, self).__init__(**kwargs)
        self.dense1 = Dense(2, input_dim=2, activation='tanh')
        self.dense2 = Dense(1, input_dim=2, activation='tanh')

    def build(self):
        self.dense1.build((1, 2))
        self.dense2.build((2, 2))
        self.trainable_weights = self.dense1.trainable_weights + self.dense2.trainable_weights
        self.built = True

    def call(self, inputs):
        return self.dense2.call(self.dense1.call(inputs))

class CustomLayer(Layer):
    def __init__(self, k, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.k = k

    def build(self, input_shape):
        print('building for', input_shape)
        self.sublayer: Layer = CustomSublayer()
        self.sublayer.build()
        self.trainable_weights = self.sublayer.trainable_weights
        self.built = True

    def compute_output_shape(self, input_shape):
        print('computing for', input_shape)
        return (None, input_shape[1] / 2)

    def call(self, inputs):
        inputShape = K.shape(inputs)
        inputs2 = K.reshape(inputs, (inputShape[0] * inputShape[1] // 2, 2))
        res1 = self.sublayer.call(inputs2)
        print('call', inputs, 'result1', res1)
        res = K.reshape(res1, (inputShape[0], inputShape[1] // 2))
        print('call', inputs, 'result', res)
        return res

class Cust:
    def __init__(self, k, setSize, n):
        self.k = k
        self.setSize = setSize
        self.n = n
        self.model: Sequential = None

    def createModelClassic(self):
        self.model = Sequential()
        self.model.add(Dense(self.k, input_dim=2 * self.k, activation='tanh'))
        opt = Adam(lr=0.01)
        self.model.compile(opt, loss='mse')

    def createModelCustom(self):
        self.model = Sequential()
        self.model.add(CustomLayer(self.k))
        opt = Adam(lr=0.01)
        self.model.compile(opt, loss='mse')

    def createSet(self, size: int):
        ax = []
        ay = []
        for i in range(size):
            x = []
            y = []
            for j in range(self.k):
                a = random.randint(0, 1)
                b = random.randint(0, 1)
                x += [a, b]
                y.append(a ^ b)
            ax.append(x)
            ay.append(y)
        return np.array(ax), np.array(ay)

    def run(self):
        self.createModelCustom()
        for i in range(self.n):
            ax, ay = self.createSet(self.setSize)
            vax, vay = self.createSet(self.setSize // 10)
            vax = np.array([[0, 0] * self.k, [0, 1] * self.k, [1, 0] * self.k, [1, 1] * self.k])
            vay = np.array([[0] * self.k, [1] * self.k, [1] * self.k, [0] * self.k])
            self.model.fit(np.array(ax), np.array(ay), validation_data=(vax, vay))
        print('weights', self.model.get_weights())
        print(self.model.predict(np.array([[0, 0] * self.k])))
        print(self.model.predict(np.array([[1, 0] * self.k])))
        print(self.model.predict(np.array([[0, 1] * self.k])))
        print(self.model.predict(np.array([[1, 1] * self.k])))

Cust(10, 100, 200).run()
