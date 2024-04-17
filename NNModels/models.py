import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from layers import *
class FNNModel(keras.Model):
    def __init__(self, hidden_nums=[32, 1], activate_funcs=["relu", "sigmoid"]):
        super().__init__()
        self.dnn_layer = DNN(hidden_nums, activate_funcs)

    def call(self, inputs):
        feature_1 = inputs["age"]
        feature_2 = inputs["sex"]
        input_feas = np.concatenate((feature_1, feature_2), axis=1)
        outputs = self.dnn_layer(input_feas)
        return outputs

model = FNNModel()
inputs = {}
feature_1 = np.array([[31.0]])
feature_2 = np.array([[0., 1.0, 0.]])
inputs["age"] = feature_1
inputs["sex"] = feature_2
result = model(inputs)
print(result.numpy())
