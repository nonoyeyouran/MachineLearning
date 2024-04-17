import os
import sys
import logging
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout


class DNN(tf.keras.Model):
    def __init__(self, hidden_dims=[], activate_funcs=[], use_dropout=False, drop_rates=[0.5]):
        super().__init__()
        if len(hidden_dims) == 0 or len(activate_funcs) == 0:
            return None
        if len(hidden_dims) != len(activate_funcs):
            return None
        self.use_dropout = use_dropout
        self.DNN_depth = len(hidden_dims)
        if not use_dropout:
            self.model = tf.keras.Sequential()
            for hidden_dim, activate_func in zip(hidden_dims, activate_funcs):
                self.model.add(Dense(units=hidden_dim, activation=activate_func, use_bias=True))
        else:
            self.model = []
            self.dropout_layers = []
            for hidden_dim, activate_func in zip(hidden_dims, activate_funcs):
                self.model.append(Dense(units=hidden_dim, activation=activate_func, use_bias=True))
            self.dropout_layers = []
            for drop_rate_ in drop_rates:
                self.dropout_layers.append(Dropout(rate=drop_rate_))

    def call(self, inputs):
        x = inputs
        if self.use_dropout:
            for i in range(self.DNN_depth):
                if i < len(self.dropout_layers):
                    x = self.dropout_layers[i](x, training=True)
                x = self.model[i](x)
        else:
            x = self.model(x)
        outputs = x
        return outputs


