import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from MLModels.NNModels.layers import *


class FNNModel(keras.Model):
    def __init__(self, hidden_nums=[32, 1], activate_funcs=["relu", "sigmoid"]):
        super().__init__()
        self.dnn_layer = DNN(hidden_nums, activate_funcs)

    def call(self, inputs):
        outputs = self.dnn_layer(inputs)
        return outputs

class DSSM(keras.Model):
    def __init__(self, user_hidden_nums=[32, 8],
                 item_hidden_nums=[32, 8],
                 user_activate_funcs=["relu", None],
                 item_activate_funcs=["relu", None]):
        super().__init__()
        self.user_dnn_layer = DNN(user_hidden_nums, user_activate_funcs)
        self.item_dnn_layer = DNN(item_hidden_nums, item_activate_funcs)
    def call(self, inputs):
        if inputs["level"] == "all":
            user_inputs = inputs["user"]
            item_inputs = inputs["item"]
            user_output = self.user_dnn_layer(user_inputs)
            item_output = self.item_dnn_layer(item_inputs)
            return user_output, item_output
        elif inputs["level"] == "user":
            user_inputs = inputs["user"]
            user_output = self.user_dnn_layer(user_inputs)
            return user_output
        elif inputs["level"] == "item":
            item_inputs = inputs["item"]
            item_output = self.item_dnn_layer(item_inputs)
            return item_output

class FMModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.FM_layer = FM()
    def call(self, inputs):
        outputs = self.FM_layer(inputs)
        return outputs

"""
feature_descs = {}
feature_descs["age"] = [10, 32]
feature_descs["sex"] = [3, 32]
embedding_layer = EmbeddingLayer(feature_descs)
feature_name = ["age", "sex"]
feature_vals = np.array([[0, 1], [5, 0]], dtype=np.int32)
print(feature_vals)
feature_inputs = {}
feature_inputs["feature_name"] = feature_name
feature_inputs["feature_value"] = feature_vals

# test FNN
model = FNNModel()
labels = [1, 0]
loss_obj = tf.keras.losses.BinaryCrossentropy (from_logits=True)
train_loss = tf.keras.metrics.Mean(name='train_loss')
optimizer = tf.keras.optimizers.Adam(1.0)
with tf.GradientTape(persistent=True) as tape:
    inputs = embedding_layer(feature_inputs)
    predictions = model(tf.concat(inputs, axis=1))
    #print(embedding_layer.trainable_variables[0])
    #print(predictions.shape)
    loss = loss_obj(labels, tf.reduce_sum(predictions, axis=1, keepdims=False))
gradients_1 = tape.gradient(loss, model.trainable_variables)
gradients_2 = tape.gradient(loss, embedding_layer.trainable_variables)
optimizer.apply_gradients(zip(gradients_1, model.trainable_variables))
optimizer.apply_gradients(zip(gradients_2, embedding_layer.trainable_variables))

embedding_init_values = {}
for item in embedding_layer.trainable_variables:
    feature_name_ = item.name.split(":")[0]
    print(feature_name_)
    embedding_init_values[feature_name_] = item

# test Embedding layer
feature_descs["income"] = [4, 32]
feature_descs["sex"] = [4, 32]
embedding_layer_new = EmbeddingLayer(feature_descs, embedding_init_values)
feature_name = ["age", "sex", "income"]
feature_vals = np.array([[0, 1, 2], [5, 0, 3]], dtype=np.int32)
feature_inputs = {}
feature_inputs["feature_name"] = feature_name
feature_inputs["feature_value"] = feature_vals
for item in embedding_layer_new.trainable_variables:
    print(item.name, item.shape)
model = FNNModel()
with tf.GradientTape(persistent=True) as tape1:
    inputs = embedding_layer_new(feature_inputs)
    predictions = model(tf.concat(inputs, axis=1))
    #print(embedding_layer.trainable_variables[0])
    #print(predictions.shape)
    loss = loss_obj(labels, tf.reduce_sum(predictions, axis=1, keepdims=False))
gradients_1 = tape1.gradient(loss, model.trainable_variables)
gradients_2 = tape1.gradient(loss, embedding_layer_new.trainable_variables)
optimizer.apply_gradients(zip(gradients_1, model.trainable_variables))
optimizer.apply_gradients(zip(gradients_2, embedding_layer_new.trainable_variables))
"""


# test FM
feature_sparse_descs = [["age", 10, 32, "int"], ["sex", 3, 32, "int"], ["income", 4, 32, "int"]]
feature_dense_descs = [["ctr", "float"], ["cost", "float"]]
feature_sparse_name = []
feature_dense_name = []
feature_descs = {}
for item in feature_sparse_descs:
    feature_desc_ = FeatureDesc()
    feature_desc_.feature_name = item[0]
    feature_desc_.feature_type = "sparse"
    feature_desc_.feature_cates_num = item[1]
    feature_desc_.embedding_dim = item[2]
    feature_desc_.data_type = item[3]
    feature_descs[feature_desc_.feature_name] = feature_desc_
    feature_sparse_name.append(feature_desc_.feature_name)
for item in feature_dense_descs:
    feature_desc_ = FeatureDesc()
    feature_desc_.feature_name = item[0]
    feature_desc_.feature_type = "dense"
    feature_desc_.data_type = item[1]
    feature_descs[feature_desc_.feature_name] = feature_desc_
    feature_dense_name.append(feature_desc_.feature_name)

embedding_init_inputs = {}
sparse_feas_cates_num = []
for feature_name_ in feature_sparse_name:
    embedding_init_inputs[feature_name_] = [feature_descs[feature_name_].feature_cates_num, feature_descs[feature_name_].embedding_dim]
    sparse_feas_cates_num.append(feature_descs[feature_name_].feature_cates_num)
embedding_layer = EmbeddingLayer(embedding_init_inputs)
one_hot_layer = OneHotLayer()

feature_dense_vals = np.array([[0.4, 3.2], [0.5, 2.3]])
feature_sparse_vals = np.array([[0, 1, 2], [5, 0, 3]], dtype=np.int32)
one_hot_inputs = {}
one_hot_inputs["feature_cates_num"] = sparse_feas_cates_num
one_hot_inputs["feature_value"] = feature_sparse_vals

embedding_feature_inputs = {}
embedding_feature_inputs["feature_name"] = feature_sparse_name
embedding_feature_inputs["feature_value"] = feature_sparse_vals
embedding_feature_inputs["is_combine"] = True

model = FMModel()
with tf.GradientTape(persistent=True) as tape1:
    sparse_feas_embeddings = embedding_layer(embedding_feature_inputs)
    print(sparse_feas_embeddings.shape)
    sparse_feas_one_hot = one_hot_layer(one_hot_inputs) # batch * feas_num * cates_num
    liner_inputs = tf.concat([feature_dense_vals, sparse_feas_one_hot], axis=1)
    print(liner_inputs.shape)
    input_dict = {}
    input_dict["liner_inputs"] = liner_inputs
    input_dict["cross_inputs"] = sparse_feas_embeddings
    predictions = model(input_dict)
    print(predictions)
    #print(embedding_layer.trainable_variables[0])
    #print(predictions.shape)
    #loss = loss_obj(labels, tf.reduce_sum(predictions, axis=1, keepdims=False))


"""
# test Var len Embedding
feature_descs = {}
feature_descs["item"] = [100, 32]
feature_descs["item2"] = [30, 16]
var_len_embedding_layer = VarLenEmbeddingLayer(feature_descs)
item_feature_vals = np.array([[0, 1, 2], [5, 0, 3]], dtype=np.int32)
user_feature_vals = np.array([[0, 1], [5, 0]], dtype=np.int32)
item_feature_inputs = {}
user_feature_inputs = {}
item_feature_inputs["feature_name"] = ["item"]
item_feature_inputs["feature_value"] = item_feature_vals
user_feature_inputs["feature_name"] = ["item2"]
user_feature_inputs["feature_value"] = user_feature_vals
with tf.GradientTape(persistent=True):
    item_embedding = var_len_embedding_layer(item_feature_inputs)
    user_embedding = var_len_embedding_layer(user_feature_inputs)
    print(item_embedding.shape)
    print(user_embedding.shape)
"""
