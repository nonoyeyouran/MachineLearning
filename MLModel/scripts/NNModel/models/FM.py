import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from MLModels.NNModels.layers import *

class FMModel(keras.Model):
    def __init__(self, feature_descs, feature_names):
        super().__init__()
        self.FM_layer = FM()
        self.feature_descs = feature_descs
        self.sparse_feas_name = feature_names["sparse"]
        self.sparse_feas_cates_num = []
        embedding_init_inputs = {}
        for feature_name_ in self.sparse_feas_name:
            embedding_init_inputs[feature_name_] = [feature_descs[feature_name_].feature_cates_num, feature_descs[feature_name_].embedding_dim]
            self.sparse_feas_cates_num.append(feature_descs[feature_name_].feature_cates_num)
        self.embedding_layer = EmbeddingLayer(embedding_init_inputs)
        self.one_hot_layer = OneHotLayer()
    def call(self, inputs):
        sparse_feas = tf.cast(inputs[:, 0:3], dtype=tf.int32)
        dense_feas = inputs[:, 3:5]

        embedding_feature_inputs = {}
        embedding_feature_inputs["feature_name"] = self.sparse_feas_name
        embedding_feature_inputs["feature_value"] = sparse_feas
        embedding_feature_inputs["is_combine"] = True
        sparse_feas_embedding = self.embedding_layer(embedding_feature_inputs) # list: num_feas * batch * dim
        sparse_feas_embedding = tf.transpose(sparse_feas_embedding, [1, 0, 2]) # batch * num_feas * dim

        one_hot_inputs = {}
        one_hot_inputs["feature_cates_num"] = self.sparse_feas_cates_num
        one_hot_inputs["feature_value"] = sparse_feas
        sparse_feas_one_hot = self.one_hot_layer(one_hot_inputs)

        input_dict = {}
        liner_inputs = tf.concat([dense_feas, sparse_feas_one_hot], axis=1)
        input_dict["liner_inputs"] = liner_inputs
        input_dict["cross_inputs"] = sparse_feas_embedding
        outputs = self.FM_layer(input_dict)
        return outputs

feature_sparse_descs = [["age", 10, 32, "int"], ["sex", 3, 32, "int"], ["income", 4, 32, "int"]]
feature_dense_descs = [["ctr", "float"], ["cost", "float"]]
feature_sparse_name = []
feature_dense_name = []
feature_descs = {}
feature_names = {}
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
feature_names["sparse"] = feature_sparse_name

feature_dense_vals = np.array([[0.4, 3.2], [0.5, 2.3]])
feature_sparse_vals = np.array([[0, 1, 2], [5, 0, 3]], dtype=np.int32)
labels = np.array([1.0, 0.0])
data_num = 1024
batch_size = 32
epoch = 2
feature_vals = []
temp_sample_age = np.random.randint(0, 10, (data_num, 1))
temp_sample_sex = np.random.randint(0, 3, (data_num, 1))
temp_sample_income = np.random.randint(0, 4, (data_num, 1))
temp_sample_ctr = np.random.rand(data_num, 1)
temp_sample_cost = np.random.rand(data_num, 1) * 4
temp_labels = np.random.randint(0, 2, (data_num))
feature_vals = np.concatenate((temp_sample_age, temp_sample_sex, temp_sample_income, temp_sample_ctr, temp_sample_cost), axis=1)
train_data = tf.data.Dataset.from_tensor_slices((feature_vals, temp_labels))
train_data = train_data.map(lambda x, y: ([x[0], x[1], x[2], x[3] * 2, x[4]], y))
# test FM
#model = tf.keras.Sequential([FMModel(feature_descs, feature_names)])
model = FMModel(feature_descs, feature_names)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
count = 0
for i in range(epoch):
    train_data_ = train_data.shuffle(buffer_size=data_num).batch(batch_size=batch_size)
    model.fit(train_data_)
#model.save("FM_model")
#reconstructed_model = tf.keras.models.load_model("FM_model")
#print(model.predict(input_dict))
#print(reconstructed_model.predict(input_dict))
#np.testing.assert_allclose(model.predict(input_dict), reconstructed_model.predict(input_dict))
"""
with tf.GradientTape(persistent=True) as tape1:
    input_dict = {}
    input_dict["sparse"] = feature_sparse_vals
    input_dict["dense"] = feature_dense_vals
    predictions = model(input_dict)
    print(predictions)
    #print(embedding_layer.trainable_variables[0])
    #print(predictions.shape)
    #loss = loss_obj(labels, tf.reduce_sum(predictions, axis=1, keepdims=False))
"""
