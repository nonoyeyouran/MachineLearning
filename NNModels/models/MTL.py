import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from MLModels.NNModels.layers import *

class MTLShareBottomModel(keras.Model):
    # multi task learning: share-bottom
    def __init__(self, feature_descs, feature_names,
                 task_num=2,
                 tasks_class_num=[2, 1],
                 fnn_dims=[[64, 32], [64, 32]],
                 fnn_activate_funcs=[["relu", "relu", None], ["relu", "relu", None]]):
        super().__init__()
        self.tasks_class_num = tasks_class_num
        self.feature_descs = feature_descs
        self.dense_feas_num = 0
        self.sparse_feas_num = 0
        self.var_sparse_feas_num = []
        self.sparse_feas_embedding_dict_name = []
        self.var_sparse_feas_embedding_dict_name = []
        self.embedding_layer = None
        self.var_embedding_layer = None
        for feature_type, feas_name in feature_names.items():
            embedding_init_inputs = {}
            if feature_type == "sparse":
                self.sparse_feas_num = len(feas_name)
                for feature_name_ in feas_name:
                    self.sparse_feas_embedding_dict_name.append(feature_descs[feature_name_].embedding_dict_name)
                    embedding_init_inputs[feature_descs[feature_name_].embedding_dict_name] = [feature_descs[feature_name_].feature_cates_num, feature_descs[feature_name_].embedding_dim]
                self.embedding_layer = EmbeddingLayer(embedding_init_inputs)
            elif feature_type == "var_sparse":
                for feature_name_ in feas_name:
                    self.var_sparse_feas_embedding_dict_name.append(feature_descs[feature_name_].embedding_dict_name)
                    self.var_sparse_feas_num.append(feature_descs[feature_name_].max_seq_length)
                    embedding_init_inputs[feature_descs[feature_name_].embedding_dict_name] = [feature_descs[feature_name_].feature_cates_num, feature_descs[feature_name_].embedding_dim]
                self.var_embedding_layer = VarLenEmbeddingLayer(embedding_init_inputs)
            elif feature_type == "dense":
                self.dense_feas_num = len(feas_name)
        self.fnn_layers = [] # tasks fnn
        self.output_layers = []
        for i in range(task_num):
            fnn_layer = tf.keras.Sequential()
            for hidden_dim_, fnn_activate_func_ in zip(fnn_dims[i], fnn_activate_funcs[i]):
                fnn_layer.add(Dense(units=hidden_dim_, activation=fnn_activate_func_, use_bias=True))
            self.fnn_layers.append(fnn_layer)
        for i in range(len(tasks_class_num)):
            self.output_layers.append(Dense(units=tasks_class_num[i], activation=None, use_bias=False))

    def call(self, inputs):
        # dense features + sparse features + [var sparse features; mask_length]
        dense_feas = inputs[:, 0:self.dense_feas_num]
        sparse_feas = tf.cast(inputs[:, self.dense_feas_num:(self.dense_feas_num + self.sparse_feas_num)], dtype=tf.int32)
        var_sparse_feas_list = []
        var_sparse_feas_seq_mask_list = []
        start_index = self.dense_feas_num + self.sparse_feas_num
        for i in range(len(self.var_sparse_feas_num)):
            var_sparse_feas_list.append(tf.cast(inputs[:, start_index:start_index + self.var_sparse_feas_num[i]], dtype=tf.int32))
            start_index = start_index + self.var_sparse_feas_num[i]
            mask_ = tf.squeeze(tf.cast(inputs[:, start_index:start_index + 1], dtype=tf.int32))
            var_sparse_feas_seq_mask_list.append(tf.sequence_mask(mask_, self.var_sparse_feas_num[i])) # batch * length
            start_index += 1

        # sparse and var sparse embedding
        embedding_feature_inputs = {}
        embedding_feature_inputs["feature_name"] = self.sparse_feas_embedding_dict_name
        embedding_feature_inputs["feature_value"] = sparse_feas
        sparse_feas_embedding = self.embedding_layer(embedding_feature_inputs) # list: num_feas * batch * dim
        sparse_feas_embedding_concat = tf.concat(sparse_feas_embedding, axis=1) # batch * [dim * n]

        embedding_feature_inputs["feature_name"] = self.var_sparse_feas_embedding_dict_name
        embedding_feature_inputs["feature_value"] = var_sparse_feas_list
        var_sparse_feas_embedding_list = self.var_embedding_layer(embedding_feature_inputs)  # list: batch * length * dim

        # var sparse handle: avg-pooling
        mask_var_sparse_feas_embedding_list = []
        for var_sparse_feas_embedding_, mask in zip(var_sparse_feas_embedding_list, var_sparse_feas_seq_mask_list):
            padding = tf.zeros_like(var_sparse_feas_embedding_)
            mask_tile = tf.tile(tf.expand_dims(mask, axis=2), [1, 1, tf.shape(var_sparse_feas_embedding_)[-1]])
            mask_var_sparse_feas_embedding = tf.where(mask_tile, var_sparse_feas_embedding_, padding)
            avg_mask_var_sparse_feas_embedding = tf.reduce_mean(mask_var_sparse_feas_embedding, axis=1) # batch * dim
            mask_var_sparse_feas_embedding_list.append(avg_mask_var_sparse_feas_embedding)
        var_sparse_feas_embedding_concat = tf.concat(mask_var_sparse_feas_embedding_list, axis=1) # batch * [dim * n]

        # fnn
        fnn_inputs = tf.concat([dense_feas, sparse_feas_embedding_concat, var_sparse_feas_embedding_concat], axis=1)
        fnn_outputs = []
        for fnn_layer in self.fnn_layers:
            fnn_outputs.append(fnn_layer(fnn_inputs))

        # outputs
        logits = []
        for output_layer, fnn_output in zip(self.output_layers, fnn_outputs):
            logits.append(output_layer(fnn_output))
        predictions = []
        for logit, class_num in zip(logits, self.tasks_class_num):
            if class_num > 1:
                predictions.append(tf.nn.softmax(logit))
            else:
                predictions.append(logit)

        return predictions, logits


data_num = 1024
batch_size = 32
seq_length = 10
epoch = 2
feature_sparse_descs = [["age", 10, 32, "int"], ["sex", 3, 32, "int"], ["income", 4, 32, "int"]]
feature_dense_descs = [["ctr", "float"], ["cost", "float"]]
feature_var_sparse_descs = [["likes", 1000, 32, "string", "items", 10], ["clicks", 1000, 32, "string", "items", 10]]
feature_sparse_name = []
feature_dense_name = []
feature_var_sparse_name = []
feature_descs = {}
feature_names = {}
for item in feature_sparse_descs:
    feature_desc_ = FeatureDesc()
    feature_desc_.feature_name = item[0]
    feature_desc_.feature_type = "sparse"
    feature_desc_.feature_cates_num = item[1]
    feature_desc_.embedding_dim = item[2]
    feature_desc_.data_type = item[3]
    feature_desc_.embedding_dict_name = item[0]
    feature_descs[feature_desc_.feature_name] = feature_desc_
    feature_sparse_name.append(feature_desc_.feature_name)
for item in feature_dense_descs:
    feature_desc_ = FeatureDesc()
    feature_desc_.feature_name = item[0]
    feature_desc_.feature_type = "dense"
    feature_desc_.data_type = item[1]
    feature_descs[feature_desc_.feature_name] = feature_desc_
    feature_dense_name.append(feature_desc_.feature_name)
for item in feature_var_sparse_descs:
    feature_desc_ = FeatureDesc()
    feature_desc_.feature_name = item[0]
    feature_desc_.feature_type = "var_sparse"
    feature_desc_.feature_cates_num = item[1]
    feature_desc_.embedding_dim = item[2]
    feature_desc_.data_type = item[3]
    feature_desc_.embedding_dict_name = item[4]
    feature_desc_.max_seq_length = item[5]
    feature_descs[feature_desc_.feature_name] = feature_desc_
    feature_var_sparse_name.append(feature_desc_.feature_name)
feature_names["var_sparse"] = feature_var_sparse_name
feature_names["sparse"] = feature_sparse_name
feature_names["dense"] = feature_dense_name

feature_dense_vals = np.array([[0.4, 3.2], [0.5, 2.3]])
feature_sparse_vals = np.array([[0, 1, 2], [5, 0, 3]], dtype=np.int32)
feature_var_sparse_vals = np.array([[1, 3, 5, 2, 23, 56, 0, 0, 0, 0], [11, 3, 51, 32, 203, 56, 67, 100, 0, 0]], dtype=np.int32)
feature_var_sparse_mask = np.array([6, 8], dtype=np.int32)
labels = np.array([7, 9])

feature_vals = []
# construct feas
task_one_class_num = 10
age_vals = np.random.randint(0, 10, (data_num, 1))
gender_vals = np.random.randint(0, 3, (data_num, 1))
income_vals = np.random.randint(0, 4, (data_num, 1))
ctr_vals = np.random.rand(data_num, 1)
cost_vals = np.random.rand(data_num, 1) * 4
likes_vals = np.random.randint(0, 1000, (data_num, seq_length))
likes_mask = np.random.randint(1, seq_length + 1, (data_num, 1))
clicks_vals = np.random.randint(0, 1000, (data_num, seq_length))
clicks_mask = np.random.randint(1, seq_length + 1, (data_num, 1))
task_one_labels = np.random.randint(0, task_one_class_num, (data_num, 1))
task_two_labels = np.random.rand(data_num) * 10
input_feas = np.concatenate((ctr_vals, cost_vals, age_vals, gender_vals, income_vals, likes_vals, likes_mask, clicks_vals, clicks_mask), axis=1)
train_data = tf.data.Dataset.from_tensor_slices((input_feas, task_one_labels, task_two_labels))
# test FM
#model = tf.keras.Sequential([FMModel(feature_descs, feature_names)])
model = MTLShareBottomModel(feature_descs, feature_names, 2, [task_one_class_num, 1], [[256, 64], [256, 64]], [["relu", "relu"], ["relu", "relu"]])
count = 0
cate_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
regress_loss_obj = tf.keras.losses.MeanSquaredError()
cate_metric = tf.keras.metrics.CategoricalAccuracy()
regress_metirc = tf.keras.metrics.MeanAbsoluteError()
optimizer = tf.keras.optimizers.Adam()
for i in range(epoch):
    train_data_ = train_data.shuffle(buffer_size=data_num).batch(batch_size=batch_size)
    for step_, (batch_, label_1, label_2) in enumerate(list(train_data_.as_numpy_iterator())):
        with tf.GradientTape(persistent=True) as tape:
            inputs = batch_
            predictions, logits = model(inputs)
            cate_loss = cate_loss_obj(label_1, logits[0])
            regress_loss = regress_loss_obj(label_2, logits[1])
            cate_metric.update_state(tf.one_hot(label_1, task_one_class_num), predictions[0])
            regress_metirc.update_state(label_2, predictions[1])
            loss = cate_loss + regress_loss
            print("[epoch:%d, step:%d]:cate_loss:%f, regeress loss:%f, loss:%f; acc:%f, mae:%f" % (i + 1, step_ + 1, cate_loss, regress_loss, loss, cate_metric.result(), regress_metirc.result()))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

