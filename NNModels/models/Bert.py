import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from MLModels.NNModels.layers import *

class BertSingleSentenceModel(keras.Model):
    # for single sentence classification task
    def __init__(self, feature_descs, feature_names, encoder_num=6, class_num=100):
        super().__init__()
        self.encoder_num = encoder_num
        self.feature_descs = feature_descs
        self.sparse_feas_name = feature_names["var_sparse"]
        self.sparse_feas_cates_num = []
        self.sparse_feas_embedding_dict_name = []
        embedding_init_inputs = {}
        for feature_name_ in self.sparse_feas_name:
            self.sparse_feas_embedding_dict_name.append(feature_descs[feature_name_].embedding_dict_name)
            embedding_init_inputs[feature_descs[feature_name_].embedding_dict_name] = [feature_descs[feature_name_].feature_cates_num, feature_descs[feature_name_].embedding_dim]
            self.sparse_feas_cates_num.append(feature_descs[feature_name_].feature_cates_num)
        self.embedding_layer = VarLenEmbeddingLayer(embedding_init_inputs)
        self.encoders = []
        for i in range(encoder_num):
            self.encoders.append(TransformerEncoder(heads_num=3, key_dim=64, value_dim=64, hidden_dims=[64, 32],
                                                 activate_funcs=["relu", None]))
        self.fnn = Dense(units=class_num, activation="sigmoid", use_bias=False)

    def call(self, inputs):
        var_sparse_feas = tf.cast(inputs["var_sparse_feas"], dtype=tf.int32) # batch * (length + 1)
        var_sparse_feas_mask = tf.cast(inputs["var_sparse_feas_mask"], dtype=tf.int32)  # batch
        var_sparse_feas_mask = tf.tile(tf.expand_dims(var_sparse_feas_mask, axis=1), [1, tf.shape(var_sparse_feas)[-1]]) # batch * length + 1

        # embedding
        embedding_feature_inputs = {}
        embedding_feature_inputs["feature_name"] = self.sparse_feas_embedding_dict_name
        embedding_feature_inputs["feature_value"] = [var_sparse_feas]
        sparse_feas_embedding = self.embedding_layer(embedding_feature_inputs)[0] # list: batch * length + 1 * dim

        # encoder
        encoder_query_inputs = sparse_feas_embedding
        for i in range(self.encoder_num):
            input_dict = {}
            input_dict["query"] = encoder_query_inputs
            input_dict["mask"] = var_sparse_feas_mask
            encoder_query_inputs = self.encoders[i](input_dict) # batch * length + 1 * dim
        encoder_outputs = encoder_query_inputs
        cls_represtation =  tf.squeeze(encoder_outputs[:, 0:1, :], axis=1) # batch * dim

        # classification
        logits = self.fnn(cls_represtation)
        predictions = tf.nn.softmax(logits)

        return predictions, logits

feature_sparse_descs = [["age", 10, 32, "int"], ["sex", 3, 32, "int"], ["income", 4, 32, "int"]]
feature_dense_descs = [["ctr", "float"], ["cost", "float"]]
feature_var_sparse_descs = [["likes", 1000, 32, "string", "text"]]
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
    feature_descs[feature_desc_.feature_name] = feature_desc_
    feature_var_sparse_name.append(feature_desc_.feature_name)
feature_names["var_sparse"] = feature_var_sparse_name

feature_dense_vals = np.array([[0.4, 3.2], [0.5, 2.3]])
feature_sparse_vals = np.array([[0, 1, 2], [5, 0, 3]], dtype=np.int32)
feature_var_sparse_vals = np.array([[1, 3, 5, 2, 23, 56, 0, 0, 0, 0], [11, 3, 51, 32, 203, 56, 67, 100, 0, 0]], dtype=np.int32)
feature_var_sparse_mask = np.array([6, 8], dtype=np.int32)
labels = np.array([7, 9])
data_num = 1024
batch_size = 32
seq_length = 10
epoch = 2
feature_vals = []
temp_text_vals = np.random.randint(0, 1000, (data_num, seq_length))
temp_text_mask = np.random.randint(1, seq_length + 1, (data_num,))
temp_labels = np.random.randint(0, 100, (data_num,))
train_data = tf.data.Dataset.from_tensor_slices((temp_text_vals, temp_text_mask, temp_labels))
# test FM
#model = tf.keras.Sequential([FMModel(feature_descs, feature_names)])
model = BertSingleSentenceModel(feature_descs, feature_names, 3)
count = 0
loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
for i in range(epoch):
    train_data_ = train_data.shuffle(buffer_size=data_num).batch(batch_size=batch_size)
    for step_, (batch_, mask_, label_) in enumerate(list(train_data_.as_numpy_iterator())):
        with tf.GradientTape(persistent=True) as tape:
            input_dict = {}
            input_dict["var_sparse_feas"] = batch_
            input_dict["var_sparse_feas_mask"] = mask_
            predictions, logits = model(input_dict)
            loss = loss_obj(label_, logits)
            print("=========epoch:%d, step:%d===========" % (i + 1, step_ + 1))
            print(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

