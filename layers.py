import os
import sys
import logging
import tensorflow as tf

from tensorflow.keras.layers import Dense, Dropout

class FeatureDesc(object):
    def __init__(self):
        self.feature_name = ""
        self.feature_type = "" # dense, sparse, var_sparse
        self.data_type = "string" # string, int, float
        self.feature_cates_num = 2 # for sparse features
        self.embedding_dim = 1 # for sparse features
        self.max_seq_length = 5 # max length for var length features
        self.embedding_dict_name = "" # label embeddings for looking up

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

class EmbeddingLayer(tf.keras.Model):
    def __init__(self, feature_descs={}, embedding_init_values={}):
        # feature_descs's key is feature_name, value are list[feature_length, feature_dim]
        # supportting increament by self
        super().__init__()
        if len(feature_descs) == 0:
            return None
        self.embeddings = {}
        initializer = tf.random_normal_initializer()
        if len(embedding_init_values) == 0:
            for key, value in feature_descs.items():
                self.embeddings[key] = tf.Variable(initializer(shape=(value[0], value[1])), dtype=tf.float32, trainable=True, name="embedding_" + key)
        else:
            for key, value in feature_descs.items():
                if "embedding_" + key in embedding_init_values.keys():
                    origin_len = embedding_init_values["embedding_" + key].shape[0]
                    if origin_len == value[0]:
                        self.embeddings[key] = tf.Variable(embedding_init_values["embedding_" + key], dtype=tf.float32,
                                                           trainable=True, name="embedding_" + key)
                    else:
                        temp = tf.Variable(initializer(shape=(value[0] - origin_len, value[1])), dtype=tf.float32,
                                                           trainable=True, name="embedding_temp_" + key)
                        temp_concat= tf.concat([embedding_init_values["embedding_" + key], temp], axis=0, name="embedding_" + key)
                        self.embeddings[key] = tf.Variable(temp_concat, dtype=tf.float32, trainable=True, name="embedding_" + key)
                else:
                    self.embeddings[key] = tf.Variable(initializer(shape=(value[0], value[1])), dtype=tf.float32,
                                                       trainable=True, name="embedding_" + key)
    def call(self, inputs):
        # inputs are dicts, elem1 is feature_names, elem2 are feature_values, feature_values shape=(batch, feature_num)
        feature_names = inputs["feature_name"] # list shape=(1)
        feature_values = inputs["feature_value"] # numpy array shape=(batch, None)
        outputs = []
        for i in range(len(feature_names)):
            outputs.append(tf.nn.embedding_lookup(params=self.embeddings[feature_names[i]], ids=feature_values[:, i]))
        return outputs # num_feas * batch * dim

class VarLenEmbeddingLayer(tf.keras.Model):
    # 支持变长的ID类特征
    def __init__(self, feature_descs={}, embedding_init_values={}):
        # feature_descs's key is feature_name, value are list[feature_length, feature_dim]
        # supportting increament by self
        super().__init__()
        if len(feature_descs) == 0:
            return None
        self.embeddings = {}
        initializer = tf.random_normal_initializer()
        if len(embedding_init_values) == 0:
            for key, value in feature_descs.items():
                self.embeddings[key] = tf.Variable(initializer(shape=(value[0], value[1])), dtype=tf.float32, trainable=True, name="embedding_" + key)
        else:
            for key, value in feature_descs.items():
                if "embedding_" + key in embedding_init_values.keys():
                    origin_len = embedding_init_values["embedding_" + key].shape[0]
                    if origin_len == value[0]:
                        self.embeddings[key] = tf.Variable(embedding_init_values["embedding_" + key], dtype=tf.float32,
                                                           trainable=True, name="embedding_" + key)
                    else:
                        temp = tf.Variable(initializer(shape=(value[0] - origin_len, value[1])), dtype=tf.float32,
                                                           trainable=True, name="embedding_temp_" + key)
                        temp_concat= tf.concat([embedding_init_values["embedding_" + key], temp], axis=0, name="embedding_" + key)
                        self.embeddings[key] = tf.Variable(temp_concat, dtype=tf.float32, trainable=True, name="embedding_" + key)
                else:
                    self.embeddings[key] = tf.Variable(initializer(shape=(value[0], value[1])), dtype=tf.float32,
                                                       trainable=True, name="embedding_" + key)
    def call(self, inputs):
        # inputs are dicts, elem1 is feature_names, elem2 are feature_values, feature_values shape=(batch, feature_num)
        outputs = []
        feature_names = inputs["feature_name"] # list shape=(1)
        feature_values = inputs["feature_value"] # list, elem shape=(batch, length)
        for i in range(len(feature_names)):
            outputs.append(tf.nn.embedding_lookup(params=self.embeddings[feature_names[i]], ids=feature_values[i]))
        return outputs # num_feas * batch * length * dim

class OneHotLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()
    def call(self, inputs):
        # inputs are dicts
        feature_cates_num = inputs["feature_cates_num"] # list shape=(1)
        feature_values = inputs["feature_value"] # numpy array shape=(batch, None)
        outputs = []
        for i in range(len(feature_cates_num)):
            outputs.append(tf.one_hot(indices=feature_values[:, i], depth=feature_cates_num[i]))
        return tf.concat(outputs, axis=1) # (batch, dim)

class FM(tf.keras.Model):
    def __init__(self):
        super().__init__()
        initializer = tf.random_normal_initializer()
        #self.bias = tf.Variable(initializer(shape=(1, 1)), trainable=True, dtype=tf.float32, name="bias")
        self.liner_layer = Dense(1)

    def call(self, inputs):
        x_liner = inputs["liner_inputs"]
        x_cross = inputs["cross_inputs"]
        liner_out = tf.reduce_sum(self.liner_layer(x_liner), axis=1)

        # compute cross output
        ## sum_square compute
        sum_square = tf.math.pow(tf.reduce_sum(x_cross, axis=1), 2)
        ## square_sum compute
        square_sum = tf.reduce_sum(tf.math.pow(x_cross, 2), axis=1)
        cross_out = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)

        outputs = liner_out + cross_out

        return outputs

class AttentionDIN(tf.keras.Model):
    # target-attention
    def __init__(self, hidden_dims=[80, 40, 1], activate_funcs=["sigmoid", "sigmoid", None]):
        super().__init__()
        if len(hidden_dims) == 0 or len(activate_funcs) == 0:
            return None
        if len(hidden_dims) != len(activate_funcs):
            return None
        self.model = tf.keras.Sequential()
        for hidden_dim, activate_func in zip(hidden_dims, activate_funcs):
            self.model.add(Dense(units=hidden_dim, activation=activate_func, use_bias=True))
    def call(self, inputs):
        # inputs are dicts
        query = inputs["query"]
        key = inputs["key"] # key is val, shape=(batch, length, dim)
        mask = inputs["mask"] # batch

        # construct din inputs
        query_expand = tf.expand_dims(query, axis=1) # batch * 1 * dim
        query_tile = tf.tile(query_expand, [1, tf.shape(key)[1], 1]) # batch * length * dim
        din_all = tf.concat([query_tile, key, query_tile - key, query_tile * key], axis=-1)

        # forward nn
        attention_score = self.model(din_all) # batch * length * 1
        attention_score = tf.transpose(attention_score, [0, 2, 1]) # batch * 1 * T

        # mask
        key_mask = tf.sequence_mask(mask, tf.shape(key)[1]) # batch * length
        key_mask = tf.expand_dims(key_mask, axis=1) # batch * 1 * length
        padding = tf.one_likes(attention_score) * (-2 ** 32 + 1)
        attention_score = tf.where(key_mask, attention_score, padding)

        # output
        attention_score = attention_score / (tf.shape(key)[-1] ** 0.5)
        attention_score = tf.nn.softmax(attention_score)
        outputs = tf.matmul(attention_score, key) # batch * 1 * dim
        return tf.squeeze(outputs) # batch * dim

class TransformerEncoder(tf.keras.Model):
    # construct transformer encoder
    def __init__(self, heads_num, key_dim, value_dim, hidden_dims=[64, 32], activate_funcs=["relu", None]):
        super().__init__()
        self.m_attention = tf.keras.layers.MultiHeadAttention(num_heads=heads_num, key_dim=key_dim, value_dim=value_dim)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.fnn = tf.keras.Sequential()
        for hidden_dim, activate_func in zip(hidden_dims, activate_funcs):
            self.fnn.add(Dense(units=hidden_dim, activation=activate_func, use_bias=True))

    def call(self, inputs):
        # inputs are dicts
        query = inputs["query"] # batch * length * dim
        mask = inputs["mask"] # batch * length
        seq_mask = tf.sequence_mask(mask, tf.shape(query)[1])

        # encoder
        m_attention_out = self.m_attention(query, query, query, seq_mask) # batch * length * dim
        x_add_m = query + m_attention_out
        norm_out_1 = self.norm1(x_add_m)
        fnn_out = self.fnn(norm_out_1) # batch * length * dim
        x_add_fnn = norm_out_1 + fnn_out
        norm_out_2 = self.norm2(x_add_fnn) # batch * length * dim

        return norm_out_2

class TransformerDecoder(tf.keras.Model):
    # construct transformer encoder
    def __init__(self, heads_num=[2, 2], key_dim=[], value_dim=[], hidden_dims=[64, 32], activate_funcs=["relu", None]):
        super().__init__()
        self.m_attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=heads_num[0], key_dim=key_dim[0], value_dim=value_dim[0])
        self.m_attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=heads_num[1], key_dim=key_dim[1], value_dim=value_dim[1])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.norm3 = tf.keras.layers.LayerNormalization()
        self.fnn = tf.keras.Sequential()
        for hidden_dim, activate_func in zip(hidden_dims, activate_funcs):
            self.fnn.add(Dense(units=hidden_dim, activation=activate_func, use_bias=True))
    def call(self, inputs):
        # inputs are dicts
        query = inputs["query"] # batch * length * dim
        encode_value = inputs["encoder_value"] # batch * length1 * dim2
        mask = inputs["mask"] # batch * length
        seq_mask = tf.sequence_mask(mask, tf.shape(query)[1])  # batch * length1 * length1
        encoder_mask = inputs["encoder_mask"] # batch
        encoder_mask = tf.tile(tf.expand_dims(encoder_mask, axis=1), [1, tf.shape(query)[1]]) # batch * length
        encoder_seq_mask = tf.sequence_mask(mask, tf.shape(encode_value)[1]) # batch * length * length1
        # encoder
        m_attention_out_1 = self.m_attention_1(query, query, query, seq_mask) # batch * length * dim
        x_add_m_1 = query + m_attention_out_1
        norm_out_1 = self.norm1(x_add_m_1)
        m_attention_out_2 = self.m_attention_2(norm_out_1, encode_value, encode_value, encoder_seq_mask) # batch * length * dim
        x_add_m_2 = norm_out_1 + m_attention_out_2
        norm_out_2 = self.norm2(x_add_m_2)
        fnn_out = self.fnn(norm_out_2) # batch * length * dim
        x_add_fnn = norm_out_2 + fnn_out
        norm_out_3 = self.norm3(x_add_fnn) # batch * length * dim

        return norm_out_3
