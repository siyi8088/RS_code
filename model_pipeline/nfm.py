#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorFlow Implementation of 《Neural Factorization Machines for Sparse Predictive Analytics》
"""

import tensorflow as tf


class DNN(tf.keras.layers.Layer):

    def __init__(self, hidden_units, activation='relu', use_bn=True,
                 dropout_rate=0.5, l2_reg=0.01, seed=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.seed = seed
        self.hidden_layers = None
        self.dropout_layers = None
        self.bn_layers = None

    def build(self, input_shape):
        self.hidden_layers = [tf.keras.layers.Dense(units=self.hidden_units[i],
                                                    use_bias=True,
                                                    activation=self.activation,
                                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                                    bias_regularizer=tf.keras.regularizers.l2(self.l2_reg))
                              for i in range(len(self.hidden_units))]
        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i)
                               for i in range(len(self.hidden_units))]

        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        super().build(input_shape)

    def call(self, inputs, training=None):
        deep_inputs = inputs
        for i in range(len(self.hidden_units)):
            fc = self.hidden_layers[i](deep_inputs)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)
            fc = self.dropout_layers[i](fc, training=training)
            deep_inputs = fc
        return deep_inputs

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            output_shape = input_shape[: -1] + (self.hidden_units[-1],)
        else:
            output_shape = input_shape
        return output_shape

    def get_config(self):
        config = {'hidden_units': self.hidden_units,
                  'activation': self.activation,
                  'use_bn': self.use_bn,
                  'dropout_rate': self.dropout_rate,
                  'l2_reg': self.l2_reg,
                  'seed': self.seed}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NFMLayer(tf.keras.layers.Layer):
    """
    包含了 Embedding layer 和 Bi-interaction layer.
    """
    def __init__(self, feature_size, field_size, embedding_size=32, l2_reg=0.01, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.l2_reg = l2_reg
        self.embeddings = None

    def build(self, input_shape):
        self.embeddings = tf.keras.layers.Embedding(input_dim=self.feature_size,
                                                    output_dim=self.embedding_size,
                                                    embeddings_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                                    input_length=self.field_size)

    def call(self, inputs):
        print(inputs)
        feature_ids = inputs['feature_ids']  # [batch_size, field_size]
        feature_vals = inputs['feature_vals']  # [batch_size, field_size]
        # feature_ids, feature_vals = inputs
        feature_vals = tf.expand_dims(feature_vals, axis=-1)  # [batch_size, field_size, 1]
        feature_embeddings = self.embeddings(feature_ids)   # [batch_size, field_size, embedding_size]
        feature_embeddings = tf.multiply(feature_vals, feature_embeddings)
        sum_square = tf.square(tf.reduce_sum(feature_embeddings, axis=1))  # [batch_size, embedding_size]
        square_sum = tf.reduce_sum(tf.square(feature_embeddings), axis=1)
        outputs = 0.5 * sum_square - square_sum

        return outputs

    def get_config(self):
        config = {'feature_size': self.field_size,
                  'field_size': self.field_size,
                  'embedding_size': self.embedding_size,
                  'l2_reg': self.l2_reg}
        base_config = super().get_config()

        return dict(list(base_config.items()) + list(config.items()))


class NFM(tf.keras.Model):

    def __init__(self, feature_size, field_size, embedding_size, hidden_units, activation='relu',
                 l2_reg=0.01, dropout_rate=0.5, use_bn=True, seed=1024, **kwargs):
        super().__init__(**kwargs)
        self.nfm_layer = NFMLayer(feature_size, field_size, embedding_size, l2_reg)
        self.dnn_layer = DNN(hidden_units, activation, use_bn, dropout_rate, l2_reg, seed)
        self.dense = tf.keras.layers.Dense(units=1, activation=None, use_bias=False)
        # self.fm = FMLayer()
        self.fm = tf.keras.layers.Dense(units=1, activation=None, use_bias=True)

    @ tf.function(input_signature=[{'feature_ids': tf.TensorSpec(shape=(None, 196), dtype=tf.int32),
                                   'feature_vals': tf.TensorSpec(shape=(None, 196), dtype=tf.float32)}])
    # @tf.function(input_signature=[tf.TensorSpec(shape=(None, 196), dtype=tf.int32),
    #                               tf.TensorSpec(shape=(None, 196), dtype=tf.float32)])
    def call(self, inputs):
        print(inputs)
        x = self.nfm_layer(inputs)
        # x = self.nfm_layer(feature_ids, feature_vals)
        x = self.dnn_layer(x, training=None)
        x = self.dense(x)
        out = self.fm(inputs['feature_vals']) + x
        # The paper dose not have this step.
        outputs = tf.sigmoid(out)
        return outputs
