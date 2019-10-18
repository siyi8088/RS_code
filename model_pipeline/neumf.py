#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TensorFlow Implementation of《Neural Collaborative Filtering》

by siyi
"""

import tensorflow as tf


class DNN(tf.keras.layers.Layer):

    def __init__(self, hidden_units, activation='relu', l2_reg=0.01, dropout_rate=0.5,
                 use_bn=True, seed=1024, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        self.seed = seed
        self.hidden_layers = None
        self.bn_layers = None
        self.dropout_layers = None

    def build(self, input_shape):
        self.hidden_layers = [tf.keras.layers.Dense(units=self.hidden_units[i],
                                                    activation=self.activation,
                                                    use_bias=True,
                                                    kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                                    bias_regularizer=tf.keras.regularizers.l2(self.l2_reg))
                              for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i)
                               for i in range(len(self.hidden_units))]
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
            output_shape = input_shape[:-1] + (self.hidden_units[-1])
        else:
            output_shape = input_shape

        return tuple(output_shape)

    def get_config(self):
        config = {'hidden_units': self.hidden_units,
                  'activation': self.activation,
                  'l2_reg': self.l2_reg,
                  'dropout_rate': self.dropout_rate,
                  'use_bn': self.use_bn,
                  'seed': self.seed}
        base_config = super().get_config()

        return dict(list(config.items()) + list(base_config.items()))


class NeuMF(tf.keras.Model):

    def __init__(self, user_size, item_size, hidden_units, embedding_size=32, activation='relu',
                 l2_reg=0.01, dropout_rate=0.5, use_bn=True, seed=1024, **kwargs):
        super().__init__()
        self.gmf_user_embedding = tf.keras.layers.Embedding(user_size, embedding_size,
                                                            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.gmf_item_embedding = tf.keras.layers.Embedding(item_size, embedding_size,
                                                            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.mlp_user_embedding = tf.keras.layers.Embedding(user_size, embedding_size,
                                                            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.mlp_item_embedding = tf.keras.layers.Embedding(item_size, embedding_size,
                                                            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dnn = DNN(hidden_units, activation=activation, l2_reg=l2_reg, dropout_rate=dropout_rate,
                       use_bn=use_bn, seed=seed, **kwargs)
        self.dense = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid, use_bias=False)

    def call(self, inputs, model_type='gmf', training=None):
        # print(inputs)
        user_ids, item_ids = inputs['user_id'], inputs['item_id']
        gmf_user_embeddings = self.gmf_user_embedding(user_ids)  # [batch_size, embedding_size]
        gmf_item_embeddings = self.gmf_item_embedding(item_ids)  # [batch_size, embedding_size]
        mlp_user_embeddings = self.mlp_user_embedding(user_ids)  # [batch_size, embedding_size]
        mlp_item_embeddings = self.mlp_item_embedding(item_ids)  # [batch_size, embedding_size]
        # if model_type == 'gmf':
        #     deep_inputs = tf.multiply(gmf_user_embeddings, gmf_item_embeddings)
        #     outputs = self.dense(deep_inputs)
        # elif model_type == 'mlp':
        #     # [batch_size, embedding_size * 2]
        #     deep_inputs = tf.concat([mlp_user_embeddings, mlp_item_embeddings], axis=1)
        #     deep_outputs = self.dnn(deep_inputs, training)
        #     outputs = self.dense(deep_outputs)
        # else:
        # [batch_size, embedding_size]
        gmf_output = tf.multiply(gmf_user_embeddings, gmf_item_embeddings)
        # [batch_size, embedding_size * 2]
        mlp_inputs = tf.concat([mlp_user_embeddings, mlp_item_embeddings], axis=-1)
        # [batch_size, hidden_units[-1]
        mlp_outputs = self.dnn(mlp_inputs, training)
        # [batch_size, embedding_size + hidden_units[-1]]
        neumf_inputs = tf.concat([gmf_output, mlp_outputs], axis=-1)
        outputs = self.dense(neumf_inputs)
        return outputs


if __name__ == '__main__':
    pass
