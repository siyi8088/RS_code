#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import date, timedelta
import tensorflow as tf
import argparse
import random
import shutil
import glob
import logging
import json
import os

logging.getLogger().setLevel(logging.INFO)
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='', help='Data dir.')
parser.add_argument('--dt_dir', default='', help='Date dir.')
parser.add_argument('--model_dir', default='', help='Model dir.')
parser.add_argument('--model_type', default='', help='Model type {wide, deep, wide&deep}')
parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs.')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')
parser.add_argument('--embedding_size', default=32, type=int, help='Embedding size')
parser.add_argument('--hidden_units', default='256,128,64', help='Hidden units for deep layers.')
parser.add_argument('--clear_existing_model', default=False, type=bool, help='Weather to clear existing model.')
parser.add_argument('--log_steps', default=1000, type=int, help='Save summary every steps.')
parser.add_argument('--task_type', default='train', help='Task type {train, predict, export}')
parser.add_argument('--servable_model_dir', default='', help='Export model dir.')
FLAGS, _ = parser.parse_known_args()


# The columns are tab separeted with the following schema:
# <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
LABEL = ['label']
I_COLUMNS = ['I' + str(i) for i in range(1, 14)]
C_COLUMNS = ['C' + str(i) for i in range(14, 40)]
COLUMNS = LABEL + I_COLUMNS + C_COLUMNS

LABEL_RECORDS = [[0.0]]
I_COLUMNS_RECORDS = [[0.0] for i in range(1, 14)]
C_COLUMNS_RECORDS = [[''] for i in range(14, 40)]
RECORDS_ALL = LABEL_RECORDS + I_COLUMNS_RECORDS + C_COLUMNS_RECORDS


# @tf.function
def input_fn(filename, batch_size=64, num_epoch=1, shuffle=False):
    print('Parsing ', filename)

    def parse_csv(line):
        columns = tf.io.decode_csv(line, record_defaults=RECORDS_ALL, field_delim='\t')
        features = dict(zip(COLUMNS, columns))
        labels = features.pop('label')
        return features, labels

    dataset = tf.data.TextLineDataset(filename).map(parse_csv, num_parallel_calls=10).prefetch(500000)
    dataset = dataset.repeat(num_epoch)
    dataset = dataset.batch(batch_size)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    # features, labels = iterator.get_next()

    return dataset


def build_features():
    """Build features."""
    numerical_features = [tf.feature_column.numeric_column(k) for k in I_COLUMNS]
    categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(c, hash_bucket_size=200) for c in
                            C_COLUMNS]
    embedding_features = [tf.feature_column.embedding_column(k, dimension=FLAGS.embedding_size, combiner='sum') for k in
                          categorical_features]
    # cross_features = tf.feature_column.crossed_column(keys=['cat_col1', 'cat_col2'], hash_bucket_size=2000)

    wide_cols = numerical_features + categorical_features
    deep_cols = numerical_features + embedding_features
    return wide_cols, deep_cols


def build_estimator(wide_cols, deep_cols):
    """Build estimator."""
    hidden_units = list(map(int, FLAGS.hidden_units.split(',')))

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, ' at clear_existing_model.')
        else:
            print('Existing model cleaned at %s' % FLAGS.model_dir)

    config = tf.estimator.RunConfig().replace(
        model_dir=FLAGS.model_dir,
        save_checkpoints_steps=FLAGS.log_steps,
        log_step_count_steps=FLAGS.log_steps,
        save_summary_steps=FLAGS.log_steps
    )

    # build_model:
    if FLAGS.model_type == 'wide':
        estimator = tf.estimator.LinearClassifier(
            feature_columns=wide_cols,
            model_dir=FLAGS.model_dir,
            config=config
        )

    elif FLAGS.model_type == 'deep':
        estimator = tf.estimator.DNNClassifier(
            feature_columns=deep_cols,
            model_dir=FLAGS.model_dir,
            hidden_units=hidden_units,
            config=config
        )

    else:
        estimator = tf.estimator.DNNLinearCombinedClassifier(
            linear_feature_columns=wide_cols,
            dnn_feature_columns=deep_cols,
            dnn_hidden_units=hidden_units,
            model_dir=FLAGS.model_dir,
            config=config
        )

    return estimator


def set_dist_env():
    pass


def main():
    # _______________ check Arguments ___________________
    if FLAGS.dt_dir == '':
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('hidden_units ', FLAGS.hidden_units)
    # print('dropout ', FLAGS.dropout)
    # print('optimizer ', FLAGS.optimizer)
    # print('learning_rate ', FLAGS.learning_rate)
    # print('batch_norm_decay ', FLAGS.batch_norm_decay)
    # print('batch_norm ', FLAGS.batch_norm)
    # print('l2_reg ', FLAGS.l2_reg)

    # ______________________ init Env _______________________
    tr_files = glob.glob('%s/tr*csv' % FLAGS.data_dir)
    random.shuffle(tr_files)
    print('tr_files: ', tr_files)
    va_files = glob.glob('%s/va*csv' % FLAGS.data_dir)
    print('va_files: ', va_files)
    te_files = glob.glob('%s/te*csv' % FLAGS.data_dir)
    print('te_files: ', te_files)

    # ______________________ build Task _______________________
    # features
    wide_cols, deep_cols = build_features()
    # estimator
    estimator = build_estimator(wide_cols, deep_cols)

    if FLAGS.task_type == 'train':
        set_dist_env()
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, batch_size=FLAGS.batch_size, num_epoch=FLAGS.num_epochs)
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, batch_size=FLAGS.batch_size, num_epoch=1)
            # steps=None  # evaluate the whole eval file
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    elif FLAGS.task_type == 'predict':
        predictions = estimator.predict(
            input_fn=lambda: input_fn(te_files, batch_size=FLAGS.batch_size, num_epoch=FLAGS.num_epochs),
            predict_keys='prob'
        )
        with open(FLAGS.data_dir + '/pred.txt', 'w') as w:
            for pred in predictions:
                w.write('%f\n' % pred['prob'][0])

    elif FLAGS.task_type == 'export':
        if FLAGS.model_type == 'wide':
            feature_columns = wide_cols
        elif FLAGS.model_type == 'deep':
            feature_columns = deep_cols
        else:
            feature_columns = wide_cols + deep_cols
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)

        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        estimator.export_saved_model(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == '__main__':
    main()
