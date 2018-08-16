# -*- coding: utf-8 -*-

__author__ = 'Pan'


# This module is for building a encoder-decoder image generator

import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_model(input_tensor):
    with tf.gfile.FastGFile('emonet/emonet_model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, input_map={'emonet/Cast':input_tensor})

if __name__ == '__main__':
    with tf.Session() as sess:
        with tf.gfile.FastGFile('emonet_model.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def)
        writer = tf.summary.FileWriter('./train_summary', sess.graph)
        writer.close()
