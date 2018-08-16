# -*- coding: utf-8 -*-

__author__ = 'Pan'


# This module is for building a encoder-decoder image generator

import tensorflow as tf
import tensorflow.contrib.slim as slim


def load_model(input_tensor):
    with tf.gfile.FastGFile('facenet/20170512-110547.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, input_map={'input': input_tensor})