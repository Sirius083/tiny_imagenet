# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:34:51 2018

@author: Sirius
"""
# Note1: 没有resize，是怎么输入的TOT
# Note2: 结果应该是64X2048，但是是1X2048
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import sys
import inception
from input_pipe_aug import *
from tensorflow.python.platform import gfile

with tf.device(':/cpu:0'):
  train_data = input_fn(True)


train_iterator = train_data.make_one_shot_iterator()
val_iterator = val_data.make_one_shot_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, train_iterator.output_shapes)
images, labels = iterator.get_next()
init = tf.global_variables_initializer()


GRAPH_PB_PATH = r'E:\transfer_tiny_imagenet\inception\classify_image_graph_def.pb'
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
         sess.graph.as_default()
         tv = tf.import_graph_def(graph_def,return_elements=['pool_3:0']) # 从graph中导入tensor的定义
         print('tv', tv)
         
         train_iterator_handle = sess.run(train_iterator.string_handle())
         val_iterator_handle = sess.run(val_iterator.string_handle())
         sess.run(init)

         for i in range(1):
             batch_transfer_values = sess.run(tv, feed_dict = {images:images})
print('batch_transfer_values',batch_transfer_values)
