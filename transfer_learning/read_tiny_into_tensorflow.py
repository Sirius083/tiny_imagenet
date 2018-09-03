# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:54:21 2018

@author: Sirius

jpeg --> np.array --> pickle
"""

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
import pickle

with tf.device(':/cpu:0'):
  train_data = input_fn(True)
  val_data = input_fn(False)
  
train_iterator = train_data.make_one_shot_iterator()
val_iterator = val_data.make_one_shot_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, train_iterator.output_shapes)
images, labels = iterator.get_next()
init = tf.global_variables_initializer()
  
with tf.Session() as sess:
    train_iterator_handle = sess.run(train_iterator.string_handle())
    val_iterator_handle = sess.run(val_iterator.string_handle())
    sess.run(init)
    stime = time.time()
    for i in range(10):
        train_decode = sess.run(images, feed_dict = {handle: train_iterator_handle}) 
        # test_decode = sess.run(images, feed_dict = {handle: val_iterator_handle})
        obj_name = 'tiny_train_' + str(i) + '.pickle'
        with open(obj_name, 'wb') as file:
             pickle.dump(train_decode, file)
    etime = time.time()
print('total time', (etime-stime)/60) # 将训练集全部转化为np.array并保存 一共用时1分钟

'''
# using pickle save numpy array to disk
import pickle
with open('tiny_test.pickle', 'wb') as handle:
    # pickle.dump(images_decode, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_decode, handle)

with open('tiny_train.pickle', 'rb') as handle:
    images_decode = pickle.load(handle)
'''
 


        

