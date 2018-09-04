# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 15:54:21 2018

@author: Sirius

将训练样本分成10组，以pickle形式保存在硬盘里
jpeg --> np.array --> pickle
Q: 无法训练??
images 和 label一定要对应起来
将输入图片大小在预处理阶段-->299X299
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


def process_images(images,model):
    num_images = len(images)
    result = [None] * num_images

    for i in range(num_images):
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)
        sys.stdout.write(msg)
        sys.stdout.flush()

    result[i] = model.transfer_values(image=images[i])
    print()
    result = np.array(result)
    return result

#==============================================================================
with tf.device(':/cpu:0'):
  train_data = input_fn(True)
  val_data = input_fn(False)
  
train_iterator = train_data.make_one_shot_iterator()
val_iterator = val_data.make_one_shot_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, train_iterator.output_shapes)
images, labels = iterator.get_next()
init = tf.global_variables_initializer()

'''
# 训练数据
with tf.Session() as sess:
    train_iterator_handle = sess.run(train_iterator.string_handle())
    val_iterator_handle = sess.run(val_iterator.string_handle())
    sess.run(init)
    stime = time.time()
    # Note1: 第一次训练没有结果，images和labels需要对应
    for i in range(10):
        train_tmp,labels_tmp = sess.run([images,labels], feed_dict = {handle: train_iterator_handle}) 
        # test_decode = sess.run(images, feed_dict = {handle: val_iterator_handle})
        with open('E:\\transfer_tiny_imagenet\\data2\\train_data_' + str(i) + '.pkl', 'wb') as file:
             pickle.dump(train_tmp, file)
        with open('E:\\transfer_tiny_imagenet\\data2\\train_labels_' + str(i) + '.pkl', 'wb') as file:
             pickle.dump(labels_tmp, file)
        etime = time.time()
print('total time', (etime-stime)/60) # 将训练集全部转化为np.array并保存 一共用时1分钟
'''
'''
# 测试数据
with tf.Session() as sess:
    val_iterator_handle = sess.run(val_iterator.string_handle())
    sess.run(init)
    stime = time.time()
    train_tmp,labels_tmp = sess.run([images,labels], feed_dict = {handle: val_iterator_handle}) 
    with open('E:\\transfer_tiny_imagenet\\data_new\\test_data.pkl', 'wb') as file:
         pickle.dump(train_tmp, file)
    with open('E:\\transfer_tiny_imagenet\\data_new\\test_labels.pkl', 'wb') as file:
         pickle.dump(labels_tmp, file)
    etime = time.time()
print('total time', (etime-stime)/60) # 将训练集全部转化为np.array并保存 一共用时1分钟
'''
model = inception.Inception()

# 训练集，直接保存transfer_value
# 由于np.array train data 太大
with tf.Session() as sess:
    train_iterator_handle = sess.run(train_iterator.string_handle())
    val_iterator_handle = sess.run(val_iterator.string_handle())
    sess.run(init)
    stime = time.time()
    # Note1: 第一次训练没有结果，images和labels需要对应
    for i in range(10):
        train_tmp,labels_tmp = sess.run([images,labels], feed_dict = {handle: train_iterator_handle}) 
        # test_decode = sess.run(images, feed_dict = {handle: val_iterator_handle})
        transfer_values_train = process_images(images = train_tmp, model=model)
        print('========== process %d done'%i)
        del train_tmp
        # save transfer value directly
        with open('E:\\transfer_tiny_imagenet\\data2\\train_data_' + str(i) + '.pkl', 'wb') as file:
             pickle.dump(transfer_values_train, file)
        with open('E:\\transfer_tiny_imagenet\\data2\\train_labels_' + str(i) + '.pkl', 'wb') as file:
             pickle.dump(labels_tmp, file)
        del labels_tmp,transfer_values_train
        
    etime = time.time()
print('total time', (etime-stime)/60) # 将训练集全部转化为np.array并保存 一共用时1分钟





        

