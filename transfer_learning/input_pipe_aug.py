# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:12:30 2018

@author: Sirius

tutorials on pytorch:
https://cs230-stanford.github.io/pytorch-getting-started.html

data API: performace guide
https://www.tensorflow.org/performance/datasets_performance

preprocessing trick:
1. shuffle filenames is cheaper than shuffle dataset
   dataset = dataset.shuffle(buffer_size=len(filenames))
   ref:https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
2. Apply batch before map if map does little work
3. shuffle -->  repeat: at bagining at each epoch, slow down (recommand)
   repeat before shuffle: epoch boundaries are blurred
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import glob
import re
import tensorflow as tf
import random
import numpy as np
from math import ceil

IMAGE_HEIGHT = 64
IMGAE_WIDTH = 64
IMAGE_CHANNELS = 3
NUM_PARALLEL_CALLS = 8

# sirius: 在验证集上也要用 train_mean
# validation_mean = tf.constant([122.54174026, 114.15707305, 101.070021])
# validation_mean = tf.expand_dims(tf.expand_dims(validation_mean, 0), 0)

def build_label_dicts():
  label_dict, class_description = {}, {}
  with open('E:/tiny_imagenet/tiny-imagenet-200/wnids.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset = line[:-1]  # remove \n
      label_dict[synset] = i
  with open('E:/tiny_imagenet/tiny-imagenet-200/words.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
      synset, desc = line.split('\t') # 同义词
      desc = desc[:-1]  # remove \n
      if synset in label_dict:
        class_description[label_dict[synset]] = desc

  return label_dict, class_description

def load_filenames_labels(mode):
  label_dict, class_description = build_label_dicts()
  filenames_labels = []
  if mode == 'train':
    filenames = glob.glob('E:\\tiny_imagenet\\tiny-imagenet-200\\train\\*\\images\\*.JPEG')
    for filename in filenames:
      match = re.search(r'n\d+', filename)
      label = str(label_dict[match.group()])
      filenames_labels.append((filename, label))
  elif mode == 'val':
    with open('E:\\tiny_imagenet\\tiny-imagenet-200\\val\\val_annotations.txt', 'r') as f:
      for line in f.readlines():
        split_line = line.split('\t')
        filename = 'E:\\tiny_imagenet\\tiny-imagenet-200\\val\\images\\' + split_line[0]
        label = str(label_dict[split_line[1]])
        filenames_labels.append((filename, label))
  return filenames_labels

def parse_fn(filename, label):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    # image = tf.image.resize_images(image,[299,299]) # resize image to 299
    return image, label

# 数据不需要预处理
def input_fn(is_training):
    mode = 'train' if is_training else 'val'
    filenames_labels = load_filenames_labels(mode)
    
    filenames = tf.constant([t[0] for t in filenames_labels])
    labels =  tf.constant([t[1] for t in filenames_labels])
    
    labels = tf.string_to_number(labels, tf.int32) # string to number
    labels  = tf.cast(labels, tf.uint8)
    # print('Inside input_fn, labels', labels)
    # Note: 注意这里不能shuffle, 否则train_data和label之间不对应
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    dataset = dataset.map(parse_fn, num_parallel_calls=NUM_PARALLEL_CALLS)
    
    # 应该是在这一步对输入结果打乱了
    dataset = dataset.batch(10000)
    return dataset


#==============================================================================
# 找到label编号与class名称之间的对应关系
# filenames_labels = load_filenames_labels('train')
# label_dict, class_description = build_label_dicts()
# 将数据resize成299X299
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
  # val_data = input_fn(False)
  
train_iterator = train_data.make_one_shot_iterator()
# val_iterator = val_data.make_one_shot_iterator()
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, train_iterator.output_shapes)
images, labels = iterator.get_next()
init = tf.global_variables_initializer()

# 训练数据
with tf.Session() as sess:
    train_iterator_handle = sess.run(train_iterator.string_handle())
    # val_iterator_handle = sess.run(val_iterator.string_handle())
    sess.run(init)
    stime = time.time()
    for i in range(10):
        train_tmp,labels_tmp = sess.run([images,labels], feed_dict = {handle: train_iterator_handle}) 
        with open('E:\\transfer_tiny_imagenet\\data3\\train_data_' + str(i) + '.pkl', 'wb') as file:
             pickle.dump(train_tmp, file)
        with open('E:\\transfer_tiny_imagenet\\data3\\train_labels_' + str(i) + '.pkl', 'wb') as file:
             pickle.dump(labels_tmp, file)
    etime = time.time()
print('total time', (etime-stime)/60) # 将训练集全部转化为np.array并保存 一共用时1分钟





