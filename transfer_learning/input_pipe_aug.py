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
    return image, label

# 数据不需要预处理
def input_fn(is_training):
    mode = 'train' if is_training else 'val'
    filenames_labels = load_filenames_labels(mode)
    
    filenames = tf.constant([t[0] for t in filenames_labels])
    labels =  tf.constant([t[1] for t in filenames_labels])
    
    labels = tf.string_to_number(labels, tf.int32) # string to number
    labels  = tf.cast(labels, tf.uint8)
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    dataset = dataset.map(parse_fn, num_parallel_calls=NUM_PARALLEL_CALLS)
    dataset = dataset.batch(10000)
    return dataset

        

