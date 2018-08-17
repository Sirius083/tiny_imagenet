# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 16:12:30 2018

@author: Sirius

iterate over batch to calculate whole dataset mean
tiny imagenet dataset
"""

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
tf.enable_eager_execution()
# import tensorflow.contrib.eager as tfe



import glob
import re
import random
import numpy as np

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



def parse_fn(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    return image


# filenames_labels = load_filenames_labels('val') # validation
filenames_labels = load_filenames_labels('train') # train
filenames = tf.constant([t[0] for t in filenames_labels])
dataset = tf.data.Dataset.from_tensor_slices((filenames))
dataset = dataset.map(parse_fn, num_parallel_calls=8)
dataset = dataset.batch(batch_size=100)

'''
# calculate mean
mean_list = []
count = 0
for batch in dataset:
    count = count + 1
    tmp = np.mean(batch,axis =(0,1,2))
    mean_list.append(tmp)
    print(count, tmp)

ml = np.array(mean_list) # change list to np.array
ml_mean = np.mean(ml,axis =0)
'''

# calculate std
from math import sqrt
TRAIN_NUM = 100000
sum_x = 0
sum_x2 = 0
count = 0
for batch in dataset:  # (100,64,64,3)
    sum_x = sum_x + np.sum(batch, axis = (0,1,2))
    batch_2 = np.square(batch)
    sum_x2 = sum_x2 + np.sum(batch_2, axis = (0,1,2)) # elementwise square
    count += 1
    print('count',count)


train_mean = sum_x/TRAIN_NUM
train_std = sqrt(sum_x2/TRAIN_NUM-train_mean**2)
print('train_mean', train_mean)
print('train_std', train_std)


