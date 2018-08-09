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
# mean over three channels
# train: [122.00180258, 113.79270265, 100.92263255]
# validation: [122.54174026, 114.15707305, 101.070021  ]
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import glob
import re
import tensorflow as tf
import random
import numpy as np

# eager execution
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

IMAGE_HEIGHT = 64
IMGAE_WIDTH = 64
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
NUM_PARALLEL_CALLS = 8
NUM_EPOCHS = 100

train_mean = tf.constant([122.00180258, 113.79270265, 100.92263255])
train_mean = tf.expand_dims(tf.expand_dims(train_mean, 0), 0)

validation_mean = tf.constant([122.54174026, 114.15707305, 101.070021])
validation_mean = tf.expand_dims(tf.expand_dims(validation_mean, 0), 0)

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
  """Gets filenames and labels

  Args:
    mode: 'train' or 'val'
      (Directory structure and file naming different for
      train and val datasets)

  Returns:
    list of tuples: (jpeg filename with path, label)
  """
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

# =============================================================================
#                          read from jpeg file
#==============================================================================
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
# tf.image.decode_image: does not return a shape
# tf.read_file(tensor): tensor: one file name


def data_preprocess(img, label, is_training):
    # only need to preprocess image instead of label
    # process one at a time
    # img = value[0]
    # label = value[1]
    if is_training:
        img = img - train_mean                           # center image
        img = tf.random_crop(img, np.array([56, 56, 3])) # random crop(v)
        img = tf.image.random_flip_left_right(img)       # input: 3D
        img = tf.image.random_hue(img, 0.05)             # input: 3D, last channel must be 3
        img = tf.image.random_saturation(img, 0.5, 2.0)  # input: 3D, last channel must be 3
        # img = tf.image.random_brightness(img, max_delta=32.0 / 255.0)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.08) # input: >= 3D
        img = tf.clip_by_value(img,0,255)  # Make sure the image is still in [0, 1]
    else:
        img = img - validation_mean                            # center image
        img = tf.image.crop_to_bounding_box(img, 4, 4, 56, 56) # center box
    return img, label


def parse_fn(filename, label):
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)
    # This will convert to float values in [0, 1]
    # sirius?? 不应该对整个图像进行scale
    image = tf.image.convert_image_dtype(image, tf.float32) # scaling --> casting
    # image = tf.image.resize_images(image, [64, 64])
    return image, label

# dataset.map: takes one element
# dataset transform:takes dataet at once
def input_fn(is_training):
    mode = 'train' if is_training else 'val'
    filenames_labels = load_filenames_labels(mode)
    
    filenames = tf.constant([t[0] for t in filenames_labels])
    labels =  tf.constant([t[1] for t in filenames_labels])
    
    labels = tf.string_to_number(labels, tf.int32) # string to number
    labels  = tf.cast(labels, tf.uint8)
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    if is_training: 
       dataset = dataset.shuffle(buffer_size=len(filenames_labels))
       dataset = dataset.repeat(count = NUM_EPOCHS)
    
    
    dataset = dataset.map(parse_fn, num_parallel_calls=NUM_PARALLEL_CALLS)
    dataset = dataset.map(lambda x,y: data_preprocess(x,y,is_training), num_parallel_calls=NUM_PARALLEL_CALLS)
    dataset = dataset.batch(BATCH_SIZE) #Note: 不显示 shape BatchDataset
    # print('dataset shape', dataset)
    dataset = dataset.prefetch(buffer_size = tf.contrib.data.AUTOTUNE)  
    return dataset

# initialize data in session
with tf.device(':/cpu:0'):
     train_data = input_fn(True)
     val_data = input_fn(False)

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
img, label = iterator.get_next()

train_init = iterator.make_initializer(train_data)
validation_init = iterator.make_initializer(val_data)
# init = tf.group(tf.global_variables_initializer()) # tf.local_variables_initializer()

# 结果没有 batch_size, 由于在下面没有print(sess.run())
with tf.Session() as sess:
    # sess.run(init)
    sess.run(train_init)
    for i in range(3):
        print(sess.run([img, label]))
        print('i',i)
        print('img', img)
        print('label', label)

'''
# example from web
# Reinitializable iterator to switch between Datasets
import tensorflow as tf
import numpy as np
EPOCHS = 10
# making fake data using numpy
train_data = (np.random.sample((100,2)), np.random.sample((100,1)))
test_data = (np.random.sample((10,2)), np.random.sample((10,1)))
# create two datasets, one for training and one for test
train_dataset = tf.data.Dataset.from_tensor_slices(train_data).batch(3)
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(3)
# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_dataset.output_types,
                                           train_dataset.output_shapes)
features, labels = iter.get_next()
# create the initialisation operations
train_init_op = iter.make_initializer(train_dataset)
test_init_op = iter.make_initializer(test_dataset)
with tf.Session() as sess:
    sess.run(train_init_op) # switch to train dataset
    for _ in range(EPOCHS):
        print(sess.run([features, labels]))
    sess.run(test_init_op) # switch to val dataset
    print(sess.run([features, labels]))
'''   
        
        
        
