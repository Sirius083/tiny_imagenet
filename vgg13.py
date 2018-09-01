# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:36:35 2018

@author: Sirius
结果：
      epoch: 100	
      train_loss: 1.335	
      train_acc:  0.9669	
      train_acc_5: 0.9971	
      val_loss:   3.378	
      val_acc:    0.4555	
      val_acc_5:  0.6763
"""
'''
Architecture is based on VGG-16 model, but the final pool-conv-conv-conv-pool
layers were discarded. The input to the network is a 56x56 RGB crop (versus
224x224 crop for the original VGG-16 model). L2 regularization is applied to
all layer weights. And dropout is applied to the first 2 fully-connected
layers.

1. conv-conv-maxpool
2. conv-conv-maxpool
3. conv-conv-maxpool
4. conv-conv-conv-maxpool
4. fc-4096 (ReLU)
5. fc-2048 (ReLU)
6. fc-200
7. softmax

按道理来说，应该过拟合，但是没有
run1: 实验结果top-1 error 基本保持在0.005
      原因1：没有加L2正则项
      原因2：conv中的权重初始化:slim.conv2d中默认初始化是2010年的Xavier中的，没有用2015年He改进的
run2: 加上l2_weight_decay: 果然是没有加上l2正则，训练结果正常多了
'''
import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
regularizer = tf.contrib.layers.l2_regularizer(1.0)

def dense(inputs, units, name=None):
    """3x3 conv layer: ReLU + He initialization"""
    # He initialization: normal dist with stdev = sqrt(2.0/fan-in)
    # fan_in: number of input units, in fully connected layers
    stddev = np.sqrt(2 / int(inputs.shape[1])) 
    inputs = tf.layers.dense(inputs, units, activation=tf.nn.relu,
                            kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                            name=name)
    return tf.identity(inputs, name)


def vgg_13(images, is_training):
  """VGG-like conv-net

  Args:
    training_batch: batch of images (N, 56, 56, 3)
    config: training configuration object

  Returns:
    class prediction scores
  """
  inputs = tf.cast(images, tf.float32)
  # out = (img - 128.0) / 128.0 # 对输入进行归一化处理

  # tf.summary.histogram('img', training_batch)
  # (N, 56, 56, 3)
  # out = conv_2d(out, 64, (3, 3), 'conv1_1')
  # out = conv_2d(out, 64, (3, 3), 'conv1_2')
  # out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool1')
  with slim.arg_scope([slim.conv2d], weights_regularizer = regularizer, biases_regularizer = regularizer,
                                     padding = 'SAME'):
    inputs = slim.conv2d(inputs, 64, [3,3], scope = 'conv1_1')
    inputs = slim.conv2d(inputs, 64, [3,3], scope = 'conv1_2')
    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2), name='pool1')

    # (N, 28, 28, 64)
    # out = conv_2d(out, 128, (3, 3), 'conv2_1')
    # out = conv_2d(out, 128, (3, 3), 'conv2_2')
    # out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool2')
    inputs = slim.conv2d(inputs, 128, [3,3], scope = 'conv2_1')
    inputs = slim.conv2d(inputs, 128, [3,3], scope = 'conv2_2')
    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2), name='pool2')


    # (N, 14, 14, 128)
    # out = conv_2d(out, 256, (3, 3), 'conv3_1')
    # out = conv_2d(out, 256, (3, 3), 'conv3_2')
    # out = conv_2d(out, 256, (3, 3), 'conv3_3')
    # out = tf.layers.max_pooling2d(out, (2, 2), (2, 2), name='pool3')
    inputs = slim.conv2d(inputs, 256, [3,3], scope = 'conv3_1')
    inputs = slim.conv2d(inputs, 256, [3,3], scope = 'conv3_2')
    inputs = slim.conv2d(inputs, 256, [3,3], scope = 'conv3_3')
    inputs = tf.layers.max_pooling2d(inputs, (2, 2), (2, 2), name='pool3')


    # (N, 7, 7, 256)
    # out = conv_2d(out, 512, (3, 3), 'conv4_1')
    # out = conv_2d(out, 512, (3, 3), 'conv4_2')
    # out = conv_2d(out, 512, (3, 3), 'conv4_3')
    inputs = slim.conv2d(inputs, 512, [3,3], scope = 'conv4_1')
    inputs = slim.conv2d(inputs, 512, [3,3], scope = 'conv4_2')
    inputs = slim.conv2d(inputs, 512, [3,3], scope = 'conv4_3')

    # fc1: flatten -> fully connected layer
    # (N, 7, 7, 512) -> (N, 25088) -> (N, 4096)
    inputs = tf.contrib.layers.flatten(inputs)
    # sirius: 没有写名称的参数按照对应位置对应，否则报错
    inputs = slim.fully_connected(inputs, 4096, scope = 'fc1')
    inputs = tf.nn.dropout(inputs, 0.5)

    # fc2
    # (N, 4096) -> (N, 2048)
    inputs = slim.fully_connected(inputs, 2048, scope = 'fc2')
    inputs = tf.nn.dropout(inputs, 0.5)

    # softmax
    # (N, 2048) -> (N, 200)
    inputs = slim.fully_connected(inputs, 200, scope = 'fc3')

  return inputs
