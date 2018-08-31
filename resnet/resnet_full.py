# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 09:36:35 2018

@author: Sirius

resnet-34版本
"""

import tensorflow as tf
import numpy as np

def batch_norm(inputs, training, name):
  # training: GraphKeys.TRAINABLE_VARIABLES; axis = 3: channel last
  # batch_norm: axis=3--> channels_last, axis=1 --> channels_first
  return tf.layers.batch_normalization(
      inputs=inputs, axis = 1, momentum=0.997, epsilon=1e-5, center=True,
      scale=True, training=training, fused=True, name = name + '_bn') 


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


def building_block(inputs, training, filters, name, kernel_size, first_layer_strides = 1,data_format='channels_first'):


    shortcuts = inputs
    shortcuts = tf.layers.conv2d(inputs, filters = filters, kernel_size = 1, 
                                 strides = first_layer_strides, data_format='channels_first')
    shortcuts = batch_norm(shortcuts, training, name + '_shortcut')
    

    stddev = np.sqrt(2 / (np.prod([kernel_size, kernel_size]) * int(inputs.shape[3])))
    inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size = kernel_size ,padding='same', 
                             strides = first_layer_strides,
                             kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                             name=name + '_1',data_format='channels_first')
    inputs = batch_norm(inputs, training, name + '_1')
    inputs = tf.nn.relu(inputs, name + '_relu')

    inputs = tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,padding='same', 
                             kernel_initializer=tf.random_normal_initializer(stddev=stddev),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0),
                             name=name + '_2',data_format='channels_first')
    inputs = batch_norm(inputs, training, name + '_2')

    inputs = inputs + shortcuts
    inputs = tf.nn.relu(inputs, name + '_relu')

    return tf.identity(inputs, name)

# sirius: 在Input换到channel first 时，conv_2d也要换 
def resnet(images, is_training):

    inputs = tf.cast(images, tf.float32)
    tf.summary.histogram('img', inputs)

    inputs = tf.transpose(inputs, [0, 3, 1, 2])  # performance boost on GPU: channel_last to channel_first
    print('inputs shape', inputs.shape)

    inputs = tf.layers.conv2d(inputs, filters = 64, kernel_size = 7, name = 'conv1', padding='same', data_format='channels_first') # 64 7X7
    inputs = batch_norm(inputs, is_training, name = 'conv1')
    inputs = tf.nn.relu(inputs, name = 'conv1')
    print('conv1 inputs shape', inputs.shape)
    
    # building blocks, 模块的重复次数分别为3,4,6,3
    inputs = building_block(inputs, is_training, filters = 64, name = 'conv2_1', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 64, name = 'conv2_2', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 64, name = 'conv2_3', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    print('conv2 inputs shape', inputs.shape)
    # (N, 64, 56, 56)
    
    inputs = building_block(inputs, is_training, filters = 128, name = 'conv3_1', kernel_size = 3, first_layer_strides = 2, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 128, name = 'conv3_2', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 128, name = 'conv3_3', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 128, name = 'conv3_4', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    print('conv3 inputs shape', inputs.shape)
    # (N, 128, 28, 28)
    
    inputs = building_block(inputs, is_training, filters = 256, name = 'conv4_1', kernel_size = 3, first_layer_strides = 2,data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 256, name = 'conv4_2', kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 256, name = 'conv4_3', kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 256, name = 'conv4_4', kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 256, name = 'conv4_5', kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 256, name = 'conv4_6', kernel_size = 3, first_layer_strides = 1,data_format='channels_first')
    print('conv4 inputs shape', inputs.shape)
    # (N, 256, 14, 14)
    
    inputs = building_block(inputs, is_training, filters = 512, name = 'conv5_1', kernel_size = 3, first_layer_strides = 2, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 512, name = 'conv5_2', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    inputs = building_block(inputs, is_training, filters = 512, name = 'conv5_3', kernel_size = 3, first_layer_strides = 1, data_format='channels_first')
    print('conv5 inputs shape', inputs.shape)
    # (N, 512, 7, 7)

    # global pooling layer
    # inputs = tf.reduce_mean(inputs, [1,2], keepdims = True) # keepdims: If true, retains reduced dimensions with length 1.
    inputs = tf.reduce_mean(inputs, [2,3], keepdims = True)
    print('global pooling inputs shape', inputs.shape)
    # (N, 512)
    
    # fc layer
    inputs = tf.reshape(inputs, [-1, 512])
    inputs = dense(inputs, 200, name='fc')
    print('fc inputs shape', inputs.shape)
    # (N, 200)
    
    return inputs

# sirius: 应该把验证集上错误率最低的模型权重保存下来

