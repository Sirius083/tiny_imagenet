# 这个程序有问题，就没有训练，不明原因
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:46:25 2018

@author: Sirius
"""

# using transfered value to retrain network
# 网络训练中自动resize input image??
import time
import numpy as np
import tensorflow as tf
from datetime import timedelta
import inception
import pickle
from random import seed

num_images = 100000
train_batch_size = 100 # report中说batch_size影响不大
num_step = int(num_images/train_batch_size)
num_epochs = 100
num_iterations = num_step * num_epochs     
num_classes = 200
transfer_len = 2048
# tiny imagenet report中调整的参数
# init_lr = 6.5*1e-4 # decay by 0.713 per epoch, 经过10个epoch模型就收敛了
init_lr   = 6.5*1e-4
lr_factor = 0.713
l2_decay  = 0.0116
regularizer = tf.contrib.layers.l2_regularizer(scale=l2_decay)


train_transfer_path = 'E:\\transfer_tiny_imagenet\\data\\train_transfer.pickle'
test_transfer_path = 'E:\\transfer_tiny_imagenet\\data\\test_transfer.pkl'

train_label_path = 'E:\\transfer_tiny_imagenet\\data\\label_train.pickle'
test_transfer_path = 'E:\\transfer_tiny_imagenet\\data\\label_test.pickle'

with open(train_transfer_path, 'rb') as handle:
	 transfer_values_train = pickle.load(handle) # (100000,2048)

with open(train_label_path,'rb') as handle:
	 labels_train = pickle.load(handle)          # (100000,)

# y_true --> one hot
labels_tmp = np.zeros([num_images, num_classes])
labels_tmp[np.arange(num_images),labels_train] = 1
labels_train = labels_tmp


def data_shuffle(ind,transfer_values_train,labels_train):
    seed(ind)
    ind_array = np.arange(num_images)
    np.random.shuffle(ind_array)
    transfer_values_train = transfer_values_train[ind_array, :]
    labels_train = labels_train[ind_array,:]
    return transfer_values_train,labels_train
    

#==============================================================================
# model defination
model = inception.Inception()
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# neural network
y_pred = tf.layers.dense(x, num_classes, activation = tf.nn.relu, name = 'fc',kernel_regularizer = regularizer)
y_pred_cls = tf.argmax(y_pred, dimension=1)
labels = tf.cast(y_true_cls, tf.int32)
loss = tf.losses.sparse_softmax_cross_entropy(labels,y_pred)

global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)
lr = tf.Variable(init_lr, trainable=False, name = 'lr', dtype=tf.float32)

optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

# 训练集上面的精度
with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    for i in range(num_epochs):
        if i != 1:
           lr.load(sess.run(lr) * lr_factor) # update learning rate
        # 在训练每个epoch前，先将数据打乱
        train_epoch,labels_epoch = data_shuffle(i,transfer_values_train,labels_train)
        print('start epoch:', i)
        for j in range(num_step): # each epoch
            x_batch = train_epoch[j:j+train_batch_size,:]
            y_true_batch = labels_epoch[j:j+train_batch_size,:]
            
            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            i_global, _ = sess.run([global_step, optimizer],feed_dict=feed_dict_train)
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                batch_acc = sess.run(accuracy,feed_dict=feed_dict_train)
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# tf.reset_default_graph()
