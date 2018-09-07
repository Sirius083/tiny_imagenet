# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:46:25 2018

@author: Sirius
"""

# using transfered value to retrain network
# 网络训练中自动resize input image??
# 训练结果 0-0.01, 即使是1/200也是0.005,这是怎么回事？？
# 将输入改成299X299，仍然没有训练
# Question：最后一个全连接层不能用relu函数，否则无法训练
# Question: 为什么打乱的训练集的顺序无法训练??
# Question: 训练精度一直保持在 20%-40% 之间，因为每个epoch训练的batch都是重复的??
# Question: 一开始就维持在比较高的水平，并且不训练
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
# init_lr   = 6.5*1e-4
init_lr = 0.00065
# init_lr = 0.01
lr_factor = 0.713
l2_decay  = 0.0116
regularizer = tf.contrib.layers.l2_regularizer(scale=l2_decay)


train_transfer_path = 'E:\\transfer_tiny_imagenet\\train_transfer_all.pickle'
# test_transfer_path = 'E:\\transfer_tiny_imagenet\\data\\test_tra.pkl'

train_label_path = 'E:\\transfer_tiny_imagenet\\train_labels_all.pickle'
# test_transfer_path = 'E:\\transfer_tiny_imagenet\\test_labels.pickle'

with open(train_transfer_path, 'rb') as handle:
	 transfer_values_train = pickle.load(handle) # (100000,2048)

with open(train_label_path,'rb') as handle:
     # 这里是one-hot编码
	 labels_array = pickle.load(handle)          # (100000,)

# y_true --> one hot
# labels_tmp = np.zeros([num_images, num_classes])
# labels_tmp[np.arange(num_images),labels_train] = 1
# labels_train = labels_tmp

def data_shuffle(ind,transfer_values_train,labels_train):
    seed(ind)
    ind_array = np.arange(num_images)
    np.random.shuffle(ind_array)
    transfer_values_train = transfer_values_train[ind_array, :]
    labels_train = labels_train[ind_array]
    return transfer_values_train,labels_train


def random_batch():
    # Create a random index.
    idx = np.random.choice(num_images,size=train_batch_size,replace=False)
    x_batch = transfer_values_train[idx,:]
    y_batch = labels_array[idx]
    return x_batch, y_batch


'''
# example:
x = np.random.rand(3,4)
ind_array = np.arange(3)
np.random.shuffle(ind_array)
x_new = x[ind_array,:]
'''

'''
# 训练前将batch打乱
ind_array = np.arange(num_images)
np.random.shuffle(ind_array)
transfer_values_train = transfer_values_train[ind_array, :]
labels_array = labels_array[ind_array]
'''
        
#==============================================================================
# model defination
model = inception.Inception()
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
# y_true_oh = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true') # one-hot
# y_true_cls = tf.argmax(y_true_oh, dimension=1) # class

# y_true = tf.placeholder(tf.float32, [num_classes], name='y_true')
# labels = tf.cast(y_true, tf.int32)
labels = tf.placeholder(tf.int64, [None], name='y_true')

# neural network
# y_pred = tf.layers.dense(x, num_classes, activation = tf.nn.relu, name = 'fc',kernel_regularizer = regularizer)
# sirius: 最后一个全连接层不能用relu函数,否则无法训练
fc_layer = tf.layers.dense(x, 1024, name = 'fc_layer', kernel_regularizer = regularizer) # logits

# 虽然report中没有说明
logits = tf.layers.dense(fc_layer, num_classes, name = 'logits', kernel_regularizer = regularizer) # logits
y_pred_cls = tf.argmax(logits, axis=1, output_type = tf.int64)

accuracy = tf.contrib.metrics.accuracy(y_pred_cls, labels)   # params: (predict_class, labels)
# 这里的第一个参数是类的数字，而不是one-hot的编码
loss = tf.losses.sparse_softmax_cross_entropy(labels,logits) # params: (class-specific labels, logits)

global_step = tf.Variable(initial_value=0,name='global_step', trainable=False)
lr = tf.Variable(init_lr, trainable=False, name = 'lr', dtype=tf.float32)
correct_prediction = tf.equal(y_pred_cls, labels)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
     optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step)

# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

# 训练集上面的精度
with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    for i in range(num_epochs):
        # if i != 1:
        #    lr.load(sess.run(lr) * lr_factor) # update learning rate
        # 在训练每个epoch前，先将数据打乱
        # train_epoch,labels_epoch = data_shuffle(i,transfer_values_train,labels_array)
        train_epoch,labels_epoch = transfer_values_train,labels_array
        print('start epoch:', i,'learning rate',sess.run(lr))
        
        # sirius: 这里有问题：每次要往后移batch_size个，不是一个
        for j in range(num_step): # each epoch
            # x_batch = train_epoch[j*train_batch_size:(j+1)*train_batch_size,:]
            # y_true_batch = labels_epoch[j*train_batch_size:(j+1)*train_batch_size]
            x_batch, y_true_batch = random_batch() # 这种选样本的方法应该没问题
                        
            feed_dict_train = {x: x_batch, labels: y_true_batch}
            i_global, _ = sess.run([global_step, optimizer],feed_dict=feed_dict_train)
            
            # 每隔100个step记录一次或者最后一步记录一次
            if (i_global % 100 == 0) or (i == num_iterations - 1):
                batch_acc = sess.run(accuracy,feed_dict=feed_dict_train)
                msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
                print(msg.format(i_global, batch_acc))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# tf.reset_default_graph()
