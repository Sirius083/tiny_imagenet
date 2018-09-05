# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 11:18:43 2018

@author: Sirius

加载数据时候检查 
1.图片和标签是否对应；
2.检查计算的transfer values的聚类是否明显
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import inception


def plot_images(images, cls_true, cls_pred=None, smooth=True):

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3, figsize=(15,15)) # change subplot figure size
    # figsize=(15,15)
    
    from matplotlib.pyplot import figure
    figure(figsize=(1,1))
    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            # cls_true_name = class_names[cls_true[i]] # cifar10
            cls_true_name = class_description[cls_true[i]].split(',')[0]   # tiny imagenet
            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
    

'''
# ================================================================ cifar10
import cifar10
from cifar10 import num_classes
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()
class_names

# cls_train:    是 0-9的类别编号
# labels_train: 是 one-hot类型的类别编号
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test    = cifar10.load_test_data() 

# Get the first images from the test-set.
images = images_test[0:9]

# Get the true classes for those images.
cls_true = cls_test[0:9]

# Plot the images and labels using our helper-function above.
# images: (9,32,32,3)
# cls_true: (9,)
plot_images(images=images, cls_true=cls_true, smooth=False)
'''

#=============================================== tiny imagenet
# 准备数据
train_data_path = r'E:\transfer_tiny_imagenet\data_new\train_data_5.pkl'
train_label_path = r'E:\transfer_tiny_imagenet\data_new\train_labels_5.pkl'

import pickle
with open(train_data_path, 'rb') as handle:
     train_data = pickle.load(handle)
     
with open(train_label_path, 'rb') as handle:
     train_label = pickle.load(handle)

# 从 100000 中任选 9 张
idx = np.random.choice(10000,size=9,replace=False)
images = train_data[idx,:,:,:]
cls_true = train_label[idx]

# 找到class
from input_pipe_aug import build_label_dicts 
label_dict, class_description = build_label_dicts()

'''
label_list = ''
for cls_ in cls_true:
    label_list = label_list + class_description[cls_] + '_'
    # print(class_description[cls_])
    # print(type(class_description[cls_]))
label_list = label_list.split('_')[:-1]
'''


# plot_images(images=images, cls_true=label_list, smooth=False) 
plot_images(images=images, cls_true=cls_true, smooth=False) 

'''
inception.maybe_download()
model = inception.Inception()
from inception import transfer_values_cache
file_path_cache_train = os.path.join(os.getcwd(), 'inception_tiny_train.pkl')
file_path_cache_test = os.path.join(os.getcwd(), 'inception_tiny_test.pkl')


# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
images_scaled = images_train * 255.0
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

images_scaled = images_test * 255.0
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)

'''
