# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 12:34:51 2018

@author: Sirius
"""
# Note1: 没有resize，是怎么输入的TOT
# Note2: 结果应该是64X2048，但是是1X2048: 由于存储模型的输出是batch_size = 1

# Get current graph: tf.get_default_graph()

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
import sys


def transfer_values_cache(cache_path, model, images=None, image_paths=None):
    """
    This function either loads the transfer-values if they have
    already been calculated, otherwise it calculates the values
    and saves them to a file that can be re-loaded again later.
    Because the transfer-values can be expensive to compute, it can
    be useful to cache the values through this function instead
    of calling transfer_values() directly on the Inception model.
    See Tutorial #08 for an example on how to use this function.
    :param cache_path:
        File containing the cached transfer-values for the images.
    :param model:
        Instance of the Inception model.
    :param images:
        4-dim array with images. [image_number, height, width, colour_channel]
    :param image_paths:
        Array of file-paths for images (must be jpeg-format).
    :return:
        The transfer-values from the Inception model for those images.
    """

    # Helper-function for processing the images if the cache-file does not exist.
    # This is needed because we cannot supply both fn=process_images
    # and fn=model.transfer_values to the cache()-function.
    def fn():
        return process_images(fn=model.transfer_values, images=images, image_paths=image_paths)

    # Read the transfer-values from a cache-file, or calculate them if the file does not exist.
    transfer_values = cache(cache_path=cache_path, fn=fn)

    return transfer_values



# Batch-processing.
def process_images(fn, images=None, image_paths=None):
    """
    Call the function fn() for each image, e.g. transfer_values() from
    the Inception model above. All the results are concatenated and returned.
    
    :param fn:
        Function to be called for each image. e.g. transfer_values()
    :param images:
        List of images to process.
    :param image_paths:
        List of file-paths for the images to process.
    :return:
        Numpy array with the results.
    """

    # Are we using images or image_paths?
    using_images = images is not None

    # Number of images.
    if using_images:
        num_images = len(images)
        # print('inside processing images len',num_images)
    else:
        num_images = len(image_paths)

    # Pre-allocate list for the results.
    # This holds references to other arrays. Initially the references are None.
    result = [None] * num_images

    # For each input image.
    for i in range(num_images):
        # Status-message. Note the \r which means the line should overwrite itself.
        msg = "\r- Processing image: {0:>6} / {1}".format(i+1, num_images)

        # Print the status message.
        sys.stdout.write(msg)
        sys.stdout.flush()

        # Process the image and store the result for later use.
        if using_images:
            result[i] = fn(image=images[i])
        else:
            result[i] = fn(image_path=image_paths[i])

    # Print newline.
    print()

    # Convert the result to a numpy array.
    result = np.array(result)

    return result

def cache(cache_path, fn, *args, **kwargs):
    """
    Cache-wrapper for a function or class. If the cache-file exists
    then the data is reloaded and returned, otherwise the function
    is called and the result is saved to cache. The fn-argument can
    also be a class instead, in which case an object-instance is
    created and saved to the cache-file.

    :param cache_path:
        File-path for the cache-file.

    :param fn:
        Function or class to be called.

    :param args:
        Arguments to the function or class-init.

    :param kwargs:
        Keyword arguments to the function or class-init.

    :return:
        The result of calling the function or creating the object-instance.
    """

    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.
        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj

model = inception.Inception()
path_data = 'E:\\transfer_tiny_imagenet\\data\\tiny_pickle\\' # save images numpy array
data_path = 'E:\\transfer_tiny_imagenet\\data'                # save transfer_value
data_path_list =  ['tiny_train_1.pickle',
                   'tiny_train_2.pickle',
                   'tiny_train_3.pickle',
                   'tiny_train_4.pickle',
                   'tiny_train_5.pickle',
                   'tiny_train_6.pickle',
                   'tiny_train_7.pickle',
                   'tiny_train_8.pickle',
                   'tiny_train_9.pickle']

for dp in data_path_list:
    with open(path_data + 'tiny_train_0.pickle', 'rb') as handle:
         tiny_train = pickle.load(handle)
    file_path_cache_train = os.path.join(data_path, dp.split('.')[0] +'.pkl')
    transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, images=tiny_train,model=model)
        
'''
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(path_graph_def,'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
         # sess.graph.as_default()
    # input: opearation name; return: operation ops
    tv = tf.import_graph_def(graph_def,return_elements=['pool_3:0']) # 从graph中导入tensor的定义
    print('tv', tv)
    
    
    # print all operation names in the graph
    # allops = [n.name for n in tf.get_default_graph().as_graph_def().node]
    # for op_ in allops:
        # print(op_)
    # 直接 feed_dict: np.array
    # for i in range(10):
    batch_transfer_values = sess.run(tv, feed_dict = {"DecodeJpeg":tiny_train_0})
    print('batch_transfer_values',batch_transfer_values)
'''


#==============================================================================
# 将不同 pickle 文件合并成一个，接下来要分batch进行训练
pickle_path = 'E:\\transfer_tiny_imagenet\\data'
pickle_list =  ['tiny_train_1.pkl',
                'tiny_train_2.pkl',
                'tiny_train_3.pkl',
                'tiny_train_4.pkl',
                'tiny_train_5.pkl',
                'tiny_train_6.pkl',
                'tiny_train_7.pkl',
                'tiny_train_8.pkl',
                'tiny_train_9.pkl']

import pickle
with open('E:\\transfer_tiny_imagenet\\data\\tiny_train_0.pkl', 'rb') as handle:
     alldata = pickle.load(handle)
     
for p_ in pickle_list:
    with open('E:\\transfer_tiny_imagenet\\data\\' + p_, 'rb') as handle:
         data = pickle.load(handle)
         alldata = np.concatenate((alldata,data),axis = 0) # (100000,2048)
         
# 存储全部训练集对应的transfer values
with open('tiny_train_all.pickle', 'wb') as handle:
    pickle.dump(alldata, handle)

