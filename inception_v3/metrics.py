"""
Tiny ImageNet: Performance Metrics
"""
# straming accuracy
# https://github.com/tensorflow/tensorflow/issues/9498
import tensorflow as tf


def accuracy(logits, labels):
  """Return batch accuracy

  Args:
    logits: logits (N, C) C = number of classes
    labels: tf.uint8 labels {0 .. 199}

  Returns:
    classification accuracy
  """
  labels = tf.cast(labels, tf.int64)
  pred = tf.argmax(logits, axis=1)

  # easy implementaion: tf.reduce_mean(tf.to_float32(predictions == labels))
  acc = tf.contrib.metrics.accuracy(pred, labels) # pred shape same as labels
  # tf.summary.scalar('acc', acc)
  
  # top_5 accuracy
  # tf.metrics.mean: Computes the (weighted) mean of the given values.
  # tf.nn.in_top_k: Says whether the targets are in the top K predictions.
  #                 top_k including ties
  #                 predictions: batch_size X all_classes(predictions for each classes)
  #                 return: batch_size bool tensor
  top_5_bool = tf.nn.in_top_k(predictions=logits, targets=labels, k=5)  # including ties
  acc_5 = tf.reduce_mean(tf.cast(top_5_bool, tf.float32))
  
  '''
  # Note: 定义好计算图以后不能从函数中间输出了
  print('top_5_bool', top_5_bool)
  print('acc_5', acc_5)
  print('Inside metrics, acc_5', acc_5)
  '''
  return acc, acc_5

'''
# 作为验证的例子
import numpy as np
import tensorflow as tf

a = np.random.rand(4,5)
b = a/a.sum(axis=1,keepdims=1)
logits = tf.constant(b)
labels = tf.constant([3,2,1,4])

logits = tf.cast(logits, tf.float32)
top_5_bool = tf.nn.in_top_k(predictions=logits, targets=labels, k=3)
acc_5 = tf.reduce_mean(tf.cast(top_5_bool, tf.float32))

with tf.Session() as sess:
    # logits_eval = sess.run(logits)
    top_5_bool_eval = sess.run(top_5_bool)
    acc_5_eval = sess.run(acc_5)

'''
