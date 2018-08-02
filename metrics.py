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

  # 
  # easy implementaion: tf.reduce_mean(tf.to_float32(predictions == labels))
  acc = tf.contrib.metrics.accuracy(pred, labels) # pred shape same as labels
  tf.summary.scalar('acc', acc)
  
  # top_5 accuracy

  # tf.metrics.mean: Computes the (weighted) mean of the given values.
  # tf.nn.in_top_k: Says whether the targets are in the top K predictions.
  #                 top_k including ties
  #                 predictions: batch_size X all_classes(predictions for each classes)
  #                 return: batch_size bool tensor
  top_5_bool = tf.nn.in_top_k(predictions=logits, targets=labels, k=5)
  acc_5 = tf.reduce_mean(tf.cast(top_5_bool, tf.float32))
  # print('acc_5, shape', len(acc_5))
  # print('acc_5', acc_5)
  # tf.summary.scalar('acc_5', acc_5)

  return acc, acc_5


