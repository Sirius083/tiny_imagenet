from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import inception_utils

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v3_base(inputs,
                      final_endpoint='Mixed_c1',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID',
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(1.0)):
      # 56 x 56 x 3
      end_point = 'Conv2d_1_3x3'
      net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('stem conv1, net', net)

      # 27 x 27 x 32
      end_point = 'Conv2d_2_3x3'
      net = slim.conv2d(net, depth(32), [3, 3], padding = 'SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('stem conv2, net', net)

      # 27 x 27 x 32
      end_point = 'Conv2d_3_3x3'
      net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('stem conv3, net', net)
      # 27 x 27 x 64

    # Inception blocks
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):
      # Mixed_a1: 27 X 27 X 64
      end_point = 'Mixed_a1'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(32), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(16), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(32), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(32), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(48), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(48), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('inception A_1, net', net)

      # Mixed_a2: 27 x 27 x 144.(32+32+48+32)
      end_point = 'Mixed_a2'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(32), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(24), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, depth(32), [5, 5],
                                 scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(32), [1, 1],
                                 scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(48), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(48), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('inception A_2, net', net)
   
      # inceptionA: grid reduction
      # inception_A_grid_reduction : 27 X 27 X 144
      # sirius: ceil[(27-3+1)/2]=13而不是14，给输入左右两边同时补零
      end_point = 'inception_A_grid_reduction'
      paddings = tf.constant([[0,0], [1, 1,], [1, 1], [0,0]]) # (64X27X27X144) 给27X27的左右上下同时补零
      net = tf.pad(net, paddings, "CONSTANT") # 29 X 29 X 144

      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(116), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(16), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(28), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(28), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 14 X 14 X 288 (116+28+144)
      print('inception A grid reducntion, net', net)

      # inception block B
      # mixed_b1: 14 X 14 X 288
      end_point = 'Mixed_b1'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(72), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(48), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(72), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(48), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(48), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(48), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(72), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(72), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('inception B_1, net', net)

      # mixed_b2: 14 X 14 X 288.
      # 在inceptionB中重复的第二个模块中间filter的个数变了
      end_point = 'Mixed_b2'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(72), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(60), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(60), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(72), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(60), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(60), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(60), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(60), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(72), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(72), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('inception B_2, net', net)

      # mixed_b3: 14 x 14 x 288.
      end_point = 'Mixed_b3'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(72), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(60), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(60), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(72), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(60), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(60), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(60), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(60), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(72), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(72), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('inception B_3, net', net)

      # Inception B:
      # sirius: grid reduction
      # 14 x 14 x 288.
      end_point = 'inception_B_grid_reduction'
      paddings = tf.constant([[0,0], [1, 1,], [1, 1], [0,0]])
      net = tf.pad(net, paddings, "CONSTANT")  # 16 x 16 x 288.
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(108), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(180), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(108), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(108), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(108), [7, 1],
                                 scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, depth(108), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 7 X 7 X 576
      print('inception B reduction grid, net', net)

      # Inception C
      # mixed_c1: 7 X 7 X 576
      end_point = 'Mixed_c1'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(144), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(160), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(160), [3, 1], scope='Conv2d_0b_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(160), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(160), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(160), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(80), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      print('inception C_1, net', net)
      # 7 X 7 X 864(160+160+160+160+80+144)
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def inception_v3(inputs,
                 is_training=True,
                 num_classes=200,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 create_aux_logits=False,
                 scope='InceptionV3',
                 global_pool=False):

  with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      
      net, end_points = inception_v3_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)
      print('After inception_v3_base net shape', net)

      # Auxiliary Head logits
      if create_aux_logits and num_classes:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
          aux_logits = end_points['Mixed_6e']
          with tf.variable_scope('AuxLogits'):
            aux_logits = slim.avg_pool2d(
                aux_logits, [5, 5], stride=3, padding='VALID',
                scope='AvgPool_1a_5x5')
            aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                     scope='Conv2d_1b_1x1')

            # Shape of feature map before the final layer.
            kernel_size = _reduced_kernel_size_for_small_input(
                aux_logits, [5, 5])
            aux_logits = slim.conv2d(
                aux_logits, depth(768), kernel_size,
                weights_initializer=trunc_normal(0.01),
                padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
            aux_logits = slim.conv2d(
                aux_logits, num_classes, [1, 1], activation_fn=None,
                normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                scope='Conv2d_2b_1x1')
            if spatial_squeeze:
              aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
            end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.
          kernel_size = [7,7]
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_{}x{}'.format(*kernel_size))
          end_points['AvgPool'] = net
          # 1 X 1 X 864
        if not num_classes:
          return net, end_points
       
        if spatial_squeeze:
          # print('Inside spatial_squeeze net shape', net)
          net = tf.squeeze(net, [1, 2], name='SpatialSqueeze') # 64 X 1 X 1 X 864
        net = slim.fully_connected(net, num_classes, scope = 'fc_200') # Sirius: last fully connected layer
    
      # end_points['Logits'] = logits
      # end_points['Predictions'] = prediction_fn(logits, scope='Predictions')

  # print('last step endpoints', end_points)
  # return logits, end_points
  print('last step net shape', net) # (64,200)
  return net

# inception_v3.default_image_size = 299
inception_v3_arg_scope = inception_utils.inception_arg_scope

# Sirius: 可以检查每层的输出
# inputs = tf.zeros([64,56,56,3])
# inception_v3(inputs)
# inception_v3_base(inputs)
