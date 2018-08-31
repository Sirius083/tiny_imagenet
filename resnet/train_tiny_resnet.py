# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:20:29 2018

@author: Sirius


feedable iterator: https://stackoverflow.com/questions/46111072/how-to-use-feedable-iterator-from-tensorflow-dataset-api-along-with-monitoredtra
train tiny imagenet on dataset representation

# 将 train 和 evaluation 放在一个session中间
# 参考链接: https://blog.csdn.net/silence1214/article/details/77876552 

Trick:
1. 只要模型能过拟合，基本都可以调整到好的结果
2. SGD的loss虽然可能不好看，波动率可能也大，但是最后的收敛效果可能比好
3. 从头开始训练时，不需要加BN, dropout,weight_decay等技巧，一个一个加
4. 输入的图像数据需要进行归一化(减均值除以方差)


需要进行的任务：
1. 加上weight decay 进行计算
2. 改变优化算法rmsprop
3. 迁移学习，将原来图像的分辨率增加，用pre-trained模型进行计算
4. 保留最优的20个ckpt，对模型进行ensemble
5. BN中的                                             

增加：
1. 按照epoch进行递减
"""

from resnet_bottleneck import *
from metrics import *
from losses import *
from input_pipe_aug import *
from datetime import datetime
import numpy as np
import os
import shutil
import glob
import time

NUM_EPOCHS = 100
BATCH_SIZE = 64

VAL_DATA_NUM = 10000
TRAIN_DATA_NUM = 100000
TRAIN_STEP = int(ceil(TRAIN_DATA_NUM/BATCH_SIZE))
VAL_STEP   = int(ceil(VAL_DATA_NUM/BATCH_SIZE))
TRAIN_STEP_ALL = int(ceil(TRAIN_DATA_NUM * NUM_EPOCHS/BATCH_SIZE))
STEP_UPDATE = int(ceil(TRAIN_DATA_NUM/BATCH_SIZE)) * 2 # update learning rate every two epoch

class TrainConfig(object):
  """Training configuration"""
  batch_size = 64
  num_epochs = 100     
  
  # 测试用
  summary_interval = 250
  eval_interval = 2000
  
  lr = 0.01     # tiny imagenet: decayed by 0.9 at every epoch  
  reg = 0.0001  # L2正则项的系数

  momentum = 0.9
  model_name = 'resnet'
  config_name = 'resnet_bottleneck'
  continue_train = False
  model = staticmethod(globals()[model_name])  # gets model by name

def options(config):
  # sirius: log and checkpoints under same directory folder
  # import os
  # from datetime import datetime

  now = datetime.now().strftime("%m%d%H%M")
  logdir = "run-{}/".format(now) # log directory name

  model_directory = os.path.join(config.model_name, config.config_name)

  ckpt_path = os.path.join(model_directory, 'checkpoints')
  log_path = os.path.join(model_directory, 'logs', logdir)
  checkpoint = None
  
  if not os.path.isdir(model_directory): # if log directory not exists, create a new one
     os.makedirs(model_directory)
     return ckpt_path, log_path, checkpoint
  else:
    if not config.continue_train:
      return ckpt_path, log_path, checkpoint
    else:
        checkpoint = tf.train.latest_checkpoint(ckpt_path)
        return ckpt_path, log_path, checkpoint

class TrainControl(object):

  def __init__(self, lr, step):
    self.val_accs = []
    self.val_accs_5 = []
    
    self.lr = lr
    self.num_lr_updates = 0
    self.lr_factor = 0.94
    self.step = step # step

  def add_val_acc(self, val_accs):
    self.val_accs.append(val_accs)

  def add_val_acc_5(self, val_accs_5):
    self.val_accs_5.append(val_accs_5)

  def update_lr(self, sess, step):
    if step % STEP_UPDATE == 0: # Note: 由于step%2000==0进入这里，这里选取的数字一定是2000的倍数
      old_lr = sess.run(self.lr)
      self.lr.load(old_lr * self.lr_factor)
      print('========================================')
      print('learning rate updates at step', step) 
      print('current learning', sess.run(self.lr))
    
def model(images, labels, is_training, config):
  
  # batch normalization: moving_mean, moving_variance need to be updated
  # x_norm = tf.layers.batch_normalization(x, training = is_training)
  # update mean and variance in BN
  # manually add dependency of these operations to train_op
  # update train_op is sufficient later on
  # moving averages are not updated by gradient descent, therefore there must be another way to update it
  # bn update its batch statistics, which are non-trainable 

  logits = config.model(images, is_training)

  with tf.name_scope('accuracy'):
       pred = tf.argmax(logits, axis=1)
       acc = tf.contrib.metrics.accuracy(pred, tf.cast(labels, tf.int64))

  with tf.name_scope('accuracy_5'):
       top_5_bool = tf.nn.in_top_k(predictions=logits, targets=tf.cast(labels, tf.int32), k=5)  # including ties
       acc_5 = tf.reduce_mean(tf.cast(top_5_bool, tf.float32))

  with tf.name_scope('loss'):
       ohe = tf.one_hot(tf.cast(labels, tf.int32), 200, dtype=tf.int32)
       train_loss = tf.losses.softmax_cross_entropy(ohe,logits,label_smoothing=0.1)
       l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss') * config.reg
       train_loss = train_loss + l2_loss
       tf.summary.scalar('train_loss', train_loss)
       tf.summary.scalar('l2_loss', l2_loss)

  return train_loss, acc, acc_5
  

def train():
    config = TrainConfig()
    ckpt_path, tflog_path, checkpoint = options(config)

    #  prepare data
    g = tf.Graph()
    with g.as_default():
      
      with tf.device(':/cpu:0'):
          train_data = input_fn(True)
          val_data = input_fn(False)
      
      # data
      train_iterator = train_data.make_one_shot_iterator()
      val_iterator = val_data.make_one_shot_iterator()

      handle = tf.placeholder(tf.string, shape=[])
      iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, 
                                                             train_iterator.output_shapes)
      images, labels = iterator.get_next()
      
      # model
      is_training = tf.placeholder(dtype=tf.bool)
      loss, acc, acc_5 = model(images, labels, is_training, config)


      # optimize
      lr = tf.Variable(config.lr, trainable=False, dtype=tf.float32)
      tf.summary.scalar('lr', lr)
      g_step = tf.Variable(0, trainable=False, name='global_step')
        
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
           train_op = tf.train.MomentumOptimizer(lr, config.momentum, use_nesterov=True).minimize(loss, global_step = g_step)
      
      # summary
      controller = TrainControl(lr, g_step)
      val_loss = tf.Variable(0.0, trainable = False)
      val_acc = tf.Variable(0.0, trainable = False)
      val_acc_5 = tf.Variable(0.0, trainable = False)

      tf.summary.scalar('val_loss',  val_loss)
      tf.summary.scalar('val_acc',   val_acc)
      tf.summary.scalar('val_acc_5', val_acc_5)

      train_loss = tf.Variable(0.0, trainable = False)
      train_acc = tf.Variable(0.0, trainable = False)
      train_acc_5 = tf.Variable(0.0, trainable = False)

      tf.summary.scalar('train_loss', train_loss)
      tf.summary.scalar('train_acc', train_acc)
      tf.summary.scalar('train_acc_5', train_acc_5)

      # init
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      # [tf.summary.histogram(v.name.replace(':','_'),v) for v in tf.trainable_variables()]
      summ = tf.summary.merge_all()
      saver = tf.train.Saver(max_to_keep=10)
      writer = tf.summary.FileWriter(tflog_path, tf.get_default_graph())
      
      # train sess
      gpu_config = tf.ConfigProto()
      gpu_config.gpu_options.allow_growth = True
      
      with tf.Session(config = gpu_config) as sess:
        start_time = time.time()
        train_iterator_handle = sess.run(train_iterator.string_handle()) # 初始化 train 的 iterator
        val_iterator_handle = sess.run(val_iterator.string_handle())     # 初始化 test  的 iterator
        sess.run(init)
        
        if config.continue_train:
           saver.restore(sess, checkpoint)

        losses,accs,accs_5 = [],[],[]
        
        while True:
          try:
             for i in range(TRAIN_STEP_ALL):
                step_loss, _, step, step_acc_5, step_acc, step_summ = sess.run([loss, train_op, g_step, acc_5, acc, summ],
                                                                                feed_dict={handle: train_iterator_handle, is_training:True})
                losses.append(step_loss)
                accs.append(step_acc)
                accs_5.append(step_acc_5)
                
                train_acc.load(step_acc)
                train_acc_5.load(step_acc_5)
                train_loss.load(step_loss)
                writer.add_summary(step_summ, step)
                # 按照inception_V3的训练方法: 初始0.045, 每两个epoch减少到原来的94%
                # controller.update_lr(sess, step)

                if step % config.summary_interval == 0:
                   mean_loss, mean_acc, mean_acc_5 = np.mean(losses), np.mean(accs), np.mean(accs_5) 
                   print('Iteration:{}, Training Loss:{:.3f}, Accuracy:{:.4f}, Accuracy_5:{:.4f}'.
                          format(step, mean_loss,mean_acc,mean_acc_5))
                   losses,accs,accs_5 = [],[],[]


                if step % config.eval_interval == 0:
                    val_losses, val_accs, val_accs_5 = [], [], []      
                    ckpt = saver.save(sess, ckpt_path + '/model', step) 
                    
                    for j in range(VAL_STEP):
                      step_val_loss, step_val_acc, step_val_acc_5, step_eval_summ = sess.run([loss, acc, acc_5, summ],
                                                                    feed_dict={handle: val_iterator_handle, is_training: False})
                      val_losses.append(step_val_loss)
                      val_accs.append(step_val_acc)
                      val_accs_5.append(step_val_acc_5)
                    
                    mean_loss, mean_acc, mean_acc_5 = np.mean(val_losses), np.mean(val_accs), np.mean(val_accs_5)
                    print('Step: {}, Validation. Loss: {:.3f}, Accuracy: {:.4f}, Accuracy_5: {:.4f}'.format(step, mean_loss, mean_acc, mean_acc_5))
                    print('============================================')

                    val_acc.load(mean_acc)      
                    val_acc_5.load(mean_acc_5)
                    val_loss.load(mean_loss)
                    writer.add_summary(step_eval_summ, step)
                    
          except tf.errors.OutOfRangeError:
            break

    print('Total time: {0} hours.'.format((time.time()-start_time)/3600))

if __name__ == "__main__":
  train()
