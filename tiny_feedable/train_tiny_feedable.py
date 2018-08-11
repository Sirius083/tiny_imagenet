# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:20:29 2018

@author: Sirius

train tiny imagenet on dataset representation

cs20 tutorial on tensorflow:
https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/03_logreg.py

Note: 报handle错误的可能是主函数中sess.run([loss]) 中间没有添加 feed_dict={handle: train_iterator_handle}
Question: train 上面的accuracy_5的值为什么都是1.0?? 但是validation上面算的都是正常取值??? excuse me??? 黑人脸

Note: sess.run([]) 中间operation的执行顺序：如果没有dependency时随机run(可能分配到不同的threads上执行)
      如果要求按照顺序，用 with tf.control_dependencies([update_a]):
                              update_b = tf.assign(b, c + a.read_value())
https://stackoverflow.com/questions/41288713/tensorflow-when-are-variable-assignments-done-in-sess-run-with-a-list
"""

from resnet import *
from metrics import *
from losses import *
from input_pipe_aug import *
from datetime import datetime
import numpy as np
import os
import shutil
import glob
import time

class TrainConfig(object):
  """Training configuration"""
  batch_size = 64
  num_epochs = 100     
  # summary_interval = 250
  # eval_interval = 2000    # must be integer multiple of summary_interval
  
  # 测试用
  summary_interval = 250
  eval_interval = 2000    # must be integer multiple of summary_interval


  lr = 0.01   
  reg = 0.0001 
  momentum = 0.9
  model_name = 'resnet_11'
  config_name = 'resnet_11_input_dataset'
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
    self.lr_factor = 1/10
    self.step = step # step

  def add_val_acc(self, val_accs):
    self.val_accs.append(val_accs)

  def add_val_acc_5(self, val_accs_5):
    self.val_accs_5.append(val_accs_5)

  def update_lr(self, sess, step):
    if len(self.val_accs) < 3:
      return
    decrease = False
    
    # sirius: 不能整除, 由于step一般都不是整数
    if step % 45000==0:
      decrease = True

    if decrease:
      old_lr = sess.run(self.lr)
      self.lr.load(old_lr * self.lr_factor)
      print('========================================')
      print('learning rate updates at step', step) 
      print('current learning', self.lr)
    
def model(images, labels, config, is_training):
  
  logits = config.model(images, is_training)
  softmax_ce_loss(logits, labels)
  acc, acc_5 = accuracy(logits, labels)
  
  # sirius: 查看到底是那一步算错了
  '''
  labels = tf.cast(labels, tf.int64)
  pred = tf.argmax(logits, axis=1)
  acc = tf.contrib.metrics.accuracy(pred, labels)
  top_5_bool = tf.nn.in_top_k(predictions=logits, targets=labels, k=5)
  acc_5 = tf.reduce_mean(tf.cast(top_5_bool, tf.float32))
  print('Inside model, top_5_bool', top_5_bool)
  print('Inside model, acc_5', acc_5)
  '''
  total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')
  total_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                         name='total_loss') * config.reg
  for l2 in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
    name = 'l2_loss_' + l2.name.split('/')[0]
    tf.summary.histogram(name, l2)

  return total_loss, acc, acc_5


def optimizer(loss, config):
  """Add training operation, global_step and learning rate variable to Graph

  Args:
    loss: model loss tensor
    config: training configuration object

  Returns:
    (train_op, global_step, lr)
  """
  lr = tf.Variable(config.lr, trainable=False, dtype=tf.float32)
  tf.summary.scalar('lr', lr) # sirius

  global_step = tf.Variable(0, trainable=False, name='global_step')

  
  optim = tf.train.MomentumOptimizer(lr, config.momentum,
                                     use_nesterov=True)
  train_op = optim.minimize(loss, global_step=global_step)

  return train_op, global_step, lr


def train():
    config = TrainConfig()
    ckpt_path, tflog_path, checkpoint = options(config)

    #  prepare data
    g = tf.Graph()
    with g.as_default():
      with tf.device(':/cpu:0'):
          train_data = input_fn(True)
          val_data = input_fn(False)
      
      # feedable iterator: train data pipeline 和 validation data pipeline在一个session中
      train_iterator = train_data.make_one_shot_iterator()
      val_iterator = val_data.make_one_shot_iterator()

      handle = tf.placeholder(tf.string, shape=[])
      iterator = tf.data.Iterator.from_string_handle(handle, train_iterator.output_types, 
                                                             train_iterator.output_shapes)
      images, labels = iterator.get_next()
      #=======================================
      # training process
      # sirius: ValueError: not enough values to unpack (expected 3, got 2)
      #         问题：模型返回3个值，忘记添加acc_5
      loss, acc, acc_5 = model(images, labels, config, is_training=True)
      
      # Note: 在validation集合上需要设置is_training = False
      # Note: 在 tensorflow官网上用的estimator中有个input_fn
      # loss_val, acc_val, acc_5_val = model(images, labels, config, is_training=False)
      train_op, g_step, lr = optimizer(loss, config)
      
      
      # 没有直接summary(loss,acc,acc_5),是想要把训练集和验证集的统计数据分开
      # validation summary variables
      # controller = TrainControl(lr, g_step) # save validation statistics
      # 只是定义，这里并没有给出更新规则
      val_loss = tf.Variable(0.0, trainable = False)
      val_acc = tf.Variable(0.0, trainable = False)
      val_acc_5 = tf.Variable(0.0, trainable = False)

      tf.summary.scalar('val_loss',  val_loss)
      tf.summary.scalar('val_acc',   val_acc)
      tf.summary.scalar('val_acc_5', val_acc_5)

      # train variables
      train_loss = tf.Variable(0.0, trainable = False)
      train_acc = tf.Variable(0.0, trainable = False)
      train_acc_5 = tf.Variable(0.0, trainable = False)

      tf.summary.scalar('train_loss', train_loss) # Note: 是这里把名字写错了，所以tensorboard画出来的图有问题
      tf.summary.scalar('train_acc', train_acc)
      tf.summary.scalar('train_acc_5', train_acc_5)

      # 是不包括train_init和test_init的
      init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      [tf.summary.histogram(v.name.replace(':','_'),v) for v in tf.trainable_variables()]
      extra_updates_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

      # Note: summary
      summ = tf.summary.merge_all()
      saver = tf.train.Saver(max_to_keep=5)
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
        # for i in range(NUM_EPOCH): 这里取1时，count(step) = 156250 
        # 这里不需要，由于在dataset中已经dataset.repeat(NUM_EPOCHS)
        
        losses,accs,accs_5 = [],[],[] # 每个summary interval的平均损失

        for i in range(TRAIN_STEP_ALL): # 总共运行多少个step
              step_loss, _, step, step_acc_5, step_acc, _ = sess.run([loss, train_op, g_step, acc_5, acc, extra_updates_ops],
                                                                      feed_dict={handle: train_iterator_handle})
              losses.append(step_loss)
              accs.append(step_acc)
              accs_5.append(step_acc_5)
              print('train, step_acc_5', step_acc_5, 'step_acc', step_acc)
              # ======================================
              #     calculate validation accuracy
              # ======================================
              # validation:
              # 1. feed_dict数据不同
              # 2. model计算时由于batch normalization需要设置为False
              # 3. 不需要更新权重参数等
              # Note: 在测试阶段，eval_interval 和 summary_inerval可以设置小一点
              # val_iterator_handle: 初始化时候也需要repeat, 或者这里重新初始化
              #                      否则会报错 OutOfRangeError: End of sequence
              # Question: 在evaluation中用model进行计算，需要传参数is_training = False, 但是这样就会重复定义权重等参数
              if step % config.eval_interval == 0:
                  losses_val, accs_val, accs_5_val = [], [], [] # 在validation上的计算也是按batch进行计算的
                  for j in range(VAL_STEP):                       # 每进行一次validation需要多少个step
                    # use trained weights to calculate outputs on validation dataset
                    # 不需要给模型的参数加上reuse = True(变量共享)
                    step_loss_val, step_acc_val, step_acc_5_val = sess.run([loss, acc, acc_5],
                                                                  feed_dict={handle: val_iterator_handle})
                    # print('validation step_acc_5_val',step_acc_5_val)
                    losses_val.append(step_loss_val)
                    accs_val.append(step_acc_val)
                    accs_5_val.append(step_acc_5_val)
                  
                  # 计算当前 eval_interval的validation
                  mean_loss, mean_acc, mean_acc_5 = np.mean(losses), np.mean(accs), np.mean(accs_5)
                  print('Step: {}, Validation. Loss: {:.3f}, Accuracy: {:.4f}, Accuracy_5: {:.4f}'.format(step, mean_loss, mean_acc, mean_acc_5))
                  print('============================================')

                  val_acc.load(mean_acc)    # 本次evaluation计算的均值，load new value to the variable
                  val_acc_5.load(mean_acc_5)
                  val_loss.load(mean_loss)
                  
                  # controller.add_val_acc(mean_acc)
                  # controller.add_val_acc_5(mean_acc_5)
                  # controller.update_lr(sess,step)
              
              if step % config.summary_interval == 0:
                 # sirius: summary 需要feed_dict 否则报错
                 # Error: You must feed a value for placeholder tensor 'Placeholder' with dtype string
                 #        summary 是 validation 数据集不够用的
                 # Note: summary 这里需要加上feed_dict计算那要保存的值
                 # Question: train上的accs_5都是1.0??
                 # Question: summary中间用train_iterator是不是会将本身需要训练的样本也用掉了
                 mean_loss, mean_acc, mean_acc_5 = np.mean(losses), np.mean(accs), np.mean(accs_5)
                 train_acc.load(mean_acc) # 只有load了才会添加operation, 所以sess.run() 才有赋值，否则没有
                 train_acc_5.load(mean_acc_5)
                 train_loss.load(mean_loss)
                 
                 writer.add_summary(sess.run(summ, feed_dict={handle: train_iterator_handle}), step)

                 print('Iteration:{}, Training Loss:{:.3f}, Accuracy:{:.4f}, Accuracy_5:{:.4f}'.
                            format(step, mean_loss,mean_acc,mean_acc_5)) # 当前summary的数据
                 losses,accs,accs_5 = [],[],[] # 每个summary interval之后重置

    print('Total time: {0} seconds.'.format(time.time()-start_time))
    
if __name__ == "__main__":
  train()
