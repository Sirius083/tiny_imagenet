# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:20:29 2018

@author: Sirius

train tiny imagenet on dataset representation
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

from inceptionV3 import *
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
  
  # 测试用
  summary_interval = 250
  eval_interval = 2000    # must be integer multiple of summary_interval
  
  lr = 0.001 # tiny imagenet: decayed by 0.9 at every epoch  
  reg = 0.08

  momentum = 0.9
  model_name = 'inception_v3'
  config_name = 'inception_v3_2'
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
    print('Inside update_lr. self.lr, step ', sess.run(self.lr), step)
    # sirius: 不能整除, 由于step一般都不是整数
    if step % LR_CHANGE_STEP == 0: # Note: 由于step%2000==0进入这里，这里选取的数字一定是2000的倍数
      old_lr = sess.run(self.lr)
      self.lr.load(old_lr * self.lr_factor)
      print('========================================')
      print('learning rate updates at step', step) 
      print('current learning', sess.run(self.lr))
    
def model(images, labels, config, is_training, reuse = None):
  
  logits = config.model(images, is_training) # sirius: 只需要給模型傳入iamges和is_training
  softmax_smooth_ce_loss(logits, labels)
  acc, acc_5 = accuracy(logits, labels)
  # print('Inside train_tiny_inception, model, tf.get_collection(tf.GraphKeys.LOSSES)', tf.get_collection(tf.GraphKeys.LOSSES))
  # print('Inside train_tiny_inception, model, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)', tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  total_loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES), name='total_loss')
  '''
  total_loss += tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                          name='total_loss') * config.reg
  for l2 in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
    name = 'l2_loss_' + l2.name.split('/')[0]
    tf.summary.histogram(name, l2)
  '''

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
      
      # handle = tf.placeholder(tf.string, shape=[])
      # train_str, val_str, images, labels = get_samples(handle)
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
      controller = TrainControl(lr, g_step) # save validation statistics
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
      
      # sirius: save summries every N step
      # summary_hook = tf.train.SummarySaverHook(save_steps=config.summary_interval,
      #                                          output_dir=tflog_path, summary_op=tf.summary.merge_all())
      
      # with tf.train.MonitoredTrainingSession(hooks=[summary_hook]) as sess:
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
        train_count = 0
        summary_count = 0
        while True:
          try:
             for i in range(TRAIN_STEP_ALL): # 总共运行多少个step
                # Note: 如果每个step都运行sess.run(step) 太慢了
                step_loss, _, step, step_acc_5, step_acc, _ = sess.run([loss, train_op, g_step, acc_5, acc, extra_updates_ops],
                                                                        feed_dict={handle: train_iterator_handle})
                losses.append(step_loss)
                accs.append(step_acc)
                accs_5.append(step_acc_5)
                train_count += 1
                # print('train, step_acc_5', step_acc_5, 'step_acc', step_acc)
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
                    val_losses, val_accs, val_accs_5 = [], [], []        # 在validation上的计算也是按batch进行计算的
                    ckpt = saver.save(sess, ckpt_path + '/model', step)  # ckpt文件,记录当前权重等
                    
                    for j in range(VAL_STEP):                       # 每进行一次validation需要多少个step
                      # use trained weights to calculate outputs on validation dataset
                      # 不需要给模型的参数加上reuse = True(变量共享)
                      step_val_loss, step_val_acc, step_val_acc_5 = sess.run([loss, acc, acc_5],
                                                                    feed_dict={handle: val_iterator_handle})
                      val_losses.append(step_val_loss)
                      val_accs.append(step_val_acc)
                      val_accs_5.append(step_val_acc_5)
                    
                    # 计算当前 eval_interval的validation
                    mean_loss, mean_acc, mean_acc_5 = np.mean(val_losses), np.mean(val_accs), np.mean(val_accs_5)
                    print('Step: {}, Validation. Loss: {:.3f}, Accuracy: {:.4f}, Accuracy_5: {:.4f}'.format(step, mean_loss, mean_acc, mean_acc_5))
                    print('============================================')

                    val_acc.load(mean_acc)          # 本次evaluation计算的均值，load new value to the variable, 不会给计算图添加新的节点
                    val_acc_5.load(mean_acc_5)
                    val_loss.load(mean_loss)
                    # controller.update_lr(sess, step) # 更新学习率
                    
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
                   print('Iteration:{}, Training Loss:{:.3f}, Accuracy:{:.4f}, Accuracy_5:{:.4f}'.
                          format(step, mean_loss,mean_acc,mean_acc_5)) # 当前summary的数据
                   losses,accs,accs_5 = [],[],[] # 每个summary interval之后重置
                   
                   # Note: 结果是155500步骤,这里summary也用到了iterator_handle中的元素
                   # writer.add_summary(sess.run(summ, feed_dict={handle: train_iterator_handle}), step)
                   writer.add_summary(sess.run(summ, feed_dict={handle: train_iterator_handle}), step)
                   summary_count += 1

          except tf.errors.OutOfRangeError:
            break

    print('Total time: {0} seconds.'.format(time.time()-start_time))
    print('summary_count',summary_count)
    print('train_count',train_count)
    
if __name__ == "__main__":
  train()
