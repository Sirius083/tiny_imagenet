# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:44:17 2018

@author: dell4mc
"""
'''
# Note: 可以从tensorboard中直接下载csv文件
# 从checkpoint的ckpt文件中读取变量的值
from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join(model_dir, "model.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key)) # Remove this is you want to print only variable names


# 从events文件中回复变量的值
import tensorflow as tf
train_accuracy_1 = []
event_dir = 'E:\\resnet\\models-master\\official\\resnet\\imagenet_train_18\\events.out.tfevents.1531975444.DESKTOP-L32SK0R'
for event in tf.train.summary_iterator(event_dir):
    for value in event.summary.value:
        if value.tag == 'train_accuracy_1':
            train_accuracy_1.append(value.simple_value)
'''        

'''
# inception_v3结果
# 从events文件中回复变量的值
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf


event_vanilla = r'E:\inception_v3_change\logs\logs_bn_l2\events.out.tfevents.1535095941.DESKTOP-L32SK0R'
event_bn_l2 =   r'E:\inception_v3_change\logs\logs_vanilla\events.out.tfevents.1534992498.DESKTOP-L32SK0R'
event_lr_decay =  r'E:\inception_v3_change\logs\logs_lr_decay\events.out.tfevents.1535336173.DESKTOP-L32SK0R'
event_side = r'E:\inception_v3_change\logs\logs_side\events.out.tfevents.1535367217.DESKTOP-L32SK0R'

train_loss_vanilla = []
val_acc_vanilla = []
for event in tf.train.summary_iterator(event_vanilla):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_vanilla.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_vanilla.append(value.simple_value)
            
train_loss_bn_l2 = []
val_acc_bn_l2 = []
for event in tf.train.summary_iterator(event_bn_l2):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_bn_l2.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_bn_l2.append(value.simple_value)
            
                      
train_loss_lr_decay = []
val_acc_lr_decay = []
for event in tf.train.summary_iterator(event_lr_decay):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_lr_decay.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_lr_decay.append(value.simple_value)
            
                    
train_loss_side = []
val_acc_side = []
for event in tf.train.summary_iterator(event_side):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_side.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_side.append(value.simple_value)

# 画图 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
c1 = '#%02x%02x%02x' % (48,186,238)
c2 = '#%02x%02x%02x' % (253,114,71)
c3 = '#%02x%02x%02x' % (41,141,198)
c4 = '#%02x%02x%02x' % (202,46,11)

sind = 100
eind = 100000
skip = 100
x = range(sind,eind,skip)

fig = plt.figure(figsize=(18,10))

plt.subplot(1, 2, 1)
plt.plot(x, train_loss_vanilla[sind:eind:skip],c = c1, label='vanilla_loss')
plt.plot(x, train_loss_bn_l2[sind:eind:skip],c = c2,label='bn_l2_loss')
plt.plot(x, train_loss_lr_decay[sind:eind:skip],c = c3,label='l2_decay_loss')
plt.plot(x, train_loss_side[sind:eind:skip],c = c4,label='side_loss')
plt.title('Inception-V3 train loss')
plt.legend(prop={'size': 11})
plt.xlabel('step')
plt.ylabel('train_loss')


plt.subplot(1, 2, 2)
plt.plot(x, val_acc_vanilla[sind:eind:skip],c = c1, label='vanilla_val_acc')
plt.plot(x, val_acc_bn_l2[sind:eind:skip],c = c2,label='bn_l2_val_acc')
plt.plot(x, val_acc_lr_decay[sind:eind:skip],c = c3,label='l2_decay_val_acc')
plt.plot(x, val_acc_side[sind:eind:skip],c = c4,label='side_val_acc')
plt.title('Inception-V3 validation top 1 accuracy')
plt.legend(prop={'size': 11})
plt.xlabel('step')
plt.ylabel('validation accuracy')
plt.savefig('inception_v3')
plt.show()
'''


# resnet 结果
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   
import tensorflow as tf


event_resnet_full =  r'E:\resnet_change\logs\log_resnet_full\events.out.tfevents.1535507362.DESKTOP-L32SK0R'
event_resnet_vanilla = r'E:\resnet_change\logs\log_resnet_vanilla\events.out.tfevents.1535553475.DESKTOP-L32SK0R'
event_resnet_bottleneck =   r'E:\resnet_change\logs\log_resnet_bottleneck\events.out.tfevents.1535598479.DESKTOP-L32SK0R'
event_resnet_bn_loc = r'E:\resnet_change\logs\log_resnet_bn_loc\events.out.tfevents.1535583416.DESKTOP-L32SK0R'

train_loss_vanilla = []
val_acc_vanilla = []
for event in tf.train.summary_iterator(event_resnet_vanilla):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_vanilla.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_vanilla.append(value.simple_value)
            
train_loss_full = []
val_acc_full = []
for event in tf.train.summary_iterator(event_resnet_full):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_full.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_full.append(value.simple_value)
            
                      
train_loss_bottleneck = []
val_acc_bottleneck = []
for event in tf.train.summary_iterator(event_resnet_bottleneck):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_bottleneck.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_bottleneck.append(value.simple_value)
            
                    
train_loss_bn_loc = []
val_acc_bn_loc = []
for event in tf.train.summary_iterator(event_resnet_bn_loc):
    for value in event.summary.value:
        if value.tag == 'train_loss':
            train_loss_bn_loc.append(value.simple_value)
        if value.tag == 'val_acc':
            val_acc_bn_loc.append(value.simple_value)

# 画图 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
c1 = '#%02x%02x%02x' % (48,186,238)
c2 = '#%02x%02x%02x' % (253,114,71)
c3 = '#%02x%02x%02x' % (41,141,198)
c4 = '#%02x%02x%02x' % (202,46,11)

sind = 100
eind = 100000
skip = 100
x = range(sind,eind,skip)

fig = plt.figure(figsize=(18,10))

plt.subplot(1, 2, 1)
plt.plot(x, train_loss_vanilla[sind:eind:skip],c = c1, label='vanilla_loss')
plt.plot(x, train_loss_bottleneck[sind:eind:skip],c = c2,label='bottleneck_loss')
plt.plot(x, train_loss_full[sind:eind:skip],c = c3,label='full_loss')
plt.plot(x, train_loss_bn_loc[sind:eind:skip],c = c4,label='bn_loc_loss')
plt.title('Resnet train loss')
plt.legend(prop={'size': 11})
plt.xlabel('step')
plt.ylabel('train_loss')


plt.subplot(1, 2, 2)
plt.plot(x, val_acc_vanilla[sind:eind:skip],c = c1, label='vanilla_val_acc')
plt.plot(x, val_acc_bottleneck[sind:eind:skip],c = c2,label='bottleneck_val_acc')
plt.plot(x, val_acc_full[sind:eind:skip],c = c3,label='full_val_acc')
plt.plot(x, val_acc_bn_loc[sind:eind:skip],c = c4,label='bn_loc_val_acc')
plt.title('Resnet validation top 1 accuracy')
plt.legend(prop={'size': 11})
plt.xlabel('step')
plt.ylabel('validation accuracy')
plt.savefig('Resnet')
plt.show()
