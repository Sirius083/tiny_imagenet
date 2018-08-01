# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:29:52 2018

@author: Sirius
"""

'''
build test set with lables: 50 out of 500 in training(does not using bounding boxes)
since test set on host cannot be connected
'''

import os
import glob
import shutil
DIR = 'E:/tiny_imagenet/tiny-imagenet-200'

dirnames = glob.glob(os.path.join(DIR, 'train/n*'))
train_new = os.path.join(DIR ,'train_new')
test_new = os.path.join(DIR,'test_new')

if not os.path.isdir(train_new):
   os.makedirs(train_new)
if not os.path.isdir(test_new):
   os.makedirs(test_new)


test_label = {} # save as python data file
count = 0       # test image count

for d in dirnames:
    clsname = d.split(os.sep)[-1]
    allfiles = os.listdir(d+'\\images')
   
    # copy train file
    train = allfiles[:450]
    train = [os.path.join(d,'images', t) for t in train]
    train_path = os.path.join(train_new, clsname)
    os.makedirs(train_path)
    for f in train:
        shutil.copy(f,train_path)
    
    # copy test file
    test = allfiles[450:]
    test = [os.path.join(d,'images', t) for t in test]
    # rename test dataset： test_0.JPEG n04067472: os.rename(src, dst)
    for i in range(len(test)):
        shutil.copy(test[i], test_new)
        
        file_new_path = os.path.join(test_new,test[i].split(os.path.sep)[-1])
        file_new_rename = os.path.join(test_new,'test_' + str(count) + '.JPEG')
        
        os.rename(file_new_path , file_new_rename)
        count += 1
        test_label['test_' + str(count)] = clsname
        
