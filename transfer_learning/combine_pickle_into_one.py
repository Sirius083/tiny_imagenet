# 将 train_labels合并为一个

import numpy as np
import os
import pickle

pickle_path = 'E:\\transfer_tiny_imagenet\\data_new\\transfer\\'
pickle_list =  ['train_data_1.pkl',
 'train_data_2.pkl',
 'train_data_3.pkl',
 'train_data_4.pkl',
 'train_data_5.pkl',
 'train_data_6.pkl',
 'train_data_7.pkl',
 'train_data_8.pkl',
 'train_data_9.pkl']


with open('E:\\transfer_tiny_imagenet\\data_new\\transfer\\train_data_0.pkl', 'rb') as handle:
     alldata = pickle.load(handle)
     
for p_ in pickle_list:
    with open('E:\\transfer_tiny_imagenet\\data_new\\transfer\\' + p_, 'rb') as handle:
         data = pickle.load(handle)
         alldata = np.concatenate((alldata,data),axis = 0)
         
with open('train_transfer_all.pickle', 'wb') as handle:
    pickle.dump(alldata, handle)
    

    
