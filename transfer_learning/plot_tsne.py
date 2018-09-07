# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 10:49:00 2018

@author: Sirius

analysis transfer values using TSNE
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# 准备数据
train_transfer_path = 'E:\\transfer_tiny_imagenet\\train_transfer_all.pickle'
train_label_path = 'E:\\transfer_tiny_imagenet\\train_labels_all.pickle'
num_classes = 200

with open(train_transfer_path, 'rb') as handle:
	 transfers = pickle.load(handle)

with open(train_label_path,'rb') as handle:
	 labels = pickle.load(handle)

transfer_values = transfers[0:3000]
cls_ = labels[0:3000] # 6 个class # [160, 5, 177, 55, 22, 183]


def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()
'''
# PCA version
pca = PCA(n_components=2)
transfer_values_reduced = pca.fit_transform(transfer_values)
plot_scatter(transfer_values_reduced, cls_)
transfer_values_reduced.shape
'''

# TSNE is very slow, first use PCA to reduce dimension
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)

tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d) 
plot_scatter(transfer_values_reduced, cls_)


