# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 01:17:22 2019

@author: MMOHTASHIM
"""
import tflearn
from tflearn.layers.core import input_data,fully_connected
from tflearn.layers.estimator import regression
from Deep_Learning_4_5 import create_feature_sets_and_labels





train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
n_classes=2
import tensorflow as tf

tf.reset_default_graph()
graph=input_data(shape=[None,len(train_x[0])],name="input")

graph=fully_connected(graph,128,activation="relu")
graph=fully_connected(graph,128,activation="relu")
graph=fully_connected(graph,128,activation="relu")

graph=fully_connected(graph,n_classes,activation="softmax")

graph=regression(graph,name="targets")

model=tflearn.DNN(graph)
model.fit({"input":train_x},{"targets":train_y},n_epoch=10,validation_set=({"input": test_x}, {"targets": test_y}))
model.save("test.model")