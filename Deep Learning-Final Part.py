# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:59:52 2019

@author: MMOHTASHIM with help of sentdex(youtube channel)
"""

import dicom
import os
import pandas as pd
import cv2
import numpy as np
import math

data_dir='../input/sample_images/'
patients=os.listdir(data_dir)

labels_df=pd.read_csv('../input/stage1_labels.csv',index_col=0)


    label=labels_df.get_value(patient,'cancer')
    path=data_dir+patient
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices),slices[0].pixel_array.shape)
import matplotlib.pyplot as plt
for patient in patients[:1]:
    label=labels_df.get_value(patient,'cancer')
    path=data_dir+patient
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    
    plt.imshow(slices[0].pixel_array)
    plt.show()




def process_data(patients,labels_df,IMG_PX_SIZE=50,HM_SLICES=20,visualize=False):
    label=labels_df.get_value(patient,'cancer')
    path=data_dir+patient
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    
    new_slices=[]
    slices=[cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE) for each_slice in slices)]
    chunk_size=math.ceil(len(slices)/HM_SLICES)
    for slice_chunk in chunks(slices, chunk_sizes):
        slice_chunk = list(map(mean, zip(*slice_chunk)))
        new_slices.append(slice_chunk)

    if len(new_slices) == HM_SLICES-1:
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES-2:
        new_slices.append(new_slices[-1])
        new_slices.append(new_slices[-1])

    if len(new_slices) == HM_SLICES+2:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val
        
    if len(new_slices) == HM_SLICES+1:
        new_val = list(map(mean, zip(*[new_slices[HM_SLICES-1],new_slices[HM_SLICES],])))
        del new_slices[HM_SLICES]
        new_slices[HM_SLICES-1] = new_val
    if visualize:
        fig = plt.figure()
        for num,each_slice in enumerate(new_slices):
            y = fig.add_subplot(4,5,num+1)
            y.imshow(each_slice, cmap='gray')
        plt.show()    
    if label==1:label=np.array([0,1])
    elif label==0:label=np.array([1,0])
    
    return np.array(new_slices),label



def chunks(l,n):
    for i in range(0,len(l),n):
        yield l[i:i+n]
def mean(l):
    return sum(l)/len(l)


much_data=[]
for num,patient in enumerate(patients):
    if num%100==0:
        print(num)
    try:
        img_data,label=process_data(patients,labels_df,IMG_PX_SIZE=50,HM_SLICES=20,visualize=False)
        much_data.append([img_data,label])
    except KeyError as e:
        print("This is unlabelled data")

np.save("much--{}--{}--{}.npy".format(IMG_SIZE,IMG_SIZE,SLICE_COUNT),much_data)

import tensorflow as tf
import numpy as np

IMG_SIZE_PIX=50
SLICE_COUNT=20
n_classes=2

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
    #                        size of window         movement of window
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')



def convolutional_neural_network(x):

    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               'W_fc':tf.Variable(tf.random_normal([54080,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, IMG_SIZE_PIX, IMG_SIZE_PIX,SLICE_COUNT,1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)
    
    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output

def train_neural_network(x):
    much_data=np.load("much_data")
	train_data=much_data[:100]
	
	prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            success_total=0
            attempt_total=0
            for data in train_data:
                attempt_total+=1 
                try:
    				X=data[0]
    				Y=data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                    success_total+=1
                except Exception as e:
                    pass
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss,"success_rate", success_total/attempt_total)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[[i[1] for i in validation_data]]}))

train_neural_network(x)

