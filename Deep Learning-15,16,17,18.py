# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 20:27:46 2019

@author: MMOHTASHIM
"""

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
TRAIN_DIR=r'C:\Users\MMOHTASHIM\Anaconda3\Deep Learning Data\Cat VS DOG\train'
TEST_DIR=r'C:\Users\MMOHTASHIM\Anaconda3\Deep Learning Data\Cat VS DOG\test'
IMG_SIZE=50
LR=1e-1
MODEL_NAME='dogsvscats-{}-{}.model'.format(LR, '2conv-basic-video')



def label_img(img):
    word_label=img.split('.')[-3]
    if word_label=="cat": return [1,0]
    elif word_label=="dog":return [0,1] 



def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy',training_data)
    return training_data
def process_test_data():
    testing_data=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        path=os.path.join(TEST_DIR,img)
        img_num=img.split('.')[0]
        img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img),img_num])
        
    np.save('test_data.npy',testing_data)
    return testing_data
train_data=np.load('train_data.npy')
test_data=np.load('test_data.npy')


tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 256, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 128, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)

#A=test_data[0][0].reshape(-1,50,50,1)
#a=model.predict(A)
#print(a)
#tensorboard --logdir=foo:C:\Users\MMOHTASHIM\Anaconda3\libs\DeepLearningAlgorithms\log\dogsvscats-0.1-2conv-basic-video.model to acess tensorboard

import matplotlib.pyplot as plt
fig=plt.figure()
for num,data in enumerate(test_data[:12]):
    #cat :[1,0],dog:[0,1]
    img_num=data[1]
    img_data=data[0]
    y=fig.add_subplot(3,4,num+1)
    
    orig=img_data
    data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    
    model_out=model.predict([data])[0]
    if np.argmax(model_out)==1:str_label='Dog'
    else:str_label='Cat'
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

with open('submission-file.csv','w') as f:
    f.write('id,label\n')
with open('submission-file.csv','a') as f:
    for data in tqdm(test_data):
        img_num=data[1]
        img_data=data[0]
        orig=img_data
        data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out=model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
        
 