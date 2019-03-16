# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 00:12:16 2019

@author: MMOHTASHIM
"""
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import gym
import pandas as pd
import numpy as np
import random
from statistics import mean
##code in alpha stage-involves a lot of mistakes
frames_per_game=500
min_score=-0.6

training_data=[]
total_games=10000


env = gym.make('MountainCar-v0')
def generating_data():
    prev_observation=[]
    for i_episode in range(total_games):
        env.reset()
        game_memory=[]
        score=[]
        for t in range(frames_per_game):
            action =random.choice([0,1,2])
            move=[]
            if len(prev_observation)==0:
                observation, reward, done, info = env.step(action)
                prev_observation=observation
            else:   
                move.append(prev_observation)
                move.append(action)
                game_memory.append(move)
#                print(game_memory)
                observation, reward, done, info = env.step(action)
                prev_observation=observation
                score.append(observation[0])
            if done:
                break
#        print(mean(score))
        if mean(score) > min_score:
            for memory in game_memory:
               training_data.append(memory)
    training_data_array=np.array(training_data)
    np.save("training_data_array.npy",training_data_array)
#    print(training_data_array)
    return training_data 


   
training_data=np.load("training_data_array_hotlabel.npy")
def hot_label_training_data():
    training_data=np.load("training_data_array.npy")
    for i in training_data:
        if i[1]==0:
            i[1]=[0,0]
        elif i[1]==1:
            i[1]=[1,0]
        else:
            i[1]=[0,1]
    np.save("training_data_array_hotlabel.npy",training_data)
    return training_data
##generating_data()
#training_data=hot_label_training_data()

X=np.array([i[0] for i in training_data[:-500]])
Y=np.array([i[1] for i in training_data[:-500]])
test_x=np.array([i[0] for i in training_data[-500:]])
test_y=np.array([i[1] for i in training_data[-500:]])
#print(X)
#print(Y)
tf.reset_default_graph()
AI=input_data(shape=[None,2],dtype='float32',name="inputs")
    
    
AI=fully_connected(AI,128,activation='relu')
AI=dropout(AI,keep_prob=0.8)
    
AI=fully_connected(AI,512,activation='relu')
AI=dropout(AI,keep_prob=0.8)
    
AI=fully_connected(AI,512,activation='relu')
AI=dropout(AI,keep_prob=0.8)

AI=fully_connected(AI,256,activation='relu')
AI=dropout(AI,keep_prob=0.8)
    
AI=fully_connected(AI,2,activation='softmax')##output
    
AI=regression(AI,name="targets")
    
model=tflearn.DNN(AI,tensorboard_dir="log")

#model.fit({'inputs': X}, {'targets': Y}, n_epoch=5, validation_set=({'inputs': test_x}, {'targets': test_y}), 
#          snapshot_step=500, show_metric=True, run_id="CAR AI")

#
#model.save("CAR AI.model")
model.load("CAR AI.model")
scores=[]
for each_game in range(100):
    negative_score=0
    game_memory=[]
    prev_obs=[]
    env.reset()
    for _ in range(500):
        env.render()
        if len(prev_obs)==0:
            action =random.choice([0,1,2])
        else:
            action=np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])
#            print(np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs)))))
        
        new_observation,reward,done,info=env.step(action)
        prev_obs=new_observation
        game_memory.append([new_observation,action])
        negative_score+=reward
    scores.append(negative_score)
print("The game performed with the follwing index:",mean(scores))

    
    
    