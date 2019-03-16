# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:36:12 2019

@author: MMOHTASHIM
"""

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
from statistics import mean,median
from collections import Counter
import tensorflow as tf
LR=1e-3
env=gym.make("CartPole-v0")
env.reset()
goal_steps=500###for how many frames was pole balanced
score_requirement=50###Learn only from score of 50 and greater
intial_games=10000
tf.reset_default_graph()


def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action=env.action_space.sample()###generatin random action
            observation,reward,done,info=env.step(action)
            if done:
                break
            
#some_random_games_first()  
def intial_population():
    training_data=[]
    score=[]
    accepted_scores=[]
    for _ in range(intial_games):
        scores=0
        game_memory=[]
        previous_observation=[]
        for _ in range(goal_steps):
            action=random.randrange(0,2)
            observation,reward,done,info=env.step(action)
            if len(previous_observation)>0:
                game_memory.append([previous_observation,action])
            previous_observation=observation
            scores+=reward
            if done:
                break
        if scores>=score_requirement:
            accepted_scores.append(scores)
            for data in game_memory:
                if data[1]==1:
                    output=[0,1]
                elif data[1]==0:
                    output=[1,0]
                training_data.append([data[0],output])
        env.reset()
        score.append(scores)
    training_data_save=np.array(training_data)
    np.save("saved.npy",training_data_save)
    
    
    print("Average accpeted score: ", mean(accepted_scores))
    print("Median accepted score: " , median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data



def neural_network_model(input_size):
    network=input_data(shape=[None,input_size,1],name="input")
    
    network=fully_connected(network,128,activation="relu")
    network=dropout(network,0.8)
    
    network=fully_connected(network,256,activation="relu")
    network=dropout(network,0.8)
    
    network=fully_connected(network,512,activation="relu")
    network=dropout(network,0.8)
    
    network=fully_connected(network,256,activation="relu")
    network=dropout(network,0.8)
    
    network=fully_connected(network,128,activation="relu")
    network=dropout(network,0.8)
    
    
    network=fully_connected(network,2,activation="softmax")
    network=regression(network,optimizer="adam",learning_rate=LR,loss="categorical_crossentropy",name="targets")
    model=tflearn.DNN(network,tensorboard_dir="log")
    
    
    return model
    
''''

Neville Chim
9 months ago (edited)
I know this is a year old but I'll just write it incase anyone else dosen't understand and stumbles upon this question. 

In i[0][0] of training_data(ie observation) it consists of 4 floats (ie. len(training_data[0][0]), you can see for youself if you print it),  these sets of 4 floats corresponds to a action(either 1 or 0). So by saying reshape(-1, len(training_data[0][0]), 1) means make a matrix that fits 1 output which in this case is Y

"-1" means I don't care what size, fit as much as you can,  in this case it means the # of observations(list of 4 floats)

e.g. 
instead of
Without reshape
X = [1,2,3,4,5,6,7,8, 9,10,11,12]
Y = [0,1,0]

do this instead
With reshape
X = [[[1,2,3,4],[5,6,7,8],[9,10,11,12]]]
Y= [0,1,0]

And therefore each index(observation) in X matches with each index(action) in Y

Extra explanation:
in this example the shape of X is  (3, 4, 1) because there is 3 observations, 4 floats per observation and the whole thing predicts 1 output which is Y﻿
 

'''
def train_model(training_data,model=False):
    X=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y=[i[1] for i in training_data]
    if not model:
        model=neural_network_model(input_size=len(X[0]))
        
    model.fit({"input":X},{"targets":y},n_epoch=5,snapshot_step=500,show_metric=True,run_id="openaistuff")
        
    
    return model


training_data=intial_population()
model=train_model(training_data)






#a=len(training_data[0][0])
#print(a)
###Run this if you are stii confused about reshaping
#X=np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
#print(X)
#print(len(X[0]))

#model.save("Game.model")
#
#model.load("Game.model")


#
scores=[]
choices=[]
for each_game in range(100):
    score=0
    game_memory=[]
    prev_obs=[]
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs)==0:
            action=random.randrange(0,2)
        else:
            action=np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            print(np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))))
        choices.append(action)
        
        new_observation,reward,done,info=env.step(action)
        prev_obs=new_observation
        game_memory.append([new_observation,action])
        score+=reward
        if done:
            break
    scores.append(score)
print("Average accpeted score: ", mean(scores))
print("Median accepted score: " , median(scores))
print("Choice 1:{},Choice 0:{}".format(choices.count(1)/len(choices),
      choices.count(0)/len(choices)))


        
            
            

