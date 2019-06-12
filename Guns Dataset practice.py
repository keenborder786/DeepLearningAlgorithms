# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 23:49:30 2019

@author: MMOHTASHIM
"""

import tensorflow as tf
import os
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import time

df=pd.read_csv("guns.csv")
print(df.head())
def processing_pipeline(df):
    df.dropna(inplace=True)
    year= pd.get_dummies(df['year'])
    sex= pd.get_dummies(df['sex'])
    race= pd.get_dummies(df['race'])
    place= pd.get_dummies(df['place'])
    month= pd.get_dummies(df['month'])
    
    for i in ["year","sex","race","place","month"]:
        df = df.drop(i,axis = 1)
    df=df.join(year)
    df=df.join(month)
    df=df.join(race)
    df=df.join(place)
    df=df.join(sex)
    df.to_csv("processed.csv")
#processing_pipeline(df)
    
    
df=pd.read_csv("processed.csv")
intent=list(set(df["intent"]))

def generate_labels():
    labels=[]
    for l in df["intent"].tolist():
            if l=='Undetermined':
                labels.append([1,0,0,0])
            elif l=='Homicide':
                labels.append([0,1,0,0])
            elif l=='Suicide':
                labels.append([0,0,1,0])
            elif l=='Accidental':
                labels.append([0,0,0,1])
    return labels
root_logdir = os.path.join(os.curdir, "my_logs")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)
run_logdir = get_run_logdir()
y=np.array(generate_labels())
X=np.array(df.drop("intent",1))

X_train_full, X_test, y_train_full, y_test = train_test_split(
X, y,test_size=0.1)
X_train, X_valid, y_train, y_valid = train_test_split(
X_train_full, y_train_full,test_size=0.1)



model=keras.models.Sequential()
model.add(keras.layers.Dense(128, activation='relu', input_shape=(37,)))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(500, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(4, activation="softmax"))

model.compile(loss="categorical_crossentropy",
optimizer="adam",
metrics=["accuracy"])
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
checkpoint_cb = keras.callbacks.ModelCheckpoint("guns.h5")


history = model.fit(X_train, y_train, epochs=35,
validation_data=(X_valid, y_valid),
callbacks=[tensorboard_cb,checkpoint_cb])





            
    

