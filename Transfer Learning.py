# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:33:47 2019

@author: MMOHTASHIM
"""
from tensorflow import keras
##transfer learning
model_A = keras.models.load_model("my_model_A.h5")

model_A_clone = keras.models.clone_model(model_A)

model_A_clone.set_weights(model_A.get_weights())


model_B_on_A = keras.models.Sequential(model_A.layers[:-1])

model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
##freezing the wieghts for reused layers
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy", optimizer="sgd",
metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
validation_data=(X_valid_B, y_valid_B))
##unfrezzing the wieghts for reused layer afer some time
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True
optimizer = keras.optimizers.SGD(lr=1e-4) # the default lr is 1e-3
model_B_on_A.compile(loss="binary_crossentropy", optimizer=optimizer,
metrics=["accuracy"])

history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
validation_data=(X_valid_B, y_valid_B))