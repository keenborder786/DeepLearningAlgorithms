# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 22:22:55 2019

@author: MMOHTASHIM
"""
###########What is tensor flow? and the design of neural network
import tensorflow as tf
#some basics
x1=tf.constant(5)
x2=tf.constant(6)

result=tf.multiply(x1,x2)
print(result)

#sess=tf.Session()
#print(sess.run(result))
#sess.close()
#or
with tf.Session() as sess:
#    output=sess.run(result)
    print(sess.run(result)) 
