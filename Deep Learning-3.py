# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:00:26 2019

@author: MMOHTASHIM
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/tmp/data",one_hot=True)###one-hot-one is on and rest is off,
###10 classes 0-9
'''
without one_hot
output_node=result
0=0
1=1
2=2

with one_hot(output defined in following terms)
0=[1,0,0,0,0,0,0,0,0,0]
1=[0,1,0,0,0,0,0,0,0,0]
2=[0,0,1,0,0,0,0,0,0,0]
'''
'''
basic design of neural network
input > weight>hidden layer1(activation function)>wieghts>hiddenl2(activation function)>wieght>output layer


this is  feed-foward neural network-passing the data straight through

compare output to intended output>cost or loss function(cross entropy)
optimization function/optimzer> minimize that cost(AdamOptimizer,SG,AdaGrad)
This optimization function in order to minimize the cost ,change the wieghts which is called:
backpropgation


feed Foward+backprop=epoch(one cycle)-lowering the cost function

'''
#hidden layer nodes
n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500


n_classes=10#0-9 digits
batch_size=100###going to go through batches of 100 of features and feed them to our network and mainipulate the wieghts

###image-heightxwidth
x=tf.placeholder("float",[None,784])###input data,flattened image pixels,shape of it(optional input to placeholder)
y=tf.placeholder("float")#####output/label of the data
### x and y are our raw input and raw output
def neural_network_model(data):
    
    #actual model
    #input_data*weights+biases,in order to ensure neuron activates if all input_Data is zero
    hidden_1_layer={"weights":tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                   "biases":tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                   "biases":tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                   "biases":tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer={"weights":tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                   "biases":tf.Variable(tf.random_normal([n_classes]))}   
    ##tf.nn.relu is the activation/threshold function
    l1=tf.add(tf.matmul(data, hidden_1_layer["weights"]),hidden_1_layer["biases"])
    l1=tf.nn.relu(l1)
    
    l2=tf.add(tf.matmul(l1, hidden_2_layer["weights"]),hidden_2_layer["biases"])
    l2=tf.nn.relu(l2)
    
    l3=tf.add(tf.matmul(l2, hidden_3_layer["weights"]),hidden_3_layer["biases"])
    l3=tf.nn.relu(l3)
    
    output=tf.add(tf.matmul(l3, output_layer["weights"]),output_layer["biases"])
    
    return output

###taking the input and passing to our model
def train_neural_network(x):
    prediction=neural_network_model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost) ##has an parameter of learning rate
    
    hm_epochs=10###one epoch-when entire dataset is passed through network(foward+back)
                ####batch-size-how many points to be trained in one iteration
                ####iteration==cycles to comepelete one epoch 
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        ##Training the data
        for epoch in range(hm_epochs):#(epoch)
            epoch_loss=0
            for  _ in range(int(mnist.train.num_examples/batch_size)):#iteration)
                epoch_x,epoch_y=mnist.train.next_batch(batch_size) ##taking batch data and labels
                _,c=sess.run([optimizer,cost],feed_dict={x:epoch_x,y:epoch_y})###running the network
                epoch_loss+=c###calculating loss
            print("Epoch",epoch,"completed out of",hm_epochs,"loss",epoch_loss)
        
        
        ###To calculate how correct are we when we test our data
        correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))##measuring accuracy
        accuracy=tf.reduce_mean(tf.cast(correct,"float"))##True=1,False=0
        print("Accuracy: ",accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
        
train_neural_network(x)
                
                
                
                
    
    
    
    
