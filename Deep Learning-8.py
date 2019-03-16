# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 21:00:26 2019

@author: MMOHTASHIM
"""




"""
for those having problems understanding the format of the input :

say , u have a 5*5 image and u have 1 such image then it is :

x = np.ones((1,5,5))

so u have , 

x  =  array([[[ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.],
                     [ 1.,  1.,  1.,  1.,  1.]]])

now for the rnn u need to convert each row of pixel into a single chunk.
so , u would have 5 chunks of 5 values each
so, u need to convert each row to an array

x = np.transpose(x,(1,0,2))

this swaps the 0th dim with the 1st dim . so, u get shape of x as (5,1,5)
which is 5 arrays of 1 chunk each of 5 elements 

x = array([[[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]],

                  [[ 1.,  1.,  1.,  1.,  1.]]])

now , u need to remove 1 pair of extra braces . so flatten by one dimension

x = np.reshape(x,(-1,chunk_size))

so, u will have :

x = array([[ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.],
                  [ 1.,  1.,  1.,  1.,  1.]])

and finally u will need to split the entire thing into 5 chunks(5 arrays)
x = np.split(x,n_chunks,0)

so, finally u have :

x = [array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]]), array([[ 1.,  1.,  1.,  1.,  1.]])]
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
tf.reset_default_graph()
hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)