# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:41:24 2019

@author: MMOHTASHIM,data from sentiment 140
"""

    
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative. 2 = neutral. 4 = positive.
id
date
query
user
tweet
'''

def init_process(fin,fout):
	outfile = open(fout,'a')
	with open(fin, buffering=200000, encoding='latin-1') as f:
		try:
			for line in f:
				line = line.replace('"','')
				initial_polarity = line.split(',')[0]
				if initial_polarity == '0':
					initial_polarity = [1,0]
				elif initial_polarity == '4':
					initial_polarity = [0,1]

				tweet = line.split(',')[-1]
				outline = str(initial_polarity)+':::'+tweet
				outfile.write(outline)
		except Exception as e:
			print(str(e))
	outfile.close()

#init_process('training.1600000.processed.noemoticon.csv','train_set.csv')
#init_process('testdata.manual.2009.06.14.csv','test_set.csv')


def create_lexicon(fin):
	lexicon = []
	with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter += 1
				if (counter/2500.0).is_integer():
					tweet = line.split(':::')[1]
					content += ' '+tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
					print(counter, len(lexicon))

		except Exception as e:
			print(str(e))

	with open('lexicon-2500-2638.pickle','wb') as f:
		pickle.dump(lexicon,f)

#create_lexicon('train_set.csv')


def convert_to_vec(fin,fout,lexicon_pickle):
	with open(lexicon_pickle,'rb') as f:
		lexicon = pickle.load(f)
	outfile = open(fout,'a')
	with open(fin, buffering=20000, encoding='latin-1') as f:
		counter = 0
		for line in f:
			counter +=1
			label = line.split(':::')[0]
			tweet = line.split(':::')[1]
			current_words = word_tokenize(tweet.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]

			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					# OR DO +=1, test both
					features[index_value] += 1

			features = list(features)
			outline = str(features)+'::'+str(label)+'\n'
			outfile.write(outline)

		print(counter)

#convert_to_vec('test_set.csv','processed-test-set.csv','lexicon-2500-2638.pickle')


def shuffle_data(fin):
	df = pd.read_csv(fin, error_bad_lines=False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('train_set_shuffled.csv', index=False)
	
#shuffle_data('train_set.csv')


def create_test_data_pickle(fin):

	feature_sets = []
	labels = []
	counter = 0
	with open(fin, buffering=20000) as f:
		for line in f:
			try:
				features = list(eval(line.split('::')[0]))
				label = list(eval(line.split('::')[1]))

				feature_sets.append(features)
				labels.append(label)
				counter += 1
			except:
				pass
	print(counter)
	feature_sets = np.array(feature_sets)
	labels = np.array(labels)

#create_test_data_pickle('processed-test-set.csv')

##########################Our  neural network design and implementation:
import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2

batch_size = 32
total_batches = int(1600000/batch_size)
hm_epochs = 10



x = tf.placeholder('float')
y = tf.placeholder('float')

hidden_1_layer = {'f_fum':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([32, n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)
    output = tf.matmul(l2,output_layer['weight']) + output_layer['bias']
    return output

saver = tf.train.Saver()
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        try:
            epoch = int(open(tf_log,'r').read().split('\n')[-2])+1
            print('STARTING:',epoch)
        except:
            epoch = 1

        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess,r"C:\Users\MMOHTASHIM\Anaconda3\libs\Deep-Learning Algorithms\model.ckpt")
            epoch_loss = 1
            with open('lexicon-2500-2638.pickle','rb') as f:
                lexicon = pickle.load(f)
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                batch_x = []
                batch_y = []
                batches_run = 0
                for line in f:
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]

                    features = np.zeros(len(lexicon))

                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            # OR DO +=1, test both
                            features[index_value] += 1
                    line_x = list(features)
                    line_y = eval(label)
                    batch_x.append(line_x)
                    batch_y.append(line_y)
                    if len(batch_x) >= batch_size:
                        _, c = sess.run([optimizer, cost], feed_dict={x: np.array(batch_x),
                                                                  y: np.array(batch_y)})
                        epoch_loss += c
                        batch_x = []
                        batch_y = []
                        batches_run +=1
                        print('Batch run:',batches_run,'/',total_batches,'| Epoch:',epoch,'| Batch Loss:',c,)

            saver.save(sess, r"C:\Users\MMOHTASHIM\Anaconda3\libs\Deep-Learning Algorithms\model.ckpt")
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
            with open(tf_log,'a') as f:
                f.write(str(epoch)+'\n') 
            epoch +=1

#train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,r"C:\Users\MMOHTASHIM\Anaconda3\libs\Deep-Learning Algorithms\model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
            
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested',counter,'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))


test_neural_network()


