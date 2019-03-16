# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:16:03 2019

@author: MMOHTASHIM
"""
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer=WordNetLemmatizer()
hm_lines=10000000
nltk.download('punkt')
def  create_lexicon(pos,neg):
    lexicon=[]
    for fi in [pos,neg]:
        with open(fi,"r") as f:
            contents=f.readlines()
            for l in contents[:hm_lines]:
                all_words=word_tokenize(l.lower())
                lexicon+=list(all_words)
    
                
    lexicon=[lemmatizer.lemmatize(i) for i in lexicon]
    w_counts=(Counter(lexicon))##returns a dic like w_counts={"the":93939","and":99999}
    l2=[]
    for w in  w_counts:
        if 1000>w_counts[w]>50:
              l2.append(w)###to take out the common words and very rare words
    return l2


def sample_handling(sample,lexicon,classification):
    featureset=[]
    
    with open(sample,"r") as f:
        contents=f.readlines()
        for l in contents[:hm_lines]:
            current_words=word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features=np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value=lexicon.index(word.lower())
                    features[index_value]+=1
            features=list(features)
            featureset.append([features,classification])
    return featureset

def create_feature_sets_and_labels(pos,neg,test_size=0.1):
    lexicon=create_lexicon(pos,neg)
    features=[]
    features+=sample_handling("pos.txt",lexicon,[1,0])##[1,0] positive
    features+=sample_handling("neg.txt",lexicon,[0,1])##[0,1] negative
    random.shuffle(features)
    
    features=np.array(features)
    testing_size=int(test_size*len(features))
    train_x=list(features[:,0][:-testing_size])
    train_y=list(features[:,1][:-testing_size])
    
    
    
    test_x=list(features[:,0][-testing_size:])
    test_y=list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('pos.txt','neg.txt')
	# if you want to pickle this data:
	with open("sentiment_set.pickle",'wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)



            




