# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 12:26:11 2018

@author: Arifhusen
Description:    Sentiment Analysis based on LSTM (Type of Recurrent Neural Network)
                We will use LSTM RNN to predict sentient of movie review.
                Resoan behind using LSTM is. LSTM not only use Vectorization method for words but also consider the sequence. 
                Sequence of work in statement make big difference in statement. 
                
Framwork: keras
 

Data Souce: Movie review data from imdb. Data set already available in the keras package.

"""

from keras import Sequential
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding,LSTM,Dropout,Dense
import numpy as np


def load_data(vocabulary_size) :
    
    #-- We have consider vocabulary size of 5000.
    #-- Vocabulary of size 5000, and for each word unique id is assigned. 
    #-- Seq: 0 to 5000
    #-- Statement is separated into word and id it allocated to word in statement.
    
    (x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=vocabulary_size)
        
    return x_train,y_train,x_test,y_test

def preview_word_in_data (data):
    wordindex = imdb.get_word_index()
    #-- In word index "word" is used as key and "id" as value.
    #-- Build new dictionary with "id" as key and "word" as value.
    wordindex = {i:word for word,i in wordindex.items()}
    wordlist = [wordindex[a] for a in data]
    print(wordlist)
    
def padding_words(data,max_len):
    #-- Since we need to feed this data to RNN. All the input must be of same size.
    #-- We will decide max_word per review.
    #-- Review having more than word more than max_word, will be removed and review with less words will have null(0) for as padding.
    data = sequence.pad_sequences(data,maxlen=max_len)
    return data 

def design_rnn_model (data,embedding_size,vocabulary_size,maxlen):
    
    
    model = Sequential()
    model.add(Embedding(vocabulary_size,embedding_size,input_length=maxlen))
    model.add(LSTM(300)) #-- First Hidden Layer
#    model.add(Dropout(0.2)) #-- Drop out to make sure model will not overfit data.
#    model.add(LSTM(100)) #-- Second Hidden Layer
#    model.add(Dropout(0.2))
    model.add(Dense(units=1,activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    print(model.summary())
    return model

def build_validation_train_dataset(x_train,y_train,batch_size):
    
    x_val = x_train[:batch_size]
    y_val = y_train [:batch_size]
    x_train2 = x_train [batch_size:]
    y_train2 = y_train [batch_size:]
    
    return x_val,y_val,x_train2,y_train2
        

if __name__ ==  "__main__":
    
#    is_train = input('Do you want to train on new data? Press [y|Y] or [n|N]\n')
    is_train='y'
    vocabulary_size= 5000
    maxlen= 500
    embedding_size = 32
    if is_train== 'Y' or is_train == 'y':
        x_train,y_train,x_test,y_test = load_data (vocabulary_size)
        print ('\nLoaded dataset with {} training sample and {} test samples.'.format(len(x_train),len(x_test)))
        #-- We can see that Data is represent as sequence of numbers. Used to LSTM sequential learning
        #-- And also Numbers are representing the position of word vocabulary.
        print ('\nSample Training Input data.')        
        print(x_train[1])
        #-- Review data by actual word.
        preview_word_in_data (x_train[1])
        
        print('\nSample Training Output data.\n0: Negetive\n1: Positive ')        
        print('\nLabel for first input is:',y_train[1])
        
        #-- Find review with maximum length
        print('Maximum review length:',len(max(x_train+x_test,key = len)))
        print('Minimun review length:',len(min(x_train+x_test,key = len)))
        
        #-- Padding words to make input with uniform size
        #-- set max_len = 500
        x_train = padding_words(x_train,maxlen)
        x_test = padding_words(x_test,maxlen)
        model = design_rnn_model(x_train,embedding_size,vocabulary_size,maxlen)    
        
        batch_size = 64 #-- Size of each batch for training.
        epochs = 5      #-- Number of time whole training process will be executed.
        
        #-- Build validation and new train dataset from train data.
        
        x_val,y_val,x_train2,y_train2 = build_validation_train_dataset(x_train,y_train,batch_size)
        
        model.fit(x = x_train2, y = y_train2,batch_size=batch_size,epochs=epochs,validation_data=(x_val,y_val))
        
        #-- Evaluate model on test data.
        score = model.evaluate(x_test,y_test,verbose=0)
        
        print('\nModel accuracy on test data is: {}'.format(score))
        
        
    elif is_train== 'N' or is_train == 'n':
        print('\n')
    


        
        