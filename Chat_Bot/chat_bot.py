# -*- coding: utf-8 -*-
"""
Created on Jul 17 14:47:51 2020

@author: Juan Romero
"""
import numpy as np
import tflearn
import tensorflow
import random
import nltk
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

class Bot:
    def __inti__(self):
        print("Bot created")
        pass
        

    def clean_up_sentence(self, sentence):
        # tokenize the pattern - split words into array
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word - create short form for word
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words
    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, words, show_details=True):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def classify_local(self, sentence):
        ERROR_THRESHOLD = 0.25
        # generate probabilities from the model
        input_data = pd.DataFrame([self.bow(sentence, self.words)], dtype=float, index=['input'])
        #print("This is input data: {}".format(input_data))
        results = self.model.predict([input_data])[0]
        # filter out predictions below a threshold, and provide intent index
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], str(r[1])))
        # return tuple of intent and probability
        return return_list

    def chat(self):
        print("Start talking with the bot (type quit to stop)!")
        while True:
            inp = input("You: ")
            if inp.lower() == "exit":
                break
            results = self.classify_local(inp)
            print(results)
            for tg in self.intents["intents"]:
                if tg['tag'] == results[0][0]:
                    responses = tg['responses']
            print(random.choice(responses))


