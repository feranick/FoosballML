#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* FoosballML - predict
* version: 20180801c
*
* run: python3 FoosballML-predict.py 0,1,2,3
*
* 0,1,2,3 are the order of the players for any given game
* Results are provided in a 1x4 array, where the 1 is winner, 0, is looser
* For ex.: [0,1,1,0] for 0,1,2,3 means that player 2,3 win, 1,4 lose
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
#print(__doc__)

import sys, math, os.path, time, keras, pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
import keras.optimizers as opt
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

#************************************
''' Main '''
#************************************
def main():
    R = np.array([np.fromstring(sys.argv[1], dtype='uint8', sep=',')])
    print([R])
    
    model = load_model("keras_MLP_model.hd5")
    predictions = model.predict(R, verbose=0)[0]
    predict_classes = model.predict_classes(R)

    pred_class = np.argmax(predictions)
    predProb = round(100*predictions[pred_class],2)
    rosterPred = np.where(predictions>0.01)[0]

    #print("predict_classes: ", predict_classes)
    #print("pred_class: ", pred_class)
    #print("predProb: ", predProb)
    #print("rosterPred: ", rosterPred)
    
    print("Estimates in match prediction mode")
    mlr = pickle.loads(open("model_mlr", "rb").read())
    print("\n",R[0])
    for i in range(rosterPred.size):
        print("{0:} {1:.2f}%".format(mlr.inverse_transform(rosterPred[i]), predictions[i]*100))


#************************************
''' MultiClassReductor '''
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
        print("totalClass: ",self.totalClass)
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return self.totalClass[int(a)]


#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
