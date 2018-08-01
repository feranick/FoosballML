#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* FoosballML
* version: 20180801a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
#print(__doc__)

#***************************************************
''' This is needed for installation through pip '''
#***************************************************
def DataSubmitter():
    main()
#***************************************************

import configparser, logging, sys, math, json, os.path, time, base64
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, ActivityRegularization, MaxPooling1D
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras.optimizers as opt
from keras import regularizers
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


#************************************
''' Main '''
#************************************
def main():
    R = np.array([list(sys.argv[1])], dtype = int)
    print(R.shape)
    print(R)
    model = load_model("model")
    predictions = model.predict(R, verbose=0)
    ynew = model.predict_classes(R)
    print(ynew)

    pred_class = np.argmax(predictions)
    predProb = round(100*predictions[0][pred_class],2)
    rosterPred = np.where(predictions[0]>0.01)[0]

    print(pred_class)
    print(predProb)
    print(rosterPred)
    

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
