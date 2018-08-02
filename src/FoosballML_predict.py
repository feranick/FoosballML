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
import keras, pickle
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
    useMLB = False

    R = np.array([list(sys.argv[1])], dtype = int)
    print(R)
    model = load_model("keras_MLP_model.hd5")
    predictions = model.predict(R, verbose=0)[0]
    predict_classes = model.predict_classes(R)

    pred_class = np.argmax(predictions)
    predProb = round(100*predictions[pred_class],2)
    rosterPred = np.where(predictions>0.01)[0]

    print("predict_classes: ", predict_classes)
    print("pred_class: ", pred_class)
    print("predProb: ", predProb)
    print("rosterPred: ", rosterPred)
    print("predictions: ", predictions)

    if useMLB == True:
        idxs = np.argsort(predictions)[::-1][:2]
        #idxs = predictions[::-1][:2]
        print(idxs)
        mlb = pickle.loads(open("model_labels", "rb").read())
        print(mlb.classes_)
        for (i, j) in enumerate(idxs):
            # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(mlb.classes_[j], predictions[j] * 100)

        # show the probabilities for each of the individual labels
        for (label, p) in zip(mlb.classes_, predictions):
            print("{}: {:.2f}%".format(label, p * 100))

        print(mlb.inverse_transform([1]))

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
