#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* FoosballML - train
* version: 20180801d
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
    ##########################
    # Parameters
    ##########################

    fullSet = True
    numCols = 10
    
    # withScore: uses the actual score as training data
    # if False it uses only win/lose
    withScore = False
    
    l_rate = 0.0001
    l_rdecay = 0.0

    HL1 = 30
    drop1 = 0.5
    l2_1 = 1e-4
    HL2 = 15
    drop2 = 0.5
    l2_2 = 1e-4
    epochs = 10000
    cv_split = 0.02

    #batch_size = A.shape[0]
    batch_size = 8
    tb_directory = "keras_MLP"
    model_directory = "."
    model_name = model_directory+"/keras_MLP_model.hd5"
    #########################

    file = sys.argv[1]

    #************************************
    ''' Read File '''
    #************************************
    df = pd.read_csv(file)
    if fullSet == True:
        numCols = len(df.columns)
    A = np.array(df.values[4:,1:numCols],dtype=np.float64)
    print(A.shape)

    data = []
    labels = []

    for i in range(A.shape[0]):
        N = np.where(~np.isnan([A[i,:]]))[1]
        if len(N)==4:
            data.append(N.tolist())
            Cl = A[i,:][~np.isnan(A[i,:])]
            if withScore == False:
                Cl = np.where(Cl==np.max(Cl),1,0)
            labels.append(Cl.tolist())

    print(data)
    print(labels)
    data = np.array(data)
    labels = np.array(labels)
    print("Labels shape:", np.array(labels).shape)
    
    classes = np.unique(labels, axis=0)
    numLabels = labels.shape[1]
    print("numLabels = ", numLabels)

    mlr = MultiClassReductor()
    mlr.fit(classes)
    Cl1 = mlr.transform(labels)
    print(Cl1)
    print("Number unique labels - classes: ", np.unique(Cl1).size)

    labels = keras.utils.to_categorical(Cl1, num_classes=np.unique(Cl1).size)
    print("labels: ",labels.shape)
    numClasses = np.unique(Cl1).size
        
    print("[INFO] label binarizer...")
    with open("model_mlr", 'ab') as f:
        f.write(pickle.dumps(mlr))

    ### Build model
    model = Sequential()
    model.add(Dense(HL1, activation = 'relu', input_dim=np.array(data).shape[1],
        kernel_regularizer=regularizers.l2(l2_1),
        name='dense1'))
    model.add(Dropout(drop1,name='drop1'))
    model.add(Dense(HL2, activation = 'relu',
        kernel_regularizer=regularizers.l2(l2_2),
        name='dense2'))
    model.add(Dropout(drop2,
        name='drop2'))
    model.add(Dense(numClasses, activation = 'softmax',
        name='dense3'))

    #optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optim = opt.Adam(lr=l_rate, beta_1=0.9,
                    beta_2=0.999, epsilon=1e-08,
                    decay=l_rdecay,
                    amsgrad=False)

    model.compile(loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'])

    tbLog = TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=batch_size,
            write_graph=True, write_grads=True, write_images=True,
            embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    tbLogs = [tbLog]
    log = model.fit(data, labels,
        epochs=epochs,
        batch_size=batch_size,
        callbacks = tbLogs,
        verbose=2,
        validation_split=cv_split)

    accuracy = np.asarray(log.history['acc'])
    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])
    val_acc = np.asarray(log.history['val_acc'])


    #score = model.evaluate(A_test, Cl2_test, batch_size=A.shape[1])
    model.save(model_name)
    plot_model(model, to_file=model_directory+'/keras_MLP_model.png', show_shapes=True)
    print('\n  =============================================')
    print('  \033[1mKeras MLP\033[0m - Model Configuration')
    print('  =============================================')
    print("\n Training set file:",file)
    print("\n Data size:", A.shape,"\n")
    for conf in model.get_config():
        print(conf,"\n")

    print('\n  ==========================================')
    print('  \033[1mKeras MLP\033[0m - Training Summary')
    print('  ==========================================')
    print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(accuracy), 100*np.amax(accuracy)))
    print("  Loss - Average: {0:.4f}; Min: {1:.4f}".format(np.average(loss), np.amin(loss)))
    print('\n\n  ==========================================')
    print('  \033[1mKerasMLP\033[0m - Validation Summary')
    print('  ==========================================')
    print("\n  Accuracy - Average: {0:.2f}%; Max: {1:.2f}%".format(100*np.average(val_acc), 100*np.amax(val_acc)))
    print("  Loss - Average: {0:.4f}; Min: {1:.4f}\n".format(np.average(val_loss), np.amin(val_loss)))
    #print("\n  Validation - Loss: {0:.2f}; accuracy: {1:.2f}%".format(score[0], 100*score[1]))
    print('  =========================================\n')

#************************************
''' MultiClassReductor '''
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
        print("totalClass: ", self.totalClass)
    
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
