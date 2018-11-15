#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* FoosballML
* 20181115c
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import pandas as pd
import sys, os.path, getopt, time, configparser, pickle, h5py, csv
from libFoosballML import *

#***************************************************
# This is needed for installation through pip
#***************************************************
def FoosballML():
    main()

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self):
        confFileName = "FoosballML.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print(" Configuration file: \""+confFileName+"\" does not exist: Creating one.\n")
            self.createConfig()
        self.readConfig(self.configFile)
            
    def datamlDef(self):
        self.conf['Parameters'] = {
            'fullSet' : False,
            'numCols' : 14,
            'withScore': False,
            'l_rate' : 0.001,
            'l_rdecay' : 1e-4,
            'HL' : [30,15],
            'drop' : 0.5,
            'l2' : 1e-4,
            'epochs' : 3000,
            'cv_split' : 0.02,
            'fullSizeBatch' : False,
            'batch_size' : 4,
            'numLabels' : 4,
            }
    def sysDef(self):
        self.conf['System'] = {
            'useTFKeras' : False,
            }

    def readConfig(self,configFile):
        try:
            self.conf.read(configFile)
            self.datamlDef = self.conf['Parameters']
            self.sysDef = self.conf['System']
            
            self.fullSet = self.conf.getboolean('Parameters','fullSet')
            self.numCols = self.conf.getint('Parameters','numCols')
            self.withScore = self.conf.getboolean('Parameters','withScore')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.HL = eval(self.datamlDef['HL'])
            self.drop = self.conf.getfloat('Parameters','drop')
            self.l2 = self.conf.getfloat('Parameters','l2')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.useTFKeras = self.conf.getboolean('System','useTFKeras')
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.datamlDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Main
#************************************
def main():
    dP = Conf()
    start_time = time.clock()
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "tph:", ["train", "predict", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-t" , "--train"):
            try:
                train(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--predict"):
            try:
                predict(sys.argv[2])
            except:
                usage()
                sys.exit(2)

    total_time = time.clock() - start_time
    print("\n Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                            total_time/60, total_time/3600),"\n")

#************************************
# Training
#************************************
def train(learnFile):
    import tensorflow as tf
    dP = Conf()
    
    # Use this to restrict GPU memory allocation in TF
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    conf = tf.ConfigProto(gpu_options=opts)
    #conf.gpu_options.allow_growth = True
    
    if dP.useTFKeras:
        print("Using tf.keras API")
        import tensorflow.keras as keras  #tf.keras
        tf.Session(config=conf)
    else:
        print("Using pure keras API")
        import keras   # pure keras
        from keras.backend.tensorflow_backend import set_session
        set_session(tf.Session(config=conf))

    tb_directory = "keras_MLP"
    model_directory = "."
    learnFileRoot = os.path.splitext(learnFile)[0]
    
    model_name = model_directory+"/keras_MLP_model.hd5"
    model_mcr = model_directory+"/keras_mcr.pkl"

    #from tensorflow.contrib.learn.python.learn import monitors as monitor_lib

    A, data, labels, names = readLearnFile(learnFile)

    classes = np.unique(labels, axis=0)
    numLabels = labels.shape[1]

    print("\n  Number of learning labels: {0:d}".format(int(dP.numLabels)))
    
    #************************************
    # Label Encoding
    #************************************

    mcr = MultiClassReductor()
    mcr.fit(classes)
    mcr.names(names)
    Cl1 = mcr.transform(labels)
    numClasses = np.unique(Cl1).size
    
    print("  Number unique classes (training): ", np.unique(Cl1).size)
    print("\n  Multi Label Reductor saved in:", model_mcr,"\n")
    with open(model_mcr, 'ab') as f:
        f.write(pickle.dumps(mcr))

    labels = keras.utils.to_categorical(Cl1, num_classes=np.unique(Cl1).size)

    #************************************
    # Training
    #************************************

    if dP.fullSizeBatch == True:
        dP.batch_size = A.shape[0]

    #************************************
    ### Define optimizer
    #************************************
    #optim = opt.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    optim = keras.optimizers.Adam(lr=dP.l_rate, beta_1=0.9,
                    beta_2=0.999, epsilon=1e-08,
                    decay=dP.l_rdecay,
                    amsgrad=False)
    #************************************
    ### Build model
    #************************************
    model = keras.models.Sequential()
    for i in range(len(dP.HL)):
        model.add(keras.layers.Dense(dP.HL[i],
            activation = 'relu',
            input_dim=np.array(data).shape[1],
            kernel_regularizer=keras.regularizers.l2(dP.l2)))
        model.add(keras.layers.Dropout(dP.drop))

    model.add(keras.layers.Dense(numClasses, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',
        optimizer=optim,
        metrics=['accuracy'])

    tbLog = keras.callbacks.TensorBoard(log_dir=tb_directory, histogram_freq=120,
            batch_size=dP.batch_size,
            write_graph=True, write_grads=True, write_images=True)
    tbLogs = [tbLog]

    log = model.fit(data, labels,
        epochs=dP.epochs,
        batch_size=dP.batch_size,
        callbacks = tbLogs,
        verbose=2,
        validation_split=dP.cv_split)

    model.save(model_name)
    keras.utils.plot_model(model, to_file=model_directory+'/keras_MLP_model.png', show_shapes=True)

    print('\n  =============================================')
    print('  \033[1mKeras MLP\033[0m - Model Configuration')
    print('  =============================================')
    #for conf in model.get_config():
    #    print(conf,"\n")
    print("  Training set file:",learnFile)
    print("  Data size:", A.shape,"\n")
    print("  Number of learning labels:",dP.numLabels)

    loss = np.asarray(log.history['loss'])
    val_loss = np.asarray(log.history['val_loss'])

    accuracy = np.asarray(log.history['acc'])
    val_acc = np.asarray(log.history['val_acc'])
    
    print("  Number unique classes (training): ", np.unique(Cl1).size)
    printParam()
    print('\n  ========================================================')
    print('  \033[1mKeras MLP - Classifier \033[0m - Training Summary')
    print('  ========================================================')
    print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(accuracy),
            100*np.amax(accuracy), 100*accuracy[-1]))
    print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}".format(np.average(loss), np.amin(loss), loss[-1]))
    print('\n\n  ========================================================')
    print('  \033[1mKeras MLP - Classifier \033[0m - Validation Summary')
    print('  ========================================================')
    print("\n  \033[1mAccuracy\033[0m - Average: {0:.2f}%; Max: {1:.2f}%; Last: {2:.2f}%".format(100*np.average(val_acc),
        100*np.amax(val_acc), 100*val_acc[-1]))
    print("  \033[1mLoss\033[0m - Average: {0:.4f}; Min: {1:.4f}; Last: {2:.4f}\n".format(np.average(val_loss), np.amin(val_loss), val_loss[-1]))

#************************************
# Prediction
#************************************
def predict(teamString):
    dP = Conf()
    if dP.useTFKeras:
        import tensorflow.keras as keras  #tf.keras
    else:
        import keras   # pure Keras
    
    R = np.array([np.fromstring(teamString, dtype='uint8', sep=',')])

    mcr = pickle.loads(open("keras_mcr.pkl", "rb").read())
    model = keras.models.load_model("keras_MLP_model.hd5")
    predictions = model.predict(R, verbose=0)[0]
    predict_classes = model.predict_classes(R)
    pred_class = np.argmax(predictions)
    predProb = round(100*predictions[pred_class],2)
    rosterPred = np.where(predictions>0.01)[0]

    names = [mcr.names[x] for x in R[0]]

    print('\n  ========================================================')
    print('  \033[1mPredicting score for game\033[0m ')
    print('  ========================================================')
    print('  {0:s} | {1:s} | {2:s} | {3:s} '.format(names[0], names[1], names[2], names[3]))
    for i in range(rosterPred.size):
        print("  {0:}:  {1:.2f}%".format(mcr.inverse_transform(rosterPred[i]), predictions[i]*100))

#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile):
    dP = Conf()
    print("\n  Opening learning file: ",learnFile)
    try:
        df = pd.read_csv(learnFile)
        if dP.fullSet == True:
            numCols = len(df.columns)
        else:
            numCols = dP.numCols
        A = np.array(df.values[4:,1:numCols],dtype=np.float64)
    except:
        print("\033[1m Learning file not found\033[0m")
        return

    names = list(df)[1:]
    data = []
    labels = []

    for i in range(A.shape[0]):
        N = np.where(~np.isnan([A[i,:]]))[1]
        if len(N)==4:
            data.append(N.tolist())
            Cl = A[i,:][~np.isnan(A[i,:])]
            if dP.withScore == False:
                Cl = np.where(Cl==np.max(Cl),1,0)
            labels.append(Cl.tolist())

    data = np.array(data)
    labels = np.array(labels)

    return A, data, labels, names

#************************************
# Print NN Info
#************************************
def printParam():
    dP = Conf()
    print('\n  ================================================')
    print('  \033[1mKeras MLP\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout:', dP.drop,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    if dP.fullSizeBatch == True:
        print('  Batch size: full')
    else:
        print('  Batch size:', dP.batch_size)
    print('  Number of labels:', dP.numLabels)
    #print('  ================================================\n')

#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Usage:\n')
    print(' Train (Random cross validation):')
    print('  python3 FoosballML.py -t <learningFile>\n')
    print(' Predict:')
    print('  python3 FoosballML.py -p <players numbers separated by comma>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
