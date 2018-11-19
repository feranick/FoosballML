# -*- coding: utf-8 -*-
'''
**********************************************************
* libDataML - Library for DataML
* 20181119a
* Uses: Keras, TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
import numpy as np
import pickle
from bisect import bisect_left

#************************************
# Normalizer
#************************************
class Normalizer(object):
    def __init__(self, M):
        self.M = M
        self.YnormTo = 1
        self.min = np.amin(self.M)
        self.max = np.amax(self.M)
    
    def transform_matrix(self,y):
        Mn = np.multiply(y - self.min,
                self.YnormTo/(self.max - self.min))
        return Mn
        
    def transform_inverse(self,Vn):
        V = (np.multiply(Vn,(self.max - self.min)/self.YnormTo) + self.min).astype(int)
        return V

    def save(self, name):
        with open(name, 'ab') as f:
            f.write(pickle.dumps(self))

#************************************
# CustomRound
#************************************
class CustomRound:
    def __init__(self,iterable):
        self.data = sorted(iterable)

    def __call__(self,x):
        data = self.data
        ndata = len(data)
        idx = bisect_left(data,x)
        if idx <= 0:
            return data[0]
        elif idx >= ndata:
            return data[ndata-1]
        x0 = data[idx-1]
        x1 = data[idx]
        if abs(x-x0) < abs(x-x1):
            return x0
        return x1

#************************************
# MultiClassReductor
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return self.totalClass[int(a)]

    def classes_(self):
        return self.totalClass

    def names(self,names):
        self.names = names
