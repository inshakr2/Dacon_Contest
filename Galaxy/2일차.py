# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:01:56 2020

@author: ChangYeol
"""
import pandas as pd
import numpy as np

data = pd.read_csv('e:/Dacon/Galaxy/train.csv')
test = pd.read_csv('e:/Dacon/Galaxy/test.csv')
def preprocess(DF,From,To,MAX,MIN):
    
    while True:
            
        DF.iloc[:,From][DF.iloc[:,From] > MAX] = np.nan
        DF.iloc[:,From][DF.iloc[:,From] < MIN] = np.nan
        
        MEAN = DF.iloc[:,From].mean()
        
        DF.iloc[:,From] = DF.iloc[:,From].fillna(MEAN)
        
        From += 1
        
        if From > To :
            break


import missingno as msno
msno.bar(data)   


preprocess(data,3,22,50,0)
preprocess(test,2,21,50,0)

(data.iloc[:,3:]).plot.box(figsize=(20,15)) 
(test.iloc[:,2:]).plot.box(figsize=(20,15)) 


def preprocess(DF,From,To,Test_DF):
    
    while True:
        
        MAX = max(Test_DF.iloc[:,From-1])
        MIN = min(Test_DF.iloc[:,From-1])
        
        DF.iloc[:,From][DF.iloc[:,From] > MAX] = np.nan
        DF.iloc[:,From][DF.iloc[:,From] < MIN] = np.nan
        
        MEAN = DF.iloc[:,From].mean()
        
        DF.iloc[:,From] = DF.iloc[:,From].fillna(MEAN)
        
        From += 1
        
        if From > To :
            break
        
preprocess(data,3,22,test)
(data.iloc[:,3:]).plot.box(figsize=(20,15)) 
# scale
# 대체 값을 평균말고 다른 값으로
# Min Max 값 조정
# fiver ID ???


data.fiberID.count()
data.loc[:,'fiberID'].groupby(data.fiberID).count()
data.info()
groupby(data.fiberID).describe()

data.loc[data.loc[:,'fiberID'] == 300, 'type']

data.iloc[:,1:8]

data.fiberID.groupby(data.type)
data.type.groupby(data.fiberID).unique()
