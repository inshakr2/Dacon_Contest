# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:07:04 2020

@author: ChangYeol
"""


import pandas as pd
import numpy as np

data = pd.read_csv('e:/Dacon/Galaxy/train.csv')

data.info()
data.psfMag_r
data.type
len(data.fiberID.unique())

test = pd.read_csv('e:/Dacon/Galaxy/test.csv')

test.info()

data.petroMag_u.head()
data.petroMag_g.head()
data.petroMag_r.head()

data.head()

data.loc[:,'petroMag_u':'petroMag_z']
data.loc[:,'psfMag_u':'psfMag_z']

data.type.unique()
len(data.type.unique())

25.85922905	22.42692865	21.67355143	19.61001198	18.37614071


 emp['SALARY'].groupby([emp['DEPARTMENT_ID'],emp['JOB_ID']]) 

data.loc[:,'fiberMag_u':'fiberMag_z'].groupby(data.fiberID).describe()
data.iloc[:,3:8].describe()
data.iloc[:,8:13].describe()
data.iloc[:,13:18].describe()
data.iloc[:,18:23].describe()

data.loc[data.modelMag_u == min(data.modelMag_u),]


data.loc[data.id == 67227,]


import matplotlib.pylab as plt

plt.boxplot([data.iloc[:,3],data.iloc[:,4]])

plt.boxplot(data.iloc[:,22])

arr =list()
for i in range(3,23):
    N = 'data.iloc[:,'+str(i)+']'
    arr.append(N)
arr    


(data.iloc[:,3:]).plot.box(figsize=(20,15)) 


test = pd.read_csv('e:/Dacon/Galaxy/test.csv')
(test.iloc[:,2:]).plot.box(figsize=(20,15)) 
test.info()
data.info()


data.iloc[:,3:].apply(lambda x : x.mean() if (x > 50) or (x < 0) else x)

data.apply(lambda x : np.mean(x))

data.loc[data.iloc[:,3] < 50 and data.iloc[:,3] > 0,]
data.mean()

data.iloc[:,3][data.iloc[:,3] > 50 ]


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


