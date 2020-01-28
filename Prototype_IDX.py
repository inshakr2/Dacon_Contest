# -*- coding: utf-8 -*-

import pickle
import re

# Prototype_유의미 컬럼 찾기
file = open('e:/Dacon/train_16R_10ET.txt','rb')
train = pickle.load(file)
file.close()

IDX = set()
for ID in train.id.unique():
    temp = train.loc[train.id == ID,:]
    for i in range(2,5124):
        if len(temp.iloc[:,i].unique()) != 1:
            IDX.add(i)
    print(ID)

file= open('e:/Dacon/IDX.txt','wb')
pickle.dump(IDX,file)
file.close()
IDX.update([0,1])
TR = train.iloc[:,list(IDX)]

file= open('e:/Dacon/IDX.txt','rb')
IDX = pickle.load(file)
file.close()

idx = list(IDX)
TR.iloc[:,4]

TR.info()
for i in range(2,2831):
    if TR.iloc[:,i].dtype == 'object':
        print(i)

TR.loc[TR.id == 30,'V0087']
TR.loc[:,'V0086'].unique()
TR.loc[TR.V0086 != 52.7, 'label'].unique()

TR.iloc[96:111,[0,1,72,-1]]
TR.iloc[112:128,[0,1,72,-1]]
TR.loc[TR.label==73, 'id']

TR.loc[TR.id==182,:]
TR.loc[TR.id==30,:]
TR.loc[:,'V1269']

for i in range(2,2831):
    if TR.iloc[:,i].dtype == 'object':
        for j in TR.iloc[:,i].unique():
            if type(j) == str:
                print(i,j)

TR.iloc[:,72].unique()  
TR.iloc[:,2274].unique()
TR.iloc[:,1315].unique()
TR.iloc[:,1315].name
TR.iloc[:,1354].unique()
TR.iloc[:,1354].name
TR.loc[TR.iloc[:,892] == 'OFF','id']
TR.info()

# KNN, RF, DT, LR, NB,
        
type(TR.iloc[:,72].unique()[39]) == 'str'
TR.iloc[:,72].unique() == 'Bad'

import numpy as np
import missingno as msno
msno.matrix(train)
plt.show()

msno.bar(train)
plt.show()

TR.iloc[:,888].unique()
TR.iloc[:,72].name

train.loc[:,'V0086'].describe()
TR.loc[TR.loc[:,'V0086'] != 'Bad','V0086'].describe()

TR.loc[TR.iloc[:,72] == 'Bad',:]

TR.loc[TR.id != 30,'V0086'].astype(float).describe()


# 1. round한 정수값을 팜고
# 2. min - max
# 3. 최빈값, 중앙값, ... 을 대체
# 4. ]

idx = list(map(lambda x : x - 2,idx))

data1 = pd.read_csv('e:/Dacon/train/30.csv')
data1.iloc[:,idx].info() # 235

data2 = pd.read_csv('e:/Dacon/test/1154.csv')
data2.iloc[:,idx].info() # 271
