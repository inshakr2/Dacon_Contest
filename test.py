# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:12:17 2020

@author: ChangYeol
"""

a = train.iloc[:100,]
a.describe()

train.info()
len(train.iloc[:,1].unique())

idx = []
for i in range(2,5124):
    if len(train.iloc[:,i].unique()) == 1:
        idx.append(i)

idx_30 = []
for i in range(2,5124):
    if len(train[train.id == 30].iloc[:,i].unique()) == 1:
        idx_30.append(i)


idx_30x = []
for i in range(2,5124):
    if len(train[train.id != 30].iloc[:,i].unique()) == 1:
        idx_30x.append(i)

train[train.id == 30]

len(idx)
len(idx_30)
len(idx_30x)

train.iloc[train.index != range(3360,3375),:]

