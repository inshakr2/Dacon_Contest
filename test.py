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

import collections
import operator
train.iloc[train.index != range(3360,3375),:]
sorted(collections.Counter(train.label), key=operator.getitem(1))


train.loc[train.label == 110, 'id']



# 110번 라벨 중 임의의 id에서 유니크 컬럼 값을 비교해봄 
# id = 0 , 122

a = train.loc[train.id == 0,:]
b = train.loc[train.id == 122,:]
a.unique()

idx = []
for i in range(2,5124):
    if len(a.iloc[:,i].unique()) == 1:
        idx.append(i)
idx_b = []
for i in range(2,5124):
    if len(b.iloc[:,i].unique()) == 1:
        idx_b.append(i)
        
idx == idx_b
len(idx)
len(idx_b)
collections.Counter(idx) - collections.Counter(idx_b)
# 결과 : Counter({1539: 1, 1540: 1, 1541: 1, 2207: 1})




# 30번확인
c = train.loc[train.id == 30,:]
idx_c = []
for i in range(2,5124):
    if len(c.iloc[:,i].unique()) == 1:
        idx_c.append(i)
len(idx_c)
c.label

# 73번 라벨을 가지고 있는 것을 확인, 동일한 라벨을 가지고 있는 아이디는 182
# 둘을 비교 ( 30 <-> 182 )
train.loc[train.label == 73,:]

d = train.loc[train.id == 182,:]
idx_d = []
for i in range(2,5124):
    if len(d.iloc[:,i].unique()) == 1:
        idx_d.append(i)

len(idx_c)  # 3795
len(idx_d)  # 2905

c.info()
d.info()



idx_whole = []
for i in range(2,5124):
    if len(train.iloc[:,i].unique()) == 1:
        idx_whole.append(i)
        
len(idx_whole) # 980

idx_whole_2 = []
for i in range(2,5124):
    if len(train.iloc[:,i].unique()) <= 828:
        idx_whole_2.append(i)
        
len(idx_whole_2) #3030

15 * 828  

train.iloc[:,5123]

idx_whole_3 = []
for j in train.id.unique():
    temp = train.loc[train.id == j,:]
    for i in range(2,5124):
        if len(temp.iloc[:,i].unique()) == 1:
            idx_whole_3.append(i)
    print(j)        

file = open('e:/Dacon/idx_whole_3.txt','wb')
pickle.dump(idx_whole_3,file)
file.close()

file = open('e:/Dacon/idx_whole_3.txt','rb')
idx_whole_3 = pickle.load(file)
file.close()


import collections
import operator
import matplotlib.pylab as plt
collections.Counter(idx_whole_3).values()
dict(collections.Counter(idx_whole_3))


y = list()
for i,j in x.items():
    if j < 828 :
        y.append(i)
len(y)

x = dict(collections.Counter(idx_whole_3))
plt.figure(figsize=(12,6))
plt.bar(x=x.keys(), height = x.values())

train
