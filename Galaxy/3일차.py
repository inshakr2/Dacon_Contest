# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:06:44 2020

@author: ChangYeol
"""

#3일차
#
#기존 preprocessing 함수에 적용시킨 train값을 keras 선형으로 예측해보기
#더 해볼 수 있는 것들
#1. scale
#2. fiverID 연구


import pandas as pd
import numpy as np

data = pd.read_csv('e:/Dacon/Galaxy/train.csv')
test = pd.read_csv('e:/Dacon/Galaxy/test.csv')

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
        

data.type.groupby(data.fiberID).unique()

