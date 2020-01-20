# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:42:10 2020

@author: ChangYeol
"""

import os
import pandas as pd 
import numpy as np
from multiprocessing import Pool 
import multiprocessing
from data_loader import data_loader #data_loader.py 파일을 다운 받아 주셔야 합니다. 
from tqdm import tqdm
from functools import partial
import pickle

def data_loader_all(func, path, train, nrows, **kwargs):
    '''
    Parameters:
    
    func: 하나의 csv파일을 읽는 함수 
    path: [str] train용 또는 test용 csv 파일들이 저장되어 있는 폴더 
    train: [boolean] train용 파일들 불러올 시 True, 아니면 False
    nrows: [int] csv 파일에서 불러올 상위 n개의 row 
    lookup_table: [pd.DataFrame] train_label.csv 파일을 저장한 변수 
    event_time: [int] 상태_B 발생 시간 
    normal: [int] 상태_A의 라벨
    
    Return:
    
    combined_df: 병합된 train 또는 test data
    '''
    
    # 읽어올 파일들만 경로 저장 해놓기 
    files_in_dir = os.listdir(path)
    
    files_path = [path+'/'+file for file in files_in_dir]
    
    if train :
        func_fixed = partial(func, nrows = nrows, train = True, lookup_table = kwargs['lookup_table'], event_time = kwargs['event_time'], normal = kwargs['normal'])
        
    else : 
        func_fixed = partial(func, nrows = nrows, train = False)
    
    
    # 여러개의 코어를 활용하여 데이터 읽기 
    if __name__ == '__main__':
        pool = Pool(processes = multiprocessing.cpu_count()) 
        df_list = list(tqdm(pool.imap(func_fixed, files_path), total = len(files_path)))
        pool.close()
        pool.join()
    
    # 데이터 병합하기 
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df

# 경로 지정
train_path = 'e:/Dacon/train'
test_path = 'e:/Dacon/test'
label = pd.read_csv('e:/Dacon/train_label.csv')

# train (event T = 10)
train = data_loader_all(data_loader, path = train_path, train = True, nrows = 100, normal = 999, event_time = 10, lookup_table = label)

file = open('e:/Dacon/train_10.txt','wb')      
pickle.dump(train,file) 
file.close() 

# test
test = data_loader_all(data_loader, path = test_path, train = False, nrows = 60)

file = open('e:/Dacon/test.txt','wb')
pickle.dump(test,file)
file.close()

train.head(100)
