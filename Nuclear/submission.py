# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import os
import pandas as pd 
import numpy as np
import multiprocessing # 여러 개의 일꾼 (cpu)들에게 작업을 분산시키는 역할
from multiprocessing import Pool 
from functools import partial # 함수가 받는 인자들 중 몇개를 고정 시켜서 새롭게 파생된 함수를 형성하는 역할
from data_loader_v2 import data_loader_v2 # 자체적으로 만든 data loader version 2.0 ([데이콘 15회 대회] 데이터 설명 및 데이터 불러오기 영상 참조)

from sklearn.ensemble import RandomForestClassifier
import joblib # 모델을 저장하고 불러오는 역할

# Set Path

train_folder = 'e:/Dacon/train/'
test_folder = 'e:/Dacon/test/'
train_label_path = 'e:/Dacon/train_label.csv'


# Load File

train_list = os.listdir(train_folder)
test_list = os.listdir(test_folder)
train_label = pd.read_csv(train_label_path, index_col=0)

# 모든 csv 파일의 상태_B로 변화는 시점이 같다라고 가정
# 하지만, 개별 csv파일의 상태_B로 변화는 시점은 상이할 수 있음
def data_loader_all_v2(func, files, folder='', train_label=None, event_time=10, nrows=60):   
    func_fixed = partial(func, folder=folder, train_label=train_label, event_time=event_time, nrows=nrows)     
    if __name__ == '__main__':
        pool = Pool(processes=multiprocessing.cpu_count()) 
        df_list = list(pool.imap(func_fixed, files)) 
        pool.close()
        pool.join()        
    combined_df = pd.concat(df_list)    
    return combined_df

train = data_loader_all_v2(data_loader_v2, train_list, folder=train_folder, train_label=train_label, event_time=10, nrows=60)


# Train
X_train = train.drop(['label'], axis=1)
y_train = train['label']
model = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1)
model.fit(X_train, y_train)
# joblib.dump(model, 'model.pkl')

test = data_loader_all_v2(data_loader_v2, test_list, folder=test_folder, train_label=None, event_time=10, nrows=60)

pred = model.predict_proba(test)


# Submission

submission = pd.DataFrame(data=pred)
submission.index = test.index
submission.index.name = 'id'
submission = submission.sort_index()
submission = submission.groupby('id').mean()
submission.to_csv('e:/Dacon/submission.csv', index=True) #제출 파일 만들기