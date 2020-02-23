# -*- coding: utf-8 -*-

import keras    
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D

import pandas as pd
import numpy as np

train = pd.read_csv('e:/Dacon/Galaxy/train.csv')
train_arr = np.array(train.iloc[:,3:]).reshape((-1,4,5,1))
train_arr.shape


model = Sequential()
model.add(Conv2D(256*2, (3, 3), padding = 'same', input_shape = (4, 5, 1)))
    # Conv2D(필터 개수, 필터의 크기, 패딩, 인풋데이터 행렬 모양)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))
model.summary()

model.add(Conv2D(256*3, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256*4, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256*2, (2, 2), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256*1, (2, 2), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (2, 2), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (2, 2), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(32, (2, 2), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(45))    # 분류해야 할 개수로 Dense 적용
model.add(Activation('softmax'))
model.summary()