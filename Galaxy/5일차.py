# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('e:/Dacon/Galaxy/train.csv', index_col='id')
test = pd.read_csv('e:/Dacon/Galaxy/test.csv').reset_index(drop=True)

unique_labels = train['type'].unique()
label_dict = {val: i for i, val in enumerate(unique_labels)}
i2lb = {v:k for k, v in label_dict.items()}


scaler = StandardScaler()
labels = train['type']
train = train.drop(columns=['fiberID', 'type']) # fiberID는 1000개의 categorical feature이며, 이 커널에서는 무시합니다.

_mat = scaler.fit_transform(train)
train = pd.DataFrame(_mat, columns=train.columns, index=train.index)

train_x = train
train_y = labels.replace(label_dict)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0003,
)

# sigmoid, tanh, relu, leakly relu, prelu, maxout
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(len(train_x.columns)),
  tf.keras.layers.Dense(256*4, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*4, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*3, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*3, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*2, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*2, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*1, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(256*1, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(128, activation='elu'),
  tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(64, activation='elu'),
  tf.keras.layers.Dense(32, activation='elu'),
  tf.keras.layers.Dense(19, activation='softmax')
])

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x,
          train_y,
          batch_size=256*3,
          validation_split=0.1,
          epochs=200)

test_ids = test['id']
test = test.drop(columns=['id', 'fiberID'])
test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

pred_mat = model.predict(test)

sample = pd.read_csv('E:/Dacon/Galaxy/sample_submission.csv')

submission = pd.DataFrame(pred_mat, index=test.index)
submission = submission.rename(columns=i2lb)
submission = pd.concat([test_ids, submission], axis=1)
submission = submission[sample.columns]
submission.to_csv("E:/Dacon/Galaxysubmission_NOPRE_cy.csv", index=False)