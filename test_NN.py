# Testing workflow on a sample
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score

# Load train set and normalizing
with open('train_ip.dat','rb') as f:
    X_train = pickle.load(f)
X_train /=255

with open('train_op.dat','rb') as f:
    Y_train = pickle.load(f)
Y_train /=255
    
# Splitting training set
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, \
                                                test_size=0.15, shuffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, \
                                                test_size=0.15, shuffle=True)

# VGG
model=Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# last layer has 14 classes=shape of Y_train
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])

model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid), batch_size=32, \
          epochs=10, verbose=1)
score = model.evaluate(X_test, Y_test)
result = model.predict(X_test)
f1_score(Y_test, result, average='weighted')

# Comparing result

    



























