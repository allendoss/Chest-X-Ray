# Testing workflow on a sample
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, ZeroPadding2D
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from sklearn.metrics import f1_score
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

# Load train set and normalizing
with open('/home/allen/hackerEarth/DL2/train_ip.dat','rb') as f:
    X_train = pickle.load(f)
# X_train =X_train / 255 fails due to it being very large

# input needs to be a 4D tensor
X_train=X_train[...,np.newaxis]    

with open('/home/allen/hackerEarth/DL2/train_op.dat','rb') as f:
    Y_train = pickle.load(f)
# WTF would yould normalize this!!!!Y_train /=255
    
# Splitting training set
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, \
                                                test_size=0.15, shuffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, \
                                                test_size=0.15, shuffle=True)

plt.hist(Y_train)
# Data generator
datagen = ImageDataGenerator()
# VGG
model=Sequential()
BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, \
                   scale=True, beta_initializer='zeros', gamma_initializer='ones', \
                   moving_mean_initializer='zeros', moving_variance_initializer='ones', \
                   beta_regularizer=None, gamma_regularizer=None, \
                   beta_constraint=None, gamma_constraint=None)
model.add(ZeroPadding2D((1,1),input_shape=(128, 128, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))

# last layer has 14 classes=shape of Y_train
model.add(Dense(Y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.summary()

with tf.device('/gpu:0'):
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), validation_data=datagen.flow(X_valid, Y_valid, batch_size=32),\
          epochs=50, verbose=1)
    #model.fit(X_train, Y_train, validation_data=(X_valid,Y_valid),batch_size=16, \
    #          epochs=10, verbose=1)
score = model.evaluate(X_test, Y_test)
result = model.predict(X_test)
f1_score(Y_test, result, average='weighted')
model.save('xrayModel.h5')
# Comparing result

    



























