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
import pickle
from sklearn.metrics import f1_score
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

# Load train set and normalizing
# X_train =X_train / 255 fails due to it being very large
with open('/home/allen/hackerEarth/DL2/train_ip_balanced.dat',"rb") as f:
    X_train = pickle.load(f)
with open('/home/allen/hackerEarth/DL2/X_valid.dat',"rb") as f:
    X_valid = pickle.load(f)
with open('/home/allen/hackerEarth/DL2/X_test.dat',"rb") as f:
    X_test = pickle.load(f)
with open('/home/allen/hackerEarth/DL2/train_op_balanced.dat',"rb") as f:
    Y_train = pickle.load(f)
with open('/home/allen/hackerEarth/DL2/Y_valid.dat',"rb") as f:
    Y_valid = pickle.load(f)
with open('/home/allen/hackerEarth/DL2/Y_test.dat',"rb") as f:
    Y_test = pickle.load(f)
# input needs to be a 4D tensor
X_train=X_train[...,np.newaxis]    
X_valid=X_valid[...,np.newaxis]
X_test=X_test[...,np.newaxis]

# Data generator
datagenTrain = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
datagenValid = ImageDataGenerator()
# VGG
model=Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(256, 256, 1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(ZeroPadding2D((1,1)))
model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))

#model.add(Dropout(0.5))
#model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))

# last layer has 14 classes=shape of Y_train
model.add(Dense(Y_train.shape[1], activation='softmax'))
sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#adamOpt=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])
model.summary()

with tf.device('/gpu:0'):
    model.fit_generator(datagenTrain.flow(X_train, Y_train, batch_size=16), validation_data=datagenValid.flow(X_valid, Y_valid, batch_size=16),\
          epochs=100, verbose=1)
    
score = model.evaluate(X_test, Y_test)
result = model.predict(X_test)
f1_score(Y_test, result, average='weighted')
model.save('xrayModel.h5')
# Comparing result

    



























