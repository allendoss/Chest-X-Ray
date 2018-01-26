# Testing workflow on a sample
import numpy as np
import tensorflow as tf
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from random import shuffle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

file_train=pd.read_csv('~/hackerEarth/DL2/csv/train.csv')

# Multiclass classification for one hot encoding
# Can you use generators instead?
Y=file_train['detected']
Y_train=[]
for yLabel in Y:
    Y_train.append(int(yLabel.split('_')[1]))

Y_train=np.array(Y_train)
Y_train=Y_train[...,np.newaxis]
print('Shape of training output set:'+str(Y_train.shape))
# +1 because there are 14 classes and class_0 that doesn't exist
Y_train=keras.utils.to_categorical(Y_train,num_classes=np.amax(Y_train)+1)
with open('train_op.dat',"wb") as f:
    pickle.dump(Y_train,f)

# Reading the training image set
# glob returns a python list of filenames for each png
# image_folder=glob.glob('/home/allen/hackerEarth/DL2/train/*.png')
# Need to match order of image name with corresponding class from Y_train
images = [cv2.resize(cv2.imread('/home/allen/hackerEarth/DL2/train/'+\
                                str(img_name),1),(128,128)) \
                                for img_name in tqdm(file_train['image_name'].values)]
X_train = np.array(images)
with open('train_ip.dat',"wb") as f:
    pickle.dump(X_train,f)

# Splitting training set
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True)

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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train, Y_train, batch_size=32, epochs=10)
score = model.evaluate(X_test, Y_test, batch_size=32)