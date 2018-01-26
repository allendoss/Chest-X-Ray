# Testing workflow on a sample
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random

file_train=pd.read_csv('~/hackerEarth/DL2/csv/train.csv')

# Need to match order of image name with corresponding class from Y_train
#images = [cv2.resize(cv2.equalizeHist(cv2.imread('/home/allen/hackerEarth/DL2/train/'+\
#                                str(img_name),0)),(256,256)) \
#                                for img_name in tqdm(file_train['image_name'].values)]
#X_train = np.array(images)
#with open('/home/allen/hackerEarth/DL2/train_ip.dat',"wb") as f:
#    pickle.dump(X_train,f)
    
with open('/home/allen/hackerEarth/DL2/train_ip.dat','rb') as f:
    X_train = pickle.load(f)
    
Y=file_train['detected']
Y_train=[]
for yLabel in Y:
    Y_train.append(int(yLabel.split('_')[1]))

# Splitting training set
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, \
                                                test_size=0.15, shuffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, \
                                                test_size=0.15, shuffle=True)
Y_train_ = np.array(Y_train)
X_train_ = np.copy(X_train) # if you use X=X1=>both point to same location=>no copy
# Cropping
# Image Transformation
class imbalancedClass:
    
    def __init__(self, X_train_, Y_train):
        self.X_train_ = X_train_
        self.Y_train = Y_train
    
    def imageTransform(self, img):
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), random.randint(-10,10), 1)
        return cv2.warpAffine(img, M, (cols,rows))

    def oversampling(self, n):
        nList=[]
        for idx,y in enumerate(self.Y_train):
            if y==n:
                nList.append(idx)        
        while self.Y_train.count(n)<1400:
            for index in nList:
                img = self.imageTransform(self.X_train_[index])
                img = img[np.newaxis,...]
                self.X_train_ = np.append(self.X_train_, img, axis=0)
                self.Y_train.append(n)
            print('Samples:',Y_train.count(n))
            
    def undersampling(self,n):
        nList=[]
        for idx,y in enumerate(self.Y_train):
            if y==n:
                nList.append(idx)
        print('Samples:',Y_train.count(n))
        for index in nList: #nList is always going to be greater than 3000
            if self.Y_train.count(n) > 3000:
                self.X_train_ = np.delete(self.X_train_, index, axis=0)
                del self.Y_train[index]
        print('Samples:',Y_train.count(n))
                        

imageSet_256x256 = imbalancedClass(X_train_, Y_train)

# Undersampling
for n in range(1,15):
    if Y_train.count(n)<1000:
        print('class:',n)
        imageSet_256x256.oversampling(n)
    elif Y_train.count(n) > 3000:
        print('class',n)
        imageSet_256x256.undersampling(n)

# One hot encoding
Y_train = imageSet_256x256.Y_train
a = [Y_train.count(i) for i in range(1,15)]
print('Y_train samples:',a)
Y_train=np.array(Y_train).reshape(-1)
Y_train=Y_train[...,np.newaxis]
print('Shape of training output set:'+str(Y_train.shape))
# +1 because there are 14 classes and class_0 that doesn't exist in real life 
Y_train = OneHotEncoder(np.amax(Y_train+1)).fit_transform(Y_train).toarray()
with open("/home/allen/hackerEarth/DL2/train_op_imbalance.dat","wb") as f:
    pickle.dump(Y_train,f)
X_train = imageSet_256x256.X_train_
with open('/home/allen/hackerEarth/DL2/train_ip_imbalance.dat',"wb") as f:
    pickle.dump(X_train,f)
    
