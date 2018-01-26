# Image processing
import cv2
from skimage import exposure
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import glob
from skimage import io

X_train_sk = [exposure.equalize_adapthist(io.imread(x,as_grey=True)) for x in glob.glob('/home/allen/hackerEarth/DL2/Code/sample_images/*.png')]
plt.imshow(X_train_sk[3], cmap='gray')

# OpenCV reads as BGR
#X_train = [cv2.equalizeHist(cv2.imread(x,0)) for x in glob.glob('/home/allen/hackerEarth/DL2/Code/sample_images/*.png')]
X_train = [cv2.cvtColor(cv2.imread(x),cv2.COLOR_BGR2HLS) for x in glob.glob('/home/allen/hackerEarth/DL2/Code/sample_images/*.png')]
X_train = np.array(X_train)
cv2.imshow('image',X_train[2])
cv2.waitKey(0)
