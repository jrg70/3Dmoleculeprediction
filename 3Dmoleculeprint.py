%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image
from keras.preprocessing.image import load_img, ImageDataGenerator
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.utils import shuffle
from mpl_toolkits import mplot3d
import os

## Example of one molecule in 3D space
# See how many pictures/slices are there
# Picture silces at: https://drive.google.com/drive/folders/1rqPIXbvQEPmllpene6wUzLWR-ygTcmRF?usp=sharing
PIC_FOLDER = 'drive/MyDrive/Colab Notebooks/3D molecules/Trans-aconiticacid/'
totalFiles = 0
totalDir = 0
for base, dirs, files in os.walk(PIC_FOLDER):
    print('Searching in : ',base)
    for Files in files:
        totalFiles += 1
    for directories in dirs:
        totalDir += 1
print('Total number of files',totalFiles,'Total Number of directories',totalDir)

# create 3D stack
im = []
pix = 250
for i in range(1,totalFiles-1):
  im.append((np.array(load_img(PIC_FOLDER+str(i)+'.jpg').resize((pix,pix))) / 255.0))

# create tensor
im = tf.stack(im)

# plot a slice
IMG_INDEX = 130
plt.imshow(im[IMG_INDEX] ,cmap=plt.cm.binary)
plt.show()

# 3D plot
# mesh
z = []
y = []
x = []

for i in range(im.shape[0]):
  a = (i+1)*np.ones(pix*pix)
  z = np.concatenate((z,a),axis=0)

for j in range(pix):
  b = (j+1)*np.ones(pix)
  y = np.concatenate((y,b),axis=0)
CO = y  
for i in range(im.shape[0]-1):  
  y = np.concatenate((y,CO),axis=0)  

for k in range(pix):
  c = (k+1)*np.ones(1)
  x = np.concatenate((x,c),axis=0)
CO = x 
for i in range((pix)*(im.shape[0])-1):  
  x = np.concatenate((x,CO),axis=0)

XYZ = np.c_[z,y,x]  

print(XYZ)

# reshape
print(tf.shape(im).numpy())
im1 = tf.reshape(im,[(im.shape[0]*(pix*pix)),3])
XYZC = np.c_[XYZ,im1]
print(XYZC)

# delete white stacks
print(XYZC.shape)
XYZC = np.delete(XYZC,np.where(XYZC == 1.0)[0],axis=0)
print(XYZC.shape)
test = XYZC[:,3] - XYZC[:,4] - XYZC[:,5]
print(test)
test1 = set(test)
print(test1)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(XYZC[:,0],XYZC[:,1],XYZC[:,2], marker='.', c=XYZC[:,3:6]**15)
plt.show()
### This shows the 3D structure of one molecule
