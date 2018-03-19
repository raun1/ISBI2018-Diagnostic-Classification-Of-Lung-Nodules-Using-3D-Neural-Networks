
#The following implements the code for the Multi_Output_Dense_Net architecture
#used in DIAGNOSTIC CLASSIFICATION OF LUNG NODULES USING 3D NEURAL NETWORKS
#accepted to ISBI-2018 conference 
#Authors @ Raunak Dey

import keras
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, MaxPooling2D,merge,MaxPooling3D,Conv3D,concatenate
from keras.layers import Activation, Dropout, Flatten, Dense,Merge,Input, Dense,Reshape,BatchNormalization
from keras import backend as K
from keras.models import Model
from keras import regularizers 
from keras import backend as K
import pprint
import random
import numpy as np
import os
import cv2
import math
from keras.layers.normalization import BatchNormalization as bn
from scipy.misc import imsave
import numpy as np
import time
from sklearn import metrics

    
#Hyper parameters to be set - 
#l2_Lambda - used for regularizing/penalizing parameters of the current layer
#Mainly used to prevent overfitting and is incorporated in the loss function
#Please see keras.io for more details
#DropP sets the % of dropout at the end of every dense block
#Kernel_size is the kernel size of the convolution filters
#Please see readme for additional resources.
#Layers such as xconv1a,xmerge1........ belong to the second branch of the architecture.
#The convolution layers's number indicates its level and so conv1a and xconv1a are at the same level
#and are parallel to each other
#Line - 718 merging of the two branches
 

l2_lambda = 0.0002
DropP = 0.3
kernel_size=3

#Input for the first branch Accepts an 3d array of 10 slices having 
#dimension of 256x256 followed by the first pooling layer 
#Every Convolution layer is followed by a Batch normalization layer

inputthree=Input(shape=(100,100,10,1), dtype='float32',name='inputthree')      
prepool= MaxPooling3D(pool_size=(2,2,2))(inputthree)

#After this point the two branches have exactly similar architecture
#The first dense block has a total of four layers

conv1a = Conv3D( 12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same', 
               kernel_regularizer=regularizers.l2(l2_lambda) )(prepool)
conv1a = bn()(conv1a)
conv1b = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(conv1a)
conv1b = bn()(conv1b)
merge1=concatenate([conv1a,conv1b])  #Merges all previous layers and then feeds it to the next layer
conv1c = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)
conv1c = bn()(conv1c)
merge2=concatenate([conv1a,conv1b,conv1c])
conv1d = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)
conv1d = bn()(conv1d)
pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1d)
#We have drop out only at the end of a particular Dense Block such as below
pool1 = Dropout(DropP)(pool1)
#First block ends here

#The following lines introduce the first intermediate output for the first branch
#We flatten the output and then connect it to a sigmoid dense block.
flatten1=Flatten()(pool1)
output1=Dense(1,activation='sigmoid')(flatten1)
#--------------------------------------------------------------------------------------------------
#Dense block two starts after this point
#It has 10 Convolution layers
conv2a = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(pool1)
conv2a = bn()(conv2a)
conv2b = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(conv2a)
conv2b = bn()(conv2b)
merge1=concatenate([conv2a,conv2b])
conv2c = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)
conv2c = bn()(conv2c)
merge2=concatenate([conv2a,conv2b,conv2c])
conv2d = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)
conv2d = bn()(conv2d)
merge3=concatenate([conv2a,conv2b,conv2c,conv2d])
conv2e = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)
conv2e = bn()(conv2e)
merge4=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e])
conv2f = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)
conv2f = bn()(conv2f)
merge5=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f])
conv2g = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)
conv2g = bn()(conv2g)
merge6=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f,conv2g])
conv2h = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)
conv2h = bn()(conv2h)
merge7=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f,conv2g,conv2h])
conv2i = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)
conv2i = bn()(conv2g)
merge8=concatenate([conv2a,conv2b,conv2c,conv2d,conv2e,conv2f,conv2g,conv2h,conv2i])
conv2j = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)
conv2j = bn()(conv2g)
pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2j)
pool2 = Dropout(DropP)(pool2)
#Intermediate output two for Branch 1
flatten2=Flatten()(pool2)
output2=Dense(1,activation='sigmoid')(flatten2)

#--------------------------------------------------------------------------------------------------
#Block 3 starts from this point on

conv3a = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(pool2)
conv3a = bn()(conv3a)
conv3b = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(conv3a)
conv3b = bn()(conv3b)
merge1=concatenate([conv3a,conv3b])
conv3c = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)
conv3c = bn()(conv3c)
merge2=concatenate([conv3a,conv3b,conv3c])
conv3d = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)
conv3d = bn()(conv3d)
merge3=concatenate([conv3a,conv3b,conv3c,conv3d])
conv3e = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)
conv3e = bn()(conv3e)
merge4=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e])
conv3f = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)
conv3f = bn()(conv3f)
merge5=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f])
conv3g = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)
conv3g = bn()(conv3g)
merge6=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g])
conv3h = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)
conv3h = bn()(conv3h)
merge7=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h])
conv3i = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)
conv3i = bn()(conv3i)
merge8=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i])
conv3j = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)
conv3j = bn()(conv3j)
merge9=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j])
conv3k = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)
conv3k = bn()(conv3k)
merge10=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k])
conv3l=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
conv3l = bn()(conv3l)
merge11=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l])
conv3m=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
conv3m = bn()(conv3m)
merge12=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m])
conv3n=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
conv3n = bn()(conv3n)
merge13=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n])
conv3o=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
conv3o = bn()(conv3o)
merge14=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o])
conv3p=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
conv3p = bn()(conv3p)
merge15=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p])
conv3q=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
conv3q = bn()(conv3q)
merge16=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q])
conv3r=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
conv3r = bn()(conv3r)
merge17=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q,conv3r])
conv3s=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
conv3s = bn()(conv3s)
merge18=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q,conv3r,conv3s])
conv3t=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
conv3t = bn()(conv3t)
merge19=concatenate([conv3a,conv3b,conv3c,conv3d,conv3e,conv3f,conv3g,conv3h,conv3i,conv3j,conv3k,conv3l,conv3m,conv3n,conv3o,conv3p,conv3q,conv3r,conv3s,conv3t])
conv3u=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
conv3u = bn()(conv3u)
pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3u)
pool3 = Dropout(DropP)(pool3)
flatten4=Flatten()(pool3)
output4=Dense(1,activation='sigmoid')(flatten4)
#--------------------------------------------------------------------------------------------------

conv4a = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(pool3)
conv4a = bn()(conv4a)
conv4b = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(conv4a)
conv4b = bn()(conv4b)
merge1=concatenate([conv4a,conv4b])
conv4c = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)
conv4c = bn()(conv4c)
merge2=concatenate([conv4a,conv4b,conv4c])
conv4d = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)
conv4d = bn()(conv4d)
merge3=concatenate([conv4a,conv4b,conv4c,conv4d])
conv4e = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)
conv4e = bn()(conv4e)
merge4=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e])
conv4f = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)
conv4f = bn()(conv4f)
merge5=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f])
conv4g = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)
conv4g = bn()(conv4g)
merge6=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g])
conv4h = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)
conv4h = bn()(conv4h)
merge7=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h])
conv4i = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)
conv4i = bn()(conv4i)
merge8=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i])
conv4j = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)
conv4j = bn()(conv4j)
merge9=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j])
conv4k = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)
conv4k = bn()(conv4k)
merge10=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k])
conv4l=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
conv4l = bn()(conv4l)
merge11=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l])
conv4m=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
conv4m = bn()(conv4m)
merge12=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m])
conv4n=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
conv4n = bn()(conv4n)
merge13=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n])
conv4o=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
conv4o = bn()(conv4o)
merge14=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o])
conv4p=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
conv4p = bn()(conv4p)
merge15=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p])
conv4q=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
conv4q = bn()(conv4q)
merge16=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q])
conv4r=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
conv4r = bn()(conv4r)
merge17=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q,conv4r])
conv4s=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
conv4s = bn()(conv4s)
merge18=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q,conv4r,conv4s])
conv4t=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
conv4t = bn()(conv4t)
merge19=concatenate([conv4a,conv4b,conv4c,conv4d,conv4e,conv4f,conv4g,conv4h,conv4i,conv4j,conv4k,conv4l,conv4m,conv4n,conv4o,conv4p,conv4q,conv4r,conv4s,conv4t])
conv4u=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
conv4u = bn()(conv4u)
pool4 = MaxPooling3D(pool_size=(2, 2, 1))(conv4u)
pool4 = Dropout(DropP)(pool4)
flatten5=Flatten()(pool4)
output5=Dense(1,activation='sigmoid')(flatten5)
#--------------------------------------------------------------------------------------------------

conv5a = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(pool4)
conv5a = bn()(conv5a)
conv5b = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(conv5a)
conv5b = bn()(conv5b)
merge1=concatenate([conv5a,conv5b])
conv5c = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge1)
conv5c = bn()(conv5c)
merge2=concatenate([conv5a,conv5b,conv5c])
conv5d = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge2)
conv5d = bn()(conv5d)
merge3=concatenate([conv5a,conv5b,conv5c,conv5d])
conv5e = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge3)
conv5e = bn()(conv5e)
merge4=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e])
conv5f = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge4)
conv5f = bn()(conv5f)
merge5=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f])
conv5g = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge5)
conv5g = bn()(conv5g)
merge6=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g])
conv5h = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge6)
conv5h = bn()(conv5h)
merge7=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h])
conv5i = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge7)
conv5i = bn()(conv5i)
merge8=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i])
conv5j = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge8)
conv5j = bn()(conv5j)
merge9=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j])
conv5k = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge9)
conv5k = bn()(conv5k)
merge10=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k])
conv5l=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge10)
conv5l = bn()(conv5l)
merge11=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l])
conv5m=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge11)
conv5m = bn()(conv5m)
merge12=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m])
conv5n=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge12)
conv5n = bn()(conv5n)
merge13=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n])
conv5o=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge13)
conv5o = bn()(conv5o)
merge14=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o])
conv5p=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge14)
conv5p = bn()(conv5p)
merge15=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p])
conv5q=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge15)
conv5q = bn()(conv5q)
merge16=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q])
conv5r=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge16)
conv5r = bn()(conv5r)
merge17=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q,conv5r])
conv5s=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge17)
conv5s = bn()(conv5s)
merge18=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q,conv5r,conv5s])
conv5t=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge18)
conv5t = bn()(conv5t)
merge19=concatenate([conv5a,conv5b,conv5c,conv5d,conv5e,conv5f,conv5g,conv5h,conv5i,conv5j,conv5k,conv5l,conv5m,conv5n,conv5o,conv5p,conv5q,conv5r,conv5s,conv5t])
conv5u=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(merge19)
conv5u = bn()(conv5u)

#End of Branch 1
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#Start of Branch 2 accepting a numpy array of size five
#Aside from the initial pooling layer of the first branch these two branches are identical 
inputthree1=Input(shape=(50,50,5,1), dtype='float32',name='inputthree1')   
xconv1a = Conv3D( 12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same', 
               kernel_regularizer=regularizers.l2(l2_lambda) )(inputthree1)
xconv1a = bn()(xconv1a)
xconv1b = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xconv1a)
xconv1b = bn()(xconv1b)
xmerge1=concatenate([xconv1a,xconv1b])
xconv1c = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge1)
xconv1c = bn()(xconv1c)
xmerge2=concatenate([xconv1a,xconv1b,xconv1c])
xconv1d = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge2)
xconv1d = bn()(xconv1d)
xpool1 = MaxPooling3D(pool_size=(2, 2, 1))(xconv1d)
xpool1 = Dropout(DropP)(xpool1)
flatten7=Flatten()(xpool1)
output7=Dense(1,activation='sigmoid')(flatten7)
#--------------------------------------------------------------------------------------------------

xconv2a = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xpool1)
xconv2a = bn()(xconv2a)
xconv2b = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xconv2a)
xconv2b = bn()(xconv2b)
xmerge1=concatenate([xconv2a,xconv2b])
xconv2c = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge1)
xconv2c = bn()(xconv2c)
xmerge2=concatenate([xconv2a,xconv2b,xconv2c])
xconv2d = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge2)
xconv2d = bn()(xconv2d)
xmerge3=concatenate([xconv2a,xconv2b,xconv2c,xconv2d])
xconv2e = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge3)
xconv2e = bn()(xconv2e)
xmerge4=concatenate([xconv2a,xconv2b,xconv2c,xconv2d,xconv2e])
xconv2f = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge4)
xconv2f = bn()(xconv2f)
xmerge5=concatenate([xconv2a,xconv2b,xconv2c,xconv2d,xconv2e,xconv2f])
xconv2g = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge5)
xconv2g = bn()(xconv2g)
xmerge6=concatenate([xconv2a,xconv2b,xconv2c,xconv2d,xconv2e,xconv2f,xconv2g])
xconv2h = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge6)
xconv2h = bn()(xconv2h)
xmerge7=concatenate([xconv2a,xconv2b,xconv2c,xconv2d,xconv2e,xconv2f,xconv2g,xconv2h])
xconv2i = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge7)
xconv2i = bn()(xconv2g)
xmerge8=concatenate([xconv2a,xconv2b,xconv2c,xconv2d,xconv2e,xconv2f,xconv2g,xconv2h,xconv2i])
xconv2j = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge8)
xconv2j = bn()(xconv2g)
xpool2 = MaxPooling3D(pool_size=(2, 2, 1))(xconv2j)
xpool2 = Dropout(DropP)(xpool2)
flatten8=Flatten()(xpool2)
output8=Dense(1,activation='sigmoid')(flatten8)
#--------------------------------------------------------------------------------------------------

xconv3a = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xpool2)
xconv3a = bn()(xconv3a)
xconv3b = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xconv3a)
xconv3b = bn()(xconv3b)
xmerge1=concatenate([xconv3a,xconv3b])
xconv3c = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge1)
xconv3c = bn()(xconv3c)
xmerge2=concatenate([xconv3a,xconv3b,xconv3c])
xconv3d = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge2)
xconv3d = bn()(xconv3d)
xmerge3=concatenate([xconv3a,xconv3b,xconv3c,xconv3d])
xconv3e = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge3)
xconv3e = bn()(xconv3e)
xmerge4=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e])
xconv3f = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge4)
xconv3f = bn()(xconv3f)
xmerge5=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f])
xconv3g = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge5)
xconv3g = bn()(xconv3g)
xmerge6=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g])
xconv3h = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge6)
xconv3h = bn()(xconv3h)
xmerge7=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h])
xconv3i = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge7)
xconv3i = bn()(xconv3i)
xmerge8=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i])
xconv3j = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge8)
xconv3j = bn()(xconv3j)
xmerge9=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j])
xconv3k = Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge9)
xconv3k = bn()(xconv3k)
xmerge10=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k])
xconv3l=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge10)
xconv3l = bn()(xconv3l)
xmerge11=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l])
xconv3m=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge11)
xconv3m = bn()(xconv3m)
xmerge12=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m])
xconv3n=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge12)
xconv3n = bn()(xconv3n)
xmerge13=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n])
xconv3o=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge13)
xconv3o = bn()(xconv3o)
xmerge14=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n,xconv3o])
xconv3p=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge14)
xconv3p = bn()(xconv3p)
xmerge15=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n,xconv3o,xconv3p])
xconv3q=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge15)
xconv3q = bn()(xconv3q)
xmerge16=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n,xconv3o,xconv3p,xconv3q])
xconv3r=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge16)
xconv3r = bn()(xconv3r)
xmerge17=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n,xconv3o,xconv3p,xconv3q,xconv3r])
xconv3s=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge17)
xconv3s = bn()(xconv3s)
xmerge18=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n,xconv3o,xconv3p,xconv3q,xconv3r,xconv3s])
xconv3t=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge18)
xconv3t = bn()(xconv3t)
xmerge19=concatenate([xconv3a,xconv3b,xconv3c,xconv3d,xconv3e,xconv3f,xconv3g,xconv3h,xconv3i,xconv3j,xconv3k,xconv3l,xconv3m,xconv3n,xconv3o,xconv3p,xconv3q,xconv3r,xconv3s,xconv3t])
xconv3u=Conv3D(12, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge19)
xconv3u = bn()(xconv3u)
xpool3 = MaxPooling3D(pool_size=(2, 2, 1))(xconv3u)
xpool3 = Dropout(DropP)(xpool3)
flatten9=Flatten()(xpool3)
output9=Dense(1,activation='sigmoid')(flatten9)
#--------------------------------------------------------------------------------------------------

xconv4a = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xpool3)
xconv4a = bn()(xconv4a)
xconv4b = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xconv4a)
xconv4b = bn()(xconv4b)
xmerge1=concatenate([xconv4a,xconv4b])
xconv4c = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge1)
xconv4c = bn()(xconv4c)
xmerge2=concatenate([xconv4a,xconv4b,xconv4c])
xconv4d = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge2)
xconv4d = bn()(xconv4d)
xmerge3=concatenate([xconv4a,xconv4b,xconv4c,xconv4d])
xconv4e = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge3)
xconv4e = bn()(xconv4e)
xmerge4=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e])
xconv4f = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge4)
xconv4f = bn()(xconv4f)
xmerge5=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f])
xconv4g = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge5)
xconv4g = bn()(xconv4g)
xmerge6=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g])
xconv4h = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge6)
xconv4h = bn()(xconv4h)
xmerge7=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h])
xconv4i = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge7)
xconv4i = bn()(xconv4i)
xmerge8=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i])
xconv4j = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge8)
xconv4j = bn()(xconv4j)
xmerge9=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j])
xconv4k = Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge9)
xconv4k = bn()(xconv4k)
xmerge10=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k])
xconv4l=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge10)
xconv4l = bn()(xconv4l)
xmerge11=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l])
xconv4m=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge11)
xconv4m = bn()(xconv4m)
xmerge12=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m])
xconv4n=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge12)
xconv4n = bn()(xconv4n)
xmerge13=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n])
xconv4o=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge13)
xconv4o = bn()(xconv4o)
xmerge14=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n,xconv4o])
xconv4p=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge14)
xconv4p = bn()(xconv4p)
xmerge15=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n,xconv4o,xconv4p])
xconv4q=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge15)
xconv4q = bn()(xconv4q)
xmerge16=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n,xconv4o,xconv4p,xconv4q])
xconv4r=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge16)
xconv4r = bn()(xconv4r)
xmerge17=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n,xconv4o,xconv4p,xconv4q,xconv4r])
xconv4s=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge17)
xconv4s = bn()(xconv4s)
xmerge18=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n,xconv4o,xconv4p,xconv4q,xconv4r,xconv4s])
xconv4t=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge18)
xconv4t = bn()(xconv4t)
xmerge19=concatenate([xconv4a,xconv4b,xconv4c,xconv4d,xconv4e,xconv4f,xconv4g,xconv4h,xconv4i,xconv4j,xconv4k,xconv4l,xconv4m,xconv4n,xconv4o,xconv4p,xconv4q,xconv4r,xconv4s,xconv4t])
xconv4u=Conv3D(24, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge19)
xconv4u = bn()(xconv4u)
xpool4 = MaxPooling3D(pool_size=(2, 2, 1))(xconv4u)
xpool4 = Dropout(DropP)(xpool4)
flatten10=Flatten()(xpool4)
output10=Dense(1,activation='sigmoid')(flatten10)
#--------------------------------------------------------------------------------------------------

xconv5a = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xpool4)
xconv5a = bn()(xconv5a)
xconv5b = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xconv5a)
xconv5b = bn()(xconv5b)
xmerge1=concatenate([xconv5a,xconv5b])
xconv5c = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge1)
xconv5c = bn()(xconv5c)
xmerge2=concatenate([xconv5a,xconv5b,xconv5c])
xconv5d = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge2)
xconv5d = bn()(xconv5d)
xmerge3=concatenate([xconv5a,xconv5b,xconv5c,xconv5d])
xconv5e = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge3)
xconv5e = bn()(xconv5e)
xmerge4=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e])
xconv5f = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge4)
xconv5f = bn()(xconv5f)
xmerge5=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f])
xconv5g = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge5)
xconv5g = bn()(xconv5g)
xmerge6=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g])
xconv5h = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge6)
xconv5h = bn()(xconv5h)
xmerge7=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h])
xconv5i = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge7)
xconv5i = bn()(xconv5i)
xmerge8=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i])
xconv5j = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge8)
xconv5j = bn()(xconv5j)
xmerge9=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j])
xconv5k = Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge9)
xconv5k = bn()(xconv5k)
xmerge10=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k])
xconv5l=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge10)
xconv5l = bn()(xconv5l)
xmerge11=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l])
xconv5m=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge11)
xconv5m = bn()(xconv5m)
xmerge12=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m])
xconv5n=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge12)
xconv5n = bn()(xconv5n)
xmerge13=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n])
xconv5o=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge13)
xconv5o = bn()(xconv5o)
xmerge14=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n,xconv5o])
xconv5p=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge14)
xconv5p = bn()(xconv5p)
xmerge15=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n,xconv5o,xconv5p])
xconv5q=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge15)
xconv5q = bn()(xconv5q)
xmerge16=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n,xconv5o,xconv5p,xconv5q])
xconv5r=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge16)
xconv5r = bn()(xconv5r)
xmerge17=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n,xconv5o,xconv5p,xconv5q,xconv5r])
xconv5s=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge17)
xconv5s = bn()(xconv5s)
xmerge18=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n,xconv5o,xconv5p,xconv5q,xconv5r,xconv5s])
xconv5t=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge18)
xconv5t = bn()(xconv5t)
xmerge19=concatenate([xconv5a,xconv5b,xconv5c,xconv5d,xconv5e,xconv5f,xconv5g,xconv5h,xconv5i,xconv5j,xconv5k,xconv5l,xconv5m,xconv5n,xconv5o,xconv5p,xconv5q,xconv5r,xconv5s,xconv5t])
xconv5u=Conv3D(48, (kernel_size, kernel_size, kernel_size), activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(l2_lambda) )(xmerge19)
xconv5u = bn()(xconv5u)

#End of second branch
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# The output of the two individual branches are merged together
merge_both_blocks=(merge([xconv5u,conv5u],mode='concat'))
flatten3=Flatten()(merge_both_blocks)

#All the penultimate layer before the intermediate outputs are merged together
finalmerge=(merge([flatten1,flatten2,flatten3,flatten4,flatten5,flatten7,flatten8,flatten9,flatten10],mode='concat'))
#We take the output of this merged layers of the intermediate outputs
output=Dense(1,activation='sigmoid',name='output')(finalmerge)
#We define our model with the inputs as mentioned before
#We can obtain the individual result of the intermediate outputs using output1,2,3,.....10 and the final consolidated output from the final output branch
#It is interesting to see how the different sections of the network optimize over time
finalmodel = Model(inputs=[inputthree,inputthree1], outputs=[output1,output2,output4,output5,output7,output8,output9,output10,output])#This line assigns the model name to the model
#The name "finalmodel" is the name to refer to this network while training/testing etc
finalmodel.compile(optimizer='adadelta',loss='binary_crossentropy',metrics=['accuracy'])
finalmodel.summary()



#Thank You and please contact me at rd31879@uga.edu if you need more information.