import matplotlib.pyplot as plt
import random
import os, sys
import numpy as np # linear algebra
import pandas as pd # data processing
import cv2
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras.backend as K
from keras.applications import Xception
from keras.layers import UpSampling2D, Conv2D, Activation, LeakyReLU, BatchNormalization
from keras import Model
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def mask2rle(img):
	tmp = np.rot90( np.flipud( img ), k=3 )
	rle = []
	lastColor = 0;
	startpos = 0
	endpos = 0

	tmp = tmp.reshape(-1,1)   
	for i in range( len(tmp) ):
		if (lastColor==0) and tmp[i]>0:
			startpos = i
			lastColor = 1
		elif (lastColor==1) and (tmp[i]==0):
			endpos = i-1
			lastColor = 0
			rle.append( str(startpos)+' '+str(endpos-startpos+1) )
	return " ".join(rle)

predict = np.load('predict.npy')
pred_rle = []
for i in range(predict.shape[0]):
# for i in range(10):
	print('Doing prediction for {}-th image'.format(i))
	img = predict[i] 
	img = cv2.resize(img, (1600, 256))
	tmp = np.copy(img)
	tmp[tmp<0.8] = 0
	tmp[tmp>0] = 1
	pred_rle.append(mask2rle(tmp))

pred_rle_4 = []
for _ in pred_rle:
	pred_rle_4.extend([_, _, _, _])

sub = pd.read_csv('sample_submission.csv')
# import pdb; pdb.set_trace()
sub['EncodedPixels'] = pred_rle_4
sub.to_csv('submission.csv', index=False)
sub.head(10)