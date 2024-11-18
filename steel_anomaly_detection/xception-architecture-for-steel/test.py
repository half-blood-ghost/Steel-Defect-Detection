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

tf.enable_eager_execution()

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

def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud(np.rot90( mask.reshape(height, width), k=1))

def rle2mask_eda(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def dice_coef(y_true, y_pred, smooth=1e-9):
    y_true_f = y_true.reshape(-1,1)
    y_pred_f = y_pred.reshape(-1,1)
    intersection = np.sum( y_true_f * y_pred_f )
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth) 

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 7

def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric


def iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.8], tf.float64)

def get_gt_from_csv(csv_data, i):
	import math
	gt = np.zeros((1600, 256))
	for cls_id in ['1', '2', '3', '4']:
		if pd.isnull(csv_data[cls_id].iloc[i]):
			continue
		else:
			print('get non-zeros gt!')
			# import pdb; pdb.set_trace()
			gt = rle2mask(csv_data[cls_id].iloc[i], imgshape=(1600,256))
	return gt


base_model = Xception(weights=None, input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False)
# base_model.load_weights('pretrain_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

base_out = base_model.output
up1 = UpSampling2D(32, interpolation='bilinear')(base_out)
conv1 = Conv2D(1, (1, 1))(up1)
conv1 = Activation('sigmoid')(conv1)
#bn1 = BatchNormalization()(conv1)
# re2 = LeakyReLU(0.2)(bn1)
# up2 = UpSampling2D(16, interpolation='bilinear')(re2)
# conv2 = Conv2D(1, (1, 1))(up2)
# conv2 = Activation('sigmoid')(conv2)

model = Model(base_model.input, conv1)
model.load_weights('./models/weights_05.hdf5')
print('model loading finished!')

# test begins
BATCH_SIZE = 64
test_img = []
testfiles=pd.read_csv('val_1.csv')['ImageId']
# import pdb; pdb.set_trace()
for fn in testfiles:
        img = cv2.imread( 'train_images/'+fn )
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))       
        test_img.append(img)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(
    np.asarray(test_img),
    batch_size=BATCH_SIZE
)

testfiles=os.listdir("test_images/")
nb_samples = len(testfiles)
predict = model.predict_generator(test_generator, steps = nb_samples / BATCH_SIZE, verbose=1)
np.save('predict_shit.npy', predict)
# predict = np.load('predict_shit.npy')

test_data = pd.read_csv('val_1.csv')

pred_rle = []
dice_scores = []
for i in range(predict.shape[0]):
    print('doint prediction for {}-th element'.format(i))
    # import pdb; pdb.set_trace()
    img = predict[i]
    img = cv2.resize(img, (1600, 256))
    tmp = np.copy(img)
    #tmp[tmp<np.mean(img)] = 0
    tmp[tmp<0.8] = 0
    tmp[tmp>0] = 1
    # pred_rle.append(mask2rle(tmp))

    gt = get_gt_from_csv(test_data, i)

    dice = dice_coef(gt, tmp)
    dice_scores.append(dice)

print('avg dice scores: ', sum(dice_scores) / len(dice_scores))



