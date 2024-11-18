import matplotlib.pyplot as plt
import random
import os, sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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

df_train = pd.read_csv('train.csv')
df_train = df_train[df_train['EncodedPixels'].notnull()].reset_index(drop=True)

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 7

def keras_generator(batch_size):
    while True:
        x_batch = []
        y_batch = []
        
        for i in range(batch_size):            
            fn = df_train['ImageId_ClassId'].iloc[i].split('_')[0]
            img = cv2.imread( 'train_images/'+fn )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            
            
            mask = rle2mask(df_train['EncodedPixels'].iloc[i], img.shape)
            
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
            
            x_batch += [img]
            y_batch += [mask]
                                    
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)

        yield x_batch, np.expand_dims(y_batch, -1)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred = K.cast(y_pred, 'float64')
    y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.8), 'float64')
    intersection = y_true_f * y_pred_f
    score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return score

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


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

def get_callback():
    ES = EarlyStopping(
        monitor='loss',
        patience=5,
        mode='max', 
        verbose=1)
    RR = ReduceLROnPlateau(
        monitor = 'loss', 
        factor = 0.5,
        patience=3,
        min_lr=0.000001,
        verbose=1,
        mode='max')
    CP = ModelCheckpoint('models/weights_new_{epoch:02d}.hdf5', monitor='loss', save_best_only=False, verbose=1, mode='max')
    return [ES, RR, CP]

base_model = Xception(weights=None, input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False)
base_model.load_weights('pretrain_weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

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
model.compile(loss=bce_dice_loss, optimizer='adam', metrics=[iou_metric])

history = model.fit_generator(keras_generator(BATCH_SIZE),
                              steps_per_epoch=len(df_train.index) // BATCH_SIZE,
                              # steps_per_epoch=2,              
                              epochs=EPOCHS,                    
                              verbose=1,
                              shuffle=True,
                              callbacks=get_callback(),
)

plt.figure(figsize=(7, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(history.history['loss']) 
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train'], loc='upper left') 
plt.savefig('plots/loss.png')

plt.figure(figsize=(7, 7), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(history.history['iou_metric']) 
plt.title('model iou_metric')  
plt.ylabel('iou_metric')  
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('plots/train_iou.png')

# test begins
BATCH_SIZE = 16
test_img = []
testfiles=os.listdir("test_images/")
for fn in testfiles:
        img = cv2.imread( 'test_images/'+fn )
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))       
        test_img.append(img)

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow(
    np.asarray(test_img),
    batch_size=BATCH_SIZE
)

testfiles=os.listdir("test_images/")
nb_samples = len(testfiles)
predict = model.predict_generator(test_generator, steps = nb_samples / BATCH_SIZE)

pred_rle = []
for img in predict:      
    img = cv2.resize(img, (1600, 256))
    tmp = np.copy(img)
    #tmp[tmp<np.mean(img)] = 0
    tmp[tmp<0.8] = 0
    tmp[tmp>0] = 1
    pred_rle.append(mask2rle(tmp))

pred_rle_4 = []
for _ in pred_rle:
    pred_rle_4.extend([_, _, _, _])

sub = pd.read_csv('sample_submission.csv')
import pdb; pdb.set_trace()
sub['EncodedPixels'] = pred_rle_4
sub.to_csv('submission.csv', index=False)
sub.head(10)