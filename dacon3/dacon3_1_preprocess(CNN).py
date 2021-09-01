import pandas as pd
import numpy as np
import cv2
from tensorflow.python.keras.layers.core import Dropout


train = pd.read_csv('../data/dacon3/train.csv',index_col=None,header=0)
test = pd.read_csv('../data/dacon3/test.csv',index_col=None,header=0)
dev = pd.read_csv('../data/dacon3/dev.csv',index_col=None,header=0)


train= pd.concat([train,dev])

x = train['SMILES']
y1 = train['S1_energy(eV)']
y2 = train['T1_energy(eV)']
y = y1 - y2
print(y)

############데이터 가져오기########################
print(x)

import codecs
from SmilesPE.tokenizer import *

spe_vob= codecs.open('../data/dacon3/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)


smiles_tok= []
for i in x : 
    smiles_tok.append(spe.tokenize(i))

print(len(smiles_tok))

'''
TypeError: 'Series' objects are mutable, thus they cannot be hashed
nan값이 있다는 건가??
'''
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# test_datagen = ImageDataGenerator(rescale=1./255)

# full = test_datagen.flow_from_directory(
#     '../data/dacon3/imgs',
#     target_size=(128,128),
#     batch_size=30345,

# )

# print(full[0][0])
# print(full[0][1])
# #{'paper,': 0, 'rock': 1, 'scissors': 2}

# np.save('./_save/_npy/image/dacon3_x.npy', arr=full[0][0])
######################이미지 가져오기##################
import PIL.Image as pilimg
import numpy as np
import matplotlib.pyplot as plt 

imgs = np.load('./_save/_npy/image/dacon3_x.npy')
print(imgs.shape)

img = imgs.reshape(30345,128,128,3).astype('float')
print(type(y))

y= np.array(y)
y =y.reshape(y.shape[0],1).astype('float')
print(type(img))
print(type(y))
print(y.shape)


x_smile_train = x[:27000]
x_smile_val = x[27000:]
y_train = y[:27000]
y_val = y[27000:]
img_train =img[:27000]
img_val =img[27000:]
# print(img_train)
# print(img_train.shape)
####################이미지 잘 가져왔나 확인#################333
import matplotlib.image as img 
# plt.imshow(img_train[0]) 
# plt.show()


#################모델 구성#########################

from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB7,ResNet50V2,ResNet50,Xception


#2. 모델 구성
efficientNetB0 =Xception(weights='imagenet',
include_top=False, input_shape=(128,128,3))
 
efficientNetB0.trainable=True
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감

# model.summary()

model = Sequential()
model.add(efficientNetB0)
model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.1))
# model.add(GlobalAveragePooling2D())
model.add(Dense(256,))
# model.add(Dropout(0.2))

# model.trainable=False

model.summary()


#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='mae', patience=10, mode="auto", verbose=1)


import time
start_time = time.time()

model.compile(loss='mae', optimizer='adam',) 
model.fit(img_train,y_train, epochs=20, batch_size=78,validation_data=(img_val,y_val),
#validation_data=([x_test, y_test], [y_test, x_test]), 벨리데이션 데이터 오류 튜플로 !!
verbose=1,callbacks=[es])

end_time = time.time() - start_time

'''
xeptio = 0.24

Epoch 20/20
347/347 [==============================] - 104s 300ms/step - loss: 0.1309 - val_loss: 0.3221
WARNING:tensorflow:Early stopping conditioned on metric `mae` which is not available. Available metrics are: loss,val_loss
과적합 드랍아웃 필요

0.2 드랍에서 
val = 0.2890

0.1드랍
 0.2911

ResNet50 = 0.2762
'''
