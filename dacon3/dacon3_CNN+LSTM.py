import pandas as pd
import numpy as np
import cv2
from tensorflow.python.keras.engine.training import concat
from tensorflow.python.keras.layers.core import Dropout


train = pd.read_csv('../data/dacon3/train.csv',index_col=None,header=0)
test = pd.read_csv('../data/dacon3/test.csv',index_col=None,header=0)
dev = pd.read_csv('../data/dacon3/dev.csv',index_col=None,header=0)
submission = pd.read_csv('../data/dacon3/sample_submission.csv',index_col=None,header=0)



train= pd.concat([train,dev])

x = train['SMILES']
x_test = test['SMILES']
y1 = train['S1_energy(eV)']
y2 = train['T1_energy(eV)']
y = y1 - y2
print(y)

############데이터 가져오기########################

import codecs
from SmilesPE.tokenizer import *
from tensorflow.keras.preprocessing.text import Tokenizer


spe_vob= codecs.open('../data/dacon3/SPE_ChEMBL.txt')
spe = SPE_Tokenizer(spe_vob)
token = Tokenizer()




smiles_tok= []
for i in x : 
    smiles_tok.append(spe.tokenize(i))

token.fit_on_texts(smiles_tok)

# test_tok= []
# for i in x_test : 
#     smiles_tok.append(spe.tokenize(i))

# token.fit_on_texts(smiles_tok)
# token.fit_on_texts(test_tok)


seq = token.texts_to_sequences(smiles_tok)
# test_seq = token.texts_to_sequences(test_tok)
# print(seq)

print("Smiles 최대길이:" ,max(len(i) for i in seq)) #102
print("Smiles 평균길이:", sum(map(len, seq)) /len(seq)) # 13
print("단어의 개수는:" ,max(seq)) #13

# # # 2. 길이 맞춰주기

from tensorflow.keras.preprocessing.sequence import pad_sequences

seq = pad_sequences(seq, maxlen=102, padding='pre',)


print(seq.shape) 

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


x_smile_train = seq[:27000]
x_smile_val = seq[27000:]
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

from tensorflow.keras.layers import Dense,Flatten,GlobalAveragePooling2D,Permute,Reshape,Input,Embedding,LSTM,Bidirectional,Concatenate
from tensorflow.keras.models import Sequential,Model
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB7,ResNet50V2,ResNet50,Xception

embedding_dim = 256
rate = 0.1

#2. 모델 구성
efficientNetB0 =Xception(weights='imagenet',
include_top=False, input_shape=(128,128,3))
 
efficientNetB0.trainable=True
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감


input1 = Input(shape=(128,128,3))
x=efficientNetB0(input1)
print(x.shape)
#(None, 4, 4, 2048)
x1=Permute((2,3,1))(x)
print(x1.shape)
# (None, 4, 2048, 4)
x2=Reshape((-1,x1.shape[3]))(x1)
print(x2.shape)
# (None, 8192, 4)
x3 =Flatten()(x2)
# print(x3.shape)
x4=Dropout(0.1)(x3)
x5=Dense(2048, activation='relu')(x4)
output=Dense(512, activation='relu')(x5)
output1=Reshape((4,128))(output)
print('너는 또 뭐야 시벙',output1)
#shape=(None, 8192, 256)

input2 = Input(shape=(102,))
x=Embedding(input_dim = 2000, output_dim=128, input_length=102)(input2)
output2=Dropout(0.1)(x)
print('여기 뭐가나오나 보자',output2)
#(shape=(None, 102, 256)


from tensorflow.keras.layers import concatenate, Concatenate,LSTM,Reshape
# 소문자는 메소드, 대문자는 클래스를 불러오는 것이다. 그러나 둘중 하나는 예전에 쓰던 것일 수 도 있기에 새로운 기능을 사용하지 못할 수 도 있다. 

merge1 = Concatenate(axis=1)([output1, output2])
print(merge1) #shape=(None, 8294, 256)
merge2 = Bidirectional(LSTM(256,return_sequences=True))(merge1)
merge3= Dropout(0.2)(merge2)
merge4 = Bidirectional(LSTM(128))(merge3)
print('LSTM의 모양',merge2.shape)
merge5= Dropout(0.2)(merge4)
merge6= Dense(20000, activation='relu')(merge5)
last_output= Dense(1, activation='relu')(merge6)

model = Model(inputs=[input1, input2], outputs=last_output)


#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='mae', patience=10, mode="auto", verbose=1)

print(y_train.shape)
print(img_train.shape)
print(x_smile_train.shape)

import time
start_time = time.time()
optimizer =tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mae', optimizer=optimizer,) 
model.fit([img_train,x_smile_train],y_train, epochs=20, batch_size=56,validation_data=([img_val,x_smile_val],y_val),
#validation_data=([x_test, y_test], [y_test, x_test]), 벨리데이션 데이터 오류 튜플로 !!
verbose=1,callbacks=[es])
model.summary()
end_time = time.time() - start_time


y_predict = model.predict(x_test)
submission['ST1_GAP(eV)'] = y_predict
submission.to_csv('/content/drive/MyDrive/dacon3/dacon_baseline.csv', index=False)


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
