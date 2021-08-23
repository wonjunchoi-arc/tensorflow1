# overfit 극복!!!
#1.  전체 훈련데이터를 마니 마니
#2.  normaliztion 정규화 시킨다. standardiztion과는 다름, minmax와 유사할지도?
# 여기서의 정규화는 레이어별로 해주는 것이 어떨가라는 접근 개념
#3.  dropout 완벽하게 fully connected 됫을때보다  node사이의 신경망을 몇개씩
#탈락 시켜준것이 오히려 더 효율적일 때가 있었다!!  잘라줄때도 일정하게 잘라주면 다시 그것에 과적합 될 수 있으므로
#랜덤하게 잘라 준다는 개념

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,Dropout,LSTM
from tensorflow.python.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#원핫 인코더할때 얘를 써도 괜찮다. 얘가 좀더 형식에서 자유롭기 때문이다. 
# example cifar100
# make perfect model in DNN

from tensorflow.keras.datasets import cifar100

import numpy as np
import matplotlib.pyplot as plt

# 1. data
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

x_train = x_train.reshape(50000, 32*32*3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32*3) # (10000, 32, 32, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
# scaler.fit(x_train) 
# x_train = scaler.transform(x_train) 
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

x_train = x_train.reshape(50000, 32*32, 3) # (50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32*32, 3) # (10000, 32, 32, 3)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

# 2. model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, GlobalAveragePooling1D, Dropout

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, padding='same',                        
                        activation='relu' ,input_shape=(32*32, 3))) 
model.add(Conv1D(32, 2, padding='same', activation='relu'))                   
model.add(MaxPool1D())                                         
model.add(Conv1D(64, 2, padding='same', activation='relu'))                   
model.add(Conv1D(64, 2, padding='same', activation='relu'))    
model.add(Flatten())                                              
model.add(Dense(256, activation='relu'))
model.add(Dense(124, activation='relu'))
# model.add(Dense(84, activation='relu'))
model.add(Dense(100, activation='softmax'))

# 3. comple fit // metrics 'acc'
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)

import time

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
    validation_split=0.05, callbacks=[es])
end_time = time.time() - start_time

# 4. predict eval -> no need to

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

loss = model.evaluate(x_test, y_test)
print('acc : ',acc[-10])
print('val_acc : ',val_acc[-10])
# print('loss : ',loss[-10])
print('val_loss : ',val_loss[-10])

'''
CNN Conv2D
loss :  4.9466118812561035
acc :  0.4169999957084656
CNN Conv1D
time :  76.25080919265747
loss :  8.077665328979492
acc :  0.27320000529289246
DNN
time =  35.59278154373169
loss :  3.2117340564727783
acc :  0.257999986410141
LSTM
time :  1009.927225112915
loss :  4.605213642120361
acc :  0.009999999776482582
'''
