import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Reshape
from tensorflow.python.keras.layers.convolutional import Conv1D

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)# (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False) # sparse의 default는 true로 matrix행렬로 반환한다. 하지만 False는 array로 반환 둘의 차이는 잘..
y_train = en.fit_transform(y_train)
y_test = en.fit_transform(y_test)
print(y_test)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(x_train.shape[0],28*28, 1)
x_test = x_test.reshape(x_test.shape[0], 28*28, 1) 


# # 모델 
# model =Sequential()
# model.add(Conv1D(20,2, input_shape=(28*28,1)))
# model.add(Flatten())    #(None, 280)                
# model.add(Dense(32,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(10, activation='softmax'))


# model.summary()


# # #3. 컴파일, 훈련 , metrics=['acc']
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience= 100, mode= 'min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto' ,
#                          filepath='./_save/ModelCheckPoint/keras48_6_MCP.hdf5')

import time
start_time = time.time()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs=100,
#  batch_size=300, validation_split=0.2,verbose=3,callbacks=[es,cp])
end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_6_model.h5')

from tensorflow.keras.models import load_model
model = load_model('./_save/ModelCheckPoint/keras48_6_MCP.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])

'''
1.
loss 0.44969457387924194
accuracy 0.958299994468689

MCP
loss 0.36109647154808044
accuracy 0.9564999938011169



'''