

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, LSTM,Conv1D
from tensorflow.python.keras.datasets.mnist import load_data


x_train = np.load('./_save/_npy/x_train_cifar10.npy')
y_train = np.load('./_save/_npy/y_train_cifar10.npy')
x_test = np.load('./_save/_npy/x_test_cifar10.npy')
y_test = np.load('./_save/_npy/y_test_cifar10.npy')


print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)



from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

# print(x_train[:1])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(y_train.shape,y_test.shape)

x_train = x_train.reshape(x_train.shape[0], 32*32, 3)
x_test = x_test.reshape(x_test.shape[0], 32*32, 3) 

#2. 모델 구성
# model = Sequential()
# # model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(LSTM(units=10, activation='relu', input_shape=(32*32,3)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(4, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model = Sequential()
# model.add(Conv1D(64, 2, input_shape=(32*32,3)))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(10, activation="softmax"))






#3. 컴파일, 훈련 , metrics=['acc']
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience= 3, mode= 'min', verbose=3)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto' ,
#                             filepath='./_save/ModelCheckPoint/keras48_8_MCP.hdf5')


import time
start_time = time.time()

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# hist = model.fit(x_train,y_train, epochs=10,
# batch_size=150, validation_split=0.3,verbose=1,callbacks=[es,cp])
end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_7_model.h5')

from tensorflow.keras.models import load_model

model = load_model('./_save/ModelCheckPoint/keras48_8_MCP.hdf5')

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print("걸린시간 : ", end_time)

