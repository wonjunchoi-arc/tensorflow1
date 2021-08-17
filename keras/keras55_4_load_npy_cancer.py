

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,GlobalAveragePooling2D,LSTM, Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer



#1. 데이터

x = np.load('./_save/_npy/k55_x_data_cancer.npy')
y = np.load('./_save/_npy/k55_y_data_cancer.npy')


print(x.shape , y.shape)



print(y[:20])
print(np.unique(y))
#unique 는 특이한 애들을 찾는다. [0 1] 밖에 없다.  인진 분류 모델 

x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.7, random_state=66)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)

y_train = y_train.reshape(398,1)
y_test = y_test.reshape(171,1)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 


from tensorflow.keras.models import load_model
#2. 모델 구성
# model = Sequential()
# # model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(LSTM(units=64, activation='relu', input_shape=(x_train.shape[1],1)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(4, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model = Sequential()

# model.add(LSTM(64, input_shape=(5,1)))
# model.add(Conv1D(64, 2, input_shape=(x_test.shape[1],1)))
# model.add(Flatten())
# model.add(Dense(10))
# model.add(Dense(1))



#3. 컴파일 및 훈련

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience= 100, mode= 'min', verbose=1)
# cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto',
#                             filepath='./_save/ModelCheckPoint/keras48_2_MCP.hdf5')

model = load_model('./_save/ModelCheckPoint/keras48_2_MCP.hdf5')

import time 
start_time =time.time()

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=100,
#  batch_size=8, validation_split=0.2, callbacks=[es,cp])


end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_2_model.h5')

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])
print('======================================')
y_predict = model.predict(x_test[-5:-1])
print('y_predict', y_predict)
print(y_test[-5:-1])

