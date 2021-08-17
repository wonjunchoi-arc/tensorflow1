from re import X
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,GlobalAveragePooling1D,Dropout,LSTM, Flatten



datasets = pd.read_csv('../data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)
#데이터가 ; 으로 분리 되어 있어서 sep 세퍼레이트 를 해준다..
# 데이터가 a열에 있었으므로 인덱스 컬럼은 없고 맨위의 행이 head기 때문에 0을 줌 

# winequality-white
# ./ : 현재 폴더
# ../ : 상위폴더 

# print(datasets)
print(datasets.shape) #(4898, 12)
print(datasets.info())
print(datasets.describe())

#1. 데이터 처리
datasets=np.array(datasets.values)
# print(datasets.shape)

x= datasets[:,:-1]
y= datasets[:,-1:]
print(x.shape, y.shape)

print(np.unique(y))

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y = en.fit_transform(y).toarray()

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

print(y.shape)
print(y[:5])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.1, random_state=66)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
# scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler =QuantileTransformer()
scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)




x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 

#2. 모델 구성
# model = Sequential()
# # model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(LSTM(units=64, activation='relu', input_shape=(x_train.shape[1],1)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(4, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# model.add(Dense(7, activation="softmax"))

# model.add(LSTM(64, input_shape=(5,1)))
model = Sequential()
model.add(Conv1D(64, 2, input_shape=(x_test.shape[1],1)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation="softmax"))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience= 30, mode= 'max', verbose=3)

import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=100,
batch_size=30, validation_split=0.3,verbose=1,callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print("걸린시간 : ", end_time)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2) # 2개를 하고 1행 2열
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()


'''
1. CNN
loss 1.2466771602630615
accuracy 0.6102041006088257

2.dnn
loss 3.9895122051239014
accuracy 0.6265305876731873

LSTM
loss 1.357404112815857
accuracy 0.40816327929496765

ConV1D
loss 1.1142525672912598
accuracy 0.5265306234359741
걸린시간 :  33.60899996757507

'''