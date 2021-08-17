import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,LSTM

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
# print(x_test.shape, y_test.shape)# (10000, 28, 28) (10000,)

x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)
## 스케일러에는 2차원의 데이터가 들어와야 한다. 때문에 스케일리 전 데이터를 2차원의 데이터로 reshape 해준 것

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)
## ONEHOTENCODER 또한 2차원의 데이터를 요구하므로 2차원의 행렬로 전환해준것

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False) # sparse의 default는 true로 matrix행렬로 반환한다. 하지만 False는 array로 반환 둘의 차이는 잘..
y_train = en.fit_transform(y_train)
y_test = en.fit_transform(y_test)
print(y_test)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler =QuantileTransformer()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)







x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(LSTM(units=64, activation='relu', input_shape=(x_train.shape[1],1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='softmax'))


# #3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 10, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10,
 batch_size=1000, validation_split=0.2,verbose=3)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])


# #0.98 이상
# print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]

'''
# vali가 0.2일때 낮을 수록 높아 보이는 경향이 있으며 

# 모델을 많이 추가해줄 수록 좋아 보인다.
# model.add(Conv2D(50, (2,2), activation='relu'))
# model.add(MaxPool2D())   
# model.add(Conv2D(70, (2,2), activation='relu'))
# model.add(Conv2D(80, (2,2), activation='relu'))
# model.add(Conv2D(900, (2,2), activation='relu'))

# Epoch 100/100
# 313/313 [==============================] - 2s 5ms/step - loss: 0.0559 - accuracy: 0.9932  
# loss 0.05588952451944351
# accuracy 0.9932000041007996



loss 2.3009793758392334
accuracy 0.11349999904632568
'''