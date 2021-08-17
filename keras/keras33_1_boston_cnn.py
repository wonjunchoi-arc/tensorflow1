from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_boston
datasets= load_boston()
x= datasets.data
y = datasets.target

print(len(np.unique(y)))

# print(x.shape) #(506, 13)
# print(y.shape) #(506,)
y = y.reshape(506,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.8)

print(x_train.shape)


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer,PowerTransformer
scaler =StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.reshape(202,2)
y_test = y_test.reshape(51,2)

# from sklearn.preprocessing import OneHotEncoder
# en = OneHotEncoder(sparse=False)
# y_train = en.fit_transform(y_train)
# y_test = en.fit_transform(y_test)

print(y_train.shape)
print(y_test.shape)


# print(x_train)
x_train = x_train.reshape(202, 13, 2, 1)
x_test = x_test.reshape(51,13,2,1)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(30, kernel_size=(2,2), padding='same', input_shape=(13, 2, 1)))
model.add(Conv2D(30, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='relu'))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience= 3, mode= 'max', verbose=3)

import time
start_time = time.time()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
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













# #3 . 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam', metrics=['acc'])
# model.fit(x_train,y_train, epochs=100, batch_size=15, verbose=3,validation_split=0.4)

# #4. 평가 예측
# loss = model.evaluate(x_test, y_test)
# print('loss', loss[0])
# print('acc', loss[1])


# y_predict = model.predict(x_test)
# print('y_predict', y_predict)


# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict)
# print('r2스코어',r2)




'''
cnn
loss 599.3861694335938
acc 0.4901960790157318

'''




# # print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
# # print(datasets.DESCR)

# #loss 출력 v , R2출력 v, 컬럼 어떻게 생격는지도 확인해보자