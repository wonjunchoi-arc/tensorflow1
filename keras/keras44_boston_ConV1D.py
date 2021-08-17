
from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,Conv1D,Flatten
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_boston
datasets= load_boston()
x= datasets.data
y = datasets.target


print(np.min(x), np.max(x)) #0.0 711.0


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.7, random_state=66)


from sklearn.preprocessing import PowerTransformer
 
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#이는 test 데이터는 train데이터에 관여하면 안된다는거

# print(np.min(x_scale), np.max(x_scale))



# print(x.shape) #(506, 13)
# print(y.shape) #(506,)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 



#2. 모델 구성

model = Sequential()

# model.add(LSTM(64, input_shape=(5,1)))
model.add(Conv1D(64, 2, input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

# model = Sequential()
# # model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(LSTM(units=64, activation='relu', input_shape=(x_train.shape[1],1)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(4, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(8, activation='relu'))

#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience= 100, mode= 'max', verbose=3)

import time
start_time = time.time()

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, epochs=100,
batch_size=30, validation_split=0.3,verbose=1,callbacks=[es])
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print("걸린시간 : ", end_time)


y_predict = model.predict(x_test)
print('y_predict', )



from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)

''' 
cnn
loss 599.3861694335938
acc 0.4901960790157318

LSTM loss 12.117149353027344
y_predict
r2스코어 0.8533336279760748

loss 15.971026420593262
r2스코어 0.806686193660181


'''

plt.scatter(x[:,6],y)
plt.show()

# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
# print(datasets.DESCR)

#loss 출력 v , R2출력 v, 컬럼 어떻게 생격는지도 확인해보자