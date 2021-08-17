import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
[5,6,7],[6,7,8],[7,8,9],[8,9,10],
[9,10,11],[10,11,12],
[20,30,40],[30,40,50,],[40,50,60,]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predict = np.array([50,60,70])


print(x.shape, y.shape)   # (4, 3) (4,)

x = x.reshape(x.shape[0], x.shape[1], 1)     #(batch_size, timesteps, features)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(SimpleRNN(units=64, activation='relu', input_shape=(3,1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='mse', patience= 100, mode= 'min', verbose=3)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x,y, epochs=300, batch_size=1,callbacks=[es])


#4. 평가, 예측
x_input = x_predict.reshape(1,3,1)
results = model.predict(x_input)
print(results)

#5 그리기

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

plt.plot(hist.history["loss"])
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss'])
plt.show()


# 결과값이 80 근접하게

'''
LSTM
[[80.44758]]
[[82.50793]]
[[79.255585]]
79.67383

GRU
[[82.9769]]
[[82.70521]]
'''


'''
480
output*output*4 +(input+bias*output*4) 
1.0068543
'''