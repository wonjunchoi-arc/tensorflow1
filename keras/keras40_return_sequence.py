import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU,Dropout


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
model.add(LSTM(units=64, activation='relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(units=16, activation='relu'))
#LSTM을 통과해서 나온 데이터가 연속적일 것이란 확신을 할 수 없기 때문에 LSTM을 두개를 연결시켜서는 잘 안쓴다. 
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='mse', patience= 20, mode= 'min', verbose=3)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit(x,y, epochs=300, batch_size=1,callbacks=[es])


#4. 평가, 예측
x_input = x_predict.reshape(1,3,1)
results = model.predict(x_input)
print(results)



'''

________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 10)                480
_________________________________________________________________
dense (Dense)                (None, 5)                 55
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
=================================================================

비교
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 10)             480
_________________________________________________________________
lstm_1 (LSTM)                (None, 7)                 504
_________________________________________________________________
dense (Dense)                (None, 5)                 40
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6
=================================================================
Total params: 1,030
Trainable params: 1,030
Non-trainable params: 0
__________________________
'''


'''
480
output*output*4 +(input+bias*output*4) 
1.0068543
'''