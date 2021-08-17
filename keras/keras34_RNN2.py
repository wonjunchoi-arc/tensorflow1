import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN


#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
y = np.array([4,5,6,7])


print(x.shape, y.shape)   # (4, 3) (4,)

x = x.reshape(4, 3, 1)     #(batch_size, timesteps, features)


#2. 모델구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(SimpleRNN(units=10, activation='relu',input_length=3, input_dim=1)) #timesteps, feature
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1, batch_size=1)


#4. 평가, 예측
x_input = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(x_input)
print(results)

'''
8.090712
'''