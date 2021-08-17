import numpy as np
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

a = np.array(range(1,101))
size = 6

print(len(a))

def split_x(dataset, size):
    aaa =[]
    for i in range(len(dataset) - size+2 ):
        subset = dataset[i : (i+size)]   # 5 6 7 8 9 10 1
        aaa.append(subset)
    return np.array(aaa)

dataset = split_x(a, size)



print(dataset)

x = dataset[: , :4]

y = dataset[:,4]

# print('x : ',x)
# print('y : ',y)


# #2. 모델구성
# model = Sequential()
# # model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
# model.add(LSTM(units=64, activation='relu', input_shape=(3,1)))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(4, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# # model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# model.summary()

# #3. 컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='mse', patience= 20, mode= 'min', verbose=3)

# model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# hist = model.fit(x,y, epochs=300, batch_size=1,callbacks=[es])


# #4. 평가, 예측
# x_input = x_predict.reshape(1,3,1)
# results = model.predict(x_input)
# print(results)