'''
# 실습
# 1~100까지의 데이터를
1, 2, 3, 4, 5   /   6
...
95, 96, 97, 98, 99   /   100
예상 결과값 -> 101, 102, 103, 104, 105, 106
평가지표 -> RMSE, R2
'''

import numpy as np
from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.layers.core import Flatten

x_data = np.array(range(1, 101))
x_pre_data = np.array(range(96, 107))  # 106 X 107 O
'''
predict의 예상 결과값
96, 97, 98, 99, 100  /  101
...
101, 102, 103, 104, 105  /  106
'''

size = 6

def split_x(dataset, size):
    arr = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i+size)]
        arr.append(subset)
    return np.array(arr)

dataset = split_x(x_data, size)

print(dataset)

x = dataset[:, :5].reshape(95,5,1)
y = dataset[:, 5]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =  train_test_split(x,y,
train_size=0.7 , shuffle=False, random_state=66)



# print("x : ", x, " / ")
# print("y : ", y)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D

model = Sequential()

# model.add(LSTM(64, input_shape=(5,1)))
model.add(Conv1D(64, 2, input_shape=(5,1)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))

model.summary()


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)

