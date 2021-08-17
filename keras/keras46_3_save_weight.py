from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape , y.shape)

# print(x[:,1])


# print(datasets.feature_names)
# #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']        
# print(datasets.DESCR)
# print(y[:30])
# print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.70,
)


from sklearn.preprocessing import StandardScaler, MinMaxScaler
# scaler = StandardScaler()
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# print(x_train.shape)

# #2. 모델 구성
model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

# model.save('./_save/keras46_3_save_model_1.h5')
model.save_weights('./_save/keras46_3_save_weight_1.h5')


# #3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=200, mode='min', verbose=3)

import time
start_time =time.time()

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.3
, verbose=1, callbacks=[es])

end_time = time.time() - start_time

# model.save('./_save/keras46_3_save_model_2.h5')
model.save_weights('./_save/keras46_3_save_weight_2.h5')

#여기서 저장

# #4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss' ,loss)

y_predict = model.predict(x_test)


'''

'''


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)



# plt.scatter(x[:,1],y)
# plt.show()

#과제 0.62까지 올리기