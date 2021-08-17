

from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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





#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3 . 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=8, verbose=3)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)
print('y_predict', )



from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)



''' 
1. Stand
loss 10.093451499938965
r2스코어 0.877828543524486

2.min max
loss 10.695916175842285
y_predict
r2스코어 0.8705362927168208

3.Robust
loss 13.279632568359375
y_predict
r2스코어 0.8392629027906773

4.Quantile
loss 12.63692569732666
y_predict
r2스코어 0.8470422541958962

5, Maxabs
loss 10.652739524841309
y_predict
r2스코어 0.8710589032814069

6. Power
loss 10.764941215515137
y_predict
r2스코어 0.8697008157689147

'''

plt.scatter(x[:,6],y)
plt.show()

# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
# print(datasets.DESCR)

#loss 출력 v , R2출력 v, 컬럼 어떻게 생격는지도 확인해보자