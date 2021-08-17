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

# print(x.shape) #(506, 13)
# print(y.shape) #(506,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.8)

# print(x_train)



#2. 모델 구성
model = Sequential()
model.add(Dense(13, input_dim=13))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(1))

#3 . 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=2000, batch_size=15, verbose=3,validation_split=0.4)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)
print('y_predict', y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)



'''
'''



# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
# print(datasets.DESCR)

#loss 출력 v , R2출력 v, 컬럼 어떻게 생격는지도 확인해보자