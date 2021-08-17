from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
from tensorflow.python.keras.backend import shape 
#1. 데이터 
x = np.array([range(100), range(301,401), range(1,101)
, range(100), range(401,501)])
x= np.transpose(x)

print(x.shape)

y = np.array([range(711,811), range(101,201)])
y = np.transpose(y)
print(y.shape)

#2. 모델 구성
# model = Sequential()
# model.add(Dense(10, input_shape=(5,)))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(2))

# 단일형 모델로 쓰기에는 간단해서 좋지만 여러개의 모델을 합치기는 어렵다.

input1 =Input(shape=(5,))
dense1= Dense(3)(input1)
dense2= Dense(3)(dense1)
dense3= Dense(3)(dense2)
dense4= Dense(3)(dense3)
output1 = Dense(2)(dense4)

model = Model(inputs = input1, outputs = output1)

model.summary()
#함수형 모델은 여러가지 모델을 합칠 수도 레이어간의 단계를 뛰어넘는 등의 자유도를 가진다. 

# model.summary()

#3 컴파일, 훈련

#4. 평가, 예측 