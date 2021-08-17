import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D



# # 모델 
model =Sequential()
model.add(Dense(units=10, activation='relu', input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(9))
model.add(Dense(9))
model.add(Dense(10))

model.summary()
# #3. 컴파일, 훈련 , metrics=['acc']



#4. 평가, 예측


