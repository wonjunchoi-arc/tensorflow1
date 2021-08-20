import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x= np.array([1,2,3,4,5])
y= np.array([1,2,3,4,5])

#2. 모델

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(2))
model.add(Dense(1))

model.summary()

print(model.weights)

print('================================')

print(model.trainable_weights)

print('===================')

print(len(model.weights))
# 6개 나오고 이는 각 층의 weight하나 bias하나 총 3층이니깐 6개 ==> 3(w+b)
print(len(model.trainable_weights))




'''
================================================================= 
dense (Dense)                (None, 3)                 6
_________________________________________________________________ 
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________ 
dense_2 (Dense)              (None, 1)                 3
================================================================= 
Total params: 17
Trainable params: 17  얘는 뭘까??????
Non-trainable params: 0
'''

'''
weight에 대한 초기값이 있고 
'dense_1/kernel 여기서 말하느 커널이 가중치를 말하는 것이다!
'''