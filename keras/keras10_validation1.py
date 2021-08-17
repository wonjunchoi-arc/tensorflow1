from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x_train=np.array([1,2,3,4,5,6,7])  # 훈련 공부하는 것
y_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])
y_test=np.array([8,9,10])
x_val = np.array([11,12,13])
y_val = np.array([11,12,13])

#validatation이란 머신이 훈련한 내용을 바탕으로 문제를 풀어보는 과정과 같다. 


#2. 모델구성
model =Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')

model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
loss = model.evaluate(x_test,y_test) 
print('loss', loss)

y_predict = model.predict([11])


# plt.scatter(x,y)
# plt.plot(x,y_predict, color='red')
# plt.show()


'''
Epoch 1000/1000
10/10 [==============================] - 0s 444us/step - loss: 2.8066
1/1 [==============================] - 0s 78ms/step - loss: 3.6655
loss 3.66550874710083
6의 예측값 [[11.551649]]

'''