from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

from sklearn.model_selection import train_test_split
x_train, x_test1, y_train, y_test1 = train_test_split(x,y,
train_size=0.6, shuffle=False)

x_tset, x_val, y_test, y_val =train_test_split(x_test1,
y_test1, train_size=0.5, shuffle=False)
print(x_tset)

## train_test_split로 만들어라

# x_train=np.array([1,2,3,4,5,6,7])  # 훈련 공부하는 것
# y_train=np.array([1,2,3,4,5,6,7])
# x_test=np.array([8,9,10])
# y_test=np.array([8,9,10])
# x_val = np.array([11,12,13])
# y_val = np.array([11,12,13])



# #2. 모델구성
# model =Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(1))

# # #3. 컴파일, 훈련
# model.compile(loss='mse',optimizer='adam')

# model.fit(x_train, y_train, epochs=300, batch_size=1, validation_data=(x_val, y_val))

# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test) 
# print('loss', loss)

# y_predict = model.predict([11])


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