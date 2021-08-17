from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random



#1. 데이터
x=np.array(range(100))
y=np.array(range(1,101))

# x_train=x[:70]
# y_train=y[:70]
# x_test=x[70:]
# y_test=y[70:]

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# s= np.arange(x_train.shape[0])
# np.random.shuffle(s)

# x_train1=


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size=0.7, shuffle=True, random_state=66)  ## random_state 66은 66번째 난수표에 있는데로 계속해서 랜덤으 섞으므로 같은 값이 나오게 되는 것이다. 

print(x_test)
print(y_test)



# #2. 모델구성
# model =Sequential()
# model.add(Dense(5, input_dim=1))
# model.add(Dense(3))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(4))
# model.add(Dense(1))

# #3. 컴파일, 훈련
# model.compile(loss='mse',optimizer='adam')

# model.fit(x_train, y_train, epochs=1, batch_size=1) #epochs 훈련의 횟수, batch_size 각 자료에 대한 훈련을 1개씩 한다. 1한번 2한번 3한번 하지만 전체 훈련양은 같다. 
# #여기서 모델에 Weight값과 bias값이 저장되어 있다. 

# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test)  #위의 mse값을 통한 loss 반환해줌
# print('loss', loss)

# y_predict = model.predict([11])


# # plt.scatter(x,y)
# # plt.plot(x,y_predict, color='red')
# plt.show()


'''
Epoch 1000/1000
10/10 [==============================] - 0s 444us/step - loss: 2.8066
1/1 [==============================] - 0s 78ms/step - loss: 3.6655
loss 3.66550874710083
6의 예측값 [[11.551649]]

'''