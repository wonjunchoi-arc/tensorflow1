from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


#1. 데이터
x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1,2,3,4,5,6,7,8,9,10])

x_train=x[:7]
y_train=y[:7]
x_test=x[7:]
y_test=y[7:]

print(x_train)

print(x_test)


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

model.fit(x_train, y_train, epochs=1000, batch_size=1) #epochs 훈련의 횟수, batch_size 각 자료에 대한 훈련을 1개씩 한다. 1한번 2한번 3한번 하지만 전체 훈련양은 같다. 
#여기서 모델에 Weight값과 bias값이 저장되어 있다. 

#4. 평가, 예측
loss = model.evaluate(x_test,y_test)  #위의 mse값을 통한 loss 반환해줌
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