from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 1 [1,2,3]
# 2 [[1,2,3]]
# 3 [[1,2],[3,4],[5,6]]
# 4 [[[1,2,3],[4,5,6]]]
# 5 [[[1,2]],[[3,4]],[[5,6]]]
# 6 [[[1],[2]],[[3],[4]]]



#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]]) ## (2, 10) 2행10열의 데이터 
# print(x.shape) x의 행열 확인 하는 방법

x1 = np.transpose(x)
# print(x1)

y = np.array([11,12,13,14,15,16,17,18,19,20])   #(10,)  10행의 데이터 1vector 출력값이 하나라는 말로 정의 할 수 있겠다리
print(y.shape)


print(x1)

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=2))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x1, y, epochs=100, batch_size=3)




#4. 평가 예측

loss = model.evaluate(x1,y)
print('loss', loss)

x_pred = np.array([[10, 1.3]]) # (1,2) 1행 2열 인데 얘를 과연 위의 모델에 넣을 수 있을까라고 묻는 다면 열의 갯수가 일치하기 때문에 당연하다. 
result =model.predict(x1)
print('[[10, 1.3]]의 예측',result)

#5. 구현
'''
#1 답변
plt.scatter(result,y)
# plt.plot(x1, result, color='red')
plt.show()
'''

plt.scatter(x1[:,0],result)
plt.scatter(x1[:,1],result)
plt.plot(x1,result, color='red')
plt.show()





'''
Epoch 1000/1000
4/4 [==============================] - 0s 333us/step - loss: 0.2334
1/1 [==============================] - 0s 65ms/step - loss: 0.1816
loss 0.1816353052854538
[[10, 1.3]]의 예측 [[19.178675]]

Epoch 2000/2000
4/4 [==============================] - 0s 333us/step - loss: 0.0142
1/1 [==============================] - 0s 69ms/step - loss: 0.0138
loss 0.013789048418402672
[[10, 1.3]]의 예측 [[19.758959]]


Epoch 3000/3000
4/4 [==============================] - 0s 334us/step - loss: 0.0081
1/1 [==============================] - 0s 69ms/step - loss: 0.0074
loss 0.007440970744937658
[[10, 1.3]]의 예측 [[19.826174]]
'''