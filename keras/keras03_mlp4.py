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
x=np.array([range(10)])


y = np.array([[1,2,3,4,5,6,7,8,9,10],
 [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],
 [10,9,8,7,6,5,4,3,2,1]]) ## (3, 10) 3행10열의 데이터 

# print(x.shape) x의 행열 확인 하는 방법
x1 = np.transpose(x)
y1 = np.transpose(y)
# print(x1)

# y = np.array([11,12,13,14,15,16,17,18,19,20])   #(10,)  10행의 데이터 1vector 출력값이 하나라는 말로 정의 할 수 있겠다리
# print(y.shape)




#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

model.fit(x1, y1, epochs=100, batch_size=3)




#4. 평가 예측

loss = model.evaluate(x1,y1)
print('loss', loss)

x_pred = np.array([[10]]) # (1,2) 1행 2열 인데 얘를 과연 위의 모델에 넣을 수 있을까라고 묻는 다면 열의 갯수가 일치하기 때문에 당연하다. 
result =model.predict(x1)
print('[[10, 1.3]]의 예측',result)


#5. 구현

plt.scatter(x1,y1[:,0])
plt.scatter(x1,y1[:,1])
plt.scatter(x1,y1[:,2])
plt.plot(x1,result[:,0], color='red')
plt.plot(x1,result[:,1], color='blue')
plt.plot(x1,result[:,2], color='green')

plt.show()




'''
Epoch 1000/1000
4/4 [==============================] - 0s 666us/step - loss: 0.0057
1/1 [==============================] - 0s 70ms/step - loss: 0.0054
loss 0.0054274857975542545
[[10, 1.3]]의 예측 [[1.1003688e+01 1.5801768e+00 1.4358610e-03]]


#1
Epoch 1000/1000
4/4 [==============================] - 0s 333us/step - loss: 10.4931
1/1 [==============================] - 0s 61ms/step - loss: 9.3785
loss 9.378477096557617
[[10, 1.3]]의 예측 [[4.269238]]
'''