#완성한뒤 출력결과값 스샷
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

#1. 데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,4,3,5])
x_pred = np.array([6])

###데이터 전처리가 가장 중요하다!!!


#2. 모델 구성

model =Sequential()
model.add(Dense(10,input_dim=1))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1000, batch_size=5) ##하이퍼 파라미터 튜닝!!

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss', loss)

y_predict =model.predict(x)
print('6의 예측값', y_predict)

from sklearn.metrics import r2_score
r2 = r2_score(y,y_predict)
print('r2스코어',r2)



# plt.scatter(x,y)
# plt.plot(x_pred,result,color='red')
# plt.show()


#과제 2 
#R2를 0.9 이상으로 올려라!!!