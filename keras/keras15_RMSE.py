#1. R2를 음수가 아닌 0.5이하로 만들어라.
#2. 데어터 건들지 마 
#3. 레이어는 인풋 아웃풋 포함 6개 이상
#4. batch size 는 =1
#5. ep는  100이상
#6 히든 레이어의 노드는 10개이상 1000개 이하 
#7 train 70%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터 
x=np.array(range(100))
y=np.array(range(1,101))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size=0.7) 



#2. 모델 구성
model = Sequential()
model.add(Dense(2, input_dim =1 ))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(2))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#두번쨰 방법
# model.compile(loss='kld', optimizer='adam')
# model.fit(x_train, y_train, epochs=100, batch_size=1)


#4. 평가예측
loss = model.evaluate(x_test, y_test) 
print('loss', loss)

y_predict = model.predict(x_test)
print('y_predict' , y_predict)

from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)



def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse =RMSE(y_test, y_predict)
print('rmse', rmse)

#r2스코어 0.9999999999999584
#rmse 6.1306499289036445e-06
'''
loss 3182.093994140625
r2 0.5366601686095063
'''
