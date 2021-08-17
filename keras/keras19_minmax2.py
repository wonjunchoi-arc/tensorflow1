##데이터 전처리의 중요성 종속변수를 종속변수의 최대값으로 나누기!!

from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_boston
datasets= load_boston()
x= datasets.data
y = datasets.target


print(np.min(x), np.max(x)) #0.0 711.0

#711.0 같은 경우에는 연산이 많이 질 수록 그 수가 한없이 커지기 때문에 얘를 
#0~1사이 값으로 바꾸는 것이 좋다. 즉 숫자를 0.~단위로 만들어주게 나눠주고 연산 결과가 끝나고 
#원래값으로 돌려주는 방법이 있다. —> 이렇게 해줘도 각 데이터 간의 비율을 바뀌지 않기 때문에 이것이 성립하는 것이다. 


#데이터 전처리
# x = x/ 711.
# x = x/np.max(x)
# x = (x - np.min(x))/ (np.max(x) - np.min(x)) 
#얘를 column별로 정규화 해주는 것이 좋다!! for 문을 가져다 쓰면 좋을 듯

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x)
x_scale = scaler.transform(x)

 #실행 시킨 애를 변환시킴
# 텐서에서 훈련시키다 실행시키다는 fit으로 조진다.

print(np.min(x_scale), np.max(x_scale))



print(x.shape) #(506, 13)
print(y.shape) #(506,)



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale,y, 
train_size=0.7, random_state=66)

#정규화 하는 것에 대해서 train할 데이터만 정규화를 해야한다. 
#왜냐하면 우리가 미래를 예측할 데이터를 어떻게 알고 전부 정규화를 시켜놓고 트레인 시키겠는가?

# print(x_train)



#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3 . 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=100, batch_size=8, verbose=3)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss)

y_predict = model.predict(x_test)
print('y_predict', y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)



''' 
전처리 이전
r2스코어 0.5861025717155626

전처리 이후
r2스코어 0.8346791942618095

validation 때는 왜 r2가 낮지?

minmax 이후
r2스코어 0.9298092798682422
'''



# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
# print(datasets.DESCR)

#loss 출력 v , R2출력 v, 컬럼 어떻게 생격는지도 확인해보자