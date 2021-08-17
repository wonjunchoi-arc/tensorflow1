#xor은 같으면 0 다르면 1

from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data= [[0,0],[0,1],[1,0],[1,1]] 
y_data = [0,1,1,0]

#2. 모델
# model = LinearSVC()
# model =SVC()
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
'''가운데 레이어에 activation='relu' 않넣어주면 값이 [[0.48836946]
 [0.49780586]
 [0.49931043]
 [0.508748  ]]
activation='relu' 넣어주면
[[0.18766126]
 [0.9441165 ]
 [0.98138267]
 [0.03379678]]
그리고 단을 적당히 많이 넣어주면 더 정확한 값을 도출해낸다. 
'''
model.add(Dense(1,activation='sigmoid'))

#3. 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

#4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data,"의 예측결과",y_predict)
predict = []
for i in y_predict:
    predict.append(np.round(i)) #반올림 함수 !!

print('수정```````````````',predict)

results =model.evaluate(x_data, y_data)
print('model score', results)

acc = accuracy_score(y_data, predict) #여기에 정수 넣을려고 round씀
print('accuacy',acc)