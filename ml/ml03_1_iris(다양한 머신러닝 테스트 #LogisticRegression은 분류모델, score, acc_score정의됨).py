## 다중분류 모델!!! y값을 어떻게 숫자의 데이터 값이 아닌 라벨화 시킬지 !
# 값이 있는 위치에만 1을 지정해준다. 즉 영이란 값은 첫번째 자리에 1의 값을
#1은 두번째 자리에서 1의 값을, 2의 값은 2번째 자리에서 1의 값을 주는 것이다. !

import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(y)
'''
데이터 확인 셔플 안하면 조지는 데이터가 있을 수 있음 이렇게
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''

#원핫인코딩 One Hot Encoding (150, 0) -> (150 , 3)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

#[0,1,2,1]
#[[1,0,0]
#[0,1,0]
#[0,0,1]
#[0,1,0]]

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)
# print(y[:5])



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)



# model = LinearSVC()
#accuracy_score 0.9333333333333333
# model = SVC()
#accuracy_score 0.9619047619047619
# model = KNeighborsClassifier()
#accuracy_score 0.9523809523809523

# model = LogisticRegression()
#accuracy_score 0.9238095238095239

# model = DecisionTreeClassifier()
# accuracy_score 0.9428571428571428

model = RandomForestClassifier()
#accuracy_score 0.9333333333333333







#3. 컴파일 및 훈련
### ml에는 컴파일까지 포함되어있당!!
model.fit(x_train, y_train)
# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience= 5, mode= 'min', verbose=1)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train,y_train, epochs=600,
#  batch_size=8, validation_split=0.2, callbacks=[es])



#4. 예측 평가

results = model.score(x_test, y_test)
print(results)
# 얘는 x_Test를 넣어서 나온 예측값과 y_test를 비교한다는 뜻이다.!!
# loss = model.evaluate(x_test, y_test)
# print('loss', loss[0])
# print('accuracy', loss[1])

y_predict = model.predict(x_test[:5])
print('y_predict', y_predict)
print(y_test[:5])

from sklearn.metrics import r2_score, accuracy_score
y_predict =model.predict(x_test)
acc = accuracy_score(y_test, y_predict)  
# 얘랑 score랑 같음 얘는 y_predict이랑 y_test직접 비교해서
print('accuracy_score', acc)



'''
1. evauation ==> score  정의된다.
2. ml ==> 대부분은 y값을 1차원으로 받아들인다.
3. model ==> 정의만 해준다.
4. predict는 그대로 유지된다. 
5. accuracy_score(y_test, y_predict) == score 


'''


'''
1. Deep Learning
loss 0.21113178133964539
accuracy 0.8666666746139526

2. ML Learning

acc : 0.9333333333333333
'''

