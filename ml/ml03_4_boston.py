#실습, 모델구성하고 완료하시오.
# 회귀데이터를 Classfier로 만들었을 경우에 에러확인!!

from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_boston
datasets= load_boston()
x= datasets.data
y = datasets.target

print(len(np.unique(y)))



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.8)

print(x_train.shape)


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer,PowerTransformer
scaler =StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# from sklearn.preprocessing import OneHotEncoder
# en = OneHotEncoder(sparse=False)
# y_train = en.fit_transform(y_train)
# y_test = en.fit_transform(y_test)

print(x_train.shape)
print(y_train.shape)



#2. 모델 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression, LinearRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)


model = KNeighborsRegressor()
#이벨류에이트대신 쓰는 SCORE!! 0.7657831432961492

model = LinearRegression()
#SCORE!! 0.7435905335464271

model = DecisionTreeRegressor()
#SCORE!! 0.6628472378551169

model = RandomForestRegressor()
#SCORE!! 0.7959515797805816

#3. 컴파일 및 훈련

model.fit(x_train,y_train)


#4. 예측 평가

y_predict = model.predict(x_test)
print('y_predict', y_predict)

predict = []
for i in y_predict:
    predict.append(np.round(i))

test = []
for i in y_test:
    test.append(np.round(i))
print(predict)
print(test)

from sklearn.metrics import r2_score, accuracy_score
results = model.score(x_test, y_test)
print('이벨류에이트대신 쓰는 SCORE!!', results)

# acc= accuracy_score(test, y_predict)
# print('acc_score', acc)

'''
머신 러닝에는 이벨류에이트 읎다!!
results = model.evaluate(x_train, y_train)
print('model score!!!!',results)

'''

