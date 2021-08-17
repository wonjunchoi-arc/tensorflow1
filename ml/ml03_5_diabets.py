#실습, 모델구성하고 완료하시오.
# 회귀데이터를 Classfier로 만들었을 경우에 에러확인!!

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression, LinearRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)

# 실습 diabets
#1.  loss와 R2로 평가함
#2. MinMax와 Stanard 결과들 표시

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,GlobalAveragePooling1D,Dropout
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape , y.shape)
print(y.shape)
print(np.unique(y))
y = y.reshape(442,1)

# # # print(x[:,1])


# print(datasets.feature_names)
# #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']        
# print(datasets.DESCR)
# print(y[:30])
# print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.7,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(y_train.shape)



# # print(x_train.shape)


#2. 모델 구성

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression, LinearRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)


model = KNeighborsRegressor()
#SCORE!! 0.4105217806695164

# model = LinearRegression()
#SCORE!! 0.493853457123357

# model = DecisionTreeRegressor()
#SCORE!! -0.0049433521750976706

# model = RandomForestRegressor()
#SCORE!! 0.4928542061515041

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

