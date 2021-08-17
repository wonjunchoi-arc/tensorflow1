#통데이터로~~

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)

import numpy as np 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


datasets = load_breast_cancer()
# print(datasets.DESCR)
# print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
# print(y)

from sklearn.model_selection import train_test_split, KFold,cross_val_score
# x_train, x_test, y_train, y_test =train_test_split(
#     x,y, test_size=0.7, random_state=66)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델 구성

model = RandomForestClassifier()
#Acc: [0.96491228 0.95614035 0.96491228 0.95614035 0.99115044] 0.9667

model = LinearSVC()
#Acc: [0.90350877 0.9122807  0.90350877 0.73684211 0.86725664] 0.8647

model = SVC()
#Acc: [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 0.921


model = KNeighborsClassifier()
#Acc: [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 0.928


model = LogisticRegression()
#Acc: [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 0.9385


model = DecisionTreeClassifier()
#Acc: [0.93859649 0.92105263 0.92105263 0.89473684 0.95575221] 0.9262



#3. 컴파일 및 훈련
#4. 예측 평가
score =cross_val_score(model, x, y, cv =kfold) 
# cross_val_score ==>여기에는 fit, score까지 포함되어있다!!'
print("Acc:",score, round(np.mean(score),4))#소수 4자리까지 반올림인듯






'''
1. Deep Learning
loss 0.21113178133964539
accuracy 0.8666666746139526

2. ML Learning

acc : 0.9333333333333333
'''

