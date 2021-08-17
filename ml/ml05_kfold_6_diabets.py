
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression, LinearRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)

import numpy as np 
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


datasets = load_diabetes()
# print(datasets.DESCR)
# print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
# print(y)

from sklearn.model_selection import train_test_split, KFold,cross_val_score


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델 구성

model = KNeighborsRegressor()
#Acc: [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 0.3673

model = LinearRegression()
#Acc: [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 0.4876

model = DecisionTreeRegressor()
#Acc: [-0.1926087  -0.28038668 -0.17030109 -0.04686408  0.10201731] -0.1176

model = RandomForestRegressor()
#Acc: [0.36191517 0.48627987 0.46895475 0.40517003 0.41298383] 0.4271

#3. 컴파일 및 훈련
#4. 예측 평가
from sklearn.metrics import r2_score
score =cross_val_score(model, x, y, cv =kfold) 
# cross_val_score ==>여기에는 fit, score까지 포함되어있다!!'
print("Acc:",score, round(np.mean(score),4))#소수 4자리까지 반올림인듯
# r2 = r2_score()





'''
1. Deep Learning
loss 0.21113178133964539
accuracy 0.8666666746139526

2. ML Learning

acc : 0.9333333333333333
'''

