
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)

import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


datasets = load_wine()
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

model = RandomForestClassifier()
#Acc: [1.         0.97222222 1.         0.97142857 1.        ] 0.9887

model = LinearSVC()
#Acc: [0.91666667 0.88888889 0.94444444 0.82857143 0.91428571] 0.8986

model = SVC()
#Acc: [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ] 0.6457


model = KNeighborsClassifier()
#Acc: [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714] 0.691


model = LogisticRegression()
#Acc: [0.97222222 0.94444444 0.94444444 0.94285714 1.        ] 0.9608

model = DecisionTreeClassifier()
#Acc: [0.91666667 0.97222222 0.91666667 0.85714286 0.94285714] 0.9211



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

