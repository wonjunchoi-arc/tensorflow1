
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)

import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
# print(y)

from sklearn.model_selection import train_test_split, KFold,cross_val_score
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#2. 모델 구성

# model = RandomForestClassifier()
#Acc: [0.93333333 0.96666667 1.         0.86666667 0.96666667] 0.9467

model = LinearSVC()
#Acc: [0.96666667 0.96666667 1.         0.9        1.        ] 0.9667
#train Acc: [1.         0.66666667 0.77777778 1.         1.        ] 0.8889 

# model = SVC()
#Acc: [0.96666667 0.96666667 1.         0.93333333 0.96666667] 0.9667
#train Acc: [1.         0.88888889 0.88888889 1.         1.        ] 0.9556

# model = KNeighborsClassifier()
#Acc: [0.96666667 0.96666667 1.         0.9        0.96666667] 0.96
#train Acc: [1.         1.         0.77777778 1.         1.        ] 0.9556

# model = LogisticRegression()
#Acc: [1.         0.96666667 1.         0.9        0.96666667] 0.9667
#train Acc: [1.         0.77777778 0.88888889 1.         1.        ] 0.9333

# model = DecisionTreeClassifier()
#Acc: [0.93333333 0.96666667 1.         0.9        0.93333333] 0.9467
# train acc Acc: [0.77777778 0.77777778 0.88888889 1.         1.        ] 0.8889




#3. 컴파일 및 훈련
#4. 예측 평가
score =cross_val_score(model, x_train, y_train, cv =kfold) 
# cross_val_score ==>여기에는 fit, score까지 포함되어있다!!'
print("Acc:",score, round(np.mean(score),4))#소수 4자리까지 반올림인듯






'''
1. Deep Learning
loss 0.21113178133964539
accuracy 0.8666666746139526

2. ML Learning

acc : 0.9333333333333333
'''

