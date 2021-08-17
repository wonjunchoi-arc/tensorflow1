
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
from sklearn.metrics import accuracy_score



datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
# print(y)
'''
Random은 CV1번당 10번만 랜덤으로 돌린다. 
'''
from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameter = [
{"C":[1, 10, 100, 1000], "kernel":["linear"]}, #교차검증 해서 여기만 (4*5)20번
{"C":[1, 10, 100, ], "kernel":["rbf"], "gamma":[0.001, 0.0001]},#교차검증 해서 여기만 (3*1*2*5)30번
{"C":[1, 10, 100, 1000], "kernel":["sigmoid"], "gamma":[0.001, 0.0001]}#교차검증 해서 여기만 (4*1*2*5)30번
] # 총 ==> 90번 돌아간다!!!

#이제 여기서 최고의 값과 파라미터를 찾아야한다!!

#2. 모델 구성
#Gridsearch랑 동일하나 파라미터를 전부 돌리는 것이 아니라 일부만 돌림
model =GridSearchCV(SVC(), parameter, cv=kfold,verbose=1) # 90 fits
# model =RandomizedSearchCV(SVC(), parameter, cv=kfold,verbose=1) #50 fits

# model = SVC()

#3. 컴파일 및 훈련
model.fit(x,y)

#4. 예측 평가
print("최적의 매개변수:", model.best_estimator_)
print("best_score :", model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", accuracy_score(y_test, y_predict))

'''
최적의 매개변수: SVC(C=1, kernel='linear')
best_score : 0.9800000000000001
model.score : 1.0
정답률 : 1.0
'''