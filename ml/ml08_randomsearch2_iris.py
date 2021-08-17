
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

from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameter = [
{'n_estimators': [100, 200]}, #epoch
{'max_depth': [6, 8, 10, 12]},
{'min_samples_leaf': [3, 5, 7, 10]},
{'min_samples_split': [2, 3, 5, 10]},#교차검증 해서 여기만 (4*1*2*5)30번
{'n_jobs': [-1, 2, 4]}] # CPU 몇 코어 쓸지

#이제 여기서 최고의 값과 파라미터를 찾아야한다!!

#2. 모델 구성

model =RandomizedSearchCV(RandomForestClassifier(), parameter, cv=kfold)

# model = SVC()

#3. 컴파일 및 훈련
model.fit(x,y)

#4. 예측 평가
print("최적의 매개변수:", model.best_estimator_)
print('best_param:',model.best_params_)
print("best_score :", model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", accuracy_score(y_test, y_predict))

'''
최적의 매개변수: RandomForestClassifier(max_depth=6)best_score : 0.9666666666666666
model.score : 1.0
정답률 : 1.0

---------------------
Random

최적의 매개변수: RandomForestClassifier(min_samples_split=5)
best_param: {'min_samples_split': 5}
best_score : 0.96
model.score : 0.9904761904761905
정답률 : 0.9904761904761905
'''