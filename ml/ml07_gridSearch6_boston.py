
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)
from time import time
import numpy as np 
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score



datasets = load_boston()
# print(datasets.DESCR)
# print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
# print(y)

from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameter = [
{'n_estimators': [100, 200], 'max_depth': [6, 8, 10, 12],'min_samples_leaf': [ 5, 7, 10],'min_samples_split': [ 3, 5, 10],'n_jobs': [-1]}, #epoch
{'n_estimators': [10, 20], 'max_depth': [3, 4, 5, 6],'min_samples_leaf': [6,  9, 11],'min_samples_split': [ 3, 5, 10],'n_jobs': [-1]},
]#이제 여기서 최고의 값과 파라미터를 찾아야한다!!

#2. 모델 구성

model =GridSearchCV(RandomForestRegressor(), parameter, cv=kfold, verbose=1)

# model = SVC()

#3. 컴파일 및 훈련
start = time()
model.fit(x,y)
end = time()


print('걸린시간', end-start)


#4. 예측 평가
print("최적의 매개변수:", model.best_estimator_)
print("best_score :", model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", r2_score(y_test, y_predict))

'''
Fitting 5 folds for each of 144 candidates, totalling 720 fits
걸린시간 64.40318155288696
최적의 매개변수: RandomForestRegressor(max_depth=10, min_samples_leaf=5, min_samples_split=5,
                      n_estimators=200, n_jobs=-1)  
best_score : 0.8364835056261135
model.score : 0.9262774997317784
정답률 : 0.9262774997317784
'''