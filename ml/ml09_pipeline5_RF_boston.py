
import numpy as np 
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_boston()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)



# 2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score, r2_score




#pipline은 여러가지 파이를 엮는다는 개념으로 모델에 여러가지를 엮을 수 있다. 
model = make_pipeline(MinMaxScaler(), RandomForestRegressor())
#accuracy_score 0.9333333333333333

#3. 컴파일 및 훈련
### ml에는 컴파일까지 포함되어있당!!
model.fit(x_train, y_train)


#4. 예측 평가
# print("최적의 매개변수:", model.best_estimator_)
# print('best_param:',model.best_params_)
# print("best_score :", model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", r2_score(y_test, y_predict))




'''
Random

Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린시간 6.908019065856934
최적의 매개변수: RandomForestRegressor(max_depth=12, min_samples_leaf=5, min_samples_split=10,
                      n_estimators=200, n_jobs=-1)
best_param: {'n_jobs': -1, 'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_depth': 12}
best_score : 0.8340608684567451
model.score : 0.9271636011625088
정답률 : 0.9271636011625088


3. PipLine
model.score : 0.8393677020736838
정답률 : 0.8393677020736838
'''

