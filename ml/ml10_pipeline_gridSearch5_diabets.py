
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor #의사결정 나무가 모여 숲을 이룸(앙상블)
from time import time
import numpy as np 
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler

datasets = load_diabetes()

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)

from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV,RandomizedSearchCV
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)


n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

#Pipeline용
parameter = [
{'rf__n_estimators': [100, 200], 'rf__max_depth': [6, 8, 10, 12],'rf__min_samples_leaf': [ 5, 7, 10],'rf__min_samples_split': [ 3, 5, 10],'rf__n_jobs': [-1]}, #epoch
{'rf__n_estimators': [10, 20], 'rf__max_depth': [3, 4, 5, 6],'rf__min_samples_leaf': [6,  9, 11],'rf__min_samples_split': [ 3, 5, 10],'rf__n_jobs': [-1]},
]

#2. 모델 구성
# pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())

#형식의 차이점을 알아보자!! make_pipeline vs Pipeline
pipe = Pipeline([("scaler",MinMaxScaler()), ("rf",RandomForestRegressor())])

# 아래와 같은 방식으로 하면 para가 pipe에 대한 para가 되기 때문에 실행이 안된다.
# 그래서 para가서 직접 적어준다. 단 소문자 + __(언더바 두개)
model =GridSearchCV(pipe, parameter, cv=kfold, verbose=1)

#3. 컴파일 및 훈련
start = time()
model.fit(x,y)
end = time()


print('걸린시간', end-start)


#4. 예측 평가
print("최적의 매개변수:", model.best_estimator_)
print('best_param:',model.best_params_)
print("best_score :", model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", r2_score(y_test, y_predict))

'''

'''