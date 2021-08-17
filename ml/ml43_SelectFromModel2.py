#실습
#데이터는 diabets

#1. 상단 모델에 그리드서치 랜덤서치로 튜닝 모델 구성
#최적의 R2값과 피처임포턴스 구할 것

#2 .위 스레드 값으로 SelectFrom돌려서
#최적의 피쳐갯수 구할 것

#3 . 위 피쳐 갯수로 피쳐 갯수를 조정한뒤
#그걸로 다시 랜덤서치 그리드 서치해서 
#최적의 R2 구할 것

# 1번값과 3번값 비교 # 0.47이상
import numpy as np 
from scipy.sparse import data
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x, y = load_diabetes(return_X_y=True)
print(x.shape , y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(

    x,y , train_size=0.8, shuffle=True, random_state=66
)


parameter = [
{"n_estimators":[1, 10, 100, 1000],"max_depth":[2,4,6],'learning_rate': [0.05,0.1,0.2]}, #교차검증 해서 여기만 (4*5)20번
{"n_estimators":[1, 10, 100, ], "gamma":[0.001, 0.0001],"max_depth":[2,4,6],'learning_rate': [0.05,0.1,0.2],},#교차검증 해서 여기만 (3*1*2*5)30번
{"n_estimators":[1, 10, 100, 1000], "gamma":[0.001, 0.0001],"max_depth":[2,4,6],'learning_rate': [0.05,0.1,0.2]}#교차검증 해서 여기만 (4*1*2*5)30번
] # 총 ==> 90번 돌아간다!!!

#2. 모델
model = RandomizedSearchCV(XGBRegressor(),parameter, verbose=1)

#3. 컴파일 및 훈련
import time

start = time.time()
model.fit(x_train,y_train)
end = time.time()

print('걸린시간:', end-start)

#4. 예측 평가
print("최적의 매개변수:", model.best_estimator_)
print("best_score:", model.best_score_)

# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", r2_score(y_test, y_predict))
