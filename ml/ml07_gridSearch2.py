#실습



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
#grid search cv  채로 걸러낸걸 찾겠다.
#내가 기존 모델이 있어 거기에 내가 넣고 싶은 하이퍼 파리미터 튜닝을 딕셔너리 형태로 만들거야!!,
#  거기에 교차검증도 할거야  
이 3가지를 랩핑해서 하나로 만들거야!
그러므로 총 진행횟수는 내부에서 진행되는 경우의 수 이다
'''




from sklearn.model_selection import train_test_split, KFold,cross_val_score, GridSearchCV
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

# model =GridSearchCV(SVC(), parameter, cv=kfold)

model = SVC(C=1 ,kernel='linear')

#3. 컴파일 및 훈련
model.fit(x,y)

#4. 예측 평가
# print("최적의 매개변수:", model.best_estimator_)
# print("best_score :", model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", model.score(x_test,y_test))

y_predict = model.predict(x_test)
print("정답률 :", accuracy_score(y_test, y_predict))

'''
model.score : 1.0
정답률 : 1.0
'''