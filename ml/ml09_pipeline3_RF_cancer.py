
import numpy as np 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(y)

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
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score




#pipline은 여러가지 파이를 엮는다는 개념으로 모델에 여러가지를 엮을 수 있다. 
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())
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
print("정답률 :", accuracy_score(y_test, y_predict))


'''
1. evauation ==> score  정의된다.
2. ml ==> 대부분은 y값을 1차원으로 받아들인다.
3. model ==> 정의만 해준다.
4. predict는 그대로 유지된다. 
5. accuracy_score(y_test, y_predict) == score 


'''


'''
1. Deep Learning
loss 0.21113178133964539
accuracy 0.8666666746139526

2. ML Learning

acc : 0.9333333333333333

3. PipLine
model.score : 0.9448621553884712
정답률 : 0.9448621553884712

'''

