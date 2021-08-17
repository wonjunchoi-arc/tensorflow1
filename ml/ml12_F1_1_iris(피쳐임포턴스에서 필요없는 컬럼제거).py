#실습
# 피쳐임포턴스가 전체 중요도에서 20%미만인 컬럼등을 제거해서 데이터 재 구성후
#각 모델별로 돌려서 결과확인
from re import L
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRFRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


#1. 데이터
datasets = load_iris()
print(type(datasets))
print(datasets.feature_names)
print(datasets.data.shape[1])
x = pd.DataFrame(data=datasets.data, columns=datasets.feature_names)
y = pd.DataFrame(datasets.target)

x.drop(['sepal width (cm)'], axis=1, inplace=True)




x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)



#2. 모델
model =DecisionTreeClassifier()
# model =RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc =model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

'''
acc: 0.9333333333333333
[0.0125026  0.         0.53835801 0.44913938]
해당 모델을 돌린 조건에서 의 컬럼이 미치는 영향

'''
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]-1
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), x.columns.tolist())
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''
1.de
acc: 0.9111111111111111
[0.         0.01906837 0.04351141 0.93742021]  

후
acc: 0.8888888888888888

2. RAN
acc: 0.9111111111111111
[0.0713492  0.0132194  0.52293176 0.39249964]   

후 acc: 0.8888888888888888

3. GB
acc: 0.8888888888888888
[0.00331243 0.01096357 0.48647516 0.49924885]
후
acc: 0.8888888888888888

4.XGB
acc: 0.9111111111111111
[0.03220411 0.02537574 0.7033289  0.23909122]       

삭제후 
acc: 0.9111111111111111


'''