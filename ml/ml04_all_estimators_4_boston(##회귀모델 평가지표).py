from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import all_estimators

from sklearn.metrics import explained_variance_score ##회귀모델 평가지표

from sklearn.datasets import load_boston
import warnings
warnings.filterwarnings('ignore')


datasets= load_boston()
x= datasets.data
y = datasets.target

print(len(np.unique(y)))



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.8)

print(x_train.shape)


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer,PowerTransformer
scaler =StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성 # all_estimators는 즉 모든 ml모델을 불러오는 것!
allAlgorithms= all_estimators(type_filter='regressor')
# print(allAlgorithms)
print('모델의 갯수',len(allAlgorithms))
for name, algorithms in allAlgorithms:
    try:
        model = algorithms()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        acc = explained_variance_score(y_test, y_predict)
        print(name, '의 정답률:', acc)
    except:
        print(name)
        continue

'''
모델의 갯수 54
ARDRegression 의 정답률: 0.7362526422787936
AdaBoostRegressor 의 정답률: 0.8380604780217842
BaggingRegressor 의 정답률: 0.8500587867385678
BayesianRidge 의 정답률: 0.7354708547890065
CCA 의 정답률: 0.7129473934570867
DecisionTreeRegressor 의 정답률: 0.6585715621244363
DummyRegressor 의 정답률: 1.1102230246251565e-16
ElasticNet 의 정답률: 0.6167693087312661
ElasticNetCV 의 정답률: 0.7354903637449552
ExtraTreeRegressor 의 정답률: 0.6784360543862535
ExtraTreesRegressor 의 정답률: 0.8894789768963703
GammaRegressor 의 정답률: 0.6037500778117424
GaussianProcessRegressor 의 정답률: 0.6224441422112199
GradientBoostingRegressor 의 정답률: 0.8922798122476526
HistGradientBoostingRegressor 의 정답률: 0.8902060511628757
HuberRegressor 의 정답률: 0.7175256131942723
IsotonicRegression
KNeighborsRegressor 의 정답률: 0.8069067146116182
KernelRidge 의 정답률: 0.7365801133168792
Lars 의 정답률: 0.736723001258307
LarsCV 의 정답률: 0.7343643377074629
Lasso 의 정답률: 0.6591190773877051
LassoCV 의 정답률: 0.7361442188458844
LassoLars 의 정답률: 1.1102230246251565e-16
LassoLarsCV 의 정답률: 0.7360711275856849
LassoLarsIC 의 정답률: 0.7364939886145783
LinearRegression 의 정답률: 0.7367230012583073
LinearSVR 의 정답률: 0.7143799015945711
MLPRegressor 의 정답률: 0.704050541889301
MultiOutputRegressor
MultiTaskElasticNet
MultiTaskElasticNetCV
MultiTaskLasso
MultiTaskLassoCV
NuSVR 의 정답률: 0.6815755109686916
OrthogonalMatchingPursuit 의 정답률: 0.5159482263126369
OrthogonalMatchingPursuitCV 의 정답률: 0.7119721785587687
PLSCanonical 의 정답률: -1.7192715470078777
PLSRegression 의 정답률: 0.711056545917359
PassiveAggressiveRegressor 의 정답률: 0.6505032838584768
PoissonRegressor 의 정답률: 0.792772887028386
RANSACRegressor 의 정답률: 0.3819553687406345
RadiusNeighborsRegressor
RandomForestRegressor 의 정답률: 0.8529583234299587
RegressorChain
Ridge 의 정답률: 0.736580113316901
RidgeCV 의 정답률: 0.7343217748165538
SGDRegressor 의 정답률: 0.7363628545731546
SVR 의 정답률: 0.6971338157410072
StackingRegressor
TheilSenRegressor 의 정답률: 0.638568356516914
TransformedTargetRegressor 의 정답률: 0.7367230012583073
TweedieRegressor 의 정답률: 0.5952870398946662
VotingRegressor
'''