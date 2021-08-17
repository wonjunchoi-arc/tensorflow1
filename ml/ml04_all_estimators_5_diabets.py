#실습, 모델구성하고 완료하시오.
# 회귀데이터를 Classfier로 만들었을 경우에 에러확인!!
from sklearn.utils import all_estimators
from sklearn.metrics import explained_variance_score ##회귀모델 평가지표
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,GlobalAveragePooling1D,Dropout
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape , y.shape)
print(y.shape)
print(np.unique(y))
y = y.reshape(442,1)

# # # print(x[:,1])


# print(datasets.feature_names)
# #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']        
# print(datasets.DESCR)
# print(y[:30])
# print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.7,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(y_train.shape)

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
ARDRegression 의 정답률: 0.541362573527007
AdaBoostRegressor 의 정답률: 0.4866899404872025
BaggingRegressor 의 정답률: 0.3460484127440989
BayesianRidge 의 정답률: 0.5447640271008016
CCA 의 정답률: 0.5325370692058508
DecisionTreeRegressor 의 정답률: -0.16943725646051777
DummyRegressor 의 정답률: 0.0
ElasticNet 의 정답률: 0.46311365926708414
ElasticNetCV 의 정답률: 0.5428209971055151
ExtraTreeRegressor 의 정답률: 0.006890248916454622
ExtraTreesRegressor 의 정답률: 0.4792734058420711
GammaRegressor 의 정답률: 0.39597354488669034
GaussianProcessRegressor 의 정답률: -0.11009280168204705
GradientBoostingRegressor 의 정답률: 0.4607847746711823
HistGradientBoostingRegressor 의 정답률: 0.4360084686472314
HuberRegressor 의 정답률: 0.5371176608782823
IsotonicRegression
KNeighborsRegressor 의 정답률: 0.4164717890487778
KernelRidge 의 정답률: -0.11941884439781303
Lars 의 정답률: 0.5399680603063577
LarsCV 의 정답률: 0.5424827412111665
Lasso 의 정답률: 0.5424250842633718
LassoCV 의 정답률: 0.544338296691636
LassoLars 의 정답률: 0.40142946860032325
LassoLarsCV 의 정답률: 0.5410379145704294
LassoLarsIC 의 정답률: 0.5436737999148791
LinearRegression 의 정답률: 0.5399680603063577
LinearSVR 의 정답률: 0.5056785515677025
MLPRegressor 의 정답률: 0.35030087580178637
MultiOutputRegressor
MultiTaskElasticNet 의 정답률: 0.46311365926708414
MultiTaskElasticNetCV 의 정답률: 0.5428209971055151
MultiTaskLasso 의 정답률: 0.5424250842633718
MultiTaskLassoCV 의 정답률: 0.5443382966916361
NuSVR 의 정답률: 0.1602180989702402
OrthogonalMatchingPursuit 의 정답률: 0.3350601709114257
OrthogonalMatchingPursuitCV 의 정답률: 0.5380841735053203
PLSCanonical 의 정답률: -0.8195388648717834
PLSRegression 의 정답률: 0.5521492578711773
PassiveAggressiveRegressor 의 정답률: 0.5365566699749996
PoissonRegressor 의 정답률: 0.53895032645715
RANSACRegressor 의 정답률: 0.14737858476652643
RadiusNeighborsRegressor
RandomForestRegressor 의 정답률: 0.435106127745169
RegressorChain
Ridge 의 정답률: 0.5441333291346027
RidgeCV 의 정답률: 0.5441333291345642
SGDRegressor 의 정답률: 0.5464985463744143
SVR 의 정답률: 0.18410234605053355
StackingRegressor
TheilSenRegressor 의 정답률: 0.5363572447701602
TransformedTargetRegressor 의 정답률: 0.5399680603063577
TweedieRegressor 의 정답률: 0.402573675973095
VotingRegressor
'''