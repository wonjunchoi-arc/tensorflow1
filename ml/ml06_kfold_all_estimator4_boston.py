from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np 
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score


datasets = load_boston()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(y)


#2. 모델 구성 # all_estimators는 즉 모든 ml모델을 불러오는 것!
allAlgorithms= all_estimators(type_filter='regressor')
# print(allAlgorithms)
kfold = KFold(n_splits=5, shuffle=True, random_state=66)
print('모델의 갯수',len(allAlgorithms))
for name, algorithms in allAlgorithms:
    try:
        model = algorithms()

        scores =cross_val_score(model, x, y, cv=kfold)

        print(name, '의 정답률 평균:', round(np.mean(scores),4))
    except:
        print(name)
        continue
'''
모델의 갯수 54
ARDRegression 의 정답률 평균: 0.6985
AdaBoostRegressor 의 정답률 평균: 0.8417
BaggingRegressor 의 정답률 평균: 0.8474
BayesianRidge 의 정답률 평균: 0.7038
CCA 의 정답률 평균: 0.6471
DecisionTreeRegressor 의 정답률 평균: 0.7543
DummyRegressor 의 정답률 평균: -0.0135
ElasticNet 의 정답률 평균: 0.6708
ElasticNetCV 의 정답률 평균: 0.6565
ExtraTreeRegressor 의 정답률 평균: 0.7422
ExtraTreesRegressor 의 정답률 평균: 0.8745
GammaRegressor 의 정답률 평균: -0.0136
GaussianProcessRegressor 의 정답률 평균: -5.9286
GradientBoostingRegressor 의 정답률 평균: 0.8844
HistGradientBoostingRegressor 의 정답률 평균: 0.8581HuberRegressor 의 정답률 평균: 0.584
IsotonicRegression 의 정답률 평균: nan
KNeighborsRegressor 의 정답률 평균: 0.5286
KernelRidge 의 정답률 평균: 0.6854
Lars 의 정답률 평균: 0.6977
LarsCV 의 정답률 평균: 0.6928
Lasso 의 정답률 평균: 0.6657
LassoCV 의 정답률 평균: 0.6779
LassoLars 의 정답률 평균: -0.0135
LassoLarsCV 의 정답률 평균: 0.6965
LassoLarsIC 의 정답률 평균: 0.713
LinearRegression 의 정답률 평균: 0.7128
LinearSVR 의 정답률 평균: 0.3901
MLPRegressor 의 정답률 평균: 0.5189
MultiOutputRegressor
MultiTaskElasticNet 의 정답률 평균: nan
MultiTaskElasticNetCV 의 정답률 평균: nan
MultiTaskLasso 의 정답률 평균: nan
MultiTaskLassoCV 의 정답률 평균: nan
NuSVR 의 정답률 평균: 0.2295
OrthogonalMatchingPursuit 의 정답률 평균: 0.5343    
OrthogonalMatchingPursuitCV 의 정답률 평균: 0.6578
PLSCanonical 의 정답률 평균: -2.2096
RadiusNeighborsRegressor 의 정답률 평균: nan        
RandomForestRegressor 의 정답률 평균: 0.8733        
RegressorChain
Ridge 의 정답률 평균: 0.7109
RidgeCV 의 정답률 평균: 0.7128
SGDRegressor 의 정답률 평균: -5.90066549574885e+25  
SVR 의 정답률 평균: 0.1963
StackingRegressor
TheilSenRegressor 의 정답률 평균: 0.6758
TransformedTargetRegressor 의 정답률 평균: 0.7128   
TweedieRegressor 의 정답률 평균: 0.6558
VotingRegressor
'''