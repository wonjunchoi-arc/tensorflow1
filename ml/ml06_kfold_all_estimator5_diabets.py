from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np 
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score


datasets = load_diabetes()
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
ARDRegression 의 정답률 평균: 0.4923
AdaBoostRegressor 의 정답률 평균: 0.4227
BaggingRegressor 의 정답률 평균: 0.3858
BayesianRidge 의 정답률 평균: 0.4893
CCA 의 정답률 평균: 0.438
DecisionTreeRegressor 의 정답률 평균: -0.1451
DummyRegressor 의 정답률 평균: -0.0033
ElasticNet 의 정답률 평균: 0.0054
ElasticNetCV 의 정답률 평균: 0.4394
ExtraTreeRegressor 의 정답률 평균: -0.1914
ExtraTreesRegressor 의 정답률 평균: 0.4604
GammaRegressor 의 정답률 평균: 0.0027
GaussianProcessRegressor 의 정답률 평균: -11.0753
GradientBoostingRegressor 의 정답률 평균: 0.4404
HistGradientBoostingRegressor 의 정답률 평균: 0.3947HuberRegressor 의 정답률 평균: 0.4822
IsotonicRegression 의 정답률 평균: nan
KNeighborsRegressor 의 정답률 평균: 0.3673
KernelRidge 의 정답률 평균: -3.5938
Lars 의 정답률 평균: -0.1495
LarsCV 의 정답률 평균: 0.4879
Lasso 의 정답률 평균: 0.3518
LassoCV 의 정답률 평균: 0.487
LassoLars 의 정답률 평균: 0.3742
LassoLarsCV 의 정답률 평균: 0.4866
LassoLarsIC 의 정답률 평균: 0.4912
LinearRegression 의 정답률 평균: 0.4876
LinearSVR 의 정답률 평균: -0.3684
MLPRegressor 의 정답률 평균: -2.9653
MultiOutputRegressor
MultiTaskElasticNet 의 정답률 평균: nan
MultiTaskElasticNetCV 의 정답률 평균: nan
MultiTaskLasso 의 정답률 평균: nan
MultiTaskLassoCV 의 정답률 평균: nan
NuSVR 의 정답률 평균: 0.1618
OrthogonalMatchingPursuit 의 정답률 평균: 0.3121    
OrthogonalMatchingPursuitCV 의 정답률 평균: 0.4857
PLSCanonical 의 정답률 평균: -1.2086
PLSRegression 의 정답률 평균: 0.4842
PassiveAggressiveRegressor 의 정답률 평균: 0.467
PoissonRegressor 의 정답률 평균: 0.3341
RANSACRegressor 의 정답률 평균: 0.0127
RadiusNeighborsRegressor 의 정답률 평균: -0.0033
RandomForestRegressor 의 정답률 평균: 0.4309
RegressorChain
Ridge 의 정답률 평균: 0.4212
RidgeCV 의 정답률 평균: 0.4884
SGDRegressor 의 정답률 평균: 0.4089
SVR 의 정답률 평균: 0.1591
StackingRegressor
TheilSenRegressor 의 정답률 평균: 0.477
TransformedTargetRegressor 의 정답률 평균: 0.4876   
TweedieRegressor 의 정답률 평균: 0.0032
VotingRegressor
'''