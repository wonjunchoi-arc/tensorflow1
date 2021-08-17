from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(y)


#2. 모델 구성 # all_estimators는 즉 모든 ml모델을 불러오는 것!
allAlgorithms= all_estimators(type_filter='classifier')
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
모델의 갯수 41
AdaBoostClassifier 의 정답률 평균: 0.8867
BaggingClassifier 의 정답률 평균: 0.9467
BernoulliNB 의 정답률 평균: 0.2933
CalibratedClassifierCV 의 정답률 평균: 0.9133
CategoricalNB 의 정답률 평균: 0.9333
ClassifierChain
ComplementNB 의 정답률 평균: 0.6667
DecisionTreeClassifier 의 정답률 평균: 0.9467       
DummyClassifier 의 정답률 평균: 0.2933
ExtraTreeClassifier 의 정답률 평균: 0.9267
ExtraTreesClassifier 의 정답률 평균: 0.9467
GaussianNB 의 정답률 평균: 0.9467
GaussianProcessClassifier 의 정답률 평균: 0.96
GradientBoostingClassifier 의 정답률 평균: 0.9667
HistGradientBoostingClassifier 의 정답률 평균: 0.94
KNeighborsClassifier 의 정답률 평균: 0.96
LabelPropagation 의 정답률 평균: 0.96
LabelSpreading 의 정답률 평균: 0.96
LinearDiscriminantAnalysis 의 정답률 평균: 0.98     
LinearSVC 의 정답률 평균: 0.9667
LogisticRegression 의 정답률 평균: 0.9667
LogisticRegressionCV 의 정답률 평균: 0.9733
MLPClassifier 의 정답률 평균: 0.98
MultiOutputClassifier
MultinomialNB 의 정답률 평균: 0.9667
NearestCentroid 의 정답률 평균: 0.9333
NuSVC 의 정답률 평균: 0.9733
OneVsOneClassifier
OneVsRestClassifier
OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률 평균: 0.8467  
Perceptron 의 정답률 평균: 0.78
QuadraticDiscriminantAnalysis 의 정답률 평균: 0.98  
RadiusNeighborsClassifier 의 정답률 평균: 0.9533    
RandomForestClassifier 의 정답률 평균: 0.96
RidgeClassifier 의 정답률 평균: 0.84
RidgeClassifierCV 의 정답률 평균: 0.84
SGDClassifier 의 정답률 평균: 0.92
SVC 의 정답률 평균: 0.9667
StackingClassifier
VotingClassifier
'''