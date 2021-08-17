from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score


datasets = load_breast_cancer()
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
AdaBoostClassifier 의 정답률 평균: 0.9649
BaggingClassifier 의 정답률 평균: 0.9438
BernoulliNB 의 정답률 평균: 0.6274
CalibratedClassifierCV 의 정답률 평균: 0.9263
CategoricalNB 의 정답률 평균: nan
ClassifierChain
ComplementNB 의 정답률 평균: 0.8963
DecisionTreeClassifier 의 정답률 평균: 0.935
DummyClassifier 의 정답률 평균: 0.6274
ExtraTreeClassifier 의 정답률 평균: 0.9227
ExtraTreesClassifier 의 정답률 평균: 0.9666
GaussianNB 의 정답률 평균: 0.942
GaussianProcessClassifier 의 정답률 평균: 0.9122
GradientBoostingClassifier 의 정답률 평균: 0.9578
HistGradientBoostingClassifier 의 정답률 평균: 0.9737
KNeighborsClassifier 의 정답률 평균: 0.928
LabelPropagation 의 정답률 평균: 0.3902
LabelSpreading 의 정답률 평균: 0.3902
LinearDiscriminantAnalysis 의 정답률 평균: 0.9614   
LinearSVC 의 정답률 평균: 0.9105
LogisticRegression 의 정답률 평균: 0.9385
LogisticRegressionCV 의 정답률 평균: 0.9578
MLPClassifier 의 정답률 평균: 0.9192
MultiOutputClassifier
MultinomialNB 의 정답률 평균: 0.8928
NearestCentroid 의 정답률 평균: 0.8893
NuSVC 의 정답률 평균: 0.8735
OneVsOneClassifier
OneVsRestClassifier
OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률 평균: 0.87    
Perceptron 의 정답률 평균: 0.7771
QuadraticDiscriminantAnalysis 의 정답률 평균: 0.9525RadiusNeighborsClassifier 의 정답률 평균: nan
RandomForestClassifier 의 정답률 평균: 0.9631
RidgeClassifier 의 정답률 평균: 0.9543
RidgeClassifierCV 의 정답률 평균: 0.9561
SGDClassifier 의 정답률 평균: 0.8454
SVC 의 정답률 평균: 0.921
StackingClassifier
VotingClassifier

'''