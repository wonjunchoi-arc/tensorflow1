from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold, cross_val_score


datasets = load_wine()
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
AdaBoostClassifier 의 정답률 평균: 0.9106
BaggingClassifier 의 정답률 평균: 0.9554
BernoulliNB 의 정답률 평균: 0.399
CalibratedClassifierCV 의 정답률 평균: 0.9156
CategoricalNB 의 정답률 평균: nan
ClassifierChain
ComplementNB 의 정답률 평균: 0.6511
DecisionTreeClassifier 의 정답률 평균: 0.9267       
DummyClassifier 의 정답률 평균: 0.399
ExtraTreeClassifier 의 정답률 평균: 0.8648
ExtraTreesClassifier 의 정답률 평균: 0.9889
GaussianNB 의 정답률 평균: 0.9721
GaussianProcessClassifier 의 정답률 평균: 0.4783
GradientBoostingClassifier 의 정답률 평균: 0.9441
HistGradientBoostingClassifier 의 정답률 평균: 0.9776
KNeighborsClassifier 의 정답률 평균: 0.691
LabelPropagation 의 정답률 평균: 0.4886
LabelSpreading 의 정답률 평균: 0.4886
LinearDiscriminantAnalysis 의 정답률 평균: 0.9887   
LinearSVC 의 정답률 평균: 0.8038
LogisticRegression 의 정답률 평균: 0.9608
LogisticRegressionCV 의 정답률 평균: 0.9662
MLPClassifier 의 정답률 평균: 0.6879
MultiOutputClassifier
MultinomialNB 의 정답률 평균: 0.8425
NearestCentroid 의 정답률 평균: 0.7251
NuSVC 의 정답률 평균: 0.8703
OneVsOneClassifier
OneVsRestClassifier
OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률 평균: 0.6056
Perceptron 의 정답률 평균: 0.6006
QuadraticDiscriminantAnalysis 의 정답률 평균: 0.9944RadiusNeighborsClassifier 의 정답률 평균: nan
RandomForestClassifier 의 정답률 평균: 0.9832
RidgeClassifier 의 정답률 평균: 0.9943
RidgeClassifierCV 의 정답률 평균: 0.9943
SGDClassifier 의 정답률 평균: 0.629
SVC 의 정답률 평균: 0.6457
StackingClassifier
VotingClassifier
'''