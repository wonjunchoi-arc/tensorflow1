import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score


datasets = load_breast_cancer()

#1. 데이터
print(datasets.DESCR)

print(datasets.feature_names)

x= datasets.data
y= datasets.target

print(x.shape , y.shape)

print(y[:20])
print(np.unique(y))
#unique 는 특이한 애들을 찾는다. [0 1] 밖에 없다.  인진 분류 모델 

x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성 # all_estimators는 즉 모든 ml모델을 불러오는 것!
allAlgorithms= all_estimators(type_filter='classifier')
# print(allAlgorithms)
print('모델의 갯수',len(allAlgorithms))
for name, algorithms in allAlgorithms:
    try:
        model = algorithms()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률:', acc)
    except:
        print(name)
        continue
'''
AdaBoostClassifier 의 정답률: 0.9448621553884712
BaggingClassifier 의 정답률: 0.9298245614035088
BernoulliNB 의 정답률: 0.6416040100250626
CalibratedClassifierCV 의 정답률: 0.9573934837092731
CategoricalNB
ClassifierChain
ComplementNB 의 정답률: 0.8295739348370927
DecisionTreeClassifier 의 정답률: 0.9122807017543859
DummyClassifier 의 정답률: 0.6390977443609023
ExtraTreeClassifier 의 정답률: 0.9022556390977443
ExtraTreesClassifier 의 정답률: 0.9573934837092731
GaussianNB 의 정답률: 0.9197994987468672
GaussianProcessClassifier 의 정답률: 0.9548872180451128
GradientBoostingClassifier 의 정답률: 0.9172932330827067
HistGradientBoostingClassifier 의 정답률: 0.949874686716792
KNeighborsClassifier 의 정답률: 0.9548872180451128
LabelPropagation 의 정답률: 0.9548872180451128
LabelSpreading 의 정답률: 0.9548872180451128
LinearDiscriminantAnalysis 의 정답률: 0.9373433583959899
LinearSVC 의 정답률: 0.9598997493734336
LogisticRegression 의 정답률: 0.9523809523809523
LogisticRegressionCV 의 정답률: 0.9649122807017544
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:614: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  warnings.warn(
MLPClassifier 의 정답률: 0.9423558897243107
MultiOutputClassifier
MultinomialNB 의 정답률: 0.8471177944862155
NearestCentroid 의 정답률: 0.924812030075188
NuSVC 의 정답률: 0.9323308270676691
OneVsOneClassifier
OneVsRestClassifier
OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률: 0.9373433583959899
Perceptron 의 정답률: 0.9373433583959899
QuadraticDiscriminantAnalysis 의 정답률: 0.9423558897243107
RadiusNeighborsClassifier
RandomForestClassifier 의 정답률: 0.9473684210526315
RidgeClassifier 의 정답률: 0.9473684210526315
RidgeClassifierCV 의 정답률: 0.9473684210526315
SGDClassifier 의 정답률: 0.9473684210526315
SVC 의 정답률: 0.9624060150375939
StackingClassifier
VotingClassifier
'''